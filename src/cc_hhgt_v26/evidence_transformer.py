"""Event-level evidence encoding for CancerLncAtlas.

The model keeps direct target assertions separate from indirect mechanism
routes.  Direct events are excluded from the discovery expert and are used only
for confidence scoring.  All train/test splits operate on OOF relation labels;
PMIDs present in held-out rows are removed from training event sets.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

KEYS = ["cancer_id", "lncrna_id", "pathway_family_id"]
CAT_FIELDS = [
    "route_type", "source_database", "experiment_family", "relation_type",
    "direction", "tissue", "cell_line", "species", "manual_review_status",
    "evidence_level", "partner_type",
]
NUM_FIELDS = [
    "event_weight", "fdr_score", "n_independent_pmids", "n_independent_events",
    "is_experimental", "is_predicted",
]
UNKNOWN = "<UNK>"
PADDING = "<PAD>"


def stable_fold(values: Iterable[object], n_folds: int = 5) -> int:
    text = "\x1f".join(str(v) for v in values)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16) % n_folds


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(labels) & np.isfinite(probability)
    labels = labels[valid]
    probability = probability[valid]
    if not len(labels):
        return {"auprc": math.nan, "auroc": math.nan, "brier": math.nan}
    binary = (labels >= 0.5).astype(int)
    return {
        "auprc": float(average_precision_score(binary, probability)) if len(np.unique(binary)) > 1 else math.nan,
        "auroc": float(roc_auc_score(binary, probability)) if len(np.unique(binary)) > 1 else math.nan,
        "brier": float(brier_score_loss(binary, probability)),
    }


@dataclass
class VocabularyBundle:
    values: dict[str, list[str]]
    fields: list[str] | None = None

    def __post_init__(self) -> None:
        if self.fields is None:
            self.fields = list(self.values)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        max_per_field: int = 5000,
        *,
        drop_constant_fields: bool = False,
        merge_redundant_fields: bool = False,
    ) -> "VocabularyBundle":
        values: dict[str, list[str]] = {}
        fields: list[str] = []
        signatures: dict[tuple[str, ...], str] = {}
        for field in CAT_FIELDS:
            series = frame.get(field, pd.Series(UNKNOWN, index=frame.index)).astype(str)
            if drop_constant_fields and series.nunique(dropna=False) <= 1:
                continue
            signature = tuple(series.tolist())
            if merge_redundant_fields and signature in signatures:
                # Exact duplicate categorical columns add parameters without
                # information; retain the first registered field only.
                continue
            signatures[signature] = field
            counts = series.value_counts()
            tokens = [PADDING, UNKNOWN] + [str(x) for x in counts.index[:max_per_field] if str(x) not in {PADDING, UNKNOWN}]
            values[field] = tokens
            fields.append(field)
        return cls(values, fields)

    def mapping(self, field: str) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.values[field])}

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps({"fields": self.fields, "values": self.values}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> "VocabularyBundle":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "values" in payload:
            return cls(payload["values"], payload.get("fields"))
        return cls(payload)


def clean_context_token(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "na", "n/a", "unknown", "not available"}:
        return UNKNOWN
    text = re.sub(r"[\s_/\\|]+", " ", text)
    text = re.sub(r"[^a-z0-9 +.-]", "", text)
    return text.strip() or UNKNOWN


def clean_evidence_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the registered tissue/cell-line token cleaning before splitting."""

    output = frame.copy()
    for field in CAT_FIELDS:
        values = output.get(field, pd.Series(UNKNOWN, index=output.index))
        if field in {"tissue", "cell_line"}:
            output[field] = values.map(clean_context_token)
        else:
            output[field] = values.astype(str).str.strip().replace(
                {"": UNKNOWN, "nan": UNKNOWN, "None": UNKNOWN, "unknown": UNKNOWN}
            )
    return output


class EvidenceStore:
    def __init__(self, frame: pd.DataFrame, vocab: VocabularyBundle) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.categorical_fields = list(vocab.fields or [])
        cat_arrays = []
        for field in self.categorical_fields:
            mapping = vocab.mapping(field)
            values = self.frame.get(field, pd.Series(UNKNOWN, index=self.frame.index)).astype(str)
            cat_arrays.append(values.map(mapping).fillna(mapping[UNKNOWN]).to_numpy(np.int64))
        self.cat = (
            np.column_stack(cat_arrays).astype(np.int64)
            if cat_arrays
            else np.zeros((len(self.frame), 0), dtype=np.int64)
        )
        numeric = self.frame.reindex(columns=NUM_FIELDS).apply(pd.to_numeric, errors="coerce").fillna(0)
        numeric["n_independent_pmids"] = np.log1p(numeric["n_independent_pmids"].clip(lower=0))
        numeric["n_independent_events"] = np.log1p(numeric["n_independent_events"].clip(lower=0))
        self.numeric = numeric.to_numpy(np.float32)
        self.direct = pd.to_numeric(self.frame.get("direct_target_evidence", pd.Series(0, index=self.frame.index)), errors="coerce").fillna(0).to_numpy(np.int8)
        self.pmids = self.frame.get("pmid", pd.Series("unknown", index=self.frame.index)).astype(str).to_numpy()
        self.weights = pd.to_numeric(self.frame.get("event_weight", pd.Series(0, index=self.frame.index)), errors="coerce").fillna(0).to_numpy(float)
        self.global_map: dict[tuple[str, str], list[int]] = {}
        self.local_map: dict[tuple[str, str, str], list[int]] = {}
        for index, row in enumerate(self.frame[["cancer_id", "lncrna_id", "pathway_family_id"]].itertuples(index=False, name=None)):
            cancer, lnc, pathway = map(str, row)
            if cancer == "PAN_CANCER":
                self.global_map.setdefault((lnc, pathway), []).append(index)
            else:
                self.local_map.setdefault((cancer, lnc, pathway), []).append(index)

    def indices_for(self, cancer: str, lnc: str, pathway: str) -> list[int]:
        return self.global_map.get((lnc, pathway), []) + self.local_map.get((cancer, lnc, pathway), [])

    def pmids_for_candidates(self, candidates: pd.DataFrame) -> set[str]:
        pmids: set[str] = set()
        for row in candidates[KEYS].itertuples(index=False, name=None):
            for index in self.indices_for(*map(str, row)):
                value = self.pmids[index]
                if value not in {"", "unknown", "nan", "None"}:
                    pmids.add(str(value))
        return pmids

    def direct_quality_for_indices(self, indices: list[int]) -> float:
        """Monotonic direct-evidence strength target, independent of relation labels.

        Direct assertions are usually positive evidence and therefore do not
        provide a meaningful negative class.  Training a binary direct head on
        relation labels would leave it untrained or make it a target-presence
        detector.  Instead, the direct head learns a bounded evidence-quality
        target derived from independent event weights.
        """
        direct = [idx for idx in indices if self.direct[idx] == 1]
        if not direct:
            return float("nan")
        total = float(np.clip(self.weights[direct], 0, 2).sum())
        return float(np.clip(1.0 - np.exp(-total), 0.0, 1.0))


class CandidateDataset(Dataset):
    def __init__(
        self,
        candidates: pd.DataFrame,
        store: EvidenceStore,
        *,
        max_events: int = 64,
        forbidden_pmids: set[str] | None = None,
        require_event: bool = True,
    ) -> None:
        self.store = store
        self.max_events = max_events
        self.forbidden_pmids = forbidden_pmids or set()
        rows = []
        for row in candidates.itertuples(index=False):
            key = (str(row.cancer_id), str(row.lncrna_id), str(row.pathway_family_id))
            indices = [
                idx for idx in store.indices_for(*key)
                if str(store.pmids[idx]) not in self.forbidden_pmids
            ]
            if require_event and not indices:
                continue
            indices = sorted(indices, key=lambda idx: store.weights[idx], reverse=True)
            retained = indices[: max_events * 2]
            pmids = {str(store.pmids[idx]) for idx in retained if str(store.pmids[idx]) not in {"", "unknown", "nan", "None"}}
            sources = {
                str(store.frame.iloc[idx].get("source_database", UNKNOWN)) for idx in retained
            }
            assays = {
                str(store.frame.iloc[idx].get("experiment_family", UNKNOWN)) for idx in retained
            }
            count_features = np.log1p(
                np.asarray([len(retained), len(pmids), len(sources), len(assays)], dtype=np.float32)
            )
            rows.append((
                key, retained, float(getattr(row, "label", np.nan)),
                store.direct_quality_for_indices(retained), count_features,
            ))
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def collate_candidates(batch, store: EvidenceStore, max_events: int):
    batch_size = len(batch)
    cat_indirect = np.zeros((batch_size, max_events, len(store.categorical_fields)), dtype=np.int64)
    num_indirect = np.zeros((batch_size, max_events, len(NUM_FIELDS)), dtype=np.float32)
    mask_indirect = np.zeros((batch_size, max_events), dtype=bool)
    cat_direct = np.zeros_like(cat_indirect)
    num_direct = np.zeros_like(num_indirect)
    mask_direct = np.zeros_like(mask_indirect)
    labels = np.empty(batch_size, dtype=np.float32)
    direct_quality = np.full(batch_size, np.nan, dtype=np.float32)
    count_features = np.zeros((batch_size, 4), dtype=np.float32)
    keys = []
    for row_index, (key, indices, label, direct_target, counts) in enumerate(batch):
        indirect = [idx for idx in indices if store.direct[idx] == 0][:max_events]
        direct = [idx for idx in indices if store.direct[idx] == 1][:max_events]
        if indirect:
            cat_indirect[row_index, : len(indirect)] = store.cat[indirect]
            num_indirect[row_index, : len(indirect)] = store.numeric[indirect]
            mask_indirect[row_index, : len(indirect)] = True
        if direct:
            cat_direct[row_index, : len(direct)] = store.cat[direct]
            num_direct[row_index, : len(direct)] = store.numeric[direct]
            mask_direct[row_index, : len(direct)] = True
        labels[row_index] = label
        direct_quality[row_index] = direct_target
        count_features[row_index] = counts
        keys.append(key)
    return {
        "keys": keys,
        "cat_indirect": torch.from_numpy(cat_indirect),
        "num_indirect": torch.from_numpy(num_indirect),
        "mask_indirect": torch.from_numpy(mask_indirect),
        "cat_direct": torch.from_numpy(cat_direct),
        "num_direct": torch.from_numpy(num_direct),
        "mask_direct": torch.from_numpy(mask_direct),
        "labels": torch.from_numpy(labels),
        "direct_quality_target": torch.from_numpy(direct_quality),
        "count_features": torch.from_numpy(count_features),
    }


class EventSetEncoder(nn.Module):
    def __init__(
        self,
        vocab_sizes: list[int],
        numeric_dim: int,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, d_model, padding_idx=0) for size in vocab_sizes])
        self.numeric_projection = nn.Sequential(nn.Linear(numeric_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 3,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        available = mask.any(dim=1)
        token = self.numeric_projection(numeric)
        for index, embedding in enumerate(self.embeddings):
            token = token + embedding(categorical[:, :, index])
        token = token * mask.unsqueeze(-1)
        cls = self.cls.expand(token.shape[0], -1, -1)
        sequence = torch.cat([cls, token], dim=1)
        padding = torch.cat([torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device), ~mask], dim=1)
        encoded = self.transformer(sequence, src_key_padding_mask=padding)
        pooled = self.output_norm(encoded[:, 0])
        pooled = pooled * available.unsqueeze(-1)
        return pooled, available


class EvidenceTransformer(nn.Module):
    def __init__(
        self,
        vocab_sizes: list[int],
        numeric_dim: int,
        d_model: int = 96,
        *,
        direct_head_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.direct_head_enabled = bool(direct_head_enabled)
        self.event_encoder = EventSetEncoder(vocab_sizes, numeric_dim, d_model=d_model)
        self.indirect_head = nn.Linear(d_model, 1)
        self.direct_head = nn.Linear(d_model, 1)
        self.combined = nn.Sequential(
            nn.Linear(d_model * 2 + 2 + 4, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(0.15), nn.Linear(d_model, 1)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        indirect, indirect_available = self.event_encoder(batch["cat_indirect"], batch["num_indirect"], batch["mask_indirect"])
        direct, direct_available = self.event_encoder(batch["cat_direct"], batch["num_direct"], batch["mask_direct"])
        if not self.direct_head_enabled:
            direct = torch.zeros_like(direct)
            direct_available = torch.zeros_like(direct_available)
        availability = torch.stack([indirect_available, direct_available], dim=-1).float()
        combined_logit = self.combined(
            torch.cat([indirect, direct, availability, batch["count_features"].float()], dim=-1)
        ).squeeze(-1)
        return {
            "indirect_logit": self.indirect_head(indirect).squeeze(-1),
            "direct_logit": self.direct_head(direct).squeeze(-1),
            "combined_logit": combined_logit,
            "indirect_available": indirect_available,
            "direct_available": direct_available,
        }


@dataclass
class EvidenceFit:
    model: EvidenceTransformer
    best_state: dict[str, torch.Tensor]
    history: list[dict[str, float]]
    metrics: dict[str, float]


def _move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _weighted_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive = labels.sum().clamp_min(1)
    negative = (labels.numel() - labels.sum()).clamp_min(1)
    pos_weight = (negative / positive).clamp(1, 20)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def fit_evidence_transformer(
    train_dataset: CandidateDataset,
    validation_dataset: CandidateDataset,
    vocab: VocabularyBundle,
    store: EvidenceStore,
    *,
    seed: int = 20260731,
    epochs: int = 40,
    patience: int = 6,
    batch_size: int = 512,
    max_events: int = 64,
    device: str | None = None,
    direct_head_enabled: bool = True,
) -> EvidenceFit:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab_sizes = [len(vocab.values[field]) for field in (vocab.fields or [])]
    model = EvidenceTransformer(
        vocab_sizes,
        len(NUM_FIELDS),
        direct_head_enabled=direct_head_enabled,
    ).to(torch_device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=lambda b: collate_candidates(b, store, max_events))
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate_candidates(b, store, max_events))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    best_state = None; best_score = -math.inf; stale = 0; history = []
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for batch in train_loader:
            batch = _move(batch, torch_device); labels = batch["labels"].float()
            result = model(batch); terms = []
            any_available = result["indirect_available"] | result["direct_available"]
            if any_available.any(): terms.append(_weighted_bce(result["combined_logit"][any_available], labels[any_available]))
            if result["indirect_available"].any(): terms.append(_weighted_bce(result["indirect_logit"][result["indirect_available"]], labels[result["indirect_available"]]))
            direct_target = batch["direct_quality_target"].float()
            direct_mask = result["direct_available"] & torch.isfinite(direct_target)
            if direct_mask.any():
                terms.append(0.5 * nn.functional.binary_cross_entropy_with_logits(
                    result["direct_logit"][direct_mask], direct_target[direct_mask]
                ))
            if not terms: continue
            loss = torch.stack(terms).sum(); optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step(); losses.append(float(loss.detach().cpu()))
        val = predict_evidence_transformer(model, validation_dataset, store, batch_size=batch_size, max_events=max_events, device=str(torch_device))
        val_metrics = metrics(val["label"].to_numpy(float), val["neural_evidence_probability"].to_numpy(float))
        score = val_metrics["auprc"] if np.isfinite(val_metrics["auprc"]) else -np.mean(losses or [999])
        history.append({"epoch": epoch, "loss": float(np.mean(losses or [np.nan])), **{f"validation_{k}": v for k, v in val_metrics.items()}})
        if score > best_score + 1e-5:
            best_score = score; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        if stale >= patience: break
    if best_state is None: raise RuntimeError("Evidence Transformer produced no checkpoint")
    model.load_state_dict(best_state)
    val = predict_evidence_transformer(model, validation_dataset, store, batch_size=batch_size, max_events=max_events, device=str(torch_device))
    return EvidenceFit(model, best_state, history, metrics(val["label"].to_numpy(float), val["neural_evidence_probability"].to_numpy(float)))


def predict_evidence_transformer(
    model: EvidenceTransformer,
    dataset: CandidateDataset,
    store: EvidenceStore,
    *, batch_size: int = 1024,
    max_events: int = 64,
    device: str | None = None,
) -> pd.DataFrame:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(torch_device); model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate_candidates(b, store, max_events))
    rows = []
    with torch.no_grad():
        for batch in loader:
            keys = batch["keys"]; label = batch["labels"].numpy(); batch = _move(batch, torch_device)
            result = model(batch)
            indirect_p = torch.sigmoid(result["indirect_logit"]).cpu().numpy()
            direct_p = torch.sigmoid(result["direct_logit"]).cpu().numpy()
            combined_p = torch.sigmoid(result["combined_logit"]).cpu().numpy()
            indirect_a = result["indirect_available"].cpu().numpy(); direct_a = result["direct_available"].cpu().numpy()
            for key, y, ip, dp, cp, ia, da in zip(keys, label, indirect_p, direct_p, combined_p, indirect_a, direct_a, strict=True):
                rows.append({
                    **dict(zip(KEYS, key, strict=True)), "label": y,
                    "indirect_mechanism_probability": float(ip) if ia else np.nan,
                    "direct_evidence_probability": float(dp) if da else np.nan,
                    "neural_evidence_probability": float(cp) if (ia or da) else np.nan,
                    "indirect_evidence_available": int(ia), "direct_evidence_available": int(da),
                })
    return pd.DataFrame(rows)
