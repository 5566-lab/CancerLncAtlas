"""Count-base plus event-quality residual models for V3.1 evidence."""
from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cc_hhgt_v26.evidence_transformer import (
    CAT_FIELDS,
    KEYS,
    NUM_FIELDS,
    UNKNOWN,
    VocabularyBundle,
    clean_evidence_frame,
)

from .metrics import binary_metrics
from .v31_residual import FALLBACK_ATOL, logit_to_probability, probability_to_logit


COUNT_FEATURES = (
    "log1p_raw_event_count",
    "log1p_deduplicated_event_count",
    "log1p_unique_pmid_count",
    "log1p_unique_source_count",
    "log1p_unique_assay_count",
    "log1p_independent_event_count",
)
QUALITY_FEATURES = (
    "mean_event_weight",
    "maximum_event_weight",
    "experimental_event_fraction",
    "predicted_event_fraction",
    "known_pmid_fraction",
    "direction_conflict",
    "retained_event_fraction",
    "retained_weight_fraction",
)
GLOBAL_FEATURES = (*COUNT_FEATURES, *QUALITY_FEATURES)
ARCHITECTURES = ("DEEPSETS_SUM", "DEEPSETS_MEAN", "DEEPSETS_MAX", "TRANSFORMER")


def stable_lnc_fold(lncrna_id: object, n_folds: int = 5) -> int:
    digest = hashlib.sha256(str(lncrna_id).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(n_folds)


def prepare_evidence_candidates(
    historical_oof: pd.DataFrame,
    *,
    cancers: Sequence[str],
    replicate_folds: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recover exact k/n labels and use a strict lncRNA-grouped split."""

    required = [*KEYS, "label"]
    missing = sorted(set(required) - set(historical_oof.columns))
    if missing:
        raise RuntimeError(f"Historical evidence OOF lacks {missing}")
    frame = historical_oof.loc[
        historical_oof.cancer_id.astype(str).isin(map(str, cancers)), required
    ].copy()
    if frame.empty or frame.duplicated(KEYS).any():
        raise RuntimeError("Evidence candidates are empty or duplicated")
    frame["replicate_fraction"] = pd.to_numeric(frame.label, errors="coerce")
    expected_k = frame.replicate_fraction * int(replicate_folds)
    if (~np.isfinite(expected_k)).any() or np.max(np.abs(expected_k - np.round(expected_k))) > 1e-8:
        raise RuntimeError("Evidence soft labels cannot be recovered as exact k/n")
    frame["replicate_k"] = np.round(expected_k).astype(np.int8)
    frame["replicate_n"] = int(replicate_folds)
    frame["binary_label"] = frame.replicate_k.ge(math.ceil(replicate_folds / 2)).astype(np.int8)
    frame["evidence_fold"] = frame.lncrna_id.map(stable_lnc_fold).astype(np.int8)
    frame = frame.drop(columns="label").sort_values(KEYS, kind="stable").reset_index(drop=True)
    leakage = frame.groupby("lncrna_id", observed=True).evidence_fold.nunique()
    if not leakage.eq(1).all():
        raise RuntimeError("An lncRNA/source-event supergroup crosses evidence folds")
    audit = {
        "status": "PASS",
        "cancers": sorted(map(str, cancers)),
        "candidate_rows": int(len(frame)),
        "unique_lncrnas": int(frame.lncrna_id.nunique()),
        "replicate_n": int(replicate_folds),
        "replicate_k_values": sorted(map(int, frame.replicate_k.unique())),
        "binary_label_threshold": f"k>={math.ceil(replicate_folds / 2)} of n={replicate_folds}",
        "split_group": "lncrna_id strict supergroup",
        "lncrna_groups_crossing_folds": 0,
        "same_pair_across_cancer_crossing_folds": 0,
    }
    return frame, audit


def _parent_event_hash(events: pd.DataFrame) -> pd.Series:
    pmid = events.pmid.astype(str)
    known = ~pmid.isin(["", "unknown", "nan", "None"])
    source_anchor = np.where(
        known,
        "PMID:" + pmid,
        "SOURCE:"
        + events.source_database.astype(str)
        + "|" + events.source_file.astype(str),
    )
    identity = pd.DataFrame(
        {
            "lncrna_id": events.lncrna_id.astype(str),
            "source_anchor": source_anchor,
            "partner_id": events.partner_id.astype(str),
            "experiment_family": events.experiment_family.astype(str),
            "relation_type": events.relation_type.astype(str),
            "direction": events.direction.astype(str),
            "tissue": events.tissue.astype(str),
            "cell_line": events.cell_line.astype(str),
        }
    )
    return pd.util.hash_pandas_object(identity, index=False).astype("uint64").astype(str)


def build_candidate_event_records(
    candidates: pd.DataFrame,
    raw_events: pd.DataFrame,
    *,
    max_events: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach deduplicated event sets and explicit overflow/count summaries."""

    if max_events < 1:
        raise ValueError("max_events must be positive")
    required = {
        *KEYS, "event_id", "pmid", "source_database", "experiment_family",
        "partner_id", "relation_type", "direction", "tissue", "cell_line",
        "source_file", "event_weight", "is_experimental", "is_predicted",
        *CAT_FIELDS, *NUM_FIELDS,
    }
    missing = sorted(required - set(raw_events.columns))
    if missing:
        raise RuntimeError(f"Evidence events lack {missing}")
    events = clean_evidence_frame(raw_events).reset_index(drop=True)
    events["parent_event_key"] = _parent_event_hash(events)
    events["event_weight"] = pd.to_numeric(events.event_weight, errors="coerce").fillna(0.0)
    candidate_pairs = set(
        candidates[["lncrna_id", "pathway_family_id"]].astype(str).itertuples(index=False, name=None)
    )
    pair_index = pd.MultiIndex.from_frame(events[["lncrna_id", "pathway_family_id"]].astype(str))
    events = events.loc[pair_index.isin(candidate_pairs)].reset_index(drop=True)
    if events.empty:
        raise RuntimeError("No evidence events apply to pilot candidates")
    global_map = {
        tuple(map(str, key)): group.index.to_numpy(np.int64)
        for key, group in events.loc[events.cancer_id.astype(str).eq("PAN_CANCER")].groupby(
            ["lncrna_id", "pathway_family_id"], observed=True, sort=False
        )
    }
    local_map = {
        tuple(map(str, key)): group.index.to_numpy(np.int64)
        for key, group in events.loc[~events.cancer_id.astype(str).eq("PAN_CANCER")].groupby(
            KEYS, observed=True, sort=False
        )
    }
    rows: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for candidate in candidates.itertuples(index=False):
        key = (str(candidate.cancer_id), str(candidate.lncrna_id), str(candidate.pathway_family_id))
        pair = key[1:]
        indices = np.concatenate(
            [global_map.get(pair, np.empty(0, dtype=np.int64)), local_map.get(key, np.empty(0, dtype=np.int64))]
        )
        if not len(indices):
            continue
        applicable = events.iloc[indices].copy()
        raw_count = int(len(applicable))
        applicable = applicable.sort_values(
            ["event_weight", "event_id"], ascending=[False, True], kind="stable"
        ).drop_duplicates("parent_event_key", keep="first")
        deduplicated_indices = applicable.index.to_numpy(np.int64)
        selected = deduplicated_indices[:max_events]
        used_indices.update(map(int, selected))
        pmid = applicable.pmid.astype(str)
        known_pmid = ~pmid.isin(["", "unknown", "nan", "None", UNKNOWN])
        independent = np.where(
            known_pmid,
            "PMID:" + pmid,
            "PARENT:" + applicable.parent_event_key.astype(str),
        )
        directions = set(applicable.direction.astype(str).str.lower())
        positive = any("posit" in value for value in directions)
        negative = any("negat" in value for value in directions)
        all_weight = np.clip(applicable.event_weight.to_numpy(float), 0, None)
        retained_weight = float(np.sum(all_weight[:max_events]))
        total_weight = float(np.sum(all_weight))
        count_values = {
            "raw_event_count": raw_count,
            "deduplicated_event_count": int(len(applicable)),
            "unique_pmid_count": int(pmid.loc[known_pmid].nunique()),
            "unique_source_count": int(applicable.source_database.astype(str).nunique()),
            "unique_assay_count": int(applicable.experiment_family.astype(str).nunique()),
            "independent_event_count": int(pd.Series(independent).nunique()),
        }
        row = {column: getattr(candidate, column) for column in candidates.columns}
        row.update(count_values)
        for name, value in count_values.items():
            row[f"log1p_{name}"] = float(np.log1p(value))
        row.update(
            {
                "mean_event_weight": float(np.mean(all_weight)),
                "maximum_event_weight": float(np.max(all_weight)),
                "experimental_event_fraction": float(pd.to_numeric(applicable.is_experimental).fillna(0).mean()),
                "predicted_event_fraction": float(pd.to_numeric(applicable.is_predicted).fillna(0).mean()),
                "known_pmid_fraction": float(known_pmid.mean()),
                "direction_conflict": float(positive and negative),
                "retained_event_fraction": float(min(len(applicable), max_events) / len(applicable)),
                "retained_weight_fraction": float(retained_weight / total_weight) if total_weight > 0 else 1.0,
                "selected_event_indices": selected,
            }
        )
        rows.append(row)
    records = pd.DataFrame(rows)
    if records.empty:
        raise RuntimeError("No event-bearing evidence candidates remain")
    used = np.asarray(sorted(used_indices), dtype=np.int64)
    remap = pd.Series(np.arange(len(used), dtype=np.int64), index=used)
    compact_events = events.iloc[used].reset_index(drop=True)
    records["selected_event_indices"] = records.selected_event_indices.map(
        lambda values: remap.reindex(values).to_numpy(np.int64)
    )
    audit = {
        "status": "PASS",
        "input_event_rows": int(len(raw_events)),
        "pair_relevant_event_rows": int(len(events)),
        "retained_unique_event_rows": int(len(compact_events)),
        "candidate_rows": int(len(records)),
        "candidates_without_events_excluded": int(len(candidates) - len(records)),
        "max_events": int(max_events),
        "raw_event_count_sum": int(records.raw_event_count.sum()),
        "deduplicated_event_count_sum": int(records.deduplicated_event_count.sum()),
        "retained_event_fraction_weighted": float(
            np.minimum(records.deduplicated_event_count, max_events).sum()
            / records.deduplicated_event_count.sum()
        ),
        "mean_retained_weight_fraction": float(records.retained_weight_fraction.mean()),
        "parent_event_deduplication": "lncRNA + PMID/source + partner + assay + relation + direction + tissue + cell_line",
        "event_selection": "event_weight_descending_then_event_id_stable",
        "outcome_used_for_event_selection": False,
    }
    return records, compact_events, audit


def _expanded_binomial_rows(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = []
    labels: list[int] = []
    for index, row in enumerate(frame[["replicate_k", "replicate_n"]].itertuples(index=False)):
        k, n = int(row.replicate_k), int(row.replicate_n)
        indices.extend([index] * n)
        labels.extend([1] * k + [0] * (n - k))
    return np.asarray(indices, dtype=np.int64), np.asarray(labels, dtype=np.int8)


def _count_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, list(COUNT_FEATURES)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise RuntimeError("Count base features contain missing values")
    return values.to_numpy(float)


def fit_count_logistic(frame: pd.DataFrame, *, c_value: float, seed: int):
    rows, labels = _expanded_binomial_rows(frame)
    if np.unique(labels).size < 2:
        raise RuntimeError("Count logistic requires both replicate outcomes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            class_weight=None,
            max_iter=5000,
            random_state=int(seed),
        ),
    )
    matrix = _count_matrix(frame)
    model.fit(matrix[rows], labels)
    return model


def select_count_base(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    c_grid: Sequence[float],
    seed: int,
) -> tuple[Any, pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []
    models: dict[float, Any] = {}
    for c_value in map(float, c_grid):
        model = fit_count_logistic(train, c_value=c_value, seed=seed)
        probability = model.predict_proba(_count_matrix(validation))[:, 1]
        metric = binary_metrics(validation.binary_label, probability)
        rows.append({"count_C": c_value, **metric})
        models[c_value] = model
    metrics = pd.DataFrame(rows)
    selected = float(
        metrics.sort_values(["auprc", "brier", "count_C"], ascending=[False, True, True]).iloc[0].count_C
    )
    return models[selected], metrics, selected


def count_crossfit_offsets(
    train: pd.DataFrame,
    *,
    c_value: float,
    seed: int,
) -> np.ndarray:
    output = np.full(len(train), np.nan)
    folds = sorted(train.evidence_fold.astype(int).unique())
    for fold in folds:
        test = train.evidence_fold.astype(int).eq(fold).to_numpy()
        model = fit_count_logistic(
            train.loc[~test], c_value=c_value, seed=seed + int(fold)
        )
        output[test] = model.predict_proba(_count_matrix(train.loc[test]))[:, 1]
    if not np.isfinite(output).all():
        raise RuntimeError("Count crossfit offsets are incomplete")
    return probability_to_logit(output)


@dataclass(frozen=True)
class EncodedEvidenceEvents:
    categorical: np.ndarray
    numeric: np.ndarray
    categorical_fields: tuple[str, ...]


def encode_evidence_events(
    events: pd.DataFrame,
    vocabulary: VocabularyBundle,
) -> EncodedEvidenceEvents:
    fields = tuple(vocabulary.fields or [])
    categorical_parts: list[np.ndarray] = []
    for field in fields:
        mapping = vocabulary.mapping(field)
        categorical_parts.append(
            events[field].astype(str).map(mapping).fillna(mapping[UNKNOWN]).to_numpy(np.int64)
        )
    categorical = (
        np.column_stack(categorical_parts)
        if categorical_parts
        else np.zeros((len(events), 0), dtype=np.int64)
    )
    numeric = events.reindex(columns=NUM_FIELDS).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for column in ("n_independent_pmids", "n_independent_events"):
        numeric[column] = np.log1p(numeric[column].clip(lower=0))
    return EncodedEvidenceEvents(categorical, numeric.to_numpy(np.float32), fields)


@dataclass(frozen=True)
class GlobalFeatureTransform:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "GlobalFeatureTransform":
        matrix = frame.loc[:, list(GLOBAL_FEATURES)].to_numpy(float)
        if not np.isfinite(matrix).all():
            raise RuntimeError("Evidence global features are non-finite")
        center = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(center, scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.loc[:, list(GLOBAL_FEATURES)].to_numpy(float)
        transformed = (matrix - self.center) / self.scale
        if not np.isfinite(transformed).all():
            raise RuntimeError("Evidence global transform produced non-finite values")
        return transformed.astype(np.float32)


class EvidenceCandidateDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        global_features: np.ndarray,
        count_logit: np.ndarray,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.global_features = np.asarray(global_features, dtype=np.float32)
        self.count_logit = np.asarray(count_logit, dtype=np.float32)
        if self.global_features.shape != (len(frame), len(GLOBAL_FEATURES)):
            raise RuntimeError("Evidence global features are not candidate aligned")
        if self.count_logit.shape != (len(frame),) or not np.isfinite(self.count_logit).all():
            raise RuntimeError("Evidence count offsets are not candidate aligned")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return {
            "key": tuple(str(row[column]) for column in KEYS),
            "event_indices": np.asarray(row.selected_event_indices, dtype=np.int64),
            "replicate_k": float(row.replicate_k),
            "replicate_n": float(row.replicate_n),
            "binary_label": int(row.binary_label),
            "global_features": self.global_features[index],
            "count_logit": float(self.count_logit[index]),
        }


def collate_evidence_candidates(
    batch: list[dict[str, Any]],
    events: EncodedEvidenceEvents,
    max_events: int,
) -> dict[str, Any]:
    n = len(batch)
    categorical = np.zeros(
        (n, max_events, events.categorical.shape[1]), dtype=np.int64
    )
    numeric = np.zeros((n, max_events, events.numeric.shape[1]), dtype=np.float32)
    mask = np.zeros((n, max_events), dtype=bool)
    for row_index, row in enumerate(batch):
        indices = row["event_indices"][:max_events]
        if len(indices):
            categorical[row_index, : len(indices)] = events.categorical[indices]
            numeric[row_index, : len(indices)] = events.numeric[indices]
            mask[row_index, : len(indices)] = True
    return {
        "keys": [row["key"] for row in batch],
        "categorical": torch.from_numpy(categorical),
        "numeric": torch.from_numpy(numeric),
        "mask": torch.from_numpy(mask),
        "global_features": torch.from_numpy(
            np.stack([row["global_features"] for row in batch]).astype(np.float32)
        ),
        "count_logit": torch.tensor([row["count_logit"] for row in batch], dtype=torch.float32),
        "replicate_k": torch.tensor([row["replicate_k"] for row in batch], dtype=torch.float32),
        "replicate_n": torch.tensor([row["replicate_n"] for row in batch], dtype=torch.float32),
        "binary_label": torch.tensor([row["binary_label"] for row in batch], dtype=torch.int64),
    }


class EventQualityResidualModel(nn.Module):
    def __init__(
        self,
        vocab_sizes: Sequence[int],
        *,
        architecture: str,
        mode: str,
        d_model: int = 32,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown evidence architecture: {architecture}")
        if mode not in {"standalone", "residual"}:
            raise ValueError(f"Unknown evidence mode: {mode}")
        self.architecture = architecture
        self.mode = mode
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(size), d_model, padding_idx=0) for size in vocab_sizes]
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(len(NUM_FIELDS), d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        if architecture == "TRANSFORMER":
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=1)
            nn.init.normal_(self.cls, std=0.02)
        self.pooled_norm = nn.LayerNorm(d_model)
        self.body = nn.Sequential(
            nn.Linear(d_model + len(GLOBAL_FEATURES), 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.output = nn.Linear(48, 1)
        if mode == "residual":
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def _tokens(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        token = self.numeric_projection(numeric)
        for index, embedding in enumerate(self.embeddings):
            token = token + embedding(categorical[:, :, index])
        return token

    def _pool(self, token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token = token * mask.unsqueeze(-1)
        if self.architecture == "DEEPSETS_SUM":
            pooled = token.sum(dim=1)
        elif self.architecture == "DEEPSETS_MEAN":
            pooled = token.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        elif self.architecture == "DEEPSETS_MAX":
            pooled = token.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(dim=1).values
            pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        else:
            cls = self.cls.expand(token.shape[0], -1, -1)
            sequence = torch.cat([cls, token], dim=1)
            padding = torch.cat(
                [
                    torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device),
                    ~mask,
                ],
                dim=1,
            )
            pooled = self.transformer(sequence, src_key_padding_mask=padding)[:, 0]
        return self.pooled_norm(pooled)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pooled = self._pool(
            self._tokens(batch["categorical"], batch["numeric"]), batch["mask"]
        )
        score = self.output(
            self.body(torch.cat([pooled, batch["global_features"]], dim=-1))
        ).squeeze(-1)
        if self.mode == "residual":
            return {
                "logit": batch["count_logit"] + score,
                "event_quality_residual_logit": score,
            }
        return {"logit": score, "event_quality_residual_logit": torch.zeros_like(score)}


def binomial_nll(logit: torch.Tensor, k: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return (n * nn.functional.softplus(logit) - k * logit).sum() / n.sum().clamp_min(1)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@dataclass
class EvidenceModelFit:
    model: EventQualityResidualModel
    best_state: dict[str, torch.Tensor]
    history: list[dict[str, float]]
    validation_metrics: dict[str, float]
    initialization_max_abs_error: float
    prediction_summary: dict[str, float]


def predict_evidence_model(
    model: EventQualityResidualModel,
    dataset: EvidenceCandidateDataset,
    encoded_events: EncodedEvidenceEvents,
    *,
    max_events: int,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda rows: collate_evidence_candidates(rows, encoded_events, max_events),
    )
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_batch in loader:
            keys = raw_batch["keys"]
            batch = _move(raw_batch, torch_device)
            result = model(batch)
            logits = result["logit"].detach().cpu().numpy()
            deltas = result["event_quality_residual_logit"].detach().cpu().numpy()
            count_logits = batch["count_logit"].detach().cpu().numpy()
            probabilities = logit_to_probability(logits)
            for index, key in enumerate(keys):
                rows.append(
                    {
                        **dict(zip(KEYS, key)),
                        "replicate_k": float(batch["replicate_k"][index].cpu()),
                        "replicate_n": float(batch["replicate_n"][index].cpu()),
                        "binary_label": int(batch["binary_label"][index].cpu()),
                        "count_logit": float(count_logits[index]),
                        "event_quality_residual_logit": float(deltas[index]),
                        "raw_logit": float(logits[index]),
                        "proxy_positive_probability": float(probabilities[index]),
                    }
                )
    return pd.DataFrame(rows)


def fit_evidence_model(
    train_dataset: EvidenceCandidateDataset,
    validation_dataset: EvidenceCandidateDataset,
    encoded_events: EncodedEvidenceEvents,
    vocabulary: VocabularyBundle,
    *,
    architecture: str,
    mode: str,
    residual_shrinkage_lambda: float,
    seed: int,
    epochs: int = 20,
    patience: int = 4,
    batch_size: int = 128,
    max_events: int = 64,
    device: str = "cpu",
) -> EvidenceModelFit:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch_device = torch.device(device)
    model = EventQualityResidualModel(
        [len(vocabulary.values[field]) for field in (vocabulary.fields or [])],
        architecture=architecture,
        mode=mode,
    ).to(torch_device)
    if mode == "residual":
        initial = predict_evidence_model(
            model,
            validation_dataset,
            encoded_events,
            max_events=max_events,
            batch_size=batch_size,
            device=device,
        )
        initialization_error = float(
            np.max(np.abs(initial.raw_logit - initial.count_logit))
        )
        if initialization_error > FALLBACK_ATOL:
            raise RuntimeError("Evidence residual zero initialization is not exact")
    else:
        initialization_error = math.nan
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        collate_fn=lambda rows: collate_evidence_candidates(rows, encoded_events, max_events),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        for raw_batch in train_loader:
            batch = _move(raw_batch, torch_device)
            result = model(batch)
            loss = binomial_nll(result["logit"], batch["replicate_k"], batch["replicate_n"])
            if mode == "residual":
                loss = loss + float(residual_shrinkage_lambda) * result[
                    "event_quality_residual_logit"
                ].square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = predict_evidence_model(
            model,
            validation_dataset,
            encoded_events,
            max_events=max_events,
            batch_size=batch_size,
            device=device,
        )
        metric = binary_metrics(
            validation.binary_label, validation.proxy_positive_probability
        )
        score = float(metric["auprc"]) if np.isfinite(metric["auprc"]) else -math.inf
        history.append(
            {
                "epoch": epoch,
                "training_binomial_nll": float(np.mean(losses)),
                **{f"validation_{key}": float(value) for key, value in metric.items()},
            }
        )
        if score > best_score + 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("Evidence model produced no checkpoint")
    model.load_state_dict(best_state)
    validation = predict_evidence_model(
        model,
        validation_dataset,
        encoded_events,
        max_events=max_events,
        batch_size=batch_size,
        device=device,
    )
    metric = binary_metrics(validation.binary_label, validation.proxy_positive_probability)
    probability = validation.proxy_positive_probability.to_numpy(float)
    logit = validation.raw_logit.to_numpy(float)
    summary = {
        "probability_mean": float(np.mean(probability)),
        "probability_sd": float(np.std(probability)),
        **{
            f"probability_p{int(q * 100):02d}": float(np.quantile(probability, q))
            for q in (0.01, 0.05, 0.5, 0.95, 0.99)
        },
        "logit_mean": float(np.mean(logit)),
        "logit_sd": float(np.std(logit)),
    }
    return EvidenceModelFit(
        model=model,
        best_state=best_state,
        history=history,
        validation_metrics={key: float(value) for key, value in metric.items()},
        initialization_max_abs_error=initialization_error,
        prediction_summary=summary,
    )
