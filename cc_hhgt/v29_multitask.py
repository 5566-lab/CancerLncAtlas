from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from .calibration import fit_temperature
from .common import (
    LOGGER,
    available_device,
    file_sha256,
    read_table,
    seed_everything,
    write_json,
    write_table,
)
from .gnn import (
    amp_autocast,
    amp_grad_scaler,
    build_model,
    candidate_tensors,
    export_node_embeddings,
    load_graph_bundle,
    move_graph,
    nnpu_loss,
    require_torch_geometric,
    runtime_bundle_for_step,
)
from .episodic_loco import EpisodicLOCOPlanner
from .metrics import proxy_binary_metrics
from .prediction_contract import candidate_universe_sha256, metric_scope_for_split
from .training_data import _pu_target_counts, split_for_fold
from .training_resume import (
    CHECKPOINT_FORMAT,
    capture_rng_state,
    restore_rng_state,
    training_resume_contract_sha256,
    training_runtime_fingerprint,
    validate_training_resume_checkpoint,
)
from .v30_integrity import atomic_write_json
from .v31_residual import FALLBACK_ATOL, attach_residual_base


STATE_MODELS = ("rgcn", "hgt", "cc_hhgt")


def _atomic_write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a table beside its destination and atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(torch, payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class StateDecoder:
    """Factory wrapper to avoid importing torch at module import time."""

    @staticmethod
    def build(
        hidden: int,
        dropout: float,
        state_ids: list[str],
        *,
        zero_initialize_membership: bool = False,
    ):
        torch = require_torch_geometric()
        from torch import nn

        if not state_ids:
            raise ValueError("At least one state-specific head is required")

        class _Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.membership = nn.Sequential(
                    nn.Linear(hidden * 5, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden // 2),
                    nn.GELU(),
                    nn.Linear(hidden // 2, 1),
                )
                self.direction = nn.Sequential(
                    nn.Linear(hidden * 5, hidden // 2),
                    nn.GELU(),
                    nn.Linear(hidden // 2, 1),
                )
                if zero_initialize_membership:
                    nn.init.zeros_(self.membership[-1].weight)
                    nn.init.zeros_(self.membership[-1].bias)

            def forward(self, z):
                return self.membership(z).squeeze(-1), self.direction(z).squeeze(-1)

        class _Decoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.state_ids = tuple(map(str, state_ids))
                self.heads = nn.ModuleList([_Head() for _ in self.state_ids])

            def forward(self, lnc, state, cancer, head_index):
                z = torch.cat(
                    [lnc, state, cancer, lnc * state, torch.abs(lnc - state)], dim=-1
                )
                # LayerNorm may promote head outputs to float32 under CUDA
                # autocast even when the concatenated representation is
                # float16.  Accumulate with differentiable out-of-place
                # index_copy in a stable dtype.
                membership = torch.zeros(len(z), dtype=torch.float32, device=z.device)
                direction = torch.zeros(len(z), dtype=torch.float32, device=z.device)
                assigned = torch.zeros(len(z), dtype=torch.bool, device=z.device)
                for index, head in enumerate(self.heads):
                    mask = head_index.eq(index)
                    if mask.any():
                        positions = torch.nonzero(mask, as_tuple=False).flatten()
                        head_membership, head_direction = head(z[mask])
                        membership = membership.index_copy(
                            0, positions, head_membership.float()
                        )
                        direction = direction.index_copy(
                            0, positions, head_direction.float()
                        )
                        assigned |= mask
                if not assigned.all():
                    raise RuntimeError("State batch contains an unmapped independent-head index")
                return membership, direction

        return _Decoder()


@dataclass
class StateBatch:
    frame: pd.DataFrame
    l: Any
    s: Any
    c: Any
    y: Any
    weak: Any
    direction: Any
    head: Any
    base_logit: Any | None = None


def _load_state_candidates(cfg: dict[str, Any]) -> pd.DataFrame:
    root = cfg["_results"] / "tables" / "strict_state_candidate"
    if not root.exists():
        raise FileNotFoundError(
            f"{root} missing. Run scripts/41_prepare_v29_state_graph.py first."
        )
    frame = read_table(root)
    required = {
        "candidate_id",
        "cancer_id",
        "lncrna_id",
        "state_id",
        "proxy_label",
        "label_class",
        "direction_label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"strict state candidates missing {sorted(missing)}")
    return frame


def _stratified_sample(
    frame: pd.DataFrame,
    n: int,
    seed: int,
    group_columns: tuple[str, ...] = ("cancer_id", "state_id"),
) -> pd.DataFrame:
    """Return an exact, deterministic sample while retaining small strata."""
    if n <= 0:
        return frame.iloc[:0].copy()
    if len(frame) <= n:
        return frame.copy()
    columns = [column for column in group_columns if column in frame.columns]
    if not columns:
        return frame.sample(n=n, random_state=seed).copy()

    groups = [group for _, group in frame.groupby(columns, sort=True, dropna=False)]
    sizes = np.asarray([len(group) for group in groups], dtype=np.int64)
    # When the budget permits, reserve one observation for every cancer/state
    # stratum, then distribute the rest proportionally to residual capacity.
    quotas = np.ones(len(groups), dtype=np.int64) if n >= len(groups) else np.zeros(len(groups), dtype=np.int64)
    remaining = int(n - quotas.sum())
    capacity = sizes - quotas
    if remaining > 0 and int(capacity.sum()) > 0:
        raw = remaining * capacity.astype(float) / float(capacity.sum())
        addition = np.floor(raw).astype(np.int64)
        addition = np.minimum(addition, capacity)
        quotas += addition
        remaining = int(n - quotas.sum())
        if remaining > 0:
            order = np.argsort(-(raw - addition), kind="stable")
            for index in order:
                if remaining == 0:
                    break
                if quotas[index] < sizes[index]:
                    quotas[index] += 1
                    remaining -= 1
    if int(quotas.sum()) != n:
        raise RuntimeError(f"Unable to allocate exact stratified sample: requested={n}, allocated={int(quotas.sum())}")

    sampled = [
        group.sample(n=int(quota), random_state=seed + offset).copy()
        for offset, (group, quota) in enumerate(zip(groups, quotas, strict=True))
        if quota > 0
    ]
    return pd.concat(sampled, ignore_index=True)


def _sample_train(
    frame: pd.DataFrame,
    cap: int,
    seed: int,
    unlabeled_to_positive_ratio: float = 4.0,
) -> pd.DataFrame:
    """Cap a PU training table without allowing positives to evict unlabeled rows."""
    if cap < 2:
        raise ValueError(f"Training cap must be at least 2 for PU learning, observed {cap}")
    if unlabeled_to_positive_ratio <= 0:
        raise ValueError("unlabeled_to_positive_ratio must be positive")
    positive = frame.loc[frame.proxy_label.astype(float).gt(0.5)]
    unlabeled = frame.loc[frame.proxy_label.astype(float).le(0.5)]
    if positive.empty or unlabeled.empty:
        raise RuntimeError(
            "State PU training requires both positive and unlabeled rows before sampling: "
            f"positive={len(positive)}, unlabeled={len(unlabeled)}"
        )
    if len(frame) <= cap:
        return frame.sample(frac=1, random_state=seed).reset_index(drop=True)

    keep_positive, keep_unlabeled = _pu_target_counts(
        len(positive), len(unlabeled), cap, unlabeled_to_positive_ratio
    )

    positive = _stratified_sample(positive, keep_positive, seed)
    unlabeled = _stratified_sample(unlabeled, keep_unlabeled, seed + 100_003)
    sampled = pd.concat([positive, unlabeled], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    sampled_positive = int(sampled.proxy_label.astype(float).gt(0.5).sum())
    if sampled_positive == 0 or sampled_positive == len(sampled):
        raise RuntimeError(
            "State PU sampler produced a single-class table: "
            f"positive={sampled_positive}, unlabeled={len(sampled) - sampled_positive}"
        )
    return sampled


def _state_candidates_in_graph(frame: pd.DataFrame, bundle) -> pd.DataFrame:
    valid = (
        frame.lncrna_id.astype(str).isin(bundle.node_maps.get("lncRNA", {}))
        & frame.state_id.astype(str).isin(bundle.node_maps.get("state", {}))
        & frame.cancer_id.astype(str).isin(bundle.node_maps.get("cancer", {}))
    )
    return frame.loc[valid].reset_index(drop=True)


def split_state_for_fold(
    cfg: dict[str, Any], fold_row: pd.Series, seed: int, bundle, state_ids: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = _load_state_candidates(cfg)
    # Graph membership is part of the candidate universe. Filter before PU
    # sampling so the configured class ratio and row cap remain true for the
    # tensors actually consumed by the model.
    frame = _state_candidates_in_graph(frame, bundle)
    if state_ids:
        frame = frame.loc[frame.state_id.astype(str).isin(set(map(str, state_ids)))].copy()
    train_cancers = set(str(fold_row.train_cancers).split(";"))
    train = frame.loc[frame.cancer_id.astype(str).isin(train_cancers)].copy()
    val = frame.loc[frame.cancer_id.astype(str).eq(str(fold_row.validation_cancer))].copy()
    test = frame.loc[frame.cancer_id.astype(str).eq(str(fold_row.test_cancer))].copy()
    if "evaluation_eligibility" in frame:
        val = val.loc[val.evaluation_eligibility.astype(str).eq("ELIGIBLE")].copy()
        test = test.loc[test.evaluation_eligibility.astype(str).eq("ELIGIBLE")].copy()
    state_cfg = cfg.get("state_training", {})
    cap = int(state_cfg.get("max_training_pairs_per_fold", 150000))
    ratio = float(state_cfg.get("unlabeled_to_positive_ratio", 4.0))
    train = _sample_train(train, cap, seed, ratio)
    return train, val.reset_index(drop=True), test.reset_index(drop=True)


def state_candidate_tensors(
    frame: pd.DataFrame,
    bundle,
    device: str,
    kind: str,
    head_map: dict[str, int] | None = None,
) -> StateBatch:
    torch = require_torch_geometric()
    mapped = _state_candidates_in_graph(frame, bundle)
    if len(mapped) != len(frame):
        raise RuntimeError(
            "State candidates were not filtered to the graph before tensorization: "
            f"input={len(frame)}, mappable={len(mapped)}"
        )
    frame = mapped
    if kind == "rgcn":
        mapping = bundle.homogeneous["global_map"]
        l = [mapping[("lncRNA", str(x))] for x in frame.lncrna_id]
        s = [mapping[("state", str(x))] for x in frame.state_id]
        c = [mapping[("cancer", str(x))] for x in frame.cancer_id]
    else:
        l = [bundle.node_maps["lncRNA"][str(x)] for x in frame.lncrna_id]
        s = [bundle.node_maps["state"][str(x)] for x in frame.state_id]
        c = [bundle.node_maps["cancer"][str(x)] for x in frame.cancer_id]
    head_map = head_map or {
        state_id: index for index, state_id in enumerate(sorted(frame.state_id.astype(str).unique()))
    }
    unknown_heads = sorted(set(frame.state_id.astype(str)) - set(head_map))
    if unknown_heads:
        raise RuntimeError(f"State candidates contain unmapped independent heads: {unknown_heads}")
    base_logit = None
    if "z_base" in frame:
        values = pd.to_numeric(frame.z_base, errors="coerce")
        if values.isna().any() or (~np.isfinite(values.to_numpy(float))).any():
            raise RuntimeError("State residual base contains non-finite logits")
        base_logit = torch.tensor(
            values.to_numpy(np.float32), dtype=torch.float32, device=device
        )
    return StateBatch(
        frame=frame,
        l=torch.tensor(l, dtype=torch.long, device=device),
        s=torch.tensor(s, dtype=torch.long, device=device),
        c=torch.tensor(c, dtype=torch.long, device=device),
        y=torch.tensor(frame.proxy_label.to_numpy(np.float32), dtype=torch.float32, device=device),
        weak=torch.tensor(frame.label_class.eq("weak_positive").to_numpy(), dtype=torch.bool, device=device),
        direction=torch.tensor(frame.direction_label.fillna(-1).to_numpy(np.float32), dtype=torch.float32, device=device),
        head=torch.tensor(
            frame.state_id.astype(str).map(head_map).to_numpy(np.int64), dtype=torch.long, device=device
        ),
        base_logit=base_logit,
    )


def _encoded_triplet(encoded, batch, kind: str, target: str):
    if kind == "rgcn":
        left = encoded[batch.l]
        middle = encoded[batch.s if target == "state" else batch.p]
        cancer = encoded[batch.c]
        return left, middle, cancer
    left = encoded["lncRNA"][batch.l]
    middle = encoded[target][batch.s] if target == "state" else encoded["pathway_family"][batch.p]
    cancer = encoded["cancer"][batch.c]
    return left, middle, cancer


def _pair_masked_batch(batch: dict[str, Any], *, zero_all: bool = False, mask_probability: float = 0.0) -> dict[str, Any]:
    """Return a shallow batch copy with leakage-safe pair features.

    Validation and test target-cancer pair evidence is always fully zeroed.
    During training, complete rows are randomly masked so CC-HHGT cannot rely
    exclusively on evidence features and must learn the graph representation.
    """
    import torch
    copied = dict(batch)
    x = batch["x"]
    if zero_all:
        copied["x"] = x.new_zeros(x.shape)
        return copied
    if mask_probability > 0 and x.numel() > 0:
        row_mask = torch.rand(x.shape[0], device=x.device) < float(mask_probability)
        masked = x.clone()
        masked[row_mask] = 0
        copied["x"] = masked
    return copied


def _pathway_forward(model, encoded, batch, kind: str):
    if kind == "rgcn":
        residual, direction = model.decoder(
            encoded[batch["l"]], encoded[batch["p"]], encoded[batch["c"]], batch["x"]
        )
    else:
        target_node_type = getattr(model, "pathway_target_node_type", "pathway_family")
        residual, direction = model.decoder(
            encoded["lncRNA"][batch["l"]],
            encoded[target_node_type][batch["p"]],
            encoded["cancer"][batch["c"]],
            batch["x"],
        )
    final = residual + batch["base_logit"] if "base_logit" in batch else residual
    return final, direction, residual


def _state_forward(decoder, encoded, batch: StateBatch, kind: str):
    if kind == "rgcn":
        residual, direction = decoder(
            encoded[batch.l], encoded[batch.s], encoded[batch.c], batch.head
        )
    else:
        residual, direction = decoder(
            encoded["lncRNA"][batch.l],
            encoded["state"][batch.s],
            encoded["cancer"][batch.c],
            batch.head,
        )
    final = residual + batch.base_logit if batch.base_logit is not None else residual
    return final, direction, residual


def _detach_encoded_for_microbatch_accumulation(encoded):
    """Create leaf embeddings for decoder-side gradient accumulation.

    Each candidate microbatch still contributes to the optimizer step, while
    the shared full-graph encoder is traversed backward only once.
    """
    if isinstance(encoded, dict):
        return {
            node_type: value.detach().requires_grad_(True)
            for node_type, value in encoded.items()
        }
    return encoded.detach().requires_grad_(True)


def _backward_encoder_once(torch, encoded, accumulated_encoded) -> int:
    """Propagate accumulated embedding gradients through the encoder once."""
    if isinstance(encoded, dict):
        pairs = [
            (original, accumulated_encoded[node_type].grad)
            for node_type, original in encoded.items()
        ]
    else:
        pairs = [(encoded, accumulated_encoded.grad)]
    usable = [(original, gradient) for original, gradient in pairs if gradient is not None]
    if not usable:
        raise RuntimeError("No accumulated candidate gradient reached encoder embeddings")
    torch.autograd.backward(
        [original for original, _ in usable],
        grad_tensors=[gradient for _, gradient in usable],
    )
    return 1


def _checkpoint_patience_improved(score: float, anchor: float, min_delta: float) -> bool:
    """Return whether a score gain is large enough to reset patience."""
    tolerance = np.finfo(float).eps * max(1.0, abs(score), abs(anchor)) * 8
    return bool(
        np.isfinite(score)
        and (
            not np.isfinite(anchor)
            or (
                score > anchor
                and score - anchor + tolerance >= min_delta
            )
        )
    )


def candidate_microbatch_indices(labels: np.ndarray, microbatch_size: int, seed: int) -> list[np.ndarray]:
    """Create deterministic, class-mixed candidate microbatches with exact coverage."""
    labels = np.asarray(labels, dtype=float)
    if microbatch_size <= 0:
        raise ValueError("candidate microbatch size must be positive")
    count = len(labels)
    if count == 0:
        return []
    n_batches = int(math.ceil(count / microbatch_size))
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels > 0.5)
    unlabeled = np.flatnonzero(labels <= 0.5)
    rng.shuffle(positive)
    rng.shuffle(unlabeled)
    batches: list[np.ndarray] = []
    for positive_part, unlabeled_part in zip(
        np.array_split(positive, n_batches), np.array_split(unlabeled, n_batches), strict=True
    ):
        batch = np.concatenate([positive_part, unlabeled_part]).astype(np.int64, copy=False)
        rng.shuffle(batch)
        if len(batch):
            batches.append(batch)
    coverage = np.concatenate(batches) if batches else np.empty(0, dtype=np.int64)
    if len(coverage) != count or len(np.unique(coverage)) != count:
        raise RuntimeError("Candidate microbatches do not cover every row exactly once")
    return batches


def _slice_pathway_batch(batch: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    torch = require_torch_geometric()
    index = torch.as_tensor(indices, dtype=torch.long, device=batch["y"].device)
    return {key: value.index_select(0, index) for key, value in batch.items()}


def _slice_state_batch(batch: StateBatch, indices: np.ndarray) -> StateBatch:
    torch = require_torch_geometric()
    index = torch.as_tensor(indices, dtype=torch.long, device=batch.y.device)
    return StateBatch(
        frame=batch.frame.iloc[indices].reset_index(drop=True),
        l=batch.l.index_select(0, index),
        s=batch.s.index_select(0, index),
        c=batch.c.index_select(0, index),
        y=batch.y.index_select(0, index),
        weak=batch.weak.index_select(0, index),
        direction=batch.direction.index_select(0, index),
        head=batch.head.index_select(0, index),
        base_logit=(
            batch.base_logit.index_select(0, index)
            if batch.base_logit is not None
            else None
        ),
    )


def state_equal_macro_metrics(
    frame: pd.DataFrame, probability: np.ndarray, bins: int
) -> tuple[float, dict[str, Any]]:
    """State-equal checkpoint metric: mean(macro AUROC, macro AUPRC lift)."""
    rows: list[dict[str, Any]] = []
    probability = np.asarray(probability, dtype=float)
    for state_id, positions in frame.groupby("state_id", observed=True).indices.items():
        index = np.asarray(positions, dtype=np.int64)
        metrics = proxy_binary_metrics(frame.iloc[index], probability[index], bins)
        positive_rate = float(metrics.get("positive_rate", math.nan))
        auprc = float(metrics.get("auprc", math.nan))
        rows.append(
            {
                "state_id": str(state_id),
                **metrics,
                "auprc_lift": auprc - positive_rate
                if np.isfinite(auprc) and np.isfinite(positive_rate)
                else math.nan,
            }
        )
    per_state = pd.DataFrame(rows)
    macro_auroc = float(per_state.auroc.mean()) if len(per_state) else math.nan
    macro_auprc_lift = float(per_state.auprc_lift.mean()) if len(per_state) else math.nan
    score = (
        float(np.mean([macro_auroc, macro_auprc_lift]))
        if np.isfinite(macro_auroc) and np.isfinite(macro_auprc_lift)
        else -math.inf
    )
    details: dict[str, Any] = {
        "state_equal_macro_auroc": macro_auroc,
        "state_equal_macro_auprc_lift": macro_auprc_lift,
        "state_equal_checkpoint_score": score,
        "state_metric_count": int(len(per_state)),
    }
    for row in per_state.to_dict("records"):
        prefix = str(row.pop("state_id"))
        for key, value in row.items():
            details[f"state::{prefix}::{key}"] = value
    return score, details


def _metric_score(
    path_frame: pd.DataFrame,
    path_prob: np.ndarray,
    state_frame: pd.DataFrame,
    state_prob: np.ndarray,
    bins: int,
    pathway_weight: float = 0.65,
    state_weight: float = 0.35,
) -> tuple[float, dict[str, Any]]:
    path_metrics = proxy_binary_metrics(path_frame, path_prob, bins)
    state_metrics = proxy_binary_metrics(state_frame, state_prob, bins)
    p = path_metrics.get("auprc", math.nan)
    s = state_metrics.get("auprc", math.nan)
    weighted = []
    if np.isfinite(p):
        weighted.append((float(pathway_weight), float(p)))
    if np.isfinite(s):
        weighted.append((float(state_weight), float(s)))
    denominator = sum(weight for weight, _ in weighted)
    score = float(sum(weight * value for weight, value in weighted) / denominator) if denominator > 0 else -math.inf
    return score, {f"pathway_{k}": v for k, v in path_metrics.items()} | {f"state_{k}": v for k, v in state_metrics.items()}


def _export_embedding_parquets(
    cfg: dict[str, Any],
    model,
    graph,
    bundle,
    kind: str,
    model_dir: Path,
    *,
    encoded=None,
) -> None:
    torch = require_torch_geometric()
    if encoded is None:
        model.eval()
        with torch.no_grad():
            encoded = model.encode(graph)
    out_dir = model_dir / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "rgcn":
        positioned = bundle.nodes.assign(_global_position=np.arange(len(bundle.nodes)))
        for node_type, group in positioned.groupby("node_type", sort=False):
            group = group.sort_values("node_index_within_type").reset_index(drop=True)
            indices = torch.tensor(group._global_position.to_numpy(np.int64), device=encoded.device)
            values = encoded.index_select(0, indices).detach().cpu().numpy().astype(np.float16)
            frame = group[["canonical_id", "node_id", "node_index_within_type"]].copy()
            for j in range(values.shape[1]):
                frame[f"embedding_{j:03d}"] = values[:, j]
            write_table(frame, out_dir / f"{node_type}.parquet")
    else:
        for node_type, values in encoded.items():
            group = bundle.nodes.loc[bundle.nodes.node_type.astype(str).eq(str(node_type))].sort_values("node_index_within_type").reset_index(drop=True)
            array = values.detach().cpu().numpy().astype(np.float16)
            frame = group[["canonical_id", "node_id", "node_index_within_type"]].copy()
            for j in range(array.shape[1]):
                frame[f"embedding_{j:03d}"] = array[:, j]
            write_table(frame, out_dir / f"{node_type}.parquet")


def train_multitask_fold(
    cfg: dict[str, Any],
    fold_row: pd.Series,
    kind: str,
    seed: int,
    _bundle: Any | None = None,
    _residual_base: dict[str, dict[str, pd.DataFrame] | pd.DataFrame] | None = None,
) -> dict[str, Any]:
    torch = require_torch_geometric()
    import torch.nn.functional as F

    if kind not in STATE_MODELS:
        raise ValueError(f"Unsupported model {kind}")
    residual_cfg = cfg.get("residual_learning", {})
    residual_enabled = bool(residual_cfg.get("enabled", False))
    residual_lambda = float(residual_cfg.get("shrinkage_lambda", 0.0))
    candidate_sampling_seed = int(
        residual_cfg.get("candidate_sampling_seed", seed)
        if residual_enabled
        else seed
    )
    if residual_lambda < 0:
        raise ValueError("residual_learning.shrinkage_lambda must be nonnegative")
    if residual_enabled and kind != "cc_hhgt":
        raise RuntimeError("V3.1 Graph residual is restricted to the primary CC-HHGT architecture")
    if residual_enabled and _residual_base is None:
        raise RuntimeError("V3.1 Graph residual requires a frozen BestSimpleLOCO base")
    seed_everything(seed)
    device = available_device(cfg["training"].get("device", "auto"))
    if device == "unavailable":
        raise RuntimeError("PyTorch is unavailable")
    # Reference-only cancers are handled by the registered graph contract and
    # must not be mixed into the held-out set.  This separation is required by
    # the shared pseudoheldout/real-test parity check.
    excluded = {str(fold_row.test_cancer), str(fold_row.validation_cancer)}
    bundle = _bundle if _bundle is not None else load_graph_bundle(cfg, seed, excluded_cancers=excluded)
    state_head_ids = list(
        map(
            str,
            cfg.get("sample_contract", {}).get("target_states")
            or list(cfg.get("state_heads", {}).get("targets", {}).keys())
            or cfg.get("state_graph", {}).get("required_states", []),
        )
    )
    required_states = set(state_head_ids)
    observed_states = set(bundle.node_maps.get("state", {}))
    missing_states = sorted(required_states - observed_states)
    if missing_states or not observed_states:
        raise RuntimeError(f"V2.9 strict graph missing state nodes: {missing_states or 'all state nodes'}")
    state_edge_count = int(
        ((bundle.edges.source_type.astype(str).eq("state")) | (bundle.edges.target_type.astype(str).eq("state"))).sum()
    )
    if state_edge_count == 0:
        raise RuntimeError("V2.9 strict graph contains state nodes but zero state edges")
    graph = move_graph(bundle, device, kind)

    data_fold_row = fold_row.copy()
    data_fold_row["split_seed"] = candidate_sampling_seed
    pathway_train, pathway_val, pathway_test = split_for_fold(cfg, data_fold_row)
    train_cancers = set(map(str, str(fold_row.train_cancers).split(";")))

    def residual_base_for(task: str, split: str) -> pd.DataFrame:
        if _residual_base is None or task not in _residual_base:
            raise RuntimeError(f"Missing {task} residual base")
        registered = _residual_base[task]
        if isinstance(registered, dict):
            if split not in registered:
                raise RuntimeError(f"Missing {task}/{split} residual base")
            return registered[split]
        return registered

    if residual_enabled:
        pathway_train = attach_residual_base(
            pathway_train,
            residual_base_for("pathway", "train"),
            split="train",
            train_cancers=train_cancers,
        )
        pathway_val = attach_residual_base(
            pathway_val,
            residual_base_for("pathway", "val"),
            split="val",
            train_cancers=train_cancers,
        )
        pathway_test = attach_residual_base(
            pathway_test,
            residual_base_for("pathway", "test"),
            split="test",
            train_cancers=train_cancers,
        )
    pathway_sampling_audit: dict[str, Any] = {}
    for split, frame in [("train", pathway_train), ("val", pathway_val), ("test", pathway_test)]:
        positive = int(frame.proxy_label.astype(float).gt(0.5).sum())
        unlabeled = int(len(frame) - positive)
        if frame.empty or positive == 0 or unlabeled == 0:
            raise RuntimeError(
                f"Pathway {split} set must contain both proxy classes: "
                f"rows={len(frame)}, positive={positive}, unlabeled={unlabeled}"
            )
        direction_leak = int(
            (frame.proxy_label.astype(float).le(0.5) & frame.direction_label.notna()).sum()
        )
        if direction_leak:
            raise RuntimeError(
                f"Pathway {split} uses {direction_leak} unlabeled rows in direction supervision"
            )
        pathway_sampling_audit[f"pathway_{split}_rows"] = len(frame)
        pathway_sampling_audit[f"pathway_{split}_positive"] = positive
        pathway_sampling_audit[f"pathway_{split}_unlabeled"] = unlabeled
        pathway_sampling_audit[f"pathway_{split}_positive_rate"] = positive / len(frame)
        if split != "train":
            policies = sorted(frame.evaluation_sampling_policy.astype(str).unique())
            if not set(policies).issubset({"full_universe", "label_independent_deterministic"}):
                raise RuntimeError(f"Invalid pathway {split} evaluation policy: {policies}")
            pathway_sampling_audit[f"pathway_{split}_evaluation_policy"] = policies
            pathway_sampling_audit[f"pathway_{split}_universe_rows"] = int(
                frame.evaluation_universe_rows.max()
            )
    features = [x for x in cfg["training"]["feature_columns"] if x in pathway_train.columns]
    pathway_frames: dict[str, pd.DataFrame] = {}
    pathway_batches: dict[str, Any] = {}
    for split, frame in [("train", pathway_train), ("val", pathway_val), ("test", pathway_test)]:
        pathway_frames[split], pathway_batches[split] = candidate_tensors(frame, bundle, features, device, kind)
        if pathway_frames[split].empty:
            raise RuntimeError(f"No pathway candidates map to V2.9 graph: {fold_row.fold_id}/{split}")

    strict_pair_mask_probability = float(cfg.get("state_training", {}).get("strict_pair_evidence_mask_probability", 0.50))
    mask_pair_evidence_for_all = bool(
        cfg.get("state_training", {}).get(
            "mask_pair_evidence_for_all_pathway_models", False
        )
    )
    zero_evaluation_pair_evidence = mask_pair_evidence_for_all or kind == "cc_hhgt"
    pathway_target_batches = {
        "train": pathway_batches["train"],
        "val": _pair_masked_batch(
            pathway_batches["val"], zero_all=zero_evaluation_pair_evidence
        ),
        "test": _pair_masked_batch(
            pathway_batches["test"], zero_all=zero_evaluation_pair_evidence
        ),
    }

    state_train, state_val, state_test = split_state_for_fold(
        cfg,
        data_fold_row,
        candidate_sampling_seed,
        bundle,
        state_ids=state_head_ids,
    )
    if residual_enabled:
        state_train = attach_residual_base(
            state_train,
            residual_base_for("state", "train"),
            split="train",
            train_cancers=train_cancers,
        )
        state_val = attach_residual_base(
            state_val,
            residual_base_for("state", "val"),
            split="val",
            train_cancers=train_cancers,
        )
        state_test = attach_residual_base(
            state_test,
            residual_base_for("state", "test"),
            split="test",
            train_cancers=train_cancers,
        )
    state_head_map = {state_id: index for index, state_id in enumerate(state_head_ids)}
    state_batches = {
        "train": state_candidate_tensors(state_train, bundle, device, kind, state_head_map),
        "val": state_candidate_tensors(state_val, bundle, device, kind, state_head_map),
        "test": state_candidate_tensors(state_test, bundle, device, kind, state_head_map),
    }
    if any(batch.frame.empty for batch in state_batches.values()):
        raise RuntimeError(f"No state candidates map to V2.9 graph for {fold_row.fold_id}")
    state_train_positive = int(state_batches["train"].frame.proxy_label.astype(float).gt(0.5).sum())
    state_train_unlabeled = int(len(state_batches["train"].frame) - state_train_positive)
    if state_train_positive == 0 or state_train_unlabeled == 0:
        raise RuntimeError(
            "State PU training requires both classes after graph mapping: "
            f"positive={state_train_positive}, unlabeled={state_train_unlabeled}"
        )
    state_unlabeled_direction_used = int(
        (
            state_batches["train"].frame.proxy_label.astype(float).le(0.5)
            & state_batches["train"].frame.direction_label.notna()
        ).sum()
    )
    if state_unlabeled_direction_used:
        raise RuntimeError(
            "State direction supervision includes unlabeled rows: "
            f"count={state_unlabeled_direction_used}. Rebuild strict state candidates before training."
        )
    state_evaluation_audit: dict[str, Any] = {}
    for split in ("val", "test"):
        state_frame = state_batches[split].frame
        policies = sorted(state_frame.evaluation_sampling_policy.astype(str).unique())
        if policies != ["full_detectable_lncrna_state_universe"]:
            raise RuntimeError(f"State {split} is not the full detectable evaluation universe: {policies}")
        state_evaluation_audit[f"state_{split}_evaluation_policy"] = policies
        state_evaluation_audit[f"state_{split}_rows"] = len(state_frame)

    episodic_enabled = bool(
        cfg.get("graph_contract", {}).get("episodic_pseudoheldout", False)
    )
    runtime_enabled = bundle.runtime_schedule is not None
    if episodic_enabled and not runtime_enabled:
        raise RuntimeError(
            "Episodic pseudoheldout LOCO requires the non-destructive runtime graph scheduler"
        )
    episode_planner = None
    formal_heldout = {str(fold_row.test_cancer), str(fold_row.validation_cancer)}
    reference_only = set(map(str, cfg.get("cancer_scope", {}).get("reference_only", [])))
    reference_only.update(map(str, cfg.get("analysis_cancers", {}).get("reference_only", [])))
    if episodic_enabled:
        pathway_cancers = set(pathway_frames["train"].cancer_id.astype(str))
        state_cancers = set(state_batches["train"].frame.cancer_id.astype(str))
        episode_cancers = sorted((pathway_cancers & state_cancers) - formal_heldout - reference_only)
        episode_planner = EpisodicLOCOPlanner(
            episode_cancers,
            formal_heldout,
            seed=seed,
        )

    model = build_model(kind, bundle, len(features), cfg).to(device)
    hidden = int(cfg["training"]["hidden_channels"])
    dropout = float(cfg["training"].get("dropout", 0.2))
    state_decoder = StateDecoder.build(
        hidden,
        dropout,
        state_head_ids,
        zero_initialize_membership=residual_enabled,
    ).to(device)
    residual_initialization_max_abs = 0.0
    if residual_enabled:
        residual_parameters = [
            model.decoder.membership[-1].weight,
            model.decoder.membership[-1].bias,
            *[
                parameter
                for head in state_decoder.heads
                for parameter in (head.membership[-1].weight, head.membership[-1].bias)
            ],
        ]
        residual_initialization_max_abs = max(
            float(parameter.detach().abs().max().cpu())
            for parameter in residual_parameters
        )
        if residual_initialization_max_abs > FALLBACK_ATOL:
            raise RuntimeError(
                "Graph residual head is not exactly zero initialized: "
                f"max_abs={residual_initialization_max_abs}"
            )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(state_decoder.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    amp_enabled = bool(cfg["training"].get("mixed_precision", False) and torch.device(device).type == "cuda")
    scaler = amp_grad_scaler(torch, amp_enabled)
    state_cfg = cfg.get("state_training", {})
    state_loss_weight = float(state_cfg.get("auxiliary_loss_weight", 0.35))
    if episodic_enabled and not np.isclose(state_loss_weight, 0.50, atol=0.0, rtol=0.0):
        raise RuntimeError(
            f"V3.1 Fixed Graph locks state auxiliary loss weight at 0.50, observed {state_loss_weight}"
        )
    state_direction_weight = float(state_cfg.get("direction_loss_weight", 0.15))
    best_pathway_score = -math.inf
    best_state_score = -math.inf
    best_pathway_epoch = 0
    best_state_epoch = 0
    pathway_patience_score = -math.inf
    state_patience_score = -math.inf
    pathway_patience_epoch = 0
    state_patience_epoch = 0
    runtime_validation_cycles = 0
    pathway_stale_validation_cycles = 0
    state_stale_validation_cycles = 0
    checkpoint_patience_cycles = int(
        cfg.get("runtime_graph_sampling", {}).get(
            "checkpoint_patience_cycles", 3
        )
    )
    if checkpoint_patience_cycles < 1:
        raise ValueError("runtime_graph_sampling.checkpoint_patience_cycles must be positive")
    checkpoint_min_delta = float(cfg["training"].get("checkpoint_min_delta", 0.0))
    if checkpoint_min_delta < 0:
        raise ValueError("training.checkpoint_min_delta must be non-negative")
    optimizer_steps = 0
    candidate_microbatch_size = int(
        cfg["training"].get(
            "candidate_microbatch_size", cfg["training"].get("batch_size", 8192)
        )
    )
    history: list[dict[str, Any]] = []
    output_model_name = kind
    strict_output_root = Path(cfg.get("_strict_output_root", cfg["_results"] / "v2_9_strict"))
    model_dir = strict_output_root / output_model_name / str(fold_row.fold_id) / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    last_training_state_path = model_dir / "last_training_state.pt"
    resume_contract_sha256 = training_resume_contract_sha256(
        cfg,
        model_name=output_model_name,
        fold_id=str(fold_row.fold_id),
        seed=seed,
    )
    resume_runtime_fingerprint = training_runtime_fingerprint(torch, device)
    resume_checkpoint_interval = int(
        cfg["training"].get("resume_checkpoint_every_epochs", 5)
    )
    if resume_checkpoint_interval < 1:
        raise ValueError("training.resume_checkpoint_every_epochs must be positive")
    resumed_from_epoch = 0
    resume_count = 0
    start_epoch = 1
    resume_stop_requested = False
    resume_checkpoint_raw = cfg.get("_resume_training_checkpoint")
    if resume_checkpoint_raw:
        resume_checkpoint_path = Path(resume_checkpoint_raw).resolve()
        if resume_checkpoint_path != last_training_state_path.resolve():
            raise RuntimeError(
                "Exact resume checkpoint must be the task-local last_training_state.pt: "
                f"expected={last_training_state_path.resolve()}, observed={resume_checkpoint_path}"
            )
        checkpoint = torch.load(
            resume_checkpoint_path, map_location=device, weights_only=False
        )
        validate_training_resume_checkpoint(
            checkpoint,
            expected_contract_sha256=resume_contract_sha256,
            expected_model_name=output_model_name,
            expected_fold_id=str(fold_row.fold_id),
            expected_seed=seed,
            expected_runtime_fingerprint=resume_runtime_fingerprint,
        )
        model.load_state_dict(checkpoint["model_state"])
        state_decoder.load_state_dict(checkpoint["state_decoder_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        early = checkpoint["early_stopping_state"]
        best_pathway_score = float(early["best_pathway_score"])
        best_state_score = float(early["best_state_score"])
        best_pathway_epoch = int(early["best_pathway_epoch"])
        best_state_epoch = int(early["best_state_epoch"])
        pathway_patience_score = float(early["pathway_patience_score"])
        state_patience_score = float(early["state_patience_score"])
        pathway_patience_epoch = int(early["pathway_patience_epoch"])
        state_patience_epoch = int(early["state_patience_epoch"])
        runtime_validation_cycles = int(early["runtime_validation_cycles"])
        pathway_stale_validation_cycles = int(
            early["pathway_stale_validation_cycles"]
        )
        state_stale_validation_cycles = int(early["state_stale_validation_cycles"])
        optimizer_steps = int(checkpoint["optimizer_steps"])
        history = list(checkpoint["history"])
        resumed_from_epoch = int(checkpoint["last_epoch"])
        resume_count = int(checkpoint.get("resume_count", 0)) + 1
        start_epoch = resumed_from_epoch + 1
        resume_stop_requested = bool(checkpoint.get("stop_requested", False))
        for role, best_epoch, filename, hash_key in (
            ("pathway", best_pathway_epoch, "best_pathway.pt", "best_pathway_sha256"),
            ("state", best_state_epoch, "best_state.pt", "best_state_sha256"),
        ):
            best_path = model_dir / filename
            expected_hash = checkpoint.get(hash_key)
            if best_epoch > 0 and (
                not best_path.is_file()
                or not expected_hash
                or file_sha256(best_path) != expected_hash
            ):
                raise RuntimeError(
                    f"Exact resume {role} checkpoint is missing or changed: {best_path}"
                )
        restore_rng_state(torch, checkpoint["rng_state"])
        if resumed_from_epoch > int(cfg["training"]["epochs"]):
            raise RuntimeError(
                "Resume checkpoint is beyond the configured epoch cap: "
                f"checkpoint={resumed_from_epoch}, cap={cfg['training']['epochs']}"
            )
    sampling_audit = {
        "fold_id": str(fold_row.fold_id),
        "seed": seed,
        "state_training_rows": len(state_batches["train"].frame),
        "state_training_positive": state_train_positive,
        "state_training_unlabeled": state_train_unlabeled,
        "state_training_positive_rate": state_train_positive / len(state_batches["train"].frame),
        "configured_unlabeled_to_positive_ratio": float(state_cfg.get("unlabeled_to_positive_ratio", 4.0)),
        "configured_max_training_pairs_per_fold": int(state_cfg.get("max_training_pairs_per_fold", 150000)),
        "state_unlabeled_direction_used_in_loss": state_unlabeled_direction_used,
        "state_head_ids": state_head_ids,
        "independent_state_heads": True,
        "candidate_microbatch_size": candidate_microbatch_size,
        "gradient_accumulation_steps": "all_candidate_microbatches_per_epoch",
        "state_auxiliary_loss_weight": state_loss_weight,
        "graph_contract": bundle.graph_contract,
        "episodic_pseudoheldout": episodic_enabled,
        "runtime_graph_sampling": runtime_enabled,
        "runtime_num_chunks": bundle.runtime_num_chunks,
        "residual_learning": residual_enabled,
        "residual_structure": "z_final=z_base+delta_graph" if residual_enabled else "standalone_logit",
        "residual_shrinkage_lambda": residual_lambda if residual_enabled else 0.0,
        "residual_initialization_max_abs": residual_initialization_max_abs,
        "candidate_sampling_seed": candidate_sampling_seed,
        "exact_interruption_resume_supported": True,
        "resume_checkpoint_every_epochs": resume_checkpoint_interval,
        "resumed_from_epoch": resumed_from_epoch,
        "resume_count": resume_count,
        **state_evaluation_audit,
        **pathway_sampling_audit,
    }
    write_json(sampling_audit, model_dir / "state_training_sample_audit.json")

    path_labels_numpy = pathway_batches["train"]["y"].detach().cpu().numpy()
    state_labels_numpy = state_batches["train"].y.detach().cpu().numpy()
    state_heads_numpy = state_batches["train"].head.detach().cpu().numpy()
    training_epochs = (
        ()
        if resume_stop_requested
        else range(start_epoch, int(cfg["training"]["epochs"]) + 1)
    )
    for epoch in training_epochs:
        episode = episode_planner.episode(epoch) if episode_planner is not None else None
        if runtime_enabled:
            epoch_bundle = runtime_bundle_for_step(
                bundle,
                epoch - 1,
                pseudoheldout_cancer=(episode.pseudoheldout_cancer if episode else None),
                formal_heldout_cancers=formal_heldout,
                reference_only=reference_only,
            )
            graph = move_graph(epoch_bundle, device, kind)
        else:
            epoch_bundle = bundle
        model.train(); state_decoder.train(); optimizer.zero_grad(set_to_none=True)
        if episode is not None:
            path_pool = np.flatnonzero(
                pathway_frames["train"].cancer_id.astype(str).to_numpy()
                == episode.pseudoheldout_cancer
            )
            state_pool_mask = (
                state_batches["train"].frame.cancer_id.astype(str).to_numpy()
                == episode.pseudoheldout_cancer
            )
        else:
            path_pool = np.arange(len(path_labels_numpy), dtype=np.int64)
            state_pool_mask = np.ones(len(state_labels_numpy), dtype=bool)
        local_path_indices = candidate_microbatch_indices(
            path_labels_numpy[path_pool], candidate_microbatch_size, seed + epoch * 1009
        )
        path_indices = [path_pool[indices] for indices in local_path_indices]
        state_indices: list[np.ndarray] = []
        state_scales: list[float] = []
        for head_index in range(len(state_head_ids)):
            head_positions = np.flatnonzero(
                (state_heads_numpy == head_index) & state_pool_mask
            )
            if not len(head_positions):
                continue
            head_batches = candidate_microbatch_indices(
                state_labels_numpy[head_positions],
                candidate_microbatch_size,
                seed + epoch * 2003 + head_index,
            )
            for local_indices in head_batches:
                state_indices.append(head_positions[local_indices])
                state_scales.append(
                    len(local_indices) / max(len(head_positions), 1) / len(state_head_ids)
                )
        work_items = [
            ("pathway", indices, len(indices) / max(len(path_pool), 1))
            for indices in path_indices
        ] + [
            ("state", indices, scale)
            for indices, scale in zip(state_indices, state_scales, strict=True)
        ]
        path_loss_value = 0.0
        state_loss_value = 0.0
        with amp_autocast(torch, device, amp_enabled):
            encoded = model.encode(graph)
        accumulated_encoded = _detach_encoded_for_microbatch_accumulation(encoded)
        for task_name, indices, task_scale in work_items:
            with amp_autocast(torch, device, amp_enabled):
                if task_name == "pathway":
                    train_path_batch = _slice_pathway_batch(pathway_target_batches["train"], indices)
                    if mask_pair_evidence_for_all or kind == "cc_hhgt":
                        train_path_batch = _pair_masked_batch(
                            train_path_batch,
                            zero_all=(episode is not None),
                            mask_probability=(0.0 if episode is not None else strict_pair_mask_probability),
                        )
                    path_logits, path_direction, path_residual = _pathway_forward(
                        model, accumulated_encoded, train_path_batch, kind
                    )
                    micro_loss = nnpu_loss(
                        path_logits,
                        train_path_batch["y"],
                        train_path_batch["weak"],
                        float(cfg["training"]["pu_prior"]),
                        float(cfg["training"]["unlabeled_weight"]),
                        float(cfg["training"]["weak_positive_weight"]),
                    )
                    known_direction = (
                        train_path_batch["y"].gt(0.5)
                        & train_path_batch["direction"].ge(0)
                    )
                    if known_direction.any():
                        micro_loss = micro_loss + float(
                            cfg["training"]["direction_loss_weight"]
                        ) * F.binary_cross_entropy_with_logits(
                            path_direction[known_direction],
                            train_path_batch["direction"][known_direction],
                        )
                    if residual_enabled:
                        micro_loss = micro_loss + residual_lambda * path_residual.square().mean()
                    scaled_loss = task_scale * micro_loss
                    path_loss_value += task_scale * float(micro_loss.detach().cpu())
                else:
                    train_state_batch = _slice_state_batch(state_batches["train"], indices)
                    state_logits, state_direction, state_residual = _state_forward(
                        state_decoder, accumulated_encoded, train_state_batch, kind
                    )
                    micro_loss = nnpu_loss(
                        state_logits,
                        train_state_batch.y,
                        train_state_batch.weak,
                        float(state_cfg.get("pu_prior", cfg["training"]["pu_prior"])),
                        float(
                            state_cfg.get(
                                "unlabeled_weight", cfg["training"]["unlabeled_weight"]
                            )
                        ),
                        float(
                            state_cfg.get(
                                "weak_positive_weight",
                                cfg["training"]["weak_positive_weight"],
                            )
                        ),
                    )
                    # The positive-membership condition is explicit even when
                    # upstream data are malformed: unlabeled rows can never
                    # contribute to direction loss.
                    known_direction = train_state_batch.y.gt(0.5) & train_state_batch.direction.ge(0)
                    if known_direction.any():
                        micro_loss = micro_loss + state_direction_weight * F.binary_cross_entropy_with_logits(
                            state_direction[known_direction],
                            train_state_batch.direction[known_direction],
                        )
                    if residual_enabled:
                        micro_loss = micro_loss + residual_lambda * state_residual.square().mean()
                    scaled_loss = state_loss_weight * task_scale * micro_loss
                    state_loss_value += task_scale * float(micro_loss.detach().cpu())
            scaler.scale(scaled_loss).backward()
        encoder_backward_calls = _backward_encoder_once(
            torch, encoded, accumulated_encoded
        )
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(state_decoder.parameters()), 5.0)
        scaler.step(optimizer); scaler.update()
        optimizer_steps += 1

        model.eval(); state_decoder.eval()
        epoch_cap = int(cfg["training"]["epochs"])
        validation_performed = bool(
            not runtime_enabled
            or epoch == 1
            or epoch % bundle.runtime_num_chunks == 0
            or epoch == epoch_cap
        )
        validation_chunk_count = 0
        if validation_performed:
            runtime_validation_cycles += int(runtime_enabled)
            # A single runtime chunk gives a noisy, topology-dependent
            # checkpoint score.  Validate only at registered checkpoints and
            # average logits across the complete non-destructive coverage
            # cycle.  This costs about one extra encoder pass per training
            # epoch while making model selection independent of chunk order.
            validation_chunks = (
                list(range(bundle.runtime_num_chunks)) if runtime_enabled else [None]
            )
            val_path_logit_sum = np.zeros(
                len(pathway_frames["val"]), dtype=np.float64
            )
            val_state_logit_sum = np.zeros(
                len(state_batches["val"].frame), dtype=np.float64
            )
            with torch.no_grad():
                for validation_chunk in validation_chunks:
                    validation_bundle = (
                        runtime_bundle_for_step(bundle, int(validation_chunk))
                        if validation_chunk is not None
                        else bundle
                    )
                    validation_graph = move_graph(validation_bundle, device, kind)
                    with amp_autocast(torch, device, amp_enabled):
                        validation_encoded = model.encode(validation_graph)
                        val_path_logits, _, _ = _pathway_forward(
                            model,
                            validation_encoded,
                            pathway_target_batches["val"],
                            kind,
                        )
                        val_state_logits, _, _ = _state_forward(
                            state_decoder,
                            validation_encoded,
                            state_batches["val"],
                            kind,
                        )
                    val_path_logit_sum += val_path_logits.float().cpu().numpy()
                    val_state_logit_sum += val_state_logits.float().cpu().numpy()
            validation_chunk_count = len(validation_chunks)
            val_path_prob = expit(val_path_logit_sum / validation_chunk_count)
            val_state_prob = expit(val_state_logit_sum / validation_chunk_count)
            bins = int(cfg["calibration"]["bins"])
            path_metrics = proxy_binary_metrics(
                pathway_frames["val"], val_path_prob, bins
            )
            pathway_score = float(path_metrics.get("auprc", math.nan))
            state_score, state_metrics = state_equal_macro_metrics(
                state_batches["val"].frame, val_state_prob, bins
            )
        else:
            path_metrics = {}
            state_metrics = {}
            pathway_score = math.nan
            state_score = math.nan
        row = {
            "epoch": epoch,
            "loss": path_loss_value + state_loss_weight * state_loss_value,
            "path_loss": path_loss_value,
            "state_loss": state_loss_value,
            "pathway_checkpoint_score": pathway_score,
            "state_checkpoint_score": state_score,
            "checkpoint_min_delta": checkpoint_min_delta,
            "pathway_microbatches": len(path_indices),
            "state_microbatches": len(state_indices),
            "decoder_microbatch_backward_calls": len(work_items),
            "encoder_backward_calls": encoder_backward_calls,
            "optimizer_steps": optimizer_steps,
            "runtime_chunk": epoch_bundle.active_runtime_chunk,
            "runtime_num_chunks": bundle.runtime_num_chunks,
            "validation_full_coverage_cycle": validation_performed,
            "validation_runtime_chunks": validation_chunk_count,
            "runtime_validation_cycle": runtime_validation_cycles,
            "pathway_stale_validation_cycles": pathway_stale_validation_cycles,
            "state_stale_validation_cycles": state_stale_validation_cycles,
            "pseudoheldout_cancer": episode.pseudoheldout_cancer if episode else None,
            "pseudoheldout_candidate_loss_only": episode is not None,
            **{f"pathway_{key}": value for key, value in path_metrics.items()},
            **state_metrics,
        }
        history.append(row)
        checkpoint_payload = {
            "model_state": model.state_dict(),
            "state_decoder_state": state_decoder.state_dict(),
            "kind": kind,
            "features": features,
            "fold": fold_row.to_dict(),
            "seed": seed,
            "analysis_version": cfg["analysis_version"],
            "state_nodes": sorted(observed_states),
            "state_head_ids": state_head_ids,
            "state_edge_count": state_edge_count,
            "multitask": True,
            "formal_training_gate_path": cfg.get("_formal_training_gate_path"),
            "formal_training_gate_sha256": cfg.get("_formal_training_gate_sha256"),
            "strict_pair_evidence_policy": "validation_and_test_fully_masked",
            "pair_evidence_masked_for_all_pathway_models": mask_pair_evidence_for_all,
            "strict_pair_evidence_train_row_mask_probability": strict_pair_mask_probability,
            "candidate_microbatch_size": candidate_microbatch_size,
            "optimizer_steps": optimizer_steps,
            "decoder_microbatch_backward_calls": len(work_items),
            "encoder_backward_calls_per_optimizer_step": encoder_backward_calls,
            "checkpoint_min_delta": checkpoint_min_delta,
            "deterministic_algorithms": True,
            "graph_contract": bundle.graph_contract,
            "episodic_pseudoheldout": episodic_enabled,
            "runtime_graph_sampling": runtime_enabled,
            "runtime_num_chunks": bundle.runtime_num_chunks,
            "runtime_chunk_at_checkpoint": epoch_bundle.active_runtime_chunk,
            "validation_runtime_chunks": validation_chunk_count,
            "validation_full_coverage_cycle": validation_performed,
            "pseudoheldout_cancer": episode.pseudoheldout_cancer if episode else None,
            "residual_learning": residual_enabled,
            "residual_structure": "z_final=z_base+delta_graph" if residual_enabled else "standalone_logit",
            "residual_shrinkage_lambda": residual_lambda if residual_enabled else 0.0,
            "residual_initialization_max_abs": residual_initialization_max_abs,
            "candidate_sampling_seed": candidate_sampling_seed,
        }
        if np.isfinite(pathway_score) and pathway_score > best_pathway_score:
            best_pathway_score = pathway_score
            best_pathway_epoch = epoch
            checkpoint_payload["checkpoint_role"] = "pathway_best"
            _atomic_torch_save(torch, checkpoint_payload, model_dir / "best_pathway.pt")
        if np.isfinite(state_score) and state_score > best_state_score:
            best_state_score = state_score
            best_state_epoch = epoch
            checkpoint_payload["checkpoint_role"] = "state_best"
            _atomic_torch_save(torch, checkpoint_payload, model_dir / "best_state.pt")
        pathway_patience_improved = _checkpoint_patience_improved(
            pathway_score, pathway_patience_score, checkpoint_min_delta
        )
        if pathway_patience_improved:
            pathway_patience_score = pathway_score
            pathway_patience_epoch = epoch
            pathway_stale_validation_cycles = 0
        elif validation_performed:
            pathway_stale_validation_cycles += 1
        state_patience_improved = _checkpoint_patience_improved(
            state_score, state_patience_score, checkpoint_min_delta
        )
        if state_patience_improved:
            state_patience_score = state_score
            state_patience_epoch = epoch
            state_stale_validation_cycles = 0
        elif validation_performed:
            state_stale_validation_cycles += 1
        history[-1]["pathway_stale_validation_cycles"] = (
            pathway_stale_validation_cycles
        )
        history[-1]["state_stale_validation_cycles"] = (
            state_stale_validation_cycles
        )
        should_stop = False
        if validation_performed and best_pathway_epoch > 0 and best_state_epoch > 0:
            if runtime_enabled:
                should_stop = (
                    pathway_stale_validation_cycles >= checkpoint_patience_cycles
                    and state_stale_validation_cycles >= checkpoint_patience_cycles
                )
            else:
                should_stop = (
                    epoch - pathway_patience_epoch >= int(cfg["training"]["patience"])
                    and epoch - state_patience_epoch >= int(cfg["training"]["patience"])
                )
        publish_resume_checkpoint = bool(
            epoch == 1
            or epoch % resume_checkpoint_interval == 0
            or validation_performed
            or should_stop
            or epoch == int(cfg["training"]["epochs"])
        )
        if publish_resume_checkpoint:
            early_stopping_state = {
                "best_pathway_score": best_pathway_score,
                "best_state_score": best_state_score,
                "best_pathway_epoch": best_pathway_epoch,
                "best_state_epoch": best_state_epoch,
                "pathway_patience_score": pathway_patience_score,
                "state_patience_score": state_patience_score,
                "pathway_patience_epoch": pathway_patience_epoch,
                "state_patience_epoch": state_patience_epoch,
                "runtime_validation_cycles": runtime_validation_cycles,
                "pathway_stale_validation_cycles": pathway_stale_validation_cycles,
                "state_stale_validation_cycles": state_stale_validation_cycles,
            }
            last_training_state = {
                "checkpoint_format": CHECKPOINT_FORMAT,
                "resume_contract_sha256": resume_contract_sha256,
                "runtime_fingerprint": resume_runtime_fingerprint,
                "formal_training_gate_sha256": cfg.get(
                    "_formal_training_gate_sha256"
                ),
                "run_id": cfg.get("_run_id"),
                "model_name": output_model_name,
                "fold_id": str(fold_row.fold_id),
                "seed": seed,
                "last_epoch": epoch,
                "configured_epoch_cap": int(cfg["training"]["epochs"]),
                "model_state": model.state_dict(),
                "state_decoder_state": state_decoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rng_state": capture_rng_state(torch),
                "history": history,
                "optimizer_steps": optimizer_steps,
                "early_stopping_state": early_stopping_state,
                "best_pathway_sha256": (
                    file_sha256(model_dir / "best_pathway.pt")
                    if (model_dir / "best_pathway.pt").is_file()
                    else None
                ),
                "best_state_sha256": (
                    file_sha256(model_dir / "best_state.pt")
                    if (model_dir / "best_state.pt").is_file()
                    else None
                ),
                "stop_requested": should_stop,
                "resume_count": resume_count,
                "resume_checkpoint_every_epochs": resume_checkpoint_interval,
            }
            _atomic_torch_save(torch, last_training_state, last_training_state_path)
        if validation_performed:
            atomic_write_json(
                model_dir / "TRAINING_PROGRESS.json",
                {
                    "status": "RUNNING" if not should_stop else "EARLY_STOP_REQUESTED",
                    "model_name": output_model_name,
                    "fold_id": str(fold_row.fold_id),
                    "seed": seed,
                    "epoch": epoch,
                    "configured_epoch_cap": int(cfg["training"]["epochs"]),
                    "runtime_validation_cycle": runtime_validation_cycles,
                    "runtime_num_chunks": bundle.runtime_num_chunks,
                    "pathway_checkpoint_score": pathway_score,
                    "state_checkpoint_score": state_score,
                    "best_pathway_epoch": best_pathway_epoch,
                    "best_state_epoch": best_state_epoch,
                    "best_pathway_validation_score": best_pathway_score,
                    "best_state_validation_score": best_state_score,
                    "pathway_stale_validation_cycles": pathway_stale_validation_cycles,
                    "state_stale_validation_cycles": state_stale_validation_cycles,
                    "checkpoint_patience_cycles": checkpoint_patience_cycles,
                    "checkpoint_min_delta": checkpoint_min_delta,
                    "stop_requested": should_stop,
                    "resume_count": resume_count,
                    "formal_training_gate_sha256": cfg.get(
                        "_formal_training_gate_sha256"
                    ),
                },
            )
        if should_stop:
            break

    if best_pathway_epoch == 0 or best_state_epoch == 0:
        raise RuntimeError(
            f"Unable to select both checkpoints: pathway_epoch={best_pathway_epoch}, state_epoch={best_state_epoch}"
        )
    pathway_checkpoint = torch.load(
        model_dir / "best_pathway.pt", map_location=device, weights_only=False
    )
    state_checkpoint = torch.load(
        model_dir / "best_state.pt", map_location=device, weights_only=False
    )
    pathway_predictions: list[pd.DataFrame] = []
    state_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    model.load_state_dict(pathway_checkpoint["model_state"])
    model.eval()
    evaluation_chunks = list(range(bundle.runtime_num_chunks)) if runtime_enabled else [None]
    path_logit_sum = {split: np.zeros(len(pathway_frames[split]), dtype=np.float64) for split in ("train", "val", "test")}
    path_residual_sum = {split: np.zeros(len(pathway_frames[split]), dtype=np.float64) for split in ("train", "val", "test")}
    path_direction_sum = {split: np.zeros(len(pathway_frames[split]), dtype=np.float64) for split in ("train", "val", "test")}
    with torch.no_grad():
        for chunk in evaluation_chunks:
            evaluation_bundle = (
                runtime_bundle_for_step(bundle, int(chunk)) if chunk is not None else bundle
            )
            evaluation_graph = move_graph(evaluation_bundle, device, kind)
            pathway_encoded = model.encode(evaluation_graph)
            for split in ("train", "val", "test"):
                logits, direction_logits, residual_logits = _pathway_forward(
                    model, pathway_encoded, pathway_target_batches[split], kind
                )
                path_logit_sum[split] += logits.float().cpu().numpy()
                path_residual_sum[split] += residual_logits.float().cpu().numpy()
                path_direction_sum[split] += direction_logits.float().cpu().numpy()
    for split in ("train", "val", "test"):
        path_logits = path_logit_sum[split] / len(evaluation_chunks)
        path_residual = path_residual_sum[split] / len(evaluation_chunks)
        path_direction = path_direction_sum[split] / len(evaluation_chunks)
        path_prob = expit(path_logits)
        path_frame = pathway_frames[split]
        path_part = path_frame[[c for c in ["candidate_id", "cancer_id", "lncrna_id", "pathway_id", "pathway_family_id", "label_class", "proxy_label", "direction", "evaluation_sampling_policy", "evaluation_universe_rows"] if c in path_frame]].copy()
        path_part["split"] = split
        path_part["model_name"] = output_model_name
        path_part["seed"] = seed
        path_part["fold_id"] = str(fold_row.fold_id)
        path_part["raw_logit"] = path_logits
        if residual_enabled:
            path_base = pathway_batches[split]["base_logit"].detach().cpu().numpy().astype(float)
            path_part["base_logit"] = path_base
            path_part["base_probability"] = expit(path_base)
            path_part["graph_residual_logit"] = path_residual
            path_part["residual_shrinkage_lambda"] = residual_lambda
            path_part["residual_structure"] = "z_final=z_base+delta_graph"
        path_part["proxy_positive_probability"] = path_prob
        path_part["raw_probability"] = path_prob
        path_part["prediction_scale"] = "raw_probability"
        path_part["direction_positive_probability"] = expit(path_direction)
        path_part["runtime_ensemble_chunks"] = len(evaluation_chunks)
        pathway_predictions.append(path_part)
        metric_rows.append(
            {
                "fold_id": str(fold_row.fold_id),
                "seed": seed,
                "model_name": output_model_name,
                "task": "pathway",
                "state_id": None,
                "split": split,
                "metric_scope": metric_scope_for_split(split, evaluation="LOCO"),
                "prediction_scale": "raw_probability",
                "runtime_ensemble_chunks": len(evaluation_chunks),
                **proxy_binary_metrics(
                    path_frame, path_prob, int(cfg["calibration"]["bins"])
                ),
            }
        )

    model.load_state_dict(state_checkpoint["model_state"])
    state_decoder.load_state_dict(state_checkpoint["state_decoder_state"])
    model.eval()
    state_decoder.eval()
    state_logit_sum = {split: np.zeros(len(state_batches[split].frame), dtype=np.float64) for split in ("train", "val", "test")}
    state_residual_sum = {split: np.zeros(len(state_batches[split].frame), dtype=np.float64) for split in ("train", "val", "test")}
    state_direction_sum = {split: np.zeros(len(state_batches[split].frame), dtype=np.float64) for split in ("train", "val", "test")}
    state_embedding_sum = None
    with torch.no_grad():
        for chunk in evaluation_chunks:
            evaluation_bundle = (
                runtime_bundle_for_step(bundle, int(chunk)) if chunk is not None else bundle
            )
            evaluation_graph = move_graph(evaluation_bundle, device, kind)
            state_encoded = model.encode(evaluation_graph)
            if kind == "rgcn":
                if state_embedding_sum is None:
                    state_embedding_sum = state_encoded.float().clone()
                else:
                    state_embedding_sum.add_(state_encoded.float())
            else:
                if state_embedding_sum is None:
                    state_embedding_sum = {
                        node_type: values.float().clone()
                        for node_type, values in state_encoded.items()
                    }
                else:
                    for node_type, values in state_encoded.items():
                        state_embedding_sum[node_type].add_(values.float())
            for split in ("train", "val", "test"):
                logits, direction_logits, residual_logits = _state_forward(
                    state_decoder, state_encoded, state_batches[split], kind
                )
                state_logit_sum[split] += logits.float().cpu().numpy()
                state_residual_sum[split] += residual_logits.float().cpu().numpy()
                state_direction_sum[split] += direction_logits.float().cpu().numpy()
    for split in ("train", "val", "test"):
        state_logits = state_logit_sum[split] / len(evaluation_chunks)
        state_residual = state_residual_sum[split] / len(evaluation_chunks)
        state_direction = state_direction_sum[split] / len(evaluation_chunks)
        state_prob = expit(state_logits)
        state_frame = state_batches[split].frame
        state_part = state_frame[[c for c in ["candidate_id", "cancer_id", "lncrna_id", "state_id", "label_class", "proxy_label", "direction", "effect", "fdr", "n_observed", "n_missing", "n_samples", "residual_design_rank", "design_rank", "correlation_df", "df", "eligibility", "unavailable_reason", "evaluation_eligibility", "evaluation_unavailable_reason", "sample_universe_sha256", "evaluation_policy", "evaluation_sampling_policy", "lineage_sha256"] if c in state_frame]].copy()
        state_part["split"] = split
        state_part["model_name"] = output_model_name
        state_part["seed"] = seed
        state_part["fold_id"] = str(fold_row.fold_id)
        state_part["raw_logit"] = state_logits
        if residual_enabled:
            state_base = state_batches[split].base_logit.detach().cpu().numpy().astype(float)
            state_part["base_logit"] = state_base
            state_part["base_probability"] = expit(state_base)
            state_part["graph_residual_logit"] = state_residual
            state_part["residual_shrinkage_lambda"] = residual_lambda
            state_part["residual_structure"] = "z_final=z_base+delta_graph"
        state_part["proxy_positive_probability"] = state_prob
        state_part["raw_probability"] = state_prob
        state_part["prediction_scale"] = "raw_probability"
        state_part["direction_positive_probability"] = expit(state_direction)
        state_part["runtime_ensemble_chunks"] = len(evaluation_chunks)
        state_predictions.append(state_part)

        for state_id, positions in state_frame.groupby("state_id", observed=True).indices.items():
            index = np.asarray(positions, dtype=np.int64)
            state_metrics = proxy_binary_metrics(
                state_frame.iloc[index],
                state_prob[index],
                int(cfg["calibration"]["bins"]),
            )
            positive_rate = float(state_metrics.get("positive_rate", math.nan))
            auprc = float(state_metrics.get("auprc", math.nan))
            metric_rows.append(
                {
                    "fold_id": str(fold_row.fold_id),
                    "seed": seed,
                    "model_name": output_model_name,
                    "task": "state",
                    "state_id": str(state_id),
                    "split": split,
                    "metric_scope": metric_scope_for_split(split, evaluation="LOCO"),
                    "prediction_scale": "raw_probability",
                    "runtime_ensemble_chunks": len(evaluation_chunks),
                    **state_metrics,
                    "auprc_lift": auprc - positive_rate
                    if np.isfinite(auprc) and np.isfinite(positive_rate)
                    else math.nan,
                }
            )

    if state_embedding_sum is None:
        raise RuntimeError("State embedding coverage-cycle ensemble was not computed")
    if kind == "rgcn":
        state_embedding_mean = state_embedding_sum / len(evaluation_chunks)
    else:
        state_embedding_mean = {
            node_type: values / len(evaluation_chunks)
            for node_type, values in state_embedding_sum.items()
        }

    path_pred = pd.concat(pathway_predictions, ignore_index=True)
    state_pred = pd.concat(state_predictions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    diagnostic_rows: list[dict[str, Any]] = []
    for task_name, prediction, group_columns in (
        ("pathway", path_pred, ["split"]),
        ("state", state_pred, ["split", "state_id"]),
    ):
        for group_key, group in prediction.groupby(group_columns, observed=True):
            keys = group_key if isinstance(group_key, tuple) else (group_key,)
            labels = pd.to_numeric(group.proxy_label, errors="coerce")
            logits = pd.to_numeric(group.raw_logit, errors="coerce")
            probability = pd.to_numeric(group.proxy_positive_probability, errors="coerce")
            row = {
                "task": task_name,
                "split": str(keys[0]),
                "state_id": str(keys[1]) if len(keys) > 1 else None,
                "n": int(len(group)),
                "label_prevalence": float(labels.mean()),
                "sampling_prevalence": float(labels.mean()),
                "pos_weight": "NOT_APPLICABLE_NNPU",
                "loss_form": "nnPU + positive-membership-only direction BCE",
                "pu_prior": float(
                    state_cfg.get("pu_prior", cfg["training"]["pu_prior"])
                    if task_name == "state"
                    else cfg["training"]["pu_prior"]
                ),
                "raw_logit_mean": float(logits.mean()),
                "raw_logit_std": float(logits.std(ddof=0)),
                "raw_logit_min": float(logits.min()),
                "raw_logit_max": float(logits.max()),
                "probability_mean": float(probability.mean()),
                "probability_std": float(probability.std(ddof=0)),
                "probability_min": float(probability.min()),
                "probability_max": float(probability.max()),
                "probability_unique_1e8": int(probability.round(8).nunique()),
            }
            row["collapse_flag"] = bool(
                row["probability_unique_1e8"] <= 1
                or row["probability_std"] < 1e-6
            )
            diagnostic_rows.append(row)
    loss_output_diagnostics = pd.DataFrame(diagnostic_rows)
    path_raw_file = model_dir / "prediction_pathway_raw.parquet"
    state_raw_file = model_dir / "prediction_state_raw.parquet"
    _atomic_write_table(path_pred, path_raw_file)
    _atomic_write_table(state_pred, state_raw_file)
    for task, prediction, prediction_file in (
        ("pathway", path_pred, path_raw_file),
        ("state", state_pred, state_raw_file),
    ):
        mask = metrics.task.astype(str).eq(task)
        metrics.loc[mask, "prediction_file_sha256"] = file_sha256(prediction_file)
        metrics.loc[mask, "candidate_universe_sha256"] = candidate_universe_sha256(prediction)
        metrics.loc[mask, "calibration_model_sha256"] = "NOT_APPLICABLE_RAW"
    _atomic_write_table(metrics, model_dir / "metrics_raw.tsv")
    _atomic_write_table(metrics, model_dir / "metrics.tsv")
    _atomic_write_table(
        loss_output_diagnostics,
        model_dir / "loss_output_diagnostics.tsv",
    )
    _atomic_write_table(pd.DataFrame(history), model_dir / "training_history.tsv")
    # The state-selected encoder is the formal representation export.
    export_node_embeddings(
        cfg,
        model,
        graph,
        bundle,
        kind,
        fold_row,
        model_dir,
        encoded=state_embedding_mean,
        runtime_ensemble_chunks=len(evaluation_chunks),
    )
    _export_embedding_parquets(
        cfg,
        model,
        graph,
        bundle,
        kind,
        model_dir,
        encoded=state_embedding_mean,
    )
    success = {
        "status": "COMPLETED",
        "analysis_version": cfg["analysis_version"],
        "model_name": output_model_name,
        "fold_id": str(fold_row.fold_id),
        "seed": seed,
        "best_pathway_epoch": best_pathway_epoch,
        "best_state_epoch": best_state_epoch,
        "best_pathway_validation_score": best_pathway_score,
        "best_state_validation_score": best_state_score,
        "checkpoint_min_delta": checkpoint_min_delta,
        "last_pathway_patience_reset_epoch": pathway_patience_epoch,
        "last_state_patience_reset_epoch": state_patience_epoch,
        "epochs_completed": len(history),
        "configured_epoch_cap": int(cfg["training"]["epochs"]),
        "configured_patience": int(cfg["training"]["patience"]),
        "configured_runtime_checkpoint_patience_cycles": checkpoint_patience_cycles,
        "runtime_validation_cycles_completed": runtime_validation_cycles,
        "pathway_stale_validation_cycles": pathway_stale_validation_cycles,
        "state_stale_validation_cycles": state_stale_validation_cycles,
        "hit_hard_epoch_cap": len(history) >= int(cfg["training"]["epochs"]),
        "stopped_by_joint_patience": len(history) < int(cfg["training"]["epochs"]),
        "candidate_microbatch_size": candidate_microbatch_size,
        "pathway_microbatches_last_epoch": history[-1]["pathway_microbatches"],
        "state_microbatches_last_epoch": history[-1]["state_microbatches"],
        "decoder_microbatch_backward_calls_last_epoch": history[-1][
            "decoder_microbatch_backward_calls"
        ],
        "encoder_backward_calls_per_optimizer_step": history[-1][
            "encoder_backward_calls"
        ],
        "optimizer_steps": optimizer_steps,
        "exact_interruption_resume_supported": True,
        "resume_checkpoint_format": CHECKPOINT_FORMAT,
        "resume_contract_sha256": resume_contract_sha256,
        "resume_checkpoint_every_epochs": resume_checkpoint_interval,
        "resumed_from_epoch": resumed_from_epoch,
        "resume_count": resume_count,
        "last_training_state_sha256": file_sha256(last_training_state_path),
        "training_progress_sha256": file_sha256(
            model_dir / "TRAINING_PROGRESS.json"
        ),
        "training_runtime_fingerprint": resume_runtime_fingerprint,
        "formal_gpu_contract": cfg.get("_formal_gpu_contract"),
        "embedding_runtime_ensemble_chunks": len(evaluation_chunks),
        "embedding_runtime_ensemble_method": "mean_encoded_representation",
        "state_head_ids": state_head_ids,
        "independent_state_heads": True,
        "state_node_count": len(observed_states),
        "state_edge_count": state_edge_count,
        "strict_pair_evidence_policy": "validation_and_test_fully_masked",
        "pair_evidence_masked_for_all_pathway_models": mask_pair_evidence_for_all,
        "strict_pair_evidence_train_row_mask_probability": strict_pair_mask_probability,
        "residual_learning": residual_enabled,
        "residual_structure": "z_final=z_base+delta_graph" if residual_enabled else "standalone_logit",
        "residual_shrinkage_lambda": residual_lambda if residual_enabled else 0.0,
        "residual_initialization_max_abs": residual_initialization_max_abs,
        "residual_exact_fallback_tolerance": FALLBACK_ATOL,
        "candidate_sampling_seed": candidate_sampling_seed,
        "formal_training_gate_path": cfg.get("_formal_training_gate_path"),
        "formal_training_gate_sha256": cfg.get("_formal_training_gate_sha256"),
        **sampling_audit,
        "pathway_checkpoint_sha256": file_sha256(model_dir / "best_pathway.pt"),
        "state_checkpoint_sha256": file_sha256(model_dir / "best_state.pt"),
        "loss_output_diagnostics_sha256": file_sha256(
            model_dir / "loss_output_diagnostics.tsv"
        ),
    }
    # The task-level formal SUCCESS marker is written by the guarded CLI only
    # after calibration and end-of-task asset verification.
    atomic_write_json(model_dir / "TRAINING_SUCCESS.json", success)
    return success


def calibrate_multitask_fold(model_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "PASS", "model_dir": str(model_dir)}
    for task in ("pathway", "state"):
        path = model_dir / f"prediction_{task}_raw.parquet"
        frame = read_table(path)
        val = frame.loc[frame.split.astype(str).eq("val")]
        if val.empty:
            raise RuntimeError(f"No validation rows for {task}: {path}")
        group_column = "state_id" if task == "state" else None
        frame["raw_probability"] = frame["proxy_positive_probability"].astype(np.float64)
        frame["proxy_positive_probability"] = frame["raw_probability"]
        groups = sorted(val[group_column].astype(str).unique()) if group_column else ["pathway"]
        calibration: dict[str, Any] = {
            "method": "temperature",
            "task": task,
            "independent_by_state": task == "state",
            "groups": {},
        }
        for group in groups:
            val_group = val.loc[val[group_column].astype(str).eq(group)] if group_column else val
            if val_group.empty or val_group.proxy_label.astype(float).nunique() < 2:
                raise RuntimeError(f"Calibration requires both proxy classes: task={task}, group={group}")
            temperature = fit_temperature(
                val_group.raw_logit.to_numpy(float), val_group.proxy_label.to_numpy(float)
            )
            output_mask = frame[group_column].astype(str).eq(group) if group_column else np.ones(len(frame), dtype=bool)
            frame.loc[output_mask, "proxy_positive_probability"] = expit(
                frame.loc[output_mask, "raw_logit"].to_numpy(float) / temperature
            )
            calibration["groups"][group] = {
                "temperature": temperature,
                "validation_rows": int(len(val_group)),
                "validation_positive": int(val_group.proxy_label.astype(float).gt(0.5).sum()),
                "validation_positive_rate": float(val_group.proxy_label.astype(float).mean()),
                "validation_sampling_policy": (
                    sorted(val_group.evaluation_sampling_policy.astype(str).unique())
                    if "evaluation_sampling_policy" in val_group
                    else ["strict_candidate_universe"]
                ),
            }
        if frame.proxy_positive_probability.isna().any():
            raise RuntimeError(f"Uncalibrated rows remain for task={task}")
        frame["calibrated_probability"] = frame["proxy_positive_probability"].astype(np.float64)
        frame["prediction_scale"] = "calibrated_probability"
        frame["uncertainty"] = 1.0 - np.abs(frame.proxy_positive_probability - 0.5) * 2.0
        calibration_file = model_dir / f"calibration_{task}.json"
        atomic_write_json(calibration_file, calibration)
        prediction_file = model_dir / f"prediction_{task}_calibrated.parquet"
        _atomic_write_table(frame, prediction_file)
        metric_rows: list[dict[str, Any]] = []
        for split, split_frame in frame.groupby("split", observed=True):
            if task == "state":
                metric_groups = split_frame.groupby("state_id", observed=True)
            else:
                metric_groups = [(None, split_frame)]
            for state_id, metric_frame in metric_groups:
                values = proxy_binary_metrics(
                    metric_frame,
                    metric_frame.proxy_positive_probability,
                    bins=10,
                )
                metric_rows.append(
                    {
                        "task": task,
                        "state_id": str(state_id) if state_id is not None else None,
                        "split": str(split),
                        "metric_scope": metric_scope_for_split(str(split), evaluation="LOCO"),
                        "prediction_scale": "calibrated_probability",
                        "prediction_file_sha256": file_sha256(prediction_file),
                        "candidate_universe_sha256": candidate_universe_sha256(frame),
                        "calibration_model_sha256": file_sha256(calibration_file),
                        **values,
                    }
                )
        _atomic_write_table(pd.DataFrame(metric_rows), model_dir / f"metrics_{task}_calibrated.tsv")
        result[f"{task}_temperatures"] = {
            key: value["temperature"] for key, value in calibration["groups"].items()
        }
    atomic_write_json(model_dir / "CALIBRATION_SUCCESS.json", result)
    return result
