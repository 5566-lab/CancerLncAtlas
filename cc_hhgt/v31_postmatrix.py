"""Shared fail-closed utilities for V3.1 post-matrix comparisons."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


EXACT_IDENTITY = [
    "candidate_id",
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
    "proxy_label",
]


def exact_identity(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(EXACT_IDENTITY) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Exact-pathway prediction lacks identity columns: {missing}")
    identity = (
        frame[EXACT_IDENTITY]
        .sort_values("candidate_id", kind="stable")
        .reset_index(drop=True)
    )
    if identity.candidate_id.astype(str).duplicated().any():
        raise RuntimeError("Exact-pathway prediction contains duplicate candidate IDs")
    return identity


def align_exact_frames(frames: Iterable[pd.DataFrame]) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    ordered: list[pd.DataFrame] = []
    reference: pd.DataFrame | None = None
    for frame in frames:
        current = frame.sort_values("candidate_id", kind="stable").reset_index(drop=True)
        identity = exact_identity(current)
        if reference is None:
            reference = identity
        elif not identity.astype(str).equals(reference.astype(str)):
            raise RuntimeError("Exact-pathway candidate identities or labels drift across predictions")
        ordered.append(current)
    if reference is None:
        raise RuntimeError("No exact-pathway predictions were supplied")
    return reference, ordered


def binary_metrics(labels: Iterable, scores: Iterable, *, allow_single_class: bool = False) -> dict:
    y = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy(float)
    p = pd.to_numeric(pd.Series(scores), errors="coerce").to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid].astype(int)
    p = p[valid]
    if len(y) == 0 or np.unique(y).size != 2:
        if allow_single_class:
            return {
                "n": int(len(y)),
                "n_positive": int(y.sum()) if len(y) else 0,
                "positive_prevalence": float(y.mean()) if len(y) else np.nan,
                "auroc": np.nan,
                "auprc": np.nan,
                "auprc_lift": np.nan,
                "both_classes": False,
            }
        raise RuntimeError("Binary comparison requires both proxy classes")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise RuntimeError("Binary comparison contains invalid probabilities")
    prevalence = float(y.mean())
    auprc = float(average_precision_score(y, p))
    return {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "positive_prevalence": prevalence,
        "auroc": float(roc_auc_score(y, p)),
        "auprc": auprc,
        "auprc_lift": auprc - prevalence,
        "both_classes": True,
    }


def assert_metric_close(observed: object, expected: float | int, name: str) -> None:
    left = float(observed)
    right = float(expected)
    if not np.isfinite(left) or not np.isfinite(right) or not np.isclose(
        left, right, rtol=1e-9, atol=1e-12
    ):
        raise RuntimeError(f"Metric mismatch for {name}: observed={left}, recomputed={right}")
