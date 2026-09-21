"""Exact additive-logit residual primitives for the V3.1 pilot."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .common import require_columns
from .prediction_contract import candidate_universe_sha256


FALLBACK_ATOL = 1e-7


def attach_residual_base(
    candidates: pd.DataFrame,
    base: pd.DataFrame,
    *,
    split: str,
    train_cancers: set[str],
) -> pd.DataFrame:
    """Attach a leakage-safe frozen base offset without changing candidates."""

    require_columns(candidates, ["candidate_id", "cancer_id"], "residual candidates")
    required = [
        "candidate_id",
        "best_simple_score",
        "z_base",
        "metric_scope",
        "simple_base_fit_cancers",
    ]
    require_columns(base, required, "residual simple base")
    if base.candidate_id.astype(str).duplicated().any():
        raise RuntimeError("residual simple base contains duplicate candidate_id")
    before = candidate_universe_sha256(candidates)
    columns = [
        *required,
        *[
            column
            for column in ("best_simple_method", "prediction_scale", "target_subtype")
            if column in base
        ],
    ]
    attached = candidates.merge(
        base[columns], on="candidate_id", how="left", validate="one_to_one"
    )
    if attached.z_base.isna().any() or attached.best_simple_score.isna().any():
        missing = attached.loc[
            attached.z_base.isna() | attached.best_simple_score.isna(), "candidate_id"
        ].head(10).tolist()
        raise RuntimeError(f"residual base lacks candidates for {split}: {missing}")
    if (~np.isfinite(pd.to_numeric(attached.z_base, errors="coerce"))).any():
        raise RuntimeError(f"residual base contains non-finite z_base for {split}")
    observed_scope = set(attached.metric_scope.astype(str))
    expected_scope = "cancer_crossfit_OOF" if split == "train" else split
    if observed_scope != {expected_scope}:
        raise RuntimeError(
            f"residual base scope mismatch for {split}: expected={expected_scope}, observed={sorted(observed_scope)}"
        )
    registered_train = set(map(str, train_cancers))
    for row in attached[["cancer_id", "simple_base_fit_cancers"]].drop_duplicates().itertuples(index=False):
        fitted = set(filter(None, str(row.simple_base_fit_cancers).split("|")))
        if str(row.cancer_id) in fitted:
            raise RuntimeError(
                f"residual base for {row.cancer_id} includes its own outcomes"
            )
        if not fitted.issubset(registered_train):
            raise RuntimeError(
                f"residual base uses cancers outside outer train set: {sorted(fitted - registered_train)}"
            )
        if split != "train" and fitted != registered_train:
            raise RuntimeError(
                f"{split} residual base must be fit on the exact outer train cancers"
            )
    after = candidate_universe_sha256(attached)
    if before != after:
        raise RuntimeError("attaching residual base changed the candidate universe")
    attached["z_base"] = pd.to_numeric(attached.z_base, errors="raise").astype(float)
    attached["best_simple_score"] = pd.to_numeric(
        attached.best_simple_score, errors="raise"
    ).astype(float)
    return attached


def probability_to_logit(probability, eps: float = 1e-7) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    clipped = np.clip(probability, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def logit_to_probability(logit) -> np.ndarray:
    logit = np.asarray(logit, dtype=float)
    positive = logit >= 0
    probability = np.empty_like(logit, dtype=float)
    probability[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    probability[~positive] = exponential / (1.0 + exponential)
    return probability


def compose_residual_logits(
    base_logit,
    residuals: Mapping[str, np.ndarray],
    availability: Mapping[str, np.ndarray | bool] | None = None,
    admitted: Mapping[str, bool] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compose strictly additive residuals without allowing base overwrite."""

    base = np.asarray(base_logit, dtype=float)
    available = availability or {}
    admission = admitted or {}
    contributions: dict[str, np.ndarray] = {}
    final = base.copy()
    for name, values in residuals.items():
        delta = np.asarray(values, dtype=float)
        if delta.shape != base.shape:
            raise ValueError(f"residual {name} shape {delta.shape} != base shape {base.shape}")
        mask = np.asarray(available.get(name, True), dtype=bool)
        if mask.ndim == 0:
            mask = np.full(base.shape, bool(mask), dtype=bool)
        if mask.shape != base.shape:
            raise ValueError(f"availability {name} shape {mask.shape} != base shape {base.shape}")
        enabled = bool(admission.get(name, True))
        contribution = np.where(mask & enabled, delta, 0.0)
        contributions[name] = contribution
        final = final + contribution
    return final, contributions


def assert_exact_fallback(base_logit, candidate_logit, *, atol: float = FALLBACK_ATOL) -> None:
    base = np.asarray(base_logit, dtype=float)
    candidate = np.asarray(candidate_logit, dtype=float)
    if base.shape != candidate.shape:
        raise RuntimeError(f"fallback shape mismatch: {base.shape} != {candidate.shape}")
    error = float(np.max(np.abs(base - candidate))) if base.size else 0.0
    if not np.isfinite(error) or error > atol:
        raise RuntimeError(f"residual fallback invariant failed: max_abs_error={error} > {atol}")


def build_additive_residual_head(input_dim: int, hidden_dim: int = 0):
    """Build a residual head whose initial contribution is exactly zero.

    Torch is imported lazily so CPU-only audit code can import this module.
    The last affine layer is always zero-initialized; therefore epoch-zero
    output equals the supplied base logit independently of upstream features.
    """

    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    if hidden_dim < 0:
        raise ValueError("hidden_dim must be nonnegative")
    import torch
    from torch import nn

    class AdditiveResidualHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if hidden_dim:
                self.feature_map = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                self.output = nn.Linear(hidden_dim, 1)
            else:
                self.feature_map = nn.Identity()
                self.output = nn.Linear(input_dim, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def residual(self, features):
            return self.output(self.feature_map(features)).squeeze(-1)

        def forward(self, base_logit, features, availability=None, *, enabled: bool = True):
            if base_logit.ndim != 1:
                raise ValueError("base_logit must be one-dimensional")
            if features.ndim != 2 or features.shape[0] != base_logit.shape[0]:
                raise ValueError("features must be [candidate, feature] aligned to base_logit")
            if not enabled:
                return base_logit, torch.zeros_like(base_logit)
            delta = self.residual(features)
            if availability is not None:
                if availability.shape != base_logit.shape:
                    raise ValueError("availability must align to base_logit")
                delta = delta * availability.to(dtype=delta.dtype)
            return base_logit + delta, delta

        @staticmethod
        def shrinkage(delta, coefficient: float):
            if coefficient < 0:
                raise ValueError("residual shrinkage coefficient must be nonnegative")
            return float(coefficient) * delta.square().mean()

    return AdditiveResidualHead()
