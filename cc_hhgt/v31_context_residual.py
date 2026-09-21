"""Leakage-safe CPU residuals for V3.1 target-context modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .common import require_columns
from .metrics import binary_metrics
from .v31_residual import FALLBACK_ATOL, assert_exact_fallback, logit_to_probability


CONTEXT_KEY_COLUMNS = {"cancer_id", "lncrna_id", "pathway_family_id"}


def context_feature_columns(
    context: pd.DataFrame,
    *,
    key_columns: Sequence[str],
) -> tuple[str, ...]:
    """Return numeric features that have explicit availability masks."""

    require_columns(context, list(key_columns), "context table")
    features = tuple(
        sorted(
            column
            for column in context.columns
            if column not in set(key_columns)
            and not column.endswith("__available")
            and f"{column}__available" in context.columns
        )
    )
    if not features:
        raise RuntimeError("Context table has no value/mask feature pairs")
    for feature in features:
        values = pd.to_numeric(context[feature], errors="coerce")
        available = context[f"{feature}__available"].fillna(False).astype(bool)
        if values.loc[~available].notna().any():
            raise RuntimeError(f"Unavailable context values are not NaN: {feature}")
    return features


def _ranked_context(
    context: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    prefix: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    features = context_feature_columns(context, key_columns=key_columns)
    frame = context.loc[:, [*key_columns, *features, *[f"{x}__available" for x in features]]].copy()
    if frame.duplicated(list(key_columns)).any():
        raise RuntimeError(f"Context keys are duplicated: {list(key_columns)}")
    output = frame.loc[:, list(key_columns)].copy()
    engineered: list[str] = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        available = frame[f"{feature}__available"].fillna(False).astype(bool)
        raw_name = f"{prefix}__{feature}"
        rank_name = f"{raw_name}__within_cancer_rank"
        mask_name = f"{raw_name}__available"
        output[raw_name] = np.where(available, values, np.nan)
        ranks = output.groupby("cancer_id", observed=True, sort=False)[raw_name].rank(
            method="average", pct=True
        )
        output[rank_name] = np.where(available, ranks, np.nan)
        output[mask_name] = available.to_numpy(bool)
        engineered.extend([raw_name, rank_name])
    return output, tuple(engineered)


def attach_context_features(
    candidates: pd.DataFrame,
    lnc_context: pd.DataFrame,
    *,
    target_context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Attach lncRNA and optional pathway context without changing candidates."""

    require_columns(candidates, ["candidate_id", "cancer_id", "lncrna_id"], "candidates")
    frame = candidates.copy()
    frame["cancer_id"] = frame.cancer_id.astype(str)
    frame["lncrna_id"] = frame.lncrna_id.astype(str)
    before_ids = frame.candidate_id.astype(str).tolist()
    lnc, lnc_features = _ranked_context(
        lnc_context,
        key_columns=("cancer_id", "lncrna_id"),
        prefix="lnc",
    )
    frame = frame.merge(
        lnc,
        on=["cancer_id", "lncrna_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    engineered = list(lnc_features)
    if target_context is not None:
        require_columns(frame, ["pathway_family_id"], "pathway candidates")
        target, target_features = _ranked_context(
            target_context,
            key_columns=("cancer_id", "pathway_family_id"),
            prefix="pathway",
        )
        frame = frame.merge(
            target,
            on=["cancer_id", "pathway_family_id"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        engineered.extend(target_features)
    if frame.candidate_id.astype(str).tolist() != before_ids:
        raise RuntimeError("Attaching context changed candidate order or identity")
    mask_columns = sorted(
        column for column in frame.columns if column.endswith("__available")
    )
    if not mask_columns:
        raise RuntimeError("Attached context has no availability masks")
    for column in mask_columns:
        frame[column] = frame[column].astype("boolean").fillna(False).astype(bool)
    for column in engineered:
        raw_mask = column.split("__within_cancer_rank")[0] + "__available"
        values = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = np.where(frame[raw_mask], values, np.nan)
    frame["context_available"] = frame[mask_columns].any(axis=1)
    return frame, tuple(engineered)


@dataclass(frozen=True)
class ContextTransform:
    feature_columns: tuple[str, ...]
    feature_masks: tuple[str, ...]
    mask_columns: tuple[str, ...]
    centers: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> "ContextTransform":
        features = tuple(feature_columns)
        feature_masks = tuple(
            column.split("__within_cancer_rank")[0] + "__available"
            for column in features
        )
        masks = tuple(dict.fromkeys(feature_masks))
        centers: list[float] = []
        scales: list[float] = []
        for feature, mask in zip(features, feature_masks):
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
            available = frame[mask].fillna(False).to_numpy(bool) & np.isfinite(values)
            if available.any():
                center = float(np.mean(values[available]))
                scale = float(np.std(values[available]))
            else:
                center, scale = 0.0, 1.0
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            centers.append(center)
            scales.append(scale)
        return cls(
            features,
            feature_masks,
            masks,
            np.asarray(centers),
            np.asarray(scales),
        )

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = np.column_stack(
            [pd.to_numeric(frame[column], errors="coerce").to_numpy(float) for column in self.feature_columns]
        )
        feature_masks = np.column_stack(
            [frame[column].fillna(False).to_numpy(bool) for column in self.feature_masks]
        )
        finite = np.isfinite(values)
        feature_masks &= finite
        standardized = np.where(
            feature_masks,
            (np.where(finite, values, self.centers) - self.centers) / self.scales,
            0.0,
        )
        # Explicit masks let the residual distinguish a biological mid-rank
        # value from unavailable context, while missing values remain zero in
        # the standardized value channel.
        availability_channels = np.column_stack(
            [frame[column].fillna(False).to_numpy(bool) for column in self.mask_columns]
        )
        matrix = np.column_stack([standardized, availability_channels.astype(float)])
        available = availability_channels.any(axis=1)
        if not np.isfinite(matrix).all():
            raise RuntimeError("Non-finite values entered context residual matrix")
        return matrix, available


@dataclass(frozen=True)
class OffsetLogisticFit:
    weights: np.ndarray
    shrinkage_lambda: float
    converged: bool
    iterations: int
    objective: float
    initialization_max_abs_error: float


def fit_offset_logistic(
    matrix: np.ndarray,
    labels,
    base_logit,
    *,
    shrinkage_lambda: float,
    max_iterations: int = 300,
) -> OffsetLogisticFit:
    """Fit ``logit(p)=z_base+Xw`` from an exact zero residual start."""

    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=float)
    offset = np.asarray(base_logit, dtype=float)
    if x.ndim != 2 or y.shape != (x.shape[0],) or offset.shape != y.shape:
        raise ValueError("Offset-logistic inputs are not row aligned")
    if x.shape[1] < 1 or not np.isfinite(x).all() or not np.isfinite(offset).all():
        raise ValueError("Offset-logistic inputs must be finite and nonempty")
    if not set(np.unique(y)).issubset({0.0, 1.0}) or np.unique(y).size < 2:
        raise ValueError("Offset-logistic training requires both binary classes")
    penalty = float(shrinkage_lambda)
    if penalty < 0 or not np.isfinite(penalty):
        raise ValueError("Residual shrinkage must be finite and nonnegative")
    zero = np.zeros(x.shape[1], dtype=float)
    initialization = offset + x @ zero
    initialization_error = float(np.max(np.abs(initialization - offset)))
    if initialization_error > FALLBACK_ATOL:
        raise RuntimeError("Context residual zero initialization is not exact")

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        eta = offset + x @ weights
        loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
        value = loss + 0.5 * penalty * float(np.dot(weights, weights))
        probability = logit_to_probability(eta)
        gradient = x.T @ (probability - y) / len(y) + penalty * weights
        return value, np.asarray(gradient, dtype=float)

    fitted = minimize(
        objective,
        zero,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iterations), "ftol": 1e-11, "gtol": 1e-7},
    )
    if not np.isfinite(fitted.fun) or not np.isfinite(fitted.x).all():
        raise RuntimeError("Context residual optimization produced non-finite output")
    return OffsetLogisticFit(
        weights=np.asarray(fitted.x, dtype=float),
        shrinkage_lambda=penalty,
        converged=bool(fitted.success),
        iterations=int(fitted.nit),
        objective=float(fitted.fun),
        initialization_max_abs_error=initialization_error,
    )


def predict_offset_logistic(
    fit: OffsetLogisticFit,
    matrix: np.ndarray,
    base_logit,
    availability,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(matrix, dtype=float)
    offset = np.asarray(base_logit, dtype=float)
    available = np.asarray(availability, dtype=bool)
    if x.shape[1] != len(fit.weights) or x.shape[0] != len(offset):
        raise ValueError("Context residual prediction is not aligned")
    delta = x @ fit.weights
    delta = np.where(available, delta, 0.0)
    final_logit = offset + delta
    return delta, final_logit, logit_to_probability(final_logit)


def select_context_residual(
    validation: pd.DataFrame,
    *,
    minimum_delta_auprc: float = 0.01,
    maximum_brier_worsening: float = 0.01,
    maximum_ece_worsening: float = 0.01,
    minimum_positive_cancers: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select shrinkage and admit modules using pooled validation only."""

    required = [
        "candidate_id", "module", "target_subtype", "loco_cancer", "seed",
        "metric_scope", "residual_shrinkage_lambda", "proxy_label",
        "current_base_probability", "proxy_positive_probability",
    ]
    require_columns(validation, required, "context validation predictions")
    if not validation.metric_scope.astype(str).eq("validation").all():
        raise RuntimeError("Context selection may only use validation rows")
    unique = [
        "module", "target_subtype", "loco_cancer", "seed",
        "residual_shrinkage_lambda", "candidate_id",
    ]
    if validation.duplicated(unique).any():
        raise RuntimeError("Duplicate context validation candidates")

    pooled_rows: list[dict[str, Any]] = []
    cancer_rows: list[dict[str, Any]] = []
    groups = ["module", "target_subtype", "residual_shrinkage_lambda"]
    for keys, group in validation.groupby(groups, observed=True, sort=True):
        base = binary_metrics(group.proxy_label, group.current_base_probability)
        final = binary_metrics(group.proxy_label, group.proxy_positive_probability)
        pooled_rows.append(
            {
                **dict(zip(groups, keys)),
                "n": int(final["n"]),
                "positive_rate": float(final["positive_rate"]),
                "base_auprc": float(base["auprc"]),
                "final_auprc": float(final["auprc"]),
                "delta_auprc": float(final["auprc"] - base["auprc"]),
                "delta_auroc": float(final["auroc"] - base["auroc"]),
                "delta_brier": float(final["brier"] - base["brier"]),
                "delta_ece": float(final["ece"] - base["ece"]),
            }
        )
        for cancer, cancer_group in group.groupby("loco_cancer", observed=True, sort=True):
            seed_deltas: list[float] = []
            for _, seed_group in cancer_group.groupby("seed", observed=True, sort=True):
                seed_base = binary_metrics(
                    seed_group.proxy_label, seed_group.current_base_probability
                )
                seed_final = binary_metrics(
                    seed_group.proxy_label, seed_group.proxy_positive_probability
                )
                seed_deltas.append(float(seed_final["auprc"] - seed_base["auprc"]))
            cancer_rows.append(
                {
                    **dict(zip(groups, keys)),
                    "loco_cancer": str(cancer),
                    "mean_seed_delta_auprc": float(np.mean(seed_deltas)),
                    "positive_direction": bool(np.mean(seed_deltas) > 0),
                    "n_seeds": len(seed_deltas),
                }
            )
    pooled = pd.DataFrame(pooled_rows)
    per_cancer = pd.DataFrame(cancer_rows)
    positive = (
        per_cancer.groupby(groups, observed=True, sort=False).positive_direction.sum().rename("positive_cancers")
    )
    pooled = pooled.merge(positive.reset_index(), on=groups, validate="one_to_one")
    selections: list[dict[str, Any]] = []
    for keys, group in pooled.groupby(["module", "target_subtype"], observed=True, sort=True):
        selected = group.sort_values(
            ["delta_auprc", "delta_brier", "delta_ece", "residual_shrinkage_lambda"],
            ascending=[False, True, True, True],
        ).iloc[0]
        admitted = bool(
            np.isfinite(selected.delta_auprc)
            and selected.delta_auprc >= minimum_delta_auprc
            and selected.delta_brier <= maximum_brier_worsening
            and selected.delta_ece <= maximum_ece_worsening
            and int(selected.positive_cancers) >= int(minimum_positive_cancers)
        )
        selections.append(
            {
                "module": str(keys[0]),
                "target_subtype": str(keys[1]),
                "selected_lambda": float(selected.residual_shrinkage_lambda),
                "validation_delta_auprc": float(selected.delta_auprc),
                "validation_delta_auroc": float(selected.delta_auroc),
                "validation_delta_brier": float(selected.delta_brier),
                "validation_delta_ece": float(selected.delta_ece),
                "positive_cancers": int(selected.positive_cancers),
                "minimum_positive_cancers": int(minimum_positive_cancers),
                "minimum_delta_auprc": float(minimum_delta_auprc),
                "maximum_brier_worsening": float(maximum_brier_worsening),
                "maximum_ece_worsening": float(maximum_ece_worsening),
                "admitted": admitted,
                "decision": "PASS" if admitted else "OFF",
                "selection_scope": "pooled_outer_validation",
                "test_metric_used_for_selection": False,
            }
        )
    return pooled, per_cancer, pd.DataFrame(selections)


def exact_module_fallback(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn a rejected module off and assert exact identity to current base."""

    result = frame.copy()
    result["context_residual_logit"] = 0.0
    result["final_logit"] = result.current_base_logit.astype(float)
    result["proxy_positive_probability"] = result.current_base_probability.astype(float)
    assert_exact_fallback(result.current_base_logit, result.final_logit)
    error = float(
        np.max(
            np.abs(
                result.current_base_probability.to_numpy(float)
                - result.proxy_positive_probability.to_numpy(float)
            )
        )
    ) if len(result) else 0.0
    if error > FALLBACK_ATOL:
        raise RuntimeError(f"Context residual probability fallback failed: {error}")
    return result
