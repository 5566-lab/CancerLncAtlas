"""Patient-OOF distal-regulatory mutation to pathway association head.

The input regulatory component must already be an outer-fold prediction of
lncRNA expression.  This head fits a second, cancer-specific outer-fold ridge
regression from that held-out component to pathway activity.  It never fits on
the pathway activity of the patient being predicted and preserves typed
unavailability for missing assays or unsupported candidate pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


COMPONENT_COLUMN = "mutation_regulatory_expression_component_z"
COMPONENT_AVAILABLE = "mutation_regulatory_expression_component_available"
PATHWAY_TARGET = "activity_score"
FOLD_COLUMN = "patient_fold_id"
OUTER_FOLD_COLUMN = "outer_fold_id"
CANDIDATE_KEYS = ("cancer_id", "lncrna_id", "pathway_id")


class DistalRegulatoryMutationHeadError(RuntimeError):
    """Raised when the independent-head leakage or availability contract fails."""


@dataclass(frozen=True)
class DistalRegulatoryMutationHeadConfig:
    folds: int = 5
    ridge_alpha: float = 1.0
    min_train_patients: int = 20
    min_heldout_patients: int = 3
    min_oof_folds: int = 3
    min_feature_sd: float = 1e-8
    min_target_sd: float = 1e-8
    candidate_batch_size: int = 2048

    def __post_init__(self) -> None:
        if self.folds != 5:
            raise ValueError("The formal distal-regulatory head requires five folds")
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        if self.min_train_patients < 5 or self.min_heldout_patients < 1:
            raise ValueError("Invalid patient-count controls")
        if not 1 <= self.min_oof_folds <= self.folds:
            raise ValueError("min_oof_folds must be in 1..folds")
        if self.candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be positive")


def _normalise_components(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cancer_id",
        "patient_id",
        "lncrna_id",
        FOLD_COLUMN,
        COMPONENT_COLUMN,
        COMPONENT_AVAILABLE,
    }
    if missing := sorted(required - set(frame.columns)):
        raise DistalRegulatoryMutationHeadError(
            f"Regulatory component input lacks: {missing}"
        )
    nested_outer = OUTER_FOLD_COLUMN in frame.columns
    selected_columns = set(required)
    if nested_outer:
        selected_columns.add(OUTER_FOLD_COLUMN)
    data = frame.loc[:, sorted(selected_columns)].copy()
    data["cancer_id"] = data.cancer_id.astype(str).str.upper()
    data["patient_id"] = data.patient_id.astype(str).str[:12]
    data["lncrna_id"] = data.lncrna_id.astype(str)
    data[FOLD_COLUMN] = pd.to_numeric(data[FOLD_COLUMN], errors="coerce")
    if data[FOLD_COLUMN].isna().any() or not data[FOLD_COLUMN].isin(range(5)).all():
        raise DistalRegulatoryMutationHeadError("Patient folds must be exact 0..4")
    data[FOLD_COLUMN] = data[FOLD_COLUMN].astype(np.int8)
    if nested_outer:
        data[OUTER_FOLD_COLUMN] = pd.to_numeric(
            data[OUTER_FOLD_COLUMN], errors="coerce"
        )
        if data[OUTER_FOLD_COLUMN].isna().any() or not data[
            OUTER_FOLD_COLUMN
        ].isin(range(5)).all():
            raise DistalRegulatoryMutationHeadError(
                "Nested outer folds must be exact 0..4"
            )
        data[OUTER_FOLD_COLUMN] = data[OUTER_FOLD_COLUMN].astype(np.int8)
    data[COMPONENT_COLUMN] = pd.to_numeric(data[COMPONENT_COLUMN], errors="coerce")
    if data[COMPONENT_AVAILABLE].isna().any():
        raise DistalRegulatoryMutationHeadError("Component availability contains null")
    data[COMPONENT_AVAILABLE] = data[COMPONENT_AVAILABLE].astype(bool)
    finite = np.isfinite(data[COMPONENT_COLUMN].to_numpy(float))
    if not np.array_equal(finite, data[COMPONENT_AVAILABLE].to_numpy(bool)):
        raise DistalRegulatoryMutationHeadError(
            "Regulatory component violates typed-null availability"
        )
    component_keys = ["cancer_id", "patient_id", "lncrna_id"]
    if nested_outer:
        component_keys.append(OUTER_FOLD_COLUMN)
    if data.duplicated(component_keys).any():
        raise DistalRegulatoryMutationHeadError("Duplicate regulatory component key")
    patient_folds = data.groupby(
        ["cancer_id", "patient_id"], observed=True
    )[FOLD_COLUMN].nunique()
    if patient_folds.gt(1).any():
        raise DistalRegulatoryMutationHeadError("One patient occurs in multiple folds")
    if nested_outer:
        outer_counts = data.groupby(
            ["cancer_id", "patient_id", "lncrna_id"], observed=True
        )[OUTER_FOLD_COLUMN].nunique()
        if not outer_counts.eq(5).all():
            raise DistalRegulatoryMutationHeadError(
                "Every nested component key must cover all five outer folds"
            )
    return data


def _normalise_activity(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"cancer_id", "patient_id", "pathway_id", PATHWAY_TARGET}
    if missing := sorted(required - set(frame.columns)):
        raise DistalRegulatoryMutationHeadError(f"Pathway activity input lacks: {missing}")
    data = frame.loc[:, sorted(required)].copy()
    data["cancer_id"] = data.cancer_id.astype(str).str.upper()
    data["patient_id"] = data.patient_id.astype(str).str[:12]
    data["pathway_id"] = data.pathway_id.astype(str)
    data[PATHWAY_TARGET] = pd.to_numeric(data[PATHWAY_TARGET], errors="coerce")
    data = data.loc[np.isfinite(data[PATHWAY_TARGET].to_numpy(float))]
    # Multiple primary-tumour aliquots are not independent patients.  Average
    # aliquots first so every patient has equal weight in the OOF evaluation.
    return (
        data.groupby(
            ["cancer_id", "patient_id", "pathway_id"], observed=True, sort=False
        )[PATHWAY_TARGET]
        .mean()
        .reset_index()
    )


def _normalise_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if missing := sorted(set(CANDIDATE_KEYS) - set(frame.columns)):
        raise DistalRegulatoryMutationHeadError(f"Candidate input lacks: {missing}")
    data = frame.loc[:, list(CANDIDATE_KEYS)].copy()
    data["cancer_id"] = data.cancer_id.astype(str).str.upper()
    data["lncrna_id"] = data.lncrna_id.astype(str)
    data["pathway_id"] = data.pathway_id.astype(str)
    if data.duplicated(list(CANDIDATE_KEYS)).any():
        raise DistalRegulatoryMutationHeadError("Candidate input has duplicate keys")
    return data.reset_index(drop=True)


def _selected_moments(
    x: np.ndarray,
    y: np.ndarray,
    lnc_index: np.ndarray,
    pathway_index: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Return pairwise moments only for selected lncRNA-pathway pairs."""

    size = len(lnc_index)
    result = {
        "n": np.zeros(size, dtype=np.int64),
        "sx": np.zeros(size, dtype=np.float64),
        "sy": np.zeros(size, dtype=np.float64),
        "sx2": np.zeros(size, dtype=np.float64),
        "sy2": np.zeros(size, dtype=np.float64),
        "sxy": np.zeros(size, dtype=np.float64),
    }
    for start in range(0, size, batch_size):
        stop = min(size, start + batch_size)
        xv = x[:, lnc_index[start:stop]]
        yv = y[:, pathway_index[start:stop]]
        observed = np.isfinite(xv) & np.isfinite(yv)
        x0 = np.where(observed, xv, 0.0)
        y0 = np.where(observed, yv, 0.0)
        result["n"][start:stop] = observed.sum(axis=0, dtype=np.int64)
        result["sx"][start:stop] = x0.sum(axis=0)
        result["sy"][start:stop] = y0.sum(axis=0)
        result["sx2"][start:stop] = np.square(x0).sum(axis=0)
        result["sy2"][start:stop] = np.square(y0).sum(axis=0)
        result["sxy"][start:stop] = (x0 * y0).sum(axis=0)
    return result


def _safe_correlation(
    n: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sx2: np.ndarray,
    sy2: np.ndarray,
    sxy: np.ndarray,
) -> np.ndarray:
    correlation = np.full(len(n), np.nan, dtype=np.float64)
    valid_n = n > 1
    covariance = sxy - np.divide(sx * sy, n, out=np.zeros_like(sxy), where=n > 0)
    variance_x = sx2 - np.divide(sx * sx, n, out=np.zeros_like(sx2), where=n > 0)
    variance_y = sy2 - np.divide(sy * sy, n, out=np.zeros_like(sy2), where=n > 0)
    denominator = np.sqrt(np.maximum(variance_x, 0.0) * np.maximum(variance_y, 0.0))
    valid = valid_n & np.isfinite(denominator) & (denominator > 0)
    correlation[valid] = covariance[valid] / denominator[valid]
    return np.clip(correlation, -1.0, 1.0)


def crossfit_distal_regulatory_pathway_head(
    components: pd.DataFrame,
    activity: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    config: DistalRegulatoryMutationHeadConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the independent second-stage patient-OOF pathway head."""

    settings = config or DistalRegulatoryMutationHeadConfig()
    component = _normalise_components(components)
    nested_outer = OUTER_FOLD_COLUMN in component.columns
    pathway = _normalise_activity(activity)
    candidate = _normalise_candidates(candidates)
    output = candidate.copy()
    output["distal_regulatory_mutation_pathway_available"] = False
    output["distal_regulatory_mutation_pathway_unavailable_reason"] = (
        "NO_DISTAL_LINKED_LNCRNA_OR_PATHWAY_ACTIVITY"
    )
    for column in (
        "oof_rows",
        "oof_folds_pass",
        "mean_oof_slope",
        "oof_rmse",
        "oof_baseline_rmse",
        "oof_delta_mse",
        "oof_r2",
        "oof_prediction_correlation",
    ):
        output[column] = np.nan

    cancer_audits: list[dict[str, Any]] = []
    for cancer, local_candidate in candidate.groupby("cancer_id", observed=True, sort=True):
        local_component = component.loc[component.cancer_id.eq(cancer)]
        local_activity = pathway.loc[pathway.cancer_id.eq(cancer)]
        audit: dict[str, Any] = {
            "cancer_id": str(cancer),
            "candidate_rows": int(len(local_candidate)),
            "status": "TYPED_UNAVAILABLE",
            "nested_outer_crossfit": nested_outer,
        }
        if local_component.empty or local_activity.empty:
            audit["reason"] = "NO_COMPONENT_OR_PATHWAY_ACTIVITY"
            cancer_audits.append(audit)
            continue

        fold_map = local_component[["patient_id", FOLD_COLUMN]].drop_duplicates()
        patients = sorted(
            set(fold_map.patient_id.astype(str))
            & set(local_activity.patient_id.astype(str))
        )
        lnc_values = sorted(
            set(local_candidate.lncrna_id.astype(str))
            & set(local_component.lncrna_id.astype(str))
        )
        pathway_values = sorted(
            set(local_candidate.pathway_id.astype(str))
            & set(local_activity.pathway_id.astype(str))
        )
        if not patients or not lnc_values or not pathway_values:
            audit["reason"] = "NO_SHARED_PATIENT_LNCRNA_PATHWAY_SCOPE"
            cancer_audits.append(audit)
            continue

        fold_series = (
            fold_map.set_index("patient_id")[FOLD_COLUMN]
            .reindex(patients)
            .astype(np.int8)
        )
        def component_matrix(outer_fold: int | None = None) -> np.ndarray:
            selected_component = local_component
            if outer_fold is not None:
                selected_component = selected_component.loc[
                    selected_component[OUTER_FOLD_COLUMN].eq(outer_fold)
                ]
            return (
                selected_component.loc[
                    selected_component.patient_id.isin(patients)
                    & selected_component.lncrna_id.isin(lnc_values),
                    ["patient_id", "lncrna_id", COMPONENT_COLUMN],
                ]
                .pivot(
                    index="patient_id", columns="lncrna_id", values=COMPONENT_COLUMN
                )
                .reindex(index=patients, columns=lnc_values)
                .to_numpy(float)
            )

        static_x = None if nested_outer else component_matrix()
        y = (
            local_activity.loc[
                local_activity.patient_id.isin(patients)
                & local_activity.pathway_id.isin(pathway_values),
                ["patient_id", "pathway_id", PATHWAY_TARGET],
            ]
            .pivot(index="patient_id", columns="pathway_id", values=PATHWAY_TARGET)
            .reindex(index=patients, columns=pathway_values)
            .to_numpy(float)
        )
        lnc_lookup = {value: index for index, value in enumerate(lnc_values)}
        pathway_lookup = {value: index for index, value in enumerate(pathway_values)}
        eligible = local_candidate.lncrna_id.isin(lnc_lookup) & local_candidate.pathway_id.isin(
            pathway_lookup
        )
        selected = local_candidate.loc[eligible]
        destination = selected.index.to_numpy(int)
        li = selected.lncrna_id.map(lnc_lookup).to_numpy(int)
        pi = selected.pathway_id.map(pathway_lookup).to_numpy(int)
        size = len(selected)
        if size == 0:
            audit["reason"] = "NO_ELIGIBLE_CANDIDATE_PAIR"
            cancer_audits.append(audit)
            continue

        oof_n = np.zeros(size, dtype=np.int64)
        oof_sse = np.zeros(size, dtype=np.float64)
        oof_baseline_sse = np.zeros(size, dtype=np.float64)
        oof_sum_prediction = np.zeros(size, dtype=np.float64)
        oof_sum_prediction2 = np.zeros(size, dtype=np.float64)
        oof_sum_target = np.zeros(size, dtype=np.float64)
        oof_sum_target2 = np.zeros(size, dtype=np.float64)
        oof_sum_prediction_target = np.zeros(size, dtype=np.float64)
        slope_weighted_sum = np.zeros(size, dtype=np.float64)
        fold_pass = np.zeros(size, dtype=np.int8)
        fold_audits: list[dict[str, Any]] = []

        fold_values = fold_series.to_numpy(np.int8)
        for fold in range(settings.folds):
            x = component_matrix(fold) if nested_outer else static_x
            if x is None:
                raise DistalRegulatoryMutationHeadError("Component matrix is unavailable")
            train_rows = fold_values != fold
            heldout_rows = fold_values == fold
            train = _selected_moments(
                x[train_rows], y[train_rows], li, pi,
                batch_size=settings.candidate_batch_size,
            )
            heldout = _selected_moments(
                x[heldout_rows], y[heldout_rows], li, pi,
                batch_size=settings.candidate_batch_size,
            )
            n_train = train["n"].astype(np.float64)
            x_mean = np.divide(train["sx"], n_train, out=np.zeros(size), where=n_train > 0)
            y_mean = np.divide(train["sy"], n_train, out=np.zeros(size), where=n_train > 0)
            sxx = train["sx2"] - np.divide(
                train["sx"] ** 2, n_train, out=np.zeros(size), where=n_train > 0
            )
            syy = train["sy2"] - np.divide(
                train["sy"] ** 2, n_train, out=np.zeros(size), where=n_train > 0
            )
            sxy = train["sxy"] - np.divide(
                train["sx"] * train["sy"], n_train,
                out=np.zeros(size), where=n_train > 0,
            )
            valid = (
                (train["n"] >= settings.min_train_patients)
                & (heldout["n"] >= settings.min_heldout_patients)
                & (sxx >= settings.min_feature_sd**2 * np.maximum(n_train, 1.0))
                & (syy >= settings.min_target_sd**2 * np.maximum(n_train, 1.0))
            )
            slope = np.zeros(size, dtype=np.float64)
            slope[valid] = sxy[valid] / (sxx[valid] + settings.ridge_alpha)
            intercept = y_mean - slope * x_mean
            nh = heldout["n"].astype(np.float64)
            prediction_sum = nh * intercept + slope * heldout["sx"]
            prediction2_sum = (
                nh * intercept**2
                + 2.0 * intercept * slope * heldout["sx"]
                + slope**2 * heldout["sx2"]
            )
            prediction_target_sum = (
                intercept * heldout["sy"] + slope * heldout["sxy"]
            )
            sse = (
                heldout["sy2"]
                - 2.0 * prediction_target_sum
                + prediction2_sum
            )
            baseline_sse = (
                heldout["sy2"]
                - 2.0 * y_mean * heldout["sy"]
                + nh * y_mean**2
            )
            oof_n[valid] += heldout["n"][valid]
            oof_sse[valid] += np.maximum(sse[valid], 0.0)
            oof_baseline_sse[valid] += np.maximum(baseline_sse[valid], 0.0)
            oof_sum_prediction[valid] += prediction_sum[valid]
            oof_sum_prediction2[valid] += prediction2_sum[valid]
            oof_sum_target[valid] += heldout["sy"][valid]
            oof_sum_target2[valid] += heldout["sy2"][valid]
            oof_sum_prediction_target[valid] += prediction_target_sum[valid]
            slope_weighted_sum[valid] += slope[valid] * heldout["n"][valid]
            fold_pass[valid] += 1
            fold_audits.append(
                {
                    "patient_fold_id": fold,
                    "train_patients": int(train_rows.sum()),
                    "heldout_patients": int(heldout_rows.sum()),
                    "candidate_pairs_pass": int(valid.sum()),
                }
            )

        available = (fold_pass >= settings.min_oof_folds) & (oof_n > 0)
        local_result = pd.DataFrame(index=destination)
        local_result["oof_rows"] = oof_n.astype(float)
        local_result["oof_folds_pass"] = fold_pass.astype(float)
        local_result["mean_oof_slope"] = np.divide(
            slope_weighted_sum, oof_n,
            out=np.full(size, np.nan), where=oof_n > 0,
        )
        local_result["oof_rmse"] = np.sqrt(
            np.divide(oof_sse, oof_n, out=np.full(size, np.nan), where=oof_n > 0)
        )
        local_result["oof_baseline_rmse"] = np.sqrt(
            np.divide(
                oof_baseline_sse, oof_n,
                out=np.full(size, np.nan), where=oof_n > 0,
            )
        )
        local_result["oof_delta_mse"] = np.divide(
            oof_baseline_sse - oof_sse, oof_n,
            out=np.full(size, np.nan), where=oof_n > 0,
        )
        local_result["oof_r2"] = 1.0 - np.divide(
            oof_sse, oof_baseline_sse,
            out=np.full(size, np.nan), where=oof_baseline_sse > 0,
        )
        local_result["oof_prediction_correlation"] = _safe_correlation(
            oof_n.astype(float),
            oof_sum_prediction,
            oof_sum_target,
            oof_sum_prediction2,
            oof_sum_target2,
            oof_sum_prediction_target,
        )
        effect_columns = [
            "mean_oof_slope",
            "oof_rmse",
            "oof_baseline_rmse",
            "oof_delta_mse",
            "oof_r2",
            "oof_prediction_correlation",
        ]
        available &= np.isfinite(local_result[effect_columns].to_numpy(float)).all(axis=1)
        metric_columns = list(local_result.columns)
        output.loc[destination, metric_columns] = local_result[metric_columns]
        passed_destination = destination[available]
        output.loc[
            passed_destination, "distal_regulatory_mutation_pathway_available"
        ] = True
        output.loc[
            passed_destination,
            "distal_regulatory_mutation_pathway_unavailable_reason",
        ] = None
        failed_destination = destination[~available]
        output.loc[failed_destination, effect_columns] = np.nan
        output.loc[
            failed_destination,
            "distal_regulatory_mutation_pathway_unavailable_reason",
        ] = "INSUFFICIENT_FIVE_FOLD_OOF_SUPPORT"
        audit.update(
            {
                "status": "PASS_PATIENT_OOF" if available.any() else "TYPED_UNAVAILABLE",
                "shared_patients": len(patients),
                "eligible_candidate_pairs": size,
                "available_candidate_pairs": int(available.sum()),
                "folds": fold_audits,
            }
        )
        cancer_audits.append(audit)

    available = output.distal_regulatory_mutation_pathway_available.to_numpy(bool)
    metric_columns = (
        "oof_rows",
        "oof_folds_pass",
        "mean_oof_slope",
        "oof_rmse",
        "oof_baseline_rmse",
        "oof_delta_mse",
        "oof_r2",
        "oof_prediction_correlation",
    )
    if output.loc[available, list(metric_columns)].isna().any(axis=None):
        raise DistalRegulatoryMutationHeadError("Available head rows contain null metrics")
    if output.loc[available, "distal_regulatory_mutation_pathway_unavailable_reason"].notna().any():
        raise DistalRegulatoryMutationHeadError("Available head row has unavailable reason")
    audit = {
        "format": "CANCERLNCATLAS_V32_DISTAL_REGULATORY_MUTATION_PATHWAY_NESTED_OOF_V2",
        "status": "PASS" if available.any() else "TYPED_UNAVAILABLE_NO_OOF_HEAD",
        "candidate_rows": int(len(output)),
        "available_candidate_rows": int(available.sum()),
        "cancers": int(output.cancer_id.nunique()),
        "patient_folds": settings.folds,
        "expression_component_source": (
            "PATIENT_OOF_FULL_MUTATION_PLUS_COVARIATES_MINUS_COVARIATES_ONLY"
        ),
        "heldout_lncrna_expression_used_for_own_component_fit": False,
        "nested_outer_crossfit": nested_outer,
        "outer_heldout_lncrna_expression_used_for_any_training_component_fit": False,
        "outer_training_components_are_inner_oof": nested_outer,
        "heldout_pathway_activity_used_for_own_prediction_fit": False,
        "missing_assay_assumed_zero": False,
        "causal_claimed": False,
        "cancer_audits": cancer_audits,
    }
    return output, audit


__all__ = [
    "DistalRegulatoryMutationHeadConfig",
    "DistalRegulatoryMutationHeadError",
    "crossfit_distal_regulatory_pathway_head",
]
