"""Leakage-safe V3.2 entity-level clinical survival associations.

The patient-risk head and an entity/outcome association answer different
questions.  This module restores the latter for lncRNAs, literal exact
pathways, and the seven historical State measurements without importing any
V3.0 checkpoint, probability, rank, or web table.

Each outer fold uses three patient folds for train-only preprocessing, the
next fold for validation, and the outer fold as an untouched OOF test set.
Associations are the one-step Newton estimate from the Breslow Cox score at
the null.  The implementation is block-vectorised so the complete entity
universe can be evaluated without fitting millions of Python model objects.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr

from cc_hhgt.common import file_sha256

from .clinical_training import (
    CLINICAL_ENDPOINTS,
    encode_fold_covariates,
    patient_fold_manifest,
    standardize_tcga_cdr_workbook,
)
from .state_training import HISTORICAL_STATE_IDS


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_CLINICAL_ENTITY_ASSOCIATION"
CALCULATION_FORMAT = "CC_HHGT_V3_2_ENTITY_CLINICAL_COX_SCORE_V1"
MODELED_ENDPOINTS = tuple(endpoint for endpoint in CLINICAL_ENDPOINTS if endpoint != "DFS")
SUBJECT_SPECS: Mapping[str, tuple[str, str]] = {
    "lncRNA": ("lncrna_id", "logcpm"),
    "exact_pathway": ("pathway_id", "activity_score"),
    "state": ("state_id", "state_value"),
}


class ClinicalEntityTrainingError(RuntimeError):
    """Raised when entity clinical inputs or split rules are invalid."""


def _explicit_patient_ids(values: Sequence[Any] | pd.Series, *, context: str) -> pd.Series:
    """Preserve explicit patient keys; whitespace trimming is the only normalization."""

    result = pd.Series(values, dtype="string")
    if result.isna().any():
        raise ClinicalEntityTrainingError(f"{context} contains null explicit patient_id")
    result = result.str.strip()
    if result.eq("").any():
        raise ClinicalEntityTrainingError(f"{context} contains empty explicit patient_id")
    return result.astype(str)


def validate_entity_clinical_contract(
    summary: pd.DataFrame, lineage: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed on generation, typed availability, DFS, and ranking rules."""

    required = {
        "cancer_id", "subject_type", "subject_id", "clinical_endpoint",
        "clinical_relevance_probability", "availability", "failure_reason",
        "meta_beta", "hazard_ratio", "p_value", "fdr", "model_version",
        "old_checkpoint_loaded", "old_predictions_used_as_features",
        "changes_primary_ranking",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ClinicalEntityTrainingError(f"Entity clinical summary lacks columns: {missing}")
    keys = ["cancer_id", "subject_type", "subject_id", "clinical_endpoint"]
    if summary.duplicated(keys).any():
        raise ClinicalEntityTrainingError("Entity clinical summary contains duplicate typed keys")
    if not set(summary.subject_type.astype(str)).issubset(set(SUBJECT_SPECS)):
        raise ClinicalEntityTrainingError("Entity clinical summary has unknown subject_type")
    if set(summary.clinical_endpoint.astype(str)) != set(CLINICAL_ENDPOINTS):
        raise ClinicalEntityTrainingError("Entity clinical endpoint coverage is incomplete")
    available = summary.availability.astype(bool)
    probability = pd.to_numeric(summary.clinical_relevance_probability, errors="coerce")
    for column in ("meta_beta", "hazard_ratio", "p_value", "fdr"):
        values = pd.to_numeric(summary[column], errors="coerce")
        if not np.isfinite(values.loc[available]).all():
            raise ClinicalEntityTrainingError(f"Available rows have invalid {column}")
    if not probability.loc[available].between(0.0, 1.0).all():
        raise ClinicalEntityTrainingError("Available clinical relevance scores are outside [0,1]")
    if probability.loc[~available].notna().any():
        raise ClinicalEntityTrainingError("Unavailable rows contain a non-null clinical relevance score")
    if summary.loc[~available, "failure_reason"].fillna("").astype(str).str.strip().eq("").any():
        raise ClinicalEntityTrainingError("Unavailable rows lack a failure reason")
    dfs = summary.clinical_endpoint.astype(str).eq("DFS")
    if summary.loc[dfs, "availability"].astype(bool).any():
        raise ClinicalEntityTrainingError("DFS was made available without a distinct source")
    if summary.loc[dfs, "failure_reason"].ne("NO_DISTINCT_DFS_SOURCE").any():
        raise ClinicalEntityTrainingError("DFS null reason drifted")
    if summary.loc[dfs, ["meta_beta", "hazard_ratio", "p_value", "fdr"]].notna().any().any():
        raise ClinicalEntityTrainingError("DFS contains non-null association statistics")
    for column in ("old_checkpoint_loaded", "old_predictions_used_as_features", "changes_primary_ranking"):
        if summary[column].astype(bool).any():
            raise ClinicalEntityTrainingError(f"Forbidden public flag is true: {column}")
    if not str(lineage.get("analysis_version", "")).startswith("CancerLncAtlas_V3.2"):
        raise ClinicalEntityTrainingError("Entity clinical lineage is not V3.2")
    if lineage.get("fresh_statistical_calculation") is not True:
        raise ClinicalEntityTrainingError("Lineage lacks fresh V3.2 calculation attestation")
    for field in ("old_checkpoint_loaded", "old_predictions_used_as_features", "old_rankings_used_as_outputs"):
        if lineage.get(field) is not False:
            raise ClinicalEntityTrainingError(f"Lineage does not exclude historical results: {field}")
    if lineage.get("changes_primary_ranking") is not False:
        raise ClinicalEntityTrainingError("Clinical entity extension may not change primary ranking")
    if lineage.get("full_entity_universe") is True:
        expected = int(lineage.get("expected_summary_rows_from_candidate_universe", -1))
        if expected != len(summary) or lineage.get("all_candidate_entities_enumerated") is not True:
            raise ClinicalEntityTrainingError("Candidate entity universe is not fully enumerated")
    return {
        "status": "PASS",
        "contract": CALCULATION_FORMAT,
        "rows": int(len(summary)),
        "available_rows": int(available.sum()),
        "unavailable_rows": int((~available).sum()),
        "typed_keys_unique": True,
        "all_six_endpoints_present": True,
        "dfs_all_null_with_reason": True,
        "all_non_null_results_newly_computed_v32": True,
        "historical_results_used": False,
        "changes_primary_ranking": False,
    }


def fold_roles(test_fold: int) -> dict[str, tuple[int, ...]]:
    """Return the fixed 3-train/1-validation/1-OOF-test allocation."""

    fold = int(test_fold)
    if fold not in range(5):
        raise ClinicalEntityTrainingError("test_fold must be in 0..4")
    validation = (fold + 1) % 5
    train = tuple(value for value in range(5) if value not in {fold, validation})
    return {"train": train, "validation": (validation,), "test": (fold,)}


def _normal_two_sided_p(z: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * ndtr(-np.abs(np.asarray(z, dtype=float))), 0.0, 1.0)


def cox_null_score_one_step(
    features: np.ndarray,
    time_days: np.ndarray,
    event: np.ndarray,
    *,
    min_patients: int,
    min_events: int,
) -> dict[str, Any]:
    """Compute vectorised Breslow Cox score and one-step beta at beta=0.

    Tied event times share the same risk set.  ``features`` must already have
    been transformed using train-only parameters when this function is used
    on validation or test patients.
    """

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ClinicalEntityTrainingError("features must be a two-dimensional matrix")
    time = np.asarray(time_days, dtype=float)
    status = np.asarray(event, dtype=float)
    valid_rows = np.isfinite(time) & (time > 0) & np.isin(status, [0.0, 1.0])
    x = x[valid_rows]
    time = time[valid_rows]
    status = status[valid_rows]
    n_patients = int(len(time))
    n_events = int(np.sum(status == 1.0))
    empty = np.full(x.shape[1], np.nan, dtype=np.float64)
    if n_patients < int(min_patients):
        return {
            "beta": empty.copy(), "standard_error": empty.copy(), "z": empty.copy(),
            "p_value": empty.copy(), "information": empty.copy(),
            "estimable": np.zeros(x.shape[1], dtype=bool), "n_patients": n_patients,
            "n_events": n_events, "failure_reason": "INSUFFICIENT_PATIENTS",
        }
    if n_events < int(min_events):
        return {
            "beta": empty.copy(), "standard_error": empty.copy(), "z": empty.copy(),
            "p_value": empty.copy(), "information": empty.copy(),
            "estimable": np.zeros(x.shape[1], dtype=bool), "n_patients": n_patients,
            "n_events": n_events, "failure_reason": "INSUFFICIENT_EVENTS",
        }

    # Descending time makes each risk set a prefix.  The last member of a tied
    # group therefore contains the complete risk set for that event time.
    order = np.argsort(-time, kind="stable")
    ordered_time = time[order]
    ordered_event = status[order]
    ordered_x = x[order]
    group_end = np.r_[ordered_time[:-1] != ordered_time[1:], True]
    end_positions = np.flatnonzero(group_end)
    starts = np.r_[0, end_positions[:-1] + 1]
    event_counts = np.add.reduceat(ordered_event, starts)
    event_groups = event_counts > 0
    end_positions = end_positions[event_groups]
    event_counts = event_counts[event_groups]
    risk_counts = (end_positions + 1).astype(np.float64)

    cumulative_x = np.cumsum(ordered_x, axis=0, dtype=np.float64)
    cumulative_x2 = np.cumsum(ordered_x * ordered_x, axis=0, dtype=np.float64)
    risk_sum = cumulative_x[end_positions]
    risk_sum2 = cumulative_x2[end_positions]
    event_sum = ordered_x[ordered_event == 1.0].sum(axis=0, dtype=np.float64)
    risk_mean = risk_sum / risk_counts[:, None]
    score = event_sum - (event_counts[:, None] * risk_mean).sum(axis=0)
    information = (
        event_counts[:, None]
        * (risk_sum2 / risk_counts[:, None] - risk_mean * risk_mean)
    ).sum(axis=0)
    estimable = np.isfinite(information) & (information > 1e-10)
    beta = np.full(x.shape[1], np.nan, dtype=np.float64)
    standard_error = beta.copy()
    z = beta.copy()
    beta[estimable] = score[estimable] / information[estimable]
    standard_error[estimable] = 1.0 / np.sqrt(information[estimable])
    z[estimable] = score[estimable] / np.sqrt(information[estimable])
    p_value = _normal_two_sided_p(z)
    p_value[~estimable] = np.nan
    return {
        "beta": beta,
        "standard_error": standard_error,
        "z": z,
        "p_value": p_value,
        "information": information,
        "estimable": estimable,
        "n_patients": n_patients,
        "n_events": n_events,
        "failure_reason": "" if bool(estimable.any()) else "ZERO_COX_SCORE_INFORMATION",
    }


def fold_local_residualize(
    values: np.ndarray,
    covariates: np.ndarray,
    train_mask: np.ndarray,
    *,
    min_train_measurements: int,
    ridge: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Train-only imputation/z-scoring and linear covariate residualisation."""

    matrix = np.asarray(values, dtype=np.float64)
    cov = np.asarray(covariates, dtype=np.float64)
    train = np.asarray(train_mask, dtype=bool)
    if matrix.ndim != 2 or cov.ndim != 2 or len(matrix) != len(cov):
        raise ClinicalEntityTrainingError("Molecular values and covariates are misaligned")
    if int(train.sum()) < 2:
        raise ClinicalEntityTrainingError("Fewer than two patients in training folds")
    observed = np.isfinite(matrix)
    train_values = matrix[train]
    train_observed = observed[train]
    n_train_measurements = train_observed.sum(axis=0).astype(np.int32)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(train_values, axis=0) / np.maximum(n_train_measurements, 1)
        centered = np.where(train_observed, train_values - mean, 0.0)
        variance = np.sum(centered * centered, axis=0) / np.maximum(n_train_measurements, 1)
    scale = np.sqrt(np.maximum(variance, 0.0))
    feature_valid = (
        (n_train_measurements >= int(min_train_measurements))
        & np.isfinite(mean)
        & np.isfinite(scale)
        & (scale > 1e-8)
    )
    safe_mean = np.where(np.isfinite(mean), mean, 0.0)
    safe_scale = np.where(feature_valid, scale, 1.0)
    filled = np.where(observed, matrix, safe_mean)
    standardized = (filled - safe_mean) / safe_scale
    standardized[:, ~feature_valid] = 0.0

    design = np.column_stack([np.ones(len(cov), dtype=np.float64), cov])
    design = np.where(np.isfinite(design), design, 0.0)
    train_design = design[train]
    penalty = np.eye(train_design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(
        train_design.T @ train_design + penalty,
        train_design.T @ standardized[train],
    )
    residual = standardized - design @ coefficient
    residual[:, ~feature_valid] = 0.0
    counts = {
        "train": n_train_measurements,
        "all": observed.sum(axis=0).astype(np.int32),
    }
    return residual.astype(np.float32), feature_valid, counts


def _role_failure(role: str, result: Mapping[str, Any]) -> str:
    reason = str(result.get("failure_reason", "")).strip()
    return f"{role.upper()}_{reason}" if reason else ""


def analyse_entity_matrix(
    *,
    cancer_id: str,
    subject_type: str,
    patient_ids: Sequence[str],
    entity_ids: Sequence[str],
    values: np.ndarray,
    fold_ids: np.ndarray,
    endpoint_frame: pd.DataFrame,
    covariate_frame: pd.DataFrame,
    block_size: int = 256,
    min_train_patients: int = 30,
    min_train_events: int = 5,
    min_validation_patients: int = 10,
    min_validation_events: int = 2,
    min_test_patients: int = 10,
    min_test_events: int = 2,
    min_train_measurements: int = 20,
) -> pd.DataFrame:
    """Run all five patient folds for one cancer and one subject matrix."""

    if subject_type not in SUBJECT_SPECS:
        raise ClinicalEntityTrainingError(f"Unsupported subject_type: {subject_type}")
    patient = pd.Index(
        _explicit_patient_ids(patient_ids, context="Entity matrix").tolist(),
        name="patient_id",
    )
    entities = np.asarray([str(value) for value in entity_ids], dtype=object)
    matrix = np.asarray(values, dtype=np.float64)
    folds = np.asarray(fold_ids, dtype=int)
    if matrix.shape != (len(patient), len(entities)):
        raise ClinicalEntityTrainingError("Entity matrix dimensions do not match IDs")
    if not set(np.unique(folds)).issubset(set(range(5))):
        raise ClinicalEntityTrainingError("Patient fold IDs must be in 0..4")
    if patient.duplicated().any():
        raise ClinicalEntityTrainingError("Entity matrix contains duplicate patients")

    endpoint = endpoint_frame.copy()
    if "patient_id" not in endpoint:
        raise ClinicalEntityTrainingError("Endpoint frame lacks explicit patient_id")
    endpoint["patient_id"] = _explicit_patient_ids(
        endpoint.patient_id, context="Endpoint frame"
    ).to_numpy()
    endpoint = endpoint.loc[endpoint.cancer_id.astype(str).eq(str(cancer_id))]
    endpoint_lookup = endpoint.set_index(["clinical_endpoint", "patient_id"])
    cov = covariate_frame.copy()
    if "patient_id" not in cov:
        raise ClinicalEntityTrainingError("Covariate frame lacks explicit patient_id")
    cov["patient_id"] = _explicit_patient_ids(
        cov.patient_id, context="Covariate frame"
    ).to_numpy()
    cov = cov.loc[cov.cancer_id.astype(str).eq(str(cancer_id))]
    cov = pd.DataFrame({"patient_id": patient}).merge(
        cov.drop_duplicates("patient_id"), on="patient_id", how="left", validate="one_to_one"
    )

    chunks: list[pd.DataFrame] = []
    for test_fold in range(5):
        roles = fold_roles(test_fold)
        train_mask = np.isin(folds, roles["train"])
        validation_mask = np.isin(folds, roles["validation"])
        test_mask = np.isin(folds, roles["test"])
        if (
            bool((train_mask & validation_mask).any())
            or bool((train_mask & test_mask).any())
            or bool((validation_mask & test_mask).any())
        ):
            raise ClinicalEntityTrainingError("Patient leakage across fold roles")
        encoded = encode_fold_covariates(cov, patient[train_mask].tolist()).set_index("patient_id")
        encoded = encoded.reindex(patient)
        cov_columns = [column for column in encoded if column.startswith("domain_")]
        cov_matrix = encoded[cov_columns].to_numpy(np.float64)

        endpoint_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for clinical_endpoint in MODELED_ENDPOINTS:
            try:
                table = endpoint_lookup.loc[clinical_endpoint]
            except KeyError:
                table = pd.DataFrame(index=pd.Index([], name="patient_id"))
            table = table.reindex(patient)
            endpoint_arrays[clinical_endpoint] = (
                pd.to_numeric(table.get("time_days"), errors="coerce").to_numpy(float),
                pd.to_numeric(table.get("event"), errors="coerce").to_numpy(float),
            )

        for start in range(0, len(entities), int(block_size)):
            stop = min(start + int(block_size), len(entities))
            block_ids = entities[start:stop]
            block_values = matrix[:, start:stop]
            residual, feature_valid, counts = fold_local_residualize(
                block_values,
                cov_matrix,
                train_mask,
                min_train_measurements=min_train_measurements,
            )
            observed = np.isfinite(block_values)
            role_measurements = {
                "train": observed[train_mask].sum(axis=0).astype(np.int32),
                "validation": observed[validation_mask].sum(axis=0).astype(np.int32),
                "test": observed[test_mask].sum(axis=0).astype(np.int32),
            }
            for clinical_endpoint in CLINICAL_ENDPOINTS:
                base = pd.DataFrame(
                    {
                        "cancer_id": str(cancer_id),
                        "subject_type": str(subject_type),
                        "subject_id": block_ids,
                        "clinical_endpoint": clinical_endpoint,
                        "patient_fold_id": np.int8(test_fold),
                        "validation_fold_id": np.int8(roles["validation"][0]),
                        "train_fold_ids": ",".join(map(str, roles["train"])),
                        "n_train_measurements": role_measurements["train"],
                        "n_validation_measurements": role_measurements["validation"],
                        "n_test_measurements": role_measurements["test"],
                    }
                )
                if clinical_endpoint == "DFS":
                    for prefix in ("train", "validation", "test"):
                        for statistic in ("beta", "standard_error", "z", "p_value"):
                            base[f"{prefix}_{statistic}"] = np.nan
                        base[f"n_{prefix}_patients"] = 0
                        base[f"n_{prefix}_events"] = 0
                    base["fold_available"] = False
                    base["failure_reason"] = "NO_DISTINCT_DFS_SOURCE"
                    base["validation_sign_concordant"] = pd.array(
                        [pd.NA] * len(base), dtype="boolean"
                    )
                    chunks.append(base)
                    continue

                time, event = endpoint_arrays[clinical_endpoint]
                results = {
                    "train": cox_null_score_one_step(
                        residual[train_mask], time[train_mask], event[train_mask],
                        min_patients=min_train_patients, min_events=min_train_events,
                    ),
                    "validation": cox_null_score_one_step(
                        residual[validation_mask], time[validation_mask], event[validation_mask],
                        min_patients=min_validation_patients, min_events=min_validation_events,
                    ),
                    "test": cox_null_score_one_step(
                        residual[test_mask], time[test_mask], event[test_mask],
                        min_patients=min_test_patients, min_events=min_test_events,
                    ),
                }
                estimable = feature_valid.copy()
                failure = np.where(
                    feature_valid, "", "INSUFFICIENT_TRAIN_MEASUREMENTS_OR_VARIANCE"
                ).astype(object)
                for role, result in results.items():
                    for statistic in ("beta", "standard_error", "z", "p_value"):
                        base[f"{role}_{statistic}"] = result[statistic]
                    base[f"n_{role}_patients"] = int(result["n_patients"])
                    base[f"n_{role}_events"] = int(result["n_events"])
                    role_ok = np.asarray(result["estimable"], dtype=bool)
                    role_reason = _role_failure(role, result)
                    if role_reason:
                        failure = np.where((failure == "") & ~role_ok, role_reason, failure)
                    else:
                        failure = np.where(
                            (failure == "") & ~role_ok,
                            f"{role.upper()}_ZERO_COX_SCORE_INFORMATION",
                            failure,
                        )
                    estimable &= role_ok
                base["fold_available"] = estimable
                base["failure_reason"] = np.where(estimable, "", failure)
                concordant = np.sign(base.train_beta.to_numpy(float)) == np.sign(
                    base.validation_beta.to_numpy(float)
                )
                base["validation_sign_concordant"] = pd.array(
                    np.where(estimable, concordant, False), dtype="boolean"
                )
                chunks.append(base)
    output = pd.concat(chunks, ignore_index=True)
    output["association_method"] = "BRESLOW_COX_NULL_SCORE_ONE_STEP"
    output["preprocessing_scope"] = "THREE_TRAIN_FOLDS_ONLY"
    output["model_version"] = "V3.2"
    output["old_checkpoint_loaded"] = False
    output["old_predictions_used_as_features"] = False
    output["changes_primary_ranking"] = False
    return output.sort_values(
        ["clinical_endpoint", "subject_id", "patient_fold_id"], kind="stable"
    ).reset_index(drop=True)


def _bh_fdr(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(numeric), np.nan, dtype=float)
    valid = np.isfinite(numeric)
    if valid.any():
        selected = numeric[valid]
        order = np.argsort(selected, kind="stable")
        ranked = selected[order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        restored = np.empty(len(ranked), dtype=float)
        restored[order] = np.clip(adjusted, 0.0, 1.0)
        output[valid] = restored
    return pd.Series(output, index=values.index)


def summarise_oof_associations(
    fold_records: pd.DataFrame,
    entity_ids: Sequence[str],
    *,
    cancer_id: str,
    subject_type: str,
    min_folds_available: int = 3,
) -> pd.DataFrame:
    """Random-effects meta-analysis of the five held-out fold estimates."""

    keys = ["cancer_id", "subject_type", "subject_id", "clinical_endpoint"]
    observed = fold_records.loc[
        fold_records.fold_available.astype(bool)
        & np.isfinite(pd.to_numeric(fold_records.test_beta, errors="coerce"))
        & np.isfinite(pd.to_numeric(fold_records.test_standard_error, errors="coerce"))
        & pd.to_numeric(fold_records.test_standard_error, errors="coerce").gt(0)
    ].copy()
    if observed.empty:
        first = pd.DataFrame(
            columns=keys
            + [
                "n_folds_available", "n_patients_test_total", "n_events_test_total",
                "n_validation_sign_concordant", "positive_folds", "negative_folds",
                "meta_beta", "meta_standard_error", "meta_z", "p_value",
                "hazard_ratio", "ci_lower", "ci_upper", "tau2", "i2",
                "n_folds_same_direction",
            ]
        )
    else:
        observed["fixed_weight"] = 1.0 / observed.test_standard_error.astype(float).pow(2)
        observed["fixed_weight_beta"] = observed.fixed_weight * observed.test_beta
        observed["fixed_weight_beta2"] = observed.fixed_weight * observed.test_beta.pow(2)
        observed["fixed_weight2"] = observed.fixed_weight.pow(2)
        first = observed.groupby(keys, observed=True, sort=False).agg(
            n_folds_available=("patient_fold_id", "nunique"),
            sum_weight=("fixed_weight", "sum"),
            sum_weight2=("fixed_weight2", "sum"),
            sum_weight_beta=("fixed_weight_beta", "sum"),
            sum_weight_beta2=("fixed_weight_beta2", "sum"),
            n_patients_test_total=("n_test_patients", "sum"),
            n_events_test_total=("n_test_events", "sum"),
            n_validation_sign_concordant=("validation_sign_concordant", "sum"),
            positive_folds=("test_beta", lambda value: int((value > 0).sum())),
            negative_folds=("test_beta", lambda value: int((value < 0).sum())),
        ).reset_index()
        first["fixed_beta"] = first.sum_weight_beta / first.sum_weight
        first["q"] = np.maximum(
            first.sum_weight_beta2 - first.sum_weight_beta.pow(2) / first.sum_weight, 0.0
        )
        first["meta_df"] = np.maximum(first.n_folds_available - 1, 0)
        first["meta_c"] = first.sum_weight - first.sum_weight2 / first.sum_weight
        first["tau2"] = np.where(
            first.meta_c > 0,
            np.maximum((first.q - first.meta_df) / first.meta_c, 0.0),
            0.0,
        )
        observed = observed.merge(first[keys + ["tau2"]], on=keys, how="left", validate="many_to_one")
        observed["random_weight"] = 1.0 / (
            observed.test_standard_error.astype(float).pow(2) + observed.tau2
        )
        observed["random_weight_beta"] = observed.random_weight * observed.test_beta
        second = observed.groupby(keys, observed=True, sort=False).agg(
            random_weight=("random_weight", "sum"),
            random_weight_beta=("random_weight_beta", "sum"),
        ).reset_index()
        first = first.merge(second, on=keys, how="left", validate="one_to_one")
        first["meta_beta"] = first.random_weight_beta / first.random_weight
        first["meta_standard_error"] = np.sqrt(1.0 / first.random_weight)
        first["meta_z"] = first.meta_beta / first.meta_standard_error
        first["p_value"] = _normal_two_sided_p(first.meta_z.to_numpy(float))
        first["hazard_ratio"] = np.exp(np.clip(first.meta_beta, -20, 20))
        first["ci_lower"] = np.exp(np.clip(first.meta_beta - 1.96 * first.meta_standard_error, -20, 20))
        first["ci_upper"] = np.exp(np.clip(first.meta_beta + 1.96 * first.meta_standard_error, -20, 20))
        first["i2"] = np.where(first.q > 0, np.maximum((first.q - first.meta_df) / first.q, 0.0), 0.0)
        first["n_folds_same_direction"] = first[["positive_folds", "negative_folds"]].max(axis=1)

    universe = pd.MultiIndex.from_product(
        [[str(cancer_id)], [str(subject_type)], list(map(str, entity_ids)), list(CLINICAL_ENDPOINTS)],
        names=keys,
    ).to_frame(index=False)
    summary = universe.merge(first, on=keys, how="left", validate="one_to_one")
    summary["n_folds_available"] = pd.to_numeric(
        summary.get("n_folds_available"), errors="coerce"
    ).fillna(0).astype("int8")
    summary["availability"] = summary.n_folds_available.ge(int(min_folds_available))
    summary.loc[summary.clinical_endpoint.eq("DFS"), "availability"] = False
    summary["failure_reason"] = np.select(
        [
            summary.clinical_endpoint.eq("DFS"),
            ~summary.availability,
        ],
        [
            "NO_DISTINCT_DFS_SOURCE",
            "FEWER_THAN_MINIMUM_ELIGIBLE_OOF_FOLDS",
        ],
        default="",
    )
    statistical_columns = [
        "meta_beta", "meta_standard_error", "meta_z", "p_value", "hazard_ratio",
        "ci_lower", "ci_upper", "tau2", "i2",
    ]
    for column in statistical_columns:
        if column not in summary:
            summary[column] = np.nan
        summary.loc[~summary.availability, column] = np.nan
    summary["fdr"] = np.nan
    for endpoint, index in summary.loc[summary.availability].groupby(
        "clinical_endpoint", observed=True
    ).groups.items():
        summary.loc[index, "fdr"] = _bh_fdr(summary.loc[index, "p_value"])
    summary["clinical_relevance_probability"] = np.where(
        summary.availability,
        np.clip(1.0 - pd.to_numeric(summary.fdr, errors="coerce"), 0.0, 1.0),
        np.nan,
    )
    summary["probability_definition"] = np.where(
        summary.availability,
        "1_MINUS_WITHIN_CANCER_ENDPOINT_BH_FDR_NOT_POSTERIOR",
        "",
    )
    summary["direction"] = np.where(
        summary.availability, np.where(summary.meta_beta.ge(0), "risk", "protective"), "unavailable"
    )
    summary["replication_tier"] = np.select(
        [
            summary.availability & summary.n_folds_same_direction.fillna(0).ge(4),
            summary.availability & summary.n_folds_same_direction.fillna(0).ge(3),
            summary.availability,
        ],
        ["robust_core", "replicated", "exploratory"],
        default="unavailable",
    )
    summary["association_method"] = "OOF_RANDOM_EFFECTS_BRESLOW_COX_SCORE"
    summary["model_version"] = "V3.2"
    summary["analysis_version"] = ANALYSIS_VERSION
    summary["old_checkpoint_loaded"] = False
    summary["old_predictions_used_as_features"] = False
    summary["changes_primary_ranking"] = False
    return summary.sort_values(
        ["clinical_endpoint", "subject_id"], kind="stable"
    ).reset_index(drop=True)


def load_long_entity_matrix(
    path: str | Path,
    *,
    cancer_id: str,
    entity_column: str,
    value_column: str,
    allowed_entities: Iterable[str] | None = None,
) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise ClinicalEntityTrainingError(f"Entity measurement file is missing: {source}")
    columns = ["cancer_id", "patient_id", entity_column, value_column]
    frame = pd.read_parquet(source, columns=columns)
    frame = frame.loc[frame.cancer_id.astype(str).eq(str(cancer_id))].copy()
    frame["patient_id"] = _explicit_patient_ids(
        frame.patient_id, context=f"Entity measurement {source}"
    ).to_numpy()
    frame[entity_column] = frame[entity_column].astype(str)
    if allowed_entities is not None:
        allowed = set(map(str, allowed_entities))
        frame = frame.loc[frame[entity_column].isin(allowed)]
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna(subset=["patient_id", entity_column, value_column])
    if frame.empty:
        raise ClinicalEntityTrainingError(
            f"No {entity_column} measurements remain for {cancer_id}: {source}"
        )
    wide = frame.pivot_table(
        index="patient_id", columns=entity_column, values=value_column, aggfunc="mean"
    )
    return wide.sort_index(axis=0).sort_index(axis=1)


def input_path_sha256(path: str | Path) -> str:
    source = Path(path)
    if source.is_file():
        return file_sha256(source)
    if not source.is_dir():
        raise ClinicalEntityTrainingError(f"Input path is missing: {source}")
    digest = hashlib.sha256()
    files = sorted(value for value in source.rglob("*") if value.is_file())
    if not files:
        raise ClinicalEntityTrainingError(f"Input directory is empty: {source}")
    for value in files:
        digest.update(value.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fold_metrics(fold_records: pd.DataFrame) -> pd.DataFrame:
    frame = fold_records.copy()
    frame["fold_available"] = frame.fold_available.astype(bool)
    frame["absolute_test_z"] = pd.to_numeric(frame.test_z, errors="coerce").abs()
    return frame.groupby(
        ["cancer_id", "subject_type", "clinical_endpoint", "patient_fold_id"],
        observed=True,
        sort=True,
    ).agg(
        entities_total=("subject_id", "nunique"),
        entities_available=("fold_available", "sum"),
        n_train_patients=("n_train_patients", "max"),
        n_train_events=("n_train_events", "max"),
        n_validation_patients=("n_validation_patients", "max"),
        n_validation_events=("n_validation_events", "max"),
        n_test_patients=("n_test_patients", "max"),
        n_test_events=("n_test_events", "max"),
        median_absolute_test_z=("absolute_test_z", "median"),
        validation_sign_concordance=("validation_sign_concordant", "mean"),
    ).reset_index()


__all__ = [
    "ANALYSIS_VERSION",
    "CALCULATION_FORMAT",
    "MODELED_ENDPOINTS",
    "SUBJECT_SPECS",
    "ClinicalEntityTrainingError",
    "fold_roles",
    "cox_null_score_one_step",
    "fold_local_residualize",
    "analyse_entity_matrix",
    "summarise_oof_associations",
    "load_long_entity_matrix",
    "input_path_sha256",
    "json_sha256",
    "fold_metrics",
    "validate_entity_clinical_contract",
    "standardize_tcga_cdr_workbook",
    "patient_fold_manifest",
    "HISTORICAL_STATE_IDS",
]
