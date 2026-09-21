"""Patient-OOF prediction of the epigenetically explained lncRNA expression.

The head is deliberately separate from HHGT.  It predicts expression from
available regulatory signals inside each cancer and lncRNA, while fitting a
second nuisance-only model.  Their difference is the regulatory expression
component that may later enter a cancer-gated fusion.  Missing assays remain
typed unavailable and held-out expression is never used to fit its own fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


KEYS = ("cancer_id", "patient_id", "lncrna_id")
FOLD_COLUMN = "patient_fold_id"
TARGET_COLUMN = "lncrna_expression"
DEFAULT_SIGNAL_FEATURES = (
    "atac_distal_accessibility",
    "distal_mutation_burden",
    "promoter_methylation_beta",
    "distal_methylation_beta",
)
DEFAULT_NUISANCE_FEATURES = ("local_cnv_log2", "tumor_purity")


class EpigeneticExpressionError(RuntimeError):
    """Raised when the patient-OOF expression contract is violated."""


@dataclass(frozen=True)
class EpigeneticExpressionConfig:
    signal_features: tuple[str, ...] = DEFAULT_SIGNAL_FEATURES
    nuisance_features: tuple[str, ...] = DEFAULT_NUISANCE_FEATURES
    folds: int = 5
    ridge_alpha: float = 1.0
    min_train_patients: int = 20
    min_target_sd: float = 1e-6
    min_feature_sd: float = 1e-8

    def __post_init__(self) -> None:
        if self.folds != 5:
            raise ValueError("The formal epigenetic expression head requires five folds")
        if self.ridge_alpha < 0 or self.min_train_patients < 5:
            raise ValueError("Invalid epigenetic expression regularisation controls")
        if not self.signal_features:
            raise ValueError("At least one regulatory signal feature is required")
        overlap = set(self.signal_features) & set(self.nuisance_features)
        if overlap:
            raise ValueError(f"Signal and nuisance features overlap: {sorted(overlap)}")


def _availability_column(feature: str) -> str:
    return f"{feature}__available"


def _normalise(frame: pd.DataFrame, config: EpigeneticExpressionConfig) -> pd.DataFrame:
    features = config.signal_features + config.nuisance_features
    required = set(KEYS + (FOLD_COLUMN, TARGET_COLUMN))
    required.update(features)
    required.update(_availability_column(feature) for feature in features)
    if missing := sorted(required - set(frame.columns)):
        raise EpigeneticExpressionError(f"Epigenetic expression input lacks: {missing}")
    result = frame.loc[:, sorted(required)].copy()
    result["cancer_id"] = result.cancer_id.astype(str).str.upper()
    result["patient_id"] = result.patient_id.astype(str).str[:12]
    result["lncrna_id"] = result.lncrna_id.astype(str)
    result[FOLD_COLUMN] = pd.to_numeric(result[FOLD_COLUMN], errors="coerce")
    if result[FOLD_COLUMN].isna().any() or not result[FOLD_COLUMN].isin(range(5)).all():
        raise EpigeneticExpressionError("Patient fold IDs must be exact integers 0..4")
    result[FOLD_COLUMN] = result[FOLD_COLUMN].astype(np.int8)
    if result.duplicated(list(KEYS)).any():
        raise EpigeneticExpressionError("Epigenetic expression input has duplicate patient keys")
    patient_fold_counts = result.groupby(
        ["cancer_id", "patient_id"], observed=True
    )[FOLD_COLUMN].nunique()
    if patient_fold_counts.gt(1).any():
        raise EpigeneticExpressionError("One patient occurs in multiple folds")
    result[TARGET_COLUMN] = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")
    for feature in features:
        available_column = _availability_column(feature)
        if result[available_column].isna().any():
            raise EpigeneticExpressionError(f"{feature} availability contains null")
        result[available_column] = result[available_column].astype(bool)
        result[feature] = pd.to_numeric(result[feature], errors="coerce")
        available = result[available_column].to_numpy(bool)
        finite = np.isfinite(result[feature].to_numpy(float))
        if not np.array_equal(available, finite):
            raise EpigeneticExpressionError(
                f"{feature} violates typed-null availability (finite iff available)"
            )
    return result.sort_values(list(KEYS), kind="stable").reset_index(drop=True)


def _design(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Iterable[str],
    *,
    min_feature_sd: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    train_columns: list[np.ndarray] = [np.ones(len(train), dtype=np.float64)]
    test_columns: list[np.ndarray] = [np.ones(len(test), dtype=np.float64)]
    records: list[dict[str, Any]] = []
    for feature in features:
        availability = _availability_column(feature)
        train_available = train[availability].to_numpy(bool)
        test_available = test[availability].to_numpy(bool)
        train_value = train[feature].to_numpy(float)
        test_value = test[feature].to_numpy(float)
        observed = train_value[train_available]
        mean = float(observed.mean()) if len(observed) else 0.0
        sd = float(observed.std(ddof=0)) if len(observed) else 0.0
        usable = bool(len(observed) >= 3 and np.isfinite(sd) and sd >= min_feature_sd)
        if usable:
            train_z = np.zeros(len(train), dtype=np.float64)
            test_z = np.zeros(len(test), dtype=np.float64)
            train_z[train_available] = (train_value[train_available] - mean) / sd
            test_z[test_available] = (test_value[test_available] - mean) / sd
        else:
            train_z = np.zeros(len(train), dtype=np.float64)
            test_z = np.zeros(len(test), dtype=np.float64)
        train_columns.extend((train_z, train_available.astype(np.float64)))
        test_columns.extend((test_z, test_available.astype(np.float64)))
        records.append(
            {
                "feature": feature,
                "train_available": int(train_available.sum()),
                "train_mean": mean,
                "train_sd": sd,
                "usable": usable,
            }
        )
    return (
        np.column_stack(train_columns),
        np.column_stack(test_columns),
        records,
    )


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    try:
        return np.linalg.solve(x.T @ x + penalty, x.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(x.T @ x + penalty) @ (x.T @ y)


def crossfit_epigenetic_expression(
    frame: pd.DataFrame,
    *,
    config: EpigeneticExpressionConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return patient-OOF full and regulatory-component expression predictions."""

    settings = config or EpigeneticExpressionConfig()
    data = _normalise(frame, settings)
    output = data.loc[:, list(KEYS) + [FOLD_COLUMN, TARGET_COLUMN]].copy()
    output["predicted_expression_z"] = np.nan
    output["regulatory_expression_component_z"] = np.nan
    output["epigenetic_expression_available"] = False
    output["epigenetic_expression_unavailable_reason"] = (
        "NO_PATIENT_OOF_REGULATORY_EXPRESSION_PREDICTION"
    )
    fit_records: list[dict[str, Any]] = []
    signal_columns = list(settings.signal_features)
    nuisance_columns = list(settings.nuisance_features)
    all_features = signal_columns + nuisance_columns

    grouped = data.groupby(["cancer_id", "lncrna_id"], observed=True, sort=True)
    for (cancer, lncrna), group in grouped:
        indices = group.index.to_numpy(int)
        for fold in range(settings.folds):
            train_mask = group[FOLD_COLUMN].ne(fold).to_numpy(bool).copy()
            test_mask = ~train_mask
            target = group[TARGET_COLUMN].to_numpy(float)
            train_mask &= np.isfinite(target)
            test_signal_available = np.zeros(len(group), dtype=bool)
            for feature in signal_columns:
                test_signal_available |= group[_availability_column(feature)].to_numpy(bool)
            heldout = test_mask & test_signal_available
            record: dict[str, Any] = {
                "cancer_id": str(cancer),
                "lncrna_id": str(lncrna),
                "patient_fold_id": fold,
                "train_patients": int(train_mask.sum()),
                "heldout_patients": int(test_mask.sum()),
                "heldout_signal_available": int(heldout.sum()),
                "status": "UNAVAILABLE",
            }
            if train_mask.sum() < settings.min_train_patients:
                record["reason"] = "INSUFFICIENT_OUTER_TRAIN_PATIENTS"
                fit_records.append(record)
                continue
            y_train = target[train_mask]
            y_mean = float(y_train.mean())
            y_sd = float(y_train.std(ddof=0))
            if not np.isfinite(y_sd) or y_sd < settings.min_target_sd:
                record["reason"] = "OUTER_TRAIN_EXPRESSION_HAS_NO_VARIANCE"
                fit_records.append(record)
                continue
            if not heldout.any():
                record["reason"] = "HELDOUT_PATIENT_HAS_NO_REGULATORY_ASSAY"
                fit_records.append(record)
                continue
            train = group.loc[train_mask]
            test = group.loc[heldout]
            full_train, full_test, feature_records = _design(
                train, test, all_features, min_feature_sd=settings.min_feature_sd
            )
            signal_usable = any(
                item["usable"] and item["feature"] in set(signal_columns)
                for item in feature_records
            )
            if not signal_usable:
                record["reason"] = "NO_USABLE_OUTER_TRAIN_REGULATORY_SIGNAL"
                fit_records.append(record)
                continue
            nuisance_train, nuisance_test, _ = _design(
                train, test, nuisance_columns, min_feature_sd=settings.min_feature_sd
            )
            y_z = (y_train - y_mean) / y_sd
            full_beta = _ridge_fit(full_train, y_z, settings.ridge_alpha)
            nuisance_beta = _ridge_fit(nuisance_train, y_z, settings.ridge_alpha)
            full_prediction = full_test @ full_beta
            nuisance_prediction = nuisance_test @ nuisance_beta
            destination = indices[np.flatnonzero(heldout)]
            output.loc[destination, "predicted_expression_z"] = full_prediction
            output.loc[destination, "regulatory_expression_component_z"] = (
                full_prediction - nuisance_prediction
            )
            output.loc[destination, "epigenetic_expression_available"] = True
            output.loc[
                destination, "epigenetic_expression_unavailable_reason"
            ] = None
            record.update(
                {
                    "status": "PASS_PATIENT_OOF",
                    "reason": None,
                    "target_train_mean": y_mean,
                    "target_train_sd": y_sd,
                    "feature_fits": feature_records,
                    "full_coefficients": full_beta.tolist(),
                    "nuisance_coefficients": nuisance_beta.tolist(),
                }
            )
            fit_records.append(record)

    available = output.epigenetic_expression_available.to_numpy(bool)
    for column in ("predicted_expression_z", "regulatory_expression_component_z"):
        finite = np.isfinite(output[column].to_numpy(float))
        if not np.array_equal(available, finite):
            raise EpigeneticExpressionError(f"{column} violates final typed availability")
    if output.loc[available, "epigenetic_expression_unavailable_reason"].notna().any():
        raise EpigeneticExpressionError("Available epigenetic expression row has a reason")
    audit = {
        "format": "CANCERLNCATLAS_V32_EPIGENETIC_EXPRESSION_PATIENT_OOF_V1",
        "status": "PASS" if available.any() else "TYPED_UNAVAILABLE_NO_OOF_PREDICTIONS",
        "rows": int(len(output)),
        "available_rows": int(available.sum()),
        "cancers": int(output.cancer_id.nunique()),
        "lncrnas": int(output.lncrna_id.nunique()),
        "patient_folds": settings.folds,
        "signal_features": signal_columns,
        "nuisance_features": nuisance_columns,
        "heldout_expression_used_for_own_prediction": False,
        "missing_assay_assumed_zero": False,
        "regulatory_component_definition": "FULL_MODEL_PREDICTION_MINUS_NUISANCE_ONLY_PREDICTION",
        "fit_records": fit_records,
    }
    return output, audit


def nested_outer_crossfit_epigenetic_expression(
    frame: pd.DataFrame,
    *,
    config: EpigeneticExpressionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build leakage-isolated expression components for a two-stage OOF head.

    The ordinary patient-OOF component is retained as the publishable
    per-patient artifact.  For every second-stage outer fold, training-patient
    components are rebuilt by inner cross-fitting after removing that entire
    outer fold.  The outer-heldout components come from a model fitted on the
    four remaining folds.  Consequently no expression from the outer-heldout
    patients enters any component used to fit or evaluate that outer fold's
    pathway model.
    """

    settings = config or EpigeneticExpressionConfig()
    normalised = _normalise(frame, settings)
    own_fold, own_audit = crossfit_epigenetic_expression(
        normalised, config=settings
    )
    nested_parts: list[pd.DataFrame] = []
    outer_audits: list[dict[str, Any]] = []
    for outer_fold in range(settings.folds):
        outer_train = normalised.loc[normalised[FOLD_COLUMN].ne(outer_fold)].copy()
        inner_component, inner_audit = crossfit_epigenetic_expression(
            outer_train, config=settings
        )
        heldout_component = own_fold.loc[
            own_fold[FOLD_COLUMN].eq(outer_fold)
        ].copy()
        combined = pd.concat(
            [inner_component, heldout_component], ignore_index=True
        )
        combined["outer_fold_id"] = np.int8(outer_fold)
        if len(combined) != len(normalised):
            raise EpigeneticExpressionError(
                f"Nested outer fold {outer_fold} does not cover every input row"
            )
        if combined.duplicated(list(KEYS)).any():
            raise EpigeneticExpressionError(
                f"Nested outer fold {outer_fold} duplicates patient keys"
            )
        nested_parts.append(combined)
        outer_audits.append(
            {
                "outer_fold_id": outer_fold,
                "outer_train_rows": int(len(outer_train)),
                "outer_heldout_rows": int(len(heldout_component)),
                "inner_component_available_rows": int(
                    inner_component.epigenetic_expression_available.sum()
                ),
                "heldout_component_available_rows": int(
                    heldout_component.epigenetic_expression_available.sum()
                ),
                "inner_status": inner_audit["status"],
                "outer_heldout_expression_present_in_inner_input": False,
            }
        )
    nested = pd.concat(nested_parts, ignore_index=True)
    expected_rows = len(normalised) * settings.folds
    if len(nested) != expected_rows:
        raise EpigeneticExpressionError("Nested component row-count contract failed")
    nested_keys = list(KEYS) + ["outer_fold_id"]
    if nested.duplicated(nested_keys).any():
        raise EpigeneticExpressionError("Nested component keys are not unique")
    audit = {
        "format": "CANCERLNCATLAS_V32_EPIGENETIC_EXPRESSION_NESTED_PATIENT_OOF_V1",
        "status": own_audit["status"],
        "rows": int(len(normalised)),
        "nested_rows": int(len(nested)),
        "patient_folds": settings.folds,
        "outer_heldout_expression_used_for_any_component_fit": False,
        "outer_training_components_are_inner_oof": True,
        "own_fold_audit": own_audit,
        "outer_folds": outer_audits,
    }
    return own_fold, nested, audit


__all__ = [
    "EpigeneticExpressionConfig",
    "EpigeneticExpressionError",
    "crossfit_epigenetic_expression",
    "nested_outer_crossfit_epigenetic_expression",
]
