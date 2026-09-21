"""Cancer-specific, nested-OOF routing for Mutation, CNV and ATAC experts.

This is an external routed *candidate*.  It does not mutate the HHGT encoder
or the frozen V3.2 primary score.  Every held-out pair fold is routed only by
gates selected on its inner training/validation folds.  Held-out performance
is reported but is never used to switch a gate after the fact.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .multimodal_fusion import (
    FUSION_FOLD_COLUMN,
    TARGET_KEYS,
    MultimodalFusionError,
    ResidualFusionConfig,
    ResidualFusionModel,
    apply_residual_fusion_model,
    artifact_sha256,
    fit_residual_fusion_model,
)


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_ROUTED_CANDIDATE"
MODULE_ID = "cancer_modality_sparse_router"
MODALITIES = ("mutation", "cnv", "atac")
SCORE_COLUMN = "discovery_adjusted_probability"


class CancerModalityRouterError(MultimodalFusionError):
    """Raised when routing would violate missingness or split isolation."""


@dataclass(frozen=True)
class CancerRouterConfig:
    folds: int = 5
    seed: int = 20260726
    min_train_rows: int = 100
    min_validation_rows: int = 30
    min_available_train_rows: int = 30
    min_available_validation_rows: int = 12
    validation_logloss_delta: float = 1e-4
    modality_ablation_logloss_delta: float = 0.0
    brier_tolerance: float = 0.0
    minimum_weight: float = 1e-8
    frozen_gate_support_folds: int = 3
    max_steps: int = 600
    learning_rate: float = 0.02
    batch_size: int = 4096
    l2: float = 1e-3
    patience: int = 10
    evaluation_interval: int = 10
    max_train_rows: int = 200_000
    max_validation_rows: int = 100_000
    clip_probability: float = 1e-6

    def validate(self) -> None:
        if self.folds != 5:
            raise CancerModalityRouterError("Routed candidate requires exactly five pair folds")
        positive = (
            self.min_train_rows,
            self.min_validation_rows,
            self.min_available_train_rows,
            self.min_available_validation_rows,
            self.frozen_gate_support_folds,
            self.max_steps,
            self.batch_size,
            self.patience,
            self.evaluation_interval,
        )
        if min(positive) < 1 or self.frozen_gate_support_folds > self.folds:
            raise CancerModalityRouterError("Router row/fold/optimiser controls are invalid")
        if min(
            self.validation_logloss_delta,
            self.modality_ablation_logloss_delta,
            self.brier_tolerance,
            self.minimum_weight,
        ) < 0:
            raise CancerModalityRouterError("Router gates require non-negative thresholds")

    def fusion_config(self, seed: int) -> ResidualFusionConfig:
        return ResidualFusionConfig(
            seed=int(seed),
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            l2=self.l2,
            patience=self.patience,
            evaluation_interval=self.evaluation_interval,
            max_train_rows=self.max_train_rows,
            max_validation_rows=self.max_validation_rows,
            clip_probability=self.clip_probability,
            increment_logloss_delta=self.validation_logloss_delta,
        )


@dataclass
class CancerRouterResult:
    oof_private: pd.DataFrame
    routed_candidate: pd.DataFrame
    metrics: pd.DataFrame
    fold_gates: list[dict[str, Any]]
    frozen_gates: list[dict[str, Any]]


@dataclass
class CancerRouterValidationResult:
    """Inner-validation outputs produced without querying an outer fold."""

    validation_private: pd.DataFrame
    fold_gates: list[dict[str, Any]]


def _validated_router_data(
    frame: pd.DataFrame, settings: CancerRouterConfig
) -> pd.DataFrame:
    settings.validate()
    data = _canonical(frame, "router frame")
    required = {"primary_probability", "fusion_target", FUSION_FOLD_COLUMN}
    required |= {
        f"{item}_{suffix}"
        for item in MODALITIES
        for suffix in ("probability", "available")
    }
    if missing := sorted(required - set(data.columns)):
        raise CancerModalityRouterError(f"Router frame lacks columns: {missing}")
    _probability(data.primary_probability, "primary_probability")
    _probability(data.fusion_target, "fusion_target")
    for modality in MODALITIES:
        available = data[f"{modality}_available"]
        if available.isna().any():
            raise CancerModalityRouterError(f"{modality} availability contains null")
        available_array = available.astype(bool).to_numpy()
        probability = _probability(
            data[f"{modality}_probability"], modality, allow_null=True
        )
        if (
            np.isnan(probability[available_array]).any()
            or np.isfinite(probability[~available_array]).any()
        ):
            raise CancerModalityRouterError(f"{modality} violates typed missingness")
    if set(pd.unique(data[FUSION_FOLD_COLUMN])) != set(range(settings.folds)):
        raise CancerModalityRouterError("Router frame does not cover pair folds 0..4")
    return data


def _canonical(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if missing := sorted(set(TARGET_KEYS) - set(frame.columns)):
        raise CancerModalityRouterError(f"{label} lacks exact keys: {missing}")
    result = frame.copy()
    result["cancer_id"] = result.cancer_id.astype(str).str.upper()
    for column in ("lncrna_id", "pathway_id"):
        result[column] = result[column].astype(str)
    if result[list(TARGET_KEYS)].duplicated().any():
        raise CancerModalityRouterError(f"{label} has duplicate exact keys")
    return result.sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)


def _probability(series: pd.Series, label: str, allow_null: bool = False) -> np.ndarray:
    value = pd.to_numeric(series, errors="coerce").to_numpy(float)
    if not allow_null and not np.isfinite(value).all():
        raise CancerModalityRouterError(f"{label} contains missing/non-finite probabilities")
    finite = value[np.isfinite(value)]
    if ((finite < 0) | (finite > 1)).any():
        raise CancerModalityRouterError(f"{label} is outside [0, 1]")
    return value


def _same_universe(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    if len(left) != len(right) or not left[list(TARGET_KEYS)].equals(right[list(TARGET_KEYS)]):
        raise CancerModalityRouterError(f"{label} differs from the primary candidate universe")


def validate_patient_oof_lineage(
    lineage: Mapping[str, Any],
    *,
    predictions_sha256: str | None = None,
) -> None:
    """Fail closed unless Mutation/CNV predictions are genuine patient OOF."""

    if not str(lineage.get("analysis_version", "")).startswith("CancerLncAtlas_V3.2"):
        raise CancerModalityRouterError("Genomic lineage is not a V3.2 run")
    if lineage.get("training_status") != "SUCCESS":
        raise CancerModalityRouterError("Genomic lineage is not a successful fresh run")
    if int(lineage.get("folds", -1)) != 5:
        raise CancerModalityRouterError("Genomic lineage does not declare five patient folds")
    if lineage.get("old_checkpoint_loaded") is not False:
        raise CancerModalityRouterError("Old genomic checkpoints are forbidden")
    if lineage.get("old_predictions_used_as_features") is not False:
        raise CancerModalityRouterError("Old genomic predictions are forbidden")
    if lineage.get("private_head_trained_from_scratch") is not True:
        raise CancerModalityRouterError("Genomic private heads were not trained from scratch")
    if lineage.get("core_parameters_frozen") is not True:
        raise CancerModalityRouterError("Genomic lineage does not prove the V3.2 core was frozen")
    if lineage.get("old_rankings_used_as_outputs") is not False:
        raise CancerModalityRouterError("Old genomic rankings are forbidden")
    declared_sha = str(lineage.get("prediction_sha256", ""))
    if len(declared_sha) != 64:
        raise CancerModalityRouterError("Genomic lineage lacks a prediction SHA256")
    if predictions_sha256 is not None and declared_sha != str(predictions_sha256):
        raise CancerModalityRouterError(
            "Genomic prediction table disagrees with its lineage SHA256"
        )
    modalities = lineage.get("modalities")
    if not isinstance(modalities, Mapping) or not {"mutation", "cnv"}.issubset(modalities):
        raise CancerModalityRouterError("Genomic lineage lacks split Mutation/CNV fold status")
    for modality in ("mutation", "cnv"):
        declared = modalities[modality]
        if not isinstance(declared, list):
            raise CancerModalityRouterError(f"{modality} fold status is malformed")
        try:
            observed = {int(item["patient_fold"]) for item in declared}
        except (KeyError, TypeError, ValueError) as exc:
            raise CancerModalityRouterError(f"{modality} fold status is malformed") from exc
        if observed != set(range(5)):
            raise CancerModalityRouterError(f"{modality} patient folds are incomplete")
        if any(item.get("status") != "SUCCESS" for item in declared):
            raise CancerModalityRouterError(f"{modality} patient fold is not successful")


def validate_atac_oof_lineage(
    lineage: Mapping[str, Any],
    *,
    predictions_sha256: str | None = None,
) -> None:
    """Fail closed unless ATAC predictions are fresh, hash-bound patient OOF."""

    if not str(lineage.get("analysis_version", "")).startswith("CancerLncAtlas_V3.2"):
        raise CancerModalityRouterError("ATAC lineage is not a V3.2 run")
    if str(lineage.get("modality", "")).lower() != "atac":
        raise CancerModalityRouterError("ATAC lineage does not declare modality=atac")
    if int(lineage.get("folds", -1)) != 5:
        raise CancerModalityRouterError("ATAC lineage does not declare five patient folds")
    if lineage.get("patient_level_modality_oof") is not True:
        raise CancerModalityRouterError("ATAC lineage is not patient-level OOF")
    if lineage.get("patient_fold_oof_predictions_not_fold_averaged") is not True:
        raise CancerModalityRouterError("ATAC lineage lacks fold-specific OOF predictions")
    if int(lineage.get("outer_test_patients_used_in_any_upstream_fit", -1)) != 0:
        raise CancerModalityRouterError("ATAC lineage used held-out patients upstream")
    if lineage.get("old_checkpoint_loaded") is not False:
        raise CancerModalityRouterError("Old ATAC checkpoints are forbidden")
    if lineage.get("old_predictions_used_as_features") is not False:
        raise CancerModalityRouterError("Old ATAC predictions are forbidden")
    declared_sha = str(lineage.get("predictions_sha256", ""))
    if len(declared_sha) != 64:
        raise CancerModalityRouterError("ATAC lineage lacks a prediction SHA256")
    if predictions_sha256 is not None and declared_sha != str(predictions_sha256):
        raise CancerModalityRouterError("ATAC prediction table disagrees with its lineage SHA256")
    fold_status = lineage.get("fold_status")
    if not isinstance(fold_status, list) or len(fold_status) != 5:
        raise CancerModalityRouterError("ATAC lineage fold status is incomplete")
    try:
        observed = {int(item["patient_fold"]) for item in fold_status}
    except (KeyError, TypeError, ValueError) as exc:
        raise CancerModalityRouterError("ATAC lineage fold status is malformed") from exc
    if observed != set(range(5)):
        raise CancerModalityRouterError("ATAC patient folds are incomplete")
    for item in fold_status:
        if item.get("status") != "SUCCESS":
            raise CancerModalityRouterError("ATAC patient fold is not successful")
        if item.get("heldout_patients_used_for_fit") is not False:
            raise CancerModalityRouterError("ATAC fold does not prove held-out isolation")


def build_cancer_router_frame(
    primary_frame: pd.DataFrame,
    genomic_predictions: pd.DataFrame,
    atac_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Split the genomic expert and materialize typed unavailable ATAC."""

    primary = _canonical(primary_frame, "router primary frame")
    required_primary = {"primary_probability", "fusion_target", FUSION_FOLD_COLUMN}
    if missing := sorted(required_primary - set(primary.columns)):
        raise CancerModalityRouterError(f"Router primary frame lacks: {missing}")
    _probability(primary.primary_probability, "primary_probability")
    _probability(primary.fusion_target, "fusion_target")
    folds = pd.to_numeric(primary[FUSION_FOLD_COLUMN], errors="raise").astype(int)
    if set(folds.unique()) != set(range(5)):
        raise CancerModalityRouterError("Router outer pair folds must be exactly 0..4")
    primary[FUSION_FOLD_COLUMN] = folds

    genomic = _canonical(genomic_predictions, "genomic predictions")
    _same_universe(primary, genomic, "genomic predictions")
    result = primary[list(TARGET_KEYS) + ["primary_probability", "fusion_target", FUSION_FOLD_COLUMN]].copy()
    for modality in ("mutation", "cnv"):
        probability_column = f"{modality}_context_probability"
        availability_column = f"{modality}_available"
        reason_column = f"{modality}_unavailable_reason"
        if missing := sorted({probability_column, availability_column} - set(genomic.columns)):
            raise CancerModalityRouterError(f"Genomic predictions lack {modality}: {missing}")
        available = genomic[availability_column]
        if available.isna().any():
            raise CancerModalityRouterError(f"{modality} availability contains null")
        available = available.astype(bool).to_numpy()
        probability = _probability(genomic[probability_column], probability_column, allow_null=True)
        if np.isnan(probability[available]).any() or np.isfinite(probability[~available]).any():
            raise CancerModalityRouterError(f"{modality} violates typed-null availability")
        result[f"{modality}_probability"] = probability
        result[f"{modality}_available"] = available
        result[f"{modality}_unavailable_reason"] = (
            genomic[reason_column].astype("string")
            if reason_column in genomic else pd.Series(pd.NA, index=genomic.index, dtype="string")
        )
        if result.loc[~available, f"{modality}_unavailable_reason"].isna().any():
            raise CancerModalityRouterError(
                f"Unavailable {modality} rows lack an explicit reason"
            )

    if atac_predictions is None:
        result["atac_probability"] = np.nan
        result["atac_available"] = False
        result["atac_unavailable_reason"] = "ATAC_NOT_TRAINED_FOR_ROUTED_CANDIDATE"
    else:
        atac = _canonical(atac_predictions, "ATAC predictions")
        _same_universe(primary, atac, "ATAC predictions")
        required = {"atac_context_probability", "atac_available"}
        if missing := sorted(required - set(atac.columns)):
            raise CancerModalityRouterError(f"ATAC predictions lack: {missing}")
        if atac.atac_available.isna().any():
            raise CancerModalityRouterError("ATAC availability contains null")
        available = atac.atac_available.astype(bool).to_numpy()
        probability = _probability(atac.atac_context_probability, "ATAC probability", allow_null=True)
        if np.isnan(probability[available]).any() or np.isfinite(probability[~available]).any():
            raise CancerModalityRouterError("ATAC violates typed-null availability")
        reason = (
            atac.atac_unavailable_reason.astype("string")
            if "atac_unavailable_reason" in atac else pd.Series(pd.NA, index=atac.index, dtype="string")
        )
        if reason.loc[~available].isna().any():
            raise CancerModalityRouterError("Unavailable ATAC rows lack an explicit reason")
        result["atac_probability"] = probability
        result["atac_available"] = available
        result["atac_unavailable_reason"] = reason
    return result


def _logloss(target: np.ndarray, probability: np.ndarray, clip: float) -> float:
    probability = np.clip(probability, clip, 1 - clip)
    return float(-np.mean(target * np.log(probability) + (1 - target) * np.log1p(-probability)))


def _brier(target: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.square(target - probability)))


def _fallback_prediction(test: pd.DataFrame, reason: str) -> pd.DataFrame:
    output = test[list(TARGET_KEYS) + ["fusion_target", FUSION_FOLD_COLUMN, "primary_probability"]].copy()
    output[SCORE_COLUMN] = output.primary_probability.to_numpy(float)
    for modality in MODALITIES:
        available = test[f"{modality}_available"].astype(bool).to_numpy()
        output[f"{modality}_probability"] = test[f"{modality}_probability"].to_numpy(float)
        output[f"{modality}_available"] = available
        output[f"{modality}_fusion_weight"] = 0.0
        output[f"{modality}_logit_contribution"] = np.where(available, 0.0, np.nan)
        output[f"{modality}_gate_enabled"] = False
    output["router_fallback"] = True
    output["router_reason"] = reason
    return output


def _fit_one_gate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    cancer: str,
    heldout: int,
    settings: CancerRouterConfig,
) -> tuple[ResidualFusionModel | None, dict[str, Any]]:
    validation_fold = (heldout + 1) % settings.folds
    record: dict[str, Any] = {
        "cancer_id": cancer,
        "heldout_pair_fold": heldout,
        "training_pair_folds": sorted(set(range(settings.folds)) - {heldout, validation_fold}),
        "validation_pair_fold": validation_fold,
        "heldout_used_for_gate_selection": False,
        "status": "PRIMARY_FALLBACK",
        "reason": None,
        "modalities": {},
    }
    if len(train) < settings.min_train_rows or len(validation) < settings.min_validation_rows:
        record["reason"] = "INSUFFICIENT_INNER_ROWS"
        return None, record
    eligible: list[str] = []
    for modality in MODALITIES:
        train_available = int(train[f"{modality}_available"].sum())
        validation_available = int(validation[f"{modality}_available"].sum())
        is_eligible = (
            train_available >= settings.min_available_train_rows
            and validation_available >= settings.min_available_validation_rows
        )
        record["modalities"][modality] = {
            "train_available_rows": train_available,
            "validation_available_rows": validation_available,
            "eligible": is_eligible,
            "enabled": False,
            "weight": 0.0,
        }
        if is_eligible:
            eligible.append(modality)
    if not eligible:
        record["reason"] = "NO_MODALITY_PASSES_INNER_COVERAGE_GATE"
        return None, record
    fit_config = settings.fusion_config(settings.seed + heldout * 1009 + sum(map(ord, cancer)))
    try:
        model = fit_residual_fusion_model(
            train,
            validation,
            eligible,
            endpoint="discovery",
            config=fit_config,
            seed=fit_config.seed,
        )
    except MultimodalFusionError as exc:
        record["reason"] = f"INNER_FIT_FAILED:{type(exc).__name__}"
        return None, record
    primary = validation.primary_probability.to_numpy(float)
    target = validation.fusion_target.to_numpy(float)
    full = apply_residual_fusion_model(model, validation, include_private_target=True)
    full_probability = full[SCORE_COLUMN].to_numpy(float)
    primary_loss = _logloss(target, primary, settings.clip_probability)
    full_loss = _logloss(target, full_probability, settings.clip_probability)
    primary_brier = _brier(target, primary)
    full_brier = _brier(target, full_probability)
    record.update(
        {
            "inner_primary_logloss": primary_loss,
            "inner_full_logloss": full_loss,
            "inner_logloss_improvement": primary_loss - full_loss,
            "inner_primary_brier": primary_brier,
            "inner_full_brier": full_brier,
        }
    )
    full_pass = (
        primary_loss - full_loss >= settings.validation_logloss_delta
        and full_brier <= primary_brier + settings.brier_tolerance
    )
    enabled: list[bool] = []
    for index, modality in enumerate(model.expert_ids):
        ablated_weights = list(model.weights)
        ablated_weights[index] = 0.0
        ablated = replace(model, weights=tuple(ablated_weights))
        ablated_probability = apply_residual_fusion_model(
            ablated, validation, include_private_target=True
        )[SCORE_COLUMN].to_numpy(float)
        conditional_delta = _logloss(
            target, ablated_probability, settings.clip_probability
        ) - full_loss
        keep = (
            full_pass
            and float(model.weights[index]) >= settings.minimum_weight
            and conditional_delta > settings.modality_ablation_logloss_delta
        )
        enabled.append(keep)
        record["modalities"][modality].update(
            {
                "conditional_ablation_logloss_delta": conditional_delta,
                "enabled": keep,
                "weight": float(model.weights[index]) if keep else 0.0,
                "centre": float(model.centres[index]),
            }
        )
    gated_weights = tuple(weight if keep else 0.0 for weight, keep in zip(model.weights, enabled))
    gated = replace(model, weights=gated_weights)
    gated_probability = apply_residual_fusion_model(
        gated, validation, include_private_target=True
    )[SCORE_COLUMN].to_numpy(float)
    gated_loss = _logloss(target, gated_probability, settings.clip_probability)
    gated_brier = _brier(target, gated_probability)
    if (
        not any(enabled)
        or primary_loss - gated_loss < settings.validation_logloss_delta
        or gated_brier > primary_brier + settings.brier_tolerance
    ):
        record["reason"] = "INNER_VALIDATION_INCREMENT_GATE_FAILED"
        for modality in record["modalities"].values():
            modality["enabled"] = False
            modality["weight"] = 0.0
        return None, record
    record["status"] = "ROUTED"
    record["reason"] = "INNER_VALIDATION_GATE_PASSED"
    record["inner_gated_logloss"] = gated_loss
    record["inner_gated_brier"] = gated_brier
    return gated, record


def _complete_prediction(
    prediction: pd.DataFrame,
    source: pd.DataFrame,
    model: ResidualFusionModel,
    record: Mapping[str, Any],
) -> pd.DataFrame:
    for modality in MODALITIES:
        available = source[f"{modality}_available"].astype(bool).to_numpy()
        prediction[f"{modality}_probability"] = source[f"{modality}_probability"].to_numpy(float)
        prediction[f"{modality}_available"] = available
        enabled = bool(record["modalities"].get(modality, {}).get("enabled", False))
        prediction[f"{modality}_gate_enabled"] = enabled
        if modality not in model.expert_ids:
            prediction[f"{modality}_fusion_weight"] = 0.0
            prediction[f"{modality}_logit_contribution"] = np.where(available, 0.0, np.nan)
    prediction["router_fallback"] = prediction[SCORE_COLUMN].eq(prediction.primary_probability)
    prediction["router_reason"] = record["reason"]
    return prediction


def _frozen_gate_records(
    fold_gates: Sequence[Mapping[str, Any]], settings: CancerRouterConfig
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cancers = sorted({str(item["cancer_id"]) for item in fold_gates})
    for cancer in cancers:
        local = [item for item in fold_gates if item["cancer_id"] == cancer]
        record: dict[str, Any] = {
            "cancer_id": cancer,
            "selection_policy": "INNER_GATE_SUPPORT_ONLY_NO_HELDOUT_REVERSE_SELECTION",
            "required_support_folds": settings.frozen_gate_support_folds,
            "modalities": {},
            "deployment_candidate_requires_external_validation": True,
        }
        for modality in MODALITIES:
            enabled = [
                item["modalities"][modality]
                for item in local
                if item["modalities"].get(modality, {}).get("enabled") is True
            ]
            support = len(enabled)
            deployed = support >= settings.frozen_gate_support_folds
            record["modalities"][modality] = {
                "enabled": deployed,
                "support_folds": support,
                "weight": float(np.median([item["weight"] for item in enabled])) if deployed else 0.0,
                "centre": float(np.median([item["centre"] for item in enabled])) if deployed else 0.0,
                "unavailable_is_exact_primary_fallback": True,
            }
        result.append(record)
    return result


def _apply_frozen(frame: pd.DataFrame, frozen: Sequence[Mapping[str, Any]], settings: CancerRouterConfig) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    lookup = {str(item["cancer_id"]): item for item in frozen}
    for cancer, local in frame.groupby("cancer_id", observed=True, sort=False):
        record = lookup[str(cancer)]
        weights = tuple(float(record["modalities"][item]["weight"]) for item in MODALITIES)
        centres = tuple(float(record["modalities"][item]["centre"]) for item in MODALITIES)
        model = ResidualFusionModel(
            endpoint="discovery",
            expert_ids=MODALITIES,
            centres=centres,
            weights=weights,
            seed=settings.seed,
            initial_parameter_sha256="FROZEN_CONSENSUS_NOT_A_FRESH_OPTIMIZER_STATE",
            final_parameter_sha256=hashlib.sha256(repr((cancer, weights, centres)).encode()).hexdigest(),
            optimiser_steps=0,
            validation_logloss=float("nan"),
            clip_probability=settings.clip_probability,
        )
        applied = apply_residual_fusion_model(model, local, include_private_target=False)
        applied["router_fallback"] = applied[SCORE_COLUMN].eq(applied.primary_probability)
        for modality in MODALITIES:
            applied[f"{modality}_gate_enabled"] = bool(record["modalities"][modality]["enabled"])
        parts.append(applied)
    return pd.concat(parts, ignore_index=True).sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)


def train_cancer_modality_router(
    frame: pd.DataFrame,
    *,
    config: CancerRouterConfig | None = None,
) -> CancerRouterResult:
    settings = config or CancerRouterConfig()
    data = _validated_router_data(frame, settings)

    predictions: list[pd.DataFrame] = []
    fold_gates: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for cancer, cancer_frame in data.groupby("cancer_id", observed=True, sort=True):
        for heldout in range(settings.folds):
            validation_fold = (heldout + 1) % settings.folds
            train = cancer_frame.loc[
                ~cancer_frame[FUSION_FOLD_COLUMN].isin([heldout, validation_fold])
            ]
            validation = cancer_frame.loc[cancer_frame[FUSION_FOLD_COLUMN].eq(validation_fold)]
            test = cancer_frame.loc[cancer_frame[FUSION_FOLD_COLUMN].eq(heldout)]
            model, record = _fit_one_gate(
                train, validation, cancer=str(cancer), heldout=heldout, settings=settings
            )
            fold_gates.append(record)
            if model is None:
                prediction = _fallback_prediction(test, str(record["reason"]))
            else:
                prediction = apply_residual_fusion_model(model, test, include_private_target=True)
                prediction = _complete_prediction(prediction, test, model, record)
            target = prediction.fusion_target.to_numpy(float)
            primary = prediction.primary_probability.to_numpy(float)
            routed = prediction[SCORE_COLUMN].to_numpy(float)
            metric_rows.append(
                {
                    "cancer_id": str(cancer),
                    "heldout_pair_fold": heldout,
                    "validation_pair_fold": validation_fold,
                    "test_rows": len(test),
                    "primary_logloss": _logloss(target, primary, settings.clip_probability),
                    "routed_logloss": _logloss(target, routed, settings.clip_probability),
                    "primary_brier": _brier(target, primary),
                    "routed_brier": _brier(target, routed),
                    "heldout_used_for_gate_selection": False,
                    "router_status": record["status"],
                }
            )
            predictions.append(prediction)
    oof = pd.concat(predictions, ignore_index=True).sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)
    if len(oof) != len(data) or oof[list(TARGET_KEYS)].duplicated().any():
        raise CancerModalityRouterError("Nested router failed to predict every candidate exactly once")
    frozen = _frozen_gate_records(fold_gates, settings)
    candidate = _apply_frozen(data, frozen, settings)
    metrics = pd.DataFrame(metric_rows)
    target = oof.fusion_target.to_numpy(float)
    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame(
                [
                    {
                        "cancer_id": "ALL_CANCERS",
                        "heldout_pair_fold": "ALL_OOF",
                        "validation_pair_fold": "INNER_ONLY",
                        "test_rows": len(oof),
                        "primary_logloss": _logloss(target, oof.primary_probability.to_numpy(float), settings.clip_probability),
                        "routed_logloss": _logloss(target, oof[SCORE_COLUMN].to_numpy(float), settings.clip_probability),
                        "primary_brier": _brier(target, oof.primary_probability.to_numpy(float)),
                        "routed_brier": _brier(target, oof[SCORE_COLUMN].to_numpy(float)),
                        "heldout_used_for_gate_selection": False,
                        "router_status": "OOF_EVALUATION_ONLY_NOT_GATE_SELECTION",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return CancerRouterResult(oof, candidate, metrics, fold_gates, frozen)


def train_cancer_modality_router_validation_only(
    frame: pd.DataFrame,
    *,
    config: CancerRouterConfig | None = None,
) -> CancerRouterValidationResult:
    """Fit all nested gates and emit inner-validation predictions only.

    Each exact candidate appears once, as the validation fold for one outer
    split.  No outer-test subset is constructed and no outer metric is
    computed.  The returned gate records are sufficient for a later,
    winner-locked outer inference stage.
    """

    settings = config or CancerRouterConfig()
    data = _validated_router_data(frame, settings)
    predictions: list[pd.DataFrame] = []
    fold_gates: list[dict[str, Any]] = []
    for cancer, cancer_frame in data.groupby("cancer_id", observed=True, sort=True):
        for heldout in range(settings.folds):
            validation_fold = (heldout + 1) % settings.folds
            train = cancer_frame.loc[
                ~cancer_frame[FUSION_FOLD_COLUMN].isin([heldout, validation_fold])
            ]
            validation = cancer_frame.loc[
                cancer_frame[FUSION_FOLD_COLUMN].eq(validation_fold)
            ]
            model, record = _fit_one_gate(
                train,
                validation,
                cancer=str(cancer),
                heldout=heldout,
                settings=settings,
            )
            fold_gates.append(record)
            if model is None:
                prediction = _fallback_prediction(validation, str(record["reason"]))
            else:
                prediction = apply_residual_fusion_model(
                    model, validation, include_private_target=True
                )
                prediction = _complete_prediction(
                    prediction, validation, model, record
                )
            prediction["selection_outer_pair_fold"] = int(heldout)
            prediction["selection_validation_pair_fold"] = int(validation_fold)
            prediction["outer_test_queried"] = False
            predictions.append(prediction)
    validation_private = (
        pd.concat(predictions, ignore_index=True)
        .sort_values(list(TARGET_KEYS), kind="stable")
        .reset_index(drop=True)
    )
    if (
        len(validation_private) != len(data)
        or validation_private[list(TARGET_KEYS)].duplicated().any()
        or not validation_private["outer_test_queried"].eq(False).all()
    ):
        raise CancerModalityRouterError(
            "Validation-only router failed to cover each candidate exactly once"
        )
    expected = data[list(TARGET_KEYS)].sort_values(
        list(TARGET_KEYS), kind="stable"
    ).reset_index(drop=True)
    if not validation_private[list(TARGET_KEYS)].equals(expected):
        raise CancerModalityRouterError(
            "Validation-only router candidate universe drift"
        )
    return CancerRouterValidationResult(validation_private, fold_gates)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def materialize_cancer_modality_router_validation_only(
    result: CancerRouterValidationResult,
    *,
    output_root: str | Path,
    run_id: str,
    config: CancerRouterConfig,
    source_artifacts: Mapping[str, str | Path],
    training_budget_id: str,
) -> dict[str, Any]:
    """Seal validation-only router outputs without any outer prediction."""

    if not run_id.startswith("v32-router-validation-"):
        raise CancerModalityRouterError(
            "validation-only run_id must start with v32-router-validation-"
        )
    output = Path(output_root).resolve()
    if output.exists():
        raise CancerModalityRouterError(
            f"Validation-only router refuses output reuse: {output}"
        )
    frame = _canonical(result.validation_private, "router validation output")
    if "outer_test_queried" not in frame or not frame.outer_test_queried.eq(False).all():
        raise CancerModalityRouterError(
            "Validation-only materialization lacks the outer-test firewall"
        )
    output.mkdir(parents=True)
    prediction_path = output / "router_validation_predictions.PRIVATE.parquet"
    gates_path = output / "VALIDATION_GATES.json"
    contract_path = output / "VALIDATION_COMPARISON_CONTRACT.json"
    _atomic_parquet(frame, prediction_path)
    _atomic_json(
        gates_path,
        {
            "status": "PASS_VALIDATION_ONLY_GATES",
            "run_id": run_id,
            "config": asdict(config),
            "fold_gates": result.fold_gates,
            "outer_test_predictions_written": False,
            "outer_test_metrics_computed": False,
            "architecture_winner_selected": False,
        },
    )
    from .routed_fair_comparison import exact_candidate_and_fold_hashes

    candidate_sha, fold_sha = exact_candidate_and_fold_hashes(frame)
    source_records = {
        label: {
            "path": str(Path(path).resolve()),
            "sha256": artifact_sha256(path),
        }
        for label, path in source_artifacts.items()
    }
    contract = {
        "format": "CANCERLNCATLAS_V32_ROUTING_VALIDATION_COMPARISON_CONTRACT_V1",
        "candidate_universe_sha256": candidate_sha,
        "validation_pair_fold_sha256": fold_sha,
        "seed": int(config.seed),
        "training_budget_id": str(training_budget_id),
        "selection_split": "INNER_VALIDATION_ONLY",
        "outer_test_queries": 0,
    }
    _atomic_json(contract_path, contract)
    lineage = {
        "status": "PASS_VALIDATION_ONLY_EXTERNAL_ROUTER",
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "run_id": run_id,
        "candidate_only": True,
        "outer_test_predictions_written": False,
        "outer_test_metrics_computed": False,
        "architecture_winner_selected": False,
        "source_artifacts": source_records,
        "outputs": {
            "validation_predictions": {
                "path": str(prediction_path),
                "sha256": artifact_sha256(prediction_path),
            },
            "validation_gates": {
                "path": str(gates_path),
                "sha256": artifact_sha256(gates_path),
            },
            "validation_comparison_contract": {
                "path": str(contract_path),
                "sha256": artifact_sha256(contract_path),
            },
        },
    }
    lineage_path = output / "LINEAGE.json"
    _atomic_json(lineage_path, lineage)
    success = {
        "status": "PASS_VALIDATION_ONLY_EXTERNAL_ROUTER",
        "run_id": run_id,
        "validation_predictions_sha256": artifact_sha256(prediction_path),
        "validation_gates_sha256": artifact_sha256(gates_path),
        "comparison_contract_sha256": artifact_sha256(contract_path),
        "lineage_sha256": artifact_sha256(lineage_path),
        "outer_test_predictions_written": False,
        "outer_test_metrics_computed": False,
        "architecture_winner_selected": False,
    }
    _atomic_json(output / "SUCCESS.json", success)
    return success


def materialize_external_router_outer_after_winner_lock(
    frame: pd.DataFrame,
    *,
    validation_gates_path: str | Path,
    external_validation_predictions_path: str | Path,
    external_validation_success_path: str | Path,
    winner_declaration_path: str | Path,
    winner_declaration_sha256: str,
    output_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Run outer inference only after an external-router validation lock."""

    from .routing_validation_winner import WINNER_FORMAT

    gates_path = Path(validation_gates_path).resolve()
    validation_predictions_path = Path(external_validation_predictions_path).resolve()
    validation_success_path = Path(external_validation_success_path).resolve()
    winner_path = Path(winner_declaration_path).resolve()
    for label, path in (
        ("validation gates", gates_path),
        ("external validation predictions", validation_predictions_path),
        ("external validation success", validation_success_path),
        ("routing winner declaration", winner_path),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise CancerModalityRouterError(f"Missing {label}: {path}")
    expected_winner_sha = str(winner_declaration_sha256).lower()
    if len(expected_winner_sha) != 64 or artifact_sha256(winner_path) != expected_winner_sha:
        raise CancerModalityRouterError("Routing winner declaration SHA256 drift")
    try:
        winner = json.loads(winner_path.read_text(encoding="utf-8"))
        validation_success = json.loads(
            validation_success_path.read_text(encoding="utf-8")
        )
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CancerModalityRouterError("Winner/gate receipt is unreadable") from exc
    winner_required = {
        "format": WINNER_FORMAT,
        "status": "PASS_ROUTING_VALIDATION_WINNER_LOCK",
        "winner_id": "external_router",
        "selection_split": "validation_only",
        "heldout_test_used_for_selection": False,
        "outer_test_predictions_available_during_selection": False,
        "test_metrics_used_for_selection": False,
        "winner_locked_before_outer_test_inference": True,
    }
    if any(winner.get(key) != value for key, value in winner_required.items()):
        raise CancerModalityRouterError(
            "Routing winner does not authorize external outer inference"
        )
    winner_success_record = winner.get("sources", {}).get("external_success", {})
    winner_prediction_record = winner.get("sources", {}).get(
        "external_predictions", {}
    )
    if (
        winner_success_record.get("sha256")
        != artifact_sha256(validation_success_path)
        or winner_prediction_record.get("sha256")
        != artifact_sha256(validation_predictions_path)
        or validation_success.get("status")
        != "PASS_VALIDATION_ONLY_EXTERNAL_ROUTER"
        or validation_success.get("outer_test_predictions_written") is not False
        or validation_success.get("outer_test_metrics_computed") is not False
        or validation_success.get("architecture_winner_selected") is not False
        or validation_success.get("validation_gates_sha256")
        != artifact_sha256(gates_path)
    ):
        raise CancerModalityRouterError(
            "External validation success/gates are not bound to the winner"
        )
    if (
        gates.get("status") != "PASS_VALIDATION_ONLY_GATES"
        or gates.get("outer_test_predictions_written") is not False
        or gates.get("outer_test_metrics_computed") is not False
        or gates.get("architecture_winner_selected") is not False
    ):
        raise CancerModalityRouterError("Validation gates violate the outer firewall")
    try:
        settings = CancerRouterConfig(**dict(gates["config"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise CancerModalityRouterError("Validation gate config is invalid") from exc
    data = _validated_router_data(frame, settings)
    validation_frame = _canonical(
        pd.read_parquet(validation_predictions_path),
        "winner-bound external validation predictions",
    )
    if (
        len(validation_frame) != len(data)
        or not validation_frame[list(TARGET_KEYS)].equals(data[list(TARGET_KEYS)])
    ):
        raise CancerModalityRouterError(
            "Outer frame differs from winner-bound validation universe"
        )
    parity_columns = (
        "fusion_target",
        "primary_probability",
        FUSION_FOLD_COLUMN,
    )
    for column in parity_columns:
        left = pd.to_numeric(validation_frame[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(data[column], errors="coerce").to_numpy(float)
        if not np.allclose(left, right, rtol=0.0, atol=1e-7, equal_nan=False):
            raise CancerModalityRouterError(f"Outer frame parity drift: {column}")
    for modality in MODALITIES:
        available_column = f"{modality}_available"
        probability_column = f"{modality}_probability"
        left_available = validation_frame[available_column].astype(bool).to_numpy()
        right_available = data[available_column].astype(bool).to_numpy()
        if not np.array_equal(left_available, right_available):
            raise CancerModalityRouterError(
                f"Outer frame availability drift: {modality}"
            )
        left_probability = pd.to_numeric(
            validation_frame[probability_column], errors="coerce"
        ).to_numpy(float)
        right_probability = pd.to_numeric(
            data[probability_column], errors="coerce"
        ).to_numpy(float)
        if not np.allclose(
            left_probability[left_available],
            right_probability[right_available],
            rtol=0.0,
            atol=1e-7,
        ):
            raise CancerModalityRouterError(
                f"Outer frame modality probability drift: {modality}"
            )
    gate_records = gates.get("fold_gates")
    if not isinstance(gate_records, list):
        raise CancerModalityRouterError("Validation gates lack fold records")
    lookup: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in gate_records:
        if not isinstance(record, Mapping):
            raise CancerModalityRouterError("Validation gate record is invalid")
        key = (str(record.get("cancer_id", "")).upper(), int(record.get("heldout_pair_fold", -1)))
        if key in lookup:
            raise CancerModalityRouterError(f"Duplicate validation gate record: {key}")
        lookup[key] = record
    expected_keys = {
        (str(cancer), fold)
        for cancer in sorted(data.cancer_id.unique())
        for fold in range(settings.folds)
    }
    if set(lookup) != expected_keys:
        raise CancerModalityRouterError("Validation gates do not cover every cancer/fold")
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for cancer, local in data.groupby("cancer_id", observed=True, sort=True):
        for heldout in range(settings.folds):
            test = local.loc[local[FUSION_FOLD_COLUMN].eq(heldout)]
            record = lookup[(str(cancer), heldout)]
            if record.get("heldout_used_for_gate_selection") is not False:
                raise CancerModalityRouterError(
                    f"Gate used heldout data during selection: {cancer}/{heldout}"
                )
            if record.get("status") == "ROUTED":
                modalities = record.get("modalities")
                if not isinstance(modalities, Mapping) or set(modalities) != set(MODALITIES):
                    raise CancerModalityRouterError(
                        f"Gate modality record drift: {cancer}/{heldout}"
                    )
                weights = tuple(
                    float(modalities[item].get("weight", 0.0)) for item in MODALITIES
                )
                centres = tuple(
                    float(modalities[item].get("centre", 0.0)) for item in MODALITIES
                )
                model = ResidualFusionModel(
                    endpoint="discovery",
                    expert_ids=MODALITIES,
                    centres=centres,
                    weights=weights,
                    seed=int(settings.seed + heldout * 1009 + sum(map(ord, str(cancer)))),
                    initial_parameter_sha256="WINNER_LOCKED_VALIDATION_GATE",
                    final_parameter_sha256=hashlib.sha256(
                        repr((cancer, heldout, weights, centres)).encode()
                    ).hexdigest(),
                    optimiser_steps=0,
                    validation_logloss=float(record.get("inner_gated_logloss", np.nan)),
                    clip_probability=settings.clip_probability,
                )
                prediction = apply_residual_fusion_model(
                    model, test, include_private_target=True
                )
                prediction = _complete_prediction(prediction, test, model, record)
            elif record.get("status") == "PRIMARY_FALLBACK":
                prediction = _fallback_prediction(test, str(record.get("reason")))
            else:
                raise CancerModalityRouterError(
                    f"Unknown validation gate status: {cancer}/{heldout}"
                )
            prediction["winner_declaration_sha256"] = expected_winner_sha
            prediction["outer_inference_after_winner_lock"] = True
            target = prediction.fusion_target.to_numpy(float)
            primary_probability = prediction.primary_probability.to_numpy(float)
            routed_probability = prediction[SCORE_COLUMN].to_numpy(float)
            metrics.append(
                {
                    "cancer_id": str(cancer),
                    "heldout_pair_fold": heldout,
                    "rows": len(prediction),
                    "primary_logloss": _logloss(
                        target, primary_probability, settings.clip_probability
                    ),
                    "routed_logloss": _logloss(
                        target, routed_probability, settings.clip_probability
                    ),
                    "primary_brier": _brier(target, primary_probability),
                    "routed_brier": _brier(target, routed_probability),
                    "outer_metric_computed_after_winner_lock": True,
                }
            )
            predictions.append(prediction)
    oof = (
        pd.concat(predictions, ignore_index=True)
        .sort_values(list(TARGET_KEYS), kind="stable")
        .reset_index(drop=True)
    )
    if (
        len(oof) != len(data)
        or oof[list(TARGET_KEYS)].duplicated().any()
        or not oof.outer_inference_after_winner_lock.eq(True).all()
    ):
        raise CancerModalityRouterError("Winner-locked external OOF coverage drift")
    metric_frame = pd.DataFrame(metrics)
    target = oof.fusion_target.to_numpy(float)
    primary_probability = oof.primary_probability.to_numpy(float)
    routed_probability = oof[SCORE_COLUMN].to_numpy(float)
    metric_frame = pd.concat(
        [
            metric_frame,
            pd.DataFrame(
                [
                    {
                        "cancer_id": "ALL_CANCERS",
                        "heldout_pair_fold": "ALL_OOF",
                        "rows": len(oof),
                        "primary_logloss": _logloss(
                            target, primary_probability, settings.clip_probability
                        ),
                        "routed_logloss": _logloss(
                            target, routed_probability, settings.clip_probability
                        ),
                        "primary_brier": _brier(target, primary_probability),
                        "routed_brier": _brier(target, routed_probability),
                        "outer_metric_computed_after_winner_lock": True,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    metric_frame["heldout_pair_fold"] = metric_frame[
        "heldout_pair_fold"
    ].astype(str)
    # Post-lock conditional modality ablations are descriptive outer metrics,
    # never inputs to gate fitting or architecture selection.  Removing one
    # modality is exact here because the external router is additive in logit
    # space and records each modality's realized contribution.
    ablation_rows: list[dict[str, Any]] = []
    routed_probability = np.clip(
        pd.to_numeric(oof[SCORE_COLUMN], errors="raise").to_numpy(float),
        settings.clip_probability,
        1.0 - settings.clip_probability,
    )
    routed_logit = np.log(routed_probability / (1.0 - routed_probability))
    target = pd.to_numeric(oof.fusion_target, errors="raise").to_numpy(float)
    ablation_groups = [("ALL_CANCERS", np.ones(len(oof), dtype=bool))] + [
        (str(cancer), oof.cancer_id.eq(cancer).to_numpy())
        for cancer in sorted(oof.cancer_id.unique())
    ]
    for modality in MODALITIES:
        contribution = pd.to_numeric(
            oof[f"{modality}_logit_contribution"], errors="coerce"
        ).to_numpy(float)
        contribution = np.where(np.isfinite(contribution), contribution, 0.0)
        ablated_probability = 1.0 / (
            1.0 + np.exp(-(routed_logit - contribution))
        )
        available = oof[f"{modality}_available"].astype(bool).to_numpy()
        enabled = oof[f"{modality}_gate_enabled"].astype(bool).to_numpy()
        for cancer, mask in ablation_groups:
            full_loss = _logloss(
                target[mask], routed_probability[mask], settings.clip_probability
            )
            ablated_loss = _logloss(
                target[mask], ablated_probability[mask], settings.clip_probability
            )
            full_brier = _brier(target[mask], routed_probability[mask])
            ablated_brier = _brier(target[mask], ablated_probability[mask])
            ablation_rows.append(
                {
                    "cancer_id": cancer,
                    "modality": modality,
                    "rows": int(mask.sum()),
                    "available_rows": int((mask & available).sum()),
                    "gate_enabled_rows": int((mask & enabled).sum()),
                    "full_logloss": full_loss,
                    "ablated_logloss": ablated_loss,
                    "conditional_ablation_logloss_delta": ablated_loss - full_loss,
                    "full_brier": full_brier,
                    "ablated_brier": ablated_brier,
                    "conditional_ablation_brier_delta": ablated_brier - full_brier,
                    "outer_metric_computed_after_winner_lock": True,
                    "used_for_model_or_gate_selection": False,
                }
            )
    ablation_frame = pd.DataFrame(ablation_rows)
    output = Path(output_root).resolve()
    staging = output.with_name(f"{output.name}.partial")
    if output.exists() or staging.exists():
        raise CancerModalityRouterError(
            f"External outer inference refuses output/staging reuse: {output}"
        )
    staging.mkdir(parents=True)
    try:
        prediction_path = staging / "external_router_winner_locked_oof.PRIVATE.parquet"
        metrics_path = staging / "external_router_winner_locked_metrics.parquet"
        ablation_path = staging / "external_router_modality_ablation_metrics.parquet"
        _atomic_parquet(oof, prediction_path)
        _atomic_parquet(metric_frame, metrics_path)
        _atomic_parquet(ablation_frame, ablation_path)
        success = {
            "status": "PASS_EXTERNAL_ROUTER_OUTER_AFTER_WINNER_LOCK",
            "run_id": run_id,
            "winner_id": "external_router",
            "winner_declaration": {
                "path": str(winner_path),
                "sha256": expected_winner_sha,
            },
            "validation_gates": {
                "path": str(gates_path),
                "sha256": artifact_sha256(gates_path),
            },
            "validation_predictions": {
                "path": str(validation_predictions_path),
                "sha256": artifact_sha256(validation_predictions_path),
            },
            "outer_predictions": {
                "path": str(output / prediction_path.name),
                "sha256": artifact_sha256(prediction_path),
            },
            "outer_metrics": {
                "path": str(output / metrics_path.name),
                "sha256": artifact_sha256(metrics_path),
            },
            "modality_ablation_metrics": {
                "path": str(output / ablation_path.name),
                "sha256": artifact_sha256(ablation_path),
            },
            "outer_inference_started_after_winner_lock": True,
            "test_metrics_used_for_selection": False,
            "formal_v32_primary_unchanged": True,
        }
        (staging / "SUCCESS.json").write_text(
            json.dumps(success, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(output)
        return success
    except Exception:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise


def materialize_cancer_modality_router(
    result: CancerRouterResult,
    *,
    output_root: str | Path,
    run_id: str,
    config: CancerRouterConfig,
    source_artifacts: Mapping[str, str | Path],
    training_budget_id: str = "v32-routed-equal-budget-v1",
) -> dict[str, Any]:
    if not run_id.startswith("v32-routed-candidate-"):
        raise CancerModalityRouterError("run_id must start with v32-routed-candidate-")
    has_atac_predictions = "atac_predictions" in source_artifacts
    has_atac_lineage = "atac_lineage" in source_artifacts
    if has_atac_predictions != has_atac_lineage:
        raise CancerModalityRouterError(
            "ATAC predictions and ATAC lineage must be supplied together"
        )
    if has_atac_predictions:
        atac_prediction_path = Path(source_artifacts["atac_predictions"]).resolve()
        atac_lineage_path = Path(source_artifacts["atac_lineage"]).resolve()
        try:
            atac_lineage = json.loads(atac_lineage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CancerModalityRouterError("ATAC lineage is unreadable") from exc
        validate_atac_oof_lineage(
            atac_lineage,
            predictions_sha256=artifact_sha256(atac_prediction_path),
        )
    if "genomic_predictions" not in source_artifacts or "genomic_lineage" not in source_artifacts:
        raise CancerModalityRouterError(
            "Genomic predictions and genomic lineage must be supplied together"
        )
    genomic_prediction_path = Path(source_artifacts["genomic_predictions"]).resolve()
    genomic_lineage_path = Path(source_artifacts["genomic_lineage"]).resolve()
    try:
        genomic_lineage = json.loads(genomic_lineage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CancerModalityRouterError("Genomic lineage is unreadable") from exc
    validate_patient_oof_lineage(
        genomic_lineage,
        predictions_sha256=artifact_sha256(genomic_prediction_path),
    )
    output = Path(output_root).resolve()
    if output.exists():
        raise CancerModalityRouterError(f"Router refuses output reuse: {output}")
    output.mkdir(parents=True)
    private_path = output / "cancer_modality_oof.PRIVATE.parquet"
    public_path = output / "routed_candidate_scores.parquet"
    metrics_path = output / "router_metrics.parquet"
    gates_path = output / "ROUTER_GATES.json"
    _atomic_parquet(result.oof_private, private_path)
    _atomic_parquet(result.routed_candidate, public_path)
    _atomic_parquet(result.metrics, metrics_path)
    gate_payload = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "run_id": run_id,
        "config": asdict(config),
        "fold_gates": result.fold_gates,
        "frozen_candidate_gates": result.frozen_gates,
        "heldout_test_metrics_never_used_to_reverse_select_gates": True,
        "frozen_gate_policy": "CONSENSUS_OF_INNER_VALIDATION_GATES",
        "requires_untouched_external_validation_before_primary_promotion": True,
    }
    _atomic_json(gates_path, gate_payload)
    sources = {
        label: {"path": str(Path(path).resolve()), "sha256": artifact_sha256(path)}
        for label, path in source_artifacts.items()
    }
    from .routed_fair_comparison import (
        exact_candidate_and_fold_hashes,
        modality_source_composite_sha256,
    )

    candidate_sha, fold_sha = exact_candidate_and_fold_hashes(result.oof_private)
    modality_hashes = {
        key: value["sha256"]
        for key, value in sources.items()
        if key in {"genomic_predictions", "atac_predictions"}
    }
    comparison_contract = {
        "candidate_universe_sha256": candidate_sha,
        "patient_modality_oof_sha256": modality_source_composite_sha256(modality_hashes),
        "outer_pair_fold_sha256": fold_sha,
        "seed": int(config.seed),
        "training_budget_id": str(training_budget_id),
    }
    comparison_contract_path = output / "COMPARISON_CONTRACT.json"
    _atomic_json(comparison_contract_path, comparison_contract)
    lineage = {
        "status": "SUCCESS",
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "run_id": run_id,
        "candidate_only": True,
        "formal_v32_primary_unchanged": True,
        "hhgt_core_modified": False,
        "outer_pair_blocked_folds": 5,
        "patient_level_modality_oof_required": True,
        "test_reverse_selection_forbidden": True,
        "typed_unavailable_atac_supported": True,
        "source_artifacts": sources,
        "outputs": {
            "oof_private": {"path": str(private_path), "sha256": artifact_sha256(private_path)},
            "routed_candidate": {"path": str(public_path), "sha256": artifact_sha256(public_path)},
            "metrics": {"path": str(metrics_path), "sha256": artifact_sha256(metrics_path)},
            "gates": {"path": str(gates_path), "sha256": artifact_sha256(gates_path)},
            "comparison_contract": {
                "path": str(comparison_contract_path),
                "sha256": artifact_sha256(comparison_contract_path),
            },
        },
    }
    lineage_path = output / "LINEAGE.json"
    _atomic_json(lineage_path, lineage)
    success = {
        "status": "SUCCESS",
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "run_id": run_id,
        "lineage_path": str(lineage_path),
        "lineage_sha256": artifact_sha256(lineage_path),
        "candidate_only": True,
        "formal_v32_primary_unchanged": True,
    }
    _atomic_json(output / "SUCCESS.json", success)
    return success


__all__ = [
    "ANALYSIS_VERSION",
    "CancerModalityRouterError",
    "CancerRouterConfig",
    "CancerRouterResult",
    "CancerRouterValidationResult",
    "MODALITIES",
    "build_cancer_router_frame",
    "materialize_cancer_modality_router",
    "materialize_cancer_modality_router_validation_only",
    "materialize_external_router_outer_after_winner_lock",
    "train_cancer_modality_router",
    "train_cancer_modality_router_validation_only",
    "validate_atac_oof_lineage",
    "validate_patient_oof_lineage",
]
