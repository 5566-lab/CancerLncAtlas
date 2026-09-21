"""Immutable input staging for the V3.2 routing architecture comparison.

This module prepares data only.  It never trains an arm, evaluates an outer
test fold, computes comparison metrics, or chooses a winner.  Both the
external router and the end-to-end hierarchical gate are bound to one shared
candidate/label/modality frame and one explicit split/budget contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


TARGET_KEYS = ("cancer_id", "lncrna_id", "pathway_id")
MODALITIES = ("mutation", "cnv", "atac")
FOLD_COLUMN = "fusion_pair_fold"
BUDGET_FORMAT = "CC_HHGT_V3_2_ROUTING_FAIR_BUDGET_V1"
FREEZE_FORMAT = "CC_HHGT_V3_2_ATAC_R3_FAIR_COMPARE_FREEZE_V1"
CONTRACT_FORMAT = "CC_HHGT_V3_2_ROUTING_FAIR_INPUT_CONTRACT_V1"
ARM_FORMAT = "CC_HHGT_V3_2_ROUTING_ARM_INPUT_V1"
SPLIT_FORMAT = "CC_HHGT_V3_2_ROUTING_NESTED_SPLIT_POLICY_V1"
SUCCESS_FORMAT = "CC_HHGT_V3_2_ROUTING_FAIR_INPUT_SUCCESS_V1"
VALIDATION_FORMAT = "CC_HHGT_V3_2_ROUTING_FAIR_INPUT_VALIDATION_V1"
ARCHITECTURES = ("external_router", "hierarchical_end_to_end")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RoutingFairInputError(RuntimeError):
    """Raised when comparison staging would violate fairness or isolation."""


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pair_blocked_fold(lncrna_id: object, pathway_id: object, *, seed: int) -> int:
    token = f"{str(lncrna_id).strip()}|{str(pathway_id).strip()}|{int(seed)}"
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") % 5


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RoutingFairInputError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoutingFairInputError(f"Unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RoutingFairInputError(f"{label} is not a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_sha(value: object, label: str) -> str:
    text = str(value).lower()
    if not _SHA256.fullmatch(text):
        raise RoutingFairInputError(f"{label} is not a SHA256")
    return text


def _validate_budget(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("format") != BUDGET_FORMAT:
        raise RoutingFairInputError("Fair budget format drift")
    scope = value.get("scope")
    seeds = value.get("seeds")
    selection = value.get("selection_policy")
    ceilings = value.get("architecture_optimizer_ceilings")
    shared_budget = value.get("shared_optimization_budget")
    if not all(
        isinstance(item, Mapping)
        for item in (scope, seeds, selection, ceilings, shared_budget)
    ):
        raise RoutingFairInputError("Fair budget lacks scope/seeds/selection/ceilings")
    cancers = [str(item).upper() for item in scope.get("formal_cancers", [])]
    if not cancers or len(cancers) != len(set(cancers)):
        raise RoutingFairInputError("Fair budget cancer scope is empty or duplicated")
    if int(scope.get("candidate_rows_per_cancer", 0)) <= 0:
        raise RoutingFairInputError("Fair budget candidate count is invalid")
    if tuple(scope.get("modalities", ())) != MODALITIES:
        raise RoutingFairInputError("Fair budget modality order drift")
    for key in ("optimizer_seed", "pair_fold_seed", "patient_fold_seed"):
        if not isinstance(seeds.get(key), int):
            raise RoutingFairInputError(f"Fair budget lacks integer {key}")
    expected_selection = {
        "outer_folds": 5,
        "validation_offset": 1,
        "training_folds_per_outer": 3,
        "validation_folds_per_outer": 1,
        "outer_test_queries_during_selection": 0,
        "outer_test_evaluations_per_arm_fold": 1,
        "winner_selection_queries": 0,
        "promotion_or_winner_selection_enabled": False,
        "selection_scope": "INNER_VALIDATION_FOLD_ONLY",
        "test_policy": "ONE_FINAL_EVALUATION_AFTER_ARM_SELECTION_IS_FROZEN",
    }
    drift = {
        key: {"expected": expected, "observed": selection.get(key)}
        for key, expected in expected_selection.items()
        if selection.get(key) != expected
    }
    if drift:
        raise RoutingFairInputError(f"Fair split/selection policy drift: {drift}")
    if set(ceilings) != set(ARCHITECTURES):
        raise RoutingFairInputError("Both architecture optimizer ceilings must be frozen")
    shared_limits = {
        "max_candidate_coverage_cycles": 10,
        "max_validation_checkpoints": 10,
        "patience_validation_checkpoints": 3,
    }
    if any(shared_budget.get(key) != expected for key, expected in shared_limits.items()):
        raise RoutingFairInputError("Shared normalized optimization budget drift")
    for architecture in ARCHITECTURES:
        ceiling = ceilings[architecture]
        if not isinstance(ceiling, Mapping) or any(
            ceiling.get(key) != expected for key, expected in shared_limits.items()
        ):
            raise RoutingFairInputError(
                f"{architecture} does not map to the same normalized optimization budget"
            )
    if not str(value.get("budget_id", "")).strip():
        raise RoutingFairInputError("Fair budget lacks budget_id")
    return dict(value)


def _validate_patient_folds(
    path: Path, *, cancers: Sequence[str], expected_seed: int
) -> dict[str, Any]:
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype={"cancer_id": "string", "sample_id": "string", "patient_id": "string"},
    )
    required = {"cancer_id", "sample_id", "patient_id", "patient_fold_id", "fold_seed"}
    if missing := sorted(required - set(frame.columns)):
        raise RoutingFairInputError(f"Patient fold manifest lacks: {missing}")
    if frame[list(required)].isna().any().any():
        raise RoutingFairInputError("Patient fold manifest contains null contract fields")
    frame["cancer_id"] = frame.cancer_id.astype(str).str.upper()
    frame["patient_fold_id"] = pd.to_numeric(frame.patient_fold_id, errors="raise").astype(int)
    frame["fold_seed"] = pd.to_numeric(frame.fold_seed, errors="raise").astype(int)
    if set(frame.patient_fold_id.unique()) != set(range(5)):
        raise RoutingFairInputError("Patient fold manifest is not exactly folds 0..4")
    if set(frame.fold_seed.unique()) != {int(expected_seed)}:
        raise RoutingFairInputError("Patient fold seed differs from the fair budget")
    absent = sorted(set(cancers) - set(frame.cancer_id.unique()))
    if absent:
        raise RoutingFairInputError(f"Patient fold manifest lacks cancers: {absent}")
    for column in ("sample_id", "patient_id"):
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise RoutingFairInputError(f"Patient fold manifest contains empty {column}")
    sample_mapping = frame.groupby(["cancer_id", "sample_id"], observed=True).patient_id.nunique()
    if sample_mapping.gt(1).any():
        raise RoutingFairInputError("One cancer/sample maps to multiple explicit patients")
    conflict = (
        frame.groupby(["cancer_id", "patient_id"], observed=True).patient_fold_id.nunique()
        > 1
    )
    if conflict.any():
        raise RoutingFairInputError("One cancer/patient is assigned to multiple patient folds")
    if frame[["cancer_id", "sample_id"]].duplicated().any():
        raise RoutingFairInputError("Patient fold manifest duplicates a cancer/sample")
    return {
        "rows": int(len(frame)),
        "patients": int(frame[["cancer_id", "patient_id"]].drop_duplicates().shape[0]),
        "cancers": sorted(frame.cancer_id.unique().tolist()),
        "fold_seed": int(expected_seed),
        "folds": 5,
    }


def _validate_genomic_lineage(
    *, lineage: Mapping[str, Any], success: Mapping[str, Any], prediction_sha: str,
    lineage_sha: str, expected_rows: int
) -> None:
    if success.get("status") != "SUCCESS" or lineage.get("training_status") != "SUCCESS":
        raise RoutingFairInputError("Mutation/CNV fresh training is not SUCCESS")
    if not str(lineage.get("analysis_version", "")).startswith("CancerLncAtlas_V3.2"):
        raise RoutingFairInputError("Mutation/CNV lineage is not V3.2")
    checks = {
        "five_folds": int(lineage.get("folds", -1)) == 5,
        "old_checkpoint_forbidden": lineage.get("old_checkpoint_loaded") is False,
        "old_predictions_forbidden": lineage.get("old_predictions_used_as_features") is False,
        "old_rankings_forbidden": lineage.get("old_rankings_used_as_outputs") is False,
        "fresh_heads": lineage.get("private_head_trained_from_scratch") is True,
        "core_frozen": lineage.get("core_parameters_frozen") is True,
        "fold_outputs_unaveraged": (
            lineage.get("patient_fold_oof_predictions_not_fold_averaged") is True
        ),
        "prediction_hash_bound": lineage.get("prediction_sha256") == prediction_sha,
        "success_prediction_hash_bound": success.get("prediction_sha256") == prediction_sha,
        "success_lineage_hash_bound": success.get("lineage_sha256") == lineage_sha,
        "candidate_rows_preserved": int(success.get("candidate_rows_preserved", -1)) == expected_rows,
        "null_never_zero": success.get("null_is_never_zero_or_wildtype") is True,
    }
    if failed := sorted(key for key, ok in checks.items() if not ok):
        raise RoutingFairInputError(f"Mutation/CNV lineage contract failed: {failed}")
    modalities = lineage.get("modalities")
    if not isinstance(modalities, Mapping):
        raise RoutingFairInputError("Mutation/CNV lineage lacks per-modality fold status")
    for modality in ("mutation", "cnv"):
        records = modalities.get(modality)
        if not isinstance(records, list) or len(records) != 5:
            raise RoutingFairInputError(f"{modality} does not have five fold records")
        if {int(item.get("patient_fold", -1)) for item in records} != set(range(5)):
            raise RoutingFairInputError(f"{modality} patient fold set differs from 0..4")
        if any(item.get("status") != "SUCCESS" for item in records):
            raise RoutingFairInputError(f"{modality} has a non-successful fold")


def _validate_atac_freeze(
    *, freeze: Mapping[str, Any], candidate_sha: str, prediction_sha: str,
    lineage_sha: str, training_success_sha: str, audit_sha: str,
    audit_success_sha: str, lineage: Mapping[str, Any],
    training_success: Mapping[str, Any], audit: Mapping[str, Any],
    audit_success: Mapping[str, Any]
) -> None:
    if freeze.get("format") != FREEZE_FORMAT:
        raise RoutingFairInputError("ATAC freeze contract format drift")
    observed = {
        "candidate_authority_sha256": candidate_sha,
        "prediction_sha256": prediction_sha,
        "lineage_sha256": lineage_sha,
        "training_success_sha256": training_success_sha,
        "independent_audit_sha256": audit_sha,
        "independent_audit_success_sha256": audit_success_sha,
    }
    for key, value in observed.items():
        if _require_sha(freeze.get(key), f"ATAC freeze {key}") != value:
            raise RoutingFairInputError(f"Frozen ATAC r3 mismatch: {key}")
    if training_success.get("status") != "SUCCESS" or training_success.get(
        "success_written_last"
    ) is not True:
        raise RoutingFairInputError("Frozen ATAC training marker is not final SUCCESS")
    if training_success.get("prediction_sha256") != prediction_sha:
        raise RoutingFairInputError("Frozen ATAC training marker prediction hash drift")
    if training_success.get("lineage_sha256") != lineage_sha:
        raise RoutingFairInputError("Frozen ATAC training marker lineage hash drift")
    if int(training_success.get("prediction_rows", -1)) != int(freeze.get("prediction_rows", -2)):
        raise RoutingFairInputError("Frozen ATAC training row count drift")
    lineage_checks = {
        "v32": str(lineage.get("analysis_version", "")).startswith("CancerLncAtlas_V3.2"),
        "atac": str(lineage.get("modality", "")).lower() == "atac",
        "five_folds": int(lineage.get("folds", -1)) == 5,
        "patient_oof": lineage.get("patient_level_modality_oof") is True,
        "unaveraged_folds": lineage.get("patient_fold_oof_predictions_not_fold_averaged") is True,
        "outer_test_never_fit": int(lineage.get("outer_test_patients_used_in_any_upstream_fit", -1)) == 0,
        "old_checkpoint_forbidden": lineage.get("old_checkpoint_loaded") is False,
        "old_prediction_forbidden": lineage.get("old_predictions_used_as_features") is False,
        "old_ranking_forbidden": lineage.get("old_rankings_used_as_outputs") is False,
        "prediction_hash_bound": lineage.get("predictions_sha256") == prediction_sha,
    }
    if failed := sorted(key for key, ok in lineage_checks.items() if not ok):
        raise RoutingFairInputError(f"Frozen ATAC lineage failed: {failed}")
    fold_status = lineage.get("fold_status")
    if not isinstance(fold_status, list) or len(fold_status) != 5:
        raise RoutingFairInputError("Frozen ATAC lineage lacks five fold records")
    if {int(item.get("patient_fold", -1)) for item in fold_status} != set(range(5)):
        raise RoutingFairInputError("Frozen ATAC patient fold set differs from 0..4")
    if any(
        item.get("status") != "SUCCESS"
        or item.get("heldout_patients_used_for_fit") is not False
        for item in fold_status
    ):
        raise RoutingFairInputError("Frozen ATAC fold isolation failed")
    audit_checks = {
        "pass": audit.get("status") == "PASS",
        "candidate_hash": audit.get("candidate_authority_sha256") == candidate_sha,
        "prediction_hash": audit.get("prediction_sha256") == prediction_sha,
        "prediction_rows": int(audit.get("prediction_rows", -1)) == int(freeze.get("prediction_rows", -2)),
        "exact_keys": audit.get("exact_candidate_keys_match_authority") is True,
        "typed_null": audit.get("typed_null_never_zero") is True,
        "no_old": audit.get("old_predictions_checkpoints_or_rankings_used") is False,
        "no_early_claim": audit.get("formal_positive_contribution_claimed") is False,
        "audit_success": audit_success.get("status") == "PASS",
        "audit_hash_bound": audit_success.get("audit_sha256") == audit_sha,
        "audit_prediction_bound": audit_success.get("prediction_sha256") == prediction_sha,
    }
    if failed := sorted(key for key, ok in audit_checks.items() if not ok):
        raise RoutingFairInputError(f"Frozen ATAC independent audit failed: {failed}")


def _parquet_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema_arrow.names)


def _read_cancer(path: Path, columns: Sequence[str], cancer: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    return pq.read_table(
        path, columns=list(columns), filters=[("cancer_id", "=", cancer)]
    ).to_pandas()


def _canonical_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if missing := sorted(set(TARGET_KEYS) - set(frame.columns)):
        raise RoutingFairInputError(f"{label} lacks exact keys: {missing}")
    value = frame.copy()
    value["cancer_id"] = value.cancer_id.astype(str).str.upper()
    value["lncrna_id"] = value.lncrna_id.astype(str)
    value["pathway_id"] = value.pathway_id.astype(str)
    if value[list(TARGET_KEYS)].duplicated().any():
        raise RoutingFairInputError(f"{label} has duplicate exact keys")
    return value.sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)


def _strict_boolean(series: pd.Series, label: str) -> np.ndarray:
    if series.isna().any():
        raise RoutingFairInputError(f"{label} contains null availability")
    if not pd.api.types.is_bool_dtype(series.dtype) and not all(
        isinstance(value, (bool, np.bool_)) for value in series.tolist()
    ):
        raise RoutingFairInputError(f"{label} is not strictly boolean")
    return series.astype(bool).to_numpy()


def _typed_modality(
    frame: pd.DataFrame, *, modality: str, probability_column: str,
    availability_column: str, reason_column: str
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    required = {probability_column, availability_column, reason_column}
    if missing := sorted(required - set(frame.columns)):
        raise RoutingFairInputError(f"{modality} prediction lacks: {missing}")
    available = _strict_boolean(frame[availability_column], f"{modality}_available")
    probability = pd.to_numeric(frame[probability_column], errors="coerce").to_numpy(float)
    reason = frame[reason_column].astype("string")
    if np.isnan(probability[available]).any() or not np.isfinite(probability[available]).all():
        raise RoutingFairInputError(f"Available {modality} probability is not finite")
    if ((probability[available] < 0) | (probability[available] > 1)).any():
        raise RoutingFairInputError(f"Available {modality} probability is outside [0,1]")
    if np.isfinite(probability[~available]).any():
        raise RoutingFairInputError(f"Unavailable {modality} is encoded as a number")
    if reason.loc[~available].isna().any():
        raise RoutingFairInputError(f"Unavailable {modality} lacks a typed reason")
    if reason.loc[available].notna().any():
        raise RoutingFairInputError(f"Available {modality} incorrectly has an unavailable reason")
    return probability, available, reason


def _split_policy(budget: Mapping[str, Any]) -> dict[str, Any]:
    selection = budget["selection_policy"]
    folds = int(selection["outer_folds"])
    offset = int(selection["validation_offset"])
    records = []
    for test_fold in range(folds):
        validation_fold = (test_fold + offset) % folds
        training_folds = sorted(set(range(folds)) - {test_fold, validation_fold})
        records.append(
            {
                "outer_test_fold": test_fold,
                "validation_fold": validation_fold,
                "training_folds": training_folds,
                "selection_reads": ["training_folds", "validation_fold"],
                "selection_may_read_outer_test": False,
                "consumer_fold_access": {
                    "scaling_fit": training_folds,
                    "early_stopping": [validation_fold],
                    "checkpoint_choice": [validation_fold],
                    "route_threshold_selection": [validation_fold],
                    "final_test_evaluation": [test_fold],
                },
                "outer_test_keys_allowed_consumers": ["final_test_evaluation"],
                "outer_test_labels_allowed_consumers": ["final_test_evaluation"],
                "outer_test_evaluations_allowed": 1,
                "outer_test_may_change_checkpoint_gate_or_hyperparameters": False,
            }
        )
    return {
        "format": SPLIT_FORMAT,
        "fold_column": FOLD_COLUMN,
        "pair_fold_seed": int(budget["seeds"]["pair_fold_seed"]),
        "patient_fold_seed": int(budget["seeds"]["patient_fold_seed"]),
        "optimizer_seed": int(budget["seeds"]["optimizer_seed"]),
        "selection_scope": "INNER_VALIDATION_FOLD_ONLY",
        "test_policy": "ONE_FINAL_EVALUATION_AFTER_ARM_SELECTION_IS_FROZEN",
        "winner_selection_enabled": False,
        "fold_records": records,
    }


def _fold_file_manifest(root: Path) -> list[dict[str, Any]]:
    records = []
    for fold in range(5):
        path = root / f"PATIENT_FOLD_{fold}.pt"
        if not path.is_file() or path.stat().st_size <= 0:
            raise RoutingFairInputError(f"Formal prepared baseline fold is missing: {path}")
        records.append(
            {
                "patient_fold": fold,
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": artifact_sha256(path),
            }
        )
    return records


def _composite_records_sha(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256(
        [
            {
                "patient_fold": int(item["patient_fold"]),
                "bytes": int(item["bytes"]),
                "sha256": str(item["sha256"]),
            }
            for item in records
        ]
    )


def stage_routing_fair_inputs(
    *, candidate_authority_path: str | Path, patient_fold_manifest_path: str | Path,
    formal_prepared_root: str | Path, primary_oof_path: str | Path,
    genomic_predictions_path: str | Path, genomic_lineage_path: str | Path,
    genomic_success_path: str | Path, atac_predictions_path: str | Path,
    atac_lineage_path: str | Path, atac_training_success_path: str | Path,
    atac_audit_path: str | Path, atac_audit_success_path: str | Path,
    budget_contract_path: str | Path, atac_freeze_contract_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Stage one immutable frame and two hash-identical arm input contracts."""

    paths = {
        "candidate_authority": Path(candidate_authority_path).resolve(),
        "patient_fold_manifest": Path(patient_fold_manifest_path).resolve(),
        "primary_oof": Path(primary_oof_path).resolve(),
        "genomic_predictions": Path(genomic_predictions_path).resolve(),
        "genomic_lineage": Path(genomic_lineage_path).resolve(),
        "genomic_success": Path(genomic_success_path).resolve(),
        "atac_predictions": Path(atac_predictions_path).resolve(),
        "atac_lineage": Path(atac_lineage_path).resolve(),
        "atac_training_success": Path(atac_training_success_path).resolve(),
        "atac_audit": Path(atac_audit_path).resolve(),
        "atac_audit_success": Path(atac_audit_success_path).resolve(),
        "budget_contract": Path(budget_contract_path).resolve(),
        "atac_freeze_contract": Path(atac_freeze_contract_path).resolve(),
    }
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise RoutingFairInputError(f"Missing fair comparison source {label}: {path}")
    prepared_root = Path(formal_prepared_root).resolve()
    output = Path(output_root).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() or staging.exists():
        raise RoutingFairInputError(f"Fair input staging refuses output/staging reuse: {output}")

    budget = _validate_budget(_read_json(paths["budget_contract"], "fair budget"))
    freeze = _read_json(paths["atac_freeze_contract"], "ATAC r3 freeze contract")
    cancers = [str(item).upper() for item in budget["scope"]["formal_cancers"]]
    rows_per_cancer = int(budget["scope"]["candidate_rows_per_cancer"])
    expected_rows = rows_per_cancer * len(cancers)
    source_hashes = {key: artifact_sha256(path) for key, path in paths.items()}
    patient_audit = _validate_patient_folds(
        paths["patient_fold_manifest"],
        cancers=cancers,
        expected_seed=int(budget["seeds"]["patient_fold_seed"]),
    )
    genomic_lineage = _read_json(paths["genomic_lineage"], "Mutation/CNV lineage")
    genomic_success = _read_json(paths["genomic_success"], "Mutation/CNV SUCCESS")
    _validate_genomic_lineage(
        lineage=genomic_lineage,
        success=genomic_success,
        prediction_sha=source_hashes["genomic_predictions"],
        lineage_sha=source_hashes["genomic_lineage"],
        expected_rows=expected_rows,
    )
    atac_lineage = _read_json(paths["atac_lineage"], "ATAC lineage")
    atac_success = _read_json(paths["atac_training_success"], "ATAC training SUCCESS")
    atac_audit = _read_json(paths["atac_audit"], "ATAC independent audit")
    atac_audit_success = _read_json(paths["atac_audit_success"], "ATAC audit SUCCESS")
    _validate_atac_freeze(
        freeze=freeze,
        candidate_sha=source_hashes["candidate_authority"],
        prediction_sha=source_hashes["atac_predictions"],
        lineage_sha=source_hashes["atac_lineage"],
        training_success_sha=source_hashes["atac_training_success"],
        audit_sha=source_hashes["atac_audit"],
        audit_success_sha=source_hashes["atac_audit_success"],
        lineage=atac_lineage,
        training_success=atac_success,
        audit=atac_audit,
        audit_success=atac_audit_success,
    )
    prepared_records = _fold_file_manifest(prepared_root)

    required_columns = {
        "candidate_authority": set(TARGET_KEYS),
        "primary_oof": set(TARGET_KEYS) | {"primary_probability", "fusion_target", FOLD_COLUMN},
        "genomic_predictions": set(TARGET_KEYS)
        | {
            "mutation_context_probability", "mutation_available", "mutation_unavailable_reason",
            "cnv_context_probability", "cnv_available", "cnv_unavailable_reason",
        },
        "atac_predictions": set(TARGET_KEYS)
        | {"atac_context_probability", "atac_available", "atac_unavailable_reason"},
    }
    for label, required in required_columns.items():
        if missing := sorted(required - _parquet_columns(paths[label])):
            raise RoutingFairInputError(f"{label} parquet lacks: {missing}")

    staging.mkdir(parents=True)
    frame_path = staging / "SHARED_FAIR_INPUT.parquet"
    writer = None
    key_digest = hashlib.sha256()
    fold_digest = hashlib.sha256()
    cancer_records: list[dict[str, Any]] = []
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        for cancer in cancers:
            candidates = _canonical_keys(
                _read_cancer(paths["candidate_authority"], TARGET_KEYS, cancer),
                f"{cancer} candidate authority",
            )
            primary_columns = list(TARGET_KEYS) + ["primary_probability", "fusion_target", FOLD_COLUMN]
            genomic_columns = list(TARGET_KEYS) + [
                "mutation_context_probability", "mutation_available", "mutation_unavailable_reason",
                "cnv_context_probability", "cnv_available", "cnv_unavailable_reason",
            ]
            atac_columns = list(TARGET_KEYS) + [
                "atac_context_probability", "atac_available", "atac_unavailable_reason"
            ]
            primary = _canonical_keys(
                _read_cancer(paths["primary_oof"], primary_columns, cancer),
                f"{cancer} primary baseline",
            )
            genomic = _canonical_keys(
                _read_cancer(paths["genomic_predictions"], genomic_columns, cancer),
                f"{cancer} Mutation/CNV",
            )
            atac = _canonical_keys(
                _read_cancer(paths["atac_predictions"], atac_columns, cancer),
                f"{cancer} ATAC",
            )
            if any(len(frame) != rows_per_cancer for frame in (candidates, primary, genomic, atac)):
                raise RoutingFairInputError(
                    f"{cancer} does not preserve {rows_per_cancer} rows in every source"
                )
            keys = candidates[list(TARGET_KEYS)]
            for label, frame in (("primary", primary), ("genomic", genomic), ("ATAC", atac)):
                if not frame[list(TARGET_KEYS)].equals(keys):
                    raise RoutingFairInputError(f"{cancer} {label} exact candidate keys differ")
            primary_probability = pd.to_numeric(primary.primary_probability, errors="raise").to_numpy(float)
            target = pd.to_numeric(primary.fusion_target, errors="raise").to_numpy(float)
            if (
                not np.isfinite(primary_probability).all()
                or ((primary_probability < 0) | (primary_probability > 1)).any()
                or not np.isfinite(target).all()
                or ((target < 0) | (target > 1)).any()
            ):
                raise RoutingFairInputError(f"{cancer} baseline probability/target is invalid")
            observed_fold = pd.to_numeric(primary[FOLD_COLUMN], errors="raise").astype(int).to_numpy()
            expected_fold = np.fromiter(
                (
                    pair_blocked_fold(lnc, pathway, seed=int(budget["seeds"]["pair_fold_seed"]))
                    for lnc, pathway in zip(keys.lncrna_id, keys.pathway_id)
                ),
                dtype=np.int8,
                count=len(keys),
            )
            if not np.array_equal(observed_fold, expected_fold):
                raise RoutingFairInputError(f"{cancer} pair-fold assignment/seed drift")
            if set(observed_fold.tolist()) != set(range(5)):
                raise RoutingFairInputError(f"{cancer} does not cover pair folds 0..4")
            mutation_probability, mutation_available, mutation_reason = _typed_modality(
                genomic,
                modality="mutation",
                probability_column="mutation_context_probability",
                availability_column="mutation_available",
                reason_column="mutation_unavailable_reason",
            )
            cnv_probability, cnv_available, cnv_reason = _typed_modality(
                genomic,
                modality="cnv",
                probability_column="cnv_context_probability",
                availability_column="cnv_available",
                reason_column="cnv_unavailable_reason",
            )
            atac_probability, atac_available, atac_reason = _typed_modality(
                atac,
                modality="atac",
                probability_column="atac_context_probability",
                availability_column="atac_available",
                reason_column="atac_unavailable_reason",
            )
            shared = keys.copy()
            shared["primary_probability"] = primary_probability.astype(np.float32)
            shared["fusion_target"] = target.astype(np.float32)
            shared[FOLD_COLUMN] = observed_fold.astype(np.int8)
            for modality, probability, available, reason in (
                ("mutation", mutation_probability, mutation_available, mutation_reason),
                ("cnv", cnv_probability, cnv_available, cnv_reason),
                ("atac", atac_probability, atac_available, atac_reason),
            ):
                shared[f"{modality}_probability"] = probability.astype(np.float32)
                shared[f"{modality}_available"] = available
                shared[f"{modality}_unavailable_reason"] = reason
            for row in shared[list(TARGET_KEYS) + [FOLD_COLUMN]].itertuples(index=False):
                key = "\t".join(map(str, row[:3])).encode("utf-8")
                key_digest.update(key + b"\n")
                fold_digest.update(key + b"\t" + str(int(row[3])).encode("ascii") + b"\n")
            table = pa.Table.from_pandas(shared, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(frame_path, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=min(100_000, len(shared)))
            cancer_records.append(
                {
                    "cancer_id": cancer,
                    "rows": int(len(shared)),
                    "pair_fold_counts": {
                        str(fold): int(np.count_nonzero(observed_fold == fold)) for fold in range(5)
                    },
                    "mutation_available_rows": int(mutation_available.sum()),
                    "cnv_available_rows": int(cnv_available.sum()),
                    "atac_available_rows": int(atac_available.sum()),
                }
            )
    finally:
        if writer is not None:
            writer.close()
    if writer is None or pq.ParquetFile(frame_path).metadata.num_rows != expected_rows:
        raise RoutingFairInputError("Shared fair input row count is incomplete")

    budget_snapshot = staging / "TRAINING_BUDGET.json"
    freeze_snapshot = staging / "ATAC_R3_FREEZE.json"
    budget_snapshot.write_bytes(paths["budget_contract"].read_bytes())
    freeze_snapshot.write_bytes(paths["atac_freeze_contract"].read_bytes())
    split_path = staging / "SPLIT_POLICY.json"
    split = _split_policy(budget)
    _atomic_json(split_path, split)
    frame_sha = artifact_sha256(frame_path)
    budget_sha = artifact_sha256(budget_snapshot)
    freeze_sha = artifact_sha256(freeze_snapshot)
    split_sha = artifact_sha256(split_path)
    final_paths = {
        "frame": output / frame_path.name,
        "budget": output / budget_snapshot.name,
        "freeze": output / freeze_snapshot.name,
        "split": output / split_path.name,
        "contract": output / "COMPARISON_INPUT_CONTRACT.json",
    }
    sources = {
        key: {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": source_hashes[key],
        }
        for key, path in paths.items()
    }
    contract = {
        "format": CONTRACT_FORMAT,
        "status": "READY_FOR_ARM_TRAINING",
        "analysis_version": "CancerLncAtlas_V3.2_ROUTING_FAIR_INPUT",
        "candidate_rows": expected_rows,
        "candidate_rows_per_cancer": rows_per_cancer,
        "formal_cancers": cancers,
        "modalities": list(MODALITIES),
        "candidate_authority_sha256": source_hashes["candidate_authority"],
        "ordered_exact_candidate_key_sha256": key_digest.hexdigest(),
        "outer_pair_fold_sha256": fold_digest.hexdigest(),
        "patient_fold_manifest_sha256": source_hashes["patient_fold_manifest"],
        "patient_fold_audit": patient_audit,
        "formal_prepared_fold_records": prepared_records,
        "formal_prepared_fold_composite_sha256": _composite_records_sha(prepared_records),
        "shared_fair_input": {"path": str(final_paths["frame"]), "sha256": frame_sha},
        "training_budget": {
            "path": str(final_paths["budget"]),
            "sha256": budget_sha,
            "budget_id": budget["budget_id"],
        },
        "atac_r3_freeze": {"path": str(final_paths["freeze"]), "sha256": freeze_sha},
        "split_policy": {"path": str(final_paths["split"]), "sha256": split_sha},
        "seeds": dict(budget["seeds"]),
        "selection_scope": "INNER_VALIDATION_FOLD_ONLY",
        "outer_test_policy": "ONE_FINAL_EVALUATION_AFTER_ARM_SELECTION_IS_FROZEN",
        "outer_test_queries_during_selection": 0,
        "winner_selection_enabled": False,
        "comparison_metrics_computed": False,
        "sources": sources,
        "cancer_records": cancer_records,
    }
    contract_path = staging / "COMPARISON_INPUT_CONTRACT.json"
    _atomic_json(contract_path, contract)
    contract_sha = artifact_sha256(contract_path)
    arm_records: dict[str, dict[str, str]] = {}
    common_arm = {
        "format": ARM_FORMAT,
        "comparison_input_contract": {
            "path": str(final_paths["contract"]),
            "sha256": contract_sha,
        },
        "shared_fair_input_sha256": frame_sha,
        "candidate_authority_sha256": source_hashes["candidate_authority"],
        "ordered_exact_candidate_key_sha256": key_digest.hexdigest(),
        "outer_pair_fold_sha256": fold_digest.hexdigest(),
        "patient_fold_manifest_sha256": source_hashes["patient_fold_manifest"],
        "training_budget_sha256": budget_sha,
        "training_budget_id": budget["budget_id"],
        "split_policy_sha256": split_sha,
        "seeds": dict(budget["seeds"]),
        "selection_scope": "INNER_VALIDATION_FOLD_ONLY",
        "outer_test_policy": "ONE_FINAL_EVALUATION_AFTER_ARM_SELECTION_IS_FROZEN",
        "outer_test_queries_during_selection": 0,
        "outer_test_evaluations_per_fold": 1,
        "test_may_change_checkpoint_gate_or_hyperparameters": False,
        "test_keys_or_labels_may_enter_scaling_early_stopping_checkpoint_or_threshold": False,
        "winner_selection_enabled": False,
        "comparison_metrics_computed": False,
    }
    for architecture in ARCHITECTURES:
        arm = {
            **common_arm,
            "architecture_id": architecture,
            "implementation_sha256": None,
            "implementation_hash_required_before_arm_training": True,
        }
        arm_path = staging / f"{architecture.upper()}_INPUT_CONTRACT.json"
        _atomic_json(arm_path, arm)
        arm_records[architecture] = {
            "path": str(output / arm_path.name),
            "sha256": artifact_sha256(arm_path),
        }
    success_path = staging / "SUCCESS.json"
    success = {
        "format": SUCCESS_FORMAT,
        "status": "READY_FOR_ARM_TRAINING",
        "comparison_input_contract": {
            "path": str(final_paths["contract"]),
            "sha256": contract_sha,
        },
        "shared_fair_input": {"path": str(final_paths["frame"]), "sha256": frame_sha},
        "arm_input_contracts": arm_records,
        "candidate_rows": expected_rows,
        "same_candidate_folds_seed_and_budget_for_both_arms": True,
        "selection_scope": "INNER_VALIDATION_FOLD_ONLY",
        "outer_test_policy": "ONE_FINAL_EVALUATION_AFTER_ARM_SELECTION_IS_FROZEN",
        "winner_selection_enabled": False,
        "winner_selection_run": False,
        "comparison_metrics_computed": False,
        "atac_r3_predictions_modified": False,
        "success_written_last": True,
    }
    _atomic_json(success_path, success)
    os.replace(staging, output)
    return success


def validate_staged_routing_fair_inputs(
    *, staging_success_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Independently validate staged inputs without reading labels as metrics."""

    success_path = Path(staging_success_path).resolve()
    success = _read_json(success_path, "fair input SUCCESS")
    output = Path(output_root).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() or staging.exists():
        raise RoutingFairInputError(f"Fair input validation refuses output reuse: {output}")
    if success.get("format") != SUCCESS_FORMAT or success.get("status") != "READY_FOR_ARM_TRAINING":
        raise RoutingFairInputError("Fair input marker is not ready for arm training")
    forbidden_true = (
        "winner_selection_enabled",
        "winner_selection_run",
        "comparison_metrics_computed",
        "atac_r3_predictions_modified",
    )
    if any(success.get(key) is not False for key in forbidden_true):
        raise RoutingFairInputError("Fair input SUCCESS claims a forbidden action")
    contract_path = Path(success["comparison_input_contract"]["path"])
    contract_sha = artifact_sha256(contract_path)
    if contract_sha != success["comparison_input_contract"]["sha256"]:
        raise RoutingFairInputError("Fair input contract hash differs from SUCCESS")
    contract = _read_json(contract_path, "fair comparison input contract")
    if contract.get("format") != CONTRACT_FORMAT or contract.get("status") != "READY_FOR_ARM_TRAINING":
        raise RoutingFairInputError("Fair comparison input contract drift")
    if any(
        contract.get(key) is not False
        for key in ("winner_selection_enabled", "comparison_metrics_computed")
    ) or int(contract.get("outer_test_queries_during_selection", -1)) != 0:
        raise RoutingFairInputError("Fair comparison contract permits test-based selection")
    source_records = contract.get("sources")
    if not isinstance(source_records, Mapping):
        raise RoutingFairInputError("Fair comparison contract lacks source records")
    source_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}
    for label, raw in source_records.items():
        if not isinstance(raw, Mapping):
            raise RoutingFairInputError(f"Malformed source record: {label}")
        path = Path(str(raw.get("path", "")))
        if not path.is_file() or path.stat().st_size != int(raw.get("bytes", -1)):
            raise RoutingFairInputError(f"Fair comparison source size drift: {label}")
        digest = artifact_sha256(path)
        if digest != _require_sha(raw.get("sha256"), f"{label} source SHA256"):
            raise RoutingFairInputError(f"Fair comparison source hash drift: {label}")
        source_paths[str(label)] = path
        source_hashes[str(label)] = digest
    prepared_records = contract.get("formal_prepared_fold_records")
    if not isinstance(prepared_records, list) or len(prepared_records) != 5:
        raise RoutingFairInputError("Formal prepared baseline fold records are incomplete")
    for record in prepared_records:
        path = Path(str(record.get("path", "")))
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or artifact_sha256(path) != _require_sha(record.get("sha256"), "prepared fold SHA256")
        ):
            raise RoutingFairInputError(f"Formal prepared baseline fold drift: {path}")
    if _composite_records_sha(prepared_records) != contract.get(
        "formal_prepared_fold_composite_sha256"
    ):
        raise RoutingFairInputError("Formal prepared baseline fold composite drift")
    frame_path = Path(contract["shared_fair_input"]["path"])
    frame_sha = artifact_sha256(frame_path)
    if frame_sha != contract["shared_fair_input"]["sha256"]:
        raise RoutingFairInputError("Shared fair input hash drift")
    budget_path = Path(contract["training_budget"]["path"])
    split_path = Path(contract["split_policy"]["path"])
    if artifact_sha256(budget_path) != contract["training_budget"]["sha256"]:
        raise RoutingFairInputError("Shared training budget hash drift")
    if artifact_sha256(split_path) != contract["split_policy"]["sha256"]:
        raise RoutingFairInputError("Nested split policy hash drift")
    freeze_path = Path(contract["atac_r3_freeze"]["path"])
    if artifact_sha256(freeze_path) != contract["atac_r3_freeze"]["sha256"]:
        raise RoutingFairInputError("Frozen ATAC r3 snapshot hash drift")
    budget = _validate_budget(_read_json(budget_path, "staged training budget"))
    freeze = _read_json(freeze_path, "staged ATAC r3 freeze")
    split = _read_json(split_path, "staged split policy")
    if split != _split_policy(budget):
        raise RoutingFairInputError("Staged split policy does not derive from the budget")
    forbidden_test_consumers = (
        "scaling_fit",
        "early_stopping",
        "checkpoint_choice",
        "route_threshold_selection",
    )
    for record in split.get("fold_records", []):
        test_fold = int(record.get("outer_test_fold", -1))
        access = record.get("consumer_fold_access")
        if not isinstance(access, Mapping):
            raise RoutingFairInputError("Nested split lacks consumer-level fold access")
        leaked = [
            consumer
            for consumer in forbidden_test_consumers
            if test_fold in {int(value) for value in access.get(consumer, [])}
        ]
        if leaked:
            raise RoutingFairInputError(
                f"Outer test fold enters selection consumers: fold={test_fold}, {leaked}"
            )
        if [int(value) for value in access.get("final_test_evaluation", [])] != [test_fold]:
            raise RoutingFairInputError("Final test consumer is not isolated to one outer fold")
        if record.get("outer_test_keys_allowed_consumers") != ["final_test_evaluation"]:
            raise RoutingFairInputError("Outer test keys are exposed to a selection consumer")
        if record.get("outer_test_labels_allowed_consumers") != ["final_test_evaluation"]:
            raise RoutingFairInputError("Outer test labels are exposed to a selection consumer")
    if source_hashes.get("budget_contract") != artifact_sha256(budget_path):
        raise RoutingFairInputError("Staged budget is not byte-identical to its source")
    if source_hashes.get("atac_freeze_contract") != artifact_sha256(freeze_path):
        raise RoutingFairInputError("Staged ATAC freeze is not byte-identical to its source")
    patient_audit = _validate_patient_folds(
        source_paths["patient_fold_manifest"],
        cancers=[str(item) for item in contract["formal_cancers"]],
        expected_seed=int(budget["seeds"]["patient_fold_seed"]),
    )
    if patient_audit != contract.get("patient_fold_audit"):
        raise RoutingFairInputError("Patient fold semantic audit drift")
    genomic_lineage = _read_json(source_paths["genomic_lineage"], "Mutation/CNV lineage")
    genomic_success = _read_json(source_paths["genomic_success"], "Mutation/CNV SUCCESS")
    _validate_genomic_lineage(
        lineage=genomic_lineage,
        success=genomic_success,
        prediction_sha=source_hashes["genomic_predictions"],
        lineage_sha=source_hashes["genomic_lineage"],
        expected_rows=int(contract["candidate_rows"]),
    )
    atac_lineage = _read_json(source_paths["atac_lineage"], "ATAC lineage")
    atac_success = _read_json(source_paths["atac_training_success"], "ATAC training SUCCESS")
    atac_audit = _read_json(source_paths["atac_audit"], "ATAC independent audit")
    atac_audit_success = _read_json(source_paths["atac_audit_success"], "ATAC audit SUCCESS")
    _validate_atac_freeze(
        freeze=freeze,
        candidate_sha=source_hashes["candidate_authority"],
        prediction_sha=source_hashes["atac_predictions"],
        lineage_sha=source_hashes["atac_lineage"],
        training_success_sha=source_hashes["atac_training_success"],
        audit_sha=source_hashes["atac_audit"],
        audit_success_sha=source_hashes["atac_audit_success"],
        lineage=atac_lineage,
        training_success=atac_success,
        audit=atac_audit,
        audit_success=atac_audit_success,
    )
    common_keys = (
        "shared_fair_input_sha256",
        "candidate_authority_sha256",
        "ordered_exact_candidate_key_sha256",
        "outer_pair_fold_sha256",
        "patient_fold_manifest_sha256",
        "training_budget_sha256",
        "training_budget_id",
        "split_policy_sha256",
        "seeds",
        "selection_scope",
        "outer_test_policy",
        "outer_test_queries_during_selection",
        "outer_test_evaluations_per_fold",
        "test_may_change_checkpoint_gate_or_hyperparameters",
        "test_keys_or_labels_may_enter_scaling_early_stopping_checkpoint_or_threshold",
        "implementation_hash_required_before_arm_training",
        "winner_selection_enabled",
        "comparison_metrics_computed",
    )
    arms: dict[str, dict[str, Any]] = {}
    for architecture in ARCHITECTURES:
        record = success["arm_input_contracts"][architecture]
        path = Path(record["path"])
        if artifact_sha256(path) != record["sha256"]:
            raise RoutingFairInputError(f"{architecture} arm contract hash drift")
        arm = _read_json(path, f"{architecture} arm input contract")
        if arm.get("format") != ARM_FORMAT or arm.get("architecture_id") != architecture:
            raise RoutingFairInputError(f"{architecture} arm identity drift")
        if arm.get("comparison_input_contract", {}).get("sha256") != contract_sha:
            raise RoutingFairInputError(f"{architecture} is not bound to the shared contract")
        arms[architecture] = arm
    canonical_arm_payloads = {}
    for architecture, arm in arms.items():
        shared = {
            key: value
            for key, value in arm.items()
            if key not in {"architecture_id", "implementation_sha256"}
        }
        canonical_arm_payloads[architecture] = json.dumps(
            shared, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    if len(set(canonical_arm_payloads.values())) != 1:
        raise RoutingFairInputError(
            "Arm shared contract bytes differ outside architecture/implementation identity"
        )
    drift = {
        key: {
            architecture: arms[architecture].get(key) for architecture in ARCHITECTURES
        }
        for key in common_keys
        if len({json.dumps(arms[item].get(key), sort_keys=True) for item in ARCHITECTURES}) != 1
    }
    if drift:
        raise RoutingFairInputError(f"Arm fairness contract drift: {drift}")
    if arms["external_router"]["shared_fair_input_sha256"] != frame_sha:
        raise RoutingFairInputError("Arm contracts do not bind the staged frame")

    columns = _parquet_columns(frame_path)
    required = set(TARGET_KEYS) | {"primary_probability", "fusion_target", FOLD_COLUMN}
    required |= {
        f"{modality}_{suffix}"
        for modality in MODALITIES
        for suffix in ("probability", "available", "unavailable_reason")
    }
    if missing := sorted(required - columns):
        raise RoutingFairInputError(f"Shared fair input lacks: {missing}")
    forbidden_outputs = {
        "discovery_adjusted_probability",
        "hierarchical_probability",
        "winner",
        "selected_architecture",
    }
    if present := sorted(forbidden_outputs & columns):
        raise RoutingFairInputError(f"Input staging contains post-selection outputs: {present}")
    key_digest = hashlib.sha256()
    fold_digest = hashlib.sha256()
    cancer_records = []
    rows_per_cancer = int(contract["candidate_rows_per_cancer"])
    read_columns = list(TARGET_KEYS) + ["primary_probability", "fusion_target", FOLD_COLUMN]
    read_columns += [
        f"{modality}_{suffix}"
        for modality in MODALITIES
        for suffix in ("probability", "available", "unavailable_reason")
    ]
    for cancer in contract["formal_cancers"]:
        frame = _canonical_keys(_read_cancer(frame_path, read_columns, cancer), f"{cancer} staged frame")
        if len(frame) != rows_per_cancer:
            raise RoutingFairInputError(f"{cancer} staged row count drift")
        candidates = _canonical_keys(
            _read_cancer(source_paths["candidate_authority"], TARGET_KEYS, cancer),
            f"{cancer} re-read candidate authority",
        )
        primary = _canonical_keys(
            _read_cancer(
                source_paths["primary_oof"],
                list(TARGET_KEYS) + ["primary_probability", "fusion_target", FOLD_COLUMN],
                cancer,
            ),
            f"{cancer} re-read primary baseline",
        )
        genomic = _canonical_keys(
            _read_cancer(
                source_paths["genomic_predictions"],
                list(TARGET_KEYS)
                + [
                    "mutation_context_probability", "mutation_available", "mutation_unavailable_reason",
                    "cnv_context_probability", "cnv_available", "cnv_unavailable_reason",
                ],
                cancer,
            ),
            f"{cancer} re-read Mutation/CNV",
        )
        atac = _canonical_keys(
            _read_cancer(
                source_paths["atac_predictions"],
                list(TARGET_KEYS)
                + ["atac_context_probability", "atac_available", "atac_unavailable_reason"],
                cancer,
            ),
            f"{cancer} re-read ATAC",
        )
        for label, source in (
            ("candidate", candidates), ("primary", primary),
            ("genomic", genomic), ("ATAC", atac),
        ):
            if len(source) != rows_per_cancer or not source[list(TARGET_KEYS)].equals(
                frame[list(TARGET_KEYS)]
            ):
                raise RoutingFairInputError(f"{cancer} {label} differs from staged exact keys")
        for column in ("primary_probability", "fusion_target"):
            staged_value = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
            source_value = pd.to_numeric(primary[column], errors="raise").to_numpy(float)
            if not np.allclose(staged_value, source_value, rtol=0.0, atol=1e-7):
                raise RoutingFairInputError(f"{cancer} staged {column} differs from baseline")
        folds = pd.to_numeric(frame[FOLD_COLUMN], errors="raise").astype(int).to_numpy()
        source_folds = pd.to_numeric(primary[FOLD_COLUMN], errors="raise").astype(int).to_numpy()
        if not np.array_equal(folds, source_folds):
            raise RoutingFairInputError(f"{cancer} staged pair folds differ from baseline")
        expected = np.fromiter(
            (
                pair_blocked_fold(lnc, pathway, seed=int(budget["seeds"]["pair_fold_seed"]))
                for lnc, pathway in zip(frame.lncrna_id, frame.pathway_id)
            ),
            dtype=np.int8,
            count=len(frame),
        )
        if not np.array_equal(folds, expected) or set(folds.tolist()) != set(range(5)):
            raise RoutingFairInputError(f"{cancer} staged pair folds drift")
        availability_counts = {}
        for modality in MODALITIES:
            staged_probability, available, staged_reason = _typed_modality(
                frame,
                modality=modality,
                probability_column=f"{modality}_probability",
                availability_column=f"{modality}_available",
                reason_column=f"{modality}_unavailable_reason",
            )
            source = atac if modality == "atac" else genomic
            source_probability, source_available, source_reason = _typed_modality(
                source,
                modality=modality,
                probability_column=(
                    "atac_context_probability" if modality == "atac"
                    else f"{modality}_context_probability"
                ),
                availability_column=f"{modality}_available",
                reason_column=f"{modality}_unavailable_reason",
            )
            if not np.array_equal(available, source_available):
                raise RoutingFairInputError(f"{cancer} staged {modality} availability drift")
            if not np.allclose(
                staged_probability[available],
                source_probability[source_available],
                rtol=0.0,
                atol=1e-7,
            ):
                raise RoutingFairInputError(f"{cancer} staged {modality} probability drift")
            if not staged_reason.reset_index(drop=True).equals(
                source_reason.reset_index(drop=True)
            ):
                raise RoutingFairInputError(f"{cancer} staged {modality} reason drift")
            availability_counts[modality] = int(available.sum())
        for row in frame[list(TARGET_KEYS) + [FOLD_COLUMN]].itertuples(index=False):
            key = "\t".join(map(str, row[:3])).encode("utf-8")
            key_digest.update(key + b"\n")
            fold_digest.update(key + b"\t" + str(int(row[3])).encode("ascii") + b"\n")
        cancer_records.append(
            {"cancer_id": cancer, "rows": len(frame), "available_rows": availability_counts}
        )
    if key_digest.hexdigest() != contract["ordered_exact_candidate_key_sha256"]:
        raise RoutingFairInputError("Staged exact candidate key digest drift")
    if fold_digest.hexdigest() != contract["outer_pair_fold_sha256"]:
        raise RoutingFairInputError("Staged outer pair fold digest drift")

    staging.mkdir(parents=True)
    report_path = staging / "VALIDATION.json"
    report = {
        "format": VALIDATION_FORMAT,
        "status": "PASS_INPUTS_ONLY_WINNER_NOT_RUN",
        "staging_success_path": str(success_path),
        "staging_success_sha256": artifact_sha256(success_path),
        "comparison_input_contract_path": str(contract_path),
        "comparison_input_contract_sha256": contract_sha,
        "shared_fair_input_path": str(frame_path),
        "shared_fair_input_sha256": frame_sha,
        "candidate_rows": int(sum(item["rows"] for item in cancer_records)),
        "cancer_records": cancer_records,
        "same_candidate_folds_seed_and_budget_for_both_arms": True,
        "selection_scope": "INNER_VALIDATION_FOLD_ONLY",
        "outer_test_policy": "ONE_FINAL_EVALUATION_AFTER_ARM_SELECTION_IS_FROZEN",
        "outer_test_queries_during_selection": 0,
        "winner_selection_enabled": False,
        "winner_selection_run": False,
        "comparison_metrics_computed": False,
        "arm_training_run": False,
        "atac_r3_predictions_modified": False,
    }
    _atomic_json(report_path, report)
    marker = {
        "format": VALIDATION_FORMAT,
        "status": report["status"],
        "validation_path": str(output / report_path.name),
        "validation_sha256": artifact_sha256(report_path),
        "winner_selection_run": False,
        "comparison_metrics_computed": False,
        "success_written_last": True,
    }
    _atomic_json(staging / "SUCCESS.json", marker)
    os.replace(staging, output)
    return marker


__all__ = [
    "ARCHITECTURES",
    "RoutingFairInputError",
    "artifact_sha256",
    "canonical_json_sha256",
    "pair_blocked_fold",
    "stage_routing_fair_inputs",
    "validate_staged_routing_fair_inputs",
]
