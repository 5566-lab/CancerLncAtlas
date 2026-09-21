"""Fail-closed semantic wrapper for a completed V3.2 Evidence binding.

The formal R2 Evidence binder proves file identity, fresh optimisation, and
pair-blocked provenance.  This post-audit wrapper adds the downstream routing
contract that must remain explicit when the Evidence result is consumed:

* direct lncRNA--partner evidence is confidence-only and reaches an exact
  pathway only through the pinned static-member mapping;
* unavailable values are null with a non-empty reason; and
* neither discovery nor the primary ranking may be changed.

The wrapper does not replace or modify the R2 binding.  It hash-binds that
immutable file to an independent audit document and writes a new local
artifact in a previously absent output directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

from .evidence_output_binding import ANALYSIS_VERSION

R2_BINDING_FORMAT = "CC_HHGT_V3_2_EVIDENCE_OUTPUT_BINDING_V1"
POST_AUDIT_FORMAT = "CC_HHGT_V3_2_EVIDENCE_INDEPENDENT_POST_AUDIT_V1"
SEMANTIC_WRAPPER_FORMAT = "CC_HHGT_V3_2_EVIDENCE_SEMANTIC_WRAPPER_V1"
SEMANTIC_WRAPPER_STATUS = "PASS_HASH_BOUND_EVIDENCE_SEMANTICS"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FOLD_IDS = {str(index) for index in range(5)}


class EvidenceSemanticWrapperError(RuntimeError):
    """Raised when Evidence artifacts do not prove the required semantics."""


def artifact_sha256(path: str | Path) -> str:
    """Return the SHA256 of a regular, non-symlink file."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise EvidenceSemanticWrapperError(f"Unsafe or missing file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path, role: str) -> tuple[Path, dict[str, Any]]:
    requested = Path(path)
    if requested.is_symlink():
        raise EvidenceSemanticWrapperError(f"{role} is missing or unsafe: {requested}")
    source = requested.resolve()
    if not source.is_file():
        raise EvidenceSemanticWrapperError(f"{role} is missing or unsafe: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceSemanticWrapperError(f"{role} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise EvidenceSemanticWrapperError(f"{role} must be a JSON object")
    return source, payload


def _require_equal(payload: Mapping[str, Any], expected: Mapping[str, Any], role: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise EvidenceSemanticWrapperError(
                f"{role} has invalid {key}: observed={payload.get(key)!r}, expected={value!r}"
            )


def _validate_r2_binding(path: Path, binding: Mapping[str, Any]) -> str:
    _require_equal(
        binding,
        {
            "format": R2_BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND",
            "release_ready": False,
            "production_deployed": False,
            "evidence_module_still_auxiliary_partial": True,
            "interaction_materialization_input_eligible": True,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "family_to_exact_broadcast": False,
            "five_fresh_private_heads_verified": True,
        },
        "R2 Evidence binding",
    )
    if int(binding.get("optimizer_steps_total", 0)) <= 0:
        raise EvidenceSemanticWrapperError("R2 Evidence binding has no optimiser updates")
    run_id = str(binding.get("evidence_training_run_id", ""))
    if not run_id.startswith("V32-EVIDENCE-TRAIN-"):
        raise EvidenceSemanticWrapperError("R2 Evidence binding lacks a fresh training_run_id")
    checkpoints = binding.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != _FOLD_IDS:
        raise EvidenceSemanticWrapperError("R2 Evidence binding does not declare five folds")

    success_path = path.parent / "SUCCESS.json"
    _, success = _read_json(success_path, "R2 Evidence binding success marker")
    binding_sha = artifact_sha256(path)
    _require_equal(
        success,
        {
            "status": binding["status"],
            "release_ready": False,
            "binding": path.name,
            "binding_sha256": binding_sha,
        },
        "R2 Evidence binding success marker",
    )
    return run_id


def _validate_semantic_policy(audit: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = audit.get("semantic_policy")
    if not isinstance(policy, Mapping):
        raise EvidenceSemanticWrapperError("Independent audit lacks semantic_policy")
    _require_equal(
        policy,
        {
            "direct_target_evidence": True,
            "direct_target_scope": "LNCRNA_PARTNER_MAPPED_VIA_EXACT_MEMBER",
            "direct_exact_pathway_assertion": False,
            "unavailable_encoding": "null_with_reason",
            "confidence_only": True,
            "changes_primary_ranking": False,
            "affects_discovery": False,
            "affects_primary_ranking": False,
            "raw_prediction_direct_fusion_allowed": False,
            "canonical_candidate_adapter_required": True,
            "lncrna_identifier_normalization": "LNC:ENSG_TO_ENSG_FOR_TRAINING__RESTORE_LNC_PREFIX_FOR_FUSION",
            "pathway_identifier_normalization": "TRIM_AND_UPPERCASE_TO_CANONICAL_PATHWAY_FOR_FUSION",
        },
        "Evidence semantic policy",
    )
    return policy


def _validate_prediction_audit(
    audit: Mapping[str, Any], binding: Mapping[str, Any], training_run_id: str
) -> Mapping[str, Any]:
    prediction = audit.get("prediction_semantics")
    if not isinstance(prediction, Mapping):
        raise EvidenceSemanticWrapperError("Independent audit lacks prediction_semantics")
    zero_fields = (
        "unavailable_probability_nonnull_rows",
        "unavailable_reason_missing_rows",
        "unavailable_failure_reason_missing_rows",
        "available_probability_null_rows",
        "probability_out_of_range_rows",
        "changes_primary_ranking_true_rows",
        "main_ranking_modified_true_rows",
        "analysis_version_mismatch_rows",
    )
    for field in zero_fields:
        if int(prediction.get(field, -1)) != 0:
            raise EvidenceSemanticWrapperError(
                f"Prediction semantic audit failed {field}: {prediction.get(field)!r}"
            )
    if int(prediction.get("distinct_training_run_ids", -1)) != 1:
        raise EvidenceSemanticWrapperError("Predictions do not have exactly one training_run_id")
    if prediction.get("training_run_id") != training_run_id:
        raise EvidenceSemanticWrapperError("Prediction training_run_id differs from binding")
    artifacts = binding.get("artifacts")
    counts = binding.get("counts")
    if not isinstance(artifacts, Mapping) or not isinstance(counts, Mapping):
        raise EvidenceSemanticWrapperError("R2 Evidence binding lacks artifacts/counts")
    prediction_artifact = artifacts.get("evidence_predictions")
    if not isinstance(prediction_artifact, Mapping):
        raise EvidenceSemanticWrapperError("R2 binding lacks evidence_predictions")
    expected_rows = int(counts.get("evidence_predictions", -1))
    if (
        expected_rows <= 0
        or int(prediction_artifact.get("rows", -1)) != expected_rows
        or int(prediction.get("rows", -1)) != expected_rows
    ):
        raise EvidenceSemanticWrapperError("Prediction row counts differ across binding/audit")
    required_columns = {
        "cancer_id",
        "lncrna_id",
        "pathway_id",
        "evidence_confidence_probability",
        "availability",
        "unavailable_reason",
        "failure_reason",
        "analysis_version",
        "training_run_id",
        "changes_primary_ranking",
        "main_ranking_modified",
    }
    columns = prediction_artifact.get("columns")
    if not isinstance(columns, list) or not required_columns.issubset(set(columns)):
        raise EvidenceSemanticWrapperError(
            "R2 binding evidence_predictions schema lacks required semantic columns"
        )
    return prediction


def _validate_fold_audit(
    audit: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    audited_folds = audit.get("folds")
    bound_folds = binding.get("checkpoints")
    if not isinstance(audited_folds, Mapping) or set(audited_folds) != _FOLD_IDS:
        raise EvidenceSemanticWrapperError("Independent audit does not contain five folds")
    if not isinstance(bound_folds, Mapping) or set(bound_folds) != _FOLD_IDS:
        raise EvidenceSemanticWrapperError("R2 binding does not contain five checkpoints")
    result: dict[str, dict[str, Any]] = {}
    required_flags = {
        "private_parameters_fresh_init": True,
        "initialized_from_checkpoint": False,
        "historical_evidence_checkpoint_allowed": False,
        "historical_evidence_result_allowed": False,
        "pair_evidence_supervision_allowed": False,
        "family_spf_allowed": False,
        "core_frozen": True,
        "core_detached": True,
        "contains_core_parameters": False,
        "contains_primary_ranking_parameters": False,
        "confidence_supervision_available": True,
        "direction_supervision_available": True,
        "provenance_residuals_zero": True,
    }
    for fold_id in sorted(_FOLD_IDS):
        observed = audited_folds[fold_id]
        bound = bound_folds[fold_id]
        if not isinstance(observed, Mapping) or not isinstance(bound, Mapping):
            raise EvidenceSemanticWrapperError(f"Evidence fold {fold_id} record is invalid")
        _require_equal(observed, required_flags, f"Evidence fold {fold_id} audit")
        initial = str(observed.get("initial_parameter_sha256", ""))
        final = str(observed.get("final_parameter_sha256", ""))
        recomputed = str(observed.get("recomputed_final_parameter_sha256", ""))
        checkpoint_sha = str(observed.get("checkpoint_sha256", ""))
        history_sha = str(observed.get("training_history_sha256", ""))
        core_checkpoint_sha = str(observed.get("core_checkpoint_sha256", ""))
        core_parameter_sha = str(observed.get("core_parameter_sha256", ""))
        steps = int(observed.get("optimizer_steps", 0))
        provenance = observed.get("provenance_residuals")
        expected_provenance = {
            "residual_pmid_overlap_count": 0,
            "residual_source_event_overlap_count": 0,
            "residual_source_record_overlap_count": 0,
            "residual_train_evaluation_provenance_overlap_count": 0,
        }
        if (
            int(observed.get("patient_fold", -1)) != int(fold_id)
            or not all(
                _SHA256.fullmatch(value)
                for value in (
                    initial,
                    final,
                    recomputed,
                    checkpoint_sha,
                    history_sha,
                    core_checkpoint_sha,
                    core_parameter_sha,
                )
            )
            or initial == final
            or recomputed != final
            or steps <= 0
            or provenance != expected_provenance
        ):
            raise EvidenceSemanticWrapperError(f"Evidence fold {fold_id} failed strong audit")
        if (
            bound.get("sha256") != checkpoint_sha
            or bound.get("training_history_sha256") != history_sha
            or bound.get("core_checkpoint_sha256") != core_checkpoint_sha
            or bound.get("core_parameter_sha256") != core_parameter_sha
            or bound.get("initial_parameter_sha256") != initial
            or bound.get("final_parameter_sha256") != final
            or int(bound.get("optimizer_steps", -1)) != steps
        ):
            raise EvidenceSemanticWrapperError(
                f"Evidence fold {fold_id} audit differs from R2 binding"
            )
        result[fold_id] = {
            "checkpoint_sha256": checkpoint_sha,
            "initial_parameter_sha256": initial,
            "final_parameter_sha256": final,
            "independently_recomputed_final_parameter_sha256": recomputed,
            "optimizer_steps": steps,
            "training_history_sha256": history_sha,
            "core_checkpoint_sha256": core_checkpoint_sha,
            "core_parameter_sha256": core_parameter_sha,
            "fresh_training_verified": True,
            "provenance_residuals": expected_provenance,
        }
    return result


def materialize_evidence_semantic_wrapper(
    *,
    r2_binding_path: str | Path,
    independent_audit_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Validate and hash-bind R2 Evidence outputs to explicit routing semantics."""

    binding_path, binding = _read_json(r2_binding_path, "R2 Evidence binding")
    audit_path, audit = _read_json(independent_audit_path, "independent Evidence audit")
    training_run_id = _validate_r2_binding(binding_path, binding)
    _require_equal(
        audit,
        {
            "format": POST_AUDIT_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS",
            "r2_binding_sha256": artifact_sha256(binding_path),
        },
        "independent Evidence audit",
    )
    policy = _validate_semantic_policy(audit)
    prediction = _validate_prediction_audit(audit, binding, training_run_id)
    folds = _validate_fold_audit(audit, binding)

    output = Path(output_root).resolve()
    if output.exists():
        raise EvidenceSemanticWrapperError(
            f"Evidence semantic wrapper output reuse is forbidden: {output}"
        )
    output.mkdir(parents=True)
    try:
        wrapper: dict[str, Any] = {
            "format": SEMANTIC_WRAPPER_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": SEMANTIC_WRAPPER_STATUS,
            "release_ready": False,
            "production_deployed": False,
            "evidence_training_run_id": training_run_id,
            "direct_target_evidence": policy["direct_target_evidence"],
            "direct_target_scope": policy["direct_target_scope"],
            "direct_exact_pathway_assertion": policy[
                "direct_exact_pathway_assertion"
            ],
            "unavailable_encoding": policy["unavailable_encoding"],
            "confidence_only": policy["confidence_only"],
            "changes_primary_ranking": policy["changes_primary_ranking"],
            "affects_discovery": policy["affects_discovery"],
            "affects_primary_ranking": policy["affects_primary_ranking"],
            "raw_prediction_direct_fusion_allowed": policy[
                "raw_prediction_direct_fusion_allowed"
            ],
            "canonical_candidate_adapter_required": policy[
                "canonical_candidate_adapter_required"
            ],
            "lncrna_identifier_normalization": policy[
                "lncrna_identifier_normalization"
            ],
            "pathway_identifier_normalization": policy[
                "pathway_identifier_normalization"
            ],
            "prediction_role": "AUXILIARY_CONFIDENCE_ONLY",
            "r2_evidence_binding": {
                "path": str(binding_path),
                "sha256": artifact_sha256(binding_path),
            },
            "independent_post_audit": {
                "path": str(audit_path),
                "sha256": artifact_sha256(audit_path),
            },
            "five_fresh_private_heads_verified": True,
            "optimizer_steps_total": sum(
                int(record["optimizer_steps"]) for record in folds.values()
            ),
            "folds": folds,
            "prediction_semantic_audit": dict(prediction),
        }
        if wrapper["optimizer_steps_total"] != int(binding["optimizer_steps_total"]):
            raise EvidenceSemanticWrapperError(
                "Aggregate optimizer steps differ from R2 Evidence binding"
            )
        wrapper_path = output / "EVIDENCE_SEMANTIC_WRAPPER.json"
        temporary = output / ".EVIDENCE_SEMANTIC_WRAPPER.json.tmp"
        temporary.write_text(
            json.dumps(wrapper, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, wrapper_path)
        success = {
            "status": SEMANTIC_WRAPPER_STATUS,
            "release_ready": False,
            "wrapper": wrapper_path.name,
            "wrapper_sha256": artifact_sha256(wrapper_path),
        }
        success_path = output / "SUCCESS.json"
        success_path.write_text(
            json.dumps(success, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return wrapper
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


__all__ = [
    "ANALYSIS_VERSION",
    "EvidenceSemanticWrapperError",
    "POST_AUDIT_FORMAT",
    "R2_BINDING_FORMAT",
    "SEMANTIC_WRAPPER_FORMAT",
    "SEMANTIC_WRAPPER_STATUS",
    "artifact_sha256",
    "materialize_evidence_semantic_wrapper",
]
