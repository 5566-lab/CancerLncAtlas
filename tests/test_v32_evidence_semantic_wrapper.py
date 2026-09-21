from __future__ import annotations

import json
from pathlib import Path

import pytest

from cc_hhgt.v32.evidence_semantic_wrapper import (
    ANALYSIS_VERSION,
    EvidenceSemanticWrapperError,
    POST_AUDIT_FORMAT,
    R2_BINDING_FORMAT,
    SEMANTIC_WRAPPER_FORMAT,
    artifact_sha256,
    materialize_evidence_semantic_wrapper,
)
from cc_hhgt.v32.evidence_output_binding import (
    ANALYSIS_VERSION as FORMAL_BINDING_ANALYSIS_VERSION,
)


def _hex(character: str) -> str:
    return character * 64


def test_wrapper_uses_exact_formal_binding_analysis_version() -> None:
    assert ANALYSIS_VERSION == FORMAL_BINDING_ANALYSIS_VERSION
    assert ANALYSIS_VERSION == "CancerLncAtlas_V3.2_FULL_MULTITASK"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    binding_root = tmp_path / "binding"
    binding_root.mkdir()
    checkpoints = {}
    audit_folds = {}
    optimizer_total = 0
    for fold in range(5):
        initial = _hex(str(fold + 1))
        final = _hex(chr(ord("a") + fold))
        checkpoint = _hex(chr(ord("f") - fold))
        steps = 100 + fold
        optimizer_total += steps
        checkpoints[str(fold)] = {
            "path": f"/formal/patient_fold={fold}/private_eventset_state.pt",
            "sha256": checkpoint,
            "optimizer_steps": steps,
            "initial_parameter_sha256": initial,
            "final_parameter_sha256": final,
            "training_history_sha256": _hex("8"),
            "core_checkpoint_sha256": _hex("7"),
            "core_parameter_sha256": _hex("6"),
        }
        audit_folds[str(fold)] = {
            "patient_fold": fold,
            "checkpoint_sha256": checkpoint,
            "initial_parameter_sha256": initial,
            "final_parameter_sha256": final,
            "recomputed_final_parameter_sha256": final,
            "optimizer_steps": steps,
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
            "training_history_sha256": _hex("8"),
            "core_checkpoint_sha256": _hex("7"),
            "core_parameter_sha256": _hex("6"),
            "provenance_residuals": {
                "residual_pmid_overlap_count": 0,
                "residual_source_event_overlap_count": 0,
                "residual_source_record_overlap_count": 0,
                "residual_train_evaluation_provenance_overlap_count": 0,
            },
        }
    run_id = "V32-EVIDENCE-TRAIN-test"
    binding = {
        "format": R2_BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND",
        "release_ready": False,
        "production_deployed": False,
        "evidence_training_run_id": run_id,
        "evidence_module_still_auxiliary_partial": True,
        "interaction_materialization_input_eligible": True,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "family_to_exact_broadcast": False,
        "five_fresh_private_heads_verified": True,
        "optimizer_steps_total": optimizer_total,
        "checkpoints": checkpoints,
        "counts": {"evidence_predictions": 330},
        "artifacts": {
            "evidence_predictions": {
                "rows": 330,
                "sha256": _hex("9"),
                "columns": [
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
                ],
            }
        },
    }
    binding_path = binding_root / "EVIDENCE_OUTPUT_BINDING.json"
    _write_json(binding_path, binding)
    _write_json(
        binding_root / "SUCCESS.json",
        {
            "status": binding["status"],
            "release_ready": False,
            "binding": binding_path.name,
            "binding_sha256": artifact_sha256(binding_path),
        },
    )
    audit = {
        "format": POST_AUDIT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS",
        "r2_binding_sha256": artifact_sha256(binding_path),
        "semantic_policy": {
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
        "prediction_semantics": {
            "rows": 330,
            "distinct_training_run_ids": 1,
            "training_run_id": run_id,
            "unavailable_probability_nonnull_rows": 0,
            "unavailable_reason_missing_rows": 0,
            "unavailable_failure_reason_missing_rows": 0,
            "available_probability_null_rows": 0,
            "probability_out_of_range_rows": 0,
            "changes_primary_ranking_true_rows": 0,
            "main_ranking_modified_true_rows": 0,
            "analysis_version_mismatch_rows": 0,
        },
        "folds": audit_folds,
    }
    audit_path = tmp_path / "EVIDENCE_INDEPENDENT_POST_AUDIT.json"
    _write_json(audit_path, audit)
    return binding_path, audit_path, binding, audit


def test_materializes_explicit_fail_closed_semantics(tmp_path: Path) -> None:
    binding_path, audit_path, binding, _ = _fixture(tmp_path)
    output = tmp_path / "wrapper"
    wrapper = materialize_evidence_semantic_wrapper(
        r2_binding_path=binding_path,
        independent_audit_path=audit_path,
        output_root=output,
    )
    assert wrapper["format"] == SEMANTIC_WRAPPER_FORMAT
    assert wrapper["direct_target_evidence"] is True
    assert wrapper["direct_target_scope"] == "LNCRNA_PARTNER_MAPPED_VIA_EXACT_MEMBER"
    assert wrapper["direct_exact_pathway_assertion"] is False
    assert wrapper["unavailable_encoding"] == "null_with_reason"
    assert wrapper["confidence_only"] is True
    assert wrapper["changes_primary_ranking"] is False
    assert wrapper["affects_discovery"] is False
    assert wrapper["affects_primary_ranking"] is False
    assert wrapper["raw_prediction_direct_fusion_allowed"] is False
    assert wrapper["canonical_candidate_adapter_required"] is True
    assert wrapper["lncrna_identifier_normalization"].startswith("LNC:ENSG_TO_ENSG")
    assert wrapper["pathway_identifier_normalization"].startswith("TRIM_AND_UPPERCASE")
    assert wrapper["optimizer_steps_total"] == binding["optimizer_steps_total"]
    assert set(wrapper["folds"]) == {"0", "1", "2", "3", "4"}
    success = json.loads((output / "SUCCESS.json").read_text(encoding="utf-8"))
    assert success["wrapper_sha256"] == artifact_sha256(
        output / "EVIDENCE_SEMANTIC_WRAPPER.json"
    )


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("direct_target_evidence", False),
        ("direct_target_scope", "DIRECT_EXACT_PATHWAY"),
        ("direct_exact_pathway_assertion", True),
        ("unavailable_encoding", "zero_fill"),
        ("confidence_only", False),
        ("raw_prediction_direct_fusion_allowed", True),
        ("canonical_candidate_adapter_required", False),
        ("changes_primary_ranking", True),
    ],
)
def test_rejects_semantic_policy_drift(tmp_path: Path, key: str, bad_value: object) -> None:
    binding_path, audit_path, _, audit = _fixture(tmp_path)
    audit["semantic_policy"][key] = bad_value
    _write_json(audit_path, audit)
    with pytest.raises(EvidenceSemanticWrapperError, match="semantic policy"):
        materialize_evidence_semantic_wrapper(
            r2_binding_path=binding_path,
            independent_audit_path=audit_path,
            output_root=tmp_path / "wrapper",
        )


def test_rejects_non_null_missing_prediction(tmp_path: Path) -> None:
    binding_path, audit_path, _, audit = _fixture(tmp_path)
    audit["prediction_semantics"]["unavailable_probability_nonnull_rows"] = 1
    _write_json(audit_path, audit)
    with pytest.raises(EvidenceSemanticWrapperError, match="Prediction semantic audit failed"):
        materialize_evidence_semantic_wrapper(
            r2_binding_path=binding_path,
            independent_audit_path=audit_path,
            output_root=tmp_path / "wrapper",
        )


def test_rejects_fold_state_hash_mismatch(tmp_path: Path) -> None:
    binding_path, audit_path, _, audit = _fixture(tmp_path)
    audit["folds"]["3"]["recomputed_final_parameter_sha256"] = _hex("0")
    _write_json(audit_path, audit)
    with pytest.raises(EvidenceSemanticWrapperError, match="fold 3 failed strong audit"):
        materialize_evidence_semantic_wrapper(
            r2_binding_path=binding_path,
            independent_audit_path=audit_path,
            output_root=tmp_path / "wrapper",
        )


def test_refuses_output_reuse(tmp_path: Path) -> None:
    binding_path, audit_path, _, _ = _fixture(tmp_path)
    output = tmp_path / "wrapper"
    output.mkdir()
    with pytest.raises(EvidenceSemanticWrapperError, match="output reuse is forbidden"):
        materialize_evidence_semantic_wrapper(
            r2_binding_path=binding_path,
            independent_audit_path=audit_path,
            output_root=output,
        )
