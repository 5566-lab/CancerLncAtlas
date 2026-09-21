from __future__ import annotations

import json
from pathlib import Path

from cc_hhgt.v32.single_cell_fusion_post_audit import (
    AUDIT_FORMAT,
    BINDING_FORMAT,
    file_sha256,
    verify_post_binding,
)


ROOT = Path(__file__).resolve().parents[1]
POST_ROOT = (
    ROOT / "artifacts"
    / "v32_single_cell_fusion_adapter_20260826_r1_independent_post_audit"
)


def test_formal_post_materialization_audit_is_pass_hash_bound_and_exact() -> None:
    report_path = POST_ROOT / "POST_MATERIALIZATION_AUDIT.json"
    binding_path = POST_ROOT / "POST_MATERIALIZATION_AUDIT_BINDING.json"
    success_path = POST_ROOT / "SUCCESS.json"
    assert report_path.is_file() and binding_path.is_file() and success_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    success = json.loads(success_path.read_text(encoding="utf-8"))
    assert report["format"] == AUDIT_FORMAT
    assert binding["format"] == BINDING_FORMAT
    assert report["status"] == binding["status"] == success["status"] == "PASS"
    assert report["failed_checks"] == []
    assert all(row["status"] == "PASS" for row in report["checks"])
    assert verify_post_binding(report_path, binding)
    assert success["post_materialization_audit_sha256"] == file_sha256(report_path)
    assert success["post_materialization_binding_sha256"] == file_sha256(binding_path)
    prediction = report["prediction"]
    assert prediction["rows"] == prediction["unique_keys"] == 3_300_000
    assert prediction["available_rows"] == 954_541
    assert prediction["unavailable_rows"] == 2_345_459
    assert prediction["prediction_minus_candidate"] == 0
    assert prediction["candidate_minus_prediction"] == 0
    assert prediction["rowwise_reconstruction_difference"] == 0
    assert prediction["filled_unavailable_rows"] == 0
    assert prediction["missing_unavailable_reason_rows"] == 0
    assert prediction["direct_target_evidence_rows"] == 0
    assert prediction["family_to_exact_broadcast_rows"] == 0
    assert prediction["changes_primary_ranking_rows"] == 0
    assert (
        report["dependencies"]["original_audit_binding"]["sha256"]
        == "5309b2eaf8e63d0daba241d6a9bca2870a382f711896916b4eb364cbfbc84829"
    )
    assert (
        report["materialization"]["directory_sha256_before"]
        == report["materialization"]["directory_sha256_after"]
    )
    assert report["release_decision"]["eligible_for_multimodal_fusion_input"] is True
    assert report["release_decision"]["primary_ranking_may_be_changed"] is False


def test_post_binding_verifier_fails_closed_if_report_hash_is_changed() -> None:
    report_path = POST_ROOT / "POST_MATERIALIZATION_AUDIT.json"
    binding = json.loads(
        (POST_ROOT / "POST_MATERIALIZATION_AUDIT_BINDING.json").read_text(encoding="utf-8")
    )
    binding["post_materialization_audit_sha256"] = "f" * 64
    assert not verify_post_binding(report_path, binding)
