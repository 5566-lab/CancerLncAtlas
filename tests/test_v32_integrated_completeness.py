from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cc_hhgt.v32.integrated_completeness import (
    IntegratedCompletenessError,
    evaluate_integrated_completeness,
    materialize_integrated_completeness_audit,
)
from cc_hhgt.v32.integrated_completeness_independent import (
    independent_audit_integrated_completeness,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/v32_historical_capability_parity.yaml"
UNIFIED = ROOT / "config/v32_unified_staging_bindings.json"
WEB = ROOT / "website/frontend/v32-capability-catalog.json"
EVALUATOR = ROOT / "cc_hhgt/v32/integrated_completeness.py"
RUNNER = ROOT / "scripts/audit_v32_integrated_completeness.py"
INDEPENDENT = ROOT / "cc_hhgt/v32/integrated_completeness_independent.py"
INDEPENDENT_RUNNER = ROOT / "scripts/audit_v32_integrated_completeness_independent.py"


def evaluate(*, unified: Path = UNIFIED, web: Path = WEB):
    return evaluate_integrated_completeness(
        repo_root=ROOT,
        parity_contract_path=CONTRACT,
        unified_bindings_path=unified,
        web_catalog_path=web,
        evaluator_code_path=EVALUATOR,
        runner_code_path=RUNNER,
    )


def test_current_release_is_truthfully_partial_and_preserves_historical_heads() -> None:
    report, inputs = evaluate()
    rows = {row["capability_id"]: row for row in report["capabilities"]}

    assert report["status"] == "PARTIAL"
    assert report["all_25_capabilities_four_gate_complete"] is False
    assert report["required_capability_count"] == 25
    assert report["complete_capability_count"] == 23
    assert report["partial_capability_count"] == 2
    assert report["blocking_capability_ids"] == ["drug", "single_cell"]
    assert rows["mutation"]["status"] == "COMPLETE"
    assert rows["cnv"]["status"] == "COMPLETE"
    assert rows["clinical"]["status"] == "COMPLETE"
    assert rows["state_rnass"]["status"] == "COMPLETE"
    assert rows["state_dnass"]["status"] == "COMPLETE"
    assert rows["state_extend"]["status"] == "COMPLETE"
    assert rows["state_ereg_expss"]["status"] == "COMPLETE"
    assert set(rows["single_cell"]["gates"]) == {"artifact", "api", "ui", "download"}
    assert all(
        gate["status"] == "FAIL" for gate in rows["single_cell"]["gates"].values()
    )
    assert all(gate["status"] == "FAIL" for gate in rows["drug"]["gates"].values())
    assert "unified_binding:historical_artifact_remediation" in inputs
    assert "unified_binding:download_catalog_independent_audit" in inputs


def test_materialized_binding_is_independently_accepted_only_as_partial_truth() -> None:
    with tempfile.TemporaryDirectory(
        prefix=".pytest_integrated_completeness_", dir=ROOT
    ) as temporary:
        base = Path(temporary)
        release = materialize_integrated_completeness_audit(
            repo_root=ROOT,
            parity_contract_path=CONTRACT,
            unified_bindings_path=UNIFIED,
            web_catalog_path=WEB,
            output_root=base / "release",
            evaluator_code_path=EVALUATOR,
            runner_code_path=RUNNER,
        )
        audit = independent_audit_integrated_completeness(
            repo_root=ROOT,
            binding_path=release["binding_path"],
            expected_binding_sha256=release["binding_sha256"],
            output_root=base / "independent",
            auditor_code_path=INDEPENDENT,
            runner_code_path=INDEPENDENT_RUNNER,
        )

        assert release["status"] == "PARTIAL"
        assert audit["status"] == "PASS"
        assert audit["audited_completeness_status"] == "PARTIAL"
        assert audit["accepted_as_truthful_staging_audit"] is True
        assert audit["accepted_as_complete"] is False
        assert audit["fail_count"] == 0


def test_unified_binding_hash_drift_is_rejected() -> None:
    with tempfile.TemporaryDirectory(
        prefix=".pytest_integrated_completeness_drift_", dir=ROOT
    ) as temporary:
        path = Path(temporary) / "unified.json"
        payload = json.loads(UNIFIED.read_text(encoding="utf-8"))
        payload["bindings"]["physical_interaction"]["sha256"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(IntegratedCompletenessError, match="SHA256 drift"):
            evaluate(unified=path)


def test_catalog_cannot_claim_complete_for_a_typed_gap() -> None:
    with tempfile.TemporaryDirectory(
        prefix=".pytest_integrated_completeness_gap_", dir=ROOT
    ) as temporary:
        path = Path(temporary) / "web.json"
        payload = json.loads(WEB.read_text(encoding="utf-8"))
        row = next(
            value
            for value in payload["capabilities"]
            if value["capability_id"] == "single_cell"
        )
        row["all_four_parity_gates_pass"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(IntegratedCompletenessError, match="falsely claims"):
            evaluate(web=path)
