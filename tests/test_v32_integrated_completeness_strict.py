from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cc_hhgt.v32.integrated_completeness_strict import (
    REQUIRED_R3_HISTORICAL_AUDIT_SHA256,
    REQUIRED_R3_HISTORICAL_BINDING_SHA256,
    REQUIRED_R4_DOWNLOAD_AUDIT_SHA256,
    REQUIRED_R4_DOWNLOAD_BINDING_SHA256,
    StrictCompletenessError,
    evaluate_strict_integrated_completeness,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def strict_report() -> dict:
    report, _ = evaluate_strict_integrated_completeness(
        repo_root=ROOT,
        parity_contract_path=ROOT / "config/v32_historical_capability_parity.yaml",
        unified_bindings_path=ROOT / "config/v32_unified_staging_bindings.json",
        web_catalog_path=ROOT / "website/frontend/v32-capability-catalog.json",
        frontend_html_path=ROOT / "website/frontend/v32-staging.html",
        frontend_javascript_path=ROOT / "website/frontend/assets/v32-staging.js",
        main_app_path=ROOT / "website/backend/app.py",
        evaluator_code_path=ROOT / "cc_hhgt/v32/integrated_completeness_strict.py",
        runner_code_path=ROOT / "scripts/audit_v32_integrated_completeness_strict.py",
    )
    return report


def _capability(report: dict, capability_id: str) -> dict:
    return next(
        row for row in report["capabilities"] if row["capability_id"] == capability_id
    )


def test_catalog_declaration_is_separate_from_runtime(strict_report: dict) -> None:
    assert strict_report["catalog_declared_is_not_implemented"] is True
    assert strict_report["required_release_revisions"] == {
        "download_catalog_r4_binding_sha256": REQUIRED_R4_DOWNLOAD_BINDING_SHA256,
        "download_catalog_r4_audit_sha256": REQUIRED_R4_DOWNLOAD_AUDIT_SHA256,
        "historical_remediation_r3_binding_sha256": REQUIRED_R3_HISTORICAL_BINDING_SHA256,
        "historical_remediation_r3_audit_sha256": REQUIRED_R3_HISTORICAL_AUDIT_SHA256,
    }
    exact_api = _capability(strict_report, "exact_pathway")["gates"]["api"]
    assert exact_api["evidence"]["catalog_state"] == "CATALOG_DECLARED"
    assert exact_api["evidence"]["runtime"]["observation_valid"] is True
    assert exact_api["status"] == "FAIL"
    assert exact_api["evidence"]["required_contract_mapping"]["status"] == "INCOMPLETE"


def test_feature_flag_main_site_is_really_servable(strict_report: dict) -> None:
    main_site = strict_report["implementation_evidence"]["frontend"]["main_site"]
    assert main_site["constructed"] is True
    assert main_site["servable"] is True
    assert main_site["staging_api_mounted_on_main_app"] is True
    assert all(row["http_status"] == 200 for row in main_site["responses"])
    assert strict_report["implementation_evidence"]["frontend"][
        "javascript_syntax_ok"
    ] is True


def test_true_four_gate_blockers_are_not_hidden(strict_report: dict) -> None:
    assert strict_report["status"] == "PARTIAL"
    assert strict_report["complete_capability_count"] == 21
    assert strict_report["blocking_capability_ids"] == [
        "drug",
        "exact_pathway",
        "mutation",
        "single_cell",
    ]
    assert strict_report["gate_counts"] == {
        "artifact": {"pass": 23, "fail": 2},
        "api": {"pass": 21, "fail": 4},
        "ui": {"pass": 23, "fail": 2},
        "download": {"pass": 23, "fail": 2},
    }
    exact_artifacts = _capability(strict_report, "exact_pathway")["gates"][
        "artifact"
    ]
    assert exact_artifacts["status"] == "PASS"
    assert all(
        row["state"] == "HASH_BOUND"
        for row in exact_artifacts["evidence"]["resolutions"]
    )
    mutation = _capability(strict_report, "mutation")
    assert mutation["gates"]["artifact"]["status"] == "PASS"
    assert mutation["gates"]["api"]["status"] == "FAIL"


def test_typed_gaps_are_observed_but_never_count_as_servable(
    strict_report: dict,
) -> None:
    single_cell = _capability(strict_report, "single_cell")
    states = {
        row["artifact_id"]: row["state"]
        for row in single_cell["gates"]["artifact"]["evidence"]["resolutions"]
    }
    assert states["v32_sc_ucell"] == "TYPED_GAP"
    assert states["v32_sc_pseudotime"] == "TYPED_GAP"
    assert states["v32_sc_figure_manifest"] == "TYPED_GAP"
    sc_runtime = single_cell["gates"]["api"]["evidence"]["runtime"]
    assert sc_runtime["observation_valid"] is True
    assert sc_runtime["servable_success"] is False
    drug_runtime = _capability(strict_report, "drug")["gates"]["api"]["evidence"][
        "runtime"
    ]
    assert drug_runtime["observation_valid"] is True
    assert drug_runtime["servable_success"] is False
    assert drug_runtime["probes"][0]["http_status"] == 503


def test_r4_binding_hash_drift_is_rejected() -> None:
    unified = json.loads(
        (ROOT / "config/v32_unified_staging_bindings.json").read_text(encoding="utf-8")
    )
    unified["bindings"]["download_catalog"]["sha256"] = "0" * 64
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        drifted = Path(temporary) / "unified.json"
        drifted.write_text(json.dumps(unified), encoding="utf-8")
        with pytest.raises(StrictCompletenessError, match="pinned r4/r3 revision"):
            evaluate_strict_integrated_completeness(
                repo_root=ROOT,
                parity_contract_path=ROOT
                / "config/v32_historical_capability_parity.yaml",
                unified_bindings_path=drifted,
                web_catalog_path=ROOT
                / "website/frontend/v32-capability-catalog.json",
                frontend_html_path=ROOT / "website/frontend/v32-staging.html",
                frontend_javascript_path=ROOT
                / "website/frontend/assets/v32-staging.js",
                main_app_path=ROOT / "website/backend/app.py",
                evaluator_code_path=ROOT
                / "cc_hhgt/v32/integrated_completeness_strict.py",
                runner_code_path=ROOT
                / "scripts/audit_v32_integrated_completeness_strict.py",
            )
