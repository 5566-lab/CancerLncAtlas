from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cc_hhgt.v32.integrated_completeness_strict_r3 import API_EQUIVALENTS
from cc_hhgt.v32.integrated_completeness_strict import sha256_file
from cc_hhgt.v32.unified_staging_bindings import create_app_from_unified_bindings


ROOT = Path(__file__).resolve().parents[1]
R3_ROOT = ROOT / "artifacts/v32_integrated_completeness_strict_audit_20260826_r3"
R3_BINDING = R3_ROOT / "INTEGRATED_COMPLETENESS_STRICT_R3_BINDING.json"
R3_REPORT = R3_ROOT / "INTEGRATED_COMPLETENESS_STRICT_R3_REPORT.json"
R3_BINDING_SHA256 = "2b1cad19c551b7eb04e29f9169f51f08325a885c4cb042e78f69eb6e550af2c1"
R2_BINDING = (
    ROOT
    / "artifacts/v32_integrated_completeness_strict_audit_20260826_r2/INTEGRATED_COMPLETENESS_STRICT_BINDING.json"
)
R2_BINDING_SHA256 = "8c6b772b466e183f31c3acdbd880b56c7758d615fc32e73b77323ded9f1ae653"


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(R3_REPORT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client():
    app = create_app_from_unified_bindings(
        ROOT / "config/v32_unified_staging_bindings.json", repo_root=ROOT
    )
    with TestClient(app) as value:
        yield value


def _capability(report: dict, capability_id: str) -> dict:
    return next(
        row for row in report["capabilities"] if row["capability_id"] == capability_id
    )


def test_r3_is_new_and_does_not_overwrite_r2(report: dict) -> None:
    assert sha256_file(R3_BINDING) == R3_BINDING_SHA256
    assert sha256_file(R2_BINDING) == R2_BINDING_SHA256
    assert report["audit_revision"] == "r3"
    assert report["supersedes_strict_r2_binding_sha256"] == R2_BINDING_SHA256


def test_r3_four_gate_result(report: dict) -> None:
    assert report["status"] == "PARTIAL"
    assert report["complete_capability_count"] == 23
    assert report["blocking_capability_ids"] == ["drug", "single_cell"]
    assert report["gate_counts"] == {
        "artifact": {"pass": 23, "fail": 2},
        "api": {"pass": 24, "fail": 1},
        "ui": {"pass": 24, "fail": 1},
        "download": {"pass": 23, "fail": 2},
    }
    assert _capability(report, "exact_pathway")["status"] == "COMPLETE"
    assert _capability(report, "mutation")["status"] == "COMPLETE"


def test_new_exact_and_mutation_routes_are_real(client: TestClient) -> None:
    assert API_EQUIVALENTS["exact_pathway"]["GET /api/site/search"] == (
        "GET",
        "/v3.2-staging/search",
    )
    search = client.get("/v3.2-staging/search", params={"q": "BRCA", "limit": 5})
    datasets = client.get("/v3.2-staging/datasets")
    cancers = client.get("/v3.2-staging/cancers")
    assert search.status_code == datasets.status_code == cancers.status_code == 200
    assert search.json()["returned_rows"] > 0
    assert datasets.json()["returned_rows"] > 0
    assert cancers.json()["returned_rows"] == 33

    mutation = client.post(
        "/v3.2-staging/predict/mutation-context",
        json={"cancer_id": "BRCA", "top_k": 1},
    )
    assert mutation.status_code == 200
    body = mutation.json()
    assert body["returned_rows"] == 1
    assert body["request_contract"] == "V3.2_MUTATION_CONTEXT_POST_V1"
    assert body["mutation_retained_despite_no_increment"] is True
    assert body["changes_primary_ranking"] is False
    assert body["old_predictions_used"] is False


def test_single_cell_routes_render_typed_unavailability_without_false_completion(
    client: TestClient, report: dict
) -> None:
    activity = client.get(
        "/v3.2-staging/single-cell/activity/HNSC", params={"limit": 2}
    )
    trajectory = client.get(
        "/v3.2-staging/single-cell/trajectory/HNSC",
        params={"level": "PATHWAY", "limit": 2},
    )
    figures = client.get("/v3.2-staging/single-cell/figures/HNSC")
    figure = client.get(
        "/v3.2-staging/single-cell/figure/HNSC/STRICT_AUDIT_PROBE"
    )
    assert activity.status_code == trajectory.status_code == 200
    assert figures.status_code == figure.status_code == 200
    assert activity.json()["returned_rows"] == 2
    assert trajectory.json()["pseudotime_numeric_values"] == 0
    assert all(
        row["pseudotime_available"] is False for row in trajectory.json()["rows"]
    )
    assert figures.json()["availability"] is False
    assert figures.json()["rows"] == []
    assert figure.json()["availability"] is False
    assert figure.json()["file"] is None

    sc = _capability(report, "single_cell")
    assert sc["status"] == "PARTIAL"
    assert sc["gates"]["api"]["status"] == "PASS"
    assert sc["gates"]["ui"]["status"] == "PASS"
    assert sc["gates"]["artifact"]["status"] == "FAIL"
    assert sc["gates"]["download"]["status"] == "FAIL"
    assert report["typed_unavailable_counts_as_artifact_complete"] is False
    assert report["typed_unavailable_counts_as_download_complete"] is False


def test_independent_r3_audit_passes() -> None:
    audit = json.loads(
        (
            ROOT
            / "artifacts/v32_integrated_completeness_strict_audit_20260826_r3_independent_audit/INDEPENDENT_AUDIT_R3_REPORT.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["status"] == "PASS"
    assert audit["fail_count"] == 0
    assert audit["recomputed_blocking_capability_ids"] == ["drug", "single_cell"]
    assert audit["accepted_as_complete"] is False
    assert audit["accepted_as_truthful_staging_audit"] is True
