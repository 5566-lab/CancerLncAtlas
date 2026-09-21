from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cc_hhgt.v32.exact_pathway_report_query import (
    ExactPathwayReportAssetError,
    ExactPathwayReportQuery,
)
from website.backend.v32_staging_api import create_staging_app


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/v32_exact_pathway_report_manifest_20260826_r1"
AUDIT = ROOT / "artifacts/v32_exact_pathway_report_manifest_20260826_r1_independent_audit"
BINDING = RELEASE / "EXACT_PATHWAY_REPORT_BINDING.json"
AUDIT_BINDING = AUDIT / "INDEPENDENT_AUDIT_BINDING.json"
BINDING_SHA256 = "2d24a623d7a8086ab966e296b1d0d1546a8ea9ba2b8282be358fb16734becd61"
AUDIT_BINDING_SHA256 = "cc2e8b44237dd0b27c2e9ddd70c5f4f5cafed9e8aefc6c60084ed44b9211c789"
REGISTRY = ROOT / "artifacts/v32_staging/core_registry_refresh_20260826_r1/RELEASE_REGISTRY.json"


def test_exact_report_is_hash_bound_and_independently_audited() -> None:
    query = ExactPathwayReportQuery(
        BINDING,
        expected_binding_sha256=BINDING_SHA256,
        audit_binding_path=AUDIT_BINDING,
        expected_audit_binding_sha256=AUDIT_BINDING_SHA256,
    )
    capability = query.capability()
    assert capability["artifact_id"] == "v32_exact_pathway_report_manifest"
    assert capability["prediction_rows"] == 3_300_000
    assert capability["fold_count"] == 5
    assert capability["old_checkpoint_loaded"] is False
    assert capability["independent_audit_pass_count"] == 66
    audit = json.loads(AUDIT_BINDING.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["fail_count"] == 0


def test_exact_report_hash_drift_fails_closed() -> None:
    with pytest.raises(ExactPathwayReportAssetError, match="binding SHA256 drift"):
        ExactPathwayReportQuery(
            BINDING,
            expected_binding_sha256="0" * 64,
            audit_binding_path=AUDIT_BINDING,
            expected_audit_binding_sha256=AUDIT_BINDING_SHA256,
        )


def test_exact_report_staging_route_is_mounted() -> None:
    app = create_staging_app(
        REGISTRY,
        exact_pathway_report_binding_path=BINDING,
        exact_pathway_report_binding_sha256=BINDING_SHA256,
        exact_pathway_report_audit_binding_path=AUDIT_BINDING,
        exact_pathway_report_audit_binding_sha256=AUDIT_BINDING_SHA256,
    )
    client = TestClient(app)
    response = client.get("/v3.2-staging/exact-pathway/report")
    assert response.status_code == 200
    assert response.json()["prediction_rows"] == 3_300_000
    health = client.get("/v3.2-staging/health").json()
    assert health["exact_pathway_report_query"] == (
        "ENABLED_HASH_BOUND_INDEPENDENTLY_AUDITED_66_OF_66"
    )
