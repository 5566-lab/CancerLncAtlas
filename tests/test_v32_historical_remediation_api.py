from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cc_hhgt.v32.historical_artifact_remediation import (
    HistoricalArtifactRemediationError,
)
from website.backend.v32_staging_api import create_staging_app


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = (
    ROOT
    / "artifacts"
    / "v32_staging"
    / "core_registry_refresh_20260826_r1"
    / "RELEASE_REGISTRY.json"
)
BINDING = (
    ROOT
    / "artifacts"
    / "v32_historical_artifact_remediation_20260826_r3_evidence_direction_closed"
    / "HISTORICAL_ARTIFACT_REMEDIATION_BINDING.json"
)
BINDING_SHA256 = "b128681bff1856ab09df21c387013b431e0da956bce6f32243b1361661e0c4d8"
AUDIT = (
    ROOT
    / "artifacts"
    / "v32_historical_artifact_remediation_20260826_r3_evidence_direction_closed_independent_audit"
    / "INDEPENDENT_AUDIT_BINDING.json"
)
AUDIT_SHA256 = "b652fcb0cdecddfd698983322c6b485025a4a656f0b86f7856b52ef20840e3ae"


def _formal_app():
    return create_staging_app(
        REGISTRY,
        historical_remediation_binding_path=BINDING,
        historical_remediation_binding_sha256=BINDING_SHA256,
        historical_remediation_audit_binding_path=AUDIT,
        historical_remediation_audit_binding_sha256=AUDIT_SHA256,
    )


@pytest.mark.skipif(
    not REGISTRY.is_file() or not BINDING.is_file() or not AUDIT.is_file(),
    reason="formal historical-remediation artifacts absent",
)
def test_historical_remediation_formal_api_smoke() -> None:
    client = TestClient(_formal_app())
    health = client.get("/v3.2-staging/health")
    assert health.status_code == 200
    assert health.json()["historical_artifact_remediation_query"] == (
        "ENABLED_HASH_PINNED_WITH_INDEPENDENT_AUDIT"
    )

    capability = client.get(
        "/v3.2-staging/historical-remediation/mutation/capability"
    )
    assert capability.status_code == 200
    assert capability.json()["status"] == "SUCCESS_HASH_BOUND_GENOMIC_LOGICAL_RELEASE"

    clinical = client.get(
        "/v3.2-staging/clinical/translational-priority",
        params={"cancer_id": "BRCA", "endpoint": "OS", "limit": 3},
    )
    assert clinical.status_code == 200
    assert clinical.json()["count"] == 3
    assert all(
        row["changes_primary_ranking"] is False
        for row in clinical.json()["rows"]
    )

    mutation = client.get(
        "/v3.2-staging/mutation/subgroups",
        params={"cancer_id": "BRCA", "limit": 3},
    )
    assert mutation.status_code == 200
    assert mutation.json()["scientific_status"] == (
        "NO_INCREMENT_DIAGNOSTIC_ONLY_RETAINED"
    )
    assert all(row["family_broadcast"] is False for row in mutation.json()["rows"])

    cnv = client.get("/v3.2-staging/cnv/coverage", params={"cancer_id": "ACC"})
    assert cnv.status_code == 503
    assert "superseded historical coverage artifact is disabled" in cnv.json()["detail"]
    cnv_download = client.get("/v3.2-staging/downloads/cnv_context")
    assert cnv_download.status_code == 200
    assert cnv_download.json()["status"] == "PENDING_FORMAL_SUCCESS"
    assert cnv_download.json()["data_present"] is False


def test_historical_remediation_bundle_is_optional_and_atomic() -> None:
    client = TestClient(create_staging_app(REGISTRY))
    for path in (
        "/v3.2-staging/historical-remediation/mutation/capability",
        "/v3.2-staging/clinical/translational-priority?cancer_id=BRCA",
        "/v3.2-staging/mutation/subgroups?cancer_id=BRCA",
        "/v3.2-staging/cnv/coverage",
    ):
        assert client.get(path).status_code == 503

    with pytest.raises(HistoricalArtifactRemediationError, match="both binding path"):
        create_staging_app(
            REGISTRY, historical_remediation_binding_path=BINDING
        )
    with pytest.raises(HistoricalArtifactRemediationError, match="both audit path"):
        create_staging_app(
            REGISTRY, historical_remediation_audit_binding_path=AUDIT
        )
    with pytest.raises(
        HistoricalArtifactRemediationError, match="both release and independent-audit"
    ):
        create_staging_app(
            REGISTRY,
            historical_remediation_binding_path=BINDING,
            historical_remediation_binding_sha256=BINDING_SHA256,
        )


@pytest.mark.skipif(not BINDING.is_file(), reason="formal binding absent")
def test_historical_remediation_outer_hash_drift_is_rejected() -> None:
    with pytest.raises(
        HistoricalArtifactRemediationError, match="external binding hash mismatch"
    ):
        create_staging_app(
            REGISTRY,
            historical_remediation_binding_path=BINDING,
            historical_remediation_binding_sha256="0" * 64,
            historical_remediation_audit_binding_path=AUDIT,
            historical_remediation_audit_binding_sha256=AUDIT_SHA256,
        )
