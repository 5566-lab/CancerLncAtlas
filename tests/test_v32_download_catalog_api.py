from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cc_hhgt.v32.download_catalog import (
    AuditedDownloadCatalog,
    DownloadCatalogError,
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
    / "v32_staging_download_catalog_20260827_r6"
    / "DOWNLOAD_CATALOG_BINDING.json"
)
BINDING_SHA256 = "7d7c22553562164d1c7e5374cda08cfd2d1818cfe9c62652b9e7d02b9374b368"
AUDIT = (
    ROOT
    / "artifacts"
    / "v32_staging_download_catalog_20260827_r6_independent_audit_r2"
    / "DOWNLOAD_CATALOG_INDEPENDENT_AUDIT_BINDING.json"
)
AUDIT_SHA256 = "7045aa4bb86efc09fc3bad05c28809a63999df072e68aaddd53a70be673e37e2"


def _app():
    return create_staging_app(
        REGISTRY,
        download_catalog_binding_path=BINDING,
        download_catalog_binding_sha256=BINDING_SHA256,
        download_catalog_audit_binding_path=AUDIT,
        download_catalog_audit_binding_sha256=AUDIT_SHA256,
    )


def _contains_private_path(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            key in {"source_path", "source_repo_relative_path", "path"}
            or _contains_private_path(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_private_path(item) for item in value)
    return False


@pytest.mark.skipif(
    not REGISTRY.is_file() or not BINDING.is_file() or not AUDIT.is_file(),
    reason="formal download catalog absent",
)
def test_audited_download_catalog_and_staging_api() -> None:
    mounted = AuditedDownloadCatalog(
        BINDING,
        expected_binding_sha256=BINDING_SHA256,
        audit_binding_path=AUDIT,
        expected_audit_binding_sha256=AUDIT_SHA256,
        repository_root=ROOT,
    )
    assert len(mounted.public_catalog()["downloads"]) == 54
    assert mounted.audit["pass_count"] == 745

    client = TestClient(_app())
    health = client.get("/v3.2-staging/health")
    assert health.status_code == 200
    assert health.json()["download_catalog_query"] == (
        "ENABLED_54_IDS_AUDITED_745_OF_745"
    )
    catalog = client.get("/v3.2-staging/downloads")
    assert catalog.status_code == 200
    assert catalog.json()["download_count"] == 54
    assert not _contains_private_path(catalog.json())

    entry = client.get("/v3.2-staging/downloads/v32_model_card")
    assert entry.status_code == 200
    assert entry.json()["status"] == "READY_FILE"
    assert entry.json()["download_url"].endswith("/v32_model_card/file")
    payload = client.get(entry.json()["download_url"])
    assert payload.status_code == 200
    assert len(payload.content) > 0
    assert payload.headers["x-content-sha256"] == entry.json()["file"]["sha256"]
    assert payload.headers["x-cancerlncatlas-version"] == "V3.2"

    parts = client.get("/v3.2-staging/downloads/gene_set_members")
    assert parts.status_code == 200
    assert parts.json()["status"] == "READY_PARTS"
    assert all("download_url" in part for part in parts.json()["parts"])

    single_cell = client.get(
        "/v3.2-staging/downloads/single_cell_associations"
    )
    assert single_cell.status_code == 200
    sc_parts = {
        part["relative_name"]: part for part in single_cell.json()["parts"]
    }
    typed = sc_parts[
        "single_cell_associations/celltype_typed_predictions.parquet"
    ]
    typed_payload = client.get(typed["download_url"])
    assert typed_payload.status_code == 200
    assert typed_payload.headers["x-content-sha256"] == typed["sha256"]
    assert typed_payload.headers["x-partition-tree-sha256"] == (
        single_cell.json()["sha256_tree"]
    )

    drug_parts = client.get("/v3.2-staging/downloads/drug_response_predictions")
    assert drug_parts.status_code == 200
    assert drug_parts.json()["status"] == "READY_PARTS"
    mechanism = client.get("/v3.2-staging/downloads/drug_mechanisms")
    assert mechanism.status_code == 200
    assert mechanism.json()["status"] == "SERVER_HASH_PINNED_DOWNLOAD"
    assert mechanism.json()["manifest_endpoint"].startswith("GET /v3.2-staging/")
    assert client.get("/v3.2-staging/downloads/not_a_download").status_code == 404


def test_download_catalog_mount_is_optional_and_atomic() -> None:
    client = TestClient(create_staging_app(REGISTRY))
    assert client.get("/v3.2-staging/downloads").status_code == 503
    with pytest.raises(DownloadCatalogError, match="both binding path"):
        create_staging_app(REGISTRY, download_catalog_binding_path=BINDING)
    with pytest.raises(DownloadCatalogError, match="both audit path"):
        create_staging_app(REGISTRY, download_catalog_audit_binding_path=AUDIT)
    with pytest.raises(
        DownloadCatalogError, match="both release and independent-audit"
    ):
        create_staging_app(
            REGISTRY,
            download_catalog_binding_path=BINDING,
            download_catalog_binding_sha256=BINDING_SHA256,
        )


@pytest.mark.skipif(not BINDING.is_file(), reason="formal binding absent")
def test_download_catalog_outer_hash_drift_is_rejected() -> None:
    with pytest.raises(DownloadCatalogError, match="binding SHA256 drift"):
        create_staging_app(
            REGISTRY,
            download_catalog_binding_path=BINDING,
            download_catalog_binding_sha256="0" * 64,
            download_catalog_audit_binding_path=AUDIT,
            download_catalog_audit_binding_sha256=AUDIT_SHA256,
        )
