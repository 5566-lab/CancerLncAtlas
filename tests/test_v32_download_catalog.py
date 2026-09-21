from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import cc_hhgt.v32.download_catalog as download_catalog_module

from cc_hhgt.v32.capability_parity import load_parity_config
from cc_hhgt.v32.download_catalog import (
    AuditedDownloadCatalog,
    DownloadCatalogError,
    _safe_relative_name,
    catalog_entry,
    load_download_catalog,
    resolve_download_part,
    resolve_download_payload,
)


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts" / "v32_staging_download_catalog_20260827_r6"
AUDIT = ROOT / "artifacts" / "v32_staging_download_catalog_20260827_r6_independent_audit_r2"
CATALOG = RELEASE / "V32_STAGING_DOWNLOAD_CATALOG.json"
CATALOG_SHA256 = "794e47c3a56e8f3f16e84ce112fcf3ce104095b343d3428bc9da95f36eee1133"
BINDING_SHA256 = "7d7c22553562164d1c7e5374cda08cfd2d1818cfe9c62652b9e7d02b9374b368"
AUDIT_BINDING_SHA256 = (
    "7045aa4bb86efc09fc3bad05c28809a63999df072e68aaddd53a70be673e37e2"
)


def _expected_downloads() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for capability_id, capability in load_parity_config()["capabilities"].items():
        for download_id in capability["gates"]["download"]["ids"]:
            result.setdefault(download_id, []).append(capability_id)
    return {key: sorted(value) for key, value in result.items()}


def test_formal_catalog_closes_all_54_contract_ids_without_hiding_gaps() -> None:
    catalog = load_download_catalog(CATALOG, expected_sha256=CATALOG_SHA256)
    expected = _expected_downloads()
    observed = {row["download_id"]: row for row in catalog["downloads"]}
    assert len(expected) == 54
    assert set(observed) == set(expected)
    assert catalog["production_deployed"] is False
    assert catalog["release_ready"] is False
    assert catalog["status_counts"] == {
        "DYNAMIC_QUERY_EXPORT": 4,
        "GAP_TYPED_UNAVAILABLE": 1,
        "READY_FILE": 39,
        "READY_PARTS": 8,
        "SERVER_HASH_PINNED_DOWNLOAD": 2,
    }
    for download_id, row in observed.items():
        assert sorted(row["capability_ids"]) == expected[download_id]
        assert row["model_version"] == "V3.2"
        assert row["family_to_exact_broadcast"] is False
        assert row["production_deployed"] is False
        assert row["availability_encoding"] == "null_with_reason"
        assert row["unavailable_fill_value"] is None

    assert observed["single_cell_pseudotime"]["status"] == "GAP_TYPED_UNAVAILABLE"
    assert observed["single_cell_pseudotime"]["download_implemented"] is False
    assert observed["single_cell_ucell"]["status"] == "SERVER_HASH_PINNED_DOWNLOAD"
    assert observed["single_cell_ucell"]["contract_complete"] is True
    sc_association_parts = {
        value["relative_name"]: value
        for value in observed["single_cell_associations"]["parts"]
    }
    assert sc_association_parts[
        "single_cell_associations/celltype_typed_predictions.parquet"
    ]["sha256"] == (
        "228ae23cf8b5d320a66ab2893d66cd017671bcd804d7a17b109da7c548bfd65c"
    )
    assert {
        "single_cell_formal_context",
        "single_cell_formal_context_independent",
    } <= set(observed["single_cell_associations"]["authority_ids"])
    assert {
        "single_cell_formal_context",
        "single_cell_formal_context_independent",
    } <= set(observed["single_cell_activity"]["authority_ids"])
    assert observed["evidence_probabilities"]["status"] == "READY_PARTS"
    assert observed["evidence_probabilities"]["contract_complete"] is True
    assert "missing_components" not in observed["evidence_probabilities"]
    evidence_parts = {
        value["relative_name"]: value
        for value in observed["evidence_probabilities"]["parts"]
    }
    assert evidence_parts[
        "evidence_probabilities/evidence_direction_probabilities.parquet"
    ]["sha256"] == (
        "a2b41290275cb57c339abe76df248a3f3f2f69ef08e8a33f16b83ae8255f40a9"
    )
    assert observed["drug_response_predictions"]["status"] == "READY_PARTS"
    assert observed["drug_evidence"]["status"] == "READY_PARTS"
    assert observed["drug_mechanisms"]["status"] == "SERVER_HASH_PINNED_DOWNLOAD"
    gap_ids = {row["artifact_id"] for row in catalog["known_capability_gaps"]}
    assert "v32_sc_figure_manifest" in gap_ids
    assert "v32_evidence_direction_probability" not in gap_ids
    sc_metadata = catalog["capability_audit_metadata"]["single_cell"]
    assert sc_metadata["download_contract_id_created"] is False
    assert sc_metadata["independent_audit_checks"] == 68
    assert sc_metadata["independent_audit_failed_checks"] == 0
    assert sc_metadata["lncrna_detection_counts"]["cancers"] == 33
    assert sc_metadata["lncrna_detection_counts"]["detected_union"] == 15_879
    assert sc_metadata["pseudotime_numeric_rows"] == 0
    assert sc_metadata["ucell_formal_cancers_covered"] == 17
    assert sc_metadata["current_v32_figure_files"] == 0
    assert sc_metadata["formal_context_binding_sha256"] == (
        "798df5cd58278e93daba3d6b7c20e47a394ce65a6c0ee3df06693c4df3c8cc4c"
    )
    assert sc_metadata["formal_context_independent_audit_binding_sha256"] == (
        "612bfeb3a74384c706d6af98dc871535f462664580ed3dd478be90555fe1410c"
    )
    assert sc_metadata["formal_context_independent_audit_checks"] == 25
    assert sc_metadata["formal_context_independent_audit_failures"] == 0
    assert sc_metadata["formal_context_cancers_available"] == 17
    assert sc_metadata["formal_context_cancers_total"] == 33
    assert sc_metadata["cell_level_ucell_remaining_cancers"] == 0
    assert sc_metadata["typed_predictions_downloaded"] is True
    assert sc_metadata["single_cell_fusion_weight_zero"] is True
    assert sc_metadata["exact_primary_modified"] is False


def test_static_parts_are_secure_and_dynamic_results_have_no_placeholders() -> None:
    catalog = load_download_catalog(CATALOG, expected_sha256=CATALOG_SHA256)
    for row in catalog["downloads"]:
        if row["status"] in {"READY_PARTS", "PARTIAL_READY_PARTS"}:
            names = [part["relative_name"] for part in row["parts"]]
            assert len(names) == len(set(names)) == row["part_count"]
            assert all(_safe_relative_name(name) == name for name in names)
            payload = "\n".join(
                f"{part['relative_name']}\t{part['sha256']}\t{part['bytes']}"
                for part in sorted(row["parts"], key=lambda value: value["relative_name"])
            ) + "\n"
            assert hashlib.sha256(payload.encode()).hexdigest() == row["sha256_tree"]
        if row["status"] == "DYNAMIC_QUERY_EXPORT":
            assert "file" not in row and "parts" not in row
            assert row["result_materialization"] == "ON_REQUEST_NO_PLACEHOLDER_FILE"
            assert row["export_endpoint"].startswith("POST /v3.2-staging/")
        if row["status"] == "SERVER_HASH_PINNED_DOWNLOAD":
            assert "file" not in row and "parts" not in row
            assert row["manifest_endpoint"].startswith("GET /v3.2-staging/")
            assert row["download_endpoint"].startswith("GET /v3.2-staging/")
            assert row["request_time_payload_rehash"] is True
    with pytest.raises(DownloadCatalogError):
        _safe_relative_name("../escape.parquet")
    with pytest.raises(DownloadCatalogError):
        _safe_relative_name("C:/escape.parquet")


def test_loader_is_hash_pinned_and_catalog_entry_is_unambiguous(tmp_path: Path) -> None:
    copied = tmp_path / "catalog.json"
    copied.write_bytes(CATALOG.read_bytes())
    catalog = load_download_catalog(copied, expected_sha256=CATALOG_SHA256)
    assert catalog_entry(catalog, "network_edges")["status"] == "READY_FILE"
    with pytest.raises(DownloadCatalogError):
        catalog_entry(catalog, "not_a_download")
    copied.write_bytes(copied.read_bytes() + b"\n")
    with pytest.raises(DownloadCatalogError, match="SHA256 drift"):
        load_download_catalog(copied, expected_sha256=CATALOG_SHA256)
    with pytest.raises(DownloadCatalogError, match="pinned catalog SHA256"):
        load_download_catalog(CATALOG, expected_sha256="")


def test_request_time_resolver_rehashes_payloads_and_fails_closed(tmp_path: Path) -> None:
    catalog = load_download_catalog(CATALOG, expected_sha256=CATALOG_SHA256)
    model_card = resolve_download_payload(
        catalog, "v32_model_card", repository_root=ROOT
    )
    assert model_card["kind"] == "file"
    assert model_card["relative_name"] == "V3.2_MODEL_REPORT.md"
    dynamic = resolve_download_payload(
        catalog, "protein_set_results", repository_root=ROOT
    )
    assert dynamic == {
        "kind": "dynamic_query_export",
        "download_id": "protein_set_results",
        "endpoint": "POST /v3.2-staging/enrichment/mixed-exact-pathway",
        "export_formats": ["json", "tsv"],
    }
    server = resolve_download_payload(
        catalog, "drug_mechanisms", repository_root=ROOT
    )
    assert server["kind"] == "server_hash_pinned_download"
    assert server["manifest_endpoint"].startswith("GET /v3.2-staging/")

    repository = tmp_path / "repo"
    repository.mkdir()
    payload = repository / "payload.tsv"
    payload.write_text("value\n", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    fixture = {
        "downloads": [
            {
                "download_id": "fixture",
                "status": "READY_FILE",
                "file": {
                    "source_path": str(payload),
                    "source_repo_relative_path": "payload.tsv",
                    "relative_name": "payload.tsv",
                    "sha256": digest,
                    "bytes": payload.stat().st_size,
                },
            }
        ]
    }
    assert resolve_download_payload(
        fixture, "fixture", repository_root=repository
    )["sha256"] == digest
    payload.write_text("drift\n", encoding="utf-8")
    with pytest.raises(DownloadCatalogError, match="drift"):
        resolve_download_payload(fixture, "fixture", repository_root=repository)


def test_partition_resolver_hashes_only_requested_part_and_keeps_tree_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    records = []
    for index in range(3):
        path = repository / f"part-{index}.tsv"
        path.write_text(f"part\t{index}\n", encoding="utf-8")
        records.append(
            {
                "source_path": str(path),
                "source_repo_relative_path": path.name,
                "relative_name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    tree_payload = "\n".join(
        f"{row['relative_name']}\t{row['sha256']}\t{row['bytes']}"
        for row in sorted(records, key=lambda value: value["relative_name"])
    ) + "\n"
    fixture = {
        "downloads": [
            {
                "download_id": "partition_fixture",
                "status": "READY_PARTS",
                "part_count": 3,
                "sha256_tree": hashlib.sha256(tree_payload.encode()).hexdigest(),
                "parts": records,
            }
        ]
    }
    real_sha256_file = download_catalog_module.sha256_file
    hashed_paths: list[Path] = []

    def tracked_sha256_file(path: str | Path) -> str:
        hashed_paths.append(Path(path).resolve())
        return real_sha256_file(path)

    monkeypatch.setattr(download_catalog_module, "sha256_file", tracked_sha256_file)
    result = resolve_download_part(
        fixture, "partition_fixture", 1, repository_root=repository
    )
    assert result["kind"] == "part"
    assert result["part_index"] == 1
    assert result["sha256_tree"] == fixture["downloads"][0]["sha256_tree"]
    assert hashed_paths == [(repository / "part-1.tsv").resolve()]

    hashed_paths.clear()
    with pytest.raises(DownloadCatalogError, match="Unknown download part index"):
        resolve_download_part(
            fixture, "partition_fixture", 3, repository_root=repository
        )
    assert hashed_paths == []

    fixture["downloads"][0]["parts"][0]["bytes"] += 1
    with pytest.raises(DownloadCatalogError, match="tree SHA256 drift"):
        resolve_download_part(
            fixture, "partition_fixture", 1, repository_root=repository
        )


def test_independent_audit_passed_and_did_not_import_materializer() -> None:
    binding = json.loads(
        (AUDIT / "DOWNLOAD_CATALOG_INDEPENDENT_AUDIT_BINDING.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (AUDIT / "DOWNLOAD_CATALOG_INDEPENDENT_AUDIT_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    assert binding["status"] == "PASS"
    assert binding["fail_count"] == 0
    assert binding["pass_count"] == 745
    assert binding["accepted_for_staging_api_integration"] is True
    assert report["all_file_hashes_recomputed"] is True
    assert report["all_partition_tree_hashes_recomputed"] is True
    assert report["file_records_rehashed"] > 200
    source = (
        ROOT / "scripts" / "audit_v32_staging_download_catalog_independent.py"
    ).read_text(encoding="utf-8")
    assert "from cc_hhgt.v32.download_catalog import" not in source
    assert report["materializer_imported"] is False

    mounted = AuditedDownloadCatalog(
        RELEASE / "DOWNLOAD_CATALOG_BINDING.json",
        expected_binding_sha256=BINDING_SHA256,
        audit_binding_path=AUDIT / "DOWNLOAD_CATALOG_INDEPENDENT_AUDIT_BINDING.json",
        expected_audit_binding_sha256=AUDIT_BINDING_SHA256,
        repository_root=ROOT,
    )
    assert mounted.entry("evidence_probabilities")["status"] == "READY_PARTS"
