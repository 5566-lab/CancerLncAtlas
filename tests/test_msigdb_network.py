import hashlib
import json

import pytest

from cla_engine.msigdb_network import build_mapping


def fixture_rows():
    registry = [{
        "source_target_id": "MSIGDB:KEGG_MEDICUS:EXAMPLE",
        "source_name": "EXAMPLE", "source_database": "KEGG_MEDICUS",
        "canonical_id": "old",
    }]
    metadata = [{
        "source_name": "EXAMPLE", "exact_source": "N00001",
        "systematic_name": "M123", "status": "VERIFIED_SOURCE_METADATA",
        "external_details": ["https://www.kegg.jp/entry/N00001"],
    }]
    return registry, metadata


def test_exact_source_and_redirect_chain():
    registry, metadata = fixture_rows()
    rows, summary = build_mapping(
        registry, metadata, [], [{"canonical_id": "middle", "integrated_canonical_id": "merged"}],
        [{"previous_internal_id": "old", "canonical_id": "middle"}],
        verify_cache=False,
    )
    assert rows[0]["mapping_status"] == "mapped"
    assert rows[0]["canonical_id"] == "merged"
    assert summary["mapped_distinct_network_ids"] == 1


def test_never_infer_from_name_or_members():
    registry, _ = fixture_rows()
    registry[0]["source_name"] = "CANCER_N00001"
    rows, _ = build_mapping(registry, [], [], [], [], verify_cache=False)
    assert rows[0]["network_id"] == ""
    assert rows[0]["mapping_status"] == "missing_metadata"


def test_explicit_conflicting_sources_preserved():
    registry, metadata = fixture_rows()
    metadata[0]["external_details"] = ["https://www.kegg.jp/entry/N00002"]
    rows, _ = build_mapping(registry, metadata, [], [], [], verify_cache=False)
    assert {r["network_id"] for r in rows} == {"N00001", "N00002"}
    assert {r["mapping_status"] for r in rows} == {"ambiguous_network"}


def test_reviewed_override_requires_exact_accession_url_and_evidence():
    registry, _ = fixture_rows()
    overrides = [{"source_name": "EXAMPLE", "network_id": "N00001",
                  "source_url": "https://www.kegg.jp/entry/N00001",
                  "evidence": "Reviewed mechanism and variant"}]
    rows, _ = build_mapping(registry, [], [], [], [], overrides, False)
    assert rows[0]["mapping_status"] == "mapped"
    overrides[0]["evidence"] = ""
    rows, _ = build_mapping(registry, [], [], [], [], overrides, False)
    assert rows[0]["mapping_status"] == "invalid_metadata"


def test_raw_cache_sha256_and_exact_source_are_verified(tmp_path):
    registry, metadata = fixture_rows()
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"EXAMPLE": {"exactSource": "N00001", "systematicName": "M123"}}))
    metadata[0]["path"] = str(path)
    metadata[0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, _ = build_mapping(registry, metadata, [], [], [])
    assert rows[0]["mapping_status"] == "mapped"
    path.write_text(json.dumps({"EXAMPLE": {"exactSource": "N00002", "systematicName": "M123"}}))
    rows, _ = build_mapping(registry, metadata, [], [], [])
    assert all(r["mapping_status"] == "ambiguous_network" for r in rows)
    assert all("metadata_checksum_mismatch" in r["mapping_issue"] for r in rows)


def test_missing_raw_cache_blocks_automatic_mapping():
    registry, metadata = fixture_rows()
    rows, _ = build_mapping(registry, metadata, [], [], [])
    assert rows[0]["mapping_status"] == "invalid_metadata"


def test_ambiguous_canonical_is_not_arbitrarily_selected():
    registry, metadata = fixture_rows()
    bindings = [{"source_target_id": registry[0]["source_target_id"], "canonical_id": cid}
                for cid in ("a", "b")]
    rows, _ = build_mapping(registry, metadata, bindings, [], [], verify_cache=False)
    assert {r["canonical_id"] for r in rows} == {"a", "b"}
    assert {r["mapping_status"] for r in rows} == {"ambiguous_canonical"}


def test_redirect_cycle_is_rejected():
    registry, metadata = fixture_rows()
    redirects = [{"previous_internal_id": "old", "canonical_id": "next"},
                 {"previous_internal_id": "next", "canonical_id": "old"}]
    rows, _ = build_mapping(registry, metadata, [], [], redirects, verify_cache=False)
    assert rows[0]["mapping_status"] == "unresolved_canonical"


def test_duplicate_source_definitions_are_rejected():
    registry, metadata = fixture_rows()
    with pytest.raises(ValueError, match="duplicate_source_target_id"):
        build_mapping(registry * 2, metadata, [], [], [], verify_cache=False)



def test_multiple_source_definitions_remain_traceable():
    registry, metadata = fixture_rows()
    registry.append(dict(registry[0], source_target_id="other_source"))
    rows, summary = build_mapping(registry, metadata, [], [], [], verify_cache=False)
    assert len(rows) == 2
    assert summary["source_definition_count"] == 2
    assert summary["mapped_distinct_network_ids"] == 1


def test_override_cannot_silently_replace_exact_source():
    registry, metadata = fixture_rows()
    overrides = [{"source_name": "EXAMPLE", "network_id": "N00002",
                  "source_url": "https://www.kegg.jp/entry/N00002",
                  "evidence": "Reviewed mechanism"}]
    rows, _ = build_mapping(registry, metadata, [], [], [], overrides, False)
    assert {r["mapping_status"] for r in rows} == {"ambiguous_network"}
    assert {r["network_id"] for r in rows} == {"N00001", "N00002"}


def test_source_accession_substring_is_not_valid():
    registry, metadata = fixture_rows()
    metadata[0]["exact_source"] = "N00001foo"
    metadata[0]["external_details"] = []
    rows, _ = build_mapping(registry, metadata, [], [], [], verify_cache=False)
    assert rows[0]["mapping_status"] == "unmapped_network"


def test_official_archive_exact_name_restores_missing_metadata(tmp_path):
    from cla_engine.msigdb_network import read_msigdb_archive
    registry, _ = fixture_rows()
    registry[0]["source_name"] = "KEGG_MEDICUS_EXAMPLE"
    path = tmp_path / "release.json"
    path.write_text(json.dumps({"KEGG_MEDICUS_EXAMPLE": {
        "exactSource": "N00001", "systematicName": "M123",
        "externalDetailsURL": "https://www.kegg.jp/entry/N00001",
        "geneSymbols": ["A"]}}))
    archive = read_msigdb_archive(path, "https://data.broadinstitute.org/release.json", "2025.1.Hs")
    rows, _ = build_mapping(registry, archive, [], [], [])
    assert rows[0]["mapping_status"] == "mapped"
    assert rows[0]["metadata_archive_version"] == "2025.1.Hs"


def test_archive_membership_difference_is_reported_without_replacing_input():
    from cla_engine.msigdb_network import audit_archive_membership
    registry, _ = fixture_rows()
    source_genes = {registry[0]["source_target_id"]: ["ENSG_OLD"]}
    archive = [{"source_name": "EXAMPLE", "members": ["A", "UNKNOWN"],
                "archive_version": "2025.1.Hs"}]
    hgnc = [{"symbol": "A", "status": "Approved", "ensembl_gene_id": "ENSG_NEW"}]
    rows = audit_archive_membership(registry, archive, source_genes, hgnc, [])
    assert rows[0]["member_comparison_status"] == "difference_or_mapping_gap"
    assert rows[0]["unmapped_archive_symbols"] == "UNKNOWN"
    assert rows[0]["original_only_ensembl"] == "ENSG_OLD"
    assert rows[0]["archive_only_ensembl"] == "ENSG_NEW"
    assert source_genes == {registry[0]["source_target_id"]: ["ENSG_OLD"]}
