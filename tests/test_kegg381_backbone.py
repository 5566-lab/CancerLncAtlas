import copy

import pytest

from cla_engine.kegg381_backbone import build_backbone, load_backbone, materialize_backbone


def data():
    source_ids = ["src_reference", "src_variant"]
    mapping = [
        {"source_target_id": sid, "source_name": sid, "systematic_id": "M" + str(i),
         "network_id": "N0000" + str(i), "canonical_id": "CLA-" + str(i),
         "mapping_status": "mapped"}
        for i, sid in enumerate(source_ids, start=1)]
    scope = [
        dict(row, network_type=kind, admitted_by_network_scope=supported,
             cancer_scope_status=status)
        for row, kind, supported, status in zip(
            mapping, ["Reference", "Variant"], ["False", "True"],
            ["outside_selected_network_categories", "direct_cancer_network"])]
    # Deliberately identical member sets must not collapse reference and variant.
    genes = {sid: ["ENSG00000000001", "ENSG00000000002"] for sid in source_ids}
    catalog = [{"canonical_id": row["canonical_id"], "display_name": row["source_name"]} for row in mapping]
    return source_ids, mapping, scope, genes, catalog


def test_fixed_list_preserves_outside_scope_and_reference_variant():
    inputs = data()
    rows, genes, summary = build_backbone(*inputs, expected_count=2)
    assert len(rows) == 2
    assert {r["network_type"] for r in rows} == {"Reference", "Variant"}
    assert summary["official_cancer_supported_count"] == 1
    assert summary["official_cancer_annotation_is_filter"] is False
    assert genes == inputs[3]


def test_production_count_is_381():
    with pytest.raises(ValueError, match="unexpected_backbone_count"):
        build_backbone(*data())


@pytest.mark.parametrize("field", ["network_id", "canonical_id"])
def test_duplicate_id_cannot_collapse_two_original_definitions(field):
    inputs = list(data())
    inputs[1][1][field] = inputs[1][0][field]
    inputs[2][1][field] = inputs[1][0][field]
    with pytest.raises(ValueError, match="duplicate_" + field):
        build_backbone(*inputs, expected_count=2)


def test_unmapped_and_conflicting_source_identity_are_rejected():
    inputs = list(data())
    inputs[1][0]["mapping_status"] = "ambiguous_network"
    with pytest.raises(ValueError, match="unverified_source_mapping"):
        build_backbone(*inputs, expected_count=2)
    inputs = list(data())
    inputs[2][0]["network_id"] = "N99999"
    with pytest.raises(ValueError, match="scope_identity_conflict"):
        build_backbone(*inputs, expected_count=2)


def test_current_canonical_must_exist():
    inputs = list(data())
    inputs[4] = inputs[4][1:]
    with pytest.raises(ValueError, match="canonical_absent"):
        build_backbone(*inputs, expected_count=2)


def test_member_order_is_preserved_and_empty_sets_are_rejected():
    inputs = list(data())
    inputs[3]["src_reference"].reverse()
    rows, genes, summary = build_backbone(*inputs, expected_count=2)
    assert genes["src_reference"] == ["ENSG00000000002", "ENSG00000000001"]
    inputs[3]["src_reference"] = []
    with pytest.raises(ValueError, match="missing_original_members"):
        build_backbone(*inputs, expected_count=2)


def test_freeze_readback_and_checksum_tampering(tmp_path):
    rows, genes, summary = build_backbone(*data(), expected_count=2)
    materialize_backbone(tmp_path, rows, genes, summary)
    loaded = load_backbone(tmp_path)
    assert loaded["gene_sets"] == genes
    assert len(loaded["rows"]) == 2
    with (tmp_path / "GENE_SETS.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="backbone_checksum_mismatch"):
        load_backbone(tmp_path)


def test_member_hash_validation_without_file_checksums(tmp_path):
    rows, genes, summary = build_backbone(*data(), expected_count=2)
    altered = copy.deepcopy(genes)
    altered["src_reference"].reverse()
    materialize_backbone(tmp_path, rows, altered, summary)
    with pytest.raises(ValueError, match="backbone_members_mismatch"):
        load_backbone(tmp_path, verify=False)
