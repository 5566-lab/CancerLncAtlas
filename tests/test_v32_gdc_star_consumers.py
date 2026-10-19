"""Actual frozen core and external evaluation consume GDC project ancestry."""
from test_v32_gdc_star_counts import assembled, no_fit
import pytest


def test_project_overlap_does_not_invent_independent_rule_evidence():
    import json
    from cc_hhgt.v32.external_validation_sources import _tokens
    from cc_hhgt.v32.evidence_rule_summary import provenance_tokens
    source = {"source_dataset": "TCGA-BRCA"}
    assert json.loads(_tokens(source)) == ["GDC_PROJECT:TCGA-BRCA"]
    assert provenance_tokens(source) == []


@pytest.mark.parametrize("kind", ["case", "sample", "aliquot"])
def test_conflicting_gdc_identity_across_files_is_rejected(tmp_path, kind):
    from pathlib import Path
    from test_v32_gdc_star_counts import inputs
    from cc_hhgt.v32.gdc_star_counts import replay_pages
    from cc_hhgt.v32.formal_expression_sources import read, write, record
    manifest, _, _ = inputs(tmp_path / "sources", n=2)
    value = read(manifest); page = Path(value["pages"][0]["path"]); content = read(page)
    first, second = content["response"]["data"]["hits"]
    if kind == "case":
        second["cases"][0]["case_id"] = first["cases"][0]["case_id"]
        second["associated_entities"][0]["case_id"] = first["cases"][0]["case_id"]
    elif kind == "sample":
        second["cases"][0]["samples"][0]["sample_id"] = first["cases"][0]["samples"][0]["sample_id"]
    else:
        shared = first["associated_entities"][0]["entity_id"]
        second["associated_entities"][0]["entity_id"] = shared
        second["cases"][0]["samples"][0]["portions"][0]["analytes"][0]["aliquots"][0]["aliquot_id"] = shared
    write(page, content); value["pages"] = [record(page)]
    with pytest.raises(ValueError, match="coverage|identity mapping"): replay_pages(value, manifest.parent)


def test_actual_frozen_core_external_consumer_excludes_same_gdc_project(assembled, tmp_path, monkeypatch):
    from test_v32_local_cnv_source_ancestry import native_inputs, test_actual_core_and_external_consumer_exclude_studies_only_in_transformation_inputs
    root, _, _, _ = assembled
    def native_factory(folder, patcher):
        return native_inputs(folder, patcher, fold_expression_root=root / "converted", patient_map_source=root / "patients.tsv")
    test_actual_core_and_external_consumer_exclude_studies_only_in_transformation_inputs(
        tmp_path / "downstream", monkeypatch, native_factory=native_factory,
        external_pmid="", external_study_metadata={"source_dataset": "TCGA-BRCA"})
