from scripts.match_cptac_gdc_star import bind, index_samples


def file_with_specimens(name, parent_ids):
    aliquot = "aliquot-" + name
    return {
        "file_id": name, "md5sum": "hash-" + name, "data_type": "Gene Expression Quantification",
        "access": "open", "analysis": {"workflow_type": "STAR - Counts"},
        "associated_entities": [{"entity_type": "aliquot", "entity_id": aliquot,
                                 "entity_submitter_id": aliquot}],
        "cases": [{"submitter_id": "C3L", "case_id": "case", "project": {"project_id": "CPTAC-3"},
                   "samples": [{"sample_id": "uuid-" + p, "submitter_id": p, "sample_type": "Primary Tumor",
                                "portions": [{"analytes": [{"aliquots": [{"aliquot_id": aliquot}]}]}]}
                               for p in parent_ids]}],
    }


def test_multi_parent_aliquot_competes_once_and_only_identical_sets_support_strict_pairing():
    files = [file_with_specimens("new", ["11"]), file_with_specimens("original", ["02", "01"])]
    idx = index_samples(files)
    column = {"native_case_id": "C3L", "source_cohort": "UCEC", "tissue_type": "Tumor"}
    key = ("UCEC", "C3L", "Primary Tumor")
    pdc = {key: [(frozenset(["01", "02"]), "study1", "protein"),
                 (frozenset(["01"]), "study2", "phospho")]}
    hit = bind(column, idx, pdc)
    assert hit["candidate_count"] == 2
    assert hit["gdc_file_id"] == "original"
    assert hit["gdc_sample_submitter_id"] == "01|02"
    assert hit["gdc_aliquot_id"] == "aliquot-original"
    assert hit["strict_pdc_protein_match"] == 1 and hit["strict_pdc_phospho_match"] == 0
    partial = bind(column, idx, {key: pdc[key][1:]})
    assert partial["gdc_file_id"] == "original"
    assert partial["strict_pdc_sample_match"] == 0
    tied = bind(column, idx, {})
    assert tied["match_status"] == "AMBIGUOUS" and not tied["gdc_file_id"]
