import numpy as np
import pandas as pd
import pytest
import json

from cc_hhgt.v32.drug_identity import DrugIdentityResolver, model_context, name_key
from cc_hhgt.v32.drug_training import _balanced_indices, normalise_cell_line_map, DrugTrainingError
from cc_hhgt.v32.drug_training_streaming import _BalancedReservoir


def resolver():
    return DrugIdentityResolver(pd.DataFrame([
        ("DC:1", "name", "compound-alpha", "curated-v1"),
        ("DC:1", "name", "brand alpha", "curated-v1"),
        ("DC:1", "source:GDSC1_AUC", "17", "curated-v1"),
        ("DC:1", "source:DRUGCENTRAL", "1", "native-v1"),
        ("DC:1", "name", "shared-name", "curated-v1"),
        ("DC:2", "name", "shared-name", "curated-v1"),
        ("DC:2", "name", "compound-beta", "curated-v1"),
        ("DC:2", "chembl", "CHEMBL2", "curated-v1"),
    ], columns=["drug_id", "identifier_type", "identifier_value", "source"]))


def test_stable_id_and_explicit_synonym_join_without_punctuation_guessing():
    r = resolver()
    assert r.resolve("GDSC1_AUC", "17", "new label")["drug_id"] == "DC:1"
    assert r.resolve("PRISM_24Q2", "BRD-x", " BRAND   ALPHA ")["drug_id"] == "DC:1"
    assert r.resolve("PRISM_24Q2", "BRD-y", "compoundalpha")["identity_available"] is False
    assert name_key("alpha-beta") != name_key("alpha beta")


def test_ambiguous_or_conflicting_identifiers_keep_separate_native_assays():
    r = resolver()
    first = r.resolve("PRISM", "a", "shared-name")
    second = r.resolve("PRISM", "b", "shared-name")
    assert not first["identity_available"] and first["drug_id"] != second["drug_id"]
    assert first["identity_unavailable_reason"] == "AMBIGUOUS_IDENTIFIER_OR_SYNONYM"
    conflict = r.resolve("GDSC1_AUC", "17", "", identifiers={"chembl": "CHEMBL2"})
    assert conflict["identity_unavailable_reason"] == "CONFLICTING_STABLE_IDENTIFIERS"
    assert r.resolve("GDSC1_AUC", "17", "compound-beta")["identity_unavailable_reason"] == "STABLE_IDENTIFIER_NAME_CONFLICT"


def test_crosswalk_requires_provenance_and_namespaced_id():
    with pytest.raises(ValueError):
        DrugIdentityResolver(pd.DataFrame({"drug_id": ["naked_name"], "identifier_type": ["name"],
                                           "identifier_value": ["a"], "source": [""]}))


def model_fixture():
    models = pd.DataFrame({"canonical_model_id": ["SIDM1", "SIDM2", "SIDM3"],
        "depmap_id": ["ACH-1", "ACH-2", "ACH-3"], "tissue": ["Lung", "Kidney", "Breast"]})
    gdsc = [dict(dataset_id="GDSC1_AUC", cell_line_id="SIDM1", tcga_desc="LUAD")]
    return models, gdsc


def test_explicit_cancer_propagates_across_assays_without_tissue_broadcast():
    models, gdsc = model_fixture()
    annotations = pd.DataFrame({"canonical_model_id": ["SIDM3"], "cancer_id": ["BRCA"],
        "source": ["molecular-metadata-v1"], "subtype_system": ["PAM50"], "subtype_id": ["Basal"]})
    mapping, audit, subtype = model_context(models, gdsc, ["ACH-1", "ACH-2", "ACH-3"],
        tissue_cancers={"Lung": ("LUAD", "LUSC"), "Kidney": ("KICH", "KIRC", "KIRP"), "Breast": ("BRCA",)},
        cancer_universe=["LUAD", "LUSC", "KICH", "KIRC", "KIRP", "BRCA"], annotations=annotations)
    assert set(mapping.loc[mapping.canonical_model_id.eq("SIDM1"), "cancer_id"]) == {"LUAD"}
    assert not audit.loc[audit.canonical_model_id.eq("SIDM2"), "context_available"].any()
    assert subtype.subtype_id.tolist() == ["Basal"]
    assert subtype.source.tolist() == ["molecular-metadata-v1"]
    assert mapping.groupby("canonical_model_id").cancer_id.nunique().max() == 1


def test_conflicting_native_cancer_and_model_annotations_are_unavailable():
    models, gdsc = model_fixture()
    annotations = pd.DataFrame({"canonical_model_id": ["SIDM1"], "cancer_id": ["LUSC"], "source": ["registry"]})
    _, audit, _ = model_context(models, gdsc, ["ACH-1"], tissue_cancers={},
                               cancer_universe=["LUAD", "LUSC"], annotations=annotations)
    assert not audit.context_available.any()
    assert audit.context_unavailable_reason.eq("CONFLICTING_EXPLICIT_CANCER_ANNOTATIONS").all()


def test_training_rejects_old_multicancer_model_expansion():
    mapping = pd.DataFrame({"dataset_id": "GDSC", "cell_line_id": "SIDM1", "canonical_model_id": "SIDM1",
                            "cancer_id": ["LUAD", "LUSC"]})
    with pytest.raises(DrugTrainingError, match="several cancer"):
        normalise_cell_line_map(mapping)


def test_uniform_sampling_preserves_rare_class_and_is_chunk_invariant():
    labels = np.r_[np.zeros(97000), np.ones(3000)].astype(np.float32)
    ids = np.arange(len(labels), dtype=np.float32).reshape(-1, 1)
    a = _BalancedReservoir(10000, 17); b = _BalancedReservoir(10000, 17)
    a.offer(ids, ids, labels)
    for start in range(0, len(labels), 1337):
        b.offer(ids[start:start+1337], ids[start:start+1337], labels[start:start+1337])
    for left, right in zip(a.arrays(1, 1), b.arrays(1, 1)): np.testing.assert_array_equal(left, right)
    assert .02 < a.arrays(1, 1)[2].mean() < .04
    assert a.audit()["seen_counts"] == {0: 97000, 1: 3000}
    sampled = _balanced_indices(labels, 10000, 17)
    assert .02 < labels[sampled].mean() < .04
    assert len(np.unique(sampled)) == 10000


def test_sampler_does_not_turn_unknown_labels_into_negatives():
    r = _BalancedReservoir(20, 1)
    r.offer(np.ones((4, 2)), np.ones((4, 1)), np.array([1., 0., np.nan, 2.]))
    assert sorted(r.arrays(2, 1)[2]) == [0., 1.]
    assert r.audit()["seen_counts"] == {0: 1, 1: 1}


def test_native_readers_use_identity_and_keep_targetless_or_unnamed_assays(tmp_path):
    from cc_hhgt.v32.drug_staging import _stage_gdsc, _stage_prism, _stage_drug_targets, DrugStagingConfig
    r = resolver(); config = DrugStagingConfig(max_models=5)
    gdsc = pd.DataFrame({"DATASET": ["GDSC1"]*3, "SANGER_MODEL_ID": ["SIDM1"]*3,
        "TCGA_DESC": ["LUAD"]*3, "DRUG_ID": ["17", "99", "100"],
        "DRUG_NAME": ["brand alpha", "novel compound", None], "AUC": [.1, .2, .3]})
    gdsc.to_csv(tmp_path/"gdsc.csv", index=False)
    _, source, _ = _stage_gdsc([tmp_path/"gdsc.csv"], tmp_path/"response", config, r)
    response = pd.read_parquet(tmp_path/"response/gdsc_auc.parquet")
    assert len(response) == 3 and response.drug_id.iloc[0] == "DC:1"
    assert source.identity_available.sum() == 1
    pd.DataFrame({"IDs": ["B1"], "Drug.Name": ["compound-alpha"]}).to_csv(tmp_path/"compounds.csv", index=False)
    pd.DataFrame({"compound": ["B1", "NO_NAME"], "ACH-1": [.1, .5]}).to_csv(tmp_path/"matrix.csv", index=False)
    _stage_prism(tmp_path/"matrix.csv", tmp_path/"compounds.csv", tmp_path/"response", config, r)
    prism = pd.read_parquet(tmp_path/"response/prism_24q2.parquet")
    assert len(prism) == 2 and prism.drug_id.iloc[0] == "DC:1" and prism.drug_id.iloc[1].startswith("ASSAY:")
    pd.DataFrame({"DRUG_NAME": ["compound-alpha"], "STRUCT_ID": ["1"], "GENE": ["G1"],
                  "ORGANISM": ["Homo sapiens"]}).to_csv(tmp_path/"target.tsv", sep="\t", index=False)
    pd.DataFrame({"symbol": ["G1"], "alias_symbol": [""], "prev_symbol": [""],
                  "ensembl_gene_id": ["ENSG00000000001"]}).to_csv(tmp_path/"hgnc.tsv", sep="\t", index=False)
    targets, _ = _stage_drug_targets(tmp_path/"target.tsv", tmp_path/"hgnc.tsv", tmp_path/"targets.parquet", r)
    assert targets.drug_id.tolist() == ["DC:1"]
    assert response.loc[response.drug_id.isin(targets.drug_id)].source_drug_id.tolist() == ["17"]


def test_full_native_staging_writes_corrected_context_and_target_mapping(tmp_path):
    from cc_hhgt.v32.drug_staging import run_native_drug_staging, DrugStagingConfig, TCGA_CANCERS
    paths = {}
    def table(name, frame, sep=","):
        path = tmp_path/(name+(".tsv" if sep == "\t" else ".csv")); frame.to_csv(path, sep=sep, index=False)
        paths[name] = path
    models = [f"SIDM{i}" for i in range(5)]
    for dataset in ["gdsc1", "gdsc2"]:
        table(dataset, pd.DataFrame({"DATASET": dataset.upper(), "SANGER_MODEL_ID": models,
            "TCGA_DESC": "BRCA", "DRUG_ID": "17", "DRUG_NAME": "brand alpha", "AUC": np.arange(5)/10}))
    table("prism_compounds", pd.DataFrame({"IDs": ["B1"], "Drug.Name": ["compound-alpha"]}))
    table("prism_matrix", pd.DataFrame({"compound": ["B1", "unknown"],
                                        **{f"ACH-{i}": [.1+i/10, .5] for i in range(5)}}))
    models_json = {"data": [], "included": [
        {"type": "samples", "id": "sample", "relationships": {"tissue": {"data": {"type": "tissues", "id": "breast"}}}},
        {"type": "tissues", "id": "breast", "attributes": {"name": "Breast"}}]}
    for i, model in enumerate(models):
        models_json["data"].append({"id": model, "attributes": {"names": [model]}, "relationships": {
            "sample": {"data": {"type": "samples", "id": "sample"}},
            "identifiers": {"data": [{"type": "identifiers", "id": str(i)}]}}})
        models_json["included"].append({"type": "identifiers", "id": str(i), "attributes": {"identifier": f"ACH-{i}"}})
    paths["models"] = tmp_path/"models.json"; paths["models"].write_text(json.dumps(models_json))
    paths["expression"] = tmp_path/"expression.csv"
    paths["expression"].write_text("model_id,Unnamed: 1,"+",".join(models)+"\n"+
        "metadata,,, , , ,\n"*3+"lnc,ENSG00000000002,1,2,3,4,5\n")
    table("targets", pd.DataFrame({"DRUG_NAME": ["compound-alpha"], "STRUCT_ID": ["1"], "GENE": ["G1"], "ORGANISM": ["Homo sapiens"]}), "\t")
    table("hgnc", pd.DataFrame({"symbol": ["G1"], "alias_symbol": [""], "prev_symbol": [""], "ensembl_gene_id": ["ENSG00000000001"]}), "\t")
    identifiers = pd.DataFrame([("DC:1", "name", name, "curated-v1") for name in ["brand alpha", "compound-alpha"]],
        columns=["drug_id", "identifier_type", "identifier_value", "source"])
    table("identifiers", identifiers, "\t")
    table("annotations", pd.DataFrame({"canonical_model_id": ["SIDM0"], "cancer_id": ["BRCA"],
        "subtype_system": ["PAM50"], "subtype_id": ["Basal"], "source": ["registry"]}), "\t")
    exact = tmp_path/"exact.parquet"
    pd.DataFrame({"cancer_id": TCGA_CANCERS, "lncrna_id": "LNC:ENSG00000000002", "pathway_id": "P1"}).to_parquet(exact, index=False)
    membership = tmp_path/"membership.parquet"
    pd.DataFrame({"pathway_id": ["P1"], "gene_id": ["ENSG00000000001"]}).to_parquet(membership, index=False)
    output = tmp_path/"staging"
    result = run_native_drug_staging(gdsc1_path=paths["gdsc1"], gdsc2_path=paths["gdsc2"],
        prism_matrix_path=paths["prism_matrix"], prism_compound_path=paths["prism_compounds"],
        cmp_expression_path=paths["expression"], cmp_models_json_path=paths["models"],
        drugcentral_target_path=paths["targets"], hgnc_path=paths["hgnc"],
        exact_candidates_path=exact, pathway_membership_path=membership,
        drug_identity_path=paths["identifiers"], model_annotation_path=paths["annotations"],
        output_root=output, staging_run_id="v32-native-synthetic", config=DrugStagingConfig(max_models=5))
    assert result["status"] == "SUCCESS" and not result["formal"]
    mapping = pd.read_parquet(output/"cell_line_map.parquet")
    assert set(mapping.cancer_id) == {"BRCA"} and len(mapping) == 15
    assert len(pd.read_parquet(output/"MODEL_SUBTYPES.parquet")) == 1
    source = pd.read_parquet(output/"source_drug_map.parquet")
    assert set(source.loc[source.identity_available, "drug_id"]) == {"DC:1"}
    response = pd.read_parquet(output/"raw_cell_line_drug_response")
    assert response.drug_id.str.startswith("ASSAY:").sum() == 5
    manifest = json.loads((output/"STAGING_MANIFEST.json").read_text())
    assert manifest["native_assayed_to_target_mapping_coverage"]["synonym_crosswalk_applied"]
    assert manifest["cancer_mapping_policy"] == "ONE_UNAMBIGUOUS_CANCER_PER_CANONICAL_MODEL"
