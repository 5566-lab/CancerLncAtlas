"""Actual chemical encoding and existing head paths, with all fitting forbidden."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import drug_training as base
from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.drug_pharmacology import (FEATURES, STATIC_FEATURES, normalize_targets, structure_table,
    normalize_design, PharmacologyInputs, load_pharmacology, concentration)


def record(path): return {"path":str(path), "sha256":sha256(path)}


@pytest.fixture(autouse=True)
def forbid_optimization(monkeypatch):
    def forbidden(*a, **kw): raise AssertionError("No optimizer or fitting in pharmacology verification")
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    monkeypatch.setattr(base, "_fit_head", forbidden)


def feature_sources(root, authority, raw_response):
    root.mkdir(parents=True, exist_ok=True)
    annotation = authority[["drug_id", "gene_id"]].drop_duplicates().copy()
    annotation = annotation.assign(action_type="INHIBITOR", affinity_type="Ki", affinity_value=10.,
        affinity_unit="nM", affinity_relation="=", source_id="synthetic_native_target")
    drugs = sorted(raw_response.drug_id.astype(str).unique())
    structures = pd.DataFrame({"drug_id":drugs, "smiles":["CCO" if i % 2 else "CC(=O)O" for i in range(len(drugs))],
                               "source_id":"synthetic_native_structure"})
    design = raw_response[["dataset_id", "cell_line_id", "drug_id"]].drop_duplicates().copy()
    design = design.assign(assay_id="protocol-1", dose_value=10., dose_unit="nM", exposure_hours=72., source_id="synthetic_design")
    records = {}
    for name, frame in {"target_annotations":annotation, "structures":structures, "assay_conditions":design}.items():
        path = root/(name+".parquet"); frame.to_parquet(path, index=False); records[name] = record(path)
    path = root/"INPUT.json"; path.write_text(json.dumps({"format":"DRUG_PHARMACOLOGY_INPUT_V1", **records}))
    return path


def simple_inputs():
    authority = pd.DataFrame({"drug_id":["D:1", "D:1"], "gene_id":["G1", "G2"]})
    annotation = authority.assign(action_type=["INHIBITOR", "AGONIST"], affinity_type=["Ki", "Kd"],
        affinity_value=[10., 1000.], affinity_unit="nM", affinity_relation="=", source_id="synthetic")
    structures = pd.DataFrame({"drug_id":["D:1"], "smiles":["CCO"], "source_id":["synthetic"]})
    response = pd.DataFrame({"dataset_id":"S", "cell_line_id":["C1", "C2"], "drug_id":"D:1"})
    mapping = response[["dataset_id", "cell_line_id"]].assign(canonical_model_id=["M1", "M2"], cancer_id="BRCA")
    design = response.assign(assay_id="protocol", dose_value=[1., 1000.], dose_unit="nM", exposure_hours=[24., 72.], source_id="synthetic")
    return authority, annotation, structures, response, mapping, design


def test_endpoint_separation_unit_conversion_and_censoring_are_not_affinity_imputation():
    authority, annotation, *_ = simple_inputs()
    extra = annotation.iloc[[0]].assign(affinity_type="IC50", affinity_value=100., affinity_relation="<")
    frame = normalize_targets(pd.concat([annotation, extra], ignore_index=True), authority)
    assert frame.p_value_molar.tolist() == pytest.approx([8., 6., 7.])
    assert frame.p_relation.tolist() == ["=", "=", ">"]
    inputs = PharmacologyInputs(frame, structure_table(simple_inputs()[2], ["D:1"]), pd.DataFrame(), authority)
    row = inputs._static.iloc[0]
    assert row.ki_exact_p_median == 8 and row.kd_exact_p_median == 6
    assert pd.isna(row.ic50_exact_p_median) and not row.ic50_exact_p_available
    assert row.ic50_censored_target_fraction == .5
    assert row.action_inhibitor_target_fraction == row.action_agonist_target_fraction == .5
    # Repeating the same database row cannot change an endpoint summary.
    repeat = normalize_targets(pd.concat([annotation]*8, ignore_index=True), authority)
    assert len(repeat) == 2
    p = annotation.iloc[[0]].assign(affinity_value=0., affinity_unit="p_molar")
    assert normalize_targets(p, authority).p_value_molar.iloc[0] == 0


@pytest.mark.parametrize("fault", ["unit", "zero", "missing_unit", "endpoint", "relation", "target"])
def test_incomplete_or_incompatible_target_measurements_fail(fault):
    authority, annotation, *_ = simple_inputs()
    if fault == "unit": annotation.loc[0,"affinity_unit"] = "mg/kg"
    if fault == "zero": annotation.loc[0,"affinity_value"] = 0
    if fault == "missing_unit": annotation.loc[0,"affinity_unit"] = None
    if fault == "endpoint": annotation.loc[0,"affinity_type"] = "PercentInhibition"
    if fault == "relation": annotation.loc[0,"affinity_relation"] = "approximately"
    if fault == "target": annotation.loc[0,"gene_id"] = "invented"
    with pytest.raises(ValueError): normalize_targets(annotation, authority)


def test_actual_rdkit_structure_equivalence_conflict_and_missing_masks():
    frame = pd.DataFrame({"drug_id":["D:1", "D:1", "D:2", "D:2", "D:3"],
        "smiles":["CCO", "OCC", "CCO", "CO", "INVALID!!"], "source_id":"synthetic"})
    result = structure_table(frame, ["D:1", "D:2", "D:3", "D:4"]).set_index("drug_id")
    assert result.loc["D:1", "structure_available"] and result.loc["D:1", "molecular_weight"] == pytest.approx(46.069, abs=.001)
    assert result.loc["D:1", list(STATIC_FEATURES[-256:])].sum()>0
    assert result.loc["D:2", "structure_unavailable_reason"] == "CONFLICTING_STRUCTURES"
    assert result.loc["D:3", "structure_unavailable_reason"] == "INVALID_SMILES"
    assert result.loc["D:4", "structure_unavailable_reason"] == "NO_STRUCTURE_SOURCE"
    assert result.loc[["D:2", "D:3", "D:4"], "molecular_weight"].isna().all()


def test_fold_local_dose_design_does_not_read_heldout_models_or_response_values():
    authority, annotation, structures, response, mapping, design = simple_inputs()
    inputs = PharmacologyInputs(normalize_targets(annotation, authority), structure_table(structures, ["D:1"]),
        normalize_design(design, response, mapping), authority)
    query = pd.DataFrame({"cancer_id":["BRCA", "BRCA"], "lncrna_id":["L1", "L2"], "drug_id":"D:1"})
    train = inputs.table(query, ["M1"]); test = inputs.table(query, ["M2"])
    assert train.log10_molar_dose_median.eq(-9).all() and test.log10_molar_dose_median.eq(-6).all()
    assert train.log1p_exposure_hours_mean.eq(np.log1p(24)).all()
    absent = inputs.table(query, [])
    assert absent.log10_molar_dose_median.isna().all() and not absent.dose_available.any()
    assert np.isfinite(inputs.encode(query, [])).all()
    design["response_value"] = [0., 1.]
    with pytest.raises(ValueError, match="response-derived"): normalize_design(design, response, mapping)


def test_vehicle_zero_and_zero_exposure_remain_observed_not_missing():
    authority, annotation, structures, response, mapping, design = simple_inputs()
    design.loc[0, ["dose_value", "exposure_hours"]] = 0
    inputs = PharmacologyInputs(normalize_targets(annotation, authority), structure_table(structures, ["D:1"]),
        normalize_design(design, response, mapping), authority)
    context = inputs.design_table(["M1"]).iloc[0]
    assert context.dose_available and not context.positive_dose_available
    assert context.vehicle_fraction == 1 and pd.isna(context.log10_molar_dose_median)
    assert context.exposure_available and context.log1p_exposure_hours_mean == 0


@pytest.mark.parametrize("fault", ["unknown_identity", "dose_unit", "negative_time", "conflicting_assay"])
def test_assay_protocol_identity_and_measurement_constraints(fault):
    _, _, _, response, mapping, design = simple_inputs()
    if fault == "unknown_identity": design.loc[0,"cell_line_id"] = "not_observed"
    if fault == "dose_unit": design.loc[0,"dose_unit"] = "unknown"
    if fault == "negative_time": design.loc[0,"exposure_hours"] = -1
    if fault == "conflicting_assay": design = pd.concat([design, design.iloc[[0]].assign(dose_value=5.)], ignore_index=True)
    with pytest.raises(ValueError): normalize_design(design, response, mapping)
    assert concentration(0., "nM", allow_zero=True) == 0


def test_frozen_forward_and_target_ablation_consume_the_new_features():
    from test_v32_drug_attribution import fixture
    from cc_hhgt.v32.drug_attribution import ablate_drug_targets
    core, queries, domain, targets, exact, membership = fixture()
    authority, annotation, structures, response, mapping, design = simple_inputs()
    for table in (authority, annotation, structures, response, design): table["drug_id"] = "DC:1"
    inputs = PharmacologyInputs(normalize_targets(annotation, authority), structure_table(structures, ["DC:1"]),
        normalize_design(design, response, mapping), authority)
    features = inputs.encode(queries, ["M1"], core=core)
    columns = [FEATURES.index("pharm_"+name) for name in
        ("action_inhibitor_target_fraction", "ki_exact_p_median", "molecular_weight", "log10_molar_dose_median")]
    class FrozenHead(torch.nn.Module):
        def __init__(self): super().__init__(); self.coefficients = torch.nn.Parameter(torch.ones(4))
        def forward(self, core, domain):
            assert not torch.is_grad_enabled()
            return (domain[:,[6+i for i in columns]]*self.coefficients).sum(1, keepdim=True)/100
    head = FrozenHead(); width = 6+len(FEATURES)
    full = np.concatenate([domain, features], axis=1)
    prediction = base._predict_head(head, np.zeros((1,6),np.float32), full, np.zeros(width,np.float32), np.ones(width,np.float32), 1)
    for i in columns:
        changed = full.copy(); changed[0,6+i] += 1
        altered = base._predict_head(head, np.zeros((1,6),np.float32), changed, np.zeros(width,np.float32), np.ones(width,np.float32), 1)
        assert altered[0] != prediction[0]
    baseline, effects, audit = ablate_drug_targets(head, core, queries, full, np.array([True]), targets, exact, membership,
        domain_mean=np.zeros(width), domain_scale=np.ones(width), pharmacology=inputs, models=["M1"])
    assert baseline.baseline_probability.iloc[0] == pytest.approx(prediction[0])
    assert effects.target_pharmacology_recomputed.all() and not effects.causal_claimed.any()
    assert effects.loc[effects.ablation_entity_id.eq("G1"), "delta_logit"].iloc[0] == pytest.approx((8+.5)/100, abs=1e-6)
    assert audit["parameters_unchanged"] and not audit["optimizer_started"]


def test_target_identity_is_coupled_to_action_and_potency_not_only_drug_level_counts():
    from test_v32_drug_attribution import fixture
    core, queries, _, _, _, _ = fixture()
    authority, annotation, structures, response, mapping, design = simple_inputs()
    for table in (authority, annotation, structures, response, design): table["drug_id"] = "DC:1"
    annotation["affinity_type"] = "Ki"
    def pack(annotation):
        return PharmacologyInputs(normalize_targets(annotation, authority), structure_table(structures, ["DC:1"]),
            normalize_design(design, response, mapping), authority)
    original = pack(annotation)
    swapped = annotation.copy(); swapped["action_type"] = list(reversed(annotation.action_type))
    swapped["affinity_value"] = list(reversed(annotation.affinity_value))
    changed = pack(swapped)
    pd.testing.assert_frame_equal(original._static, changed._static)
    first, available = original.target_core(queries, core)
    second, _ = changed.target_core(queries, core)
    from cc_hhgt.v32.drug_pharmacology import TARGET_POOLS
    assert first[0,TARGET_POOLS.index("action_inhibitor")*2] == 1
    assert second[0,TARGET_POOLS.index("action_inhibitor")*2] == 3
    assert first[0,TARGET_POOLS.index("exact_ki_p")*2] == 13
    assert second[0,TARGET_POOLS.index("exact_ki_p")*2] == 15
    assert available[0,TARGET_POOLS.index("exact_ki_p")] == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_actual_existing_training_entry_receives_extended_features_before_fitting(tmp_path, monkeypatch, streaming):
    from test_v32_drug_training import _write_inputs, _write_streaming_formal_fixture, _expected_provenance_hashes
    from cc_hhgt.v32.drug_training_streaming import run_streaming_drug_training, StreamingDrugTrainingConfig
    fixture = _write_streaming_formal_fixture(tmp_path) if streaming else None
    paths = fixture["paths"] if streaming else _write_inputs(tmp_path)
    manifest = feature_sources(tmp_path/"pharmacology", pd.read_parquet(paths["targets"]), pd.read_parquet(paths["response"]))
    class ReachedFit(Exception): pass
    def inspect(train_core, train_domain, train_labels, val_core, val_domain, val_labels, **kwargs):
        from cc_hhgt.v32.drug_pharmacology import TARGET_POOLS
        assert train_core.shape[1] == val_core.shape[1] == (3+len(TARGET_POOLS))*4
        assert train_domain.shape[1] == val_domain.shape[1] == 6+len(FEATURES)
        assert np.isfinite(train_domain).all() and len(train_domain)>0
        assert train_domain[:,6+FEATURES.index("pharm_structure_available")].all()
        assert train_domain[:,6+FEATURES.index("pharm_dose_available")].all()
        raise ReachedFit("Input inspection complete; no fitting executed")
    monkeypatch.setattr(base, "_fit_head", inspect)
    args = dict(exact_candidates_path=fixture["formal_exact"] if streaming else paths["exact"],
        raw_drug_response_path=paths["response"], raw_lncrna_expression_path=paths["expression"],
        cell_line_map_path=paths["mapping"], drug_gene_target_path=paths["targets"], core_embedding_manifest_path=paths["core"],
        pharmacology_manifest_path=manifest, output_root=tmp_path/"stopped_before_fit", training_run_id="v32-pharmacology-synthetic")
    if streaming:
        args.update(factored_drug_candidates_path=fixture["factor_root"], **_expected_provenance_hashes(fixture),
            config=StreamingDrugTrainingConfig(seed=19, min_cell_lines_per_dataset=3, min_total_cell_lines=6, max_train_rows=100,
                max_validation_rows=100, candidate_chunk_rows=32))
    else:
        args.update(config=base.DrugTrainingConfig(seed=19, min_cell_lines_per_dataset=3, min_total_cell_lines=6))
    with pytest.raises(ReachedFit): (run_streaming_drug_training if streaming else base.run_drug_training)(**args)
    assert (args["output_root"]/"PHARMACOLOGY_FEATURE_RECEIPT.json").is_file()
    assert not list(args["output_root"].rglob("private_head.pt"))
    if streaming:
        budget=json.loads((args["output_root"]/"EXECUTION_BUDGET_AUDIT.json").read_text())
        assert budget["additional_pharmacology_domain_features"] == len(FEATURES)


def test_preparation_command_materializes_actual_structures_affinity_and_five_fold_design(tmp_path):
    from test_v32_drug_training import _write_inputs
    from scripts.materialize_v32_drug_pharmacology import main
    paths = _write_inputs(tmp_path)
    manifest = feature_sources(tmp_path/"pharm", pd.read_parquet(paths["targets"]), pd.read_parquet(paths["response"]))
    prep = {"format":"DRUG_PHARMACOLOGY_PREPARATION_V1", "seed":19, "pharmacology_manifest":record(manifest),
        "drug_gene_target":record(paths["targets"]), "raw_drug_response":record(paths["response"]), "cell_line_map":record(paths["mapping"])}
    path=tmp_path/"PREP.json"; path.write_text(json.dumps(prep))
    assert main(["--manifest",str(path),"--output",str(tmp_path/"features")]) == 0
    audit=json.loads((tmp_path/"features/AUDIT.json").read_text())
    static=pd.read_parquet(audit["outputs"]["drug_static_features"]["path"])
    assert static.structure_available.all() and static.ki_exact_p_median.eq(8).all()
    context=pd.read_parquet(audit["outputs"]["assay_design_fold_context"]["path"])
    assert set(context.cell_line_fold)==set(range(5)) and set(context.split)=={"train","validation","test"}
    assert not {"response_value", "canonical_model_id", "cell_line_id"} & set(context)
    assert not audit["training_started"] and not audit["response_values_used_as_features"]
    source=Path(json.loads(manifest.read_text())["structures"]["path"])
    source.write_bytes(source.read_bytes()+b"changed")
    with pytest.raises(ValueError): load_pharmacology(manifest, targets=pd.read_parquet(paths["targets"]),
        response=base.normalise_raw_response(pd.read_parquet(paths["response"])),
        mapping=base.normalise_cell_line_map(pd.read_parquet(paths["mapping"])))


def test_existing_native_staging_keeps_pharmacology_without_guessing_scale(tmp_path):
    from cc_hhgt.v32.drug_staging import _stage_drug_targets
    hgnc = tmp_path/"hgnc.tsv"
    pd.DataFrame({"symbol":["G1"], "alias_symbol":[None], "prev_symbol":[None],
        "ensembl_gene_id":["ENSG00000000001"]}).to_csv(hgnc, sep="\t", index=False)
    native = tmp_path/"native.tsv"
    pd.DataFrame({"DRUG_NAME":["compound"], "GENE":["G1"], "ORGANISM":["Homo sapiens"],
        "ACT_VALUE":[8.], "ACT_TYPE":["Ki"], "ACT_UNIT":["p_molar"], "RELATION":[">"],
        "ACTION_TYPE":["INHIBITOR"], "ACT_SOURCE":["synthetic_source"]}).to_csv(native, sep="\t", index=False)
    frame, audit = _stage_drug_targets(native, hgnc, tmp_path/"staged.parquet")
    assert frame.native_act_value.iloc[0] == "8.0" and frame.native_relation.iloc[0] == ">"
    assert frame.native_action_type.iloc[0] == "INHIBITOR" and frame.native_act_unit.iloc[0] == "p_molar"
    assert not audit["native_activity_units_or_scale_inferred"]
    assert "ACT_SOURCE" in audit["native_pharmacology_fields_preserved"]
