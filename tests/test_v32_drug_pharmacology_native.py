"""Native producer -> actual encoder checks; no optimizer or real cohort data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import drug_training as base
from cc_hhgt.v32.drug_identity import DrugIdentityResolver
from cc_hhgt.v32.drug_pharmacology import load_pharmacology, normalize_targets
from cc_hhgt.v32.drug_pharmacology_native import activity_rows, native_structures, native_protocols
from cc_hhgt.v32.input_lineage import artifact_sha256


@pytest.fixture(autouse=True)
def forbid_optimization(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("No training in native pharmacology checks")
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    monkeypatch.setattr(base, "_fit_head", forbidden)


def record(path): return {"path":str(path), "sha256":artifact_sha256(path)}


def declaration(**kwargs):
    return {"value_scale":"NEGATIVE_LOG10_MOLAR", "relation_scale":"CONCENTRATION",
            "schema_reference":"synthetic-native-scale-dictionary", **kwargs}


def staged():
    return pd.DataFrame({"drug_id":["DC:1"]*4, "gene_id":["ENSG00000000001"]*4, "native_source_row":range(4),
        "native_action_type":["INHIBITOR", "AGONIST", "BLOCKER", None],
        "native_act_type":["Ki", "IC50", "percent_inhibition", None], "native_act_value":[8., 7., 50., None],
        "native_act_unit":[None]*4, "native_relation":["<", "=", "=", None]})


def test_native_activity_separates_value_scale_relation_scale_and_unavailable_endpoints():
    raw=staged(); authority=raw[["drug_id", "gene_id"]].drop_duplicates()
    frame,audit=activity_rows(raw,authority,declaration(),source_sha="a"*64)
    normalized=normalize_targets(frame,authority)
    assert normalized.p_value_molar.iloc[:2].tolist()==[8.,7.]
    assert normalized.p_relation.iloc[:2].tolist()==[">","="]
    assert normalized.p_value_molar.iloc[2:].isna().all()
    assert frame.action_type.iloc[2]=="BLOCKER" and audit.target_retained.all()
    assert audit.activity_available.tolist()==[True,True,False,False]
    assert audit.reason.iloc[2]=="UNSUPPORTED_ACTIVITY_ENDPOINT"
    other,_=activity_rows(raw,authority,declaration(relation_scale="NEGATIVE_LOG10_MOLAR"),source_sha="a"*64)
    assert normalize_targets(other,authority).p_relation.iloc[0]=="<"
    raw=raw.iloc[[0]].assign(native_act_unit="nM", native_act_value=10.)
    frame,_=activity_rows(raw,authority,declaration(value_scale="CONCENTRATION"),source_sha="a"*64)
    result=normalize_targets(frame,authority).iloc[0]
    assert result.p_value_molar==8 and result.p_relation==">"


@pytest.mark.parametrize("fault",["no_value_scale","no_relation_scale","unit_conflict","p_unit_conflict","infinite","negative_concentration"])
def test_ambiguous_scale_or_invalid_native_measurement_does_not_become_a_feature(fault):
    raw=staged().iloc[[0]]; authority=raw[["drug_id","gene_id"]]; config=declaration()
    if fault=="no_value_scale": config.pop("value_scale")
    if fault=="no_relation_scale": config.pop("relation_scale")
    if fault=="unit_conflict": raw=raw.assign(native_act_unit="nM")
    if fault=="p_unit_conflict": raw=raw.assign(native_act_unit="p_molar"); config["value_scale"]="CONCENTRATION"
    if fault=="infinite": raw=raw.assign(native_act_value=np.inf)
    if fault=="negative_concentration":
        raw=raw.assign(native_act_value=-2.,native_act_unit="nM"); config["value_scale"]="CONCENTRATION"
    with pytest.raises(ValueError): activity_rows(raw,authority,config,source_sha="a"*64)


def test_native_structure_uses_stable_identity_and_retains_conflict_reasons():
    ids=pd.DataFrame([("DC:1","source:DRUGCENTRAL","1","synthetic"),
        ("DC:1","name","alpha","synthetic"),("DC:2","name","beta","synthetic")],
        columns=["drug_id","identifier_type","identifier_value","source"])
    raw=pd.DataFrame({"ID":["1","1","99"],"INN":["alpha","beta","unknown"],"SMILES":["CCO","CCC","CO"]})
    frame,audit=native_structures(raw,DrugIdentityResolver(ids),{"DC:1","DC:2"},source_sha="a"*64)
    assert frame.drug_id.tolist()==["DC:1"] and frame.smiles.tolist()==["CCO"]
    assert audit.reason.iloc[1]=="STABLE_IDENTIFIER_NAME_CONFLICT"
    assert audit.reason.iloc[2]=="NO_CURATED_IDENTITY_MATCH"


def test_missing_native_measurement_fields_preserve_actions_with_explicit_reasons():
    raw=staged().iloc[:2].copy(); raw.loc[0,"native_relation"]=None
    authority=raw[["drug_id","gene_id"]].drop_duplicates()
    frame,audit=activity_rows(raw,authority,declaration(value_scale="CONCENTRATION"),source_sha="a"*64)
    assert frame.affinity_value.isna().all() and frame.action_type.tolist()==["INHIBITOR","AGONIST"]
    assert audit.reason.tolist()==["MISSING_OR_UNSUPPORTED_CENSORING_RELATION","MISSING_CONCENTRATION_UNIT"]


def test_native_protocol_keeps_targetless_assays_and_never_turns_ranges_into_doses():
    source=pd.DataFrame({"dataset_id":["S","S"],"source_drug_id":["17","99"],"drug_id":["DC:1","ASSAY:99"]})
    response=pd.DataFrame({"dataset_id":["S","S"],"cell_line_id":["C1","C1"],"drug_id":["DC:1","ASSAY:99"]})
    frame=pd.DataFrame({"compound":["17","99","bad"],"model":"C1","plate":"P1","dose":[0.,10.,1.],"response_value":[.2,.4,.6]})
    spec={"condition_kind":"POINT_CONCENTRATION","columns":{"source_drug_id":"compound","cell_line_id":"model",
        "assay_id":"plate","dose_value":"dose"},"constants":{"dataset_id":"S","dose_unit":"nM","exposure_hours":0.}}
    result,audit=native_protocols(frame,spec,source,response,source_sha="a"*64)
    assert result.drug_id.tolist()==["DC:1","ASSAY:99"] and result.dose_value.tolist()==[0.,10.]
    assert "response_value" not in result and audit.reason.iloc[2]=="NO_STAGED_NATIVE_DRUG_ID"
    result,audit=native_protocols(frame,{**spec,"condition_kind":"FITTED_CONCENTRATION_RANGE"},source,response,source_sha="a"*64)
    assert result.empty and audit.reason.iloc[:2].eq("RANGE_IS_NOT_POINT_DOSE").all()


def native_fixture(tmp_path):
    from test_v32_drug_identity import native_staging_fixture
    staged_root,paths=native_staging_fixture(tmp_path,pharmacology=True)
    structures=tmp_path/"structures.smiles.tsv"
    pd.DataFrame({"ID":["1"],"INN":["compound-alpha"],"SMILES":["CCO"],"InChIKey":[None]}).to_csv(structures,sep="\t",index=False)
    protocol=tmp_path/"protocol.csv"
    pd.DataFrame({"DRUG_ID":["17"]*5,"SANGER_MODEL_ID":[f"SIDM{i}" for i in range(5)],
        "ASSAY_ID":"plate1","DOSE":[0.,1.,10.,100.,1000.]}).to_csv(protocol,index=False)
    manifest={"format":"NATIVE_DRUG_PHARMACOLOGY_V1","staging_manifest":record(staged_root/"STAGING_MANIFEST.json"),
        "drugcentral_structures":record(structures),"drugcentral_activity":declaration(),
        "assay_protocols":[{"source":record(protocol),"condition_kind":"POINT_CONCENTRATION",
            "columns":{"source_drug_id":"DRUG_ID","cell_line_id":"SANGER_MODEL_ID","assay_id":"ASSAY_ID","dose_value":"DOSE"},
            "constants":{"dataset_id":"GDSC1_AUC","dose_unit":"nM","exposure_hours":72.}}]}
    path=tmp_path/"native_input.json"; path.write_text(json.dumps(manifest))
    return path,staged_root,paths,structures


def test_actual_native_staging_to_preparation_encoder_and_raw_source_tamper(tmp_path):
    from scripts.materialize_v32_native_drug_pharmacology import main
    native,staging,paths,structures=native_fixture(tmp_path)
    output=tmp_path/"pharmacology"
    assert main(["--manifest",str(native),"--output",str(output)])==0
    inputs=load_pharmacology(output/"PHARMACOLOGY_INPUT.json",
        targets=base.normalise_drug_targets(pd.read_parquet(staging/"drug_gene_target.static.parquet")),
        response=base.normalise_raw_response(pd.read_parquet(staging/"raw_cell_line_drug_response")),
        mapping=base.normalise_cell_line_map(pd.read_parquet(staging/"cell_line_map.parquet")))
    assert inputs.targets.p_value_molar.tolist()==[8.] and inputs.targets.p_relation.tolist()==[">"]
    assert inputs.structures.loc[inputs.structures.drug_id.eq("DC:1"),"molecular_weight"].iloc[0]==pytest.approx(46.069)
    context=inputs.design_table(["SIDM0"]).iloc[0]
    assert context.dose_available and not context.positive_dose_available and context.vehicle_fraction==1
    assert not inputs._static.ki_exact_p_available.any() # Censored measurements are not point estimates.
    assert inputs.receipt()["native_origin_sources"]["native_raw_cell_line_drug_response"]["sha256"]==artifact_sha256(staging/"raw_cell_line_drug_response")
    audit=json.loads((output/"NATIVE_ADAPTATION_AUDIT.json").read_text())
    assert not audit["training_started"] and not audit["dose_response_analysis_performed"]
    disposition=pd.read_parquet(output/"native_target_disposition.parquet")
    assert len(disposition)==3 and disposition.target_mapped.tolist()==[True,False,False]
    assert disposition.reason.iloc[1]=="NON_HUMAN_OR_UNSPECIFIED_ORGANISM"
    assert disposition.reason.iloc[2]=="MISSING_OR_AMBIGUOUS_NATIVE_COMPOUND_OR_GENE"
    # Missing protocol metadata remains unavailable through the same consumer.
    specification=json.loads(native.read_text()); specification["assay_protocols"]=[]
    without_protocol=tmp_path/"no_protocol.json"; without_protocol.write_text(json.dumps(specification))
    assert main(["--manifest",str(without_protocol),"--output",str(tmp_path/"no_protocol")])==0
    absent=load_pharmacology(tmp_path/"no_protocol/PHARMACOLOGY_INPUT.json",
        targets=base.normalise_drug_targets(pd.read_parquet(staging/"drug_gene_target.static.parquet")),
        response=base.normalise_raw_response(pd.read_parquet(staging/"raw_cell_line_drug_response")),
        mapping=base.normalise_cell_line_map(pd.read_parquet(staging/"cell_line_map.parquet")))
    query=pd.DataFrame({"cancer_id":["BRCA"],"drug_id":["DC:1"]})
    missing=absent.table(query,["SIDM0"])
    assert not missing.dose_available.any() and missing.log10_molar_dose_median.isna().all()
    structures.write_text(structures.read_text()+"\n")
    with pytest.raises(ValueError,match="SHA256"): inputs.verify_sources()


def test_native_adapter_rejects_changes_to_the_actual_staging_input(tmp_path):
    from cc_hhgt.v32.drug_pharmacology_native import materialize_native_pharmacology
    native,staging,paths,_=native_fixture(tmp_path)
    paths["targets"].write_text(paths["targets"].read_text()+"\n")
    with pytest.raises(ValueError,match="SHA256"):
        materialize_native_pharmacology(native,tmp_path/"rejected")
    assert not (tmp_path/"rejected/SUCCESS.json").exists()
