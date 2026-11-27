from __future__ import annotations
import csv
import copy
import json
from pathlib import Path
import pytest
from cla_engine.common import Stop,sha,read_json,atomic_json
from cla_engine.kegg381_backbone import build_backbone,materialize_backbone
from cla_engine.target_policy import REQUIRED_FIELDS
from cla_engine.prepared_target_policy import (load_bound_policy,select_pathways,
    select_state_targets,seal_prepared_policy,validate_prepared_policy,resolve_identity)

def write_tsv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,delimiter="\t")
        writer.writeheader();writer.writerows(rows)

def policy_row(cid,kind="pathway",role="primary_pathway",display="Pathway"):
    group={"primary_pathway":"pathway","state_auxiliary":"state","regulator_auxiliary":"state","context_only":"none"}[role]
    return dict(canonical_id=cid,roles=display,biological_type=kind,analysis_role=role,
        loss_group=group,loss_weight={"pathway":1,"state":.5,"none":0}[group],
        active_in_analysis=True,cancer_scope_status="user_fixed_backbone",backbone_anchor_ids="")

def fixture_binding(root,cfg=None):
    root.mkdir(parents=True,exist_ok=True)
    if cfg is None:
        cfg={"synthetic_fixture":True,"targets":[
            {"id":"S1","column":"s1","role":"primary","family":"state"},
            {"id":"TF1","column":"tf1","role":"primary","family":"regulator"},
            {"id":"PROGENY1","column":"prog1","role":"primary","family":"regulator"}]}
    source=["P0","P1"]
    mapping=[{"source_target_id":sid,"source_name":sid,"systematic_id":"M"+str(i),
              "network_id":f"N{i+1:05d}","canonical_id":"CAN:"+sid,"mapping_status":"mapped"}
              for i,sid in enumerate(source)]
    scopes=[{**row,"admitted_by_network_scope":"True","cancer_scope_status":"direct_cancer_network"} for row in mapping]
    genes={sid:[f"ENSG{i+1:011d}"] for i,sid in enumerate(source)}
    catalog=[{"canonical_id":row["canonical_id"],"display_name":row["source_name"]} for row in mapping]
    rows,genes,summary=build_backbone(source,mapping,scopes,genes,catalog,expected_count=2)
    materialize_backbone(root/"backbone",rows,genes,summary)
    policies=[policy_row("CAN:"+sid) for sid in source]
    identity=[{"source_target_id":sid,"source_column":"","canonical_id":"CAN:"+sid} for sid in source]
    primary=[t for t in cfg["targets"] if t["role"]=="primary"]
    excluded=set(t["id"] for t in primary[-2:])
    for row in cfg["targets"]:
        cid="CAN:"+row["id"]
        role="regulator_auxiliary" if row["id"] in excluded else "state_auxiliary"
        policies.append(policy_row(cid,"tf" if row["id"] in excluded else "state",role,
                                   "Pathway" if row["id"] in excluded else "State"))
        identity.append({"source_target_id":row["id"],"source_column":row["column"],"canonical_id":cid})
    write_tsv(root/"TARGET_POLICY.tsv",policies,REQUIRED_FIELDS)
    write_tsv(root/"SOURCE_IDENTITY_MAP.tsv",identity,["source_target_id","source_column","canonical_id"])
    cfg["relation_target_policy"]={"path":str(root/"TARGET_POLICY.tsv"),"sha256":sha(root/"TARGET_POLICY.tsv"),
        "backbone_path":str(root/"backbone"),"backbone_manifest_sha256":sha(root/"backbone/MANIFEST.json"),
        "source_identity_map":str(root/"SOURCE_IDENTITY_MAP.tsv"),"source_identity_map_sha256":sha(root/"SOURCE_IDENTITY_MAP.tsv")}
    return cfg,excluded

def test_binding_requires_all_hashes_and_verified_dynamic_count(tmp_path):
    cfg,_=fixture_binding(tmp_path)
    bad=copy.deepcopy(cfg);bad["relation_target_policy"].pop("sha256")
    with pytest.raises(Stop,match="All policy"):load_bound_policy(bad)
    cfg["synthetic_fixture"]=False
    assert load_bound_policy(cfg)["backbone"]["rows"]

def test_admission_preserves_source_ids_and_excludes_regulators(tmp_path):
    cfg,excluded=fixture_binding(tmp_path)
    binding=load_bound_policy(cfg)
    assert select_pathways(binding,["P2","P1","P0","TF1"])==["P0","P1"]
    states=select_state_targets(cfg,binding)
    assert [x["id"] for x in states]==["S1"]
    assert not excluded.intersection(x["id"] for x in states)
    assert resolve_identity(binding,"unknown","s1")=="CAN:S1"
    with pytest.raises(Stop,match="absent from graph"):select_pathways(binding,["P0"])

def test_tampered_sources_fail_checksum(tmp_path):
    cfg,_=fixture_binding(tmp_path)
    Path(cfg["relation_target_policy"]["path"]).write_text("wrong")
    with pytest.raises(Stop,match="target_policy"):load_bound_policy(cfg)

def test_many_to_one_supervised_states_require_explicit_aggregation(tmp_path):
    cfg,_=fixture_binding(tmp_path)
    cfg["targets"].append({"id":"S2","column":"s2","role":"primary","family":"state"})
    p=Path(cfg["relation_target_policy"]["source_identity_map"])
    with p.open("a") as handle:handle.write("S2\ts2\tCAN:S1\n")
    cfg["relation_target_policy"]["source_identity_map_sha256"]=sha(p)
    with pytest.raises(Stop,match="Multiple original training targets"):
        select_state_targets(cfg,load_bound_policy(cfg))

def test_sealed_policy_works_without_original_files_and_rejects_axis_drift(tmp_path):
    cfg,_=fixture_binding(tmp_path/"sources")
    binding=load_bound_policy(cfg)
    states=select_state_targets(cfg,binding)
    out=tmp_path/"prepared"
    receipt=seal_prepared_policy(out,cfg,binding,["P0","P1"],states,"formal")
    spec={"nodes":{"pathway":["P0","P1"]},"states":["S1"],
          "pathway_canonical_ids":receipt["pathway_canonical_ids"],
          "state_canonical_ids":receipt["state_canonical_ids"],
          "relation_target_policy_sha256":sha(out/"RELATION_TARGET_POLICY.json")}
    Path(cfg["relation_target_policy"]["path"]).rename(tmp_path/"old_policy.tsv")
    assert validate_prepared_policy(out,cfg,spec)["state_targets"]==states
    drift=copy.deepcopy(spec);drift["nodes"]["pathway"]=["P0"]
    with pytest.raises(Stop,match="omitted fixed backbone"):validate_prepared_policy(out,cfg,drift)
    drift=copy.deepcopy(spec);drift["states"].append("TF1")
    with pytest.raises(Stop,match="State axis differs"):validate_prepared_policy(out,cfg,drift)

def test_diagnostic_cap_allowed_formal_cap_rejected(tmp_path):
    cfg,_=fixture_binding(tmp_path/"sources")
    binding=load_bound_policy(cfg);states=select_state_targets(cfg,binding)
    seal_prepared_policy(tmp_path/"diagnostic",cfg,binding,["P0"],states,"diagnostic")
    with pytest.raises(Stop,match="cannot omit"):
        seal_prepared_policy(tmp_path/"formal",cfg,binding,["P0"],states,"formal")

def test_legacy_unbound_policy_remains_optional(tmp_path):
    cfg={"targets":[{"id":"S1","role":"primary"}]}
    assert load_bound_policy(cfg) is None
    assert select_pathways(None,["P1","P0","P1"])==["P0","P1"]
    assert select_state_targets(cfg,None)==cfg["targets"]
    assert validate_prepared_policy(tmp_path,cfg,{}) is None

def test_real_prepare_and_lineage_consume_sealed_policy(tmp_path):
    from cla_engine.fixture import make_fixture
    from cla_engine.prepare import prepare
    from cla_engine.lineage import validate_prepared
    cfg=make_fixture(tmp_path/"fixture")
    cfg,excluded=fixture_binding(tmp_path/"policy",cfg)
    cfg["baseline"]["identity_mode"]="numeric_only"
    out=tmp_path/"prepared"
    prepare(cfg,out,fold=0,profile="diagnostic")
    manifest,spec,stored,lineage=validate_prepared(out)
    assert spec["nodes"]["pathway"]==["P0","P1"]
    assert not set(spec["states"]) & excluded
    assert spec["pathway_canonical_ids"]==["CAN:P0","CAN:P1"]
    assert lineage["relation_target_policy_sha256"]==spec["relation_target_policy_sha256"]
    assert "relation_target_policy/TARGET_POLICY.tsv" in manifest["files"]
    assert read_json(out/"BASELINE_CONTRACT.json")["identity_mode"]=="numeric_only"
    assert read_json(out/"baseline/pathway_identity_vocabulary.json")["dimensions"]==0
    assert "+ sparse identities" not in read_json(out/"BASELINES.json")["pathway"]["input_semantics"]
    assert read_json(out/"TASK_CONTRACT.json")["pathway_label"]=="abs(r)>=.20 & q<=.05 OR abs(r)>=.15 & q<=.10"
    assert read_json(out/"RELATION_TARGET_POLICY.json")["fused_y_converted_to_binary"] is False



def test_shared_state_identity_does_not_admit_routed_progeny(tmp_path):
    cfg,_=fixture_binding(tmp_path)
    cfg["targets"].append({"id":"PROGENY:Hypoxia","column":"progeny_hypoxia",
                          "role":"primary","family":"state","source_kind":"routed_program"})
    p=Path(cfg["relation_target_policy"]["source_identity_map"])
    with p.open("a") as handle:handle.write("PROGENY:Hypoxia\tprogeny_hypoxia\tCAN:S1\n")
    cfg["relation_target_policy"]["source_identity_map_sha256"]=sha(p)
    assert [x["id"] for x in select_state_targets(cfg,load_bound_policy(cfg))]==["S1"]

def test_baseline_identity_mode_is_explicit_and_legacy_contract_unchanged():
    from cla_engine.execution_contract import baseline_contract
    root=Path(__file__).resolve().parents[1]
    cfg=read_json(root/"config/149.json")
    cfg["baseline"].pop("identity_mode",None)
    legacy=baseline_contract(cfg)
    assert "identity_mode" not in legacy
    cfg["baseline"]["identity_mode"]="numeric_only"
    contract=baseline_contract(cfg)
    assert contract["identity_mode"]=="numeric_only"
    assert contract["identity_fit_scope"]=="NONE_NUMERIC_ONLY"
    cfg["baseline"]["identity_mode"]="unknown"
    with pytest.raises(Stop,match="exact or numeric_only"):baseline_contract(cfg)


def test_host_component_readouts_cannot_supervise_conditional_mechanisms():
    from cla_engine.prepared_target_policy import host_readout_qualified_axis
    b={'backbone':{'manifest':{'format':'CLA_KEGG_CANCER_BACKBONE_V2'},'rows':[
        {'source_target_id':'r','network_type':'Reference'},
        {'source_target_id':'v','network_type':'Variant'},
        {'source_target_id':'p','network_type':'Pathogen'},
        {'source_target_id':'e','network_type':'Env factor'}]}}
    assert host_readout_qualified_axis(b,['r','v','p','e'])==[True,False,False,False]
