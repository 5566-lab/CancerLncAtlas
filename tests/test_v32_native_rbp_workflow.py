"""Cached KD/RBNS -> actual chain -> rule publication and typed graph; no fit."""
from pathlib import Path
import socket

import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import native_rbp_workflow as workflow, native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read,write,record
from test_v32_current_state_context import current_sources


def forbidden(*a,**k):raise AssertionError("No optimizer or download is allowed")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression,"fit",forbidden)
    import subprocess
    monkeypatch.setattr(subprocess,"run",forbidden)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    from test_v32_encode_functional import test_complete_command_consumes_cached_native_data_without_training as kd
    from test_v32_rbns_sequence import test_full_command_streams_native_scores_and_writes_predicted_binding as rbns
    root=tmp_path_factory.mktemp("native_rbp_sources");result={}
    for kind,create in (("encode_functional",kd),("rbns_sequence",rbns)):
        folder=root/kind;folder.mkdir();create(folder);manifest=folder/"manifest.json";spec=read(manifest)
        roles=("physical_binding","exact_candidates") if kind=="encode_functional" else ("lncrnas","transcript_crosswalk")
        for role in roles:
            target=Path(spec["artifacts"][role]["path"]);table=pd.read_parquet(target)
            old="L1" if kind=="encode_functional" else "LNC:ENSG0001"
            table.loc[table.lncrna_id.eq(old),"lncrna_id"]="LNC:Z"
            if "cancer_id" in table:table["cancer_id"]="BRCA"
            table.to_parquet(target,index=False);spec["artifacts"][role]=record(target)
        write(manifest,spec)
        path=folder/"WORKFLOW.json";write(path,{"format":workflow.INPUT,"kind":kind,"analysis":record(manifest),"cache":str(folder/"cache"),
            **({"chunk_rows":1} if kind=="rbns_sequence" else {})})
        result[kind]=path
    return root,result


@pytest.fixture(scope="module")
def completed(inputs,current_sources,tmp_path_factory):
    root=tmp_path_factory.mktemp("native_rbp_chain");_,paths=inputs
    candidates=root/"candidates.parquet";pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"pathway_id":["P1"]}).to_parquet(candidates,index=False)
    members=root/"members.parquet";pd.DataFrame({"pathway_id":["P1"],"member_id":["RBP1"]}).to_parquet(members,index=False)
    def ref(stage,pointer):return {"stage_ref":stage,"pointer":pointer}
    rule={"format":"CURRENT_RULE_EVIDENCE_INPUT_V1","candidates":record(candidates),"exact_membership":record(members),
        "sources":[{"source_id":"kd","kind":"encode_functional","audit":ref("kd","/audit"),
            "table":ref("kd","/outputs/FUNCTIONAL_EVIDENCE.parquet")},
            {"source_id":"rbns","kind":"rbns_sequence","audit":ref("rbns","/audit"),
            "table":ref("rbns","/outputs/GENE_SEQUENCE_PREFERENCE.parquet")}]}
    from cc_hhgt.v32.current_analysis_web import COMPONENTS
    components={k:{"status":"NOT_PRODUCED","reason":"Not part of this cached evidence fixture"} for k in COMPONENTS}
    components["evidence"]={"status":"BUILD","input_manifest":{"json_file":rule}}
    stages=[{"id":name,"operation":"rbp.run","inputs":read(paths[kind])}
        for name,kind in (("kd","encode_functional"),("rbns","rbns_sequence"))]
    stages.append({"id":"publication","operation":"publication.run","inputs":{
        "format":"CURRENT_ANALYSIS_WORKFLOW_INPUT_V1","release_id":"native-rbp-fixture",
        "selected":{"status":"REUSE","binding":record(current_sources[1])},"components":components}})
    manifest=root/"CHAIN_INPUT.json";write(manifest,{"format":chain.INPUT,"chain_id":"rbp-evidence","stages":stages})
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam","AdamW","SGD"):patch.setattr(torch.optim,name,forbidden)
        import subprocess
        patch.setattr(subprocess,"run",forbidden)
        chain.initialize(manifest,root/"chain");chain.advance(root/"chain")
    state=read(root/"chain/CHAIN.json");assert state["status"]=="DECLARED_STAGES_COMPLETE",state
    return root,state,members


def test_cached_analyses_reach_rule_query_and_actual_publication(completed,monkeypatch):
    root,state,_=completed
    from cc_hhgt.v32.current_analysis_web import CurrentAnalysisWebStore
    from cc_hhgt.v32.evidence_rule_release import RuleEvidenceQuery
    publication=read(state["results"]["publication"]["path"])
    web=publication["web_binding"];store=CurrentAnalysisWebStore(web["path"],expected_sha=web["sha256"])
    pin=store.binding["component_bindings"]["evidence"]
    row=RuleEvidenceQuery(pin["path"],expected_sha=pin["sha256"]).query(cancer_id="BRCA",lncrna_id="LNC:Z")["rows"][0]
    assert row["functional_bridge_studies"]==1 and row["prediction_sources"]==1
    assert row["direct_studies"]==0 and row["observed_direct_direction"]=="unknown"
    before={str(p):record(p)["sha256"] for p in (root/"chain/stages").rglob("*") if p.is_file()}
    monkeypatch.setattr(chain,"dispatch",forbidden);chain.advance(root/"chain")
    assert before=={str(p):record(p)["sha256"] for p in (root/"chain/stages").rglob("*") if p.is_file()}


def test_actual_rbns_output_enters_only_predicted_graph_arm(completed):
    from scripts.prepare_v32_current_binding_comparison import load_sources
    from test_v32_current_binding_expansion import current_bundle,source
    from cc_hhgt.v32.current_binding_graph import normalize_binding_source,build_current_binding_arms,binding_arm_runtime_bundle,ARMS
    _,state,_=completed;binding=read(state["results"]["rbns"]["path"])
    spec={"source_id":"rbns","kind":"rbns_sequence","audit":binding["audit"],"binding":binding["outputs"]["RBNS_TYPED_BINDING.parquet"]}
    tables,_=load_sources([spec],Path("/"))
    tables.append(normalize_binding_source(source([("LNC:A","UNIPROT:A","rip",1.)]),source_id="synthetic-experimental"))
    arms=build_current_binding_arms(current_bundle(),tables,fold=0,edge_chunk_size=2)
    predicted=[binding_arm_runtime_bundle(arms,arm).edges.relation_type.eq("binds_protein_predicted").sum() for arm in ARMS]
    assert predicted==[0,0,1]


def test_explicit_predicted_relation_cannot_be_relabelled_physical():
    from test_v32_current_binding_expansion import source
    from cc_hhgt.v32.current_binding_graph import normalize_binding_source
    frame=source([("LNC:Z","PROTEIN:P1","predicted",.4)])
    frame["relation_type"]="binds_protein_rip"
    with pytest.raises(ValueError,match="contradicts its assay"):
        normalize_binding_source(frame,source_id="invalid")


@pytest.mark.parametrize("kind",["encode_functional","rbns_sequence"])
def test_plan_missing_cached_file_never_downloads_or_creates_output(inputs,tmp_path,kind):
    _,paths=inputs;spec=read(paths[kind]);cache=tmp_path/"empty";cache.mkdir();spec["cache"]=str(cache)
    source=tmp_path/"INPUT.json";write(source,spec)
    with pytest.raises(FileNotFoundError,match="Missing selected ENCODE"):workflow.run(source,tmp_path/"out")
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("kind,table,column",[("kd","RBP_PATHWAY_SUPPORT.parquet","pvalue"),
    ("rbns","GENE_SEQUENCE_PREFERENCE.parquet","prediction_score"),("rbns","RBNS_TYPED_BINDING.parquet","weight")])
def test_rehashed_native_outputs_are_rejected_at_actual_consumer(completed,kind,table,column):
    from cc_hhgt.v32.evidence_rule_sources import load_rule_sources
    from scripts.prepare_v32_current_binding_comparison import load_sources
    _,state,members=completed;binding=read(state["results"][kind]["path"])
    audit_path=Path(binding["audit"]["path"]);audit=read(audit_path);target=Path(audit["outputs"][table]["path"])
    success=audit_path.parent/"SUCCESS.json";original={p:p.read_bytes() for p in (target,audit_path,success)}
    try:
        frame=pd.read_parquet(target);frame.loc[frame[column].notna(),column]+=0.1;frame.to_parquet(target,index=False)
        audit["outputs"][table]={**audit["outputs"][table],**record(target)};write(audit_path,audit)
        write(success,{"status":"PASS","audit_sha256":record(audit_path)["sha256"]})
        with pytest.raises(ValueError,match="RBP native output differs"):
            if column=="weight":
                load_sources([{"source_id":"rbns","kind":"rbns_sequence","audit":record(audit_path),"binding":record(target)}],audit_path.parent)
            else:
                source_kind="encode_functional" if kind=="kd" else "rbns_sequence"
                name="FUNCTIONAL_EVIDENCE.parquet" if kind=="kd" else "GENE_SEQUENCE_PREFERENCE.parquet"
                load_rule_sources([{"source_id":kind,"kind":source_kind,"audit":record(audit_path),"table":audit["outputs"][name]}],
                    audit_path.parent,pd.read_parquet(members))
    finally:
        for path,data in original.items():path.write_bytes(data)


@pytest.mark.parametrize("consumer",["workflow","rule_query"])
def test_changed_raw_cache_rejected_even_when_native_output_is_unchanged(completed,consumer):
    _,state,_=completed;binding=read(state["results"]["rbns"]["path"]);audit=read(binding["audit"]["path"])
    path=Path(next(iter(audit["source_files"].values()))["path"]);original=path.read_bytes()
    try:
        path.write_bytes(original+b"\n")
        with pytest.raises(ValueError,match="checksum"):
            if consumer=="workflow":workflow.validate(state["results"]["rbns"],Path("/"))
            else:
                from cc_hhgt.v32.evidence_rule_release import RuleEvidenceQuery
                publication=read(state["results"]["publication"]["path"])
                evidence=next(s["binding"] for s in publication["stages"] if s["stage"]=="evidence")
                RuleEvidenceQuery(evidence["path"],expected_sha=evidence["sha256"])
    finally:path.write_bytes(original)


def test_plan_cli_is_readonly_and_host_checked(inputs,tmp_path,monkeypatch):
    from scripts.run_native_rbp_workflow import main
    source=inputs[1]["rbns_sequence"]
    assert main(["--manifest",str(source),"--output",str(tmp_path/"out"),"--plan"])==0
    assert not (tmp_path/"out").exists()
    monkeypatch.setattr(socket,"gethostname",lambda:"windows")
    with pytest.raises(RuntimeError,match="149"):workflow.run(source,tmp_path/"out")
