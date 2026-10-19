"""Actual native command orchestration preserves all three CNV decisions."""
from pathlib import Path
import json
import socket
import subprocess
import sys

import pandas as pd
import pytest

from cc_hhgt.v32 import native_cnv_workflow as workflow
from cc_hhgt.v32.formal_expression_sources import read,write,record
from cc_hhgt.v32.formal_graph_authority import GRAPH_INPUT_RECEIPT_FORMAT,GRAPH_INPUT_RECEIPT_STATUS
from test_v32_local_cnv_cohort import fixture_inputs,no_fit,ROOT,CANCERS


def input_manifest(root,confounded=True,variable_low=False):
    args=fixture_inputs(root,confounded,variable_low=variable_low)
    graph=root/"graph_receipt.json"
    write(graph,{"format":GRAPH_INPUT_RECEIPT_FORMAT,"status":GRAPH_INPUT_RECEIPT_STATUS,
        "synthetic_static_control_only":True,"cancer_scope":CANCERS,"counts":{},"gates":{},
        "artifacts":{k:record(args.graph_root/"static"/(k+".parquet")) for k in ("signed_membership","protein_gene_encoding")},
        "fold_artifacts":{str(f):{"train_coexpression":{}} for f in range(5)}})
    manifest=root/"workflow.json"
    write(manifest,{"format":workflow.INPUT,"run_id":"SYNTHETIC_NATIVE_WORKFLOW", "candidates":record(args.candidates),
        "patient_map":record(args.patient_map),"source_config":record(args.source_config),
        "directional_success":record(args.directional_root/"SUCCESS.json"),"graph_root":str(args.graph_root),"graph_receipt":record(graph)})
    return manifest,args


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    assert socket.gethostname()=="149"
    root=tmp_path_factory.mktemp("native_cnv_workflow");outputs={}
    for name,confounded,variable_low in (("correction",True,False),("small",False,True),("review",False,False)):
        manifest,args=input_manifest(root/name,confounded,variable_low)
        output=root/(name+"_output")
        command=[sys.executable,"-m","scripts.run_native_cnv_workflow","--manifest",str(manifest),"--output",str(output)]
        if name=="correction":
            result=subprocess.run([*command,"--plan"],cwd=ROOT,capture_output=True,text=True,timeout=30)
            assert result.returncode==0,result.stderr
            plan=json.loads(result.stdout)
            assert len(plan["commands"])==10 and not output.exists()
            assert not plan["training_started"] and not plan["baseline_fitting_started"]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=150)
        assert result.returncode==0,result.stderr
        outputs[name]=(output,manifest,args)
    return outputs


@pytest.mark.parametrize("kind,decision",[("correction",workflow.CORRECTION),("small","LOCAL_CNV_IMPACT_SMALL"),
                                          ("review","LOCAL_CNV_SENSITIVITY_ONLY_REVIEW")])
def test_actual_cli_decision_and_scientific_outputs(completed,kind,decision):
    root,manifest,args=completed[kind]
    _,binding=workflow.validate(record(root/"NATIVE_CNV_BINDING.json"),root)
    assert binding["decision"]==decision and not binding["training_started"] and not binding["baseline_fitting_started"]
    rows=pd.read_parquet(root/"association/local_cnv_confounding_pair_level.parquet")
    assert len(rows)==30 and set(rows.run_id)=={"SYNTHETIC_NATIVE_WORKFLOW"}
    state=read(root/"STAGE_STATUS.json")
    if kind=="correction":
        assert binding["corrected_inputs_available"] and all(s["status"]=="PASS" for s in state["stages"])
        assert read(root/"validation/INDEPENDENT_AUDIT.json")["rows"]==30
        graph=read(root/"corrected_graph/GRAPH_INPUT_AUTHORITY_RECEIPT.json")
        for fold in range(5):
            path=Path(graph["fold_artifacts"][str(fold)]["train_coexpression"]["path"])
            assert len(list(path.glob("part_*.parquet")))==2
        assert rows.loc[rows.lncrna_id.eq("LNC:LOW"),"availability"].eq("typed_unavailable").all()
    else:
        assert not binding["corrected_inputs_available"] and not (root/"corrected_graph").exists()
        assert all(s["status"]=="NOT_RUN_BY_DECISION" and s["decision"]==decision for s in state["stages"][3:])
    assert binding["core_preparation_status"]=="NOT_RUN_INCLUDES_L1_FITTING"
    assert not (root/"prepared").exists() and not (root/"training").exists()


@pytest.mark.parametrize("fault",["candidate_definition","static_hash","cancer","directional_failed","covariate_path","extra_command"])
def test_bad_inputs_stop_before_output(tmp_path,fault):
    manifest,args=input_manifest(tmp_path/"input");spec=read(manifest)
    if fault=="candidate_definition":
        config=read(args.source_config);config.pop("candidate_definition");write(args.source_config,config);spec["source_config"]=record(args.source_config)
    elif fault=="static_hash":
        graph=Path(spec["graph_receipt"]["path"]);value=read(graph);value["artifacts"]["signed_membership"]["sha256"]="0"*64;write(graph,value);spec["graph_receipt"]=record(graph)
    elif fault=="cancer":
        path=Path(spec["directional_success"]["path"]);value=read(path);value["cancers"].pop("LUAD");write(path,value);spec["directional_success"]=record(path)
    elif fault=="directional_failed":
        path=Path(spec["directional_success"]["path"]);value=read(path);value["status"]="FAILED";write(path,value);spec["directional_success"]=record(path)
    elif fault=="covariate_path":
        config=read(args.source_config);config["sources"]["covariates"]["path"]="covariates.parquet";write(args.source_config,config);spec["source_config"]=record(args.source_config)
    else:spec["commands"]=["train"]
    write(manifest,spec)
    with pytest.raises(ValueError):workflow.run(manifest,tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_actual_worker_failure_leaves_stage_diagnostics(tmp_path):
    manifest,args=input_manifest(tmp_path/"input")
    cov=pd.read_parquet(args.covariates).drop(columns=["purity"]);cov.to_parquet(args.covariates,index=False)
    config=read(args.source_config);config["sources"]["covariates"]=record(args.covariates);write(args.source_config,config)
    spec=read(manifest);spec["source_config"]=record(args.source_config);write(manifest,spec)
    with pytest.raises(ValueError,match="association:BRCA"):workflow.run(manifest,tmp_path/"bad")
    state=read(tmp_path/"bad/STAGE_STATUS.json")
    assert state["failed_stage"]=="association:BRCA" and state["stages"][0]["status"]=="FAILED"
    assert not (tmp_path/"bad/SUCCESS.json").exists() and not (tmp_path/"bad/corrected_graph").exists()


def test_changed_input_rejected_after_completion(completed):
    root,_,args=completed["correction"]
    path=args.covariates;original=path.read_bytes()
    try:
        path.write_bytes(original+b"changed")
        with pytest.raises(ValueError):workflow.validate(record(root/"NATIVE_CNV_BINDING.json"),root)
    finally:path.write_bytes(original)


def test_repinned_skipped_stage_cannot_claim_execution(completed):
    root,_,_=completed["small"]
    paths=[root/n for n in ("STAGE_STATUS.json","NATIVE_CNV_BINDING.json","SUCCESS.json")]
    originals={p:p.read_bytes() for p in paths}
    try:
        state=read(paths[0]);state["stages"][3]["status"]="PASS";write(paths[0],state)
        binding=read(paths[1]);binding["stage_status"]=record(paths[0]);write(paths[1],binding)
        success=read(paths[2]);success["binding"]=record(paths[1]);write(paths[2],success)
        with pytest.raises(ValueError,match="decision was bypassed"):workflow.validate(record(paths[1]),root)
    finally:
        for p,value in originals.items():p.write_bytes(value)


def test_wrong_host_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(socket,"gethostname",lambda:"windows")
    with pytest.raises(RuntimeError,match="149"):workflow.preflight(tmp_path/"missing",tmp_path/"out")
