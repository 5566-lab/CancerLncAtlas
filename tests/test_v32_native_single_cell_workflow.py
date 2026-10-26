"""Actual SC serializers, frozen models, signed fusion and current publication."""
from dataclasses import asdict,replace
from pathlib import Path
import socket

import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import native_single_cell_workflow as workflow
from cc_hhgt.v32.formal_expression_sources import read,write,record
from cc_hhgt.v32.single_cell_signed_fusion import load_signed_sc_fusion
from scripts import train_single_cell_hhgt_149 as native
from test_v32_single_cell_numeric_replay import frozen_fit,reseal
from test_v32_current_state_context import current_sources
import test_single_cell_hhgt_training as fixtures


def forbidden(*a,**k):raise AssertionError("SC workflow fitting is forbidden")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    monkeypatch.setattr(native,"fit_fold_model",forbidden)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root=tmp_path_factory.mktemp("native_sc_workflow")
    # Define the desired identities in the input generator, before any model
    # sees them. Completed predictions are never renamed to fit a publisher.
    original_static=fixtures._write_static_asset;original_payload=fixtures._episode_payload
    def static(root,cancer_id="BRCA"):return original_static(root,cancer_id)
    def payload(*args,**kwargs):
        kwargs.setdefault("cancer_id","BRCA");value=original_payload(*args,**kwargs)
        value["row_metadata"]["lncrna_id"]=["LNC:Z" if v=="L0" else v for v in value["row_metadata"]["lncrna_id"]]
        return value
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fixtures,"_write_static_asset",static);patch.setattr(fixtures,"_episode_payload",payload)
        episode,graph=fixtures._write_episode_fixture(root)
    settings=replace(fixtures._tiny_settings(episode,graph,root/"unused"),threads_per_worker=1)
    descriptors,_=native.load_episode_manifest(settings);donors=[]
    for parts in descriptors.values():
        for descriptor in parts:
            item=native.load_episode(descriptor,"with_lasso");dataset=item.rows.dataset_id.iloc[0]
            donors.extend({"dataset_id":dataset,"query_donor_id":d,"canonical_donor_id":"CANONICAL_"+d}
                for d in (*item.outer_train_donors,*item.outer_query_donors))
    pd.DataFrame(donors).drop_duplicates().to_parquet(root/"crosswalk.parquet",index=False)
    pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"pathway_id":["P1"]}).to_parquet(root/"candidates.parquet",index=False)
    spec={"format":workflow.INPUT,"run_id":"SC-WORKFLOW-FROZEN-FIXTURE",
        "episode_manifest":record(episode/"SC_HHGT_EPISODE_MANIFEST.json"),
        "graph_asset_manifest":record(graph/"GRAPH_ASSET_MANIFEST.json"),
        "candidates":record(root/"candidates.parquet"),"donor_crosswalk":record(root/"crosswalk.parquet"),
        "training_config":{k:v for k,v in asdict(settings).items() if k not in {"episode_root","graph_assets_root","output_root","prepare_only"}}}
    path=root/"WORKFLOW.json";write(path,spec);return path


def complete(prepared,**changes):
    args=read(prepared.parent/"EXECUTION_COMMAND.json")["argv"][3:];settings=replace(native.parse_args(args),**changes)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native,"fit_fold_model",frozen_fit);native.run_training(settings)
    return settings.output_root/"SUCCESS.json"


@pytest.fixture(scope="module")
def collected(source,tmp_path_factory):
    root=tmp_path_factory.mktemp("sc_collected")
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam","AdamW","SGD"):patch.setattr(torch.optim,name,forbidden)
        prepared=workflow.prepare(source,root/"prepared");success=complete(prepared)
        result=workflow.collect(record(prepared),record(success),root/"result")
    return root,prepared,success,result


def test_original_cli_preparation_no_model_and_no_fit_dispatch(source,tmp_path,monkeypatch):
    from scripts.run_native_single_cell_workflow import main
    args=["prepare","--manifest",str(source),"--output",str(tmp_path/"prepared")]
    assert main([*args,"--plan"])==0 and not (tmp_path/"prepared").exists()
    monkeypatch.setattr(native,"_initialize_model",forbidden);monkeypatch.setattr(torch.nn.Module,"__call__",forbidden)
    monkeypatch.setattr(native,"_train_fold_job",forbidden)
    assert main(args)==0
    prepared=tmp_path/"prepared/PREPARED.json";_,_,value,ready=workflow.validate_preparation(record(prepared),tmp_path)
    assert len(ready["models"])==10 and not ready["model_constructed"] and not ready["model_forward_started"]
    assert not (prepared.parent/"training").exists()
    assert not value["execution_command"]["automatically_executed"]
    with pytest.raises(ValueError,match="cannot dispatch"):workflow.execute_preflight(value["execution_command"]["argv"],tmp_path/"forbidden.log")
    assert not (tmp_path/"forbidden.log").exists()


def test_actual_collection_cli_signed_context_and_existing_publication(source,tmp_path,current_sources):
    from scripts.run_native_single_cell_workflow import main
    from cc_hhgt.v32.current_single_cell_context import materialize_current_single_cell_context,audit_current_single_cell_context,INPUT_FORMAT
    prepared=workflow.prepare(source,tmp_path/"prepared");success=complete(prepared)
    before={str(p):record(p)["sha256"] for p in success.parent.rglob("*") if p.is_file()}
    assert main(["collect","--preparation",str(prepared),"--preparation-sha256",record(prepared)["sha256"],
        "--completed",str(success),"--completed-sha256",record(success)["sha256"],"--output",str(tmp_path/"result")])==0
    binding=read(tmp_path/"result/NATIVE_SINGLE_CELL_BINDING.json")
    audit,_,_,dependencies=load_signed_sc_fusion(binding["signed_fusion"]["path"],expected_sha=binding["signed_fusion"]["sha256"])
    assert audit["native_numeric_replay_bound"] and audit["available_rows"]==1 and audit["donor_context_rows"]==5
    assert record(prepared)["sha256"] in {r["sha256"] for r in dependencies}
    context=tmp_path/"CONTEXT.json"
    write(context,{"format":INPUT_FORMAT,"selected_analysis":record(current_sources[1]),"native_binding":binding["signed_fusion"]})
    result=materialize_current_single_cell_context(context,tmp_path/"context");value=read(result["binding_path"])
    assert value["native_model_forwards_replayed"] and not value["used_in_selected_score"]
    assert record(prepared) in value["source_artifacts"]
    public=pd.read_parquet(value["outputs"]["selected_member_support"]["path"])
    assert public.single_cell_available.all() and public.single_cell_donors.eq(5).all()
    rechecked=audit_current_single_cell_context(result["binding_path"],expected_sha=result["binding_sha256"])
    assert rechecked["native_model_forwards_replayed_in_this_audit"] and not rechecked["training_started"]
    from cc_hhgt.v32.current_analysis_workflow import run_current_analysis_workflow,INPUT_FORMAT as PUBLICATION
    from cc_hhgt.v32.current_analysis_web import COMPONENTS,CurrentAnalysisWebStore
    components={k:{"status":"NOT_PRODUCED","reason":"Outside SC synthetic workflow"} for k in COMPONENTS}
    components["single_cell"]={"status":"REUSE","binding":record(result["binding_path"])}
    release=tmp_path/"PUBLICATION.json"
    write(release,{"format":PUBLICATION,"release_id":"sc-workflow-synthetic","selected":{"status":"REUSE","binding":record(current_sources[1])},"components":components})
    publication=run_current_analysis_workflow(release,tmp_path/"publication");web=publication["web_binding"]
    store=CurrentAnalysisWebStore(web["path"],expected_sha=web["sha256"])
    assert store.binding["component_status"]["single_cell"]["status"]=="AVAILABLE"
    assert before=={str(p):record(p)["sha256"] for p in success.parent.rglob("*") if p.is_file()}
    assert not binding["global_pointer_updated"] and not binding["independent_predictive_performance_verified"]


def test_actual_fusion_model_use_keeps_no_forward_and_complete_numeric_dependencies(collected,tmp_path,monkeypatch):
    from test_v32_single_cell_model_use import fusion_fixture
    from cc_hhgt.v32.current_model_use import materialize_current_model_use
    _,prepared,_,binding=collected;result=read(binding)
    # The existing fixture uses actual fusion serialization with fixed zero
    # weights. It explicitly substitutes fitting; no benefit claim is made.
    spec=read(prepared.parent/"LAUNCH_PREFLIGHT.json")
    pd.read_parquet(spec["files"]["candidates"]["path"]).to_parquet(tmp_path/"candidates.parquet",index=False)
    source=fusion_fixture(tmp_path,Path(result["signed_fusion"]["path"]))
    monkeypatch.setattr(native,"_initialize_model",forbidden);monkeypatch.setattr(torch.nn.Module,"__call__",forbidden)
    output=materialize_current_model_use(source,tmp_path/"use");value=read(output["binding_path"])
    assert not value["model_forwards_executed"] and not value["model_use_inventory_completeness_verified"]
    dependencies=pd.read_parquet(value["outputs"]["dependencies"]["path"])
    assert {result["numeric_replay"]["sha256"],record(prepared)["sha256"]}<=set(dependencies.sha256)


@pytest.mark.parametrize("fault",["run_id","unknown","bool","fraction","nan","minimum","candidate_null","crosswalk_duplicate","manifest_name"])
def test_invalid_plan_fails_before_output(source,tmp_path,fault):
    spec=read(source)
    if fault=="run_id":spec["run_id"]="../escape"
    elif fault=="unknown":spec["training_config"]["not_a_setting"]=1
    elif fault=="bool":spec["training_config"]["epochs"]=True
    elif fault=="fraction":spec["training_config"]["hidden_channels"]=1.5
    elif fault=="nan":spec["training_config"]["learning_rate"]=float("nan")
    elif fault=="minimum":spec["min_donors"]=1
    elif fault=="manifest_name":
        path=tmp_path/"different.json";write(path,read(spec["episode_manifest"]["path"]));spec["episode_manifest"]=record(path)
    else:
        key="candidates" if fault=="candidate_null" else "donor_crosswalk";frame=pd.read_parquet(spec[key]["path"])
        if fault=="candidate_null":frame.loc[0,"lncrna_id"]=None
        else:frame=pd.concat([frame,frame.iloc[[0]]],ignore_index=True)
        path=tmp_path/"bad.parquet";frame.to_parquet(path,index=False);spec[key]=record(path)
    path=tmp_path/"INPUT.json";write(path,spec)
    with pytest.raises((ValueError,RuntimeError)):workflow.prepare(path,tmp_path/"out")
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("fault",["command","input_tensor","row_count","plan_parameter"])
def test_rehashed_preparation_cannot_change_original_inputs_or_commands(source,tmp_path,fault):
    path=workflow.prepare(source,tmp_path/"prepared");binding=read(path)
    key="execution_command" if fault=="command" else "plan" if fault=="plan_parameter" else "input_preflight"
    target=Path(binding[key]["path"]);value=read(target)
    if fault=="command":value["argv"][0]="some-other-interpreter"
    elif fault=="plan_parameter":value["run_config"]["epochs"]+=1
    elif fault=="input_tensor":value["models"][0]["episodes"][0]["input_tensor_sha256"]="0"*64
    else:value["models"][0]["rows_by_role"]["query"]+=1
    write(target,value);binding[key]=record(target);write(path,binding)
    with pytest.raises(ValueError,match="SC .*differ"):workflow.validate_preparation(record(path),tmp_path)


def test_completed_configuration_must_match_preparation(source,tmp_path):
    prepared=workflow.prepare(source,tmp_path/"prepared");success=complete(prepared,seed=12345)
    with pytest.raises(ValueError,match="configuration differ"):workflow.collect(record(prepared),record(success),tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_rehashed_query_prediction_is_rejected_by_collection(source,tmp_path):
    prepared=workflow.prepare(source,tmp_path/"prepared");success=complete(prepared)
    model=read(success.parent/"CHECKPOINT_MANIFEST.json")["models"][0]
    target=Path(model["files"]["predictions"]["path"]);frame=pd.read_parquet(target)
    frame.predicted_rho=frame.predicted_rho*.5+.1
    frame.predicted_association_strength=frame.predicted_rho.abs();frame.residual=frame.rho-frame.predicted_rho
    frame.to_parquet(target,index=False);reseal(success.parent)
    with pytest.raises(ValueError,match="frozen query predictions differ"):workflow.collect(record(prepared),record(success),tmp_path/"out")
    assert (tmp_path/"out/RUN_FAILED.json").exists() and not (tmp_path/"out/NATIVE_SINGLE_CELL_BINDING.json").exists()


def test_wrong_output_host_and_existing_directory_are_refused(source,collected,tmp_path,monkeypatch):
    _,prepared,success,_=collected
    with pytest.raises(ValueError,match="outside immutable"):workflow.collect(record(prepared),record(success),success.parent/"audit")
    with pytest.raises(FileExistsError):workflow.prepare(source,prepared.parent)
    with pytest.raises(ValueError,match="inside an input"):workflow.plan(source,Path(read(source)["episode_manifest"]["path"]).parent/"bad")
    monkeypatch.setattr(socket,"gethostname",lambda:"windows")
    with pytest.raises(RuntimeError,match="149"):workflow.plan(source,tmp_path/"out")
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("fault",["candidate","minimum","numerical_semantics"])
def test_signed_loader_follows_preparation_and_numerical_receipt(collected,tmp_path,fault):
    _,_,_,binding=collected;result=read(binding);original=Path(result["signed_fusion"]["path"]);value=read(original)
    if fault=="candidate":
        path=tmp_path/"candidates.parquet";pd.read_parquet(value["inputs"]["candidates"]["path"]).to_parquet(path,index=False)
        value["inputs"]["candidates"]=record(path)
    elif fault=="minimum":value["min_donors"]+=1
    else:value["native_numeric_replay_bound"]=False
    path=tmp_path/"AUDIT.json";write(path,value);write(tmp_path/"SUCCESS.json",{"status":"PASS","audit_sha256":record(path)["sha256"]})
    with pytest.raises(ValueError,match="SC .*differ"):load_signed_sc_fusion(path,expected_sha=record(path)["sha256"])


@pytest.mark.parametrize("fault",["missing","ambiguous","inner_alias","outer_alias"])
def test_preparation_requires_complete_canonical_episode_partitions(source,tmp_path,fault):
    spec=read(source);frame=pd.read_parquet(spec["donor_crosswalk"]["path"])
    if fault=="missing":frame=frame.loc[~frame.query_donor_id.eq("BRCA_F0_D2")].copy()
    elif fault=="ambiguous":
        row=frame.loc[frame.query_donor_id.eq("BRCA_F0_D2")].copy();row.dataset_id="OTHER";row.canonical_donor_id="OTHER_PERSON"
        frame=pd.concat([frame,row],ignore_index=True)
    elif fault=="inner_alias":frame.loc[frame.query_donor_id.eq("BRCA_F0_D2"),"canonical_donor_id"]="CANONICAL_BRCA_F0_Q0"
    else:frame.loc[frame.query_donor_id.eq("BRCA_F1_Q0"),"canonical_donor_id"]="CANONICAL_BRCA_F0_Q0"
    cross=tmp_path/"cross.parquet";frame.to_parquet(cross,index=False);spec["donor_crosswalk"]=record(cross)
    path=tmp_path/"INPUT.json";write(path,spec)
    with pytest.raises(ValueError,match="SC .*canonical"):workflow.prepare(path,tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_rehashed_numeric_receipt_does_not_replace_actual_forward(collected,tmp_path):
    from cc_hhgt.v32.single_cell_signed_fusion import materialize_signed_sc_fusion
    _,_,_,binding=collected;result=read(binding);value=read(result["numeric_replay"]["path"])
    value["models"][0]["validation_loss"]+=.1
    numerical=tmp_path/"NUMERIC_REPLAY.json";write(numerical,value)
    pointer=read(result["pointer"]["path"]);pointer["numeric_replay"]=record(numerical)
    path=tmp_path/"POINTER.json";write(path,pointer)
    prepared=read(read(pointer["workflow_preparation"]["path"])["plan"]["path"])
    with pytest.raises(ValueError,match="saved numerical replay differs"):
        materialize_signed_sc_fusion(path,Path(prepared["files"]["candidates"]["path"]),
            Path(prepared["files"]["donor_crosswalk"]["path"]),tmp_path/"out")
    assert not (tmp_path/"out").exists()
