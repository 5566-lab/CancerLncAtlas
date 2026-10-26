"""Actual State CLI collectors/serializer/publication, with fitting forbidden."""
from copy import deepcopy
from pathlib import Path
import socket

import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import native_state_workflow as workflow, state_training as native
from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.contracts import canonical_json_sha256
from test_v32_current_state_context import current_sources, native_fixture, frozen_state_fold, native_state_driver


def forbidden(*a,**k): raise AssertionError("State fitting/optimization is forbidden")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"): monkeypatch.setattr(torch.optim,name,forbidden)
    monkeypatch.setattr(native,"train_fold_private_head",forbidden)
    monkeypatch.setattr(native,"train_state_models",forbidden)


@pytest.fixture(scope="module")
def source(tmp_path_factory,current_sources):
    root=tmp_path_factory.mktemp("native_state_workflow_sources")
    path=native_fixture(root/"frozen_container",current_sources,input_style="dataset")
    context=read(path); lineage=read(context["state_training_lineage"]["path"])
    inputs={r["role"]:r for r in lineage["input_artifacts"]}
    # Use the actual patient-first authority builder, not altered frozen-count
    # constants or a hand-written approval receipt.
    from cc_hhgt.v32.patient_fold_authority import build_patient_fold_authority
    activity=root/"activity"; activity.mkdir()
    mapping=pd.read_parquet(inputs["v32_patient_fold_manifest"]["path"])
    mapping[["cancer_id","sample_id","patient_id"]].to_parquet(activity/"part.parquet",index=False)
    build_patient_fold_authority(activity_root=activity,output_root=root/"authority",expected_cancers=["BRCA"],expected_patients=40)
    spec={"format":workflow.INPUT,"state_run_id":"v32-state-workflow-fixture","core_run_id":context["core_run_id"],
        "core_manifest":context["core_manifest"],"selected_analysis":context["selected_analysis"],
        "state_measurements":{k:inputs["tumor_state_long_measurement"][k] for k in ("path","sha256")},
        "lncrna_expression":{k:inputs["lncrna_expression"][k] for k in ("path","sha256")},
        "patient_folds":record(root/"authority/SAMPLE_PATIENT_FOLD_MAP.tsv"),
        "patient_authority_receipt":record(root/"authority/PATIENT_FOLD_AUTHORITY_RECEIPT.json"),
        "config":{**lineage["config"],"validation_offset":2},"state_sets":context["state_sets"]}
    manifest=root/"WORKFLOW.json"; write(manifest,spec); return manifest


def complete(prepared):
    """Call the actual generated CLI with only fitting replaced by frozen heads."""
    command=read(prepared.parent/"TRAINING_COMMAND.json")["argv"]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native,"train_state_models",native_state_driver)
        patch.setattr(native,"train_fold_private_head",frozen_state_fold)
        assert native.main(command[3:])==0
    return prepared.parent/"training/STATE_TRAINING_LINEAGE.json"


def repin_lineage(path,value):
    value.pop("lineage_sha256");value["lineage_sha256"]=canonical_json_sha256(value);write(path,value)


def test_actual_prepare_collect_cli_and_existing_publication(source,tmp_path,monkeypatch):
    from scripts.run_native_state_workflow import main
    args=["prepare","--manifest",str(source),"--output",str(tmp_path/"prepared")]
    assert main([*args,"--plan"])==0 and not (tmp_path/"prepared").exists()
    original=native.compute_fold_local_associations; roles=[]
    def collect(*a,**k):
        roles.append(k["split"]); assert k["split"] in {"train","validation"}
        return original(*a,**k)
    with monkeypatch.context() as patch:
        patch.setattr(native,"compute_fold_local_associations",collect)
        patch.setattr(native,"build_state_private_head",forbidden)
        patch.setattr(native,"_predict_probabilities",forbidden)
        assert main(args)==0
    assert set(roles)=={"train","validation"}
    prepared=tmp_path/"prepared/PREPARED.json"
    _,binding,value,ready=workflow.validate_preparation(record(prepared),tmp_path)
    assert not (prepared.parent/"training").exists()
    assert len(ready["folds"])==5 and all(r["max_train_pairs"]==24 for r in ready["folds"])
    assert ready["patient_authority"]["audit"]["patients"]==40
    command=read(prepared.parent/"TRAINING_COMMAND.json")
    parsed=native.build_arg_parser().parse_args(command["argv"][3:])
    assert parsed.validation_offset==2 and not parsed.prepare_only and parsed.patient_authority_mode=="current"
    assert not command["automatically_executed"] and command["requires_separate_explicit_execution"]
    with pytest.raises(ValueError,match="cannot dispatch"): workflow.execute_preflight(command["argv"],tmp_path/"forbidden.log")
    assert not (tmp_path/"forbidden.log").exists()
    completed=complete(prepared)
    before={str(p):record(p)["sha256"] for p in completed.parent.rglob("*") if p.is_file()}
    output=tmp_path/"collected"
    assert main(["collect","--preparation",str(prepared),"--preparation-sha256",record(prepared)["sha256"],
        "--completed",str(completed),"--completed-sha256",record(completed)["sha256"],"--output",str(output)])==0
    _,result=workflow.validate(record(output/"NATIVE_STATE_BINDING.json"),tmp_path)
    assert before=={str(p):record(p)["sha256"] for p in completed.parent.rglob("*") if p.is_file()}
    assert binding==workflow.validate_preparation(record(prepared),tmp_path)[1]
    context=read(result["state_context"]["path"])
    assert context["private_head_predictions_replayed"] and context["native_numeric_replay"]["rows"]==35
    assert record(prepared) in context["source_artifacts"]
    # Actual pre-existing workflow consumes the collected binding as a State
    # stage, with other producers explicitly absent in this scoped fixture.
    from cc_hhgt.v32.current_analysis_workflow import run_current_analysis_workflow, INPUT_FORMAT
    from cc_hhgt.v32.current_analysis_web import COMPONENTS, CurrentAnalysisWebStore
    components={k:{"status":"NOT_PRODUCED","reason":"Outside this State integration fixture"} for k in COMPONENTS}
    components["state"]={"status":"REUSE","binding":result["state_context"]}
    publication=tmp_path/"publication.json"
    write(publication,{"format":INPUT_FORMAT,"release_id":"state-workflow-synthetic",
        "selected":{"status":"REUSE","binding":value["files"]["selected_analysis"]},"components":components})
    published=run_current_analysis_workflow(publication,tmp_path/"publication")
    web=published["web_binding"]; store=CurrentAnalysisWebStore(web["path"],expected_sha=web["sha256"])
    assert store.binding["component_status"]["state"]["status"]=="AVAILABLE"
    assert not result["training_started"] and not result["independent_predictive_performance_verified"]


@pytest.mark.parametrize("fault",["run","core_run","config_unknown","config_boolean","config_fraction","config_nan","state_sets","authority_hash"])
def test_invalid_plan_is_rejected_before_output(source,tmp_path,fault):
    spec=deepcopy(read(source))
    if fault=="run":spec["state_run_id"]="../escape"
    elif fault=="core_run":spec["core_run_id"]="another-run"
    elif fault=="config_unknown":spec["config"]["unknown"]=1
    elif fault=="config_boolean":spec["config"]["seed"]=True
    elif fault=="config_fraction":spec["config"]["max_epochs"]=1.5
    elif fault=="config_nan":spec["config"]["learning_rate"]=float("nan")
    elif fault=="state_sets":spec["state_sets"]["minimum_set_members"]=0
    else:spec["patient_authority_receipt"]["sha256"]="0"*64
    path=tmp_path/"input.json";write(path,spec)
    with pytest.raises((ValueError,RuntimeError)):workflow.prepare(path,tmp_path/"out")
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("fault",["input_arrays","split","authority","command"])
def test_rehashed_preparation_still_replays_actual_collector(source,tmp_path,fault):
    prepared=workflow.prepare(source,tmp_path/"ready");binding=read(prepared)
    if fault=="command":
        target=Path(binding["training_command"]["path"]);data=read(target);data["automatically_executed"]=True
        write(target,data);binding["training_command"]=record(target)
    else:
        target=Path(binding["head_preflight"]["path"]);data=read(target)
        if fault=="input_arrays":data["folds"][0]["training_input_sha256"]="0"*64
        elif fault=="split":data["folds"][0]["split_manifest_sha256"]="0"*64
        else:data["patient_authority"]["audit"]["patients"]+=1
        write(target,data);binding["head_preflight"]=record(target)
    write(prepared,binding)
    with pytest.raises(ValueError,match="State.*(preflight|command)"):workflow.validate_preparation(record(prepared),tmp_path)


@pytest.mark.parametrize("fault",["run","config","authority","source","input_arrays","split","programs"])
def test_collection_rejects_completed_result_different_from_preparation(source,tmp_path,fault):
    prepared=workflow.prepare(source,tmp_path/"ready");completed=complete(prepared);data=read(completed)
    if fault=="run":data["training_run_id"]="v32-other"
    elif fault=="config":data["config"]["max_epochs"]+=1
    elif fault=="authority":data["patient_authority"]=None
    elif fault=="source":data["input_artifacts"][0]["sha256"]="0"*64
    elif fault=="input_arrays":data["fold_lineage"]["0"]["training_input_sha256"]="0"*64
    elif fault=="split":data["fold_lineage"]["0"]["split_manifest_sha256"]="0"*64
    else:data["missing_state_ids"]=[]
    repin_lineage(completed,data)
    with pytest.raises(ValueError,match="Completed State"):workflow.collect(record(prepared),record(completed),tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_context_auditor_rechecks_workflow_preparation_dependency(source,tmp_path):
    from cc_hhgt.v32.current_state_context import audit_current_state_context
    prepared=workflow.prepare(source,tmp_path/"ready");completed=complete(prepared)
    result=read(workflow.collect(record(prepared),record(completed),tmp_path/"out"))
    value=read(prepared);value["status"]="FAILED";write(prepared,value)
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        audit_current_state_context(result["state_context"]["path"],expected_sha=result["state_context"]["sha256"])


def test_current_cli_requires_core_identity_and_preserves_frozen_default(source,tmp_path):
    command=workflow.plan(source,tmp_path/"ready")["training_command"]["argv"][3:]
    i=command.index("--expected-core-run-id");del command[i:i+2]
    with pytest.raises(RuntimeError,match="expected-core-run-id"):native.main([*command,"--prepare-only"])
    assert not (tmp_path/"ready").exists()
    command=workflow.plan(source,tmp_path/"ready")["training_command"]["argv"][3:]
    i=command.index("--patient-authority-mode");del command[i:i+2]
    with pytest.raises(RuntimeError):native.main([*command,"--prepare-only"])
    assert not (tmp_path/"ready").exists()


def test_host_and_existing_output_guards_precede_work(source,tmp_path,monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(socket,"gethostname",lambda:"local")
        with pytest.raises(RuntimeError,match="149"):workflow.prepare(source,tmp_path/"missing")
    existing=tmp_path/"existing";existing.mkdir();(existing/"original").write_text("keep")
    with pytest.raises(FileExistsError):workflow.prepare(source,existing)
    assert (existing/"original").read_text()=="keep"
