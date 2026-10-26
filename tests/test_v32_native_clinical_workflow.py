"""Actual Clinical collection/serialization/publication with optimization forbidden."""
from copy import deepcopy
from pathlib import Path
import socket

import pytest
import torch

from cc_hhgt.v32 import native_clinical_workflow as workflow
from cc_hhgt.v32.formal_expression_sources import read,write,record
from scripts import run_v32_clinical_training as risk,run_v32_clinical_entity_training as entity
from test_v32_current_clinical_context import current_sources,clinical_fixture,frozen_clinical_head


def forbidden(*a,**k):raise AssertionError("No Clinical fitting or entity statistics during preparation")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    monkeypatch.setattr("cc_hhgt.v32.clinical_training.train_private_clinical_head",forbidden)


@pytest.fixture(scope="module")
def source(tmp_path_factory,current_sources):
    root=tmp_path_factory.mktemp("native_clinical_workflow")
    path=clinical_fixture(root/"native",current_sources);manifest=read(path)
    clinical=read(manifest["patient_risk_lineage"]["path"]);association=read(manifest["entity_lineage"]["path"])
    files={**clinical["source_contract"]["files"],**association["source_contract"]["files"],"selected_analysis":manifest["selected_analysis"]}
    spec={"format":workflow.INPUT,"patient_risk_run_id":"CLINICAL-WORKFLOW","entity_run_id":"ENTITY-WORKFLOW",
        "core_run_id":manifest["core_run_id"],**files,"risk_config":clinical["training_config"]["parameters"],
        "entity_config":{"block_size":association["calculation_config"]["block_size"],
            **{"min_"+k:v for k,v in association["calculation_config"]["minimums"].items()}},"fdr_threshold":.08}
    target=root/"WORKFLOW.json";write(target,spec);return target


def complete(prepared):
    commands=read(prepared.parent/"EXECUTION_COMMANDS.json")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("cc_hhgt.v32.clinical_training.train_private_clinical_head",frozen_clinical_head)
        assert risk.main(commands["risk"]["argv"][3:])==0
    assert entity.main(commands["entity"]["argv"][3:])==0
    return prepared.parent/"risk_training/MODULE_LINEAGE.json",prepared.parent/"entity_calculation/ENTITY_CLINICAL_LINEAGE.json"


def test_native_chain_clinical_prepare_collect(source,tmp_path):
    from native_chain_fixture import exercise
    result=exercise("clinical",source,tmp_path,complete)
    assert read(result["clinical_context"]["path"])["raw_patient_feature_projection_replayed"]


def test_actual_both_preparation_collectors_and_existing_publication(source,tmp_path,monkeypatch):
    from scripts.run_native_clinical_workflow import main
    args=["prepare","--manifest",str(source),"--output",str(tmp_path/"prepared")]
    assert main([*args,"--plan"])==0 and not (tmp_path/"prepared").exists()
    with monkeypatch.context() as patch:
        patch.setattr("cc_hhgt.v32.clinical_training.build_clinical_head",forbidden)
        patch.setattr(entity,"analyse_entity_matrix",forbidden)
        patch.setattr("cc_hhgt.v32.clinical_entity_training.analyse_entity_matrix",forbidden)
        patch.setattr(torch.nn.Module,"__call__",forbidden)
        assert main(args)==0
    prepared=tmp_path/"prepared/PREPARED.json"
    _,_,value,risk_ready,entity_ready=workflow.validate_preparation(record(prepared),tmp_path)
    assert not (prepared.parent/"risk_training").exists() and not (prepared.parent/"entity_calculation").exists()
    assert len(risk_ready["folds"])==5 and len(entity_ready["partitions"])==3
    assert not risk_ready["model_forward_started"] and not entity_ready["entity_statistics_computed"]
    assert {p["patients"] for p in entity_ready["partitions"]}=={40}
    commands=read(prepared.parent/"EXECUTION_COMMANDS.json")
    for kind,parser in (("risk",risk._parser()),("entity",entity._parser())):
        command=commands[kind]
        parsed=parser.parse_args(command["argv"][3:]);assert not parsed.prepare_only and parsed.patient_authority_mode=="current"
        assert not command["automatically_executed"] and command["requires_separate_explicit_execution"]
        with pytest.raises(ValueError,match="cannot dispatch"):workflow.execute_preflight(command["argv"],tmp_path/"forbidden.log")
    assert not (tmp_path/"forbidden.log").exists()
    completed_risk,completed_entity=complete(prepared)
    original={str(p):record(p)["sha256"] for directory in (completed_risk.parent,completed_entity.parent) for p in directory.rglob("*") if p.is_file()}
    args=["collect","--preparation",str(prepared),"--preparation-sha256",record(prepared)["sha256"],
        "--risk-completed",str(completed_risk),"--risk-completed-sha256",record(completed_risk)["sha256"],
        "--entity-completed",str(completed_entity),"--entity-completed-sha256",record(completed_entity)["sha256"],"--output",str(tmp_path/"out")]
    assert main(args)==0
    _,result=workflow.validate(record(tmp_path/"out/NATIVE_CLINICAL_BINDING.json"),tmp_path)
    assert original=={str(p):record(p)["sha256"] for directory in (completed_risk.parent,completed_entity.parent) for p in directory.rglob("*") if p.is_file()}
    context=read(result["clinical_context"]["path"])
    assert record(prepared) in context["source_artifacts"] and context["fdr_threshold"]==.08
    assert context["raw_patient_feature_projection_replayed"] and context["entity_raw_patient_statistics_replayed"]
    from cc_hhgt.v32.current_analysis_workflow import run_current_analysis_workflow,INPUT_FORMAT
    from cc_hhgt.v32.current_analysis_web import COMPONENTS,CurrentAnalysisWebStore
    components={k:{"status":"NOT_PRODUCED","reason":"Outside this Clinical workflow fixture"} for k in COMPONENTS}
    components["clinical"]={"status":"REUSE","binding":result["clinical_context"]}
    publication=tmp_path/"publication.json"
    write(publication,{"format":INPUT_FORMAT,"release_id":"clinical-workflow-synthetic",
        "selected":{"status":"REUSE","binding":value["files"]["selected_analysis"]},"components":components})
    published=run_current_analysis_workflow(publication,tmp_path/"publication");web=published["web_binding"]
    store=CurrentAnalysisWebStore(web["path"],expected_sha=web["sha256"])
    assert store.binding["component_status"]["clinical"]["status"]=="AVAILABLE"
    assert not result["training_started"] and not result["independent_predictive_performance_verified"]


@pytest.mark.parametrize("fault",["risk_run","entity_run","core_run","config_unknown","config_bool","config_fraction","min_folds","fdr_nan","candidate","authority"])
def test_invalid_plan_rejected_before_output(source,tmp_path,fault):
    spec=deepcopy(read(source))
    if fault=="risk_run":spec["patient_risk_run_id"]="../escape"
    elif fault=="entity_run":spec["entity_run_id"]=""
    elif fault=="core_run":spec["core_run_id"]="other"
    elif fault=="config_unknown":spec["risk_config"]["unknown"]=1
    elif fault=="config_bool":spec["risk_config"]["seed"]=True
    elif fault=="config_fraction":spec["entity_config"]["min_train_events"]=1.2
    elif fault=="min_folds":spec["entity_config"]["min_folds_available"]=6
    elif fault=="fdr_nan":spec["fdr_threshold"]=float("nan")
    elif fault=="candidate":spec["candidates"]=spec["patient_folds"]
    else:spec["patient_authority_receipt"]["sha256"]="0"*64
    path=tmp_path/"input.json";write(path,spec)
    with pytest.raises((ValueError,RuntimeError)):workflow.prepare(path,tmp_path/"out")
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("fault",["risk_arrays","risk_patients","entity_arrays","entity_patients","command"])
def test_rehashed_preparation_reconstructs_actual_inputs(source,tmp_path,fault):
    prepared=workflow.prepare(source,tmp_path/"ready");binding=read(prepared)
    key="execution_commands" if fault=="command" else "risk_preflight" if fault.startswith("risk") else "entity_preflight"
    target=Path(binding[key]["path"]);data=read(target)
    if fault=="command":data["entity"]["automatically_executed"]=True
    elif fault=="risk_arrays":data["folds"][0]["training_input_sha256"]="0"*64
    elif fault=="risk_patients":data["patient_fold_authority"]["patients"]+=1
    elif fault=="entity_arrays":data["partitions"][0]["array_sha256"]="0"*64
    else:data["partitions"][0]["patients"]+=1
    write(target,data);binding[key]=record(target);write(prepared,binding)
    with pytest.raises(ValueError,match="Clinical (preparation numerical replay|execution commands)"):
        workflow.validate_preparation(record(prepared),tmp_path)


@pytest.mark.parametrize("fault",["risk_run","risk_config","risk_source","risk_arrays","entity_run","entity_config","entity_source","entity_authority","entity_arrays"])
def test_completed_results_must_match_prepared_inputs_and_parameters(source,tmp_path,fault):
    prepared=workflow.prepare(source,tmp_path/"ready");rp,ep=complete(prepared)
    target=rp if fault.startswith("risk") else ep;data=read(target)
    if fault.endswith("run"):data["training_run_id"]="OTHER"
    elif fault=="risk_config":data["training_config"]["parameters"]["epochs"]+=1
    elif fault=="entity_config":data["calculation_config"]["minimums"]["test_events"]+=1
    elif fault.endswith("source"):data["source_contract"]["files"]["workbook"]["sha256"]="0"*64
    elif fault=="risk_arrays":data["task_lineage"][0]["input_description"]["training_input_sha256"]="0"*64
    elif fault=="entity_authority":data["patient_fold_authority"]=None
    else:
        cp=ep.parent/"CALCULATION_MANIFEST.json";calc=read(cp);calc["partitions"][0]["input_description"]["array_sha256"]="0"*64
        write(cp,calc);data["calculation_manifest_sha256"]=record(cp)["sha256"]
    write(target,data)
    with pytest.raises(ValueError,match="Completed Clinical"):
        workflow.collect(record(prepared),record(rp),record(ep),tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_actual_context_auditor_follows_preparation_dependency(source,tmp_path):
    from cc_hhgt.v32.current_clinical_context import audit_current_clinical_context
    prepared=workflow.prepare(source,tmp_path/"ready");rp,ep=complete(prepared)
    result=read(workflow.collect(record(prepared),record(rp),record(ep),tmp_path/"out"))
    data=read(prepared);data["status"]="FAILED";write(prepared,data)
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        audit_current_clinical_context(result["clinical_context"]["path"],expected_sha=result["clinical_context"]["sha256"])


def test_preparation_failure_keeps_diagnostics_and_no_ready_binding(source,tmp_path,monkeypatch):
    monkeypatch.setattr(entity,"collect_partition",lambda *a,**k:(_ for _ in ()).throw(ValueError("missing measurement")))
    with pytest.raises(ValueError,match="missing measurement"):workflow.prepare(source,tmp_path/"out")
    assert (tmp_path/"out/RUN_FAILED.json").exists() and not (tmp_path/"out/PREPARED.json").exists()
    assert not (tmp_path/"out/risk_training").exists()


def test_host_and_existing_output_guards_precede_work(source,tmp_path,monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(socket,"gethostname",lambda:"windows")
        with pytest.raises(RuntimeError,match="149"):workflow.prepare(source,tmp_path/"out")
    assert not (tmp_path/"out").exists()
    out=tmp_path/"existing";out.mkdir();(out/"original").write_text("keep")
    with pytest.raises(FileExistsError):workflow.prepare(source,out)
    assert (out/"original").read_text()=="keep"
