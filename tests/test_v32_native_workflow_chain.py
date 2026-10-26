"""Actual native preparation/collection/publication and restart safety; no fit."""
from copy import deepcopy
from pathlib import Path
import socket

import pytest

from cc_hhgt.v32 import native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_native_fusion_workflow import source as fusion_source, no_fit, complete, forbidden


def request(path, stages):
    write(path,{"format":chain.INPUT,"chain_id":"synthetic-native-chain","stages":stages});return path


def row(name, operation, inputs):return {"id":name,"operation":operation,"inputs":inputs}
def ref(name,pointer=""):return {"stage_ref":name,"pointer":pointer}
def external(name,pointer=""):return {"external_ref":name,"pointer":pointer}


@pytest.fixture(scope="module")
def module_no_fit():
    import torch
    from sklearn.linear_model import LogisticRegression
    from cc_hhgt.v32 import multimodal_fusion
    from scripts import run_v32_current_multimodal_fusion as runner
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam","AdamW","SGD"):patch.setattr(torch.optim,name,forbidden)
        patch.setattr(LogisticRegression,"fit",forbidden)
        patch.setattr(multimodal_fusion,"fit_residual_fusion_model",forbidden)
        patch.setattr(runner,"train_pair_blocked_crossfit_fusion",forbidden)
        yield


@pytest.fixture(scope="module")
def completed_chain(module_no_fit,fusion_source,tmp_path_factory):
    root=tmp_path_factory.mktemp("native_workflow_chain")
    from cc_hhgt.v32.current_analysis_web import COMPONENTS
    publication={"format":"CURRENT_ANALYSIS_WORKFLOW_INPUT_V1","release_id":"chain-publication",
        "selected":{"status":"REUSE","binding":ref("fusion","/selected_analysis")},
        "components":{k:{"status":"NOT_PRODUCED","reason":"Explicitly absent in this publication fixture"} for k in COMPONENTS}}
    source=request(root/"chain.json",[
        row("prepared","fusion.prepare",read(fusion_source)),
        row("fusion","fusion.collect",{"preparation":ref("prepared"),"completed":external("fusion-result")}),
        row("independent","fusion.prepare",read(fusion_source)),
        row("publication","publication.run",publication)])
    output=root/"chain"; assert chain.plan(source)["external_names"]==["fusion-result"]
    from scripts.run_native_workflow_chain import main
    assert main(["plan","--manifest",str(source)])==0 and not output.exists()
    assert main(["init","--manifest",str(source),"--output",str(output)])==0
    first=chain.advance(output);state=read(first)
    assert state["status"]=="WAITING_EXTERNAL",state
    assert set(state["stages"])=={"prepared","independent"}
    prep=Path(state["results"]["prepared"]["path"])
    assert not (prep.parent/"training").exists()
    result=complete(prep)  # Explicit test-only frozen serializer outside chain dispatcher.
    registry=root/"external.json";write(registry,{"format":chain.EXTERNAL,"inputs":{"fusion-result":record(result)}})
    chain.advance(output,registry);state=read(first)
    assert state["status"]=="DECLARED_STAGES_COMPLETE",state
    return root,source,output


def test_actual_fusion_completion_publication_and_resume(completed_chain,monkeypatch):
    root,_,output=completed_chain;state=read(output/"CHAIN.json")
    assert len(state["stages"])==4 and not state["training_started"]
    assert state["completion_scope"]=="DECLARED_NATIVE_WORKFLOW_STAGES_ONLY"
    publication=read(state["results"]["publication"]["path"])
    fusion=read(state["results"]["fusion"]["path"])
    assert publication["selected_analysis"]==fusion["selected_analysis"]
    before={str(p):p.stat().st_mtime_ns for p in (output/"stages").rglob("*") if p.is_file()}
    monkeypatch.setattr(chain,"dispatch",forbidden)
    chain.advance(output)
    assert before=={str(p):p.stat().st_mtime_ns for p in (output/"stages").rglob("*") if p.is_file()}
    assert read(output/"CHAIN.json")["stages"]==state["stages"]


@pytest.mark.parametrize("fault",["operation","inputs"])
def test_rehashed_stage_declaration_drift_is_rejected(completed_chain,fault):
    _,_,output=completed_chain;path=output/"CHAIN.json";original=path.read_bytes();state=read(path)
    receipt=Path(state["stages"]["prepared"]["path"]);raw=receipt.read_bytes();value=read(receipt)
    if fault=="operation":value["operation"]="single_cell.prepare"
    else:value["inputs"]["selection_rationale"]="changed after preparation"
    try:
        write(receipt,value);state["stages"]["prepared"]=record(receipt);write(path,state)
        with pytest.raises(ValueError,match="stage differs"):chain.advance(output)
        assert read(path)["status"]=="FAILED"
    finally:receipt.write_bytes(raw);path.write_bytes(original)


def test_external_identity_cannot_be_replaced(completed_chain,tmp_path):
    _,_,output=completed_chain;state=read(output/"CHAIN.json")
    original=Path(state["externals"]["fusion-result"]["path"]);copy=tmp_path/"copy.json";copy.write_bytes(original.read_bytes())
    registry=tmp_path/"external.json";write(registry,{"format":chain.EXTERNAL,"inputs":{"fusion-result":record(copy)}})
    with pytest.raises(ValueError,match="cannot be replaced"):chain.advance(output,registry)


def test_failure_keeps_independent_progress_and_retries_in_new_directory(fusion_source,tmp_path,monkeypatch):
    source=request(tmp_path/"chain.json",[row("first","fusion.prepare",read(fusion_source)),row("second","fusion.prepare",read(fusion_source))])
    output=tmp_path/"chain";chain.initialize(source,output);original=chain.dispatch;calls=[]
    def intermittent(operation,inputs,folder):
        calls.append(folder)
        if folder.parent.name=="first":raise OSError("Synthetic terminal I/O failure before dispatch")
        return original(operation,inputs,folder)
    with monkeypatch.context() as patch:
        patch.setattr(chain,"dispatch",intermittent);chain.advance(output)
    state=read(output/"CHAIN.json");assert state["status"]=="FAILED" and set(state["stages"])=={"second"}
    failed=output/"stages/first/0001";assert (failed/"RUN_FAILED.json").is_file() and not (failed/"STAGE_BINDING.json").exists()
    old=state["stages"]["second"];chain.advance(output);state=read(output/"CHAIN.json")
    assert state["status"]=="DECLARED_STAGES_COMPLETE" and state["stages"]["second"]==old
    assert Path(state["stages"]["first"]["path"]).parent.name=="0002"


def test_unfinished_attempt_is_never_inferred_terminal_or_restarted(fusion_source,tmp_path,monkeypatch):
    source=request(tmp_path/"chain.json",[row("first","fusion.prepare",read(fusion_source))]);output=tmp_path/"chain"
    chain.initialize(source,output);attempt=output/"stages/first/0001";attempt.mkdir(parents=True)
    monkeypatch.setattr(chain,"dispatch",forbidden);chain.advance(output)
    state=read(output/"CHAIN.json");assert state["status"]=="NEEDS_INSPECTION" and not state["stages"]
    assert not (attempt.parent/"0002").exists()


def test_live_os_lock_prevents_a_second_coordinator(fusion_source,tmp_path):
    source=request(tmp_path/"chain.json",[row("first","fusion.prepare",read(fusion_source))]);output=tmp_path/"chain"
    chain.initialize(source,output)
    with chain.locked(output):
        with pytest.raises(RuntimeError,match="Another coordinator"):chain.advance(output)
    assert read(output/"CHAIN.json")["status"]=="READY"


def test_recover_completed_native_output_without_restarting_producer(fusion_source,tmp_path,monkeypatch):
    source=request(tmp_path/"chain.json",[row("first","fusion.prepare",read(fusion_source))]);output=tmp_path/"chain"
    chain.initialize(source,output);chain.advance(output);state=read(output/"CHAIN.json")
    native=state["results"]["first"];receipt=Path(state["stages"]["first"]["path"])
    receipt.rename(receipt.with_name("SIMULATED_UNCOMMITTED_RECEIPT.json"));state["stages"]={};write(output/"CHAIN.json",state)
    monkeypatch.setattr(chain,"dispatch",forbidden);chain.advance(output)
    assert read(output/"CHAIN.json")["status"]=="NEEDS_INSPECTION"
    chain.recover(output,"first",native);chain.advance(output)
    assert read(output/"CHAIN.json")["status"]=="DECLARED_STAGES_COMPLETE"
    assert len(list((output/"stages/first").iterdir()))==1


@pytest.mark.parametrize("fault",["fit","core_run","unknown","duplicate","forward","cycle","wrong_collection","extra_reference"])
def test_invalid_or_training_chain_never_creates_output(tmp_path,fault):
    stages=[row("first","fusion.prepare",{})]
    if fault in {"fit","core_run","unknown"}:stages[0]["operation"]={"fit":"fusion.fit","core_run":"core.run","unknown":"shell"}[fault]
    elif fault=="duplicate":stages.append(deepcopy(stages[0]))
    elif fault in {"forward","cycle"}:stages[0]["inputs"]={"source":ref("later" if fault=="forward" else "first")}
    elif fault=="wrong_collection":stages[0]["operation"]="clinical.collect"
    else:stages[0]["inputs"]={"source":{"external_ref":"completed","shell":"anything"}}
    source=request(tmp_path/"chain.json",stages)
    with pytest.raises(ValueError):chain.initialize(source,tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_external_payload_json_file_and_relative_pins(fusion_source,tmp_path):
    pin=tmp_path/"payload.json";payload=read(fusion_source)
    original=Path(payload["primary_fusion"]["path"])
    import os
    payload["primary_fusion"]["path"]=os.path.relpath(original,tmp_path)
    write(pin,{"nested":payload})
    source=request(tmp_path/"chain.json",[row("first","fusion.prepare",{"external_ref":"input","pointer":"/nested"})])
    chain.initialize(source,tmp_path/"chain");registry=tmp_path/"external.json"
    write(registry,{"format":chain.EXTERNAL,"inputs":{"input":{"path":pin.name,"sha256":record(pin)["sha256"]}}})
    chain.advance(tmp_path/"chain",registry);state=read(tmp_path/"chain/CHAIN.json")
    assert state["status"]=="DECLARED_STAGES_COMPLETE"
    folder=tmp_path/"generated";folder.mkdir()
    generated=chain.resolve({"json_file":{"value":external("input","/nested/format")}}, {},state["externals"],folder,materialize=True)
    assert read(generated["path"])["value"]==read(fusion_source)["format"]
    assert generated==chain.resolve({"json_file":{"value":external("input","/nested/format")}}, {},state["externals"],folder,materialize=False)


def test_host_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(socket,"gethostname",lambda:"local-windows")
    with pytest.raises(RuntimeError,match="149"):chain.initialize(tmp_path/"missing",tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_actual_cached_gdc_expression_workflow_through_chain(tmp_path,module_no_fit,monkeypatch):
    from test_v32_native_expression_workflow import fixture_input
    from cc_hhgt.v32.native_expression_workflow import validate
    manifest=fixture_input(tmp_path/"raw",two_cancers=True)
    source=request(tmp_path/"chain.json",[row("expression","expression.run",read(manifest))]);output=tmp_path/"chain"
    chain.initialize(source,output);chain.advance(output);state=read(output/"CHAIN.json")
    assert state["status"]=="DECLARED_STAGES_COMPLETE",state
    _,binding=validate(state["results"]["expression"],tmp_path)
    assert len(binding["outputs"]["partitions"])==30
    assert binding["association_analysis_executed"] is False and binding["training_started"] is False
    monkeypatch.setattr(chain,"dispatch",forbidden);chain.advance(output)
