"""Actual core orchestration; only frozen cohort controls and L1 logits are fixtures."""
from pathlib import Path
from types import SimpleNamespace
import copy
import shutil
import socket

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from cc_hhgt.v32 import native_core_workflow as workflow,native_cnv_workflow as cnv
from cc_hhgt.v32.formal_expression_sources import read,write,record
from test_v32_typed_graph_preparation import normalized,scoped,native_authority,patch_controls,no_fit
from test_v32_fold_local_expression import forbidden


@pytest.fixture(scope="module")
def inputs(native_authority,scoped,tmp_path_factory):
    root=tmp_path_factory.mktemp("native_core_workflow")
    prior_root,prior_manifest,_=native_authority
    original=scoped/"native_corrected_graph/native"
    config=read(original/"source_config.json")
    config["candidate_definition"]=record(scoped/"normal/CANDIDATE_SOURCES.json")
    config_path=root/"sources.json";write(config_path,config)
    directional=root/"directional";shutil.copytree(original/"directional",directional)
    part=directional/"cancer=BRCA";ids=read(part/"patient_ids.json");lncs=read(part/"lncrna_ids.json")
    # The same latent signal drives the synthetic raw counts and pathway
    # activity. CNV captures that signal, while normalized expression retains
    # its nonlinear/count variation, so adjustment remains estimable.
    t=np.arange(len(ids),dtype=float)
    signed=np.load(part/"lncrna_signed_mean.npy");signed[:,lncs.index("LNC:Z")]=np.sin(t*.63)+np.cos(t*.27)
    np.save(part/"lncrna_signed_mean.npy",signed)
    raw=root/"synthetic_correlated_cnv.tsv";pd.DataFrame({"patient_id":ids,"cnv":signed[:,lncs.index("LNC:Z")]}).to_csv(raw,sep="\t",index=False)
    write(part/"SOURCE_FILES.json",{"records":[record(raw)]})
    write(part/"SUCCESS.json",{"status":"SUCCESS","files":{p.name:record(p)["sha256"] for p in part.iterdir() if p.name!="SUCCESS.json"}})
    write(directional/"SUCCESS.json",{"status":"SUCCESS","cancers":{"BRCA":{"processed_segment_patients":120,"typed_unavailable_patients":0}}})
    graph=read(prior_root/"source.json");static=root/"graph/static";static.mkdir(parents=True)
    for key in ("signed_membership","protein_gene_encoding"):
        path=static/(key+".parquet");shutil.copyfile(graph["artifacts"][key]["path"],path);graph["artifacts"][key].update(record(path))
    graph_path=root/"graph.json";write(graph_path,graph)
    cnv_spec=root/"cnv.json"
    write(cnv_spec,{"format":cnv.INPUT,"run_id":"SYNTHETIC_CORE_WORKFLOW_CNV","candidates":record(scoped/"normal/candidates.parquet"),
        "patient_map":record(scoped.parent/"patient_map.tsv"),"source_config":record(config_path),
        "directional_success":record(directional/"SUCCESS.json"),"graph_root":str(static.parent),"graph_receipt":record(graph_path)})
    cnv_binding=cnv.run(cnv_spec,root/"cnv")
    assert read(cnv_binding)["corrected_inputs_available"],read(cnv_binding)["decision"]
    typed=read(prior_manifest)
    typed["source_receipt"]=record(root/"cnv/corrected_graph/GRAPH_INPUT_AUTHORITY_RECEIPT.json")
    typed["fold_coexpression_pattern"]=str(root/"cnv/corrected_graph/folds/fold_{fold}/train_coexpression")
    typed_path=root/"typed.json";write(typed_path,typed)
    from test_v32_current_binding_expansion import native_source,source
    edges,lineage=native_source();edges.to_parquet(root/"edges.parquet",index=False);lineage.to_parquet(root/"lineage.parquet",index=False)
    source([("LNC:Z","UNIPROT:A","predicted",.4)]).to_parquet(root/"rbns.parquet",index=False)
    write(root/"rbns_audit.json",{"format":"RBNS_SEQUENCE_ANALYSIS_AUDIT_V1","status":"PASS","annotation_id":"synthetic",
        "lncrna_binding_measured":False,"outputs":{"RBNS_TYPED_BINDING.parquet":record(root/"rbns.parquet")}})
    cfg=root/"config.yaml";cfg.write_text(yaml.safe_dump({"task_contract":{"rbp_evidence_mode":workflow.NATIVE_MODE,"graph_variant":"G2"},
        "runtime_profile":{"max_coverage_cycles":2,"device":"cpu","target_host":"149","mixed_precision":False},
        "primary_model":{"hidden_channels":96,"dropout":.2},"training_io":{}}))
    write(root/"hashes.json",{"synthetic_control":True})
    manifest=root/"core.json"
    write(manifest,{"format":workflow.INPUT,"release":"synthetic-core","typed_graph_manifest":record(typed_path),
        "candidates":record(scoped/"normal/candidates.parquet"),"fold_local_scope":record(scoped/"normal/SCOPE_RELEASE.json"),
        "local_cnv_success":record(root/"cnv/association/SUCCESS.json"),"validation_success":record(root/"cnv/validation/SUCCESS.json"),
        "authorization_hashes":record(root/"hashes.json"),"comparison_config":record(cfg),"native_cnv_workflow":record(cnv_binding),
        "expansion_sources":[{"source_id":"native","kind":"native_october","edges":record(root/"edges.parquet"),"lineage":record(root/"lineage.parquet")},
            {"source_id":"rbns","kind":"rbns_sequence","audit":record(root/"rbns_audit.json"),"binding":record(root/"rbns.parquet")}]})
    return root,manifest


def controls(patch,native_authority,scoped):
    patch_controls(patch,scoped.parent/"patient_map.tsv",native_authority[0]/"patients.json")
    for name in ("Adam","AdamW","SGD"):patch.setattr(torch.optim,name,forbidden)
    from sklearn.linear_model import LogisticRegression
    patch.setattr(LogisticRegression,"fit",forbidden)


@pytest.fixture(scope="module")
def completed(inputs,native_authority,scoped):
    root,manifest=inputs
    import scripts.prepare_v32_formal as formal
    with pytest.MonkeyPatch.context() as patch:
        controls(patch,native_authority,scoped)
        calls=[]
        def fixed_l1(frames,**kwargs):
            calls.append(len(frames["train"]))
            from frozen_formal_baseline import frozen_l1
            return frozen_l1(frames, **kwargs)
        patch.setattr(formal,"_fast_l1",fixed_l1)
        plan=workflow.preflight(manifest,root/"output")
        assert len(plan["commands"])==3 and not (root/"output").exists()
        binding=workflow.run(manifest,root/"output",allow_baseline_fitting=True)
        assert calls==[3]*5 and read(binding)["neural_checkpoints_created"] is False
    return root/"output"


def test_actual_workflow_loaders_sources_and_thirty_blocked_commands(completed,native_authority,scoped,monkeypatch):
    controls(monkeypatch,native_authority,scoped)
    _,binding=workflow.validate(record(completed/"NATIVE_CORE_BINDING.json"),completed)
    summary=read(binding["formal_summary"]["path"])
    assert summary["local_cnv_audit"]["cohort"]["candidate_rows"]==3
    assert summary["validation_a1_authority"]["cohort"]==summary["local_cnv_audit"]["cohort"]
    commands=read(completed/"TRAINING_COMMANDS.json")["commands"]
    assert len(commands)==30 and {c["fold"] for c in commands}==set(range(5))
    from cc_hhgt.v32.cli import main as pipeline
    for item in commands:
        assert pipeline(item["argv"][2:])==13
        args=item["argv"];task=pd.read_csv(args[args.index("--task-manifest")+1],sep="\t")
        assert task.status.eq("BLOCKED").all() and task.owner.eq("server149").all()
    assert not list(completed.rglob("TRAINING_APPROVAL.json")) and not list(completed.rglob("best_checkpoint.pt"))


@pytest.mark.parametrize("index",range(30))
def test_actual_generated_command_model_forward_preserves_parameters(completed,native_authority,scoped,monkeypatch,index):
    controls(monkeypatch,native_authority,scoped)
    from cc_hhgt.v32.training import load_prepared_artifact_from_authorized_handle,_runtime_graph_for_chunk
    from cc_hhgt.v32.native_graph_input import load_training_graph_inputs
    from cc_hhgt.gnn import build_model
    from cc_hhgt.v32.model import build_v32_cc_hhgt_residual
    from cc_hhgt.v32.integrated_model import module_state_sha256
    command=read(completed/"TRAINING_COMMANDS.json")["commands"][index];args=command["argv"];fold=command["fold"]
    manifest=Path(args[args.index("--input-manifest")+1]);config=yaml.safe_load(Path(args[args.index("--config")+1]).read_text())
    prepared=Path(read(manifest)["fold_inputs"][fold]["path"])
    payload,digest=load_prepared_artifact_from_authorized_handle(torch,manifest,fold=fold,prepared_path=prepared)
    payload,_=load_training_graph_inputs(torch,payload=payload,config=config,input_manifest_path=manifest,fold=fold,root=completed,
        prepared_sha256=digest,task_seed=20260726)
    encoder=build_model("cc_hhgt",payload["bundle"],payload["feature_dim"],payload["legacy_model_config"])
    model=build_v32_cc_hhgt_residual(encoder,hidden_channels=96,context_features=4,dropout=.2).eval()
    before=module_state_sha256(model)
    with torch.inference_mode():
        graph=_runtime_graph_for_chunk(payload["bundle"],0,device="cpu");batch=payload["train_batches"][0]
        result=model(graph,batch["candidate_batch"],batch["base_logit"],batch["conservation_context"])
    assert result["final_logit"].shape==(3,) and all(torch.isfinite(v).all() for v in result.values())
    assert module_state_sha256(model)==before


def test_execution_without_baseline_switch_never_calls_a_producer(tmp_path,monkeypatch):
    import scripts.prepare_v32_formal as formal
    monkeypatch.setattr(formal,"main",forbidden)
    with pytest.raises(ValueError,match="includes L1 fitting"):workflow.run(tmp_path/"missing",tmp_path/"output")
    assert not (tmp_path/"output").exists()


@pytest.mark.parametrize("fault",["config_mode","config_expanded","config_seed","cnv_binding","scope","unknown_field"])
def test_preflight_rejects_mixed_authorities(inputs,native_authority,scoped,tmp_path,monkeypatch,fault):
    controls(monkeypatch,native_authority,scoped);root,manifest=inputs;spec=copy.deepcopy(read(manifest))
    if fault in {"config_mode","config_expanded","config_seed"}:
        config=yaml.safe_load(Path(spec["comparison_config"]["path"]).read_text())
        if fault=="config_mode":config["task_contract"]["rbp_evidence_mode"]="C_TYPED_ENCODE_V2"
        elif fault=="config_expanded":config["training_io"]["binding_expansion_overlay_pattern"]="old"
        else:config["crossfit"]={"seed":7}
        path=tmp_path/"config.yaml";path.write_text(yaml.safe_dump(config));spec["comparison_config"]=record(path)
    elif fault=="cnv_binding":spec["native_cnv_workflow"]["sha256"]="0"*64
    elif fault=="scope":spec["fold_local_scope"]=record(scoped/"changed/SCOPE_RELEASE.json")
    else:spec["commands"]=["train"]
    copied=tmp_path/"bad.json";write(copied,spec)
    with pytest.raises((ValueError,RuntimeError)):workflow.preflight(copied,tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_changed_input_invalidates_completed_core(completed,inputs,native_authority,scoped,monkeypatch):
    controls(monkeypatch,native_authority,scoped)
    path=inputs[0]/"synthetic_correlated_cnv.tsv";original=path.read_bytes()
    try:
        path.write_bytes(original+b"changed")
        with pytest.raises(ValueError):workflow.validate(record(completed/"NATIVE_CORE_BINDING.json"),completed)
    finally:path.write_bytes(original)


@pytest.mark.parametrize("fault",["duplicate_fold","task_seed","task_fold"])
def test_acceptance_rejects_miswired_commands(completed,native_authority,scoped,monkeypatch,fault):
    controls(monkeypatch,native_authority,scoped)
    commands=read(completed/"TRAINING_COMMANDS.json")
    if fault=="duplicate_fold":
        path=completed/"TRAINING_COMMANDS.json";original=path.read_bytes()
        commands["commands"][1]=commands["commands"][0]
        write(path,commands)
    else:
        args=commands["commands"][0]["argv"];path=Path(args[args.index("--task-manifest")+1]);original=path.read_bytes()
        task=pd.read_csv(path,sep="\t");task.loc[0,"seed" if fault=="task_seed" else "patient_fold"]=7
        task.to_csv(path,sep="\t",index=False)
    try:
        with pytest.raises(ValueError):workflow.accept_inputs(completed,read(completed/"LAUNCH_PREFLIGHT.json"))
    finally:path.write_bytes(original)


def test_producer_failure_records_stage_without_success(inputs,native_authority,scoped,tmp_path,monkeypatch):
    controls(monkeypatch,native_authority,scoped)
    import scripts.prepare_v32_formal as formal
    def fail(*args):raise RuntimeError("synthetic formal preparation failure")
    monkeypatch.setattr(formal,"main",fail)
    with pytest.raises(RuntimeError,match="synthetic formal preparation failure"):
        workflow.run(inputs[1],tmp_path/"failed",allow_baseline_fitting=True)
    state=read(tmp_path/"failed/STAGE_STATUS.json")
    assert state["status"]=="FAILED" and state["failed_stage"]=="formal_preparation"
    assert [s["status"] for s in state["stages"]]==["PASS","FAILED"]
    assert not (tmp_path/"failed/SUCCESS.json").exists() and not (tmp_path/"failed/comparison").exists()
