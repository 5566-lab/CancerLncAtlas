"""Actual raw prepare/collect/fusion entry points, explicit frozen fit substitutes."""
from copy import deepcopy
from pathlib import Path
import socket

import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import native_genomic_workflow as workflow
from cc_hhgt.v32 import genomic_partition_training as trainer
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_native_genomic_inputs import inputs


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    def forbidden(*a,**k):raise AssertionError("No optimizer or genomic prediction permitted")
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    from cc_hhgt.v32 import genomic_training
    for module in (trainer,genomic_training):
        monkeypatch.setattr(module,"_fit_head",forbidden);monkeypatch.setattr(module,"_predict_head",forbidden)
    import scripts.run_v32_current_multimodal_fusion as fusion
    monkeypatch.setattr(fusion,"train_pair_blocked_crossfit_fusion",forbidden)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root=tmp_path_factory.mktemp("genomic_workflow_inputs")
    genomic,args=inputs(root/"raw")
    manifest=root/"workflow.json"
    write(manifest,{"format":workflow.INPUT,"native_run_id":"v32-genomic-synthetic","genomic_input":record(genomic),
        "resource_limits":{"adapter_budget":{"max_declared_adapter_peak_rss_bytes":10**12,"max_memmap_bytes":10**8,"max_staging_disk_bytes":10**8},
                           "downstream_budget":{"max_declared_process_rss_bytes":10**12,"minimum_host_available_memory_bytes":1,"minimum_host_free_disk_bytes":1}},
        "config":{"min_pair_callable":2}})
    return manifest,args


def complete(prepared, root):
    """Write actual seals/completion serializer using explicit fixture predictions."""
    from test_v32_native_auxiliary_fusion import fixture
    from test_v32_native_partitioned_genomic import partitioned
    _,_,value=workflow.validate_preparation(record(prepared),root)
    files=value["files"];keys=pd.read_parquet(files["candidates"]["path"])
    path,_=fixture(root,core_override=(Path(files["core_embedding_manifest"]["path"]),keys,value["core_run_id"]))
    manifest=read(path);manifest.update(candidate_authority=files["candidates"],patient_folds=files["patient_folds"]);write(path,manifest)
    lp=root/"native/LINEAGE.json";lineage=read(lp)
    ready=read(prepared.parent/"head_preflight/HEAD_PREFLIGHT.json")
    lineage["input_artifacts"]=ready["input_lineage_audit"]["artifacts"];write(lp,lineage)
    partitioned(root,existing=path,streaming_success=prepared.parent/"streaming/SUCCESS.json",resource_success=prepared.parent/"resources/SUCCESS.json")
    lineage=read(lp);lineage.update(configuration=value["config"])
    checkpoints=read(root/"native/CHECKPOINT_MANIFEST.json")
    for row in checkpoints["records"]:
        counts=next(r for r in ready["folds"] if (r["modality"],r["patient_fold"])==(row["modality"],row["patient_fold"]))
        row.update(train_rows=counts["train_rows"],validation_rows=counts["validation_rows"])
    output=prepared.parent/"training";output.mkdir()
    success=read(root/"native/TRAINING_SUCCESS.json")
    success.update(lineage_path=str(output/"LINEAGE.json"),checkpoint_manifest_path=str(output/"CHECKPOINT_MANIFEST.json"))
    trainer.write_partitioned_completion(output,output,source_audit=ready["input_lineage_audit"],split_policy=read(root/"native/SPLIT_POLICY.json"),
        checkpoint_manifest=checkpoints,resource_audit=read(root/"native/RESOURCE_AUDIT.json"),lineage=lineage,training_success=success)
    return output/"TRAINING_SUCCESS.json"


def test_native_chain_genomic_prepare_collect(source,tmp_path):
    from native_chain_fixture import exercise
    result=exercise("genomic",source[0],tmp_path,lambda prepared:complete(prepared,tmp_path/"fixture"))
    assert result["candidate_rows"]==6 and result["fold_rows"]==60


def repin_completed(path):
    value=read(path);lp=Path(value["lineage_path"]);lineage=read(lp)
    for name,item in lineage["completion_records"].items():lineage["completion_records"][name]=record(Path(item["path"]))
    write(lp,lineage);value["lineage_sha256"]=record(lp)["sha256"]
    value["checkpoint_manifest_sha256"]=lineage["completion_records"]["CHECKPOINT_MANIFEST.json"]["sha256"];write(path,value)


def test_actual_prepare_collect_cli_and_fusion_keep_mutable_arrays_separate_from_preparation(source,tmp_path):
    from scripts.run_native_genomic_workflow import main
    from cc_hhgt.v32.native_auxiliary_fusion import load_native_auxiliary
    args=["prepare","--manifest",str(source[0]),"--output",str(tmp_path/"prepare")]
    assert main([*args,"--plan"])==0 and not (tmp_path/"prepare").exists()
    assert main(args)==0
    prepared=tmp_path/"prepare/PREPARED.json";_,binding,value=workflow.validate_preparation(record(prepared),tmp_path)
    assert not (prepared.parent/"training").exists()
    command=read(prepared.parent/"TRAINING_COMMAND.json")
    from scripts.train_v32_partitioned_genomic_heads import _parser
    parsed=_parser().parse_args(command["argv"][3:])
    assert not parsed.prepare_only and parsed.output_root==str(prepared.parent/"training")
    assert not command["automatically_executed"] and command["requires_separate_explicit_execution"]
    with pytest.raises(ValueError,match="cannot dispatch"):workflow.execute_preflight(command["argv"],tmp_path/"forbidden.log")
    assert not (tmp_path/"forbidden.log").exists()
    mutable=list((prepared.parent/"resources/memmaps").glob("*.npy"))
    initial={p.name:record(p)["sha256"] for p in mutable}
    completed=complete(prepared,tmp_path/"fixture")
    assert initial!={p.name:record(p)["sha256"] for p in mutable}
    assert binding==workflow.validate_preparation(record(prepared),tmp_path)[1]
    before={str(p):record(p)["sha256"] for p in completed.parent.rglob("*") if p.is_file()}
    collected=tmp_path/"collected"
    assert main(["collect","--preparation",str(prepared),"--preparation-sha256",record(prepared)["sha256"],
        "--completed",str(completed),"--completed-sha256",record(completed)["sha256"],"--output",str(collected)])==0
    _,result=workflow.validate(record(collected/"NATIVE_GENOMIC_BINDING.json"),tmp_path)
    assert result["candidate_rows"]==6 and result["fold_rows"]==60
    assert not result["training_started"] and not result["formal_increment_established"]
    assert before=={str(p):record(p)["sha256"] for p in completed.parent.rglob("*") if p.is_file()}
    frame,adapter=load_native_auxiliary(result["fusion_binding"]["path"],expected_sha=result["fusion_binding"]["sha256"],expected_core_run_id="current-G2")
    assert len(frame)==6 and record(prepared) in adapter["sources"] and result["collection_audit"] in adapter["sources"]
    # Actual fusion preparation; primary scores/labels are explicit substitutes.
    import scripts.run_v32_current_multimodal_fusion as fusion
    from cc_hhgt.v32.multimodal_fusion import ANALYSIS_VERSION,PRIMARY_COLUMN
    keys=pd.read_parquet(source[1].candidates);primary=tmp_path/"primary.parquet"
    keys.assign(**{PRIMARY_COLUMN:.25,"analysis_version":ANALYSIS_VERSION}).to_parquet(primary,index=False)
    folds=[]
    for fold in range(5):
        p=tmp_path/f"labels_{fold}.parquet";keys.assign(patient_fold_id=fold,held_out_proxy_label=fold%2).to_parquet(p,index=False)
        folds.append({"fold_id":fold,"artifact":record(p)})
    primary_binding=tmp_path/"primary.json";write(primary_binding,{"status":"PASS","run_id":"current-G2","prediction":record(primary),
        "fold_prediction_sha256":[r["artifact"]["sha256"] for r in folds]})
    spec=tmp_path/"fusion.json";write(spec,{"format":"CURRENT_MULTIMODAL_FUSION_MANIFEST_V1","core_run_id":"current-G2",
        "primary_prediction":record(primary),"primary_binding_audit":record(primary_binding),"primary_folds":folds,"experts":read(result["fusion_experts"]["path"])})
    assert fusion.main(["--manifest",str(spec),"--expected-core-run-id","current-G2","--output",str(tmp_path/"fusion")])==0
    assert not read(tmp_path/"fusion/PREPARED.json")["eligible_for_current_fitting"]


@pytest.mark.parametrize("fault",["run","missing_limit","unknown_limit","boolean_limit","unknown_config","fractional_config","threshold","genomic_hash"])
def test_invalid_manifest_cannot_create_preparation(source,tmp_path,fault):
    spec=deepcopy(read(source[0]))
    if fault=="run":spec["native_run_id"]="../escape"
    elif fault=="missing_limit":spec["resource_limits"]["adapter_budget"].pop("max_memmap_bytes")
    elif fault=="unknown_limit":spec["resource_limits"]["adapter_budget"]["extra"]=1
    elif fault=="boolean_limit":spec["resource_limits"]["downstream_budget"]["minimum_host_free_disk_bytes"]=True
    elif fault=="unknown_config":spec["config"]["unknown"]=1
    elif fault=="fractional_config":spec["config"]["seed"]=1.2
    elif fault=="threshold":spec["config"]["cnv_event_threshold"]=.6
    elif fault=="genomic_hash":spec["genomic_input"]["sha256"]="0"*64
    path=tmp_path/"input.json";write(path,spec)
    with pytest.raises((ValueError,RuntimeError)):workflow.prepare(path,tmp_path/"prepared")
    assert not (tmp_path/"prepared").exists()


@pytest.mark.parametrize("fault",["command","raw_audit","budget","split"])
def test_repinned_preparation_controls_fail_semantic_recheck(source,tmp_path,fault):
    prepared=workflow.prepare(source[0],tmp_path/"prepared");binding=read(prepared)
    if fault=="command":
        path=prepared.parent/"TRAINING_COMMAND.json";value=read(path);value["argv"][-1]="999";write(path,value);binding["training_command"]=record(path)
    else:
        path=prepared.parent/{"raw_audit":"RAW_CNV_AUDIT.json","budget":"RESOURCE_BUDGET.json","split":"head_preflight/HEAD_PREFLIGHT.json"}[fault]
        value=read(path)
        if fault=="raw_audit":value["source"]["sha256"]="0"*64
        elif fault=="budget":value["scope"]["candidate_rows"]+=1
        else:value["folds"][0]["split_policy"]["patient_counts"]["train"]+=1
        write(path,value)
        binding["consumed_sources"]=[record(path) if r["path"]==str(path) else r for r in binding["consumed_sources"]]
    write(prepared,binding)
    with pytest.raises((ValueError,RuntimeError)):workflow.validate_preparation(record(prepared),tmp_path)


@pytest.mark.parametrize("fault",["config","run","stage","sample_count","input","location","stage_only"])
def test_completion_must_match_actual_prepared_run_and_inputs(source,tmp_path,fault):
    prepared=workflow.prepare(source[0],tmp_path/"prepared");completed=complete(prepared,tmp_path/"fixture")
    lp=completed.parent/"LINEAGE.json";value=read(lp)
    if fault=="config":value["configuration"]["seed"]+=1
    elif fault=="run":value["training_run_id"]="v32-other"
    elif fault=="stage":value["resource_stage_success_path"]=str(tmp_path/"another/SUCCESS.json")
    elif fault=="sample_count":
        p=completed.parent/"CHECKPOINT_MANIFEST.json";cp=read(p);cp["records"][0]["train_rows"]+=1;write(p,cp)
    elif fault=="input":
        p=completed.parent/"INPUT_LINEAGE_AUDIT.json";audit=read(p);audit["artifacts"].pop();write(p,audit)
    write(lp,value);repin_completed(completed)
    if fault=="location":p=tmp_path/"TRAINING_SUCCESS.json";p.write_bytes(completed.read_bytes());completed=p
    elif fault=="stage_only":completed=prepared.parent/"resources/SUCCESS.json"
    with pytest.raises((ValueError,RuntimeError)):workflow.collect(record(prepared),record(completed),tmp_path/"result")
    assert not (tmp_path/"result/NATIVE_GENOMIC_BINDING.json").exists()


@pytest.mark.parametrize("fault",["audit_changed","audit_repinned","audit_missing","experts"])
def test_actual_consumer_requires_matching_collection_audit_and_expert_declarations(source,tmp_path,fault):
    from cc_hhgt.v32.native_auxiliary_fusion import materialize_native_auxiliary
    prepared=workflow.prepare(source[0],tmp_path/"prepared");completed=complete(prepared,tmp_path/"fixture")
    result=workflow.collect(record(prepared),record(completed),tmp_path/"result");value=read(result)
    if fault=="experts":
        p=Path(value["fusion_experts"]["path"]);experts=read(p);experts[0]["estimand"]="WRONG";write(p,experts)
        value["fusion_experts"]=record(p);write(result,value)
        with pytest.raises(ValueError):workflow.validate(record(result),tmp_path)
    else:
        p=Path(value["adapter_input"]["path"]);spec=read(p);audit=Path(value["collection_audit"]["path"])
        if fault=="audit_missing":spec.pop("workflow_audit")
        else:
            content=read(audit);content["completed"]["sha256"]="0"*64;write(audit,content)
            if fault=="audit_repinned":spec["workflow_audit"]=record(audit)
        write(p,spec)
        with pytest.raises(ValueError):materialize_native_auxiliary(p,tmp_path/"reconsume",expected_core_run_id="current-G2")
