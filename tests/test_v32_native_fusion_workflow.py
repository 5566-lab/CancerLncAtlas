"""Actual native inputs, crossfit bookkeeping and publication; no optimizer."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import native_fusion_workflow as workflow,multimodal_fusion as fusion
from cc_hhgt.v32.fusion_numeric_replay import replay_release
from cc_hhgt.v32.formal_expression_sources import read,write,record
from scripts import run_v32_current_multimodal_fusion as runner

NATIVE_CROSSFIT=fusion.train_pair_blocked_crossfit_fusion


def forbidden(*a,**k):raise AssertionError("Fusion fitting is forbidden")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    monkeypatch.setattr(fusion,"fit_residual_fusion_model",forbidden)
    monkeypatch.setattr(runner,"train_pair_blocked_crossfit_fusion",forbidden)


def frozen_fit(train,validation,expert_ids,*,endpoint,config,seed,feature_transforms):
    """Explicit fixed nonzero weights, native centres/validation, zero fit steps."""
    train=fusion._subsample(train,config.max_train_rows,seed)
    validation=fusion._subsample(validation,config.max_validation_rows,seed+1)
    _,_,centres=fusion._expert_matrix(train,expert_ids,None,config.clip_probability,feature_transforms)
    matrix,_,_=fusion._expert_matrix(validation,expert_ids,centres,config.clip_probability,feature_transforms)
    weights=np.arange(1,len(expert_ids)+1,dtype=float)*.08
    initial=np.random.default_rng(seed).uniform(.0025,.0125,len(expert_ids)).astype(float)
    probability=fusion._sigmoid(fusion._logit(validation.primary_probability.to_numpy(float),config.clip_probability)+matrix@weights)
    loss=fusion._logloss(validation.fusion_target.to_numpy(float),probability,config.clip_probability)+config.l2*float(np.square(weights).sum())
    return fusion.ResidualFusionModel(endpoint,tuple(expert_ids),tuple(centres),tuple(weights),seed,
        fusion._parameter_sha(initial),fusion._parameter_sha(weights),0,loss,config.clip_probability,tuple(feature_transforms))


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    from test_v32_current_core_prediction import current_fixture
    import test_v32_current_binding_expansion as fixtures
    from cc_hhgt.v32.current_core_prediction import materialize_current_predictions
    from cc_hhgt.v32.current_primary_fusion import materialize_primary_fusion_inputs
    from scripts.export_v32_current_core_embeddings import main as export
    from cc_hhgt.v32.single_cell_signed_fusion import materialize_signed_sc_fusion
    from cc_hhgt.v32.native_auxiliary_fusion import materialize_native_auxiliary
    from test_v32_native_auxiliary_fusion import fixture as auxiliary_fixture
    from cc_hhgt.v32.current_analysis_web import COMPONENTS
    root=tmp_path_factory.mktemp("native_fusion_workflow")
    identities=["LNC:Z",*[f"LNC:L{i}" for i in range(79)]]
    original=fixtures.verified_payload
    def bundle(fold=0):
        expression=pd.DataFrame({"cancer_id":"BRCA","lncrna_id":identities,"detection_rate":.8})
        coexp=pd.DataFrame({"cancer_id":"BRCA","lncrna_id":identities,"gene_id":"G1","rho":.5,"source_split":"train","edge_outer_fold":fold})
        authority=fixtures.build_formal_graph_authority([
            fixtures.materialize_expressed_in(expression,candidate_pairs=expression),
            fixtures.materialize_signed_coexpression(coexp,outer_fold=fold),
            fixtures.materialize_signed_membership(pd.DataFrame({"gene_id":["G1","G2"],"pathway_id":"P1","weight":[1.,-1.]})),
            fixtures.materialize_pathway_hierarchy(pd.DataFrame({"pathway_id":["P1"],"pathway_family_id":["F1"],"weight":[1.]})),
            fixtures.materialize_protein_gene_encoding(pd.DataFrame({"protein_id":["UNIPROT:Z"],"gene_id":["G1"],"weight":[1.]})),
            fixtures.materialize_typed_lnc_protein_binding(fixtures.source([(lnc,"UNIPROT:Z","eclip",1.) for lnc in identities]),require_global_binding=True)],outer_fold=fold)
        return fixtures.build_variant_runtime_bundle(authority,"G2",edge_chunk_size=128)
    def payload(fold=0):
        value=original(fold);batch=value["train_batches"][0]
        for key,item in list(batch.items()):
            if torch.is_tensor(item):batch[key]=item.repeat((80,)+(1,)*(item.ndim-1))
        batch["candidate_batch"]={"l":torch.arange(80),"p":torch.zeros(80,dtype=torch.long),"c":torch.zeros(80,dtype=torch.long)}
        batch["base_logit"]=torch.linspace(-.8,.8,80)
        batch["proxy_label"]=(torch.arange(80)%2).float()
        value["validation_batches"]=[deepcopy(batch)]
        validation=value["validation_batches"][0];validation["proxy_label"]=(torch.arange(80).add(fold)%2).float()
        validation["association_available"][0]=False
        validation["association_available"][1]=fold==0
        value["input_scope"]["candidate_rows_per_split"]=80
        return value
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam","AdamW","SGD"):patch.setattr(torch.optim,name,forbidden)
        patch.setattr(fixtures,"current_bundle",bundle);patch.setattr(fixtures,"verified_payload",payload)
        manifest=current_fixture(root/"core_inputs")
        spec=read(manifest);keys=pd.DataFrame({"cancer_id":"BRCA","lncrna_id":sorted(identities),"pathway_id":"P1"})
        candidate=Path(spec["candidate_authority"]["path"]);keys.assign(pathway_family_id="F1").to_parquet(candidate,index=False)
        scope=Path(spec["lncrna_scope"]["path"]);keys[["cancer_id","lncrna_id"]].assign(shared_or_local_scope="shared").to_parquet(scope,index=False)
        spec["candidate_authority"]=record(candidate);spec["lncrna_scope"]=record(scope);write(manifest,spec)
        primary=materialize_current_predictions(manifest,root/"primary",expected_run_id=spec["core_run_id"])
        primary_fusion=materialize_primary_fusion_inputs(primary,root/"primary_fusion",expected_sha=record(primary)["sha256"],expected_core_run_id=spec["core_run_id"])
        export_input=root/"EXPORT.json";write(export_input,{**spec,"format":"CURRENT_CORE_EXPORT_INPUT_V1"})
        assert export(["--manifest",str(export_input),"--expected-core-run-id",spec["core_run_id"],"--output",str(root/"core_export")])==0
        auxiliary,_=auxiliary_fixture(root/"genomic",core_override=(root/"core_export/CORE_EMBEDDING_MANIFEST.json",keys,spec["core_run_id"]))
        auxiliary_binding=materialize_native_auxiliary(auxiliary,root/"genomic_fusion",expected_core_run_id=spec["core_run_id"])
    donors=pd.DataFrame({"dataset_id":"S1","query_donor_id":[f"D{f}" for f in range(5)],"canonical_donor_id":[f"P{f}" for f in range(5)]})
    cross=root/"cross.parquet";donors.to_parquet(cross,index=False)
    rows=[]
    for fold in range(5):
        for i,row in keys.iloc[1:].iterrows():rows.append({**row.to_dict(),"dataset_id":"S1","cell_key":"T","query_donor_id":f"D{fold}",
            "outer_fold_id":fold,"arm":"with_lasso","predicted_rho":float(np.sin(i*.2+fold*.1)*.8)})
    prediction=root/"SC.parquet";pd.DataFrame(rows).to_parquet(prediction,index=False)
    validation=root/"SC_VALIDATION.json";write(validation,{"status":"PASS","synthetic_fixed_predictions":True,"optimizer_actually_run":False})
    pointer=root/"SC_POINTER.json";write(pointer,{"status":"PASS_ACTIVATED_SC_HHGT","run_id":"SC_SYNTHETIC_FUSION_PARENT",
        "estimand":"signed_within_donor_metacell_spearman_rho","split_policy":"donor_blocked_outer_oof_with_nested_inner_validation",
        "training_complete":True,"outer_folds":5,"primary_arm":"with_lasso","oof_predictions":record(prediction),"independent_validation":record(validation),
        "synthetic_fixed_predictions":True,"optimizer_actually_run":False})
    materialize_signed_sc_fusion(pointer,candidate,cross,root/"sc_fusion")
    sc_context=root/"SC_CONTEXT.json";write(sc_context,{"format":"CURRENT_SINGLE_CELL_CONTEXT_INPUT_V1",
        "selected_analysis":{"workflow_ref":"selected_analysis"},"native_binding":record(root/"sc_fusion/AUDIT.json")})
    components={k:{"status":"NOT_PRODUCED","reason":"Outside synthetic fusion workflow"} for k in COMPONENTS}
    components["single_cell"]={"status":"BUILD","input_manifest":record(sc_context)}
    workflow_input=root/"WORKFLOW.json"
    write(workflow_input,{"format":workflow.INPUT,"fusion_run_id":"V32-FUSION-NATIVE_WORKFLOW_TEST",
        "primary_fusion":record(primary_fusion),"experts":[record(auxiliary_binding),record(root/"sc_fusion/AUDIT.json")],
        "selected_score_column":"discovery_adjusted_probability","selection_rationale":"Explicit frozen synthetic fusion registration; no winner or benefit claim",
        "geneset":{"min_members":1,"max_members":80,"score_min":0.},
        "subtype":{"score_column":"selected_score","min_members":1,"min_cancers":2,"bootstrap_replicates":2},
        "training_config":{"seed":79,"max_steps":20,"patience":2,"batch_size":20,"max_train_rows":30,"max_validation_rows":10},
        "publication_components":components})
    return workflow_input


def complete(prepared):
    command=read(prepared.parent/"EXECUTION_COMMAND.json")["argv"]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner,"train_pair_blocked_crossfit_fusion",NATIVE_CROSSFIT)
        patch.setattr(fusion,"fit_residual_fusion_model",frozen_fit)
        assert runner.main(command[3:])==0
    return prepared.parent/"training/release/MULTIMODAL_FUSION_BINDING.json"


def test_original_cli_preparation_and_frozen_completion_to_selected_web(source,tmp_path,monkeypatch):
    from scripts.run_native_fusion_workflow import main
    args=["prepare","--manifest",str(source),"--output",str(tmp_path/"prepared")]
    assert main([*args,"--plan"])==0 and not (tmp_path/"prepared").exists()
    with monkeypatch.context() as patch:
        patch.setattr(fusion,"apply_residual_fusion_model",forbidden)
        assert main(args)==0
    prepared=tmp_path/"prepared/PREPARED.json";_,_,value,frame,audit=workflow.validate_preparation(record(prepared),tmp_path)
    assert len(frame)==80 and len(audit["expert_bindings"])==4 and audit["eligible_for_current_fitting"]
    assert frame.fusion_target_available.sum()==79
    assert frame.loc[~frame.single_cell_positive_available,"single_cell_positive_probability"].isna().all()
    with pytest.raises(ValueError,match="cannot dispatch"):workflow.execute_preflight(value["execution_command"]["argv"],tmp_path/"bad.log")
    assert not (tmp_path/"bad.log").exists() and not (prepared.parent/"training").exists()
    completed=complete(prepared);before={str(p):record(p)["sha256"] for p in completed.parent.parent.rglob("*") if p.is_file()}
    assert main(["collect","--preparation",str(prepared),"--preparation-sha256",record(prepared)["sha256"],
        "--completed",str(completed),"--completed-sha256",record(completed)["sha256"],"--output",str(tmp_path/"result")])==0
    result=read(tmp_path/"result/NATIVE_FUSION_BINDING.json");numerical=read(result["numeric_replay"]["path"])
    assert numerical["models_replayed"]==12 and numerical["candidate_rows"]==80
    assert numerical["frozen_oof_predictions_metrics_and_final_outputs_replayed"] and not numerical["training_started"]
    from cc_hhgt.v32.current_analysis_web import CurrentAnalysisWebStore
    store=CurrentAnalysisWebStore(result["web_binding"]["path"],expected_sha=result["web_binding"]["sha256"])
    assert store.binding["component_status"]["single_cell"]["status"]=="AVAILABLE"
    published=read(result["publication"]["path"])
    sc=read(next(s["binding"]["path"] for s in published["stages"] if s["stage"]=="single_cell"))
    assert sc["used_in_selected_score"] and not sc["native_model_forwards_replayed"]
    from cc_hhgt.v32.selected_genesets import SelectedAnalysisQuery
    query=SelectedAnalysisQuery(result["selected_analysis"]["path"],expected_sha=result["selected_analysis"]["sha256"])
    members=pd.read_parquet(query.paths["geneset_members"])
    assert members.score_source_run_id.eq(value["fusion_input"]["fusion_run_id"]).all()
    assert members.score_scientific_status.eq("DEVELOPMENT_ONLY_PRIMARY_VALIDATION_REUSED").all()
    assert before=={str(p):record(p)["sha256"] for p in completed.parent.parent.rglob("*") if p.is_file()}


@pytest.mark.parametrize("field,value",[("seed",True),("max_steps",2.5),("l2",float("nan")),("tolerance",-1.),("increment_logloss_delta",float("inf"))])
def test_invalid_configuration_before_any_output(source,tmp_path,field,value):
    spec=read(source);spec["training_config"][field]=value;path=tmp_path/"bad.json";write(path,spec)
    with pytest.raises(fusion.MultimodalFusionError):workflow.prepare(path,tmp_path/"output")
    assert not (tmp_path/"output").exists()


@pytest.mark.parametrize("fault",["experts","run","component","score","rationale","geneset","subtype","unsupported_result_ref"])
def test_invalid_workflow_contract(source,tmp_path,fault):
    spec=read(source)
    if fault=="experts":spec["experts"].append(spec["experts"][0])
    elif fault=="run":spec["fusion_run_id"]="V32-FUSION-"+"A"*100
    elif fault=="component":spec["publication_components"].pop("drug")
    elif fault=="score":spec["selected_score_column"]="primary_probability"
    elif fault=="geneset":spec["geneset"]["max_members"]=0
    elif fault=="subtype":spec["subtype"]["min_cancers"]=1
    elif fault=="unsupported_result_ref":
        component=read(spec["publication_components"]["single_cell"]["input_manifest"]["path"])
        component["native_binding"]={"result_ref":"core_export"}
        template=tmp_path/"context.json";write(template,component)
        spec["publication_components"]["single_cell"]["input_manifest"]=record(template)
    else:spec["selection_rationale"]=""
    path=tmp_path/"bad.json";write(path,spec)
    with pytest.raises(ValueError):workflow.prepare(path,tmp_path/"out")
    assert not (tmp_path/"out").exists()


def reseal(path):
    value=read(path)
    for item in value["artifacts"].values():
        target=Path(item["path"]);item.update(record(target),bytes=target.stat().st_size)
    write(path,value);success=read(path.parent/"SUCCESS.json");success["binding_sha256"]=record(path)["sha256"];write(path.parent/"SUCCESS.json",success)
    parent=read(path.parent.parent/"SUCCESS.json");parent["binding_sha256"]=record(path)["sha256"];write(path.parent.parent/"SUCCESS.json",parent)


@pytest.mark.parametrize("fault",["oof","metric","public_contribution","validation_loss","centre","input","status"])
def test_rehashed_fusion_result_corruption_rejected_by_native_consumer(source,tmp_path,fault):
    prepared=workflow.prepare(source,tmp_path/"prepared");completed=complete(prepared);binding=read(completed)
    key={"oof":"discovery_oof_private","metric":"crossfit_metrics","public_contribution":"secondary_scores", "input":"fusion_input_private"}.get(fault,"checkpoint")
    target=Path(binding["artifacts"][key]["path"])
    if key=="checkpoint":
        value=read(target)
        if fault=="validation_loss":value["fold_models"]["discovery"][0]["validation_logloss"]+=.1
        elif fault=="centre":value["discovery"]["centres"][0]+=.1
        else:
            target=Path(binding["artifacts"]["module_lineage"]["path"]);value=read(target);value["discovery_performance_outcome"]="INCREMENT"
        write(target,value)
    else:
        frame=pd.read_parquet(target)
        column={"oof":"discovery_adjusted_probability","metric":"adjusted_logloss","public_contribution":"discovery_single_cell_positive_logit_contribution","input":"primary_probability"}[fault]
        frame.loc[2,column]+=.02;frame.to_parquet(target,index=False)
    reseal(completed)
    from cc_hhgt.v32.external_validation_current import producer_scores
    spec={"producer_kind":"fusion_release","model_id":"test","producer_binding":record(completed),
        "run_id":binding["training_run_id"],"score_column":"discovery_adjusted_probability"}
    with pytest.raises(ValueError,match="Fusion .*differ"):producer_scores(spec,tmp_path)


def test_rehashed_prepared_matrix_and_execution_command_rejected(source,tmp_path):
    prepared=workflow.prepare(source,tmp_path/"prepared");binding=read(prepared)
    command=Path(binding["execution_command"]["path"]);value=read(command);value["argv"][0]="other-python";write(command,value)
    binding["execution_command"]=record(command);write(prepared,binding)
    with pytest.raises(ValueError,match="command differs"):workflow.validate_preparation(record(prepared),tmp_path)


@pytest.mark.parametrize("fault",["frame","audit"])
def test_prepared_values_cannot_be_changed_by_rehashing(source,tmp_path,fault):
    prepared=workflow.prepare(source,tmp_path/"prepared");binding=read(prepared)
    if fault=="frame":
        path=Path(binding["input_frame"]["path"]);frame=pd.read_parquet(path);frame.loc[2,"fusion_target"]+=.1;frame.to_parquet(path,index=False)
        binding["input_frame"]=record(path)
        ready=Path(binding["ready"]["path"]);value=read(ready);value["input_sha256"]=record(path)["sha256"];write(ready,value);binding["ready"]=record(ready)
    else:
        path=Path(binding["input_audit"]["path"]);value=read(path);value["training_config"]["parameters"]["seed"]+=1;write(path,value);binding["input_audit"]=record(path)
    write(prepared,binding)
    with pytest.raises(ValueError,match="Fusion .*differ"):workflow.validate_preparation(record(prepared),tmp_path)


def test_new_fusion_model_use_has_no_neural_forward_and_keeps_unresolved_ancestry(source,tmp_path,monkeypatch):
    from cc_hhgt.v32.current_model_use import materialize_current_model_use,INPUT_FORMAT
    prepared=workflow.prepare(source,tmp_path/"prepared");completed=complete(prepared);binding=read(completed)
    path=tmp_path/"MODEL_USE.json";write(path,{"format":INPUT_FORMAT,"models":[{"model_id":"frozen-native-fusion",
        "producer_kind":"fusion_release","producer_binding":record(completed),"run_id":binding["training_run_id"],
        "score_column":"discovery_adjusted_probability"}]})
    monkeypatch.setattr(torch.nn.Module,"__call__",forbidden)
    result=materialize_current_model_use(path,tmp_path/"use");value=read(result["binding_path"])
    assert not value["neural_model_forwards_executed"] and not value["model_use_inventory_completeness_verified"]


def test_host_gate_and_existing_outputs(source,tmp_path,monkeypatch):
    prepared=workflow.prepare(source,tmp_path/"prepared")
    with pytest.raises(FileExistsError):workflow.prepare(source,prepared.parent)
    monkeypatch.setattr(socket,"gethostname",lambda:"windows")
    with pytest.raises(RuntimeError,match="149"):workflow.plan(source,tmp_path/"out")
