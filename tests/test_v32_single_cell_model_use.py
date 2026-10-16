"""Actual native SC episode loading and downstream isolation; no model fit."""
from pathlib import Path
import json
import socket
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_model_use import Inventory,INPUT_FORMAT,record,read,write,derive_manifest,materialize_current_model_use
from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.single_cell_signed_fusion import materialize_signed_sc_fusion,load_signed_sc_fusion,SIGNED_SC_EXPERTS
from cc_hhgt.v32.current_single_cell_model_use import inspect_signed_sources
from test_single_cell_hhgt_training import _write_episode_fixture,_tiny_settings,_record
from scripts import train_single_cell_hhgt_149 as trainer


def forbidden(*a,**kw):raise AssertionError("No fitting or model inference during source inspection")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    monkeypatch.setattr(trainer,"run_training",forbidden)
    monkeypatch.setattr(trainer,"train_fold_arm",forbidden)
    import cc_hhgt.v32.multimodal_fusion as fusion
    monkeypatch.setattr(fusion,"fit_residual_fusion_model",forbidden)
    monkeypatch.setattr(fusion,"train_pair_blocked_crossfit_fusion",forbidden)


def native_fixture(root,*,canonical_alias=False):
    torch.set_num_threads(1)
    root.mkdir(parents=True,exist_ok=True)
    episode_root,graph_root=_write_episode_fixture(root)
    raw=root/"source_study.parquet"
    pd.DataFrame({"pmid":["22223333"],"dataset_id":["GSE_SYNTHETIC_SOURCE_ONLY"],
        "patient_id":["PRIVATE_RAW_PATIENT"],"value":[1.]}).to_parquet(raw,index=False)
    upstream=root/"upstream.json";write(upstream,{"source_files":{"PRIVATE_RAW_DONOR_KEY":_record(raw)}})
    manifest_path=episode_root/"SC_HHGT_EPISODE_MANIFEST.json";manifest=read(manifest_path)
    static_records={}
    for fold in manifest["folds"]:
        for item in fold["episodes"]:
            path=Path(item["path"]);payload=torch.load(path,weights_only=True)
            static_path=Path(payload["static_graph_asset"]["path"])
            if str(static_path) not in static_records:
                static=torch.load(static_path,weights_only=True)
                static["node_identifiers"]={"lncRNA":["L0","L1"]}
                torch.save(static,static_path);static_records[str(static_path)]=_record(static_path)
            payload["static_graph_asset"]=static_records[str(static_path)]
            payload["source_receipts"]=[_record(upstream)]
            torch.save(payload,path);item.update(_record(path))
    write(manifest_path,manifest)
    output=root/"training";output.mkdir()
    settings=_tiny_settings(episode_root,graph_root,output)
    descriptors,lineage=trainer.load_episode_manifest(settings)
    write(output/"INPUT_LINEAGE.json",{"format":trainer.LINEAGE_FORMAT,"status":"PASS","target_host":"149",**lineage,
        "trainer_source":_record(Path(trainer.__file__)),
        "model_source":_record(Path(trainer.__file__).resolve().parents[1]/"cc_hhgt/v32/single_cell_hhgt.py")})
    write(output/"RUN_CONFIG.json",trainer._run_config(settings))
    write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,"synthetic_fixed_checkpoint_fixture":True})
    models=[];all_predictions=[];crosswalk=[]
    for fold,items in descriptors.items():
        for arm in trainer.ARMS:
            fold_root=output/arm/str(fold);fold_root.mkdir(parents=True)
            loaded=[trainer.load_episode(d,arm) for d in items]
            audits=[e.audit(arm) for e in loaded]
            # Instantiate the actual native architecture with fixed initial
            # parameters only. No optimizer, training loop or fitted result.
            sample=loaded[0]
            model=trainer._initialize_model(sample.graph,settings,sample.pair_features.shape[1],7301+fold)
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            item={"arm":arm,"outer_fold_id":fold,"seed":7301+fold,
                "initial_state_sha256":trainer._state_hash(state),"final_state_sha256":trainer._state_hash(state),
                "initial_hgt_state_sha256":trainer._state_hash(state,"encoder.layers."),
                "final_hgt_state_sha256":trainer._state_hash(state,"encoder.layers."),
                "model_config":trainer._model_config(settings,sample.pair_features.shape[1]),
                "continuous_feature_dims":{k:int(sample.graph.data[k].continuous_x.shape[1]) for k in trainer.NODE_TYPES},
                "edge_schema":[list(v) for v in sample.graph.edge_schema],"best_epoch":0,"best_validation_loss":.5}
            trainer.atomic_torch_save(fold_root/"checkpoint.pt",{"format":trainer.CHECKPOINT_FORMAT,**item,"state_dict":state,
                "synthetic_fixed_checkpoint_fixture":True,"optimizer_actually_run":False})
            query=[]
            for e in loaded:
                dataset=e.rows.dataset_id.iloc[0]
                for donor in (*e.outer_train_donors,*e.outer_query_donors):
                    crosswalk.append({"dataset_id":dataset,"query_donor_id":donor,"canonical_donor_id":"CANONICAL_"+donor})
                if e.descriptor.role=="query":
                    frame=e.rows.copy();frame["arm"]=arm;frame["outer_fold_id"]=fold;frame["episode_id"]=e.descriptor.episode_id
                    frame["predicted_rho"]=np.float32(.25 if arm=="with_lasso" else -.25);query.append(frame)
            prediction=pd.concat(query,ignore_index=True);prediction.to_parquet(fold_root/"predictions.parquet",index=False)
            all_predictions.append(prediction)
            report={"status":"PASS","arm":arm,"outer_fold_id":fold,"model_selection_source":"inner_validation_only",
                "outer_query_used_for_model_selection":False,"episode_audits":audits,"optimizer_actually_run":False}
            write(fold_root/"report.json",report);write(fold_root/"history.json",{"fixture_only":True})
            files={k:_record(fold_root/name) for k,name in {"checkpoint":"checkpoint.pt","predictions":"predictions.parquet",
                "report":"report.json","history":"history.json"}.items()}
            write(fold_root/"SUCCESS.json",{"format":trainer.SUCCESS_FORMAT,"state":"FOLD_TRAINING_COMPLETE",
                "target_host":"149","arm":arm,"outer_fold_id":fold,"files":files,"synthetic_only":True})
            models.append({**item,"files":files,"success":_record(fold_root/"SUCCESS.json")})
    pd.concat(all_predictions,ignore_index=True).to_parquet(output/"OOF_PREDICTIONS.parquet",index=False)
    write(output/"CHECKPOINT_MANIFEST.json",{"format":trainer.CHECKPOINT_MANIFEST_FORMAT,"status":"SUCCESS","models":models})
    trainer._write_checksums(output)
    files={k:_record(output/name) for k,name in {"run_config":"RUN_CONFIG.json","preflight":"LAUNCH_PREFLIGHT.json",
        "input_lineage":"INPUT_LINEAGE.json","checkpoint_manifest":"CHECKPOINT_MANIFEST.json",
        "oof_predictions":"OOF_PREDICTIONS.parquet","checksums":"SHA256SUMS.tsv"}.items()}
    write(output/"SUCCESS.json",{"format":trainer.SUCCESS_FORMAT,"state":"TRAINING_COMPLETE","target_host":"149",
        "outer_folds":5,"arms":list(trainer.ARMS),"trained_model_count":10,"files":files,"synthetic_only":True})
    validation=root/"validation.json";write(validation,{"format":"CANCERLNCATLAS_V32_SC_HHGT_INDEPENDENT_VALIDATION_V1",
        "status":"PASS","training_success_sha256":sha256(output/"SUCCESS.json"),
        "checkpoint_manifest_sha256":sha256(output/"CHECKPOINT_MANIFEST.json"),"synthetic_receipt_only":True})
    pointer=root/"pointer.json";write(pointer,{"status":"PASS_ACTIVATED_SC_HHGT",
        "estimand":"signed_within_donor_metacell_spearman_rho","split_policy":"donor_blocked_outer_oof_with_nested_inner_validation",
        "training_complete":True,"outer_folds":5,"primary_arm":"with_lasso","run_id":"SC_NATIVE_SYNTHETIC",
        "input_manifest":_record(manifest_path),"training_success":_record(output/"SUCCESS.json"),
        "oof_predictions":files["oof_predictions"],"independent_validation":_record(validation)})
    crosswalk=pd.DataFrame(crosswalk).drop_duplicates()
    if canonical_alias:
        # Distinct raw IDs pass the native loader but represent the same person.
        crosswalk.loc[crosswalk.query_donor_id.eq("C1_F0_D2"),"canonical_donor_id"]="CANONICAL_C1_F0_Q0"
    crosswalk.to_parquet(root/"crosswalk.parquet",index=False)
    keys=pd.DataFrame([("C1",f"L{i}",p) for i in range(80) for p in ("P0","P1")],columns=["cancer_id","lncrna_id","pathway_id"])
    keys.to_parquet(root/"candidates.parquet",index=False)
    materialize_signed_sc_fusion(pointer,root/"candidates.parquet",root/"crosswalk.parquet",root/"adapter")
    return root/"adapter/AUDIT.json",raw


def fusion_fixture(root,binding_path):
    from scripts.run_v32_current_multimodal_fusion import main
    import cc_hhgt.v32.multimodal_fusion as fusion
    native=read(binding_path);keys=pd.read_parquet(root/"candidates.parquet")
    primary=root/"primary.parquet";keys.assign(**{fusion.PRIMARY_COLUMN:.25,"analysis_version":fusion.ANALYSIS_VERSION}).to_parquet(primary,index=False)
    folds=[]
    for f in range(5):
        p=root/f"labels{f}.parquet";keys.assign(patient_fold_id=f,held_out_proxy_label=f%2).to_parquet(p,index=False)
        folds.append({"fold_id":f,"artifact":record(p)})
    parent=root/"primary.json";write(parent,{"status":"PASS","run_id":"current-G2","prediction":record(primary),
        "fold_prediction_sha256":[r["artifact"]["sha256"] for r in folds]})
    experts=[{"spec":asdict(s),"run_id":native["run_id"],"estimand":native["estimand"],
        "binding_audit":record(binding_path),"prediction":native["outputs"]["single_cell_signed_fusion"]} for s in SIGNED_SC_EXPERTS]
    manifest=root/"fusion.json";write(manifest,{"format":"CURRENT_MULTIMODAL_FUSION_MANIFEST_V1","core_run_id":"current-G2",
        "primary_prediction":record(primary),"primary_binding_audit":record(parent),"primary_folds":folds,"experts":experts})
    assert main(["--manifest",str(manifest),"--expected-core-run-id","current-G2","--output",str(root/"prepared")])==0
    prepared=read(root/"prepared/INPUT_AUDIT.json")
    frame,audit=fusion.build_pair_aggregated_fusion_frame_from_parquet(primary,[Path(r["artifact"]["path"]) for r in folds],
        {s:Path(experts[0]["prediction"]["path"]) for s in SIGNED_SC_EXPERTS})
    audit.update(core_run_id="current-G2",expert_bindings=prepared["expert_bindings"],synthetic_only=True)
    results=[]
    for endpoint in ("discovery","confidence"):
        model=fusion.ResidualFusionModel(endpoint,tuple(s.expert_id for s in SIGNED_SC_EXPERTS),(0.,0.),(0.,0.),1,"0"*64,"1"*64,0,0,
            feature_transforms=("identity_unit","identity_unit"))
        results.append(fusion.CrossfitFusionResult(endpoint,SIGNED_SC_EXPERTS,fusion.apply_residual_fusion_model(model,frame),
            pd.DataFrame({"synthetic_only":[True]}),(model,)*5,model,"SYNTHETIC_NOT_ESTIMATED","SYNTHETIC_NOT_ESTIMATED"))
    files=[primary,*[Path(r["artifact"]["path"]) for r in folds],binding_path,Path(experts[0]["prediction"]["path"])]
    result=fusion.materialize_multimodal_fusion_release(root/"fusion_release",frame,*results,
        source_declarations=[{**record(p),"source_role":"current_v32_oof_prediction","generation":"current_v32"} for p in files],
        training_run_id="V32-FUSION-SIGNED_SC_SOURCE_TEST",input_audit=audit)
    spec={"model_id":"fusion","producer_kind":"fusion_release","producer_binding":record(Path(result["binding_path"])),
        "run_id":"V32-FUSION-SIGNED_SC_SOURCE_TEST","score_column":"discovery_adjusted_probability"}
    source=root/"use.json";write(source,{"format":INPUT_FORMAT,"models":[spec]});return source


def inspect(binding,root):
    native,inputs,outputs,_=load_signed_sc_fusion(binding,expected_sha=sha256(binding))
    inventory=Inventory([],root)
    with inventory.context("FUSION_EXPERT:single_cell_positive"):
        inspect_signed_sources(inventory,native,inputs,outputs,"fusion","fusion.expert.single_cell_positive")
    return inventory


def test_native_checkpoint_episode_source_and_external_isolation(tmp_path,monkeypatch):
    binding,raw=native_fixture(tmp_path);source=fusion_fixture(tmp_path,binding)
    monkeypatch.setattr(trainer,"_initialize_model",forbidden)
    result=materialize_current_model_use(source,tmp_path/"inventory")
    use=read(Path(result["binding_path"]));usage=pd.read_parquet(use["outputs"]["usage"]["path"])
    deps=pd.read_parquet(use["outputs"]["dependencies"]["path"])
    assert len(deps.loc[deps.role.str.endswith(".checkpoint") & deps.dependency_context.str.startswith("FUSION_EXPERT:")])==20
    assert not usage[["model_id","source_event_id"]].duplicated().any()
    assert {"SC_HHGT_FIT","SC_HHGT_VALIDATION","SC_HHGT_QUERY","SC_HHGT_GRAPH_NODES"}.issubset(set(usage.use_view))
    assert usage.study_tokens_json.str.contains("22223333").any()
    assert {"FUSION_EXPERT:single_cell_positive","FUSION_EXPERT:single_cell_negative"}.issubset(set(usage.dependency_context))
    assert not usage.astype(str).apply(lambda c:c.str.contains("CANONICAL_|PRIVATE_RAW_PATIENT|C1_F0_D2",regex=True)).any().any()
    assert not deps.drop(columns=["path"]).astype(str).apply(lambda c:c.str.contains("PRIVATE_RAW_DONOR_KEY",regex=False)).any().any()
    assert not use["model_use_inventory_completeness_verified"] and not use["model_forwards_executed"]
    from cc_hhgt.v32.external_validation_current import materialize_current_external,CurrentExternalValidationQuery
    cross=tmp_path/"external_crosswalk.parquet"
    pd.DataFrame([("lncrna","new","LNC:NEW"),("cancer","c","C1"),("pathway","p","P0")],
        columns=["namespace","raw_id","canonical_id"]).to_parquet(cross,index=False)
    assay=tmp_path/"external_assay.parquet"
    pd.DataFrame({"record_id":["synthetic"],"lncrna":["new"],"cancer":["c"],"pathway":["p"],"label":[1],"pmid":["22223333"],
        "is_experimental":[True],"is_predicted":[False],"assay":["Synthetic assay"]}).to_parquet(assay,index=False)
    external=tmp_path/"external.json";write(external,{"format":"CURRENT_EXTERNAL_VALIDATION_INPUT_V2","models":read(source)["models"],
        "model_use_inventory":record(Path(result["binding_path"])),"identity_crosswalk":record(cross),
        "external_sources":[{"kind":"exact_assay","source_id":"sc_only_study","table":record(assay)}],
        "evaluation":{"primary_model":"fusion","cutoffs":[1],"lncrna_aggregation":"MAX","bootstrap_replicates":0,"seed":0}})
    released=materialize_current_external(external,tmp_path/"external")
    query=CurrentExternalValidationQuery(released["binding_path"],expected_sha=released["binding_sha256"])
    rows=query.query(table="isolation")["rows"]
    assert rows and all(r["study_overlap"] and not r["lncrna_overlap"] and not r["provenance_held_out"] for r in rows)
    rows=query.query(table="model_dependencies",filters={"dependency_context":"FUSION_EXPERT:single_cell_positive"})["rows"]
    assert rows and all("path" not in r for r in rows)


def test_canonical_donor_alias_leak_is_rejected(tmp_path,monkeypatch):
    binding,_=native_fixture(tmp_path,canonical_alias=True)
    monkeypatch.setattr(trainer,"_initialize_model",forbidden)
    with pytest.raises(ValueError,match="canonical donor aliases"):inspect(binding,tmp_path)


@pytest.mark.parametrize("fault",["upstream_file","checkpoint","episode","missing_training_binding"])
def test_signed_source_drift_and_historical_gap(tmp_path,monkeypatch,fault):
    binding,raw=native_fixture(tmp_path)
    monkeypatch.setattr(trainer,"_initialize_model",forbidden)
    native,inputs,outputs,_=load_signed_sc_fusion(binding,expected_sha=sha256(binding))
    if fault=="upstream_file":pd.DataFrame({"pmid":["99999999"]}).to_parquet(raw,index=False)
    elif fault=="checkpoint":
        p=tmp_path/"training/with_lasso/0/checkpoint.pt";p.write_bytes(p.read_bytes()+b"changed")
    elif fault=="episode":
        p=tmp_path/"episodes/outer_fold=0/fit.pt";p.write_bytes(p.read_bytes()+b"changed")
    else:
        pointer=read(inputs["pointer"]);pointer.pop("training_success");write(inputs["pointer"],pointer)
    inventory=Inventory([],tmp_path)
    if fault=="missing_training_binding":
        inspect_signed_sources(inventory,native,inputs,outputs,"fusion","sc")
        assert {r["reason"] for r in inventory.gaps}=={"SC_NATIVE_TRAINING_SUCCESS_NOT_BOUND"}
    else:
        with pytest.raises((ValueError,RuntimeError),match="mismatch|drift|SHA256|byte"):
            inspect_signed_sources(inventory,native,inputs,outputs,"fusion","sc")


@pytest.mark.parametrize("fault",["checkpoint_state","ablation_report_omission","query_identity"])
def test_rehashed_native_receipts_do_not_hide_semantic_drift(tmp_path,monkeypatch,fault):
    binding,_=native_fixture(tmp_path)
    monkeypatch.setattr(trainer,"_initialize_model",forbidden)
    native,inputs,outputs,_=load_signed_sc_fusion(binding,expected_sha=sha256(binding))
    root=tmp_path/"training";manifest=read(root/"CHECKPOINT_MANIFEST.json")
    arm="without_lasso" if fault=="ablation_report_omission" else "with_lasso"
    item=next(r for r in manifest["models"] if r["arm"]==arm and r["outer_fold_id"]==0)
    if fault=="checkpoint_state":
        path=Path(item["files"]["checkpoint"]["path"]);value=torch.load(path,weights_only=True)
        key=next(k for k,v in value["state_dict"].items() if v.numel())
        value["state_dict"][key].reshape(-1)[0]+=1;torch.save(value,path)
        item["files"]["checkpoint"]=_record(path)
    elif fault=="ablation_report_omission":
        path=Path(item["files"]["report"]["path"]);value=read(path);value["episode_audits"].pop()
        write(path,value);item["files"]["report"]=_record(path)
    else:
        path=Path(item["files"]["predictions"]["path"]);value=pd.read_parquet(path)
        value.loc[0,"query_donor_id"]="DIFFERENT_DONOR";value.to_parquet(path,index=False)
        item["files"]["predictions"]=_record(path)
        pd.concat([pd.read_parquet(r["files"]["predictions"]["path"]) for r in manifest["models"]],ignore_index=True).to_parquet(root/"OOF_PREDICTIONS.parquet",index=False)
    path=Path(item["success"]["path"]);value=read(path);value["files"]=item["files"];write(path,value);item["success"]=_record(path)
    write(root/"CHECKPOINT_MANIFEST.json",manifest)
    success=read(root/"SUCCESS.json")
    for key,name in (("checkpoint_manifest","CHECKPOINT_MANIFEST.json"),("oof_predictions","OOF_PREDICTIONS.parquet")):
        success["files"][key]=_record(root/name)
    write(root/"SUCCESS.json",success)
    pointer=read(inputs["pointer"]);validation_path=Path(pointer["independent_validation"]["path"])
    validation=read(validation_path);validation.update(training_success_sha256=sha256(root/"SUCCESS.json"),
        checkpoint_manifest_sha256=sha256(root/"CHECKPOINT_MANIFEST.json"));write(validation_path,validation)
    pointer.update(training_success=_record(root/"SUCCESS.json"),independent_validation=_record(validation_path),
        oof_predictions=_record(root/"OOF_PREDICTIONS.parquet"));write(inputs["pointer"],pointer)
    # Exercise the semantic inspector after repinning native hashes. A mere
    # hash-list check would accept these inconsistent records.
    with pytest.raises(ValueError,match="state digest|omits|query rows"):
        inspect_signed_sources(Inventory([],tmp_path),native,inputs,outputs,"fusion","sc")
