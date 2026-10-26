"""Actual SC-HHGT serializers and frozen forwards; optimization is forbidden."""
from dataclasses import asdict,replace
from pathlib import Path
import hashlib
import socket

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import single_cell_numeric_replay as replay
from cc_hhgt.v32.formal_expression_sources import read,write,record
from scripts import train_single_cell_hhgt_149 as native,validate_single_cell_hhgt_training_149 as validator
from test_single_cell_hhgt_training import _write_episode_fixture,_tiny_settings


def forbidden(*a,**k):raise AssertionError("SC-HHGT optimization is forbidden")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    monkeypatch.setattr(native,"fit_fold_model",forbidden)


def frozen_fit(settings,fold,arm,fit,validation,model,seed):
    """A marked serialization fixture, with actual forwards and no optimizer."""
    before=native._state_hash(model.state_dict(),"encoder.layers.")
    with torch.no_grad():
        next(p for name,p in model.named_parameters() if name.startswith("encoder.layers.")).add_(.025)
    after=native._state_hash(model.state_dict(),"encoder.layers.")
    audits={};receipts=[];available=sampled=0
    for i,descriptor in enumerate(fit):
        episode=native.load_episode(descriptor,arm);audits[descriptor.episode_id]=episode.audit(arm)
        indices=native._sample_rows(len(episode.rows),settings.max_rows_per_episode_step,seed+i)
        receipts.append({"episode_id":descriptor.episode_id,"available_rows":len(episode.rows),"sampled_rows":len(indices),
            "sampling_fraction":len(indices)/len(episode.rows),"sample_index_sha256":hashlib.sha256(indices.tobytes()).hexdigest()})
        available+=len(episode.rows);sampled+=len(indices)
    val_rows=0;val_sum=0.
    for descriptor in validation:
        episode=native.load_episode(descriptor,arm);_,loss=native._evaluate_episode(model,episode,settings.prediction_batch_size)
        val_sum+=loss;val_rows+=len(episode.rows);audits[descriptor.episode_id]=episode.audit(arm)
    loss=val_sum/val_rows
    history=[{"epoch":0,"optimizer_steps":len(fit),"mean_train_loss":.5,"validation_loss":loss,
        "validation_rows":val_rows,"selected_as_best":True,"stale_epochs":0,
        "available_training_rows":available,"sampled_training_rows":sampled,"sampling_coverage":sampled/available,
        "sampling_receipts":receipts,"fit_episode_population":len(fit),"selected_fit_episode_count":len(fit),
        "selected_fit_episode_ids_sha256":native.canonical_hash([d.episode_id for d in fit]),
        "cumulative_fit_episode_count":len(fit),"cumulative_fit_episode_coverage":1.,"full_fit_episode_coverage":True,
        "fit_episode_cycle":0,"fit_episode_cycle_length":1,"model_selection_eligible":True,
        "validation_skipped_until_full_fit_episode_coverage":False,
        **{key+"_"+stat:.1 for key in ("hgt_gradient_l1","weighted_projection_gradient_l1","continuous_projection_gradient_l1","decoder_gradient_l1") for stat in ("min","mean","max")},
        "synthetic_serialization_fixture":True,"optimizer_actually_run":False}]
    return {"history":history,"episode_audits":audits,"best_epoch":0,"best_validation":loss,
        "first_step":{"episode_id":fit[0].episode_id,"hgt_parameter_sha256_before":before,"hgt_parameter_sha256_after":after,
            "hgt_parameters_changed":True,"hgt_gradient_l1":.1,"synthetic_serialization_fixture":True,"optimizer_actually_run":False},
        "synthetic_serialization_fixture":True,"optimizer_actually_run":False}


def completed(root):
    episode_root,graph_root=_write_episode_fixture(root)
    settings=_tiny_settings(episode_root,graph_root,root/"training")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native,"fit_fold_model",frozen_fit)
        success=native.run_training(settings)
    return settings,success


def command(settings):
    return [part for key,value in asdict(settings).items() if key!="prepare_only" and value is not None for part in ("--"+key.replace("_","-"),str(value))]


def reseal(root):
    manifest=read(root/"CHECKPOINT_MANIFEST.json");frames=[]
    for model in manifest["models"]:
        frame=pd.read_parquet(model["files"]["predictions"]["path"]);frames.append(frame)
        for name,item in model["files"].items():model["files"][name]=native.file_record(Path(item["path"]))
        success_path=Path(model["success"]["path"]);success=read(success_path);success["files"]=model["files"]
        write(success_path,success);model["success"]=native.file_record(success_path)
    pd.concat(frames,ignore_index=True).to_parquet(root/"OOF_PREDICTIONS.parquet",index=False)
    write(root/"CHECKPOINT_MANIFEST.json",manifest);native._write_checksums(root)
    success=read(root/"SUCCESS.json")
    for key,item in success["files"].items():success["files"][key]=native.file_record(Path(item["path"]))
    write(root/"SUCCESS.json",success)


def test_actual_prepare_only_has_both_arms_all_roles_and_no_model(tmp_path,monkeypatch):
    episode_root,graph_root=_write_episode_fixture(tmp_path)
    settings=_tiny_settings(episode_root,graph_root,tmp_path/"prepared")
    monkeypatch.setattr(native,"_initialize_model",forbidden)
    monkeypatch.setattr(native,"_train_fold_job",forbidden)
    monkeypatch.setattr(torch.nn.Module,"__call__",forbidden)
    assert native.main([*command(settings),"--prepare-only"])==0
    ready=read(settings.output_root/"INPUT_PREFLIGHT.json")
    assert ready["format"]==replay.PREFLIGHT and len(ready["models"])==10
    assert not ready["training_started"] and not ready["model_constructed"] and not ready["model_forward_started"]
    for model in ready["models"]:
        assert model["rows_by_role"]=={"fit":8,"validation":4,"query":4}
        assert len(model["episodes"])==4
        assert all(len(e["input_tensor_sha256"])==64 for e in model["episodes"])
    for pattern in ("best_model_state.pt","SUCCESS.json","OOF_PREDICTIONS.parquet","CHECKPOINT_MANIFEST.json"):
        assert not list(settings.output_root.rglob(pattern))


def test_actual_completion_serializer_independent_validator_and_numeric_cli(tmp_path):
    from scripts.audit_single_cell_numeric_replay_149 import main
    settings,success=completed(tmp_path);root=settings.output_root
    assert success["trained_model_count"]==10 and success["oof_prediction_rows"]==40
    before={str(p):record(p)["sha256"] for p in root.rglob("*") if p.is_file()}
    output=tmp_path/"audit"
    assert main(["--training-root",str(root),"--training-success-sha256",record(root/"SUCCESS.json")["sha256"],"--output",str(output)])==0
    value=read(output/"NUMERIC_REPLAY.json")
    assert value["native_validation"]["validated_model_count"]==10 and len(value["models"])==10
    assert value["model_forwards_executed"] and value["parameters_unchanged"] and not value["training_started"]
    assert all(m["query_rows"]==4 and m["validation_rows"]==4 and m["maximum_prediction_error"]==0 for m in value["models"])
    assert not value["optimizer_execution_independently_verified"] and not value["independent_predictive_performance_verified"]
    assert before=={str(p):record(p)["sha256"] for p in root.rglob("*") if p.is_file()}
    for model in read(root/"CHECKPOINT_MANIFEST.json")["models"]:
        cp=torch.load(model["files"]["checkpoint"]["path"],weights_only=True)
        assert cp["synthetic_serialization_fixture"] and cp["optimizer_actually_run"] is False


@pytest.mark.parametrize("fault",["predicted_rho","residual","absolute_strength","validation_loss","preflight_tensor"])
def test_rehashed_consistent_containers_cannot_replace_actual_frozen_values(tmp_path,fault):
    settings,_=completed(tmp_path);root=settings.output_root;manifest=read(root/"CHECKPOINT_MANIFEST.json");model=manifest["models"][0]
    if fault in {"predicted_rho","residual","absolute_strength"}:
        path=Path(model["files"]["predictions"]["path"]);frame=pd.read_parquet(path)
        if fault=="predicted_rho":
            frame.predicted_rho=(frame.predicted_rho*.5+.1).astype(np.float32)
            frame.predicted_association_strength=frame.predicted_rho.abs();frame.residual=frame.rho-frame.predicted_rho
        else:frame.loc[0,"residual" if fault=="residual" else "predicted_association_strength"]+=.1
        frame.to_parquet(path,index=False)
    elif fault=="validation_loss":
        cp=Path(model["files"]["checkpoint"]["path"]);checkpoint=torch.load(cp,weights_only=True)
        checkpoint["best_validation_loss"]+=.1;checkpoint["history"][0]["validation_loss"]+=.1;torch.save(checkpoint,cp)
        for key in ("report","history"):
            path=Path(model["files"][key]["path"]);value=read(path);value["history"]=checkpoint["history"]
            if key=="report":value["best_validation_loss"]=checkpoint["best_validation_loss"]
            write(path,value)
        model["best_validation_loss"]=checkpoint["best_validation_loss"];write(root/"CHECKPOINT_MANIFEST.json",manifest)
    else:
        path=root/"INPUT_PREFLIGHT.json";value=read(path);value["models"][0]["episodes"][0]["input_tensor_sha256"]="0"*64;write(path,value)
    reseal(root)
    # The old structural/history validator accepts these internally consistent
    # containers. Numerical reconstruction must provide the missing rejection.
    assert validator.validate_training(root)["status"]=="PASS"
    with pytest.raises(ValueError,match="SC-HHGT (frozen|prepared graph)"):
        replay.replay_completed(root,expected_success_sha256=record(root/"SUCCESS.json")["sha256"])


@pytest.mark.parametrize("key,value",[("epochs",True),("hidden_channels",1.5),("learning_rate",float("nan")),("seed",-1)])
def test_invalid_preparation_settings_are_rejected_before_output(tmp_path,key,value):
    episode_root,graph_root=_write_episode_fixture(tmp_path);settings=_tiny_settings(episode_root,graph_root,tmp_path/"out")
    with pytest.raises(RuntimeError):native.run_training(replace(settings,prepare_only=True,**{key:value}))
    assert not settings.output_root.exists()


def test_insufficient_epoch_coverage_fails_without_training_or_ready(tmp_path):
    episode_root,graph_root=_write_episode_fixture(tmp_path);settings=_tiny_settings(episode_root,graph_root,tmp_path/"out")
    with pytest.raises(ValueError,match="cannot cover every fit episode"):
        native.run_training(replace(settings,prepare_only=True,max_fit_episodes_per_epoch=1))
    assert not (settings.output_root/"INPUT_PREFLIGHT.json").exists() and not (settings.output_root/"SUCCESS.json").exists()


def test_numerical_audit_cannot_write_into_training_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(replay,"replay_completed",forbidden)
    with pytest.raises(ValueError,match="outside immutable"):
        replay.audit(tmp_path,expected_success_sha256="0"*64,output=tmp_path/"audit")
    assert not (tmp_path/"audit").exists()
    monkeypatch.setattr(socket,"gethostname",lambda:"windows")
    with pytest.raises(RuntimeError,match="149"):
        replay.audit(tmp_path,expected_success_sha256="0"*64,output=tmp_path.parent/"outside")
