"""Reconstruct native fusion inputs, twelve frozen models and full serialization.

This is same-implementation numerical replay, not optimization or independent
performance estimation. Primary validation labels retain development-only scope.
"""
from dataclasses import fields
from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd

from . import multimodal_fusion as native
from .cnv_input_sources import require_host
from .current_core_export import verified_path,sha256
from .formal_expression_sources import read,record

FORMAT="NATIVE_FUSION_NUMERIC_REPLAY_V1"


def same_frame(actual,expected,label):
    try:
        pd.testing.assert_frame_equal(actual.isna(),expected.isna(),check_dtype=False)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
    except AssertionError as error:raise ValueError("Fusion numerical replay differs: "+label) from error


def frozen_result(frame,audit,checkpoint,endpoint):
    """Use the original crossfit bookkeeping with an immutable model provider."""
    config=native.ResidualFusionConfig(**audit["training_config"]["parameters"]);config.validate()
    specs=tuple(native.ExpertSpec(**{**audit["expert_bindings"][name]["spec"],
        "endpoint_roles":tuple(audit["expert_bindings"][name]["spec"]["endpoint_roles"])}) for name in audit["expert_order"])
    models=checkpoint["fold_models"][endpoint]
    if len(models)!=5:raise ValueError("Fusion requires all five saved fold models")
    visited=[]
    def provider(train,validation,expert_ids,*,endpoint,config,seed,feature_transforms):
        index=seed-config.seed
        if index not in {*range(5),10000} or index in visited:raise ValueError("Fusion model seed or fold reused")
        visited.append(index);value=checkpoint[endpoint] if index==10000 else models[index]
        args={f.name:value[f.name] for f in fields(native.ResidualFusionModel) if f.name in value}
        for key in ("expert_ids","centres","weights","feature_transforms"):
            if key in args:args[key]=tuple(args[key])
        model=native.ResidualFusionModel(**args)
        if (json.loads(json.dumps(model.to_dict()))!=value or model.endpoint!=endpoint or model.seed!=seed
                or model.expert_ids!=tuple(expert_ids) or model.feature_transforms!=tuple(feature_transforms)
                or model.clip_probability!=config.clip_probability
                or type(model.optimiser_steps) is not int or not 0<=model.optimiser_steps<=config.max_steps):
            raise ValueError("Fusion saved model identity or configuration differs")
        train=native._subsample(train,config.max_train_rows,seed)
        validation=native._subsample(validation,config.max_validation_rows,seed+1)
        _,_,centres=native._expert_matrix(train,expert_ids,None,config.clip_probability,feature_transforms)
        weights=np.asarray(model.weights)
        if (not np.isfinite(centres).all() or not np.array_equal(centres,np.asarray(model.centres))
                or not np.isfinite(weights).all() or (weights<0).any()):
            raise ValueError("Fusion frozen training centres or weights differ")
        initial=np.random.default_rng(seed).uniform(.0025,.0125,size=len(expert_ids)).astype(float)
        if (model.initial_parameter_sha256!=native._parameter_sha(initial)
                or model.final_parameter_sha256!=native._parameter_sha(weights)):
            raise ValueError("Fusion saved parameter digests differ")
        matrix,_,_=native._expert_matrix(validation,expert_ids,centres,config.clip_probability,feature_transforms)
        probability=native._sigmoid(native._logit(validation.primary_probability.to_numpy(float),config.clip_probability)+matrix@weights)
        loss=native._logloss(validation.fusion_target.to_numpy(float),probability,config.clip_probability)+config.l2*float(np.square(weights).sum())
        if not np.isclose(loss,model.validation_logloss,rtol=0,atol=1e-12):
            raise ValueError("Fusion frozen inner-validation loss differs")
        return model
    result=native._pair_blocked_fusion(frame,specs,endpoint=endpoint,config=config,model_provider=provider)
    if set(visited)!={*range(5),10000}:raise ValueError("Fusion saved model coverage is incomplete")
    return result


def replay_release(binding_path,*,expected_sha,replay_expert_models=True):
    from scripts import run_v32_current_multimodal_fusion as runner
    require_host();path=verified_path({"path":str(binding_path),"sha256":expected_sha},Path.cwd());binding=read(path)
    if (binding.get("format")!="CC_HHGT_V3_2_MULTIMODAL_FUSION_BINDING_V1"
            or binding.get("status")!="SUCCESS_SECONDARY_FUSION"):
        raise ValueError("Native fusion numerical replay requires a completed producer")
    success_path=path.parent/"SUCCESS.json";success=read(success_path)
    if success.get("binding_sha256")!=expected_sha or success.get("training_run_id")!=binding["training_run_id"]:
        raise ValueError("Native fusion completion is inconsistent")
    paths={k:verified_path(v,path.parent) for k,v in binding["artifacts"].items()}
    audit=read(paths["fusion_input_audit"]);contract=audit.get("native_input_contract",{})
    if contract.get("format")!="CURRENT_FUSION_NATIVE_INPUTS_V1":raise ValueError("Fusion has no native input reconstruction contract")
    manifest=verified_path(contract["manifest"],path.parent)
    if read(manifest).get("fusion_run_id")!=binding["training_run_id"]:raise ValueError("Fusion manifest run identity differs")
    with tempfile.TemporaryDirectory(prefix="native_fusion_replay_") as folder:
        frame,expected_audit,declarations,rules=runner.collect_inputs(manifest,expected_core_run_id=audit["core_run_id"],
            temp_directory=Path(folder)/"duckdb",require_fit_ready=True,replay_expert_models=replay_expert_models)
        if audit!=expected_audit:raise ValueError("Fusion original input collection or configuration differs")
        same_frame(pd.read_parquet(paths["fusion_input_private"]),frame,"native input matrix/targets")
        checkpoint=read(paths["checkpoint"])
        results={endpoint:frozen_result(frame,audit,checkpoint,endpoint) for endpoint in ("discovery","confidence")}
        release=native.materialize_multimodal_fusion_release(Path(folder)/"expected",frame,results["discovery"],results["confidence"],
            source_declarations=declarations,training_run_id=binding["training_run_id"],input_audit=audit,rule_evidence_binding=rules)
        expected=read(release["binding_path"])
        if set(binding["artifacts"])!=set(expected["artifacts"]):raise ValueError("Fusion output inventory differs")
        for key,item in expected["artifacts"].items():
            if Path(item["path"]).suffix==".parquet":same_frame(pd.read_parquet(paths[key]),pd.read_parquet(item["path"]),key)
            elif read(paths[key])!=read(item["path"]):raise ValueError("Fusion serialized metadata differs: "+key)
            for field in ("public","rows"):
                if binding["artifacts"][key][field]!=item[field]:raise ValueError("Fusion artifact semantics differ")
        if {k:v for k,v in binding.items() if k!="artifacts"}!={k:v for k,v in expected.items() if k!="artifacts"}:
            raise ValueError("Fusion native binding differs from reconstructed serialization")
    for item in [*binding["artifacts"].values(),*binding["source_inputs"],record(success_path),contract["manifest"]]:verified_path(item,path.parent)
    verified_path({"path":str(path),"sha256":expected_sha},path.parent)
    return {"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
        "binding":record(path),"native_input_manifest":record(manifest),"models_replayed":12,"candidate_rows":len(frame),
        "native_inputs_reconstructed":True,"frozen_inner_validation_loss_replayed":True,
        "frozen_oof_predictions_metrics_and_final_outputs_replayed":True,"source_expert_neural_replay_requested":replay_expert_models,
        "same_implementation_replay":True,"optimizer_execution_independently_verified":False,
        "independent_predictive_performance_verified":False,"source_ancestry_complete":False,
        "implementation_sha256":{str(Path(__file__).relative_to(Path(__file__).resolve().parents[2])):sha256(Path(__file__))}}
