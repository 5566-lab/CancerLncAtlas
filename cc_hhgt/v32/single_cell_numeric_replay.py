"""Actual SC-HHGT input collection and frozen prediction reconstruction.

The separate native validator audits serialization, history and donor geometry.
This module additionally checks loaded graph/row tensors and frozen forwards.
It does not replay optimization or establish raw-study/biological independence.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from .cnv_input_sources import require_host
from .current_core_export import sha256,verified_path
from .formal_expression_sources import read,write,record

ROOT=Path(__file__).resolve().parents[2]
PREFLIGHT="NATIVE_SC_HHGT_INPUT_PREFLIGHT_V1"
FORMAT="NATIVE_SC_HHGT_NUMERIC_REPLAY_V1"
CODE=("scripts/train_single_cell_hhgt_149.py","cc_hhgt/v32/single_cell_hhgt.py",
      "cc_hhgt/v32/single_cell_numeric_replay.py","scripts/audit_single_cell_numeric_replay_149.py")


def tensor_record(value):
    array=np.ascontiguousarray(value.detach().cpu().numpy());digest=hashlib.sha256()
    if array.size:
        view=memoryview(array.reshape(-1)).cast("B")
        for start in range(0,len(view),8*1024*1024):digest.update(view[start:start+8*1024*1024])
    return {"shape":list(array.shape),"dtype":str(array.dtype),"sha256":digest.hexdigest()}


def episode_description(episode,arm):
    from scripts import train_single_cell_hhgt_149 as native
    tensors={k:getattr(episode,k) for k in ("lnc_index","pathway_index","pair_features","targets")}
    for node in native.NODE_TYPES:tensors["node."+node]=episode.graph.data[node].continuous_x
    for edge in episode.graph.edge_schema:
        for key in ("edge_index","edge_weight"):
            tensors[json.dumps(list(edge))+"."+key]=getattr(episode.graph.data[edge],key)
    return {**episode.audit(arm),"input_tensor_sha256":native.canonical_hash({k:tensor_record(v) for k,v in sorted(tensors.items())}),
            "row_order_sha256":native.canonical_hash(episode.rows.to_dict("records"))}


def source_inventory(lineage):
    from scripts import train_single_cell_hhgt_149 as native
    items=[lineage[k] for k in ("episode_manifest","graph_asset_manifest")]+lineage["episode_assets"]
    items.extend(item for bundle in lineage["shared_static_assets"] for item in bundle.values())
    records={}
    for item in items:
        path,bound=native.verify_file_record(item,base=ROOT,label="SC-HHGT actual input collection")
        records[str(path)]=bound
    return [records[k] for k in sorted(records)]


def collect_input_preflight(settings,descriptors,lineage):
    """Load one actual graph at a time for each arm/fold, without a neural model."""
    from scripts import train_single_cell_hhgt_149 as native
    require_host();native._validate_settings(settings);sources=source_inventory(lineage);parts=[]
    for fold,items in sorted(descriptors.items()):
        fit=[d for d in items if d.role=="fit"]
        if (len(fit)+settings.max_fit_episodes_per_epoch-1)//settings.max_fit_episodes_per_epoch>settings.epochs:
            raise ValueError("SC-HHGT epoch budget cannot cover every fit episode")
        for arm in native.ARMS:
            audits=[]
            for descriptor in items:
                loaded=native.load_episode(descriptor,arm)
                audits.append(episode_description(loaded,arm));del loaded
            parts.append({"outer_fold_id":fold,"arm":arm,"episodes":audits,
                "rows_by_role":{role:sum(a["row_count"] for a in audits if a["role"]==role) for role in native.ROLES}})
    if sources!=source_inventory(lineage):raise ValueError("SC-HHGT sources changed during input collection")
    return {"format":PREFLIGHT,"status":"PASS_READY","target_host":"149","training_started":False,
        "model_constructed":False,"model_forward_started":False,"run_config":native._run_config(settings),
        "input_lineage":lineage,"source_files":sources,"implementation_sha256":{p:sha256(ROOT/p) for p in CODE},
        "models":parts,"maximum_live_episode_graphs":1,"source_acquisition_ancestry_verified":False,
        "cross_module_patient_closure_verified":False}


def settings_from_config(root,config,lineage):
    from scripts import train_single_cell_hhgt_149 as native
    values={"episode_root":Path(lineage["episode_manifest"]["path"]).parent,
        "graph_assets_root":Path(lineage["graph_asset_manifest"]["path"]).parent,"output_root":Path(root),"seed":config["base_seed"]}
    for key in native.TrainingSettings.__dataclass_fields__:
        if key in values or key=="prepare_only":continue
        values[key]=config["architecture"][key] if key in config["architecture"] else config[key]
    settings=native.TrainingSettings(**values);native._validate_settings(settings)
    if native._run_config(settings)!=config:raise ValueError("SC-HHGT recorded configuration cannot be reconstructed")
    return settings


def replay_completed(root,*,expected_success_sha256):
    from scripts import train_single_cell_hhgt_149 as native,validate_single_cell_hhgt_training_149 as validator
    require_host();root=Path(root).resolve();success_path=verified_path({"path":str(root/"SUCCESS.json"),"sha256":expected_success_sha256},root)
    success=read(success_path)
    # This independent structural/history audit remains separate from the
    # same-loader numerical reconstruction below; neither proves optimizer use.
    independent=validator.validate_training(root)
    config=read(root/"RUN_CONFIG.json");lineage=read(root/"INPUT_LINEAGE.json")
    settings=settings_from_config(root,config,lineage);descriptors,actual_lineage=native.load_episode_manifest(settings)
    if any(lineage.get(k)!=v for k,v in actual_lineage.items()):raise ValueError("SC-HHGT native input lineage differs")
    preflight_path=verified_path(success["files"]["input_preflight"],root);preflight=read(preflight_path)
    if preflight!=collect_input_preflight(settings,descriptors,actual_lineage):
        raise ValueError("SC-HHGT prepared graph/row tensor replay differs")
    manifest=read(root/"CHECKPOINT_MANIFEST.json");models=[]
    for entry in manifest["models"]:
        fold=entry["outer_fold_id"];arm=entry["arm"];paths={k:verified_path(v,root) for k,v in entry["files"].items()}
        checkpoint=torch.load(paths["checkpoint"],map_location="cpu",weights_only=True)
        fit=next(d for d in descriptors[fold] if d.role=="fit");example=native.load_episode(fit,arm)
        model=native._initialize_model(example.graph,settings,example.pair_features.shape[1],entry["seed"]);del example
        model.load_state_dict(checkpoint["state_dict"],strict=True);model.requires_grad_(False);model.eval()
        before=native._state_hash(model.state_dict());observed=pd.read_parquet(paths["predictions"])
        maximum=0.;rows=0;validation_rows=0;validation_sum=0.;episode_ids=set()
        for descriptor in descriptors[fold]:
            if descriptor.role not in {"validation","query"}:continue
            episode=native.load_episode(descriptor,arm)
            if descriptor.role=="validation":
                _,loss=native._evaluate_episode(model,episode,settings.prediction_batch_size)
                validation_sum+=loss;validation_rows+=len(episode.rows)
            else:
                expected=native._prediction_frame(model,episode,arm,settings.prediction_batch_size)
                actual=observed.loc[observed.episode_id.eq(descriptor.episode_id)].reset_index(drop=True)
                if len(actual)!=len(expected) or set(actual)!=set(expected):
                    raise ValueError("SC-HHGT frozen query schema or row count differs")
                actual=actual[expected.columns]
                try:
                    pd.testing.assert_frame_equal(actual.isna(),expected.isna(),check_dtype=False)
                    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=0,atol=1e-6)
                except AssertionError as error:raise ValueError("SC-HHGT frozen query predictions differ") from error
                maximum=max(maximum,float(np.max(np.abs(actual.predicted_rho.to_numpy()-expected.predicted_rho.to_numpy()))))
                rows+=len(expected);episode_ids.add(descriptor.episode_id)
            del episode
        loss=validation_sum/validation_rows
        if not np.isclose(loss,checkpoint["best_validation_loss"],rtol=0,atol=1e-6):
            raise ValueError("SC-HHGT frozen inner-validation loss differs")
        if rows!=len(observed) or set(observed.episode_id)!=episode_ids:raise ValueError("SC-HHGT frozen query episode scope differs")
        if native._state_hash(model.state_dict())!=before:raise ValueError("SC-HHGT parameters changed during frozen replay")
        models.append({"arm":arm,"outer_fold_id":fold,"query_rows":rows,"validation_rows":validation_rows,
            "validation_loss":loss,"maximum_prediction_error":maximum,"checkpoint":record(paths["checkpoint"]),
            "predictions":record(paths["predictions"])})
        del model
    # Include the actual completion inventory and implementation, not just its
    # success flag, and verify nothing changed during the replay.
    validator.verify_checksum_manifest(root)
    if sha256(success_path)!=expected_success_sha256:raise ValueError("SC-HHGT completion changed during replay")
    source_inventory(actual_lineage)
    return {"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
        "model_forwards_executed":True,"parameters_unchanged":True,"same_implementation_replay":True,
        "training_success":record(success_path),"input_preflight":record(preflight_path),
        "native_validation":{k:v for k,v in independent.items() if k!="validated_at_utc"},"models":models,
        "implementation_sha256":{p:sha256(ROOT/p) for p in (*CODE,"scripts/validate_single_cell_hhgt_training_149.py")},
        "optimizer_execution_independently_verified":False,"source_acquisition_ancestry_verified":False,
        "cross_module_patient_closure_verified":False,"independent_predictive_performance_verified":False}


def audit(root,*,expected_success_sha256,output):
    require_host();root=Path(root).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError("Use a new SC-HHGT numerical audit directory")
    if root==output or root in output.parents:raise ValueError("SC-HHGT numerical audit must be outside immutable training results")
    output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,"training_root":str(root),"success_sha256":expected_success_sha256})
    try:
        value=replay_completed(root,expected_success_sha256=expected_success_sha256)
        path=output/"NUMERIC_REPLAY.json";write(path,value);return path
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise
