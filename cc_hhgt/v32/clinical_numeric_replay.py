"""Reconstruct native Clinical calculations from their actually consumed inputs.

This uses the original feature/statistic implementations. It verifies numeric
consistency, not acquisition ancestry, optimizer history or independent effect.
"""
from pathlib import Path
import json

import pandas as pd

from .current_core_export import sha256
from .current_state_context import _assert_same_table
from .clinical_training import standardize_tcga_cdr_workbook, patient_fold_manifest
from .clinical_entity_training import input_path_sha256, analyse_entity_matrix
from .input_lineage import artifact_sha256


def contract_sources(value, kind, sources, root):
    common={"workbook","patient_folds","patient_authority_receipt","expression_root","activity_root"}
    required=common|({"core_manifest"} if kind=="RISK" else {"candidates","state_measurements","state_lineage"})
    if (not isinstance(value,dict) or set(value)!={"format","patient_authority_mode","files"}
            or value["format"]!="NATIVE_CLINICAL_"+kind+"_INPUTS_V1" or set(value["files"])!=required
            or value["patient_authority_mode"] not in {"current","frozen"}):
        raise ValueError("Clinical numeric replay requires complete native source roles")
    method=artifact_sha256 if kind=="RISK" else input_path_sha256
    paths={}
    for name,item in value["files"].items():
        path=(root/item["path"]).resolve()
        if method(path)!=item["sha256"]: raise ValueError("Clinical numeric source checksum differs: "+name)
        sources.records[str(path)]={"path":str(path),"sha256":item["sha256"],"hash_method":method.__name__}
        paths[name]=path
    from .patient_fold_authority import validate_patient_fold_authority, validate_frozen_v32_patient_fold_binding
    validate=validate_patient_fold_authority if value["patient_authority_mode"]=="current" else validate_frozen_v32_patient_fold_binding
    authority=validate(paths["patient_folds"],paths["patient_authority_receipt"])
    return paths,authority


def risk_sources(context,lineage,audit,root,artifacts):
    from scripts import run_v32_clinical_training as producer
    contract=lineage.get("source_contract"); paths,authority=contract_sources(contract,"RISK",context["sources"],root)
    if (authority!=audit.get("patient_fold_authority") or sha256(paths["core_manifest"])!=context["manifest"]["core_manifest"]["sha256"]):
        raise ValueError("Clinical numeric replay patient authority or core differs")
    declared={str(Path(r["path"]).resolve()):r["sha256"] for r in [*artifacts,*audit["artifacts"]]}
    for name in set(paths)-{"patient_authority_receipt"}:
        if declared.get(str(paths[name]))!=contract["files"][name]["sha256"]:
            raise ValueError("Clinical replay source differs from actual input inventory")
    config=lineage.get("training_config",{}); params=config.get("parameters",{})
    if (set(params)!={"seed","epochs","patience","n_bins","min_patients","min_events"}
            or any(type(v) is not int or (k!="seed" and v<1) for k,v in params.items())
            or config.get("runner_sha256")!=sha256(Path(producer.__file__))
            or config.get("implementation_sha256")!=sha256(Path(__file__).with_name("clinical_training.py"))
            or producer._canonical_hash(config)!=lineage.get("training_config_sha256")):
        raise ValueError("Clinical numeric replay lacks the recorded complete training settings")
    endpoints,covariates=standardize_tcga_cdr_workbook(paths["workbook"])
    folds,endpoints,covariates=producer.align_clinical_cohort(pd.read_csv(paths["patient_folds"],sep="\t"),endpoints,covariates)
    return dict(paths=paths,authority=authority,config=params,config_sha=lineage["training_config_sha256"],
        core=context["core"],folds=folds,endpoints=endpoints,covariates=covariates,verified_folds=[])


def patient_fold(replay,row,checkpoint,observed):
    from scripts import run_v32_clinical_training as producer
    paths=replay["paths"]; params=replay["config"]
    collected=producer.collect_fold_inputs(row["patient_fold"],replay["folds"],replay["covariates"],replay["endpoints"],
        paths["expression_root"],paths["activity_root"],paths["core_manifest"].parent,replay["core"],
        **{k:params[k] for k in ("n_bins","min_patients","min_events")})
    try: _assert_same_table(observed,collected["features"])
    except AssertionError as error: raise ValueError("Clinical raw patient feature projection replay differs") from error
    if (row.get("input_description")!=collected["description"] or row.get("training_config_sha256")!=replay["config_sha"]
            or row.get("seed")!=params["seed"]
            or any(row.get(k)!=collected["description"][k] for k in ("n_train","n_validation","n_test"))
            or checkpoint["endpoint_time_bins"]!=collected["edges"]
            or checkpoint["core_columns"]!=collected["core_columns"] or checkpoint["domain_columns"]!=collected["domain_columns"]):
        raise ValueError("Clinical replay endpoint bins, labels, splits or feature columns differ")
    replay["verified_folds"].append({"fold":row["patient_fold"],**collected["description"]})


def entity_sources(context,lineage,calculation,root,artifacts,risk,config):
    from scripts.run_v32_clinical_entity_training import _state_frame
    contract=lineage.get("source_contract"); paths,authority=contract_sources(contract,"ENTITY",context["sources"],root)
    if calculation.get("source_contract")!=contract or lineage.get("patient_fold_authority")!=authority:
        raise ValueError("Clinical entity source/authority contract differs")
    declared={str(Path(r["path"]).resolve()):r["sha256"] for r in artifacts}
    if any(declared.get(str(path))!=contract["files"][key]["sha256"] for key,path in paths.items()):
        raise ValueError("Clinical entity replay source differs from input inventory")
    for key in ("workbook","patient_folds","patient_authority_receipt"):
        if paths[key]!=risk["numeric_replay"]["paths"][key]: raise ValueError("Clinical risk/entity raw patient inputs differ")
    if sha256(paths["candidates"])!=context["candidate_sha256"]: raise ValueError("Clinical entity replay candidate scope differs")
    anchor=json.loads(paths["state_lineage"].read_text(encoding="utf-8"))
    if not str(anchor.get("analysis_version","")).startswith("CancerLncAtlas_V3.2") or anchor.get("old_checkpoint_loaded") is not False:
        raise ValueError("Clinical entity State generation anchor differs")
    minimum=config.get("minimums",{})
    required={"train_patients","train_events","validation_patients","validation_events","test_patients","test_events","train_measurements","folds_available"}
    if (set(minimum)!=required or any(type(v) is not int or v<1 for v in minimum.values())
            or type(config.get("block_size")) is not int or config["block_size"]<1):
        raise ValueError("Clinical entity replay lacks complete calculation settings")
    endpoints,covariates=standardize_tcga_cdr_workbook(paths["workbook"])
    folds=patient_fold_manifest(pd.read_csv(paths["patient_folds"],sep="\t"))
    return dict(paths=paths,authority=authority,config=config,endpoints=endpoints,covariates=covariates,
        folds=folds,state=_state_frame(paths["state_measurements"]),verified_partitions=[])


def entity_partition(replay,cancer,kind,ids,observed,part):
    from scripts.run_v32_clinical_entity_training import _measurement_matrix
    paths=replay["paths"]; cfg=replay["config"]
    cancer_folds=replay["folds"].loc[replay["folds"].cancer_id.astype(str).eq(cancer)].copy().drop_duplicates("patient_id").set_index("patient_id")
    wide=_measurement_matrix(kind,cancer,lncrna_root=paths["expression_root"],pathway_root=paths["activity_root"],
        state=replay["state"],entity_universe=ids)
    patients=cancer_folds.index.intersection(wide.index.astype(str),sort=False); wide=wide.reindex(index=patients)
    expected=analyse_entity_matrix(cancer_id=cancer,subject_type=kind,patient_ids=patients.tolist(),
        entity_ids=wide.columns.astype(str).tolist(),values=wide.to_numpy(float),
        fold_ids=cancer_folds.reindex(patients).patient_fold_id.to_numpy(int),endpoint_frame=replay["endpoints"],
        covariate_frame=replay["covariates"],block_size=cfg["block_size"],
        **{"min_"+k:v for k,v in cfg["minimums"].items() if k!="folds_available"})
    try: _assert_same_table(observed,expected)
    except AssertionError as error: raise ValueError("Clinical raw entity patient statistic replay differs") from error
    if part.get("patients")!=len(patients) or part.get("entities")!=len(ids):
        raise ValueError("Clinical entity replay patient/entity counts differ")
    replay["verified_partitions"].append({"cancer_id":cancer,"subject_type":kind,"patients":len(patients),"entities":len(ids),"fold_rows":len(expected)})


def summary(risk,entity):
    return {"format":"NATIVE_CLINICAL_NUMERIC_REPLAY_V1","patient_folds":risk["verified_folds"],
        "entity_partitions":entity["verified_partitions"],"same_implementation_replay":True,
        "source_acquisition_ancestry_verified":False,"cross_module_patient_closure_verified":False,
        "independent_predictive_performance_verified":False,"training_started":False}
