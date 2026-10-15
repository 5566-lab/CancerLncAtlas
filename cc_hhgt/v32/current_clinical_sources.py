"""Read native Clinical heads and entity Cox results against a selected core.

Patient risk and entity survival associations are distinct estimands. This
reader never broadcasts patient risk to lncRNA/pathway pairs and never treats
one minus FDR as a model posterior. No optimizer or raw-cohort fitting is used.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clinical_training import (CLINICAL_CHECKPOINT_FORMAT, CLINICAL_ENDPOINTS,
    build_clinical_head, patient_fold_manifest, reshape_endpoint_logits, survival_risk_probability)
from .clinical_entity_training import (CALCULATION_FORMAT, HISTORICAL_STATE_IDS, SUBJECT_SPECS,
    input_path_sha256, json_sha256, summarise_oof_associations, validate_entity_clinical_contract)
from .current_core_export import sha256, validate_current_export
from .current_core_prediction import FORMAT as CORE_PREDICTION_FORMAT
from .current_state_context import _assert_same_table
from .evidence_rule_sources import read_table
from .input_lineage import artifact_sha256, _canonical_json_sha256
from .integrated_model import module_state_sha256
from .native_auxiliary_fusion import Sources
from .selected_genesets import SelectedAnalysisQuery, audit_selected_analysis

INPUT_FORMAT = "CURRENT_CLINICAL_CONTEXT_INPUT_V1"
ENTITY_KEYS = ["cancer_id", "subject_type", "subject_id", "clinical_endpoint"]
RISK_KEYS = ["cancer_id", "subject_id", "clinical_endpoint"]


def _json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def _boolean(frame, column):
    if not pd.api.types.is_bool_dtype(frame[column]) or frame[column].isna().any():
        raise ValueError("Clinical Boolean column is invalid: "+column)


def _inputs(records, sources, root, *, method):
    result = []
    for record in records:
        path = Path(record["path"])
        path = (root/path).resolve() if not path.is_absolute() else path.resolve()
        if method(path) != record["sha256"]: raise ValueError("Native Clinical input SHA256 mismatch")
        sources.records[str(path)] = {"path":str(path),"sha256":record["sha256"],"hash_method":method.__name__}
        result.append({**record,"path":str(path)})
    return result


def _patient_folds(path):
    data = read_table(path)
    values = pd.to_numeric(data.patient_fold_id, errors="raise")
    if pd.api.types.is_bool_dtype(values) or not values.isin(range(5)).all():
        raise ValueError("Clinical folds must be integers 0..4")
    result = patient_fold_manifest(data)
    if set(result.patient_fold_id) != set(range(5)) or result.groupby("patient_id").patient_fold_id.nunique().gt(1).any():
        raise ValueError("Clinical explicit patient identity crosses or omits folds")
    return result.sort_values(["cancer_id","patient_id"]).reset_index(drop=True)


def selected_sources(manifest_path):
    path = Path(manifest_path).resolve(); manifest = _json(path); sources = Sources()
    if manifest.get("format") != INPUT_FORMAT or not all(manifest.get(k) for k in ("core_run_id","patient_risk_run_id","entity_run_id")):
        raise ValueError("Invalid current Clinical manifest")
    selected_path = sources.bind(manifest["selected_analysis"],path.parent)
    selected = SelectedAnalysisQuery(selected_path,expected_sha=manifest["selected_analysis"]["sha256"])
    audit_selected_analysis(selected_path,expected_sha=selected.binding_sha256)
    score = _json(selected.score_binding_path)
    _, primary = sources.document(score["sources"]["reference_exact_producer"]["producer_binding"],selected.score_binding_path.parent)
    if primary.get("format") != CORE_PREDICTION_FORMAT or primary.get("training_run_id") != manifest["core_run_id"]:
        raise ValueError("Clinical selected reference is not the requested native current core")
    core_path = sources.bind(manifest["core_manifest"],path.parent)
    core,_,_ = validate_current_export(core_path,expected_run_id=manifest["core_run_id"])
    folds = primary["ensemble_source_records"]
    if len(folds)!=5 or {r["patient_fold"] for r in folds}!=set(range(5)):
        raise ValueError("Clinical selected reference lacks five core folds")
    for row in folds:
        parent = core["folds"][str(row["patient_fold"])]
        if row["checkpoint_sha256"]!=parent["checkpoint_sha256"] or row["core_parameter_sha256"]!=parent["core_parameter_sha256"]:
            raise ValueError("Clinical and selected scores use different core checkpoints")
        sources.bind(parent["training_receipt"],core_path.parent)
        for export in parent["exports"].values(): sources.bind(export,core_path.parent)
    candidate = sources.bind(primary["input_artifacts"][0],path.parent)
    return dict(manifest=manifest,root=path.parent,sources=sources,selected=selected,core=core,
        candidates=pd.read_parquet(candidate),candidate_sha256=sha256(candidate))


def patient_risk_sources(context):
    """Verify five native heads; replay forwards when native feature hashes exist."""
    import torch
    sources = context["sources"]; manifest = context["manifest"]
    lp,lineage = sources.document(manifest["patient_risk_lineage"],context["root"])
    if (lineage.get("module_id")!="clinical" or lineage.get("training_status")!="SUCCESS" or lineage.get("folds")!=5
            or lineage.get("training_run_id")!=manifest["patient_risk_run_id"]
            or lineage.get("core_run_id")!=manifest["core_run_id"]
            or lineage.get("core_manifest_sha256")!=manifest["core_manifest"]["sha256"]
            or lineage.get("labels_in_public_output") is not False):
        raise ValueError("Native Clinical patient risk has another run/core or is incomplete")
    for key in ("private_head_trained_from_scratch","core_parameters_frozen"):
        if lineage.get(key) is not True: raise ValueError("Clinical private-head isolation missing")
    for key in ("old_checkpoint_loaded","old_predictions_used_as_features","old_rankings_used_as_outputs"):
        if lineage.get(key) is not False: raise ValueError("Clinical historical result use is not excluded")
    artifacts = _inputs(lineage["input_artifacts"],sources,lp.parent,method=artifact_sha256)
    split = [r for r in artifacts if r["artifact_kind"]=="split_manifest"]
    if len(split)!=1: raise ValueError("Clinical risk lacks one explicit patient authority")
    patients = _patient_folds(split[0]["path"])
    workbook = [r for r in artifacts if r["artifact_kind"]=="training_label"]
    if len(workbook)!=1: raise ValueError("Clinical risk lacks its original outcome workbook")
    audit_path = lp.parent/"INPUT_LINEAGE_AUDIT.json"; audit = _json(audit_path)
    # Native runner appends patient_fold_authority after hashing its input audit.
    if (audit.get("status")!="PASS" or audit.get("lineage_sha256")!=lineage["input_manifest_sha256"]
            or audit.get("input_count")!=len(audit.get("artifacts",[]))
            or any(r.get("status")!="PASS" for r in audit.get("artifacts",[]))
            or _canonical_json_sha256({k:v for k,v in audit.items() if k not in {"lineage_sha256","patient_fold_authority"}})!=audit["lineage_sha256"]):
        raise ValueError("Native Clinical input audit is inconsistent")
    sources.bind({"path":str(audit_path),"sha256":sha256(audit_path)},lp.parent)
    _inputs(audit["artifacts"],sources,lp.parent,method=artifact_sha256)
    tasks = lineage["task_lineage"]
    if (len(tasks)!=5 or {r["patient_fold"] for r in tasks}!=set(range(5))
            or json_sha256(tasks)!=lineage.get("checkpoint_manifest_sha256")):
        raise ValueError("Native Clinical lacks five bound fold receipts")
    all_public=[]; bins=[]; replayed=[]
    for row in sorted(tasks,key=lambda r:r["patient_fold"]):
        fold=row["patient_fold"]; parent=context["core"]["folds"][str(fold)]
        if (type(fold) is not int or row.get("status")!="SUCCESS"
                or row.get("training_run_id")!=manifest["patient_risk_run_id"]
                or row.get("core_manifest_sha256")!=manifest["core_manifest"]["sha256"]
                or row.get("input_lineage_sha256")!=lineage["input_manifest_sha256"]
                or row.get("core_parameters_unchanged") is not True
                or row.get("private_head_trained_from_scratch") is not True
                or row.get("core_parameters_frozen") is not True or row.get("source_checkpoint_sha256") is not None):
            raise ValueError("Native Clinical fold receipt mismatch")
        cp_path=sources.bind({"path":row["checkpoint_path"],"sha256":row["checkpoint_sha256"]},lp.parent)
        checkpoint=torch.load(cp_path,map_location="cpu",weights_only=True)
        if (checkpoint.get("checkpoint_format")!=CLINICAL_CHECKPOINT_FORMAT or checkpoint.get("patient_fold")!=fold
                or checkpoint.get("training_run_id")!=manifest["patient_risk_run_id"]
                or checkpoint.get("core_run_id")!=manifest["core_run_id"]
                or checkpoint.get("core_manifest_sha256")!=manifest["core_manifest"]["sha256"]
                or checkpoint.get("seed")!=row["seed"] or checkpoint.get("module_id")!="clinical"
                or checkpoint.get("source_checkpoint_sha256") is not None
                or checkpoint.get("core_parameters_frozen") is not True
                or checkpoint.get("old_checkpoint_loaded") is not False
                or checkpoint.get("old_predictions_used_as_features") is not False):
            raise ValueError("Native Clinical checkpoint identity or isolation mismatch")
        if (checkpoint.get("parent_v32_core_checkpoint_sha256")!=parent["checkpoint_sha256"]
                or row.get("parent_v32_core_checkpoint_sha256")!=parent["checkpoint_sha256"]):
            raise ValueError("Native Clinical head uses another core checkpoint")
        for key in ("parent_v32_core_parameter_sha256","core_parameters_before_sha256","core_parameters_after_sha256"):
            if checkpoint.get(key)!=parent["core_parameter_sha256"]: raise ValueError("Native Clinical frozen core changed")
        state=checkpoint.get("model_state",{})
        if not state or not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Native Clinical checkpoint has invalid tensors")
        edges=checkpoint["endpoint_time_bins"]; trained=row["trained_endpoints"]
        if set(edges)!=set(trained) or "DFS" in trained or not set(trained).issubset(CLINICAL_ENDPOINTS):
            raise ValueError("Clinical endpoint availability differs from trained time bins")
        lengths={len(v) for v in edges.values()}
        if len(lengths)!=1 or not lengths or min(lengths)<1: raise ValueError("Clinical bins have inconsistent widths")
        n_bins=next(iter(lengths))
        for endpoint,values in edges.items():
            if not np.isfinite(values).all() or np.min(values)<=0 or (np.diff(values)<=0).any():
                raise ValueError("Clinical time-bin edges are invalid")
        head,init=build_clinical_head(core_features=len(checkpoint["core_columns"]),domain_features=len(checkpoint["domain_columns"]),
            n_bins=n_bins,seed=checkpoint["seed"])
        if any(checkpoint["initialization"].get(k)!=v for k,v in init.as_dict().items()):
            raise ValueError("Clinical deterministic private-head initialization differs")
        head.load_state_dict(state,strict=True); head.eval(); head.requires_grad_(False)
        if module_state_sha256(head)==init.initial_parameter_sha256: raise ValueError("Clinical private head never changed")
        public_path=sources.bind({"path":str(cp_path.parent/"clinical_patient_risk.parquet"),"sha256":row["public_prediction_sha256"]},lp.parent)
        public=pd.read_parquet(public_path); _validate_risk(public,fold,manifest,patients,trained)
        if len(public)!=row["n_test"]*len(CLINICAL_ENDPOINTS): raise ValueError("Native risk fold row count differs")
        if row.get("feature_artifact"):
            features=pd.read_parquet(sources.bind(row["feature_artifact"],lp.parent))
            test=features.loc[features.patient_fold_id.eq(fold)].sort_values(["cancer_id","patient_id"]).reset_index(drop=True)
            if test.duplicated(["cancer_id","patient_id"]).any(): raise ValueError("Clinical feature rows duplicate patients")
            core_values=test[checkpoint["core_columns"]].to_numpy(np.float32,copy=True)
            domain_values=test[checkpoint["domain_columns"]].fillna(0).to_numpy(np.float32,copy=True)
            if not np.isfinite(core_values).all() or not np.isfinite(domain_values).all(): raise ValueError("Clinical replay features are nonfinite")
            with torch.inference_mode():
                logits=reshape_endpoint_logits(head(torch.from_numpy(core_values),torch.from_numpy(domain_values)),n_bins).numpy()
            for i,endpoint in enumerate(CLINICAL_ENDPOINTS):
                observed=public.loc[public.clinical_endpoint.eq(endpoint)].sort_values(["cancer_id","subject_id"]).reset_index(drop=True)
                if list(zip(observed.cancer_id,observed.subject_id))!=list(zip(test.cancer_id,test.patient_id)):
                    raise ValueError("Clinical replay patient identities differ")
                if endpoint in trained and not np.allclose(observed.clinical_relevance_probability,survival_risk_probability(logits[:,i]),rtol=0,atol=1e-6):
                    raise ValueError("Clinical frozen checkpoint replay disagrees with native risk")
            replayed.append(fold)
        for endpoint in CLINICAL_ENDPOINTS:
            bins.append({"patient_fold_id":fold,"clinical_endpoint":endpoint,"trained":endpoint in trained,
                "time_bins_days_json":json.dumps(edges.get(endpoint,[])),"last_training_time_bin_days":edges[endpoint][-1] if endpoint in trained else np.nan})
        all_public.append(public)
    public=pd.concat(all_public,ignore_index=True).sort_values(RISK_KEYS).reset_index(drop=True)
    prediction=sources.bind(manifest["patient_risk_predictions"],context["root"])
    if lineage.get("public_prediction_artifact") and lineage["public_prediction_artifact"]["sha256"]!=sha256(prediction):
        raise ValueError("Native root risk prediction identity drift")
    _assert_same_table(pd.read_parquet(prediction).sort_values(RISK_KEYS).reset_index(drop=True),public)
    if len(public)!=lineage["public_prediction_rows"]: raise ValueError("Native root risk count differs")
    return dict(patients=patients,workbook_sha256=workbook[0]["sha256"],public=public,bins=pd.DataFrame(bins),
        replayed_folds=replayed,fold_input_sha256=split[0]["sha256"])


def _validate_risk(data,fold,manifest,patients,trained):
    required=set(RISK_KEYS)|{"subject_type","clinical_relevance_probability","availability","failure_reason","patient_fold_id",
        "analysis_version","training_run_id","core_run_id","model_artifact_id","changes_primary_ranking"}
    if set(data)!=required or data[RISK_KEYS].isna().any().any() or data.duplicated(RISK_KEYS).any():
        raise ValueError("Native patient-risk schema is incomplete, duplicated or contains private/unregistered columns")
    for key in ("availability","changes_primary_ranking"): _boolean(data,key)
    if (not data.subject_type.eq("patient").all() or not data.patient_fold_id.eq(fold).all()
            or not data.training_run_id.eq(manifest["patient_risk_run_id"]).all()
            or not data.core_run_id.eq(manifest["core_run_id"]).all() or data.changes_primary_ranking.any()):
        raise ValueError("Native patient risk identity differs")
    keys=data.rename(columns={"subject_id":"patient_id"}).merge(patients,on=["cancer_id","patient_id","patient_fold_id"],how="left",indicator=True)
    if not keys._merge.eq("both").all(): raise ValueError("Native patient risk crosses its explicit patient-fold authority")
    if not data.groupby(["cancer_id","subject_id"]).clinical_endpoint.agg(set).map(lambda x:x==set(CLINICAL_ENDPOINTS)).all():
        raise ValueError("Clinical patient lacks all six typed endpoints")
    if not data.availability.eq(data.clinical_endpoint.isin(trained)).all(): raise ValueError("Native risk availability differs from trained endpoints")
    score=pd.to_numeric(data.clinical_relevance_probability,errors="coerce")
    if (not pd.api.types.is_numeric_dtype(data.clinical_relevance_probability)
            or not np.isfinite(score[data.availability]).all() or not score[data.availability].between(0,1).all()
            or data.loc[~data.availability,"clinical_relevance_probability"].notna().any()
            or data.loc[~data.availability,"failure_reason"].fillna("").str.strip().eq("").any()
            or data.loc[data.availability,"failure_reason"].fillna("").ne("").any()):
        raise ValueError("Native Clinical risk has invalid probabilities or null reasons")


def entity_sources(context,risk):
    sources=context["sources"]; manifest=context["manifest"]
    lp,lineage=sources.document(manifest["entity_lineage"],context["root"])
    if (lineage.get("format")!=CALCULATION_FORMAT or lineage.get("training_status")!="SUCCESS"
            or lineage.get("training_run_id")!=manifest["entity_run_id"] or lineage.get("folds")!=5
            or lineage.get("full_entity_universe") is not True or lineage.get("all_candidate_entities_enumerated") is not True
            or set(lineage.get("subject_types",[]))!=set(SUBJECT_SPECS)):
        raise ValueError("Native entity Clinical result lacks its full typed universe")
    cp,calculation=sources.document(manifest["entity_calculation_manifest"],context["root"])
    if (sha256(cp)!=lineage["calculation_manifest_sha256"] or calculation.get("training_run_id")!=manifest["entity_run_id"]
            or calculation.get("format")!=CALCULATION_FORMAT or calculation.get("folds")!=5
            or calculation.get("method")!="TRAIN_LOCAL_COVARIATE_RESIDUALIZED_BRESLOW_COX_NULL_SCORE_ONE_STEP"):
        raise ValueError("Clinical calculation manifest identity differs")
    config=lineage.get("calculation_config")
    if config is None:
        _,config=sources.document(manifest["entity_config"],context["root"])
    if json_sha256(config)!=lineage["config_sha256"] or (calculation.get("calculation_config") is not None and calculation["calculation_config"]!=config):
        raise ValueError("Clinical calculation configuration differs")
    minimum=config["minimums"]["folds_available"]
    if type(minimum) is not int or not 1<=minimum<=5 or config.get("max_entities") is not None:
        raise ValueError("Clinical native fold threshold or complete-universe policy is invalid")
    for fold in range(5):
        expected={"test":[fold],"validation":[(fold+1)%5],"train":[i for i in range(5) if i not in {fold,(fold+1)%5}]}
        if calculation["split_policy"].get(str(fold))!=expected: raise ValueError("Clinical split policy differs")
    if json_sha256(lineage["input_artifacts"])!=lineage["input_manifest_sha256"]: raise ValueError("Clinical entity input inventory differs")
    artifacts=_inputs(lineage["input_artifacts"],sources,lp.parent,method=input_path_sha256)
    split=[r for r in artifacts if r["artifact_kind"]=="split_manifest"]
    raw=[r for r in artifacts if r["artifact_kind"]=="raw_data"]
    candidates=[r for r in artifacts if r["artifact_kind"]=="annotation"]
    if (len(split)!=1 or len(raw)!=1 or len(candidates)!=1 or raw[0]["sha256"]!=risk["workbook_sha256"]
            or candidates[0]["sha256"]!=context["candidate_sha256"]):
        raise ValueError("Clinical native inputs differ from selected candidates or patient-risk outcomes")
    _assert_same_table(_patient_folds(split[0]["path"]),risk["patients"])
    output=[]; partitions=calculation["partitions"]; seen=set()
    candidate=context["candidates"]
    for part in partitions:
        cancer=part["cancer_id"]; kind=part["subject_type"]; identity=(cancer,kind)
        if identity in seen or part.get("status")!="SUCCESS" or kind not in SUBJECT_SPECS:
            raise ValueError("Clinical native partition is duplicated or incomplete")
        seen.add(identity)
        ids=(list(HISTORICAL_STATE_IDS) if kind=="state" else sorted(candidate.loc[candidate.cancer_id.eq(cancer),"lncrna_id" if kind=="lncRNA" else "pathway_id"].unique()))
        fp=sources.bind({"path":part["fold_records_path"],"sha256":part["fold_records_sha256"]},lp.parent)
        folds=pd.read_parquet(fp); _validate_entity_folds(folds,cancer,kind,ids,risk["patients"])
        if len(folds)!=part["fold_rows"]: raise ValueError("Clinical fold partition count differs")
        summary=summarise_oof_associations(folds,ids,cancer_id=cancer,subject_type=kind,min_folds_available=minimum)
        sp=sources.bind({"path":part["summary_part_path"],"sha256":part["summary_part_sha256"]},lp.parent)
        _assert_same_table(pd.read_parquet(sp),summary)
        if len(summary)!=part["summary_rows"]: raise ValueError("Clinical summary partition count differs")
        output.append(summary)
    expected={(c,k) for c in config["cancers"] for k in SUBJECT_SPECS}
    if seen!=expected or set(config["subject_types"])!=set(SUBJECT_SPECS): raise ValueError("Clinical native partition universe is incomplete")
    summary=pd.concat(output,ignore_index=True).sort_values(ENTITY_KEYS).reset_index(drop=True)
    path=sources.bind({"path":lineage["summary_path"],"sha256":lineage["summary_sha256"]},lp.parent)
    observed=pd.read_parquet(path).sort_values(ENTITY_KEYS).reset_index(drop=True)
    _boolean(observed,"availability"); _assert_same_table(observed,summary)
    validate_entity_clinical_contract(observed,lineage)
    return summary


def _validate_entity_folds(data,cancer,kind,ids,patients):
    _boolean(data,"fold_available")
    if (not data.cancer_id.eq(cancer).all() or not data.subject_type.eq(kind).all()
            or data.duplicated([*ENTITY_KEYS,"patient_fold_id"]).any()
            or not data.patient_fold_id.isin(range(5)).all()
            or set(data.subject_id)!=set(ids) or set(data.clinical_endpoint)!=set(CLINICAL_ENDPOINTS)
            or len(data)!=len(ids)*6*5):
        raise ValueError("Clinical entity fold identity or universe is invalid")
    for key in ("old_checkpoint_loaded","old_predictions_used_as_features","changes_primary_ranking"):
        _boolean(data,key)
        if data[key].any(): raise ValueError("Clinical entity historical/ranking flag is invalid")
    if (not data.preprocessing_scope.eq("THREE_TRAIN_FOLDS_ONLY").all()
            or not data.association_method.eq("BRESLOW_COX_NULL_SCORE_ONE_STEP").all()):
        raise ValueError("Clinical entity preprocessing/statistical estimand drift")
    for fold,part in data.groupby("patient_fold_id"):
        validation=(int(fold)+1)%5; train=[i for i in range(5) if i not in {fold,validation}]
        if not part.validation_fold_id.eq(validation).all() or not part.train_fold_ids.astype(str).eq(",".join(map(str,train))).all():
            raise ValueError("Clinical fold roles drift")
        for role,roles in [("train",train),("validation",[validation]),("test",[fold])]:
            n=part[f"n_{role}_patients"]; events=part[f"n_{role}_events"]
            maximum=len(patients.loc[patients.cancer_id.eq(cancer)&patients.patient_fold_id.isin(roles)])
            if (not np.isfinite(n).all() or not np.isfinite(events).all() or (n<0).any() or (n>maximum).any()
                    or (n!=np.floor(n)).any() or (events<0).any() or (events>n).any() or (events!=np.floor(events)).any()):
                raise ValueError("Clinical fold patient/event counts exceed authority")
    available=data.loc[data.fold_available]
    if (not np.isfinite(available.test_beta).all() or not np.isfinite(available.test_standard_error).all()
            or not available.test_standard_error.gt(0).all()
            or available.failure_reason.fillna("").ne("").any()
            or not pd.api.types.is_bool_dtype(data.validation_sign_concordant)
            or available.validation_sign_concordant.isna().any()
            or data.loc[~data.fold_available,"failure_reason"].fillna("").str.strip().eq("").any()):
        raise ValueError("Clinical eligible folds lack finite estimates or missing reasons")
    dfs=data.loc[data.clinical_endpoint.eq("DFS")]
    if dfs.fold_available.any() or dfs.test_beta.notna().any() or not dfs.failure_reason.eq("NO_DISTINCT_DFS_SOURCE").all():
        raise ValueError("Clinical DFS was inferred without a distinct source")
