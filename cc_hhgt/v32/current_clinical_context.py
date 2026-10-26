"""Selected GeneSet clinical context with native patient/entity units intact."""
from __future__ import annotations
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd

from .clinical_training import CLINICAL_ENDPOINTS
from .current_clinical_sources import selected_sources, patient_risk_sources, entity_sources, ENTITY_KEYS
from .current_core_export import verified_path, sha256
from .current_state_context import _assert_same_table

FORMAT="CURRENT_CLINICAL_CONTEXT_RELEASE_V1"
STATISTICS=["meta_beta","meta_standard_error","hazard_ratio","ci_lower","ci_upper","p_value","fdr","tau2","i2"]
COUNTS=["n_folds_available","n_patients_test_total","n_events_test_total","n_folds_same_direction"]


def _json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def public_entities(summary):
    columns=[*ENTITY_KEYS,"availability","failure_reason",*STATISTICS,*COUNTS,"direction","replication_tier","association_method"]
    data=summary[columns].rename(columns={"direction":"native_direction"}).copy()
    beta=data.meta_beta
    data["clinical_effect_direction"]=np.select([data.availability & beta.gt(0),data.availability & beta.lt(0),data.availability & beta.eq(0)],
        ["risk","protective","neutral"],default="unavailable")
    data["one_minus_fdr_score"]=1-data.fdr
    data["score_semantics"]="ONE_MINUS_FULL_NATIVE_CANCER_ENDPOINT_SUBJECT_TYPE_BH_FDR_NOT_POSTERIOR"
    data["fdr_recomputed_after_geneset_selection"]=False
    data["model_based_entity_prediction"]=False
    return data


def selected_clinical_tables(members,coverage,entities,*,fdr_threshold=.05):
    """Selected-score mass with supported risk/protective associations; no mean HR."""
    if type(fdr_threshold) not in (int,float) or not np.isfinite(fdr_threshold) or not 0<fdr_threshold<=1:
        raise ValueError("Clinical FDR threshold must be finite in (0,1]")
    keep=["geneset_id","geneset_rank","cancer_id","lncrna_id","pathway_id","selected_score","association_direction"]
    if (not set(keep).issubset(members) or members.duplicated(["geneset_id","lncrna_id"]).any()
            or not np.isfinite(pd.to_numeric(members.selected_score,errors="coerce")).all()
            or not members.selected_score.between(0,1).all()):
        raise ValueError("Clinical selected members or scores are invalid")
    endpoints=pd.DataFrame({"clinical_endpoint":list(CLINICAL_ENDPOINTS)})
    lnc=entities.loc[entities.subject_type.eq("lncRNA")].drop(columns="subject_type").rename(columns={"subject_id":"lncrna_id"})
    data=members[keep].rename(columns={"association_direction":"pathway_association_direction"}).merge(endpoints,how="cross")
    data=data.merge(lnc,on=["cancer_id","lncrna_id","clinical_endpoint"],how="left",validate="many_to_one")
    absent=data.availability.isna(); data.loc[absent,"failure_reason"]="NO_NATIVE_ENTITY_ASSOCIATION_FOR_SELECTED_MEMBER"
    data["availability"]=data.availability.eq(True)
    data["clinical_effect_direction"]=data.clinical_effect_direction.fillna("unavailable")
    supported=data.availability & data.fdr.le(fdr_threshold)
    for direction in ("risk","protective","neutral"):
        data[f"supported_{direction}"]=supported & data.clinical_effect_direction.eq(direction)
    data["available_score_weight"]=data.selected_score.where(data.availability,0.)
    for direction in ("risk","protective","neutral"):
        data[f"supported_{direction}_score_weight"]=data.selected_score.where(data[f"supported_{direction}"],0.)
    groups=["geneset_id","cancer_id","pathway_id","pathway_association_direction","clinical_endpoint"]
    aggregate=dict(member_count=("lncrna_id","size"),available_members=("availability","sum"),
        total_selected_score_weight=("selected_score","sum"),available_selected_score_weight=("available_score_weight","sum"))
    for direction in ("risk","protective","neutral"):
        aggregate[f"supported_{direction}_members"]=(f"supported_{direction}","sum")
        aggregate[f"supported_{direction}_score_weight"]=(f"supported_{direction}_score_weight","sum")
    profiles=data.groupby(groups,sort=True,dropna=False).agg(**aggregate).reset_index()
    denominator=profiles.available_selected_score_weight.where(profiles.available_selected_score_weight.gt(0))
    profiles["member_coverage"]=profiles.available_members/profiles.member_count
    profiles["selected_weight_coverage"]=profiles.available_selected_score_weight/profiles.total_selected_score_weight.where(profiles.total_selected_score_weight.gt(0))
    for direction in ("risk","protective","neutral"):
        profiles[f"supported_{direction}_weight_fraction"]=profiles[f"supported_{direction}_score_weight"]/denominator
    profiles["availability"]=profiles.available_selected_score_weight.gt(0)
    profiles["failure_reason"]=np.select([profiles.available_members.eq(0),~profiles.availability],
        ["NO_MEMBERS_WITH_AVAILABLE_ENTITY_ASSOCIATION","ZERO_AVAILABLE_SELECTED_SCORE_WEIGHT"],default=None)
    profiles["fdr_threshold"]=float(fdr_threshold)
    profiles["summary_semantics"]="SELECTED_SCORE_WEIGHTED_MEMBER_SUPPORT_NOT_JOINT_GENESET_SURVIVAL_EFFECT"
    profiles["gene_set_hazard_ratio_available"]=False
    profiles["gene_set_p_value_available"]=False
    data=data.drop(columns=["available_score_weight",*[f"supported_{d}_score_weight" for d in ("risk","protective","neutral")]])
    cov=coverage[["cancer_id","pathway_id","association_direction","geneset_available","unavailable_reason","exported_members"]].rename(
        columns={"association_direction":"pathway_association_direction","unavailable_reason":"geneset_unavailable_reason"}).merge(endpoints,how="cross")
    cov=cov.merge(profiles[[*groups,"available_members","member_coverage","selected_weight_coverage","availability","failure_reason"]],
        on=groups[1:],how="left",validate="one_to_one")
    cov.loc[~cov.geneset_available,"failure_reason"]="SELECTED_GENESET_UNAVAILABLE"
    cov["availability"]=cov.availability.eq(True); cov["available_members"]=cov.available_members.fillna(0).astype(int)
    paths=entities.loc[entities.subject_type.eq("exact_pathway")].drop(columns="subject_type").rename(columns={"subject_id":"pathway_id"})
    selected_paths=coverage[["cancer_id","pathway_id"]].drop_duplicates().merge(endpoints,how="cross")
    selected_paths=selected_paths.merge(paths,on=["cancer_id","pathway_id","clinical_endpoint"],how="left",validate="one_to_one")
    absent=selected_paths.availability.isna()
    selected_paths.loc[absent,"failure_reason"]="NO_NATIVE_EXACT_PATHWAY_ASSOCIATION"
    selected_paths["availability"]=selected_paths.availability.eq(True)
    selected_paths["association_unit"]="MEASURED_EXACT_PATHWAY_ACTIVITY_NOT_SELECTED_LNCRNA_GENESET"
    return dict(selected_member_clinical=data,selected_geneset_clinical_support=profiles,
        selected_clinical_coverage=cov,selected_pathway_clinical=selected_paths)


def patient_risk_coverage(risk):
    """Fold-specific distributions, retaining different native time grids."""
    authority=risk["patients"].rename(columns={"patient_id":"subject_id"}).merge(
        pd.DataFrame({"clinical_endpoint":list(CLINICAL_ENDPOINTS)}),how="cross")
    keys=["cancer_id","subject_id","patient_fold_id","clinical_endpoint"]
    data=authority.merge(risk["public"][[*keys,"clinical_relevance_probability","availability","failure_reason"]],
        on=keys,how="left",validate="one_to_one")
    absent=data.availability.isna(); data["native_prediction_row_present"]=~absent
    data.loc[absent,"failure_reason"]="NO_NATIVE_RISK_FOR_AUTHORIZED_PATIENT"
    data["availability"]=data.availability.eq(True)
    rows=[]
    for key,group in data.groupby(["cancer_id","patient_fold_id","clinical_endpoint"],sort=True):
        observed=group.loc[group.availability,"clinical_relevance_probability"]
        reasons=group.loc[~group.availability,"failure_reason"].value_counts().sort_index().to_dict()
        rows.append(dict(zip(["cancer_id","patient_fold_id","clinical_endpoint"],key))|
            {"authorized_patients":len(group),"patients_with_native_row":int(group.native_prediction_row_present.sum()),
             "available_patients":len(observed),"patient_coverage":len(observed)/len(group),
             "risk_p10":observed.quantile(.1) if len(observed) else np.nan,
             "risk_median":observed.median() if len(observed) else np.nan,
             "risk_p90":observed.quantile(.9) if len(observed) else np.nan,
             "availability":bool(len(observed)),"missing_reasons_json":json.dumps(reasons,sort_keys=True)})
    result=pd.DataFrame(rows).merge(risk["bins"],on=["patient_fold_id","clinical_endpoint"],validate="many_to_one")
    result["risk_semantics"]="NATIVE_DISCRETE_TIME_CUMULATIVE_RISK_OVER_FOLD_TRAINED_BINS"
    result["common_time_horizon_calibrated"]=False
    result["fold_risks_pooled"]=False
    return result


def clinical_tables(manifest_path):
    context=selected_sources(manifest_path)
    risk=patient_risk_sources(context); native=entity_sources(context,risk)
    entities=public_entities(native); selected=context["selected"]
    alpha=context["manifest"].get("fdr_threshold",.05)
    tables={"native_entity_clinical":entities,"patient_risk_fold_coverage":patient_risk_coverage(risk),
        **selected_clinical_tables(pd.read_parquet(selected.paths["geneset_members"]),
            pd.read_parquet(selected.paths["geneset_coverage"]),entities,fdr_threshold=alpha)}
    for data in tables.values():
        data["selected_analysis_sha256"]=selected.binding_sha256
        data["selected_score_run_id"]=selected.binding["selected_score_run_id"]
        data["patient_risk_run_id"]=context["manifest"]["patient_risk_run_id"]
        data["entity_analysis_run_id"]=context["manifest"]["entity_run_id"]
        data["changes_primary_ranking"]=False
    return context,risk,tables


def implementation_hashes():
    root=Path(__file__).resolve().parent
    names=["current_clinical_context.py","current_clinical_sources.py","clinical_training.py","clinical_entity_training.py",
        "current_state_context.py","current_core_export.py","integrated_model.py","input_lineage.py",
        "clinical_numeric_replay.py","patient_fold_authority.py"]
    return {**{name:sha256(root/name) for name in names},
        **{name:sha256(root.parents[1]/name) for name in
           ("scripts/run_v32_clinical_training.py","scripts/run_v32_clinical_entity_training.py")}}


def materialize_current_clinical_context(manifest_path,output):
    if socket.gethostname()!="149": raise RuntimeError("Current Clinical context must execute on target host 149")
    path=Path(manifest_path).resolve(); output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    _write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,"stage":"CURRENT_CLINICAL_CONTEXT","input_manifest_sha256":sha256(path)})
    context,risk,tables=clinical_tables(path); outputs={}
    for name,data in tables.items():
        target=output/(name+".parquet"); data.to_parquet(target,index=False)
        outputs[name]={"path":str(target),"sha256":sha256(target),"rows":len(data)}
    binding={"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
        "input_manifest":{"path":str(path),"sha256":sha256(path)},"selected_analysis":context["manifest"]["selected_analysis"],
        "selected_score_run_id":context["selected"].binding["selected_score_run_id"],"core_run_id":context["manifest"]["core_run_id"],
        "patient_risk_run_id":context["manifest"]["patient_risk_run_id"],"entity_run_id":context["manifest"]["entity_run_id"],
        "fdr_threshold":float(context["manifest"].get("fdr_threshold",.05)),"outputs":outputs,
        "source_artifacts":list(context["sources"].records.values()),"implementation_sha256":implementation_hashes(),
        "patient_risk_frozen_replay_folds":risk["replayed_folds"],"raw_patient_feature_projection_replayed":True,
        "native_entity_meta_analysis_recomputed":True,"entity_raw_patient_statistics_replayed":True,
        "native_numeric_replay":context["native_numeric_replay"],
        "context_changes_ranking":False,"joint_geneset_survival_effect_estimated":False,"patient_identifiers_in_public_context":False,
        "production_deployed":False}
    bp=output/"CURRENT_CLINICAL_CONTEXT_BINDING.json"; _write(bp,binding)
    ap=output/"RECOMPUTATION_AUDIT.json"; _write(ap,audit_current_clinical_context(bp,expected_sha=sha256(bp)))
    result={"status":"PASS","target_host":"149","training_started":False,"binding_path":str(bp),"binding_sha256":sha256(bp),"audit_sha256":sha256(ap)}
    _write(output/"SUCCESS.json",result); return result


def _load(binding_path,expected_sha):
    path=verified_path({"path":str(binding_path),"sha256":expected_sha},Path.cwd()); binding=_json(path)
    if binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149":
        raise ValueError("Invalid current Clinical binding")
    for key in ("training_started",
        "context_changes_ranking","joint_geneset_survival_effect_estimated","patient_identifiers_in_public_context","production_deployed"):
        if binding.get(key) is not False: raise ValueError("Invalid Clinical release semantics: "+key)
    if binding.get("native_entity_meta_analysis_recomputed") is not True: raise ValueError("Clinical entity replay missing")
    for key in ("raw_patient_feature_projection_replayed","entity_raw_patient_statistics_replayed"):
        if binding.get(key) is not True: raise ValueError("Clinical numerical replay missing: "+key)
    paths={name:verified_path(r,path.parent) for name,r in binding["outputs"].items()}
    return path,binding,paths


def audit_current_clinical_context(binding_path,*,expected_sha):
    path,binding,outputs=_load(binding_path,expected_sha)
    if binding["implementation_sha256"]!=implementation_hashes(): raise ValueError("Clinical replay implementation changed")
    context,risk,tables=clinical_tables(verified_path(binding["input_manifest"],path.parent))
    for key in ("selected_analysis","core_run_id","patient_risk_run_id","entity_run_id"):
        if binding[key]!=context["manifest"][key]: raise ValueError("Clinical input identity changed")
    if (binding["selected_score_run_id"]!=context["selected"].binding["selected_score_run_id"]
            or binding["fdr_threshold"]!=context["manifest"].get("fdr_threshold",.05)
            or binding["source_artifacts"]!=list(context["sources"].records.values())
            or binding.get("native_numeric_replay")!=context["native_numeric_replay"]
            or binding["patient_risk_frozen_replay_folds"]!=risk["replayed_folds"] or set(outputs)!=set(tables)):
        raise ValueError("Clinical output policy or source inventory changed")
    for name,expected in tables.items():
        observed=pd.read_parquet(outputs[name])
        if len(observed)!=binding["outputs"][name]["rows"]: raise ValueError("Clinical context row count differs")
        try: _assert_same_table(observed,expected)
        except AssertionError as error: raise ValueError("Clinical context source replay differs: "+name) from error
    return {"status":"PASS","binding_sha256":expected_sha,"training_started":False,
        "native_entity_meta_analysis_recomputed":True,"entity_replay_same_statistical_implementation":True,
        "selected_member_support_recomputed":True,"patient_risk_frozen_replay_folds":risk["replayed_folds"],
        "raw_patient_feature_projection_replayed":True,"entity_raw_patient_statistics_replayed":True,
        "native_numeric_replay":context["native_numeric_replay"]}


class CurrentClinicalContextQuery:
    def __init__(self,binding_path,*,expected_sha,selected_analysis_sha):
        path,self.binding,self.paths=_load(binding_path,expected_sha); self.sha=expected_sha
        if self.binding["selected_analysis"]["sha256"]!=selected_analysis_sha: raise ValueError("Clinical selected-analysis version differs")
        success=_json(path.parent/"SUCCESS.json"); ap=path.parent/"RECOMPUTATION_AUDIT.json"; audit=_json(ap)
        if (success.get("status")!="PASS" or success.get("binding_sha256")!=expected_sha
                or success.get("audit_sha256")!=sha256(ap) or audit.get("status")!="PASS" or audit.get("binding_sha256")!=expected_sha):
            raise ValueError("Clinical context lacks completed source replay")

    def download(self,name):
        if name not in self.paths: raise ValueError("Unknown Clinical download")
        record=self.binding["outputs"][name]; verified_path(record,Path.cwd())
        return {**record,"binding_sha256":self.sha,"selected_analysis_sha256":self.binding["selected_analysis"]["sha256"]}

    def query(self,*,table="selected_geneset_clinical_support",filters=None,limit=100,offset=0):
        import duckdb
        if table not in self.paths: raise ValueError("Unknown Clinical table")
        if type(limit) is not int or type(offset) is not int or not 1<=limit<=1000 or not 0<=offset<=1000000:
            raise ValueError("Invalid Clinical pagination")
        filters=filters or {}; allowed={"cancer_id","lncrna_id","pathway_id","geneset_id","clinical_endpoint","subject_id",
            "subject_type","clinical_effect_direction","pathway_association_direction"}
        if not set(filters).issubset(allowed) or not all(isinstance(v,str) for v in filters.values()):
            raise ValueError("Clinical filters require exact string identities")
        source=self.download(table)["path"]; connection=duckdb.connect()
        try:
            columns=[r[0] for r in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)",[source]).fetchall()]
            if not set(filters).issubset(columns): raise ValueError("Clinical filter is not a column in this table")
            where=" AND ".join('"'+key+'"=?' for key in filters) or "true"
            order=",".join('"'+key+'"' for key in ["cancer_id","pathway_id","geneset_id","clinical_endpoint","patient_fold_id","geneset_rank","lncrna_id","subject_type","subject_id"] if key in columns)
            frame=connection.execute(f"SELECT * FROM read_parquet(?) WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",[source,*filters.values(),limit,offset]).df()
        finally: connection.close()
        return {"binding_sha256":self.sha,"selected_analysis_sha256":self.binding["selected_analysis"]["sha256"],
            "rows":frame.astype(object).where(frame.notna(),None).to_dict("records")}
