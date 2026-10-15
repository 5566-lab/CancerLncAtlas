"""Patient-level conditional lncRNA--pathway association and heterogeneity.

Predictive assessment is patient-OOF. Full-cohort interaction inference is a
separate descriptive association analysis with HC3 covariance; it is not
causal inference or independent validation. No neural optimizer is involved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


PAIR_KEYS = ["cancer_id", "lncrna_id", "pathway_id"]
PATIENT_KEYS = ["cancer_id", "patient_id"]


@dataclass(frozen=True)
class HeterogeneityConfig:
    min_train_patients: int = 20
    min_group_patients: int = 8
    confidence: float = .95
    ridge_alpha: float = 1.
    covariates: tuple[str, ...] = ("local_cnv_log2", "tumor_purity")
    folds: int = 5

    def __post_init__(self):
        if self.folds != 5 or self.min_train_patients < 5 or self.min_group_patients < 3:
            raise ValueError("Patient heterogeneity requires five folds and adequate group counts")
        if not 0 < self.confidence < 1 or not np.isfinite(self.ridge_alpha) or self.ridge_alpha < 0:
            raise ValueError("Invalid confidence or ridge penalty")
        if len(set(self.covariates)) != len(self.covariates):
            raise ValueError("Duplicate nuisance covariates")


def _keys(frame: pd.DataFrame, columns: Sequence[str], label: str):
    if not set(columns).issubset(frame) or frame[list(columns)].isna().any().any():
        raise ValueError(f"{label} lacks complete identifiers")
    result=frame.copy()
    for column in columns:
        result[column]=result[column].astype(str).str.strip()
        if result[column].eq("").any():raise ValueError(f"{label} has empty identifiers")
    if "cancer_id" in result:result["cancer_id"]=result.cancer_id.str.upper()
    if result.duplicated(list(columns)).any():raise ValueError(f"Duplicate {label} identifiers")
    return result


def _boolean(values: pd.Series, label: str) -> np.ndarray:
    if values.isna().any() or not values.isin([True,False]).all():
        raise ValueError(f"{label} must be non-null boolean")
    return values.to_numpy(bool)


def mutation_subgroups(calls: pd.DataFrame, *, group_id: str, entity_column="lncrna_id") -> pd.DataFrame:
    """Separate observed mutants, confirmed callable negatives and unknowns.

    Input: cancer_id, patient_id, entity_column, reported_variant_count,
    mutation_assay_available, locus_callable. A positive report demonstrates a
    mutation without requiring complete locus coverage. Zero WES reports alone
    cannot demonstrate a nonmutant locus.
    """
    keys=PATIENT_KEYS+[entity_column]
    data=_keys(calls,keys,"mutation calls")
    assay=_boolean(data.mutation_assay_available,"mutation assay availability")
    callable_=_boolean(data.locus_callable,"locus callability")
    count=pd.to_numeric(data.reported_variant_count,errors="raise").to_numpy(float)
    if np.isinf(count).any() or (count[np.isfinite(count)]<0).any() or not np.all(count[np.isfinite(count)]==np.floor(count[np.isfinite(count)])):
        raise ValueError("Variant counts must be non-negative integers or typed null")
    if (np.isfinite(count)&~assay).any() or (assay&~np.isfinite(count)).any() or (callable_&~assay).any():
        raise ValueError("Mutation count/assay/callability contradiction")
    positive=assay&(count>0)
    negative=assay&callable_&(count==0)
    result=data[keys].copy()
    result["group_id"]=group_id
    result["group_value"]=np.where(positive,1.,np.where(negative,0.,np.nan))
    result["group_available"]=positive|negative
    result["group_unavailable_reason"]=np.where(positive|negative,None,
        np.where(~assay,"NO_MUTATION_ASSAY","NO_CONFIRMED_NONMUTANT_LOCUS"))
    result["group_zero_semantics"]="CALLABLE_LOCUS_WITH_NO_REPORTED_VARIANT"
    result["group_one_semantics"]="OBSERVED_REPORTED_VARIANT"
    return result


def assemble_patient_pair(expression: pd.DataFrame, activity: pd.DataFrame, groups: pd.DataFrame,
                          covariates: pd.DataFrame, folds: pd.DataFrame, *, cancer: str,
                          lncrna: str, pathway: str, group_id: str, config: HeterogeneityConfig) -> pd.DataFrame:
    """Join literal patient/entity keys for one prespecified candidate and group."""
    cancer=cancer.upper()
    x=expression.loc[expression.cancer_id.astype(str).str.upper().eq(cancer)&expression.lncrna_id.eq(lncrna)].copy()
    y=activity.loc[activity.cancer_id.astype(str).str.upper().eq(cancer)&activity.pathway_id.eq(pathway)].copy()
    g=groups.loc[groups.cancer_id.astype(str).str.upper().eq(cancer)&groups.group_id.eq(group_id)].copy()
    if "lncrna_id" in g:g=g.loc[g.lncrna_id.eq(lncrna)]
    c=covariates.loc[covariates.cancer_id.astype(str).str.upper().eq(cancer)].copy()
    if "lncrna_id" in c:c=c.loc[c.lncrna_id.eq(lncrna)]
    f=folds.loc[folds.cancer_id.astype(str).str.upper().eq(cancer)].copy()
    parts=[_keys(value,PATIENT_KEYS,label) for value,label in [(x,"expression"),(y,"activity"),(g,"subgroups"),(c,"covariates"),(f,"folds")]]
    x,y,g,c,f=parts
    # Frozen folds enumerate the patient universe, including missing outcomes.
    result=f[PATIENT_KEYS+["patient_fold_id"]].merge(x[PATIENT_KEYS+["lncrna_expression"]],on=PATIENT_KEYS,how="left",validate="one_to_one")
    result=result.merge(y[PATIENT_KEYS+["activity_score"]],on=PATIENT_KEYS,how="left",validate="one_to_one")
    result=result.merge(g[PATIENT_KEYS+["group_value","group_available","group_unavailable_reason"]],on=PATIENT_KEYS,how="left",validate="one_to_one")
    missing_group=result.group_available.isna()
    result.loc[missing_group,"group_available"]=False
    result.loc[missing_group,"group_unavailable_reason"]="NO_SUBGROUP_MEASUREMENT"
    wanted=list(config.covariates)+[name+"__available" for name in config.covariates]
    result=result.merge(c[PATIENT_KEYS+wanted],on=PATIENT_KEYS,how="left",validate="one_to_one")
    for feature in config.covariates:
        flag=result[feature+"__available"]
        if not flag.dropna().isin([True,False]).all():raise ValueError(f"{feature} availability must be boolean")
        result[feature+"__available"]=flag.fillna(False).astype(bool)
    return result.assign(lncrna_id=lncrna,pathway_id=pathway,group_id=group_id)


def _normalise(frame: pd.DataFrame, config: HeterogeneityConfig):
    required=[*PATIENT_KEYS,"lncrna_id","pathway_id","group_id","patient_fold_id","lncrna_expression",
              "activity_score","group_value","group_available","group_unavailable_reason"]
    required+=list(config.covariates)+[name+"__available" for name in config.covariates]
    if missing:=sorted(set(required)-set(frame)):raise ValueError(f"Patient-pair input lacks {missing}")
    data=_keys(frame,PATIENT_KEYS,"patient-pair rows")
    if data[PAIR_KEYS+["group_id"]].isna().any().any():raise ValueError("Missing candidate/group identity")
    if data[PAIR_KEYS+["group_id"]].drop_duplicates().shape[0]!=1:
        raise ValueError("Analyze one exact candidate and subgroup definition at a time")
    fold=pd.to_numeric(data.patient_fold_id,errors="raise")
    if not fold.isin(range(5)).all():raise ValueError("Patient folds must be integers 0..4")
    data["patient_fold_id"]=fold.astype(int)
    g=pd.to_numeric(data.group_value,errors="raise").to_numpy(float)
    available=_boolean(data.group_available,"subgroup availability")
    data["group_available"]=available
    if np.isinf(g).any() or not np.array_equal(available,np.isfinite(g)) or not np.isin(g[available],[0,1]).all():
        raise ValueError("Subgroups must be binary and preserve typed-null availability")
    if data.loc[~available,"group_unavailable_reason"].fillna("").eq("").any():
        raise ValueError("Unavailable subgroup lacks a reason")
    for feature in ["lncrna_expression","activity_score",*config.covariates]:
        data[feature]=pd.to_numeric(data[feature],errors="raise")
        if np.isinf(data[feature]).any():raise ValueError(f"Infinite {feature}")
        if feature in config.covariates:
            mask=_boolean(data[feature+"__available"],feature+" availability")
            if not np.array_equal(mask,np.isfinite(data[feature])):
                raise ValueError(f"{feature} violates typed-null availability")
    data["group_value"]=g
    return data.sort_values(PATIENT_KEYS).reset_index(drop=True)


def _design(train: pd.DataFrame, test: pd.DataFrame, covariates: Sequence[str]):
    """Train-only centring/scaling and nuisance missingness indicators."""
    x=train.lncrna_expression.to_numpy(float)
    mean=float(x.mean());sd=float(x.std())
    if not np.isfinite(sd) or sd<1e-8:raise ValueError("NO_TRAIN_EXPRESSION_VARIATION")
    z=(x-mean)/sd;zt=(test.lncrna_expression.to_numpy(float)-mean)/sd
    g=train.group_value.to_numpy(float);gt=test.group_value.to_numpy(float)
    train_columns=[np.ones(len(train)),z,g,z*g]
    test_columns=[np.ones(len(test)),zt,gt,zt*gt]
    names=["intercept","expression_z","group","expression_z_by_group"]
    records=[]
    for feature in covariates:
        observed=train[feature].dropna().to_numpy(float)
        centre=float(observed.mean()) if len(observed) else 0.
        scale=float(observed.std()) if len(observed) else 0.
        if scale>=1e-8:
            train_columns.append((train[feature].fillna(centre).to_numpy()-centre)/scale)
            test_columns.append((test[feature].fillna(centre).to_numpy()-centre)/scale)
            names.append(feature+"_z")
        missing=train[feature].isna().to_numpy(float)
        if len(np.unique(missing))>1:
            train_columns.append(missing);test_columns.append(test[feature].isna().to_numpy(float));names.append(feature+"_missing")
        records.append({"feature":feature,"train_mean":centre,"train_sd":scale,"train_observed":len(observed)})
    return np.column_stack(train_columns),np.column_stack(test_columns),{
        "expression_mean":mean,"expression_sd":sd,"columns":names,"covariates":records}


def _ridge(x,y,alpha):
    penalty=np.eye(x.shape[1])*alpha;penalty[0,0]=0
    return np.linalg.pinv(x.T@x+penalty)@(x.T@y)


def interaction_hc3(data: pd.DataFrame, config: HeterogeneityConfig) -> dict:
    """Unpenalized full-cohort association, HC3 covariance and Wald intervals."""
    x,_,transform=_design(data,data,config.covariates)
    if len(data)<=x.shape[1]+2 or np.linalg.matrix_rank(x)!=x.shape[1]:
        return {"inference_available":False,"inference_reason":"INSUFFICIENT_OR_RANK_DEFICIENT_DESIGN"}
    y=data.activity_score.to_numpy(float)
    inverse=np.linalg.inv(x.T@x);beta=inverse@x.T@y
    residual=y-x@beta;leverage=np.einsum("ij,jk,ik->i",x,inverse,x)
    if (leverage>=1-1e-8).any():
        return {"inference_available":False,"inference_reason":"UNIT_LEVERAGE_PREVENTS_HC3"}
    adjusted=np.square(residual/(1-leverage))
    covariance=inverse@((x.T*adjusted)@x)@inverse
    cutoff=norm.ppf((1+config.confidence)/2)
    result={"inference_available":True,"inference_reason":None,"inference_patients":len(data),
            "inference_method":"FULL_COHORT_OLS_HC3_ASYMPTOTIC_WALD","confidence_level":config.confidence}
    for label,contrast in [("group0_slope",[0,1,0,0]),("group1_slope",[0,1,0,1]),("slope_difference",[0,0,0,1])]:
        c=np.pad(contrast,(0,x.shape[1]-4))/transform["expression_sd"]
        effect=float(c@beta);se=float(np.sqrt(max(c@covariance@c,0)))
        result.update({label:effect,label+"_se":se,label+"_ci_low":effect-cutoff*se,label+"_ci_high":effect+cutoff*se})
    effect,se=result["slope_difference"],result["slope_difference_se"]
    result["interaction_p_value"]=float(2*norm.sf(abs(effect/se))) if se>0 else (0. if effect else 1.)
    return result


def analyze_patient_heterogeneity(frame: pd.DataFrame, *, config: HeterogeneityConfig | None = None):
    settings=config or HeterogeneityConfig()
    data=_normalise(frame,settings)
    keys=PAIR_KEYS+["group_id"]
    identity=data.iloc[0][keys].to_dict() if len(data) else {}
    if not identity:raise ValueError("Patient-pair universe is empty")
    known=data.group_available.to_numpy(bool)&np.isfinite(data.lncrna_expression)&np.isfinite(data.activity_score)
    usable=data.loc[known].copy()
    patient=data[PATIENT_KEYS+["patient_fold_id","lncrna_id","pathway_id","group_id","group_value","group_available"]].copy()
    patient["observed_activity"]=data.activity_score
    for column in ["predicted_activity_oof","nuisance_prediction_oof","association_component_oof","conditional_slope_oof","prediction_residual"]:
        patient[column]=np.nan
    patient["prediction_available"]=False
    patient["prediction_unavailable_reason"]="INSUFFICIENT_TRAIN_SUPPORT"
    patient.loc[~data.group_available,"prediction_unavailable_reason"]=data.loc[~data.group_available,"group_unavailable_reason"]
    patient.loc[data.group_available & data.lncrna_expression.isna(),"prediction_unavailable_reason"]="NO_EXPRESSION_MEASUREMENT"
    audits=[]
    for fold in range(5):
        train=usable.loc[usable.patient_fold_id.ne(fold)]
        # Outcomes are unnecessary for prediction. Missing held-out outcomes
        # remain valid predictions, but are excluded from evaluation metrics.
        test=data.loc[data.patient_fold_id.eq(fold)&data.group_available&data.lncrna_expression.notna()]
        counts=train.groupby("group_value").size().reindex([0.,1.],fill_value=0)
        record={"fold":fold,"train_patients":len(train),"group0_train":int(counts[0.]),
                "group1_train":int(counts[1.]),"heldout_patients":len(test),"status":"UNAVAILABLE"}
        if len(train)<settings.min_train_patients or counts.min()<settings.min_group_patients:
            audits.append(record);continue
        try:x,xt,transform=_design(train,test,settings.covariates)
        except ValueError as error:record["reason"]=str(error);audits.append(record);continue
        if any(train.loc[train.group_value.eq(g),"lncrna_expression"].std(ddof=0)<1e-8 for g in [0,1]):
            record["reason"]="NO_WITHIN_GROUP_EXPRESSION_VARIATION";audits.append(record);continue
        y=train.activity_score.to_numpy(float)
        beta=_ridge(x,y,settings.ridge_alpha)
        nuisance=[i for i,name in enumerate(transform["columns"]) if name not in {"expression_z","expression_z_by_group"}]
        nuisance_beta=_ridge(x[:,nuisance],y,settings.ridge_alpha)
        prediction=xt@beta;baseline=xt[:,nuisance]@nuisance_beta
        patient.loc[test.index,"predicted_activity_oof"]=prediction
        patient.loc[test.index,"nuisance_prediction_oof"]=baseline
        patient.loc[test.index,"association_component_oof"]=prediction-baseline
        patient.loc[test.index,"conditional_slope_oof"]=(beta[1]+test.group_value.to_numpy()*beta[3])/transform["expression_sd"]
        patient.loc[test.index,"prediction_residual"]=test.activity_score.to_numpy()-prediction
        patient.loc[test.index,"prediction_available"]=True
        patient.loc[test.index,"prediction_unavailable_reason"]=None
        record.update(status="PASS_PATIENT_OOF",transform=transform,coefficients=beta.tolist(),nuisance_coefficients=nuisance_beta.tolist())
        audits.append(record)
    summary={**identity,"patient_universe":len(data),"group_unknown_patients":int((~data.group_available).sum()),
             "group0_patients":int(usable.group_value.eq(0).sum()),"group1_patients":int(usable.group_value.eq(1).sum()),
             "prediction_available_patients":int(patient.prediction_available.sum()),"causal_claimed":False,
             "subject_specific_effect_claimed":False,"external_validation_claimed":False}
    group_metrics=[]
    for group in [0,1]:
        test=patient.loc[patient.group_value.eq(group)&patient.prediction_available&patient.observed_activity.notna()]
        metric={**identity,"group_value":group,"evaluated_patients":len(test)}
        for name,column in [("model","predicted_activity_oof"),("nuisance","nuisance_prediction_oof")]:
            metric[name+"_mse"]=float(np.mean((test.observed_activity-test[column])**2)) if len(test) else np.nan
        metric["delta_mse"]=metric["nuisance_mse"]-metric["model_mse"]
        group_metrics.append(metric)
    if min(summary["group0_patients"],summary["group1_patients"])>=settings.min_group_patients:
        try:summary.update(interaction_hc3(usable,settings))
        except ValueError as error:summary.update(inference_available=False,inference_reason=str(error))
    else:summary.update(inference_available=False,inference_reason="INSUFFICIENT_GROUP_SUPPORT")
    if not summary["inference_available"]:
        for name in ["group0_slope","group1_slope","slope_difference"]:
            for suffix in ["","_se","_ci_low","_ci_high"]:summary[name+suffix]=np.nan
        summary["interaction_p_value"]=np.nan
    audit={"config":asdict(settings),"folds":audits,"heldout_outcomes_used_for_own_fit":False,
           "missing_mutation_assumed_wildtype":False,"patient_prediction_semantics":"GROUP_CONDITIONAL_ASSOCIATION_NOT_INDIVIDUAL_CAUSAL_EFFECT"}
    return patient,pd.DataFrame([summary]),pd.DataFrame(group_metrics),audit


def add_heterogeneity_fdr(summary: pd.DataFrame) -> pd.DataFrame:
    """BH within the full declared cancer x subgroup family, including null tests."""
    result=summary.copy();result["interaction_fdr"]=np.nan
    for _,local in result.groupby(["cancer_id","group_id"],sort=True):
        p=pd.to_numeric(local.interaction_p_value,errors="raise")
        available=local.inference_available.astype(bool)
        finite=np.isfinite(p)
        if not np.array_equal(available,finite) or not p[finite].between(0,1).all():
            raise ValueError("Interaction tests violate typed-null semantics")
        ordered=p[finite].sort_values(kind="stable")
        if len(ordered):
            # Unavailable prespecified tests contribute to the family size.
            q=np.minimum.accumulate((ordered.to_numpy()*len(local)/np.arange(1,len(ordered)+1))[::-1])[::-1]
            result.loc[ordered.index,"interaction_fdr"]=np.clip(q,0,1)
    return result
