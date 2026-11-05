"""RNA-quality-only Lasso with explicit failure for empty RNA signatures."""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
import pandas as pd
from cc_hhgt.m0_joint_lasso import fit_x_context, select_ratio, fit_final, predict
from cc_hhgt.m0_covariates import complete_rows, uses_purity

G = {}
_CONTEXTS = OrderedDict()
FAIL_ZERO = "FAILED_NO_RNA_SELECTED"

def context(rows, use_purity):
    key = (tuple(rows.tolist()), bool(use_purity))
    if key not in _CONTEXTS:
        _CONTEXTS[key] = fit_x_context(G["x"][rows], G["cohort"][rows],
            mask=None, covariates=G["covariates"].iloc[rows], use_purity=use_purity)
        if len(_CONTEXTS) > 4:
            _CONTEXTS.popitem(last=False)
    _CONTEXTS.move_to_end(key)
    return _CONTEXTS[key]

def worker(j):
    x=G["x"]; cov=G["covariates"]; cohort=G["cohort"]; tid=G["target_ids"][j]
    purity=uses_purity(tid); eligible=complete_rows(cov,purity); y=G["yf"][:,j]
    ft=G["ft"]; ft=ft[np.isfinite(y[ft]) & eligible[ft]]
    base=dict(j=j,n_fit=len(ft),purity_covariate_used=purity,
        adjustment="AGE_SEX_STAGE_COHORT_FIXED_DNA_PURITY",
        screening_applied=False,n_candidates=x.shape[1])
    if len(ft)<12:
        return dict(**base,status="INSUFFICIENT_OBSERVED_TRAINING")
    pa={}
    try:
        if G["final"]:
            ratio=G["ratios"][j]
            if not np.isfinite(ratio):
                return dict(**base,status="INSUFFICIENT_OUTER_VALIDATION")
        else:
            tr=G["tr"]; va=G["va"]
            tr=tr[np.isfinite(y[tr]) & eligible[tr]]
            va=va[np.isfinite(y[va]) & eligible[va]]
            if len(tr)<12 or len(va)<3:
                return dict(**base,status="INSUFFICIENT_OBSERVED_TUNING")
            inner=context(tr,purity)
            if len(inner["feature_index"])==0:
                return dict(**base,status="FAILED_NO_VARIABLE_RNA_INNER",
                            n_eligible_inner=0,n_selected=0)
            ratio,pa=select_ratio(inner,y[tr],x[va],y[va],cohort[va],cov.iloc[va])
            pa["n_eligible_inner"]=pa.pop("n_eligible")
        outer=context(ft,purity)
        if len(outer["feature_index"])==0:
            return dict(**base,status="FAILED_NO_VARIABLE_RNA_OUTER",
                        n_eligible=0,n_selected=0,ratio=ratio,**pa)
        model=fit_final(outer,y[ft],ratio)
    except ValueError as ex:
        if str(ex)=="INSUFFICIENT_SUPPORTED_VALIDATION":
            return dict(**base,status=str(ex))
        raise
    except RuntimeError as ex:
        if str(ex)=="LASSO_NOT_CONVERGED":
            return dict(**base,status="FAILED_LASSO_NOT_CONVERGED")
        raise
    te=G["te"]; n_observed_test=int(np.isfinite(y[te]).sum())
    te=te[np.isfinite(y[te]) & eligible[te]]
    pred=predict(model,x[te],cohort[te],cov.iloc[te]) if len(te) else np.empty(0)
    cb=predict(model,x[te],cohort[te],cov.iloc[te],covariate_only=True) if len(te) else np.empty(0)
    ok=np.isfinite(pred)&np.isfinite(cb);te=te[ok];pred=pred[ok];cb=cb[ok]
    baseline=np.array([model["outcome_mean"][model["groups"].index(c)] for c in cohort[te]])
    return dict(**base,status="FITTED",model=model,rows=te,prediction=pred,
        baseline=baseline,covariate_baseline=cb,ratio=ratio,
        n_observed_test=n_observed_test,n_supported_test=len(te),**pa)

def success_status(audits, targets):
    """Keep every target in the denominator; all five folds plus final must fit."""
    rows=[]
    for arm,frame in audits.items():
        if frame.duplicated(["target_id","phase"]).any():
            raise ValueError("Duplicate target phase audit")
        lookup=frame.set_index(["target_id","phase"])
        for tid in targets:
            row=dict(arm=arm,target_id=tid)
            statuses=[]
            for phase in [f"outer{k}" for k in range(5)]+["final"]:
                key=(tid,phase)
                if key not in lookup.index:
                    status="NOT_TRAINABLE_OR_NO_MEASURED_Y"; selected=0
                else:
                    a=lookup.loc[key]
                    status=str(a.status); selected=int(a.get("n_selected",0) or 0) if pd.notna(a.get("n_selected",0)) else 0
                    if status=="FITTED" and selected==0:
                        raise ValueError("Empty RNA signature incorrectly marked FITTED")
                row[phase+"_status"]=status;row[phase+"_n_selected"]=selected
                statuses.append(status)
            row["successful_outer_folds"]=sum(s=="FITTED" for s in statuses[:5])
            row["zero_rna_outer_folds"]=sum(s in [FAIL_ZERO,"FAILED_NO_VARIABLE_RNA_INNER","FAILED_NO_VARIABLE_RNA_OUTER"] for s in statuses[:5])
            row["final_success"]=statuses[-1]=="FITTED"
            row["training_success"]=row["successful_outer_folds"]==5 and row["final_success"]
            row["attempted"]=any(s!="NOT_TRAINABLE_OR_NO_MEASURED_Y" for s in statuses)
            row["training_status"]="SUCCESS" if row["training_success"] else "FAILED" if row["attempted"] else "UNAVAILABLE"
            rows.append(row)
    return pd.DataFrame(rows)

def summarize(out,meta,features):
    import joblib
    from cc_hhgt.m0_three_arm import ARMS, matched_predictions,compare_target,regression_metrics
    import hashlib
    audits={a:pd.concat([pd.read_parquet(out/p/a/"AUDIT.parquet") for p in [f"outer{k}" for k in range(5)]+["final"]],ignore_index=True) for a in ARMS}
    status=success_status(audits,meta.target_id.tolist())
    frames={a:pd.concat([pd.read_parquet(out/f"outer{k}"/a/"PREDICTIONS.parquet") for k in range(5)],ignore_index=True) for a in ARMS}
    # Successful phase models must provide held-out predictions in every fold.
    for a in ARMS:
        count=frames[a].groupby("target_id").phase.nunique() if len(frames[a]) else pd.Series(dtype=int)
        mask=status.arm.eq(a)
        status.loc[mask,"evaluated_outer_folds"]=status.loc[mask,"target_id"].map(count).fillna(0).to_numpy()
    status["complete_fivefold"]=status.evaluated_outer_folds.eq(5)&status.successful_outer_folds.eq(5)
    status["training_success"] &= status.complete_fivefold
    status.loc[status.attempted & ~status.training_success,"training_status"]="FAILED"
    status=status.merge(meta,on="target_id",validate="many_to_one")
    status.to_csv(out/"TARGET_TRAINING_STATUS.tsv",sep="\t",index=False)
    successful={a:set(status.loc[status.arm.eq(a)&status.training_success,"target_id"]) for a in ARMS}
    common=set.intersection(*(successful[a] for a in ARMS))
    paired=matched_predictions({a:frames[a].loc[frames[a].target_id.isin(common)].copy() for a in ARMS})
    paired.to_parquet(out/"PAIRED_OOF_PREDICTIONS.parquet",index=False,compression="zstd")
    byarm=[];foldrows=[]
    for a in ARMS:
        for tid,f in frames[a].groupby("target_id"):
            for phase,ff in f.groupby("phase"):
                foldrows.append(dict(target_id=tid,arm=a,phase=phase,**regression_metrics(ff.observed,ff.predicted)))
            if tid in successful[a]:
                byarm.append(dict(target_id=tid,arm=a,**regression_metrics(f.observed,f.predicted)))
    metric=pd.DataFrame(byarm,columns=["target_id","arm","n","r2","rmse","mse","spearman"])
    status.merge(metric,on=["target_id","arm"],how="left",validate="one_to_one").to_csv(out/"ARM_TARGET_METRICS.tsv",sep="\t",index=False)
    pd.DataFrame(foldrows,columns=["target_id","arm","phase","n","r2","rmse","mse","spearman"]).to_csv(out/"FOLD_COMPARISON.tsv",sep="\t",index=False)
    comparisons=[]
    for tid,f in paired.groupby("target_id"):
        seed=int(hashlib.sha256(tid.encode()).hexdigest()[:8],16)
        comparisons.append(dict(target_id=tid,**compare_target(f,seed=seed)))
    # Include failed/unavailable pathways with missing metrics, never substitute
    # clinical-only predictions for a failed RNA model.
    result=meta.copy()
    columns=["n_patients","evaluated_folds","r2_covariates","delta_r2_combined_vs_mrna",
        "delta_r2_mrna_vs_lncrna","delta_r2_combined_vs_covariates",
        "best_arm","folds_combined_lower_mse_than_mrna",
        "delta_r2_conditional_bootstrap_low","delta_r2_conditional_bootstrap_high","bootstrap_replicates"]
    columns += [k+"_"+a for a in ARMS for k in ["r2","rmse","mse","spearman"]]
    tab=pd.DataFrame(comparisons,columns=["target_id"]+columns)
    result=result.merge(tab,on="target_id",how="left",validate="one_to_one")
    result["complete_fivefold"]=result.target_id.isin(common)
    result["comparison_status"]=np.where(result.complete_fivefold,"ALL_ARMS_SUCCESS","NOT_COMPARABLE_TRAINING_FAILURE_OR_UNAVAILABLE")
    result.to_csv(out/"TARGET_COMPARISON.tsv",sep="\t",index=False)
    aggregate=[]
    for layer in meta.layer.unique():
        f=result.loc[result.layer.eq(layer)&result.complete_fivefold]
        for a in ARMS:
            s=status.loc[status.layer.eq(layer)&status.arm.eq(a)]
            aggregate.append(dict(layer=layer,arm=a,n_targets=len(f),
                attempted_targets=int(s.attempted.sum()),successful_targets=int(s.training_success.sum()),
                failed_targets=int((s.attempted&~s.training_success).sum()),
                unavailable_targets=int((~s.attempted).sum()),
                median_oof_r2=f["r2_"+a].median(),median_oof_spearman=f["spearman_"+a].median(),
                positive_r2_targets=int(f["r2_"+a].gt(0).sum()),
                better_than_covariates_targets=int((f["r2_"+a]>f.r2_covariates+1e-9).sum()),
                uniquely_best_targets=int(f.best_arm.eq(a).sum()),tied_targets=int(f.best_arm.eq("TIE").sum()),
                median_delta_r2_combined_vs_mrna=f.delta_r2_combined_vs_mrna.median(),
                combined_better_than_mrna_targets=int(f.delta_r2_combined_vs_mrna.gt(1e-9).sum())))
    pd.DataFrame(aggregate).to_csv(out/"LAYER_SUMMARY.tsv",sep="\t",index=False)
    genes=[];nuisance=[]
    for a in ARMS:
        f=pd.read_parquet(out/"final"/a/"SELECTED_GENES.parquet")
        fold=pd.concat([pd.read_parquet(out/f"outer{k}"/a/"SELECTED_GENES.parquet") for k in range(5)],ignore_index=True)
        if len(fold):
            freq=fold.groupby(["target_id","feature_id"]).phase.nunique().rename("selected_outer_folds").reset_index()
            f=f.merge(freq,on=["target_id","feature_id"],how="left",validate="one_to_one")
            f["selected_outer_folds"]=f.selected_outer_folds.fillna(0).astype(int)
        else:f["selected_outer_folds"]=0
        f["selection_frequency"]=f.selected_outer_folds/5
        f["complete_training_success"]=f.target_id.isin(successful[a]);genes.append(f)
        for tid,m in joblib.load(out/"final"/a/"MODELS.joblib").items():
            for i,col in enumerate(m["design_spec"]["columns"]):
                nuisance.append(dict(arm=a,target_id=tid,covariate=col,coefficient=m["nuisance_coefficients"][i],penalized=False,complete_training_success=tid in successful[a]))
    allgenes=pd.concat(genes,ignore_index=True).merge(features,on="feature_id",how="left",validate="many_to_one")
    allgenes.to_csv(out/"ALL_NONEMPTY_FINAL_SIGNATURES.tsv.gz",sep="\t",index=False)
    allgenes.loc[allgenes.complete_training_success].to_csv(out/"FINAL_SIGNATURES.tsv.gz",sep="\t",index=False)
    pd.DataFrame(nuisance,columns=["arm","target_id","covariate","coefficient","penalized","complete_training_success"]).to_csv(out/"FINAL_COVARIATE_COEFFICIENTS.tsv",sep="\t",index=False)
    return dict(n_paired_targets=len(common),n_complete_fivefold_targets=len(common),
        n_paired_predictions=len(paired),by_layer=result.loc[result.complete_fivefold].groupby("layer").size().to_dict(),
        attempted_by_arm=status.loc[status.attempted].groupby("arm").size().to_dict(),
        successful_by_arm=status.loc[status.training_success].groupby("arm").size().to_dict(),
        failed_by_arm=status.loc[status.attempted&~status.training_success].groupby("arm").size().to_dict(),
        failure_rule="ANY_ZERO_RNA_OR_OTHER_FAILURE_IN_FIVE_OUTER_FOLDS_OR_FINAL_EXCLUDES_SUCCESS",
        performance_scope="COMMON_SUCCESSFUL_TARGETS_ONLY;FAILURE_COUNTS_RETAIN_ALL_ATTEMPTED_TARGETS")

def append_failure_reports(root,previous=None):
    """Separate failure coverage from conditional performance comparisons."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    from cc_hhgt.m0_three_arm import ARMS,ARM_LABELS
    root=Path(root);allstatus=[];allmetrics=[];reference=[]
    colors={"lncrna":"#3C9277","mrna":"#2878B5","combined":"#D97724"}
    for cancer in ["BRCA","LUAD","COAD"]:
        d=root/cancer;s=pd.read_csv(d/"TARGET_TRAINING_STATUS.tsv",sep="\t")
        m=pd.read_csv(d/"ARM_TARGET_METRICS.tsv",sep="\t")
        s["cancer_id"]=cancer;m["cancer_id"]=cancer;allstatus.append(s);allmetrics.append(m)
        layers=s.loc[s.attempted,"layer"].drop_duplicates().tolist()
        fig,ax=plt.subplots(figsize=(9,4.8));xx=np.arange(len(layers));width=.25
        for i,arm in enumerate(ARMS):
            values=[]
            for layer in layers:
                z=s.loc[s.arm.eq(arm)&s.layer.eq(layer)&s.attempted]
                values.append(100*(~z.training_success).mean() if len(z) else np.nan)
            ax.bar(xx+(i-1)*width,values,width,color=colors[arm],label=ARM_LABELS[arm])
        ax.set_xticks(xx,[z.replace("_"," ") for z in layers],rotation=15,ha="right")
        ax.set_ylim(0,100);ax.set_ylabel("Failed targets among all attempted (%)")
        ax.set_title(cancer+" | Failure includes zero RNA in any fold or final fit")
        ax.legend(frameon=False)
        for ext in ["png","pdf"]:fig.savefig(root/"figures"/(cancer+"_05_training_failure."+ext),dpi=300,bbox_inches="tight")
        plt.close(fig)
        if previous is None:continue
        old=Path(previous)/cancer
        ot=pd.read_csv(old/"TARGET_COMPARISON.tsv",sep="\t").set_index("target_id")
        for arm in ARMS:
            aud=pd.concat([pd.read_parquet(old/phase/arm/"AUDIT.parquet").assign(phase=phase)
                for phase in [f"outer{k}" for k in range(5)]+["final"]],ignore_index=True)
            old_attempted=set(aud.target_id)
            good=aud.status.eq("FITTED") & aud.n_selected.fillna(0).gt(0)
            counts=aud.loc[good].groupby("target_id").phase.nunique()
            old_success=set(counts.index[counts.eq(6)]) & set(ot.index[ot.complete_fivefold])
            for layer in layers:
                z=m.loc[m.arm.eq(arm)&m.layer.eq(layer)].set_index("target_id")
                old_ids=set(ot.index[ot.layer.eq(layer)])
                before=old_success & old_ids;after=set(z.index[z.training_success])
                common=sorted(before & after)
                if common:
                    zz=z.loc[common];oo=ot.loc[common]
                    if not np.array_equal(zz.n.to_numpy(),oo.n_patients.to_numpy()):
                        raise ValueError("Reference/new patient coverage mismatch")
                    old_r2=oo["r2_"+arm].to_numpy(float);new_r2=zz.r2.to_numpy(float)
                    finite=np.isfinite(old_r2)&np.isfinite(new_r2)
                    old_r2=old_r2[finite];new_r2=new_r2[finite]
                else:old_r2=new_r2=np.array([])
                delta=new_r2-old_r2
                reference.append(dict(cancer_id=cancer,layer=layer,arm=arm,
                    old_attempted_targets=len(old_attempted&old_ids),old_successful_targets=len(before),
                    new_attempted_targets=int(z.attempted.sum()),new_successful_targets=len(after),
                    both_successful_targets=len(delta),
                    old_median_r2_same_targets=float(np.median(old_r2)) if len(delta) else np.nan,
                    new_median_r2_same_targets=float(np.median(new_r2)) if len(delta) else np.nan,
                    median_paired_delta_r2=float(np.median(delta)) if len(delta) else np.nan,
                    improved_targets=int((delta>1e-9).sum()),worse_targets=int((delta< -1e-9).sum()),
                    tied_targets=int((np.abs(delta)<=1e-9).sum()),
                    old_zero_rna_reclassified_as_failure=True))
    st=pd.concat(allstatus,ignore_index=True);mt=pd.concat(allmetrics,ignore_index=True)
    st.to_csv(root/"ALL_TRAINING_STATUS.tsv",sep="\t",index=False)
    mt.to_csv(root/"ALL_ARM_TARGET_METRICS.tsv",sep="\t",index=False)
    rs=pd.DataFrame(reference)
    if len(rs):rs.to_csv(root/"SCREENING_POLICY_COMPARISON.tsv",sep="\t",index=False)
    failure=st.groupby(["cancer_id","layer","arm"]).agg(listed_targets=("target_id","size"),
        attempted_targets=("attempted","sum"),successful_targets=("training_success","sum")).reset_index()
    failure["failed_targets"]=failure.attempted_targets-failure.successful_targets
    failure["failure_fraction"]=failure.failed_targets/failure.attempted_targets.replace(0,np.nan)
    failure.to_csv(root/"TRAINING_FAILURE_SUMMARY.tsv",sep="\t",index=False)
    return dict(failed_targets_by_arm=st.loc[st.attempted&~st.training_success].groupby("arm").size().to_dict(),
        successful_targets_by_arm=st.loc[st.training_success].groupby("arm").size().to_dict(),
        compared_to_screened_reference=previous is not None,
        success_policy="Every outer fold and final refit must have at least one RNA coefficient",
        conditional_performance_warning="Common successful targets only; consult failure rates for all attempted targets")
