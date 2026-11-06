"""Compare the topology-motivated resolution-four follow-up with frozen controls."""
from __future__ import annotations
import argparse,html,json,socket,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cla_engine.common import sha
from scripts.prepare_graph_ridge import dump
from scripts.run_network_validation import check_receipt
from scripts.summarize_network_validation import sufficient_statistics
from scripts.bootstrap_network_validation import family_mean_bootstrap
from scripts.plot_m0_increment import classify,ARMS,CANCERS,LABELS,COLORS
from scripts.plot_network_validation import points,save,LAYERS,SHORT
from cc_hhgt.graph_ridge import target_policy
from cc_hhgt.m0_covariates import complete_rows

MAIN="module_resolution4_union"
NULLS=tuple(MAIN+"_random_"+str(s) for s in (101,202,303,404,505))
FINE=(MAIN,)+NULLS
BASES=("module_ridge_union","ridge_adaptive_omics","ridge_equal","rna_laplacian_union")
METHODS=BASES+FINE
KEYS=["target_id","variant","arm"]
PAIRKEYS=["cancer_id","repeat","target_id","display_layer","family","arm"]
PRIMARY=("PROTEIN_PATHWAY","PHOSPHO_PTMSEA")

def verified_file(folder,name):
    receipt=json.loads((folder/"COMPLETE.json").read_text())
    if sha(folder/name)!=receipt["files"][name]:
        raise ValueError("PHASE_FILE_CHECKSUM_DRIFT: "+str(folder/name))

def verify_pairing(fine,original_file):
    cols=["patient_id","target_id","variant","arm","observed"]
    original=pd.read_parquet(original_file,columns=cols,filters=[("variant","in",list(BASES))])
    base=original[original.variant.eq("ridge_equal")][["patient_id","target_id","arm","observed"]]
    base=base.rename(columns={"observed":"reference_observed"})
    candidates=pd.concat([original,fine[cols]],ignore_index=True)
    key=["patient_id","target_id","arm"]
    if base.duplicated(key).any() or candidates.duplicated(key+["variant"]).any():
        raise ValueError("DUPLICATE_PAIRED_PREDICTION")
    paired=candidates.merge(base,on=key,how="left",validate="many_to_one")
    if len(paired)!=len(candidates) or paired.reference_observed.isna().any():
        raise ValueError("UNPAIRED_PATIENT")
    if not np.array_equal(paired.observed.to_numpy(),paired.reference_observed.to_numpy()):
        raise ValueError("DIFFERENT_OBSERVED_TARGET")
    counts=candidates.groupby(["target_id","variant","arm"]).size().rename("actual").reset_index()
    expected=base.groupby(["target_id","arm"]).size().rename("expected").reset_index()
    check=counts.merge(expected,on=["target_id","arm"],how="left",validate="many_to_one")
    if not check.actual.eq(check.expected).all():
        raise ValueError("DIFFERENT_PATIENT_SUPPORT")
    return dict(paired_prediction_rows=len(paired),fine_prediction_rows=len(fine),
                patient_sets_exact=True,observed_values_exact=True)

def two_repeat_mean(frame,keys,value):
    count=frame.groupby(keys,dropna=False)["repeat"].nunique()
    keep=count[count.eq(2)].index
    if not len(keep):
        return pd.DataFrame(columns=keys+[value])
    selected=frame.set_index(keys).loc[keep].reset_index()
    return selected.groupby(keys,dropna=False,as_index=False)[value].mean()

def summarize(original_training,fine_training,inputs,original_summary,output):
    original_training=Path(original_training);fine_training=Path(fine_training)
    inputs=Path(inputs);original_summary=Path(original_summary);out=Path(output)
    original_receipt=json.loads((original_summary/"SUMMARY_COMPLETE.json").read_text())
    original_metrics=original_summary/"ALL_TARGET_METRICS.parquet"
    if sha(original_metrics)!=original_receipt["files"]["ALL_TARGET_METRICS.parquet"]:
        raise ValueError("ORIGINAL_SUMMARY_METRICS_CHANGED")
    met=[];statuses=[];phases=[];pairing=[];hyp=[];codes=[]
    for cancer in CANCERS:
        meta=classify(pd.read_csv(fine_training/cancer/"final/TARGETS.tsv",sep="\t"))
        meta["family"]=meta.family.fillna("UNSPECIFIED")
        native=pd.read_parquet(inputs/cancer/"Y.parquet")
        cov=pd.read_parquet(inputs/cancer/"COVARIATES.parquet").reindex(native.index)
        if np.isinf(native.to_numpy(float)).any():raise ValueError("INFINITE_NATIVE_TARGET")
        meta["n_observed_y"]=meta.target_id.map(native.notna().sum())
        meta["n_covariate_eligible_y"]=[
            int((native[r["target_id"]].notna().to_numpy()&complete_rows(cov,target_policy(r)["use_purity"])).sum())
            for r in meta.to_dict("records")]
        audits={};stats={0:[],1:[]}
        names=[f"repeat{r}_outer{f}" for r in range(2) for f in range(5)]+["final"]
        for phase in names:
            folder=fine_training/cancer/phase;original=original_training/cancer/phase
            if not (folder/"COMPLETE.json").exists():raise ValueError("INCOMPLETE_FINE_PHASE")
            check_receipt(folder)
            for name in ("PREDICTIONS.parquet","SPLIT.json","LAUNCH_PREFLIGHT.json"):
                verified_file(original,name)
            launch=json.loads((folder/"LAUNCH_PREFLIGHT.json").read_text())
            oldlaunch=json.loads((original/"LAUNCH_PREFLIGHT.json").read_text())
            if launch["target_host"]!="149" or oldlaunch["target_host"]!="149":
                raise ValueError("WRONG_EXECUTION_HOST")
            if launch["input_receipt"]!=oldlaunch["input_receipt"]:
                raise ValueError("PATIENT_INPUT_BINDING_CHANGED")
            if any(launch["code"].get(k)!=v for k,v in oldlaunch["code"].items()):
                raise ValueError("SHARED_TRAINING_CORE_CHANGED")
            if json.loads((folder/"SPLIT.json").read_text())!=json.loads((original/"SPLIT.json").read_text()):
                raise ValueError("ORIGINAL_FINE_SPLITS_DIFFER")
            codes.append(launch["code"])
            a=pd.read_parquet(folder/"AUDIT.parquet")
            if a.duplicated(KEYS).any() or set(a.variant)!=set(FINE) or set(a.arm)!=set(ARMS):
                raise ValueError("FINE_AUDIT_AXIS_ERROR")
            if set(a.target_id)!=set(meta.target_id) or not a.groupby("target_id").size().eq(18).all():
                raise ValueError("MISSING_FINE_TARGET_ATTEMPTS")
            audits[phase]=a[KEYS+["status"]]
            fitted=a[a.status.eq("FITTED")].merge(meta[["target_id","display_layer","family"]],on="target_id",validate="many_to_one")
            if not fitted.inner_validations.eq(6).all() or not fitted.n_inner_train_min.ge(12).all() or not fitted.n_inner_validation_min.ge(3).all():
                raise ValueError("NESTED_CV_THRESHOLD_FAILED")
            for layer,excluded in (("PROTEIN_PATHWAY","PROTEIN"),("PHOSPHO_PTMSEA","PHOSPHO")):
                if fitted.loc[fitted.display_layer.eq(layer),"allowed_blocks"].str.split(";").map(lambda v:excluded in v).any():
                    raise ValueError("TARGET_ASSAY_LEAK")
            h=fitted.groupby(["variant","arm","display_layer","parameter","lam"]).size().rename("fits").reset_index()
            h["cancer_id"]=cancer;h["phase"]=phase;hyp.append(h)
            phases.append(dict(cancer_id=cancer,phase=phase,attempts=len(a),fitted=len(fitted),
                               failed=int(a.status.ne("FITTED").sum()),checksums_verified=True,
                               shared_core_matches=True,split_matches=True))
            if phase!="final":
                p=pd.read_parquet(folder/"PREDICTIONS.parquet")
                if not np.isfinite(p[["observed","predicted"]].to_numpy()).all():
                    raise ValueError("NONFINITE_PREDICTION")
                record=verify_pairing(p,original/"PREDICTIONS.parquet")
                pairing.append(dict(cancer_id=cancer,phase=phase,**record))
                stats[int(phase[6])].append(sufficient_statistics(p))
                print("FINE_PHASE_VERIFIED",cancer,phase,len(p),flush=True)
                del p
        for repeat in (0,1):
            a=pd.concat([audits[f"repeat{repeat}_outer{k}"].assign(phase=str(k)) for k in range(5)]+[audits["final"].assign(phase="final")],ignore_index=True)
            success=a.assign(fitted=a.status.eq("FITTED")).groupby(KEYS).agg(fitted_phases=("fitted","sum"),attempted_phases=("phase","size"))
            success["success"]=success.fitted_phases.eq(6)&success.attempted_phases.eq(6)
            status=success.reset_index().merge(meta,on="target_id",validate="many_to_one")
            status["repeat"]=repeat;status["cancer_id"]=cancer;statuses.append(status)
            s=pd.concat(stats[repeat]).groupby(level=KEYS).sum().join(success[["success"]])
            s["sst"]=s.sy2-s.sy**2/s.n
            s["r2"]=1-s.sse/s.sst;s["rmse"]=np.sqrt(s.sse/s.n)
            denominator=np.sqrt(np.maximum(s.sst*(s.sp2-s.sp**2/s.n),0))
            s["pearson"]=(s.syp-s.sy*s.sp/s.n)/denominator
            s=s.reset_index().merge(meta,on="target_id",validate="many_to_one")
            s["repeat"]=repeat;s["cancer_id"]=cancer;met.append(s)
    if any(c!=codes[0] for c in codes):raise ValueError("MIXED_FINE_TRAINING_CODE")
    fine=pd.concat(met,ignore_index=True);status=pd.concat(statuses,ignore_index=True)
    old=pd.read_parquet(original_metrics)
    joined=pd.concat([old,fine],ignore_index=True,sort=False)
    for name,frame in (("FINE_TARGET_METRICS.parquet",fine),("FINE_TRAINING_STATUS.parquet",status),("COMBINED_TARGET_METRICS.parquet",joined)):
        frame.to_parquet(out/name,index=False)
    pd.DataFrame(phases).to_csv(out/"FINE_PHASE_AUDIT.tsv",sep="\t",index=False)
    pd.DataFrame(pairing).to_csv(out/"PAIRED_PHASE_IDENTITIES.tsv",sep="\t",index=False)
    pd.concat(hyp,ignore_index=True).to_csv(out/"FINE_HYPERPARAMETERS.tsv",sep="\t",index=False)
    good=joined[joined.success&np.isfinite(joined.r2)&joined.variant.isin(METHODS)&joined.arm.isin(ARMS)]
    wide=good.pivot(index=PAIRKEYS,columns="variant",values="r2").reset_index()
    counts=good.pivot(index=PAIRKEYS,columns="variant",values="n").reset_index()
    paired=[]
    for baseline in BASES+NULLS:
        eligible=counts.dropna(subset=[MAIN,baseline])
        if not eligible[MAIN].eq(eligible[baseline]).all():raise ValueError("PAIRED_METRIC_SUPPORT_DIFFERS")
        p=wide.dropna(subset=[MAIN,baseline]).copy()
        p["method"]=MAIN;p["baseline"]=baseline;p["delta_r2"]=p[MAIN]-p[baseline]
        paired.append(p[PAIRKEYS+["method","baseline","delta_r2"]])
    p=wide.dropna(subset=[MAIN,*NULLS]).copy()
    p["method"]=MAIN;p["baseline"]="mean_five_resolution4_nulls";p["delta_r2"]=p[MAIN]-p[list(NULLS)].mean(axis=1)
    paired.append(p[PAIRKEYS+["method","baseline","delta_r2"]])
    paired=pd.concat(paired,ignore_index=True)
    paired.to_parquet(out/"FINE_PAIRED_MODEL_DELTAS.parquet",index=False)
    paired.groupby(["cancer_id","repeat","arm","display_layer","baseline"]).agg(
        targets=("target_id","size"),median_delta_r2=("delta_r2","median"),
        mean_delta_r2=("delta_r2","mean"),improved=("delta_r2",lambda x:int(x.gt(1e-9).sum()))
    ).reset_index().to_csv(out/"FINE_GAIN_SUMMARY.tsv",sep="\t",index=False)
    selected=good[good.variant.eq(MAIN)]
    three=selected.pivot(index=["cancer_id","repeat","target_id","display_layer","family"],columns="arm",values="r2").dropna().reset_index()
    three["combined_minus_mrna"]=three.combined-three.mrna
    three["lncrna_minus_mrna"]=three.lncrna-three.mrna
    three.to_parquet(out/"FINE_THREE_ARM_COMPARISONS.parquet",index=False)
    three.groupby(["cancer_id","repeat","display_layer"]).agg(
        targets=("target_id","size"),median_lncrna=("lncrna","median"),median_mrna=("mrna","median"),
        median_combined=("combined","median"),median_combined_minus_mrna=("combined_minus_mrna","median"),
        combined_improved=("combined_minus_mrna",lambda x:int(x.gt(1e-9).sum()))
    ).reset_index().to_csv(out/"FINE_THREE_ARM_SUMMARY.tsv",sep="\t",index=False)
    good.groupby(["cancer_id","repeat","variant","arm","display_layer"]).agg(
        targets=("target_id","size"),median_r2=("r2","median"),mean_r2=("r2","mean"),
        positive=("r2",lambda x:int(x.gt(0).sum())),median_patients=("n","median")
    ).reset_index().to_csv(out/"FINE_MODEL_SUMMARY.tsv",sep="\t",index=False)
    coverage=status.groupby(["cancer_id","repeat","variant","arm","display_layer"]).agg(
        attempted=("target_id","size"),successful=("success","sum")).reset_index()
    coverage["failed"]=coverage.attempted-coverage.successful
    coverage.to_csv(out/"FINE_COVERAGE_SUMMARY.tsv",sep="\t",index=False)
    valid=fine[fine.success&np.isfinite(fine.r2)].copy()
    valid["evaluated_fraction"]=valid.n/valid.n_observed_y
    valid.groupby(["cancer_id","repeat","variant","arm","display_layer"]).agg(
        targets=("target_id","size"),median_observed_y=("n_observed_y","median"),
        median_clinical_eligible_y=("n_covariate_eligible_y","median"),
        median_evaluated_patients=("n","median"),minimum_evaluated_patients=("n","min"),
        median_evaluated_fraction=("evaluated_fraction","median")
    ).reset_index().to_csv(out/"FINE_PATIENT_COVERAGE.tsv",sep="\t",index=False)
    dump(out/"FINE_SUMMARY_COMPLETE.json",dict(target_host="149",status="COMPLETE",phases=33,
        paired_prediction_phases=30,original_summary_sha256=sha(original_summary/"SUMMARY_COMPLETE.json"),
        fine_training_code=codes[0],files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in (".parquet",".tsv")}))
    return joined

def bootstrap(original_training,fine_training,inputs,output,replicates=2000):
    original_training=Path(original_training);fine_training=Path(fine_training)
    inputs=Path(inputs);out=Path(output)
    met=pd.read_parquet(out/"COMBINED_TARGET_METRICS.parquet")
    met=met[met.success&np.isfinite(met.r2)&met.display_layer.isin(PRIMARY)&met.variant.isin(METHODS)&met.arm.isin(ARMS)]
    rows=[];aggregate={};cancer_sets={}
    for ci,cancer in enumerate(CANCERS):
        m=met[met.cancer_id.eq(cancer)]
        counts=m.groupby("target_id").size()
        tids=sorted(counts[counts.eq(2*len(METHODS)*len(ARMS))].index)
        if not tids:continue
        meta=m.drop_duplicates("target_id").set_index("target_id").loc[tids]
        pids=pd.read_parquet(inputs/cancer/"COVARIATES.parquet").index
        ti={v:i for i,v in enumerate(tids)};pi={v:i for i,v in enumerate(pids)}
        mi={v:i for i,v in enumerate(METHODS)};ai={v:i for i,v in enumerate(ARMS)}
        predicted=np.full((2,len(METHODS),3,len(tids),len(pids)),np.nan)
        for repeat in range(2):
            for fold in range(5):
                phase=f"repeat{repeat}_outer{fold}"
                for root,methods in ((original_training,BASES),(fine_training,FINE)):
                    p=pd.read_parquet(root/cancer/phase/"PREDICTIONS.parquet",
                        columns=["patient_id","target_id","variant","arm","predicted"],
                        filters=[("target_id","in",tids),("variant","in",list(methods)),("arm","in",list(ARMS))])
                    predicted[repeat,p.variant.map(mi).to_numpy(int),p.arm.map(ai).to_numpy(int),
                        p.target_id.map(ti).to_numpy(int),p.patient_id.map(pi).to_numpy(int)]=p.predicted.to_numpy()
        y=pd.read_parquet(inputs/cancer/"Y.parquet",columns=tids).reindex(pids).to_numpy(float).T
        valid=np.isfinite(predicted).all(axis=(0,1,2))&np.isfinite(y)
        nn=valid.sum(1);mean=np.nansum(np.where(valid,y,np.nan),axis=1)/np.maximum(nn,1)
        variance=np.nansum(np.where(valid,(y-mean[:,None])**2,np.nan),axis=1)/np.maximum(nn,1)
        eligible=(nn>=12)&(variance>1e-12)
        errors=np.mean((predicted-y[None,None,None])**2,axis=0)/np.maximum(variance[None,None,:,None],1e-30)
        rng=np.random.default_rng(20261106+ci)
        draw=rng.multinomial(len(pids),np.repeat(1/len(pids),len(pids)),size=replicates)
        contrasts=[]
        for arm in ARMS:
            for baseline in BASES:
                contrasts.append((arm,MAIN,baseline,errors[mi[baseline],ai[arm]]-errors[mi[MAIN],ai[arm]]))
            null=np.mean(errors[[mi[n] for n in NULLS],ai[arm]],axis=0)
            contrasts.append((arm,MAIN,"mean_five_resolution4_nulls",null-errors[mi[MAIN],ai[arm]]))
        for comparison,arm in (("combined_vs_mrna","combined"),("lncrna_vs_mrna","lncrna")):
            contrasts.append((comparison,MAIN,MAIN,errors[mi[MAIN],ai["mrna"]]-errors[mi[MAIN],ai[arm]]))
        for layer in PRIMARY:
            take=(meta.display_layer.to_numpy()==layer)&eligible
            if not take.any():continue
            families=meta.family.fillna("UNSPECIFIED").to_numpy()[take]
            for arm,method,baseline,delta in contrasts:
                point,boots=family_mean_bootstrap(delta[take],valid[take],families,draw)
                key=(layer,arm,method,baseline)
                aggregate.setdefault(key,[]).append((point,boots));cancer_sets.setdefault(key,[]).append(cancer)
                rows.append(dict(cancer_id=cancer,layer=layer,arm=arm,method=method,baseline=baseline,
                    targets=int(take.sum()),candidate_patients=len(pids),minimum_common_patients=int(nn[take].min()),
                    median_common_patients=float(np.median(nn[take])),family_balanced_mean_delta_r2=point,
                    ci_low=float(np.percentile(boots,2.5)),ci_high=float(np.percentile(boots,97.5)),replicates=replicates))
        print("FINE_PATIENT_BOOTSTRAP",cancer,int(eligible.sum()),flush=True)
    for key,values in aggregate.items():
        point=float(np.mean([p for p,b in values]));boots=np.mean([b for p,b in values],axis=0)
        layer,arm,method,baseline=key
        rows.append(dict(cancer_id="EQUAL_CANCER_MEAN",layer=layer,arm=arm,method=method,baseline=baseline,
            family_balanced_mean_delta_r2=point,ci_low=float(np.percentile(boots,2.5)),
            ci_high=float(np.percentile(boots,97.5)),replicates=replicates,cancers=";".join(cancer_sets[key])))
    pd.DataFrame(rows).to_csv(out/"FINE_PATIENT_BOOTSTRAP.tsv",sep="\t",index=False)
    dump(out/"FINE_BOOTSTRAP_COMPLETE.json",dict(target_host="149",status="COMPLETE",replicates=replicates,
        pairing="Same patient counts shared by both outer repetitions and all compared models",
        variance="Original common-support target variance held fixed",
        null="Mean of five fitted prediction errors; not an averaged-prediction ensemble",
        interval="Conditional on fitted CV models; no retraining or external validation",
        files={"FINE_PATIENT_BOOTSTRAP.tsv":sha(out/"FINE_PATIENT_BOOTSTRAP.tsv")}))
    return pd.DataFrame(rows)

def plot(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    out=Path(output)
    fonts={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family":"Arial" if "Arial" in fonts else "DejaVu Sans","font.size":11,
                        "axes.titlesize":13,"pdf.fonttype":42,"ps.fonttype":42})
    met=pd.read_parquet(out/"FINE_TARGET_METRICS.parquet")
    good=met[met.success&np.isfinite(met.r2)&met.variant.eq(MAIN)]
    avg=two_repeat_mean(good,["cancer_id","target_id","arm","display_layer"],"r2")
    limits=(min(-.1,float(avg.r2.min())*1.05),max(.1,float(avg.r2.max())*1.05)) if len(avg) else (-.1,.1)
    for i,arm in enumerate(ARMS,1):
        fig,axes=plt.subplots(1,3,figsize=(16,5.5),sharey=True)
        for ax,cancer in zip(axes,CANCERS):
            data=avg[avg.arm.eq(arm)&avg.cancer_id.eq(cancer)]
            points(ax,data,"r2",COLORS[arm]);ax.set_ylim(limits)
            ax.set_title(cancer+" | "+str(len(data))+" targets")
        axes[0].set_ylabel("Held-out R2, mean of two outer CV repeats")
        fig.suptitle(LABELS[arm]+" arm: resolution-four network modules\nSame measured assays, targets and clinical covariates")
        save(fig,out,f"{i:02d}_fine_{arm}_performance")
    three=pd.read_parquet(out/"FINE_THREE_ARM_COMPARISONS.parquet")
    fig,axes=plt.subplots(2,3,figsize=(16,10),sharey="row")
    for row,(field,title,color) in enumerate((("combined_minus_mrna","lncRNA + mRNA minus mRNA",COLORS["combined"]),
                                             ("lncrna_minus_mrna","lncRNA minus mRNA",COLORS["lncrna"]))):
        averaged=two_repeat_mean(three,["cancer_id","target_id","display_layer"],field)
        for ax,cancer in zip(axes[row],CANCERS):
            points(ax,averaged[averaged.cancer_id.eq(cancer)],field,color);ax.set_title(cancer+" | "+title)
        axes[row,0].set_ylabel("Paired Delta R2")
    fig.suptitle("Three RNA arms under the finer network prior\nSame eligible patient omics and unpenalized clinical covariates")
    save(fig,out,"04_fine_three_arm_comparison")
    paired=pd.read_parquet(out/"FINE_PAIRED_MODEL_DELTAS.parquet")
    selected=paired[paired.baseline.isin(("module_ridge_union","ridge_adaptive_omics"))]
    averaged=two_repeat_mean(selected,["cancer_id","target_id","arm","display_layer","baseline"],"delta_r2")
    fig,axes=plt.subplots(2,3,figsize=(16,10),sharey="row")
    for row,(baseline,title) in enumerate((("module_ridge_union","Fine minus coarse modules"),
                                           ("ridge_adaptive_omics","Fine minus adaptive omics"))):
        for ax,arm in zip(axes[row],ARMS):
            points(ax,averaged[averaged.arm.eq(arm)&averaged.baseline.eq(baseline)],"delta_r2",COLORS[arm])
            ax.set_title(LABELS[arm]+" | "+title)
        axes[row,0].set_ylabel("Paired Delta R2")
    fig.suptitle("Finer modules compared on matching patients and targets\nCancer-target pairs; mean of two outer CV repeats")
    save(fig,out,"05_fine_vs_coarse_and_adaptive")
    boot=pd.read_csv(out/"FINE_PATIENT_BOOTSTRAP.tsv",sep="\t")
    pooled=boot[boot.cancer_id.eq("EQUAL_CANCER_MEAN")]
    bases=BASES+("mean_five_resolution4_nulls",)
    names=["Coarse network modules","Adaptive omics ridge","Equal omics ridge","RNA Laplacian","Five fine random networks"]
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    for row,layer in enumerate(PRIMARY):
        for ax,arm in zip(axes[row],ARMS):
            for i,baseline in enumerate(bases):
                f=pooled[pooled.layer.eq(layer)&pooled.arm.eq(arm)&pooled.baseline.eq(baseline)]
                if len(f):
                    v=f.iloc[0];ax.plot([v.ci_low,v.ci_high],[i,i],color=COLORS[arm],lw=2)
                    ax.scatter(v.family_balanced_mean_delta_r2,i,color=COLORS[arm],s=35)
            ax.set_yticks(range(len(bases)),names);ax.invert_yaxis()
            ax.axvline(0,color="gray",ls="--");ax.set_title(SHORT[layer]+" | "+LABELS[arm])
            ax.set_xlabel("Mean paired Delta R2 (95% patient bootstrap)")
    fig.suptitle("Fine network modules versus five controls\nEqual cancer and score-family weighting; intervals conditional on fitted models")
    save(fig,out,"06_fine_primary_patient_bootstrap")
    tables=["FINE_THREE_ARM_SUMMARY.tsv","FINE_GAIN_SUMMARY.tsv","FINE_PATIENT_BOOTSTRAP.tsv",
            "FINE_MODEL_SUMMARY.tsv","FINE_COVERAGE_SUMMARY.tsv","FINE_PATIENT_COVERAGE.tsv",
            "FINE_HYPERPARAMETERS.tsv","FINE_PHASE_AUDIT.tsv","PAIRED_PHASE_IDENTITIES.tsv"]
    page="""<!doctype html><meta charset="utf-8"><title>Fine network validation</title>
<style>body{font:16px Arial;max-width:1250px;margin:32px auto;padding:20px;color:#213547}p{line-height:1.6}img{width:100%;border:1px solid #ddd}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd}a{color:#176294}</style>
<h1>Resolution-four network validation</h1>"""
    page+="<p>This exploratory follow-up was motivated by a topology audit: at resolution one, roughly 99% of connected lncRNA anchors belonged to one module. A topology-only probe selected resolution four before predictive metrics were inspected; the largest lncRNA group then contained 17%?22% of connected lncRNAs. This structural choice was not part of the original prespecified experiment and is not evidence of better prediction by itself.</p>"
    page+="<p>The same three cancers, all candidate targets, three RNA arms, measured patient assays, clinical adjustment and target-source exclusions are retained. Both outer repetitions contain five folds and every fit uses six inner validations. Every fine prediction is paired with the same held-out patient and observed label in the original controls. Successful evaluation requires all five folds and final fit; failed targets and negative R2 remain in the tables.</p>"
    page+="<p>Native scores retain their original cohort preprocessing. Protein abundance scores and unadjusted phosphosite scores are retrospective assay-derived targets, not absolute pathway activation. RNA-derived targets in mRNA-containing arms are score reconstruction. These results are not independent external or causal validation; ridge coefficients are dense rather than a Lasso-selected signature.</p>"
    page+="<p>Plots 1?5 average each cancer-target result over both outer repetitions. Bootstrap intervals use patient-level paired resampling, common support across both repeats and models, fixed original target variance, equal target weights within score families, and equal cancer/family weights. They condition on fitted models and do not refit training. The null reference averages five prediction errors, not five predictions. Nulls preserve node degrees and edge-weight multisets, but not weighted strengths or community sizes.</p>"
    page+="<p><a href='../OPEN_REPORT.html'>Original coarse-module and Laplacian report</a></p>"
    page+="<h2>Primary controls</h2>"+pooled[pooled.arm.isin(ARMS)].to_html(index=False,float_format=lambda x:f"{x:.5f}")
    page+="<h2>Three RNA arms</h2>"+pd.read_csv(out/"FINE_THREE_ARM_SUMMARY.tsv",sep="\t").to_html(index=False,float_format=lambda x:f"{x:.5f}")
    page+="<h2>Tables</h2><ul>"+"".join("<li><a href='"+n+"'>"+n+"</a></li>" for n in tables)+"</ul>"
    for path in sorted(out.glob("*.png")):
        page+="<h2>"+html.escape(path.stem.replace("_"," "))+"</h2><p><a href='"+path.with_suffix(".pdf").name+"'>PDF</a></p><img src='"+path.name+"'>"
    (out/"OPEN_REPORT.html").write_text(page,encoding="utf-8")
    dump(out/"FINE_PLOT_COMPLETE.json",dict(target_host="149",status="COMPLETE",figures=6,
        files={p.name:sha(p) for p in out.iterdir() if p.suffix in (".png",".pdf",".html")}))

def report(original_training,fine_training,inputs,original_summary,output,replicates=2000):
    if socket.gethostname()!="149":raise RuntimeError("HOST_149_REQUIRED")
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    dump(out/"PREFLIGHT.json",dict(target_host="149",stage="RESOLUTION4_COMPARATIVE_REPORT",
        code_sha256=sha(Path(__file__)),original_training=str(original_training),
        fine_training=str(fine_training),inputs=str(inputs),original_summary=str(original_summary),
        replicates=replicates,review_complete=False))
    summarize(original_training,fine_training,inputs,original_summary,out)
    bootstrap(original_training,fine_training,inputs,out,replicates)
    plot(out)
    dump(out/"REPORT_READY_FOR_REVIEW.json",dict(target_host="149",status="GENERATED_REVIEW_PENDING",
        original_and_fine_paired_phases=30,fine_phases=33,figures=6,
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="REPORT_READY_FOR_REVIEW.json"}))
    print("FINE_REPORT_READY_FOR_REVIEW",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("original_training","fine_training","inputs","original_summary","output"):
        parser.add_argument("--"+name.replace("_","-"),required=True)
    parser.add_argument("--replicates",type=int,default=2000)
    report(**vars(parser.parse_args()))
