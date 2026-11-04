#!/usr/bin/env python3
"""Run M0 lncRNA-only Ridge regressions against current native Pathway/State scores."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cc_hhgt.m0_linear import grouped_oof, metrics, LAMBDAS
from cla_engine.cptac_binding import load_assay, load_scores
from cla_engine.common import sha

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=True,allow_nan=False)+"\n")
    tmp.replace(path)

def source_read(path, reader, sources, expected=None):
    path=Path(path);before=sha(path)
    if expected and before!=expected:raise ValueError("SOURCE_HASH_CHANGED: "+str(path))
    result=reader(path)
    if sha(path)!=before:raise ValueError("SOURCE_CHANGED_WHILE_READING: "+str(path))
    sources.append(dict(path=str(path),sha256=before,bytes=path.stat().st_size))
    return result

def code_hash():
    paths=[Path(__file__),ROOT/"cc_hhgt/m0_linear.py",ROOT/"cc_hhgt/v31_continuous_baselines.py",
           ROOT/"cla_engine/cptac_binding.py"]
    return hashlib.sha256("".join(sha(p) for p in paths).encode()).hexdigest()

def tasks(cfg,datasets,cancer_type,cancer_list):
    all_tasks=[]
    if "TCGA" in datasets:
        folds=pd.read_csv(cfg["sources"]["patient_map"]["path"],sep="\t")
        all_tasks.extend(dict(dataset="TCGA",cohort=c,cancer_id=c) for c in sorted(folds.cancer_id.unique()))
    if "CPTAC" in datasets:
        b=Path(cfg["cptac_input"]["complete"]["path"]).parent
        pts=pd.read_csv(b/"PATIENTS.tsv",sep="\t")
        all_tasks.extend(dict(dataset="CPTAC",cohort=r.source_cohort,cancer_id=r.cancer_id)
                         for r in pts[["source_cohort","cancer_id"]].drop_duplicates().itertuples())
    if cancer_type=="list":
        if not cancer_list:raise ValueError("cancer_type=list requires --cancer-list")
        unknown=set(cancer_list)-{t["cancer_id"] for t in all_tasks}-{t["cohort"] for t in all_tasks}
        if unknown:raise ValueError("Unknown cancers: "+str(sorted(unknown)))
        all_tasks=[t for t in all_tasks if t["cancer_id"] in cancer_list or t["cohort"] in cancer_list]
    elif cancer_list:raise ValueError("Do not supply cancer_list when cancer_type=all")
    if not all_tasks:raise ValueError("Empty scope")
    return all_tasks

def load_case(cfg,task,sources):
    dataset,cohort=task["dataset"],task["cohort"]
    metadata=[]
    # GENCODE v36 annotation, independent of patient expression/outcomes.
    binding=cfg["cptac_input"];bp=Path(binding["complete"]["path"])
    complete=source_read(bp,lambda p:json.loads(p.read_text()),sources,binding["complete"]["sha256"])
    ap=Path(complete["config"]["aligned_complete"]["path"])
    ar=source_read(ap,lambda p:json.loads(p.read_text()),sources,complete["config"]["aligned_complete"]["sha256"])
    feature_path=ap.parent/"FEATURE_ALIGNMENT.parquet"
    ann=source_read(feature_path,pd.read_parquet,sources,ar["files"][feature_path.name]["sha256"])
    ann=ann.loc[ann.layer.eq("rna") & ann.gene_biotype.eq("lncRNA")].drop_duplicates("gene_id")
    lnc=set("LNC:"+ann.gene_id.astype(str));symbols=dict(zip("LNC:"+ann.gene_id.astype(str),ann.reference_symbol))
    score_columns=[r["column"] for r in cfg["targets"]]
    if dataset=="TCGA":
        f=cfg["sources"]["patient_map"]
        authority=source_read(f["path"],lambda p:pd.read_csv(p,sep="\t"),sources,f["sha256"])
        authority=authority.loc[authority.cancer_id.eq(cohort)].sort_values("patient_id").copy()
        if authority.patient_id.duplicated().any() or authority.sample_id.duplicated().any():
            raise ValueError("Duplicate patient/sample authority")
        ep=Path(cfg["sources"]["lnc_expression_root"]["path"])/f"cancer_id={cohort}/part-0.parquet"
        raw=source_read(ep,pd.read_parquet,sources)
        raw=raw.merge(authority[["sample_id","patient_id"]],on=["sample_id","patient_id"],validate="many_to_one")
        if raw.duplicated(["patient_id","lncrna_id"]).any():raise ValueError("Duplicate lncRNA expression")
        pre_count=raw.lncrna_id.nunique()
        raw=raw.loc[raw.lncrna_id.isin(lnc)]
        x=raw.pivot(index="patient_id",columns="lncrna_id",values="logcpm")
        authority=authority.loc[authority.patient_id.isin(x.index)]
        x=x.reindex(authority.patient_id).sort_index(axis=1)
        pa=Path(cfg["sources"]["pathway_activity_root"]["path"])/f"cancer_id={cohort}/part-0.parquet"
        act=source_read(pa,pd.read_parquet,sources)
        act=act.merge(authority[["sample_id","patient_id"]],on=["sample_id","patient_id"],validate="many_to_one")
        if not set(act.score_method.astype(str)) <= {"ssGSEA_GSVA_2.4.7", "weighted_mean_DoRothEA_ABC", "weighted_mean_PROGENy_full"}:
            raise ValueError("M0 expects verified current RNA scores; unknown or retired methods rejected")
        if act.duplicated(["patient_id","pathway_id"]).any():raise ValueError("Duplicate pathway labels")
        if "quality_flag" in act:act.loc[~act.quality_flag.eq("pass"),"activity_score"]=np.nan
        yp=act.pivot(index="patient_id",columns="pathway_id",values="activity_score").reindex(x.index)
        yp.columns=["RNA_PATHWAY|"+str(t) for t in yp.columns]
        for tid,methods in act.groupby("pathway_id",sort=True).score_method:
            metadata.append(dict(target_id="RNA_PATHWAY|"+tid,native_target_id=tid,layer="RNA_PATHWAY",
                                 family="RNA_TF" if tid.startswith("DOROTHEA:") else "RNA_PROGENY" if tid.startswith("PROGENY:") else "RNA_PATHWAY",method=";".join(sorted(set(methods)))))
        st=cfg["sources"]["state"]
        state=source_read(st["path"],pd.read_parquet,sources,st["sha256"])
        state=state.loc[state.cancer_id.eq(cohort)].set_index("patient_id")
        ys=state.reindex(x.index)[score_columns]
        folds=authority.set_index("patient_id").patient_fold_id.reindex(x.index).to_numpy(int)
        unit="native_edgeR_TMM_logCPM"
        exposure="TCGA current frozen cohort TMM and expression eligibility; not fold-local raw RNA normalization"
    else:
        assay=load_assay(binding,cohort=cohort,layer="rna")
        raw=assay["values"].copy()
        pre_count=len(raw)
        raw.index="LNC:"+raw.index.astype(str)
        x=raw.loc[raw.index.isin(lnc)].T
        x.index=x.index.str.removesuffix("::T")
        x=x.sort_index().sort_index(axis=1)
        rel=f"matrices/{cohort}/rna_TumorNormal.parquet"
        sources.append(dict(path=str(ap.parent/rel),sha256=ar["files"][rel]["sha256"]))
        mats=[]
        idx=source_read(bp.parent/"SCORE_INDEX.tsv",lambda p:pd.read_csv(p,sep="\t"),sources,
                        complete["files"]["SCORE_INDEX.tsv"]["sha256"])
        for layer in ["RNA_PATHWAY","PROTEIN_PATHWAY","METHYLATION_PATHWAY","PHOSPHO_PTMSEA"]:
            got=load_scores(binding,cohort=cohort,layer=layer)["values"].T.reindex(x.index)
            row=idx.loc[idx.source_cohort.eq(cohort)&idx.layer.eq(layer)].iloc[0]
            sources.append(dict(path=str(bp.parent/row.path),sha256=row.sha256))
            for tid in got.columns:
                metadata.append(dict(target_id=layer+"|"+tid,native_target_id=tid,layer=layer,
                                     family=layer,method=row.method))
            got.columns=[layer+"|"+str(t) for t in got.columns]
            mats.append(got)
        yp=pd.concat(mats,axis=1)
        ys=load_scores(binding,cohort=cohort,layer="STATE")["values"].T.reindex(x.index)[score_columns]
        sources.append(dict(path=str(bp.parent/"STATE_PATIENT.parquet"),
                            sha256=complete["files"]["STATE_PATIENT.parquet"]["sha256"]))
        # Balanced, deterministic assignment independent of X and Y; fixed patient IDs.
        order=sorted(x.index,key=lambda p:hashlib.sha256(("M0:20261104:"+p).encode()).hexdigest())
        fmap={p:i%5 for i,p in enumerate(order)}
        folds=np.array([fmap[p] for p in x.index])
        unit="log2(TPM+1)"
        exposure="CPTAC supplied log2(TPM+1); some frozen targets use cohort/global normalization"
    for r in cfg["targets"]:
        metadata.append(dict(target_id="STATE|"+r["column"],native_target_id=r["column"],
                             layer="STATE",family=r["family"],method=r["method"]))
    ys.columns=["STATE|"+c for c in ys.columns]
    y=pd.concat([yp,ys],axis=1)
    if not x.index.is_unique or not x.columns.is_unique or not y.columns.is_unique:
        raise ValueError("Duplicate model axes")
    if not x.columns.str.startswith("LNC:ENSG").all():raise ValueError("Non-lncRNA predictors")
    info=dict(**task,n_patients=len(x),n_features=x.shape[1],features_before_annotation=pre_count,
              n_targets=y.shape[1],expression_unit=unit,upstream_exposure=exposure,
              cohort_pooling=False,model_inputs=["lncRNA_expression"],
              labels="FROZEN_NATIVE_SCORES_NOT_ABSOLUTE_ACTIVATION",
              fully_inductive_raw_pipeline=False,normalization_fit_in_m0="training_patients_only",
              target_weights={"RNA_PATHWAY":1.0,"STATE":0.5},
              weight_note="Independent regressions: each task weight is a constant and cancels in its objective.")
    feature=pd.DataFrame(dict(lncrna_id=x.columns,gene_symbol=[symbols.get(c,"") for c in x.columns]))
    return x,y,folds,pd.DataFrame(metadata),feature,info

def run_case(config_path,task,root):
    t0=time.time();out=Path(root)/(task["dataset"]+"_"+task["cohort"])
    if (out/"COMPLETE.json").exists():
        rec=json.loads((out/"COMPLETE.json").read_text())
        if rec["code_hash"]!=code_hash() or rec["config_sha256"]!=sha(config_path):
            raise ValueError("Existing M0 result was made with different code/config")
        for rel,digest in rec["files"].items():
            if sha(out/rel)!=digest:raise ValueError("M0 output drift: "+rel)
        return rec["summary"]
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"LAUNCH_PREFLIGHT.json",dict(target_host=socket.gethostname(),task=task,
               config_sha256=sha(config_path),code_hash=code_hash(),predictors="lncRNA_expression_only"))
    cfg=json.loads(Path(config_path).read_text());sources=[]
    x,y,folds,meta,features,info=load_case(cfg,task,sources)
    write_json(out/"INPUTS.json",dict(info=info,sources=sources))
    meta.to_csv(out/"TARGETS.tsv",sep="\t",index=False)
    features.to_csv(out/"FEATURES.tsv",sep="\t",index=False)
    pd.DataFrame(dict(patient_id=x.index,outer_fold=folds)).to_csv(out/"PATIENT_FOLDS.tsv",sep="\t",index=False)
    y.to_parquet(out/"OBSERVED_SCORES.parquet",compression="zstd")
    pred,base,audit,coef=grouped_oof(x.to_numpy(),y.to_numpy(),folds,output=out/"models",
                 feature_ids=x.columns.tolist(),patient_ids=x.index.tolist(),target_ids=y.columns.tolist())
    pd.DataFrame(pred,index=x.index,columns=y.columns).to_parquet(out/"OOF_PREDICTIONS.parquet",compression="zstd")
    pd.DataFrame(base,index=x.index,columns=y.columns).to_parquet(out/"OOF_TRAIN_MEAN_BASELINE.parquet",compression="zstd")
    audit.to_parquet(out/"FOLD_AUDIT.parquet",index=False,compression="zstd")
    coef.to_parquet(out/"TOP_LNCRNA_COEFFICIENTS.parquet",index=False,compression="zstd")
    rows=[]
    for j,t in enumerate(y.columns):
        observed=y.iloc[:,j].to_numpy()
        stat=metrics(observed,pred[:,j],base[:,j])
        nobs=int(np.isfinite(observed).sum())
        stat.update(target_id=t,n_observed=nobs,n_patients=len(x),observed_fraction=nobs/len(x),
                    oof_fraction_of_observed=stat["n_oof"]/nobs if nobs else np.nan,
                    status="COMPLETE_OOF" if nobs and stat["n_oof"]==nobs else "PARTIAL_OOF" if stat["n_oof"] else "UNAVAILABLE")
        rows.append(stat)
    metric=pd.DataFrame(rows).merge(meta,on="target_id",validate="one_to_one")
    for k,v in task.items():metric[k]=v
    metric.to_csv(out/"METRICS.tsv",sep="\t",index=False)
    summary=dict(**info,targets_evaluated=int(metric.r2.notna().sum()),
                 n_heldout_predictions=int(np.isfinite(pred).sum()),seconds=round(time.time()-t0,2))
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob("*") if p.is_file() and p.name!="COMPLETE.json"}
    write_json(out/"COMPLETE.json",dict(status="COMPLETE",target_host="149",code_hash=code_hash(),
               config_sha256=sha(config_path),files=files,summary=summary))
    print(json.dumps(dict(completed=task,seconds=summary["seconds"],predictions=summary["n_heldout_predictions"])),flush=True)
    return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",required=True);ap.add_argument("--output",required=True)
    ap.add_argument("--datasets",nargs="+",choices=["TCGA","CPTAC"],default=["TCGA","CPTAC"])
    ap.add_argument("--cancer-type",choices=["all","list"],default="all")
    ap.add_argument("--cancer-list",nargs="*",default=[])
    ap.add_argument("--workers",type=int,default=2)
    args=ap.parse_args()
    if socket.gethostname()!="149":raise RuntimeError("CancerLncAtlas workloads must execute on host 149")
    if args.workers<1 or args.workers>4:raise ValueError("Use 1-4 workers")
    cfg=json.loads(Path(args.config).read_text());root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    selected=tasks(cfg,args.datasets,args.cancer_type,args.cancer_list)
    launch=dict(target_host="149",model="M0_RIDGE_LNCRNA_ONLY",config_sha256=sha(args.config),
                code_hash=code_hash(),tasks=selected,lambda_grid=LAMBDAS.tolist(),
                outer_folds=5,inner_validation="next_fold",refit="train_plus_validation",
                min_observed_patients=dict(train=10,validation=3,test=3),cohort_pooling=False,
                no_PCA=True,no_outcome_imputation=True,paid_compute=False,workers=args.workers,
                interpretation="Internal prediction of existing frozen cohort scores; not a prospective clinical validation")
    if (root/"LAUNCH_PREFLIGHT.json").exists():
        previous=json.loads((root/"LAUNCH_PREFLIGHT.json").read_text())
        if previous!=launch:raise ValueError("Do not reuse output directory with another experiment")
    write_json(root/"LAUNCH_PREFLIGHT.json",launch)
    summary=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_case,args.config,t,root) for t in selected]
        for f in concurrent.futures.as_completed(futures):summary.append(f.result())
    pd.DataFrame(summary).sort_values(["dataset","cancer_id"]).to_csv(root/"RUN_SUMMARY.tsv",sep="\t",index=False)
    all_metrics=pd.concat([pd.read_csv(root/(t["dataset"]+"_"+t["cohort"])/"METRICS.tsv",sep="\t") for t in selected],ignore_index=True)
    all_metrics.to_parquet(root/"ALL_METRICS.parquet",index=False,compression="zstd")
    write_json(root/"COMPLETE.json",dict(status="COMPLETE",target_host="149",code_hash=code_hash(),
               studies=len(summary),patients=sum(r["n_patients"] for r in summary),
               evaluated_targets=int(all_metrics.r2.notna().sum()),
               heldout_predictions=sum(r["n_heldout_predictions"] for r in summary),
               config_sha256=sha(args.config),run_summary_sha256=sha(root/"RUN_SUMMARY.tsv"),
               metrics_sha256=sha(root/"ALL_METRICS.parquet")))
    print("ALL_M0_CASES_COMPLETE",flush=True)

if __name__=="__main__":main()
