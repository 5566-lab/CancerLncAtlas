"""Three-cancer demo: training-only correlation screen -> Lasso -> fixed signatures."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,multiprocessing,os,socket,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from cc_hhgt.m0_screened_lasso import screen,path_predictions,fit_fixed,ALPHA_RATIOS,SOLVER
from cc_hhgt.m0_linear import metrics
from cla_engine.common import sha
from run_m0_linear import write_json
G={}

def worker(j):
    x,y=G["x"],G["y"][:,j];phase=G["phase"]
    if phase=="final":
        fit=np.flatnonzero(np.isfinite(y))
        if len(fit)<10 or not np.isfinite(G["ratios"][j]):
            return dict(j=j,status="INSUFFICIENT_CV_OR_OBSERVED_PATIENTS")
        _,st=fit_fixed(x[fit],y[fit],G["sf"]["passed"][:,j],x[:0],G["ratios"][j])
        return dict(j=j,status="FITTED",state=st,ratio=G["ratios"][j],n_fit=len(fit))
    tr,va,te,ft=G["tr"],G["va"],G["te"],G["ft"]
    tr=tr[np.isfinite(y[tr])];va=va[np.isfinite(y[va])];te=te[np.isfinite(y[te])]
    ft=ft[np.isfinite(y[ft])]
    base=dict(j=j,n_train=len(tr),n_validation=len(va),n_test=len(te))
    if len(tr)<10 or len(va)<3 or len(te)<3:
        return dict(**base,status="INSUFFICIENT_OBSERVED_PATIENTS")
    pv,pa=path_predictions(x[tr],y[tr],G["si"]["passed"][:,j],x[va],ratios=ALPHA_RATIOS)
    mse=np.mean((pv-y[va,None])**2,axis=0)
    best=int(np.flatnonzero(mse<=np.min(mse)+1e-12*max(1.,np.min(mse)))[0])
    pred,st=fit_fixed(x[ft],y[ft],G["sf"]["passed"][:,j],x[te],ALPHA_RATIOS[best])
    return dict(**base,status="FITTED",state=st,test_rows=te,prediction=pred,
                baseline=np.full(len(te),y[ft].mean()),ratio=float(ALPHA_RATIOS[best]),
                validation_mse=float(mse[best]),n_inner_screened=pa["n_screened"],
                path_max_dual_gap=pa["max_dual_gap"],path_max_iterations=pa["max_iterations"])


def code_hash():
    files=[Path(__file__),ROOT/"cc_hhgt/m0_screened_lasso.py",ROOT/"cc_hhgt/m0_linear.py"]
    return hashlib.sha256("".join(sha(p) for p in files).encode()).hexdigest()


def coefficient_rows(st,j,fold,features,targets,sf):
    nz=np.flatnonzero(st["coefficient"]!=0)
    idx=st["feature_index"][nz]
    return pd.DataFrame(dict(target_id=np.repeat(targets[j],len(idx)),outer_fold=np.repeat(fold,len(idx)),
        lncrna_id=np.asarray(features)[idx],coefficient=st["coefficient"][nz],
        standardized_coefficient=st["standardized_coefficient"][nz],imputation_median=st["median"][nz],
        screening_rho=sf["rho"][idx,j],screening_fdr=sf["q"][idx,j],screening_n=sf["n"][idx,j]))


def run_case(input_root,case,output_root,min_abs_rho,max_fdr,workers):
    global G
    if socket.gethostname()!="149":raise RuntimeError("Run on 149")
    t0=time.time();src=Path(input_root)/case;out=Path(output_root)/case
    source=json.loads((src/"COMPLETE.json").read_text())
    if source["status"]!="COMPLETE" or source["target_host"]!="149":raise ValueError("Unfinished inputs")
    for rel,digest in source["files"].items():
        if sha(src/rel)!=digest:raise ValueError("Input drift: "+rel)
    experiment=dict(target_host="149",model="M0_CORRELATION_SCREENED_LASSO",case=case,
       input_complete_sha256=sha(src/"COMPLETE.json"),code_hash=code_hash(),min_abs_rho=min_abs_rho,
       max_fdr=max_fdr,min_pairs=10,alpha_ratios=ALPHA_RATIOS.tolist(),outer_folds=5,
       final_alpha_rule="median_of_inner_selected_alpha_ratios_across_outer_folds",
       solver=SOLVER,cohort_pooling=False,full_common_gene_axis=True,score_semantics="FROZEN_NATIVE_LABELS")
    if (out/"COMPLETE.json").exists():
        rec=json.loads((out/"COMPLETE.json").read_text())
        if rec["experiment"]!=experiment:raise ValueError("Existing experiment differs")
        for rel,digest in rec["files"].items():
            if sha(out/rel)!=digest:raise ValueError("Output drift: "+rel)
        return rec["summary"]
    if out.exists() and any(out.iterdir()):raise ValueError("Partial run: use a new case output")
    out.mkdir(parents=True)
    write_json(out/"LAUNCH_PREFLIGHT.json",experiment)
    xf=pd.read_parquet(src/"X.parquet");yf=pd.read_parquet(src/"Y.parquet")
    f=pd.read_csv(src/"PATIENT_FOLDS.tsv",sep="\t").set_index("patient_id").outer_fold.reindex(xf.index)
    if not xf.index.equals(yf.index) or f.isna().any() or xf.index.has_duplicates:raise ValueError("Invalid patient axes")
    x=xf.to_numpy(float);y=yf.to_numpy(float);folds=f.to_numpy(int)
    feature_ids=xf.columns.tolist();target_ids=yf.columns.tolist()
    pd.read_csv(src/"FEATURES.tsv",sep="\t").to_csv(out/"FEATURES.tsv",sep="\t",index=False)
    pd.read_csv(src/"TARGETS.tsv",sep="\t").to_csv(out/"TARGETS.tsv",sep="\t",index=False)
    pd.read_csv(src/"PATIENT_FOLDS.tsv",sep="\t").to_csv(out/"PATIENT_FOLDS.tsv",sep="\t",index=False)
    write_json(out/"INPUTS.json",json.loads((src/"INPUTS.json").read_text()))
    valid=np.flatnonzero(np.isfinite(y).sum(axis=0)>=16)
    pred=np.full(y.shape,np.nan);baseline=np.full(y.shape,np.nan)
    chosen=np.full((5,y.shape[1]),np.nan);audit=[];models=[];coefs=[]
    for fold in range(5):
        tr=np.flatnonzero((folds!=fold)&(folds!=(fold+1)%5));va=np.flatnonzero(folds==(fold+1)%5)
        ft=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold)
        print(case,"SCREEN",fold,"targets",len(valid),flush=True)
        # Untestable targets remain in all output axes, but do not consume matrix products.
        si=screen(x[tr],y[np.ix_(tr,valid)],min_abs_rho=min_abs_rho,max_fdr=max_fdr)
        sf=screen(x[ft],y[np.ix_(ft,valid)],min_abs_rho=min_abs_rho,max_fdr=max_fdr)
        si.pop("p");sf.pop("p")
        G=dict(x=x,y=y[:,valid],phase="oof",tr=tr,va=va,te=te,ft=ft,si=si,sf=sf)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("fork")) as pool:
            for count,res in enumerate(pool.map(worker,range(len(valid)),chunksize=4),1):
                jj=res["j"];j=valid[jj]
                row={k:v for k,v in res.items() if k not in ("j","state","test_rows","prediction","baseline")}
                row.update(target_id=target_ids[j],outer_fold=fold)
                if res["status"]=="FITTED":
                    st=res["state"];ii=res["test_rows"]
                    pred[ii,j]=res["prediction"];baseline[ii,j]=res["baseline"];chosen[fold,j]=res["ratio"]
                    row.update(n_refit_screened=int(sf["passed"][:,jj].sum()),n_selected=int(np.count_nonzero(st["coefficient"])),
                               fit_dual_gap=st["dual_gap"],fit_iterations=st["n_iter"],alpha=st["alpha"],intercept=st["intercept"])
                    coefs.append(coefficient_rows(st,jj,fold,feature_ids,[target_ids[v] for v in valid],sf))
                    models.append(dict(target_id=target_ids[j],outer_fold=fold,intercept=st["intercept"],
                              alpha=st["alpha"],alpha_ratio=res["ratio"],n_selected=row["n_selected"]))
                audit.append(row)
                if count%50==0 or count==len(valid):
                    print(case,"FOLD",fold,"targets",count,"/",len(valid),"seconds",round(time.time()-t0),flush=True)
        G={}
        pd.DataFrame(audit).to_parquet(out/"FOLD_AUDIT.parquet",index=False,compression="zstd")
        pd.DataFrame(pred,index=xf.index,columns=yf.columns).to_parquet(out/"OOF_PREDICTIONS.parquet",compression="zstd")
        del si,sf
    fold_coef=pd.concat(coefs,ignore_index=True) if coefs else pd.DataFrame(columns=["target_id","outer_fold","lncrna_id","coefficient","standardized_coefficient","imputation_median","screening_rho","screening_fdr","screening_n"])
    fold_coef.to_parquet(out/"FOLD_SELECTED_GENES.parquet",index=False,compression="zstd")
    pd.DataFrame(models).to_csv(out/"FOLD_MODELS.tsv",sep="\t",index=False)
    pd.DataFrame(baseline,index=xf.index,columns=yf.columns).to_parquet(out/"OOF_TRAIN_MEAN_BASELINE.parquet",compression="zstd")
    rows=[]
    for j,t in enumerate(target_ids):
        m=metrics(y[:,j],pred[:,j],baseline[:,j]);nobs=int(np.isfinite(y[:,j]).sum())
        m.update(target_id=t,n_observed=nobs,n_patients=len(y),observed_fraction=nobs/len(y),
                 oof_fraction_of_observed=m["n_oof"]/nobs if nobs else np.nan,
                 status="COMPLETE_OOF" if nobs and m["n_oof"]==nobs else "PARTIAL_OOF" if m["n_oof"] else "UNAVAILABLE")
        rows.append(m)
    mm=pd.DataFrame(rows).merge(pd.read_csv(src/"TARGETS.tsv",sep="\t"),on="target_id",validate="one_to_one")
    for k in ("dataset","cohort","cancer_id"):mm[k]=source["info"][k]
    mm.to_csv(out/"METRICS.tsv",sep="\t",index=False)
    # Final deployment fit uses all observed patients, after OOF evaluation is complete.
    ratios=np.full(y.shape[1],np.nan)
    for j in valid:
        vv=chosen[:,j][np.isfinite(chosen[:,j])]
        if len(vv)>=3:ratios[j]=float(np.median(vv))
    print(case,"FINAL_SCREEN",flush=True)
    sf=screen(x,y[:,valid],min_abs_rho=min_abs_rho,max_fdr=max_fdr);sf.pop("p")
    G=dict(x=x,y=y[:,valid],phase="final",sf=sf,ratios=ratios[valid])
    finals=[];final_models=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("fork")) as pool:
        for count,res in enumerate(pool.map(worker,range(len(valid)),chunksize=4),1):
            jj=res["j"];j=valid[jj]
            row=dict(target_id=target_ids[j],status=res["status"],n_screened=int(sf["passed"][:,jj].sum()))
            if res["status"]=="FITTED":
                st=res["state"];nz=int(np.count_nonzero(st["coefficient"]))
                row.update(intercept=st["intercept"],alpha=st["alpha"],alpha_ratio=res["ratio"],
                      n_fit=res["n_fit"],n_selected=nz,dual_gap=st["dual_gap"],iterations=st["n_iter"],
                      signature_status="NONEMPTY" if nz else "NO_CORRELATION_PASS" if row["n_screened"]==0 else "LASSO_INTERCEPT_ONLY")
                finals.append(coefficient_rows(st,jj,-1,feature_ids,[target_ids[v] for v in valid],sf))
            final_models.append(row)
            if count%50==0 or count==len(valid):print(case,"FINAL",count,"/",len(valid),flush=True)
    G={}
    fin=pd.concat(finals,ignore_index=True) if finals else fold_coef.iloc[:0].copy()
    if len(fold_coef):
        stab=fold_coef.assign(positive=fold_coef.coefficient.gt(0)).groupby(["target_id","lncrna_id"]).agg(
            selected_folds=("outer_fold","nunique"),positive_folds=("positive","sum")).reset_index()
        nfold=pd.DataFrame(audit).query("status=='FITTED'").groupby("target_id").outer_fold.nunique()
        stab["eligible_folds"]=stab.target_id.map(nfold)
        stab["selection_frequency"]=stab.selected_folds/stab.eligible_folds
        stab["direction_agreement"]=np.maximum(stab.positive_folds,stab.selected_folds-stab.positive_folds)/stab.selected_folds
        stab.to_csv(out/"GENE_SELECTION_STABILITY.tsv.gz",sep="\t",index=False)
        fin=fin.merge(stab,on=["target_id","lncrna_id"],how="left",validate="one_to_one")
        for c in ("selected_folds","positive_folds","selection_frequency"):fin[c]=fin[c].fillna(0)
    symbols=pd.read_csv(src/"FEATURES.tsv",sep="\t")
    fin=fin.merge(symbols,on="lncrna_id",how="left",validate="many_to_one").drop(columns="outer_fold")
    fin.to_parquet(out/"FINAL_GENE_SETS.parquet",index=False,compression="zstd")
    fin.to_csv(out/"FINAL_GENE_SETS.tsv.gz",sep="\t",index=False)
    fm=pd.DataFrame({"target_id":target_ids}).merge(pd.DataFrame(final_models),on="target_id",how="left",validate="one_to_one")
    fm["status"]=fm["status"].fillna("INSUFFICIENT_OBSERVED_PATIENTS")
    fm.to_csv(out/"FINAL_MODELS.tsv",sep="\t",index=False)
    # Preserve all passing final correlations, including genes Lasso shrinks to zero.
    import pyarrow as pa, pyarrow.parquet as pq
    writer=None
    for start in range(0,len(valid),64):
        vv=valid[start:start+64];ix,jx=np.where(sf["passed"][:,start:start+len(vv)])
        frame=pd.DataFrame(dict(target_id=np.asarray(target_ids)[vv[jx]],lncrna_id=np.asarray(feature_ids)[ix],
             rho=sf["rho"][ix,start+jx],fdr=sf["q"][ix,start+jx],n_pairs=sf["n"][ix,start+jx]))
        table=pa.Table.from_pandas(frame,preserve_index=False)
        if writer is None:writer=pq.ParquetWriter(out/"FINAL_CORRELATION_PASS.parquet",table.schema,compression="zstd")
        writer.write_table(table)
    if writer:writer.close()
    summary=dict(**source["info"],targets_evaluated=int(mm.r2.notna().sum()),
             final_nonempty_signatures=int(fm.get("n_selected",pd.Series(dtype=float)).gt(0).sum()),
             final_selected_gene_rows=len(fin),heldout_predictions=int(np.isfinite(pred).sum()),
             min_abs_rho=min_abs_rho,max_fdr=max_fdr,seconds=round(time.time()-t0,2))
    for rel,digest in source["files"].items():
        if sha(src/rel)!=digest:raise ValueError("Input changed during fitting: "+rel)
    files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="COMPLETE.json"}
    write_json(out/"COMPLETE.json",dict(status="COMPLETE",target_host="149",experiment=experiment,files=files,summary=summary))
    print("CASE_COMPLETE",case,json.dumps(summary),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--inputs",required=True);p.add_argument("--output",required=True)
    p.add_argument("--cases",nargs="*");p.add_argument("--min-abs-rho",type=float,default=.2);p.add_argument("--max-fdr",type=float,default=.05)
    p.add_argument("--workers",type=int,default=4);p.add_argument("--case-workers",type=int,default=2)
    a=p.parse_args()
    if socket.gethostname()!="149":raise RuntimeError("CancerLncAtlas workloads require host 149")
    inp=json.loads((Path(a.inputs)/"COMPLETE.json").read_text())
    cases=[r["case"] for r in inp["cases"]]
    if a.cases:
        if set(a.cases)-set(cases):raise ValueError("Unknown case")
        cases=a.cases
    if len({c.split("_",1)[1] for c in cases})>3:raise ValueError("Demo cannot exceed 3 cancers")
    if not 1<=a.workers<=8 or not 1<=a.case_workers<=3:raise ValueError("Invalid CPU parallelism")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    launch=dict(target_host="149",operation="THREE_CANCER_SCREENED_LASSO_DEMO",cases=cases,
                min_abs_rho=a.min_abs_rho,max_fdr=a.max_fdr,code_hash=code_hash(),paid_compute=False,
                input_complete_sha256=sha(Path(a.inputs)/"COMPLETE.json"),cohort_pooling=False)
    if (out/"LAUNCH_PREFLIGHT.json").exists() and json.loads((out/"LAUNCH_PREFLIGHT.json").read_text())!=launch:
        raise ValueError("Different prior launch; use a new output directory")
    write_json(out/"LAUNCH_PREFLIGHT.json",launch)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.case_workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        fs=[pool.submit(run_case,a.inputs,c,a.output,a.min_abs_rho,a.max_fdr,a.workers) for c in cases]
        results=[f.result() for f in concurrent.futures.as_completed(fs)]
    pd.DataFrame(results).to_csv(out/"RUN_SUMMARY.tsv",sep="\t",index=False)
    allm=pd.concat([pd.read_csv(out/c/"METRICS.tsv",sep="\t") for c in cases],ignore_index=True)
    allm.to_parquet(out/"ALL_METRICS.parquet",index=False)
    write_json(out/"COMPLETE.json",dict(status="COMPLETE",target_host="149",launch=launch,
       cases=cases,run_summary_sha256=sha(out/"RUN_SUMMARY.tsv"),metrics_sha256=sha(out/"ALL_METRICS.parquet")))
    print("ALL_LASSO_DEMO_CASES_COMPLETE",flush=True)
if __name__=="__main__":main()
