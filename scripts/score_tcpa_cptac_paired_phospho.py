"""Fold-fitted total-protein adjustment and cross-platform site diagnostics.

These are measured-site residuals, not phosphorylation occupancy or causal
activity. No common phospho pathway is fabricated when PTM signature coverage
fails. Both PROTEIN and PHOSPHORYLATION are label dependencies.
"""
import argparse,json,hashlib,socket
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import pearsonr,spearmanr
from scripts.score_tcpa_cptac_paired_protein_pilot import standardize_train,calibration_fit

def residualize(phospho,total,train):
    phospho,total=np.asarray(phospho,float),np.asarray(total,float)
    use=np.asarray(train,bool)&np.isfinite(phospho)&np.isfinite(total)
    n=int(use.sum())
    output=np.full(len(phospho),np.nan)
    if n<3:return output,dict(status="TOO_FEW_TRAIN_PAIRS",n=n,intercept=np.nan,slope=np.nan)
    t=total[use];p=phospho[use]
    if np.var(t)<=1e-15:return output,dict(status="NO_TOTAL_PROTEIN_VARIANCE",n=n,intercept=np.nan,slope=np.nan)
    b=float(np.sum((t-t.mean())*(p-p.mean()))/np.sum((t-t.mean())**2));a=float(p.mean()-b*t.mean())
    valid=np.isfinite(phospho)&np.isfinite(total);output[valid]=phospho[valid]-(a+b*total[valid])
    return output,dict(status="FITTED",n=n,intercept=a,slope=b)

def run(root,rppa,epitopes,specimens,output):
    assert socket.gethostname()=="149"
    root,rppa,epitopes,specimens,out=map(Path,[root,rppa,epitopes,specimens,output])
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError("Use new output directory")
    (out/"LAUNCH_PREFLIGHT.json").write_text(json.dumps(dict(target_host="149",action="BRCA_FIVE_FOLD_PAIRED_SITE_RESIDUAL_DIAGNOSTIC",
      folds=[0,1,2,3,4],residual="P - (intercept + slope*T), separately fit within each platform training fold",
      score_comparison="Raw and adjusted have different outcomes; their R2 differences are not predictive improvement on one Y.",
      missing="No imputation; missing total gives missing residual; raw phospho retained separately",
      label_dependencies=["PROTEIN","PHOSPHORYLATION"],training_input_exclusion_for_future_use=["PROTEIN","PHOSPHORYLATION"],
      production_Y=False),indent=2))
    sources={}
    def bind(p,h=None):
        s=hashlib.sha256(p.read_bytes()).hexdigest()
        if h and h!=s:raise ValueError("Source checksum drift")
        sources[str(p)]=s;return p
    def product(r,f):
        d=json.loads(bind(r/"COMPLETE.json").read_text())
        return bind(r/f,d["files"][f]["sha256"])
    sites=pd.read_csv(product(epitopes,"CURATED_PAIRED_SITES.tsv"),sep="\t")
    proof=pd.read_csv(product(specimens,"PATIENT_SPECIMEN_CONCORDANCE.tsv"),sep="\t")
    ph=set(proof.loc[proof.dataset.eq("BRCA_BI_phospho")&proof.pdc_vial_concordant,"patient_id"])
    pr=set(proof.loc[proof.dataset.eq("BRCA_BI_protein")&proof.pdc_vial_concordant,"patient_id"])
    ids=sorted(ph&pr)
    authority=pd.read_parquet(product(rppa,"PATIENT_AUTHORITY.parquet")).set_index("patient_id").loc[ids]
    folds=pd.to_numeric(authority.patient_fold_id).to_numpy(int)
    rp=pd.read_parquet(product(rppa,"PHOSPHORYLATION.parquet")).set_index("patient_id").loc[ids]
    rt=pd.read_parquet(product(rppa,"PROTEIN.parquet")).set_index("patient_id").loc[ids]
    manifest=json.loads(bind(root/"DOWNLOAD_COMPLETE.json").read_text())
    def ms(layer):
        e=next(r for r in manifest["files"] if r["cancer"]=="BRCA" and r["layer"]==layer)
        d=pd.read_csv(bind(Path(e["path"]),e["sha256"]),sep="\t",index_col=0)
        return d[[s.replace("-",".") for s in ids]]
    mp,mt=ms("phospho"),ms("protein")
    metrics=[];parameters=[];predictions=[]
    for site in sites.itertuples():
        pv=rp[site.rppa_phospho].to_numpy(float);tv=rt[site.rppa_total].to_numpy(float)
        mv=mp.loc[site.retrospective_feature_id].to_numpy(float);tt=mt.loc[site.gene_symbol].to_numpy(float)
        complete=np.isfinite(pv)&np.isfinite(tv)&np.isfinite(mv)&np.isfinite(tt)
        for fold in range(5):
            train=folds!=fold;test=~train
            ar,rr=residualize(pv,tv,train);am,mr=residualize(mv,tt,train)
            for mode,rvalue,mvalue in [("RAW",pv,mv),("TOTAL_PROTEIN_ADJUSTED",ar,am)]:
                rz,rmean,rsd,rvalid=standardize_train(rvalue[:,None],train)
                mz,mmean,msd,mvalid=standardize_train(mvalue[:,None],train)
                x=rz[:,0];y=mz[:,0]
                a,b,ntrain=calibration_fit(x,y,train&complete);pred=a+b*x
                keep=test&complete&np.isfinite(y)&np.isfinite(pred);ntest=int(keep.sum())
                row=dict(feature_id=site.rppa_phospho,gene_symbol=site.gene_symbol,human_site=site.residue+str(site.human_position),
                  mode=mode,fold=fold,train_pairs=ntrain,heldout_pairs=ntest,heldout_r2=np.nan,heldout_pearson=np.nan,
                  heldout_spearman=np.nan,heldout_mse=np.nan,status="INSUFFICIENT_SUPPORT")
                if ntest>=3 and np.var(y[keep])>0:
                    row["heldout_r2"]=float(1-np.sum((y[keep]-pred[keep])**2)/np.sum((y[keep]-y[keep].mean())**2))
                    row["heldout_mse"]=float(np.mean((y[keep]-pred[keep])**2));row["status"]="HELDOUT_DIAGNOSTIC"
                    if np.var(pred[keep])>0:
                        row["heldout_pearson"]=float(pearsonr(pred[keep],y[keep]).statistic)
                        row["heldout_spearman"]=float(spearmanr(pred[keep],y[keep]).statistic)
                metrics.append(row)
                parameters.append(dict(feature_id=site.rppa_phospho,fold=fold,mode=mode,rppa_residual_intercept=rr["intercept"],
                  rppa_residual_slope=rr["slope"],rppa_residual_train_n=rr["n"],rppa_residual_status=rr["status"],
                  ms_residual_intercept=mr["intercept"],ms_residual_slope=mr["slope"],ms_residual_train_n=mr["n"],ms_residual_status=mr["status"],
                  rppa_reference_mean=float(rmean[0]),rppa_reference_sd=float(rsd[0]),ms_reference_mean=float(mmean[0]),ms_reference_sd=float(msd[0]),
                  calibration_intercept=a,calibration_slope=b,calibration_train_pairs=ntrain))
                for i in np.flatnonzero(test):
                    predictions.append(dict(patient_id=ids[i],feature_id=site.rppa_phospho,mode=mode,fold=fold,
                      rppa_raw_phospho=pv[i],rppa_total_protein=tv[i],ms_raw_phospho=mv[i],ms_total_protein=tt[i],
                      rppa_adjusted_phospho=ar[i],ms_adjusted_phospho=am[i],rppa_fold_score=x[i],ms_fold_score=y[i],
                      calibrated_prediction=pred[i],complete_four_measurements=bool(complete[i]),included_in_metric=bool(keep[i])))
    frame=pd.DataFrame(metrics);frame.to_csv(out/"SITE_FOLD_METRICS.tsv",sep="\t",index=False)
    pd.DataFrame(parameters).to_csv(out/"TRAIN_FOLD_PARAMETERS.tsv",sep="\t",index=False)
    pd.DataFrame(predictions).to_parquet(out/"HELDOUT_SITE_VALUES.parquet",index=False)
    summary=frame.groupby(["feature_id","gene_symbol","human_site","mode"]).agg(completed_folds=("heldout_r2","count"),
      mean_fold_r2=("heldout_r2","mean"),positive_folds=("heldout_r2",lambda x:int(x.gt(0).sum())),
      min_train_pairs=("train_pairs","min"),min_test_pairs=("heldout_pairs","min")).reset_index()
    summary.to_csv(out/"SITE_CV_SUMMARY.tsv",sep="\t",index=False)
    d=dict(target_host="149",status="COMPLETE_FIVE_FOLD_SITE_RESIDUAL_DIAGNOSTIC",patients=len(ids),sites=len(sites),
       label_dependencies=["PROTEIN","PHOSPHORYLATION"],common_phospho_pathway_scores=False,
       reason="Curated common-site panel supports at most3 members per PTMsigDB signature, below existing minimum5.",
       production_calibration=False,summary=summary.to_dict("records"),source_sha256=sources,
       limitations=["Five sites are a bounded first reviewed panel, not all RPPA500 phospho markers.",
         "Residuals are relative phospho signals after linear total-protein adjustment, not phosphorylation occupancy.",
         "Different fold references are never pooled into one cross-fold Y.",
         "PDC metadata establish vial concordance; direct processed-column-to-aliquot lineage remains unverified."])
    d["files"]={p.name:{"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir() if p.is_file()}
    (out/"COMPLETE.json").write_text(json.dumps(d,indent=2));print(json.dumps(d["summary"]),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["root","rppa","epitopes","specimens","output"]:p.add_argument("--"+key,required=True)
    a=p.parse_args();run(a.root,a.rppa,a.epitopes,a.specimens,a.output)
