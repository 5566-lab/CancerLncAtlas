"""Held-out diagnostic for RPPA500 versus retrospective TCGA-CPTAC protein scores.

Equal nonnegative weights over the same named protein panel. All learned
standardization/calibration uses the frozen training patients. Outputs remain
exploratory common-panel abundance scores, not unified activity/training Y.
"""
from __future__ import annotations
import argparse,json,socket,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr

def standardize_train(values,train):
    values=np.asarray(values,float);reference=values[train]
    count=np.isfinite(reference).sum(axis=0)
    mean=np.nanmean(reference,axis=0);scale=np.nanstd(reference,axis=0,ddof=1)
    valid=(count>=2)&np.isfinite(scale)&(scale>0)
    result=np.full(values.shape,np.nan)
    result[:,valid]=(values[:,valid]-mean[valid])/scale[valid]
    return result,mean,scale,valid

def observed_panel_mean(values):
    count=np.isfinite(values).sum(axis=1)
    mean=np.divide(np.nansum(values,axis=1),count,out=np.full(len(values),np.nan),where=count>0)
    return mean,count

def calibration_fit(x,y,train):
    keep=np.asarray(train,bool)&np.isfinite(x)&np.isfinite(y)
    if keep.sum()<3 or np.var(x[keep])<=1e-15:return np.nan,np.nan,int(keep.sum())
    slope=np.sum((x[keep]-x[keep].mean())*(y[keep]-y[keep].mean()))/np.sum((x[keep]-x[keep].mean())**2)
    return float(y[keep].mean()-slope*x[keep].mean()),float(slope),int(keep.sum())

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(rppa,bundle,mapping,retrospective,output,specimens,heldout_fold=0):
    assert socket.gethostname()=="149"
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError("Use a new output directory")
    rppa=Path(rppa);bundle=Path(bundle);mapping=Path(mapping)
    sources={}
    def bind(p,expected=None):
        p=Path(p);h=digest(p)
        if expected is not None and h!=expected:raise ValueError("Changed source: "+str(p))
        sources[str(p)]=h;return p
    def product(root,rel,complete="COMPLETE.json"):
        man=json.loads(bind(root/complete).read_text())
        return bind(root/rel,man["files"][rel]["sha256"])
    (out/"LAUNCH_PREFLIGHT.json").write_text(json.dumps(dict(target_host="149",action="BRCA_PAIRED_COMMON_PROTEIN_HELDOUT_DIAGNOSTIC",
        reference="Frozen patient_fold_id; selected fold held out; independent scalers per assay fitted on train only",
        weights="Equal nonnegative; no fitted gene weights",score_type="COMMON_PANEL_ABUNDANCE_NOT_ABSOLUTE_ACTIVATION",
        missing="NA kept; mean divides by observed fixed-panel members; counts exported",
        acceptance="No calibrated unified Y is promoted; all pathway rows including failures retained",
        pooling="No application to current prospective CPTAC cohort and no model input changes"),indent=2))
    pairs=pd.read_csv(product(mapping,"RETROSPECTIVE_PATIENT_PAIR_CANDIDATES.tsv"),sep="\t")
    pairs=pairs.loc[pairs.dataset.eq("BRCA_BI_protein") & pairs.in_current_rppa_rna_scope].sort_values("patient_id")
    assert pairs.patient_id.is_unique
    specimens=Path(specimens)
    specimen_receipt=json.loads(bind(specimens/"COMPLETE.json").read_text())
    assert specimen_receipt["status"]=="COMPLETE_PDC_COHORT_METADATA_VIAL_CONCORDANCE"
    proof=pd.read_csv(product(specimens,"PATIENT_SPECIMEN_CONCORDANCE.tsv"),sep="\t")
    eligible=proof.loc[proof.dataset.eq("BRCA_BI_protein")].copy()
    assert eligible.patient_id.is_unique
    pairs=pairs.merge(eligible[["patient_id","pdc_vial_concordant","status"]],on="patient_id",how="left",validate="one_to_one")
    if pairs.status.isna().any():raise ValueError("MISSING_SPECIMEN_AUDIT")
    assert pairs.pdc_vial_concordant.eq(pairs.status.eq("SAME_VIAL_IN_PDC_COHORT_METADATA")).all()
    pairs.to_csv(out/"PAIR_ELIGIBILITY.tsv",sep="\t",index=False)
    pairs=pairs.loc[pairs.pdc_vial_concordant].copy()
    ids=pairs.patient_id.tolist()
    authority=pd.read_parquet(product(rppa,"PATIENT_AUTHORITY.parquet")).set_index("patient_id").loc[ids]
    folds=pd.to_numeric(authority.patient_fold_id).to_numpy(int);test=folds==heldout_fold;train=~test
    assert test.any() and train.any()
    fitmap=pd.read_csv(product(mapping,"RETROSPECTIVE_COMMON_PROTEIN_CANDIDATES.tsv"),sep="\t")
    fitmap=fitmap.loc[fitmap.dataset.eq("BRCA_BI") & fitmap.primary_candidate].sort_values("gene_id")
    assert fitmap.gene_id.is_unique and fitmap.rppa_feature_id.is_unique and fitmap.ms_gene_symbol.is_unique
    rx=pd.read_parquet(product(rppa,"PROTEIN.parquet")).set_index("patient_id").loc[ids,fitmap.rppa_feature_id].to_numpy(float)
    manifest=json.loads(bind(retrospective).read_text());rec=next(r for r in manifest["files"] if r["cancer"]=="BRCA" and r["layer"]=="protein")
    ms=pd.read_csv(bind(rec["path"],rec["sha256"]),sep="\t",index_col=0)
    cx=ms.loc[fitmap.ms_gene_symbol,pairs.native_patient_id].to_numpy(float).T
    rz,rm,rs,rv=standardize_train(rx,train);cz,cm,cs,cv=standardize_train(cx,train)
    usable=rv&cv;rz[:,~usable]=np.nan;cz[:,~usable]=np.nan
    scalers=fitmap.copy()
    for name,val in [("rppa_train_mean",rm),("rppa_train_sd",rs),("ms_train_mean",cm),("ms_train_sd",cs),("train_usable",usable)]:scalers[name]=val
    scalers.to_csv(out/"TRAIN_REFERENCE.tsv",sep="\t",index=False)
    genes=fitmap.gene_id.to_numpy()
    bridge=pd.read_parquet(product(bundle,"PATHWAY_GENE_PROTEIN_BRIDGE.parquet"))
    rows=[];rvalues={};cvalues={};predictions={};rn={};cn={}
    for pathway,g in bridge.groupby("pathway_id"):
        members=set(g.gene_id.dropna());take=np.isin(genes,list(members))&usable
        rscore,rcount=observed_panel_mean(rz[:,take]);cscore,ccount=observed_panel_mean(cz[:,take])
        a,b,ntrain=calibration_fit(rscore,cscore,train);prediction=a+b*rscore
        keep=test&np.isfinite(prediction)&np.isfinite(cscore)
        ntest=int(keep.sum())
        row=dict(pathway_id=pathway,full_pathway_genes=len(members),panel_genes=int(take.sum()),
            train_pairs=ntrain,heldout_pairs=ntest,intercept=a,slope=b,
            status="NO_FIT" if not np.isfinite(b) else "INSUFFICIENT_HELDOUT" if ntest<3 else "HELDOUT_DIAGNOSTIC",
            heldout_pearson=np.nan,heldout_spearman=np.nan,heldout_r2=np.nan,heldout_ccc=np.nan,
            heldout_mse=np.nan,heldout_uncalibrated_mse=np.nan,
            heldout_rppa_observed_mean=float(rcount[test].mean()),heldout_ms_observed_mean=float(ccount[test].mean()))
        if ntest>=3 and np.isfinite(b):
            y=cscore[keep];pr=prediction[keep];x=rscore[keep]
            row["heldout_mse"]=float(np.mean((y-pr)**2))
            row["heldout_uncalibrated_mse"]=float(np.mean((y-x)**2))
            if np.var(y)>0:
                row["heldout_r2"]=float(1-np.sum((y-pr)**2)/np.sum((y-y.mean())**2))
                if np.var(pr)>0:
                    row["heldout_pearson"]=float(pearsonr(pr,y).statistic)
                    row["heldout_spearman"]=float(spearmanr(pr,y).statistic)
                    row["heldout_ccc"]=float(2*np.mean((pr-pr.mean())*(y-y.mean()))/(np.var(pr)+np.var(y)+(pr.mean()-y.mean())**2))
        rows.append(row);rvalues[pathway]=rscore;cvalues[pathway]=cscore;predictions[pathway]=prediction;rn[pathway]=rcount;cn[pathway]=ccount
    metrics=pd.DataFrame(rows);metrics.to_csv(out/"HELDOUT_PATHWAY_METRICS.tsv",sep="\t",index=False)
    for name,data in [("RPPA_COMMON_PANEL_SCORE",rvalues),("MS_COMMON_PANEL_SCORE",cvalues),("RPPA_TO_MS_PREDICTED_SCORE",predictions),
                      ("RPPA_OBSERVED_COUNTS",rn),("MS_OBSERVED_COUNTS",cn)]:
        d=pd.DataFrame(data,index=ids);d.index.name="patient_id";d.reset_index().to_parquet(out/(name+".parquet"),index=False)
    pd.DataFrame(dict(patient_id=ids,patient_fold_id=folds,heldout=test)).to_csv(out/"PATIENT_SPLIT.tsv",sep="\t",index=False)
    summaries=[]
    for minimum in [1,3,5,10]:
        d=metrics.loc[metrics.panel_genes.ge(minimum)&np.isfinite(metrics.heldout_r2)]
        summaries.append(dict(min_panel_genes_descriptive_only=minimum,pathways=len(d),
             positive_heldout_r2=int(d.heldout_r2.gt(0).sum()),median_heldout_r2=float(d.heldout_r2.median()) if len(d) else None,
             median_heldout_pearson=float(d.heldout_pearson.median()) if len(d) else None,
             calibration_lower_mse_than_raw_panel_mean=int(d.heldout_mse.lt(d.heldout_uncalibrated_mse).sum())))
    result=dict(target_host="149",status="COMPLETE_EXPLORATORY_HELDOUT_DIAGNOSTIC",
        patients=len(ids),heldout_fold=int(heldout_fold),train_patients=int(train.sum()),heldout_patients=int(test.sum()),primary_panel_genes=int(usable.sum()),
        pathway_axis=len(metrics),summaries=summaries,calibration_validated_for_production=False,
        pdc_cohort_metadata_same_vial=True,processed_column_to_aliquot_lineage_verified=False,same_aliquot_verified=False,
        prospective_cptac_transfer_validated=False,source_sha256=sources,
        limitations=["Single held-out fold; no repeated or external validation or inferential pathway selection.",
          "Scores depend on observed members; patient-specific coverage exported. Missing is never zero.",
          "Published assays include upstream cohort normalization; this is not fully inductive raw-assay validation.",
          "Pathway scores measure shared-panel protein abundance, not absolute pathway activation.",
          "No single universal TCGA/CPTAC Y or formal training input is created."])
    result["files"]={p.name:{"sha256":digest(p),"bytes":p.stat().st_size} for p in out.iterdir() if p.is_file()}
    (out/"COMPLETE.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ["status","patients","train_patients","heldout_patients","primary_panel_genes","summaries"]}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["rppa","bundle","mapping","retrospective","output","specimens"]:p.add_argument("--"+key,required=True)
    p.add_argument("--heldout-fold",type=int,default=0)
    a=p.parse_args();run(a.rppa,a.bundle,a.mapping,a.retrospective,a.output,a.specimens,a.heldout_fold)
