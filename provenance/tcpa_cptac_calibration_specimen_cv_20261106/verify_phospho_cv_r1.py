import pathlib,socket,json,hashlib,sys,numpy as np,pandas as pd
assert socket.gethostname()=="149"
P=pathlib.Path("/dell_2/DSC/CancerLncAtlas/runtime/tcpa_cptac_calibration_20261106_r1");Q=P/"phospho_cv_r1";B=P.parents[1]
(P/"PHOSPHO_INDEPENDENT_REVIEW_PREFLIGHT.json").write_text(json.dumps({"target_host":"149","action":"RECONSTRUCT_ALL_FIVE_SITE_FOLD_RESIDUALS_AND_METRICS"}))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
complete=json.loads((Q/"COMPLETE.json").read_text())
for p,h in complete["source_sha256"].items():assert sha(p)==h,p
for f,r in complete["files"].items():assert sha(Q/f)==r["sha256"],f
proof=pd.read_csv(P/"specimen_audit_r1/PATIENT_SPECIMEN_CONCORDANCE.tsv",sep="\t")
ids=sorted(set(proof.loc[proof.dataset.eq("BRCA_BI_phospho")&proof.pdc_vial_concordant,"patient_id"])&
           set(proof.loc[proof.dataset.eq("BRCA_BI_protein")&proof.pdc_vial_concordant,"patient_id"]))
assert len(ids)==102
rp=pd.read_parquet(B/"inputs/rppa500_20261104_r4/PHOSPHORYLATION.parquet").set_index("patient_id").loc[ids]
rt=pd.read_parquet(B/"inputs/rppa500_20261104_r4/PROTEIN.parquet").set_index("patient_id").loc[ids]
mp=pd.read_csv(P/"retrospective_sources/BRCA_BI_phospho.tsv.gz",sep="\t",index_col=0)[[x.replace("-",".") for x in ids]]
mt=pd.read_csv(P/"retrospective_sources/BRCA_BI_protein.tsv.gz",sep="\t",index_col=0)[[x.replace("-",".") for x in ids]]
folds=pd.read_parquet(B/"inputs/rppa500_20261104_r4/PATIENT_AUTHORITY.parquet").set_index("patient_id").loc[ids,"patient_fold_id"].to_numpy(int)
parameters=pd.read_csv(Q/"TRAIN_FOLD_PARAMETERS.tsv",sep="\t").set_index(["feature_id","fold","mode"])
metrics=pd.read_csv(Q/"SITE_FOLD_METRICS.tsv",sep="\t").set_index(["feature_id","fold","mode"])
predictions=pd.read_parquet(Q/"HELDOUT_SITE_VALUES.parquet")
assert not predictions.duplicated(["patient_id","feature_id","mode"]).any()
assert len(predictions)==102*5*2
sites=pd.read_csv(P/"epitope_audit_r1/CURATED_PAIRED_SITES.tsv",sep="\t")
maxcoef=maxvalue=maxr2=0.;checked=0
def difference(actual,expected):
    assert np.array_equal(np.isnan(actual),np.isnan(expected))
    return float(np.nanmax(np.abs(actual-expected))) if np.isfinite(expected).any() else 0.
for site in sites.itertuples():
    pv=rp[site.rppa_phospho].to_numpy(float);tv=rt[site.rppa_total].to_numpy(float)
    mv=mp.loc[site.retrospective_feature_id].to_numpy(float);tt=mt.loc[site.gene_symbol].to_numpy(float)
    eligible=np.isfinite(pv)&np.isfinite(tv)&np.isfinite(mv)&np.isfinite(tt)
    for fold in range(5):
        train=folds!=fold;test=~train;residuals=[]
        for prefix,p,t in [("rppa",pv,tv),("ms",mv,tt)]:
            use=train&np.isfinite(p)&np.isfinite(t)
            coef=np.linalg.lstsq(np.column_stack([np.ones(use.sum()),t[use]]),p[use],rcond=None)[0]
            param=parameters.loc[(site.rppa_phospho,fold,"TOTAL_PROTEIN_ADJUSTED")]
            maxcoef=max(maxcoef,abs(coef[0]-param[prefix+"_residual_intercept"]),abs(coef[1]-param[prefix+"_residual_slope"]))
            assert use.sum()==param[prefix+"_residual_train_n"]
            residuals.append(p-coef[0]-coef[1]*t)
        for mode,rx,my in [("RAW",pv,mv),("TOTAL_PROTEIN_ADJUSTED",residuals[0],residuals[1])]:
            z=(rx-pd.Series(rx[train]).mean())/pd.Series(rx[train]).std()
            y=(my-pd.Series(my[train]).mean())/pd.Series(my[train]).std()
            use=train&eligible&np.isfinite(z)&np.isfinite(y)
            coef=np.linalg.lstsq(np.column_stack([np.ones(use.sum()),z[use]]),y[use],rcond=None)[0]
            pred=coef[0]+coef[1]*z
            actual=predictions.loc[predictions.feature_id.eq(site.rppa_phospho)&predictions.fold.eq(fold)&predictions["mode"].eq(mode)].set_index("patient_id").loc[np.array(ids)[test]]
            for field,expected in [("rppa_raw_phospho",pv),("rppa_total_protein",tv),("ms_raw_phospho",mv),("ms_total_protein",tt),
               ("rppa_adjusted_phospho",residuals[0]),("ms_adjusted_phospho",residuals[1]),
               ("rppa_fold_score",z),("ms_fold_score",y),("calibrated_prediction",pred)]:
                maxvalue=max(maxvalue,difference(actual[field].to_numpy(float),expected[test]))
            keep=test&eligible&np.isfinite(y)&np.isfinite(pred)
            assert actual.included_in_metric.to_numpy(bool).tolist()==keep[test].tolist()
            metric=metrics.loc[(site.rppa_phospho,fold,mode)]
            assert metric.train_pairs==use.sum() and metric.heldout_pairs==keep.sum()
            r2=1-np.sum((y[keep]-pred[keep])**2)/np.sum((y[keep]-y[keep].mean())**2)
            maxr2=max(maxr2,abs(r2-metric.heldout_r2));checked+=1
assert checked==50 and maxcoef<1e-10 and maxvalue<1e-10 and maxr2<1e-10
result={"target_host":"149","status":"PASS_ALL_PHOSPHO_SITE_FOLDS","patients":102,"site_mode_fold_metrics":checked,
  "heldout_rows_checked":len(predictions),"max_coefficient_error":maxcoef,"max_value_error":maxvalue,"max_r2_error":maxr2,
  "producer_sha256":sha(P/"repo/scripts/score_tcpa_cptac_paired_phospho.py"),
  "interpretation":"Both raw and residual values are numerically verified; they represent different outcomes, not treatment efficacy or causal activation."}
(P/"PHOSPHO_INDEPENDENT_REVIEW.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))
