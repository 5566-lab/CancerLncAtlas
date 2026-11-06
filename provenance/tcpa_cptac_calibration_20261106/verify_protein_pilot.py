import pathlib,socket,json,hashlib,sys,numpy as np,pandas as pd
assert socket.gethostname()=="149"
P=pathlib.Path("/dell_2/DSC/CancerLncAtlas/runtime/tcpa_cptac_calibration_20261106_r1");Q=P/"protein_pilot_r1"
(P/"INDEPENDENT_REVIEW_PREFLIGHT.json").write_text(json.dumps({"target_host":"149","action":"RECONSTRUCT_PAIRED_PROTEIN_SCORES_AND_HELDOUT_ERRORS"}))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((Q/"COMPLETE.json").read_text())
for p,h in m["source_sha256"].items():assert sha(p)==h,p
for f,r in m["files"].items():assert sha(Q/f)==r["sha256"],f
split=pd.read_csv(Q/"PATIENT_SPLIT.tsv",sep="\t");ids=split.patient_id.tolist();train=~split.heldout.to_numpy(bool);test=~train
sc=pd.read_csv(Q/"TRAIN_REFERENCE.tsv",sep="\t")
B=pathlib.Path("/dell_2/DSC/CancerLncAtlas")
rx=pd.read_parquet(B/"inputs/rppa500_20261104_r4/PROTEIN.parquet").set_index("patient_id").loc[ids,sc.rppa_feature_id];rx.columns=sc.gene_id
ms=pd.read_csv(P/"retrospective_sources/BRCA_BI_protein.tsv.gz",sep="\t",index_col=0)
cx=ms.loc[sc.ms_gene_symbol,[x.replace("-",".") for x in ids]].T
cx.index=ids;cx.columns=sc.gene_id
max_ref=0.
for frame,prefix in [(rx,"rppa"),(cx,"ms")]:
 for key,expected in [("mean",frame.iloc[train].mean()),("sd",frame.iloc[train].std(ddof=1))]:
  actual=sc[prefix+"_train_"+key].to_numpy(float);a=expected.to_numpy(float)
  assert np.array_equal(np.isnan(actual),np.isnan(a))
  max_ref=max(max_ref,float(np.nanmax(np.abs(actual-a))))
rz=rx.subtract(rx.iloc[train].mean()).divide(rx.iloc[train].std(ddof=1))
cz=cx.subtract(cx.iloc[train].mean()).divide(cx.iloc[train].std(ddof=1))
valid=sc.loc[sc.train_usable,"gene_id"].tolist();rz=rz[valid];cz=cz[valid]
rout=pd.read_parquet(Q/"RPPA_COMMON_PANEL_SCORE.parquet").set_index("patient_id").loc[ids]
cout=pd.read_parquet(Q/"MS_COMMON_PANEL_SCORE.parquet").set_index("patient_id").loc[ids]
pout=pd.read_parquet(Q/"RPPA_TO_MS_PREDICTED_SCORE.parquet").set_index("patient_id").loc[ids]
metrics=pd.read_csv(Q/"HELDOUT_PATHWAY_METRICS.tsv",sep="\t").set_index("pathway_id")
bridge=pd.read_parquet(B/"inputs/cptac_20261104_r1/bundle_r1/PATHWAY_GENE_PROTEIN_BRIDGE.parquet")
max_score=max_pred=max_r2=0.;tested=0;fitted=0
for pathway,g in bridge.groupby("pathway_id"):
 hit=sorted(set(g.gene_id)&set(valid))
 x=rz[hit].mean(axis=1).to_numpy();y=cz[hit].mean(axis=1).to_numpy()
 for expected,actual in [(x,rout[pathway].to_numpy()),(y,cout[pathway].to_numpy())]:
  assert np.array_equal(np.isnan(expected),np.isnan(actual))
  if np.isfinite(expected).any():max_score=max(max_score,float(np.nanmax(np.abs(expected-actual))))
 keep=train&np.isfinite(x)&np.isfinite(y)
 if keep.sum()<3 or np.var(x[keep])<=1e-15:continue
 coef=np.linalg.lstsq(np.column_stack([np.ones(keep.sum()),x[keep]]),y[keep],rcond=None)[0]
 prediction=coef[0]+coef[1]*x;actual=pout[pathway].to_numpy()
 assert np.array_equal(np.isnan(prediction),np.isnan(actual))
 max_pred=max(max_pred,float(np.nanmax(np.abs(prediction-actual))));fitted+=1
 keep=test&np.isfinite(prediction)&np.isfinite(y)
 if keep.sum()>=3 and np.var(y[keep])>0:
  r2=1-np.sum((y[keep]-prediction[keep])**2)/np.sum((y[keep]-y[keep].mean())**2)
  max_r2=max(max_r2,abs(float(r2)-float(metrics.loc[pathway,"heldout_r2"])));tested+=1
assert max_ref<1e-10 and max_score<1e-10 and max_pred<1e-10 and max_r2<1e-10
report={"target_host":"149","status":"PASS_INDEPENDENT_NUMERICAL_RECONSTRUCTION","fitted_pathways":fitted,"heldout_r2_checked":tested,
"max_reference_error":max_ref,"max_score_error":max_score,"max_prediction_error":max_pred,"max_r2_error":max_r2,
"code_sha256":{str(f.relative_to(P/"repo")):sha(f) for f in [P/"repo/cla_engine/rppa500.py",P/"repo/scripts/audit_tcpa_cptac_common_panel.py",P/"repo/scripts/score_tcpa_cptac_paired_protein_pilot.py"]},
"scientific_limit":"Numerical verification of the one-fold diagnostic, not validated cross-platform activity or prospective transfer."}
(P/"INDEPENDENT_REVIEW.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
