
"""Compare fine report against independent raw-prediction reconstruction."""
import argparse,hashlib,json,os,pathlib,socket,sys,time
assert socket.gethostname()=="149"
for k in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:os.environ[k]="1"
sys.path.insert(0,"/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps")
import numpy as np,pandas as pd
MAIN="module_resolution4_union"
NULLS=[MAIN+"_random_"+str(s) for s in [101,202,303,404,505]]
BASES=["module_ridge_union","ridge_adaptive_omics","ridge_equal","rna_laplacian_union"]
METHODS=BASES+[MAIN]+NULLS;ARMS=["lncrna","mrna","combined"]
K=["cancer_id","repeat","target_id","variant","arm"];U=["cancer_id","repeat","target_id","arm"]
PRIMARY=["PROTEIN_PATHWAY","PHOSPHO_PTMSEA"]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def match(expected,actual,keys,numeric,exact=()):
 assert not expected.duplicated(keys).any() and not actual.duplicated(keys).any()
 m=expected.merge(actual,on=keys,how="outer",validate="one_to_one",suffixes=("_expected","_actual"),indicator=True)
 assert m._merge.eq("both").all(),("DIFFERENT_ROW_AXIS",m._merge.value_counts().to_dict())
 differences={}
 for name in exact:assert m[name+"_expected"].eq(m[name+"_actual"]).all(),name
 for name in numeric:
  a=m[name+"_expected"].to_numpy(float);b=m[name+"_actual"].to_numpy(float)
  np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-8,equal_nan=True,err_msg=name)
  finite=np.isfinite(a)&np.isfinite(b);differences[name]=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
 return dict(rows=len(m),max_absolute_difference=differences)
def run(raw,report,original,out,expected_targets,synthetic=False):
 out.mkdir(exist_ok=False)
 (out/"LAUNCH_PREFLIGHT.json").write_text(json.dumps(dict(target_host="149",action="COMPARE_INDEPENDENT_RAW_METRICS_TO_FINE_REPORT",raw=str(raw),report=str(report),original=str(original),synthetic=synthetic,expected_targets_per_cancer=expected_targets,producer_sha256=sha(pathlib.Path(__file__))),indent=2))
 sources={}
 def load(root,name):
  f=root/name;sources[str(f)]=sha(f)
  return pd.read_parquet(f) if f.suffix==".parquet" else pd.read_csv(f,sep="\t")
 mr=[];sr=[]
 for ca in ["BRCA","LUAD","COAD"]:
  for rep in [0,1]:
   folder=raw/(ca+"_repeat"+str(rep));rec=json.loads((folder/"COMPLETE.json").read_text());assert rec["status"]=="COMPLETE_INDEPENDENT_RAW_REPEAT"
   for f,h in rec["files"].items():assert sha(folder/f)==h
   mr.append(load(folder,"INDEPENDENT_METRICS.parquet").assign(cancer_id=ca,repeat=rep))
   sr.append(load(folder,"INDEPENDENT_STATUS.parquet").assign(cancer_id=ca,repeat=rep))
 im=pd.concat(mr,ignore_index=True);ist=pd.concat(sr,ignore_index=True)
 met=load(report,"FINE_TARGET_METRICS.parquet");st=load(report,"FINE_TRAINING_STATUS.parquet")
 checks={}
 checks["raw_metrics"]=match(im,met,K,["r2","rmse","pearson","sse","sst"],["n","success","fitted_phases","attempted_phases"] if "fitted_phases" in met else ["n","success"])
 checks["raw_status"]=match(ist,st,K,[],["success","fitted_phases","attempted_phases"])
 assert len(ist)==3*2*expected_targets*18
 assert ist.groupby(["cancer_id","repeat","target_id"],observed=True).size().eq(18).all()
 meta=met[["cancer_id","target_id","display_layer","family"]].drop_duplicates();assert not meta.duplicated(["cancer_id","target_id"]).any()
 im=im.merge(meta,on=["cancer_id","target_id"],validate="many_to_one")
 old=load(original,"ALL_TARGET_METRICS.parquet")
 allm=pd.concat([old.loc[old.variant.isin(BASES)&old.arm.isin(ARMS)],im],ignore_index=True)
 good=allm.loc[allm.success&np.isfinite(allm.r2)].copy()
 main=good.loc[good.variant.eq(MAIN)]
 deltas=[]
 for base in BASES+NULLS:
  reference=good.loc[good.variant.eq(base)]
  p=main[U+["r2","n","display_layer"]].merge(reference[U+["r2","n"]],on=U,validate="one_to_one",suffixes=("_main","_base"))
  assert p.n_main.eq(p.n_base).all()
  p["delta_r2"]=p.r2_main-p.r2_base;p["baseline"]=base;deltas.append(p[U+["display_layer","baseline","delta_r2"]])
 p=main[U+["r2","display_layer"]].rename(columns={"r2":"main"})
 for j,base in enumerate(NULLS):
  p=p.merge(good.loc[good.variant.eq(base),U+["r2"]].rename(columns={"r2":"null"+str(j)}),on=U,validate="one_to_one")
 p["delta_r2"]=p.main-sum(p["null"+str(j)] for j in range(5))/5;p["baseline"]="mean_five_resolution4_nulls"
 deltas.append(p[U+["display_layer","baseline","delta_r2"]]);expected_delta=pd.concat(deltas,ignore_index=True)
 checks["paired_deltas"]=match(expected_delta,load(report,"FINE_PAIRED_MODEL_DELTAS.parquet"),U+["baseline"],["delta_r2"],["display_layer"])
 gain=expected_delta.groupby(["cancer_id","repeat","arm","display_layer","baseline"],observed=True).agg(targets=("target_id","size"),median_delta_r2=("delta_r2","median"),mean_delta_r2=("delta_r2","mean"),improved=("delta_r2",lambda s:int(s.gt(1e-9).sum()))).reset_index()
 checks["gain_summary"]=match(gain,load(report,"FINE_GAIN_SUMMARY.tsv"),["cancer_id","repeat","arm","display_layer","baseline"],["median_delta_r2","mean_delta_r2"],["targets","improved"])
 three=None;keys=["cancer_id","repeat","target_id"]
 for arm in ARMS:
  a=main.loc[main.arm.eq(arm),keys+["r2"]].rename(columns={"r2":arm})
  three=a if three is None else three.merge(a,on=keys,validate="one_to_one")
 three["combined_minus_mrna"]=three.combined-three.mrna;three["lncrna_minus_mrna"]=three.lncrna-three.mrna
 checks["three_arms"]=match(three,load(report,"FINE_THREE_ARM_COMPARISONS.parquet"),keys,ARMS+["combined_minus_mrna","lncrna_minus_mrna"])
 independent_status=ist.merge(st[K+["display_layer"]],on=K,validate="one_to_one")
 coverage=independent_status.groupby(["cancer_id","repeat","variant","arm","display_layer"],observed=True).agg(attempted=("target_id","size"),successful=("success","sum")).reset_index()
 coverage["failed"]=coverage.attempted-coverage.successful
 checks["coverage"]=match(coverage,load(report,"FINE_COVERAGE_SUMMARY.tsv"),["cancer_id","repeat","variant","arm","display_layer"],[],["attempted","successful","failed"])
 # Bootstrap point validation: same observations across models/repeats is also
 # checked by the complete full-record patient-identity audits in production.
 primary=good.loc[good.display_layer.isin(PRIMARY)&good.variant.isin(METHODS)&good.arm.isin(ARMS)]
 complete=primary.groupby(["cancer_id","target_id"],observed=True).size()
 complete=complete.loc[complete.eq(2*len(METHODS)*3)].index
 primary=primary.set_index(["cancer_id","target_id"]).loc[complete].reset_index()
 groups=primary.groupby(["cancer_id","target_id"],observed=True)
 assert (groups.n.max()-groups.n.min()).eq(0).all()
 assert ((groups.sst.max()-groups.sst.min())<=1e-8*np.maximum(1,groups.sst.max().abs())).all()
 assert primary.n.ge(12).all() and (primary.sst/primary.n).gt(1e-12).all()
 idx=["cancer_id","target_id","display_layer","family"]
 wide=primary.groupby(idx+["variant","arm"],observed=True).r2.mean().unstack(["variant","arm"])
 assert wide.notna().all().all()
 rows=[]
 def point_rows(delta,arm,baseline):
  d=delta.rename("delta").reset_index()
  v=d.groupby(["cancer_id","display_layer","family"],observed=True).delta.mean().groupby(["cancer_id","display_layer"]).mean()
  for (ca,layer),point in v.items():rows.append(dict(cancer_id=ca,layer=layer,arm=arm,method=MAIN,baseline=baseline,independent_point=float(point)))
 for arm in ARMS:
  for base in BASES:point_rows(wide[(MAIN,arm)]-wide[(base,arm)],arm,base)
  point_rows(wide[(MAIN,arm)]-sum(wide[(base,arm)] for base in NULLS)/5,arm,"mean_five_resolution4_nulls")
 for label,arm in [("combined_vs_mrna","combined"),("lncrna_vs_mrna","lncrna")]:
  point_rows(wide[(MAIN,arm)]-wide[(MAIN,"mrna")],label,MAIN)
 points=pd.DataFrame(rows);pool=points.groupby(["layer","arm","method","baseline"]).independent_point.mean().reset_index();pool["cancer_id"]="EQUAL_CANCER_MEAN"
 points=pd.concat([points,pool],ignore_index=True)
 boot=load(report,"FINE_PATIENT_BOOTSTRAP.tsv");keys=["cancer_id","layer","arm","method","baseline"]
 checks["bootstrap_points"]=match(points.rename(columns={"independent_point":"family_balanced_mean_delta_r2"}),boot,keys,["family_balanced_mean_delta_r2"])
 assert np.isfinite(boot[["ci_low","ci_high"]]).all().all() and boot.ci_low.le(boot.ci_high).all()
 expected_replicates=50 if synthetic else 2000;assert boot.replicates.eq(expected_replicates).all()
 points.merge(boot,on=keys,validate="one_to_one").to_csv(out/"INDEPENDENT_BOOTSTRAP_POINT_CHECK.tsv",sep="\t",index=False,na_rep="NA")
 rec=dict(target_host="149",status="PASS_INDEPENDENT_RAW_METRICS_AND_SCIENTIFIC_REPORT_COMPARISON",synthetic=synthetic,checks=checks,
  negative_finite_metric_rows_retained=int((im.r2.lt(0)&np.isfinite(im.r2)).sum()),failure_state_rows_retained=int((~ist.success).sum()),
  common_primary_cancer_targets=len(complete),source_files=sources,producer_sha256=sha(pathlib.Path(__file__)),
  bootstrap_interval_scope="Checked complete point estimates, interval ordering and recorded replicate count. Intervals use previously tested producer; no independent resampling in this review.",
  time=time.time())
 (out/"COMPLETE.json").write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 for name in ["raw","report","original","out"]:p.add_argument("--"+name,type=pathlib.Path,required=True)
 p.add_argument("--expected-targets",type=int,default=8750);p.add_argument("--synthetic",action="store_true")
 a=p.parse_args();run(a.raw,a.report,a.original,a.out,a.expected_targets,a.synthetic)
