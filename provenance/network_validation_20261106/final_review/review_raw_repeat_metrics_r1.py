
"""Independent centered-moment review of every available complete CV repeat."""
import argparse,hashlib,json,os,pathlib,socket,sys,time
assert socket.gethostname()=="149"
for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:os.environ[key]="1"
sys.path.insert(0,"/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps")
import numpy as np,pandas as pd
P=pathlib.Path("/dell_2/DSC/CancerLncAtlas/runtime/network_validation_20261106_r1")
KEYS=["target_id","variant","arm"]
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(8388608),b""):h.update(block)
 return h.hexdigest()
def dump(path,record):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(record,indent=2));tmp.replace(path)
def centered_metrics(frame):
 if frame.duplicated(KEYS+["patient_id"]).any():raise ValueError("REPEATED_PATIENT_WITHIN_CV_REPEAT")
 if not np.isfinite(frame[["observed","predicted"]].to_numpy()).all():raise ValueError("NONFINITE_PREDICTION")
 # Center directly over all held-out patients rather than subtract sum(y)^2/n.
 means=frame.groupby(KEYS,observed=True)[["observed","predicted"]].transform("mean")
 dy=frame.observed.to_numpy()-means.observed.to_numpy()
 dp=frame.predicted.to_numpy()-means.predicted.to_numpy()
 values=frame[KEYS].copy()
 values["sse"]=(frame.predicted.to_numpy()-frame.observed.to_numpy())**2
 values["sst"]=dy*dy;values["prediction_sst"]=dp*dp;values["cross_deviation"]=dy*dp
 g=values.groupby(KEYS,observed=True).agg(n=("sse","size"),sse=("sse","sum"),sst=("sst","sum"),
     prediction_sst=("prediction_sst","sum"),cross_deviation=("cross_deviation","sum"))
 g["r2"]=1-g.sse/g.sst;g["rmse"]=np.sqrt(g.sse/g.n)
 g["pearson"]=g.cross_deviation/np.sqrt(g.sst*g.prediction_sst)
 return g.reset_index()
def review_repeat(training,cancer,repeat,out):
 names=[f"repeat{repeat}_outer{k}" for k in range(5)]+["final"]
 root=training/cancer
 if not all((root/name/"COMPLETE.json").exists() for name in names):return None
 source={str(root/name/"COMPLETE.json"):sha(root/name/"COMPLETE.json") for name in names}
 if (out/"COMPLETE.json").exists():
  old=json.loads((out/"COMPLETE.json").read_text())
  assert old["source_receipts"]==source and old["producer_sha256"]==sha(pathlib.Path(__file__))
  for name,h in old["files"].items():assert sha(out/name)==h
  return old
 out.mkdir(exist_ok=False)
 dump(out/"LAUNCH_PREFLIGHT.json",dict(target_host="149",action="INDEPENDENT_RAW_HELDOUT_CENTERED_METRICS",cancer=cancer,repeat=repeat,source_receipts=source,producer_sha256=sha(pathlib.Path(__file__))))
 pred=[];audits=[];hashes={}
 for phase in names:
  folder=root/phase;receipt=json.loads((folder/"COMPLETE.json").read_text())
  for name in (["AUDIT.parquet"] if phase=="final" else ["AUDIT.parquet","PREDICTIONS.parquet"]):
   h=sha(folder/name);assert receipt["files"][name]==h;hashes[str(folder/name)]=h
  a=pd.read_parquet(folder/"AUDIT.parquet",columns=KEYS+["status"])
  assert not a.duplicated(KEYS).any();a["phase"]=phase;audits.append(a)
  if phase!="final":
   f=pd.read_parquet(folder/"PREDICTIONS.parquet",columns=KEYS+["patient_id","observed","predicted"])
   pred.append(f)
  print("RAW_REVIEW_READ_PHASE",cancer,repeat,phase,flush=True)
 a=pd.concat(audits,ignore_index=True)
 a["fitted"]=a.status.eq("FITTED")
 status=a.groupby(KEYS,observed=True).agg(attempted_phases=("phase","nunique"),fitted_phases=("fitted","sum")).reset_index()
 assert status.attempted_phases.eq(6).all()
 status["success"]=status.fitted_phases.eq(6)
 f=pd.concat(pred,ignore_index=True)
 for key in KEYS+["patient_id"]:f[key]=f[key].astype("category")
 m=centered_metrics(f).merge(status,on=KEYS,validate="one_to_one")
 m.to_parquet(out/"INDEPENDENT_METRICS.parquet",index=False)
 status.to_parquet(out/"INDEPENDENT_STATUS.parquet",index=False)
 result=dict(target_host="149",status="COMPLETE_INDEPENDENT_RAW_REPEAT",cancer=cancer,repeat=repeat,
  raw_prediction_rows=len(f),metric_rows=len(m),candidate_model_targets=len(status),success_rows=int(m.success.sum()),
  negative_finite_metric_rows=int((m.r2.lt(0)&np.isfinite(m.r2)).sum()),
  source_receipts=source,source_files=hashes,producer_sha256=sha(pathlib.Path(__file__)),
  method="Direct centering over all five outer folds; no report-producer statistics helper imported",
  files={name:sha(out/name) for name in ["INDEPENDENT_METRICS.parquet","INDEPENDENT_STATUS.parquet","LAUNCH_PREFLIGHT.json"]},time=time.time())
 dump(out/"COMPLETE.json",result);print("RAW_REPEAT_REVIEW_COMPLETE",cancer,repeat,len(m),flush=True);return result
def selfcheck(output):
 output.mkdir(exist_ok=False);dump(output/"LAUNCH_PREFLIGHT.json",dict(target_host="149",action="CENTERED_METRIC_INDEPENDENT_ARITHMETIC_CHECK",producer_sha256=sha(pathlib.Path(__file__))))
 f=pd.DataFrame(dict(target_id=["t"]*3,variant=["v"]*3,arm=["a"]*3,patient_id=["p1","p2","p3"],observed=[1.,2.,4.],predicted=[4.,2.,1.]))
 for offset in [0.,1e8]:
  g=f.copy();g[["observed","predicted"]]+=offset
  m=centered_metrics(g).iloc[0]
  np.testing.assert_allclose([m.r2,m.rmse,m.pearson],[-20/7,np.sqrt(6),-13/14],rtol=1e-12,atol=1e-12)
 try:centered_metrics(pd.concat([f,f.iloc[[0]]],ignore_index=True))
 except ValueError as e:assert "REPEATED_PATIENT" in str(e)
 else:raise AssertionError("Duplicate patient not rejected")
 dump(output/"COMPLETE.json",dict(target_host="149",status="PASS_CLOSED_FORM_NEGATIVE_AND_OFFSET_AND_DUPLICATE_CHECK",producer_sha256=sha(pathlib.Path(__file__))))
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--training",type=pathlib.Path);parser.add_argument("--output",type=pathlib.Path,required=True);parser.add_argument("--cancers",nargs="+",default=["BRCA","LUAD","COAD"]);parser.add_argument("--repeats",nargs="+",type=int,default=[0,1]);parser.add_argument("--selfcheck",action="store_true");args=parser.parse_args()
 if args.selfcheck:selfcheck(args.output);print("SELFCHECK_PASS",flush=True);return
 assert args.training is not None
 args.output.mkdir(exist_ok=True)
 dump(args.output/"PREFLIGHT.json",dict(target_host="149",action="INCREMENTAL_RAW_REPEAT_REVIEW",training=str(args.training),producer_sha256=sha(pathlib.Path(__file__)),time=time.time()))
 results=[];pending=[]
 for cancer in args.cancers:
  for repeat in args.repeats:
   r=review_repeat(args.training,cancer,repeat,args.output/(cancer+"_repeat"+str(repeat)))
   if r is None:pending.append([cancer,repeat])
   else:results.append(r)
   dump(args.output/"STATUS.json",dict(target_host="149",status="PARTIAL" if pending or len(results)<len(args.cancers)*len(args.repeats) else "COMPLETE_RAW_RECONSTRUCTION",completed=len(results),required=len(args.cancers)*len(args.repeats),pending=pending,raw_prediction_rows=sum(r["raw_prediction_rows"] for r in results),metric_rows=sum(r["metric_rows"] for r in results),time=time.time()))
 print("RAW_REVIEW_EXIT",len(results),pending,flush=True)
if __name__=="__main__":main()
