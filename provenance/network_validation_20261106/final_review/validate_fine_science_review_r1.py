
"""Challenge the report reviewer with deliberately corrupted synthetic reports."""
import pathlib,json,socket,subprocess,sys,shutil,time,hashlib
assert socket.gethostname()=="149"
P=pathlib.Path("/dell_2/DSC/CancerLncAtlas/runtime/network_validation_20261106_r1")
sys.path.insert(0,"/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps")
import pandas as pd
F=pathlib.Path(json.loads((P/"raw_metric_synthetic_fine_r1/REPORT_COMPARISON.json").read_text())["source_fixture"])
Q=P/"fine_science_negative_checks_r1";Q.mkdir(exist_ok=False)
(Q/"LAUNCH_PREFLIGHT.json").write_text(json.dumps(dict(target_host="149",action="DELIBERATE_CORRUPTION_OF_SYNTHETIC_REPORT_COPIES",producer_sha256=hashlib.sha256((P/"review_fine_science_r1.py").read_bytes()).hexdigest())))
names=["FINE_TARGET_METRICS.parquet","FINE_TRAINING_STATUS.parquet","FINE_PAIRED_MODEL_DELTAS.parquet","FINE_GAIN_SUMMARY.tsv","FINE_THREE_ARM_COMPARISONS.parquet","FINE_COVERAGE_SUMMARY.tsv","FINE_PATIENT_BOOTSTRAP.tsv"]
cases=[
 ("wrong_r2","FINE_TARGET_METRICS.parquet","r2"),
 ("missing_metric_row","FINE_TARGET_METRICS.parquet","DIFFERENT_ROW_AXIS"),
 ("wrong_failure_flag","FINE_TRAINING_STATUS.parquet","success"),
 ("wrong_null_difference","FINE_PAIRED_MODEL_DELTAS.parquet","delta_r2"),
 ("wrong_bootstrap_point","FINE_PATIENT_BOOTSTRAP.tsv","family_balanced_mean_delta_r2")]
checks=[]
for case,file,expected in cases:
 r=Q/case;r.mkdir()
 for name in names:shutil.copy2(F/"SIMULATED_FINE_REPORT"/name,r/name)
 f=r/file;d=pd.read_parquet(f) if f.suffix==".parquet" else pd.read_csv(f,sep="\t")
 if case=="wrong_r2":d.loc[0,"r2"]+=.25
 elif case=="missing_metric_row":d=d.iloc[1:].copy()
 elif case=="wrong_failure_flag":d.loc[0,"success"]=not bool(d.loc[0,"success"])
 elif case=="wrong_null_difference":
  j=d.loc[d.baseline.eq("mean_five_resolution4_nulls")].index[0];d.loc[j,"delta_r2"]+=.25
 else:d.loc[0,"family_balanced_mean_delta_r2"]+=.25
 if f.suffix==".parquet":d.to_parquet(f,index=False)
 else:d.to_csv(f,sep="\t",index=False)
 out=Q/(case+"_review")
 cmd=[sys.executable,str(P/"review_fine_science_r1.py"),"--raw",str(P/"raw_metric_synthetic_fine_r1"),"--report",str(r),"--original",str(F/"SIMULATED_REPORT"),"--out",str(out),"--expected-targets","8","--synthetic"]
 result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (Q/(case+".log")).write_text(result.stdout)
 assert result.returncode!=0 and expected in result.stdout,(case,result.returncode,result.stdout[-2000:])
 assert not (out/"COMPLETE.json").exists()
 checks.append(dict(case=case,status="CORRECTLY_REJECTED",expected_error=expected))
 print("NEGATIVE_CHECK_PASS",case,flush=True)
(Q/"COMPLETE.json").write_text(json.dumps(dict(target_host="149",status="PASS_ALL_FIVE_CORRUPTION_CHECKS",checks=checks,time=time.time()),indent=2))
