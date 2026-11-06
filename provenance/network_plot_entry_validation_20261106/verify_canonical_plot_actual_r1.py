
"""Validate normal plot entry on a copy of actual reviewed original results."""
import pathlib,socket,json,hashlib,os,sys,time,shutil
assert socket.gethostname()=="149"
P=pathlib.Path("/dell_2/DSC/CancerLncAtlas/runtime/network_validation_20261106_r1")
C=P/"plot_default_entry_candidate_r1";Q=P/"canonical_plot_actual_r1"
R=pathlib.Path("/dell_2/DSC/CancerLncAtlas/results/model/NETWORK_VALIDATION_REPORT_demo_20261106_r2")
T=pathlib.Path("/dell_2/DSC/CancerLncAtlas/results/model/NETWORK_VALIDATION_demo_20261106_r2")
Q.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
code={f:sha(C/f) for f in ["scripts/plot_network_validation.py","scripts/plot_network_validation_complete_repeats.py","tests/test_network_plot_repeat_coverage.py"]}
(Q/"LAUNCH_PREFLIGHT.json").write_text(json.dumps(dict(target_host="149",action="ACTUAL_REPORT_COPY_CANONICAL_FIGURE_REVIEW",code=code,source=str(R),time=time.time()),indent=2))
for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1";os.environ["MPLCONFIGDIR"]=str(P/"matplotlib_cache")
sys.path.insert(0,str(C))
for v in ["cla_m0_joint_20261105_r1","cla_m0_lasso_20261104_r1","cla_m0_linear_20261104_r1"]:sys.path.append("/dsk2/user/dengsc/"+v+"/python_deps")
import numpy as np,pandas as pd
from scripts import plot_network_validation as plotter
names=["ALL_TARGET_METRICS.parquet","PAIRED_MODEL_DELTAS.parquet","THREE_ARM_COMPARISONS.parquet","SUMMARY_COMPLETE.json","BOOTSTRAP_COMPLETE.json"]
names+=["RNA_INCREMENT_OVER_OMICS_SUMMARY.tsv","THREE_ARM_SUMMARY.tsv","NETWORK_GAIN_SUMMARY.tsv","PATIENT_BOOTSTRAP.tsv","MODEL_SUMMARY.tsv","COVERAGE_SUMMARY.tsv","PATIENT_COVERAGE_SUMMARY.tsv","NETWORK_FEATURE_COVERAGE.tsv","NETWORK_PARTITIONS.tsv","PHASE_AUDIT.tsv","HYPERPARAMETERS.tsv"]
bound={}
for f in names:
 bound[f]=sha(R/f);shutil.copy2(R/f,Q/f);assert sha(Q/f)==bound[f]
(Q/"SOURCE_COPY_RECEIPT.json").write_text(json.dumps(dict(target_host="149",source=str(R),files=bound),indent=2))
def independent_join(frame,keys,value):
 left=frame.loc[frame.repeat.eq(0),keys+[value]].rename(columns={value:"repeat0"})
 right=frame.loc[frame.repeat.eq(1),keys+[value]].rename(columns={value:"repeat1"})
 paired=left.merge(right,on=keys,how="inner",validate="one_to_one")
 paired[value]=(paired.repeat0+paired.repeat1)/2.
 return paired
met=pd.read_parquet(Q/"ALL_TARGET_METRICS.parquet");good=met.loc[met.success&np.isfinite(met.r2)]
expected07=independent_join(good,["cancer_id","target_id","variant","arm","display_layer"],"r2")
d=pd.read_parquet(Q/"PAIRED_MODEL_DELTAS.parquet")
expected05=independent_join(d,["cancer_id","target_id","arm","display_layer","method","baseline"],"delta_r2")
checks=[];real_save=plotter.save
def checked_save(fig,folder,name):
 if name=="05_network_ablation":
  assert len(fig.axes)==6
  for row,layer in enumerate(["PROTEIN_PATHWAY","PHOSPHO_PTMSEA"]):
   for col,arm in enumerate(plotter.ARMS):
    ax=fig.axes[row*3+col];assert len(ax.collections)==3
    for i,method in enumerate(["module_ridge_ppi","module_ridge_rpi","module_ridge_union"]):
     expected=expected05.loc[expected05.display_layer.eq(layer)&expected05.arm.eq(arm)&expected05.method.eq(method)&expected05.baseline.eq("ridge_adaptive_omics"),"delta_r2"].to_numpy()
     actual=np.asarray(ax.collections[i].get_offsets()[:,1],float)
     assert len(actual)==len(expected)
     np.testing.assert_allclose(np.sort(actual),np.sort(expected),rtol=1e-13,atol=1e-13)
     checks.append(dict(figure=name,layer=layer,arm=arm,method=method,points=len(actual),negative=int((actual<0).sum()),max_difference=float(np.max(np.abs(np.sort(actual)-np.sort(expected)))) if len(actual) else 0.))
 elif name=="07_model_performance_table":
  methods=["rna_only","omics_only","ridge_equal","ridge_adaptive_omics","rna_laplacian_ppi","rna_laplacian_rpi","rna_laplacian_union","module_ridge_ppi","module_ridge_rpi","module_ridge_union"]
  for ax,layer in zip(fig.axes,["PROTEIN_PATHWAY","PHOSPHO_PTMSEA"]):
   assert len(ax.images)==1;actual=np.asarray(ax.images[0].get_array(),float);expected=np.full((10,3),np.nan)
   for row,method in enumerate(methods):
    for col,arm in enumerate(plotter.ARMS):
     values=expected07.loc[expected07.display_layer.eq(layer)&expected07.variant.eq(method)&expected07.arm.eq("shared" if method=="omics_only" else arm),"r2"]
     expected[row,col]=np.median(values)
   np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)
   checks.append(dict(figure=name,layer=layer,cells=60//2,max_difference=float(np.max(np.abs(actual-expected)))))
 real_save(fig,folder,name)
 print("FIGURE_CHECKED_AND_EXPORTED",name,flush=True)
plotter.save=checked_save
plotter.plot(Q,T)
assert all(sha(R/f)==h for f,h in bound.items())
assert all(sha(C/f)==h for f,h in code.items())
frozen=json.loads((P/"RESOLUTION4_POSTPROCESS_PREFLIGHT.json").read_text())
assert all(sha(P/"reporting_repo"/f)==h for f,h in frozen["code"].items())
pd.DataFrame(checks).to_csv(Q/"FIGURE_DATA_CHECKS.tsv",sep="\t",index=False)
result=dict(target_host="149",status="PASS_NUMERICAL_CANONICAL_ENTRY_VISUAL_REVIEW_PENDING",figures=10,
 fig05_independent_repeat_join_rows=len(expected05),fig07_independent_repeat_join_rows=len(expected07),
 checked_fig05_panels=18,checked_fig07_cells=60,checks=checks,original_source_unchanged=True,frozen_live_reporting_code_unchanged=True,
 code=code,files={f.name:sha(f) for f in Q.iterdir() if f.suffix in [".png",".pdf",".html"]},time=time.time())
(Q/"NUMERICAL_REVIEW.json").write_text(json.dumps(result,indent=2));print("ACTUAL_CANONICAL_REVIEW_COMPLETE",flush=True)
