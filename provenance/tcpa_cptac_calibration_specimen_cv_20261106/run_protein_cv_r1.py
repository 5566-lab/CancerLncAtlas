import os,pathlib,socket,subprocess,json,hashlib,sys
assert socket.gethostname()=="149"
P=pathlib.Path("/dell_2/DSC/CancerLncAtlas/runtime/tcpa_cptac_calibration_20261106_r1");C=P/"repo";B=P.parents[1]
sys.path.insert(0,str(C));sys.path.insert(0,"/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps")
import pandas as pd,numpy as np
Q=P/"protein_cv_r1";Q.mkdir(exist_ok=False)
(Q/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host":"149","action":"BRCA_ALL_FIVE_FROZEN_FOLDS_AFTER_SPECIMEN_CONCORDANCE",
 "folds":[0,1,2,3,4],"additional_random_repeats":False,"tuning":False,
 "aggregation":"Per-fold metrics, then equal-fold mean; never pool different fold reference scores as one Y.",
 "source_code_sha256":hashlib.sha256((C/"scripts/score_tcpa_cptac_paired_protein_pilot.py").read_bytes()).hexdigest()},indent=2))
tables=[]
for fold in range(5):
 dest=Q/("fold"+str(fold))
 cmd=[sys.executable,str(C/"scripts/score_tcpa_cptac_paired_protein_pilot.py"),"--rppa",str(B/"inputs/rppa500_20261104_r4"),"--bundle",str(B/"inputs/cptac_20261104_r1/bundle_r1"),"--mapping",str(P/"mapping_r1"),"--retrospective",str(P/"DOWNLOAD_COMPLETE.json"),"--specimens",str(P/"specimen_audit_r1"),"--heldout-fold",str(fold),"--output",str(dest)]
 with (Q/("fold"+str(fold)+".log")).open("w") as log:subprocess.run(cmd,cwd=C,stdout=log,stderr=subprocess.STDOUT,check=True)
 d=pd.read_csv(dest/"HELDOUT_PATHWAY_METRICS.tsv",sep="\t");d["fold"]=fold;tables.append(d)
 print("FOLD_COMPLETE",fold,json.loads((dest/"COMPLETE.json").read_text())["heldout_patients"],flush=True)
allrows=pd.concat(tables,ignore_index=True);allrows.to_csv(Q/"ALL_FOLD_PATHWAY_METRICS.tsv",sep="\t",index=False)
summary=allrows.groupby("pathway_id").agg(completed_folds=("heldout_r2","count"),mean_fold_r2=("heldout_r2","mean"),
 median_fold_r2=("heldout_r2","median"),positive_folds=("heldout_r2",lambda x:int(x.gt(0).sum())),
 min_panel_genes=("panel_genes","min"),min_heldout_pairs=("heldout_pairs","min")).reset_index()
summary.to_csv(Q/"PATHWAY_CV_SUMMARY.tsv",sep="\t",index=False)
stats=[]
for n in [1,3,5,10]:
 d=summary.loc[summary.completed_folds.eq(5)&summary.min_panel_genes.ge(n)]
 stats.append(dict(minimum_panel_genes_descriptive=n,all_five_fold_pathways=len(d),positive_mean_fold_r2=int(d.mean_fold_r2.gt(0).sum()),
   positive_in_all_five=int(d.positive_folds.eq(5).sum()),median_mean_fold_r2=float(d.mean_fold_r2.median())))
result=dict(target_host="149",status="COMPLETE_FIVE_FOLD_DIAGNOSTIC",patients=102,fold_count=5,pathway_axis=len(summary),summary=stats,
 unified_production_Y=False,prospective_transfer_validated=False,
 files={str(p.relative_to(Q)):{"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in Q.rglob("*") if p.is_file()})
(Q/"COMPLETE.json").write_text(json.dumps(result,indent=2));print("PROTEIN_CV_COMPLETE",json.dumps(stats),flush=True)
