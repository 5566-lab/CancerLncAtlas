import pathlib,socket,subprocess,os,json,hashlib,sys
assert socket.gethostname()=="149"
P=pathlib.Path("/dell_2/DSC/CancerLncAtlas/runtime/tcpa_cptac_calibration_20261106_r1");C=P/"repo";O=P/"protein_cv_review_r1";O.mkdir(exist_ok=False)
sys.path.insert(0,"/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps")
import pandas as pd
proof=pd.read_csv(P/"specimen_audit_r1/PATIENT_SPECIMEN_CONCORDANCE.tsv",sep="\t")
expected=set(proof.loc[proof.dataset.eq("BRCA_BI_protein")&proof.pdc_vial_concordant,"patient_id"])
old=(P/"verify_protein_pilot.py").read_text()
heldout=[];results=[]
for fold in range(5):
 q=P/"protein_cv_r1"/("fold"+str(fold))
 split=pd.read_csv(q/"PATIENT_SPLIT.tsv",sep="\t")
 assert set(split.patient_id)==expected and split.patient_id.is_unique
 assert split.heldout.eq(split.patient_fold_id.eq(fold)).all()
 heldout+=split.loc[split.heldout,"patient_id"].tolist()
 source=old.replace('Q=P/"protein_pilot_r1"','Q=P/"protein_cv_r1"/"fold'+str(fold)+'"')
 source=source.replace('P/"INDEPENDENT_REVIEW_PREFLIGHT.json"','P/"protein_cv_review_r1"/"fold'+str(fold)+'_PREFLIGHT.json"')
 source=source.replace('P/"INDEPENDENT_REVIEW.json"','P/"protein_cv_review_r1"/"fold'+str(fold)+'.json"')
 f=O/("review_fold"+str(fold)+".py");f.write_text(source)
 with (O/("fold"+str(fold)+".log")).open("w") as log:subprocess.run([sys.executable,str(f)],stdout=log,stderr=subprocess.STDOUT,check=True)
 results.append(json.loads((O/("fold"+str(fold)+".json")).read_text()))
 print("INDEPENDENT_FOLD_PASS",fold,flush=True)
assert len(heldout)==len(set(heldout))==len(expected)==102
result={"target_host":"149","status":"PASS_ALL_FIVE_PROTEIN_FOLDS","patients":102,"all_heldout_patients_exactly_once":True,
 "fitted_pathways_checked":sum(r["fitted_pathways"] for r in results),"heldout_r2_checked":sum(r["heldout_r2_checked"] for r in results),
 "maximum_r2_error":max(r["max_r2_error"] for r in results),"folds":results}
(O/"COMPLETE.json").write_text(json.dumps(result,indent=2));print("CV_REVIEW_COMPLETE",result["heldout_r2_checked"],result["maximum_r2_error"],flush=True)
