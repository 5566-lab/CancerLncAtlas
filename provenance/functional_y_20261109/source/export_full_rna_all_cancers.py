from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,socket,os
import pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'full_tcga_tpm_r1';O.mkdir(exist_ok=True)
assert not (O/'COMPLETE.json').exists()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
raw=Path('/dsk2/user/xumj/ShuJuKu/data/01_bulk/GDCdata')
patientmap=Path('/dell_2/DSC/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv')
assert sha(patientmap)=='e05c20085be532159bb6a51f200a27923975c3caa6a90910290e10a5b60e6253'
auth=pd.read_csv(patientmap,sep='\t');cancers=sorted(auth.cancer_id.unique());assert len(cancers)==33
for c in cancers:assert (raw/('TCGA-'+c+'.RDS')).is_file(),c
preflight={'target_host':'149','operation':'RESTORE_UNFILTERED_RNA_FOR_RELATION_Y','training_started':False,'cancers':cancers,'no_association_admission_filter':True,'patient_map_sha256':sha(patientmap),'source_script_sha256':sha(R/'export_unfiltered_tcga_tpm.R'),'existing_three_cancer_exports_reused_after_hash_checks':True}
(O/'LAUNCH_PREFLIGHT.json').write_text(json.dumps(preflight,indent=2))
old=Path('/dsk2/user/dengsc/cla_m0_lasso_20261104_r1/tcga_full_tpm_r1');om=json.loads((old/'COMPLETE.json').read_text());records={};failures={}
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
for c in cancers:
 try:
  if c in om['cancers']:
   z=om['cancers'][c]
   assert sha(old/c/'GENE_TPM.tsv.gz')==z['outputs']['tpm_sha256']
   assert sha(old/c/'GENE_ANNOTATION.tsv')==z['outputs']['annotation_sha256']
   assert z['patients']==int(auth.cancer_id.eq(c).sum())
   records[c]={'path':str(old/c),'source':'REUSED_VERIFIED_FULL_EXPORT','receipt':z}
  else:
   out=O/'runs'/c;log=O/(c+'.log')
   with log.open('w') as f:q=subprocess.run(['/usr/bin/Rscript',str(R/'export_unfiltered_tcga_tpm.R'),str(raw),str(patientmap),str(out),c],stdout=f,stderr=subprocess.STDOUT,env=env,timeout=1800)
   if q.returncode:raise RuntimeError('R_EXPORT_FAILED: '+log.read_text()[-1500:])
   z=json.loads((out/'COMPLETE.json').read_text())['cancers'][c]
   records[c]={'path':str(out/c),'source':'EXPORTED_FULL_RAW_TPM','receipt':z}
  print('VERIFIED_FULL_RNA',c,records[c]['receipt']['patients'],records[c]['receipt']['source_lncRNAs'],flush=True)
 except Exception as exc:
  failures[c]=str(exc);print('RNA_EXPORT_UNAVAILABLE',c,str(exc),flush=True)
 (O/'PROGRESS.json').write_text(json.dumps({'target_host':'149','records':records,'failures':failures,'completed_cancers':len(records),'expected_cancers':len(cancers)},indent=2))
result={'target_host':'149','status':'COMPLETE' if not failures else 'INCOMPLETE_WITH_EXPLICIT_FAILURES','records':records,'failures':failures,'completed_cancers':len(records),'expected_cancers':33,'patients':sum(x['receipt']['patients'] for x in records.values()),'no_rho_or_FDR_filter':True,'updated_utc':datetime.now(timezone.utc).isoformat()}
(O/('COMPLETE.json' if not failures else 'INCOMPLETE.json')).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
