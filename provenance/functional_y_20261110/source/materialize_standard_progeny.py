from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,socket,sys,hashlib,time,os
import numpy as np,pandas as pd
from standard_progeny import score_progeny
R=Path(__file__).resolve().parent;O=R/'progeny_standard_r1';assert socket.gethostname()=='149'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def work(key):
 t=time.monotonic();d=O/'cohorts'/key;d.mkdir(parents=True,exist_ok=False);model=pd.read_csv(O/'OFFICIAL_TOP100_MODEL.tsv',sep='\t',index_col=0)
 src=R/'common_rna_scores_r1/aligned'/key/'SYMBOL_LOGTPM.parquet';e=pd.read_parquet(src)
 # Raw scores require no patient-derived fitting. Scaling references are fitted downstream by context.
 raw,_,rec=score_progeny(e,model,e.index.tolist());raw.columns=['STATE|native_program::PROGENY:'+s for s in raw.columns];raw.to_parquet(d/'RAW_PROGENY.parquet',compression='zstd')
 rec={k:v for k,v in rec.items() if k not in ['fit_patients','mean','sd','scaling']};rec.update(target_host='149',cohort=key,patient_scaling='DEFERRED_TO_SOURCE_PATIENTS_PER_CONTEXT',source_sha256=sha(src),model_sha256=sha(O/'OFFICIAL_TOP100_MODEL.tsv'),raw_sha256=sha(d/'RAW_PROGENY.parquet'),patients=len(raw),pathways=len(raw.columns),seconds=time.monotonic()-t)
 (d/'COMPLETE.json').write_text(json.dumps(rec,indent=2));print('PROGENY_COMPLETE',key,len(raw),round(rec['seconds'],2),flush=True);return key,rec
if __name__=='__main__':
 v=json.loads((O/'IMPLEMENTATION_VALIDATION.json').read_text());assert v['status']=='PASS';assert v['implementation_sha256']==sha(R/'standard_progeny.py');assert v['model_sha256']==sha(O/'OFFICIAL_TOP100_MODEL.tsv')
 cases=sorted(p.name for p in (R/'common_rna_scores_r1/aligned').iterdir() if (p/'COMPLETE.json').exists());assert len(cases)==43
 done={};fail={}
 with ProcessPoolExecutor(max_workers=2) as ex:
  fs={ex.submit(work,k):k for k in cases}
  for f in as_completed(fs):
   try:k,rec=f.result();done[k]=rec
   except Exception as e:fail[fs[f]]=repr(e)
   (O/'PROGRESS.json').write_text(json.dumps({'target_host':'149','completed':len(done),'total':len(cases),'failures':fail},indent=2))
 (O/'COMPLETE.json').write_text(json.dumps({'target_host':'149','status':'COMPLETE' if len(done)==43 and not fail else 'FAILED','cohorts':done,'failures':fail},indent=2))
