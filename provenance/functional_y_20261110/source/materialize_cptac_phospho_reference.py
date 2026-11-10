from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import socket,json,sys,subprocess,hashlib,argparse,shutil,time
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'shared_relation_demo_r1';P=R/'source_integration/repo';sys.path.insert(0,str(P))
from cla_engine.cptac_bundle import read_gct,ptm_observed_mask
from cla_engine.cptac_omics_prepare import gmt
from materialize_shared_tcga_relations import canon
B=Path('/dell_2/DSC/CancerLncAtlas/inputs/cptac_20261104_r1');CFG=json.loads((P/'config/cptac_149_sources_20261104.json').read_text());BIND=pd.read_csv(R/'canonical_targets_r6/SOURCE_ID_BINDINGS.tsv',sep='\t').set_index('source_target_id').canonical_id.to_dict();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();js=lambda p,d:Path(p).write_text(json.dumps(d,indent=2));CONTEXTS=['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test']
def work(c):
 assert c in ['BRCA','LUAD','COAD'];out=O/'CPTAC_SCORES'/c;assert (out/'RNA_STATE_COMPLETE.json').exists();patients=json.loads((out/'AXES.json').read_text())['patient_ids'];src=B/'omics_inputs_r1'/c;centered=read_gct(src/'PHOSPHO_SITE_CENTERED.gct');med=pd.read_csv(src/'PHOSPHO_REFERENCE_MEDIANS.tsv',sep='\t',index_col=0).iloc[:,0];assert med.index.is_unique;raw=centered.add(med.reindex(centered.index),axis=0);wanted={x+'::T' for x in patients};raw=raw.loc[:,[x for x in raw.columns if x in wanted]];assert raw.columns.is_unique;sets=gmt(CFG['references']['ptm_gmt']['path']);prod=O/'CPTAC_SCORES/ptm_producer';prod.mkdir(exist_ok=True)
 for name in ['score_cptac_ptm.R','cptac_ptm_compat.R']:
  text=(B/'bundle_r1/PRODUCERS'/name).read_text()
  if name=='score_cptac_ptm.R':text=text.replace('frozen_tumor_cohort_site_median','frozen_source_patient_site_median').replace('relative_to_CPTAC_cohort','relative_to_CPTAC_source_patients')
  target=prod/name
  if not target.exists() or target.read_text()!=text:
   tmp=prod/(name+'.'+str(os.getpid())+'.tmp');tmp.write_text(text);tmp.replace(target)
 js(out/'PHOSPHO_PREFLIGHT.json',{'target_host':'149','cancer':c,'operation':'PTM_SEA_RECOMPUTE_WITH_SOURCE_PATIENT_SITE_MEDIANS','source_centered_sha256':sha(src/'PHOSPHO_SITE_CENTERED.gct'),'original_site_medians_sha256':sha(src/'PHOSPHO_REFERENCE_MEDIANS.tsv'),'site_raw_reconstruction':'stored centered + stored original median; missingness preserved','per_context_source_fit':True})
 for name in CONTEXTS:
  dest=out/name/'PHOSPHO';dest.mkdir(exist_ok=True);split=json.loads((out/name/'SPLIT.json').read_text());sp=split['source_patients'];lp=split['label_patients'];assert not set(sp)&set(lp)
  if (dest/'SCORES_COMPLETE.json').exists():
   receipt=json.loads((dest/'SCORES_COMPLETE.json').read_text())
   for filename,digest in receipt['files'].items():assert sha(dest/filename)==digest
   print('PHOSPHO_SCORE_REUSE',c,name,flush=True);continue
  if name=='sealed_test':
   other=out/'validation/PHOSPHO';assert json.loads((out/'validation/SPLIT.json').read_text())['source_patients']==sp
   for filename in ['PATIENT_S.parquet','TARGETS.tsv','REFERENCE.json']:(dest/filename).write_bytes((other/filename).read_bytes())
   row=json.loads((other/'SCORES_COMPLETE.json').read_text());row['context']=name;row['reference_reused_from']='validation';row['files']={f.name:sha(f) for f in dest.iterdir() if f.is_file() and f.name!='SCORES_COMPLETE.json'};js(dest/'SCORES_COMPLETE.json',row);continue
  tick=time.monotonic();train=[x+'::T' for x in sp if x+'::T' in raw.columns];assert len(train)>=3;mu=raw[train].median(axis=1,skipna=True);data=raw.sub(mu,axis=0);data=data.loc[mu.notna()];inputdir=out/name/'phospho_input';inputdir.mkdir(exist_ok=True);inputfile=inputdir/'PHOSPHO_SOURCE_CENTERED.gct';mu.rename('source_patient_median_log2').to_csv(inputdir/'REFERENCE_MEDIANS.tsv',sep='\t',index_label='ptm_sequence_id');expanded=data.copy();expanded.insert(0,'Description','NA')
  with inputfile.open('w') as f:f.write('#1.2\n'+str(len(data))+'\t'+str(data.shape[1])+'\n');expanded.to_csv(f,sep='\t',index_label='Name',na_rep='NaN',float_format='%.17g')
  rout=out/name/'ptmsea';cmd=['/usr/bin/Rscript',str(prod/'score_cptac_ptm.R'),'--input',str(inputfile),'--output',str(rout),'--gmt',CFG['references']['ptm_gmt']['path'],'--broad-source',CFG['broad_source']['path'],'--r-library',':'.join(CFG['R_libraries'])]
  if not (rout/'COMPLETE.json').exists():
   with (out/name/'ptmsea_runner.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
  rc=json.loads((rout/'COMPLETE.json').read_text());assert rc['input_md5']==hashlib.md5(inputfile.read_bytes()).hexdigest();result=read_gct(rout/'PTMSEA-scores.gct');counts,usable=ptm_observed_mask(data,sets);result=result.where(usable.reindex(index=result.index,columns=result.columns));result=result.reindex(index=list(sets));result.columns=[x[:-3] for x in result.columns];assert all(x in patients for x in result.columns);scores=result.T.reindex(patients);scores.columns=['PHOSPHO_PTMSEA|'+x for x in scores];md=pd.DataFrame([{'target_id':'PHOSPHO_PTMSEA|'+sid,'source_column':sid,'source_target_id':sid,'canonical_id':canon(BIND[sid]),'family':'phospho_signature','measurement_layer':'PHOSPHO','method':'BROAD_PTM_SEA_DIRECTIONAL_ES_SOURCE_PATIENT_SITE_MEDIAN','native_reference':False,'role':'primary','dataset':'CPTAC'} for sid in sets]);assert md.target_id.tolist()==scores.columns.tolist();scores.to_parquet(dest/'PATIENT_S.parquet',compression='zstd');md.to_csv(dest/'TARGETS.tsv',sep='\t',index=False);js(dest/'REFERENCE.json',{'method':'per-site source-patient median; PTM-SEA ES','source_patients':sp,'measured_source_samples':train,'input_sha256':sha(inputfile),'reference_medians_sha256':sha(inputdir/'REFERENCE_MEDIANS.tsv'),'raw_source_sha256':sha(src/'PHOSPHO_SITE_CENTERED.gct'),'original_medians_sha256':sha(src/'PHOSPHO_REFERENCE_MEDIANS.tsv'),'source_sites':len(data),'r_method_receipt_sha256':sha(rout/'COMPLETE.json'),'minimum_observed_sites':5,'one_site_direction_mask_preserved':True})
  row={'target_host':'149','dataset':'CPTAC','cancer_id':c,'context':name,'layer':'PHOSPHO','targets':len(md),'patients':len(patients),'finite_values':int(np.isfinite(scores.to_numpy(float)).sum()),'source_patients':len(sp),'label_patients':len(lp),'seconds':time.monotonic()-tick,'files':{f.name:sha(f) for f in dest.iterdir() if f.is_file() and f.name!='SCORES_COMPLETE.json'}};js(dest/'SCORES_COMPLETE.json',row);print('PHOSPHO_SOURCE_SCORES_COMPLETE',c,name,row['finite_values'],round(row['seconds'],1),flush=True)
 js(out/'PHOSPHO_COMPLETE.json',{'target_host':'149','status':'COMPLETE','cancer_id':c,'contexts':CONTEXTS,'validation_test_same_frozen_reference':True})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--cancer',choices=['BRCA','LUAD','COAD'],required=True);args=a.parse_args();work(args.cancer)
