from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import sys,json,socket,time,subprocess,argparse,hashlib
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'shared_relation_demo_r1';P=R/'source_integration/repo';sys.path[:0]=[str(P),str(R)]
from cc_hhgt.m0_joint_scores import JointScoreBuilder
from materialize_shared_tcga_relations import canon
B=Path('/dell_2/DSC/CancerLncAtlas/inputs/cptac_20261104_r1/bundle_r1');J=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1');SOURCECFG=json.loads((J/'repo/config/cptac_149_sources_20261104.json').read_text());CFG=json.loads((O/'SCORING_ADJUSTMENT_CONFIG.json').read_text());CAT=json.loads(Path('/dsk2/user/dengsc/cla_state_signature_update_20261104_r1/CURRENT_149_CONFIG.json').read_text())['targets'];BIND=pd.read_csv(R/'canonical_targets_r6/SOURCE_ID_BINDINGS.tsv',sep='\t').set_index('source_target_id').canonical_id.to_dict();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encode(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Path):return str(x)
 raise TypeError(type(x).__name__)
def js(p,d):Path(p).write_text(json.dumps(d,indent=2,default=encode))
def contexts():return [(f'train_label_fold{j}',[q for q in [2,3,4] if q!=j],[j]) for j in [2,3,4]]+[('validation',[2,3,4],[1]),('sealed_test',[2,3,4],[0])]
def work(c):
 assert c in CFG['cancer_list'];out=O/'CPTAC_SCORES'/c;out.mkdir(parents=True,exist_ok=True);ad=R/'common_rna_scores_r1/aligned'/('CPTAC_'+c);ca=R/'common_rna_scores_r1/cache'/('CPTAC_'+c)
 if (out/'RNA_STATE_COMPLETE.json').exists():print('SCORE_REUSE',c,flush=True);return
 js(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','cancer':c,'dataset':'CPTAC','task':'source-fitted pathway and State readouts; published samplewise protein and promoter methylation scores','source_split_authority':str(J/'aligned_r1'/c/'PATIENTS.tsv'),'shared_adjustment':'shared_relation_demo_r1/SCORING_ADJUSTMENT_CONFIG.json'})
 for folder in [ad,ca]:
  receipt=json.loads((folder/'COMPLETE.json').read_text())
  for name,h in receipt['files'].items():assert sha(folder/name)==h,name
 e=pd.read_parquet(ad/'SYMBOL_LOGTPM.parquet');e.index.name='patient_id';patients=e.index.tolist();auth=pd.read_csv(J/'aligned_r1'/c/'PATIENTS.tsv',sep='\t');auth=auth.set_index('index').reindex(patients);assert auth.outer_fold.notna().all() and auth.cohort.eq('CPTAC').all();folds=auth.outer_fold.to_numpy(int);defs=json.loads((ca/'DEFINITIONS.json').read_text());ap=out/'apm_inputs';ap.mkdir(exist_ok=True);splits={name:{'train':np.flatnonzero(np.isin(folds,sf)).tolist(),'label':np.flatnonzero(np.isin(folds,lf)).tolist()} for name,sf,lf in contexts()}
 js(ap/'SPLITS.json',{k:v for k,v in splits.items() if k!='sealed_test'});js(ap/'APM_GENES.json',[x['SYMBOL'] for x in defs['APM_Wang']]);e.reset_index().to_csv(ap/'APM_EXPRESSION.tsv.gz',sep='\t',index=False,float_format='%.17g',compression={'method':'gzip','compresslevel':1});rscript=J/'repo/scripts/materialize_m0_joint_apm.R';js(ap/'APM_INPUT.json',{'target_host':'149','expression_sha256':sha(ap/'APM_EXPRESSION.tsv.gz'),'input_complete_sha256':sha(ad/'COMPLETE.json'),'script_sha256':sha(rscript)})
 apm=ap/'apm'
 if not (apm/'COMPLETE.json').exists():
  with (out/'apm.log').open('w') as log:subprocess.run(['/usr/bin/Rscript',str(rscript),str(ap),':'.join(SOURCECFG['R_libraries']),'2'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
 (apm/'sealed_test.tsv').write_bytes((apm/'validation.tsv').read_bytes());rec=json.loads((apm/'validation.json').read_text());rec['split']='sealed_test';js(apm/'sealed_test.json',rec)
 native=pd.read_parquet(B/'STATE_PATIENT.parquet').set_index('patient_id').reindex(index=patients,columns=[d['column'] for d in CAT]);native.columns=['STATE|'+k for k in native]
 b=JointScoreBuilder.__new__(JointScoreBuilder);b.aligned=ad;b.cache=ca;b.apm=apm;b.e=e;b.frames={k:pd.read_parquet(ca/(k+'.parquet')).reindex(patients) for k in ['PATHWAY_RAW','STATE_SSGSEA_RAW','HALLMARK_RAW','STATE_FIXED_RAW','XCELL_RAW']};b.defs=defs;b.xref=json.loads((ca/'XCELL_REFERENCE.json').read_text());b.network=pd.read_csv(ca/'PROGRAM_NETWORK.tsv',sep='\t');b.catalog=CAT;b.native={'CPTAC':native,'TCGA':pd.DataFrame(index=e.index)};b.target_metadata=None;colspec={d['column']:d for d in CAT};progeny=pd.read_parquet(R/'progeny_standard_r1/cohorts'/('CPTAC_'+c)/'RAW_PROGENY.parquet').reindex(patients)
 extra={}
 for layer in ['PROTEIN_PATHWAY','METHYLATION_PATHWAY']:
  f=pd.read_parquet(B/'scores'/layer/(c+'.parquet')).set_index('target_id').T.reindex(patients);f.columns=[layer+'|'+k for k in f];extra[layer]=f
 records=[];validation=None
 for name,sf,lf in contexts():
  dest=out/name;dest.mkdir(exist_ok=True);si=np.asarray(splits[name]['train'],int);yi=np.asarray(splits[name]['label'],int);sp=np.asarray(patients)[si].tolist();lp=np.asarray(patients)[yi].tolist();assert not set(sp)&set(lp);scores,reference=b.fit_apply(si,name);mu=progeny.iloc[si].mean();sd=progeny.iloc[si].std(ddof=1);assert set(progeny.columns)<=set(scores.columns);scores.loc[:,progeny.columns]=(progeny-mu)/sd.mask(sd.le(1e-12));reference['PROGENY_REPLACEMENT']={'method':'OFFICIAL_TOP100','source_patients':sp,'mean':mu.to_dict(),'sd':sd.to_dict(),'raw_sha256':sha(R/'progeny_standard_r1/cohorts'/('CPTAC_'+c)/'RAW_PROGENY.parquet')}
  js(dest/'SPLIT.json',{'target_host':'149','source_folds':sf,'label_folds':lf,'source_patients':sp,'label_patients':lp,'outer_fold':0,'minimum_pairs':12});js(dest/'SCORE_REFERENCE.json',reference)
  if name=='validation':validation=scores.copy()
  if name=='sealed_test':assert scores.equals(validation),'Score reference changed for outer heldout patients'
  for layer,frame in [('RNA',scores[[k for k in scores if k.startswith('RNA_PATHWAY|')]]),('STATE',scores[[k for k in scores if k.startswith('STATE')]]),('PROTEIN',extra['PROTEIN_PATHWAY']),('METHYLATION',extra['METHYLATION_PATHWAY'])]:
   folder=dest/layer;folder.mkdir(exist_ok=True);metadata=[]
   for col in frame:
    key=col.split('|',1)[1]
    if layer=='STATE':
     d=colspec[key];sid=d['id'];family=d['family'];measurement=d['measurement'];method='SOURCE_FITTED_COMMON_RNA_DEFINITION' if not col.startswith('STATE_NATIVE') else 'PUBLISHED_CPTAC_ENDPOINT'
     if sid=='ABSOLUTE_purity':method='PUBLISHED_WES_PURITY_FROM_PHENOTYPE;legacy_compatibility_column_name'
     if sid.startswith('PROGENY:'):method='OFFICIAL_PROGENY_TOP100_SOURCE_REFERENCE'
    else:sid=key;family='pathway';measurement={'RNA':'RNA_EXPRESSION','PROTEIN':'PROTEIN','METHYLATION':'DNA_METHYLATION'}[layer];method={'RNA':'GSVA_ssGSEA_ALPHA025_COMMON_PC_BACKGROUND_SOURCE_RANGE','PROTEIN':'GSVA_ssGSEA_ALPHA025_SAMPLEWISE_RAW_ES','METHYLATION':'OBSERVED_PROMOTER_GENE_MEAN_BETA'}[layer]
    metadata.append({'target_id':col,'source_column':key,'source_target_id':sid,'canonical_id':canon(BIND[sid]),'family':family,'measurement_layer':measurement,'method':method,'native_reference':col.startswith('STATE_NATIVE'),'role':'primary','dataset':'CPTAC'})
   md=pd.DataFrame(metadata);frame.to_parquet(folder/'PATIENT_S.parquet',compression='zstd');md.to_csv(folder/'TARGETS.tsv',sep='\t',index=False);js(folder/'SCORES_COMPLETE.json',{'target_host':'149','dataset':'CPTAC','cancer_id':c,'context':name,'layer':layer,'targets':len(md),'patients':len(patients),'finite_values':int(np.isfinite(frame.to_numpy(float)).sum()),'source_patients':len(sp),'label_patients':len(lp),'files':{f.name:sha(f) for f in folder.iterdir() if f.name!='SCORES_COMPLETE.json'}})
  records.append({'context':name,'source':len(si),'label':len(yi),'RNA_targets':sum(k.startswith('RNA_PATHWAY|') for k in scores),'STATE_targets':sum(k.startswith('STATE') for k in scores)});print('CPTAC_SCORE_CONTEXT',c,records[-1],flush=True)
 js(out/'AXES.json',{'patient_ids':patients,'lncrna_ids':json.loads((R/'full_lnc_local_cnv_r1/lncrna_ids.json').read_text())});js(out/'RNA_STATE_COMPLETE.json',{'target_host':'149','status':'COMPLETE','cancer_id':c,'contexts':records,'frozen_validation_test_reference_identical':True,'PHOSPHO':'separate source-fitted site median recalculation required'})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--cancer',choices=CFG['cancer_list'],required=True);args=a.parse_args();work(args.cancer)
