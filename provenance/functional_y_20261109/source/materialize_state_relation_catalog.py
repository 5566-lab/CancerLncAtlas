"""Frozen-reference State S and target-adjusted relation evidence on server 149."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import os,sys,json,socket,time,subprocess,traceback
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'state_relation_evidence_catalog_r1';P=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/repo')
sys.path[:0]=[str(P),str(P/'scripts')]
from cla_engine.common import sha
from cla_engine.statistics import design
from cc_hhgt.m0_joint_scores import JointScoreBuilder
from cc_hhgt.v32.adjustment_policy import resolve_policy
from block_associations import associate,bh
CFG=Path('/dsk2/user/dengsc/cla_state_signature_update_20261104_r1/CURRENT_149_CONFIG.json')
COV=Path('/dell_2/DSC/CancerLncAtlas/inputs/v32_full_pyg_r6_2/covariates_merged/covariates_r62_20260908.parquet')
AUTH=Path('/dell_2/DSC/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv')
def encode(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,Path):return str(v)
 raise TypeError(type(v).__name__)
def js(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,default=encode))
def contexts():return [(f'train_label_fold{j}',[q for q in [2,3,4] if q!=j],[j]) for j in [2,3,4]]+[('validation',[2,3,4],[1]),('sealed_test',[2,3,4],[0])]
def work(cancer):
 t=time.monotonic();out=O/cancer;out.mkdir();a=pd.read_csv(AUTH,sep='\t');a=a.loc[a.cancer_id.eq(cancer)];patients=a.patient_id.tolist();folds=a.patient_fold_id.to_numpy(int)
 ad=R/'common_rna_scores_r1/aligned'/('TCGA_'+cancer);ca=R/'common_rna_scores_r1/cache'/('TCGA_'+cancer)
 for folder in [ad,ca]:
  receipt=json.loads((folder/'COMPLETE.json').read_text())
  for name,h in receipt['files'].items():assert sha(folder/name)==h,name
 cfg=json.loads(CFG.read_text());cov=pd.read_parquet(COV);e=pd.read_parquet(ad/'SYMBOL_LOGTPM.parquet').reindex(patients);e.index.name='patient_id';assert not e.isna().any().any()
 defs=json.loads((ca/'DEFINITIONS.json').read_text());ap=out/'apm_inputs';ap.mkdir();spec={}
 for name,sf,lf in contexts():spec[name]={'train':np.flatnonzero(np.isin(folds,sf)).tolist(),'label':np.flatnonzero(np.isin(folds,lf)).tolist()}
 # validation/test share exactly the same fitted score reference.
 apmspec={k:v for k,v in spec.items() if k!='sealed_test'}
 js(ap/'SPLITS.json',apmspec);js(ap/'APM_GENES.json',[x['SYMBOL'] for x in defs['APM_Wang']]);e.reset_index().to_csv(ap/'APM_EXPRESSION.tsv.gz',sep='\t',index=False,float_format='%.17g',compression={'method':'gzip','compresslevel':1})
 js(ap/'APM_INPUT.json',{'target_host':'149','expression_sha256':sha(ap/'APM_EXPRESSION.tsv.gz'),'input_complete_sha256':sha(ad/'COMPLETE.json'),'script_sha256':sha(P/'scripts/materialize_m0_joint_apm.R')})
 libs=json.loads((P/'config/cptac_149_sources_20261104.json').read_text())['R_libraries'];cmd=['/usr/bin/Rscript',str(P/'scripts/materialize_m0_joint_apm.R'),str(ap),':'.join(libs),'2']
 print('STATE_APM_STARTED',cancer,flush=True)
 with (out/'apm.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=3600)
 apm=ap/'apm';(apm/'sealed_test.tsv').write_bytes((apm/'validation.tsv').read_bytes());rec=json.loads((apm/'validation.json').read_text());rec['split']='sealed_test';js(apm/'sealed_test.json',rec)
 nativepath=Path(cfg['sources']['state']['path']);assert sha(nativepath)==cfg['sources']['state']['sha256'];native=pd.read_parquet(nativepath).set_index('patient_id').reindex(patients);native=native[[d['column'] for d in cfg['targets']]].rename(columns=lambda c:'STATE|'+c)
 # Reuse the existing score builder, supplying the verified cohort-specific inputs.
 b=JointScoreBuilder.__new__(JointScoreBuilder);b.aligned=ad;b.cache=ca;b.apm=apm;b.e=e;b.frames={k:pd.read_parquet(ca/(k+'.parquet')).reindex(patients) for k in ['PATHWAY_RAW','STATE_SSGSEA_RAW','HALLMARK_RAW','STATE_FIXED_RAW','XCELL_RAW']};b.defs=defs;b.xref=json.loads((ca/'XCELL_REFERENCE.json').read_text());b.network=pd.read_csv(ca/'PROGRAM_NETWORK.tsv',sep='\t');b.catalog=cfg['targets'];b.native={'TCGA':native,'CPTAC':pd.DataFrame(index=e.index)};b.target_metadata=None
 x=pd.read_parquet(ad/'LNC_LOGTPM_FULL.parquet').reindex(patients);lncs=x.columns.tolist();X=x.to_numpy(float);cn=R/'full_lnc_local_cnv_r1'/cancer;cp=json.loads((cn/'patient_ids.json').read_text());cl=json.loads((cn/'lncrna_ids.json').read_text());pi=np.array([cp.index(p) for p in patients]);li=np.array([cl.index(l) for l in lncs]);C=np.load(cn/'lncrna_signed_mean.npy',mmap_mode='r')[np.ix_(pi,li)].astype(float);call=np.load(cn/'lncrna_callable.npy',mmap_mode='r')[np.ix_(pi,li)].astype(bool)
 colspec={d['column']:d for d in cfg['targets']};reg=pd.read_csv(R/'canonical_targets_r2/SOURCE_ID_BINDINGS.tsv',sep='\t');idmap=reg.set_index('source_target_id').canonical_id.to_dict();summary=[];frozen=None
 for name,sf,lf in contexts():
  st=time.monotonic();dest=out/name;dest.mkdir();si=np.array(spec[name]['train']);yi=np.array(spec[name]['label']);sp=[patients[i] for i in si];lp=[patients[i] for i in yi];assert not set(sp)&set(lp)
  scores,state=b.fit_apply(si,name);scores=scores[[c for c in scores if c.startswith('STATE')]];scores.to_parquet(dest/'PATIENT_S.parquet',compression='zstd');js(dest/'SCORE_REFERENCE.json',state)
  if name=='validation':frozen=scores
  if name=='sealed_test':assert scores.equals(frozen),'Held-out patients changed score reference'
  definitions=[]
  for col in scores:
   orig=col.split('|',1)[1];d=colspec[orig];definitions.append({'target_id':col,'source_column':orig,'source_target_id':d['id'],'canonical_id':idmap[d['id']],'family':d['family'],'measurement_layer':d['measurement'],'native_reference':col.startswith('STATE_NATIVE'),'role':d['role']})
  md=pd.DataFrame(definitions);assert len(md)==len(cfg['targets']);assert md.source_target_id.is_unique
  md.to_csv(dest/'TARGETS.tsv',sep='\t',index=False);Y=scores.to_numpy(float);shape=(len(lncs),Y.shape[1]);stats={k:np.full(shape,np.nan) for k in ['r','p']};stats.update(n=np.zeros(shape,np.int32),rank=np.zeros(shape,np.int16),available=np.zeros(shape,bool))
  groups={}
  for j,d in enumerate(definitions):
   _,policy=resolve_policy(d['source_target_id'],contract=cfg['adjustment_contract']);key=json.dumps(policy,sort_keys=True);groups.setdefault(key,[]).append(j)
  adjustments=[]
  for jslice in groups.values():
   first=definitions[jslice[0]]
   try:Z,report=design(cov,patients,sp,family=first['family'],target_id=first['source_target_id'],cfg=cfg,admission_cohorts={'source':sp,'label':lp})
   except Exception as err:
    adjustments.append({'targets':[definitions[j]['source_target_id'] for j in jslice],'error':str(err)});continue
   sub=associate(X,Y[:,jslice],C,call,Z,yi,min_n=cfg['minimum_pairs'],compute_fdr=False)
   for k in stats:stats[k][:,jslice]=sub[k]
   adjustments.append({'targets':[definitions[j]['source_target_id'] for j in jslice],'report':report})
  stats['q']=bh(stats['p']);strict=stats['available']&(np.abs(stats['r'])>=.20)&(stats['q']<=.05);weak=stats['available']&(np.abs(stats['r'])>=.15)&(stats['q']<=.10)&~strict
  np.savez_compressed(dest/'ASSOCIATIONS.npz',**stats,strict_statistical_support=strict,weak_statistical_support=weak,historical_state_rho_support=stats['available']&(np.abs(stats['r'])>=.20))
  ix,jx=np.where(strict|weak);pd.DataFrame({'lncrna_id':np.asarray(lncs)[ix],'canonical_id':md.canonical_id.to_numpy()[jx],'source_target_id':md.source_target_id.to_numpy()[jx],'rho':stats['r'][ix,jx],'p':stats['p'][ix,jx],'q':stats['q'][ix,jx],'n':stats['n'][ix,jx],'statistical_support':np.where(strict[ix,jx],'strict','weak'),'evidence_layer':md.measurement_layer.to_numpy()[jx]}).to_parquet(dest/'SUPPORTED_RELATIONS.parquet',compression='zstd')
  counts=md.copy();counts['eligible_lncrna_pairs']=stats['available'].sum(0);counts['strict_statistical_support']=strict.sum(0);counts['weak_statistical_support']=weak.sum(0);counts['patients_with_score']=scores.notna().sum().to_numpy();counts.to_csv(dest/'TARGET_COUNTS.tsv',sep='\t',index=False)
  js(dest/'ADJUSTMENT.json',adjustments);js(dest/'SPLIT.json',{'target_host':'149','outer_fold':0,'source_folds':sf,'label_folds':lf,'source_patients':sp,'label_patients':lp,'minimum_pairs':cfg['minimum_pairs'],'fdr_family':'ALL_LNCRNAS_X_ALL_STATE_TARGETS_WITH_FINITE_TESTS'})
  row={'cancer_id':cancer,'context':name,'patients_source':len(si),'patients_label':len(yi),'state_targets':len(md),'eligible_relations':int(stats['available'].sum()),'strict_statistical_support':int(strict.sum()),'weak_statistical_support':int(weak.sum()),'adjustment_groups_rejected':sum('error' in a for a in adjustments),'seconds':time.monotonic()-st};js(dest/'COMPLETE.json',dict(row,target_host='149',files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}));summary.append(row);pd.DataFrame(summary).to_csv(out/'CONTEXT_SUMMARY.tsv',sep='\t',index=False);print('STATE_CONTEXT_COMPLETE',row,flush=True)
 js(out/'AXES.json',{'patient_ids':patients,'lncrna_ids':lncs});js(out/'COMPLETE.json',{'target_host':'149','cancer':cancer,'contexts':summary,'seconds':time.monotonic()-t,'score_builder_sha256':sha(P/'cc_hhgt/m0_joint_scores.py')});return cancer,summary
if __name__=='__main__':
 import shutil
 O.mkdir(exist_ok=False);a=pd.read_csv(AUTH,sep='\t');cancers=sorted(a.cancer_id.unique());demo=R/'state_relation_evidence_demo_r1'
 js(O/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'STATE'+'_ALL_CANCER_SCORE_AND_RELATION_MATERIALIZATION','cancers':cancers,'outer_fold':0,'training_started':False,'minimum_free_disk_gib':150})
 assert shutil.disk_usage(O).free/2**30>150
 if 'STATE'=='RPPA':
  v=json.loads((R/'RPPA_SCORING_VALIDATION.json').read_text());assert v['status']=='PASS'
  for p in demo.iterdir():
   if p.is_file() and p.name not in ['COMPLETE.json','PROGRESS.json','LAUNCH_PREFLIGHT.json']:shutil.copy2(p,O/p.name)
 results={};failures={};reused={}
 with ProcessPoolExecutor(max_workers=2) as ex:
  futures={ex.submit(work,c):c for c in cancers if c not in ['BRCA','LUAD','COAD']}
  for f in as_completed(futures):
   c=futures[f]
   try:_,rows=f.result();results[c]=rows
   except Exception as err:failures[c]=repr(err);print('CATALOG_FAILED',c,traceback.format_exc(),flush=True)
   js(O/'PROGRESS.json',{'target_host':'149','completed':sorted(results),'total':len(cancers),'failures':failures})
 deadline=time.monotonic()+3600
 for c in ['BRCA','LUAD','COAD']:
  src=demo/c
  while not (src/'COMPLETE.json').exists() and time.monotonic()<deadline:time.sleep(10)
  if not (src/'COMPLETE.json').exists():failures[c]='DEMO_SOURCE_RECEIPT_UNAVAILABLE';continue
  d=json.loads((src/'COMPLETE.json').read_text());assert d['cancer']==c;(O/c).symlink_to(src,target_is_directory=True);results[c]=d['contexts'];reused[c]={'path':str(src),'complete_sha256':sha(src/'COMPLETE.json')}
 js(O/'REUSED_INPUTS.json',reused);pd.DataFrame([r for rows in results.values() for r in rows]).to_csv(O/'CANCER_CONTEXT_COUNTS.tsv',sep='\t',index=False)
 js(O/'COMPLETE.json',{'target_host':'149','status':'COMPLETE' if not failures else 'FAILED','results':results,'failures':failures})
