"""Materialize RNA discovery relationship evidence with the existing R6.2 split roles."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,socket,os,json,hashlib,time,traceback
import pandas as pd,numpy as np
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'rna_relation_evidence_demo_r1';REPO=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/repo');sys.path.insert(0,str(REPO))
from cla_engine.statistics import design
from cla_engine.common import sha
from block_associations import associate
AUTH=Path('/dell_2/DSC/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv')
COV=Path('/dell_2/DSC/CancerLncAtlas/inputs/v32_full_pyg_r6_2/covariates_merged/covariates_r62_20260908.parquet')
CFG=Path('/dsk2/user/dengsc/cla_state_signature_update_20261104_r1/CURRENT_149_CONFIG.json')
def js(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False))
def work(cancer):
 out=O/cancer;out.mkdir();src=R/'common_rna_scores_r1';key='TCGA_'+cancer;ad=src/'aligned'/key;sc=src/'cache'/key
 for folder in [ad,sc]:
  d=json.loads((folder/'COMPLETE.json').read_text())
  for n,h in d['files'].items():assert sha(folder/n)==h,n
 valid=json.loads((R/'BLOCK_ASSOCIATION_VALIDATION.json').read_text());assert valid['status']=='PASS';assert valid['implementation_sha256']==sha(R/'block_associations.py')
 a=pd.read_csv(AUTH,sep='\t');a=a.loc[a.cancer_id.eq(cancer)].reset_index(drop=True);patients=a.patient_id.tolist();folds=a.patient_fold_id.to_numpy(int)
 x=pd.read_parquet(ad/'LNC_LOGTPM_FULL.parquet').reindex(patients);s=pd.read_parquet(sc/'PATHWAY_RAW.parquet').reindex(patients)
 assert not x.isna().all(1).any();lncs=x.columns.tolist();targets=s.columns.tolist();X=x.to_numpy(float);Y=s.to_numpy(float)
 cn=R/'full_lnc_local_cnv_r1'/cancer;cp=json.loads((cn/'patient_ids.json').read_text());cl=json.loads((cn/'lncrna_ids.json').read_text());pi=np.array([cp.index(p) for p in patients]);li=np.array([cl.index(l) for l in lncs]);C=np.load(cn/'lncrna_signed_mean.npy',mmap_mode='r')[np.ix_(pi,li)].astype(float);call=np.load(cn/'lncrna_callable.npy',mmap_mode='r')[np.ix_(pi,li)].astype(bool)
 cfg=json.loads(CFG.read_text());cov=pd.read_parquet(COV)
 js(out/'AXES.json',{'patient_ids':patients,'lncrna_ids':lncs,'target_ids':targets});pd.DataFrame({'target_id':targets,'canonical_id':[None]*len(targets)}).to_csv(out/'TARGETS_INITIAL.tsv',sep='\t',index=False)
 pmap=pd.read_csv(R/'catalog_inventory/PATHWAY_SOURCE_ID_REGISTRY_DRAFT.tsv',sep='\t').set_index('source_target_id').canonical_id.to_dict();canon=[pmap[t.removeprefix('RNA_PATHWAY|')] for t in targets];pd.DataFrame({'target_id':targets,'canonical_id':canon}).to_csv(out/'TARGETS.tsv',sep='\t',index=False)
 contexts=[(f'train_label_fold{j}',[q for q in [2,3,4] if q!=j],[j]) for j in [2,3,4]]+[('validation',[2,3,4],[1]),('sealed_test',[2,3,4],[0])]
 rows=[]
 for name,source_folds,label_folds in contexts:
  start=time.monotonic();dest=out/name;dest.mkdir();si=np.flatnonzero(np.isin(folds,source_folds));yi=np.flatnonzero(np.isin(folds,label_folds));sp=[patients[i] for i in si];lp=[patients[i] for i in yi];assert not set(sp)&set(lp)
  spec={'outer_fold':0,'source_folds':source_folds,'label_folds':label_folds,'source_patients':sp,'label_patients':lp,'label_patient_role':name,'patient_score':'FIXED_BACKGROUND_RNA_SSGSEA_RAW','fdr_family':'ALL_LNCRNAS_X_ALL_PATHWAY_TARGETS_WITH_FINITE_TESTS','lncrna_admission':'ALL_16889_MEASURED_RNA_ANNOTATIONS','target_host':'149'};js(dest/'SPLIT.json',spec)
  try:Z,report=design(cov,patients,sp,family='pathway',cfg=cfg,admission_cohorts={'source':sp,'label':lp})
  except Exception as e:
   js(dest/'CONTRACT_REJECTION.json',{'error':str(e),'target_host':'149'});rows.append({'cancer_id':cancer,'context':name,'admissible':False,'error':str(e)});print('CONTRACT_REJECTION',cancer,name,str(e),flush=True);continue
  js(dest/'ADJUSTMENT.json',report)
  st=associate(X,Y,C,call,Z,yi,min_n=cfg['minimum_pairs']);ar=np.abs(st['r']);strict=st['available']&(ar>=.20)&(st['q']<=.05);weak=st['available']&(ar>=.15)&(st['q']<=.10)&~strict
  np.savez_compressed(dest/'ASSOCIATIONS.npz',**{k:st[k] for k in ['r','p','q','n','rank','available']},strict_statistical_support=strict,weak_statistical_support=weak)
  counts=pd.DataFrame({'canonical_id':canon,'source_target_id':targets,'eligible_lncrna_pairs':st['available'].sum(0),'strict_statistical_support':strict.sum(0),'weak_statistical_support':weak.sum(0),'positive_direction':((strict|weak)&(st['r']>0)).sum(0),'negative_direction':((strict|weak)&(st['r']<0)).sum(0)})
  counts.to_csv(dest/'TARGET_COUNTS.tsv',sep='\t',index=False)
  ix,jx=np.where(strict|weak);pd.DataFrame({'lncrna_id':np.asarray(lncs)[ix],'canonical_id':np.asarray(canon)[jx],'rho':st['r'][ix,jx],'p':st['p'][ix,jx],'q':st['q'][ix,jx],'n':st['n'][ix,jx],'statistical_support':np.where(strict[ix,jx],'strict','weak'),'evidence_layer':'RNA'}).to_parquet(dest/'SUPPORTED_RELATIONS.parquet',compression='zstd')
  row={'cancer_id':cancer,'context':name,'admissible':True,'patients_source':len(si),'patients_label':len(yi),'lncrnas':len(lncs),'pathway_definitions':len(targets),'eligible_relations':int(st['available'].sum()),'strict_statistical_support':int(strict.sum()),'weak_statistical_support':int(weak.sum()),'mask_groups':st['mask_groups'],'seconds':time.monotonic()-start}
  js(dest/'COMPLETE.json',dict(row,target_host='149',files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}));rows.append(row);pd.DataFrame(rows).to_csv(out/'CONTEXT_SUMMARY.tsv',sep='\t',index=False);print('RELATION_CONTEXT_COMPLETE',row,flush=True)
 js(out/'COMPLETE.json',{'target_host':'149','cancer':cancer,'contexts':rows,'source_hashes':{str(p):sha(p) for p in [AUTH,COV,CFG,ad/'COMPLETE.json',sc/'COMPLETE.json',cn/'COMPLETE.json',R/'block_associations.py']}})
 return cancer,rows
if __name__=='__main__':
 O.mkdir(exist_ok=False);js(O/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'R62_RNA_RELATION_EVIDENCE','cancers':['BRCA','LUAD','COAD'],'outer_fold':0,'training_started':False,'independent_molecular_validation':'Additional measured protein/phospho evidence is integrated separately'})
 pending={'BRCA','LUAD','COAD'};running={};results={};failures={};deadline=time.monotonic()+3*3600
 with ProcessPoolExecutor(max_workers=2) as pool:
  while pending or running:
   for c in list(pending):
    if (R/'common_rna_scores_r1/cache'/('TCGA_'+c)/'COMPLETE.json').exists():running[pool.submit(work,c)]=c;pending.remove(c);print('RELATION_STARTED',c,flush=True)
   for f,c in list(running.items()):
    if f.done():
     try:_,rows=f.result();results[c]=rows
     except Exception as e:failures[c]=repr(e);print('RELATION_FAILED',c,traceback.format_exc(),flush=True)
     del running[f]
   js(O/'PROGRESS.json',{'target_host':'149','pending':sorted(pending),'running':list(running.values()),'complete':results,'failures':failures})
   if time.monotonic()>deadline:
    failures['source_wait']='Three-hour input deadline reached';break
   if pending or running:time.sleep(5)
 js(O/'COMPLETE.json',{'target_host':'149','status':'COMPLETE' if not failures else 'FAILED','results':results,'failures':failures})
