"""Remove direct RNA score-member self associations, then recompute State FDR."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,socket,time,traceback,zipfile
import pandas as pd,numpy as np
import pyarrow.parquet as pq
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'state_relation_evidence_scope_adjusted_r2';P=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/repo');sys.path[:0]=[str(P),'/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps']
from cla_engine.common import sha
from block_associations import bh,associate
from canonical_quantity_adjustment.design import design
from canonical_quantity_adjustment.adjustment_policy import validate_adjustment_contract,default_adjustment_contract
CTX=['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test']
def js(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False))
def save_npz(path,**arrays):
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for name,v in arrays.items():
   with z.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(v),allow_pickle=False)
def prepare():
 O.mkdir(exist_ok=False);js(O/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'CANONICAL_STATE_QUANTITY_ADJUSTMENT','training_started':False})
 cfg=json.loads(Path('/dsk2/user/dengsc/cla_state_signature_update_20261104_r1/CURRENT_149_CONFIG.json').read_text());cfg['adjustment_contract']=default_adjustment_contract(cfg['adjustment_contract']['policies'])
 for name in ['TumorPurity','ABSOLUTE_purity']:
  pol=cfg['adjustment_contract']['policies'][name];pol['excluded']['purity']='TARGET_SAME_BIOLOGICAL_QUANTITY';pol['required']=[x for x in pol['required'] if x!='purity'];pol['purity_dependent_source_families']=[]
 validate_adjustment_contract(cfg['adjustment_contract']);js(O/'SCORING_ADJUSTMENT_CONFIG.json',cfg)
 for name in ['DIRECT_RNA_MEMBERS.tsv','MIXED_CODING_LNC_SYMBOLS.tsv','DEFINITION_RECEIPT.json']:(O/name).symlink_to(R/'state_relation_evidence_direct_excluded_r1'/name)
 js(O/'CANONICAL_SCOPE_POLICY.json',{'target_host':'149','canonical_registry':'canonical_targets_r6','target_quantity':'Tumor cell fraction','readouts':['TumorPurity','ABSOLUTE_purity'],'excluded_covariate':'purity','basis':'Any estimate of the same target quantity is excluded regardless of measurement source','other_adjustments':'Existing R6.2 contract and lncRNA-side signed local CNV'})
def source(c):return R/'state_relation_evidence_direct_excluded_r1'/c
def work(cancer):
 src=source(cancer);out=O/cancer;out.mkdir();axes=json.loads((src/'AXES.json').read_text());lncs=axes['lncrna_ids'];lp={x:i for i,x in enumerate(lncs)};direct=pd.read_csv(O/'DIRECT_RNA_MEMBERS.tsv',sep='\t');rows=[];js(out/'AXES.json',axes)
 patients=axes['patient_ids'];pi={p:i for i,p in enumerate(patients)};x=pd.read_parquet(R/'common_rna_scores_r1/aligned'/('TCGA_'+cancer)/'LNC_LOGTPM_FULL.parquet').reindex(patients);assert x.columns.tolist()==lncs;X=x.to_numpy(float)
 cn=R/'full_lnc_local_cnv_r1'/cancer;cp=pd.Index(json.loads((cn/'patient_ids.json').read_text()));cl=pd.Index(json.loads((cn/'lncrna_ids.json').read_text()));ri=cp.get_indexer(patients);ci=cl.get_indexer(lncs);assert (ri>=0).all() and (ci>=0).all();C=np.load(cn/'lncrna_signed_mean.npy',mmap_mode='r')[np.ix_(ri,ci)].astype(float);call=np.load(cn/'lncrna_callable.npy',mmap_mode='r')[np.ix_(ri,ci)].astype(bool)
 cfg=json.loads((O/'SCORING_ADJUSTMENT_CONFIG.json').read_text());cov=pd.read_parquet('/dell_2/DSC/CancerLncAtlas/inputs/v32_full_pyg_r6_2/covariates_merged/covariates_r62_20260908.parquet')

 for context in CTX:
  inp=src/context;dest=out/context;dest.mkdir();receipt=json.loads((inp/'COMPLETE.json').read_text());assert sha(inp/'ASSOCIATIONS.npz')==receipt['files']['ASSOCIATIONS.npz'];md=pd.read_csv(inp/'TARGETS.tsv',sep='\t');tp={x:j for j,x in enumerate(md.source_target_id)};exclude=direct.loc[direct.state_id.isin(tp)&direct.lncrna_id.isin(lp)].copy();ix=exclude.lncrna_id.map(lp).to_numpy(int);jx=exclude.state_id.map(tp).to_numpy(int)
  raw=np.load(inp/'ASSOCIATIONS.npz');stats={k:raw[k] for k in ['r','p','q','n','rank','available']}
  split=json.loads((inp/'SPLIT.json').read_text());sp=split['source_patients'];lp2=split['label_patients'];yi=np.array([pi[p] for p in lp2]);sel=np.array([tp['TumorPurity'],tp['ABSOLUTE_purity']]);scores=pd.read_parquet(inp/'PATIENT_S.parquet').reindex(patients);ys=scores[md.target_id.iloc[sel]].to_numpy(float)
  adjustments=json.loads((inp/'ADJUSTMENT.json').read_text());adjustments=[dict(g,targets=[t for t in g['targets'] if t not in ['TumorPurity','ABSOLUTE_purity']]) for g in adjustments];adjustments=[g for g in adjustments if g['targets']]
  try:
   Z,report=design(cov,patients,sp,family='purity',target_id='ABSOLUTE_purity',cfg=cfg,admission_cohorts={'source':sp,'label':lp2})
  except Exception as err:
   for k in ['r','p']:stats[k][:,sel]=np.nan
   for k in ['n','rank','available']:stats[k][:,sel]=0
   adjustments.append({'targets':['TumorPurity','ABSOLUTE_purity'],'error':str(err),'canonical_quantity':'Tumor cell fraction'})
  else:
   assert 'purity' not in report['encoded'];sub=associate(X,ys,C,call,Z,yi,min_n=cfg['minimum_pairs'],compute_fdr=False);adjustments.append({'targets':['TumorPurity','ABSOLUTE_purity'],'report':report,'canonical_quantity':'Tumor cell fraction'})
   for k in ['r','p','n','rank','available']:stats[k][:,sel]=sub[k]
  js(dest/'ADJUSTMENT.json',adjustments);previous=int(stats['available'][ix,jx].sum());stats['available'][ix,jx]=False;stats['r'][ix,jx]=np.nan;stats['p'][ix,jx]=np.nan;stats['q']=bh(stats['p']);strict=stats['available']&(np.abs(stats['r'])>=.20)&(stats['q']<=.05);weak=stats['available']&(np.abs(stats['r'])>=.15)&(stats['q']<=.10)&~strict
  assert not np.isfinite(stats['p'][ix,jx]).any();assert not strict[ix,jx].any();assert not weak[ix,jx].any();save_npz(dest/'ASSOCIATIONS.npz',**stats,strict_statistical_support=strict,weak_statistical_support=weak,historical_state_rho_support=stats['available']&(np.abs(stats['r'])>=.20))
  ei,ej=np.where(strict|weak);pd.DataFrame({'lncrna_id':np.asarray(lncs)[ei],'canonical_id':md.canonical_id.to_numpy()[ej],'source_target_id':md.source_target_id.to_numpy()[ej],'rho':stats['r'][ei,ej],'p':stats['p'][ei,ej],'q':stats['q'][ei,ej],'n':stats['n'][ei,ej],'statistical_support':np.where(strict[ei,ej],'strict','weak'),'evidence_layer':md.measurement_layer.to_numpy()[ej]}).to_parquet(dest/'SUPPORTED_RELATIONS.parquet',compression='zstd')
  counts=pd.read_csv(inp/'TARGET_COUNTS.tsv',sep='\t');counts['eligible_lncrna_pairs']=stats['available'].sum(0);counts['strict_statistical_support']=strict.sum(0);counts['weak_statistical_support']=weak.sum(0);counts['direct_rna_members_excluded']=counts.source_target_id.map(exclude.groupby('state_id').size()).fillna(0).astype(int);counts.to_csv(dest/'TARGET_COUNTS.tsv',sep='\t',index=False)
  for name in ['PATIENT_S.parquet','SCORE_REFERENCE.json','TARGETS.tsv','SPLIT.json']:(dest/name).symlink_to(inp/name)
  exclude.to_csv(dest/'DIRECT_RNA_EXCLUSIONS.tsv',sep='\t',index=False);row={'cancer_id':cancer,'context':context,'state_targets':len(md),'direct_definition_pairs':len(exclude),'previously_estimable_self_pairs_removed':previous,'canonical_quantity_adjusted_readouts':2,'eligible_relations':int(stats['available'].sum()),'strict_statistical_support':int(strict.sum()),'weak_statistical_support':int(weak.sum())};js(dest/'COMPLETE.json',dict(row,target_host='149',source_complete_sha256=sha(inp/'COMPLETE.json'),files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}));rows.append(row);print('STATE_SELF_EXCLUSION_COMPLETE',row,flush=True)
 pd.DataFrame(rows).to_csv(out/'CONTEXT_SUMMARY.tsv',sep='\t',index=False);js(out/'COMPLETE.json',{'target_host':'149','cancer':cancer,'contexts':rows});return cancer,rows
if __name__=='__main__':
 prepare();a=pd.read_csv('/dell_2/DSC/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv',sep='\t');pending=set(a.cancer_id.unique());running={};results={};failures={};deadline=time.monotonic()+6*3600
 with ProcessPoolExecutor(max_workers=2) as pool:
  while pending or running:
   for c in sorted(pending,key=lambda x:(x not in ['BRCA','LUAD','COAD'],x)):
    if (source(c)/'COMPLETE.json').exists():running[pool.submit(work,c)]=c;pending.remove(c)
   for f,c in list(running.items()):
    if f.done():
     try:_,rows=f.result();results[c]=rows
     except Exception as err:failures[c]=repr(err);print('STATE_SELF_EXCLUSION_FAILED',c,traceback.format_exc(),flush=True)
     del running[f]
   js(O/'PROGRESS.json',{'target_host':'149','completed':sorted(results),'pending':sorted(pending),'running':sorted(running.values()),'failures':failures})
   if time.monotonic()>deadline:failures['source_wait']='Six-hour deadline';break
   if pending or running:time.sleep(10)
 pd.DataFrame([r for rows in results.values() for r in rows]).to_csv(O/'CANCER_CONTEXT_COUNTS.tsv',sep='\t',index=False);js(O/'COMPLETE.json',{'target_host':'149','status':'COMPLETE' if not failures else 'FAILED','results':results,'failures':failures})
