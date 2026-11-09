"""Exclude mixed-biotype self contributions from RNA pathway supervision."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,socket,time,traceback,sys
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'rna_relation_evidence_direct_excluded_r1'
from exclude_state_direct_rna_members import save_npz,js,sha,bh,CTX

def source(c):return R/('rna_relation_evidence_demo_r1' if c in ['BRCA','LUAD','COAD'] else 'rna_relation_evidence_catalog_r1')/c

def work(c):
 src=source(c);out=O/c;out.mkdir();axes=json.loads((src/'AXES.json').read_text());lncs=axes['lncrna_ids'];md=pd.read_csv(src/'TARGETS.tsv',sep='\t');md['source_target_id']=md.target_id;js(out/'AXES.json',axes);(out/'TARGETS.tsv').symlink_to(src/'TARGETS.tsv');mixed=pd.read_csv(O/'MIXED_CODING_LNC_SYMBOLS.tsv',sep='\t');ids=set('LNC:'+x for x in mixed.gene_id);ix=np.array([i for i,x in enumerate(lncs) if x in ids],int);assert len(ix)==len(ids);rows=[]
 for ctx in CTX:
  inp=src/ctx;dest=out/ctx;dest.mkdir()
  if (inp/'CONTRACT_REJECTION.json').exists():
   for fn in ['CONTRACT_REJECTION.json','SPLIT.json']:(dest/fn).symlink_to(inp/fn)
   row={'cancer_id':c,'context':ctx,'admissible':False,'eligible_relations':0,'strict_statistical_support':0,'weak_statistical_support':0};js(dest/'COMPLETE.json',dict(row,target_host='149',files={}));rows.append(row);continue
  receipt=json.loads((inp/'COMPLETE.json').read_text());assert sha(inp/'ASSOCIATIONS.npz')==receipt['files']['ASSOCIATIONS.npz']
  with np.load(inp/'ASSOCIATIONS.npz') as a:stats={k:a[k] for k in ['r','p','q','n','rank','available']}
  old=int(stats['available'][ix,:].sum());stats['available'][ix,:]=False;stats['r'][ix,:]=np.nan;stats['p'][ix,:]=np.nan;stats['q']=bh(stats['p']);strict=stats['available']&(np.abs(stats['r'])>=.20)&(stats['q']<=.05);weak=stats['available']&(np.abs(stats['r'])>=.15)&(stats['q']<=.10)&~strict
  assert not np.isfinite(stats['p'][ix,:]).any();save_npz(dest/'ASSOCIATIONS.npz',**stats,strict_statistical_support=strict,weak_statistical_support=weak)
  ei,ej=np.where(strict|weak);pd.DataFrame({'lncrna_id':np.asarray(lncs)[ei],'canonical_id':md.canonical_id.to_numpy()[ej],'source_target_id':md.source_target_id.to_numpy()[ej],'rho':stats['r'][ei,ej],'p':stats['p'][ei,ej],'q':stats['q'][ei,ej],'n':stats['n'][ei,ej],'statistical_support':np.where(strict[ei,ej],'strict','weak'),'evidence_layer':'RNA'}).to_parquet(dest/'SUPPORTED_RELATIONS.parquet',compression='zstd',index=False)
  counts=pd.read_csv(inp/'TARGET_COUNTS.tsv',sep='\t');counts['eligible_lncrna_pairs']=stats['available'].sum(0);counts['strict_statistical_support']=strict.sum(0);counts['weak_statistical_support']=weak.sum(0);counts['positive_direction']=((strict|weak)&(stats['r']>0)).sum(0);counts['negative_direction']=((strict|weak)&(stats['r']<0)).sum(0);counts['direct_rna_members_excluded']=len(ix);counts.to_csv(dest/'TARGET_COUNTS.tsv',sep='\t',index=False)
  for fn in ['SPLIT.json','ADJUSTMENT.json']:(dest/fn).symlink_to(inp/fn)
  row={'cancer_id':c,'context':ctx,'admissible':True,'lncrnas':len(lncs),'pathway_definitions':len(md),'direct_definition_pairs':len(ix)*len(md),'previously_estimable_self_pairs_removed':old,'eligible_relations':int(stats['available'].sum()),'strict_statistical_support':int(strict.sum()),'weak_statistical_support':int(weak.sum())};js(dest/'COMPLETE.json',dict(row,target_host='149',source_complete_sha256=sha(inp/'COMPLETE.json'),files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}));rows.append(row);print('RNA_DIRECT_EXCLUSION_COMPLETE',row,flush=True)
 pd.DataFrame(rows).to_csv(out/'CONTEXT_SUMMARY.tsv',sep='\t',index=False);js(out/'COMPLETE.json',{'target_host':'149','cancer':c,'contexts':rows});return c,rows

if __name__=='__main__':
 O.mkdir(exist_ok=False);js(O/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'RNA_PATHWAY_DIRECT_CONTRIBUTION_EXCLUSION_AND_FDR','training_started':False})
 ann=pd.read_csv(R/'common_rna_scores_r1/COMMON_GENE_ANNOTATION.tsv',sep='\t');coding=set(ann.loc[ann.gene_biotype.eq('protein_coding'),'reference_symbol']);mixed=ann.loc[ann.gene_biotype.eq('lncRNA')&ann.reference_symbol.isin(coding)];mixed.to_csv(O/'MIXED_CODING_LNC_SYMBOLS.tsv',sep='\t',index=False)
 js(O/'DEFINITION_RECEIPT.json',{'target_host':'149','mixed_biotype_lncrnas':len(mixed),'handling':'Exclude direct RNA background contributors from RNA pathway supervision; preserve lncRNA feature axes and measured non-RNA evidence','common_annotation_sha256':sha(R/'common_rna_scores_r1/COMMON_GENE_ANNOTATION.tsv')})
 auth=pd.read_csv('/dell_2/DSC/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv',sep='\t');pending=set(auth.cancer_id.unique());running={};results={};failures={};deadline=time.monotonic()+6*3600
 with ProcessPoolExecutor(max_workers=2) as pool:
  while pending or running:
   for c in sorted(pending,key=lambda x:(x not in ['BRCA','LUAD','COAD'],x)):
    if (source(c)/'COMPLETE.json').exists():running[pool.submit(work,c)]=c;pending.remove(c)
   for f,c in list(running.items()):
    if f.done():
     try:_,rows=f.result();results[c]=rows
     except Exception as e:failures[c]=repr(e);print('RNA_DIRECT_EXCLUSION_FAILED',c,traceback.format_exc(),flush=True)
     del running[f]
   js(O/'PROGRESS.json',{'target_host':'149','completed':sorted(results),'pending':sorted(pending),'running':sorted(running.values()),'failures':failures})
   if time.monotonic()>deadline:failures['source_wait']='Six-hour deadline';break
   if pending or running:time.sleep(10)
 pd.DataFrame([r for rows in results.values() for r in rows]).to_csv(O/'CANCER_CONTEXT_COUNTS.tsv',sep='\t',index=False);js(O/'COMPLETE.json',{'target_host':'149','status':'COMPLETE' if not failures else 'FAILED','results':results,'failures':failures})
