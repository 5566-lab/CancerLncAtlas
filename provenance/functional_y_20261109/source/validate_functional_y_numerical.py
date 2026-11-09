"""Verify canonical Y arrays, scope adjustments and scientific count tables."""
from pathlib import Path
import socket,json,time,hashlib,traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;Y=R/'functional_relation_y_catalog_r6';O=R/'functional_y_validation_r2';C=R/'canonical_targets_r6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,d):Path(p).write_text(json.dumps(d,indent=2))
M=pd.read_csv(C/'TARGET_MASTER.tsv',sep='\t');target_ids=M.loc[M.has_scored_readout.astype(str).str.lower().eq('true'),'canonical_id'].tolist();mixed=set('LNC:'+x for x in pd.read_csv(R/'rna_relation_evidence_direct_excluded_r1/MIXED_CODING_LNC_SYMBOLS.tsv',sep='\t').gene_id);pop=np.array([i.bit_count() for i in range(256)],np.uint8);seen={};failures={};deadline=time.monotonic()+8*3600

def check(path):
 rec=json.loads((path/'COMPLETE.json').read_text());axes=json.loads((path/'AXES.json').read_text());lncs=axes['lncrna_ids'];assert len(lncs)==16889 and len(set(lncs))==len(lncs);assert axes['canonical_ids']==target_ids;assert len(target_ids)==3456
 for n,h in rec['files'].items():assert sha(path/n)==h,(path,n)
 with np.load(path/'Y.npz') as a:arrays={k:a[k] for k in a.files}
 for a in arrays.values():assert a.shape==(len(lncs),len(target_ids)) and a.dtype==np.uint8
 m=arrays['membership'];dire=arrays['direction'];obs=arrays['observed_assay_bits'];strict=arrays['strict_assay_bits'];support=arrays['supported_assay_bits'];assert np.isin(m,[0,1,2]).all();assert np.isin(dire,[0,1,2,3]).all();assert not (strict & ~support).any();assert not (support & ~obs).any();assert np.array_equal(m==2,pop[strict]>=2);assert np.array_equal(m>0,support>0);assert np.array_equal(m==0,dire==0)
 ix=np.array([i for i,l in enumerate(lncs) if l in mixed]);assert len(ix)==6
 for a in [obs,strict,support]:assert not (a[ix,:]&1).any(),'Mixed-biotype RNA self contribution retained'
 cnt=pd.read_csv(path/'TARGET_COUNTS.tsv',sep='\t');assert cnt.canonical_id.tolist()==target_ids
 expect={'observable_relations':(obs>0).sum(axis=0),'strong_positive':(m==2).sum(axis=0),'weak_positive':(m==1).sum(axis=0),'unlabeled_observed':((obs>0)&(m==0)).sum(axis=0),'direction_discordant':(dire==3).sum(axis=0)}
 for k,v in expect.items():assert np.array_equal(cnt[k].to_numpy(),v),k;assert int(v.sum())==rec[k],k
 frame=pd.read_parquet(path/'POSITIVE_RELATIONS.parquet');assert len(frame)==int((m>0).sum());li=pd.Index(lncs).get_indexer(frame.lncrna_id);ti=pd.Index(target_ids).get_indexer(frame.canonical_id);assert (li>=0).all() and (ti>=0).all();assert np.array_equal(m[li,ti],np.where(frame.membership.eq('strong_positive'),2,1));assert np.array_equal(dire[li,ti],frame.direction.map({'positive':1,'negative':2,'discordant':3}).to_numpy());assert np.array_equal(obs[li,ti],frame.observed_assay_bits.to_numpy());assert np.array_equal(strict[li,ti],frame.strict_assay_bits.to_numpy());assert not frame.duplicated(['lncrna_id','canonical_id']).any()
 split=json.loads((path/'SPLIT.json').read_text());assert not set(split['source_patients'])&set(split['label_patients'])
 numerical=R/'numerical_relation_evidence_r1'/rec['cancer_id']/rec['context'];TP={v:i for i,v in enumerate(target_ids)};rd=pd.read_csv(C/'ID_REDIRECTS.tsv',sep='\t');red=dict(zip(rd.previous_internal_id,rd.canonical_id))
 def canonical(x):
  while x in red:x=red[x]
  assert x in TP;return x
 def bit(layer):
  t=str(layer).upper()
  if t.startswith('RNA'):return 1
  if 'PROTEIN' in t:return 2
  if 'PHOSPH' in t:return 4
  if 'METHYL' in t:return 8
  if 'COPY_NUMBER' in t:return 16
  if any(k in t for k in ['SNV','SEQUENC','MUTATION']):return 32
  return 64
 expected_obs=np.zeros_like(obs);expected_strict=np.zeros_like(strict);expected_support=np.zeros_like(support);expected_direction=np.zeros_like(dire);removed_total=0
 for layer in ['RNA','STATE','PROTEIN','PHOSPHO']:
  b=numerical/layer;br=json.loads((b/'COMPLETE.json').read_text());assert br['target_host']=='149'
  if br.get('admissible') is False:continue
  for f in ['ASSOCIATIONS.npz','NUMERICAL_EXCLUSIONS.npz','SUPPORTED_RELATIONS.parquet','TARGETS.tsv','PATIENT_S.parquet']:assert sha(b/f)==br['files'][f],f
  md=pd.read_csv(b/'TARGETS.tsv',sep='\t');assert md.source_target_id.is_unique;adj=json.loads((b/'NUMERICAL_ADJUSTMENT.json').read_text());base=Path(adj['base_associations']);assert sha(base)==adj['base_associations_sha256'];assert adj['estimator_sha256']==sha(R/'residual_estimability.py')
  with np.load(b/'ASSOCIATIONS.npz') as aa:st={k:aa[k] for k in ['r','p','q','n','rank','available','strict_statistical_support','weak_statistical_support']}
  with np.load(b/'NUMERICAL_EXCLUSIONS.npz') as cc:code=cc['reason_bits']
  assert code.dtype==np.uint8 and (code<64).all();changed=code>0;removed_total+=int(changed.sum())
  with np.load(base) as old:
   old_av=old['available'];assert np.array_equal(st['available'],old_av&~changed);assert not (changed&~old_av).any()
   for k in ['n','rank']:assert np.array_equal(st[k],old[k]),k
   for k in ['r','p']:
    a=old[k];assert np.array_equal(st[k][~changed],a[~changed],equal_nan=True);assert not np.isfinite(st[k][changed]).any()
  assert np.array_equal(np.isfinite(st['q']),st['available']);ss=st['available']&(np.abs(st['r'])>=.2)&(st['q']<=.05);ws=st['available']&(np.abs(st['r'])>=.15)&(st['q']<=.10)&~ss;assert np.array_equal(ss,st['strict_statistical_support']);assert np.array_equal(ws,st['weak_statistical_support'])
  ev=pd.read_parquet(b/'SUPPORTED_RELATIONS.parquet');assert len(ev)==int((ss|ws).sum());assert not ev.duplicated(['lncrna_id','source_target_id']).any();ei=pd.Index(lncs).get_indexer(ev.lncrna_id);ej=pd.Index(md.source_target_id).get_indexer(ev.source_target_id);assert (ei>=0).all() and (ej>=0).all()
  for k,arr in [('rho','r'),('p','p'),('q','q'),('n','n')]:assert np.array_equal(ev[k].to_numpy(),st[arr][ei,ej],equal_nan=True),k
  assert np.array_equal(ev.statistical_support.eq('strict').to_numpy(),ss[ei,ej])
  for j,row in enumerate(md.itertuples()):
   if layer=='STATE' and row.source_target_id in ['OXPHOS','DNARepair_Hallmark']:continue
   ti=TP[canonical(row.canonical_id)];bv=bit(row.measurement_layer if layer=='STATE' else layer);available=st['available'][:,j];supported=ss[:,j]|ws[:,j];expected_obs[:,ti]|=available.astype(np.uint8)*bv;expected_strict[:,ti]|=ss[:,j].astype(np.uint8)*bv;expected_support[:,ti]|=supported.astype(np.uint8)*bv;expected_direction[:,ti]|=np.where(supported,np.where(st['r'][:,j]>0,1,2),0).astype(np.uint8)
  if layer=='STATE':
   groups=json.loads((b/'ADJUSTMENT.json').read_text());hit=[g for g in groups if set(g['targets'])&{'TumorPurity','ABSOLUTE_purity'}];assert len(hit)==1 and set(hit[0]['targets'])=={'TumorPurity','ABSOLUTE_purity'}
   if 'report' in hit[0]:assert 'purity' not in hit[0]['report']['encoded']
 assert np.array_equal(obs,expected_obs);assert np.array_equal(strict,expected_strict);assert np.array_equal(support,expected_support);assert np.array_equal(dire,expected_direction)
 result={'target_host':'149','status':'PASS','cancer_id':rec['cancer_id'],'context':rec['context'],'canonical_targets':len(target_ids),'lncrnas':len(lncs),'positive_relations':len(frame),'numerically_undefined_pairs_removed':removed_total,'source_complete_sha256':sha(path/'COMPLETE.json'),'checks':['AXIS_IDENTITY','ARRAY_PARQUET_COUNTS','DIRECT_RNA_EXCLUSION','SOURCE_LABEL_DISJOINT','CANONICAL_PURITY_ADJUSTMENT','ALL_ASSAY_EVIDENCE_RECONSTRUCTS_FINAL_Y','NUMERICAL_EXCLUSIONS_PRESERVE_OTHER_EFFECTS','STATISTICAL_THRESHOLDS_AND_SOURCE_LABELS']};js(O/(rec['cancer_id']+'__'+rec['context']+'.json'),result);return result
if __name__=='__main__':
 O.mkdir(exist_ok=False);js(O/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'VERIFY_NUMERICALLY_DEFINED_RELATION_Y','workers':4});a=pd.read_csv('/dell_2/DSC/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv',sep='\t');expected={c+'/'+k for c in a.cancer_id.unique() for k in ['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test']};pending=set(expected);running={}
 with ProcessPoolExecutor(max_workers=4) as pool:
  while pending or running:
   for key in sorted(pending,key=lambda k:(k.split('/')[0] not in ['BRCA','LUAD','COAD'],k)):
    if (Y/key/'COMPLETE.json').exists():running[pool.submit(check,Y/key)]=key;pending.remove(key)
   for f,key in list(running.items()):
    if f.done():
     try:seen[key]=f.result();print('Y_NUMERICAL_VALIDATED',key,flush=True)
     except Exception as e:failures[key]=repr(e);print('Y_NUMERICAL_VALIDATION_FAILED',key,traceback.format_exc(),flush=True)
     del running[f]
   js(O/'PROGRESS.json',{'target_host':'149','validated_contexts':len(seen),'expected_contexts':len(expected),'pending':len(pending),'running_or_queued':len(running),'failures':failures})
   if time.monotonic()>deadline:failures['deadline']='Eight-hour input/validation deadline';break
   if pending or running:time.sleep(10)
 assert set(seen)|set(failures)==expected or 'deadline' in failures
 if (Y/'COMPLETE.json').exists():q=json.loads((Y/'COMPLETE.json').read_text());assert q['status']=='COMPLETE',q.get('failures')
 js(O/'COMPLETE.json',{'target_host':'149','status':'PASS' if not failures and len(seen)==165 else 'FAILED','validated_contexts':len(seen),'failures':failures,'canonical_definition_sha256':sha(C/'COMPLETE.json'),'estimator_sha256':sha(R/'residual_estimability.py')})
