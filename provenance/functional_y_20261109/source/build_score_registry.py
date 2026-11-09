"""Inventory measured/derived S on host149. Does not turn S into relation Y."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,socket,os
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'score_registry_r1'
assert socket.gethostname()=='149'
OUT.mkdir(exist_ok=True)
assert not (OUT/'COMPLETE.json').exists(), 'Immutable completed output'
(OUT/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'SCORE_IDENTITY_AND_ACTUAL_FINITE_COVERAGE','relation_Y_created':False,'training_started':False},indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
receipts={}
def record(p,expected=None):
 p=Path(p);h=sha(p)
 if expected and h!=expected:raise ValueError('SOURCE_HASH_MISMATCH '+str(p))
 receipts[str(p)]={'sha256':h,'bytes':p.stat().st_size}
 return p
cfg=json.loads(Path('/dsk2/user/dengsc/cla_state_signature_update_20261104_r1/CURRENT_149_CONFIG.json').read_text())
pathway=pd.read_csv(ROOT/'catalog_inventory/PATHWAY_SOURCE_ID_REGISTRY_DRAFT.tsv',sep='\t').fillna('')
state=pd.read_csv(ROOT/'catalog_inventory/STATE_SOURCE_ID_REGISTRY_DRAFT.tsv',sep='\t').fillna('')
pmap=dict(zip(pathway.source_target_id,pathway.canonical_id));smap=dict(zip(state.source_column,state.canonical_id))
def sid(kind,key):return 'CLA-'+kind+'-'+hashlib.sha256(key.encode()).hexdigest()[:16].upper()
extra={};rows=[];covrows=[]
def canonical(t):
 if t in pmap:return pmap[t]
 if t not in extra:
  typ='kinase_substrate_program' if t.startswith('KINASE-') else 'phospho_response_signature' if t.startswith('PATH-') else 'legacy_pathway_definition'
  key=('PTMSIGDB:2.0:' if typ!='legacy_pathway_definition' else 'LEGACY_SOURCE:')+t
  extra[t]={'canonical_id':sid('P',key),'source_target_id':t,'target_type':typ,'authority_id':key,'integration_status':'SOURCE_ID_PRESERVED_REQUIRES_MECHANISM_EQUIVALENCE_REVIEW'}
 return extra[t]['canonical_id']
def add_matrix(matrix,cohort,cancer,layer,method,source,kind='pathway',definitions=None):
 assert matrix.index.is_unique and matrix.columns.is_unique
 vals=matrix.to_numpy(dtype=float);assert not np.isinf(vals).any()
 good=np.isfinite(vals);n=good.sum(axis=0);vary=np.zeros(len(n),bool)
 for j in np.flatnonzero(n>1):vary[j]=np.ptp(vals[good[:,j],j])>0
 for j,target in enumerate(matrix.columns):
  d=(definitions or {}).get(target,{})
  cid=smap.get(target) if kind=='state' else canonical(str(target))
  if cid is None:raise ValueError('UNREGISTERED_STATE '+str(target))
  rows.append(dict(canonical_id=cid,source_target_id=str(target),target_class=kind,cohort=cohort,cancer_id=cancer,measurement_layer=d.get('measurement',layer),method=d.get('method',method),score_source_file=str(source),patients_in_scope=len(matrix),patients_with_finite_score=int(n[j]),patient_coverage_fraction=float(n[j]/len(matrix)) if len(matrix) else 0,nonconstant=bool(vary[j]),at_least_12_observed_and_variable=bool(n[j]>=12 and vary[j]),role=d.get('role','primary'),score_unit=d.get('unit','SOURCE_DEFINED_NO_CROSS_PLATFORM_EQUIVALENCE'),label_status='S_ONLY_REQUIRES_RELATION_ESTIMATION',structural_support_does_not_validate_activity=True))
 covrows.append(dict(cohort=cohort,cancer_id=cancer,layer=layer,method=method,patients_in_scope=len(matrix),patients_any_finite_score=int(good.any(axis=1).sum()),score_definitions=len(matrix.columns),targets_any_finite=int((n>0).sum()),targets_at_least_12_and_variable=int(((n>=12)&vary).sum()),source=str(source)))
patientmap=pd.read_csv(record(cfg['sources']['patient_map']['path'],cfg['sources']['patient_map']['sha256']),sep='\t')
assert not patientmap.patient_id.duplicated().any()
state_defs={d['column']:d for d in cfg['targets']}
sp=record(cfg['sources']['state']['path'],cfg['sources']['state']['sha256']);S=pd.read_parquet(sp)
assert not S.patient_id.duplicated().any()
for cancer,sub in S.groupby('cancer_id',sort=True):add_matrix(sub.set_index('patient_id')[list(smap)],'TCGA',cancer,'STATE_MIXED','SOURCE_DEFINED',sp,'state',state_defs)
print('TCGA_STATE',len(S),len(smap),flush=True)
cp=Path('/dell_2/DSC/CancerLncAtlas/inputs/cptac_20261104_r1/bundle_r1');cm=json.loads(record(cp/'COMPLETE.json').read_text())
scoreindex=pd.read_csv(record(cp/'SCORE_INDEX.tsv'),sep='\t')
for row in scoreindex.itertuples():
 p=record(cp/row.path,row.sha256);M=pd.read_parquet(p).set_index('target_id').T
 add_matrix(M,'CPTAC:'+row.source_cohort,row.cancer_id,row.layer,row.method,p)
 print('CPTAC_SCORE',row.source_cohort,row.layer,flush=True)
sp=record(cp/'STATE_PATIENT.parquet',cm['files']['STATE_PATIENT.parquet']['sha256']);S=pd.read_parquet(sp)
for (cohort,cancer),sub in S.groupby(['source_cohort','cancer_id'],sort=True):
 add_matrix(sub.set_index('patient_id')[[c for c in smap if c in sub]],'CPTAC:'+cohort,cancer,'STATE_MIXED','SOURCE_DEFINED_CPTAC_RECOMPUTATION',sp,'state',state_defs)
 print('CPTAC_STATE',cohort,flush=True)
# Current unaltered TCGA RNA scores, only exact authority patients; no MOGSA sources.
root=Path(cfg['sources']['pathway_activity_root']['path'])
for cancer,auth in patientmap.groupby('cancer_id',sort=True):
 pp=sorted((root/('cancer_id='+cancer)).glob('*.parquet'))
 if not pp:raise ValueError('MISSING_TCGA_SCORES '+cancer)
 pieces=[]
 for p in pp:pieces.append(pd.read_parquet(record(p),columns=['patient_id','pathway_id','activity_score','score_method']))
 d=pd.concat(pieces,ignore_index=True);d=d.loc[d.patient_id.isin(set(auth.patient_id))]
 assert not d.duplicated(['patient_id','pathway_id','score_method']).any()
 for method,part in d.groupby('score_method'):
  M=part.pivot(index='patient_id',columns='pathway_id',values='activity_score').reindex(auth.patient_id)
  add_matrix(M,'TCGA',cancer,'RNA_PATHWAY',str(method),root/('cancer_id='+cancer))
 print('TCGA_RNA',cancer,len(d),flush=True)
# Three-cancer shared definition raw RNA scores are separate readouts, not duplicate identities.
for cancer in ['BRCA','LUAD','COAD']:
 p=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/score_cache_r2')/cancer/'PATHWAY_RAW.parquet'
 M=pd.read_parquet(record(p));M.columns=[str(c).removeprefix('RNA_PATHWAY|') for c in M.columns]
 for cohort in ['TCGA','CPTAC']:
  mask=M.index.astype(str).str.startswith('TCGA-' if cohort=='TCGA' else 'CPTAC:')
  add_matrix(M.loc[mask],cohort if cohort=='TCGA' else 'CPTAC:'+cancer,cancer,'RNA_PATHWAY','SHARED_DEFINITION_RAW_SSGSEA_M0_20261105',p)
R=pd.DataFrame(rows);assert not R.duplicated(['canonical_id','cohort','cancer_id','measurement_layer','method','score_source_file']).any()
R.to_parquet(OUT/'SCORE_REGISTRY.parquet',index=False)
R.to_csv(OUT/'SCORE_REGISTRY.tsv.gz',sep='\t',index=False)
C=pd.DataFrame(covrows);C.to_csv(OUT/'CANCER_SCORE_COVERAGE.tsv',sep='\t',index=False)
pd.DataFrame(extra.values()).to_csv(OUT/'ADDITIONAL_SOURCE_IDENTITIES.tsv',sep='\t',index=False)
(OUT/'INPUT_SHA256.json').write_text(json.dumps(receipts,indent=2))
status={'target_host':'149','status':'COMPLETE_SCORE_INVENTORY_NOT_FINAL_Y','rows':len(R),'source_identity_count':R.canonical_id.nunique(),'cancer_count':R.cancer_id.nunique(),'cohorts':sorted(R.cohort.unique()),'extra_identities':len(extra),'finite_score_definitions':R.loc[R.patients_with_finite_score>0].canonical_id.nunique(),'relation_Y_rows':0,'MOGSA_used':False,'PathFA_used':False,'limitations':['Finite score coverage does not establish independent label validity','No pooling of platform scales','Old TCGA RNA and shared-definition demo RNA remain distinct readouts','CPTAC/TCGA recomputed State numerical contracts require separate checks','No causal activation or final semantic equivalence claimed'],'created_utc':datetime.now(timezone.utc).isoformat()}
status['files']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.is_file() and p.name!='COMPLETE.json'}
(OUT/'COMPLETE.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2),flush=True)
