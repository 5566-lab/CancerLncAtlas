from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
from functools import lru_cache
import socket,hashlib,json,gzip,csv
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'measured_site_evidence_r1';O.mkdir(exist_ok=True)
assert not (O/'COMPLETE.json').exists()
(O/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'SEQUENCE_VERIFIED_SITE_MEASUREMENT_MASKS_AND_REACTION_ROLES','patient_activity_or_relation_Y_fitting':False},indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
source=R/'sequence_validation_r1/CPTAC_REACTOME_SITE_SEQUENCE_VALIDATION.parquet';sites=pd.read_parquet(source);sites=sites.loc[sites.canonical_site_and_flank_verified].copy()
by_state=defaultdict(set)
for state,fid in sites[['protein_state_node','feature_id']].drop_duplicates().itertuples(index=False,name=None):by_state[state].add(fid)
S=R/'structure_reactome98';mem=pd.read_csv(S/'COMPLEX_AND_ENTITY_SET_MEMBERS.tsv.gz',sep='\t').fillna('');ch=defaultdict(list)
for a,b in mem[['container_node','member_node']].itertuples(index=False,name=None):ch[a].append(b)
visiting=set()
@lru_cache(None)
def measured_descendants(node):
 if node in visiting:raise ValueError('CYCLE '+node)
 visiting.add(node);out=set(by_state.get(node,[]))
 for c in ch[node]:out.update(measured_descendants(c))
 visiting.remove(node);return frozenset(out)
roles=pd.read_csv(S/'REACTION_ENTITY_ROLES.tsv.gz',sep='\t').fillna('');links=[]
for row in roles.itertuples():
 for fid in measured_descendants(row.entity_node):links.append(dict(feature_id=fid,reaction_stable_id=row.reaction_stable_id,reaction_node=row.reaction_node,role=row.role,control_type=row.control_type,direct_entity_node=row.entity_node,direct_entity_class=row.entity_class,measured_full_complex=False,measured_full_modified_state=False))
L=pd.DataFrame(links).drop_duplicates();L.to_parquet(O/'SEQUENCE_VERIFIED_FEATURE_REACTION_ROLES.parquet',index=False)
pr=pd.read_csv(S/'PATHWAY_REACTION.tsv.gz',sep='\t');pr=pr[['pathway_stable_id','reaction_stable_id']].drop_duplicates()
F=L[['feature_id','reaction_stable_id']].drop_duplicates().merge(pr,on='reaction_stable_id').drop_duplicates()
F.to_parquet(O/'SEQUENCE_VERIFIED_FEATURE_REACTION_PATHWAY.parquet',index=False)
authority=pd.read_csv(R/'catalog_inventory/REACTOME_PATHWAY_AUTHORITY.tsv',sep='\t');pknown=set(authority.loc[authority.in_existing_score_catalog,'reactome_id'])
a=Path('/dell_2/DSC/CancerLncAtlas/inputs/cptac_20261104_r1/aligned_r4');manifest=json.loads((a/'COMPLETE.json').read_text());patients=pd.read_csv('/dell_2/DSC/CancerLncAtlas/inputs/cptac_20261104_r1/bundle_r1/PATIENTS.tsv',sep='\t')
receipts={str(source):{'sha256':sha(source)}};summ=[];allcov=[];pathcov=[]
for cohort in manifest['cohorts']:
 key='matrices/'+cohort+'/phosphorylation_Tumor.parquet';p=a/key;h=sha(p);assert h==manifest['files'][key]['sha256'];receipts[str(p)]={'sha256':h,'bytes':p.stat().st_size}
 M=pd.read_parquet(p).set_index('feature_id');assert M.index.is_unique
 # Keep the native tumor sample identifiers, never collapse unidentified aliquots.
 present=sites.feature_id.unique();M=M.reindex(present)
 V=np.isfinite(M.to_numpy(float));n=V.sum(axis=1);good=set(M.index[n>0]);observed=M.loc[n>0]
 observed.to_parquet(O/(cohort+'_VERIFIED_SITE_ABUNDANCE.parquet'))
 coverage=pd.DataFrame({'source_cohort':cohort,'feature_id':M.index,'patients_in_native_tumor_matrix':M.shape[1],'patients_with_finite_measurement':n});allcov.append(coverage)
 sub=F.loc[F.feature_id.isin(good)];grp=sub.groupby('pathway_stable_id')
 for pid,df in grp:
  members=sorted(set(df.feature_id));mask=np.isfinite(M.loc[members].to_numpy(float));counts=mask.sum(axis=0)
  pathcov.append(dict(source_cohort=cohort,reactome_id=pid,in_existing_score_catalog=pid in pknown,measured_sequence_verified_sites=len(members),reactions_with_at_least_one_verified_site=df.reaction_stable_id.nunique(),native_tumor_samples=M.shape[1],samples_any_site=int((counts>0).sum()),samples_at_least_3_sites=int((counts>=3).sum()),median_observed_sites=float(np.median(counts)),activity_score_validated=False))
 summ.append(dict(source_cohort=cohort,native_tumor_samples=M.shape[1],sequence_verified_features_any_measured=len(good),samples_with_any_verified_feature=int(V.any(axis=0).sum()),reactions_with_measured_site=sub.reaction_stable_id.nunique(),pathways_with_measured_site=sub.pathway_stable_id.nunique(),existing_score_pathways_with_measured_site=len(set(sub.pathway_stable_id)&pknown)))
 print('MEASURED_SITE_COHORT',json.dumps(summ[-1]),flush=True)
pd.concat(allcov,ignore_index=True).to_csv(O/'VERIFIED_SITE_PATIENT_COVERAGE.tsv',sep='\t',index=False)
pd.DataFrame(pathcov).to_csv(O/'VERIFIED_SITE_PATHWAY_COVERAGE.tsv',sep='\t',index=False)
pd.DataFrame(summ).to_csv(O/'CPTAC_VERIFIED_SITE_SUMMARY.tsv',sep='\t',index=False)
(O/'INPUT_SHA256.json').write_text(json.dumps(receipts,indent=2))
summary={'target_host':'149','status':'COMPLETE_MEASURED_SEQUENCE_VERIFIED_SITE_BRIDGE','site_features':sites.feature_id.nunique(),'reaction_role_links':len(L),'reaction_count':L.reaction_stable_id.nunique(),'cohort_summary':summ,'score_aggregation_applied':False,'missing_filled':False,'relation_Y_created':False,'note':'The abundance of a verified site is an observation; it is not the occupancy of a whole modified protein or the flux of a reaction','created_utc':datetime.now(timezone.utc).isoformat()}
summary['files']={x.name:{'sha256':sha(x),'bytes':x.stat().st_size} for x in O.iterdir() if x.is_file() and x.name!='COMPLETE.json'}
(O/'COMPLETE.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',flush=True)
