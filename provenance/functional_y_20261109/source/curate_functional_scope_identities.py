"""Bind equivalent functional scopes while retaining distinct biological contexts."""
from pathlib import Path
import json,socket,hashlib,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;SRC=R/'canonical_targets_r4';O=R/'canonical_targets_r5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
O.mkdir(exist_ok=False);(O/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'FUNCTIONAL_SCOPE_IDENTITY_CURATION'}))
W='CLA-P-0C25477C8427F283';T='CLA-P-366F48557175A084';WP='CLA-P-F90D2D45B29169FE';RK='CLA-P-7D82EE7CD2C905DA';mapping={W:T}
for p in SRC.iterdir():
 if p.name in ['COMPLETE.json','LAUNCH_PREFLIGHT.json']:continue
 if p.name.endswith(('.tsv','.tsv.gz','.parquet')):
  d=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,sep='\t',keep_default_na=False)
  if p.name=='TARGET_MASTER.tsv':d=d.loc[d.canonical_id.ne(W)].copy()
  for c in d:
   if c in ['canonical_id','source_canonical_id','target_canonical_id','parent_id','child_id','source_id','target_id','left_id','right_id']:d[c]=d[c].replace(mapping)
  if p.name=='ID_REDIRECTS.tsv':d=pd.concat([d,pd.DataFrame([{'previous_internal_id':W,'canonical_id':T,'relation':'equivalent_functional_scope','evidence':'ComPath equivalentTo plus primary descriptions of canonical, PCP and calcium WNT branches'}])],ignore_index=True)
  if p.name=='SOURCE_ALIASES.tsv':d=pd.concat([d,pd.DataFrame([{'canonical_id':T,'alias':'Wnt signaling (WP428)','alias_type':'CROSS_DATABASE_EQUIVALENT_SCOPE','source_database':'WikiPathways'}])],ignore_index=True)
  d=d.drop_duplicates()
  if p.suffix=='.parquet':d.to_parquet(O/p.name,index=False,compression='zstd')
  else:d.to_csv(O/p.name,sep='\t',index=False)
 else:(O/p.name).write_bytes(p.read_bytes())
rows=[{'source_authority':'WikiPathways:WP428','target_authority':'Reactome:R-HSA-195721','source_canonical_id':T,'target_canonical_id':T,'relation':'equivalent_functional_scope','shared_function':'Canonical beta-catenin transcription, PCP cytoskeleton and WNT calcium signaling','scope_difference':'Database-specific reaction and molecular coverage retained as readouts','identity_result':'One canonical WNT signaling identity','evidence':'ComPath equivalentTo and concordant primary biological scope descriptions','source_url':'https://sandbox.wikipathways.org/pathways/WP428.html','target_url':'https://reactome.org/content/detail/R-HSA-195721'}, {'source_authority':'Reactome:R-HSA-198203','target_authority':'WikiPathways:WP4172','source_canonical_id':RK,'target_canonical_id':WP,'relation':'shares_functional_core','shared_function':'PI3K activation and phosphoinositide signaling upstream of AKT','scope_difference':'Reactome: NTRK1/TRKA-IRS branch; WikiPathways: multiple stimuli and downstream cellular programs','identity_result':'Distinct scoped identities linked through a shared functional core','evidence':'Primary Reactome parent NTRK1 and reaction list contrasted with WikiPathways pathway scope','source_url':'https://reactome.org/content/detail/R-HSA-198203','target_url':'https://sandbox.wikipathways.org/pathways/WP4172.html'}]
pd.DataFrame(rows).to_csv(O/'FUNCTIONAL_SCOPE_RELATIONS.tsv',sep='\t',index=False)
m=pd.read_csv(O/'TARGET_MASTER.tsv',sep='\t');bind=pd.read_csv(O/'SOURCE_ID_BINDINGS.tsv',sep='\t');assert m.canonical_id.is_unique;assert set(bind.canonical_id)<=set(m.canonical_id)
assert bind.loc[bind.source_target_id.eq('PATH-WP_Wnt_signaling'),'canonical_id'].tolist()==[T]
q=m.has_scored_readout.astype(str).str.lower().eq('true');d={'target_host':'149','status':'COMPLETE','canonical_identities':len(m),'identities_with_readouts':int(q.sum()),'pathway_roles':int((q&m.roles.str.contains('Pathway')).sum()),'state_roles':int((q&m.roles.str.contains('State')).sum()),'source_bindings':len(bind),'merged_identity':{W:T},'source_registry_sha256':sha(SRC/'COMPLETE.json'),'functional_scope_mapping_sha256':sha(O/'FUNCTIONAL_SCOPE_RELATIONS.tsv'),'files':{p.name:sha(p) for p in O.iterdir() if p.is_file()}};(O/'COMPLETE.json').write_text(json.dumps(d,indent=2));print('CANONICAL_SCOPE_COMPLETE',json.dumps(d),flush=True)
