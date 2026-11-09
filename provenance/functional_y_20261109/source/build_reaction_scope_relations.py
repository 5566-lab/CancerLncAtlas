from pathlib import Path
from collections import defaultdict
from itertools import combinations
import socket,json,hashlib,time
import pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;C=R/'canonical_targets_r6';O=R/'reaction_scope_relations_r1';O.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
js(O/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'EXACT_REACTION_SCOPE_RELATIONS','source_version':'Reactome98','identity_basis':'Explicit BioPAX reaction IDs with entity states and compartments','canonical_registry_sha256':sha(C/'COMPLETE.json')})
rd=pd.read_csv(C/'ID_REDIRECTS.tsv',sep='\t');red=dict(zip(rd.previous_internal_id,rd.canonical_id))
def canon(x):
 seen=set()
 while x in red:
  assert x not in seen;seen.add(x);x=red[x]
 return x
M=pd.read_csv(C/'TARGET_MASTER.tsv',sep='\t');scored=set(M.loc[M.has_scored_readout.astype(str).str.lower().eq('true'),'canonical_id']);names=dict(zip(M.canonical_id,M.display_name))
a=pd.read_csv(R/'catalog_inventory/REACTOME_PATHWAY_AUTHORITY.tsv',sep='\t');a.canonical_id=a.canonical_id.map(canon);a['is_score_target']=a.canonical_id.isin(scored);cols=['canonical_id','reactome_id','name','authority_version','is_score_target'];a[cols].drop_duplicates().to_csv(O/'STRUCTURE_NODES.tsv',sep='\t',index=False)
d=pd.read_csv(C/'PATHWAY_REACTION_COMPOSITION.tsv.gz',sep='\t');d.canonical_id=d.canonical_id.map(canon);d=d.loc[d.canonical_id.isin(scored)];sets={cid:set(df.reaction_stable_id) for cid,df in d.groupby('canonical_id')};rnames=d.drop_duplicates('reaction_stable_id').set_index('reaction_stable_id').reaction_name.to_dict()
h=pd.read_csv(R/'catalog_inventory/REACTOME_PARENT_CHILD.tsv',sep='\t');parents=defaultdict(set)
for row in h.itertuples():parents[canon(row.child_id)].add(canon(row.parent_id))
def ancestors(cid):
 seen=set();stack=list(parents[cid])
 while stack:
  x=stack.pop()
  if x in seen:continue
  assert x!=cid,'Hierarchy cycle';seen.add(x);stack.extend(parents[x])
 return seen
anc={cid:ancestors(cid) for cid in sets};byreaction=defaultdict(set)
for cid,ss in sets.items():
 for r in ss:byreaction[r].add(cid)
pairs=set()
for ids in byreaction.values():pairs.update(combinations(sorted(ids),2))
relations=[];cores={};coremembers=[];extensionmembers=[]
for left,right in sorted(pairs):
 lset,rset=sets[left],sets[right];core=lset&rset;extra_l=lset-core;extra_r=rset-core;cid='CLA-C-'+hashlib.sha256(('Reactome98|'+'|'.join(sorted(core))).encode()).hexdigest()[:16].upper()
 if cid not in cores:
  cores[cid]=frozenset(core)
  for rid in sorted(core):coremembers.append({'core_id':cid,'reaction_id':rid,'reaction_name':rnames.get(rid,''),'source_version':'Reactome98'})
 else:assert cores[cid]==frozenset(core),'Core hash collision'
 if left in anc[right]:relation='left_is_annotated_ancestor'
 elif right in anc[left]:relation='right_is_annotated_ancestor'
 elif not extra_l and not extra_r:relation='same_reaction_membership'
 elif not extra_l:relation='right_contains_left_reaction_scope'
 elif not extra_r:relation='left_contains_right_reaction_scope'
 else:relation='shared_reaction_core'
 if relation=='left_is_annotated_ancestor':assert rset<=lset
 if relation=='right_is_annotated_ancestor':assert lset<=rset
 relations.append({'left_id':left,'right_id':right,'left_name':names[left],'right_name':names[right],'relation':relation,'shared_core_id':cid,'shared_reactions':len(core),'left_additional_reactions':len(extra_l),'right_additional_reactions':len(extra_r),'reaction_jaccard':len(core)/len(lset|rset),'common_annotated_ancestors':'|'.join(sorted(anc[left]&anc[right])),'source_version':'Reactome98'})
 # Extension sets identify the exact additional reactions for each pair.
 for side,extra in [('left',extra_l),('right',extra_r)]:
  if not extra:continue
  eid='CLA-E-'+hashlib.sha256(('Reactome98|'+'|'.join(sorted(extra))).encode()).hexdigest()[:16].upper()
  relations[-1][side+'_extension_id']=eid
  if eid not in cores:
   cores[eid]=frozenset(extra)
   for rid in sorted(extra):extensionmembers.append({'extension_id':eid,'reaction_id':rid,'reaction_name':rnames.get(rid,''),'source_version':'Reactome98'})
  else:assert cores[eid]==frozenset(extra)
pd.DataFrame(relations).to_csv(O/'REACTION_SCOPE_RELATIONS.tsv.gz',sep='\t',index=False);pd.DataFrame(coremembers).to_csv(O/'REACTION_CORE_MEMBERS.tsv.gz',sep='\t',index=False);pd.DataFrame(extensionmembers).to_csv(O/'REACTION_EXTENSION_MEMBERS.tsv.gz',sep='\t',index=False)
# Cross-database, manually checked core and additional reactions retain their source identities.
for src,dst in [('semantic_relations_r1/CROSS_DATABASE_CURATED_RELATIONS.tsv','CROSS_DATABASE_FUNCTIONAL_SCOPE.tsv'),('semantic_relations_r1/CURATED_TYROSINE_CORE_REACTIONS.tsv','CROSS_DATABASE_CORE_REACTIONS.tsv')]:
 f=pd.read_csv(R/src,sep='\t')
 for col in [c for c in f.columns if c.endswith('_id')]:f[col]=f[col].map(lambda x:canon(x) if isinstance(x,str) and x.startswith('CLA-') else x)
 f.to_csv(O/dst,sep='\t',index=False)
counts=pd.Series([r['relation'] for r in relations]).value_counts().to_dict();js(O/'COMPLETE.json',{'target_host':'149','status':'COMPLETE','scored_pathways_with_reactions':len(sets),'pairs':len(relations),'relation_types':counts,'shared_core_sets':len({r['shared_core_id'] for r in relations}),'core_members':len(coremembers),'extension_members':len(extensionmembers),'identity_merges':0,'validation':['All shared cores are exact reaction-set intersections','All extension sets are exact differences','Annotated ancestry implies reaction-set containment','Stable core/extension IDs uniquely bind exact member sets'],'files':{p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name!='COMPLETE.json'}});print('REACTION_SCOPE_COMPLETE',len(relations),counts,flush=True)
