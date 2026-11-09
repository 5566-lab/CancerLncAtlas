"""Index molecular composition and explicit reaction branches by canonical target."""
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
import json,socket,hashlib
import pandas as pd
assert socket.gethostname() == '149'
R=Path(__file__).resolve().parent;C=R/'canonical_targets_r6';O=R/'target_composition_index_r1'
O.mkdir(exist_ok=False)
(O/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'INDEX_TARGET_COMPOSITION_AND_REACTION_BRANCHES'}))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
master=pd.read_csv(C/'TARGET_MASTER.tsv',sep='\t').fillna('');targets=master.loc[master.has_scored_readout.astype(str).str.lower().eq('true')].copy();assert len(targets)==3456
nodes=pd.read_csv(R/'structure_reactome98/BIOPAX_NODES.tsv.gz',sep='\t').fillna('');names=nodes.set_index('node_id').name.to_dict();stable=nodes.set_index('node_id').reactome_stable_id.to_dict();classes=nodes.set_index('node_id')['class'].to_dict()
edges=pd.read_csv(R/'structure_reactome98/BIOPAX_EDGES.tsv.gz',sep='\t',usecols=['subject_id','relation','object_id','subject_class','object_class'])
hierarchy=edges.loc[edges.relation.eq('pathwayComponent')&edges.subject_class.eq('Pathway')&edges.object_class.eq('Pathway')]
children=hierarchy.groupby('subject_id').object_id.agg(set).to_dict();direct=edges.loc[edges.relation.eq('pathwayOrder')].groupby('subject_id').object_id.agg(set).to_dict();active=set()
@lru_cache(None)
def steps(pathway):
 if pathway in active:raise ValueError('Cyclic pathway hierarchy')
 active.add(pathway);result=set(direct.get(pathway,set()))
 for child in children.get(pathway,set()):result.update(steps(child))
 active.remove(pathway);return frozenset(result)
node_by_reactome=nodes.loc[nodes['class'].eq('Pathway')&nodes.reactome_stable_id.ne('')].set_index('reactome_stable_id').node_id.to_dict();step_targets=defaultdict(set);scope={}
for row in targets.loc[targets.reactome_id.ne('')].itertuples():
 node=node_by_reactome.get(row.reactome_id)
 if node:
  scope[row.canonical_id]=steps(node)
  for step in scope[row.canonical_id]:step_targets[step].add(row.canonical_id)
branches=pd.read_csv(R/'catalog_inventory/REACTION_ORDER_BRANCHES.tsv.gz',sep='\t').fillna('');assert not branches.duplicated(['fork_step','downstream_step']).any()
rows=[]
def processes(text):
 tokens=text.split('|');reactions=[v for v in tokens if classes.get(v) in ['BiochemicalReaction','Transport','TransportWithBiochemicalReaction','Degradation','ComplexAssembly']]
 return '|'.join(sorted({stable[v] for v in reactions if stable.get(v)})), ' | '.join(names.get(v,v) for v in reactions)
for row in branches.itertuples():
 shared=step_targets[row.fork_step]&step_targets[row.downstream_step]
 up_ids,up_names=processes(row.upstream_processes);down_ids,down_names=processes(row.downstream_processes)
 for cid in sorted(shared):
  assert row.fork_step in scope[cid] and row.downstream_step in scope[cid]
  rows.append({'canonical_id':cid,'fork_step':row.fork_step,'downstream_step':row.downstream_step,'upstream_reaction_ids':up_ids,'upstream_reaction_names':up_names,'downstream_reaction_ids':down_ids,'downstream_reaction_names':down_names,'relation':'reaction_order_fork','n_next_steps':row.n_next_steps,'membership_scope':'EXPLICIT_PATHWAY_ORDER_OR_DESCENDANT','source_version':'Reactome98'})
bridge=pd.DataFrame(rows);assert not bridge.duplicated(['canonical_id','fork_step','downstream_step']).any();bridge.to_csv(O/'PATHWAY_REACTION_BRANCHES.tsv.gz',sep='\t',index=False)
# Count composition types separately; gene and phosphosite memberships have different units.
summary=targets[['canonical_id','display_name','roles','source_database']].copy()
def counts(column,frame,key,member=None):
 f=frame.loc[frame[key].notna()].copy()
 if member:
  f=f.loc[f[member].notna()&f[member].astype(str).ne('')];n=f.groupby(key)[member].nunique()
 else:n=f.groupby(key).size()
 summary[column]=summary.canonical_id.map(n).fillna(0).astype(int)
bind=pd.read_csv(C/'SOURCE_ID_BINDINGS.tsv',sep='\t');counts('source_definitions',bind,'canonical_id','source_target_id')
genes=pd.read_csv(C/'PATHWAY_GENE_COMPOSITION.tsv.gz',sep='\t');counts('pathway_gene_members',genes,'canonical_id','member_id')
states=pd.read_csv(C/'STATE_MOLECULAR_COMPOSITION.tsv.gz',sep='\t');counts('state_gene_members',states,'canonical_id','gene_id');counts('state_probe_members',states,'canonical_id','source_probe_id')
ptm=pd.read_csv(C/'PHOSPHO_READOUT_COMPOSITION.tsv.gz',sep='\t').merge(pd.read_csv(C/'PTM_READOUT_BINDINGS.tsv',sep='\t')[['source_readout_id','canonical_id']],on='source_readout_id',validate='many_to_one');counts('phosphosite_sequence_members',ptm,'canonical_id','site_identifier');counts('phosphosite_readouts',ptm,'canonical_id','source_readout_id')
reactions=pd.read_csv(C/'PATHWAY_REACTION_COMPOSITION.tsv.gz',sep='\t');counts('reaction_members',reactions,'canonical_id','reaction_stable_id');counts('reaction_order_branches',bridge,'canonical_id')
parents=pd.read_csv(R/'catalog_inventory/REACTOME_PARENT_CHILD.tsv',sep='\t');counts('annotated_direct_subpathways',parents,'parent_id','child_id');counts('annotated_direct_parent_pathways',parents,'child_id','parent_id')
state_components=pd.read_csv(C/'STATE_COMPONENT_RELATIONS.tsv',sep='\t');counts('state_component_relations',state_components,'parent_id','child_id')
summary.to_csv(O/'TARGET_COMPOSITION_COUNTS.tsv',sep='\t',index=False)
record={'target_host':'149','status':'COMPLETE','canonical_targets':len(summary),'pathway_branch_links':len(bridge),'canonical_targets_with_reaction_branches':int(summary.reaction_order_branches.gt(0).sum()),'distinct_annotated_forks':int(bridge.fork_step.nunique()),'composition_coverage':{k:int(summary[k].gt(0).sum()) for k in summary.columns[4:]},'inputs':{'canonical_registry':sha(C/'COMPLETE.json'),'biopax_edges':sha(R/'structure_reactome98/BIOPAX_EDGES.tsv.gz'),'annotated_branches':sha(R/'catalog_inventory/REACTION_ORDER_BRANCHES.tsv.gz')},'files':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'COMPLETE.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
