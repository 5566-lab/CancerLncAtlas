"""Connect assay definitions to mechanisms without claiming state/epitope equivalence."""
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from datetime import datetime,timezone
import json,hashlib,re,socket,gzip,csv
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'measurement_mechanism_r1';OUT.mkdir(exist_ok=True)
assert not (OUT/'COMPLETE.json').exists()
(OUT/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'MEASURED_FEATURE_TO_REACTION_PROVENANCE','activity_fitting':False},indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
inputs={}
def record(p,expected=None):
 p=Path(p);h=sha(p)
 if expected and expected!=h:raise ValueError('SOURCE_HASH_MISMATCH '+str(p))
 inputs[str(p)]={'sha256':h,'bytes':p.stat().st_size};return p
def tokens(s):return [x for x in re.split(r'[|;,]',str(s)) if x and x!='nan']
def labels(value):
 # Same corrected terminal-suffix parser as current cla_engine/rppa500.py.
 key=re.sub('[^A-Z0-9]','',re.sub(r'^X(?=\d)','',str(value).upper()))
 m=re.search(r'P[STY]\d+(?:P?[STY]\d+)*$',key)
 return re.findall(r'[STY]\d+',m.group(0)) if m else []
S=ROOT/'structure_reactome98';rp=Path('/dell_2/DSC/CancerLncAtlas/inputs/rppa500_20261104_r4');cp=Path('/dell_2/DSC/CancerLncAtlas/inputs/cptac_20261104_r1/bundle_r1')
rm=json.loads(record(rp/'COMPLETE.json').read_text());cm=json.loads(record(cp/'COMPLETE.json').read_text())
cat=pd.read_csv(record(rp/'FEATURE_CATALOG.tsv',rm['files']['FEATURE_CATALOG.tsv']['sha256']),sep='\t').fillna('')
fc=pd.read_csv(record(rp/'FEATURE_COVERAGE.tsv',rm['files']['FEATURE_COVERAGE.tsv']['sha256']),sep='\t');cat=cat.merge(fc[['feature_id','observed_patients']],validate='one_to_one')
cat['corrected_site_labels']=['|'.join(labels(v)) if layer=='PHOSPHORYLATION' else '' for v,layer in zip(cat.source_column,cat.molecular_layer)]
cat['old_catalog_site_label_changed']=cat.reported_phosphosite_labels.ne(cat.corrected_site_labels)
cat['current_code_parser_already_corrected']=True
cat['sequence_validated']=False
cat.to_csv(OUT/'RPPA_FEATURE_SITE_REVIEW.tsv',sep='\t',index=False)
fa=pd.read_parquet(record(cp/'FEATURE_ALIGNMENT.parquet',cm['files']['FEATURE_ALIGNMENT.parquet']['sha256'])).fillna('')
gmap=defaultdict(set)
for g,u in fa[['gene_id','uniprot_accessions']].drop_duplicates().itertuples(index=False,name=None):
 if g:gmap[g].update(tokens(u))
proteins=pd.read_csv(record(S/'PROTEIN_STATES.tsv.gz'),sep='\t').fillna('')
mods=pd.read_csv(record(S/'PROTEIN_MODIFICATIONS.tsv.gz'),sep='\t').fillna('')
mem=pd.read_csv(record(S/'COMPLEX_AND_ENTITY_SET_MEMBERS.tsv.gz'),sep='\t').fillna('')
roles=pd.read_csv(record(S/'REACTION_ENTITY_ROLES.tsv.gz'),sep='\t').fillna('')
children=defaultdict(list)
for a,b,rel in mem[['container_node','member_node','relation']].itertuples(index=False,name=None):children[a].append((b,rel))
punip={row.protein_state_node:set(tokens(row.uniprot_accessions)) for row in proteins.itertuples()}
# Keep generic sets and complexes distinct; the flattened table is provenance only.
visiting=set()
@lru_cache(None)
def leaves(node):
 if node in visiting:raise ValueError('ENTITY_MEMBERSHIP_CYCLE '+node)
 visiting.add(node);out=set()
 if node in punip:out.add(node)
 for child,rel in children.get(node,[]):out.update(leaves(child))
 visiting.remove(node);return frozenset(out)
rppa_by_uni=defaultdict(set);features=cat.set_index('feature_id').to_dict('index')
for row in cat.itertuples():
 if row.observed_patients<=0:continue
 for gene in tokens(row.gene_ids):
  for u in gmap.get(gene,[]):rppa_by_uni[u].add(row.feature_id)
# Strictly retain terminal residue labels; matching is still provisional without sequence validation.
residue={'O-phospho-L-serine':'S','O-phospho-L-threonine':'T',"O4'-phospho-L-tyrosine":'Y'}
site_rows=[]
for row in mods.itertuples():
 aa=residue.get(row.modification_terms)
 if not aa or row.position_status!='EQUAL' or row.sequence_position=='':continue
 site=aa+str(int(float(row.sequence_position)))
 for u in tokens(row.uniprot_accessions):site_rows.append(dict(protein_state_node=row.protein_state_node,uniprot_accession=u,site=site,feature_node=row.feature_node))
siteframe=pd.DataFrame(site_rows).drop_duplicates();siteframe.to_csv(OUT/'REACTOME_EXPLICIT_PHOSPHOSITES.tsv.gz',sep='\t',index=False)
sites=defaultdict(set)
for st,u,site in siteframe[['protein_state_node','uniprot_accession','site']].itertuples(index=False,name=None):sites[(st,u)].add(site)
# Common gene/site fields alone are not sufficient to establish isoform sequence equivalence.
ph=fa.loc[fa.layer.eq('phosphorylation'),['feature_id','gene_id','uniprot_accessions','protein_isoform','site','flanking_sequence']].drop_duplicates()
ph['uniprot_accession']=ph.uniprot_accessions.map(tokens);ph=ph.explode('uniprot_accession')
ph=ph.loc[ph.uniprot_accession.notna() & ph.site.ne('')]
match=ph.merge(siteframe,on=['uniprot_accession','site'],how='inner')
match['match_status']='REFERENCE_ACCESSION_AND_SITE_CANDIDATE_REQUIRES_ISOFORM_SEQUENCE_VALIDATION'
match['eligible_as_exact_state_measurement']=False
match.to_parquet(OUT/'CPTAC_REACTOME_PHOSPHOSITE_CANDIDATES.parquet',index=False)
# Pathway/reaction membership retains the authority's direct/descendant distinction.
pr=pd.read_csv(record(S/'PATHWAY_REACTION.tsv.gz'),sep='\t').fillna('');print('PATHWAY_REACTION_COLUMNS',list(pr),flush=True)
reaction_features=defaultdict(set);reaction_site_features=defaultdict(set);role_counts=defaultdict(int);linked_states=set();n=0
cols=['reaction_stable_id','reaction_node','reaction_role','control_type','direct_entity_node','direct_entity_class','protein_state_node','uniprot_accession','rppa_feature_id','molecular_layer','observed_patients','match_kind','contains_entity_set','exact_state_measurement','activation_direction_inferred']
set_containers=set(mem.loc[mem.relation.ne('component'),'container_node'])
@lru_cache(None)
def contains_set(node):return node in set_containers or any(contains_set(child) for child,rel in children.get(node,[]))
with gzip.open(OUT/'RPPA_REACTION_READOUT_LINKS.tsv.gz','wt',newline='') as f:
 w=csv.DictWriter(f,fieldnames=cols,delimiter='\t');w.writeheader()
 for row in roles.itertuples():
  for state in leaves(row.entity_node):
   for u in punip.get(state,[]):
    for fid in rppa_by_uni.get(u,[]):
     fe=features[fid];layer=fe['molecular_layer']
     if layer=='PROTEIN':kind='PROTEIN_REFERENCE_CONTEXT_NOT_MODIFICATION_OR_COMPLEX_MEASUREMENT'
     elif layer=='PHOSPHORYLATION':
      ls=set(tokens(fe['corrected_site_labels']));hits=ls & sites.get((state,u),set())
      if not hits:continue
      kind='SINGLE_GENE_SITE_LABEL_CANDIDATE' if int(fe['gene_count'])==1 and len(ls)==1 else 'AMBIGUOUS_FAMILY_OR_MULTISITE_CANDIDATE'
      reaction_site_features[row.reaction_stable_id].add(fid)
     else:continue
     w.writerow(dict(reaction_stable_id=row.reaction_stable_id,reaction_node=row.reaction_node,reaction_role=row.role,control_type=row.control_type,direct_entity_node=row.entity_node,direct_entity_class=row.entity_class,protein_state_node=state,uniprot_accession=u,rppa_feature_id=fid,molecular_layer=layer,observed_patients=fe['observed_patients'],match_kind=kind,contains_entity_set=contains_set(row.entity_node),exact_state_measurement=False,activation_direction_inferred=False))
     reaction_features[row.reaction_stable_id].add(fid);role_counts[row.role]+=1;linked_states.add(state);n+=1
print('LINK_ROWS',n,flush=True)
# Explicitly count coverage as any candidate readout; not an admission threshold.
ridcol='reaction_stable_id';pidcol='pathway_stable_id'
if ridcol not in pr:raise ValueError('REACTION_SCHEMA '+str(list(pr)))
if pidcol not in pr:
 candidates=[c for c in pr if c.startswith('pathway') and ('stable' in c or c.endswith('_id'))]
 if len(candidates)!=1:raise ValueError('PATHWAY_SCHEMA '+str(list(pr)))
 pidcol=candidates[0]
out=[]
for pid,part in pr.groupby(pidcol):
 reactions=set(part[ridcol]);linked=reactions & set(reaction_features);site_linked=reactions & set(reaction_site_features)
 out.append(dict(reactome_id=pid,reaction_count=len(reactions),reactions_any_rppa_reference_context=len(linked),reactions_any_rppa_site_candidate=len(site_linked),rppa_features='|'.join(sorted(set().union(*(reaction_features[x] for x in linked)))) if linked else '',exact_state_coverage_validated=False,latent_activity_identified=False))
pd.DataFrame(out).to_csv(OUT/'PATHWAY_RPPA_MECHANISM_COVERAGE.tsv',sep='\t',index=False)
(OUT/'INPUT_SHA256.json').write_text(json.dumps(inputs,indent=2))
status={'target_host':'149','status':'COMPLETE_CANDIDATE_MEASUREMENT_BRIDGE','rppa_feature_count':len(cat),'stale_catalog_site_labels_corrected':int(cat.old_catalog_site_label_changed.sum()),'current_code_parser_already_corrected':True,'rppa_reaction_context_links':n,'reactions_any_rppa_context':len(reaction_features),'reactions_any_rppa_phosphosite_candidate':len(reaction_site_features),'protein_states_linked':len(linked_states),'role_link_counts':dict(role_counts),'cptac_phospho_feature_candidates':int(match.feature_id.nunique()),'cptac_site_candidate_rows':len(match),'exact_state_equivalence_validated':0,'patient_activity_scores_created':0,'relation_Y_created':0,'scope':'Mechanistic mapping with measured assay provenance; no automatic activation or cross-platform calibration','created_utc':datetime.now(timezone.utc).isoformat()}
status['files']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.is_file() and p.name!='COMPLETE.json'}
(OUT/'COMPLETE.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2),flush=True)
