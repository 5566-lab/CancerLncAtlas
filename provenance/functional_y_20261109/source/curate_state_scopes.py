"""Curate State identity, biological facets and exact score compositions."""
from pathlib import Path
import json,socket,hashlib,pandas as pd,numpy as np
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;S=R/'canonical_targets_r5';O=R/'canonical_targets_r6';O.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(O/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'STATE_BIOLOGICAL_SCOPE_AND_COMPONENT_CURATION'}))
B=pd.read_csv(S/'SOURCE_ID_BINDINGS.tsv',sep='\t').fillna('');lookup=B.set_index('source_target_id').canonical_id.to_dict();M=pd.read_csv(S/'TARGET_MASTER.tsv',sep='\t').fillna('');mapping={};merges=[];labels={}
for keep,other,label,scope,ref in [
 ('ABSOLUTE_purity','TumorPurity','Tumor cell fraction','Fraction of malignant cells in the bulk tumor sample','https://www.nature.com/articles/ncomms3612'),
 ('xCell::CD8+ T-cells','MCPcounter::CD8 T cells','CD8 T-cell abundance in bulk tissue','Bulk-tissue abundance of the CD8 T-cell population','https://pmc.ncbi.nlm.nih.gov/articles/PMC5073889/'),
 ('xCell::Endothelial cells','MCPcounter::Endothelial cells','Endothelial-cell abundance in bulk tissue','Bulk-tissue abundance of endothelial cells','https://pmc.ncbi.nlm.nih.gov/articles/PMC5073889/'),
 ('xCell::Fibroblasts','MCPcounter::Fibroblasts','Fibroblast abundance in bulk tissue','Bulk-tissue abundance of fibroblasts','https://pmc.ncbi.nlm.nih.gov/articles/PMC5073889/'),
 ('xCell::NK cells','MCPcounter::NK cells','NK-cell abundance in bulk tissue','Bulk-tissue abundance of natural killer cells','https://pmc.ncbi.nlm.nih.gov/articles/PMC5073889/'),
 ('xCell::Neutrophils','MCPcounter::Neutrophils','Neutrophil abundance in bulk tissue','Bulk-tissue abundance of neutrophils','https://pmc.ncbi.nlm.nih.gov/articles/PMC5073889/')]:
 old,new=lookup[other],lookup[keep];mapping[old]=new;labels[new]=label;merges.append({'source_target_id':other,'reference_target_id':keep,'previous_canonical_id':old,'canonical_id':new,'relation':'same_biological_quantity','scope':scope,'source_url':ref})
hyp=M.loc[M.authority_id.eq('MSIGDB:HALLMARK:HALLMARK_HYPOXIA'),'canonical_id'].item();labels[hyp]='Hypoxia transcriptional response'
for state in ['signifinder::Hypoxia_Buffa','PROGENY:Hypoxia']:
 old=lookup[state];mapping[old]=hyp;merges.append({'source_target_id':state,'reference_target_id':'HALLMARK_HYPOXIA','previous_canonical_id':old,'canonical_id':hyp,'relation':'same_response_program','scope':'Transcriptional response to low oxygen; each signature remains a distinct readout','source_url':'https://www.gsea-msigdb.org/gsea/msigdb/cards/HALLMARK_HYPOXIA'})
for p in S.iterdir():
 if p.name in ['COMPLETE.json','LAUNCH_PREFLIGHT.json']:continue
 if p.name.endswith(('.tsv','.tsv.gz','.parquet')):
  d=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,sep='\t',keep_default_na=False,low_memory=False)
  if p.name=='TARGET_MASTER.tsv':
   d=d.loc[~d.canonical_id.isin(mapping)].copy();d.loc[d.canonical_id.eq(hyp),'roles']='Pathway|State'
   for k,v in labels.items():d.loc[d.canonical_id.eq(k),'display_name']=v
   for src,cid in lookup.items():
    if src.startswith('cibersort_relative::') and cid not in mapping:d.loc[d.canonical_id.eq(cid),'display_name']=src.split('::',1)[1].replace('.',' ').strip()+' fraction within the LM22 immune compartment'
  for col in d:
   if col in ['canonical_id','source_canonical_id','target_canonical_id','parent_id','child_id','source_id','target_id','left_id','right_id']:d[col]=d[col].replace(mapping)
  if p.name=='ID_REDIRECTS.tsv':d=pd.concat([d,pd.DataFrame([{'previous_internal_id':x,'canonical_id':y,'relation':'STATE_FUNCTIONAL_SCOPE_EQUIVALENCE','evidence':'Measured biological quantity and population scope matched; numerical readouts remain source-specific'} for x,y in mapping.items()])],ignore_index=True)
  if p.name=='SOURCE_ALIASES.tsv':d=pd.concat([d,pd.DataFrame([{'canonical_id':mapping[x],'alias':M.set_index('canonical_id').loc[x,'display_name'],'alias_type':'EQUIVALENT_STATE_READOUT','source_database':'CancerLncAtlas State scope curation'} for x in mapping])],ignore_index=True)
  if p.suffix=='.parquet':d.drop_duplicates().to_parquet(O/p.name,index=False,compression='zstd')
  else:d.drop_duplicates().to_csv(O/p.name,sep='\t',index=False)
 else:(O/p.name).write_bytes(p.read_bytes())
pd.DataFrame(merges).to_csv(O/'STATE_SCOPE_EQUIVALENCES.tsv',sep='\t',index=False)
L={k:mapping.get(v,v) for k,v in lookup.items()};master=pd.read_csv(O/'TARGET_MASTER.tsv',sep='\t').fillna('');parents=[];links=[]
def parent(key,name):
 authority='CLA:STATE_CONCEPT:'+key;cid='CLA-S-'+hashlib.sha256(authority.encode()).hexdigest()[:16].upper();row={c:'' for c in master};row.update(canonical_id=cid,display_name=name,entity_type='state_concept',authority_id=authority,source_database='CancerLncAtlas',roles='State',definition_version='2026-11-09',has_scored_readout=False);parents.append(row);return cid
def edge(parent_id,child,relation,formula='',weight='',reference=''):
 links.append({'parent_id':parent_id,'child_id':L[child],'source_child_score':child,'relation':relation,'composition_formula':formula,'component_weight':weight,'source_url':reference})
stem=parent('ONCOGENIC_DEDIFFERENTIATION','Oncogenic dedifferentiation')
for n in ['RNAss','EREG.EXPss','DNAss','DMPss','ENHss','EREG-METHss']:edge(stem,n,'has_molecular_facet',reference='https://pmc.ncbi.nlm.nih.gov/articles/PMC5902191/')
for n in ['DMPss','ENHss','EREG-METHss']:edge(L['DNAss'],n,'signature_component','OCLR fitted to the combined 219-CpG signature',reference='https://pmc.ncbi.nlm.nih.gov/articles/PMC5902191/')
emt=parent('EPITHELIAL_MESENCHYMAL_STATE','Epithelial and mesenchymal programs')
for n in ['signifinder::EMT_Miow_Epithelial','signifinder::EMT_Miow_Mesenchymal']:edge(emt,n,'has_program_branch')
ecm=parent('CANCER_ASSOCIATED_ECM','Cancer-associated extracellular-matrix programs')
for n in ['signifinder::ECM_Chakravarthy_up','signifinder::ECM_Chakravarthy_down']:edge(ecm,n,'has_directional_signature')
for n in ['HRD_LOH','HRD_TAI','HRD_LST']:edge(L['HRD_total'],n,'score_sum_component','HRD_total = HRD_LOH + HRD_TAI + HRD_LST',1)
for n in ['ImmuneScore','StromalScore']:edge(L['ESTIMATEScore'],n,'score_sum_component','ESTIMATEScore = ImmuneScore + StromalScore',1,'https://www.nature.com/articles/ncomms3612')
# Cell abundance hierarchy preserves the denominator of each measurement.
for base,children in [('xCell::CD4+ T-cells',['xCell::CD4+ Tcm','xCell::CD4+ Tem','xCell::CD4+ memory T-cells','xCell::CD4+ naive T-cells']),('xCell::CD8+ T-cells',['xCell::CD8+ Tcm','xCell::CD8+ Tem','xCell::CD8+ naive T-cells']),('xCell::B-cells',['xCell::naive B-cells','xCell::Memory B-cells']),('xCell::Memory B-cells',['xCell::Class-switched memory B-cells']),('xCell::Macrophages',['xCell::Macrophages M1','xCell::Macrophages M2']),('xCell::Endothelial cells',['xCell::ly Endothelial cells','xCell::mv Endothelial cells']),('xCell::DC',['xCell::aDC','xCell::cDC','xCell::iDC','xCell::pDC'])]:
 for child in children:edge(L[base],child,'has_cell_subpopulation')
master=pd.concat([master,pd.DataFrame(parents)],ignore_index=True);assert master.canonical_id.is_unique;master.to_csv(O/'TARGET_MASTER.tsv',sep='\t',index=False);pd.DataFrame(links).to_csv(O/'STATE_COMPONENT_RELATIONS.tsv',sep='\t',index=False)
raw=pd.read_parquet('/dell_2/DSC/CancerLncAtlas/inputs/state_augmented_20261104_r1/STATE_AUGMENTED_PATIENT.parquet');checks=[]
for total,parts,tol in [('hrd::HRD',['hrd::hrd-loh','hrd::ai1','hrd::lst1'],1e-9),('ESTIMATE::ESTIMATEScore',['ESTIMATE::ImmuneScore','ESTIMATE::StromalScore'],.010001)]:
 a=raw[[total]+parts].dropna();err=(a[total]-a[parts].sum(axis=1)).abs();assert err.max()<=tol;checks.append({'score':total,'matched_patients':len(a),'maximum_difference':float(err.max()),'tolerance':tol})
(O/'STATE_COMPOSITION_VALIDATION.json').write_text(json.dumps({'target_host':'149','status':'PASS','checks':checks},indent=2))
q=master.has_scored_readout.astype(str).str.lower().eq('true');counts={'target_host':'149','status':'COMPLETE','identities_with_readouts':int(q.sum()),'structural_context_identities':int((~q).sum()),'pathway_roles':int((q&master.roles.str.contains('Pathway')).sum()),'state_roles':int((q&master.roles.str.contains('State')).sum()),'shared_roles':int((q&master.roles.eq('Pathway|State')).sum()),'state_identity_merges':len(mapping),'state_component_links':len(links),'source_registry_sha256':sha(S/'COMPLETE.json'),'files':{p.name:sha(p) for p in O.iterdir() if p.is_file()}};(O/'COMPLETE.json').write_text(json.dumps(counts,indent=2));print('STATE_SCOPE_COMPLETE',json.dumps({k:v for k,v in counts.items() if k!='files'}),flush=True)
