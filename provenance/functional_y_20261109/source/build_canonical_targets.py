"""Versioned target identities and source composition, separate from score readouts."""
from pathlib import Path
import json,hashlib,socket,re,collections
import pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'canonical_targets_r2';O.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sid(kind,key):return 'CLA-'+kind+'-'+hashlib.sha256(key.encode()).hexdigest()[:16].upper()
def save(name,df):df.to_csv(O/name,sep='\t',index=False)
P=pd.read_csv(R/'catalog_inventory/PATHWAY_SOURCE_ID_REGISTRY_DRAFT.tsv',sep='\t').fillna('');S=pd.read_csv(R/'catalog_inventory/STATE_SOURCE_ID_REGISTRY_DRAFT.tsv',sep='\t').fillna('');A=pd.read_csv(R/'catalog_inventory/REACTOME_PATHWAY_AUTHORITY.tsv',sep='\t').fillna('')
names=A.set_index('reactome_id').name.to_dict();raw=json.loads((R/'CROSSDATABASE_DEFINITION_SOURCES.json').read_text())['rows'];meta={d['source_name']:d for d in raw if d.get('status')=='VERIFIED_SOURCE_METADATA'}
manual={d['source_name']:d for d in json.loads((R/'sources/MSIGDB_CURATED_EXACT_XREFS.json').read_text())['records']}
master=[];bindings=[];redirect=[];aliases=[];pidmap={}
for row in P.itertuples():
 authority=row.authority_id;rid=row.reactome_id;url='';evidence=row.identity_evidence;systematic='';version='';entry_class=row.target_type
 if row.source_name in manual:
  m=manual[row.source_name];rid=m['reactome_id'];authority='REACTOME:'+rid;url=m['source_url'];evidence='MSigDB exact source field';systematic=m['systematic_name'];version=m['source_version']
 elif row.source_name in meta:
  m=meta[row.source_name];url=m['url'];systematic=m.get('systematic_name','')
  if row.source_database=='KEGG_MEDICUS' and re.fullmatch('N[0-9]+',m.get('exact_source','')):
   authority='KEGG.NETWORK:'+m['exact_source'];entry_class='network_module';evidence='MSigDB exact source field'
 canonical=sid('P',authority);pidmap[row.source_target_id]=canonical
 if canonical!=row.canonical_id:redirect.append({'previous_internal_id':row.canonical_id,'canonical_id':canonical,'relation':'same_identity','evidence':evidence})
 display=names.get(rid,row.source_name.removeprefix('REACTOME_').removeprefix('HALLMARK_').removeprefix('KEGG_MEDICUS_').replace('_',' '))
 master.append({'canonical_id':canonical,'display_name':display,'entity_type':entry_class,'authority_id':authority,'source_database':row.source_database,'reactome_id':rid,'membership_gene_count':row.member_count,'roles':'Pathway','definition_version':version,'systematic_source_id':systematic,'source_url':url})
 bindings.append({'source_target_id':row.source_target_id,'source_column':'','canonical_id':canonical,'source_name':row.source_name,'source_database':row.source_database,'relation':'same_identity','identity_evidence':evidence})
 aliases.append({'canonical_id':canonical,'alias':row.source_name,'alias_type':'source_name','source_database':row.source_database})
 if rid:aliases.append({'canonical_id':canonical,'alias':rid,'alias_type':'authority_id','source_database':'Reactome'})
for row in S.itertuples():
 t=json.loads(row.original_definition);canonical=row.canonical_id;src=row.source_column
 if src.startswith('HALLMARK::'):
  target='MSIGDB:HALLMARK:'+src.split('::',1)[1]
  if target in pidmap:
   canonical=pidmap[target];redirect.append({'previous_internal_id':row.canonical_id,'canonical_id':canonical,'relation':'same_identity','evidence':'Same named Hallmark source definition in State and pathway catalogs'})
   for m in master:
    if m['canonical_id']==canonical:m['roles']='Pathway|State'
 if canonical==row.canonical_id:
  master.append({'canonical_id':canonical,'display_name':row.source_target_id,'entity_type':'state','authority_id':'STATE:'+row.source_target_id,'source_database':t.get('source',''),'reactome_id':'','membership_gene_count':'','roles':'State','definition_version':'','systematic_source_id':'','source_url':t.get('source','') if str(t.get('source','')).startswith('http') else ''})
 bindings.append({'source_target_id':row.source_target_id,'source_column':src,'canonical_id':canonical,'source_name':row.source_target_id,'source_database':t.get('source',''),'relation':'same_identity','identity_evidence':'Exact State source definition' if canonical==row.canonical_id else 'Same Hallmark definition'})
 aliases.append({'canonical_id':canonical,'alias':row.source_target_id,'alias_type':'State_identifier','source_database':t.get('source','')})
M=pd.DataFrame(master);B=pd.DataFrame(bindings);assert not M.canonical_id.duplicated().any();assert not B.source_target_id.duplicated().any()
D=pd.DataFrame(redirect);legacy=pd.read_csv(R/'score_registry_r2/EXACT_ID_REDIRECTS.tsv',sep='\t');legacy=legacy.rename(columns={'mapping_evidence':'evidence'})[['previous_internal_id','canonical_id','relation','evidence']];D=pd.concat([D,legacy],ignore_index=True);assert not D.previous_internal_id.duplicated().any()
redirects=D.set_index('previous_internal_id').canonical_id.to_dict();D.canonical_id=[redirects.get(x,x) for x in D.canonical_id]
save('TARGET_MASTER.tsv',M);save('SOURCE_ID_BINDINGS.tsv',B);save('ID_REDIRECTS.tsv',D);save('SOURCE_ALIASES.tsv',pd.DataFrame(aliases))
catpath=Path('/dsk2/user/dengsc/cla_pathway_fusion_20261101_r1/raw_inputs/CATALOG.json');cat=json.loads(catpath.read_text());cat_hash=sha(catpath);members=[]
for source,genes in cat['gene_sets'].items():
 for gene in genes:members.append({'canonical_id':pidmap[source],'source_target_id':source,'member_id':gene,'member_namespace':'Ensembl_gene','member_role':'source_gene_set_member','source_definition_sha256':cat_hash})
save('PATHWAY_GENE_COMPOSITION.tsv.gz',pd.DataFrame(members))
pr=pd.read_csv(R/'structure_reactome98/PATHWAY_REACTION.tsv.gz',sep='\t');ids=set(P.reactome_id)|set(manual[x]['reactome_id'] for x in manual);pr=pr.loc[pr.pathway_stable_id.isin(ids)].copy();pr['canonical_id']=[sid('P','REACTOME:'+v) for v in pr.pathway_stable_id];save('PATHWAY_REACTION_COMPOSITION.tsv.gz',pr)
# The 624 PTMsigDB sets are measured readout definitions; retain their exact provenance here.
extra=pd.read_csv(R/'score_registry_r2/ADDITIONAL_SOURCE_IDENTITIES.tsv',sep='\t');gmt=Path('/dsk2/user/dengsc/cla_cptac_integration_20261104_r1/sources/PTMSIGDB_PATH_KINASE_V2.gmt');reads=[];sites=[]
for line in gmt.read_text().splitlines():
 f=line.split('\t');term,description=f[:2];wids=sorted(set(re.findall(r'Pathway:(WP[0-9]+)',description)));reads.append({'source_readout_id':term,'source_database':'PTMsigDB','source_version':'2.0.0','source_family':term.split('_')[0],'data_form':'Directional phosphosite signature','site_count':len(f)-2,'external_authority_ids':'|'.join(wids)})
 for token in f[2:]:
  peptide,direction=token.rsplit(';',1);sites.append({'source_readout_id':term,'site_identifier':peptide,'direction':direction,'member_namespace':'Phosphosite_flanking_sequence'})
save('PHOSPHO_READOUT_DEFINITIONS.tsv',pd.DataFrame(reads));save('PHOSPHO_READOUT_COMPOSITION.tsv.gz',pd.DataFrame(sites))
reg=pd.read_parquet(R/'score_registry_r2/SCORE_REGISTRY.parquet');reg['canonical_id']=[redirects.get(x,x) for x in reg.canonical_id];reg.to_parquet(O/'SCORE_REGISTRY.parquet',compression='zstd')
summary={'target_host':'149','target_identities':len(M),'pathway_role':int(M.roles.str.contains('Pathway').sum()),'state_role':int(M.roles.str.contains('State').sum()),'shared_pathway_state_identities':int(M.roles.eq('Pathway|State').sum()),'source_definition_bindings':len(B),'id_redirects':len(D),'pathway_gene_memberships':len(members),'pathway_reaction_memberships':len(pr),'phosphosite_readout_definitions':len(reads),'phosphosite_readout_memberships':len(sites),'exact_reactome_source_ids':int(M.reactome_id.ne('').sum()),'mapped_kegg_network_ids':int(M.authority_id.str.startswith('KEGG.NETWORK:').sum()),'source_hashes':{str(p):sha(p) for p in [catpath,gmt,R/'sources/MSIGDB_CURATED_EXACT_XREFS.json',R/'CROSSDATABASE_DEFINITION_SOURCES.json']}}
(O/'COMPLETE.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
