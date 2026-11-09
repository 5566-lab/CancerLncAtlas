"""Bind PTM readouts to enzyme, complex, isoform and pathway identities."""
from pathlib import Path
import sys,json,re,hashlib,shutil,socket
import pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;SRC=R/'canonical_targets_r2';OUT=R/'canonical_targets_r4';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cid(kind,key):return 'CLA-'+kind+'-'+hashlib.sha256(key.encode()).hexdigest()[:16].upper()
def js(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2))
js(OUT/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'PTM_BIOLOGICAL_IDENTITY_BINDING','sources':['GENCODE_v36','HGNC_20261105','PTMsigDB_2.0.0','ComPath']})
for p in SRC.iterdir():
 if p.is_file() and p.name!='COMPLETE.json':shutil.copy2(p,OUT/p.name)
M=pd.read_csv(SRC/'TARGET_MASTER.tsv',sep='\t').fillna('');B=pd.read_csv(SRC/'SOURCE_ID_BINDINGS.tsv',sep='\t').fillna('');RED=pd.read_csv(SRC/'ID_REDIRECTS.tsv',sep='\t').fillna('');ALIASES=pd.read_csv(SRC/'SOURCE_ALIASES.tsv',sep='\t').fillna('')
H=pd.read_csv(R/'sources/hgnc_complete_set_20261109.tsv',sep='\t',dtype=str).fillna('');H=H.loc[H.locus_group.eq('protein-coding gene')];assert sha(R/'sources/hgnc_complete_set_20261109.tsv')==json.loads((R/'sources/HGNC_RECEIPT.json').read_text())['sha256'];hrows=H.set_index('hgnc_id');primary={r.symbol.upper():r.hgnc_id for r in H.itertuples()};aliases={}
for r in H.itertuples():
 for x in [r.symbol]+r.alias_symbol.split('|')+r.prev_symbol.split('|'):
  if x:aliases.setdefault(x.upper(),set()).add(r.hgnc_id)
def gene(tokens):
 tokens=[t.upper() for t in tokens if t];sets=[aliases[t] for t in tokens if t in aliases]
 if not sets:return None
 hit=set.intersection(*sets)
 if len(hit)==1:return next(iter(hit))
 official={primary[t] for t in tokens if t in primary}&hit
 return next(iter(official)) if len(official)==1 else None
D=pd.read_csv(SRC/'PHOSPHO_READOUT_DEFINITIONS.tsv',sep='\t').fillna('');OLD=pd.read_csv(R/'score_registry_r2/ADDITIONAL_SOURCE_IDENTITIES.tsv',sep='\t').set_index('source_target_id').canonical_id.to_dict();new={};bind=[];redirect=[];component=[];hier=[];al=[];unresolved=[]
qualifiers={'PRKCB1.PRKCB':'PRKCB1','PRKCB2.PRKCB':'PRKCB2','LYNA.LYN':'LYNA','LYNB.LYN':'LYNB','smMLCK/MYLK':'smooth_muscle_MLCK'}
fusions={'NPM-ALK/NPM/ALK':['NPM1','ALK'],'NPM1:ALK':['NPM1','ALK'],'BCR-ABL1/BCR/ABL':['BCR','ABL1']}
def add_entity(key,name,typ,role,genes,source,qualifier=''):
 k='S' if role=='State' else 'P';i=cid(k,key)
 if i not in new:
  new[i]={'canonical_id':i,'display_name':name,'entity_type':typ,'authority_id':key,'source_database':source,'reactome_id':'','membership_gene_count':len(genes),'roles':role,'definition_version':'PTMsigDB 2.0.0; HGNC 2026-11-05','systematic_source_id':'|'.join(genes),'source_url':'https://www.gsea-msigdb.org/gsea/msigdb/ptmsig_genesets.jsp','qualifier':qualifier}
  for h in genes:
   rr=hrows.loc[h];component.append({'canonical_id':i,'hgnc_id':h,'gene_symbol':rr.symbol,'ensembl_gene_id':rr.ensembl_gene_id,'uniprot_accessions':rr.uniprot_ids,'component_role':'enzyme_or_enzyme_complex_subunit','qualifier':qualifier})
 return i
for r in D.itertuples():
 name=r.source_readout_id.split('_',1)[1];known=[];qual='';evidence='';parents=[]
 if r.source_family.startswith('KINASE'):
  if name in fusions:
   known=[gene([t]) for t in fusions[name]];assert all(known);key='KINASE_FUSION:'+':'.join(known);display='::'.join(hrows.loc[h,'symbol'] for h in known)+' kinase program';typ='fusion_kinase_program';evidence='Source explicitly names the same ordered gene fusion; genes resolved through HGNC'
  elif '-' in name and '/' not in name and ':' not in name:
   parts=name.split('-');stripped=[re.sub(r'p(?:25|35)$','',t) for t in parts];known=[gene([t]) for t in stripped];qual='p25' if name.endswith('p25') else 'p35' if name.endswith('p35') else ''
   if all(known):
    key='KINASE_COMPLEX:'+'+'.join(sorted(known))+(':'+qual if qual else '');display='?'.join(hrows.loc[h,'symbol'] for h in known)+((' '+qual) if qual else '')+' kinase program';typ='kinase_complex_program';evidence='Explicit source complex; HGNC-resolved subunits; processed subunit retained'
   else:known=[]
  else:
   h=gene(re.split(r'[/\.]',name));known=[h] if h else []
   if known:
    qual=qualifiers.get(name,'');key='KINASE_ACTIVITY:'+h+(':'+qual if qual else '');display=hrows.loc[h,'symbol']+((' '+qual) if qual else '')+' kinase program';typ='isoform_kinase_program' if qual else 'kinase_substrate_program';evidence='HGNC approved, previous and alias symbols; explicit isoform qualifier retained'
  if not known:
   key='PTMSIGDB:2.0:SOURCE_KINASE_PROGRAM:'+r.source_readout_id;display=name+' kinase program';typ='source_kinase_program';evidence='Named source enzyme program';unresolved.append({'source_readout_id':r.source_readout_id,'name':name})
  new_id=add_entity(key,display,typ,'State',known,'PTMsigDB; HGNC',qual)
  if known and (len(known)>1 or qual):
   # Only the catalytic enzyme has a kinase-activity parent; cyclins and other partners stay molecular components.
   catalytic=[known[-1]] if typ=='fusion_kinase_program' else [known[0]]
   for h in catalytic:
    parent=add_entity('KINASE_ACTIVITY:'+h,hrows.loc[h,'symbol']+' kinase program','kinase_substrate_program','State',[h],'HGNC')
    hier.append({'source_id':new_id,'target_id':parent,'relation':'has_molecular_component','evidence':'Explicit source enzyme construct or isoform'})
 else:
  external=r.external_authority_ids
  key='WIKIPATHWAYS:'+external if re.fullmatch(r'WP\d+',external) else 'PTMSIGDB:2.0:SOURCE_PATHWAY_PROGRAM:'+r.source_readout_id
  display=name.replace('_',' ');typ='pathway' if external.startswith('WP') else 'phospho_response_program';new_id=add_entity(key,display,typ,'Pathway',[],'WikiPathways; PTMsigDB' if external.startswith('WP') else 'PTMsigDB')
  evidence='Exact WikiPathways authority in source definition' if external.startswith('WP') else 'Source-specific directional phosphorylation response definition'
 bind.append({'source_readout_id':r.source_readout_id,'canonical_id':new_id,'source_family':r.source_family,'external_authority_ids':r.external_authority_ids,'entity_type':typ,'evidence':evidence})
 redirect.append({'previous_internal_id':OLD[r.source_readout_id],'canonical_id':new_id,'relation':'same_biological_identity','evidence':evidence});al.append({'canonical_id':new_id,'alias':r.source_readout_id,'alias_type':'PTMsigDB_source_readout','source_database':'PTMsigDB'})
 B.loc[len(B)]={'source_target_id':r.source_readout_id,'source_column':r.source_readout_id,'canonical_id':new_id,'source_name':name,'source_database':'PTMsigDB','relation':'readout_of','identity_evidence':evidence}
N=pd.DataFrame(new.values());used=set(x['canonical_id'] for x in bind);N['has_scored_readout']=N.canonical_id.isin(used);M['has_scored_readout']=True;M=pd.concat([M,N],ignore_index=True).fillna('');assert M.canonical_id.is_unique;assert B.source_target_id.is_unique
# Correct the external source identifier without changing an existing public internal ID.
for r in json.loads((R/'sources/KEGG_CURATED_NETWORK_XREFS.json').read_text())['records']:
 q=B.source_target_id.eq(r['source_name'])|B.source_name.eq(r['source_name']);ids=B.loc[q,'canonical_id'].unique()
 for i in ids:M.loc[M.canonical_id.eq(i),['authority_id','systematic_source_id','source_url']]=['KEGG_NETWORK:'+r['network_id'],r['network_id'],r['source_url']]
RED=pd.concat([RED,pd.DataFrame(redirect)],ignore_index=True);assert not RED.previous_internal_id.duplicated().any();ALIASES=pd.concat([ALIASES,pd.DataFrame(al)],ignore_index=True).drop_duplicates()
for fn,df in [('TARGET_MASTER.tsv',M),('SOURCE_ID_BINDINGS.tsv',B),('ID_REDIRECTS.tsv',RED),('SOURCE_ALIASES.tsv',ALIASES),('PTM_READOUT_BINDINGS.tsv',pd.DataFrame(bind)),('KINASE_MOLECULAR_COMPONENTS.tsv',pd.DataFrame(component)),('KINASE_CONSTRUCT_RELATIONS.tsv',pd.DataFrame(hier).drop_duplicates()),('SOURCE_KINASE_PROGRAMS.tsv',pd.DataFrame(unresolved))]:df.to_csv(OUT/fn,sep='\t',index=False)
lookup=dict(zip(RED.previous_internal_id,RED.canonical_id))
def redirect_id(x):
 seen=set()
 while x in lookup:
  assert x not in seen;seen.add(x);x=lookup[x]
 return x
reg=pd.read_parquet(SRC/'SCORE_REGISTRY.parquet');reg['canonical_id']=reg.canonical_id.map(redirect_id);assert set(reg.canonical_id)<=set(M.canonical_id);reg.to_parquet(OUT/'SCORE_REGISTRY.parquet',index=False,compression='zstd')
# Keep independently curated cross-database relations at their stated granularity.
external={}
for r in M.itertuples():
 if r.reactome_id:external[('reactome',r.reactome_id)]=r.canonical_id
 if r.authority_id.startswith('WIKIPATHWAYS:'):external[('wikipathways',r.authority_id.split(':',1)[1])]=r.canonical_id
cross=[]
for f in ['compath_wikipathways_reactome.csv','compath_kegg_reactome.csv','compath_kegg_wikipathways.csv']:
 for r in pd.read_csv(R/'sources/crossdatabase'/f).to_dict('records'):
  l=external.get((r['Source Resource'],r['Source ID']));rr=external.get((r['Target Resource'],r['Target ID']))
  if l or rr:cross.append({'source_canonical_id':l or '', 'source_resource':r['Source Resource'],'source_external_id':r['Source ID'],'source_name':r['Source Name'],'relation':r['Mapping Type'],'target_canonical_id':rr or '', 'target_resource':r['Target Resource'],'target_external_id':r['Target ID'],'target_name':r['Target Name'],'mapping_source':'ComPath'})
pd.DataFrame(cross).drop_duplicates().to_csv(OUT/'CURATED_EXTERNAL_RELATIONS.tsv',sep='\t',index=False)
# Existing source IDs and their meanings remain recoverable through redirects.
assert len(bind)==624;assert all(x in set(M.canonical_id) for x in pd.DataFrame(bind).canonical_id)
rec={'target_host':'149','identities_with_readouts':int(M.has_scored_readout.eq(True).sum()),'identities_including_construct_components':len(M),'pathway_identities_with_readouts':int((M.has_scored_readout.eq(True)&M.roles.str.contains('Pathway')).sum()),'state_identities_with_readouts':int((M.has_scored_readout.eq(True)&M.roles.str.contains('State')).sum()),'ptm_readouts':len(bind),'ptm_biological_identities':len(used),'kinase_source_programs_without_unique_gene_resolution':unresolved,'source_bindings':len(B),'id_redirects':len(RED),'hgnc_sha256':sha(R/'sources/hgnc_complete_set_20261109.tsv'),'parent_registry_sha256':sha(SRC/'COMPLETE.json')};js(OUT/'COMPLETE.json',rec);print(json.dumps(rec,ensure_ascii=False),flush=True)
