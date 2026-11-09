"""Remove direct RNA score-member self associations, then recompute State FDR."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,socket,time,traceback,zipfile
import pandas as pd,numpy as np
import pyarrow.parquet as pq
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'state_relation_evidence_direct_excluded_r1';P=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/repo');sys.path[:0]=[str(P),'/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps']
from cla_engine.common import sha
from block_associations import bh
CTX=['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test']
def js(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False))
def save_npz(path,**arrays):
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for name,v in arrays.items():
   with z.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(v),allow_pickle=False)
def prepare():
 O.mkdir(exist_ok=False);js(O/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'STATE_DIRECT_RNA_MEMBER_EXCLUSION_AND_FDR','scope':'Explicit nonzero scoring inputs; patient scores and full lncRNA feature axes retained','training_started':False})
 ann=pd.read_csv(R/'common_rna_scores_r1/COMMON_GENE_ANNOTATION.tsv',sep='\t').fillna('');lnc=ann.loc[ann.gene_biotype.eq('lncRNA')];lookup=lnc.groupby('reference_symbol').gene_id.agg(list).to_dict();pairs=[]
 def add(state,symbol,role):
  for gid in lookup.get(symbol,[]):pairs.append({'state_id':state,'lncrna_id':'LNC:'+gid,'source_symbol':symbol,'scoring_role':role})
 net=pd.read_csv(R/'common_rna_scores_r1/cache/TCGA_BRCA/PROGRAM_NETWORK.tsv',sep='\t')
 for row in net.loc[net.weight.ne(0)].itertuples():add(row.state_id,row.gene,'NONZERO_PROGRAM_WEIGHT')
 defs=json.loads(Path('/dsk2/user/dengsc/cla_state_signature_update_20261104_r1/sources/SIGNIFINDER_EXACT_DEFINITIONS.json').read_text());simple=['Hypoxia_Buffa','CellCycle_Lundberg','MitoticIndex_Yang','CIN_Carter','DNArep_Kang','Glycolysis_Zhang','LipidMetabolism_Zheng','VEGF_Hu','IFN_Ayers','Tinflam_Ayers','TLS_Cabrita','LRRC15CAF_Dominguez','APM_Wang']
 for name in simple:
  target='signifinder::'+('CIN_Carter_70' if name=='CIN_Carter' else name)
  for row in defs[name]:
   if row.get('coeff') is None or float(row['coeff'])!=0:add(target,row['SYMBOL'],'SIGNATURE_OR_HOUSEKEEPING_COMPONENT')
 for name,klass,target in [('EMT_Miow','Epithelial-like','EMT_Miow_Epithelial'),('EMT_Miow','Mesenchymal-like','EMT_Miow_Mesenchymal'),('ECM_Chakravarthy','ECMup','ECM_Chakravarthy_up'),('ECM_Chakravarthy','ECMdown','ECM_Chakravarthy_down')]:
  for row in defs[name]:
   if row.get('class')==klass:add('signifinder::'+target,row['SYMBOL'],'SSGSEA_SIGNATURE_MEMBER')
 refs=json.loads((P/'config/cptac_149_sources_20261104.json').read_text())['references']
 for symbol in json.loads(Path(refs['senmayo']['path']).read_text())['SAUL_SEN_MAYO']['geneSymbols']:add('SenMayo',symbol,'SSGSEA_SIGNATURE_MEMBER')
 for symbol in ['C18orf54','CCT6P1','CEP72','POLE2','MCM4','HELLS','TERC','CCT6AP1','C21orf81','LOC81691','AC004381.6','ANKRD20A11P','EHMT2','LIN9','PAXIP1','TERT']:add('EXTEND',symbol,'TEL0MERASE_SIGNATURE_COMPONENT')
 gids=set(lnc.gene_id)
 for name,sheet,header in [('RNAss','mRNAsi',0),('EREG.EXPss','EREG-mRNAsi',1)]:
  w=pd.read_excel(refs['malta']['path'],sheet_name=sheet,header=header).dropna(subset=['ENSEMBL']);w['gene_id']=w.ENSEMBL.astype(str).str.replace(r'\.[0-9]+$','',regex=True)
  for row in w.loc[w.gene_id.isin(gids)&pd.to_numeric(w.Weight,errors='coerce').ne(0)].itertuples():pairs.append({'state_id':name,'lncrna_id':'LNC:'+row.gene_id,'source_symbol':'','scoring_role':'OCLR_WEIGHT_COMPONENT'})
 table=pd.DataFrame(pairs).drop_duplicates(['state_id','lncrna_id']);table.to_csv(O/'DIRECT_RNA_MEMBERS.tsv',sep='\t',index=False)
 # Explicitly examine symbol aggregation across biotypes in the common RNA background.
 coding=set(ann.loc[ann.gene_biotype.eq('protein_coding'),'reference_symbol']);mixed=lnc.loc[lnc.reference_symbol.isin(coding)];mixed.to_csv(O/'MIXED_CODING_LNC_SYMBOLS.tsv',sep='\t',index=False)
 js(O/'DEFINITION_RECEIPT.json',{'target_host':'149','direct_pairs':len(table),'state_targets':table.state_id.nunique(),'mixed_coding_lnc_symbols':len(mixed),'member_table_sha256':sha(O/'DIRECT_RNA_MEMBERS.tsv')});print('DIRECT_RNA_MEMBERS',len(table),table.state_id.nunique(),'mixed',len(mixed),flush=True)
 if len(mixed):
  md=pd.read_csv(R/'state_relation_evidence_demo_r1/BRCA/sealed_test/TARGETS.tsv',sep='\t')
  rna_states=md.loc[md.measurement_layer.str.startswith('RNA'),'source_target_id']
  extra=[{'state_id':state,'lncrna_id':'LNC:'+row.gene_id,'source_symbol':row.reference_symbol,'scoring_role':'MIXED_BIOTYPE_RNA_BACKGROUND_CONTRIBUTION'} for state in rna_states for row in mixed.itertuples()]
  table=pd.concat([table,pd.DataFrame(extra)],ignore_index=True).drop_duplicates(['state_id','lncrna_id']);table.to_csv(O/'DIRECT_RNA_MEMBERS.tsv',sep='\t',index=False)
  js(O/'DEFINITION_RECEIPT.json',{'target_host':'149','direct_pairs':len(table),'state_targets':table.state_id.nunique(),'mixed_coding_lnc_symbols':len(mixed),'mixed_symbol_handling':'Exclude this lncRNA from RNA score supervision; retain its features and non-RNA evidence','member_table_sha256':sha(O/'DIRECT_RNA_MEMBERS.tsv')})
  print('DIRECT_RNA_MEMBERS_RESOLVED',len(table),table.state_id.nunique(),flush=True)
def source(c):return R/('state_relation_evidence_demo_r1' if c in ['BRCA','LUAD','COAD'] else 'state_relation_evidence_catalog_r1')/c
def work(cancer):
 src=source(cancer);out=O/cancer;out.mkdir();axes=json.loads((src/'AXES.json').read_text());lncs=axes['lncrna_ids'];lp={x:i for i,x in enumerate(lncs)};direct=pd.read_csv(O/'DIRECT_RNA_MEMBERS.tsv',sep='\t');rows=[];js(out/'AXES.json',axes)
 for context in CTX:
  inp=src/context;dest=out/context;dest.mkdir();receipt=json.loads((inp/'COMPLETE.json').read_text());assert sha(inp/'ASSOCIATIONS.npz')==receipt['files']['ASSOCIATIONS.npz'];md=pd.read_csv(inp/'TARGETS.tsv',sep='\t');tp={x:j for j,x in enumerate(md.source_target_id)};exclude=direct.loc[direct.state_id.isin(tp)&direct.lncrna_id.isin(lp)].copy();ix=exclude.lncrna_id.map(lp).to_numpy(int);jx=exclude.state_id.map(tp).to_numpy(int)
  raw=np.load(inp/'ASSOCIATIONS.npz');stats={k:raw[k] for k in ['r','p','q','n','rank','available']};previous=int(stats['available'][ix,jx].sum());stats['available'][ix,jx]=False;stats['r'][ix,jx]=np.nan;stats['p'][ix,jx]=np.nan;stats['q']=bh(stats['p']);strict=stats['available']&(np.abs(stats['r'])>=.20)&(stats['q']<=.05);weak=stats['available']&(np.abs(stats['r'])>=.15)&(stats['q']<=.10)&~strict
  assert not np.isfinite(stats['p'][ix,jx]).any();assert not strict[ix,jx].any();assert not weak[ix,jx].any();save_npz(dest/'ASSOCIATIONS.npz',**stats,strict_statistical_support=strict,weak_statistical_support=weak,historical_state_rho_support=stats['available']&(np.abs(stats['r'])>=.20))
  ei,ej=np.where(strict|weak);pd.DataFrame({'lncrna_id':np.asarray(lncs)[ei],'canonical_id':md.canonical_id.to_numpy()[ej],'source_target_id':md.source_target_id.to_numpy()[ej],'rho':stats['r'][ei,ej],'p':stats['p'][ei,ej],'q':stats['q'][ei,ej],'n':stats['n'][ei,ej],'statistical_support':np.where(strict[ei,ej],'strict','weak'),'evidence_layer':md.measurement_layer.to_numpy()[ej]}).to_parquet(dest/'SUPPORTED_RELATIONS.parquet',compression='zstd')
  counts=pd.read_csv(inp/'TARGET_COUNTS.tsv',sep='\t');counts['eligible_lncrna_pairs']=stats['available'].sum(0);counts['strict_statistical_support']=strict.sum(0);counts['weak_statistical_support']=weak.sum(0);counts['direct_rna_members_excluded']=counts.source_target_id.map(exclude.groupby('state_id').size()).fillna(0).astype(int);counts.to_csv(dest/'TARGET_COUNTS.tsv',sep='\t',index=False)
  for name in ['PATIENT_S.parquet','SCORE_REFERENCE.json','TARGETS.tsv','ADJUSTMENT.json','SPLIT.json']:(dest/name).symlink_to(inp/name)
  exclude.to_csv(dest/'DIRECT_RNA_EXCLUSIONS.tsv',sep='\t',index=False);row={'cancer_id':cancer,'context':context,'state_targets':len(md),'direct_definition_pairs':len(exclude),'previously_estimable_self_pairs_removed':previous,'eligible_relations':int(stats['available'].sum()),'strict_statistical_support':int(strict.sum()),'weak_statistical_support':int(weak.sum())};js(dest/'COMPLETE.json',dict(row,target_host='149',source_complete_sha256=sha(inp/'COMPLETE.json'),files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}));rows.append(row);print('STATE_SELF_EXCLUSION_COMPLETE',row,flush=True)
 pd.DataFrame(rows).to_csv(out/'CONTEXT_SUMMARY.tsv',sep='\t',index=False);js(out/'COMPLETE.json',{'target_host':'149','cancer':cancer,'contexts':rows});return cancer,rows
if __name__=='__main__':
 prepare();a=pd.read_csv('/dell_2/DSC/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv',sep='\t');pending=set(a.cancer_id.unique());running={};results={};failures={};deadline=time.monotonic()+6*3600
 with ProcessPoolExecutor(max_workers=2) as pool:
  while pending or running:
   for c in sorted(pending,key=lambda x:(x not in ['BRCA','LUAD','COAD'],x)):
    if (source(c)/'COMPLETE.json').exists():running[pool.submit(work,c)]=c;pending.remove(c)
   for f,c in list(running.items()):
    if f.done():
     try:_,rows=f.result();results[c]=rows
     except Exception as err:failures[c]=repr(err);print('STATE_SELF_EXCLUSION_FAILED',c,traceback.format_exc(),flush=True)
     del running[f]
   js(O/'PROGRESS.json',{'target_host':'149','completed':sorted(results),'pending':sorted(pending),'running':sorted(running.values()),'failures':failures})
   if time.monotonic()>deadline:failures['source_wait']='Six-hour deadline';break
   if pending or running:time.sleep(10)
 pd.DataFrame([r for rows in results.values() for r in rows]).to_csv(O/'CANCER_CONTEXT_COUNTS.tsv',sep='\t',index=False);js(O/'COMPLETE.json',{'target_host':'149','status':'COMPLETE' if not failures else 'FAILED','results':results,'failures':failures})
