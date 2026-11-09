"""Materialize fixed-background RNA scoring components from measured TCGA/CPTAC RNA."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import socket,sys,os,json,hashlib,time,traceback
import pandas as pd,numpy as np
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'common_rna_scores_r1';REPO=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/repo')
sys.path[:0]=[str(REPO),str(REPO/'scripts'),'/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps']
from cache_m0_joint_scores import cache_case
from cla_engine.common import sha
CP=Path('/dell_2/DSC/CancerLncAtlas/inputs/cptac_20261104_r1/aligned_r4')
ANN=Path('/dsk2/user/dengsc/cla_m0_lasso_20261104_r1/tcga_full_tpm_r1/BRCA/GENE_ANNOTATION.tsv')
REF=REPO/'config/cptac_149_sources_20261104.json'
DEFS=Path('/dsk2/user/dengsc/cla_state_signature_update_20261104_r1/sources')
XREF=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/XCELL_FROZEN_REFERENCE.json')
CFG=Path('/dsk2/user/dengsc/cla_state_signature_update_20261104_r1/CURRENT_149_CONFIG.json')
def js(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False))
def prepare_axis():
 O.mkdir(exist_ok=False);(O/'aligned').mkdir();(O/'cache').mkdir()
 refs=json.loads(REF.read_text())['references']
 # Preserve original source hashes and scoring code.
 for name in ['pathway_gmt','historical_networks','senmayo','mcp','progeny','dorothea','malta']:
  v=refs[name];assert sha(v['path'])==v['sha256'],name
 cpcomplete=json.loads((CP/'COMPLETE.json').read_text());aa=pd.read_parquet(CP/'FEATURE_ALIGNMENT.parquet')
 assert sha(CP/'FEATURE_ALIGNMENT.parquet')==cpcomplete['files']['FEATURE_ALIGNMENT.parquet']['sha256']
 a=aa.loc[aa.layer.eq('rna')&aa.reference_presence.eq('GENCODE_V36'),['gene_id','reference_symbol','gene_biotype']].drop_duplicates()
 assert not a.gene_id.duplicated().any();a=a.set_index('gene_id');ta=pd.read_csv(ANN,sep='\t').set_index('gene_id')
 genes=set(ta.index)&set(a.index);cohorts=['BRCA','CCRCC','COAD','GBM','HNSCC','LSCC','LUAD','OV','PDAC','UCEC']
 for cohort in cohorts:
  ids=pd.read_parquet(CP/f'matrices/{cohort}/rna_TumorNormal.parquet',columns=['feature_id']).feature_id
  assert ids.is_unique;genes&=set(ids)
 genes=sorted(genes);a=a.loc[genes]
 assert len(genes)>50000;assert (a.gene_biotype==ta.loc[genes,'gene_type']).all()
 assert (a.reference_symbol.fillna('')==ta.loc[genes,'gene_symbol'].fillna('')).all()
 a.to_csv(O/'COMMON_GENE_ANNOTATION.tsv',sep='\t');pc=sorted(set(a.loc[a.gene_biotype.eq('protein_coding')&a.reference_symbol.notna(),'reference_symbol']))
 pd.DataFrame({'symbol':pc}).to_csv(O/'PROTEIN_CODING_SYMBOLS.tsv',sep='\t',index=False)
 rec=json.loads((R/'full_tcga_tpm_r1/COMPLETE.json').read_text())['records']
 cases=[{'dataset':'TCGA','cohort':c,'source':str(Path(v['path'])/'GENE_TPM.tsv.gz'),'sha256':v['receipt']['outputs']['tpm_sha256']} for c,v in rec.items()]
 for c in cohorts:
  rel=f'matrices/{c}/rna_TumorNormal.parquet';cases.append({'dataset':'CPTAC','cohort':c,'source':str(CP/rel),'sha256':cpcomplete['files'][rel]['sha256']})
 js(O/'LAUNCH_PREFLIGHT.json',{'target_host':'149','action':'COMMON_RNA_COMPONENTS','cases':cases,'common_genes':len(genes),'coding_symbols':len(pc),'annotation':'GENCODE_v36','reference_sha256':sha(REF),'score_code_sha256':sha(REPO/'cc_hhgt/m0_joint_scores.py'),'cache_code_sha256':sha(REPO/'scripts/cache_m0_joint_scores.py'),'expression_filter':'NONE','scoring_background':'MEASUREMENT_AXIS_INTERSECTION_ALL_TCGA_CPTAC_COHORTS','learned_transforms':'SOURCE_TRAIN_PATIENTS_ONLY_IN_DOWNSTREAM_SPLIT_MATERIALIZER'})
 return cases

def work(case):
 key=case['dataset']+'_'+case['cohort'];d=O/'aligned'/key;d.mkdir(exist_ok=False);t=time.monotonic();f=Path(case['source']);assert sha(f)==case['sha256'],key
 a=pd.read_csv(O/'COMMON_GENE_ANNOTATION.tsv',sep='\t').set_index('gene_id');genes=a.index.tolist()
 if case['dataset']=='TCGA':
  full=pd.read_csv(f,sep='\t').set_index('gene_id');ta=pd.read_csv(ANN,sep='\t').set_index('gene_id');lnc=ta.index[ta.gene_type.eq('lncRNA')]
  x=np.log2(full.reindex(lnc).T+1);x.columns='LNC:'+x.columns;x.index.name='patient_id';x.to_parquet(d/'LNC_LOGTPM_FULL.parquet',compression='zstd')
 else:
  log=pd.read_parquet(f).set_index('feature_id');log=log.loc[:,[c for c in log if c.endswith('::T')]];log.columns=log.columns.str.removesuffix('::T');full=np.expm1(log*np.log(2))
  allann=pd.read_parquet(CP/'FEATURE_ALIGNMENT.parquet');lnc=allann.loc[allann.layer.eq('rna')&allann.gene_biotype.eq('lncRNA'),'gene_id'].drop_duplicates();lnc=[g for g in lnc if g in log.index]
  x=log.loc[lnc].T;x.columns='LNC:'+x.columns;x.index.name='patient_id';x.to_parquet(d/'LNC_LOGTPM_FULL.parquet',compression='zstd')
 if (full.to_numpy()<0).any() or np.isinf(full.to_numpy()).any():raise ValueError('Invalid TPM '+key)
 v=full.loc[genes];log=np.log2(v+1).T;log.index.name='patient_id';log.to_parquet(d/'GENE_LOGTPM.parquet',compression='zstd')
 usable=a.reference_symbol.notna()&a.reference_symbol.ne('');z=v.loc[a.index[usable]].copy();z.index=a.loc[usable,'reference_symbol']
 summed=z.groupby(level=0,sort=True).sum(min_count=1);missing=z.isna().groupby(level=0,sort=True).any();summed=summed.mask(missing)
 e=np.log2(summed+1).T;e.index.name='patient_id';e.to_parquet(d/'SYMBOL_LOGTPM.parquet',compression='zstd');(d/'PROTEIN_CODING_SYMBOLS.tsv').write_bytes((O/'PROTEIN_CODING_SYMBOLS.tsv').read_bytes())
 js(d/'COMPLETE.json',{'target_host':'149','dataset':case['dataset'],'cohort':case['cohort'],'source':case,'patients':len(e),'full_lncrnas':x.shape[1],'common_genes':len(genes),'missing_rna_cells':int(log.isna().sum().sum()),'files':{p.name:sha(p) for p in d.iterdir() if p.is_file()}})
 cache_case(O/'aligned',key,O/'cache',REF,DEFS,XREF,CFG)
 row={'dataset':case['dataset'],'cohort':case['cohort'],'patients':len(e),'full_lncrnas':x.shape[1],'common_genes':len(genes),'missing_rna_cells':int(log.isna().sum().sum()),'seconds':time.monotonic()-t,'aligned_sha256':sha(d/'COMPLETE.json'),'cache_sha256':sha(O/'cache'/key/'COMPLETE.json')}
 print('CASE_COMPLETE',row,flush=True);return key,row
if __name__=='__main__':
 cases=prepare_axis();records={};failures={}
 with ProcessPoolExecutor(max_workers=2) as ex:
  fs={ex.submit(work,c):c for c in cases}
  for f in as_completed(fs):
   c=fs[f];key=c['dataset']+'_'+c['cohort']
   try:key,row=f.result();records[key]=row
   except Exception as e:failures[key]=repr(e);print('CASE_FAILED',key,traceback.format_exc(),flush=True)
   js(O/'PROGRESS.json',{'target_host':'149','completed':len(records),'total':len(cases),'records':records,'failures':failures})
 js(O/'COMPLETE.json',{'target_host':'149','status':'COMPLETE' if not failures else 'FAILED','records':records,'failures':failures})
