
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import sys,socket,json,time,zipfile,argparse,hashlib
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'shared_relation_demo_r1';sys.path.insert(0,str(R/'source_integration/repo'))
from shared_core_adjustment.design import design
from shared_core_adjustment.adjustment_policy import resolve_policy
from block_associations import associate,bh
from residual_estimability import residual_estimability
CFG=json.loads((O/'SCORING_ADJUSTMENT_CONFIG.json').read_text());assert CFG['cancer_type']=='list' and CFG['cancer_list']==['BRCA','LUAD','COAD']
RED=dict(pd.read_csv(R/'canonical_targets_r6/ID_REDIRECTS.tsv',sep='\t')[['previous_internal_id','canonical_id']].itertuples(index=False,name=None))
def canon(k):
 seen=set()
 while k in RED:
  assert k not in seen;seen.add(k);k=RED[k]
 return k
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def save(path, /, **arrays):
 tmp=Path(str(path)+'.tmp')
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for k,v in arrays.items():
   with z.open(k+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(v),allow_pickle=False)
 tmp.replace(path)
def direct_members(lncs):
 d=pd.read_csv(R/'state_relation_evidence_scope_adjusted_r2/DIRECT_RNA_MEMBERS.tsv',sep='\t');d=d[~d.state_id.str.startswith('PROGENY:')].copy()
 a=pd.read_csv(R/'common_rna_scores_r1/COMMON_GENE_ANNOTATION.tsv',sep='\t');a['lncrna_id']='LNC:'+a.gene_id;a=a[a.lncrna_id.isin(lncs)]
 m=pd.read_csv(R/'progeny_standard_r1/OFFICIAL_TOP100_MODEL.tsv',sep='\t',index_col=0);parts=[d]
 for pathway in m:
  sy=set(m.index[m[pathway].ne(0)]);f=a[a.reference_symbol.isin(sy)].copy();f['state_id']='PROGENY:'+pathway;f['source_symbol']=f.reference_symbol;f['scoring_role']='OFFICIAL_TOP100_NONZERO_WEIGHT';parts.append(f[d.columns])
 return pd.concat(parts,ignore_index=True).drop_duplicates()
def work(cancer,context,layers):
 if cancer not in CFG['cancer_list']:raise ValueError('OUTSIDE_USER_THREE_CANCER_SCOPE')
 start=time.monotonic();base=R/'numerical_relation_evidence_r1'/cancer;old=base/context;dest=O/'TCGA'/cancer/context;dest.mkdir(exist_ok=True,parents=True)
 axes=json.loads((base/'AXES.json').read_text());patients=axes['patient_ids'];lncs=axes['lncrna_ids'];split=json.loads((old/'SPLIT.json').read_text());sp=split['source_patients'];lp=split['label_patients'];assert not set(sp)&set(lp)
 idx=pd.Index(patients);si=idx.get_indexer(sp);yi=idx.get_indexer(lp);assert min(si.min(),yi.min())>=0
 cov=pd.read_parquet(O/'cohorts'/f'TCGA_{cancer}'/'COVARIATES.parquet');assert set(cov.patient_id)==set(patients)
 X=pd.read_parquet(R/'common_rna_scores_r1/aligned'/f'TCGA_{cancer}'/'LNC_LOGTPM_FULL.parquet').reindex(index=patients,columns=lncs).to_numpy(float)
 cn=R/'full_lnc_local_cnv_r1'/cancer;ri=pd.Index(json.loads((cn/'patient_ids.json').read_text())).get_indexer(patients);ci=pd.Index(json.loads((cn/'lncrna_ids.json').read_text())).get_indexer(lncs);assert min(ri.min(),ci.min())>=0
 C=np.load(cn/'lncrna_signed_mean.npy',mmap_mode='r')[np.ix_(ri,ci)].astype(float);call=np.load(cn/'lncrna_callable.npy',mmap_mode='r')[np.ix_(ri,ci)].astype(bool)
 direct=direct_members(lncs);js(dest/'AXES.json',axes);js(dest/'SPLIT.json',split);direct.to_csv(dest/'DIRECT_RNA_MEMBERS.tsv',sep='\t',index=False)
 for layer in layers:
  out=dest/layer;out.mkdir(exist_ok=True)
  if (out/'COMPLETE.json').exists():
   done=json.loads((out/'COMPLETE.json').read_text());assert done['config_sha256']==sha(O/'SCORING_ADJUSTMENT_CONFIG.json')
   for filename,digest in done['files'].items():assert sha(out/filename)==digest,filename
   print('REUSE_VERIFIED',cancer,context,layer,flush=True);continue
  tick=time.monotonic();md=pd.read_csv(old/layer/'TARGETS.tsv',sep='\t');md['canonical_id']=md.canonical_id.map(canon);scores=pd.read_parquet(old/layer/'PATIENT_S.parquet').reindex(patients);assert scores.columns.tolist()==md.target_id.tolist();reference={}
  if layer=='STATE':
   raw=pd.read_parquet(R/'progeny_standard_r1/cohorts'/f'TCGA_{cancer}'/'RAW_PROGENY.parquet').reindex(patients)
   assert set(raw.columns)<=set(scores.columns);mu=raw.iloc[si].mean();sd=raw.iloc[si].std(ddof=1);scores.loc[:,raw.columns]=(raw-mu)/sd.mask(sd.le(1e-12))
   reference={'method':'OFFICIAL_PROGENY_TOP100','source_patients':sp,'means':mu.to_dict(),'sd':sd.to_dict(),'raw_sha256':sha(R/'progeny_standard_r1/cohorts'/f'TCGA_{cancer}'/'RAW_PROGENY.parquet')};js(out/'PROGENY_REFERENCE.json',reference)
  scores.to_parquet(out/'PATIENT_S.parquet',compression='zstd');md.to_csv(out/'TARGETS.tsv',sep='\t',index=False);Y=scores.to_numpy(float);shape=(len(lncs),len(md));stats={k:np.full(shape,np.nan) for k in ['r','p']};stats.update(n=np.zeros(shape,np.int32),rank=np.zeros(shape,np.int16),available=np.zeros(shape,bool));groups={}
  for j,row in md.iterrows():
   target=row.source_target_id if layer=='STATE' else '__pathway__';_,policy=resolve_policy(target,contract=CFG['adjustment_contract']);groups.setdefault(json.dumps(policy,sort_keys=True),[]).append(j)
  reports=[]
  for jj in groups.values():
   row=md.iloc[jj[0]];target=row.source_target_id if layer=='STATE' else '__pathway__';family=row['family'] if layer=='STATE' else 'pathway'
   Z,report=design(cov,patients,sp,family=family,target_id=target,cfg=CFG,admission_cohorts={'source':sp,'label':lp})
   if target not in ('TumorPurity','ABSOLUTE_purity'):assert 'purity' in report['encoded'] or 'purity' in report.get('redundant',{})
   sub=associate(X,Y[:,jj],C,call,Z,yi,min_n=12,compute_fdr=False);keep,codes,numeric=residual_estimability(X,Y[:,jj],C,call,Z,yi,sub['available'],min_n=12,previous_n=sub['n'],previous_rank=sub['rank'])
   sub['available']=keep;sub['r'][~keep]=np.nan;sub['p'][~keep]=np.nan
   for k in stats:stats[k][:,jj]=sub[k]
   reports.append({'targets':md.source_target_id.iloc[jj].tolist(),'report':report,'numeric':numeric})

  excluded=0
  if layer=='STATE':
   ix=pd.Index(lncs).get_indexer(direct.lncrna_id);jx=pd.Index(md.source_target_id).get_indexer(direct.state_id);ok=(ix>=0)&(jx>=0);ix=ix[ok];jx=jx[ok];excluded=len(ix)
   stats['available'][ix,jx]=False;stats['r'][ix,jx]=np.nan;stats['p'][ix,jx]=np.nan
  if layer=='RNA':
   mixed=pd.read_csv(R/'rna_relation_evidence_direct_excluded_r1/MIXED_CODING_LNC_SYMBOLS.tsv',sep='\t');ix=pd.Index(lncs).get_indexer('LNC:'+mixed.gene_id);ix=ix[ix>=0];excluded=len(ix)*len(md)
   stats['available'][ix,:]=False;stats['r'][ix,:]=np.nan;stats['p'][ix,:]=np.nan
  stats['q']=bh(stats['p']);strict=stats['available']&(np.abs(stats['r'])>=.20)&(stats['q']<=.05);weak=stats['available']&(np.abs(stats['r'])>=.15)&(stats['q']<=.10)&~strict
  save(out/'ASSOCIATIONS.npz',**stats,strict_statistical_support=strict,weak_statistical_support=weak)
  ix,jx=np.where(strict|weak);df=pd.DataFrame({'lncrna_id':np.asarray(lncs)[ix],'canonical_id':md.canonical_id.to_numpy()[jx],'source_target_id':md.source_target_id.to_numpy()[jx],'rho':stats['r'][ix,jx],'p':stats['p'][ix,jx],'q':stats['q'][ix,jx],'n':stats['n'][ix,jx],'statistical_support':np.where(strict[ix,jx],'strict','weak'),'evidence_layer':md.measurement_layer.to_numpy()[jx]});df.to_parquet(out/'SUPPORTED_RELATIONS.parquet',index=False,compression='zstd')
  counts=md.copy();counts['eligible_lncrna_pairs']=stats['available'].sum(0);counts['strict']=strict.sum(0);counts['weak']=weak.sum(0);counts.to_csv(out/'TARGET_COUNTS.tsv',sep='\t',index=False);js(out/'ADJUSTMENT.json',reports)
  row={'target_host':'149','dataset':'TCGA','cancer_id':cancer,'context':context,'layer':layer,'patients_source':len(si),'patients_label':len(yi),'targets':len(md),'eligible_relations':int(stats['available'].sum()),'strict_relations':int(strict.sum()),'weak_relations':int(weak.sum()),'direct_member_excluded_pairs':excluded,'seconds':time.monotonic()-tick,'config_sha256':sha(O/'SCORING_ADJUSTMENT_CONFIG.json'),'source_score_sha256':sha(old/layer/'PATIENT_S.parquet'),'progeny_replaced':layer=='STATE','files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}};js(out/'COMPLETE.json',row);print('RELATIONS_COMPLETE',cancer,context,layer,row['eligible_relations'],row['strict_relations'],row['weak_relations'],'seconds',round(row['seconds'],1),flush=True)
  del stats,Y,scores,df,strict,weak
 if all((dest/k/'COMPLETE.json').exists() for k in ['RNA','STATE','PROTEIN','PHOSPHO']):js(dest/'COMPLETE.json',{'target_host':'149','cancer_id':cancer,'context':context,'status':'COMPLETE','layers':['RNA','STATE','PROTEIN','PHOSPHO'],'seconds':time.monotonic()-start})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--cancer',required=True,choices=CFG['cancer_list']);a.add_argument('--context',default='sealed_test',choices=['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test']);a.add_argument('--layers',nargs='+',choices=['RNA','STATE','PROTEIN','PHOSPHO'],default=['RNA','STATE','PROTEIN','PHOSPHO']);args=a.parse_args();work(args.cancer,args.context,args.layers)
