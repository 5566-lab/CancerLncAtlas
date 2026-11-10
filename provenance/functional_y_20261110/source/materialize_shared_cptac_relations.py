from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import socket,sys,json,time,argparse
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'shared_relation_demo_r1';sys.path.insert(0,str(R/'source_integration/repo'))
from materialize_shared_tcga_relations import save,js,sha,direct_members,canon
from shared_core_adjustment.design import design
from shared_core_adjustment.adjustment_policy import resolve_policy
from block_associations import associate,bh
from residual_estimability import residual_estimability
CFG=json.loads((O/'SCORING_ADJUSTMENT_CONFIG.json').read_text());assert CFG['cancer_list']==['BRCA','LUAD','COAD'];LAYERS=['RNA','STATE','PROTEIN','PHOSPHO','METHYLATION']
def work(c,context,layers,output_dataset='CPTAC',exclude_label_patients=()):
 assert c in CFG['cancer_list'];base=O/'CPTAC_SCORES'/c;source=base/context;dest=O/output_dataset/c/context;dest.mkdir(parents=True,exist_ok=True);axes=json.loads((base/'AXES.json').read_text());patients=axes['patient_ids'];lncs=axes['lncrna_ids'];split=json.loads((source/'SPLIT.json').read_text());sp=split['source_patients'];lp=split['label_patients'];excluded=sorted(set(lp)&set(exclude_label_patients));split['original_label_patients']=list(lp);lp=[pid for pid in lp if pid not in set(exclude_label_patients)];split['label_patients']=lp;split['excluded_overlap_uncertain_patients']=excluded;assert not set(sp)&set(lp);yi=pd.Index(patients).get_indexer(lp);assert yi.min()>=0
 covpath=O/'cohorts'/('CPTAC_'+c)/'COVARIATES.parquet';cov=pd.read_parquet(covpath);assert set(cov.patient_id)==set(patients);x=pd.read_parquet(R/'common_rna_scores_r1/aligned'/('CPTAC_'+c)/'LNC_LOGTPM_FULL.parquet').reindex(index=patients,columns=lncs).to_numpy(float);cn=O/'cptac_local_cnv'/c;ri=pd.Index(json.loads((cn/'patient_ids.json').read_text())).get_indexer(patients);ci=pd.Index(json.loads((cn/'lncrna_ids.json').read_text())).get_indexer(lncs);assert min(ri.min(),ci.min())>=0;C=np.load(cn/'lncrna_signed_mean.npy')[np.ix_(ri,ci)].astype(float);call=np.load(cn/'lncrna_callable.npy')[np.ix_(ri,ci)].astype(bool);assert np.array_equal(np.isfinite(C),call);direct=direct_members(lncs);js(dest/'AXES.json',axes);js(dest/'SPLIT.json',split);direct.to_csv(dest/'DIRECT_RNA_MEMBERS.tsv',sep='\t',index=False)
 for layer in layers:
  out=dest/layer;out.mkdir(exist_ok=True)
  if (out/'COMPLETE.json').exists():
   done=json.loads((out/'COMPLETE.json').read_text());assert done.get('excluded_overlap_uncertain_patients',[])==split['excluded_overlap_uncertain_patients'];assert done['config_sha256']==sha(O/'SCORING_ADJUSTMENT_CONFIG.json') and done['covariates_sha256']==sha(covpath)
   for filename,digest in done['files'].items():assert sha(out/filename)==digest
   print('CPTAC_RELATION_REUSE',c,context,layer,flush=True);continue
  tick=time.monotonic();src=source/layer;receipt=json.loads((src/'SCORES_COMPLETE.json').read_text())
  for filename,digest in receipt['files'].items():assert sha(src/filename)==digest
  md=pd.read_csv(src/'TARGETS.tsv',sep='\t');scores=pd.read_parquet(src/'PATIENT_S.parquet').reindex(patients);assert list(scores)==md.target_id.tolist();md.canonical_id=md.canonical_id.map(canon);Y=scores.to_numpy(float);shape=(len(lncs),len(md));stats={k:np.full(shape,np.nan) for k in ['r','p']};stats.update(n=np.zeros(shape,np.int32),rank=np.zeros(shape,np.int16),available=np.zeros(shape,bool));groups={}
  for j,row in md.iterrows():
   target=row.source_target_id if layer=='STATE' else '__pathway__';_,policy=resolve_policy(target,contract=CFG['adjustment_contract']);groups.setdefault(json.dumps(policy,sort_keys=True),[]).append(j)
  reports=[]
  for jj in groups.values():
   row=md.iloc[jj[0]];target=row.source_target_id if layer=='STATE' else '__pathway__';family=row['family'] if layer=='STATE' else 'pathway';Z,report=design(cov,patients,sp,family=family,target_id=target,cfg=CFG,admission_cohorts={'source':sp,'label':lp})
   if target not in ['TumorPurity','ABSOLUTE_purity']:assert 'purity' in report['encoded'] or 'purity' in report.get('redundant',{})
   sub=associate(x,Y[:,jj],C,call,Z,yi,min_n=12,compute_fdr=False);keep,codes,numeric=residual_estimability(x,Y[:,jj],C,call,Z,yi,sub['available'],min_n=12,previous_n=sub['n'],previous_rank=sub['rank']);sub['available']=keep;sub['r'][~keep]=np.nan;sub['p'][~keep]=np.nan
   for k in stats:stats[k][:,jj]=sub[k]
   reports.append({'targets':md.source_target_id.iloc[jj].tolist(),'report':report,'numeric':numeric})
  excluded=0
  if layer=='STATE':
   ix=pd.Index(lncs).get_indexer(direct.lncrna_id);jx=pd.Index(md.source_target_id).get_indexer(direct.state_id);ok=(ix>=0)&(jx>=0);ix=ix[ok];jx=jx[ok];excluded=len(ix);stats['available'][ix,jx]=False;stats['r'][ix,jx]=np.nan;stats['p'][ix,jx]=np.nan
  if layer=='RNA':
   mixed=pd.read_csv(R/'rna_relation_evidence_direct_excluded_r1/MIXED_CODING_LNC_SYMBOLS.tsv',sep='\t');ix=pd.Index(lncs).get_indexer('LNC:'+mixed.gene_id);ix=ix[ix>=0];excluded=len(ix)*len(md);stats['available'][ix,:]=False;stats['r'][ix,:]=np.nan;stats['p'][ix,:]=np.nan
  stats['q']=bh(stats['p']);strict=stats['available']&(abs(stats['r'])>=.20)&(stats['q']<=.05);weak=stats['available']&(abs(stats['r'])>=.15)&(stats['q']<=.10)&~strict;save(out/'ASSOCIATIONS.npz',**stats,strict_statistical_support=strict,weak_statistical_support=weak);ix,jx=np.where(strict|weak);df=pd.DataFrame({'lncrna_id':np.asarray(lncs)[ix],'canonical_id':md.canonical_id.to_numpy()[jx],'source_target_id':md.source_target_id.to_numpy()[jx],'rho':stats['r'][ix,jx],'p':stats['p'][ix,jx],'q':stats['q'][ix,jx],'n':stats['n'][ix,jx],'statistical_support':np.where(strict[ix,jx],'strict','weak'),'evidence_layer':md.measurement_layer.to_numpy()[jx],'dataset':'CPTAC'});df.to_parquet(out/'SUPPORTED_RELATIONS.parquet',index=False,compression='zstd');scores.to_parquet(out/'PATIENT_S.parquet',compression='zstd');md.to_csv(out/'TARGETS.tsv',sep='\t',index=False);counts=md.copy();counts['eligible_lncrna_pairs']=stats['available'].sum(0);counts['strict']=strict.sum(0);counts['weak']=weak.sum(0);counts.to_csv(out/'TARGET_COUNTS.tsv',sep='\t',index=False);js(out/'ADJUSTMENT.json',reports)
  row={'target_host':'149','excluded_overlap_uncertain_patients':split['excluded_overlap_uncertain_patients'],'dataset':'CPTAC','cancer_id':c,'context':context,'layer':layer,'patients_source':len(sp),'patients_label':len(lp),'targets':len(md),'eligible_relations':int(stats['available'].sum()),'strict_relations':int(strict.sum()),'weak_relations':int(weak.sum()),'direct_member_excluded_pairs':excluded,'seconds':time.monotonic()-tick,'config_sha256':sha(O/'SCORING_ADJUSTMENT_CONFIG.json'),'covariates_sha256':sha(covpath),'local_cnv_complete_sha256':sha(cn/'COMPLETE.json'),'source_score_receipt_sha256':sha(src/'SCORES_COMPLETE.json'),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json'}};js(out/'COMPLETE.json',row);print('CPTAC_RELATIONS_COMPLETE',c,context,layer,row['eligible_relations'],row['strict_relations'],row['weak_relations'],flush=True);del stats,Y,scores,strict,weak,df
 if all((dest/k/'COMPLETE.json').exists() for k in LAYERS):js(dest/'COMPLETE.json',{'target_host':'149','dataset':'CPTAC','cancer_id':c,'context':context,'layers':LAYERS,'status':'COMPLETE'})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--cancer',choices=CFG['cancer_list'],required=True);a.add_argument('--context',choices=['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test'],default='train_label_fold2');a.add_argument('--layers',nargs='+',choices=LAYERS,default=LAYERS);a.add_argument('--output-dataset',choices=['CPTAC','CPTAC_DEDUPLICATED'],default='CPTAC');a.add_argument('--exclude-label-patient',nargs='*',default=[]);args=a.parse_args();work(args.cancer,args.context,args.layers,args.output_dataset,args.exclude_label_patient)
