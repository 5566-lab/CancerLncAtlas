from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import socket,sys,json,time,argparse,hashlib
from datetime import datetime,timezone
import numpy as np,pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parent;O=R/'shared_relation_demo_r1';F=R/'functional_relation_y_fused_r1';assert socket.gethostname()=='149';sys.path.insert(0,str(R/'source_integration/repo'))
from relationship_fusion import empirical_fisher,duplicate_groups
from shared_core_adjustment.design import design
from shared_core_adjustment.adjustment_policy import resolve_policy
from block_associations import basis,bh
from materialize_shared_tcga_relations import save,canon
CFG=json.loads((O/'SCORING_ADJUSTMENT_CONFIG.json').read_text());CANCERS=['BRCA','LUAD','COAD'];CONTEXTS=['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test'];assert CFG['cancer_list']==CANCERS
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,d):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));q.replace(p)
def assay(value):
 v=str(value).upper()
 if 'PHOSPH' in v:return 4
 if 'PROTEIN' in v or 'PROTEOM' in v:return 2
 if 'RNA' in v or 'TRANSCRIPT' in v:return 1
 if 'METHYL' in v:return 8
 if 'COPY' in v or 'CNV' in v:return 16
 if any(s in v for s in ['VARIANT','SNV','MUTATION','SEQUENC']):return 32
 return 64

def run(cancer,context):
 assert cancer in CANCERS and context in CONTEXTS
 method_receipt=O/'fusion_method_r1/METHOD_ADOPTION.json'
 adoption=json.loads(method_receipt.read_text());assert adoption['status']=='ADOPT_EMPIRICALLY_CALIBRATED_FISHER' and adoption['algorithm_sha256']==sha(R/'relationship_fusion.py')
 F.mkdir(exist_ok=True);dest=F/cancer/context;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():
  d=json.loads((dest/'COMPLETE.json').read_text())
  for f,h in d['files'].items():assert sha(dest/f)==h,f
  assert d['algorithm_sha256']==sha(R/'relationship_fusion.py');print('FUSED_Y_REUSE',cancer,context,flush=True);return
 start=time.monotonic();js(dest/'LAUNCH_PREFLIGHT.json',{'target_host':'149','cancer_id':cancer,'context':context,'method':'EMPIRICALLY_CALIBRATED_FISHER','created_utc':datetime.now(timezone.utc).isoformat(),'method_adoption_sha256':sha(method_receipt)})
 master=pd.read_csv(R/'canonical_targets_r6/TARGET_MASTER.tsv',sep='\t');master=master[master.has_scored_readout.eq(True)].copy();assert master.canonical_id.is_unique;master=master.sort_values('canonical_id').reset_index(drop=True);ids=master.canonical_id.tolist();indices={x:i for i,x in enumerate(ids)}
 references=[];streams=[];splits={};lncs=None;covariates={};patient_axes={};groups={};z_cache={};basis_cache={}
 for ds,layers in [('TCGA',['RNA','STATE','PROTEIN','PHOSPHO']),('CPTAC',['RNA','STATE','PROTEIN','METHYLATION','PHOSPHO'])]:
  source_dataset='CPTAC_DEDUPLICATED' if ds=='CPTAC' and cancer=='BRCA' and context=='sealed_test' else ds
  root=O/source_dataset/cancer/context;ax=json.loads((root/'AXES.json').read_text());split=json.loads((root/'SPLIT.json').read_text());assert not set(split['source_patients'])&set(split['label_patients'])
  if cancer=='BRCA' and ds=='CPTAC':assert 'CPTAC:BRCA:604' not in split['source_patients']+split['label_patients'],'UNRESOLVED_RETROSPECTIVE_CASE_NOT_EXCLUDED'
  if lncs is None:lncs=ax['lncrna_ids']
  assert lncs==ax['lncrna_ids'];splits[ds]=split;patients=ax['patient_ids'];patient_axes[ds]=patients;covariates[ds]=pd.read_parquet(O/'cohorts'/(ds+'_'+cancer)/'COVARIATES.parquet')
  for layer in layers:
   path=root/layer;receipt=json.loads((path/'COMPLETE.json').read_text());assert receipt['target_host']=='149' and receipt['config_sha256']==sha(O/'SCORING_ADJUSTMENT_CONFIG.json')
   for name in ['TARGETS.tsv','ASSOCIATIONS.npz','PATIENT_S.parquet']:assert sha(path/name)==receipt['files'][name],str(path/name)
   md=pd.read_csv(path/'TARGETS.tsv',sep='\t');md.canonical_id=md.canonical_id.map(canon);scores=pd.read_parquet(path/'PATIENT_S.parquet').reindex(patients);assert list(scores)==md.target_id.tolist()
   with np.load(path/'ASSOCIATIONS.npz',allow_pickle=False) as a:stats={k:a[k] for k in ['r','p','q','n','available']}
   assert stats['r'].shape==(len(lncs),len(md));si=pd.Index(patients).get_indexer(split['source_patients']);stream={'dataset':ds,'layer':layer,'path':path,'metadata':md,'scores':scores.to_numpy(float),'stats':stats,'source_index':si,'label_index':pd.Index(patients).get_indexer(split['label_patients'])};streams.append(stream);sid=len(streams)-1
   references.append({'dataset':ds,'source_dataset':source_dataset,'layer':layer,'directory':str(path),'receipt_sha256':sha(path/'COMPLETE.json')})
   for j,row in md.iterrows():
    assert row.canonical_id in indices;groups.setdefault(row.canonical_id,[]).append((sid,j,row.to_dict()))
   print('FUSION_INPUT_READY',cancer,context,ds,layer,flush=True)
 n=len(lncs);m=len(ids);shape=(n,m);arrays={k:np.full(shape,np.nan,np.float64) for k in ['combined_p','combined_q']}
 arrays.update({k:np.full(shape,np.nan,np.float32) for k in ['rms_effect','max_abs_effect']});arrays.update({k:np.zeros(shape,np.uint8) for k in ['observed','membership','observed_assay_bits','individually_strict_assay_bits','individually_supported_assay_bits','cohort_count','individually_supported_cohort_count','calibration_method','direction']});arrays.update({k:np.zeros(shape,np.uint16) for k in ['readout_count','distinct_readout_count','positive_evidence_count','negative_evidence_count']})
 dependence=[];readout_index=[];target_progress=[]
 def residual_label(sid,j,row):
  st=streams[sid];ds=st['dataset'];patients=patient_axes[ds];split=splits[ds];target=row['source_target_id'] if st['layer']=='STATE' else '__pathway__';family=row['family'] if st['layer']=='STATE' else 'pathway';_,policy=resolve_policy(target,contract=CFG['adjustment_contract']);key=(ds,json.dumps(policy,sort_keys=True))
  if key not in z_cache:z_cache[key]=design(covariates[ds],patients,split['source_patients'],family=family,target_id=target,cfg=CFG,admission_cohorts={'source':split['source_patients'],'label':split['label_patients']})[0]
  si=st['label_index'];Z=z_cache[key][si];y=st['scores'][si,j];ok=np.isfinite(Z).all(1)&np.isfinite(y);out=np.full(len(si),np.nan)
  if ok.sum()<12:return out
  bk=(key,ok.tobytes())
  if bk not in basis_cache:basis_cache[bk]=basis(Z[ok])
  Q,rank=basis_cache[bk]
  if ok.sum()-rank<3:return out
  v=rankdata(y[ok]);v-=Q@(Q.T@v)
  if np.linalg.norm(v)>1e-8:out[ok]=v
  return out
 for target_no,(cid,parts) in enumerate(groups.items(),1):
  col=indices[cid];k=len(parts);p=np.column_stack([streams[s]['stats']['p'][:,j] for s,j,row in parts]);rho=np.column_stack([streams[s]['stats']['r'][:,j] for s,j,row in parts]);av=np.column_stack([streams[s]['stats']['available'][:,j] for s,j,row in parts]);av &=np.isfinite(p)&np.isfinite(rho)
  if not av.any():target_progress.append({'canonical_id':cid,'observed_pairs':0,'readouts_registered':k});continue
  source_vectors=[residual_label(s,j,row) for s,j,row in parts];corr=np.eye(k);pair_n=np.zeros((k,k),int)
  for i in range(k):
   pair_n[i,i]=np.isfinite(source_vectors[i]).sum()
   for j in range(i):
    if streams[parts[i][0]]['dataset']!=streams[parts[j][0]]['dataset']:r=0.;cnt=0
    else:
     ok=np.isfinite(source_vectors[i])&np.isfinite(source_vectors[j]);cnt=int(ok.sum());vi=np.nan_to_num(source_vectors[i]);vj=np.nan_to_num(source_vectors[j]);den=np.linalg.norm(vi)*np.linalg.norm(vj);r=float(vi@vj/den) if den>1e-12 else np.nan
    corr[i,j]=corr[j,i]=r;pair_n[i,j]=pair_n[j,i]=cnt
  signature_groups={}
  for ai,(ss,jj,row) in enumerate(parts):
   st=streams[ss];yy=st['scores'][st['label_index'],jj];ok=np.isfinite(yy);rv=np.zeros(len(yy));rv[ok]=rankdata(yy[ok])-(ok.sum()+1)/2
   nz=np.flatnonzero(rv)
   if len(nz) and rv[nz[0]]>0:rv=-rv
   rv[rv==0]=0.
   target=row['source_target_id'] if st['layer']=='STATE' else '__pathway__';_,policy=resolve_policy(target,contract=CFG['adjustment_contract'])
   signature=(st['dataset'],ok.tobytes(),rv.tobytes(),json.dumps(policy,sort_keys=True));signature_groups.setdefault(signature,[]).append(ai)
  dup=list(signature_groups.values());fused=empirical_fisher(p,av,corr,groups=dup,simulations=100000,seed=20261110,cache=dest/'null_references');valid=fused['available'];arrays['observed'][:,col]=valid;arrays['combined_p'][:,col]=fused['p'];arrays['calibration_method'][:,col]=fused['calibration_method'];arrays['readout_count'][:,col]=av.sum(1)
  w=av.astype(float);distinct=np.zeros(n,int)
  for g in dup:
   number=av[:,g].sum(1);distinct+=number>0;w[:,g]=np.divide(w[:,g],number[:,None],out=np.zeros_like(w[:,g]),where=number[:,None]>0)
  ws=w.sum(1);clean=np.where(av,rho,0.);arrays['distinct_readout_count'][:,col]=distinct;arrays['rms_effect'][:,col]=np.sqrt(np.divide((w*clean*clean).sum(1),ws,out=np.full(n,np.nan),where=ws>0));arrays['max_abs_effect'][:,col]=np.where(valid,np.max(abs(clean),axis=1),np.nan)
  signs=av&(p<=.05)&(abs(clean)>=.15);positive=((clean>0)&signs).sum(1);negative=((clean<0)&signs).sum(1);arrays['positive_evidence_count'][:,col]=positive;arrays['negative_evidence_count'][:,col]=negative
  direction=np.where((positive>0)&(negative>0),3,np.where(positive>0,1,np.where(negative>0,2,np.where((w*clean).sum(1)>=0,1,2))));arrays['direction'][:,col]=np.where(valid,direction,0)
  raw_q=np.column_stack([streams[s]['stats']['q'][:,j] for s,j,row in parts]);raw_strict=av&(abs(clean)>=.20)&(raw_q<=.05);raw_support=raw_strict|(av&(abs(clean)>=.15)&(raw_q<=.10))
  for ds in ['TCGA','CPTAC']:
   ii=[a for a,(s,j,row) in enumerate(parts) if streams[s]['dataset']==ds]
   if ii:
    arrays['cohort_count'][:,col]+=av[:,ii].any(1);arrays['individually_supported_cohort_count'][:,col]+=raw_support[:,ii].any(1)
  for a,(s,j,row) in enumerate(parts):
   bit=assay(row['measurement_layer']);arrays['observed_assay_bits'][:,col]|=np.where(av[:,a],bit,0).astype(np.uint8);arrays['individually_strict_assay_bits'][:,col]|=np.where(raw_strict[:,a],bit,0).astype(np.uint8);arrays['individually_supported_assay_bits'][:,col]|=np.where(raw_support[:,a],bit,0).astype(np.uint8);readout_index.append({'canonical_id':cid,'dataset':streams[s]['dataset'],'layer':streams[s]['layer'],'source_target_id':row['source_target_id'],'target_id':row['target_id'],'measurement_layer':row['measurement_layer'],'directory':str(streams[s]['path']),'column_index':j,'label_null_observations':int(pair_n[a,a])})
  dependence.append({'canonical_id':cid,'readouts':[{'dataset':streams[s]['dataset'],'target_id':row['target_id']} for s,j,row in parts],'label_residual_gram_correlation':corr.tolist(),'label_pair_n':pair_n.tolist(),'null_reference_role':'label patients only; score transformations and input covariate encoding remain source-fitted','duplicate_rule':'identical (up to reversal) raw score ranks on the same label patients and same covariate policy','duplicate_groups':dup,'calibration_references':fused['references']});target_progress.append({'canonical_id':cid,'observed_pairs':int(valid.sum()),'readouts_registered':k,'joint_cohort_pairs':int((arrays['cohort_count'][:,col]==2).sum())})
  if target_no%100==0:js(dest/'PROGRESS.json',{'target_host':'149','cancer_id':cancer,'context':context,'targets_processed':target_no,'targets_with_readouts':len(groups),'seconds':time.monotonic()-start});print('FUSED_TARGETS',cancer,context,target_no,len(groups),flush=True)
 primary=np.where(master.roles.str.contains('Pathway'),'Pathway','State');multiplicity=[]
 for task in ['Pathway','State']:
  mask=primary==task;values=arrays['combined_p'][:,mask];arrays['combined_q'][:,mask]=bh(values);multiplicity.append({'task':task,'tested_pairs':int(np.isfinite(values).sum()),'canonical_targets':int(mask.sum())})
 strong=(arrays['observed']>0)&(arrays['combined_q']<=.05)&(arrays['max_abs_effect']>=.20);weak=(arrays['observed']>0)&(arrays['combined_q']<=.10)&(arrays['max_abs_effect']>=.15)&~strong;arrays['membership']=np.where(strong,2,np.where(weak,1,0)).astype(np.uint8);arrays['direction'][arrays['membership']==0]=0
 axes={'lncrna_ids':lncs,'canonical_ids':ids,'target_roles':master.roles.tolist(),'primary_tasks':primary.tolist()};js(dest/'AXES.json',axes);js(dest/'SPLIT.json',splits);js(dest/'INPUT_REFERENCES.json',references);js(dest/'DEPENDENCE_REFERENCES.json',dependence);js(dest/'MULTIPLICITY.json',multiplicity);pd.DataFrame(readout_index).to_csv(dest/'READOUT_INDEX.tsv',sep='\t',index=False);save(dest/'Y.npz',**arrays)
 counts=master.copy();counts['observed_relations']=arrays['observed'].sum(0);counts['supported_relations']=(strong|weak).sum(0);counts['strong_relations']=strong.sum(0);counts['weak_relations']=weak.sum(0);counts['joint_cohort_relations']=(arrays['cohort_count']==2).sum(0);counts['mixed_direction_supported_relations']=(arrays['direction']==3).sum(0);counts['supported_with_individually_supported_nonRNA']=((strong|weak)&((arrays['individually_supported_assay_bits']&126)>0)).sum(0);counts['supported_with_only_individually_supported_RNA']=((strong|weak)&(arrays['individually_supported_assay_bits']==1)).sum(0);counts.to_csv(dest/'TARGET_COUNTS.tsv',sep='\t',index=False)
 record={'target_host':'149','format':'CLA_FUNCTIONAL_RELATION_FUSED_V1','cancer_id':cancer,'context':context,'status':'COMPLETE','canonical_targets':m,'observed_targets':int((counts.observed_relations>0).sum()),'supported_targets':int((counts.supported_relations>0).sum()),'observed_relations':int(arrays['observed'].sum()),'strong_relations':int(strong.sum()),'weak_relations':int(weak.sum()),'joint_cohort_relations':int((arrays['cohort_count']==2).sum()),'mixed_direction_supported_relations':int((arrays['direction']==3).sum()),'algorithm_sha256':sha(R/'relationship_fusion.py'),'method_adoption_sha256':sha(method_receipt),'seconds':time.monotonic()-start,'files':{f.name:sha(f) for f in dest.iterdir() if f.is_file() and f.name not in ['COMPLETE.json','PROGRESS.json']}}
 js(dest/'COMPLETE.json',record);print('FUSED_Y_COMPLETE',json.dumps({k:v for k,v in record.items() if k!='files'}),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--cancer',choices=CANCERS,required=True);a.add_argument('--context',choices=CONTEXTS,default='train_label_fold2');args=a.parse_args();run(args.cancer,args.context)
