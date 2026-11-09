"""Join independent assay evidence into one cancer-lncRNA-target label."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,socket,sys,time,traceback,zipfile
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'functional_relation_y_catalog_r6';P=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/repo');sys.path.insert(0,str(P))
from cla_engine.common import sha
C=R/'canonical_targets_r6';MASTER=pd.read_csv(C/'TARGET_MASTER.tsv',sep='\t');MASTER=MASTER.loc[MASTER.has_scored_readout.astype(str).str.lower().eq('true')].reset_index(drop=True);assert MASTER.canonical_id.is_unique
TARGETS=MASTER.canonical_id.tolist();TPOS={x:i for i,x in enumerate(TARGETS)};RED=pd.read_csv(C/'ID_REDIRECTS.tsv',sep='\t')
print('REDIRECT_COLUMNS',list(RED),flush=True)
def canon(x):
 seen=set()
 while x in REDIRECT:
  assert x not in seen;seen.add(x);x=REDIRECT[x]
 return x
# The registry records old and surviving canonical identities explicitly.
oldcol=next(k for k in ['previous_internal_id','old_canonical_id','from_canonical_id','previous_canonical_id'] if k in RED)
newcol=next(k for k in ['canonical_id','new_canonical_id','to_canonical_id'] if k in RED)
REDIRECT=dict(zip(RED[oldcol],RED[newcol]));BIND=pd.read_csv(C/'SOURCE_ID_BINDINGS.tsv',sep='\t').set_index('source_target_id').canonical_id.to_dict()
LAYER_BITS={'RNA':1,'PROTEIN':2,'PHOSPHO':4,'DNA_METHYLATION':8,'DNA_COPY_NUMBER':16,'DNA_VARIANT':32,'OTHER':64}
def measurement_bit(layer):
 s=str(layer).upper()
 if s.startswith('RNA'):return 1
 if 'PROTEIN' in s:return 2
 if 'PHOSPH' in s:return 4
 if 'METHYL' in s:return 8
 if 'COPY_NUMBER' in s:return 16
 if any(k in s for k in ['SNV','SEQUENC','MUTATION']):return 32
 return 64
def js(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False))
def save_npz(path,**arrays):
 temp=path.with_suffix('.tmp')
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for key,v in arrays.items():
   with z.open(key+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(v),allow_pickle=False)
 temp.replace(path)
def inputs(cancer,context):
 base=R/'numerical_relation_evidence_r1'/cancer/context
 return [base/layer/'COMPLETE.json' for layer in ['RNA','STATE','PROTEIN','PHOSPHO']]
def work(cancer,context):
 start=time.monotonic();dest=O/cancer/context;dest.mkdir(parents=True,exist_ok=False);base=R/'numerical_relation_evidence_r1'/cancer;axes=json.loads((base/'AXES.json').read_text());lncs=axes['lncrna_ids'];n=len(lncs);m=len(TARGETS);idx=pd.Index(lncs);shape=(n,m);available_bits=np.zeros(shape,np.uint8);strict_bits=np.zeros(shape,np.uint8);support_bits=np.zeros(shape,np.uint8);direction=np.zeros(shape,np.uint8);source_records=[];refsplit=json.loads((base/context/'SPLIT.json').read_text())
 shared=set(MASTER.loc[MASTER.roles.fillna('').eq('Pathway|State'),'canonical_id'])
 streams=[(layer,base/context/layer,pd.read_csv(base/context/layer/'TARGETS.tsv',sep='\t'),None if layer=='STATE' else layer) for layer in ['RNA','STATE','PROTEIN','PHOSPHO']]
 for source,path,md,fixedlayer in streams:
  receipt=json.loads((path/'COMPLETE.json').read_text());assert receipt['target_host']=='149'
  if receipt.get('admissible') is False:
   source_records.append({'source':source,'eligible_relations':0,'adjustment_record':str(path/'COMPLETE.json')});continue
  for fn in ['ASSOCIATIONS.npz','SUPPORTED_RELATIONS.parquet']:
   expected=receipt['files'][fn];assert sha(path/fn)==expected,fn
  splitpath=path/'SPLIT.json' if (path/'SPLIT.json').exists() else path.parent/'SPLIT.json';split=json.loads(splitpath.read_text());assert split['source_patients']==refsplit['source_patients'] and split['label_patients']==refsplit['label_patients'];assert not set(split['source_patients'])&set(split['label_patients'])
  a=np.load(path/'ASSOCIATIONS.npz');av=a['available'];assert av.shape==(n,len(md))
  cids=[canon(str(x)) for x in md.canonical_id];bits=np.array([measurement_bit(fixedlayer or x) for x in (md.measurement_layer if fixedlayer is None else [fixedlayer]*len(md))],np.uint8)
  keep=np.array([not(source=='STATE' and md.source_target_id.iloc[j] in ['OXPHOS','DNARepair_Hallmark']) for j,c in enumerate(cids)])
  for j,(cid,bit) in enumerate(zip(cids,bits)):
   if keep[j]:available_bits[:,TPOS[cid]]|=av[:,j].astype(np.uint8)*bit
  df=pd.read_parquet(path/'SUPPORTED_RELATIONS.parquet');df['canonical_id']=df.canonical_id.map(canon)
  if source=='STATE':df=df.loc[~df.source_target_id.isin(['OXPHOS','DNARepair_Hallmark'])].copy()
  if len(df):
   ix=idx.get_indexer(df.lncrna_id);assert (ix>=0).all();jx=df.canonical_id.map(TPOS).to_numpy(int);bv=np.array([measurement_bit(fixedlayer or x) for x in df.evidence_layer],np.uint8);st=df.statistical_support.eq('strict').to_numpy();np.bitwise_or.at(support_bits,(ix,jx),bv);np.bitwise_or.at(strict_bits,(ix[st],jx[st]),bv[st]);dv=np.where(df.rho.to_numpy()>0,1,2).astype(np.uint8);np.bitwise_or.at(direction,(ix,jx),dv)
  source_records.append({'source':source,'file':str(path/'SUPPORTED_RELATIONS.parquet'),'file_sha256':sha(path/'SUPPORTED_RELATIONS.parquet'),'score_targets':len(md),'supported_readouts':len(df),'target_axis_sha256':sha(path/'TARGETS.tsv') if (path/'TARGETS.tsv').exists() else None})
 # A repeated algorithm on the same assay is one evidence family.
 pop=np.array([int(i).bit_count() for i in range(256)],np.uint8);strong=pop[strict_bits]>=2;positive=support_bits>0;membership=np.where(strong,2,np.where(positive,1,0)).astype(np.uint8);eligible=available_bits>0
 assert not np.any(positive&~eligible);assert not np.any((membership==0)&(direction!=0));assert not np.any((membership>0)&(direction==0))
 save_npz(dest/'Y.npz',membership=membership,direction=direction,observed_assay_bits=available_bits,strict_assay_bits=strict_bits,supported_assay_bits=support_bits)
 ix,jx=np.where(membership>0);df=pd.DataFrame({'cancer_id':cancer,'lncrna_id':np.asarray(lncs)[ix],'canonical_id':np.asarray(TARGETS)[jx],'membership':np.where(membership[ix,jx]==2,'strong_positive','weak_positive'),'direction':np.array(['unknown','positive','negative','discordant'])[direction[ix,jx]],'strict_assay_count':pop[strict_bits[ix,jx]],'supported_assay_count':pop[support_bits[ix,jx]],'observed_assay_bits':available_bits[ix,jx],'strict_assay_bits':strict_bits[ix,jx],'supported_assay_bits':support_bits[ix,jx]});df.to_parquet(dest/'POSITIVE_RELATIONS.parquet',compression='zstd',index=False)
 counts=MASTER[['canonical_id','display_name','roles']].copy();counts['observable_relations']=eligible.sum(0);counts['strong_positive']=(membership==2).sum(0);counts['weak_positive']=(membership==1).sum(0);counts['unlabeled_observed']=(eligible&(membership==0)).sum(0);counts['direction_discordant']=(direction==3).sum(0);counts.to_csv(dest/'TARGET_COUNTS.tsv',sep='\t',index=False)
 js(dest/'AXES.json',{'lncrna_ids':lncs,'canonical_ids':TARGETS});js(dest/'SPLIT.json',refsplit);js(dest/'EVIDENCE_SOURCES.json',source_records)
 check=np.load(dest/'Y.npz');assert np.array_equal(check['membership'],membership);assert int((check['membership']>0).sum())==len(df)
 row={'cancer_id':cancer,'context':context,'target_identities':m,'observable_relations':int(eligible.sum()),'strong_positive':int((membership==2).sum()),'weak_positive':int((membership==1).sum()),'unlabeled_observed':int((eligible&(membership==0)).sum()),'direction_discordant':int((direction==3).sum()),'seconds':time.monotonic()-start};js(dest/'COMPLETE.json',dict(row,target_host='149',source_split_disjoint=True,files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}));print('Y_CONTEXT_COMPLETE',row,flush=True);return row
if __name__=='__main__':
 CANCERS=sorted(pd.read_csv('/dell_2/DSC/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv',sep='\t').cancer_id.unique())
 O.mkdir(exist_ok=False);js(O/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'ASSAY_EVIDENCE_RELATION_Y','cancers':CANCERS,'outer_fold':0,'training_started':False});MASTER.to_csv(O/'TARGET_MASTER.tsv',sep='\t',index=False)
 js(O/'Y_DEFINITION.json',{'unit':'cancer x lncRNA x canonical target','membership_codes':{'0':'unlabeled','1':'weak_positive','2':'strong_positive'},'weak_positive':'At least one measured-assay family meets its declared strict or weak association criterion','strong_positive':'At least two distinct measured-assay families meet strict association criteria','strict_association':'absolute rho >= 0.20 and BH q <= 0.05','weak_association':'absolute rho >= 0.15 and BH q <= 0.10, excluding strict','direction_codes':{'0':'unknown','1':'positive','2':'negative','3':'discordant'},'direction':'Sign of the measured-score association; conflicting supported signs retained','assay_bits':LAYER_BITS,'unobserved':'observed_assay_bits == 0','supervised_negative_class':None,'input_patient_role':'source_patients','label_patient_role':'label_patients','numerical_estimability':{'definition':'Nonconstant rank vectors and nonzero adjusted residual norms','implementation_sha256':sha(R/'residual_estimability.py'),'source':str(R/'numerical_relation_evidence_r1')},'dataset_scope':'TCGA RNA and State with matched TCPA RPPA500 evidence','task_weights':{'Pathway':1.0,'State':0.5},'shared_pathway_state_identity':'Count once; Pathway primary task, both roles retained in the dictionary','within_assay_algorithms':'Count as one family','source_definition_hash':sha(C/'COMPLETE.json'),'rna_direct_member_exclusion':{'State':str(R/'state_relation_evidence_scope_adjusted_r2/DIRECT_RNA_MEMBERS.tsv'),'Pathway':str(R/'rna_relation_evidence_direct_excluded_r1/MIXED_CODING_LNC_SYMBOLS.tsv')}})
 pending={(c,k) for c in CANCERS for k in ['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test']};running={};rows=[];failures={};deadline=time.monotonic()+6*3600
 with ProcessPoolExecutor(max_workers=2) as ex:
  while pending or running:
   for c,k in sorted(pending,key=lambda t:(t[0] not in ['BRCA','LUAD','COAD'],t[0],t[1])):
    if all(p.exists() for p in inputs(c,k)):running[ex.submit(work,c,k)]=(c,k);pending.remove((c,k))
   for f,key in list(running.items()):
    if f.done():
     try:rows.append(f.result())
     except Exception as err:failures['/'.join(key)]=repr(err);print('Y_FAILED',key,traceback.format_exc(),flush=True)
     del running[f]
   js(O/'PROGRESS.json',{'target_host':'149','complete':len(rows),'total':len(CANCERS)*5,'pending':[list(k) for k in sorted(pending)],'failures':failures});pd.DataFrame(rows).to_csv(O/'CANCER_CONTEXT_COUNTS.tsv',sep='\t',index=False)
   if time.monotonic()>deadline:failures['input_wait']='Six-hour input deadline';break
   if pending or running:time.sleep(5)
 js(O/'COMPLETE.json',{'target_host':'149','status':'COMPLETE' if not failures else 'FAILED','contexts':rows,'failures':failures})
