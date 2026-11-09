from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import os,socket,json,time,traceback,sys,zipfile,shutil
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'numerical_relation_evidence_r1';P=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/repo');sys.path.insert(0,str(P))
from cla_engine.common import sha
from block_associations import bh
from residual_estimability import residual_estimability
from canonical_quantity_adjustment.design import design
CTX=['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test']
CFG=json.loads((R/'state_relation_evidence_scope_adjusted_r2/SCORING_ADJUSTMENT_CONFIG.json').read_text())

def js(p,d):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
def save(path,**arrays):
 temp=path.with_suffix('.tmp')
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for name,arr in arrays.items():
   with z.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(arr),allow_pickle=False)
 temp.replace(path)
def link(src,dst):
 assert src.exists();dst.symlink_to(src.resolve())
def case(cancer,contexts=CTX):
 start=time.monotonic();base=R/'rna_relation_evidence_direct_excluded_r1'/cancer;axes=json.loads((base/'AXES.json').read_text());patients=axes['patient_ids'];lncs=axes['lncrna_ids'];out=O/cancer;out.mkdir(exist_ok=False);js(out/'AXES.json',axes)
 x=pd.read_parquet(R/'common_rna_scores_r1/aligned'/('TCGA_'+cancer)/'LNC_LOGTPM_FULL.parquet').reindex(patients);assert x.columns.tolist()==lncs;X=x.to_numpy(float)
 cn=R/'full_lnc_local_cnv_r1'/cancer;pi=pd.Index(json.loads((cn/'patient_ids.json').read_text())).get_indexer(patients);li=pd.Index(json.loads((cn/'lncrna_ids.json').read_text())).get_indexer(lncs);assert (pi>=0).all() and (li>=0).all();C=np.load(cn/'lncrna_signed_mean.npy',mmap_mode='r')[np.ix_(pi,li)].astype(float);call=np.load(cn/'lncrna_callable.npy',mmap_mode='r')[np.ix_(pi,li)].astype(bool)
 cov=pd.read_parquet('/dell_2/DSC/CancerLncAtlas/inputs/v32_full_pyg_r6_2/covariates_merged/covariates_r62_20260908.parquet');patient_index={p:i for i,p in enumerate(patients)};records=[]
 rmd=pd.read_csv(base/'TARGETS.tsv',sep='\t');rmd['source_target_id']=rmd.target_id.str.split('|',n=1).str[-1];rmd['measurement_layer']='RNA_EXPRESSION';rmd['family']='pathway'
 pmd=pd.read_csv(R/'rppa_functional_evidence_demo_r1/PROTEIN_TARGETS.tsv',sep='\t');pmd['source_target_id']=pmd.canonical_id;pmd['target_id']=pmd.canonical_id;pmd['family']='pathway'
 fmd=pd.read_csv(R/'rppa_functional_evidence_demo_r1/PHOSPHO_TARGETS.tsv',sep='\t');fmd['source_target_id']=fmd.source_readout_id;fmd['target_id']=fmd.canonical_id;fmd['measurement_layer']='PHOSPHORYLATION';fmd['family']='pathway'
 for context in contexts:
  if shutil.disk_usage(R).free<150*2**30:raise RuntimeError('Remaining disk below 150 GiB')
  split=json.loads((base/context/'SPLIT.json').read_text());sp=split['source_patients'];lp=split['label_patients'];yi=np.array([patient_index[p] for p in lp]);dest=out/context;dest.mkdir();js(dest/'SPLIT.json',split)
  state=R/'state_relation_evidence_scope_adjusted_r2'/cancer/context;rp=R/('rppa_functional_evidence_demo_r1' if cancer in ['BRCA','LUAD','COAD'] else 'rppa_functional_evidence_catalog_r1')/cancer/context
  streams=[('RNA',base/context,R/'common_rna_scores_r1/cache'/('TCGA_'+cancer)/'PATHWAY_RAW.parquet',rmd),('STATE',state,state/'PATIENT_S.parquet',pd.read_csv(state/'TARGETS.tsv',sep='\t')),('PROTEIN',rp/'PROTEIN',rp/'PROTEIN/PATIENT_S.parquet',pmd),('PHOSPHO',rp/'PHOSPHO',rp/'PHOSPHO/PATIENT_S.parquet',fmd)]
  for layer,src,scorepath,md in streams:
   tick=time.monotonic();dd=dest/layer;dd.mkdir();receipt=json.loads((src/'COMPLETE.json').read_text());md.to_csv(dd/'TARGETS.tsv',sep='\t',index=False);link(scorepath,dd/'PATIENT_S.parquet');js(dd/'SPLIT.json',split)
   if receipt.get('admissible') is False:
    js(dd/'COMPLETE.json',{'target_host':'149','admissible':False,'cancer_id':cancer,'context':context,'layer':layer,'source_complete_sha256':sha(src/'COMPLETE.json'),'files':{p.name:sha(p) for p in dd.iterdir() if p.is_file()}});records.append({'cancer_id':cancer,'context':context,'layer':layer,'previous_available':0,'retained_available':0,'removed':0});continue
   assert sha(src/'ASSOCIATIONS.npz')==receipt['files']['ASSOCIATIONS.npz'];a=np.load(src/'ASSOCIATIONS.npz');stats={k:a[k] for k in ['r','p','q','n','rank','available']};a.close();old_available=stats['available'].copy();codes=np.zeros(old_available.shape,np.uint8);scores=pd.read_parquet(scorepath).reindex(patients);assert scores.columns.tolist()==md.target_id.tolist();Y=scores.to_numpy(float)
   reportpath=src/'ADJUSTMENT.json' if (src/'ADJUSTMENT.json').exists() else src.parent/'ADJUSTMENT.json';old_report=json.loads(reportpath.read_text());used=[]
   if layer=='STATE':groups=old_report
   else:groups=[{'targets':md.source_target_id.tolist(),'report':old_report}] if old_available.any() else []
   reports=[]
   for group in groups:
    jj=np.flatnonzero(md.source_target_id.isin(group['targets']));used.extend(jj.tolist())
    if not old_available[:,jj].any():continue
    assert 'report' in group,'Available pairs for a rejected adjustment group'
    target=md.source_target_id.iloc[jj[0]] if layer=='STATE' else '__pathway__';family=md.family.iloc[jj[0]] if layer=='STATE' else 'pathway';Z,report=design(cov,patients,sp,family=family,target_id=target,cfg=CFG,admission_cohorts={'source':sp,'label':lp});assert report['column_names']==group['report']['column_names'],'Encoder columns changed'
    keep,cc,summary=residual_estimability(X,Y[:,jj],C,call,Z,yi,old_available[:,jj],min_n=CFG['minimum_pairs'],previous_n=stats['n'][:,jj],previous_rank=stats['rank'][:,jj]);stats['available'][:,jj]=keep;codes[:,jj]=cc;reports.append(dict(targets=md.source_target_id.iloc[jj].tolist(),summary=summary))
   assert len(used)==len(set(used)),'Duplicate adjustment target group'
   if old_available.any():assert set(np.flatnonzero(old_available.any(axis=0)))<=set(used)
   changed=codes>0;assert np.array_equal(changed,old_available&~stats['available']);stats['r'][changed]=np.nan;stats['p'][changed]=np.nan;stats['q']=bh(stats['p']);strict=stats['available']&(np.abs(stats['r'])>=.20)&(stats['q']<=.05);weak=stats['available']&(np.abs(stats['r'])>=.15)&(stats['q']<=.10)&~strict
   save(dd/'ASSOCIATIONS.npz',**stats,strict_statistical_support=strict,weak_statistical_support=weak);save(dd/'NUMERICAL_EXCLUSIONS.npz',reason_bits=codes)
   ei,ej=np.where(strict|weak);frame=pd.DataFrame({'lncrna_id':np.asarray(lncs)[ei],'canonical_id':md.canonical_id.to_numpy()[ej],'source_target_id':md.source_target_id.to_numpy()[ej],'rho':stats['r'][ei,ej],'p':stats['p'][ei,ej],'q':stats['q'][ei,ej],'n':stats['n'][ei,ej],'statistical_support':np.where(strict[ei,ej],'strict','weak'),'evidence_layer':md.measurement_layer.to_numpy()[ej]});frame.to_parquet(dd/'SUPPORTED_RELATIONS.parquet',index=False,compression='zstd')
   counts=pd.DataFrame({'canonical_id':md.canonical_id,'source_target_id':md.source_target_id,'eligible_lncrna_pairs':stats['available'].sum(axis=0),'strict_statistical_support':strict.sum(axis=0),'weak_statistical_support':weak.sum(axis=0),'numerical_exclusions':changed.sum(axis=0)});counts.to_csv(dd/'TARGET_COUNTS.tsv',sep='\t',index=False);link(reportpath,dd/'ADJUSTMENT.json');js(dd/'NUMERICAL_ADJUSTMENT.json',{'target_host':'149','base_associations':str(src/'ASSOCIATIONS.npz'),'base_associations_sha256':receipt['files']['ASSOCIATIONS.npz'],'estimator_sha256':sha(R/'residual_estimability.py'),'groups':reports,'reason_bits':{'1':'insufficient_patient_pairs','2':'nonpositive_residual_degrees_of_freedom','4':'constant_input_ranks','8':'constant_target_ranks','16':'input_residual_numerically_zero','32':'target_residual_numerically_zero'}})
   row={'cancer_id':cancer,'context':context,'layer':layer,'previous_available':int(old_available.sum()),'retained_available':int(stats['available'].sum()),'removed':int(changed.sum()),'strict_statistical_support':int(strict.sum()),'weak_statistical_support':int(weak.sum()),'seconds':time.monotonic()-tick};js(dd/'COMPLETE.json',dict(row,target_host='149',admissible=True,files={p.name:sha(p) for p in dd.iterdir() if p.is_file()}));records.append(row);print('NUMERICAL_CONTEXT_COMPLETE',row,flush=True)
  js(dest/'COMPLETE.json',{'target_host':'149','cancer_id':cancer,'context':context,'layers':['RNA','STATE','PROTEIN','PHOSPHO']})
 pd.DataFrame(records).to_csv(out/'CORRECTION_COUNTS.tsv',sep='\t',index=False);js(out/'COMPLETE.json',{'target_host':'149','status':'COMPLETE','cancer_id':cancer,'seconds':time.monotonic()-start,'records':records});return cancer,records
if __name__=='__main__':
 v=json.loads((R/'RESIDUAL_ESTIMABILITY_VALIDATION.json').read_text());assert v['status']=='PASS' and v['implementation_sha256']==sha(R/'residual_estimability.py');O.mkdir(exist_ok=False);js(O/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'CORRECT_NUMERICAL_ESTIMABILITY_AND_RECOMPUTE_FDR','feature_axis':'16889 measured lncRNAs, unchanged','patient_scores':'unchanged','workers':4,'minimum_free_disk_gib':150,'validation':v});cancers=sorted([p.name for p in (R/'rna_relation_evidence_direct_excluded_r1').iterdir() if (p/'AXES.json').exists()],key=lambda c:(c not in ['BRCA','LUAD','COAD'],c));assert len(cancers)==33;results={};failures={}
 original_output=O;O=R/'numerical_relation_pilot_r1';O.mkdir(exist_ok=False);js(O/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'REAL_LUAD_NUMERICAL_ESTIMABILITY_PILOT'});case('LUAD',contexts=['sealed_test']);O=original_output;print('NUMERICAL_PILOT_COMPLETE',flush=True)
 with ProcessPoolExecutor(max_workers=4) as pool:
  fs={pool.submit(case,c):c for c in cancers}
  for f in as_completed(fs):
   c=fs[f]
   try:_,rows=f.result();results[c]=rows
   except Exception as e:failures[c]=repr(e);print('NUMERICAL_CORRECTION_FAILED',c,traceback.format_exc(),flush=True)
   js(O/'PROGRESS.json',{'target_host':'149','complete_cancers':len(results),'failures':failures})
 pd.DataFrame([r for rows in results.values() for r in rows]).to_csv(O/'CORRECTION_COUNTS.tsv',sep='\t',index=False);js(O/'COMPLETE.json',{'target_host':'149','status':'COMPLETE' if not failures and len(results)==33 else 'FAILED','cancers':len(results),'failures':failures})
