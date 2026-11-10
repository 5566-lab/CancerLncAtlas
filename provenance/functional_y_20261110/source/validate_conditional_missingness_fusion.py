from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import socket,json,sys,hashlib,time
import numpy as np,pandas as pd
from scipy.stats import rankdata,t as student_t,binom
R=Path(__file__).resolve().parent;O=R/'shared_relation_demo_r1';P=O/'fusion_method_r1/conditional_missingness_validation';P.mkdir(exist_ok=True);assert socket.gethostname()=='149';sys.path.insert(0,str(R/'source_integration/repo'))
from shared_core_adjustment.design import design
from block_associations import basis
from relationship_fusion import empirical_fisher
cfg=json.loads((O/'SCORING_ADJUSTMENT_CONFIG.json').read_text());js=lambda p,d:Path(p).write_text(json.dumps(d,indent=2))
js(P/'LAUNCH_PREFLIGHT.json',{'target_host':'149','context':'train_label_fold2','cancers':cfg['cancer_list'],'heldout_test_used':False,'score_parameters':'source patients only','null_dependence':'conditional label outcome residual Gram, missing measurements padded with zero after normalization','permutations_per_gene':1000,'calibration_draws':100000,'prespecified_checks':'Hypoxia/WNT/OXPHOS plus first three measured multi-readout phosphosite identities'})
rng=np.random.default_rng(20261112);records=[]
for ds in ['TCGA','CPTAC']:
 for c in cfg['cancer_list']:
  root=O/ds/c/'train_label_fold2';axes=json.loads((root/'AXES.json').read_text());split=json.loads((root/'SPLIT.json').read_text());patients=axes['patient_ids'];lncs=axes['lncrna_ids'];sp=split['source_patients'];lp=split['label_patients'];si=pd.Index(patients).get_indexer(sp);li=pd.Index(patients).get_indexer(lp)
  cov=pd.read_parquet(O/'cohorts'/(ds+'_'+c)/'COVARIATES.parquet');Z,_=design(cov,patients,sp,family='pathway',target_id='__pathway__',cfg=cfg,admission_cohorts={'source':sp,'label':lp})
  x=pd.read_parquet(R/'common_rna_scores_r1/aligned'/(ds+'_'+c)/'LNC_LOGTPM_FULL.parquet').reindex(index=patients,columns=lncs).to_numpy(float);cn=R/'full_lnc_local_cnv_r1'/c if ds=='TCGA' else O/'cptac_local_cnv'/c;ri=pd.Index(json.loads((cn/'patient_ids.json').read_text())).get_indexer(patients);ci=pd.Index(json.loads((cn/'lncrna_ids.json').read_text())).get_indexer(lncs);C=np.load(cn/'lncrna_signed_mean.npy')[np.ix_(ri,ci)];call=np.load(cn/'lncrna_callable.npy')[np.ix_(ri,ci)].astype(bool)
  scores=[];metas=[]
  for layer in ['RNA','STATE','PROTEIN','METHYLATION','PHOSPHO']:
   p=root/layer
   if not (p/'COMPLETE.json').exists():continue
   md=pd.read_csv(p/'TARGETS.tsv',sep='\t');md['layer']=layer;sc=pd.read_parquet(p/'PATIENT_S.parquet').reindex(patients);scores.append(sc);metas.append(md)
  S=pd.concat(scores,axis=1);md=pd.concat(metas,ignore_index=True);assert list(S)==md.target_id.tolist();usable=(np.isfinite(S.iloc[si].to_numpy(float))&np.isfinite(Z[si]).all(1)[:,None]).sum(0)>=12;vm=md[usable].copy();vm['is_phospho']=vm.layer.eq('PHOSPHO');g=vm.groupby('canonical_id').agg(n=('target_id','size'),phos=('is_phospho','sum'));extra=sorted(g.index[(g.n>=2)&(g.phos>0)])[:3]
  targets={'Hypoxia':'CLA-P-3423B99760CD551B','WNT':'CLA-P-366F48557175A084','OXPHOS':'CLA-P-97C440D7FBA8755B',**{f'Phosphosite_{i+1}':cid for i,cid in enumerate(extra)}}
  validx=(np.isfinite(x[si])&call[si]&np.isfinite(C[si])).sum(0)>=12;eligible=np.flatnonzero(validx&(np.nanstd(x[si],axis=0)>0));chosen=rng.choice(eligible,min(10,len(eligible)),replace=False)
  for name,cid in targets.items():
   ids=vm.loc[vm.canonical_id.eq(cid),'target_id'].tolist()
   if len(ids)<2:continue
   Y=S[ids].to_numpy(float);k=len(ids);vectors=[];counts=[]
   for a in range(k):
    ok=np.isfinite(Z[li]).all(1)&np.isfinite(Y[li,a]);jj=li[ok];v=np.zeros(len(li));counts.append(len(jj))
    if len(jj)>=12:
     Q,q=basis(Z[jj]);yy=rankdata(Y[jj,a]);yy-=Q@(Q.T@yy);norm=np.linalg.norm(yy)
     if norm>1e-8 and len(jj)-q>=3:v[ok]=yy/norm
    vectors.append(v)
   V=np.column_stack(vectors);corr=V.T@V;norms=np.linalg.norm(V,axis=0);bad=norms<1e-8;corr[bad,:]=np.nan;corr[:,bad]=np.nan;np.fill_diagonal(corr,1.);ps=[];pairs=[]
   for j in chosen:
    ok=np.isfinite(x[li,j])&np.isfinite(C[li,j])&call[li,j]&np.isfinite(Z[li]).all(1);idx=li[ok]
    if len(idx)<12:continue
    Q,q=basis(Z[idx]);xx=rankdata(x[idx,j]);xx-=Q@(Q.T@xx);cc=C[idx,j].astype(float);cr=cc-Q@(Q.T@cc);den=cr@cr
    if den>1e-18*max(1,cc@cc):xx-=cr*(cr@xx)/den
    if np.linalg.norm(xx)<1e-8:continue
    perm=np.column_stack([rng.permutation(xx) for _ in range(1000)]);out=np.full((1000,k),np.nan);ns=[];dfs=[]
    for a in range(k):
     use=np.isfinite(Y[idx,a]);jj=idx[use];n=len(jj)
     if n<12:continue
     Q,q=basis(Z[jj]);yy=rankdata(Y[jj,a]);yy-=Q@(Q.T@yy);cc=C[jj,j].astype(float);cr=cc-Q@(Q.T@cc);den=cr@cr;active=den>1e-18*max(1,cc@cc);A=np.column_stack([Q,cr/np.sqrt(den)]) if active else Q;df=n-q-int(active)-1
     if df<=0 or np.linalg.norm(yy)<1e-8:continue
     px=perm[use].copy();px-=A@(A.T@px);rho=np.clip(px.T@yy/np.sqrt(np.sum(px*px,axis=0)*(yy@yy)),-1,1);out[:,a]=2*student_t.sf(abs(rho)*np.sqrt(df/np.maximum(1-rho*rho,1e-15)),df);ns.append(n);dfs.append(df)
    if np.isfinite(out).any():ps.append(out);pairs.append({'lncrna_id':lncs[j],'readout_n':ns,'readout_df':dfs})
   if not ps:continue
   raw=np.concatenate(ps);fused=empirical_fisher(raw,np.isfinite(raw),corr,simulations=100000,seed=20261110,cache=P/'mc_cache');pv=fused['p'][np.isfinite(fused['p'])];rows={'dataset':ds,'cancer':c,'target':name,'canonical_id':cid,'readout_ids':ids,'measurement_layers':vm.set_index('target_id').loc[ids,'measurement_layer'].tolist(),'label_reference_n':counts,'label_residual_gram':corr.tolist(),'null_trials':len(pv),'reject05':float(np.mean(pv<=.05)),'reject01':float(np.mean(pv<=.01)),'reject001':float(np.mean(pv<=.001)),'raw_reject05':np.nanmean(np.where(np.isfinite(raw),raw<=.05,np.nan),axis=0).tolist(),'pairs':pairs,'calibration_references':fused['references']};records.append(rows);js(P/(ds+'_'+c+'_'+name+'.json'),rows);js(P/'PROGRESS.json',{'target_host':'149','records':records});print('CONDITIONAL_MISSING_NULL',ds,c,name,len(pv),rows['reject05'],rows['reject01'],flush=True)
checks=[];mtests=len(records)*3
for row in records:
 for alpha,key in [(.05,'reject05'),(.01,'reject01'),(.001,'reject001')]:
  n=row['null_trials'];count=round(row[key]*n);limit=int(binom.ppf(1-.01/mtests,n,alpha));checks.append({'dataset':row['dataset'],'cancer':row['cancer'],'target':row['target'],'alpha':alpha,'count':count,'upper_null_limit_99pct_familywise':limit,'pass':count<=limit})
js(P/'VALIDATION.json',{'target_host':'149','status':'PASS' if all(x['pass'] for x in checks) else 'FAILED_NULL_CALIBRATION','heldout_test_used':False,'records':records,'checks':checks,'total_null_trials':sum(r['null_trials'] for r in records),'source_algorithm_sha256':hashlib.sha256((R/'relationship_fusion.py').read_bytes()).hexdigest(),'scope':'Conditional label-outcome missingness reference; examples across three cancers and two cohorts. Statistical formula assumes valid marginal association p-values and calibrated conditional dependence.'})
