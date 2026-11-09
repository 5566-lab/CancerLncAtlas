from pathlib import Path
import os,sys,json,socket,hashlib,time
import numpy as np
import pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;REPO=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/repo');sys.path.insert(0,str(REPO))
from cla_engine.statistics import pairs,design
from block_associations import associate
REF=REPO/'cla_engine/statistics.py';assert hashlib.sha256(REF.read_bytes()).hexdigest()=='7183a6d777043064078224db4d40f3885de3b025d4f49ab43a1e3e20cabed06a'
records=[]
def compare(name,X,Y,C,call,Z,cohort):
 t=time.monotonic();a=associate(X,Y,C,call,Z,cohort);block_time=time.monotonic()-t
 li=np.repeat(np.arange(X.shape[1]),Y.shape[1]);ti=np.tile(np.arange(Y.shape[1]),X.shape[1]);t=time.monotonic();b=pairs(np.asarray(X,float),np.asarray(Y,float),np.asarray(C,float),np.asarray(call,bool),Z,li,ti,cohort);ref_time=time.monotonic()-t
 errors={}
 for k in ['r','p','q','n','rank','available']:
  av=np.asarray(a[k]).ravel();bv=b[k]
  if k in ['r','p','q']:
   np.testing.assert_allclose(av,bv,atol=2e-12,rtol=2e-11,equal_nan=True);errors[k]=float(np.nanmax(np.abs(av-bv))) if np.isfinite(av).any() else None
  else:np.testing.assert_array_equal(av,bv)
 records.append({'case':name,'pairs':len(li),'finite_pairs':int(a['available'].sum()),'max_absolute_error':errors,'block_seconds':block_time,'reference_seconds':ref_time,'masks':a['mask_groups']});print(records[-1],flush=True)
rg=np.random.default_rng(4091)
for mode in ['complete','missing_ties','rank_deficient','constant_cnv','low_n','all_missing']:
 n=83;L=33;T=17;X=rg.normal(size=(n,L));Y=rg.normal(size=(n,T));C=rg.normal(size=(n,L));call=np.ones((n,L),bool);Z=np.column_stack([np.ones(n),rg.normal(size=(n,3))]);cohort=np.arange(n)
 if mode=='missing_ties':
  X=np.round(X);Y=np.round(Y);X[:9,:4]=np.nan;Y[2:16,:5]=np.nan;call[5:13,7:12]=False;C[18:24,20:23]=np.nan;Z[-5:,2]=np.nan;X[:,0]=3;Y[:,0]=2
 if mode=='rank_deficient':Z=np.column_stack([Z,Z[:,1],np.ones(n)])
 if mode=='constant_cnv':C[:,:8]=2;C[:,8:13]=0;C[:,13]=Z[:,1]
 if mode=='low_n':cohort=np.arange(9)
 if mode=='all_missing':Y[:]=np.nan
 compare(mode,X,Y,C,call,Z,cohort)
# Independent real-data fixture: raw full lncRNA, signed local CNV, common RNA scores and R6.2 clinical design.
cancer='BRCA';cdir=R/'full_lnc_local_cnv_r1'/cancer;patients=json.loads((cdir/'patient_ids.json').read_text());lncs=json.loads((cdir/'lncrna_ids.json').read_text());chosen=np.linspace(0,len(lncs)-1,67,dtype=int)
x=pd.read_csv('/dsk2/user/dengsc/cla_m0_lasso_20261104_r1/tcga_full_tpm_r1/BRCA/GENE_TPM.tsv.gz',sep='\t',index_col=0)
X=np.log2(x.reindex([lncs[i].removeprefix('LNC:') for i in chosen])[patients].T.to_numpy(float)+1)
y=pd.read_parquet('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/score_cache_r2/BRCA/PATHWAY_RAW.parquet').reindex(patients);Y=y.loc[:,y.notna().any()].iloc[:,:19].to_numpy(float)
C=np.load(cdir/'lncrna_signed_mean.npy',mmap_mode='r')[:,chosen];call=np.load(cdir/'lncrna_callable.npy',mmap_mode='r')[:,chosen]
cov=pd.read_parquet('/dell_2/DSC/CancerLncAtlas/inputs/v32_full_pyg_r6_2/covariates_merged/covariates_r62_20260908.parquet');cfg=json.loads(Path('/dsk2/user/dengsc/cla_state_signature_update_20261104_r1/CURRENT_149_CONFIG.json').read_text())
try:
 Z,report=design(cov,patients,patients,family='pathway',cfg=cfg)
 compare('BRCA_real_R62',X,Y,C,call,Z,np.arange(len(patients)))
except Exception as e:
 # Contract failure is recorded, never replaced by a reduced adjustment.
 records.append({'case':'BRCA_real_R62','contract_error':str(e)});raise
out={'target_host':'149','status':'PASS','reference_sha256':hashlib.sha256(REF.read_bytes()).hexdigest(),'implementation_sha256':hashlib.sha256((R/'block_associations.py').read_bytes()).hexdigest(),'checks':records}
(R/'BLOCK_ASSOCIATION_VALIDATION.json').write_text(json.dumps(out,indent=2));print('VALIDATION_PASS',flush=True)
