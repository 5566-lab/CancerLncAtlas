"""Graph ridge solver; reuse M0's clinical design and patient split contracts."""
from __future__ import annotations
import warnings
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from cc_hhgt.m0_covariates import fit_design,query_design,complete_rows,uses_purity
ARMS=('lncrna','mrna','combined')
ETAS=(0.,.1,1.,10.)
LAMBDAS=np.logspace(-4,2,13)
BLOCKS=('CNV','MUTATION','PROTEIN','PHOSPHO')

def target_policy(row):
    layer=row['layer'];tid=row['target_id'];family=str(row.get('family',''));method=str(row.get('method',''))
    exclude=set();role='NON_RNA_ASSAY_PREDICTION'
    if layer=='PROTEIN_PATHWAY':exclude={'PROTEIN'}
    elif layer=='PHOSPHO_PTMSEA':exclude={'PHOSPHO'}
    elif layer=='METHYLATION_PATHWAY':pass
    elif layer=='RNA_PATHWAY':role='RNA_DERIVED_TARGET'
    elif layer=='STATE':
        if not uses_purity(tid):exclude={'CNV','MUTATION'}
        elif family in ['mutation_burden','sbs_spectrum','msi']:exclude={'MUTATION'}
        elif family=='ploidy':exclude={'CNV'}
        elif family=='hrd_scar':exclude={'CNV','MUTATION'}
        elif 'stemness_dna' in method or 'leukocyte_fraction' in method:pass
        else:role='RNA_DERIVED_TARGET'
    else:raise ValueError('UNCLASSIFIED_TARGET_LAYER: '+str(layer))
    return dict(allowed_blocks=tuple(b for b in BLOCKS if b not in exclude),excluded_blocks=tuple(sorted(exclude)),role=role,use_purity=uses_purity(tid))

def fit_transform(raw,train,add_availability=False):
    raw=np.asarray(raw,float)
    if np.isinf(raw).any():raise ValueError('INFINITE_INPUT')
    if add_availability:raw=np.column_stack([raw,np.isfinite(raw).any(axis=1).astype(float)])
    idx=np.flatnonzero(np.isfinite(raw[train]).sum(axis=0)>=2)
    if not len(idx):raise ValueError('NO_MEASURED_TRAINING_FEATURES')
    x=raw[:,idx]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(x[train],axis=0)
    z=np.where(np.isfinite(x),x,med);mu=z[train].mean(axis=0);sd=z[train].std(axis=0)
    ok=sd>1e-10;idx=idx[ok];med=med[ok];mu=mu[ok];sd=sd[ok];z=(z[:,ok]-mu)/sd
    if not len(idx):raise ValueError('NO_VARIABLE_TRAINING_FEATURES')
    return z,dict(index=idx,median=med,mean=mu,scale=sd,add_availability=add_availability,original_width=raw.shape[1]-int(add_availability))

def transform(raw,spec):
    raw=np.asarray(raw,float)
    if raw.shape[1]!=spec['original_width']:raise ValueError('FEATURE_WIDTH_MISMATCH')
    if spec['add_availability']:raw=np.column_stack([raw,np.isfinite(raw).any(axis=1).astype(float)])
    x=raw[:,spec['index']];return (np.where(np.isfinite(x),x,spec['median'])-spec['mean'])/spec['scale']

_FACTORS={}
def solve_graph(lap,b,eta,tol=1e-9,maxiter=200):
    from scipy.sparse.linalg import splu
    if eta<0:raise ValueError('NEGATIVE_GRAPH_PENALTY')
    if eta==0:return np.array(b,copy=True),dict(iterations=0,max_relative_residual=0.)
    key=(id(lap),float(eta))
    if key not in _FACTORS:
        q=(sparse.eye(lap.shape[0],format='csr')+eta*lap).tocsc()
        _FACTORS[key]=(lap,q,splu(q,permc_spec='MMD_AT_PLUS_A'))
    _,q,lu=_FACTORS[key];x=lu.solve(np.asarray(b,float))
    norm=np.maximum(np.linalg.norm(b,axis=0),1e-30)
    err=float((np.linalg.norm(q@x-b,axis=0)/norm).max())
    if err>tol*5:raise RuntimeError('GRAPH_SOLVER_NOT_CONVERGED: '+str(err))
    return x,dict(iterations=1,max_relative_residual=err,solver='CACHED_SPARSE_LU')

def normalized_kernel(x,train,lap=None,eta=0.,indices=None):
    if lap is None:k=x@x.T;info=dict(iterations=0,max_relative_residual=0.)
    else:
        rhs=np.zeros((lap.shape[0],x.shape[0]));rhs[indices,:]=x.T
        solved,info=solve_graph(lap,rhs,eta);k=x@solved[indices,:]
    k=(k+k.T)/2;kt=k[np.ix_(train,train)];scale=float(np.trace(kt)/len(train)-kt.mean())
    if scale<=1e-12:raise ValueError('ZERO_KERNEL_VARIATION')
    return k/scale,dict(scale=scale,**info)

def make_context(blocks,features,lap,cov,train,use_purity,progress=None):
    train=np.asarray(train,int);train=train[complete_rows(cov,use_purity)[train]]
    if len(train)<12:raise ValueError('INSUFFICIENT_COVARIATE_COMPLETE_TRAINING')
    z,spec=fit_design(cov.iloc[train],use_purity);za,supported=query_design(cov,spec)
    kernels={};pre={};stats={}
    xr,sr=fit_transform(blocks['RNA'],train);pre['RNA']=sr
    types=features.feature_type.to_numpy()[sr['index']]
    for arm in ARMS:
        selected=np.arange(len(types)) if arm=='combined' else np.flatnonzero(types==arm)
        if not len(selected):raise ValueError('NO_VARIABLE_RNA: '+arm)
        for eta in ETAS:
            key=(arm,eta);k,st=normalized_kernel(xr[:,selected],train,lap,eta,sr['index'][selected]);kernels[key]=k;stats[key]=st
            if progress:progress(arm,eta,st)
    for block in BLOCKS:
        xb,sb=fit_transform(blocks[block],train,True);pre[block]=sb
        kernels[block],stats[block]=normalized_kernel(xb,train)
    return dict(kernels=kernels,preprocess=pre,stats=stats,design=za,design_spec=spec,supported=supported,train_preprocessing=train)

def combined_kernel(ctx,arm,eta,allowed,variant):
    keys=[] if variant=='omics_only' else [(arm,float(eta))]
    if variant!='rna_only':keys+=list(allowed)
    if not keys:raise ValueError('EMPTY_INPUT_BLOCK_SET')
    return sum(ctx['kernels'][k] for k in keys)/len(keys)

def ridge_path(k,z,train,query,y,lambdas=LAMBDAS):
    train=np.asarray(train,int);query=np.asarray(query,int);y=np.asarray(y,float)
    if y.ndim==1:y=y[:,None]
    zz=z[train];zi=np.linalg.pinv(zz);m=np.eye(len(train))-zz@zi
    kt=k[np.ix_(train,train)];kr=m@kt@m;kr=(kr+kr.T)/2
    vals,u=eigh(kr,check_finite=True);vals=np.maximum(vals,0)
    yy=m@y;uy=u.T@yy;covbeta=zi@y;base=z[query]@covbeta
    cross=k[np.ix_(query,train)]-z[query]@zi@kt
    preds=[]
    for lam in lambdas:
        dual=u@(uy/(vals[:,None]+len(train)*float(lam)))
        preds.append(base+cross@dual)
    return np.stack(preds),dict(u=u,vals=vals,uy=uy,zi=zi,kt=kt,covbeta=covbeta,train=train)

def extract_fit(state,lam,y):
    dual=state['u']@(state['uy']/(state['vals'][:,None]+len(state['train'])*float(lam)))
    return dict(train=state['train'],dual=dual,nuisance=state['zi']@(y-state['kt']@dual),lam=float(lam))
