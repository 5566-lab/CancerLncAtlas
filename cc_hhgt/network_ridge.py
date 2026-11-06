"""Train-only network-module ridge; coefficients keep independent signs.

The biological prior groups assay features. Group kernel weights are learned
inside every training split; no outcome-derived edges or test-set scaling.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from cc_hhgt.graph_ridge import fit_transform,normalized_kernel,ridge_path
from cc_hhgt.m0_covariates import fit_design,query_design,complete_rows
ARMS=('lncrna','mrna','combined')
BLOCKS=('CNV','MUTATION','PROTEIN','PHOSPHO')
MODES=('ppi','rpi','union','random_101','random_202','random_303','random_404','random_505')
LAMBDAS=10.**np.arange(-5,3.01,.5)
ETAS=(0.,.01,.1,1.,10.,100.)
POWERS=(0.,.5,1.,2.)

def center_trace(k,train):
    a=k[np.ix_(train,train)];return float(np.trace(a)/len(train)-a.mean())

def make_context(blocks,features,fmap,cov,train,purity=True):
    train=np.asarray(train,int);train=train[complete_rows(cov,purity)[train]]
    if len(train)<12:raise ValueError('INSUFFICIENT_CONTEXT_TRAIN')
    _,ds=fit_design(cov.iloc[train],purity);z,supported=query_design(cov,ds)
    ctx=dict(train=train,design=z,design_spec=ds,supported=supported,preprocess={},base={},parts={m:{} for m in MODES},legacy={},scales={})
    for b in ('RNA',)+BLOCKS:
        x,spec=fit_transform(blocks[b],train,b!='RNA',b in ('PROTEIN','PHOSPHO'));ctx['preprocess'][b]=spec
        fm=fmap[fmap.block.eq(b)].sort_values('position');ix=spec['index'];orig=spec['original_width']
        if b=='RNA':
            ctx['_rna_x']=x;ctx['_rna_index']=ix;types=features.feature_type.to_numpy()[ix];ctx['_rna_types']=types
            selectors={a:np.flatnonzero(types==a) for a in ('lncrna','mrna')}
        else:selectors={b:np.arange(x.shape[1])}
        for key,sel in selectors.items():
            xx=x[:,sel];gram=xx@xx.T;scale=center_trace(gram,train)
            if scale<=1e-12:raise ValueError('EMPTY_BLOCK_'+key)
            ctx['base'][key]=gram/scale;ctx['scales'][key]=scale
            positions=ix[sel]
            for mode in MODES:
                modules=np.full(len(positions),-1,int);valid=positions<orig;modules[valid]=fm['module_'+mode].to_numpy()[positions[valid]]
                part={}
                for group in np.unique(modules):
                    keep=np.flatnonzero(modules==group);gx=xx[:,keep]
                    label='module:'+str(group) if group>=0 else 'background:'+key
                    part[label]=(gx@gx.T)/scale
                ctx['parts'][mode][key]=part
        if b=='RNA':
            sl=ctx['scales']['lncrna'];sm=ctx['scales']['mrna'];ctx['scales']['combined']=sl+sm
            ctx['base']['combined']=(sl*ctx['base']['lncrna']+sm*ctx['base']['mrna'])/(sl+sm)
    return ctx

def base_kernel(ctx,arm,allowed,variant='ridge_equal'):
    keys=[] if variant=='omics_only' else [arm]
    if variant!='rna_only':keys+=list(allowed)
    if not keys:raise ValueError('NO_INPUTS')
    return sum(ctx['base'][k] for k in keys)/len(keys)

def group_kernels(ctx,arm,allowed,mode):
    keys=[arm]+list(allowed);groups={}
    if mode=='omics':groups={k:ctx['base'][k]/len(keys) for k in keys}
    else:
        rnakeys=('lncrna','mrna') if arm=='combined' else (arm,)
        for key in rnakeys+tuple(allowed):
            factor=(ctx['scales'][key]/ctx['scales']['combined'] if arm=='combined' and key in rnakeys else 1.)/len(keys)
            for label,k in ctx['parts'][mode][key].items():groups[label]=groups.get(label,0)+factor*k
    labels=sorted(groups);raw=np.stack([groups[g] for g in labels]);mass=np.array([center_trace(k,ctx['train']) for k in raw]);keep=mass>1e-15
    labels=[v for v,q in zip(labels,keep) if q];raw=raw[keep];mass=mass[keep]
    # Each modality has unit centered trace, so prior masses sum exactly to one.
    if abs(mass.sum()-1)>1e-8:raise ValueError('GROUP_MASS_NOT_ONE')
    kg=raw/mass[:,None,None]
    if not np.allclose(raw.sum(axis=0),base_kernel(ctx,arm,allowed),atol=2e-10,rtol=2e-10):raise ValueError('POWER_ZERO_BASELINE_DRIFT')
    return kg,mass,labels

def alignment_weights(kg,mass,z,train,y,power):
    y=np.asarray(y,float)
    if y.ndim==1:y=y[:,None]
    train=np.asarray(train,int);zz=z[train];m=np.eye(len(train))-zz@np.linalg.pinv(zz);yr=m@y
    if power==0:return np.broadcast_to(mass[:,None],(len(mass),y.shape[1])).copy()
    kt=kg[:,train][:,:,train];res=np.einsum('ij,gjk,kl->gil',m,kt,m,optimize=True)
    numer=np.einsum('it,gij,jt->gt',yr,res,yr,optimize=True)
    denom=np.linalg.norm(res,axis=(1,2))[:,None]*np.sum(yr*yr,axis=0)[None,:]
    a=np.divide(np.maximum(numer,0),denom,out=np.zeros_like(numer),where=denom>1e-20)
    weights=mass[:,None]*(a+1e-12)**power
    weights/=weights.sum(axis=0,keepdims=True)
    return weights

def weighted_path(kg,weights,z,train,query,y,lambdas=LAMBDAS,chunk=32,return_fits=False):
    """Batch target-specific dual ridge with unpenalized clinical covariates."""
    train=np.asarray(train,int);query=np.asarray(query,int);y=np.asarray(y,float)
    if y.ndim==1:y=y[:,None]
    zz=z[train];zi=np.linalg.pinv(zz);m=np.eye(len(train))-zz@zi
    yr=m@y;covbeta=zi@y;ktg=kg[:,train][:,:,train];kqg=kg[:,query][:,:,train]
    pred=np.empty((len(lambdas),len(query),y.shape[1]));fits=[]
    for start in range(0,y.shape[1],chunk):
        end=min(start+chunk,y.shape[1]);w=weights[:,start:end];kt=np.einsum('gt,gij->tij',w,ktg,optimize=True);kq=np.einsum('gt,gij->tij',w,kqg,optimize=True)
        kr=np.einsum('ij,tjk,kl->til',m,kt,m,optimize=True);kr=(kr+kr.transpose(0,2,1))/2
        vals,u=np.linalg.eigh(kr);vals=np.maximum(vals,0)
        uy=np.einsum('tij,jt->ti',u.transpose(0,2,1),yr[:,start:end],optimize=True)
        cross=kq-np.einsum('ij,jk,tkl->til',z[query],zi,kt,optimize=True)
        cu=np.einsum('tqi,tij->tqj',cross,u,optimize=True)
        for li,lam in enumerate(lambdas):
            coeff=uy/(vals+len(train)*float(lam))
            pred[li,:,start:end]=(np.einsum('tqj,tj->tq',cu,coeff,optimize=True)).T+z[query]@covbeta[:,start:end]
        if return_fits:
            for t in range(end-start):fits.append(dict(u=u[t],vals=vals[t],uy=uy[t],kt=kt[t],zi=zi,train=train,y=y[:,start+t]))
    return pred,fits

def extract_weighted_fit(st,lam,weights):
    dual=st['u']@(st['uy']/(st['vals']+len(st['train'])*float(lam)))
    return dict(train=st['train'],dual=dual,nuisance=st['zi']@(st['y']-st['kt']@dual),lam=float(lam),group_weights=weights)

def add_legacy_kernels(contexts,laps,progress=None):
    """One sparse factor at a time; disconnected graph nodes solved exactly."""
    for mode,lap in laps.items():
        active=np.flatnonzero(lap.diagonal()>0);sub=lap[active][:,active].tocsc();n=lap.shape[0]
        for eta in ETAS[1:]:
            q=sparse.eye(len(active),format='csc')+eta*sub;lu=splu(q,permc_spec='MMD_AT_PLUS_A')
            for ctx in contexts:
                x=ctx['_rna_x'];ix=ctx['_rna_index'];types=ctx['_rna_types'];all_solved={};rawk={}
                for arm in ('lncrna','mrna'):
                    sel=np.flatnonzero(types==arm);rhs=np.zeros((n,x.shape[0]));rhs[ix[sel]]=x[:,sel].T
                    solved=rhs.copy();solved[active]=lu.solve(rhs[active]);res=q@solved[active]-rhs[active]
                    err=np.max(np.linalg.norm(res,axis=0)/np.maximum(np.linalg.norm(rhs[active],axis=0),1e-30))
                    if err>5e-9:raise ValueError('LEGACY_SOLVER_RESIDUAL')
                    rawk[arm]=x[:,sel]@solved[ix[sel]];all_solved[arm]=solved
                cross=np.zeros_like(rawk['lncrna'])
                sl=np.flatnonzero(types=='lncrna');cross=x[:,sl]@all_solved['mrna'][ix[sl]]
                rawk['combined']=rawk['lncrna']+rawk['mrna']+cross+cross.T
                for arm,k in rawk.items():
                    k=(k+k.T)/2;scale=center_trace(k,ctx['train']);ctx['legacy'][(mode,arm,float(eta))]=k/scale
            if progress:progress(mode,eta)
            del lu,q
    for ctx in contexts:
        del ctx['_rna_x'];del ctx['_rna_index'];del ctx['_rna_types']

def legacy_kernel(ctx,mode,arm,allowed,eta):
    kr=ctx['base'][arm] if eta==0 else ctx['legacy'][(mode,arm,float(eta))]
    return (kr+sum(ctx['base'][b] for b in allowed))/(1+len(allowed))
