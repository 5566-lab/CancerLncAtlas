"""Shared lncRNA Lasso, with unpenalized clinical/purity/batch coefficients."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from cc_hhgt.m0_screened_lasso import solve_path,ALPHA_RATIOS,SOLVER
from cc_hhgt.m0_covariates import fit_design,query_design,complete_rows,uses_purity

def fit_x_context(x,cohort,mask=None,covariates=None,use_purity=True):
    x=np.asarray(x,float);cohort=np.asarray(cohort,str);groups=sorted(set(cohort));p=x.shape[1]
    frame=pd.DataFrame({'cohort':cohort}) if covariates is None else covariates.reset_index(drop=True)
    design,spec=fit_design(frame,use_purity,clinical=covariates is not None)
    eligible=np.ones(p,bool) if mask is None else np.asarray(mask,bool).copy()
    original=np.flatnonzero(eligible);a=x[:,original];med=[];imputed=np.empty_like(a);weights=np.empty(len(x))
    good=np.ones(len(original),bool)
    for g in groups:
        rows=cohort==g;v=a[rows];good &= np.isfinite(v).sum(axis=0)>=2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);m=np.nanmedian(np.where(np.isfinite(v),v,np.nan),axis=0)
        med.append(m);imputed[rows]=np.where(np.isfinite(v),v,m);weights[rows]=1/(len(groups)*rows.sum())
    ids=original[good];imputed=imputed[:,good];med=np.asarray(med)[:,good]
    root=np.sqrt(weights*len(x));weighted_design=design*root[:,None];pinv=np.linalg.pinv(weighted_design)
    nuisance_x=pinv@(imputed*root[:,None]);residual=imputed-design@nuisance_x
    scale=np.sqrt(np.sum(weights[:,None]*residual**2,axis=0));keep=np.isfinite(scale)&(scale>1e-12)
    return dict(z=np.asfortranarray(residual[:,keep]/scale[keep]*root[:,None]),feature_index=ids[keep],scale=scale[keep],
        groups=groups,median=med[:,keep],weights=weights,root=root,cohort=cohort,n_input=p,
        design=design,design_spec=spec,design_pinv=pinv,nuisance_x=nuisance_x[:,keep])

def design_for_query(cohort,context,covariates=None):
    if context['design_spec']['clinical'] and covariates is None:raise ValueError('CLINICAL_AND_PURITY_INPUT_REQUIRED')
    frame=pd.DataFrame({'cohort':np.asarray(cohort,str)}) if covariates is None else covariates.reset_index(drop=True)
    return query_design(frame,context['design_spec'])

def query_x(x,cohort,context,covariates=None):
    a=np.asarray(x,float)[:,context['feature_index']];cohort=np.asarray(cohort,str);imputed=np.empty_like(a)
    for g in set(cohort):
        rows=cohort==g;k=context['groups'].index(g) if g in context['groups'] else None
        med=context['median'][k] if k is not None else context['median'].mean(axis=0)
        imputed[rows]=np.where(np.isfinite(a[rows]),a[rows],med)
    design,valid=design_for_query(cohort,context,covariates)
    z=(imputed-design@context['nuisance_x'])/context['scale'];z[~valid]=np.nan
    return z

def center_y(y,context):
    y=np.asarray(y,float);gamma=context['design_pinv']@(y*context['root']);residual=y-context['design']@gamma
    sd=np.sqrt(np.sum(context['weights']*residual**2))
    return residual,gamma,float(sd)

def query_mean(cohort,context,gamma,covariates=None):
    z,valid=design_for_query(cohort,context,covariates);out=z@gamma;out[~valid]=np.nan;return out

def solve(context,y,ratios):
    residual,gamma,sy=center_y(y,context);z=context['z'];ratio=np.asarray(ratios,float)
    if sy<=1e-12 or z.shape[1]==0:return np.zeros((z.shape[1],len(ratio))),gamma,sy,0.,0.
    yn=residual/sy*context['root'];amax=float(np.max(np.abs(z.T@yn))/len(yn))
    if amax<=1e-14:return np.zeros((z.shape[1],len(ratio))),gamma,sy,0.,0.
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter('always',ConvergenceWarning)
        _,coefs,gaps,_=solve_path(z,yn,alphas=amax*ratio,tol=1e-4,max_iter=20000,precompute=False,return_n_iter=True,selection='cyclic')
    if np.any(gaps>1.01e-4) or any(issubclass(v.category,ConvergenceWarning) for v in ww):
        _,coefs,gaps,_=solve_path(z,yn,alphas=amax*ratio,tol=1e-4,max_iter=100000,precompute=False,return_n_iter=True,selection='cyclic')
        if np.any(gaps>1.01e-4):raise RuntimeError('LASSO_NOT_CONVERGED')
    coefs[:,ratio==1]=0
    return coefs*sy,gamma,sy,amax,float(np.max(gaps))

def select_ratio(context,y,x_val,y_val,cohort_val,covariates_val=None):
    coefs,gamma,sy,amax,gap=solve(context,y,ALPHA_RATIOS)
    pred=query_x(x_val,cohort_val,context,covariates_val)@coefs+query_mean(cohort_val,context,gamma,covariates_val)[:,None]
    ok=np.isfinite(pred).all(axis=1)&np.isfinite(y_val)
    if ok.sum()<3:raise ValueError('INSUFFICIENT_SUPPORTED_VALIDATION')
    losses=[np.mean((pred[ok & (np.asarray(cohort_val)==g)]-np.asarray(y_val)[ok & (np.asarray(cohort_val)==g),None])**2,axis=0) for g in sorted(set(np.asarray(cohort_val)[ok]))]
    mse=np.mean(losses,axis=0);best=np.flatnonzero(mse<=np.min(mse)+1e-12*max(1.,np.min(mse)))[0]
    return float(ALPHA_RATIOS[best]),dict(validation_mse=float(mse[best]),path_dual_gap=gap,n_eligible=len(context['feature_index']),n_supported_validation=int(ok.sum()))

def fit_final(context,y,ratio):
    coefs,gamma0,sy,amax,gap=solve(context,y,[ratio]);beta=coefs[:,0]/context['scale'];nz=np.flatnonzero(beta!=0)
    gamma=gamma0-context['nuisance_x'][:,nz]@beta[nz]
    # Intercepts below are at reference clinical covariate values. Full
    # prediction always uses design_spec and all nuisance coefficients.
    spec=context['design_spec'];intercept=[]
    for g in context['groups']:
        value=gamma[0];name='cohort='+g
        if name in spec['columns']:value+=gamma[spec['columns'].index(name)]
        intercept.append(value)
    return dict(feature_index=context['feature_index'][nz],coefficient=beta[nz],groups=context['groups'],
        median=context['median'][:,nz],unknown_median=context['median'][:,nz].mean(axis=0),intercept=np.asarray(intercept),unknown_intercept=float(np.mean(intercept)),
        design_spec=spec,nuisance_coefficients=gamma,covariate_only_coefficients=gamma0,
        outcome_mean=np.array([np.mean(np.asarray(y)[context['cohort']==g]) for g in context['groups']]),outcome_scale=sy,
        alpha=amax*ratio,alpha_ratio=ratio,dual_gap=gap,n_eligible=len(context['feature_index']),n_input=context['n_input'])

def predict(model,x,cohort,covariates=None,covariate_only=False):
    design,valid=design_for_query(cohort,model,covariates)
    gamma=model['covariate_only_coefficients'] if covariate_only else model['nuisance_coefficients']
    out=design@gamma
    if not covariate_only:
        a=np.asarray(x,float)[:,model['feature_index']]
        for g in set(cohort):
            rows=np.asarray(cohort)==g;k=model['groups'].index(g) if g in model['groups'] else None
            med=model['median'][k] if k is not None else model['unknown_median']
            out[rows]+=np.where(np.isfinite(a[rows]),a[rows],med)@model['coefficient']
    out[~valid]=np.nan;return out

def cohort_screen(x,y,cohort,*,min_abs_rho=.10,max_fdr=.05,min_pairs=12,covariates=None,target_ids=None):
    """Rank X/Y, regress on training-only clinical/purity/cohort design, then BH.

    Complete required covariates and observed molecular pairs only. Per target,
    BH includes all supplied lncRNAs; untestable pairs have p=1. DNA purity is
    excluded for its own target. This is not full HHGT A1/local-CNV adjustment.
    """
    from scipy.stats import rankdata,t as student_t,false_discovery_control
    x=np.asarray(x,float);y=np.asarray(y,float);cohort=np.asarray(cohort,str)
    if y.ndim==1:y=y[:,None]
    target_ids=target_ids if target_ids is not None else ['']*y.shape[1]
    frame=pd.DataFrame({'cohort':cohort}) if covariates is None else covariates.reset_index(drop=True)
    shape=(x.shape[1],y.shape[1]);rho=np.zeros(shape,np.float32);p=np.ones(shape);n=np.zeros(shape,np.int32);dof=np.zeros(shape,np.int16)
    yg={};xg={}
    for j in range(y.shape[1]):yg.setdefault((np.packbits(np.isfinite(y[:,j])).tobytes(),uses_purity(target_ids[j])),[]).append(j)
    for k in range(x.shape[1]):xg.setdefault(np.packbits(np.isfinite(x[:,k])).tobytes(),[]).append(k)
    for (_,use_purity),js in yg.items():
        yr=np.isfinite(y[:,js[0]])
        if covariates is not None:yr &= complete_rows(frame,use_purity)
        for ks in xg.values():
            rows=yr & np.isfinite(x[:,ks[0]]);nn=int(rows.sum());n[np.ix_(ks,js)]=nn
            if nn<min_pairs:continue
            design,spec=fit_design(frame.loc[rows],use_purity,clinical=covariates is not None);df=nn-spec['rank']-1;dof[np.ix_(ks,js)]=max(df,0)
            if df<=0:continue
            a=rankdata(x[np.ix_(rows,ks)],axis=0,method='average');b=rankdata(y[np.ix_(rows,js)],axis=0,method='average')
            qz=np.linalg.svd(design,full_matrices=False)[0][:,:spec['rank']];a-=qz@(qz.T@a);b-=qz@(qz.T@b)
            den=np.sqrt((a*a).sum(axis=0)[:,None]*(b*b).sum(axis=0)[None,:])
            rr=np.divide(a.T@b,den,out=np.zeros_like(den),where=den>1e-8);rr=np.clip(rr,-1,1)
            with np.errstate(divide='ignore',invalid='ignore'):stat=np.abs(rr)*np.sqrt(df/np.maximum(1-rr*rr,0))
            pp=2*student_t.sf(stat,df);pp[den<=1e-8]=1
            rho[np.ix_(ks,js)]=rr;p[np.ix_(ks,js)]=pp
    q=false_discovery_control(p,axis=0).astype(np.float32)
    return dict(rho=rho,q=q,n=n,df=dof,passed=(np.abs(rho)>=min_abs_rho)&(q<=max_fdr)&(n>=min_pairs))
