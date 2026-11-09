"""Fold-local rank associations with actual rank, finite-pair masks and target-aware covariates."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t
from .common import fail
from .residual_variation import rank_residual_estimable

# Shared implementation; retained module callers resolve the very same policy.
import sys
from pathlib import Path
_CODE = str(Path(__file__).resolve().parents[1])
if _CODE not in sys.path: sys.path.insert(0, _CODE)
from cc_hhgt.v32.adjustment_policy import (BASE, COMPOSITION, NUMERIC, CONTRACT_ID,
    default_adjustment_contract, resolve_policy, validate_adjustment_contract, effective_policy)
from cc_hhgt.v32.covariate_contract import patient_covariate_design, CovariateContractError

def bh(p):
    p=np.asarray(p,float);out=np.full(p.shape,np.nan);idx=np.flatnonzero(np.isfinite(p));order=idx[np.argsort(p[idx],kind='stable')]
    if len(order):out[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
    return out

def design(cov, patients, fit_patients, *, family, target_id=None, cfg=None, admission_cohorts=None):
    """Fit once on feature patients; enforce explicit roles for this target.

    admission_cohorts are source/label groups only. The outer test never selects
    an encoder or relaxes a threshold. Returned Z marks unsupported patients NaN.
    """
    fallback={'pathway':'__pathway__','stemness_telomere':'RNAss','immune':'ImmuneScore',
        'stromal':'StromalScore','cytotoxicity':'CYT','composition':'ESTIMATEScore',
        'purity':'TumorPurity','ploidy':'ABSOLUTE_ploidy'}
    target=target_id or fallback.get(family)
    if target is None:fail('ADJUSTMENT_TARGET_UNDECLARED',str(family))
    contract=cfg.get('adjustment_contract') if cfg is not None else default_adjustment_contract()
    try:
        validate_adjustment_contract(contract)
        target,policy=resolve_policy(target,contract=contract)
        if len(set(patients))!=len(patients) or len(set(fit_patients))!=len(fit_patients) or set(fit_patients)-set(patients):
            fail('COVARIATE_PATIENT_SCOPE_INVALID','Fit/transform patient identities must be unique and aligned')
        Z,report=patient_covariate_design(cov,patients=patients,fit_patients=fit_patients,
            target_id=target,columns=policy['requested'],required=policy['required'],contract=contract)
        _,policy=effective_policy(target,contract=contract,purity_families=report.get('purity_source_families',[]))
    except CovariateContractError as exc:
        fail(exc.code,str(exc),**exc.details)
    except ValueError as exc:
        fail('COVARIATE_CONTRACT_INVALID',str(exc))
    groups={'source':list(fit_patients)} if admission_cohorts is None else admission_cohorts
    index={p:i for i,p in enumerate(patients)};cohorts={}
    for name,group in groups.items():
        if set(group)-set(patients) or not group:fail('COVARIATE_COHORT_INVALID',str(name))
        ix=np.asarray([index[p] for p in group],int);valid=np.isfinite(Z[ix]).all(1)
        n=int(valid.sum());fraction=n/len(ix)
        # Cohort admission follows the adjustment contract; association pair thresholds apply later.
        minimum=int(policy.get('minimum_observed_count',3))
        cohorts[name]={'patients':len(ix),'valid':n,'valid_fraction':fraction,'minimum_observed_count':minimum}
        if n<minimum or fraction<policy['minimum_joint_valid_fraction']:
            fail('COVARIATE_COHORT_NOT_ADMISSIBLE','Required adjustments unsupported in a source/label cohort',
                target=target,cohort=name,report=cohorts[name])
    # Report full-cohort transformation counts outside the capsule only if needed;
    # the optimizer contract contains source-fit and admitted cohort counts, no outcomes.
    report.update(family=family,fit_patient_count=len(fit_patients),cohorts=cohorts,
        policy='SINGLE_TRAIN_FITTED_TARGET_CONTRACT_R62',admission_passed=True)
    return Z,report

def basis(Z):
    U,s,_=np.linalg.svd(Z,full_matrices=False)
    tol=max(Z.shape)*np.finfo(float).eps*(s[0] if len(s) else 1)
    r=int((s>tol).sum());return U[:,:r],r

def pairs(X,Y,C,call,Z,li,ti,cohort,min_n=12,local_mode='lncrna_only',cnv_transform='raw_signed'):
    """Ranks and nuisance coefficients use only the requested cohort; encoder schema uses training.
    Historical A1 local-CNV action is on lncRNA residual, not both sides. This is an
    adjusted rank association, not an intervention or an identified causal effect.
    """
    call=np.asarray(call,bool)
    cohort=np.asarray(cohort,int);li=np.asarray(li,int);ti=np.asarray(ti,int)
    nrow=len(li);r=np.full(nrow,np.nan);pv=r.copy();nn=np.zeros(nrow,np.int32);rank=np.zeros(nrow,np.int16)
    # The nuisance design basis depends only on the cohort and the finite /
    # callable mask, not on the target.  Formal mode reuses the same lncRNA
    # masks across many pathway targets; cache the SVD basis by packed mask to
    # avoid recomputing an identical basis once per target group.
    basis_cache={}
    for target in np.unique(ti):
        rows=np.flatnonzero(ti==target);y=Y[cohort,int(target)];x=X[np.ix_(cohort,li[rows])];cn=C[np.ix_(cohort,li[rows])]
        mask=np.isfinite(x)&np.isfinite(cn)&call[np.ix_(cohort,li[rows])]&np.isfinite(y)[:,None]&np.isfinite(Z[cohort]).all(1)[:,None]
        # Separate missingness groups; no imputed molecular values participate in associations.
        packed=np.packbits(mask.T,axis=1)
        _,inverse=np.unique(packed,axis=0,return_inverse=True)
        for g in np.unique(inverse):
            ix=np.flatnonzero(inverse==g);valid=mask[:,ix[0]];pos=rows[ix];n=int(valid.sum());nn[pos]=n
            if n<min_n:continue
            mask_key=packed[ix[0]].tobytes()
            cached=basis_cache.get(mask_key)
            if cached is None:
                z=Z[cohort][valid];cached=basis(z);basis_cache[mask_key]=cached
            Q,q=cached
            raw_x=rankdata(x[valid][:,ix],axis=0);raw_y=rankdata(y[valid])
            xx=raw_x.copy();yy=raw_y.copy();cc=cn[valid][:,ix].copy()
            if cnv_transform=='rank':cc=rankdata(cc,axis=0)
            elif cnv_transform!='raw_signed':fail('CNV_TRANSFORM_INVALID',cnv_transform)
            xx-=Q@(Q.T@xx);yy-=Q@(Q.T@yy);cc-=Q@(Q.T@cc)
            den=np.sum(cc*cc,axis=0);tol=1e-18*np.maximum(1,np.sum(cn[valid][:,ix]**2,axis=0))
            active=den>tol;beta=np.divide(np.sum(cc*xx,axis=0),den,out=np.zeros(len(ix)),where=active)
            xr=xx-cc*beta
            if local_mode=='both':
                by=np.divide(cc.T@yy,den,out=np.zeros(len(ix)),where=active);yr=yy[:,None]-cc*by
            else:yr=np.broadcast_to(yy[:,None],xr.shape)
            rr=q+active.astype(int);df=n-rr-1;rank[pos]=rr
            norm=np.sqrt(np.sum(xr*xr,axis=0)*np.sum(yr*yr,axis=0))
            good=((norm>1e-10)&(df>0)
                  &rank_residual_estimable(raw_x,xr,Z.shape[1])
                  &rank_residual_estimable(raw_y,yr,Z.shape[1]))
            corr=np.divide(np.sum(xr*yr,axis=0),norm,out=np.full(len(ix),np.nan),where=good)
            corr=np.clip(corr,-1,1);r[pos]=corr
            stat=np.abs(corr)*np.sqrt(np.maximum(df,0)/np.maximum(1-corr*corr,1e-15));pv[pos]=np.where(good,2*student_t.sf(stat,np.maximum(df,1)),np.nan)
    return {'r':r,'p':pv,'q':bh(pv),'n':nn,'rank':rank,'available':np.isfinite(r)&np.isfinite(pv)}

def labels(stats, task):
    r=np.abs(stats['r']);q=stats['q'];ok=stats['available']
    if task=='pathway':
        strong=(r>=.20)&(q<=.05);weak=(r>=.15)&(q<=.10)&~strong;positive=strong|weak
    else:positive=r>=.20;weak=np.zeros(len(r),bool)
    return np.where(ok,positive,0).astype(np.float32),weak

def raw_features(stats):
    r=stats['r'];ok=stats['available'];q=stats['q']
    return np.column_stack([np.nan_to_num(r),np.nan_to_num(np.abs(r)),np.nan_to_num(-np.log10(np.clip(q,1e-12,1))),np.log1p(stats['n']),ok.astype(float)]).astype(np.float32)
