"""Target-aware fixed clinical covariates, with training-fitted encodings only."""
from __future__ import annotations
import numpy as np
import pandas as pd

PURITY_SELF = ('absolute_purity_ploidy::purity', 'published::WES_purity', 'WES_purity', 'ABSOLUTE_purity')

def uses_purity(target_id):
    # Primary covariate is source DNA purity. RNA ESTIMATE is NOT that source.
    return str(target_id).split('|', 1)[-1] not in PURITY_SELF

def complete_rows(frame, use_purity=True):
    cols=['cohort','age_years','sex','stage'] + (['purity'] if use_purity else [])
    valid=frame[cols].notna().all(axis=1).to_numpy(copy=True)
    for col in ['age_years']+(['purity'] if use_purity else []):
        valid &= np.isfinite(frame[col].to_numpy(float))
    return valid

def fit_design(frame, use_purity=True, clinical=True):
    """Complete cases only; clinical and batch coefficients are never penalized.

    Age, sex and stage have common effects. Purity has a cohort-specific slope
    when both cohorts are observed, avoiding an assumed identical DNA assay
    response. All offsets/scales/category levels come only from training rows.
    """
    if clinical and not complete_rows(frame,use_purity).all():
        raise ValueError('REQUIRED_COVARIATE_MISSING_NO_IMPUTATION')
    numeric=['age_years']+(['purity'] if use_purity else []) if clinical else []
    categorical=['sex','stage','cohort'] if clinical else ['cohort']
    spec=dict(clinical=clinical,use_purity=use_purity,numeric={},categories={},columns=['intercept'])
    parts=[np.ones(len(frame))]
    for col in numeric:
        a=frame[col].to_numpy(float);mu=float(a.mean());sd=float(a.std())
        spec['numeric'][col]={'mean':mu,'scale':sd if sd>1e-12 else 1.,'constant':bool(sd<=1e-12),'constant_value':float(a[0])}
        if sd>1e-12:parts.append((a-mu)/sd);spec['columns'].append(col)
    for col in categorical:
        levels=sorted(frame[col].astype(str).unique());spec['categories'][col]=levels
        for lev in levels[1:]:parts.append(frame[col].astype(str).eq(lev).to_numpy(float));spec['columns'].append(col+'='+lev)
    if clinical and use_purity and not spec['numeric']['purity']['constant']:
        a=(frame.purity.to_numpy(float)-spec['numeric']['purity']['mean'])/spec['numeric']['purity']['scale']
        for lev in spec['categories']['cohort'][1:]:
            parts.append(a*frame.cohort.astype(str).eq(lev).to_numpy(float));spec['columns'].append('purity:cohort='+lev)
    z=np.column_stack(parts);spec['rank']=int(np.linalg.matrix_rank(z))
    return z,spec

def query_design(frame,spec):
    n=len(frame);valid=complete_rows(frame,spec['use_purity']) if spec['clinical'] else np.ones(n,bool)
    values={'intercept':np.ones(n)}
    for col,st in spec['numeric'].items():
        a=frame[col].to_numpy(float);values[col]=(a-st['mean'])/st['scale']
        if st['constant']:valid &= np.isclose(a,st['constant_value'],rtol=0,atol=1e-12)
    for col,levels in spec['categories'].items():
        a=frame[col].astype(str);known=a.isin(levels).to_numpy()
        if col!='cohort':valid &= known
        for lev in levels[1:]:
            v=a.eq(lev).to_numpy(float)
            # Unknown cohort uses the equally weighted TRAINING reference.
            # No parameters are estimated from held-out patients.
            if col=='cohort':v[~known]=1/len(levels)
            values[col+'='+lev]=v
    for col in spec['columns']:
        if col.startswith('purity:cohort='):values[col]=values['purity']*values[col.split(':',1)[1]]
    z=np.column_stack([values[c] for c in spec['columns']]);z[~valid]=np.nan
    return z,valid
