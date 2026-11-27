"""Protein-coding LM22 reference and per-patient linear nu-SVR deconvolution.

This is a coding-member adaptation of the published relative-fraction method,
not a claim of numerical equivalence to published CIBERSORT results.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.svm import NuSVR

METHOD='LM22_PC_LINEAR_NUSVR_RELATIVE_V1'
REFERENCE='https://pmc.ncbi.nlm.nih.gov/articles/PMC4739640/'


def coding_reference(signature, annotation, hgnc):
    if not signature.index.is_unique or not annotation.gene_id.is_unique:
        raise ValueError('LM22_DUPLICATE_INPUT')
    coding=annotation[annotation.gene_biotype.eq('protein_coding')]
    pc=set(coding.reference_symbol)-{''}
    direct=set(annotation.reference_symbol)-{''}
    by_id=coding.set_index('gene_id').reference_symbol.to_dict()
    aliases={}
    for row in hgnc.to_dict('records'):
        if row.get('status')!='Approved':continue
        gene=str(row.get('ensembl_gene_id','')).split('.')[0]
        target=by_id.get(gene)
        if not target:continue
        for field in ('symbol','alias_symbol','prev_symbol'):
            for name in str(row.get(field,'')).split('|'):
                if name:aliases.setdefault(name,set()).add(target)
    audit=[];kept=[]
    for symbol in signature.index:
        target='';reason=''
        if symbol in pc:target=symbol;reason='DIRECT_CODING_SYMBOL'
        elif symbol in direct:reason='DIRECT_NONCODING_SYMBOL'
        else:
            candidates=aliases.get(symbol,set())
            if len(candidates)==1:target=next(iter(candidates));reason='HGNC_TO_CODING_GENE_ID'
            else:reason='AMBIGUOUS_ALIAS' if candidates else 'NO_CODING_GENE_ID_MATCH'
        audit.append({'source_symbol':symbol,'coding_symbol':target,'decision':reason,'retained':bool(target)})
        if target:
            row=signature.loc[[symbol]].copy();row.index=[target];kept.append(row)
    if not kept:raise ValueError('LM22_NO_CODING_REFERENCE')
    reference=pd.concat(kept).groupby(level=0,sort=True).mean()
    if not np.isfinite(reference.to_numpy(float)).all() or (reference.to_numpy(float)<0).any():
        raise ValueError('LM22_INVALID_REFERENCE')
    if len(reference)<=reference.shape[1]:raise ValueError('LM22_INSUFFICIENT_REFERENCE_GENES')
    return reference,pd.DataFrame(audit)


def relative_nusvr(reference, mixture, nus=(.25,.5,.75)):
    """Apply the fixed reference to one linear-expression mixture only."""
    B=np.asarray(reference,float);m=np.asarray(mixture,float)
    if B.ndim!=2 or m.shape!=(B.shape[0],):raise ValueError('LM22_GENE_AXIS')
    if not np.isfinite(B).all():raise ValueError('LM22_REFERENCE_NOT_FINITE')
    if (B<0).any() or (m[np.isfinite(m)]<0).any():raise ValueError('LM22_EXPECTS_LINEAR_NONNEGATIVE_EXPRESSION')
    if not np.isfinite(m).all():return np.full(B.shape[1],np.nan),{'reason':'MISSING_MEMBER_MEASUREMENT'}
    bs=B.std(ddof=1);ms=m.std(ddof=1)
    if bs<=1e-12 or ms<=1e-12:return np.full(B.shape[1],np.nan),{'reason':'CONSTANT_REFERENCE_OR_MIXTURE'}
    X=(B-B.mean())/bs;y=(m-m.mean())/ms
    candidates=[]
    for nu in nus:
        model=NuSVR(nu=nu,C=1.,kernel='linear',tol=1e-3,shrinking=True,cache_size=64)
        model.fit(X,y)
        weight=np.maximum(model.coef_.ravel(),0)
        if weight.sum()<=1e-12:continue
        weight/=weight.sum()
        rmse=float(np.sqrt(np.mean((X@weight-y)**2)))
        candidates.append((rmse,float(nu),weight))
    if not candidates:return np.full(B.shape[1],np.nan),{'reason':'NO_POSITIVE_COEFFICIENT'}
    rmse,nu,weight=min(candidates,key=lambda t:(t[0],t[1]))
    return weight,{'reason':'COMPUTED','selected_nu':nu,'reference_reconstruction_rmse':rmse}


def cell_id(name):
    return re.sub(r'[^A-Za-z0-9_.]','.',str(name))


def score_coding_lm22(logtpm, reference, progress=None):
    if not logtpm.index.is_unique or not logtpm.columns.is_unique:
        raise ValueError('LM22_PATIENT_OR_GENE_DUPLICATE')
    if not set(reference.index)<=set(logtpm.columns):raise ValueError('LM22_MISSING_CODING_AXIS')
    x=np.expm1(logtpm.loc[:,reference.index].to_numpy(float)*np.log(2))
    result=[];qc=[]
    for patient,m in zip(logtpm.index,x):
        f,r=relative_nusvr(reference.to_numpy(float),m)
        result.append(f);qc.append(dict(patient_id=patient,**r))
        if progress and (len(result)%100==0 or len(result)==len(logtpm)):
            progress(len(result),len(logtpm))
    frame=pd.DataFrame(result,index=logtpm.index,columns=[cell_id(c) for c in reference.columns])
    return frame,pd.DataFrame(qc)
