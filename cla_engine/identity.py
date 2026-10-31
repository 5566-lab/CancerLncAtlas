"""Explicit, lossless identity handling shared by all current CPU readers.

Only the supplied GENE:Ensembl namespace alias is collapsed. No barcode
truncation, symbol lookup, float-to-ID coercion, or fuzzy cancer matching.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .common import fail, digest

ID_NAMES = {'sample','Sample','SampleID','sampleID','sample_barcode','bcr_sample_barcode',
            'patient','Patient','patient_barcode','bcr_patient_barcode','cancer','Cancer',
            'cancer_type','CancerType','project_id'}

def is_id_column(name):
    return str(name).lower().endswith(('_id','_ids','_barcode')) or name in ID_NAMES

def token(v, role='identity', allow_missing=False):
    if pd.isna(v) or str(v).strip()=='':
        if allow_missing:return None
        fail('IDENTITY_MISSING',role)
    if isinstance(v,(float,np.floating,bool,np.bool_)):
        fail('IDENTITY_TYPE_INVALID','IDs must not be inferred from floats/booleans',role=role,value=str(v))
    return str(v).strip()

def gene(v, allow_missing=False):
    s=token(v,'gene_id',allow_missing)
    if s is None:return None
    if s.startswith('GENE:ENSG'):return s[5:]
    return s

def cancer(v, allow_global=False, explicit_global=False):
    s=token(v,'cancer_id',allow_missing=allow_global)
    if s is None:
        if explicit_global:return 'GLOBAL'
        fail('CANCER_CONTEXT_UNKNOWN','Null context is not automatically a global claim')
    if s in {'global','GLOBAL','PAN_CANCER','PAN-CANCER','pan-cancer','PANCAN','ALL','all','*'}:
        if allow_global:return 'GLOBAL'
        fail('CANCER_ID_DRIFT','Global context is not a patient cancer')
    return s.removeprefix('TCGA-')

def frame_ids(frame):
    f=frame.copy()
    for col in f:
        if col=='gene_id':f[col]=f[col].map(lambda x:gene(x,True))
        elif col=='cancer_id':
            f[col]=f[col].map(lambda x:None if pd.isna(x) or not str(x).strip() else str(x).strip().removeprefix('TCGA-'))
        elif is_id_column(col):f[col]=f[col].map(lambda x:token(x,col,True))
    return f

def unique_values(frame, keys, values=None, role='table'):
    """Identical copies are harmless; conflicting measurements must not depend on row order."""
    cols=keys+(values if values is not None else [c for c in frame if c not in keys])
    f=frame[cols].drop_duplicates().copy()
    if f[keys].isna().any().any():fail('IDENTITY_MISSING',role)
    if f.duplicated(keys).any():
        fail('DUPLICATE_VALUE_CONFLICT',role,keys=keys,
             examples=f.loc[f.duplicated(keys,keep=False),keys].drop_duplicates().head(8).to_dict('records'))
    return f

def coverage(requested, observed, *, role, minimum=1.0, allow_empty=False):
    req={token(x,role) for x in requested};obs={token(x,role) for x in observed if not pd.isna(x)}
    missing=sorted(req-obs);fraction=len(req&obs)/len(req) if req else 1.0
    report={'role':role,'requested':len(req),'observed_requested':len(req&obs),
            'missing_count':len(missing),'missing_examples':missing[:12],
            'coverage':fraction,'minimum':float(minimum),'requested_sha256':digest(sorted(req))}
    if req and ((not req&obs and not allow_empty) or fraction<float(minimum)):
        fail('ID_COVERAGE_FAIL','Identity coverage is below the frozen threshold; do not lower it after seeing results',**report)
    return report

def detection_eligibility(frame, min_cancers, threshold=.1):
    f=frame_ids(frame)
    keys=['cancer_id','lncrna_id'];col='detection_rate_logcpm_gt0'
    if set(keys+[col])-set(f):fail('SCHEMA_MISMATCH','detection columns')
    f[col]=pd.to_numeric(f[col],errors='coerce')
    if not np.isfinite(f[col]).all() or not f[col].between(0,1).all():
        fail('DETECTION_RATE_INVALID','Detection rates must be finite in [0,1]')
    rows=len(f);f=unique_values(f,keys,[col],'detection')
    f['eligible']=f[col].ge(threshold)
    counts=f.loc[f.eligible].groupby('lncrna_id',observed=True).cancer_id.nunique()
    eligible=set(counts[counts>=min_cancers].index)
    return f,eligible,{'source_rows':rows,'unique_pairs':len(f),'identical_duplicates_removed':rows-len(f),
                     'count_unit':'DISTINCT_CANCER_NOT_ROWS','minimum_cancers':min_cancers}
