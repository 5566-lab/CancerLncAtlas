"""Collision-free sparse identity design for the matched L1 baseline.

Six original identity terms are retained. A TRAIN-only vocabulary replaces the
unsigned 4096-bin hash. Numeric columns are separately scaled; no dense identity
matrix is constructed. Validation/prediction are transformed in bounded batches.
Budget failures never silently reduce features or fall back to hashing.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from .common import fail,digest

ENCODING='EXACT_CANCER_LNCRNA_TARGET_INTERACTIONS_PLUS_EXPLICIT_SCALED_NUMERIC'
TERMS=('c','l','t','c_l','c_t','l_t')


def identity_records(rows):
    for row in rows:
        if len(row)!=3 or any(not isinstance(x,(str,np.str_)) or not str(x).strip() for x in row):
            fail('BASELINE_IDENTITY_INVALID','Identity rows require nonempty string cancer/lncRNA/target')
        c,l,t=map(str,row)
        # Structured keys prevent IDs containing delimiters from aliasing terms.
        yield {json.dumps(x,ensure_ascii=False,separators=(',',':')):1. for x in
               (('c',c),('l',l),('t',t),('c_l',c,l),('c_t',c,t),('l_t',l,t))}


def available_memory_bytes():
    limits=[]
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):limits.append(int(line.split()[1])*1024)
    except (OSError,ValueError):pass
    try:
        maximum=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        current=int(Path('/sys/fs/cgroup/memory.current').read_text().strip())
        if maximum!='max':limits.append(max(0,int(maximum)-current))
    except (OSError,ValueError):pass
    return min(limits) if limits else None


def resource_plan(n_train,numeric_dim,n_valid,contract):
    # upper bound: six identities plus every numeric term nonzero; liblinear's
    # transpose/workspace and validation batches get explicit overhead allowance.
    nnz=int(n_train)*(6+int(numeric_dim))
    csr=12*nnz+8*(int(n_train)+1)
    batch=min(int(contract['batch_rows']),max(int(n_valid),1))
    validation=12*batch*(6+int(numeric_dim))+8*(batch+1)
    estimate=5*csr+4*validation+int(n_train+int(n_valid))*int(numeric_dim)*16
    cap=int(float(contract['max_working_gib'])*(1024**3))
    free=available_memory_bytes()
    if free is not None:cap=min(cap,int(free*float(contract['max_available_memory_fraction'])))
    report={'train_rows':int(n_train),'validation_rows':int(n_valid),'numeric_dimensions':int(numeric_dim),
            'csr_upper_bound_bytes':csr,'estimated_extra_working_bytes':estimate,'allowed_extra_working_bytes':cap,
            'observed_available_memory_bytes':free,'estimator_memory_is_upper_budget_model_not_measured_peak':True}
    if estimate>cap:fail('BASELINE_RESOURCE_LIMIT_EXCEEDED','Exact baseline does not fit the current CPU resource allowance; no hashed fallback or hidden sampling',**report)
    return report


def fit_vocabulary(rows,contract):
    seen=set();name_bytes=0
    max_features=int(contract['max_identity_features']);max_bytes=int(float(contract['max_vocabulary_gib'])*(1024**3))
    def guarded():
        nonlocal name_bytes
        for record in identity_records(rows):
            for key in record:
                if key not in seen:
                    seen.add(key);name_bytes+=len(key.encode('utf-8'))+160
                    if len(seen)>max_features or name_bytes>max_bytes:
                        fail('BASELINE_VOCABULARY_RESOURCE_LIMIT','Train-only exact vocabulary exceeds allowed size; no feature dropping',features=len(seen),estimated_bytes=name_bytes)
            yield record
    v=DictVectorizer(sparse=True,sort=True,dtype=np.float64)
    v.fit(guarded())
    if not v.vocabulary_:fail('BASELINE_EMPTY_VOCABULARY','No train identities')
    return v,{'features':len(v.vocabulary_),'estimated_vocabulary_bytes':name_bytes,
              'fit_scope':'AVAILABLE_TRAIN_ROWS_ONLY','feature_names_sha256':digest(v.feature_names_)}


def design_block(rows,numeric,scaler,vectorizer):
    numeric=np.asarray(numeric)
    if numeric.ndim!=2 or numeric.shape[0]!=len(rows) or not np.isfinite(numeric).all():
        fail('BASELINE_NUMERIC_INVALID','Finite row-aligned numeric features required')
    identities=vectorizer.transform(identity_records(rows))
    return sparse.hstack([identities,sparse.csr_matrix(scaler.transform(numeric))],format='csr',dtype=np.float64)


def training_design(rows,numeric,scaler,vectorizer,batch_rows):
    n=len(rows);d=numeric.shape[1];width=len(vectorizer.vocabulary_)+d
    if width>=np.iinfo(np.int32).max:fail('BASELINE_FEATURE_SPACE_TOO_LARGE','No unsafe 32-bit CSR indices')
    upper=n*(6+d);data=np.empty(upper,np.float64);indices=np.empty(upper,np.int32);indptr=np.zeros(n+1,np.int64);offset=0
    for start in range(0,n,batch_rows):
        end=min(n,start+batch_rows);b=design_block(rows[start:end],numeric[start:end],scaler,vectorizer)
        data[offset:offset+b.nnz]=b.data;indices[offset:offset+b.nnz]=b.indices
        indptr[start+1:end+1]=b.indptr[1:]+offset;offset+=b.nnz
    return sparse.csr_matrix((data[:offset],indices[:offset],indptr),shape=(n,width),copy=False)


def decision_batched(clf,rows,numeric,scaler,vectorizer,batch_rows):
    out=np.empty(len(rows),np.float64)
    for start in range(0,len(rows),batch_rows):
        end=min(len(rows),start+batch_rows)
        out[start:end]=clf.decision_function(design_block(rows[start:end],numeric[start:end],scaler,vectorizer))
    if not np.isfinite(out).all():fail('BASELINE_PREDICTION_NONFINITE','Finite baseline logits required')
    return out


def unseen_audit(rows,vectorizer):
    counts={t:0 for t in TERMS};total=0
    for record in identity_records(rows):
        total+=1
        for key in record:
            if key not in vectorizer.vocabulary_:counts[json.loads(key)[0]]+=1
    return {'rows':total,'unseen_terms_by_type':counts,'policy':'ZERO_FOR_UNFITTED_IDENTITY_TERM;NUMERIC_FEATURES_REMAIN;REPORTED_NOT_SILENT',
            'validation_fitted_into_vocabulary':False}
