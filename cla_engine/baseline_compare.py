"""Validation-only exact versus 4096-hash identity LASSO comparison."""
from __future__ import annotations

import inspect
import socket
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression

from .common import atomic_json,commit_manifest,digest,fail,read_json,sha,status
from .exact_baseline import fit_vocabulary,identity_records
from .execution_contract import baseline_contract
from .lineage import validate_prepared
from .prepare import macro_ap


FORMAT='CLA_R62_MATCHED_EXACT_VS_HASH4096_IDENTITY_BASELINE_V1'


def _csr32(matrix):
    """Return CSR with liblinear-compatible 32-bit sparse index arrays."""
    matrix=matrix.tocsr()
    if matrix.indices.dtype!=np.int32:
        matrix.indices=matrix.indices.astype(np.int32)
    if matrix.indptr.dtype!=np.int32:
        matrix.indptr=matrix.indptr.astype(np.int32)
    return matrix


def load_npz(path):
    with np.load(path,allow_pickle=False) as handle:
        return {key:handle[key] for key in handle.files}


def hashed_design(rows,numeric,n_features=4096):
    """Hash the six identity terms; keep the same scaled numeric columns explicit."""
    numeric=np.asarray(numeric,dtype=np.float64)
    if numeric.ndim!=2 or numeric.shape[0]!=len(rows) or not np.isfinite(numeric).all():
        fail('BASELINE_NUMERIC_INVALID','Prepared standardized numeric rows must be finite and aligned')
    hasher=FeatureHasher(n_features=int(n_features),input_type='dict',alternate_sign=False)
    identity=hasher.transform(identity_records(rows)).tocsr()
    return _csr32(sparse.hstack([identity,sparse.csr_matrix(numeric)],format='csr',dtype=np.float64))


def hash_collision_report(rows,n_features=4096):
    terms=sorted({key for record in identity_records(rows) for key in record})
    hasher=FeatureHasher(n_features=int(n_features),input_type='dict',alternate_sign=False)
    mapped=hasher.transform(({term:1.} for term in terms)).tocsr()
    occupied=int(np.unique(mapped.indices).size)
    unique=len(terms)
    return {
        'hash_dimensions':int(n_features),
        'unique_train_identity_terms':unique,
        'occupied_hash_buckets':occupied,
        'colliding_identity_terms':unique-occupied,
        'collision_fraction':float((unique-occupied)/max(unique,1)),
        'alternate_sign':False,
        'same_six_identity_terms_as_exact':True,
        'standardized_numeric_columns_are_explicit_not_hashed':True,
    }


def _assemble(prepared,views,spec,task):
    target_names=spec['nodes']['pathway'] if task=='pathway' else spec['states']
    assembled={}
    for split in ('train','validation'):
        pieces=[]
        for view in views:
            if view['split']!=split:
                continue
            arr=load_npz(Path(prepared)/'views'/view['id']/f'{task}.npz')
            cancer=spec['cancers'][view['cancer_index']]
            rows=[(cancer,spec['nodes']['lncRNA'][int(lnc)],target_names[int(target)])
                  for lnc,target in zip(arr['lnc'],arr['target'])]
            pieces.append((arr,cancer,rows))
        if not pieces:
            fail('BASELINE_SPLIT_EMPTY','Matched comparison requires train and validation rows',task=task,split=split)
        assembled[split]={
            'rows':[row for _,_,rows in pieces for row in rows],
            'numeric':np.concatenate([arr['numeric'] for arr,_,_ in pieces]),
            'label':np.concatenate([arr['label'] for arr,_,_ in pieces]),
            'available':np.concatenate([arr['available'] for arr,_,_ in pieces]).astype(bool),
            'weight':np.concatenate([arr['weight'] for arr,_,_ in pieces]),
            'base':np.concatenate([arr['base'] for arr,_,_ in pieces]),
            'cancer':np.concatenate([np.repeat(cancer,len(arr['lnc'])) for arr,cancer,_ in pieces]),
            'target_name':np.concatenate([[target_names[int(target)] for target in arr['target']]
                                          for arr,_,_ in pieces]),
        }
    return assembled


def _score(task,data,spec,values,mask=None):
    take=np.ones(len(data['label']),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    cancer=data['cancer'][take]
    target=data['target_name'][take]
    groups=np.asarray([spec['state_families'][name] for name in target]) if task=='state' else None
    target_ids=target if task=='state' else None
    return macro_ap(data['label'][take],np.asarray(values)[take],cancer,data['available'][take],groups,target_ids)


def _fit_hash(task,assembled,spec,cfg,n_features):
    train=assembled['train']
    validation=assembled['validation']
    take=np.flatnonzero(train['available'])
    train_rows=[train['rows'][index] for index in take]
    x=hashed_design(train_rows,train['numeric'][take],n_features)
    xv=hashed_design(validation['rows'],validation['numeric'],n_features)
    weights=train['weight'][take].astype(float)
    weights/=max(float(weights.mean()),1e-12)
    chosen=None
    trials=[]
    for cpen in cfg['baseline']['C']:
        l1args={'l1_ratio':1.0} if inspect.signature(LogisticRegression).parameters['penalty'].default=='deprecated' else {'penalty':'l1'}
        model=LogisticRegression(C=float(cpen),solver='liblinear',max_iter=cfg['baseline']['max_iter'],random_state=cfg['seed'],**l1args)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model.fit(x,train['label'][take],sample_weight=weights)
        except ConvergenceWarning:
            trials.append({'C':float(cpen),'status':'NONCONVERGED_EXCLUDED'})
            continue
        logits=model.decision_function(xv)
        score=_score(task,validation,spec,logits)
        trials.append({'C':float(cpen),'macro_cancer_auprc':score})
        if chosen is None or score>chosen[0]:
            chosen=(score,float(cpen),logits)
    if chosen is None:
        fail('HASH_BASELINE_NONCONVERGENCE','No converged matched hash4096 LASSO candidate',task=task)
    return chosen,trials,hash_collision_report(train_rows,n_features)


def _ablation_design(rows,numeric,vectorizer,numeric_columns,use_identity):
    parts=[]
    if use_identity:parts.append(vectorizer.transform(identity_records(rows)))
    columns=np.asarray(numeric_columns,dtype=np.int64)
    if columns.size:parts.append(sparse.csr_matrix(np.asarray(numeric)[:,columns],dtype=np.float64))
    if not parts:fail('BASELINE_ABLATION_EMPTY','At least one declared feature block is required')
    matrix=sparse.hstack(parts,format='csr',dtype=np.float64) if len(parts)>1 else parts[0]
    return _csr32(matrix)


def _fit_ablation(task,assembled,spec,cfg,vectorizer,numeric_columns,use_identity,label,return_logits=False):
    train=assembled['train'];validation=assembled['validation'];take=np.flatnonzero(train['available'])
    train_rows=[train['rows'][index] for index in take]
    x=_ablation_design(train_rows,train['numeric'][take],vectorizer,numeric_columns,use_identity)
    xv=_ablation_design(validation['rows'],validation['numeric'],vectorizer,numeric_columns,use_identity)
    weights=train['weight'][take].astype(float);weights/=max(float(weights.mean()),1e-12)
    chosen=None;trials=[]
    for cpen in cfg['baseline']['C']:
        l1args={'l1_ratio':1.0} if inspect.signature(LogisticRegression).parameters['penalty'].default=='deprecated' else {'penalty':'l1'}
        model=LogisticRegression(C=float(cpen),solver='liblinear',max_iter=cfg['baseline']['max_iter'],random_state=cfg['seed'],**l1args)
        try:
            with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);model.fit(x,train['label'][take],sample_weight=weights)
        except ConvergenceWarning:
            trials.append({'C':float(cpen),'status':'NONCONVERGED_EXCLUDED'});continue
        logits=model.decision_function(xv);score=_score(task,validation,spec,logits)
        trials.append({'C':float(cpen),'macro_cancer_auprc':score})
        if chosen is None or score>chosen[0]:chosen=(score,float(cpen),model)
    if chosen is None:fail('BASELINE_ABLATION_NONCONVERGENCE','No converged candidate',task=task,ablation=label)
    score,cpen,model=chosen;coef=model.coef_.ravel();identity_width=len(vectorizer.vocabulary_) if use_identity else 0
    report={'label':label,'validation_macro_ap':score,'C':cpen,'trials':trials,'identity_enabled':bool(use_identity),
        'numeric_columns':[int(v) for v in numeric_columns],'feature_count':int(x.shape[1]),'nonzero_coefficients':int(np.count_nonzero(coef)),
        'nonzero_identity_coefficients':int(np.count_nonzero(coef[:identity_width])),'nonzero_numeric_coefficients':int(np.count_nonzero(coef[identity_width:])),
        'coefficient_l1_norm':float(np.abs(coef).sum()),'same_rows_labels_weights_and_C_grid':True,'outer_test_read':False}
    return (report,model.decision_function(xv)) if return_logits else report


def compare(prepared,out,hash_dimensions=4096):
    if socket.gethostname()!='149':
        fail('WRONG_HOST','Baseline comparison is authorized only on host 149',host=socket.gethostname())
    prepared=Path(prepared)
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        fail('OUTPUT_EXISTS','Baseline comparison never overwrites an audit directory',path=str(out))
    out.mkdir(parents=True,exist_ok=True)
    manifest,spec,cfg,lineage=validate_prepared(prepared)
    views=read_json(prepared/'VIEWS.json')
    baseline=read_json(prepared/'BASELINES.json')
    results={}
    for task in ('pathway','state'):
        assembled=_assemble(prepared,views,spec,task)
        validation=assembled['validation']
        exact=_score(task,validation,spec,validation['base'])
        expected=float(baseline[task]['validation_macro_ap'])
        if abs(exact-expected)>1e-6:
            fail('EXACT_BASELINE_REPLAY_DRIFT','Stored exact logits do not reproduce their receipt',task=task,expected=expected,actual=exact)
        (hashed,cpen,hash_logits),trials,collisions=_fit_hash(task,assembled,spec,cfg,int(hash_dimensions))
        train=assembled['train'];train_take=np.flatnonzero(train['available'])
        train_rows=[train['rows'][index] for index in train_take]
        vectorizer,_=fit_vocabulary(train_rows,baseline_contract(cfg))
        all_numeric=list(range(train['numeric'].shape[1]));association_slice=spec.get('singlecell_association_slice')
        without_association=all_numeric
        if association_slice is not None:
            start,stop=map(int,association_slice);without_association=[index for index in all_numeric if not start<=index<stop]
        ablations={
            'full_exact_refit':_fit_ablation(task,assembled,spec,cfg,vectorizer,all_numeric,True,'EXACT_IDENTITY_PLUS_ALL_STANDARDIZED_NUMERIC'),
            'identity_only':_fit_ablation(task,assembled,spec,cfg,vectorizer,[],True,'EXACT_IDENTITY_ONLY'),
            'numeric_only':_fit_ablation(task,assembled,spec,cfg,vectorizer,all_numeric,False,'ALL_STANDARDIZED_NUMERIC_ONLY'),
            'identity_plus_discovery':_fit_ablation(task,assembled,spec,cfg,vectorizer,list(range(min(5,len(all_numeric)))),True,'EXACT_IDENTITY_PLUS_FIVE_DISCOVERY_STATISTICS'),
        }
        if association_slice is not None:
            ablations['without_singlecell_association']=_fit_ablation(task,assembled,spec,cfg,vectorizer,without_association,True,'EXACT_IDENTITY_PLUS_ALL_NUMERIC_EXCEPT_SINGLECELL_ASSOCIATION')
        by_cancer=[]
        for cancer in spec['cancers']:
            take=validation['cancer']==cancer
            if not take.any():
                continue
            row={
                'cancer_id':cancer,
                'exact_auprc':_score(task,validation,spec,validation['base'],take),
                'hash4096_auprc':_score(task,validation,spec,hash_logits,take),
            }
            row['exact_minus_hash4096']=row['exact_auprc']-row['hash4096_auprc']
            by_cancer.append(row)
        results[task]={
            'exact_validation_macro_ap':exact,
            'exact_selected_C':baseline[task]['C'],
            'exact_identity_features':baseline[task]['identity_feature_count'],
            'hash4096_validation_macro_ap':hashed,
            'hash4096_selected_C':cpen,
            'exact_minus_hash4096':exact-hashed,
            'hash4096_trials':trials,
            'collision_report':collisions,
            'feature_block_ablations':ablations,
            'interpretation_note':'The primary comparator is a multimodal sparse logistic L1 model, not an expression-only LASSO.',
            'by_cancer':by_cancer,
        }
    report={
        'format':FORMAT,
        'host':'149',
        'prepared_manifest_sha256':sha(prepared/'MANIFEST.json'),
        'config_sha256':digest(cfg),
        'lineage':lineage,
        'comparison_scope':'SAME_R62_PREPARED_VALIDATION_ROWS_LABELS_WEIGHTS_C_GRID_AND_STANDARDIZED_NUMERIC_FEATURES;IDENTITY_ENCODING_ONLY',
        'historical_hash_baseline_exact_reproduction':False,
        'historical_difference':'Earlier pathway baseline also hashed family and numeric terms and used a different candidate universe; this is the controlled collision ablation.',
        'outer_test_read':False,
        'historical_score_directly_comparable':False,
        'results':results,
    }
    atomic_json(out/'BASELINE_COMPARISON.json',report)
    commit_manifest(out,{'role':'VALIDATION_ONLY_BASELINE_IDENTITY_ENCODING_AUDIT','prepared_manifest_sha256':report['prepared_manifest_sha256'],'outer_test_read':False})
    status(out,'BASELINE_COMPARISON_COMPLETE_NOT_TRAINING',report_sha256=sha(out/'BASELINE_COMPARISON.json'))
    return report
