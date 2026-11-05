"""Matched non-RNA outcomes for a three-arm RNA predictor comparison.

Native CPTAC protein / phosphosite / methylation outcomes retain their upstream
normalization. This is a retrospective comparison, not an inductive validation
of that upstream preprocessing, and not evidence of causal regulation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ARMS = ('lncrna', 'mrna', 'combined')
ARM_LABELS = {'lncrna': 'lncRNA', 'mrna': 'mRNA', 'combined': 'lncRNA + mRNA'}
LAYERS = ('PROTEIN_PATHWAY', 'PHOSPHO_PTMSEA', 'METHYLATION_PATHWAY')


def feature_axes(annotation):
    a = annotation.copy()
    if a.gene_id.duplicated().any():
        raise ValueError('Duplicate gene annotation')
    a = a.loc[a.gene_biotype.isin(['lncRNA', 'protein_coding'])].copy()
    a['feature_type'] = a.gene_biotype.map({'lncRNA': 'lncrna', 'protein_coding': 'mrna'})
    a['feature_id'] = np.where(a.feature_type.eq('lncrna'), 'LNC:', 'MRNA:') + a.gene_id
    a = pd.concat([a.loc[a.feature_type.eq(t)].sort_values('gene_id') for t in ARMS[:2]], ignore_index=True)
    if not all(a.feature_type.eq(t).any() for t in ARMS[:2]):
        raise ValueError('Both measured RNA feature classes are required')
    ids = np.arange(len(a))
    axes = {t: ids[a.feature_type.eq(t)] for t in ARMS[:2]}
    axes['combined'] = ids
    return a, axes


def select_targets(metadata, cohort='CPTAC'):
    # Explicit assay-layer whitelist. RNA-derived States, TF activities and RNA
    # pathways never enter this comparison merely because of their names.
    m = metadata.loc[metadata.cohort.eq(cohort) & metadata.layer.isin(LAYERS)].copy()
    if m.target_id.duplicated().any():
        raise ValueError('Duplicate native assay target')
    if not m.target_id.str.split('|').str[0].eq(m.layer).all():
        raise ValueError('Target layer/provenance mismatch')
    m['scope'] = 'NATIVE_CPTAC_ONLY'
    m['analysis_role'] = np.where(m.layer.eq('METHYLATION_PATHWAY'), 'SECONDARY', 'PRIMARY')
    m['normalization_limit'] = 'Upstream native assay preprocessing retained; not fitted anew within outer folds'
    return m.reset_index(drop=True)


def restrict_split(spec, full_index, kept_index):
    lookup = {pid: i for i, pid in enumerate(kept_index)}
    out = {}
    for phase, entry in spec.items():
        out[phase] = {}
        for key, values in entry.items():
            if key in ('train', 'validation', 'test'):
                out[phase][key] = [lookup[full_index[k]] for k in values if full_index[k] in lookup]
            else:
                out[phase][key] = values
        tr = set(out[phase].get('train', []))
        va = set(out[phase].get('validation', []))
        te = set(out[phase].get('test', []))
        if tr & va or tr & te or va & te:
            raise ValueError('Patient leakage in supplied split')
    return out


def matched_predictions(frames):
    keys = ['patient_id', 'target_id', 'phase', 'cohort']
    result = None
    for arm in ARMS:
        f = frames[arm].copy()
        if f.duplicated(keys).any():
            raise ValueError('Duplicate held-out predictions')
        f = f.set_index(keys).sort_index()
        if result is None:
            result = f[['observed', 'train_mean_baseline', 'covariate_only_baseline']].copy()
        if not f.index.equals(result.index):
            raise ValueError('Three arms did not evaluate identical held-out patients')
        for c in ('observed', 'train_mean_baseline', 'covariate_only_baseline'):
            if not np.allclose(f[c], result[c], rtol=1e-10, atol=1e-10, equal_nan=False):
                raise ValueError('Three-arm outcome/covariate-baseline mismatch: ' + c)
        result['prediction_' + arm] = f.predicted
    return result.reset_index()


def regression_metrics(y, p):
    from scipy.stats import spearmanr
    y = np.asarray(y, float); p = np.asarray(p, float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite paired outcome/prediction')
    sst = float(np.sum((y-y.mean())**2))
    mse = float(np.mean((y-p)**2))
    return dict(n=len(y), r2=1-float(np.sum((y-p)**2))/sst if sst > 1e-20 else np.nan,
                rmse=np.sqrt(mse), mse=mse,
                spearman=float(spearmanr(y,p).statistic) if len(y)>2 and np.std(y)>1e-12 and np.std(p)>1e-12 else np.nan)


def compare_target(frame, seed=1405, bootstrap=1000):
    y = frame.observed.to_numpy(float)
    row = dict(n_patients=len(y), evaluated_folds=frame.phase.nunique())
    for arm in ARMS:
        row.update({k+'_'+arm: v for k,v in regression_metrics(y, frame['prediction_'+arm]).items() if k!='n'})
    base = regression_metrics(y, frame.covariate_only_baseline)
    row['r2_covariates'] = base['r2']
    row['delta_r2_combined_vs_mrna'] = row['r2_combined']-row['r2_mrna']
    row['delta_r2_mrna_vs_lncrna'] = row['r2_mrna']-row['r2_lncrna']
    row['delta_r2_combined_vs_covariates'] = row['r2_combined']-base['r2']
    score = np.array([row['r2_'+arm] for arm in ARMS])
    if not np.isfinite(score).all():
        row['best_arm'] = 'UNDEFINED_CONSTANT_Y'
    else:
        winners = np.flatnonzero(np.isclose(score, score.max(), rtol=0, atol=1e-9))
        row['best_arm'] = ARMS[winners[0]] if len(winners)==1 else 'TIE'
    wins = []
    for _,f in frame.groupby('phase'):
        d = regression_metrics(f.observed, f.prediction_combined)['mse'] - regression_metrics(f.observed,f.prediction_mrna)['mse']
        wins.append(d < -1e-12)
    row['folds_combined_lower_mse_than_mrna'] = int(sum(wins))
    # A paired patient bootstrap of fixed OOF errors. Conditional uncertainty
    # only: it does not re-fit the models or correct multiple target selection.
    rng = np.random.default_rng(seed)
    draw = rng.integers(0,len(y),size=(bootstrap,len(y)))
    delta_error = ((y-frame.prediction_mrna.to_numpy())**2 - (y-frame.prediction_combined.to_numpy())**2)
    yb=y[draw]; den=np.sum((yb-yb.mean(axis=1,keepdims=True))**2,axis=1)
    valid=den>1e-20
    delta=np.sum(delta_error[draw[valid]],axis=1)/den[valid]
    lo,hi=np.quantile(delta,[.025,.975]) if len(delta) else [np.nan,np.nan]
    row.update(delta_r2_conditional_bootstrap_low=float(lo),delta_r2_conditional_bootstrap_high=float(hi),bootstrap_replicates=bootstrap)
    return row
