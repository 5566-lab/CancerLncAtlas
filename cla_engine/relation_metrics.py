"""Paired validation for observed fused relation-strength regression."""
from __future__ import annotations

import numpy as np


def paired_relation_strength_metrics(y, score, baseline, available, target, *, task,
                                     targets, families=None):
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    available = np.asarray(available, dtype=bool)
    target = np.asarray(target, dtype=int)
    if y.ndim != 1 or any(x.shape != y.shape for x in (score, baseline, available, target)):
        raise ValueError('RELATION_METRIC_AXIS_MISMATCH')
    if not np.isfinite(score).all() or not np.isfinite(baseline).all():
        raise ValueError('RELATION_METRIC_PREDICTION_NONFINITE')
    if not np.isfinite(y[available]).all() or np.any(y[available] < 0) or np.any(y[available] > 1 + 1e-6):
        raise ValueError('RELATION_METRIC_LABEL_INVALID')
    if np.any(target < 0) or np.any(target >= len(targets)):
        raise ValueError('RELATION_METRIC_TARGET_OUT_OF_RANGE')
    if task not in {'pathway', 'state'} or (task == 'state' and families is None):
        raise ValueError('RELATION_METRIC_TASK_INVALID')
    units = []
    for index, name in enumerate(targets):
        mask = available & (target == index)
        n = int(mask.sum())
        if n < 2:
            units.append({'target_id': name, 'n': n, 'available': False})
            continue
        err = y[mask] - score[mask]
        base_err = y[mask] - baseline[mask]
        units.append({'target_id': name, 'n': n, 'available': True,
                      'mse': float(np.mean(err * err)),
                      'baseline_mse': float(np.mean(base_err * base_err)),
                      'mae': float(np.mean(np.abs(err))),
                      'baseline_mae': float(np.mean(np.abs(base_err))),
                      'family': families[name] if task == 'state' else 'pathway'})
    good = [row for row in units if row['available']]
    result = {'n': int(available.sum()), 'available': bool(good), 'targets': units,
              'unit': 'EXACT_TARGET_MACRO_WITHIN_CANCER' if task == 'pathway'
                      else 'EXACT_TARGET_MACRO_WITHIN_FAMILY_THEN_FAMILY_MACRO_WITHIN_CANCER'}
    if not good:
        return {**result, 'reason': 'NO_TARGET_WITH_TWO_OBSERVED_RELATIONS'}
    names = ('mse', 'baseline_mse', 'mae', 'baseline_mae')
    if task == 'state':
        groups = []
        for family in sorted({row['family'] for row in good}):
            rows = [row for row in good if row['family'] == family]
            groups.append({'family': family, 'n_evaluable_targets': len(rows),
                           **{key: float(np.mean([row[key] for row in rows])) for key in names}})
        result['families'] = groups
        values = groups
    else:
        values = good
    result.update({key: float(np.mean([row[key] for row in values])) for key in names})
    result['delta_mse'] = result['baseline_mse'] - result['mse']
    return result
