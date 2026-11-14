"""Observed fused relationship strength for the exact-pathway training task."""
from __future__ import annotations

import numpy as np

OBJECTIVE = 'OBSERVED_FUSED_RMS_EFFECT_V1'


def context_for_view(view: str) -> str:
    if view == 'validation':
        return view
    if view.startswith('train') and view[5:].isdigit() and 0 <= int(view[5:]) < 5:
        return 'train_label_fold' + view[5:]
    raise ValueError('RELATION_STRENGTH_UNKNOWN_VIEW: ' + view)


def verify_patient_roles(context, source_patients_by_cohort, *, expected_context=None):
    """Require the actual feature patients to be the frozen source partition."""
    if expected_context is not None and context.context != expected_context:
        raise ValueError('RELATION_STRENGTH_CONTEXT_MISMATCH')
    if set(source_patients_by_cohort) != set(context.cohort_splits):
        raise ValueError('RELATION_STRENGTH_COHORT_SET_MISMATCH')
    for cohort, split in context.cohort_splits.items():
        feature = tuple(source_patients_by_cohort[cohort])
        frozen = tuple(split['source_patients'])
        labels = set(split['label_patients'])
        if len(feature) != len(set(feature)) or set(feature) != set(frozen):
            raise ValueError('RELATION_STRENGTH_SOURCE_PATIENT_MISMATCH: ' + cohort)
        if set(feature) & labels:
            raise ValueError('RELATION_STRENGTH_PATIENT_LEAK: ' + cohort)


def exact_relation_strength(context, lncrna_ids, canonical_ids, *, source_patients_by_cohort,
                            expected_context=None):
    """Return measured RMS effect and its mask; unobserved pairs stay missing."""
    verify_patient_roles(context, source_patients_by_cohort, expected_context=expected_context)
    if len(lncrna_ids) != len(canonical_ids):
        raise ValueError('RELATION_STRENGTH_PAIR_AXIS_MISMATCH')
    if len(set(context.lncrna_ids)) != len(context.lncrna_ids) or len(set(context.canonical_ids)) != len(context.canonical_ids):
        raise ValueError('RELATION_STRENGTH_CONTEXT_ID_DUPLICATE')
    lnc_axis = {value: index for index, value in enumerate(context.lncrna_ids)}
    target_axis = {value: index for index, value in enumerate(context.canonical_ids)}
    unknown_lnc = set(lncrna_ids) - set(lnc_axis)
    unknown_target = set(canonical_ids) - set(target_axis)
    if unknown_lnc or unknown_target:
        raise ValueError('RELATION_STRENGTH_ID_NOT_IN_FROZEN_AXIS')
    li = np.fromiter((lnc_axis[value] for value in lncrna_ids), dtype=np.int64, count=len(lncrna_ids))
    ti = np.fromiter((target_axis[value] for value in canonical_ids), dtype=np.int64, count=len(canonical_ids))
    observed = np.asarray(context.arrays['observed'][li, ti], dtype=bool)
    effect = np.asarray(context.arrays['rms_effect'][li, ti], dtype=np.float32)
    q = np.asarray(context.arrays['combined_q'][li, ti], dtype=np.float32)
    membership = np.asarray(context.arrays['membership'][li, ti], dtype=np.uint8)
    if not np.array_equal(np.isfinite(effect), observed) or not np.array_equal(np.isfinite(q), observed):
        raise ValueError('RELATION_STRENGTH_OBSERVATION_MISMATCH')
    if np.any(effect[observed] < 0) or np.any(effect[observed] > 1 + 1e-6):
        raise ValueError('RELATION_STRENGTH_EFFECT_RANGE')
    if np.any(membership[~observed] != 0):
        raise ValueError('RELATION_STRENGTH_UNOBSERVED_SUPPORT')
    return {'label': np.where(observed, effect, 0).astype(np.float32),
            'available': observed,
            'combined_q': q,
            'membership': membership}
