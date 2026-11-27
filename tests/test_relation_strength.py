import numpy as np
import pytest

from cla_engine.functional_fusion import FusedFunctionalContext
from cla_engine.relation_strength import context_for_view, exact_relation_strength


def fixture():
    split = {'source_patients': ['a', 'b'], 'label_patients': ['c']}
    empty = {'source_patients': [], 'label_patients': ['p']}
    arrays = {'observed': np.array([[1, 0], [1, 1]], dtype=np.uint8),
              'rms_effect': np.array([[.25, np.nan], [.3, .4]], dtype=float),
              'combined_q': np.array([[.9, np.nan], [.01, .2]], dtype=float),
              'membership': np.array([[0, 0], [2, 0]], dtype=np.uint8)}
    return FusedFunctionalContext('BRCA', 'train_label_fold2', ('lnc1', 'lnc2'), ('path', 'state'),
                                  ('Pathway', 'State'), ('Pathway', 'State'),
                                  {'TCGA': split, 'CPTAC': empty}, arrays)


def test_view_mapping_and_continuous_observed_unlabelled():
    assert context_for_view('train2') == 'train_label_fold2'
    assert context_for_view('validation') == 'validation'
    result = exact_relation_strength(fixture(), ['lnc1', 'lnc2', 'lnc1'],
                                     ['path', 'path', 'state'],
                                     source_patients_by_cohort={'TCGA': ['b', 'a'], 'CPTAC': []},
                                     expected_context='train_label_fold2')
    assert result['label'].tolist() == pytest.approx([.25, .3, 0])
    assert result['available'].tolist() == [True, True, False]
    assert result['membership'].tolist() == [0, 2, 0]


def test_patient_overlap_and_changed_source_are_rejected():
    with pytest.raises(ValueError, match='SOURCE_PATIENT_MISMATCH'):
        exact_relation_strength(fixture(), ['lnc1'], ['path'],
                                source_patients_by_cohort={'TCGA': ['a', 'c'], 'CPTAC': []})


def test_unknown_target_cannot_be_silently_zero_filled():
    with pytest.raises(ValueError, match='ID_NOT_IN_FROZEN_AXIS'):
        exact_relation_strength(fixture(), ['lnc1'], ['other'],
                                source_patients_by_cohort={'TCGA': ['a', 'b'], 'CPTAC': []})
