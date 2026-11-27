import numpy as np
import pytest

from cla_engine.relation_metrics import paired_relation_strength_metrics


def test_exact_target_macro_does_not_let_large_target_dominate():
    y = np.array([.1, .2, .3, .4, .9, 1.0, 0.0])
    score = np.array([.1, .2, .3, .4, .8, .9, 999.0])
    baseline = np.zeros_like(y)
    result = paired_relation_strength_metrics(y, score, baseline,
                                              np.array([1, 1, 1, 1, 1, 1, 0]),
                                              np.array([0, 0, 0, 0, 1, 1, 1]),
                                              task='pathway', targets=('a', 'b'))
    assert result['mse'] == pytest.approx(.005)
    assert result['n'] == 6
    assert result['delta_mse'] > 0


def test_state_family_macro_and_missing_labels():
    y = np.array([.1, .2, .4, .5, .8, .9, np.nan])
    score = np.array([.1, .2, .4, .5, .8, .9, 0.])
    result = paired_relation_strength_metrics(y, score, np.zeros(7),
                                              [1, 1, 1, 1, 1, 1, 0],
                                              [0, 0, 1, 1, 2, 2, 2], task='state',
                                              targets=('a', 'b', 'c'),
                                              families={'a': 'f1', 'b': 'f1', 'c': 'f2'})
    assert result['mse'] == 0
    assert len(result['families']) == 2


def test_unobserved_is_not_zero_label():
    with pytest.raises(ValueError, match='LABEL_INVALID'):
        paired_relation_strength_metrics([np.nan, .5], [.1, .2], [.1, .2],
                                         [1, 1], [0, 0], task='pathway', targets=('p',))
