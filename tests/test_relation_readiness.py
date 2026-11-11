"""Continuous fused labels retain observation masks in prepared validation."""
import numpy as np
import pytest

from cla_engine.common import Stop
from cla_engine.readiness import validate_task_arrays


def block():
    return {'lnc': np.array([0, 1, 2]), 'target': np.array([0, 0, 1]),
            'numeric': np.ones((3, 2)), 'base': np.zeros(3),
            'label': np.array([.2, .7, 999.]),
            'available': np.array([True, True, False]),
            'weight': np.ones(3), 'direction_label': np.zeros(3),
            'direction_available': np.zeros(3, dtype=bool)}


def test_observed_strength_keeps_unobserved_label_masked():
    result = validate_task_arrays(block(), 'BRCA__train2', 'pathway', 2,
                                  objective='observed_fused_rms_effect')
    assert result['usable'] == 2
    assert result['positive'] is None
    assert result['label_min'] == .2
    assert result['label_max'] == .7


def test_observed_strength_rejects_bad_value_and_direction_head():
    bad = block()
    bad['label'][0] = 1.2
    with pytest.raises(Stop) as error:
        validate_task_arrays(bad, 'BRCA__train2', 'pathway', 2,
                             objective='observed_fused_rms_effect')
    assert error.value.code == 'TASK_LABEL_OR_WEIGHT_INVALID'
    bad = block()
    bad['direction_available'][0] = True
    with pytest.raises(Stop) as error:
        validate_task_arrays(bad, 'BRCA__train2', 'pathway', 2,
                             objective='observed_fused_rms_effect')
    assert error.value.code == 'RELATION_STRENGTH_DIRECTION_HEAD_NOT_DECLARED'


def test_binary_contract_still_requires_zero_one():
    with pytest.raises(Stop) as error:
        validate_task_arrays(block(), 'BRCA__train2', 'pathway', 2)
    assert error.value.code == 'TASK_LABEL_OR_WEIGHT_INVALID'
