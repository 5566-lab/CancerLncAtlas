import hashlib
import json
import numpy as np
import pytest
from cla_engine.functional_y import (validate_relation_arrays, functional_task_weights,
                                     load_functional_context)


def fixture_arrays():
    # Columns: no measurement, observed without support, one RNA family,
    # RNA+protein strict, discordant RNA+phospho strict.
    return {name: np.array([value], dtype=np.uint8) for name, value in {
        'membership': [0, 0, 1, 2, 2], 'direction': [0, 0, 1, 2, 3],
        'observed_assay_bits': [0, 1, 1, 3, 5],
        'strict_assay_bits': [0, 0, 1, 3, 5],
        'supported_assay_bits': [0, 0, 1, 3, 5],
    }.items()}


def test_distinct_assay_count_and_unlabelled_semantics():
    arrays = fixture_arrays()
    validate_relation_arrays(arrays, (1, 5))
    assert arrays['membership'][0, 1] == 0
    assert arrays['observed_assay_bits'][0, 1] != 0
    arrays['membership'][0, 2] = 2
    with pytest.raises(ValueError, match='distinct assay'):
        validate_relation_arrays(arrays, (1, 5))


@pytest.mark.parametrize('field,value', [('supported_assay_bits', 2), ('strict_assay_bits', 2)])
def test_support_requires_observed_assay(field, value):
    arrays = fixture_arrays()
    arrays[field][0, 2] = value
    with pytest.raises(ValueError, match='Supported assay'):
        validate_relation_arrays(arrays, (1, 5))


def test_shared_identity_has_one_weight():
    np.testing.assert_array_equal(functional_task_weights(['Pathway', 'State', 'Pathway|State']),
                                  [1, 0.5, 1])


def make_context(tmp_path, overlap=False):
    base = tmp_path / 'BRCA' / 'sealed_test'
    base.mkdir(parents=True)
    (base / 'AXES.json').write_text(json.dumps({'lncrna_ids': ['L1'],
                                               'canonical_ids': ['T'+str(i) for i in range(5)]}))
    (base / 'SPLIT.json').write_text(json.dumps({'source_patients': ['P1'],
        'label_patients': ['P1' if overlap else 'P2']}))
    np.savez_compressed(base / 'Y.npz', **fixture_arrays())
    (base / 'COMPLETE.json').write_text(json.dumps({'cancer_id': 'BRCA', 'context': 'sealed_test',
        'files': {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                  for name in ['AXES.json', 'SPLIT.json', 'Y.npz']}}))
    return base


def test_context_round_trip_and_changed_file_rejected(tmp_path):
    base = make_context(tmp_path)
    loaded = load_functional_context(tmp_path, 'BRCA', 'sealed_test')
    assert loaded.counts()['strong_positive'] == 2
    assert loaded.counts()['unlabeled_observed'] == 1
    (base / 'AXES.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        load_functional_context(tmp_path, 'BRCA', 'sealed_test')


def test_source_label_overlap_rejected(tmp_path):
    make_context(tmp_path, overlap=True)
    with pytest.raises(ValueError, match='overlap'):
        load_functional_context(tmp_path, 'BRCA', 'sealed_test')
