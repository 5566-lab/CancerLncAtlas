import json
import numpy as np
import pytest

from cla_engine.common import atomic_json, atomic_npz, sha, digest
from cla_engine.prepare import deterministic_pairs
from cla_engine.relation_components import load_relation_components, FORMAT
from cla_engine.relation_features import COHORTS, LAYERS, ENTITY_FIELDS
from cla_engine.relation_source import SOURCE_NUMERIC_FIELDS


def seal(root, update=None):
    path = root / 'COMPLETE.json'
    receipt = json.loads(path.read_text()) if path.exists() else {}
    receipt.update(update or {})
    receipt['files'] = {p.name: sha(p) for p in root.iterdir() if p.name != 'COMPLETE.json'}
    atomic_json(path, receipt)


@pytest.fixture
def components(tmp_path):
    lncs, targets, groups = ['l1', 'l2'], ['p1', 's1'], ['pathway', 'state']
    li, ti = deterministic_pairs(lncs, targets, 4, 1, 'BRCA')
    atomic_json(tmp_path / 'LAUNCH_PREFLIGHT.json', {'target_host': '149', 'cancer': 'BRCA',
        'context': 'train_label_fold2', 'sealed_test_opened': False, 'candidate_budget': 4, 'seed': 1})
    atomic_json(tmp_path / 'AXES.json', {'lncrna_ids': lncs, 'canonical_ids': targets,
        'loss_groups': groups, 'candidate_selection': 'DETERMINISTIC_LABEL_INDEPENDENT', 'seed': 1})
    atomic_json(tmp_path / 'SPLIT.json', {c: {'source_patients': [c + '_s'],
        'label_patients': [c + '_l'], 'outer_fold': 0, 'source_folds': [3, 4], 'label_folds': [2]} for c in COHORTS})
    names = [n for c in COHORTS for n in ([f'{c}/{layer}/{field}' for layer in LAYERS for field in SOURCE_NUMERIC_FIELDS]
        + [f'{c}/{field}' for field in ENTITY_FIELDS])]
    atomic_json(tmp_path / 'FEATURE_NAMES.json', names)
    atomic_json(tmp_path / 'READOUT_AXES.json', {c + '/' + layer: {'readout_ids': [], 'canonical_ids': [], 'metadata': []}
        for c in COHORTS for layer in LAYERS})
    atomic_json(tmp_path / 'AUDIT.json', [{'cohort': c, 'layer': layer} for c in COHORTS for layer in LAYERS])
    for c in COHORTS:
        for layer in LAYERS:
            atomic_npz(tmp_path / f'{c}_{layer}_SOURCE_ASSOCIATIONS.npz',
                lncrna_index=np.empty(0, np.int64), readout_index=np.empty(0, np.int64),
                rho=np.empty(0, np.float32), p=np.empty(0, np.float32), n=np.empty(0, np.int32),
                rank=np.empty(0, np.int16), available=np.empty(0, bool))
    atomic_npz(tmp_path / 'LNC_NODE_FEATURES.npz', **{c: np.zeros((2, 6), np.float32) for c in COHORTS})
    atomic_npz(tmp_path / 'CANDIDATES.npz', lnc=li, target=ti, numeric=np.zeros((4, 72), np.float32),
               label=np.full(4, .2, np.float32), available=np.ones(4, bool))
    seal(tmp_path, {'target_host': '149', 'format': FORMAT, 'cancer': 'BRCA',
        'context': 'train_label_fold2', 'full_prepared_capsule': False,
        'candidates': 4, 'numeric_features': 72, 'lncrna_axis': 2, 'target_axis': 2,
        'observed_labels': 4, 'observed_labels_by_task': {'pathway': 2, 'state': 2},
        'targets_in_candidates': 2, 'lncrnas_in_candidates': 2,
        'candidate_hash': digest(list(zip(li.tolist(), ti.tolist())))})
    return tmp_path


def test_missing_readouts_remain_explicitly_empty(components):
    data = load_relation_components(components)
    assert data['arrays']['available'].all()
    assert not data['arrays']['numeric'].any()
    assert all(len(a['rho']) == 0 for a in data['associations'].values())


def test_numeric_label_injection_rejected_even_after_rehash(components):
    with np.load(components / 'CANDIDATES.npz') as archive:
        arrays = dict(archive)
    arrays['numeric'][:, 0] = arrays['label']
    atomic_npz(components / 'CANDIDATES.npz', **arrays)
    seal(components)
    with pytest.raises(ValueError, match='NUMERIC_SOURCE_MISMATCH'):
        load_relation_components(components)


def test_patient_overlap_rejected_even_after_rehash(components):
    splits = json.loads((components / 'SPLIT.json').read_text())
    splits['TCGA']['label_patients'] = splits['TCGA']['source_patients']
    atomic_json(components / 'SPLIT.json', splits)
    seal(components)
    with pytest.raises(ValueError, match='PATIENT_SPLIT'):
        load_relation_components(components)


def test_candidate_order_is_label_independent_and_fixed(components):
    with np.load(components / 'CANDIDATES.npz') as archive:
        arrays = {key: archive[key][::-1].copy() for key in archive.files}
    atomic_npz(components / 'CANDIDATES.npz', **arrays)
    seal(components, {'candidate_hash': digest(list(zip(arrays['lnc'].tolist(), arrays['target'].tolist())))})
    with pytest.raises(ValueError, match='CANDIDATE_SELECTION'):
        load_relation_components(components)


def test_changed_file_is_rejected(components):
    (components / 'FEATURE_NAMES.json').write_text('[]')
    with pytest.raises(ValueError, match='FILE_DRIFT'):
        load_relation_components(components)


def test_missing_label_storage_requires_explicit_mask(components):
    with np.load(components / 'CANDIDATES.npz') as archive:
        arrays = dict(archive)
    arrays['available'][0] = False
    atomic_npz(components / 'CANDIDATES.npz', **arrays)
    seal(components)
    with pytest.raises(ValueError, match='LABEL_MASK'):
        load_relation_components(components)
