"""Continuous held-out evaluation uses observed relation effects and paired MSE."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cla_engine import evaluate as outer
from cla_engine.common import Stop, digest
from cla_engine.execution_contract import CONTINUOUS_METRIC_CONTRACT
from cla_engine.lineage import FIELDS, completion_state


def config():
    return {'seed': 17, 'relation_objective': 'observed_fused_rms_effect',
            'metrics_contract': CONTINUOUS_METRIC_CONTRACT,
            'gates': {'diagnostic_min_delta_mse': .001,
                      'formal_min_delta_mse': .001,
                      'minimum_evaluable_cancers': 3}}


def test_diagnostic_completion_uses_mse_threshold():
    cfg = config()
    metrics = {'pathway': {'delta': .002}, 'state': {'delta': .0015}}
    assert completion_state(cfg, 'diagnostic', 1., metrics,
                            'observed_fused_rms_effect') == 'GO_TO_FORMAL_EVALUATION'
    metrics['state']['delta'] = .0005
    assert completion_state(cfg, 'diagnostic', 1., metrics,
                            'observed_fused_rms_effect') == 'NO_GO_WITHIN_DIAGNOSTIC_BUDGET'


def test_continuous_outer_test_masks_unobserved_effect(tmp_path, monkeypatch):
    from cla_engine import test_access

    prepared, training, output = (tmp_path / name for name in ('prepared', 'training', 'outer'))
    view = prepared / 'views' / 'BRCA__validation'
    sealed = tmp_path / 'prepared__sealed_test'
    prediction = training / 'predictions'
    for path in (view, sealed, prediction):
        path.mkdir(parents=True)
    spec = {'cancers': ['BRCA'], 'nodes': {'lncRNA': ['L1', 'L2', 'L3'],
            'pathway': ['P1']}, 'states': ['S1'], 'state_families': {'S1': 'state'},
            'fold': 0, 'relation_objective': 'observed_fused_rms_effect'}
    cfg = config()
    (prepared / 'MANIFEST.json').write_text(json.dumps({'metadata': {'profile': 'diagnostic'}}))
    (sealed / 'MANIFEST.json').write_text(json.dumps({'metadata': {'role': 'sealed'}}))
    (prepared / 'CANDIDATES.json').write_text(json.dumps({'BRCA': {'pathway': {}, 'state': {}}}))
    (training / 'TRAINING_RECEIPT.json').write_text(json.dumps({'alpha': 1.0}))
    for task in ('pathway', 'state'):
        common = {'lnc': np.array([0, 1, 2]), 'target': np.zeros(3, dtype=int)}
        np.savez(view / f'{task}.npz', **common)
        np.savez(sealed / f'BRCA__validation__{task}.npz', **common,
                 label=np.array([.2, .8, 999.]),
                 available=np.array([True, True, False]))
        np.savez(prediction / f'BRCA__{task}.npz', **common,
                 score=np.array([.25, .75, .3]), baseline=np.array([.4, .4, .4]))
    monkeypatch.setattr(outer, 'validate_prepared', lambda _: ({}, spec, cfg, {}))
    monkeypatch.setattr(outer, 'verify_completion', lambda *_: ({'alpha': 1.,
        'state': 'NO_GO_WITHIN_DIAGNOSTIC_BUDGET', 'signature': 'model'}, {}))
    monkeypatch.setattr(outer, 'verify_sealed', lambda *_: None)
    monkeypatch.setattr(test_access, 'freeze_model', lambda *_: {'freeze_sha256': 'frozen'})
    monkeypatch.setattr(test_access, 'record_outer_open', lambda *_, **__: None)
    monkeypatch.setattr(outer, 'commit_manifest', lambda *_, **__: None)
    monkeypatch.setattr(outer, 'sign_file', lambda *_, **__: None)
    monkeypatch.setattr(outer, 'status', lambda *_, **__: None)
    result = outer.evaluate_fold(prepared, training, output)
    assert result['relation_objective'] == 'observed_fused_rms_effect'
    assert all(m['available'] and m['delta_mse'] > 0 for m in result['metrics'])
    assert all('auroc' not in m and 'direction_accuracy' not in m for m in result['metrics'])
    rows = pd.read_csv(output / 'OOF_ASSOCIATIONS.tsv.gz', sep='\t')
    assert rows.loc[~rows.availability, 'heldout_rms_effect'].isna().all()
    assert set(rows.task) == {'pathway', 'state'}


def test_formal_continuous_gate_uses_mse_and_checks_objective(tmp_path, monkeypatch):
    cfg = config()
    common_lineage = {field: 'same' for field in FIELDS}
    roots = []
    for fold in range(5):
        path = tmp_path / f'eval_{fold}'
        path.mkdir()
        metrics = [{'fold': fold, 'cancer_id': cancer, 'task': task,
                    'available': True, 'delta_mse': .01}
                   for cancer in ('BRCA', 'LUAD', 'COAD')
                   for task in ('pathway', 'state')]
        receipt = {'fold': fold, 'profile': 'formal', 'synthetic_fixture': False,
                   'relation_objective': 'observed_fused_rms_effect',
                   'state': 'OUTER_TEST_EVALUATED_NOT_DEPLOYED',
                   'training_state': 'TRAINED_FORMAL_CANDIDATE',
                   'model_signature': f'model_{fold}', 'sealed_manifest_sha256': 'sealed',
                   'model_freeze_sha256': 'frozen', 'metrics_contract': CONTINUOUS_METRIC_CONTRACT,
                   'config_sha256': digest(cfg), 'lineage': {**common_lineage, 'fold': fold},
                   'candidate_catalog': {'same': True},
                   'expected_cancers': ['BRCA', 'LUAD', 'COAD'], 'metrics': metrics}
        (path / 'EVALUATION_RECEIPT.json').write_text(json.dumps(receipt))
        roots.append(path)
    monkeypatch.setattr(outer, 'verify_manifest', lambda *_: None)
    monkeypatch.setattr(outer, 'verify_signed_file', lambda *_, **__: None)
    monkeypatch.setattr(outer, 'status', lambda *_, **__: None)
    result = outer.gate(roots, cfg, tmp_path / 'gate')
    assert result['state'] == 'PREDICTION_GATE_PASS_NOT_DEPLOYED'
    assert all(item['mean_delta_mse'] == .01 for item in result['tasks'].values())
    receipt_path = roots[0] / 'EVALUATION_RECEIPT.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['relation_objective'] = 'binary'
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(Stop) as error:
        outer.gate(roots, cfg, tmp_path / 'bad_gate')
    assert error.value.code == 'GATE_RELATION_OBJECTIVE_MISMATCH'
