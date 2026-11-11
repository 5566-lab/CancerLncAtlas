"""Meaningful isolation checks for the policy-scoped relation comparator."""
from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cla_engine import m0_relation as m0
from cla_engine.exact_baseline import ENCODING
from cla_engine.target_policy import REQUIRED_FIELDS, policy_from_rows


def config():
    return {'seed': 73, 'baseline': {'C': [.1, 1.], 'max_iter': 2000,
            'identity_encoding': ENCODING, 'batch_rows': 19, 'max_identity_features': 10000,
            'max_working_gib': .5, 'max_vocabulary_gib': .1, 'max_available_memory_fraction': .5}}


def policy_rows():
    def row(target, role, group, weight, active, biological_type='pathway'):
        return dict(zip(REQUIRED_FIELDS, [target, 'Pathway', biological_type, role, group,
                        weight, active, 'backbone' if active else 'excluded_zero_overlap', 'hsa05200']))
    return [row('P1', 'primary_pathway', 'pathway', 1., True),
            row('P2', 'primary_pathway', 'pathway', 1., True),
            row('TF1', 'regulator_auxiliary', 'state', .5, True, 'tf'),
            row('DROP', 'excluded', 'none', 0., False)]


def block(seed, n=160):
    rng = np.random.default_rng(seed)
    numeric = rng.normal(size=(n, 5)).astype(np.float32)
    target = np.tile(np.arange(4), n // 4)
    lnc = np.arange(n) // 4
    identities = [('BRCA', 'L'+str(i), ['P1','P2','TF1','DROP'][int(j)]) for i,j in zip(lnc,target)]
    return {'numeric': numeric, 'label': (numeric[:, 0] > 0).astype(np.float32),
            'available': np.ones(n, bool), 'weight': np.ones(n),
            'direction_label': (numeric[:, 1] > 0).astype(np.float32),
            'direction_available': np.arange(n) % 5 != 0,
            'lnc': lnc, 'target': target, 'identities': identities}


def assert_same_models(a, b):
    for head in ('membership', 'direction'):
        x, y = a['heads'][head], b['heads'][head]
        assert x['selected_C'] == y['selected_C']
        assert x['vocabulary'] == y['vocabulary']
        for key in ('coef', 'intercept', 'numeric_mean', 'numeric_scale', 'logits'):
            np.testing.assert_array_equal(x[key], y[key])


def test_excluded_and_auxiliary_rows_never_influence_fit_or_selection():
    train, valid = block(1), block(2)
    a = m0.fit_scoped_cancer([train], [valid], ['P1','P2'], config())
    changed_train, changed_valid = block(1), block(2)
    for item in (changed_train, changed_valid):
        excluded = item['target'] >= 2
        item['numeric'][excluded] = 1e8
        item['label'][excluded] = 1 - item['label'][excluded]
        item['direction_label'][excluded] = 1 - item['direction_label'][excluded]
        item['weight'][excluded] = 1e9
    b = m0.fit_scoped_cancer([changed_train], [changed_valid], ['P1','P2'], config())
    assert_same_models(a, b)
    assert {r['canonical_id'] for r in a['exact_target_metrics']} == {'P1','P2'}
    for model in a['heads'].values():
        assert all('DROP' not in key and 'TF1' not in key for key in model['vocabulary'])


def test_scope_uses_analysis_role_not_pathway_display_role():
    policies = policy_from_rows(policy_rows())
    _, admitted = m0._target_axis(['P1','P2','TF1','DROP'], policies, {})
    assert admitted == ('P1','P2')
    assert policies['TF1'].roles == 'Pathway'
    assert policies['TF1'].analysis_role == 'regulator_auxiliary'


def test_no_multisource_label_fusion_by_renaming():
    policies = policy_from_rows(policy_rows())
    with pytest.raises(ValueError, match='MANY_TO_ONE_LABEL_REQUIRES_REMATERIALIZATION'):
        m0._target_axis(['a','b'], policies, {'a':'P1','b':'P1'})
    with pytest.raises(ValueError, match='POLICY_ID_MISSING'):
        m0._target_axis(['unknown'], policies, {})


def test_masked_label_changes_cannot_change_fit():
    train, valid = block(1), block(2)
    train['available'][::7] = False
    valid['available'][::7] = False
    a = m0.fit_scoped_cancer([train], [valid], ['P1','P2'], config())
    for item in (train, valid):
        item['label'][~item['available']] = 999
        item['direction_label'][~item['available']] = 999
    b = m0.fit_scoped_cancer([train], [valid], ['P1','P2'], config())
    assert_same_models(a, b)


def test_direction_does_not_treat_unavailable_or_discordant_as_negative():
    train, valid = block(1), block(2)
    for item in (train, valid):
        item['direction_label'][~item['direction_available']] = 3
    fit = m0.fit_scoped_cancer([train], [valid], ['P1','P2'], config())
    assert fit['heads']['direction']['status'] == 'FITTED'
    assert fit['heads']['direction']['train_rows'] < fit['heads']['membership']['train_rows']


def test_cancer_models_cannot_be_pooled_accidentally():
    train, valid = block(1), block(2)
    valid['identities'] = [('LUAD',l,t) for c,l,t in valid['identities']]
    with pytest.raises(ValueError, match='ONE_CANCER_AT_A_TIME'):
        m0.fit_scoped_cancer([train], [valid], ['P1','P2'], config())


def fixture_capsule(tmp_path, monkeypatch):
    prepared = tmp_path/'prepared'
    prepared.mkdir()
    spec = {'cancers':['BRCA'], 'nodes':{'pathway':['P1','P2','TF1','DROP'],
             'lncRNA':['L'+str(i) for i in range(40)]}, 'fold':0,
            'validation_fold':1, 'train_folds':[2,3,4]}
    views = [{'id':'BRCA__train2','cancer_index':0,'split':'train'},
             {'id':'BRCA__validation','cancer_index':0,'split':'validation'}]
    for i, view in enumerate(views):
        folder = prepared/'views'/view['id']
        folder.mkdir(parents=True)
        data = block(i + 1)
        data.pop('identities')
        data.update(base=np.full(160, 9999.), heldout_rho=np.full(160, .9999),
                    heldout_p=np.full(160, 1e-30), heldout_q=np.full(160, 1e-30))
        np.savez(folder/'pathway.npz', **data)
    for file, obj in [('VIEWS.json',views), ('SPEC.json',spec), ('MANIFEST.json',{'fixture':True}),
                      ('TASK_CONTRACT.json',{'pathway_label':'EXISTING_PREPARED_BINARY_ENDPOINT'})]:
        (prepared/file).write_text(json.dumps(obj))
    policy = tmp_path/'TARGET_POLICY.tsv'
    with policy.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=REQUIRED_FIELDS,delimiter='\t')
        writer.writeheader();writer.writerows(policy_rows())
    # The production path always validates full capsule integrity; this fixture
    # substitutes only that expensive independent validator, not the fit/reader.
    monkeypatch.setattr(m0, 'validate_prepared', lambda _: ({},spec,config(),{}))
    monkeypatch.setattr(m0, 'require_host_149', lambda:'149')
    return prepared, policy


def test_sealed_test_changes_cannot_change_fit_or_hyperparameters(tmp_path, monkeypatch):
    prepared, policy = fixture_capsule(tmp_path, monkeypatch)
    sealed = tmp_path/'prepared__sealed_test'
    sealed.mkdir()
    labels = sealed/'BRCA__validation__pathway.npz'
    np.savez(labels,label=np.zeros(160),rho=np.zeros(160),p=np.ones(160),q=np.ones(160))
    first = m0.run_m0_relation(prepared,policy,tmp_path/'run1',cancers=['BRCA'])
    np.savez(labels,label=np.ones(160),rho=np.ones(160),p=np.zeros(160),q=np.zeros(160))
    second = m0.run_m0_relation(prepared,policy,tmp_path/'run2',cancers=['BRCA'])
    assert not first['outer_test_read'] and not second['outer_test_read']
    assert first['fit_contract_sha256'] == second['fit_contract_sha256']
    for head in ('membership','direction'):
        assert first['results']['BRCA'][head]['selected_C'] == second['results']['BRCA'][head]['selected_C']
        with np.load(tmp_path/'run1'/'BRCA'/(head+'_model.npz')) as a, np.load(tmp_path/'run2'/'BRCA'/(head+'_model.npz')) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key],b[key])
    assert set(first['admitted_targets']) == {'P1','P2'}
    assert first['cancers'] == ('BRCA',)


def test_extra_heldout_columns_are_never_features(tmp_path, monkeypatch):
    prepared, policy = fixture_capsule(tmp_path, monkeypatch)
    spec = json.loads((prepared/'SPEC.json').read_text())
    blocks, _ = m0._load_blocks(prepared,spec,tuple(spec['nodes']['pathway']),'BRCA')
    assert set(blocks['train'][0]) == set(m0._FIELDS) | {'identities'}
    assert blocks['train'][0]['numeric'].shape[1] == 5


def test_raw_fused_catalog_is_not_implicitly_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(m0,'require_host_149',lambda:'149')
    catalog = tmp_path/'fused'
    catalog.mkdir()
    (catalog/'Y_DEFINITION.json').write_text('{"format":"CLA_FUNCTIONAL_RELATION_FUSED_V1"}')
    with pytest.raises(ValueError, match='UNLABELLED_IS_NOT_NEGATIVE'):
        m0.run_m0_relation(catalog,tmp_path/'unused_policy',tmp_path/'out')
