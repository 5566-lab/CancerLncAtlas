"""The same prepared baseline API supports explicit numeric-only ablations."""
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from cla_engine import exact_baseline as baseline


def contract(mode=None):
    c = dict(batch_rows=3, max_identity_features=10000,
             max_working_gib=.5, max_vocabulary_gib=.1,
             max_available_memory_fraction=.5)
    if mode is not None:
        c['identity_mode'] = mode
    return c


def inputs():
    x = np.random.default_rng(4).normal(size=(8, 4))
    rows = [('BRCA', 'L' + str(i), 'P' + str(i % 2)) for i in range(8)]
    return rows, x, StandardScaler().fit(x)


def test_missing_identity_mode_preserves_exact_legacy_design():
    rows, x, scaler = inputs()
    old, old_report = baseline.fit_vocabulary(rows, contract())
    explicit, new_report = baseline.fit_vocabulary(rows, contract('exact'))
    assert old.vocabulary_ == explicit.vocabulary_
    assert old_report == new_report
    assert len(old.vocabulary_) > 0
    np.testing.assert_array_equal(
        baseline.training_design(rows, x, scaler, old, 3).toarray(),
        baseline.training_design(rows, x, scaler, explicit, 3).toarray())


def test_numeric_only_design_has_no_identity_features_or_identity_memory():
    rows, x, scaler = inputs()
    vectorizer, report = baseline.fit_vocabulary(rows, contract('numeric_only'))
    assert vectorizer.vocabulary_ == {} and vectorizer.feature_names_ == []
    assert report['features'] == 0
    design = baseline.training_design(rows, x, scaler, vectorizer, 3)
    np.testing.assert_array_equal(design.toarray(), scaler.transform(x))
    renamed = [('LUAD', 'OTHER' + str(i), 'UNSEEN') for i in range(8)]
    np.testing.assert_array_equal(baseline.design_block(renamed, x, scaler, vectorizer).toarray(), design.toarray())
    assert not baseline.unseen_audit(renamed, vectorizer)['identity_terms_used']


def test_numeric_only_prediction_uses_existing_batched_interface():
    rows, x, scaler = inputs()
    vectorizer, _ = baseline.fit_vocabulary(rows, contract('numeric_only'))
    class Linear:
        def decision_function(self, design):
            return np.asarray(design @ np.arange(1., 5.)).ravel() + .7
    pred = baseline.decision_batched(Linear(), rows, x, scaler, vectorizer, 3)
    np.testing.assert_allclose(pred, scaler.transform(x) @ np.arange(1., 5.) + .7)


def test_numeric_only_resource_budget_omits_identity_columns():
    exact = baseline.resource_plan(200, 4, 20, contract())
    numeric = baseline.resource_plan(200, 4, 20, contract('numeric_only'))
    assert numeric['csr_upper_bound_bytes'] < exact['csr_upper_bound_bytes']


def test_unknown_identity_mode_cannot_silently_select_another_model():
    with pytest.raises(Exception, match='unknown'):
        baseline.fit_vocabulary([], contract('unknown'))


def test_numeric_only_adapter_cannot_be_misreported_as_exact_identity():
    rows, _, _ = inputs()
    vectorizer, _ = baseline.fit_vocabulary(rows, contract('numeric_only'))
    with pytest.raises(Exception, match='numeric_only cannot supply'):
        vectorizer.transform(baseline.identity_records(rows))
