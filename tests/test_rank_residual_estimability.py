import numpy as np
import pytest
from scipy.stats import rankdata, t
from cla_engine.statistics import pairs
from cla_engine.graph import coexpression


def fixture():
    rng = np.random.default_rng(20261109)
    n = 120
    return (rng.normal(size=(n, 4)), rng.normal(size=(n, 3)),
            rng.normal(size=(n, 4)), np.ones((n, 4), dtype=bool),
            np.column_stack([np.ones(n), rng.normal(size=(n, 3))]))


def all_pairs(data, **kwargs):
    X, Y, C, call, Z = data
    li, ti = np.indices((X.shape[1], Y.shape[1]))
    return pairs(X, Y, C, call, Z, li.ravel(), ti.ravel(), np.arange(len(X)), **kwargs)


@pytest.mark.parametrize("constant", [0.0, 7.0])
def test_constant_x_or_y_has_no_association(constant):
    X, Y, C, call, Z = fixture()
    X[:, 0] = constant
    Y[:, 0] = constant
    result = all_pairs((X, Y, C, call, Z))
    observed = result['available'].reshape(4, 3)
    assert not observed[0, :].any()
    assert not observed[:, 0].any()
    assert observed[1:, 1:].all()
    assert np.isnan(result['q'][~result['available']]).all()


def test_expression_unit_does_not_filter_rank_variation():
    data = fixture()
    original = all_pairs(data)
    tiny = all_pairs((data[0] * 1e-100, *data[1:]))
    assert original['available'].all()
    np.testing.assert_allclose(original['r'], tiny['r'])
    np.testing.assert_array_equal(original['available'], tiny['available'])


@pytest.mark.parametrize('null_side', ['x', 'y', 'cnv_x'])
def test_exact_adjustment_span_has_no_association(null_side):
    X, Y, C, call, Z = fixture()
    if null_side == 'x':
        Z = np.column_stack([Z, rankdata(X[:, 1])])
    elif null_side == 'y':
        Z = np.column_stack([Z, rankdata(Y[:, 1])])
    else:
        C[:, 1] = rankdata(X[:, 1])
    result = all_pairs((X, Y, C, call, Z))['available'].reshape(4, 3)
    if null_side == 'y':
        assert not result[:, 1].any()
        assert result[:, [0, 2]].all()
    else:
        assert not result[1, :].any()
        assert result[[0, 2, 3], :].all()


def test_uint8_callability_and_pairwise_missingness_match_boolean():
    X, Y, C, call, Z = fixture()
    X[:3, 0] = np.nan
    Y[3:7, 1] = np.nan
    call[7:12, 0] = False
    expected = all_pairs((X, Y, C, call, Z))
    actual = all_pairs((X, Y, C, call.astype(np.uint8), Z))
    for key in ['r', 'p', 'q', 'n', 'rank', 'available']:
        np.testing.assert_allclose(actual[key], expected[key], equal_nan=True)
    assert actual['n'].reshape(4, 3)[0, 1] == 108


@pytest.mark.parametrize('mode', ['lncrna_only', 'both'])
def test_valid_pair_matches_independent_least_squares(mode):
    X, Y, C, call, Z = fixture()
    result = pairs(X, Y, C, call, Z, [0], [1], np.arange(len(X)), local_mode=mode)
    x, y, c = rankdata(X[:, 0]), rankdata(Y[:, 1]), C[:, 0]
    residual = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    xr, yr, cr = residual(x), residual(y), residual(c)
    xr -= cr * np.dot(cr, xr) / np.dot(cr, cr)
    if mode == 'both':
        yr -= cr * np.dot(cr, yr) / np.dot(cr, cr)
    expected = np.dot(xr, yr) / (np.linalg.norm(xr) * np.linalg.norm(yr))
    df = len(X) - np.linalg.matrix_rank(Z) - 2
    expected_p = 2 * t.sf(abs(expected) * np.sqrt(df / (1 - expected**2)), df)
    np.testing.assert_allclose(result['r'], [expected], atol=1e-13)
    np.testing.assert_allclose(result['p'], [expected_p], atol=1e-13)


def test_coexpression_excludes_null_vectors_but_retains_signal():
    X, Y, C, call, Z = fixture()
    X[:, 0] = 0
    Y[:, 0] = 5
    Y[:, 1] = X[:, 1]
    edges, audit = coexpression(X, Y, C, call, Z, np.arange(len(X)),
                               topk=3, return_audit=True, lnc_batch=2, gene_batch=2)
    all_edges = np.concatenate([value[0] for value in edges.values()], axis=1)
    assert not (all_edges[0] == 0).any()
    assert not (all_edges[1] == 0).any()
    assert ((all_edges[0] == 1) & (all_edges[1] == 1)).any()
    assert audit['estimable_pairs'] == 6
