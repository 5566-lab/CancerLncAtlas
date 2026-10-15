from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Lasso

from cc_hhgt.v32.single_cell_batched_lasso import (
    BatchedLassoPolicy,
    donor_balanced_weights,
    fit_batched_lasso,
)


def _fixture(seed: int = 14):
    rng = np.random.default_rng(seed)
    donors = np.concatenate(
        [np.repeat(f"d{index}", count) for index, count in enumerate((7, 9, 11, 13, 15, 17))]
    )
    coding = rng.normal(size=(len(donors), 8))
    truth = np.zeros((8, 3))
    truth[[0, 2], 0] = [0.9, -0.45]
    truth[[1, 5], 1] = [-0.8, 0.35]
    truth[[3, 6], 2] = [0.7, 0.4]
    response = coding @ truth + rng.normal(scale=0.12, size=(len(donors), 3))
    return coding, response, donors


def _policy(**changes):
    values = {
        "alphas": (0.05,),
        "max_genes": 8,
        "min_donors": 4,
        "min_detection": 0.0,
        "inner_folds": 3,
        "bootstraps": 4,
        "min_bootstrap_successes": 2,
        "max_iter": 5000,
        "tolerance": 1e-8,
        "kkt_check_interval": 5,
        "response_batch_size": 2,
        "seed": 812,
    }
    values.update(changes)
    return BatchedLassoPolicy(**values)


def _dense_coefficients(result):
    return result.coefficient_coo().toarray()


def test_batched_independent_objectives_match_scalar_sklearn_lasso():
    coding, response, donors = _fixture()
    result = fit_batched_lasso(
        coding,
        response,
        donors,
        source_donors=np.unique(donors),
        excluded_donors=(),
        policy=_policy(),
    )
    assert result.available.all()

    weights = donor_balanced_weights(donors)
    donor_weight = [weights[donors == donor].sum() for donor in np.unique(donors)]
    np.testing.assert_allclose(donor_weight, np.repeat(1 / 6, 6))
    x_mean = weights @ coding
    x_scale = np.sqrt(weights @ ((coding - x_mean) ** 2))
    x_standardized = (coding - x_mean) / x_scale
    expected = np.zeros((response.shape[1], coding.shape[1]))
    for index in range(response.shape[1]):
        target = response[:, index]
        target_mean = weights @ target
        target_scale = np.sqrt(weights @ ((target - target_mean) ** 2))
        estimator = Lasso(
            alpha=0.05,
            max_iter=100000,
            tol=1e-10,
            selection="cyclic",
        ).fit(
            x_standardized,
            (target - target_mean) / target_scale,
            sample_weight=weights * len(weights),
        )
        expected[index] = estimator.coef_
    np.testing.assert_allclose(_dense_coefficients(result), expected, atol=2e-5, rtol=2e-5)


def test_query_rows_never_affect_preprocessing_selection_or_fit():
    coding, response, source_donor = _fixture(32)
    rng = np.random.default_rng(91)
    query_donor = np.repeat(["q1", "q2"], 5)
    coding = np.vstack((coding, rng.normal(size=(10, coding.shape[1]))))
    response = np.vstack((response, rng.normal(size=(10, response.shape[1]))))
    donors = np.concatenate((source_donor, query_donor))
    arguments = {
        "source_donors": np.unique(source_donor),
        "excluded_donors": np.unique(query_donor),
        "policy": _policy(),
    }
    first = fit_batched_lasso(coding, response, donors, **arguments)
    coding[-10:] = np.nan
    response[-10:] = np.nan
    second = fit_batched_lasso(coding, response, donors, **arguments)
    np.testing.assert_array_equal(first.available, second.available)
    np.testing.assert_array_equal(first.response_index, second.response_index)
    np.testing.assert_array_equal(first.gene_index, second.gene_index)
    np.testing.assert_allclose(first.coefficient, second.coefficient, atol=0, rtol=0)
    np.testing.assert_allclose(first.stability, second.stability, atol=0, rtol=0)
    assert first.audit["source_coding_sha256"] == second.audit["source_coding_sha256"]
    assert first.audit["held_out_donors_used"] == 0

    with pytest.raises(ValueError, match="DONOR_LEAK"):
        fit_batched_lasso(
            coding,
            response,
            donors,
            source_donors=[*np.unique(source_donor), "q1"],
            excluded_donors=["q1", "q2"],
            policy=_policy(),
        )


def test_deterministic_signed_stable_edges_and_unavailable_mask():
    rng = np.random.default_rng(19)
    donors = np.repeat([f"d{index}" for index in range(8)], 15)
    coding = rng.normal(size=(len(donors), 7))
    response = np.column_stack(
        (
            1.4 * coding[:, 0] + rng.normal(scale=0.03, size=len(donors)),
            -1.2 * coding[:, 1] + rng.normal(scale=0.03, size=len(donors)),
            np.ones(len(donors)),
        )
    )
    policy = _policy(max_genes=7, seed=44)
    first = fit_batched_lasso(
        coding,
        response,
        donors,
        source_donors=np.unique(donors),
        excluded_donors=(),
        response_ids=["positive", "negative", "constant"],
        policy=policy,
    )
    second = fit_batched_lasso(
        coding,
        response,
        donors,
        source_donors=np.unique(donors),
        excluded_donors=(),
        response_ids=["positive", "negative", "constant"],
        policy=policy,
    )
    np.testing.assert_array_equal(first.available, [True, True, False])
    assert first.unavailable_reason[2] in {
        "INNER_CV_UNAVAILABLE", "NON_ESTIMABLE_RESPONSE"
    }
    np.testing.assert_array_equal(first.response_index, second.response_index)
    np.testing.assert_array_equal(first.gene_index, second.gene_index)
    np.testing.assert_allclose(first.coefficient, second.coefficient, atol=0, rtol=0)
    np.testing.assert_allclose(first.stability, second.stability, atol=0, rtol=0)
    dense = _dense_coefficients(first)
    assert dense[0, 0] > 0
    assert dense[1, 1] < 0
    edge = first.edge_frame()
    assert edge.loc[(edge.response_index == 0) & (edge.gene_index == 0), "stability"].iloc[0] == 1
    assert edge.loc[(edge.response_index == 1) & (edge.gene_index == 1), "stability"].iloc[0] == 1


def test_top_k_is_disabled_by_default_and_only_applied_when_configured():
    coding, response, donors = _fixture(55)
    arguments = {
        "coding": coding,
        "responses": response[:, :1],
        "donors": donors,
        "source_donors": np.unique(donors),
        "excluded_donors": (),
    }
    complete = fit_batched_lasso(**arguments, policy=_policy())
    limited = fit_batched_lasso(**arguments, policy=_policy(top_k_per_response=1))
    assert complete.audit["top_k_per_response"] is None
    assert complete.coefficient.size > 1
    assert limited.coefficient.size == 1


def test_dual_lipschitz_and_descending_warm_path_preserve_scalar_solutions():
    rng = np.random.default_rng(202)
    donors = np.repeat([f"d{index}" for index in range(6)], 10)
    coding = rng.normal(size=(len(donors), 90))
    truth = np.zeros((90, 4))
    truth[[0, 7, 14], 0] = [0.8, -0.5, 0.3]
    truth[[2, 9], 1] = [-0.9, 0.4]
    truth[[4, 11, 22], 2] = [0.6, 0.4, -0.2]
    truth[[6, 19], 3] = [-0.7, -0.35]
    response = coding @ truth + rng.normal(scale=0.2, size=(len(donors), 4))
    policy = _policy(
        alphas=(0.03, 0.2, 0.01, 0.08),
        max_genes=90,
        max_iter=7000,
        tolerance=2e-8,
        response_batch_size=4,
    )
    result = fit_batched_lasso(
        coding,
        response,
        donors,
        source_donors=np.unique(donors),
        excluded_donors=(),
        policy=policy,
    )
    assert result.available.all()
    assert result.audit["final_lipschitz_method"] == "EXACT_DUAL_GRAM_EIGENVALUE"
    assert result.audit["alpha_warm_start_order"] == [0.2, 0.08, 0.03, 0.01]

    weights = donor_balanced_weights(donors)
    x_mean = weights @ coding
    x_scale = np.sqrt(weights @ ((coding - x_mean) ** 2))
    x_standardized = (coding - x_mean) / x_scale
    expected = np.zeros_like(_dense_coefficients(result))
    for index, alpha in enumerate(result.selected_alpha):
        target = response[:, index]
        target_mean = weights @ target
        target_scale = np.sqrt(weights @ ((target - target_mean) ** 2))
        expected[index] = Lasso(
            alpha=float(alpha), max_iter=100000, tol=1e-10, selection="cyclic"
        ).fit(
            x_standardized,
            (target - target_mean) / target_scale,
            sample_weight=weights * len(weights),
        ).coef_
    np.testing.assert_allclose(_dense_coefficients(result), expected, atol=5e-5, rtol=5e-5)

