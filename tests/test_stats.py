import numpy as np
import pytest
from scipy import stats

from cc_hhgt.stats import bh_fdr, correlation_p_values, design_rank, expected_calibration_error, residualize, signed_support

def test_bh_monotonic_and_bounded():
    q=bh_fdr([0.01,0.02,0.5,1.0])
    assert np.all((q>=0)&(q<=1))
    assert q[0] <= q[1] <= q[2] <= q[3]

def test_signed_support_prefers_strong_significant_effect():
    score=signed_support([0.5,0.1],[1e-5,0.5])
    assert score[0] > score[1]

def test_ece_zero_for_perfect_predictions():
    assert expected_calibration_error(np.array([0,1]),np.array([0,1]),bins=2) == 0


def test_partial_correlation_p_value_uses_effective_residual_df():
    r = np.array([0.4])
    observed = correlation_p_values(r, 100, residual_design_rank=6)[0]
    expected_df = 100 - 6 - 1
    expected_t = r[0] * np.sqrt(expected_df / (1 - r[0] ** 2))
    assert np.isclose(observed, 2 * stats.t.sf(abs(expected_t), df=expected_df))
    assert observed > correlation_p_values(r, 100)[0]


def test_rank_deficient_design_only_removes_its_true_column_space():
    x = np.column_stack([np.ones(8), np.arange(8), 2 * np.arange(8)])
    y = np.column_stack([np.arange(8), np.array([0, 1, 0, 1, 0, 1, 0, 1])])
    residual = residualize(y, x)
    assert design_rank(x) == 2
    assert np.allclose(residual[:, 0], 0, atol=1e-5)
    assert np.linalg.norm(residual[:, 1]) > 0


def test_partial_correlation_p_value_matches_frozen_base_r_qr_reference():
    # Independent base-R qr.resid/cor/pt authority:
    # ./data/CancerLncAtlas/runtime/audits/
    # v32_correlation_df_reference_20260829_r1/R_REFERENCE.json
    # SHA-256 daf0a8926e4d98ef040a0a5eb88d0656bd05c5c1f2645cfefdaf3c6a7d598ec8.
    effect = np.array([0.61356918769728719])
    observed = correlation_p_values(effect, 12, residual_design_rank=2)[0]
    assert observed == pytest.approx(0.044668865202101976, rel=1e-12)
