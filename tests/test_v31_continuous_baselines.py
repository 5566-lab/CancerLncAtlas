from __future__ import annotations

import numpy as np

from cc_hhgt.v31_continuous_baselines import fit_nested_continuous_baselines


def test_nested_continuous_baselines_tune_without_outer_test() -> None:
    rng = np.random.default_rng(20260823)
    x = rng.normal(size=(90, 12))
    y = 2.5 * x[:, 0] - 1.7 * x[:, 3] + rng.normal(scale=0.2, size=90)
    result = fit_nested_continuous_baselines(
        x[:50],
        y[:50],
        x[50:70],
        y[50:70],
        x[70:],
        y[70:],
        lasso_alpha_ratios=np.logspace(0, -3, 12),
        ridge_alphas=np.logspace(2, -3, 10),
    )
    assert result["audit"]["hyperparameter_selection_split"] == "validation_only"
    assert result["audit"]["outer_test_used_for_tuning"] is False
    assert result["lasso"]["test_r2"] > 0.9
    assert result["ridge"]["test_r2"] > 0.9
    assert result["lasso"]["n_nonzero"] < x.shape[1]
    assert result["lasso"]["prediction"].shape == (20,)
