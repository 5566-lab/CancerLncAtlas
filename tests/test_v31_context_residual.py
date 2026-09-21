from __future__ import annotations

import numpy as np
import pandas as pd

from cc_hhgt.v31_context_residual import (
    ContextTransform,
    attach_context_features,
    exact_module_fallback,
    fit_offset_logistic,
    predict_offset_logistic,
    select_context_residual,
)


def _context() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cancer_id": ["A", "A", "B"],
            "lncrna_id": ["L1", "L2", "L1"],
            "signal": [1.0, np.nan, 3.0],
            "signal__available": [True, False, True],
        }
    )


def test_context_attach_rank_mask_and_offset_fallback() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2", "c3", "c4"],
            "cancer_id": ["A", "A", "B", "B"],
            "lncrna_id": ["L1", "L2", "L1", "MISSING"],
        }
    )
    attached, features = attach_context_features(candidates, _context())
    assert attached.candidate_id.tolist() == candidates.candidate_id.tolist()
    assert attached.context_available.tolist() == [True, False, True, False]
    transform = ContextTransform.fit(attached, features)
    matrix, available = transform.transform(attached)
    base = np.zeros(4)
    fit = fit_offset_logistic(
        matrix,
        [1, 0, 1, 0],
        base,
        shrinkage_lambda=0.1,
    )
    delta, final, probability = predict_offset_logistic(
        fit, matrix, base, available
    )
    assert fit.initialization_max_abs_error <= 1e-7
    assert delta[1] == 0 and delta[3] == 0
    assert final[1] == base[1] and probability[1] == 0.5


def _validation() -> pd.DataFrame:
    rows = []
    labels = [0, 0, 1, 1]
    base = [0.4, 0.6, 0.4, 0.6]
    improved = [0.05, 0.05, 0.95, 0.95]
    worse = [0.9, 0.8, 0.2, 0.1]
    for shrinkage, scores in ((0.1, improved), (1.0, worse)):
        for cancer in ("BRCA", "COAD", "KIRP"):
            for seed in (1, 2):
                for index, label in enumerate(labels):
                    rows.append(
                        {
                            "candidate_id": f"{cancer}-{index}",
                            "module": "RNA",
                            "target_subtype": "RNAss",
                            "loco_cancer": cancer,
                            "seed": seed,
                            "metric_scope": "validation",
                            "residual_shrinkage_lambda": shrinkage,
                            "proxy_label": label,
                            "current_base_probability": base[index],
                            "proxy_positive_probability": scores[index],
                        }
                    )
    return pd.DataFrame(rows)


def test_context_admission_is_global_validation_only() -> None:
    metrics, cancers, selection = select_context_residual(_validation())
    assert len(metrics) == 2
    assert len(cancers) == 6
    assert selection.iloc[0].selected_lambda == 0.1
    assert selection.iloc[0].positive_cancers == 3
    assert bool(selection.iloc[0].admitted)
    assert not bool(selection.iloc[0].test_metric_used_for_selection)


def test_rejected_context_is_exact_current_base() -> None:
    frame = pd.DataFrame(
        {
            "current_base_logit": [-1.0, 1.0],
            "current_base_probability": [0.26894142137, 0.73105857863],
            "context_residual_logit": [4.0, -3.0],
            "final_logit": [3.0, -2.0],
            "proxy_positive_probability": [0.9, 0.1],
        }
    )
    result = exact_module_fallback(frame)
    assert np.array_equal(result.final_logit, result.current_base_logit)
    assert np.array_equal(
        result.proxy_positive_probability, result.current_base_probability
    )
