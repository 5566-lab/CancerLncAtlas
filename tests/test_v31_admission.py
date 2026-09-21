from __future__ import annotations

import pandas as pd
import pytest

from cc_hhgt.v31_admission import select_residual_lambda


def _validation() -> pd.DataFrame:
    rows = []
    labels = [0, 0, 1, 1]
    base = [0.1, 0.6, 0.4, 0.9]
    probabilities = {
        0.01: [0.1, 0.2, 0.8, 0.9],
        0.10: [0.2, 0.5, 0.5, 0.8],
    }
    for shrinkage, scores in probabilities.items():
        for index, label in enumerate(labels):
            rows.append(
                {
                    "candidate_id": f"c{index}",
                    "cancer_id": "BRCA",
                    "seed": 1,
                    "target_subtype": "Pathway",
                    "metric_scope": "validation",
                    "residual_shrinkage_lambda": shrinkage,
                    "proxy_positive_probability": scores[index],
                    "base_probability": base[index],
                    "proxy_label": label,
                }
            )
    return pd.DataFrame(rows)


def test_lambda_selection_is_validation_only_and_deterministic() -> None:
    metrics, selection = select_residual_lambda(
        _validation(),
        group_columns=["cancer_id", "seed", "target_subtype"],
    )
    assert len(metrics) == 2
    assert selection.iloc[0].selected_lambda == 0.01
    assert bool(selection.iloc[0].admitted)
    assert not bool(selection.iloc[0].test_metric_used_for_selection)


def test_lambda_selection_rejects_test_rows() -> None:
    frame = _validation()
    frame["metric_scope"] = "LOCO_test"
    with pytest.raises(RuntimeError, match="validation"):
        select_residual_lambda(
            frame, group_columns=["cancer_id", "seed", "target_subtype"]
        )


def test_lambda_selection_rejects_candidate_drift() -> None:
    frame = _validation()
    frame.loc[frame.residual_shrinkage_lambda.eq(0.10).idxmax(), "candidate_id"] = "changed"
    with pytest.raises(RuntimeError, match="drift"):
        select_residual_lambda(
            frame, group_columns=["cancer_id", "seed", "target_subtype"]
        )
