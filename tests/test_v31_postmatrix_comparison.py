from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v31_postmatrix import align_exact_frames, binary_metrics


COMPARE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "82_compare_v31_selected_deep_to_sparse_baselines.py"
)
SPEC = importlib.util.spec_from_file_location("v31_postmatrix_compare", COMPARE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def _frame(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["C0", "C1", "C2", "C3"],
            "cancer_id": "TEST",
            "lncrna_id": ["L0", "L1", "L2", "L3"],
            "pathway_id": ["P0", "P1", "P2", "P3"],
            "pathway_family_id": ["F0", "F0", "F1", "F1"],
            "proxy_label": [0, 0, 1, 1],
            "score": scores,
        }
    )


def test_exact_alignment_reorders_without_changing_identity() -> None:
    left = _frame([0.1, 0.2, 0.8, 0.9])
    right = left.iloc[::-1].reset_index(drop=True)
    identity, ordered = align_exact_frames([left, right])
    assert identity.candidate_id.tolist() == ["C0", "C1", "C2", "C3"]
    assert ordered[0].candidate_id.tolist() == ordered[1].candidate_id.tolist()


def test_exact_alignment_rejects_label_drift() -> None:
    left = _frame([0.1, 0.2, 0.8, 0.9])
    right = left.copy()
    right.loc[0, "proxy_label"] = 1
    with pytest.raises(RuntimeError, match="drift"):
        align_exact_frames([left, right])


def test_binary_metrics_reports_prevalence_adjusted_auprc() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["positive_prevalence"] == 0.5
    assert metrics["auprc_lift"] == 0.5


def test_family_summary_uses_preregistered_stability_threshold() -> None:
    rows = []
    for index in range(6):
        fold = f"LOCO_C{index}"
        for model, lift in (
            ("selected_deep", 0.30),
            ("lasso", 0.10 if index < 5 else 0.40),
            ("ridge", 0.05),
        ):
            rows.append(
                {
                    "fold_id": fold,
                    "pathway_family_id": "F",
                    "model": model,
                    "both_classes": True,
                    "auroc": lift + 0.5,
                    "auprc": lift + 0.2,
                    "auprc_lift": lift,
                }
            )
    deltas, summary = COMPARE._delta_summary(
        pd.DataFrame(rows), ["fold_id", "pathway_family_id"]
    )
    assert len(deltas) == 12
    lasso = summary.loc[summary.baseline.eq("lasso")].iloc[0]
    ridge = summary.loc[summary.baseline.eq("ridge")].iloc[0]
    assert int(lasso.eligible_folds) == 6
    assert float(lasso.fraction_positive_delta_auprc_lift) == pytest.approx(5 / 6)
    assert bool(lasso.stable_deep_benefit)
    assert bool(ridge.stable_deep_benefit)


def test_overall_delta_summary_groups_only_by_baseline() -> None:
    rows = []
    for index in range(3):
        for model, value in (("selected_deep", 0.4), ("lasso", 0.2), ("ridge", 0.1)):
            rows.append(
                {
                    "fold_id": f"LOCO_C{index}",
                    "model": model,
                    "both_classes": True,
                    "auroc": value + 0.4,
                    "auprc": value + 0.1,
                    "auprc_lift": value,
                }
            )
    deltas, summary = COMPARE._delta_summary(pd.DataFrame(rows), ["fold_id"])
    assert len(deltas) == 6
    assert set(summary.baseline) == {"lasso", "ridge"}
    assert set(summary.eligible_folds.astype(int)) == {3}


def test_single_class_family_is_retained_but_ineligible() -> None:
    metrics = binary_metrics([1, 1], [0.8, 0.9], allow_single_class=True)
    assert metrics["n"] == 2
    assert metrics["n_positive"] == 2
    assert metrics["both_classes"] is False
    assert np.isnan(metrics["auroc"])
