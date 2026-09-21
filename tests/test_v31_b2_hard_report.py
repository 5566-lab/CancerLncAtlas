from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v31_b2_hard_report import (
    PILOT_CANCERS,
    PILOT_SEEDS,
    TARGET_SUBTYPES,
    build_b2_hard_report,
)


def _fixtures(delta: float = 0.25):
    rows = []
    for cancer in PILOT_CANCERS:
        for seed in PILOT_SEEDS:
            for subtype in TARGET_SUBTYPES:
                for index in range(20):
                    label = int(index >= 10)
                    base = 0.5
                    final = np.clip(base + delta * (1 if label else -1), 1e-6, 1 - 1e-6)
                    rows.append(
                        {
                            "candidate_id": f"{cancer}|{subtype}|{index}",
                            "loco_cancer": cancer,
                            "seed": seed,
                            "target_subtype": subtype,
                            "split": "test",
                            "prediction_scale": "raw_probability",
                            "proxy_label": label,
                            "base_probability": base,
                            "proxy_positive_probability": final,
                            "module_admitted": True,
                            "test_metric_used_for_selection": False,
                        }
                    )
    predictions = pd.DataFrame(rows)
    calibrated = predictions.copy()
    calibrated["prediction_scale"] = "calibrated_probability"
    predictions = pd.concat([predictions, calibrated], ignore_index=True)
    admission = pd.DataFrame(
        {
            "target_subtype": TARGET_SUBTYPES,
            "admitted": True,
            "validation_delta_auprc": 0.02,
            "validation_delta_brier": 0.0,
            "validation_delta_ece": 0.0,
            "minimum_delta_auprc": 0.01,
            "maximum_brier_worsening": 0.01,
            "maximum_ece_worsening": 0.01,
            "test_metric_used_for_selection": False,
        }
    )
    task_rows = []
    for cancer in PILOT_CANCERS:
        for seed in PILOT_SEEDS:
            for value in (0.001, 0.01, 0.1):
                task_rows.append(
                    {
                        "loco_cancer": cancer,
                        "seed": seed,
                        "residual_shrinkage_lambda": value,
                        "status": "PASS",
                        "residual_initialization_max_abs": 0.0,
                    }
                )
    gate = {
        "status": "PASS",
        "tasks_completed": 27,
        "tasks_expected": 27,
        "selection_used_test_labels": False,
    }
    return predictions, admission, pd.DataFrame(), gate, pd.DataFrame(task_rows)


def test_hard_report_authorizes_strong_consistent_gain():
    metrics, cancer, summary, payload = build_b2_hard_report(
        *_fixtures(), iterations=500, seed=7
    )
    assert len(metrics) == 27
    assert len(cancer) == 12
    assert len(summary) == 4
    assert payload["decision"] == "HARD_GO_B3"
    assert payload["b3_authorized"] is True
    assert payload["primary_scope"] == "Pathway"
    assert payload["primary_best_simple_auprc"] == 0.5
    assert payload["primary_selected_b2_auprc"] == 1.0
    assert payload["primary_prevalence"] == 0.5
    assert payload["primary_best_simple_auprc_over_prevalence"] == 1.0
    assert payload["primary_selected_b2_auprc_over_prevalence"] == 2.0
    assert all(payload["gates"].values())


def test_hard_report_stops_when_selected_model_is_exact_base():
    predictions, admission, fallback, gate, audits = _fixtures()
    predictions["proxy_positive_probability"] = predictions.base_probability
    admission["admitted"] = False
    admission["validation_delta_auprc"] = 0.0
    predictions["module_admitted"] = False
    predictions = predictions.loc[
        predictions.prediction_scale.eq("raw_probability")
    ].copy()
    _, _, _, payload = build_b2_hard_report(
        predictions, admission, fallback, gate, audits, iterations=500, seed=7
    )
    assert payload["decision"] == "HARD_STOP_AFTER_B2"
    assert payload["b3_authorized"] is False
    assert payload["off_max_abs_probability_error"] == 0.0


def test_state_gain_cannot_compensate_for_no_pathway_gain():
    predictions, admission, fallback, gate, audits = _fixtures()
    pathway = predictions.target_subtype.eq("Pathway")
    predictions.loc[pathway, "proxy_positive_probability"] = predictions.loc[
        pathway, "base_probability"
    ]
    _, _, summary, payload = build_b2_hard_report(
        predictions,
        admission,
        fallback,
        gate,
        audits,
        iterations=500,
        seed=7,
    )
    all_targets = summary.loc[summary.scope.eq("ALL_TARGETS_MACRO")].iloc[0]
    assert all_targets.mean_delta_auprc > 0.02
    assert payload["primary_scope"] == "Pathway"
    assert payload["primary_mean_delta_auprc"] == 0.0
    assert payload["decision"] == "HARD_STOP_AFTER_B2"
    assert payload["b3_authorized"] is False


def test_hard_report_rejects_candidate_drift_across_seeds():
    predictions, admission, fallback, gate, audits = _fixtures()
    drift = (
        predictions.loco_cancer.eq("BRCA")
        & predictions.target_subtype.eq("Pathway")
        & predictions.seed.eq(PILOT_SEEDS[-1])
        & predictions.candidate_id.str.endswith("|0")
    )
    predictions.loc[drift, "candidate_id"] += "|DRIFT"
    with pytest.raises(RuntimeError, match="candidate keys or labels drift"):
        build_b2_hard_report(
            predictions,
            admission,
            fallback,
            gate,
            audits,
            iterations=100,
            seed=7,
        )
