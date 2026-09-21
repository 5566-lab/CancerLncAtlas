from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v31_simple_loco import (
    CrossCancerSufficientStatistics,
    FEATURE_COLUMNS,
    SimpleLocoContractError,
    aggregate_cross_cancer_features,
    fit_best_simple_loco,
    nested_cancer_crossfit_predictions,
)


def _candidates(cancers=("A", "B", "C", "D")) -> pd.DataFrame:
    labels = {
        "A": (1, 0, 1, 0),
        "B": (1, 0, 0, 0),
        "C": (1, 1, 0, 0),
        "D": (0, 1, 1, 0),
    }
    effects = {
        "A": (0.5, -0.1, 0.3, 0.0),
        "B": (0.4, -0.2, -0.2, 0.0),
        "C": (0.6, 0.3, -0.1, 0.0),
        "D": (-0.4, 0.4, 0.2, 0.0),
    }
    rows = []
    for cancer in cancers:
        for index, (lnc, target, kind) in enumerate(
            (
                ("L1", "P1", "pathway"),
                ("L2", "P1", "pathway"),
                ("L1", "stemness_rna::RNAss", "state"),
                ("L2", "stemness_dna::DNAss", "state"),
            )
        ):
            rows.append(
                {
                    "candidate_id": f"{cancer}:{lnc}:{target}",
                    "cancer_id": cancer,
                    "lncrna_id": lnc,
                    "target_id": target,
                    "target_type": kind,
                    "label": labels[cancer][index],
                    "effect": effects[cancer][index],
                    "n_observed": 50 + index,
                }
            )
    return pd.DataFrame(rows)


def test_aggregate_is_query_outcome_invariant_and_reports_required_features() -> None:
    frame = _candidates()
    source = frame.loc[frame.cancer_id.isin(["A", "B"])]
    query = frame.loc[frame.cancer_id.eq("C")]
    observed = aggregate_cross_cancer_features(source, query, "C")
    permuted = query.copy()
    permuted["label"] = 1 - permuted.label
    permuted["effect"] = permuted.effect.iloc[::-1].to_numpy()
    invariant = aggregate_cross_cancer_features(source, permuted, "C")
    pd.testing.assert_frame_equal(observed, invariant)
    assert set(FEATURE_COLUMNS).issubset(observed.columns)
    assert observed.n_evaluable_cancers.eq(2).all()


def test_query_cancer_in_source_fails_closed() -> None:
    frame = _candidates()
    with pytest.raises(SimpleLocoContractError, match="leaked"):
        aggregate_cross_cancer_features(frame, frame.loc[frame.cancer_id.eq("C")], "C")


def test_sufficient_statistics_exactly_match_direct_cancer_exclusion() -> None:
    frame = _candidates()
    query = frame.loc[frame.cancer_id.eq("C")]
    direct = aggregate_cross_cancer_features(
        frame.loc[frame.cancer_id.isin(["A", "B", "D"])], query, "C"
    ).sort_values("candidate_id").reset_index(drop=True)
    statistics = CrossCancerSufficientStatistics.from_frame(frame)
    scalable = statistics.features(
        query, "C", excluded_cancers={"C"}
    ).sort_values("candidate_id").reset_index(drop=True)
    np.testing.assert_allclose(
        direct[list(FEATURE_COLUMNS)].to_numpy(float),
        scalable[list(FEATURE_COLUMNS)].to_numpy(float),
        atol=1e-12,
        rtol=1e-12,
    )
    assert direct.source_cancers.tolist() == scalable.source_cancers.tolist()


def test_nested_crossfit_prediction_for_cancer_is_outcome_invariant() -> None:
    frame = _candidates()
    original = nested_cancer_crossfit_predictions(frame, seed=17)
    changed = frame.copy()
    mask = changed.cancer_id.eq("C")
    changed.loc[mask, "label"] = 1 - changed.loc[mask, "label"]
    changed.loc[mask, "effect"] = -changed.loc[mask, "effect"]
    repeated = nested_cancer_crossfit_predictions(changed, seed=17)
    score_columns = [column for column in original if column.startswith("score_")]
    left = original.loc[original.cancer_id.eq("C")].sort_values("candidate_id")
    right = repeated.loc[repeated.cancer_id.eq("C")].sort_values("candidate_id")
    np.testing.assert_allclose(left[score_columns], right[score_columns], atol=0, rtol=0)


def test_best_simple_uses_training_only_and_emits_frozen_logit() -> None:
    frame = _candidates(("A", "B", "C", "D"))
    train = frame.loc[frame.cancer_id.isin(["A", "B", "C"])]
    query = frame.loc[frame.cancer_id.eq("D")]
    fit = fit_best_simple_loco(train, {"validation": query}, seed=23)
    prediction = fit.query_predictions["validation"]
    assert fit.oof_predictions.z_base.notna().all()
    assert fit.oof_predictions.metric_scope.eq("cancer_crossfit_OOF").all()
    for row in fit.oof_predictions.itertuples(index=False):
        assert row.cancer_id not in row.simple_base_fit_cancers.split("|")
    assert len(prediction) == len(query)
    assert prediction.best_simple_method.notna().all()
    assert np.isfinite(prediction.z_base).all()
    assert prediction.simple_base_fit_cancers.eq("A|B|C").all()
    changed = query.copy()
    changed["label"] = 1 - changed.label
    changed["effect"] = 0.999
    repeated = fit_best_simple_loco(train, {"validation": changed}, seed=23)
    np.testing.assert_allclose(
        prediction.best_simple_score,
        repeated.query_predictions["validation"].best_simple_score,
        atol=0,
        rtol=0,
    )


def test_best_simple_can_be_selected_on_validation_but_never_test() -> None:
    frame = _candidates(("A", "B", "C", "D"))
    train = frame.loc[frame.cancer_id.isin(["A", "B", "C"])]
    validation = frame.loc[frame.cancer_id.eq("D")]
    fit = fit_best_simple_loco(
        train, {"val": validation}, seed=29, selection_split="val"
    )
    assert fit.selected_methods.selection_scope.eq("val").all()
    assert fit.oof_predictions.best_simple_selection_scope.eq("val").all()
    assert fit.query_predictions["val"].best_simple_selection_scope.eq("val").all()
    with pytest.raises(SimpleLocoContractError, match="test is forbidden"):
        fit_best_simple_loco(
            train, {"test": validation}, seed=29, selection_split="test"
        )
