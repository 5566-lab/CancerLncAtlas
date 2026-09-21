from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cc_hhgt.v32.associations import compute_fold_associations
from cc_hhgt.v32.patient_folds import assign_outer_split, build_patient_fold_manifest


def _fixture():
    samples = pd.DataFrame(
        [("BRCA", f"S{i}", f"P{i}") for i in range(25)],
        columns=["cancer_id", "sample_id", "patient_id"],
    )
    folds = assign_outer_split(build_patient_fold_manifest(samples), 0)
    expression = pd.DataFrame(
        [
            ("BRCA", f"S{i}", lnc, float(i if lnc == "L1" else 25 - i))
            for i in range(25)
            for lnc in ["L1", "L2"]
        ],
        columns=["cancer_id", "sample_id", "lncrna_id", "logcpm"],
    )
    activity = pd.DataFrame(
        [
            (f"S{i}", pathway, float(i if pathway == "P1" else np.sin(i)))
            for i in range(25)
            for pathway in ["P1", "P2"]
        ],
        columns=["sample_id", "pathway_id", "pathway_activity_scaled"],
    )
    hierarchy = pd.DataFrame({"pathway_id": ["P1", "P2"], "pathway_family_id": ["F1", "F2"]})
    eligible = pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA"],
            "lncrna_id": ["L1", "L2"],
            "within_cancer_eligible": [True, True],
            "shared_or_local_scope": ["shared", "cancer_local"],
            "detection_rate": [1.0, 1.0],
        }
    )
    return folds, expression, activity, hierarchy, eligible


def test_validation_test_mutations_cannot_change_train_associations() -> None:
    folds, expression, activity, hierarchy, eligible = _fixture()
    first = compute_fold_associations(expression, activity, folds, hierarchy, eligible)
    heldout = set(folds.loc[folds.split.ne("train"), "sample_id"])
    mutated_expression = expression.copy()
    mutated_expression.loc[mutated_expression.sample_id.isin(heldout), "logcpm"] += 10000
    mutated_activity = activity.copy()
    mutated_activity.loc[mutated_activity.sample_id.isin(heldout), "pathway_activity_scaled"] *= -999
    second = compute_fold_associations(mutated_expression, mutated_activity, folds, hierarchy, eligible)
    assert first.attrs["association_sha256"] == second.attrs["association_sha256"]
    pd.testing.assert_frame_equal(first, second)


def test_association_keys_are_exact_pathway() -> None:
    folds, expression, activity, hierarchy, eligible = _fixture()
    result = compute_fold_associations(expression, activity, folds, hierarchy, eligible)
    assert result.pathway_id.nunique() == 2
    assert result.pathway_family_id.nunique() == 2
    assert not result.duplicated(["cancer_id", "lncrna_id", "pathway_id"]).any()


def test_partial_correlation_df_uses_n_minus_numerical_design_rank_minus_one() -> None:
    folds, expression, activity, hierarchy, eligible = _fixture()
    covariates = pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(25)],
            "age": np.arange(25, dtype=float),
            # The duplicate column must not consume another degree of freedom.
            "age_duplicate": np.arange(25, dtype=float) * 2.0,
        }
    )
    result = compute_fold_associations(
        expression,
        activity,
        folds,
        hierarchy,
        eligible,
        covariates=covariates,
    )
    assert result.residual_design_rank.eq(2).all()
    assert result.correlation_df.eq(result.association_n_samples - 3).all()
    row = result.iloc[0]
    expected_t = row.discovery_effect * np.sqrt(
        row.correlation_df / (1.0 - row.discovery_effect**2)
    )
    expected_p = 2.0 * stats.t.sf(abs(expected_t), df=row.correlation_df)
    assert row.discovery_pvalue == pytest.approx(expected_p)
