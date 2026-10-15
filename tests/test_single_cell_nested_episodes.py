from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cc_hhgt.v32.single_cell_nested_episodes import (
    FIT_GEOMETRY_ISOLATION,
    SUPPORT_LIMITED_FIT_GEOMETRY_ISOLATION,
    VALIDATION_GEOMETRY_ISOLATION,
    fit_nested_metacells,
    lasso_observations,
    nested_fold_plans,
    source_node_features,
    validate_nested_isolation,
)


def _folds(donors_per_fold: int = 2) -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for donor in range(donors_per_fold):
            rows.append(
                {
                    "cancer_id": "C1",
                    "donor_id": f"D{fold}_{donor}",
                    "outer_fold_id": fold,
                }
            )
    return pd.DataFrame(rows)


def test_global_nested_plan_reserves_next_fold_from_every_fit_source():
    plans = nested_fold_plans(_folds(), cancer_id="C1", outer_fold_id=0)
    validation = plans[0]
    assert validation.role == "validation"
    assert validation.inner_target_fold_id == 1
    assert validation.inner_validation_fold_id == 1
    assert validation.geometry_isolation == VALIDATION_GEOMETRY_ISOLATION
    assert set(validation.source_donors) == {
        "D2_0", "D2_1", "D3_0", "D3_1", "D4_0", "D4_1"
    }

    fit = next(plan for plan in plans if plan.inner_target_fold_id == 2)
    assert fit.role == "fit"
    assert fit.geometry_isolation == FIT_GEOMETRY_ISOLATION
    assert set(fit.source_donors) == {"D3_0", "D3_1", "D4_0", "D4_1"}
    assert set(fit.excluded_donors) == {
        "D0_0", "D0_1", "D1_0", "D1_1", "D2_0", "D2_1"
    }
    assert not set(fit.source_donors) & set(fit.inner_validation_donors)


def test_small_cohort_uses_leave_one_target_fit_and_does_not_supply_validation():
    plans = nested_fold_plans(_folds(1), cancer_id="C1", outer_fold_id=0)
    assert len(plans) == 4
    assert {plan.role for plan in plans} == {"fit"}
    assert all(plan.available for plan in plans)
    assert all(plan.support_limited_fit for plan in plans)
    assert all(not plan.validation_provider for plan in plans)
    assert all(not plan.inner_validation_donors for plan in plans)
    assert all(plan.minimum_source_donors == 2 for plan in plans)
    assert all(
        plan.geometry_isolation == SUPPORT_LIMITED_FIT_GEOMETRY_ISOLATION
        for plan in plans
    )
    assert all(len(plan.source_donors) == 3 for plan in plans)


def test_four_donor_cohort_keeps_two_source_donors_in_every_fit():
    folds = _folds(1).loc[lambda value: value.outer_fold_id.lt(4)].copy()
    plans = nested_fold_plans(folds, cancer_id="C1", outer_fold_id=0)
    assert len(plans) == 3
    assert all(plan.available and plan.minimum_source_donors == 2 for plan in plans)
    assert all(len(plan.source_donors) == 2 for plan in plans)
    assert {plan.inner_target_fold_id for plan in plans} == {1, 2, 3}


def _synthetic_inputs():
    folds = _folds()
    donors = folds.donor_id.tolist()
    metadata_rows = []
    for donor in donors:
        for cell in range(9):
            metadata_rows.append(
                {
                    "cell_id": f"{donor}:C{cell}",
                    "dataset_id": "DSET",
                    "cancer_id": "C1",
                    "donor_id": donor,
                    "cell_type": "T",
                    "state_id": "N",
                }
            )
    metadata = pd.DataFrame(metadata_rows)
    rng = np.random.default_rng(7)
    coding = sparse.csr_matrix(rng.gamma(2.0, 1.0, size=(len(metadata), 8)))
    lnc = sparse.csr_matrix(rng.gamma(1.5, 1.0, size=(len(metadata), 3)))
    return folds, metadata, coding, lnc


def test_excluded_donor_values_do_not_change_source_geometry_or_features():
    folds, metadata, coding, lnc = _synthetic_inputs()
    plan = next(
        plan
        for plan in nested_fold_plans(folds, cancer_id="C1", outer_fold_id=0)
        if plan.role == "fit" and plan.inner_target_fold_id == 2
    )
    kwargs = dict(
        all_gene_library=np.ones(len(metadata)),
        metadata=metadata,
        plan=plan,
        measurement_scale="log_normalized",
        max_genes=6,
        components=3,
        max_cells_per_donor=9,
        target_size=3,
        min_size=2,
        radius_quantile=1.0,
        seed=19,
    )
    first = fit_nested_metacells(coding, coding, lnc, **kwargs)
    persisted = set(first["metadata"].donor_id.astype(str))
    assert persisted == set(plan.source_donors) | set(plan.inner_target_donors)
    assert not persisted & set(plan.outer_query_donors)
    assert not persisted & set(plan.inner_validation_donors)
    changed_coding = coding.toarray()
    changed_lnc = lnc.toarray()
    excluded = metadata.donor_id.isin(plan.excluded_donors).to_numpy()
    changed_coding[excluded] = changed_coding[excluded] * 1000 + 500
    changed_lnc[excluded] = changed_lnc[excluded] * 1000 + 500
    second = fit_nested_metacells(
        sparse.csr_matrix(changed_coding),
        sparse.csr_matrix(changed_coding),
        sparse.csr_matrix(changed_lnc),
        **kwargs,
    )
    assert first["geometry_audit"]["pca_training_donors"] == list(
        plan.source_donors
    )
    assert first["geometry_audit"]["radius_training_donors"] == list(
        plan.source_donors
    )
    source_first = first["metadata"].donor_id.isin(plan.source_donors).to_numpy()
    source_second = second["metadata"].donor_id.isin(plan.source_donors).to_numpy()
    np.testing.assert_allclose(
        first["coding"][source_first].toarray(),
        second["coding"][source_second].toarray(),
    )
    np.testing.assert_allclose(
        first["lncrna"][source_first].toarray(),
        second["lncrna"][source_second].toarray(),
    )

    activity_first = first["coding"][:, :2].toarray()
    activity_second = second["coding"][:, :2].toarray()
    target_rows = pd.DataFrame({"cell_key": ["T"]})
    family = sparse.csr_matrix([[1.0], [1.0]])
    features_first, contexts_first = source_node_features(
        first["coding"], first["lncrna"], activity_first, first["metadata"],
        plan, family, target_rows=target_rows,
    )
    features_second, contexts_second = source_node_features(
        second["coding"], second["lncrna"], activity_second, second["metadata"],
        plan, family, target_rows=target_rows,
    )
    pd.testing.assert_frame_equal(contexts_first, contexts_second)
    for node_type in features_first:
        np.testing.assert_allclose(features_first[node_type], features_second[node_type])


def test_lasso_rows_and_isolation_audit_reject_any_excluded_source():
    folds, metadata, coding, lnc = _synthetic_inputs()
    plan = nested_fold_plans(folds, cancer_id="C1", outer_fold_id=0)[0]
    result = fit_nested_metacells(
        coding,
        coding,
        lnc,
        np.ones(len(metadata)),
        metadata,
        plan,
        measurement_scale="log_normalized",
        max_genes=6,
        components=3,
        max_cells_per_donor=9,
        target_size=3,
        min_size=2,
        radius_quantile=1.0,
        seed=23,
    )
    observations = lasso_observations(result["metadata"], plan)
    allowed = set(observations.loc[observations.lasso_input_allowed, "donor_id"])
    forbidden = set(observations.loc[~observations.lasso_input_allowed, "donor_id"])
    assert allowed == set(plan.source_donors)
    assert not allowed & set(plan.excluded_donors)
    assert forbidden == set(plan.inner_target_donors)
    assert not set(observations.donor_id) & set(plan.outer_query_donors)
    universe = pd.DataFrame(
        {
            "candidate_selection_query_donor_used": [False],
            "candidate_selection_target_donor_used": [False],
            "candidate_selection_validation_donor_used": [False],
        }
    )
    targets = pd.DataFrame(
        {"query_donor_id": [plan.inner_target_donors[0]], "cell_key": ["T"]}
    )
    validate_nested_isolation(
        plan, result["geometry_audit"], observations, universe, targets
    )
    leaking = observations.copy()
    row = leaking.index[leaking.donor_id.eq(plan.inner_target_donors[0])][0]
    leaking.loc[row, "lasso_input_allowed"] = True
    with pytest.raises(ValueError, match="LASSO_SOURCE_LEAK"):
        validate_nested_isolation(
            plan, result["geometry_audit"], leaking, universe, targets
        )
    with pytest.raises(ValueError, match="UNIVERSE_LEAK_FLAG"):
        validate_nested_isolation(
            plan,
            result["geometry_audit"],
            observations,
            universe.assign(candidate_selection_validation_donor_used=True),
            targets,
        )


def test_lasso_scope_allows_planned_donor_without_estimable_metacells():
    folds, metadata, coding, lnc = _synthetic_inputs()
    plan = nested_fold_plans(folds, cancer_id="C1", outer_fold_id=0)[0]
    result = fit_nested_metacells(
        coding,
        coding,
        lnc,
        np.ones(len(metadata)),
        metadata,
        plan,
        measurement_scale="log_normalized",
        max_genes=6,
        components=3,
        max_cells_per_donor=9,
        target_size=3,
        min_size=2,
        radius_quantile=1.0,
        seed=29,
    )
    missing_source = plan.source_donors[0]
    metacells = result["metadata"].loc[
        ~result["metadata"].donor_id.astype(str).eq(missing_source)
    ].reset_index(drop=True)
    observations = lasso_observations(metacells, plan)
    assert missing_source not in set(observations.donor_id.astype(str))
    universe = pd.DataFrame(
        {
            "candidate_selection_query_donor_used": [False],
            "candidate_selection_target_donor_used": [False],
            "candidate_selection_validation_donor_used": [False],
        }
    )
    targets = pd.DataFrame(
        {"query_donor_id": [plan.inner_target_donors[0]], "cell_key": ["T"]}
    )
    validate_nested_isolation(
        plan, result["geometry_audit"], observations, universe, targets
    )

    undeclared = metacells.iloc[[0]].copy()
    undeclared["donor_id"] = "undeclared-donor"
    with pytest.raises(ValueError, match="LASSO_DONOR_SCOPE_DRIFT"):
        lasso_observations(pd.concat([metacells, undeclared], ignore_index=True), plan)
