from __future__ import annotations

import numpy as np
import pandas as pd

from cc_hhgt.v32.clinical_entity_training import (
    analyse_entity_matrix,
    cox_null_score_one_step,
    fold_local_residualize,
    fold_roles,
    summarise_oof_associations,
    validate_entity_clinical_contract,
)


def test_fold_roles_are_three_one_one_and_disjoint() -> None:
    for fold in range(5):
        roles = fold_roles(fold)
        assert len(roles["train"]) == 3
        assert roles["test"] == (fold,)
        assert roles["validation"] == ((fold + 1) % 5,)
        assert set(roles["train"]).isdisjoint(roles["validation"])
        assert set(roles["train"]).isdisjoint(roles["test"])
        assert set(roles["validation"]).isdisjoint(roles["test"])
        assert set(roles["train"] + roles["validation"] + roles["test"]) == set(range(5))


def test_vectorised_cox_score_recovers_risk_direction() -> None:
    rng = np.random.default_rng(812)
    x = rng.normal(size=180)
    time = rng.exponential(scale=np.exp(-1.1 * x)) + 0.1
    event = np.ones(len(x), dtype=float)
    result = cox_null_score_one_step(
        np.column_stack([x, -x]), time, event, min_patients=30, min_events=10
    )
    assert result["failure_reason"] == ""
    assert result["estimable"].all()
    assert result["beta"][0] > 0
    assert result["beta"][1] < 0
    assert result["p_value"][0] < 1e-5


def test_preprocessing_is_invariant_to_test_value_changes() -> None:
    rng = np.random.default_rng(32)
    values = rng.normal(size=(30, 3))
    covariates = rng.normal(size=(30, 2))
    train = np.zeros(30, dtype=bool)
    train[:18] = True
    first, valid_first, _ = fold_local_residualize(
        values, covariates, train, min_train_measurements=10
    )
    changed = values.copy()
    changed[~train] += 10000.0
    second, valid_second, _ = fold_local_residualize(
        changed, covariates, train, min_train_measurements=10
    )
    np.testing.assert_allclose(first[train], second[train])
    np.testing.assert_array_equal(valid_first, valid_second)


def _toy_inputs() -> tuple[list[str], np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(991)
    patients = [f"TCGA-AA-{index:04d}" for index in range(50)]
    fold_ids = np.arange(50) % 5
    endpoint_rows = []
    x = rng.normal(size=50)
    for endpoint in ("OS", "DSS", "PFI", "PFS", "DFI"):
        time = rng.exponential(scale=np.exp(-0.6 * x)) + 1.0
        for patient, duration in zip(patients, time, strict=True):
            endpoint_rows.append(
                {
                    "cancer_id": "TOY", "patient_id": patient,
                    "clinical_endpoint": endpoint, "time_days": duration,
                    "event": 1.0, "endpoint_available": True,
                }
            )
    for patient in patients:
        endpoint_rows.append(
            {
                "cancer_id": "TOY", "patient_id": patient,
                "clinical_endpoint": "DFS", "time_days": np.nan,
                "event": np.nan, "endpoint_available": False,
            }
        )
    covariates = pd.DataFrame(
        {
            "cancer_id": "TOY", "patient_id": patients,
            "age_years": 50 + rng.normal(size=50), "sex": ["MALE", "FEMALE"] * 25,
            "pathologic_stage": ["Stage II"] * 50, "clinical_stage": [pd.NA] * 50,
        }
    )
    return patients, fold_ids, pd.DataFrame(endpoint_rows), covariates


def test_entity_analysis_has_all_folds_endpoints_and_dfs_null() -> None:
    patients, folds, endpoints, covariates = _toy_inputs()
    rng = np.random.default_rng(4)
    values = rng.normal(size=(50, 2))
    records = analyse_entity_matrix(
        cancer_id="TOY", subject_type="lncRNA", patient_ids=patients,
        entity_ids=["LNC:A", "LNC:B"], values=values, fold_ids=folds,
        endpoint_frame=endpoints, covariate_frame=covariates,
        block_size=1, min_train_patients=20, min_train_events=5,
        min_validation_patients=5, min_validation_events=2,
        min_test_patients=5, min_test_events=2, min_train_measurements=10,
    )
    assert len(records) == 2 * 6 * 5
    assert set(records.patient_fold_id) == set(range(5))
    dfs = records.loc[records.clinical_endpoint.eq("DFS")]
    assert not dfs.fold_available.any()
    assert dfs.failure_reason.eq("NO_DISTINCT_DFS_SOURCE").all()
    assert dfs.test_beta.isna().all()
    assert not records.old_checkpoint_loaded.any()
    assert not records.old_predictions_used_as_features.any()
    assert not records.changes_primary_ranking.any()

    summary = summarise_oof_associations(
        records, ["LNC:A", "LNC:B"], cancer_id="TOY", subject_type="lncRNA",
        min_folds_available=3,
    )
    assert len(summary) == 2 * 6
    dfs_summary = summary.loc[summary.clinical_endpoint.eq("DFS")]
    assert not dfs_summary.availability.any()
    assert dfs_summary.meta_beta.isna().all()
    assert dfs_summary.clinical_relevance_probability.isna().all()
    assert dfs_summary.failure_reason.eq("NO_DISTINCT_DFS_SOURCE").all()
    assert not summary.changes_primary_ranking.any()

    lineage = {
        "analysis_version": "CancerLncAtlas_V3.2_CLINICAL_ENTITY_ASSOCIATION",
        "fresh_statistical_calculation": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "changes_primary_ranking": False,
    }
    validation = validate_entity_clinical_contract(summary, lineage)
    assert validation["status"] == "PASS"
    assert validation["dfs_all_null_with_reason"] is True


def test_explicit_patient_ids_are_not_unconditionally_truncated_to_12() -> None:
    patients, folds, endpoints, covariates = _toy_inputs()
    replacements = {
        patients[0]: "PATIENT-0001-A",
        patients[1]: "PATIENT-0001-B",
    }
    patients = [replacements.get(value, value) for value in patients]
    endpoints["patient_id"] = endpoints.patient_id.replace(replacements)
    covariates["patient_id"] = covariates.patient_id.replace(replacements)
    records = analyse_entity_matrix(
        cancer_id="TOY",
        subject_type="lncRNA",
        patient_ids=patients,
        entity_ids=["LNC:A"],
        values=np.random.default_rng(721).normal(size=(50, 1)),
        fold_ids=folds,
        endpoint_frame=endpoints,
        covariate_frame=covariates,
        min_train_patients=20,
        min_train_events=5,
        min_validation_patients=5,
        min_validation_events=2,
        min_test_patients=5,
        min_test_events=2,
        min_train_measurements=10,
    )
    assert len(records) == 6 * 5
