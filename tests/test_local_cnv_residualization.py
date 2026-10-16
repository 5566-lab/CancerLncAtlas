import json

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.local_cnv_confounding import (
    COVARIATE_CONTRACT_VERSION,
    DEFAULT_COVARIATES,
    fold_local_pair_models,
)
from cc_hhgt.v32.local_cnv_core_preparation import (
    adjusted_association_frame,
    validate_local_cnv_audit,
    validate_validation_a1_authority,
)
from scripts.run_v32_local_cnv_confounding_audit_r2 import edge_candidate_frame


def _pair(local, pathway, *, design=None):
    n = len(local)
    t = np.arange(n, dtype=float)
    expression = (np.sin(t / 2.7) + 0.07 * t)[:, None]
    activity = (np.cos(t / 3.1) - 0.03 * t)[:, None]
    design = np.ones((n, 1), dtype=float) if design is None else design
    return fold_local_pair_models(
        expression,
        activity,
        np.asarray(local, dtype=float)[:, None],
        np.asarray(pathway, dtype=float)[:, None],
        design,
        local_callable=np.ones((n, 1), dtype=bool),
        pathway_callable=np.ones((n, 1), dtype=bool),
        min_patients=12,
    )


def test_constant_local_and_pathway_cnv_are_noop_and_remain_available():
    n = 20
    result = _pair(np.zeros(n), np.zeros(n))

    assert np.isfinite(result["rho_a0_matched"][0])
    assert np.isfinite(result["rho_a1_local"][0])
    assert np.isfinite(result["rho_a2_local_pathway"][0])
    np.testing.assert_allclose(result["rho_a1_local"], result["rho_a0_matched"])
    np.testing.assert_allclose(result["rho_a2_local_pathway"], result["rho_a1_on_a2_matched"])
    assert result["residual_design_rank_a0"][0] == 1
    assert result["residual_design_rank_a1"][0] == 1
    assert result["residual_design_rank_a2"][0] == 1
    assert result["correlation_df_a1"][0] == n - 2
    assert result["correlation_df_a2"][0] == n - 2
    assert bool(result["available"][0])
    assert bool(result["available_a2"][0])


def test_variable_local_and_pathway_cnv_increase_rank_independently():
    n = 24
    t = np.arange(n, dtype=float)
    result = _pair(np.sin(t / 2.0), np.cos(t / 4.0))

    assert result["residual_design_rank_a0"][0] == 1
    assert result["residual_design_rank_a1"][0] == 2
    assert result["residual_design_rank_a2"][0] == 3
    assert result["correlation_df_a1"][0] == n - 3
    assert result["correlation_df_a2"][0] == n - 4
    assert np.isfinite(result["local_cnv_beta"][0])
    assert np.isfinite(result["pathway_cnv_beta"][0])


def test_constant_pathway_cnv_does_not_add_a2_rank_after_local_adjustment():
    n = 18
    t = np.arange(n, dtype=float)
    result = _pair(np.sin(t / 2.0), np.zeros(n))

    assert result["residual_design_rank_a0"][0] == 1
    assert result["residual_design_rank_a1"][0] == 2
    assert result["residual_design_rank_a2"][0] == 2
    assert result["correlation_df_a1"][0] == n - 3
    assert result["correlation_df_a2"][0] == n - 3
    np.testing.assert_allclose(
        result["rho_a2_local_pathway"], result["rho_a1_on_a2_matched"]
    )


def test_design_collinear_cnv_does_not_increase_rank():
    n = 22
    t = np.arange(n, dtype=float)
    design = np.column_stack([np.ones(n), t])
    result = _pair(3.0 + 2.0 * t, np.zeros(n), design=design)

    assert result["residual_design_rank_a0"][0] == 2
    assert result["residual_design_rank_a1"][0] == 2
    assert result["residual_design_rank_a2"][0] == 2
    assert result["correlation_df_a1"][0] == n - 3
    assert result["correlation_df_a2"][0] == n - 3
    assert np.isfinite(result["rho_a1_local"][0])


def _association_frame(rank_values=(1, 2)):
    n = np.array([20, 20])
    rank = np.array(rank_values)
    return pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA"],
            "lncrna_id": ["L1", "L2"],
            "pathway_id": ["P1", "P2"],
            "fold_id": [0, 0],
            "rho_A1_local": [0.3, 0.4],
            "p_A1_local": [0.01, 0.02],
            "fdr_official_A1_local": [0.02, 0.03],
            "label_A1_local": ["strong_positive", "strong_positive"],
            "direction_A1_local": ["positive", "positive"],
            "n_A1_local": n,
            "residual_design_rank_A1_local": rank,
            "correlation_df_A1_local": n - rank - 1,
            "availability": ["available", "available"],
            "unavailable_reason": ["", ""],
        }
    )


def test_core_adapter_preserves_per_candidate_rank_and_df():
    result = adjusted_association_frame(
        _association_frame(), prefix="A1_local", expected_cancer="BRCA", expected_fold=0
    )
    assert result.residual_design_rank.tolist() == [1, 2]
    assert result.correlation_df.tolist() == [18, 17]


def test_core_adapter_rejects_missing_per_candidate_rank_and_df():
    frame = _association_frame().drop(
        columns=["residual_design_rank_A1_local", "correlation_df_A1_local"]
    )
    with pytest.raises(ValueError, match="per-candidate residual rank"):
        adjusted_association_frame(frame, prefix="A1_local")


def test_g0_edge_p_values_use_per_lncrna_rank():
    n = 60
    vector = np.linspace(-1.0, 1.0, n)
    vector *= np.sqrt(n - 1) / np.linalg.norm(vector)
    x = np.column_stack([vector, vector])
    genes = vector[:, None]
    frame, total = edge_candidate_frame(
        x,
        genes,
        np.array(["L1", "L2"]),
        np.array(["G1"]),
        n=n,
        rank=np.array([1, 2]),
        model="E1",
    )
    assert total == 2
    assert frame.residual_design_rank_E1.tolist() == [1, 2]
    assert frame.correlation_df_E1.tolist() == [n - 2, n - 3]


def test_nonpositive_df_is_typed_unavailable_even_when_rho_is_finite():
    n = 12
    design = np.eye(n, dtype=float)[:, : n - 1]
    result = _pair(np.sin(np.arange(n)), np.cos(np.arange(n)), design=design)
    assert np.isfinite(result["rho_a0_matched"][0])
    assert np.isnan(result["p_a0_matched"][0])
    assert np.isnan(result["p_a1_local"][0])
    assert result["correlation_df_a0"][0] == 0
    assert result["correlation_df_a1"][0] == 0
    assert not bool(result["available"][0])


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_authority_validators_fail_closed_on_shared_covariate_contract(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    success = {
        "status": "SUCCESS",
        "decision": "LOCAL_CNV_CORE_RETRAIN_REQUIRED",
        "association_C0": "shared_covariate_current_v32_formal_association",
        "covariate_contract_version": COVARIATE_CONTRACT_VERSION,
        "covariate_contract": {"version": COVARIATE_CONTRACT_VERSION},
        "covariates": {"declared": list(DEFAULT_COVARIATES)},
    }
    _write_json(local / "SUCCESS.json", success)
    _write_json(local / "LOCAL_CNV_INDEPENDENT_AUDIT.json", {"status": "PASS"})
    _write_json(local / "LOCAL_CNV_DECISION.json", {"status": "PASS", "decision": success["decision"]})
    assert validate_local_cnv_audit(local)["covariates_declared"] == list(DEFAULT_COVARIATES)

    for field, bad in (
        ("covariate_contract_version", "OLD"),
        ("covariates", {"declared": list(DEFAULT_COVARIATES[:-1])}),
    ):
        mutated = dict(success)
        mutated[field] = bad
        _write_json(local / "SUCCESS.json", mutated)
        with pytest.raises(RuntimeError, match="shared|declaration"):
            validate_local_cnv_audit(local)
    _write_json(local / "SUCCESS.json", success)

    validation = tmp_path / "validation"
    validation.mkdir()
    validation_success = {
        "status": "PASS",
        "covariate_contract_version": COVARIATE_CONTRACT_VERSION,
        "covariate_contract": {"version": COVARIATE_CONTRACT_VERSION},
        "covariates": {"declared": list(DEFAULT_COVARIATES)},
        "gates": {
            "validation_label_only": True,
            "designated_oof_partition_read": False,
            "sealed_test_accessed": False,
            "mutation_used": False,
            "candidate_closure": True,
            "typed_unavailable_preserved": True,
            "shared_covariate_contract": COVARIATE_CONTRACT_VERSION,
            "covariates_declared": list(DEFAULT_COVARIATES),
        },
    }
    _write_json(validation / "SUCCESS.json", validation_success)
    _write_json(validation / "INDEPENDENT_AUDIT.json", {"status": "PASS"})
    assert validate_validation_a1_authority(validation)["covariates_declared"] == list(DEFAULT_COVARIATES)
    for key, bad in (
        ("covariate_contract_version", "OLD"),
        ("covariates", {"declared": list(DEFAULT_COVARIATES[:-1])}),
    ):
        mutated = dict(validation_success)
        mutated[key] = bad
        _write_json(validation / "SUCCESS.json", mutated)
        with pytest.raises(RuntimeError, match="shared|declaration"):
            validate_validation_a1_authority(validation)
