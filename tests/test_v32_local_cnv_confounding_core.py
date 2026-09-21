from __future__ import annotations

import numpy as np
import pandas as pd

from cc_hhgt.v32.local_cnv_confounding import (
    association_label,
    bh_fdr,
    classify_impact,
    fold_local_pair_models,
    partial_spearman,
    patient_first_matrix,
)
from scripts.finalize_and_audit_v32_local_cnv_r3 import edge_direction_metrics


def _intercept(n: int) -> np.ndarray:
    return np.ones((n, 1), dtype=float)


def test_no_cnv_effect_preserves_matched_association() -> None:
    rng = np.random.default_rng(7)
    n = 240
    x = rng.normal(size=n)
    y = 0.7 * x + rng.normal(scale=0.7, size=n)
    cnv = rng.normal(size=n)
    base = partial_spearman(x, y, _intercept(n))
    adjusted = partial_spearman(x, y, _intercept(n), x_extras=[cnv])
    assert base.available and adjusted.available
    assert abs(base.rho - adjusted.rho) < 0.03


def test_synthetic_local_cnv_confounding_is_attenuated() -> None:
    rng = np.random.default_rng(11)
    n = 400
    cnv = rng.normal(size=n)
    x = 1.4 * cnv + rng.normal(scale=0.35, size=n)
    y = 1.2 * cnv + rng.normal(scale=0.35, size=n)
    base = partial_spearman(x, y, _intercept(n))
    adjusted = partial_spearman(x, y, _intercept(n), x_extras=[cnv])
    assert abs(base.rho) > 0.80
    assert abs(adjusted.rho) < 0.15


def test_vectorized_four_models_use_identical_matched_patients_and_side_adjustment() -> None:
    rng = np.random.default_rng(13)
    n = 180
    pairs = 4
    local = rng.normal(size=(n, pairs))
    pathway = rng.normal(size=(n, pairs))
    x = 1.3 * local + rng.normal(scale=0.4, size=(n, pairs))
    y = 0.8 * local + 1.1 * pathway + rng.normal(scale=0.4, size=(n, pairs))
    lc = np.ones_like(x, dtype=bool)
    pc = np.ones_like(x, dtype=bool)
    lc[:10, 0] = False
    pc[10:25, 0] = False
    result = fold_local_pair_models(
        x, y, local, pathway, _intercept(n),
        local_callable=lc, pathway_callable=pc,
    )
    assert result["n_a0_matched"].tolist() == [170, 180, 180, 180]
    assert result["n_a1_local"].tolist() == [170, 180, 180, 180]
    assert result["n_a2_local_pathway"].tolist() == [155, 180, 180, 180]
    assert result["available"].all()
    assert np.nanmean(np.abs(result["rho_a1_local"])) < np.nanmean(
        np.abs(result["rho_a0_matched"])
    )
    assert np.nanmean(np.abs(result["rho_a2_local_pathway"])) < np.nanmean(
        np.abs(result["rho_a1_on_a2_matched"])
    )


def test_unavailable_cnv_is_excluded_and_never_imputed_as_zero() -> None:
    x = np.arange(20, dtype=float)
    y = x.copy()
    cnv = np.linspace(-1, 1, 20)
    available = np.ones(20, dtype=bool)
    available[:9] = False
    result = partial_spearman(
        x, y, _intercept(20), x_extras=[cnv], available_mask=available,
        min_patients=12,
    )
    assert not result.available
    assert result.n_patients == 11
    assert result.unavailable_reason == "INSUFFICIENT_MATCHED_PATIENTS"


def test_heldout_changes_do_not_change_train_fit() -> None:
    rng = np.random.default_rng(19)
    n = 100
    fold = np.arange(n) % 5
    train = (fold != 0) & (fold != 1)
    x = rng.normal(size=n)
    y = x + rng.normal(size=n)
    cnv = rng.normal(size=n)
    first = partial_spearman(x, y, _intercept(n), x_extras=[cnv], available_mask=train)
    y[~train] = 1e9
    cnv[~train] = -1e9
    second = partial_spearman(x, y, _intercept(n), x_extras=[cnv], available_mask=train)
    assert first.available and second.available
    assert first.n_patients == second.n_patients
    assert first.residual_design_rank == second.residual_design_rank
    assert first.rho == second.rho
    assert first.p_value == second.p_value
    assert first.x_beta == second.x_beta


def test_duplicate_aliquots_are_averaged_patient_first() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["TCGA-AA-0001-01A", "TCGA-AA-0001-02A", "TCGA-AA-0002-01A"],
            "lncrna_id": ["L1", "L1", "L1"],
            "logcpm": [1.0, 3.0, 7.0],
        }
    )
    matrix, duplicates = patient_first_matrix(
        frame, feature_column="lncrna_id", value_column="logcpm"
    )
    assert duplicates == 2
    assert matrix.loc["TCGA-AA-0001", "L1"] == 2.0
    assert matrix.loc["TCGA-AA-0002", "L1"] == 7.0


def test_bh_and_label_semantics() -> None:
    observed = bh_fdr([0.001, 0.01, np.nan, 0.2])
    assert np.isnan(observed[2])
    assert np.allclose(observed[[0, 1, 3]], [0.003, 0.015, 0.2])
    assert association_label(0.25, 0.04) == "strong_positive"
    assert association_label(-0.16, 0.08) == "weak_positive"
    assert association_label(np.nan, np.nan) == "unavailable"


def test_preregistered_impact_thresholds() -> None:
    small = [{
        "positive_flip_rate": 0.01,
        "direction_flip_rate": 0.01,
        "median_rank_correlation": 0.98,
        "top100_jaccard": 0.90,
        "g0_sign_flip_rate": 0.01,
        "g0_edge_jaccard": 0.90,
    }]
    assert classify_impact(small) == "IMPACT_SMALL"
    middle = [dict(small[0], top100_jaccard=0.80)]
    assert classify_impact(middle) == "SENSITIVITY_ONLY_REVIEW"
    core = [dict(small[0], g0_edge_jaccard=0.70)]
    assert classify_impact(core) == "CORE_RETRAIN_REQUIRED"


def test_g0_stability_keeps_positive_negative_and_per_lncrna_topk_separate() -> None:
    edge = pd.DataFrame({
        "lncrna_id": ["L1", "L1", "L1", "L2"],
        "gene_id": ["G1", "G2", "G3", "G4"],
        "rho_E0": [0.8, -0.7, 0.6, 0.5],
        "rho_E1": [0.7, -0.6, np.nan, -0.4],
        "direction_E0": ["positive", "negative", "positive", "positive"],
        "direction_E1": ["positive", "negative", np.nan, "negative"],
        "rank_E0": [1, 1, 2, 1], "rank_E1": [1, 1, 3, 1],
    })
    result = edge_direction_metrics(edge)
    assert result["positive_edge_jaccard"] == 1 / 3
    assert result["negative_edge_jaccard"] == 1 / 2
    assert 0 <= result["median_lncrna_direction_topk_jaccard"] <= 1
