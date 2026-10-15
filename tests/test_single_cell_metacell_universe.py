import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.single_cell_metacell_universe import (
    DiscoveryPolicy,
    MetacellUniverseError,
    audit_node_coverage,
    attach_query_donor_targets,
    build_dual_universe_partition,
)


CONTEXT = {"dataset_id": "GSE1", "cancer_id": "BRCA", "cell_key": "malignant"}


def evidence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": ["GSE1"] * 4,
            "cancer_id": ["BRCA"] * 4,
            "cell_key": ["malignant"] * 4,
            "lncrna_id": ["L1", "L1", "L1", "L2"],
            "pathway_id": ["P1", "P2", "P3", "P2"],
            "outer_fold_id": [0] * 4,
            "lasso_available": [True] * 4,
            "pathway_stable_support": [0.4, 0.9, 0.2, 0.8],
            "pathway_null_z": [2.0, 3.0, 4.0, 2.5],
            "inner_relative_mse_gain": [0.2, 0.3, 0.4, 0.1],
            "pathway_calibration_available": [True] * 4,
            "pathway_selected_member_count": [1, 2, 1, 2],
            "training_donor_count": [5] * 4,
            "held_out_donor_rows_used": [0] * 4,
            "source_donor_hash": ["train-donors-fold-0"] * 4,
        }
    )


def build(**overrides) -> pd.DataFrame:
    values = {
        "context": CONTEXT,
        "outer_fold_id": 0,
        "bulk_candidates": pd.DataFrame(
            {"cancer_id": ["BRCA", "BRCA"], "lncrna_id": ["L1", "L2"], "pathway_id": ["P1", "P3"]}
        ),
        "lasso_evidence": evidence(),
        "measured_lncrna_ids": ["L1", "L2"],
        "available_pathway_ids": ["P1", "P2", "P3"],
        "core_lncrna_ids": ["L1", "L2"],
        "core_pathway_ids": ["P1", "P2", "P3"],
        "policy": DiscoveryPolicy(top_k_pathways_per_lncrna=1),
    }
    values.update(overrides)
    return build_dual_universe_partition(**values)


def test_bulk_rescore_is_retained_and_lasso_adds_novel_exact_pair():
    result = build()
    roles = result.set_index(["lncrna_id", "pathway_id"]).universe_role.to_dict()
    assert roles[("L1", "P1")] == "bulk_rescore"
    assert roles[("L2", "P3")] == "bulk_rescore"
    assert roles[("L1", "P2")] == "sc_discovery"
    assert roles[("L2", "P2")] == "sc_discovery"
    assert not result.candidate_selection_target_used.any()
    assert result.held_out_donor_rows_used.eq(0).all()


def test_topk_is_deterministic_and_measured_core_domain_is_enforced():
    shuffled = evidence().sample(frac=1, random_state=91).reset_index(drop=True)
    first = build(lasso_evidence=shuffled)
    second = build(lasso_evidence=shuffled.iloc[::-1].reset_index(drop=True))
    pd.testing.assert_frame_equal(first, second)
    with pytest.raises(MetacellUniverseError, match="outside frozen core"):
        build(core_pathway_ids=["P1", "P3"])
    filtered = build(
        core_pathway_ids=["P1", "P3"],
        policy=DiscoveryPolicy(
            top_k_pathways_per_lncrna=1, allow_core_limited_discovery=True
        ),
    )
    assert "P2" not in set(filtered.pathway_id)
    assert filtered.discovery_node_scope.eq("EXPLICIT_FROZEN_CORE_INTERSECTION").all()


def test_node_coverage_audit_lists_oov_nodes_without_silent_drop():
    audit = audit_node_coverage(
        measured_lncrna_ids=["L1", "L2", "L3"],
        available_pathway_ids=["P1", "P2"],
        core_lncrna_ids=["L1", "L2"],
        core_pathway_ids=["P1"],
    )
    assert audit["missing_core_lncrnas"] == ["L3"]
    assert audit["missing_core_pathways"] == ["P2"]
    assert audit["complete"] is False


def test_selector_rejects_target_columns_and_heldout_donor_use():
    with pytest.raises(MetacellUniverseError, match="target-derived"):
        build(lasso_evidence=evidence().assign(rho=0.9))
    leaking = evidence()
    leaking.loc[0, "held_out_donor_rows_used"] = 1
    with pytest.raises(MetacellUniverseError, match="held-out"):
        build(lasso_evidence=leaking)


def test_bulk_and_discovery_overlap_keeps_both_roles():
    result = build(
        bulk_candidates=pd.DataFrame(
            {"cancer_id": ["BRCA"], "lncrna_id": ["L1"], "pathway_id": ["P2"]}
        )
    )
    row = result.loc[(result.lncrna_id == "L1") & (result.pathway_id == "P2")].iloc[0]
    assert row.universe_role == "bulk_and_sc_discovery"
    assert row.is_bulk_candidate and row.is_sc_discovery_candidate


def test_query_targets_attach_after_selection_and_remain_donor_resolved():
    universe = build()
    target = pd.DataFrame(
        {
            "dataset_id": ["GSE1", "GSE1"],
            "cancer_id": ["BRCA", "BRCA"],
            "cell_key": ["malignant", "malignant"],
            "query_donor_id": ["D1", "D2"],
            "lncrna_id": ["L1", "L1"],
            "pathway_id": ["P2", "P2"],
            "rho": [-0.7, 0.4],
            "association_target": [0.7, 0.4],
            "n_metacells": [25, 31],
            "association_method": ["WITHIN_DONOR_METACELL_SPEARMAN"] * 2,
            "association_observation_unit": ["within_donor_metacell"] * 2,
            "association_target_level": ["donor_x_celltype_x_lncrna_x_exact_pathway"] * 2,
            "donor_resolved_target": [True, True],
            "donor_target_row_id": ["D1:L1:P2", "D2:L1:P2"],
        }
    )
    result = attach_query_donor_targets(universe, target, expected_outer_fold_id=0)
    assert result.query_donor_id.tolist() == ["D1", "D2"]
    np.testing.assert_allclose(result.association_target, result.rho.abs())
    assert result.universe_role.eq("sc_discovery").all()


def test_query_target_contract_rejects_aggregate_or_too_small_targets():
    universe = build()
    target = pd.DataFrame(
        {
            "dataset_id": ["GSE1"], "cancer_id": ["BRCA"], "cell_key": ["malignant"],
            "query_donor_id": ["D1"], "lncrna_id": ["L1"], "pathway_id": ["P2"],
            "rho": [0.2], "association_target": [0.2], "n_metacells": [19],
            "association_method": ["CROSS_DONOR_PSEUDOBULK_SPEARMAN"],
            "association_observation_unit": ["donor_mean"],
            "association_target_level": ["donor_x_celltype_x_lncrna_x_exact_pathway"],
            "donor_resolved_target": [True], "donor_target_row_id": ["one"],
        }
    )
    with pytest.raises(MetacellUniverseError, match="wrong association method"):
        attach_query_donor_targets(universe, target, expected_outer_fold_id=0)
