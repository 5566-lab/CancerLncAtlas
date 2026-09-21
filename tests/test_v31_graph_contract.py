from __future__ import annotations

import pandas as pd

from cc_hhgt.graph_contract import (
    DeploymentContract,
    FoldGraphMode,
    assert_pseudoheldout_real_test_parity,
    build_fold_graph,
    materialize_relation_provenance,
)


def fixture_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edge_id": ["e1", "e2", "e3", "e4", "e5", "e6", "e7"],
            "source_type": ["lncRNA", "lncRNA", "gene", "lncRNA", "lncRNA", "protein", "lncRNA"],
            "target_type": ["cancer", "gene", "state", "gene", "gene", "gene", "gene"],
            "source_canonical_id": ["L1", "L1", "G1", "L2", "L3", "P1", "L4"],
            "target_canonical_id": ["BRCA", "G1", "S1", "G2", "G3", "G1", "G4"],
            "relation_type": [
                "expressed_in",
                "coexpressed_with",
                "positively_associated_with_state",
                "curated_relation",
                "coexpressed_with",
                "encoded_by",
                "coexpressed_with",
            ],
            "cancer_id": ["BRCA", "BRCA", "BRCA", "BRCA", "REF_A", None, "COAD"],
            "is_context_specific": [True, True, True, True, True, False, True],
            "source_database": ["TCGA", "TCGA", "TCGA_sample_scores", "RNAInter", "TCGA", "UniProt", "TCGA"],
            "weight": [1.0] * 7,
            "raw_effect": [None, 0.4, 0.6, None, -0.5, None, 0.3],
        }
    )


def test_provenance_is_three_way_and_state_edges_are_outcome_derived() -> None:
    result = materialize_relation_provenance(fixture_edges())
    assert set(result.relation_provenance) == {
        "global_static",
        "cancer_unlabeled_context",
        "cancer_outcome_derived",
    }
    state = result.loc[result.target_type.eq("state")]
    assert state.relation_provenance.eq("cancer_outcome_derived").all()


def test_contract_s_and_t_are_separate_and_both_drop_outcomes() -> None:
    edges = fixture_edges()
    strict = build_fold_graph(
        edges,
        heldout_cancers={"BRCA"},
        mode=FoldGraphMode.REAL_TEST,
        contract=DeploymentContract.STRICT,
        reference_only={"REF_A", "REF_B"},
    ).edges
    target = build_fold_graph(
        edges,
        heldout_cancers={"BRCA"},
        mode=FoldGraphMode.REAL_TEST,
        contract=DeploymentContract.TARGET_CONTEXT,
        reference_only={"REF_A", "REF_B"},
    ).edges
    assert set(strict.loc[strict.cancer_id.eq("BRCA"), "relation_type"]) == {"expressed_in"}
    assert set(target.loc[target.cancer_id.eq("BRCA"), "relation_type"]) == {
        "expressed_in",
        "coexpressed_with",
        "curated_relation",
    }
    assert not target.target_type.eq("state").any()
    assert not target.cancer_id.eq("REF_A").any()


def test_pseudoheldout_and_real_test_have_exact_policy_parity() -> None:
    for contract in DeploymentContract:
        audit = assert_pseudoheldout_real_test_parity(
            fixture_edges(),
            heldout_cancers={"BRCA"},
            contract=contract,
            reference_only={"REF_A", "REF_B"},
        )
        assert audit["status"] == "PASS"
        assert audit["pseudoheldout_sha256"] == audit["real_test_sha256"]
