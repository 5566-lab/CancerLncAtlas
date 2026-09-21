from __future__ import annotations

import pandas as pd
import pytest

from cc_hhgt.v32.genesets import (
    FROZEN_MEMBER_COLUMNS,
    GeneSetContractError,
    GeneSetMaterializationConfig,
    genesets_to_gmt_lines,
    materialize_ranked_genesets,
)


def _fold_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        for index in range(15):
            rows.append(
                {
                    "patient_fold_id": f"F{fold}",
                    "cancer_id": "BRCA",
                    "lncrna_id": f"ENSG_LNC_{index:03d}",
                    "pathway_id": "MSIGDB:HALLMARK:DNA_REPAIR",
                    "pathway_family_id": "PF:DNA_REPAIR",
                    "pathway_target_level": "exact_pathway",
                    "association_membership_probability": 0.95 - index * 0.01,
                    "association_direction": "positive",
                    "l1_probability": 0.80 - index * 0.005,
                    "ridge_probability": 0.78 - index * 0.004,
                    "graph_residual": 0.1 - index * 0.001,
                    "graph_gate": 0.4,
                    "shared_or_local_scope": "shared" if index < 12 else "cancer_local",
                    "regulatory_evidence_confidence": (index + fold) / 25.0,
                }
            )
    return pd.DataFrame(rows)


def test_ranked_geneset_materialization_has_frozen_exact_pathway_schema() -> None:
    config = GeneSetMaterializationConfig(min_members=10, max_members=12)
    master, members = materialize_ranked_genesets(_fold_predictions(), config)

    assert len(master) == 1
    assert master.iloc[0].pathway_id == "MSIGDB:HALLMARK:DNA_REPAIR"
    assert master.iloc[0].pathway_family_id == "PF:DNA_REPAIR"
    assert master.iloc[0].pathway_target_level == "exact_pathway"
    assert master.iloc[0].ranking_uses_regulatory_evidence == False  # noqa: E712
    assert master.iloc[0].member_count == 12
    assert master.iloc[0].shared_member_count == 12
    assert set(FROZEN_MEMBER_COLUMNS).issubset(members.columns)
    assert members.geneset_rank.tolist() == list(range(1, 13))
    assert members.lncrna_id.tolist() == [f"ENSG_LNC_{i:03d}" for i in range(12)]
    assert members.fold_selection_frequency.eq(1.0).all()
    assert members.pathway_target_level.eq("exact_pathway").all()
    assert genesets_to_gmt_lines(master, members)[0].split("\t")[2:] == members[
        "lncrna_id"
    ].tolist()


def test_regulatory_evidence_is_carried_but_cannot_change_member_ranking() -> None:
    original = _fold_predictions()
    changed = original.copy()
    changed["regulatory_evidence_confidence"] = (
        1.0 - changed["regulatory_evidence_confidence"]
    )
    config = GeneSetMaterializationConfig(min_members=10, max_members=12)
    _, members_a = materialize_ranked_genesets(original, config)
    _, members_b = materialize_ranked_genesets(changed, config)

    pd.testing.assert_frame_equal(
        members_a[
            [
                "cancer_id",
                "pathway_id",
                "lncrna_id",
                "geneset_rank",
                "association_membership_probability",
            ]
        ],
        members_b[
            [
                "cancer_id",
                "pathway_id",
                "lncrna_id",
                "geneset_rank",
                "association_membership_probability",
            ]
        ],
    )
    assert not members_a.regulatory_evidence_confidence.equals(
        members_b.regulatory_evidence_confidence
    )


def test_family_target_is_rejected_instead_of_broadcast_to_exact_pathway() -> None:
    invalid = _fold_predictions()
    invalid["pathway_target_level"] = "pathway_family"
    with pytest.raises(GeneSetContractError, match="exact_pathway"):
        materialize_ranked_genesets(invalid)

    invalid = _fold_predictions()
    invalid["pathway_id"] = invalid["pathway_family_id"]
    with pytest.raises(GeneSetContractError, match="family-level targets"):
        materialize_ranked_genesets(invalid)


def test_materialization_is_invariant_to_input_row_permutation() -> None:
    config = GeneSetMaterializationConfig(min_members=10, max_members=12)
    master_a, members_a = materialize_ranked_genesets(_fold_predictions(), config)
    master_b, members_b = materialize_ranked_genesets(
        _fold_predictions().sample(frac=1.0, random_state=71).reset_index(drop=True),
        config,
    )
    pd.testing.assert_frame_equal(master_a, master_b)
    pd.testing.assert_frame_equal(members_a, members_b)
