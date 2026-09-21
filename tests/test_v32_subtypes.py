from __future__ import annotations

import pandas as pd
import pytest

from cc_hhgt.v32.subtypes import (
    RankedSubtypeConfig,
    SubtypeContractError,
    classify_pathway_context_subtypes,
    infer_ranked_subtypes,
    rank_biased_overlap,
    weighted_jaccard,
)


CANCERS = ["BRCA", "COAD", "KIRC", "LUAD", "LUSC", "STAD"]


def _members(
    *,
    cancers: list[str] | None = None,
    separated: bool = False,
    pathway: str = "REACTOME:DNA_REPAIR",
) -> pd.DataFrame:
    cancers = cancers or CANCERS
    rows: list[dict[str, object]] = []
    for cancer_index, cancer in enumerate(cancers):
        block = "A" if not separated or cancer_index < 3 else "B"
        for rank in range(1, 21):
            rows.append(
                {
                    "geneset_id": f"GS:{cancer}:{pathway}",
                    "geneset_rank": rank,
                    "cancer_id": cancer,
                    "lncrna_id": f"LNC_{block}_{rank:03d}",
                    "pathway_id": pathway,
                    "pathway_family_id": "PF:DNA_REPAIR",
                    "pathway_target_level": "exact_pathway",
                    "association_membership_probability": 1.0 - rank / 100.0,
                    "association_direction": "positive",
                    "shared_or_local_scope": "shared",
                    "regulatory_evidence_confidence": cancer_index / 10.0,
                }
            )
        # Local-only members must never manufacture cross-cancer subtypes.
        rows.append(
            {
                "geneset_id": f"GS:{cancer}:{pathway}",
                "geneset_rank": 21,
                "cancer_id": cancer,
                "lncrna_id": f"LOCAL_{cancer}",
                "pathway_id": pathway,
                "pathway_family_id": "PF:DNA_REPAIR",
                "pathway_target_level": "exact_pathway",
                "association_membership_probability": 0.99,
                "association_direction": "positive",
                "shared_or_local_scope": "cancer_local",
                "regulatory_evidence_confidence": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _config() -> RankedSubtypeConfig:
    return RankedSubtypeConfig(bootstrap_replicates=30)


def test_frozen_similarity_primitives() -> None:
    assert rank_biased_overlap(["A", "B", "C"], ["A", "B", "C"], p=0.98) == pytest.approx(
        1.0
    )
    assert rank_biased_overlap(["A", "B"], ["C", "D"], p=0.98) == 0.0
    assert weighted_jaccard({"A": 1.0, "B": 0.5}, {"A": 0.5, "C": 1.0}) == pytest.approx(
        0.5 / 2.5
    )
    defaults = RankedSubtypeConfig()
    assert defaults.top_n == 200
    assert defaults.rbo_p == 0.98
    assert defaults.k_min == 1 and defaults.k_max == 4
    assert defaults.program_k_max == 6
    assert defaults.silhouette_threshold == 0.25
    assert defaults.bootstrap_ari_threshold == 0.75
    assert defaults.bootstrap_replicates == 200
    assert defaults.min_cancers == 6 and defaults.min_members == 10


def test_conserved_pathway_returns_k1() -> None:
    result = infer_ranked_subtypes(_members(), _config())
    context = result.pathway_context_subtype
    conservation = result.pathway_conservation.iloc[0]

    assert len(context) == 6
    assert context.selected_k.eq(1).all()
    assert context.classification_status.eq("CONSERVED_K1").all()
    assert context.pathway_context_subtype_id.nunique() == 1
    assert conservation.selected_k == 1
    assert conservation.mean_pairwise_similarity == pytest.approx(1.0)
    assert result.cancer_program_subtype.selected_k.eq(1).all()


def test_two_gene_set_composition_groups_return_stable_k2_at_both_levels() -> None:
    result = infer_ranked_subtypes(_members(separated=True), _config())
    context = result.pathway_context_subtype.set_index("cancer_id")
    conservation = result.pathway_conservation.iloc[0]

    assert conservation.selected_k == 2
    assert conservation.silhouette >= 0.25
    assert conservation.bootstrap_ari >= 0.75
    assert context.loc[CANCERS[:3], "pathway_context_subtype_id"].nunique() == 1
    assert context.loc[CANCERS[3:], "pathway_context_subtype_id"].nunique() == 1
    assert (
        context.loc[CANCERS[0], "pathway_context_subtype_id"]
        != context.loc[CANCERS[3], "pathway_context_subtype_id"]
    )

    program = result.cancer_program_subtype.set_index("cancer_id")
    assert program.selected_k.eq(2).all()
    assert program.loc[CANCERS[:3], "cancer_program_subtype_id"].nunique() == 1
    assert program.loc[CANCERS[3:], "cancer_program_subtype_id"].nunique() == 1


def test_low_cancer_coverage_is_explicitly_unavailable() -> None:
    result = infer_ranked_subtypes(_members(cancers=CANCERS[:5]), _config())
    assert result.pathway_context_subtype.classification_status.eq("UNAVAILABLE").all()
    assert result.pathway_context_subtype.selected_k.eq(0).all()
    assert result.pathway_conservation.iloc[0].classification_status == "UNAVAILABLE"
    assert result.cancer_program_subtype.classification_status.eq("UNAVAILABLE").all()


def test_local_only_cancer_pathway_is_retained_as_unavailable_not_zero() -> None:
    members = _members()
    local_only = members.iloc[[0]].copy()
    local_only["cancer_id"] = "UVM"
    local_only["lncrna_id"] = "LOCAL_UVM_ONLY"
    local_only["shared_or_local_scope"] = "cancer_local"
    result = infer_ranked_subtypes(pd.concat([members, local_only], ignore_index=True), _config())
    uvm = result.pathway_context_subtype.loc[
        result.pathway_context_subtype.cancer_id.eq("UVM")
    ].iloc[0]
    assert uvm.n_shared_members == 0
    assert uvm.pathway_context_subtype_id == "UNAVAILABLE"
    assert uvm.selected_k == 0


def test_subtype_results_are_deterministic_under_row_permutation_and_evidence_change() -> None:
    original = _members(separated=True)
    permuted = original.sample(frac=1.0, random_state=90210).reset_index(drop=True)
    permuted["regulatory_evidence_confidence"] = 1.0 - permuted[
        "regulatory_evidence_confidence"
    ]
    result_a = infer_ranked_subtypes(original, _config())
    result_b = infer_ranked_subtypes(permuted, _config())

    pd.testing.assert_frame_equal(
        result_a.pathway_context_subtype, result_b.pathway_context_subtype
    )
    pd.testing.assert_frame_equal(
        result_a.pathway_conservation, result_b.pathway_conservation
    )
    pd.testing.assert_frame_equal(
        result_a.pathway_pairwise_similarity,
        result_b.pathway_pairwise_similarity,
    )
    pd.testing.assert_frame_equal(
        result_a.cancer_program_subtype, result_b.cancer_program_subtype
    )


def test_pathway_sharding_matches_monolithic_context_results() -> None:
    members = pd.concat(
        [
            _members(pathway="REACTOME:DNA_REPAIR"),
            _members(separated=True, pathway="HALLMARK:EMT"),
        ],
        ignore_index=True,
    )
    config = RankedSubtypeConfig(bootstrap_replicates=8)
    expected = classify_pathway_context_subtypes(members, config)
    shards = [
        classify_pathway_context_subtypes(group.copy(), config)
        for _, group in members.groupby("pathway_id", observed=True, sort=True)
    ]
    sort_keys = (
        ["pathway_id", "cancer_id"],
        ["pathway_id"],
        ["pathway_id", "cancer_a", "cancer_b"],
    )
    for expected_frame, position, keys in zip(expected, range(3), sort_keys):
        observed = (
            pd.concat([item[position] for item in shards], ignore_index=True)
            .sort_values(keys, kind="stable")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(expected_frame, observed)


def test_family_target_is_rejected_for_subtype_classification() -> None:
    invalid = _members()
    invalid["pathway_target_level"] = "pathway_family"
    with pytest.raises(SubtypeContractError, match="exact_pathway"):
        infer_ranked_subtypes(invalid, _config())
