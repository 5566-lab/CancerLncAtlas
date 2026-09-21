from __future__ import annotations

import pandas as pd
import pytest

from cc_hhgt.v31_exact_materialize import (
    build_exact_pathway_genesets,
    classify_exact_pathway_scores,
)


def _policy() -> dict:
    return {
        "observed_core_evidence_min": 0.70,
        "model_supported_probability_min": 0.80,
        "predicted_candidate_probability_min": 0.90,
        "predicted_candidate_max_observed_evidence": 0.35,
        "max_uncertainty": 0.25,
    }


def _scores() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append(
            {
                "candidate_id": f"C{index}",
                "cancer_id": "BRCA",
                "lncrna_id": f"L{index:02d}",
                "pathway_id": "MSIGDB:HALLMARK:EMT",
                "pathway_family_id": "PF:EMT",
                "calibrated_probability": 0.95 - index / 1000,
                "seed_probability_std": 0.02,
                "direction_positive_probability": 0.8,
                "observed_evidence_score": 0.1,
                "observed_direction": "",
                "gene_symbol": f"GENE{index}",
                "pathway_name": "EMT",
            }
        )
    rows[0]["observed_evidence_score"] = 0.8
    rows[0]["observed_direction"] = "negative"
    rows[-1]["calibrated_probability"] = 0.81
    rows[-1]["observed_evidence_score"] = 0.4
    return pd.DataFrame(rows)


def test_exact_pathway_classification_preserves_observed_direction() -> None:
    classified = classify_exact_pathway_scores(_scores(), _policy())
    assert classified.loc[0, "relationship_class"] == "observed_core"
    assert classified.loc[0, "final_direction"] == "negative"
    assert classified.loc[1, "relationship_class"] == "predicted_candidate"
    assert classified.loc[1, "final_direction"] == "positive"
    assert classified.loc[11, "relationship_class"] == "model_supported"


def test_exact_pathway_materializer_rejects_label_leakage() -> None:
    scores = _scores()
    scores["label_class"] = "negative"
    with pytest.raises(ValueError, match="Held-out label"):
        classify_exact_pathway_scores(scores, _policy())


def test_genesets_target_exact_pathway_not_family() -> None:
    classified = classify_exact_pathway_scores(_scores(), _policy())
    master, member = build_exact_pathway_genesets(
        classified,
        analysis_version="V3.1-test",
        min_members=5,
        max_members=10,
    )
    positive = master.loc[master.direction.eq("positive")].iloc[0]
    assert positive.geneset_type == "cancer_exact_pathway"
    assert positive.pathway_id == "MSIGDB:HALLMARK:EMT"
    assert positive.pathway_family_id == "PF:EMT"
    assert positive.pathway_target_level == "exact_pathway"
    assert positive.member_count == 10
    assert member.loc[member.geneset_id.eq(positive.geneset_id), "rank"].tolist() == list(
        range(1, 11)
    )
