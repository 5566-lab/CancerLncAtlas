from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.interaction_release import (
    GLOBAL_SCOPE,
    InteractionReleaseError,
    build_interaction_pathway_enrichment,
    build_physical_relationships,
    materialize_interaction_release,
    validate_current_exact_authority,
)


def _facts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "physical_fact_id": "F1",
                "cancer_id": None,
                "lncrna_id": "LNC:ENSG00000000001",
                "partner_id": "GENE:ENSG00000000101.3",
                "relation_type": "physical_binding",
                "experiment_type": "RIP",
                "source_database": "DB1",
                "source_dataset": "D1",
                "source_record_id": "R1",
                "pmid": "1",
                "source_row_sha256": "a" * 64,
                "source_occurrence_count": 2,
                "is_prediction": False,
            },
            {
                "physical_fact_id": "F2",
                "cancer_id": "BRCA",
                "lncrna_id": "ENSG00000000001.2",
                "partner_id": "ENSG00000000102",
                "relation_type": "physical_binding",
                "experiment_type": "CLIP",
                "source_database": "DB2",
                "source_dataset": "D2",
                "source_record_id": "R2",
                "pmid": "2",
                "source_row_sha256": "b" * 64,
                "source_occurrence_count": 1,
                "is_prediction": False,
            },
            {
                "physical_fact_id": "F3",
                "cancer_id": "GLOBAL",
                "lncrna_id": "LNC:ENSG00000000001",
                "partner_id": "GENE:ENSG00000000999",
                "relation_type": "physical_binding",
                "experiment_type": "RIP",
                "source_database": "DB1",
                "source_dataset": "D1",
                "source_record_id": "R3",
                "pmid": "3",
                "source_row_sha256": "c" * 64,
                "source_occurrence_count": 1,
                "is_prediction": False,
            },
        ]
    )


def _membership() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("P1", "GENE:ENSG00000000101"),
            ("P1", "GENE:ENSG00000000103"),
            ("P2", "ENSG00000000102"),
            ("P2", "ENSG00000000103"),
            ("P3", "ENSG00000000104"),
        ],
        columns=["pathway_id", "gene_id"],
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("BRCA", "LNC:ENSG00000000001", "P1"),
            ("BRCA", "LNC:ENSG00000000001", "P2"),
            ("BRCA", "LNC:ENSG00000000001", "P3"),
        ],
        columns=["cancer_id", "lncrna_id", "pathway_id"],
    )


def test_relationships_keep_global_scope_and_drilldown() -> None:
    relationships, evidence, rejected = build_physical_relationships(_facts())
    assert rejected.empty
    assert set(relationships.cancer_scope) == {GLOBAL_SCOPE, "BRCA"}
    assert len(relationships) == 3
    assert len(evidence) == 3
    assert not evidence.is_prediction.any()
    assert evidence.relationship_id.isin(relationships.relationship_id).all()


def test_predicted_interaction_is_rejected() -> None:
    facts = _facts()
    facts.loc[0, "is_prediction"] = True
    with pytest.raises(InteractionReleaseError, match="Predicted interactions"):
        build_physical_relationships(facts)


def test_exact_authority_filters_membership_and_rejects_duplicates() -> None:
    member, pathways, audit = validate_current_exact_authority(
        _membership(), _candidates(), expected_pathways=3, expected_candidates=3, expected_cancers=1
    )
    assert pathways == {"P1", "P2", "P3"}
    assert len(member) == 5
    assert audit["candidate_lncrnas"] == 1
    duplicated = pd.concat([_candidates(), _candidates().iloc[[0]]], ignore_index=True)
    with pytest.raises(InteractionReleaseError, match="duplicate"):
        validate_current_exact_authority(
            _membership(), duplicated, expected_pathways=3
        )


def test_enrichment_uses_all_pathways_for_bh_and_does_not_broadcast() -> None:
    relationships, _, _ = build_physical_relationships(_facts())
    member, _, _ = validate_current_exact_authority(
        _membership(), _candidates(), expected_pathways=3
    )
    enrichment, unmapped = build_interaction_pathway_enrichment(
        relationships, member, total_pathways=3
    )
    assert set(enrichment.cancer_scope) == {GLOBAL_SCOPE, "BRCA"}
    assert set(enrichment.pathway_id) == {"P1", "P2"}
    assert not enrichment.family_to_exact_broadcast.any()
    assert enrichment.ora_p_value.between(0, 1).all()
    assert enrichment.ora_fdr.between(0, 1).all()
    assert np.isfinite(enrichment.fold_enrichment).all()
    assert set(unmapped.partner_gene_id) == {"ENSG00000000999"}


def test_materialization_writes_immutable_hashed_release(tmp_path: Path) -> None:
    facts = tmp_path / "physical_facts.parquet"
    membership = tmp_path / "exact_membership.parquet"
    candidates = tmp_path / "exact_candidates.parquet"
    _facts().to_parquet(facts, index=False)
    _membership().to_parquet(membership, index=False)
    _candidates().to_parquet(candidates, index=False)
    output = tmp_path / "release"
    manifest = materialize_interaction_release(
        physical_facts_path=facts,
        membership_path=membership,
        candidates_path=candidates,
        output_root=output,
        strict_formal_authority=False,
    )
    assert manifest["status"] == "SUCCESS_NEWLY_MATERIALIZED_V32"
    assert manifest["global_facts_broadcast_to_cancers"] is False
    assert manifest["counts"]["interaction_exact_pathway_rows"] > 0
    success = json.loads((output / "SUCCESS.json").read_text(encoding="utf-8"))
    assert success["release_ready"] is False
    with pytest.raises(InteractionReleaseError, match="output reuse"):
        materialize_interaction_release(
            physical_facts_path=facts,
            membership_path=membership,
            candidates_path=candidates,
            output_root=output,
            strict_formal_authority=False,
        )


def test_forbidden_historical_support_input_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "interaction_pathway_support.parquet"
    _facts().to_parquet(bad, index=False)
    membership = tmp_path / "membership.parquet"
    candidates = tmp_path / "candidates.parquet"
    _membership().to_parquet(membership, index=False)
    _candidates().to_parquet(candidates, index=False)
    with pytest.raises(InteractionReleaseError, match="forbidden"):
        materialize_interaction_release(
            physical_facts_path=bad,
            membership_path=membership,
            candidates_path=candidates,
            output_root=tmp_path / "release",
            strict_formal_authority=False,
        )
