from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.interaction_query import (
    InteractionQueryAssetError,
    InteractionQueryInputError,
    InteractionReleaseQuery,
)
from cc_hhgt.v32.interaction_release import (
    FORMAL_CANDIDATE_SHA256,
    FORMAL_MEMBERSHIP_SHA256,
    materialize_interaction_release,
)
from cc_hhgt.v32.release_registry import artifact_sha256


def _release(tmp_path: Path) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    facts = pd.DataFrame(
        [
            {
                "physical_fact_id": "F1",
                "cancer_id": None,
                "lncrna_id": "ENSG00000000001",
                "partner_id": "ENSG00000000101",
                "source_database": "DB",
                "source_record_id": "R1",
                "pmid": "1",
                "is_prediction": False,
            },
            {
                "physical_fact_id": "F2",
                "cancer_id": "BRCA",
                "lncrna_id": "ENSG00000000001",
                "partner_id": "ENSG00000000102",
                "source_database": "DB",
                "source_record_id": "R2",
                "pmid": "2",
                "is_prediction": False,
            },
            {
                "physical_fact_id": "F3",
                "cancer_id": None,
                "lncrna_id": "ENSG00000000001",
                "partner_id": "ENSG00000999999",
                "source_database": "DB",
                "source_record_id": "R3",
                "pmid": "3",
                "is_prediction": False,
            },
        ]
    )
    membership = pd.DataFrame(
        [("P1", "ENSG00000000101"), ("P2", "ENSG00000000102")],
        columns=["pathway_id", "gene_id"],
    )
    candidates = pd.DataFrame(
        [("BRCA", "ENSG00000000001", "P1"), ("BRCA", "ENSG00000000001", "P2")],
        columns=["cancer_id", "lncrna_id", "pathway_id"],
    )
    for frame, name in (
        (facts, "physical_facts.parquet"),
        (membership, "membership.parquet"),
        (candidates, "candidates.parquet"),
    ):
        frame.to_parquet(tmp_path / name, index=False)
    output = tmp_path / "release"
    materialize_interaction_release(
        physical_facts_path=tmp_path / "physical_facts.parquet",
        membership_path=tmp_path / "membership.parquet",
        candidates_path=tmp_path / "candidates.parquet",
        output_root=output,
        strict_formal_authority=False,
    )
    manifest = output / "INTERACTION_RELEASE_MANIFEST.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["formal_authority_enforced"] = True
    payload["fresh_evidence_output_binding"] = {
        "path": "/formal/EVIDENCE_OUTPUT_BINDING.json",
        "sha256": "1" * 64,
        "success_path": "/formal/SUCCESS.json",
        "success_sha256": "2" * 64,
        "evidence_training_run_id": "V32-EVIDENCE-TRAIN-TEST",
        "physical_facts_sha256": "3" * 64,
        "five_fresh_private_heads_verified": True,
    }
    payload["fresh_evidence_semantic_wrapper"] = {
        "path": "/formal/EVIDENCE_SEMANTIC_WRAPPER.json",
        "sha256": "4" * 64,
        "success_path": "/formal/SEMANTIC_SUCCESS.json",
        "success_sha256": "5" * 64,
        "independent_post_audit_path": "/formal/EVIDENCE_INDEPENDENT_POST_AUDIT.json",
        "independent_post_audit_sha256": "6" * 64,
        "evidence_training_run_id": "V32-EVIDENCE-TRAIN-TEST",
        "confidence_only": True,
        "affects_discovery": False,
        "affects_primary_ranking": False,
        "direct_exact_pathway_assertion": False,
    }
    payload["inputs"]["exact_membership"]["sha256"] = FORMAL_MEMBERSHIP_SHA256
    payload["inputs"]["exact_candidates"]["sha256"] = FORMAL_CANDIDATE_SHA256
    payload["inputs"]["evidence_output_binding"] = {
        "path": "/formal/EVIDENCE_OUTPUT_BINDING.json",
        "sha256": "1" * 64,
    }
    payload["inputs"]["evidence_semantic_wrapper"] = {
        "path": "/formal/EVIDENCE_SEMANTIC_WRAPPER.json",
        "sha256": "4" * 64,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, artifact_sha256(manifest)


def test_queries_relationship_enrichment_and_evidence(tmp_path: Path) -> None:
    manifest, digest = _release(tmp_path)
    query = InteractionReleaseQuery(manifest, expected_manifest_sha256=digest)
    relationships = query.query_relationships(lncrna_id="ENSG00000000001")
    assert relationships["returned_rows"] == 3
    assert relationships["provenance"]["global_facts_broadcast_to_cancers"] is False
    global_rows = query.query_relationships(
        lncrna_id="LNC:ENSG00000000001", cancer_scope="GLOBAL"
    )
    assert global_rows["returned_rows"] == 2
    enrichment = query.query_enrichment(
        lncrna_id="ENSG00000000001", max_fdr=1.0
    )
    assert enrichment["returned_rows"] == 2
    relationship_id = relationships["rows"][0]["relationship_id"]
    evidence = query.query_evidence(relationship_id=relationship_id)
    assert evidence["returned_rows"] == 1
    unmapped = query.query_unmapped_partners(
        lncrna_id="ENSG00000000001", partner_gene_id="GENE:ENSG00000999999"
    )
    assert unmapped["returned_rows"] == 1
    assert unmapped["exact_pathway_negative_claimed"] is False
    assert unmapped["affects_primary_ranking"] is False


def test_query_requires_pinned_manifest_and_rejects_hash_drift(tmp_path: Path) -> None:
    manifest, digest = _release(tmp_path)
    with pytest.raises(InteractionQueryAssetError, match="requires"):
        InteractionReleaseQuery(manifest, expected_manifest_sha256=None)
    with pytest.raises(InteractionQueryAssetError, match="mismatch"):
        InteractionReleaseQuery(manifest, expected_manifest_sha256="0" * 64)
    assert digest != "0" * 64


def test_query_requires_audited_evidence_semantic_wrapper(tmp_path: Path) -> None:
    manifest, _ = _release(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("fresh_evidence_semantic_wrapper")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InteractionQueryAssetError, match="semantic wrapper"):
        InteractionReleaseQuery(
            manifest, expected_manifest_sha256=artifact_sha256(manifest)
        )


def test_query_rejects_artifact_drift_and_manifest_path_escape(tmp_path: Path) -> None:
    manifest, _ = _release(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    relationship = manifest.parent / "physical_interaction_relationships.parquet"
    frame = pd.read_parquet(relationship)
    frame.loc[0, "physical_fact_count"] = 999
    frame.to_parquet(relationship, index=False)
    with pytest.raises(InteractionQueryAssetError, match="SHA drift"):
        InteractionReleaseQuery(
            manifest, expected_manifest_sha256=artifact_sha256(manifest)
        )

    manifest, _ = _release(tmp_path / "second")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    declaration = payload["artifacts"]["physical_interaction_relationships.parquet"]
    declaration["path"] = "../physical_interaction_relationships.parquet"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InteractionQueryAssetError, match="escaped"):
        InteractionReleaseQuery(
            manifest, expected_manifest_sha256=artifact_sha256(manifest)
        )


def test_query_input_bounds_fail_closed(tmp_path: Path) -> None:
    manifest, digest = _release(tmp_path)
    query = InteractionReleaseQuery(manifest, expected_manifest_sha256=digest)
    with pytest.raises(InteractionQueryInputError, match="lncrna_id"):
        query.query_relationships(lncrna_id="")
    with pytest.raises(InteractionQueryInputError, match="limit"):
        query.query_enrichment(lncrna_id="ENSG1", limit=0)
    with pytest.raises(InteractionQueryInputError, match="max_fdr"):
        query.query_enrichment(lncrna_id="ENSG1", max_fdr=1.1)
