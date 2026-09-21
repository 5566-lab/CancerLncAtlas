from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.network_query import (
    NetworkQueryAssetError,
    NetworkQueryInputError,
    UnifiedNetworkQuery,
)
from cc_hhgt.v32.network_release import (
    ANALYSIS_VERSION,
    EDGE_TYPES,
    FORMAL_CANDIDATE_SHA256,
    FORMAL_EXACT_LINEAGE_SHA256,
    FORMAL_EXPERIMENT_AUDIT_SHA256,
    FORMAL_EXPERIMENT_BRIDGE_SHA256,
    FORMAL_FUSION_AUDIT_SHA256,
    FORMAL_FUSION_BINDING_SHA256,
    FORMAL_FUSION_SCORES_SHA256,
    FORMAL_MEMBERSHIP_SHA256,
    FORMAL_PHYSICAL_AUDIT_SHA256,
    FORMAL_PHYSICAL_MANIFEST_SHA256,
    FORMAL_PHYSICAL_RELATIONSHIPS_SHA256,
    FORMAL_PRIMARY_SHA256,
    MANIFEST_FILE,
    MEMBERSHIP_EDGE_TYPE,
    MODEL_EDGE_TYPE,
    NETWORK_EDGE_FILE,
    NETWORK_NODE_FILE,
    NODE_TYPES,
    PHYSICAL_EDGE_TYPE,
    RELEASE_FORMAT,
    RELEASE_STATUS,
    SUCCESS_FILE,
    NetworkReleaseError,
    materialize_network_tables,
)


def _write(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_parquet(path, index=False)
    return path


def _fixture_release(tmp_path: Path) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    keys = [
        ("BRCA", "LNC:ENSG00000000001", "P1"),
        ("BRCA", "LNC:ENSG00000000001", "P2"),
        ("LUAD", "LNC:ENSG00000000002", "P1"),
    ]
    candidate = pd.DataFrame(keys, columns=["cancer_id", "lncrna_id", "pathway_id"])
    primary = candidate.copy()
    primary["association_membership_probability"] = [0.8, 0.3, 0.6]
    primary["analysis_version"] = ANALYSIS_VERSION
    # The exact table itself defines the primary ranking; the network
    # projection below still declares changes_primary_ranking=False.
    primary["changes_primary_ranking"] = True
    fusion = candidate.copy()
    fusion["primary_probability"] = [0.8, 0.3, 0.6]
    fusion["discovery_adjusted_probability"] = [0.81, 0.3, 0.6]
    fusion["fused_confidence_probability"] = [0.82, 0.3, 0.6]
    fusion["genomic_native_available"] = [True, False, False]
    fusion["genomic_native_probability"] = [0.7, None, None]
    fusion["single_cell_native_available"] = [False, False, True]
    fusion["single_cell_native_probability"] = [None, None, 0.65]
    fusion["evidence_transformer_native_available"] = [True, False, False]
    fusion["evidence_transformer_native_probability"] = [0.9, None, None]
    fusion["analysis_version"] = ANALYSIS_VERSION
    fusion["primary_ranking_unchanged"] = True
    fusion["adjusted_ranking_is_secondary"] = True
    fusion["used_for_primary_release"] = False
    fusion["historical_predictions_used"] = False
    fusion["historical_rankings_used"] = False
    membership = pd.DataFrame(
        [("P1", "ENSG00000000101"), ("P1", "ENSG00000000102.4"),
         ("P2", "GENE:ENSG00000000102"), ("OLD", "ENSG00000999999")],
        columns=["pathway_id", "gene_id"],
    )
    physical = pd.DataFrame(
        [
            {
                "relationship_id": "REL1",
                "cancer_scope": "GLOBAL_NOT_CANCER_SPECIFIC",
                "lncrna_id": "LNC:ENSG00000000001",
                "partner_gene_id": "ENSG00000000101",
                "physical_fact_count": 2,
                "source_occurrence_count": 2,
                "independent_pmid_count": 1,
                "independent_source_database_count": 1,
                "independent_source_record_count": 2,
                "availability": True,
                "is_prediction": False,
                "analysis_version": ANALYSIS_VERSION,
            },
            {
                "relationship_id": "REL2",
                "cancer_scope": "GLOBAL_NOT_CANCER_SPECIFIC",
                "lncrna_id": "LNC:ENSG00000000003",
                "partner_gene_id": "ENSG00000000103",
                "physical_fact_count": 1,
                "source_occurrence_count": 1,
                "independent_pmid_count": 0,
                "independent_source_database_count": 1,
                "independent_source_record_count": 1,
                "availability": True,
                "is_prediction": False,
                "analysis_version": ANALYSIS_VERSION,
            },
        ]
    )
    experiment = pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "lncrna_id": "LNC:ENSG00000000001",
                "pathway_id": "P1",
                "experiment_event_count": 4,
                "source_record_count": 3,
                "absorption_status": "FULLY_ABSORBED",
                "confidence_route": "EXISTING_EVIDENCE_TRANSFORMER_ONLY",
                "separate_fusion_forbidden": True,
                "used_for_separate_fusion": False,
                "changes_primary_ranking": False,
                "changes_discovery_ranking": False,
            }
        ]
    )
    paths = {
        "primary": _write(primary, tmp_path / "primary.parquet"),
        "fusion": _write(fusion, tmp_path / "fusion.parquet"),
        "membership": _write(membership, tmp_path / "membership.parquet"),
        "candidate": _write(candidate, tmp_path / "candidate.parquet"),
        "physical": _write(physical, tmp_path / "physical.parquet"),
        "experiment": _write(experiment, tmp_path / "experiment.parquet"),
    }
    release = tmp_path / "release"
    result = materialize_network_tables(
        primary_path=paths["primary"],
        fusion_scores_path=paths["fusion"],
        membership_path=paths["membership"],
        candidate_path=paths["candidate"],
        physical_relationships_path=paths["physical"],
        experiment_exact_query_path=paths["experiment"],
        output_root=release,
        expected_candidate_rows=None,
        expected_cancers=None,
        expected_pathways=None,
        expected_membership_rows=None,
        expected_physical_rows=None,
        memory_limit="1GB",
    )
    assert result["node_type_rows"] == {
        "lncRNA": 3,
        "exact_pathway": 2,
        "protein_gene": 3,
    }
    assert result["edge_type_rows"] == {
        MODEL_EDGE_TYPE: 3,
        MEMBERSHIP_EDGE_TYPE: 3,
        PHYSICAL_EDGE_TYPE: 2,
    }
    input_hashes = {
        "primary": FORMAL_PRIMARY_SHA256,
        "exact_lineage": FORMAL_EXACT_LINEAGE_SHA256,
        "fusion_scores": FORMAL_FUSION_SCORES_SHA256,
        "fusion_binding": FORMAL_FUSION_BINDING_SHA256,
        "fusion_audit": FORMAL_FUSION_AUDIT_SHA256,
        "membership": FORMAL_MEMBERSHIP_SHA256,
        "candidate": FORMAL_CANDIDATE_SHA256,
        "physical_relationships": FORMAL_PHYSICAL_RELATIONSHIPS_SHA256,
        "physical_manifest": FORMAL_PHYSICAL_MANIFEST_SHA256,
        "physical_audit": FORMAL_PHYSICAL_AUDIT_SHA256,
        "experiment_bridge": FORMAL_EXPERIMENT_BRIDGE_SHA256,
        "experiment_audit": FORMAL_EXPERIMENT_AUDIT_SHA256,
    }
    manifest = {
        "format": RELEASE_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "unified_network",
        "status": RELEASE_STATUS,
        "release_ready": False,
        "production_deployed": False,
        "formal_authority_enforced": True,
        "source_generation": "CURRENT_V3.2_ONLY",
        "primary_frozen": True,
        "primary_ranking_unchanged": True,
        "secondary_scores_remain_secondary": True,
        "native_expert_probabilities_public": True,
        "native_missingness_encoding": "availability_boolean_plus_nullable_probability",
        "missing_native_probability_imputed_to_zero": False,
        "family_to_exact_broadcast": False,
        "global_physical_facts_broadcast_to_cancers": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "experiment_route": "SUPPORT_ATTRIBUTE_ALREADY_ABSORBED_IN_EVIDENCE_TRANSFORMER",
        "experiment_separate_fusion_created": False,
        "experiment_changes_primary_ranking": False,
        "edge_types": list(EDGE_TYPES),
        "node_types": list(NODE_TYPES),
        "counts": {
            "nodes": result["nodes_rows"],
            "edges": result["edges_rows"],
            "node_type_rows": result["node_type_rows"],
            "edge_type_rows": result["edge_type_rows"],
        },
        "inputs": {role: {"path": f"/formal/{role}", "sha256": digest}
                   for role, digest in input_hashes.items()},
        "experiment_query_link": {
            "path": str(paths["experiment"]),
            "sha256": artifact_sha256(paths["experiment"]),
            "bridge_path": "/formal/bridge.json",
            "bridge_sha256": FORMAL_EXPERIMENT_BRIDGE_SHA256,
            "separate_probability_head": False,
            "double_counted_in_network_score": False,
        },
        "artifacts": {
            NETWORK_NODE_FILE: {
                "path": NETWORK_NODE_FILE,
                "rows": result["nodes_rows"],
                "sha256": artifact_sha256(release / NETWORK_NODE_FILE),
            },
            NETWORK_EDGE_FILE: {
                "path": NETWORK_EDGE_FILE,
                "rows": result["edges_rows"],
                "sha256": artifact_sha256(release / NETWORK_EDGE_FILE),
            },
        },
    }
    manifest_path = release / MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = artifact_sha256(manifest_path)
    success = {
        "status": RELEASE_STATUS,
        "analysis_version": ANALYSIS_VERSION,
        "release_ready": False,
        "production_deployed": False,
        "manifest": MANIFEST_FILE,
        "manifest_sha256": digest,
    }
    (release / SUCCESS_FILE).write_text(json.dumps(success), encoding="utf-8")
    return manifest_path, digest


def test_network_materialization_preserves_edge_semantics(tmp_path: Path) -> None:
    manifest, digest = _fixture_release(tmp_path)
    query = UnifiedNetworkQuery(manifest, expected_manifest_sha256=digest)
    nodes = query.query_nodes(node_type="exact_pathway")
    assert nodes["returned_rows"] == 2
    assert {row["entity_id"] for row in nodes["rows"]} == {"P1", "P2"}
    associations = query.query_model_associations(
        lncrna_id="ENSG00000000001", cancer_id="brca"
    )
    assert associations["returned_rows"] == 2
    p1 = next(row for row in associations["rows"] if row["pathway_id"] == "P1")
    assert p1["primary_probability"] == pytest.approx(0.8)
    assert p1["single_cell_native_available"] is False
    assert p1["single_cell_native_probability"] is None
    assert p1["experiment_support_available"] is True
    assert p1["experiment_event_count"] == 4
    p2 = next(row for row in associations["rows"] if row["pathway_id"] == "P2")
    assert p2["genomic_native_available"] is False
    assert p2["genomic_native_probability"] is None
    assert p2["experiment_support_available"] is False
    supported = query.query_experiment_supported_associations(
        lncrna_id="LNC:ENSG00000000001"
    )
    assert supported["returned_rows"] == 1
    assert supported["separate_experiment_probability_head"] is False
    assert supported["double_counted_in_score"] is False


def test_network_neighborhood_has_membership_and_physical_edges(tmp_path: Path) -> None:
    manifest, digest = _fixture_release(tmp_path)
    query = UnifiedNetworkQuery(manifest, expected_manifest_sha256=digest)
    pathway = query.query_neighborhood(
        node_id="P1", node_type="exact_pathway", edge_type=MEMBERSHIP_EDGE_TYPE
    )
    assert pathway["returned_rows"] == 2
    gene = query.query_neighborhood(node_id="GENE:ENSG00000000101")
    assert {row["edge_type"] for row in gene["rows"]} == {
        MEMBERSHIP_EDGE_TYPE,
        PHYSICAL_EDGE_TYPE,
    }


def test_network_query_requires_hash_pin_and_rejects_drift(tmp_path: Path) -> None:
    manifest, digest = _fixture_release(tmp_path)
    with pytest.raises(NetworkQueryAssetError, match="requires"):
        UnifiedNetworkQuery(manifest, expected_manifest_sha256=None)
    with pytest.raises(NetworkQueryAssetError, match="mismatch"):
        UnifiedNetworkQuery(manifest, expected_manifest_sha256="0" * 64)
    edge_path = manifest.parent / NETWORK_EDGE_FILE
    frame = pd.read_parquet(edge_path)
    frame.loc[0, "primary_probability"] = 0.01
    frame.to_parquet(edge_path, index=False)
    with pytest.raises(NetworkQueryAssetError, match="SHA256 drift"):
        UnifiedNetworkQuery(manifest, expected_manifest_sha256=digest)


def test_network_query_rejects_unpinned_authority_and_bad_input(tmp_path: Path) -> None:
    manifest, _ = _fixture_release(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inputs"]["fusion_audit"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NetworkQueryAssetError, match="fusion_audit"):
        UnifiedNetworkQuery(
            manifest, expected_manifest_sha256=artifact_sha256(manifest)
        )
    manifest, digest = _fixture_release(tmp_path / "second")
    query = UnifiedNetworkQuery(manifest, expected_manifest_sha256=digest)
    with pytest.raises(NetworkQueryInputError, match="node_id"):
        query.query_neighborhood(node_id="P1")
    with pytest.raises(NetworkQueryInputError, match="edge_type"):
        query.query_neighborhood(node_id="PATHWAY:P1", edge_type="OLD_NETWORK")
    with pytest.raises(NetworkQueryInputError, match="limit"):
        query.query_model_associations(lncrna_id="ENSG1", limit=0)


def test_network_materializer_rejects_missing_to_zero(tmp_path: Path) -> None:
    # Build the fixture inputs, then corrupt one unavailable native probability.
    manifest, _ = _fixture_release(tmp_path)
    fusion_path = tmp_path / "fusion.parquet"
    fusion = pd.read_parquet(fusion_path)
    fusion.loc[~fusion["genomic_native_available"], "genomic_native_probability"] = 0.0
    fusion.to_parquet(fusion_path, index=False)
    with pytest.raises(NetworkReleaseError, match="closure/semantic"):
        materialize_network_tables(
            primary_path=tmp_path / "primary.parquet",
            fusion_scores_path=fusion_path,
            membership_path=tmp_path / "membership.parquet",
            candidate_path=tmp_path / "candidate.parquet",
            physical_relationships_path=tmp_path / "physical.parquet",
            experiment_exact_query_path=tmp_path / "experiment.parquet",
            output_root=tmp_path / "corrupt_release",
            expected_candidate_rows=None,
            expected_cancers=None,
            expected_pathways=None,
            expected_membership_rows=None,
            expected_physical_rows=None,
            memory_limit="1GB",
        )
