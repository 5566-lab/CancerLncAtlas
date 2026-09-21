from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cc_hhgt.v32.patient_fold_authority import (
    FROZEN_V32_RECEIPT_SHA256,
    FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
    FORMAL_PREPARED_BINDING_FORMAT,
    validate_frozen_v32_patient_fold_binding,
)
from cc_hhgt.v32.formal_graph_authority import (
    FOLD_ARTIFACT_IDS,
    GRAPH_PAYLOAD_BINDING_FORMAT,
    GRAPH_PAYLOAD_BINDING_STATUS,
    STATIC_ARTIFACT_IDS,
)
from cc_hhgt.v32.training import PREPARED_FORMAT, _validate_prepared
from cc_hhgt.v32.contracts import dataframe_sha256
from cc_hhgt.v32.safe_graph import EDGE_KEYS


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "artifacts/v32_patient_fold_authority_20260829_r1"


def _binding() -> dict:
    audit = validate_frozen_v32_patient_fold_binding(
        AUTHORITY / "SAMPLE_PATIENT_FOLD_MAP.tsv",
        AUTHORITY / "PATIENT_FOLD_AUTHORITY_RECEIPT.json",
    )
    return {
        "format": FORMAL_PREPARED_BINDING_FORMAT,
        "status": "PASS_FROZEN_V32_PATIENT_FIRST_AUTHORITY",
        "authority": audit,
        "sample_patient_fold_map": {"sha256": audit["manifest_sha256"]},
        "authority_receipt": {"sha256": audit["receipt_sha256"]},
        "sample_id_patient_fallback_used": False,
        "legacy_patient_fold_manifest_used": False,
    }


def _payload() -> dict:
    bundle = _graph_bundle()
    batch = {
        "candidate_batch": {},
        "base_logit": object(),
        "conservation_context": object(),
        "proxy_label": object(),
        "weak_positive": object(),
        "direction_label": object(),
        "direction_available": object(),
    }
    return {
        "prepared_format": PREPARED_FORMAT,
        "patient_fold": 0,
        "bundle": bundle,
        "feature_dim": 2,
        "legacy_model_config": {},
        "train_batches": [batch],
        "validation_batches": [batch],
        "patient_fold_authority": _binding(),
        "formal_graph_variant": "G2",
        "formal_graph_authority": _graph_binding(bundle),
        "label_contract": {
            "train_direction_label_source": "train_discovery_effect",
            "validation_direction_label_source": "validation_replication_effect",
            "heldout_direction_is_model_input": False,
            "test_labels_in_training_payload": False,
            "test_logits_in_training_payload": False,
            "test_metrics_computed_before_winner_lock": False,
        },
    }


def _graph_bundle():
    nodes = pd.DataFrame({"node_type": ["lncRNA", "pathway", "cancer"], "canonical_id": ["L1", "P1", "BRCA"]})
    edges = pd.DataFrame({
        "source_type": ["lncRNA"], "source_id": ["L1"],
        "relation_type": ["expressed_in"], "target_type": ["cancer"],
        "target_id": ["BRCA"],
    })
    return SimpleNamespace(nodes=nodes, edges=edges)


def _graph_binding(bundle) -> dict:
    node_sha = dataframe_sha256(bundle.nodes, ["node_type", "canonical_id"])
    edge_sha = dataframe_sha256(bundle.edges, list(EDGE_KEYS))
    return {
        "format": GRAPH_PAYLOAD_BINDING_FORMAT,
        "status": GRAPH_PAYLOAD_BINDING_STATUS,
        "outer_fold": 0,
        "variant": "G2",
        "receipt": {"sha256": "a" * 64},
        "patient_fold_authority": {
            "manifest_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
            "receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        },
        "static_artifacts": {
            key: {"path": f"/authority/{key}.parquet", "sha256": "d" * 64}
            for key in STATIC_ARTIFACT_IDS
        },
        "fold_artifacts": {
            key: {"path": f"/authority/fold0/{key}.parquet", "sha256": "e" * 64}
            for key in FOLD_ARTIFACT_IDS
        },
        "graph": {
            "node_sha256": node_sha,
            "edge_sha256": edge_sha,
            "master_edge_sha256": edge_sha,
            "edge_count": len(bundle.edges),
            "same_relation_schema_all_variants": True,
        },
        "gates": {
            "frozen_patient_first_authority": True,
            "outer_train_expression_only": True,
            "outer_train_coexpression_only": True,
            "static_evidence_outcome_free": True,
            "historical_graph_rows_used": False,
            "legacy_graph_root_fallback_used": False,
            "toy_or_synthetic_fallback_used": False,
            "same_node_and_relation_schema_all_variants": True,
            "all_required_fold_declarations_present": True,
        },
    }


def test_training_accepts_only_the_frozen_patient_first_payload_binding() -> None:
    payload = _payload()
    _validate_prepared(payload, fold=0, artifact_hashes={})

    missing = _payload()
    missing.pop("patient_fold_authority")
    with pytest.raises(RuntimeError, match="frozen patient-first authority"):
        _validate_prepared(missing, fold=0, artifact_hashes={})

    drifted = _payload()
    drifted["patient_fold_authority"]["sample_patient_fold_map"]["sha256"] = (
        "0" * 64
    )
    with pytest.raises(RuntimeError, match="frozen patient-first authority"):
        _validate_prepared(drifted, fold=0, artifact_hashes={})
