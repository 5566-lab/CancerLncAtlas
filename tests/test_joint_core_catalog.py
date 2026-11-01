from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from cla_engine.common import Stop, read_json
from cla_engine.core_export import _context_node_arrays, _fold_node_arrays
from cla_engine.execution_contract import validate_config_contract
from cla_engine.model import ARCHITECTURE, JointHGT, to_graph
from cc_hhgt.v32.core_export_contract_r4 import (
    ARCHITECTURE as EXPORT_ARCHITECTURE,
    CHECKPOINT_FORMAT,
    EXPORT_FORMAT,
    file_hash,
    validate_export,
)


ROOT = Path(__file__).resolve().parents[1]


def _model_fixture():
    torch.manual_seed(31)
    nodes = {
        "lncRNA": ["L0", "L1"],
        "pathway": ["P0", "P1"],
        "cancer": ["C0"],
        "pathway_family": ["F0"],
        "gene": ["G0", "G1"],
        "protein": ["R0", "R1"],
    }
    relations = [
        ("lncRNA", "targets", "pathway"),
        ("pathway", "rev_targets", "lncRNA"),
        ("cancer", "conditions", "lncRNA"),
        ("lncRNA", "rev_conditions", "cancer"),
        ("gene", "gene_context", "lncRNA"),
        ("protein", "protein_context", "pathway"),
    ]
    spec = {
        "architecture": ARCHITECTURE,
        "nodes": nodes,
        "relations": relations,
        "cancers": ["C0"],
        "states": ["RNAss"],
        "state_families": {"RNAss": "stemness_telomere"},
        "feature_dims": {kind: 6 for kind in nodes},
        "sc_dim": 1,
        "singlecell_role": "interpretation_only",
        "genomics_enabled": True,
        "numeric_features": 8,
    }
    config = {
        "task_weights": {"pathway": 1.0, "state": 0.5},
        "hidden": 8,
        "heads": 2,
        "layers": 2,
        "dropout": 0.0,
        "residual_bound": 1.0,
        "gate_penalty": 0.0,
    }
    arrays = {}
    for offset, (kind, ids) in enumerate(nodes.items(), start=1):
        values = np.arange(len(ids) * 6, dtype=np.float32).reshape(len(ids), 6)
        arrays[f"x::{kind}"] = (values + offset) / 13.0
    for relation in relations:
        key = "|".join(relation)
        edge = np.array([[0, 1], [0, 1]], dtype=np.int64)
        if relation[0] == "cancer":
            edge = np.array([[0, 0], [0, 1]], dtype=np.int64)
        elif relation[2] == "cancer":
            edge = np.array([[0, 1], [0, 0]], dtype=np.int64)
        arrays[f"edge::{key}"] = edge
        arrays[f"weight::{key}"] = np.ones(edge.shape[1], dtype=np.float32)
    return JointHGT(spec, config), to_graph(arrays, "cpu"), spec


def test_real_joint_encoder_gene_values_are_exported_and_fold_averaged():
    model, graph, spec = _model_fixture()
    model.eval()
    with torch.no_grad():
        encoded = model.encode(graph, 0)[0]
    contexts = _context_node_arrays(encoded, spec)
    expected = encoded["gene"].float().cpu().numpy()
    np.testing.assert_array_equal(contexts["gene"], expected)
    assert contexts["gene"].shape == (2, 8)
    assert np.isfinite(contexts["gene"]).all()

    totals = {node: value.astype(np.float64) * 2 for node, value in contexts.items()}
    counts = {node: 2 for node in contexts}
    fold = _fold_node_arrays(totals, counts, contexts["cancer"])
    np.testing.assert_allclose(fold["gene"], expected, rtol=0, atol=0)


def _write_manifest(root: Path, *, include_gene: bool):
    root.mkdir()
    prepared = {str(fold): f"{fold + 1:064x}" for fold in range(5)}
    graph = root / "GRAPH_AUTHORITY_RECEIPT.json"
    graph.write_text(
        json.dumps(
            {
                "format": "R62_CONTEXT_GRAPH_AUTHORITY_V1",
                "variant": "G2",
                "prepared_manifest_sha256": prepared,
            }
        ),
        encoding="utf-8",
    )
    folds = {}
    for fold in range(5):
        exports = {}
        nodes = ["cancer", "lncRNA", "pathway"] + (["gene"] if include_gene else [])
        for node in nodes:
            path = root / f"fold_{fold}_{node}.parquet"
            path.write_bytes(f"{fold}:{node}".encode())
            exports[node] = {
                "path": path.name,
                "sha256": file_hash(path),
                "rows": 1,
                "features": 8,
            }
        folds[str(fold)] = {
            "patient_fold": fold,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "old_checkpoint_loaded": False,
            "trained_from_scratch": True,
            "core_parameter_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "prepared_sha256": prepared[str(fold)],
            "training_receipt_sha256": "c" * 64,
            "exports": exports,
            "context_exports": {},
        }
    payload = {
        "export_format": EXPORT_FORMAT,
        "core_architecture": EXPORT_ARCHITECTURE,
        "training_generation": "V3.2",
        "all_embeddings_from_newly_trained_v32_core": True,
        "historical_checkpoint_loaded": False,
        "historical_prediction_loaded": False,
        "test_labels_used_in_export": False,
        "task_weights": {"pathway": 1.0, "state": 0.5},
        "formal_graph_variant": "G2",
        "graph_authority_receipt": {"path": graph.name, "sha256": file_hash(graph)},
        "folds": folds,
    }
    manifest = root / "CORE_EMBEDDING_MANIFEST.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, payload


def test_r62_manifest_validates_declared_gene_and_detects_gene_hash_drift(tmp_path):
    manifest, payload = _write_manifest(tmp_path / "with_gene", include_gene=True)
    validate_export(manifest)
    gene = manifest.parent / payload["folds"]["2"]["exports"]["gene"]["path"]
    gene.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="R62_CORE_EXPORT_MISMATCH:2:gene"):
        validate_export(manifest)


def test_legacy_three_node_manifest_remains_generically_valid(tmp_path):
    manifest, _ = _write_manifest(tmp_path / "three_nodes", include_gene=False)
    validate_export(manifest)


def test_expanded_state_catalog_seals_measurement_definitions_and_roles():
    cfg = read_json(ROOT / "config/149.json")
    validate_config_contract(cfg)

    swapped = copy.deepcopy(cfg)
    by_id = {row["id"]: row for row in swapped["targets"]}
    by_id["EXTEND"]["role"] = "auxiliary"
    by_id["ESTIMATEScore"]["role"] = "primary"
    with pytest.raises(Stop) as error:
        validate_config_contract(swapped)
    assert error.value.code == "TARGET_ROLE_CATALOG_DRIFT"

    remapped = copy.deepcopy(cfg)
    for index, row in enumerate(remapped["targets"]):
        row["column"] = f"source_specific_column_{index}"
    with pytest.raises(Stop) as error:
        validate_config_contract(remapped)
    assert error.value.code == "TARGET_ROLE_CATALOG_DRIFT"
