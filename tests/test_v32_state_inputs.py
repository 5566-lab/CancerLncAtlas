import json

import pandas as pd
import pytest

from cc_hhgt.v32.core_export_contract_r4 import (
    ARCHITECTURE,
    CHECKPOINT_FORMAT,
    EXPORT_FORMAT,
    file_hash,
)
from cc_hhgt.v32.state_inputs import (
    StateInputError,
    load_v32_core_embeddings,
    validate_fold_manifest,
)


def _folds():
    return pd.DataFrame({
        "cancer_id": ["ACC"] * 5,
        "sample_id": [f"S{i}" for i in range(5)],
        "patient_id": [f"P{i}" for i in range(5)],
        "patient_fold_id": range(5),
    })


def test_fold_validation_preserves_explicit_patient_identity():
    observed = validate_fold_manifest(_folds())
    assert observed.patient_id.tolist() == [f"P{i}" for i in range(5)]
    with pytest.raises(StateInputError, match="patient_id"):
        validate_fold_manifest(_folds().drop(columns="patient_id"))


def test_joint_r62_core_export_is_the_supported_state_embedding_parent(tmp_path):
    prepared = {str(i): format(i + 1, "064x") for i in range(5)}
    graph = tmp_path / "GRAPH_AUTHORITY_RECEIPT.json"
    graph.write_text(json.dumps({"format": "R62_CONTEXT_GRAPH_AUTHORITY_V1",
        "variant": "G2", "prepared_manifest_sha256": prepared}))
    folds = {}
    for fold in range(5):
        exports = {}
        for node in ("lncRNA", "pathway", "cancer"):
            path = tmp_path / f"{fold}_{node}.parquet"
            pd.DataFrame({"node_id": [f"{node}:1"], "core_feature_0000": [fold + 0.5]}).to_parquet(path, index=False)
            exports[node] = {"path": path.name, "sha256": file_hash(path), "rows": 1, "features": 1}
        folds[str(fold)] = {"patient_fold": fold, "checkpoint_format": CHECKPOINT_FORMAT,
            "old_checkpoint_loaded": False, "trained_from_scratch": True,
            "core_parameter_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
            "prepared_sha256": prepared[str(fold)], "training_receipt_sha256": "c" * 64,
            "exports": exports}
    manifest = {"export_format": EXPORT_FORMAT, "core_architecture": ARCHITECTURE,
        "training_generation": "V3.2", "all_embeddings_from_newly_trained_v32_core": True,
        "historical_checkpoint_loaded": False, "historical_prediction_loaded": False,
        "test_labels_used_in_export": False, "task_weights": {"pathway": 1.0, "state": 0.5},
        "formal_graph_variant": "G2", "graph_authority_receipt": {"path": graph.name, "sha256": file_hash(graph)},
        "folds": folds}
    path = tmp_path / "CORE_EMBEDDING_MANIFEST.json"
    path.write_text(json.dumps(manifest))
    embeddings, lineage, loaded = load_v32_core_embeddings(path, repo_root=tmp_path)
    assert set(embeddings) == set(range(5)) and set(lineage) == set(range(5))
    assert loaded["task_weights"] == {"pathway": 1.0, "state": 0.5}
    assert all(frame.columns.tolist() == ["node_id", "core_feature_0000"] for frame in embeddings.values())


def test_retired_frozen_state_core_manifest_is_rejected(tmp_path):
    path = tmp_path / "CORE_EMBEDDING_MANIFEST.json"
    path.write_text(json.dumps({"export_format": "CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1"}))
    with pytest.raises(StateInputError, match="current joint export"):
        load_v32_core_embeddings(path)
