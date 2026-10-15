"""Current V4 core inference, with explicit run identity and all-chunk semantics.

No labels or optimizers are consumed here. V2 export provenance describes a
completed selected training run; it does not invent a from-scratch attestation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Mapping

EXPORT_FORMAT = "CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V2"
CHECKPOINT_FORMAT = "CC_HHGT_V3_2_FULL_TRAINING_STATE_V4_GROUP_SCHEDULE"
ESTIMAND = "REGISTERED_CHUNK_WEIGHTED_MEAN_OF_ENCODER_OUTPUTS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def checked_sha(value, role: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{role} requires a SHA256")
    return value


def verified_path(record: Mapping, base: Path) -> Path:
    path = Path(record["path"])
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    if sha256(path) != checked_sha(record.get("sha256"), str(path)):
        raise ValueError(f"Artifact SHA256 mismatch: {path}")
    return path


def validate_weights(chunks, weights) -> None:
    if not chunks or len(set(chunks)) != len(chunks) or set(chunks) != set(weights):
        raise ValueError("All registered chunks require exactly one weight")
    if any(not math.isfinite(float(w)) or float(w) <= 0 for w in weights.values()):
        raise ValueError("Chunk weights must be finite and positive")
    if not math.isclose(math.fsum(map(float, weights.values())), 1., rel_tol=0, abs_tol=1e-12):
        raise ValueError("Complete chunk weights must sum to one; partial exports are forbidden")


def stream_embeddings(model, bundle, chunks, weights, graph_for_chunk, *, device="cpu"):
    """Accumulate one chunk at a time in float64, retaining no per-chunk stack."""
    import torch
    from .integrated_model import freeze_v32_core, assert_v32_core_unchanged

    validate_weights(chunks, weights)
    before = freeze_v32_core(model)
    expected_types = set(bundle.node_maps)
    if not {"lncRNA", "pathway"}.issubset(expected_types):
        raise ValueError("Current core graph requires exact lncRNA and pathway node maps")
    for kind, node_map in bundle.node_maps.items():
        if (not node_map or any(type(i) is not int for i in node_map.values())
                or sorted(node_map.values()) != list(range(len(node_map)))
                or len({str(name) for name in node_map}) != len(node_map)):
            raise ValueError(f"{kind} node map is not a complete zero-based alignment")
    means = {}
    compensation = {}
    with torch.inference_mode():
        for chunk in chunks:
            graph = graph_for_chunk(bundle, chunk, device=device)
            encoded = model.encoder.encode(graph)
            if set(encoded) != expected_types:
                raise ValueError(f"Chunk {chunk} has inconsistent encoded node types")
            for kind, tensor in encoded.items():
                value = tensor.detach().to(device="cpu", dtype=torch.float64)
                if (value.ndim != 2 or value.shape[0] != len(bundle.node_maps[kind])
                        or value.shape[1] == 0 or not torch.isfinite(value).all()):
                    raise ValueError(f"Chunk {chunk} has invalid {kind} embedding values or alignment")
                if kind not in means:
                    means[kind] = torch.zeros_like(value)
                    compensation[kind] = torch.zeros_like(value)
                if value.shape != means[kind].shape:
                    raise ValueError(f"Chunk {chunk} changes {kind} embedding shape")
                # Compensated weighted sum, preserving the original weights.
                increment = value * weights[chunk] - compensation[kind]
                total = means[kind] + increment
                compensation[kind] = (total - means[kind]) - increment
                means[kind] = total
            del graph, encoded
    assert_v32_core_unchanged(model, before)
    return means, before


def node_frame(values, node_map):
    import numpy as np
    import pandas as pd

    array = values.numpy().astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Embedding cannot be represented as finite float32")
    ordered = sorted(node_map, key=node_map.get)
    frame = pd.DataFrame(array, columns=[f"core_feature_{i:03d}" for i in range(array.shape[1])])
    frame.insert(0, "node_id", [str(key) for key in ordered])
    frame.insert(0, "node_index", np.arange(len(ordered), dtype=np.int64))
    unique = len(np.unique(array, axis=0))
    return frame, {"unique_embedding_vectors": unique,
                   "usable_for_node_discrimination": len(array) == 1 or unique > 1,
                   "constant_embedding": len(array) > 1 and unique == 1}


def validate_training_receipt(receipt, *, run_id, fold, checkpoint_sha, variant):
    expected = {"status": "SUCCESS", "run_id": run_id, "patient_fold": fold,
                "best_model_state_sha256": checkpoint_sha, "graph_variant": variant}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"Training receipt identity mismatch: {key}")
    if (type(receipt.get("optimizer_steps")) is not int or receipt["optimizer_steps"] <= 0
            or receipt.get("parameter_delta_positive") is not True
            or receipt.get("validation_full_runtime_coverage") is not True):
        raise ValueError("Selected core lacks evidence of completed optimizer work/full coverage")
    if not receipt.get("task_id"):
        raise ValueError("Training receipt lacks task identity")


def validate_checkpoint(checkpoint, receipt, *, fold, prepared_sha, config_sha, input_manifest_sha):
    if checkpoint.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError("Current exporter requires the V4 group-schedule checkpoint")
    for key in ["patient_fold", "task_id", "graph_variant", "artifact_hashes", "input_authority_hashes"]:
        if checkpoint.get(key) is None or checkpoint.get(key) != receipt.get(key):
            raise ValueError(f"Checkpoint / training receipt differ: {key}")
    if checkpoint["patient_fold"] != fold or checkpoint.get("prepared_artifact_sha256") != prepared_sha:
        raise ValueError("Checkpoint uses a different prepared fold")
    hashes = checkpoint["artifact_hashes"]
    if hashes.get("config_sha256") != config_sha or hashes.get("input_manifest_sha256") != input_manifest_sha:
        raise ValueError("Checkpoint uses different configuration / input manifest")


def validate_current_export(path: Path, *, expected_run_id=None):
    """Shared native-head reader for V2; checks small bound training receipts."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("export_format") != EXPORT_FORMAT or manifest.get("status") != "PASS"
            or manifest.get("embedding_estimand") != ESTIMAND
            or manifest.get("training_generation") != "V3.2"
            or manifest.get("all_registered_chunks_exported") is not True
            or manifest.get("labels_used_for_export") is not False
            or manifest.get("optimizer_started_during_export") is not False):
        raise ValueError("Incomplete or invalid current core export")
    run_id = manifest.get("core_run_id")
    if not run_id or (expected_run_id is not None and run_id != expected_run_id):
        raise ValueError("Current core run identity mismatch")
    folds = manifest.get("folds", {})
    if set(folds) != {str(i) for i in range(5)}:
        raise ValueError("Current core export requires all five folds")
    parameter_hashes = []
    for fold in range(5):
        item = folds[str(fold)]
        if (item.get("patient_fold") != fold or item.get("checkpoint_format") != CHECKPOINT_FORMAT
                or item.get("core_unchanged_after_export") is not True):
            raise ValueError("Current core fold identity or isolation mismatch")
        parameter_hashes.append(checked_sha(item.get("core_parameter_sha256"), "core parameters"))
        checkpoint_sha = checked_sha(item.get("checkpoint_sha256"), "checkpoint")
        receipt_path = verified_path(item["training_receipt"], path.parent)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_training_receipt(receipt, run_id=run_id, fold=fold,
                                  checkpoint_sha=checkpoint_sha, variant=manifest["graph_variant"])
        weights = item["chunk_weights"]
        validate_weights(tuple(weights), weights)
        if (weights != receipt.get("runtime_chunk_weights")
                or len(weights) != receipt.get("runtime_num_chunks")
                or len(weights) != item.get("runtime_chunks_total")):
            raise ValueError("Exported chunk weights differ from training")
    composite = hashlib.sha256(json.dumps(parameter_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return manifest, sha256(path), composite
