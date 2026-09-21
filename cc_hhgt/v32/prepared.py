from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PREPARED_FORMAT = "CC_HHGT_V3_2_PREPARED_FOLD_V1"


def validate_prepared_batch(batch: Mapping[str, Any]) -> None:
    required = {
        "candidate_batch",
        "base_logit",
        "conservation_context",
        "proxy_label",
        "weak_positive",
    }
    if missing := sorted(required - set(batch)):
        raise ValueError(f"Prepared candidate batch lacks fields: {missing}")
    indices = batch["candidate_batch"]
    if not isinstance(indices, Mapping) or set(indices) < {"l", "p", "c"}:
        raise ValueError("candidate_batch requires l, p and c node indices")
    lengths = []
    for value in (
        indices["l"],
        indices["p"],
        indices["c"],
        batch["base_logit"],
        batch["proxy_label"],
        batch["weak_positive"],
    ):
        if not hasattr(value, "__len__"):
            raise ValueError("Prepared batch arrays must expose a length")
        lengths.append(len(value))
    if len(set(lengths)) != 1:
        raise ValueError(f"Prepared batch candidate arrays are misaligned: {lengths}")


def build_prepared_fold_payload(
    *,
    patient_fold: int,
    bundle: Any,
    feature_dim: int,
    legacy_model_config: Mapping[str, Any],
    train_batches: Sequence[Mapping[str, Any]],
    validation_batches: Sequence[Mapping[str, Any]],
    artifact_hashes: Mapping[str, str],
    conservation_context_features: int,
    graph: Any | None = None,
) -> dict[str, Any]:
    """Create the serialization payload consumed by the guarded trainer.

    This is a pure packaging function: it does not import torch, fit a model,
    initialize CUDA, or write a checkpoint.
    """

    if int(patient_fold) not in range(5):
        raise ValueError("patient_fold must be in 0..4")
    if int(feature_dim) < 1 or int(conservation_context_features) < 1:
        raise ValueError("feature dimensions must be positive")
    if not train_batches or not validation_batches:
        raise ValueError("Prepared fold requires non-empty train and validation batches")
    for batch in [*train_batches, *validation_batches]:
        validate_prepared_batch(batch)
    required_hashes = {
        "code_sha256",
        "config_sha256",
        "input_manifest_sha256",
        "task_manifest_sha256",
    }
    if missing := sorted(required_hashes - set(artifact_hashes)):
        raise ValueError(f"Prepared fold lacks authorization hashes: {missing}")
    return {
        "prepared_format": PREPARED_FORMAT,
        "patient_fold": int(patient_fold),
        "bundle": bundle,
        "graph": graph,
        "feature_dim": int(feature_dim),
        "conservation_context_features": int(conservation_context_features),
        "legacy_model_config": dict(legacy_model_config),
        "train_batches": list(train_batches),
        "validation_batches": list(validation_batches),
        "artifact_hashes": {key: str(value) for key, value in sorted(artifact_hashes.items())},
        "contains_optimizer_state": False,
        "contains_trained_parameters": False,
    }
