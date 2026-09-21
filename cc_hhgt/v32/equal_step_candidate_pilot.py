"""Guarded G0/G1/G2 equal-step comparison for the group-shared estimator.

This module is deliberately comparison-only.  It consumes three independently
guarded Fold-0 tasks, runs one 403-batch/101-step pass per estimator and graph
variant, evaluates both arms with the production exact all-29-chunk validation
path, and emits one typed JSON receipt to stdout.  It never resolves a formal
output root and contains no checkpoint, prediction, winner-selection, SUCCESS,
or FAILURE writer.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


RECEIPT_SCHEMA = "CC_HHGT_V3_2_EQUAL_STEP_CANDIDATE_PILOT_V1"
AUTHORIZED_TRAINER_SPECIFICATION = (
    "cc_hhgt.v32.equal_step_candidate_pilot:"
    "run_authorized_equal_step_candidate_pilot"
)
PASS_STATUS = "PASS_EQUAL_STEP_CANDIDATE_ONLY_NOT_AUTHORIZED_FOR_FORMAL_TRAINING"
SCIENTIFIC_FAIL_STATUS = "SCIENTIFIC_FAIL_EQUAL_STEP_CANDIDATE_ONLY"
RUNTIME_FAIL_STATUS = "RUNTIME_FAIL_EQUAL_STEP_CANDIDATE_ONLY"
NONZERO_RETURN_CODE = 42

EXPECTED_VARIANTS = ("G0", "G1", "G2")
EXPECTED_FOLD = 0
EXPECTED_SEED = 20260726
EXPECTED_BATCH_COUNT = 403
EXPECTED_BATCH_ROWS = 8192
EXPECTED_GROUP_BATCHES = 4
EXPECTED_OPTIMIZER_STEPS = 101
EXPECTED_RUNTIME_CHUNKS = 29

REFERENCE_ESTIMATOR = "balanced_cyclic_single_pass_v1"
PROPOSED_ESTIMATOR = "balanced_group_latin_shared_encoder_v1"
VALIDATION_LOGLOSS_ABSOLUTE_DIFFERENCE_MAX = 0.002
MEAN_LOGIT_PEARSON_MIN = 0.995
MEAN_LOGIT_SPEARMAN_MIN = 0.995


@dataclass(frozen=True)
class EqualStepPilotAuthorizationContext:
    """Three separately guarded formal-task contexts for one pilot."""

    g0: Any
    g1: Any
    g2: Any
    pilot_id: str = "v32-g012-equal-step-candidate-pilot-20260901-r1"

    @property
    def contexts(self) -> tuple[Any, Any, Any]:
        return (self.g0, self.g1, self.g2)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _base_receipt(status: str) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "artifact_type": "CANDIDATE_ONLY_EQUAL_STEP_COMPARISON",
        "comparison_only": True,
        "candidate_only": True,
        "formal_training_authorized_by_this_receipt": False,
        "winner_selection_input": False,
        "formal_artifacts_written": 0,
        "formal_checkpoint_written": False,
        "formal_success_marker_written": False,
        "formal_failure_marker_written": False,
        "formal_prediction_written": False,
        "test_or_outer_artifact_written": False,
        "output_channel": "STDOUT_JSON_ONLY",
        "test_rows_read": 0,
        "test_labels_read": False,
        "outer_metrics_read": False,
        "outer_predictions_read": False,
    }


def _emit_receipt(receipt: Mapping[str, Any]) -> None:
    print(json.dumps(dict(receipt), sort_keys=True), flush=True)


def build_equal_step_pilot_authorization(
    contexts: Sequence[Any],
    *,
    pilot_id: str = "v32-g012-equal-step-candidate-pilot-20260901-r1",
) -> EqualStepPilotAuthorizationContext:
    """Assemble three already guarded contexts without importing Torch."""

    from .training_guard import TrainingAuthorizationContext, load_structured_mapping

    if len(contexts) != len(EXPECTED_VARIANTS):
        raise RuntimeError("EQUAL_STEP_PILOT_REQUIRES_EXACTLY_THREE_CONTEXTS")
    by_variant: dict[str, Any] = {}
    endpoint_hardware = set()
    for context in contexts:
        if type(context) is not TrainingAuthorizationContext:
            raise RuntimeError("EQUAL_STEP_PILOT_REQUIRES_GUARDED_CONTEXTS")
        if context.authorized_trainer != AUTHORIZED_TRAINER_SPECIFICATION:
            raise RuntimeError(
                "EQUAL_STEP_PILOT_AUTHORIZED_TRAINER_DRIFT="
                f"{context.authorized_trainer!r}"
            )
        config = load_structured_mapping(context.config_path)
        variant = str(config.get("task_contract", {}).get("graph_variant", ""))
        if variant not in EXPECTED_VARIANTS or variant in by_variant:
            raise RuntimeError(
                f"EQUAL_STEP_PILOT_CONTEXT_VARIANT_SET_DRIFT={variant!r}"
            )
        by_variant[variant] = context
        endpoint_hardware.add((context.endpoint_id, context.hardware_class))
    if tuple(sorted(by_variant)) != EXPECTED_VARIANTS:
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_VARIANTS_DRIFT={tuple(sorted(by_variant))}"
        )
    if len(endpoint_hardware) != 1:
        raise RuntimeError("EQUAL_STEP_PILOT_ENDPOINT_OR_HARDWARE_DRIFT")
    return EqualStepPilotAuthorizationContext(
        g0=by_variant["G0"],
        g1=by_variant["G1"],
        g2=by_variant["G2"],
        pilot_id=str(pilot_id),
    )


def _reverify_context(context: Any) -> Any:
    from .training_guard import TrainingAuthorizationContext, guard_training_entry

    if type(context) is not TrainingAuthorizationContext:
        raise RuntimeError("EQUAL_STEP_PILOT_REQUIRES_GUARDED_CONTEXTS")
    if context.authorized_trainer != AUTHORIZED_TRAINER_SPECIFICATION:
        raise RuntimeError(
            "EQUAL_STEP_PILOT_AUTHORIZED_TRAINER_DRIFT="
            f"{context.authorized_trainer!r}"
        )
    verified = guard_training_entry(
        allow_training=True,
        repo_root=context.repo_root,
        config_path=context.config_path,
        input_manifest_path=context.input_manifest_path,
        task_manifest_path=context.task_manifest_path,
        approval_path=context.approval_path,
        run_id=context.run_id,
        task_id=context.task_id,
        endpoint_id=context.endpoint_id,
        hardware_class=context.hardware_class,
        trainer_specification=context.authorized_trainer,
    )
    if verified != context:
        raise RuntimeError("EQUAL_STEP_PILOT_REVERIFIED_CONTEXT_DRIFT")
    return verified


def _require_pilot_task(task: Mapping[str, Any]) -> None:
    required = {
        "task_type": "CC_HHGT_PATIENT_FOLD",
        "model": "CC-HHGT",
        "status": "PENDING",
        "blocked_reason": "",
        "paid_task": "true",
        "patient_fold": str(EXPECTED_FOLD),
        "seed": str(EXPECTED_SEED),
    }
    drift = {
        key: {"observed": str(task.get(key, "")), "required": value}
        for key, value in required.items()
        if str(task.get(key, "")).strip().lower() != value.lower()
    }
    if drift:
        raise RuntimeError(f"EQUAL_STEP_PILOT_TASK_CONTRACT_DRIFT={drift}")


def _optimization_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    primary = config.get("primary_model", {})
    runtime = config.get("runtime_profile", {})
    integration = primary.get("evidence_integration", {})
    return {
        "learning_rate": float(primary.get("learning_rate", 1e-3)),
        "weight_decay": float(primary.get("weight_decay", 1e-5)),
        "direction_loss_weight": float(
            primary.get("direction_loss_weight", 0.25)
        ),
        "residual_shrinkage": float(primary.get("residual_shrinkage", 1e-4)),
        "hidden_channels": int(primary.get("hidden_channels", 96)),
        "dropout": float(primary.get("dropout", 0.20)),
        "evidence_integration_mode": str(
            integration.get("mode", "external_router")
        ).lower(),
        "evidence_modalities": list(integration.get("modalities", ())),
        "mixed_precision": runtime.get("mixed_precision"),
        "candidate_microbatch_size": int(
            runtime.get("candidate_microbatch_size", 0)
        ),
        "gradient_accumulation": int(runtime.get("gradient_accumulation", 0)),
    }


def _fold_role() -> dict[str, Any]:
    return {
        "patient_fold": EXPECTED_FOLD,
        "selection_outer_pair_fold": EXPECTED_FOLD,
        "selection_validation_pair_fold": (EXPECTED_FOLD + 1) % 5,
        "role": "INNER_VALIDATION_ONLY",
    }


def _forbidden_payload_keys(payload: Mapping[str, Any]) -> list[str]:
    exact = {
        "outer_metrics",
        "outer_predictions",
        "heldout_test_labels",
        "sealed_test_labels",
        "sealed_test_predictions",
    }
    return sorted(
        str(key)
        for key in payload
        if str(key).lower().startswith("test") or str(key).lower() in exact
    )


def _validation_identity(
    payload: Mapping[str, Any],
    *,
    runtime_value_sha256,
) -> dict[str, Any]:
    """Bind ordered biological keys through node maps plus integer triplets."""

    if forbidden := _forbidden_payload_keys(payload):
        raise RuntimeError(f"EQUAL_STEP_PILOT_SEALED_TEST_FIREWALL={forbidden}")
    bundle = payload["bundle"]
    node_maps = getattr(bundle, "node_maps", None)
    if not isinstance(node_maps, Mapping):
        raise RuntimeError("EQUAL_STEP_PILOT_PREPARED_GRAPH_LACKS_NODE_MAPS")
    bound_maps = {}
    for kind in ("cancer", "lncRNA", "pathway"):
        mapping = node_maps.get(kind)
        if not isinstance(mapping, Mapping) or not mapping:
            raise RuntimeError(f"EQUAL_STEP_PILOT_NODE_MAP_MISSING={kind}")
        values = [int(value) for value in mapping.values()]
        if len(set(values)) != len(values):
            raise RuntimeError(f"EQUAL_STEP_PILOT_NODE_MAP_NOT_ONE_TO_ONE={kind}")
        bound_maps[kind] = {str(key): int(value) for key, value in mapping.items()}

    key_batches = []
    label_batches = []
    model_input_batches = []
    mask_weight_batches = []
    row_counts = []
    proxy_parts = []
    for batch_index, raw in enumerate(payload["validation_batches"]):
        candidate = raw.get("candidate_batch")
        if not isinstance(candidate, Mapping):
            raise RuntimeError(
                f"EQUAL_STEP_PILOT_VALIDATION_CANDIDATE_MALFORMED={batch_index}"
            )
        keys = {}
        for key in ("l", "p", "c"):
            value = candidate.get(key)
            if value is None or not hasattr(value, "detach"):
                raise RuntimeError(
                    f"EQUAL_STEP_PILOT_VALIDATION_KEY_MISSING={batch_index}:{key}"
                )
            keys[key] = value
        rows = int(len(keys["l"]))
        if rows < 1 or any(len(keys[key]) != rows for key in ("p", "c")):
            raise RuntimeError(
                f"EQUAL_STEP_PILOT_VALIDATION_KEY_ALIGNMENT_DRIFT={batch_index}"
            )
        labels = {
            key: raw[key]
            for key in (
                "proxy_label",
                "weak_positive",
                "direction_label",
                "direction_available",
            )
        }
        if "association_available" in raw:
            labels["association_available"] = raw["association_available"]
        if any(len(value) != rows for value in labels.values()):
            raise RuntimeError(
                f"EQUAL_STEP_PILOT_VALIDATION_LABEL_ALIGNMENT_DRIFT={batch_index}"
            )
        proxy = raw["proxy_label"].detach().to(device="cpu").reshape(-1)
        if not bool(((proxy == 0) | (proxy == 1)).all()):
            raise RuntimeError(
                f"EQUAL_STEP_PILOT_VALIDATION_PROXY_NOT_BINARY={batch_index}"
            )
        key_batches.append(keys)
        label_batches.append(labels)
        for required_input in ("base_logit", "conservation_context"):
            value = raw.get(required_input)
            if value is None or not hasattr(value, "detach") or len(value) != rows:
                raise RuntimeError(
                    "EQUAL_STEP_PILOT_VALIDATION_MODEL_INPUT_ALIGNMENT_DRIFT="
                    f"{batch_index}:{required_input}"
                )
        graph_available_present = "graph_available" in raw
        graph_available = raw.get("graph_available")
        if graph_available_present and (
            not hasattr(graph_available, "detach")
            or len(graph_available) != rows
        ):
            raise RuntimeError(
                "EQUAL_STEP_PILOT_VALIDATION_GRAPH_AVAILABLE_ALIGNMENT_DRIFT="
                f"{batch_index}"
            )
        # Bind every candidate-side model input, including any future admitted
        # field, while avoiding a second scan of the already bound l/p/c keys.
        candidate_extras = {
            str(key): value
            for key, value in candidate.items()
            if str(key) not in {"l", "p", "c"}
        }
        model_input_batches.append(
            {
                "base_logit": raw["base_logit"],
                "conservation_context": raw["conservation_context"],
                "graph_available_present": graph_available_present,
                "graph_available": graph_available,
                "candidate_extras": candidate_extras,
            }
        )
        mask_weights = {
            "weak_positive": raw["weak_positive"],
            "direction_available": raw["direction_available"],
            "graph_available_present": graph_available_present,
            "graph_available": graph_available,
        }
        if "association_available" in raw:
            mask_weights["association_available"] = raw["association_available"]
        mask_weight_batches.append(mask_weights)
        row_counts.append(rows)
        proxy_parts.append(proxy)

    node_maps_sha256 = runtime_value_sha256(bound_maps)
    candidate_key_sha256 = runtime_value_sha256(
        {
            "node_maps_sha256": node_maps_sha256,
            "ordered_candidate_index_triplets": key_batches,
            "batch_row_counts": row_counts,
        }
    )
    label_sha256 = runtime_value_sha256(
        {
            "ordered_labels": label_batches,
            "batch_row_counts": row_counts,
        }
    )
    model_input_sha256 = runtime_value_sha256(
        {
            "ordered_model_inputs": model_input_batches,
            "batch_row_counts": row_counts,
        }
    )
    mask_weight_sha256 = runtime_value_sha256(
        {
            "ordered_masks_and_row_weights": mask_weight_batches,
            "batch_row_counts": row_counts,
        }
    )
    role = _fold_role()
    validation_loss_payload_sha256 = runtime_value_sha256(
        {
            "candidate_key_sha256": candidate_key_sha256,
            "label_sha256": label_sha256,
            "model_input_sha256": model_input_sha256,
            "mask_weight_sha256": mask_weight_sha256,
            "fold_role": role,
        }
    )
    identity_sha256 = runtime_value_sha256(
        {
            "candidate_key_sha256": candidate_key_sha256,
            "label_sha256": label_sha256,
            "model_input_sha256": model_input_sha256,
            "mask_weight_sha256": mask_weight_sha256,
            "validation_loss_payload_sha256": validation_loss_payload_sha256,
            "fold_role": role,
        }
    )
    return {
        "node_maps_sha256": node_maps_sha256,
        "ordered_candidate_key_sha256": candidate_key_sha256,
        "ordered_label_sha256": label_sha256,
        "ordered_model_input_sha256": model_input_sha256,
        "ordered_mask_weight_sha256": mask_weight_sha256,
        "validation_loss_payload_sha256": validation_loss_payload_sha256,
        "ordered_identity_sha256": identity_sha256,
        "fold_role": role,
        "batch_row_counts": row_counts,
        "candidate_rows": int(sum(row_counts)),
        "proxy_label": __import__("torch").cat(proxy_parts).to(dtype=__import__("torch").float64),
    }


def _training_identity(
    payload: Mapping[str, Any],
    *,
    runtime_value_sha256,
) -> dict[str, Any]:
    """Bind every ordered row consumed by the optimizer across variants.

    Equal batch counts and row counts are not sufficient fairness authority:
    independently materialized G0/G1/G2 payloads can retain the same geometry
    while differing in candidates, labels, masks or candidate-side inputs.
    """

    bundle = payload["bundle"]
    node_maps = getattr(bundle, "node_maps", None)
    if not isinstance(node_maps, Mapping):
        raise RuntimeError("EQUAL_STEP_PILOT_PREPARED_GRAPH_LACKS_NODE_MAPS")
    bound_maps = {}
    for kind in ("cancer", "lncRNA", "pathway"):
        mapping = node_maps.get(kind)
        if not isinstance(mapping, Mapping) or not mapping:
            raise RuntimeError(f"EQUAL_STEP_PILOT_NODE_MAP_MISSING={kind}")
        values = [int(value) for value in mapping.values()]
        if len(set(values)) != len(values):
            raise RuntimeError(f"EQUAL_STEP_PILOT_NODE_MAP_NOT_ONE_TO_ONE={kind}")
        bound_maps[kind] = {str(key): int(value) for key, value in mapping.items()}

    key_batches = []
    label_batches = []
    model_input_batches = []
    mask_weight_batches = []
    row_counts = []
    for batch_index, raw in enumerate(payload["train_batches"]):
        candidate = raw.get("candidate_batch")
        if not isinstance(candidate, Mapping):
            raise RuntimeError(
                f"EQUAL_STEP_PILOT_TRAIN_CANDIDATE_MALFORMED={batch_index}"
            )
        keys = {}
        for key in ("l", "p", "c"):
            value = candidate.get(key)
            if value is None or not hasattr(value, "detach"):
                raise RuntimeError(
                    f"EQUAL_STEP_PILOT_TRAIN_KEY_MISSING={batch_index}:{key}"
                )
            keys[key] = value
        rows = int(len(keys["l"]))
        if rows < 1 or any(len(keys[key]) != rows for key in ("p", "c")):
            raise RuntimeError(
                f"EQUAL_STEP_PILOT_TRAIN_KEY_ALIGNMENT_DRIFT={batch_index}"
            )
        labels = {}
        for key in (
            "proxy_label",
            "weak_positive",
            "direction_label",
            "direction_available",
        ):
            value = raw.get(key)
            if value is None or not hasattr(value, "detach") or len(value) != rows:
                raise RuntimeError(
                    "EQUAL_STEP_PILOT_TRAIN_LABEL_ALIGNMENT_DRIFT="
                    f"{batch_index}:{key}"
                )
            labels[key] = value
        if "association_available" in raw:
            value = raw["association_available"]
            if not hasattr(value, "detach") or len(value) != rows:
                raise RuntimeError(
                    "EQUAL_STEP_PILOT_TRAIN_LABEL_ALIGNMENT_DRIFT="
                    f"{batch_index}:association_available"
                )
            labels["association_available"] = value
        proxy = raw["proxy_label"].detach().to(device="cpu").reshape(-1)
        if not bool(((proxy == 0) | (proxy == 1)).all()):
            raise RuntimeError(
                f"EQUAL_STEP_PILOT_TRAIN_PROXY_NOT_BINARY={batch_index}"
            )
        for required_input in ("base_logit", "conservation_context"):
            value = raw.get(required_input)
            if value is None or not hasattr(value, "detach") or len(value) != rows:
                raise RuntimeError(
                    "EQUAL_STEP_PILOT_TRAIN_MODEL_INPUT_ALIGNMENT_DRIFT="
                    f"{batch_index}:{required_input}"
                )
        graph_available_present = "graph_available" in raw
        graph_available = raw.get("graph_available")
        if graph_available_present and (
            not hasattr(graph_available, "detach") or len(graph_available) != rows
        ):
            raise RuntimeError(
                "EQUAL_STEP_PILOT_TRAIN_GRAPH_AVAILABLE_ALIGNMENT_DRIFT="
                f"{batch_index}"
            )
        candidate_extras = {
            str(key): value
            for key, value in candidate.items()
            if str(key) not in {"l", "p", "c"}
        }
        key_batches.append(keys)
        label_batches.append(labels)
        model_input_batches.append(
            {
                "base_logit": raw["base_logit"],
                "conservation_context": raw["conservation_context"],
                "graph_available_present": graph_available_present,
                "graph_available": graph_available,
                "candidate_extras": candidate_extras,
            }
        )
        mask_weights = {
            "weak_positive": raw["weak_positive"],
            "direction_available": raw["direction_available"],
            "graph_available_present": graph_available_present,
            "graph_available": graph_available,
        }
        if "association_available" in raw:
            mask_weights["association_available"] = raw["association_available"]
        mask_weight_batches.append(mask_weights)
        row_counts.append(rows)

    node_maps_sha256 = runtime_value_sha256(bound_maps)
    candidate_key_sha256 = runtime_value_sha256(
        {
            "node_maps_sha256": node_maps_sha256,
            "ordered_candidate_index_triplets": key_batches,
            "batch_row_counts": row_counts,
        }
    )
    label_sha256 = runtime_value_sha256(
        {"ordered_labels": label_batches, "batch_row_counts": row_counts}
    )
    model_input_sha256 = runtime_value_sha256(
        {"ordered_model_inputs": model_input_batches, "batch_row_counts": row_counts}
    )
    mask_weight_sha256 = runtime_value_sha256(
        {
            "ordered_masks_and_row_weights": mask_weight_batches,
            "batch_row_counts": row_counts,
        }
    )
    loss_payload_sha256 = runtime_value_sha256(
        {
            "candidate_key_sha256": candidate_key_sha256,
            "label_sha256": label_sha256,
            "model_input_sha256": model_input_sha256,
            "mask_weight_sha256": mask_weight_sha256,
        }
    )
    identity_sha256 = runtime_value_sha256(
        {
            "candidate_key_sha256": candidate_key_sha256,
            "label_sha256": label_sha256,
            "model_input_sha256": model_input_sha256,
            "mask_weight_sha256": mask_weight_sha256,
            "training_loss_payload_sha256": loss_payload_sha256,
        }
    )
    return {
        "node_maps_sha256": node_maps_sha256,
        "ordered_candidate_key_sha256": candidate_key_sha256,
        "ordered_label_sha256": label_sha256,
        "ordered_model_input_sha256": model_input_sha256,
        "ordered_mask_weight_sha256": mask_weight_sha256,
        "training_loss_payload_sha256": loss_payload_sha256,
        "ordered_identity_sha256": identity_sha256,
        "batch_row_counts": row_counts,
        "candidate_rows": int(sum(row_counts)),
    }


def _validation_loss_contract(
    training: Any, primary: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind every non-payload scalar used by the production validation loss."""

    signature = inspect.signature(training._global_loss_from_outputs)
    defaults = {}
    for name in ("positive_prior", "unlabeled_weight", "weak_positive_weight"):
        value = signature.parameters[name].default
        if value is inspect.Parameter.empty or not isinstance(value, (int, float)):
            raise RuntimeError(
                f"EQUAL_STEP_PILOT_VALIDATION_LOSS_DEFAULT_DRIFT={name}:{value!r}"
            )
        defaults[name] = float(value)
    return {
        "production_helper": "training._global_loss_from_outputs",
        "positive_prior": defaults["positive_prior"],
        "unlabeled_weight": defaults["unlabeled_weight"],
        "weak_positive_weight": defaults["weak_positive_weight"],
        "direction_loss_weight": float(
            primary.get("direction_loss_weight", 0.25)
        ),
        "residual_shrinkage": float(primary.get("residual_shrinkage", 1e-4)),
        "chunk_aggregation": "CANONICAL_KAHAN_FLOAT64_BEFORE_NONLINEAR_LOSS",
        "ordinary_logloss_reduction": "BINARY_CROSS_ENTROPY_WITH_LOGITS_MEAN",
    }


def _clone_state(model) -> dict[str, Any]:
    return {
        str(name): value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def _synchronize(torch, device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _timing_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values or not all(math.isfinite(float(value)) and value >= 0 for value in values):
        raise RuntimeError("EQUAL_STEP_PILOT_TIMING_INVALID")
    ordered = sorted(float(value) for value in values)
    array = np.asarray(ordered, dtype=np.float64)
    return {
        "count": len(ordered),
        "total_seconds": float(math.fsum(ordered)),
        "mean_seconds": float(statistics.fmean(ordered)),
        "median_seconds": float(np.median(array)),
        "p95_seconds": float(np.percentile(array, 95)),
        "minimum_seconds": ordered[0],
        "maximum_seconds": ordered[-1],
    }


def _binary_logloss(logit, label, *, torch) -> float:
    logits = logit.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    labels = label.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if len(logits) != len(labels) or not len(logits):
        raise RuntimeError("EQUAL_STEP_PILOT_LOGLOSS_ROWS_MISALIGNED")
    if not bool(torch.isfinite(logits).all() and torch.isfinite(labels).all()):
        raise RuntimeError("EQUAL_STEP_PILOT_LOGLOSS_NONFINITE")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise RuntimeError("EQUAL_STEP_PILOT_LOGLOSS_LABEL_NOT_BINARY")
    value = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    result = float(value.detach().cpu())
    if not math.isfinite(result):
        raise RuntimeError("EQUAL_STEP_PILOT_LOGLOSS_NONFINITE_RESULT")
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise RuntimeError("EQUAL_STEP_PILOT_RANK_INPUT_INVALID")
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    stops = np.r_[starts[1:], len(array)]
    average = (starts + stops - 1).astype(np.float64) / 2.0 + 1.0
    ranked_sorted = np.repeat(average, stops - starts)
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = ranked_sorted
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or len(x) < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise RuntimeError("EQUAL_STEP_PILOT_CORRELATION_INPUT_INVALID")
    x = x - x.mean(dtype=np.float64)
    y = y - y.mean(dtype=np.float64)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if not denominator > 0:
        raise RuntimeError("EQUAL_STEP_PILOT_CORRELATION_UNDEFINED_CONSTANT_LOGITS")
    value = float(np.dot(x, y) / denominator)
    if not math.isfinite(value):
        raise RuntimeError("EQUAL_STEP_PILOT_CORRELATION_NONFINITE")
    return max(-1.0, min(1.0, value))


def compare_equal_step_arms(
    reference: Mapping[str, Any], proposed: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen downstream thresholds without borrowing an arm."""

    torch = __import__("torch")
    exact_fields = (
        "ordered_candidate_key_sha256",
        "ordered_label_sha256",
        "ordered_identity_sha256",
        "ordered_model_input_sha256",
        "ordered_mask_weight_sha256",
        "validation_loss_payload_sha256",
        "validation_loss_contract_sha256",
        "precision_contract_sha256",
        "fold_role",
        "runtime_chunk_weights",
        "runtime_chunk_weights_sha256",
        "candidate_rows",
    )
    missing = {
        side: sorted(field for field in exact_fields if field not in arm)
        for side, arm in (("reference", reference), ("proposed", proposed))
    }
    missing = {side: fields for side, fields in missing.items() if fields}
    if missing:
        raise RuntimeError(f"EQUAL_STEP_PILOT_ARM_IDENTITY_FIELD_MISSING={missing}")
    exact_gate = {
        field: reference.get(field) == proposed.get(field) for field in exact_fields
    }
    proxy_equal = bool(
        torch.equal(reference["proxy_label"], proposed["proxy_label"])
    )
    reference_logit = reference["mean_final_logit"].detach().to(
        device="cpu", dtype=torch.float64
    )
    proposed_logit = proposed["mean_final_logit"].detach().to(
        device="cpu", dtype=torch.float64
    )
    if tuple(reference_logit.shape) != tuple(proposed_logit.shape):
        raise RuntimeError("EQUAL_STEP_PILOT_MEAN_LOGIT_SHAPE_DRIFT")
    left = reference_logit.numpy()
    right = proposed_logit.numpy()
    pearson = _pearson(left, right)
    spearman = _pearson(_average_ranks(left), _average_ranks(right))
    difference = abs(
        float(reference["validation_logloss"])
        - float(proposed["validation_logloss"])
    )
    gates = {
        "validation_logloss_absolute_difference": (
            difference <= VALIDATION_LOGLOSS_ABSOLUTE_DIFFERENCE_MAX
        ),
        "mean_logit_pearson": pearson >= MEAN_LOGIT_PEARSON_MIN,
        "mean_logit_spearman": spearman >= MEAN_LOGIT_SPEARMAN_MIN,
        "ordered_candidate_keys_labels_fold_role_chunk_weights_exact": (
            all(exact_gate.values()) and proxy_equal
        ),
    }
    return {
        "pass": bool(all(gates.values())),
        "thresholds": {
            "validation_logloss_absolute_difference_max": (
                VALIDATION_LOGLOSS_ABSOLUTE_DIFFERENCE_MAX
            ),
            "mean_logit_pearson_min": MEAN_LOGIT_PEARSON_MIN,
            "mean_logit_spearman_min": MEAN_LOGIT_SPEARMAN_MIN,
        },
        "reference_validation_logloss": float(reference["validation_logloss"]),
        "proposed_validation_logloss": float(proposed["validation_logloss"]),
        "validation_logloss_absolute_difference": difference,
        "mean_logit_pearson": pearson,
        "mean_logit_spearman": spearman,
        "exact_identity_field_gates": exact_gate,
        "proxy_label_tensor_exact": proxy_equal,
        "gates": gates,
    }


def _execute_arm(
    *,
    pilot_id: str,
    authorization_run_id: str,
    authorization_task_id: str,
    variant: str,
    estimator: str,
    model: Any,
    initial_model_state: Mapping[str, Any],
    initial_model_sha256: str,
    initial_rng_state: Mapping[str, Any],
    initial_rng_sha256: str,
    train_batches: Sequence[Mapping[str, Any]],
    validation_batches: Sequence[Mapping[str, Any]],
    validation_identity: Mapping[str, Any],
    bundle: Any,
    runtime_chunks: tuple[int, ...],
    runtime_chunk_weights: Mapping[int, float],
    runtime_permutation: tuple[int, ...],
    pass_offset_permutation: tuple[int, ...],
    runtime_plan: Any,
    precision: Mapping[str, Any],
    primary: Mapping[str, Any],
    graph_for_chunk,
    torch: Any,
    training: Any,
    runtime_value_sha256,
    device: str,
) -> dict[str, Any]:
    if estimator not in {REFERENCE_ESTIMATOR, PROPOSED_ESTIMATOR}:
        raise RuntimeError(f"EQUAL_STEP_PILOT_ESTIMATOR_INVALID={estimator}")
    arm = "reference" if estimator == REFERENCE_ESTIMATOR else "proposed"
    model.load_state_dict(initial_model_state)
    training._restore_torch_rng_state(torch, initial_rng_state)
    observed_initial_model_sha256 = runtime_value_sha256(model.state_dict())
    observed_initial_rng_sha256 = runtime_value_sha256(
        training._capture_torch_rng_state(torch)
    )
    if observed_initial_model_sha256 != initial_model_sha256:
        raise RuntimeError(f"EQUAL_STEP_PILOT_INIT_MODEL_RESTORE_DRIFT={variant}:{estimator}")
    if observed_initial_rng_sha256 != initial_rng_sha256:
        raise RuntimeError(f"EQUAL_STEP_PILOT_INIT_RNG_RESTORE_DRIFT={variant}:{estimator}")

    batch_row_counts = tuple(training._batch_row_count(batch) for batch in train_batches)
    cycle_schedule = training._candidate_chunk_cycle_schedules(
        batch_count=len(train_batches),
        chunks=runtime_chunks,
        permutation=runtime_permutation,
        pass_offset_permutation=pass_offset_permutation,
        max_cycles=1,
        mode=estimator,
        gradient_accumulation=runtime_plan.gradient_accumulation,
    )
    schedule_payload = training._candidate_chunk_schedule_payload(
        mode=estimator,
        chunks=runtime_chunks,
        weights=runtime_chunk_weights,
        permutation=runtime_permutation,
        pass_offset_permutation=pass_offset_permutation,
        batch_count=len(train_batches),
        cycle_schedules=cycle_schedule,
        gradient_accumulation=runtime_plan.gradient_accumulation,
        batch_row_counts=batch_row_counts,
    )
    schedule_sha256 = training._candidate_chunk_schedule_sha256(schedule_payload)
    assignment_sha256 = _json_sha256(schedule_payload["cycles"])
    pass_pairs = cycle_schedule[0][0]
    work_items = training.materialize_pass_work_items(
        pass_pairs,
        train_batches,
        microbatch_size=runtime_plan.candidate_microbatch_size,
    )
    optimizer_groups = training._canonical_optimizer_batch_groups(
        len(work_items), runtime_plan.gradient_accumulation
    )
    expected_groups = tuple(
        tuple(range(start, min(start + runtime_plan.gradient_accumulation, len(work_items))))
        for start in range(0, len(work_items), runtime_plan.gradient_accumulation)
    )
    if optimizer_groups != expected_groups:
        raise RuntimeError("EQUAL_STEP_PILOT_OPTIMIZER_GROUP_AUTHORITY_DRIFT")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(primary.get("learning_rate", 1e-3)),
        weight_decay=float(primary.get("weight_decay", 1e-5)),
    )
    direction_weight = float(primary.get("direction_loss_weight", 0.25))
    shrinkage = float(primary.get("residual_shrinkage", 1e-4))
    telemetry = {
        "optimizer_steps": 0,
        "encoder_forward_calls": 0,
        "decoder_forward_calls": 0,
        "global_loss_calls": 0,
        "backward_calls": 0,
    }
    step_seconds = []
    objectives = []
    last_guard = None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    for group_index, indices in enumerate(optimizer_groups):
        group = [work_items[index] for index in indices]
        optimizer.zero_grad(set_to_none=True)
        _synchronize(torch, device)
        step_started = time.perf_counter()
        try:
            if estimator == PROPOSED_ESTIMATOR:
                plan, calls = training.shared_encoder_conventional_group_backward(
                    model,
                    group,
                    graph_for_chunk,
                    torch=torch,
                    device=device,
                    precision=precision,
                    direction_loss_weight=direction_weight,
                    shrinkage=shrinkage,
                )
            else:
                plan, calls = training.streaming_group_backward(
                    model,
                    group,
                    graph_for_chunk,
                    torch=torch,
                    device=device,
                    precision=precision,
                    direction_loss_weight=direction_weight,
                    shrinkage=shrinkage,
                    return_call_telemetry=True,
                )
        except BaseException as exc:
            if training._is_cuda_oom(torch, exc):
                optimizer.zero_grad(set_to_none=True)
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA_OOM_EQUAL_STEP_PILOT_TRAINING_FAIL_CLOSED="
                    f"{variant}:{estimator}:group={group_index}"
                ) from exc
            raise
        try:
            last_guard = training.optimizer_step_with_guards(
                model,
                optimizer,
                torch=torch,
                objective_value=plan.objective_value,
            )
        except BaseException as exc:
            if training._is_cuda_oom(torch, exc):
                raise RuntimeError(
                    "CUDA_OOM_EQUAL_STEP_PILOT_OPTIMIZER_FAIL_CLOSED="
                    f"{variant}:{estimator}:group={group_index}"
                ) from exc
            raise
        _synchronize(torch, device)
        step_seconds.append(time.perf_counter() - step_started)
        objectives.append(float(plan.objective_value))
        telemetry["optimizer_steps"] += 1
        telemetry["encoder_forward_calls"] += int(calls.encoder_forward_calls)
        telemetry["decoder_forward_calls"] += int(calls.decoder_forward_calls)
        telemetry["global_loss_calls"] += int(calls.global_loss_calls)
        telemetry["backward_calls"] += int(calls.backward_calls)
        optimizer.zero_grad(set_to_none=True)
        _emit_receipt(
            {
                **_base_receipt("EQUAL_STEP_OPTIMIZER_PROGRESS"),
                "pilot_id": str(pilot_id),
                "authorization_run_id": str(authorization_run_id),
                "authorization_task_id": str(authorization_task_id),
                "graph_variant": variant,
                "arm": arm,
                "phase": "OPTIMIZER",
                "estimator": estimator,
                "optimizer_step": telemetry["optimizer_steps"],
                "required_optimizer_steps": len(optimizer_groups),
                "completed": telemetry["optimizer_steps"],
                "total": len(optimizer_groups),
                "progress_event_id": (
                    f"{pilot_id}|{authorization_run_id}|{authorization_task_id}|"
                    f"{variant}|{arm}|OPTIMIZER|{telemetry['optimizer_steps']}"
                ),
                "objective_value": float(plan.objective_value),
                "step_seconds": float(step_seconds[-1]),
                "grad_finite": last_guard.get("grad_finite") is True,
                "parameters_finite": last_guard.get("parameters_finite") is True,
                "parameter_delta_positive": (
                    last_guard.get("parameter_delta_positive") is True
                ),
                "training_started": True,
                "gpu_telemetry": {
                    "device": str(device),
                    "cuda_available": bool(torch.cuda.is_available()),
                    "memory_allocated_bytes": int(
                        torch.cuda.memory_allocated()
                        if str(device).startswith("cuda")
                        else 0
                    ),
                    "memory_reserved_bytes": int(
                        torch.cuda.memory_reserved()
                        if str(device).startswith("cuda")
                        else 0
                    ),
                },
            }
        )
    _synchronize(torch, device)
    training_seconds = time.perf_counter() - training_started
    if telemetry["optimizer_steps"] != len(optimizer_groups):
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_STEP_COUNT_DRIFT={variant}:{estimator}:{telemetry}"
        )
    if last_guard is None or any(
        last_guard.get(key) is not True
        for key in ("grad_finite", "parameters_finite", "parameter_delta_positive")
    ):
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_OPTIMIZER_GUARD_FAILED={variant}:{estimator}"
        )
    final_model_sha256 = runtime_value_sha256(model.state_dict())
    final_rng_sha256 = runtime_value_sha256(training._capture_torch_rng_state(torch))

    split_validation = training.split_batches_for_evaluation(
        validation_batches,
        microbatch_size=runtime_plan.candidate_microbatch_size,
    )
    _synchronize(torch, device)
    validation_started = time.perf_counter()
    validation_completed_chunks: set[int] = set()
    validation_chunk_ordinals = {
        int(chunk): ordinal for ordinal, chunk in enumerate(runtime_chunks, start=1)
    }

    def validation_progress(chunk: int, batch_index: int, total_batches: int) -> None:
        if int(batch_index) != int(total_batches):
            return
        resolved_chunk = int(chunk)
        if resolved_chunk in validation_completed_chunks:
            raise RuntimeError(
                "EQUAL_STEP_PILOT_VALIDATION_PROGRESS_DUPLICATE_CHUNK="
                f"{variant}:{estimator}:{resolved_chunk}"
            )
        validation_completed_chunks.add(resolved_chunk)
        _emit_receipt(
            {
                **_base_receipt("EQUAL_STEP_VALIDATION_CHUNK_PROGRESS"),
                "pilot_id": str(pilot_id),
                "authorization_run_id": str(authorization_run_id),
                "authorization_task_id": str(authorization_task_id),
                "graph_variant": variant,
                "arm": arm,
                "phase": "VALIDATION",
                "estimator": estimator,
                "runtime_chunk": resolved_chunk,
                "validation_chunk_ordinal": validation_chunk_ordinals[resolved_chunk],
                "validation_chunks_completed": len(validation_completed_chunks),
                "required_runtime_chunks": len(runtime_chunks),
                "completed": len(validation_completed_chunks),
                "total": len(runtime_chunks),
                "progress_event_id": (
                    f"{pilot_id}|{authorization_run_id}|{authorization_task_id}|"
                    f"{variant}|{arm}|VALIDATION|"
                    f"{len(validation_completed_chunks)}"
                ),
                "candidate_microbatches_completed_for_chunk": int(total_batches),
            }
        )

    try:
        detailed = training.evaluate_full_runtime_coverage_detailed(
            model,
            bundle,
            split_validation,
            chunks=runtime_chunks,
            weights=runtime_chunk_weights,
            torch=torch,
            device=device,
            direction_loss_weight=direction_weight,
            shrinkage=shrinkage,
            precision=precision,
            progress_callback=validation_progress,
        )
    except BaseException as exc:
        if training._is_cuda_oom(torch, exc):
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA_OOM_EQUAL_STEP_PILOT_VALIDATION_FAIL_CLOSED="
                f"{variant}:{estimator}"
            ) from exc
        raise
    _synchronize(torch, device)
    validation_seconds = time.perf_counter() - validation_started
    if detailed.runtime_chunks_evaluated != len(runtime_chunks):
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_VALIDATION_CHUNK_DRIFT={variant}:{estimator}"
        )
    if validation_completed_chunks != set(runtime_chunks):
        raise RuntimeError(
            "EQUAL_STEP_PILOT_VALIDATION_PROGRESS_COVERAGE_DRIFT="
            f"{variant}:{estimator}:{sorted(validation_completed_chunks)}"
        )
    expected_proxy = validation_identity["proxy_label"]
    if not torch.equal(detailed.proxy_label, expected_proxy):
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_VALIDATION_LABEL_ORDER_DRIFT={variant}:{estimator}"
        )
    if detailed.candidate_rows != int(validation_identity["candidate_rows"]):
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_VALIDATION_ROW_DRIFT={variant}:{estimator}"
        )
    logloss = _binary_logloss(
        detailed.mean_final_logit, detailed.proxy_label, torch=torch
    )
    weights_receipt = {
        str(chunk): float(runtime_chunk_weights[chunk]) for chunk in runtime_chunks
    }
    validation_loss_contract = _validation_loss_contract(training, primary)
    precision_contract = dict(precision)
    peak_reserved = (
        int(torch.cuda.max_memory_reserved()) if str(device).startswith("cuda") else 0
    )
    result = {
        "estimator": estimator,
        "gradient_algorithm": training._gradient_algorithm_for_schedule_mode(estimator),
        "schedule_sha256": schedule_sha256,
        "assignment_sha256": assignment_sha256,
        "selected_pass_offset": int(pass_offset_permutation[0]),
        "initial_model_sha256": observed_initial_model_sha256,
        "initial_rng_sha256": observed_initial_rng_sha256,
        "final_model_sha256": final_model_sha256,
        "final_rng_sha256": final_rng_sha256,
        "training_call_telemetry_scope": "OPTIMIZER_GROUPS_ONLY_EXCLUDES_VALIDATION",
        "training_call_telemetry_source": "PRODUCTION_HELPER_RUNTIME_COUNTERS",
        "training_call_telemetry": telemetry,
        "training_step_timing": _timing_summary(step_seconds),
        "training_wall_seconds": training_seconds,
        "training_objective_summary": {
            "count": len(objectives),
            "minimum": float(min(objectives)),
            "maximum": float(max(objectives)),
            "mean": float(statistics.fmean(objectives)),
            "last": float(objectives[-1]),
        },
        "last_optimizer_guard": dict(last_guard),
        "validation_exact_all_chunks": True,
        "validation_runtime_chunks": int(detailed.runtime_chunks_evaluated),
        "validation_call_telemetry": {
            "encoder_forward_calls": int(detailed.encoder_forward_calls),
            "decoder_forward_calls": int(detailed.decoder_forward_calls),
            "global_loss_calls": int(detailed.global_loss_calls),
        },
        "validation_wall_seconds": validation_seconds,
        "validation_production_objective": float(detailed.objective_value),
        "validation_logloss": logloss,
        "candidate_rows": int(detailed.candidate_rows),
        "mean_final_logit_sha256": runtime_value_sha256(detailed.mean_final_logit),
        "runtime_chunk_weights": weights_receipt,
        "runtime_chunk_weights_sha256": _json_sha256(weights_receipt),
        "validation_loss_contract": validation_loss_contract,
        "validation_loss_contract_sha256": _json_sha256(
            validation_loss_contract
        ),
        "precision_contract_sha256": _json_sha256(precision_contract),
        "peak_reserved_bytes": peak_reserved,
        "ordered_candidate_key_sha256": validation_identity[
            "ordered_candidate_key_sha256"
        ],
        "ordered_label_sha256": validation_identity["ordered_label_sha256"],
        "ordered_model_input_sha256": validation_identity[
            "ordered_model_input_sha256"
        ],
        "ordered_mask_weight_sha256": validation_identity[
            "ordered_mask_weight_sha256"
        ],
        "validation_loss_payload_sha256": validation_identity[
            "validation_loss_payload_sha256"
        ],
        "ordered_identity_sha256": validation_identity["ordered_identity_sha256"],
        "fold_role": dict(validation_identity["fold_role"]),
        # Internal tensors are removed before receipt emission.
        "mean_final_logit": detailed.mean_final_logit,
        "proxy_label": detailed.proxy_label,
    }
    del optimizer
    return result


def _public_arm_receipt(arm: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in arm.items()
        if key not in {"mean_final_logit", "proxy_label"}
    }


def _run_variant(
    *,
    pilot_id: str,
    context: Any,
    torch: Any,
    training: Any,
    runtime_value_sha256,
    build_formal_model,
    device: str,
) -> dict[str, Any]:
    variant_started = time.perf_counter()
    ctx = training._context_mapping(context)
    root = Path(ctx["repo_root"])
    config = training._load_mapping(Path(ctx["config_path"]))
    task = training._task_row(Path(ctx["task_manifest_path"]), ctx["task_id"])
    _require_pilot_task(task)
    fold = int(task["patient_fold"])
    seed = int(task["seed"])
    prepared_path = training._resolve_prepared_path(config, fold, root)
    payload, prepared_sha256 = training.load_prepared_artifact_from_authorized_handle(
        torch,
        ctx["input_manifest_path"],
        fold=fold,
        prepared_path=prepared_path,
    )
    if not isinstance(payload, Mapping):
        raise RuntimeError("EQUAL_STEP_PILOT_PREPARED_NOT_MAPPING")
    training._validate_prepared(
        payload, fold=fold, artifact_hashes=ctx["artifact_hashes"]
    )
    observed_variant = training.validate_formal_graph_variant_binding(
        run_id=str(ctx["run_id"]),
        config=config,
        payload=payload,
        prepared_path=prepared_path,
    )
    if observed_variant not in EXPECTED_VARIANTS:
        raise RuntimeError(f"EQUAL_STEP_PILOT_VARIANT_INVALID={observed_variant}")
    runtime = config.get("runtime_profile", {})
    precision = training.resolve_mixed_precision_contract(torch, runtime, device=device)
    runtime_plan = training.resolve_runtime_training_plan(runtime)
    if (
        runtime_plan.candidate_microbatch_size != EXPECTED_BATCH_ROWS
        or runtime_plan.gradient_accumulation != EXPECTED_GROUP_BATCHES
    ):
        raise RuntimeError(
            "EQUAL_STEP_PILOT_PRIMARY_GROUP_CONTRACT_DRIFT="
            f"{runtime_plan.candidate_microbatch_size}x"
            f"{runtime_plan.gradient_accumulation}"
        )
    train_batches = payload["train_batches"]
    if len(train_batches) != EXPECTED_BATCH_COUNT:
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_REQUIRES_B403={observed_variant}:{len(train_batches)}"
        )
    train_row_counts = [training._batch_row_count(batch) for batch in train_batches]
    if train_row_counts[:-1] != [EXPECTED_BATCH_ROWS] * (EXPECTED_BATCH_COUNT - 1):
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_FULL_BATCH_ROW_DRIFT={observed_variant}"
        )
    if not (1 <= train_row_counts[-1] <= EXPECTED_BATCH_ROWS):
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_FINAL_BATCH_ROW_DRIFT={observed_variant}:{train_row_counts[-1]}"
        )
    optimizer_group_count = len(
        training._canonical_optimizer_batch_groups(
            len(train_batches), runtime_plan.gradient_accumulation
        )
    )
    if optimizer_group_count != EXPECTED_OPTIMIZER_STEPS:
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_REQUIRES_101_STEPS={observed_variant}:{optimizer_group_count}"
        )

    bundle = payload["bundle"]
    runtime_chunks, runtime_chunk_weights = training._runtime_chunk_contract(bundle)
    if len(runtime_chunks) != EXPECTED_RUNTIME_CHUNKS:
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_REQUIRES_K29={observed_variant}:{len(runtime_chunks)}"
        )
    if getattr(bundle, "runtime_schedule", None) is None or payload.get("graph") is not None:
        raise RuntimeError("EQUAL_STEP_PILOT_REQUIRES_RUNTIME_SCHEDULE_WITHOUT_FIXED_GRAPH")
    runtime_permutation = training._frozen_chunk_permutation(
        runtime_chunks, seed=seed, fold=fold
    )
    pass_offset_permutation = training._frozen_pass_offset_permutation(
        runtime_chunks, seed=seed, fold=fold
    )

    training_identity = _training_identity(
        payload, runtime_value_sha256=runtime_value_sha256
    )
    validation_identity = _validation_identity(
        payload, runtime_value_sha256=runtime_value_sha256
    )
    input_authority_hashes = training._prepared_input_authority_hashes(payload)
    training._seed_everything(torch, seed)
    model, architecture_id = build_formal_model(
        training, payload, config, device=device
    )
    if architecture_id != "HHGT_FORMAL_CORE_EXTERNAL_ROUTER":
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_REQUIRES_CORE_EXTERNAL_ROUTER={architecture_id}"
        )
    initial_model_state = _clone_state(model)
    initial_model_sha256 = runtime_value_sha256(initial_model_state)
    initial_rng_state = training._capture_torch_rng_state(torch)
    initial_rng_sha256 = runtime_value_sha256(initial_rng_state)

    def graph_for_chunk(chunk: int):
        return training._runtime_graph_for_chunk(bundle, int(chunk), device=device)

    primary = config.get("primary_model", {})
    reference = _execute_arm(
        pilot_id=pilot_id,
        authorization_run_id=str(ctx["run_id"]),
        authorization_task_id=str(ctx["task_id"]),
        variant=observed_variant,
        estimator=REFERENCE_ESTIMATOR,
        model=model,
        initial_model_state=initial_model_state,
        initial_model_sha256=initial_model_sha256,
        initial_rng_state=initial_rng_state,
        initial_rng_sha256=initial_rng_sha256,
        train_batches=train_batches,
        validation_batches=payload["validation_batches"],
        validation_identity=validation_identity,
        bundle=bundle,
        runtime_chunks=runtime_chunks,
        runtime_chunk_weights=runtime_chunk_weights,
        runtime_permutation=runtime_permutation,
        pass_offset_permutation=pass_offset_permutation,
        runtime_plan=runtime_plan,
        precision=precision,
        primary=primary,
        graph_for_chunk=graph_for_chunk,
        torch=torch,
        training=training,
        runtime_value_sha256=runtime_value_sha256,
        device=device,
    )
    proposed = _execute_arm(
        pilot_id=pilot_id,
        authorization_run_id=str(ctx["run_id"]),
        authorization_task_id=str(ctx["task_id"]),
        variant=observed_variant,
        estimator=PROPOSED_ESTIMATOR,
        model=model,
        initial_model_state=initial_model_state,
        initial_model_sha256=initial_model_sha256,
        initial_rng_state=initial_rng_state,
        initial_rng_sha256=initial_rng_sha256,
        train_batches=train_batches,
        validation_batches=payload["validation_batches"],
        validation_identity=validation_identity,
        bundle=bundle,
        runtime_chunks=runtime_chunks,
        runtime_chunk_weights=runtime_chunk_weights,
        runtime_permutation=runtime_permutation,
        pass_offset_permutation=pass_offset_permutation,
        runtime_plan=runtime_plan,
        precision=precision,
        primary=primary,
        graph_for_chunk=graph_for_chunk,
        torch=torch,
        training=training,
        runtime_value_sha256=runtime_value_sha256,
        device=device,
    )
    if not (
        reference["initial_model_sha256"]
        == proposed["initial_model_sha256"]
        == initial_model_sha256
    ) or not (
        reference["initial_rng_sha256"]
        == proposed["initial_rng_sha256"]
        == initial_rng_sha256
    ):
        raise RuntimeError(f"EQUAL_STEP_PILOT_ARM_INIT_DRIFT={observed_variant}")
    comparison = compare_equal_step_arms(reference, proposed)
    identity_after = _validation_identity(
        payload, runtime_value_sha256=runtime_value_sha256
    )
    if identity_after["ordered_identity_sha256"] != validation_identity[
        "ordered_identity_sha256"
    ]:
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_VALIDATION_INPUT_MUTATED={observed_variant}"
        )
    training_identity_after = _training_identity(
        payload, runtime_value_sha256=runtime_value_sha256
    )
    if training_identity_after["ordered_identity_sha256"] != training_identity[
        "ordered_identity_sha256"
    ]:
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_TRAIN_INPUT_MUTATED={observed_variant}"
        )
    elapsed = time.perf_counter() - variant_started
    result = {
        "graph_variant": observed_variant,
        "authorization_run_id": str(ctx["run_id"]),
        "authorization_task_id": str(ctx["task_id"]),
        "authorization_endpoint_id": str(ctx["endpoint_id"]),
        "authorization_hardware_class": str(ctx["hardware_class"]),
        "patient_fold": fold,
        "seed": seed,
        "architecture_id": architecture_id,
        "precision_contract": dict(precision),
        "precision_contract_sha256": _json_sha256(dict(precision)),
        "prepared_path": str(prepared_path),
        "prepared_artifact_sha256": prepared_sha256,
        "authorization_artifact_hashes": dict(ctx["artifact_hashes"]),
        "input_authority_hashes": input_authority_hashes,
        "optimization_contract": _optimization_contract(config),
        "optimization_contract_sha256": _json_sha256(
            _optimization_contract(config)
        ),
        "initial_model_sha256": initial_model_sha256,
        "initial_rng_sha256": initial_rng_sha256,
        "candidate_batch_count": len(train_batches),
        "candidate_batch_row_counts_sha256": _json_sha256(train_row_counts),
        "candidate_rows": int(sum(train_row_counts)),
        "ordered_train_candidate_key_sha256": training_identity[
            "ordered_candidate_key_sha256"
        ],
        "ordered_train_label_sha256": training_identity["ordered_label_sha256"],
        "ordered_train_model_input_sha256": training_identity[
            "ordered_model_input_sha256"
        ],
        "ordered_train_mask_weight_sha256": training_identity[
            "ordered_mask_weight_sha256"
        ],
        "training_loss_payload_sha256": training_identity[
            "training_loss_payload_sha256"
        ],
        "ordered_train_identity_sha256": training_identity[
            "ordered_identity_sha256"
        ],
        "optimizer_group_count": optimizer_group_count,
        "runtime_chunks": list(runtime_chunks),
        "runtime_chunk_permutation": list(runtime_permutation),
        "runtime_chunk_permutation_sha256": _json_sha256(
            list(runtime_permutation)
        ),
        "runtime_chunk_weights": {
            str(key): float(value) for key, value in runtime_chunk_weights.items()
        },
        "ordered_validation_candidate_key_sha256": validation_identity[
            "ordered_candidate_key_sha256"
        ],
        "ordered_validation_label_sha256": validation_identity[
            "ordered_label_sha256"
        ],
        "ordered_validation_model_input_sha256": validation_identity[
            "ordered_model_input_sha256"
        ],
        "ordered_validation_mask_weight_sha256": validation_identity[
            "ordered_mask_weight_sha256"
        ],
        "validation_loss_payload_sha256": validation_identity[
            "validation_loss_payload_sha256"
        ],
        "validation_loss_contract_sha256": reference[
            "validation_loss_contract_sha256"
        ],
        "ordered_validation_identity_sha256": validation_identity[
            "ordered_identity_sha256"
        ],
        "validation_fold_role": validation_identity["fold_role"],
        "validation_candidate_rows": validation_identity["candidate_rows"],
        "reference": _public_arm_receipt(reference),
        "proposed": _public_arm_receipt(proposed),
        "execution_gates": {
            "reference_optimizer_steps_exact_101": reference[
                "training_call_telemetry"
            ]["optimizer_steps"]
            == EXPECTED_OPTIMIZER_STEPS,
            "proposed_optimizer_steps_exact_101": proposed[
                "training_call_telemetry"
            ]["optimizer_steps"]
            == EXPECTED_OPTIMIZER_STEPS,
            "reference_exact_29_chunk_validation": reference[
                "validation_runtime_chunks"
            ]
            == EXPECTED_RUNTIME_CHUNKS,
            "proposed_exact_29_chunk_validation": proposed[
                "validation_runtime_chunks"
            ]
            == EXPECTED_RUNTIME_CHUNKS,
            "same_initial_model": reference["initial_model_sha256"]
            == proposed["initial_model_sha256"]
            == initial_model_sha256,
            "same_initial_rng": reference["initial_rng_sha256"]
            == proposed["initial_rng_sha256"]
            == initial_rng_sha256,
        },
        "comparison": comparison,
        "variant_scientific_pass": False,
        "variant_wall_seconds": elapsed,
    }
    result["variant_scientific_pass"] = bool(
        comparison["pass"] and all(result["execution_gates"].values())
    )
    del reference, proposed, model, payload
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _projection_from_observed_timing(
    variants: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    step_seconds = {
        variant: max(
            float(variants[variant]["proposed"]["training_step_timing"]["mean_seconds"]),
            float(variants[variant]["proposed"]["training_step_timing"]["p95_seconds"]),
        )
        for variant in EXPECTED_VARIANTS
    }
    validation_g2 = max(
        float(variants["G2"][arm]["validation_wall_seconds"])
        for arm in ("reference", "proposed")
    )
    setup_seconds = {
        variant: max(
            0.0,
            float(variants[variant]["variant_wall_seconds"])
            - float(variants[variant]["reference"]["training_wall_seconds"])
            - float(variants[variant]["reference"]["validation_wall_seconds"])
            - float(variants[variant]["proposed"]["training_wall_seconds"])
            - float(variants[variant]["proposed"]["validation_wall_seconds"]),
        )
        for variant in EXPECTED_VARIANTS
    }
    core = math.fsum(
        5
        * 6
        * (
            EXPECTED_OPTIMIZER_STEPS * step_seconds[variant]
            + validation_g2
            + 30.0
        )
        for variant in EXPECTED_VARIANTS
    )
    core += math.fsum(5 * setup_seconds[variant] for variant in EXPECTED_VARIANTS)
    core += 1800.0
    upper_seconds = 1.25 * core
    return {
        "source": "OBSERVED_EQUAL_STEP_PILOT_TIMING_INFORMATIONAL_ONLY",
        "authoritative_paid_time_gate": False,
        "still_requires_separate_bounded_probe_receipt": True,
        "shared_step_seconds_by_variant_max_of_mean_and_p95": step_seconds,
        "g2_exact_validation_seconds_upper": validation_g2,
        "setup_seconds_by_variant": setup_seconds,
        "formula": (
            "1.25*[sum_v 5*6*(101*step_seconds_v+validation_seconds_G2+30)"
            "+sum_v 5*setup_seconds_v+1800]"
        ),
        "upper_seconds": upper_seconds,
        "upper_hours": upper_seconds / 3600.0,
        "within_50_hours_informational": upper_seconds / 3600.0 <= 50.0,
    }


def _cross_variant_comparison_gates(
    variants: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    """Fail closed on every preregistered cross-variant fairness invariant.

    G0/G1/G2 intentionally differ in graph edges, but share the node universe,
    complete relation schema, model constructor, seed and optimizer contract.
    Therefore their initial parameter/RNG states must be identical; a mismatch
    indicates architecture or RNG-consumption drift rather than a legitimate
    graph ablation.
    """

    if tuple(sorted(variants)) != EXPECTED_VARIANTS:
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_CROSS_VARIANT_SET_DRIFT={tuple(sorted(variants))}"
        )
    exact_fields = (
        "architecture_id",
        "initial_model_sha256",
        "initial_rng_sha256",
        "candidate_batch_row_counts_sha256",
        "candidate_rows",
        "ordered_train_candidate_key_sha256",
        "ordered_train_label_sha256",
        "ordered_train_model_input_sha256",
        "ordered_train_mask_weight_sha256",
        "training_loss_payload_sha256",
        "ordered_train_identity_sha256",
        "runtime_chunks",
        "runtime_chunk_permutation_sha256",
        "precision_contract_sha256",
        "ordered_validation_candidate_key_sha256",
        "ordered_validation_label_sha256",
        "ordered_validation_model_input_sha256",
        "ordered_validation_mask_weight_sha256",
        "validation_loss_payload_sha256",
        "validation_loss_contract_sha256",
        "ordered_validation_identity_sha256",
        "validation_fold_role",
        "validation_candidate_rows",
        "optimization_contract_sha256",
    )
    missing = {
        variant: sorted(field for field in exact_fields if field not in variants[variant])
        for variant in EXPECTED_VARIANTS
    }
    missing = {variant: fields for variant, fields in missing.items() if fields}
    if missing:
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_CROSS_VARIANT_IDENTITY_FIELD_MISSING={missing}"
        )
    gates = {
        field: len(
            {
                json.dumps(variants[variant][field], sort_keys=True)
                for variant in EXPECTED_VARIANTS
            }
        )
        == 1
        for field in exact_fields
    }
    for arm in ("reference", "proposed"):
        missing_arm = {
            variant: sorted(
                field
                for field in ("schedule_sha256", "runtime_chunk_weights_sha256")
                if field not in variants[variant].get(arm, {})
            )
            for variant in EXPECTED_VARIANTS
        }
        missing_arm = {
            variant: fields for variant, fields in missing_arm.items() if fields
        }
        if missing_arm:
            raise RuntimeError(
                "EQUAL_STEP_PILOT_CROSS_VARIANT_ARM_FIELD_MISSING="
                f"{arm}:{missing_arm}"
            )
        gates[f"{arm}_schedule_byte_identical"] = len(
            {variants[variant][arm]["schedule_sha256"] for variant in EXPECTED_VARIANTS}
        ) == 1
        gates[f"{arm}_chunk_weights_byte_identical"] = len(
            {
                variants[variant][arm]["runtime_chunk_weights_sha256"]
                for variant in EXPECTED_VARIANTS
            }
        ) == 1
    return gates


def _run_verified_equal_step_pilot(
    authorization: EqualStepPilotAuthorizationContext,
) -> dict[str, Any]:
    import torch

    from . import training
    from .gpu_backward_probe import _build_formal_model, _resolve_gpu_identity
    from .group_shared_encoder_oracle import _runtime_value_sha256

    if not torch.cuda.is_available():
        raise RuntimeError("REAL_EQUAL_STEP_PILOT_REQUIRES_CUDA")
    device = "cuda"
    gpu_identity = _resolve_gpu_identity(torch)
    started = time.perf_counter()
    protocol_path = (
        Path(authorization.g0.repo_root)
        / "docs"
        / "v32_group_shared_encoder_estimator_preregistration_20260901.md"
    )
    runner_path = Path(__file__).resolve()
    training_path = Path(training.__file__).resolve()
    if not protocol_path.is_file():
        raise RuntimeError(f"EQUAL_STEP_PILOT_PREREGISTRATION_MISSING={protocol_path}")
    frozen_hashes = {
        "preregistration_sha256": training._file_sha256(protocol_path),
        "runner_sha256": training._file_sha256(runner_path),
        "production_training_sha256": training._file_sha256(training_path),
    }
    variants: dict[str, dict[str, Any]] = {}
    for expected_variant, context in zip(
        EXPECTED_VARIANTS, authorization.contexts, strict=True
    ):
        _emit_receipt(
            {
                **_base_receipt("EQUAL_STEP_VARIANT_SETUP_START"),
                "pilot_id": authorization.pilot_id,
                "authorization_run_id": str(context.run_id),
                "authorization_task_id": str(context.task_id),
                "graph_variant": expected_variant,
                "arm": "none",
                "phase": "VARIANT_SETUP",
                "completed": 0,
                "total": 1,
                "progress_event_id": (
                    f"{authorization.pilot_id}|{context.run_id}|{context.task_id}|"
                    f"{expected_variant}|none|VARIANT_SETUP|0"
                ),
                "optimizer_progress_observed": False,
                "training_started": False,
            }
        )
        result = _run_variant(
            pilot_id=authorization.pilot_id,
            context=context,
            torch=torch,
            training=training,
            runtime_value_sha256=_runtime_value_sha256,
            build_formal_model=_build_formal_model,
            device=device,
        )
        variant = result["graph_variant"]
        if variant in variants:
            raise RuntimeError(f"EQUAL_STEP_PILOT_DUPLICATE_VARIANT={variant}")
        variants[variant] = result
        _emit_receipt(
            {
                **_base_receipt("EQUAL_STEP_VARIANT_HEARTBEAT"),
                "pilot_id": authorization.pilot_id,
                "authorization_run_id": result["authorization_run_id"],
                "authorization_task_id": result["authorization_task_id"],
                "graph_variant": variant,
                "arm": "both",
                "phase": "VARIANT_COMPLETE",
                "completed": 1,
                "total": 1,
                "progress_event_id": (
                    f"{authorization.pilot_id}|{result['authorization_run_id']}|"
                    f"{result['authorization_task_id']}|{variant}|both|"
                    "VARIANT_COMPLETE|1"
                ),
                "variant_scientific_pass": result["variant_scientific_pass"],
                "reference_optimizer_steps": result["reference"][
                    "training_call_telemetry"
                ]["optimizer_steps"],
                "proposed_optimizer_steps": result["proposed"][
                    "training_call_telemetry"
                ]["optimizer_steps"],
            }
        )
    if tuple(sorted(variants)) != EXPECTED_VARIANTS:
        raise RuntimeError(
            f"EQUAL_STEP_PILOT_COMPLETED_VARIANTS_DRIFT={tuple(sorted(variants))}"
        )

    cross_variant_gates = _cross_variant_comparison_gates(variants)
    cross_variant_pass = bool(all(cross_variant_gates.values()))
    per_variant_pass = {
        variant: bool(variants[variant]["variant_scientific_pass"])
        for variant in EXPECTED_VARIANTS
    }
    scientific_pass = bool(all(per_variant_pass.values()) and cross_variant_pass)
    status = PASS_STATUS if scientific_pass else SCIENTIFIC_FAIL_STATUS

    current_hashes = {
        "preregistration_sha256": training._file_sha256(protocol_path),
        "runner_sha256": training._file_sha256(runner_path),
        "production_training_sha256": training._file_sha256(training_path),
    }
    if current_hashes != frozen_hashes:
        raise RuntimeError("EQUAL_STEP_PILOT_CODE_OR_PROTOCOL_CHANGED_DURING_RUN")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        **_base_receipt(status),
        "scientific_pass": scientific_pass,
        "pilot_id": authorization.pilot_id,
        "patient_fold": EXPECTED_FOLD,
        "seed": EXPECTED_SEED,
        "graph_variants": list(EXPECTED_VARIANTS),
        "reference_estimator": REFERENCE_ESTIMATOR,
        "proposed_estimator": PROPOSED_ESTIMATOR,
        "required_optimizer_steps_per_arm": EXPECTED_OPTIMIZER_STEPS,
        "required_runtime_chunks": EXPECTED_RUNTIME_CHUNKS,
        "thresholds": {
            "validation_logloss_absolute_difference_max": (
                VALIDATION_LOGLOSS_ABSOLUTE_DIFFERENCE_MAX
            ),
            "mean_logit_pearson_min": MEAN_LOGIT_PEARSON_MIN,
            "mean_logit_spearman_min": MEAN_LOGIT_SPEARMAN_MIN,
            "ordered_candidate_keys_labels_fold_role_chunk_weights_exact": True,
        },
        "per_variant_pass": per_variant_pass,
        "cross_variant_gates": cross_variant_gates,
        "cross_variant_pass": cross_variant_pass,
        "variants": variants,
        "observed_timing_projection": _projection_from_observed_timing(variants),
        "gpu_identity": gpu_identity,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "elapsed_seconds": elapsed,
        **frozen_hashes,
    }


def run_authorized_equal_step_candidate_pilot(context: Any) -> int:
    """Reverify all three authorities, run, and emit only typed stdout JSON."""

    if type(context) is not EqualStepPilotAuthorizationContext:
        raise RuntimeError(
            "EQUAL_STEP_PILOT_REQUIRES_EQUAL_STEP_AUTHORIZATION_CONTEXT"
        )
    try:
        verified_contexts = tuple(_reverify_context(item) for item in context.contexts)
        verified = build_equal_step_pilot_authorization(
            verified_contexts, pilot_id=context.pilot_id
        )
        if verified != context:
            raise RuntimeError("EQUAL_STEP_PILOT_COMPOSITE_CONTEXT_DRIFT")
        receipt = _run_verified_equal_step_pilot(verified)
    except Exception as exc:
        receipt = {
            **_base_receipt(RUNTIME_FAIL_STATUS),
            "scientific_pass": False,
            "pilot_id": str(getattr(context, "pilot_id", "UNKNOWN")),
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }
        _emit_receipt(receipt)
        return NONZERO_RETURN_CODE
    _emit_receipt(receipt)
    return 0 if receipt.get("scientific_pass") is True else NONZERO_RETURN_CODE


__all__ = [
    "AUTHORIZED_TRAINER_SPECIFICATION",
    "EqualStepPilotAuthorizationContext",
    "build_equal_step_pilot_authorization",
    "compare_equal_step_arms",
    "run_authorized_equal_step_candidate_pilot",
]
