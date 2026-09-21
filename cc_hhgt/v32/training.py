"""Authorized V3.2 training loop.

This module is never imported by the CODE_ONLY path.  The CLI imports it only
after :func:`cc_hhgt.v32.training_guard.guard_training_entry` validates an
explicit approval whose hashes cover this file, the config, inputs and task
manifest.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PREPARED_FORMAT = "CC_HHGT_V3_2_PREPARED_FOLD_V2_SEALED_TEST"
LEGACY_CHECKPOINT_FORMAT_V3 = "CC_HHGT_V3_2_FULL_TRAINING_STATE_V3_STREAMING_BF16"
CHECKPOINT_FORMAT = "CC_HHGT_V3_2_FULL_TRAINING_STATE_V4_GROUP_SCHEDULE"
AUTHORIZED_FORMAL_TRAINER_SPECIFICATION = (
    "cc_hhgt.v32.training:run_authorized_task"
)
CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN = "exact_cartesian_v1"
CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS = (
    "balanced_cyclic_single_pass_v1"
)
CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER = (
    "balanced_group_latin_shared_encoder_v1"
)
_CANDIDATE_CHUNK_SCHEDULE_MODES = frozenset(
    {
        CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN,
        CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS,
        CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER,
    }
)
TWO_PASS_STREAMING_GRADIENT_ALGORITHM = "TWO_PASS_GLOBAL_LOSS_STREAMING_BACKWARD_V1"
SHARED_ENCODER_GRADIENT_ALGORITHM = (
    "SINGLE_SHARED_ENCODER_CONVENTIONAL_GROUP_BACKWARD_V1"
)


@dataclass(frozen=True)
class RuntimeTrainingPlan:
    """Effective candidate-memory plan for one authorized training task."""

    candidate_microbatch_size: int
    gradient_accumulation: int
    fallback_microbatch_size: int
    fallback_gradient_accumulation: int


@dataclass(frozen=True)
class GlobalLossPlan:
    """Frozen denominators and nnPU branch for exact streaming backward."""

    positive_weight_denominator: float
    unlabeled_count: int
    direction_count: int
    residual_count: int
    nnpu_correction_active: bool
    objective_value: float


@dataclass(frozen=True)
class BackwardCallTelemetry:
    """Observable model-call budget for one successful optimizer group."""

    unique_chunks: int
    group_rows: int
    encoder_forward_calls: int
    decoder_forward_calls: int
    global_loss_calls: int
    backward_calls: int


@dataclass(frozen=True)
class RuntimeCoverageEvaluation:
    """Exact all-chunk validation result exposed without changing its estimand.

    ``mean_final_logit`` and ``proxy_label`` stay on CPU in canonical candidate
    order.  They are needed by comparison-only scientific gates that must
    compute ordinary validation logloss and logit correlations from the same
    exact runtime-graph aggregation used by formal training.  Formal training
    continues to consume only ``objective_value`` through the compatibility
    wrapper below.
    """

    objective_value: float
    mean_final_logit: Any
    proxy_label: Any
    candidate_rows: int
    encoder_forward_calls: int
    decoder_forward_calls: int
    global_loss_calls: int
    runtime_chunks_evaluated: int


def _context_mapping(context: Any) -> dict[str, Any]:
    if is_dataclass(context):
        return asdict(context)
    if isinstance(context, Mapping):
        return dict(context)
    raise TypeError("Authorized training context must be a dataclass or mapping")


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Expected a mapping in {path}")
    return dict(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authorized_prepared_record(
    input_manifest_path: str | Path,
    *,
    fold: int,
    prepared_path: str | Path,
) -> tuple[Path, str]:
    """Resolve the one authorized fold row without opening the large artifact."""

    manifest_path = Path(input_manifest_path).resolve()
    prepared = Path(prepared_path).resolve()
    manifest = _load_mapping(manifest_path)
    fold_inputs = manifest.get("fold_inputs")
    if not isinstance(fold_inputs, list):
        raise RuntimeError("Authorized input manifest lacks fold_inputs")
    matches = [
        item
        for item in fold_inputs
        if isinstance(item, Mapping) and int(item.get("fold", -1)) == int(fold)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Authorized input manifest expected one fold {fold} row, observed {len(matches)}"
        )
    row = matches[0]
    registered_path = Path(str(row.get("path", ""))).resolve()
    if registered_path != prepared:
        raise RuntimeError(
            "Prepared fold path differs from the authorized input manifest: "
            f"registered={registered_path}, resolved={prepared}"
        )
    expected = str(row.get("sha256", "")).lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise RuntimeError(f"Authorized fold {fold} SHA256 is missing or malformed")
    if not prepared.is_file():
        raise RuntimeError(f"Prepared fold artifact is missing: {prepared}")
    return prepared, expected


def _stable_file_identity(stat_result: os.stat_result) -> tuple[int, ...]:
    """Return fields that must remain stable while an authorized PT is consumed."""

    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_mode),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def load_prepared_artifact_from_authorized_handle(
    torch,
    input_manifest_path: str | Path,
    *,
    fold: int,
    prepared_path: str | Path,
) -> tuple[Any, str]:
    """Hash and deserialize the authorized PT through one stable file handle.

    Opening once prevents a rename/replacement between the authorization hash
    check and ``torch.load``.  Repeated ``fstat`` checks also fail closed when
    the opened inode is modified in place while it is being consumed.
    """

    prepared, expected = _authorized_prepared_record(
        input_manifest_path,
        fold=fold,
        prepared_path=prepared_path,
    )
    digest = hashlib.sha256()
    with prepared.open("rb") as handle:
        before = _stable_file_identity(os.fstat(handle.fileno()))
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after_hash = _stable_file_identity(os.fstat(handle.fileno()))
        if after_hash != before:
            raise RuntimeError(
                f"Prepared fold artifact changed while hashing fold {fold}"
            )
        observed = digest.hexdigest()
        if observed != expected:
            raise RuntimeError(
                f"Prepared fold SHA256 differs from authorization for fold {fold}: "
                f"expected={expected}, observed={observed}"
            )
        handle.seek(0)
        use_mmap = os.environ.get("V32_PREPARED_ARTIFACT_MMAP", "0") == "1"
        if use_mmap:
            if os.name != "posix" or not Path("/proc/self/fd").is_dir():
                raise RuntimeError(
                    "V32_PREPARED_ARTIFACT_MMAP requires Linux /proc/self/fd"
                )
            # Resolve through the already-authorized open descriptor.  This
            # retains the single-inode/TOCTOU guarantee while letting Torch
            # map tensor storages lazily instead of duplicating a multi-GB PT
            # artifact in anonymous host RAM during deserialization.
            descriptor_path = f"/proc/self/fd/{handle.fileno()}"
            payload = torch.load(
                descriptor_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        else:
            payload = torch.load(handle, map_location="cpu", weights_only=False)
        after_load = _stable_file_identity(os.fstat(handle.fileno()))
        if after_load != before:
            raise RuntimeError(
                f"Prepared fold artifact changed while loading fold {fold}"
            )
    return payload, observed


def validate_prepared_artifact_against_input_manifest(
    input_manifest_path: str | Path,
    *,
    fold: int,
    prepared_path: str | Path,
) -> str:
    """Compatibility hash-only gate; formal loading uses the handle-bound API."""

    prepared, expected = _authorized_prepared_record(
        input_manifest_path,
        fold=fold,
        prepared_path=prepared_path,
    )
    observed = _file_sha256(prepared)
    if observed != expected:
        raise RuntimeError(
            f"Prepared fold SHA256 differs from authorization for fold {fold}: "
            f"expected={expected}, observed={observed}"
        )
    return observed


def _task_row(task_manifest: Path, task_id: str) -> dict[str, str]:
    from .orchestration import read_task_manifest

    rows = [row for row in read_task_manifest(task_manifest) if row["task_id"] == task_id]
    if len(rows) != 1:
        raise RuntimeError(f"Prepared trainer expected one task row, observed {len(rows)}")
    return rows[0]


def _resolve_prepared_path(config: Mapping[str, Any], fold: int, root: Path) -> Path:
    io = config.get("training_io", {})
    pattern = io.get("prepared_fold_pattern", "artifacts/prepared/PATIENT_FOLD_{fold}.pt")
    path = Path(str(pattern).format(fold=int(fold)))
    return path if path.is_absolute() else root / path


def _resolve_output_root(config: Mapping[str, Any], root: Path) -> Path:
    value = Path(str(config.get("training_io", {}).get("output_root", "artifacts/training")))
    return value if value.is_absolute() else root / value


def validate_formal_graph_variant_binding(
    *,
    run_id: str,
    config: Mapping[str, Any],
    payload: Mapping[str, Any],
    prepared_path: Path,
) -> str:
    """Bind the formal G0/G1/G2 identity across every execution authority.

    The graph authority embedded in a prepared payload proves that the payload
    is internally consistent.  It does not by itself prove that a G0 run was
    not accidentally pointed at a valid G1/G2 payload.  Keep that cross-source
    identity check in the production trainer as well as the bounded probe.
    """

    expected = str(config.get("task_contract", {}).get("graph_variant", ""))
    if expected not in {"G0", "G1", "G2"}:
        raise RuntimeError(f"FORMAL_CONFIG_GRAPH_VARIANT_INVALID={expected!r}")
    observed = {
        "run_id": str(run_id),
        "prepared_parent": prepared_path.resolve().parent.name,
        "payload": str(payload.get("formal_graph_variant", "")),
    }
    drift: dict[str, str] = {}
    if not str(run_id).lower().startswith(f"v32-g012-{expected.lower()}-"):
        drift["run_id"] = observed["run_id"]
    if observed["prepared_parent"] != expected:
        drift["prepared_parent"] = observed["prepared_parent"]
    if observed["payload"] != expected:
        drift["payload"] = observed["payload"]
    if drift:
        raise RuntimeError(
            "FORMAL_GRAPH_VARIANT_BINDING_DRIFT: "
            f"expected={expected}, observed={drift}"
        )
    return expected


def _validate_prepared(payload: Mapping[str, Any], *, fold: int, artifact_hashes: Mapping[str, str]) -> None:
    if payload.get("prepared_format") != PREPARED_FORMAT:
        raise RuntimeError("Prepared fold format mismatch")
    if int(payload.get("patient_fold", -1)) != int(fold):
        raise RuntimeError("Prepared fold ID does not match the task")
    required = {
        "bundle",
        "feature_dim",
        "legacy_model_config",
        "train_batches",
        "validation_batches",
        "label_contract",
    }
    if missing := sorted(required - set(payload)):
        raise RuntimeError(f"Prepared fold payload lacks fields: {missing}")
    try:
        from .patient_fold_authority import (
            validate_frozen_v32_patient_fold_payload_binding,
        )

        validate_frozen_v32_patient_fold_payload_binding(
            payload.get("patient_fold_authority")
        )
    except Exception as exc:
        raise RuntimeError(
            "Prepared fold lacks the frozen patient-first authority binding"
        ) from exc
    try:
        from .formal_graph_authority import validate_formal_graph_payload_binding

        validate_formal_graph_payload_binding(
            payload.get("formal_graph_authority"),
            outer_fold=fold,
            variant=str(payload.get("formal_graph_variant", "")),
            bundle=payload.get("bundle"),
        )
    except Exception as exc:
        raise RuntimeError(
            "Prepared fold lacks a fresh fold-local G0/G1/G2 graph authority binding"
        ) from exc
    registered = payload.get("artifact_hashes")
    if registered is not None:
        drift = {
            key: {"prepared": registered.get(key), "authorized": artifact_hashes.get(key)}
            for key in artifact_hashes
            if key in registered and registered.get(key) != artifact_hashes.get(key)
        }
        if drift:
            raise RuntimeError(f"Prepared fold authorization hash mismatch: {drift}")
    forbidden_test_keys = sorted(
        key
        for key in payload
        if str(key).lower().startswith("test")
        or str(key).lower() in {"heldout_test_labels", "sealed_test_labels"}
    )
    if forbidden_test_keys:
        raise RuntimeError(
            "Training payload crosses the sealed-test firewall: "
            f"{forbidden_test_keys}"
        )
    label_contract = payload["label_contract"]
    if not isinstance(label_contract, Mapping):
        raise RuntimeError("Prepared label_contract must be a mapping")
    fixed_label_contract = {
        "heldout_direction_is_model_input": False,
        "test_labels_in_training_payload": False,
        "test_logits_in_training_payload": False,
        "test_metrics_computed_before_winner_lock": False,
    }
    drift = {
        key: {"prepared": label_contract.get(key), "required": value}
        for key, value in fixed_label_contract.items()
        if label_contract.get(key) != value
    }
    direction_sources = (
        label_contract.get("train_direction_label_source"),
        label_contract.get("validation_direction_label_source"),
    )
    allowed_direction_sources = {
        ("train_discovery_effect", "validation_replication_effect"),
        (
            "train_local_cnv_adjusted_effect",
            "validation_local_cnv_adjusted_replication_effect",
        ),
    }
    if direction_sources not in allowed_direction_sources:
        drift["direction_label_sources"] = {
            "prepared": direction_sources,
            "required_one_of": sorted(allowed_direction_sources),
        }
    local_cnv_adjusted = direction_sources == (
        "train_local_cnv_adjusted_effect",
        "validation_local_cnv_adjusted_replication_effect",
    )
    if local_cnv_adjusted and (
        label_contract.get("association_target") != "A1_LOCAL"
        or label_contract.get("association_available_mask")
        != "association_available"
    ):
        drift["local_cnv_adjustment"] = {
            "association_target": label_contract.get("association_target"),
            "association_available_mask": label_contract.get(
                "association_available_mask"
            ),
        }
    if drift:
        raise RuntimeError(f"Prepared label/test governance contract is unsafe: {drift}")
    for split in ("train_batches", "validation_batches"):
        batches = payload[split]
        if not isinstance(batches, list) or not batches:
            raise RuntimeError(f"{split} must be a non-empty list")
        required_batch = {
            "candidate_batch",
            "base_logit",
            "conservation_context",
            "proxy_label",
            "weak_positive",
            "direction_label",
            "direction_available",
        }
        if local_cnv_adjusted:
            required_batch.add("association_available")
        for index, batch in enumerate(batches):
            if missing := sorted(required_batch - set(batch)):
                raise RuntimeError(f"{split}[{index}] lacks fields: {missing}")
            association_available = batch.get("association_available")
            if association_available is not None:
                rows = _batch_row_count(batch)
                if (
                    not hasattr(association_available, "shape")
                    or int(association_available.shape[0]) != rows
                ):
                    raise RuntimeError(
                        f"{split}[{index}] association_available is not row-aligned"
                    )
            candidate = batch.get("candidate_batch")
            if not isinstance(candidate, Mapping):
                raise RuntimeError(f"{split}[{index}] candidate_batch must be a mapping")
            forbidden_features = {
                "replication_effect",
                "replication_direction_label",
                "test_effect",
                "test_label",
            } & set(candidate)
            if forbidden_features:
                raise RuntimeError(
                    f"{split}[{index}] exposes held-out labels as model inputs: "
                    f"{sorted(forbidden_features)}"
                )


def _prepared_input_authority_hashes(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable data/graph authority carried by a prepared fold.

    The 2026-08-30 R3 materializer used ``artifact_hashes`` for its input
    authority (patient map, graph receipt and source configuration).  The
    guarded trainer also uses the name ``artifact_hashes``, but for a different
    authority: code/config/input-manifest/task-manifest hashes computed at
    training launch.  Keep both lineages explicitly instead of pretending
    those independently generated mappings are equal.
    """

    value = payload.get("input_authority_hashes", payload.get("artifact_hashes"))
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError("Prepared fold lacks immutable input-authority hashes")
    return {str(key): item for key, item in value.items()}


def validate_hierarchical_modality_contract(
    payload: Mapping[str, Any],
    modality_names: tuple[str, ...],
) -> None:
    """Require patient-level OOF modalities before end-to-end gating."""

    contract = payload.get("modality_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("Hierarchical HHGT requires modality_contract")
    if tuple(contract.get("modality_names", ())) != tuple(modality_names):
        raise RuntimeError("Hierarchical modality order differs from the prepared contract")
    required = {
        "source_scope": "PATIENT_LEVEL_OOF",
        "patient_folds": 5,
        "old_predictions_used": False,
        "missing_values_typed_unavailable": True,
    }
    drift = {key: (contract.get(key), value) for key, value in required.items() if contract.get(key) != value}
    if drift:
        raise RuntimeError(f"Hierarchical modality contract is unsafe: {drift}")
    for split in ("train_batches", "validation_batches"):
        for index, batch in enumerate(payload[split]):
            candidate = batch.get("candidate_batch")
            if not isinstance(candidate, Mapping):
                raise RuntimeError(f"{split}[{index}] candidate_batch is malformed")
            missing = {"modality_probability", "modality_available"} - set(candidate)
            if missing:
                raise RuntimeError(f"{split}[{index}] lacks hierarchical modalities: {sorted(missing)}")


def _move(value: Any, device: str):
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


def _atomic_torch_save(torch, payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _runtime_fingerprint(
    torch,
    hardware_class: str,
    device: str,
    *,
    seed: int,
    mixed_precision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import numpy as np

    record = {
        "hardware_class": str(hardware_class),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "python": str(sys.version),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(),
        "seed": int(seed),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "mixed_precision": dict(mixed_precision or {}),
    }
    if str(device).startswith("cuda"):
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        record.update(
            {
                "cuda_device_name": properties.name,
                "cuda_total_memory": int(properties.total_memory),
                "cuda_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return record


def _seed_everything(torch, seed: int) -> None:
    """Seed the complete V3.2 training runtime before model construction."""

    import numpy as np

    resolved = int(seed)
    random.seed(resolved)
    np.random.seed(resolved % (2**32))
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # PyG/torch-scatter may warn for a CUDA kernel without a deterministic
    # implementation.  The warning is preserved in the log while all kernels
    # with a deterministic implementation are forced onto that path.
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_mixed_precision_contract(
    torch,
    runtime: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    """Resolve the configured precision without requiring CUDA in CPU tests."""

    raw = runtime.get("mixed_precision", False)
    if raw is False or raw is None or str(raw).strip().lower() in {
        "false",
        "off",
        "none",
        "fp32",
        "float32",
    }:
        requested = "fp32"
    elif raw is True or str(raw).strip().lower() in {"true", "bf16", "bfloat16"}:
        requested = "bf16"
    else:
        raise RuntimeError(f"Unsupported runtime_profile.mixed_precision={raw!r}")
    device_type = torch.device(device).type
    enabled = requested == "bf16" and device_type == "cuda"
    if enabled:
        supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(supported) and not bool(supported()):
            raise RuntimeError("Configured bf16 autocast is unsupported by the CUDA device")
    return {
        "requested": requested,
        "active": "bf16" if enabled else "fp32",
        "autocast_enabled": enabled,
        "device_type": device_type,
        "cpu_test_fallback": bool(requested == "bf16" and device_type == "cpu"),
    }


def _autocast_context(torch, device: str, precision: Mapping[str, Any]):
    if not bool(precision.get("autocast_enabled", False)):
        return nullcontext()
    device_type = torch.device(device).type
    if device_type != "cuda" or precision.get("active") != "bf16":
        raise RuntimeError("Invalid mixed-precision execution contract")
    try:
        return torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        )
    except AttributeError:  # pragma: no cover - old supported PyTorch builds
        return torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True)


def resolve_runtime_training_plan(runtime: Mapping[str, Any]) -> RuntimeTrainingPlan:
    """Validate the primary and OOM retry microbatch capacities.

    Runtime graphs are immutable prepared inputs.  An edge-chunk fallback
    cannot be applied here without changing the registered graph estimand, so
    the obsolete option is rejected instead of being silently ignored.
    """

    if "oom_fallback_edge_chunk_size" in runtime:
        raise RuntimeError(
            "runtime_profile.oom_fallback_edge_chunk_size is unsupported for "
            "sealed prepared graphs; rematerialize the graph instead"
        )
    plan = RuntimeTrainingPlan(
        candidate_microbatch_size=int(runtime.get("candidate_microbatch_size", 8192)),
        gradient_accumulation=int(runtime.get("gradient_accumulation", 4)),
        fallback_microbatch_size=int(
            runtime.get(
                "oom_fallback_microbatch_size",
                runtime.get("candidate_microbatch_size", 8192),
            )
        ),
        fallback_gradient_accumulation=int(
            runtime.get(
                "oom_fallback_gradient_accumulation",
                runtime.get("gradient_accumulation", 4),
            )
        ),
    )
    values = (
        plan.candidate_microbatch_size,
        plan.gradient_accumulation,
        plan.fallback_microbatch_size,
        plan.fallback_gradient_accumulation,
    )
    if min(values) < 1:
        raise RuntimeError("Runtime microbatch and accumulation values must be positive")
    if plan.fallback_microbatch_size > plan.candidate_microbatch_size:
        raise RuntimeError("OOM fallback microbatch must not exceed the primary microbatch")
    primary_capacity = plan.candidate_microbatch_size * plan.gradient_accumulation
    fallback_capacity = (
        plan.fallback_microbatch_size * plan.fallback_gradient_accumulation
    )
    if fallback_capacity < primary_capacity:
        raise RuntimeError(
            "OOM fallback capacity must preserve one primary optimizer group's row capacity"
        )
    return plan


def _batch_row_count(batch: Mapping[str, Any]) -> int:
    base = batch.get("base_logit")
    if not hasattr(base, "shape") or getattr(base, "ndim", 0) < 1:
        raise RuntimeError("Training batch base_logit must have a row dimension")
    rows = int(base.shape[0])
    if rows < 1:
        raise RuntimeError("Training batch cannot be empty")
    return rows


def _slice_tensor_rows(value: Any, start: int, stop: int, rows: int, label: str):
    if not hasattr(value, "shape") or getattr(value, "ndim", 0) < 1:
        raise RuntimeError(f"Row-aligned training field is not a tensor: {label}")
    if int(value.shape[0]) != rows:
        raise RuntimeError(
            f"Row-aligned training field has the wrong length: {label}="
            f"{int(value.shape[0])}!={rows}"
        )
    return value[start:stop]


def slice_training_batch_rows(
    batch: Mapping[str, Any], start: int, stop: int
) -> dict[str, Any]:
    """Slice every candidate/label field without guessing on scalar metadata."""

    rows = _batch_row_count(batch)
    if not (0 <= start < stop <= rows):
        raise ValueError(f"Invalid training row slice {start}:{stop} for {rows}")
    result = dict(batch)
    candidate = batch.get("candidate_batch")
    if not isinstance(candidate, Mapping):
        raise RuntimeError("Training candidate_batch must be a mapping")
    result["candidate_batch"] = {
        key: _slice_tensor_rows(value, start, stop, rows, f"candidate_batch.{key}")
        for key, value in candidate.items()
    }
    for key in (
        "base_logit",
        "conservation_context",
        "graph_available",
        "proxy_label",
        "weak_positive",
        "association_available",
        "direction_label",
        "direction_available",
    ):
        if key in batch:
            result[key] = _slice_tensor_rows(batch[key], start, stop, rows, key)
    return result


def materialize_pass_work_items(
    pass_pairs: Sequence[tuple[int, int]],
    train_batches: Sequence[Mapping[str, Any]],
    *,
    microbatch_size: int,
) -> list[tuple[dict[str, Any], int]]:
    """Expand frozen batch/chunk pairs into deterministic row microbatches."""

    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    work_items: list[tuple[dict[str, Any], int]] = []
    for batch_index, chunk in pass_pairs:
        raw = train_batches[int(batch_index)]
        rows = _batch_row_count(raw)
        for start in range(0, rows, microbatch_size):
            work_items.append(
                (
                    slice_training_batch_rows(
                        raw, start, min(start + microbatch_size, rows)
                    ),
                    int(chunk),
                )
            )
    return work_items


def split_batches_for_evaluation(
    batches: Sequence[Mapping[str, Any]], *, microbatch_size: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in batches:
        rows = _batch_row_count(raw)
        for start in range(0, rows, microbatch_size):
            result.append(
                slice_training_batch_rows(
                    raw, start, min(start + microbatch_size, rows)
                )
            )
    return result


def split_work_items_for_oom_fallback(
    work_items: Sequence[tuple[Mapping[str, Any], int]],
    *,
    microbatch_size: int,
    maximum_fragments: int,
) -> list[tuple[dict[str, Any], int]]:
    """Split one primary optimizer group while preserving its single step."""

    fragments: list[tuple[dict[str, Any], int]] = []
    for raw, chunk in work_items:
        rows = _batch_row_count(raw)
        for start in range(0, rows, microbatch_size):
            fragments.append(
                (
                    slice_training_batch_rows(
                        raw, start, min(start + microbatch_size, rows)
                    ),
                    int(chunk),
                )
            )
    if len(fragments) > int(maximum_fragments):
        raise RuntimeError(
            "OOM fallback would change optimizer-step semantics: "
            f"fragments={len(fragments)}, configured_accumulation={maximum_fragments}"
        )
    return fragments


def _capture_torch_rng_state(torch) -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_torch_rng_state(torch, state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["cpu"].cpu())
    cuda_states = state.get("cuda", [])
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG topology changed during streaming backward")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def _torch_rng_states_equal(torch, left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if not bool(torch.equal(left["cpu"].cpu(), right["cpu"].cpu())):
        return False
    left_cuda = left.get("cuda", [])
    right_cuda = right.get("cuda", [])
    return len(left_cuda) == len(right_cuda) and all(
        bool(torch.equal(first.cpu(), second.cpu()))
        for first, second in zip(left_cuda, right_cuda, strict=True)
    )


def _capture_model_buffers(model) -> dict[str, Any]:
    return {
        name: value.detach().clone()
        for name, value in model.named_buffers()
    }


def _restore_model_buffers(model, state: Mapping[str, Any]) -> None:
    current = dict(model.named_buffers())
    if set(current) != set(state):
        raise RuntimeError("Model buffer topology changed during streaming backward")
    for name, value in current.items():
        value.detach().copy_(state[name].to(device=value.device, dtype=value.dtype))


def _model_buffers_equal(torch, model, state: Mapping[str, Any]) -> bool:
    current = dict(model.named_buffers())
    return set(current) == set(state) and all(
        bool(torch.equal(value.detach(), state[name].to(device=value.device, dtype=value.dtype)))
        for name, value in current.items()
    )


def _loss_plan_metadata_from_outputs(
    outputs: Sequence[Mapping[str, Any]],
    batches: Sequence[Mapping[str, Any]],
    *,
    torch,
    objective_value: float,
    positive_prior: float = 0.10,
    weak_positive_weight: float = 0.35,
) -> GlobalLossPlan:
    """Extract denominators/branch metadata without a second global reduction."""

    if len(outputs) != len(batches) or not outputs:
        raise RuntimeError("Streaming loss probe requires aligned non-empty fragments")
    logits = torch.cat([item["final_logit"].reshape(-1) for item in outputs]).to(
        dtype=torch.float32
    )
    proxy = torch.cat([item["proxy_label"].reshape(-1) for item in batches]).to(
        device=logits.device
    )
    weak = torch.cat([item["weak_positive"].reshape(-1) for item in batches]).to(
        device=logits.device, dtype=torch.bool
    )
    association_available = torch.cat(
        [
            item.get(
                "association_available",
                torch.ones_like(item["proxy_label"], dtype=torch.bool),
            ).reshape(-1)
            for item in batches
        ]
    ).to(device=logits.device, dtype=torch.bool)
    if len(association_available) != len(logits):
        raise RuntimeError("Association-availability rows are misaligned")
    positive = (proxy > 0.5) & association_available
    unlabeled = (proxy <= 0.5) & association_available
    if bool(positive.any()):
        weights = torch.where(
            weak[positive],
            torch.full_like(logits[positive], float(weak_positive_weight)),
            torch.ones_like(logits[positive]),
        )
        positive_denominator = float(weights.sum())
        positive_negative_risk = float(positive_prior) * (
            torch.nn.functional.softplus(logits[positive]) * weights
        ).sum() / positive_denominator
    else:
        positive_denominator = 0.0
        positive_negative_risk = logits.sum() * 0.0
    unlabeled_count = int(unlabeled.sum().item())
    unlabeled_risk = (
        torch.nn.functional.softplus(logits[unlabeled]).mean()
        if unlabeled_count
        else logits.sum() * 0.0
    )
    correction = unlabeled_risk - positive_negative_risk
    direction_count = 0
    for batch in batches:
        if "direction_label" not in batch:
            continue
        label = batch["direction_label"].reshape(-1)
        available = batch.get(
            "direction_available", torch.ones_like(label, dtype=torch.bool)
        ).reshape(-1).to(dtype=torch.bool)
        association_available = batch.get(
            "association_available", torch.ones_like(label, dtype=torch.bool)
        ).reshape(-1).to(dtype=torch.bool)
        available = available & association_available
        direction_count += int(available.sum().item())
    residual_count = sum(
        int(item["raw_graph_residual"].numel()) for item in outputs
    )
    value = float(objective_value)
    if not math.isfinite(value):
        raise RuntimeError(f"NONFINITE_TRAINING_OBJECTIVE={value}")
    return GlobalLossPlan(
        positive_weight_denominator=positive_denominator,
        unlabeled_count=unlabeled_count,
        direction_count=direction_count,
        residual_count=residual_count,
        nnpu_correction_active=bool(float(correction.detach().cpu()) >= 0.0),
        objective_value=value,
    )


def _loss_plan_from_probe_outputs(
    outputs: Sequence[Mapping[str, Any]],
    batches: Sequence[Mapping[str, Any]],
    *,
    torch,
    direction_loss_weight: float,
    shrinkage: float,
    positive_prior: float = 0.10,
    weak_positive_weight: float = 0.35,
) -> GlobalLossPlan:
    """Compute the probe objective once, then extract replay metadata."""

    objective = _global_loss_from_outputs(
        list(outputs),
        list(batches),
        torch=torch,
        direction_loss_weight=direction_loss_weight,
        shrinkage=shrinkage,
        positive_prior=positive_prior,
        weak_positive_weight=weak_positive_weight,
        reduction_dtype=torch.float32,
    )
    return _loss_plan_metadata_from_outputs(
        outputs,
        batches,
        torch=torch,
        objective_value=float(objective.detach().cpu()),
        positive_prior=positive_prior,
        weak_positive_weight=weak_positive_weight,
    )


def _streaming_fragment_loss(
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    plan: GlobalLossPlan,
    *,
    torch,
    direction_loss_weight: float,
    shrinkage: float,
    positive_prior: float = 0.10,
    unlabeled_weight: float = 0.12,
    weak_positive_weight: float = 0.35,
):
    """Additive fragment of the frozen global objective in ``plan``."""

    functional = torch.nn.functional
    logits = output["final_logit"].reshape(-1).to(dtype=torch.float32)
    proxy = batch["proxy_label"].reshape(-1).to(device=logits.device)
    weak = batch["weak_positive"].reshape(-1).to(
        device=logits.device, dtype=torch.bool
    )
    association_available = batch.get(
        "association_available", torch.ones_like(proxy, dtype=torch.bool)
    ).reshape(-1).to(device=logits.device, dtype=torch.bool)
    if len(association_available) != len(logits):
        raise RuntimeError("Streaming association-availability rows are misaligned")
    positive = (proxy > 0.5) & association_available
    unlabeled = (proxy <= 0.5) & association_available
    loss = logits.sum() * 0.0
    if bool(positive.any()):
        if plan.positive_weight_denominator <= 0:
            raise RuntimeError("Streaming positive denominator is invalid")
        positive_logits = logits[positive]
        weights = torch.where(
            weak[positive],
            torch.full_like(positive_logits, float(weak_positive_weight)),
            torch.ones_like(positive_logits),
        )
        loss = loss + float(positive_prior) * (
            functional.softplus(-positive_logits) * weights
        ).sum() / plan.positive_weight_denominator
        if plan.nnpu_correction_active:
            loss = loss - float(unlabeled_weight) * float(positive_prior) * (
                functional.softplus(positive_logits) * weights
            ).sum() / plan.positive_weight_denominator
    if plan.nnpu_correction_active and bool(unlabeled.any()):
        if plan.unlabeled_count <= 0:
            raise RuntimeError("Streaming unlabeled denominator is invalid")
        loss = loss + float(unlabeled_weight) * functional.softplus(
            logits[unlabeled]
        ).sum() / plan.unlabeled_count

    if "direction_label" in batch:
        direction_logit = output["direction_logit"].reshape(-1).to(dtype=torch.float32)
        label = batch["direction_label"].reshape(-1).to(
            device=direction_logit.device, dtype=direction_logit.dtype
        )
        available = batch.get(
            "direction_available", torch.ones_like(label, dtype=torch.bool)
        ).reshape(-1).to(device=label.device, dtype=torch.bool)
        association_available = batch.get(
            "association_available", torch.ones_like(label, dtype=torch.bool)
        ).reshape(-1).to(device=label.device, dtype=torch.bool)
        available = available & association_available
        if bool(available.any()):
            if plan.direction_count <= 0:
                raise RuntimeError("Streaming direction denominator is invalid")
            loss = loss + float(direction_loss_weight) * functional.binary_cross_entropy_with_logits(
                direction_logit[available],
                label[available],
                reduction="sum",
            ) / plan.direction_count
    residual = output["raw_graph_residual"].reshape(-1).to(dtype=torch.float32)
    if residual.numel():
        if plan.residual_count <= 0:
            raise RuntimeError("Streaming residual denominator is invalid")
        loss = loss + float(shrinkage) * residual.square().sum() / plan.residual_count
    return loss


def streaming_group_backward(
    model,
    work_items: Sequence[tuple[Mapping[str, Any], int]],
    graph_for_chunk: Callable[[int], Any],
    *,
    torch,
    device: str,
    precision: Mapping[str, Any],
    direction_loss_weight: float,
    shrinkage: float,
    return_call_telemetry: bool = False,
) -> GlobalLossPlan | tuple[GlobalLossPlan, BackwardCallTelemetry]:
    """Compute an exact group objective while retaining one encoder graph.

    The no-grad probe freezes global nnPU denominators and the clamp branch and
    records each fragment's Torch RNG state.  The gradient pass replays one
    fragment at a time with the same dropout mask and immediately calls
    backward.  Decoder and encoder gradients equal a conventional single
    global loss, while at most one full-graph encoder activation is live.
    """

    if not work_items:
        raise RuntimeError("Streaming optimizer group is empty")
    probe_outputs: list[dict[str, Any]] = []
    probe_batches: list[dict[str, Any]] = []
    replay_rng: list[dict[str, Any]] = []
    replay_buffers_before: list[dict[str, Any]] = []
    replay_buffers_after: list[dict[str, Any]] = []
    encoder_forward_calls = 0
    decoder_forward_calls = 0
    global_loss_calls = 0
    backward_calls = 0
    for raw_batch, chunk in work_items:
        batch = _move(raw_batch, device)
        replay_rng.append(_capture_torch_rng_state(torch))
        replay_buffers_before.append(_capture_model_buffers(model))
        graph = graph_for_chunk(int(chunk))
        with torch.no_grad(), _autocast_context(torch, device, precision):
            output = model(
                graph,
                batch["candidate_batch"],
                batch["base_logit"],
                batch["conservation_context"],
                batch.get("graph_available"),
                admitted=True,
            )
            encoder_forward_calls += 1
            decoder_forward_calls += 1
        replay_buffers_after.append(_capture_model_buffers(model))
        probe_outputs.append(
            {
                key: output[key].detach().to(dtype=torch.float32)
                for key in ("final_logit", "direction_logit", "raw_graph_residual")
            }
        )
        probe_batches.append(
            {
                key: value.detach()
                for key, value in batch.items()
                if key
                in {
                    "proxy_label",
                    "weak_positive",
                    "association_available",
                    "direction_label",
                    "direction_available",
                }
            }
        )
        del output, graph, batch
    post_probe_rng = _capture_torch_rng_state(torch)
    plan = _loss_plan_from_probe_outputs(
        probe_outputs,
        probe_batches,
        torch=torch,
        direction_loss_weight=direction_loss_weight,
        shrinkage=shrinkage,
    )
    global_loss_calls += 1
    final_probe_buffers = replay_buffers_after[-1]
    replay_outputs: list[dict[str, Any]] = []
    try:
        for index, ((raw_batch, chunk), rng_state) in enumerate(
            zip(work_items, replay_rng, strict=True)
        ):
            _restore_model_buffers(model, replay_buffers_before[index])
            batch = _move(raw_batch, device)
            _restore_torch_rng_state(torch, rng_state)
            graph = graph_for_chunk(int(chunk))
            with _autocast_context(torch, device, precision):
                output = model(
                    graph,
                    batch["candidate_batch"],
                    batch["base_logit"],
                    batch["conservation_context"],
                    batch.get("graph_available"),
                    admitted=True,
                )
                encoder_forward_calls += 1
                decoder_forward_calls += 1
                fragment_loss = _streaming_fragment_loss(
                    output,
                    batch,
                    plan,
                    torch=torch,
                    direction_loss_weight=direction_loss_weight,
                    shrinkage=shrinkage,
                )
            replay_outputs.append(
                {
                    key: output[key].detach().to(dtype=torch.float32)
                    for key in (
                        "final_logit",
                        "direction_logit",
                        "raw_graph_residual",
                    )
                }
            )
            if not bool(torch.isfinite(fragment_loss.detach()).all()):
                raise RuntimeError("NONFINITE_STREAMING_FRAGMENT_LOSS")
            fragment_loss.backward()
            backward_calls += 1
            expected_rng = (
                replay_rng[index + 1]
                if index + 1 < len(replay_rng)
                else post_probe_rng
            )
            if not _torch_rng_states_equal(
                torch, _capture_torch_rng_state(torch), expected_rng
            ):
                raise RuntimeError("STREAMING_REPLAY_RNG_DRIFT")
            if not _model_buffers_equal(torch, model, replay_buffers_after[index]):
                raise RuntimeError("STREAMING_REPLAY_BUFFER_DRIFT")
            del fragment_loss, output, graph, batch
    finally:
        _restore_model_buffers(model, final_probe_buffers)
    replay_plan = _loss_plan_from_probe_outputs(
        replay_outputs,
        probe_batches,
        torch=torch,
        direction_loss_weight=direction_loss_weight,
        shrinkage=shrinkage,
    )
    global_loss_calls += 1
    if replay_plan.nnpu_correction_active != plan.nnpu_correction_active:
        raise RuntimeError("STREAMING_REPLAY_NNPU_BRANCH_DRIFT")
    tolerance = 1e-6 * max(1.0, abs(plan.objective_value))
    if abs(replay_plan.objective_value - plan.objective_value) > tolerance:
        raise RuntimeError(
            "STREAMING_REPLAY_OBJECTIVE_DRIFT="
            f"{replay_plan.objective_value}!={plan.objective_value}"
        )
    if not return_call_telemetry:
        return replay_plan
    return replay_plan, BackwardCallTelemetry(
        unique_chunks=len({int(chunk) for _, chunk in work_items}),
        group_rows=sum(_batch_row_count(batch) for batch, _ in work_items),
        encoder_forward_calls=encoder_forward_calls,
        decoder_forward_calls=decoder_forward_calls,
        global_loss_calls=global_loss_calls,
        backward_calls=backward_calls,
    )


def shared_encoder_conventional_group_backward(
    model,
    work_items: Sequence[tuple[Mapping[str, Any], int]],
    graph_for_chunk: Callable[[int], Any],
    *,
    torch,
    device: str,
    precision: Mapping[str, Any],
    direction_loss_weight: float,
    shrinkage: float,
) -> tuple[GlobalLossPlan, BackwardCallTelemetry]:
    """Backpropagate one conventional global loss through one shared encoder.

    This path is intentionally available only to an optimizer group whose
    candidate fragments all bind the same runtime chunk.  It performs exactly
    one graph materialization, one encoder call, the original-order decoder
    calls, and one backward call.  It never silently falls back to mixed-chunk
    or two-pass streaming semantics.
    """

    if not work_items:
        raise RuntimeError("Shared-encoder optimizer group is empty")
    chunks = {int(chunk) for _, chunk in work_items}
    if len(chunks) != 1:
        raise RuntimeError(
            "SHARED_ENCODER_GROUP_REQUIRES_UNIQUE_CHUNK="
            f"{sorted(chunks)}"
        )
    chunk = next(iter(chunks))
    batches = [_move(raw_batch, device) for raw_batch, _ in work_items]
    group_rows = sum(_batch_row_count(batch) for batch in batches)
    encoder_forward_calls = 0
    decoder_forward_calls = 0
    global_loss_calls = 0
    backward_calls = 0
    graph = graph_for_chunk(chunk)
    with _autocast_context(torch, device, precision):
        encoded = model.encoder.encode(graph)
        encoder_forward_calls += 1
        outputs = []
        for batch in batches:
            outputs.append(
                model(
                    graph,
                    batch["candidate_batch"],
                    batch["base_logit"],
                    batch["conservation_context"],
                    batch.get("graph_available"),
                    admitted=True,
                    encoded=encoded,
                )
            )
            decoder_forward_calls += 1
        loss = _global_loss_from_outputs(
            outputs,
            batches,
            torch=torch,
            direction_loss_weight=direction_loss_weight,
            shrinkage=shrinkage,
            reduction_dtype=torch.float32,
        )
        global_loss_calls += 1
    if not bool(torch.isfinite(loss.detach()).all()):
        raise RuntimeError("NONFINITE_SHARED_ENCODER_GROUP_LOSS")
    probe_outputs = [
        {
            key: output[key].detach().to(dtype=torch.float32)
            for key in ("final_logit", "direction_logit", "raw_graph_residual")
        }
        for output in outputs
    ]
    probe_batches = [
        {
            key: value.detach()
            for key, value in batch.items()
            if key
            in {
                "proxy_label",
                "weak_positive",
                "association_available",
                "direction_label",
                "direction_available",
            }
        }
        for batch in batches
    ]
    observed = float(loss.detach().to(dtype=torch.float32).cpu())
    plan = _loss_plan_metadata_from_outputs(
        probe_outputs,
        probe_batches,
        torch=torch,
        objective_value=observed,
    )
    loss.backward()
    backward_calls += 1
    return plan, BackwardCallTelemetry(
        unique_chunks=1,
        group_rows=int(group_rows),
        encoder_forward_calls=encoder_forward_calls,
        decoder_forward_calls=decoder_forward_calls,
        global_loss_calls=global_loss_calls,
        backward_calls=backward_calls,
    )


def optimizer_step_with_guards(
    model,
    optimizer,
    *,
    torch,
    objective_value: float,
) -> dict[str, Any]:
    """Fail before/after a step on non-finite gradients or a no-op update."""

    if not math.isfinite(float(objective_value)):
        raise RuntimeError(f"NONFINITE_OPTIMIZER_OBJECTIVE={objective_value}")
    gradient_rows: list[tuple[str, Any, Any]] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_rows.append((name, parameter, gradient))
    gradient_tensors = len(gradient_rows)
    if gradient_tensors == 0:
        raise RuntimeError("OPTIMIZER_STEP_HAS_NO_GRADIENTS")
    finite_flags = torch.stack(
        [torch.isfinite(gradient).all() for _, _, gradient in gradient_rows]
    )
    if not bool(finite_flags.all()):
        bad = [
            name
            for (name, _, _), finite
            in zip(gradient_rows, finite_flags.detach().cpu().tolist(), strict=True)
            if not finite
        ]
        raise RuntimeError(f"NONFINITE_GRADIENT={bad}")
    nonzero_flags = torch.stack(
        [gradient.detach().ne(0).any() for _, _, gradient in gradient_rows]
    ).detach().cpu().tolist()
    probes = [
        (int(parameter.numel()), name, parameter)
        for (name, parameter, _), nonzero
        in zip(gradient_rows, nonzero_flags, strict=True)
        if nonzero
    ]
    if not probes:
        raise RuntimeError("OPTIMIZER_STEP_HAS_ONLY_ZERO_GRADIENTS")
    _, probe_name, probe_parameter = min(probes, key=lambda item: (item[0], item[1]))
    before = probe_parameter.detach().clone()
    optimizer.step()
    delta = probe_parameter.detach() - before
    if not bool(torch.isfinite(delta).all()):
        raise RuntimeError(f"NONFINITE_PARAMETER_DELTA={probe_name}")
    maximum_delta = float(delta.abs().max().detach().cpu())
    if not maximum_delta > 0.0:
        raise RuntimeError(f"ZERO_PARAMETER_DELTA_AFTER_STEP={probe_name}")
    parameter_rows = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameter_finite_flags = torch.stack(
        [torch.isfinite(parameter.detach()).all() for _, parameter in parameter_rows]
    )
    if not bool(parameter_finite_flags.all()):
        bad = [
            name
            for (name, _), finite in zip(
                parameter_rows,
                parameter_finite_flags.detach().cpu().tolist(),
                strict=True,
            )
            if not finite
        ]
        raise RuntimeError(f"NONFINITE_PARAMETER_AFTER_STEP={bad}")
    return {
        "grad_finite": True,
        "gradient_tensors": gradient_tensors,
        "parameters_finite": True,
        "trainable_parameter_tensors": len(parameter_rows),
        "parameter_delta_positive": maximum_delta > 0.0,
        "update_probe_parameter": probe_name,
        "update_probe_max_abs_delta": maximum_delta,
    }


def _is_cuda_oom(torch, exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", ())
    return bool(
        (oom_type and isinstance(exc, oom_type))
        or "cuda out of memory" in str(exc).lower()
        or "cuda error: out of memory" in str(exc).lower()
    )


def _batch_loss(model, graph, batch, *, torch, direction_loss_weight: float, shrinkage: float, encoded=None):
    output = model(
        graph,
        batch["candidate_batch"],
        batch["base_logit"],
        batch["conservation_context"],
        batch.get("graph_available"),
        admitted=True,
        encoded=encoded,
    )
    loss = _global_loss_from_outputs(
        [output],
        [batch],
        torch=torch,
        direction_loss_weight=direction_loss_weight,
        shrinkage=shrinkage,
    )
    return loss, output


def _global_loss_from_outputs(
    outputs,
    batches,
    *,
    torch,
    direction_loss_weight: float,
    shrinkage: float,
    positive_prior: float = 0.10,
    unlabeled_weight: float = 0.12,
    weak_positive_weight: float = 0.35,
    reduction_dtype=None,
):
    """Compute one nnPU/BCE/shrinkage objective over the complete row set.

    nnPU contains a nonlinear non-negative clamp, so averaging independently
    clamped batch losses changes the estimand when the same rows are merely
    repartitioned.  The caller therefore supplies every output in the intended
    optimization/evaluation unit and this function applies each denominator
    and the clamp exactly once.
    """

    import torch.nn.functional as functional

    if len(outputs) != len(batches) or not outputs:
        raise RuntimeError("Global loss requires aligned, non-empty outputs and batches")
    logits = torch.cat([item["final_logit"].reshape(-1) for item in outputs])
    if reduction_dtype is not None:
        logits = logits.to(dtype=reduction_dtype)
    proxy = torch.cat([item["proxy_label"].reshape(-1) for item in batches]).to(
        device=logits.device
    )
    weak = torch.cat([item["weak_positive"].reshape(-1) for item in batches]).to(
        device=logits.device, dtype=torch.bool
    )
    if not (len(logits) == len(proxy) == len(weak)):
        raise RuntimeError("Global nnPU rows are misaligned")

    association_available = torch.cat(
        [
            item.get(
                "association_available",
                torch.ones_like(item["proxy_label"], dtype=torch.bool),
            ).reshape(-1)
            for item in batches
        ]
    ).to(device=logits.device, dtype=torch.bool)
    if len(association_available) != len(logits):
        raise RuntimeError("Global association-availability rows are misaligned")
    positive = (proxy > 0.5) & association_available
    unlabeled = (proxy <= 0.5) & association_available
    zero = logits.sum() * 0.0
    if bool(positive.any()):
        positive_logits = logits[positive]
        weights = torch.where(
            weak[positive],
            torch.full_like(positive_logits, float(weak_positive_weight)),
            torch.ones_like(positive_logits),
        )
        denominator = weights.sum().clamp_min(1e-8)
        positive_risk = float(positive_prior) * (
            functional.softplus(-positive_logits) * weights
        ).sum() / denominator
        positive_negative_risk = float(positive_prior) * (
            functional.softplus(positive_logits) * weights
        ).sum() / denominator
    else:
        positive_risk = zero
        positive_negative_risk = zero
    if bool(unlabeled.any()):
        unlabeled_risk = functional.softplus(logits[unlabeled]).mean()
    else:
        unlabeled_risk = zero
    membership = positive_risk + float(unlabeled_weight) * torch.clamp(
        unlabeled_risk - positive_negative_risk, min=0.0
    )

    direction_sum = zero
    direction_count = 0
    for output, batch in zip(outputs, batches):
        if "direction_label" not in batch:
            continue
        direction_logit = output["direction_logit"].reshape(-1)
        if reduction_dtype is not None:
            direction_logit = direction_logit.to(dtype=reduction_dtype)
        label = batch["direction_label"].reshape(-1).to(
            device=direction_logit.device, dtype=direction_logit.dtype
        )
        available = batch.get(
            "direction_available", torch.ones_like(label, dtype=torch.bool)
        ).reshape(-1).to(device=label.device, dtype=torch.bool)
        association_available = batch.get(
            "association_available", torch.ones_like(label, dtype=torch.bool)
        ).reshape(-1).to(device=label.device, dtype=torch.bool)
        available = available & association_available
        if len(label) != len(direction_logit):
            raise RuntimeError("Direction rows are misaligned")
        if bool(available.any()):
            direction_sum = direction_sum + functional.binary_cross_entropy_with_logits(
                direction_logit[available],
                label[available],
                reduction="sum",
            )
            direction_count += int(available.sum().item())
    direction = direction_sum / direction_count if direction_count else zero

    residuals = torch.cat([item["raw_graph_residual"].reshape(-1) for item in outputs])
    if reduction_dtype is not None:
        residuals = residuals.to(dtype=reduction_dtype)
    penalty = float(shrinkage) * residuals.square().mean() if residuals.numel() else zero
    return membership + float(direction_loss_weight) * direction + penalty


def _evaluate(
    model,
    graph,
    batches,
    *,
    torch,
    device: str,
    direction_loss_weight: float,
    shrinkage: float,
    precision: Mapping[str, Any] | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> float:
    model.eval()
    outputs = []
    moved_batches = []
    resolved_precision = precision or {
        "active": "fp32",
        "autocast_enabled": False,
    }
    with torch.no_grad():
        with _autocast_context(torch, device, resolved_precision):
            encoded = model.encoder.encode(graph)
        total_batches = len(batches)
        for batch_index, raw in enumerate(batches):
            batch = _move(raw, device)
            with _autocast_context(torch, device, resolved_precision):
                output = model(
                    graph,
                    batch["candidate_batch"],
                    batch["base_logit"],
                    batch["conservation_context"],
                    batch.get("graph_available"),
                    admitted=True,
                    encoded=encoded,
                )
            outputs.append({
                key: output[key].detach().to(device="cpu", dtype=torch.float64)
                for key in ("final_logit", "direction_logit", "raw_graph_residual")
            })
            moved_batches.append({
                key: value.detach().to("cpu")
                for key, value in batch.items()
                if key in {
                    "proxy_label",
                    "weak_positive",
                    "association_available",
                    "direction_label",
                    "direction_available",
                }
            })
            if progress_callback is not None:
                progress_callback(0, batch_index + 1, total_batches)
    if not outputs:
        raise RuntimeError("Validation produced no rows")
    loss = _global_loss_from_outputs(
        outputs,
        moved_batches,
        torch=torch,
        direction_loss_weight=direction_loss_weight,
        shrinkage=shrinkage,
    )
    return float(loss.detach().cpu())


def _runtime_chunk_contract(bundle: Any) -> tuple[tuple[int, ...], dict[int, float]]:
    """Validate and return the formal resident-backbone chunk estimand."""

    schedule = getattr(bundle, "runtime_schedule", None)
    if schedule is None:
        return (0,), {0: 1.0}
    if not isinstance(schedule, __import__("pandas").DataFrame) or schedule.empty:
        raise RuntimeError("Runtime graph schedule is empty or malformed")
    required = {"runtime_chunk", "runtime_residency", "canonical_edge_id"}
    if missing := sorted(required - set(schedule)):
        raise RuntimeError(
            "Formal training requires a resident-backbone schedule; missing "
            f"{missing}"
        )
    from ..relation_sampling import runtime_chunk_positions

    position_map = runtime_chunk_positions(schedule)
    chunks = tuple(sorted(position_map))
    resident = schedule.loc[schedule.runtime_residency.astype(str).eq("resident_backbone")]
    rotating = schedule.loc[schedule.runtime_residency.astype(str).eq("rotating_variable")]
    if (
        resident.canonical_edge_id.astype(str).duplicated().any()
        or not resident.runtime_chunk.eq(-1).all()
    ):
        raise RuntimeError("Compact resident backbone rows are duplicated/misassigned")
    expected_resident = set(resident.canonical_edge_id.astype(str))
    for chunk, positions in position_map.items():
        active = schedule.iloc[positions]
        observed = set(
            active.loc[
                active.runtime_residency.astype(str).eq("resident_backbone"),
                "canonical_edge_id",
            ].astype(str)
        )
        if observed != expected_resident:
            raise RuntimeError(f"Runtime chunk {chunk} lacks the complete resident backbone")
    if rotating.canonical_edge_id.astype(str).duplicated().any():
        raise RuntimeError("A rotating variable edge occurs in more than one runtime chunk")
    counts = {
        chunk: int(rotating.runtime_chunk.eq(chunk).sum()) for chunk in chunks
    }
    total = sum(counts.values())
    if total == 0:
        if len(chunks) != 1:
            raise RuntimeError("Backbone-only formal graph must have exactly one chunk")
        weights = {chunks[0]: 1.0}
    else:
        if any(value <= 0 for value in counts.values()):
            raise RuntimeError("A variable runtime chunk is empty")
        weights = {chunk: counts[chunk] / total for chunk in chunks}
    if abs(sum(weights.values()) - 1.0) > 1e-15:
        raise RuntimeError("Runtime chunk weights do not sum to one")
    return chunks, weights


def _frozen_chunk_permutation(chunks: tuple[int, ...], *, seed: int, fold: int) -> tuple[int, ...]:
    """Hash-sort chunks without consuming the model RNG stream."""

    return tuple(
        sorted(
            chunks,
            key=lambda chunk: hashlib.sha256(
                f"V32_CHUNK_PERMUTATION|{int(seed)}|{int(fold)}|{int(chunk)}".encode()
            ).hexdigest(),
        )
    )


def _candidate_chunk_schedule(
    *,
    batch_count: int,
    chunks: tuple[int, ...],
    permutation: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return the exact B x K Cartesian supercycle from the frozen design."""

    if batch_count < 1 or not chunks or set(chunks) != set(permutation):
        raise ValueError("Invalid candidate/chunk schedule inputs")
    passes = []
    width = len(permutation)
    for pass_index in range(width):
        pairs = tuple(
            (batch_index, permutation[(batch_index + pass_index) % width])
            for batch_index in range(batch_count)
        )
        passes.append(pairs)
    flattened = [pair for pairs in passes for pair in pairs]
    expected = {(batch, chunk) for batch in range(batch_count) for chunk in chunks}
    if len(flattened) != len(expected) or set(flattened) != expected:
        raise AssertionError("Candidate/chunk supercycle is not exact Cartesian coverage")
    return tuple(passes)


def _frozen_pass_offset_permutation(
    chunks: tuple[int, ...], *, seed: int, fold: int
) -> tuple[int, ...]:
    """Hash-sort Cartesian pass offsets without consuming model RNG state."""

    return tuple(
        sorted(
            range(len(chunks)),
            key=lambda offset: hashlib.sha256(
                (
                    "V32_CANDIDATE_CHUNK_PASS_OFFSET|"
                    f"{int(seed)}|{int(fold)}|{int(offset)}"
                ).encode()
            ).hexdigest(),
        )
    )


def _resolve_candidate_chunk_schedule_mode(runtime: Mapping[str, Any]) -> str:
    """Resolve the explicitly configured schedule, retaining the legacy default."""

    mode = str(
        runtime.get(
            "candidate_chunk_schedule_mode",
            CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN,
        )
    ).strip().lower()
    if mode not in _CANDIDATE_CHUNK_SCHEDULE_MODES:
        raise RuntimeError(
            "Unsupported runtime_profile.candidate_chunk_schedule_mode="
            f"{mode!r}"
        )
    return mode


def _resolve_partial_rotation_authorization(
    runtime: Mapping[str, Any], *, schedule_mode: str
) -> bool:
    """Require a literal opt-in whenever training may stop before K rotations."""

    raw = runtime.get("allow_partial_candidate_chunk_rotation")
    if schedule_mode == CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN:
        return False
    if raw is not True:
        raise RuntimeError(
            "BALANCED_CANDIDATE_CHUNK_ROTATION_REQUIRES_EXPLICIT_PARTIAL_AUTHORIZATION"
        )
    return True


def _gradient_algorithm_for_schedule_mode(mode: str) -> str:
    if mode == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER:
        return SHARED_ENCODER_GRADIENT_ALGORITHM
    if mode in {
        CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN,
        CANDIDATE_CHUNK_SCHEDULE_BALANCED_SINGLE_PASS,
    }:
        return TWO_PASS_STREAMING_GRADIENT_ALGORITHM
    raise ValueError(f"Unknown candidate/chunk schedule mode: {mode!r}")


def _canonical_optimizer_batch_groups(
    batch_count: int, gradient_accumulation: int
) -> tuple[tuple[int, ...], ...]:
    """Freeze consecutive candidate batches into optimizer assignment units."""

    if batch_count < 1 or gradient_accumulation < 1:
        raise ValueError("Optimizer grouping requires positive B and accumulation")
    return tuple(
        tuple(range(start, min(start + gradient_accumulation, batch_count)))
        for start in range(0, batch_count, gradient_accumulation)
    )


def _optimizer_group_row_counts(
    batch_row_counts: Sequence[int],
    optimizer_groups: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    if not batch_row_counts or any(int(value) < 1 for value in batch_row_counts):
        raise ValueError("Candidate batch row counts must be positive")
    flattened = [int(batch) for group in optimizer_groups for batch in group]
    if flattened != list(range(len(batch_row_counts))):
        raise ValueError("Optimizer groups do not partition candidate batches canonically")
    return tuple(
        sum(int(batch_row_counts[int(batch)]) for batch in group)
        for group in optimizer_groups
    )


def _candidate_chunk_cycle_schedules(
    *,
    batch_count: int,
    chunks: tuple[int, ...],
    permutation: tuple[int, ...],
    pass_offset_permutation: tuple[int, ...],
    max_cycles: int,
    mode: str,
    gradient_accumulation: int = 1,
) -> tuple[tuple[tuple[tuple[int, int], ...], ...], ...]:
    """Precompute every authorized cycle's candidate/chunk work.

    ``exact_cartesian_v1`` preserves the legacy B x K oracle in every cycle.
    ``balanced_cyclic_single_pass_v1`` selects one existing Cartesian pass per
    cycle.  Each selected pass visits every candidate batch exactly once and,
    when B >= K, exposes every chunk either floor(B/K) or ceil(B/K) times.  The
    frozen pass-offset permutation makes the first K cycles a deterministic
    without-replacement traversal of the exact Cartesian oracle.

    ``balanced_group_latin_shared_encoder_v1`` freezes consecutive optimizer
    groups of width A and assigns one chunk to the complete group.  Its first K
    cycles also cover every candidate-batch/chunk pair once, but its per-step
    objective is deliberately not the legacy mixed-chunk oracle.
    """

    if mode not in _CANDIDATE_CHUNK_SCHEDULE_MODES:
        raise ValueError(f"Unknown candidate/chunk schedule mode: {mode!r}")
    if max_cycles < 1:
        raise ValueError("Candidate/chunk schedule requires a positive cycle count")
    cartesian = _candidate_chunk_schedule(
        batch_count=batch_count,
        chunks=chunks,
        permutation=permutation,
    )
    if (
        len(pass_offset_permutation) != len(chunks)
        or set(pass_offset_permutation) != set(range(len(chunks)))
    ):
        raise ValueError("Invalid frozen Cartesian pass-offset permutation")
    if mode == CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN:
        return tuple(cartesian for _ in range(max_cycles))
    if mode == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER:
        optimizer_groups = _canonical_optimizer_batch_groups(
            batch_count, gradient_accumulation
        )
        if len(optimizer_groups) < len(chunks):
            raise RuntimeError(
                "Balanced group scheduling requires optimizer groups >= runtime "
                f"chunks: G={len(optimizer_groups)}, K={len(chunks)}"
            )
        cycles = []
        for cycle in range(max_cycles):
            offset = pass_offset_permutation[cycle % len(chunks)]
            pairs = []
            for group_index, group in enumerate(optimizer_groups):
                chunk = permutation[(group_index + offset) % len(chunks)]
                pairs.extend((batch_index, chunk) for batch_index in group)
            cycles.append((tuple(pairs),))
        cycles = tuple(cycles)
        expected_batches = list(range(batch_count))
        low, remainder = divmod(len(optimizer_groups), len(chunks))
        high = low + (1 if remainder else 0)
        for cycle_index, cycle_schedule in enumerate(cycles):
            pairs = cycle_schedule[0]
            if [batch for batch, _ in pairs] != expected_batches:
                raise AssertionError(
                    f"Group-Latin cycle {cycle_index} changed canonical batch order"
                )
            group_chunk_counts = {chunk: 0 for chunk in chunks}
            for group in optimizer_groups:
                assigned = {pairs[batch][1] for batch in group}
                if len(assigned) != 1:
                    raise AssertionError(
                        f"Group-Latin cycle {cycle_index} mixed chunks within a group"
                    )
                group_chunk_counts[next(iter(assigned))] += 1
            if set(group_chunk_counts.values()) - {low, high}:
                raise AssertionError(
                    f"Group-Latin cycle {cycle_index} is unbalanced: "
                    f"{group_chunk_counts}"
                )
    else:
        if batch_count < len(chunks):
            raise RuntimeError(
                "Balanced single-pass scheduling requires candidate batches >= "
                f"runtime chunks: B={batch_count}, K={len(chunks)}"
            )

        cycles = tuple(
            (cartesian[pass_offset_permutation[cycle % len(chunks)]],)
            for cycle in range(max_cycles)
        )
        expected_batches = set(range(batch_count))
        expected_chunks = set(chunks)
        low, remainder = divmod(batch_count, len(chunks))
        high = low + (1 if remainder else 0)
        for cycle_index, cycle_schedule in enumerate(cycles):
            pairs = cycle_schedule[0]
            if len(pairs) != batch_count or {batch for batch, _ in pairs} != expected_batches:
                raise AssertionError(
                    f"Balanced cycle {cycle_index} does not cover each candidate batch once"
                )
            counts = {chunk: 0 for chunk in chunks}
            for _, chunk in pairs:
                counts[chunk] += 1
            if set(counts) != expected_chunks or set(counts.values()) - {low, high}:
                raise AssertionError(
                    f"Balanced cycle {cycle_index} has unbalanced chunk exposure: {counts}"
                )
    first_rotation = cycles[: min(len(chunks), len(cycles))]
    if len(first_rotation) == len(chunks):
        flattened = [
            pair
            for cycle_schedule in first_rotation
            for pass_pairs in cycle_schedule
            for pair in pass_pairs
        ]
        expected = {
            (batch, chunk)
            for batch in range(batch_count)
            for chunk in chunks
        }
        if len(flattened) != len(expected) or set(flattened) != expected:
            raise AssertionError(
                "The first K balanced cycles do not partition exact Cartesian coverage"
            )
    return cycles


def _candidate_chunk_schedule_payload(
    *,
    mode: str,
    chunks: tuple[int, ...],
    weights: Mapping[int, float],
    permutation: tuple[int, ...],
    pass_offset_permutation: tuple[int, ...],
    batch_count: int,
    cycle_schedules: Sequence[Sequence[Sequence[tuple[int, int]]]],
    gradient_accumulation: int = 1,
    batch_row_counts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return the complete, hashable training-schedule authority."""

    optimizer_groups = _canonical_optimizer_batch_groups(
        batch_count, gradient_accumulation
    )
    if batch_row_counts is None:
        if mode == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER:
            raise ValueError("Group-shared schedule authority requires batch row counts")
        group_row_counts: tuple[int, ...] = ()
    else:
        if len(batch_row_counts) != batch_count:
            raise ValueError("Schedule batch-row authority count changed")
        group_row_counts = _optimizer_group_row_counts(
            batch_row_counts, optimizer_groups
        )
    return {
        "format": "CC_HHGT_V3_2_CANDIDATE_CHUNK_TRAINING_SCHEDULE_V3",
        "mode": mode,
        "gradient_algorithm": _gradient_algorithm_for_schedule_mode(mode),
        "assignment_unit": (
            "optimizer_group"
            if mode == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
            else "candidate_batch"
        ),
        "runtime_chunks": list(chunks),
        "runtime_chunk_weights": {str(key): weights[key] for key in chunks},
        "runtime_permutation": list(permutation),
        "pass_offset_permutation": list(pass_offset_permutation),
        "candidate_batch_count": int(batch_count),
        "gradient_accumulation": int(gradient_accumulation),
        "optimizer_group_count": len(optimizer_groups),
        "optimizer_groups": [list(group) for group in optimizer_groups],
        "candidate_batch_row_counts": (
            [int(value) for value in batch_row_counts]
            if batch_row_counts is not None
            else None
        ),
        "optimizer_group_row_counts": list(group_row_counts),
        "full_optimizer_group_rows": (
            max(group_row_counts) if group_row_counts else None
        ),
        "final_optimizer_group_rows": (
            group_row_counts[-1] if group_row_counts else None
        ),
        "configured_cycles": len(cycle_schedules),
        "graph_variant_independent_schedule": True,
        "cycles": [
            [
                [list(pair) for pair in pass_pairs]
                for pass_pairs in cycle_schedule
            ]
            for cycle_schedule in cycle_schedules
        ],
    }


def _candidate_chunk_schedule_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _completed_rotation_receipt(
    cycle_schedules: Sequence[Sequence[Sequence[tuple[int, int]]]],
    runtime_permutation: Sequence[int],
) -> dict[str, Any]:
    """Summarize completed pass offsets without claiming estimator exactness."""

    permutation = tuple(int(chunk) for chunk in runtime_permutation)
    if not permutation or len(set(permutation)) != len(permutation):
        raise ValueError("Rotation receipt requires a unique runtime permutation")
    by_cycle = []
    for cycle_schedule in cycle_schedules:
        offsets = []
        for pass_pairs in cycle_schedule:
            if not pass_pairs or int(pass_pairs[0][0]) != 0:
                raise RuntimeError(
                    "Rotation receipt requires non-empty canonical passes"
                )
            offsets.append(permutation.index(int(pass_pairs[0][1])))
        by_cycle.append(offsets)
    completed = [offset for offsets in by_cycle for offset in offsets]
    unique = sorted(set(completed))
    return {
        "completed_selected_pass_offsets_by_cycle": by_cycle,
        "completed_selected_pass_offsets": completed,
        "completed_unique_pass_offsets": unique,
        "completed_rotation_fraction": min(1.0, len(unique) / len(permutation)),
        "full_pair_rotation_completed": set(unique) == set(range(len(permutation))),
    }


_RESUME_TRAINING_CALL_FIELDS = (
    "encoder_forward_calls",
    "decoder_forward_calls",
    "global_loss_calls",
    "backward_calls",
)


def _strict_nonnegative_counter(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"RESUME_COUNTER_INVALID={label}:{value!r}")
    return int(value)


def validate_resume_history_and_counters(
    state: Mapping[str, Any],
    *,
    checkpoint_cycle: int,
    schedule_mode: str,
    optimizer_group_count: int,
    candidate_batch_count: int,
) -> None:
    """Fail before model restore when checkpoint progress is not self-consistent.

    Every V4 checkpoint carries per-cycle deltas and cumulative totals.  The
    generic modes are checked for non-negative, monotone accounting; the
    no-fallback shared-encoder estimator additionally has a closed-form exact
    call budget that is recomputed from completed cycles.
    """

    history = state.get("history")
    if not isinstance(history, list) or len(history) != checkpoint_cycle + 1:
        raise RuntimeError(
            "RESUME_HISTORY_CYCLE_COVERAGE_DRIFT="
            f"cycle={checkpoint_cycle}, rows="
            f"{len(history) if isinstance(history, list) else 'NOT_A_LIST'}"
        )
    state_totals = {
        "optimizer_steps": _strict_nonnegative_counter(
            state.get("optimizer_steps"), label="state.optimizer_steps"
        )
    }
    for field in _RESUME_TRAINING_CALL_FIELDS:
        state_totals[field] = _strict_nonnegative_counter(
            state.get(field), label=f"state.{field}"
        )
    previous = {key: 0 for key in state_totals}
    previous_oom_events = 0
    previous_oom_active = False
    for expected_cycle, raw_row in enumerate(history):
        if not isinstance(raw_row, Mapping):
            raise RuntimeError(f"RESUME_HISTORY_ROW_INVALID={expected_cycle}")
        if raw_row.get("cycle") != expected_cycle:
            raise RuntimeError(
                "RESUME_HISTORY_CYCLE_ORDER_DRIFT="
                f"expected={expected_cycle}, observed={raw_row.get('cycle')!r}"
            )
        optimizer_delta = _strict_nonnegative_counter(
            raw_row.get("optimizer_steps_this_cycle"),
            label=f"history[{expected_cycle}].optimizer_steps_this_cycle",
        )
        optimizer_total = _strict_nonnegative_counter(
            raw_row.get("optimizer_steps"),
            label=f"history[{expected_cycle}].optimizer_steps",
        )
        if optimizer_total != previous["optimizer_steps"] + optimizer_delta:
            raise RuntimeError(
                "RESUME_HISTORY_OPTIMIZER_ACCOUNTING_DRIFT="
                f"cycle={expected_cycle}"
            )
        previous["optimizer_steps"] = optimizer_total
        for field in _RESUME_TRAINING_CALL_FIELDS:
            delta = _strict_nonnegative_counter(
                raw_row.get(field), label=f"history[{expected_cycle}].{field}"
            )
            total_key = f"{field}_total"
            total = _strict_nonnegative_counter(
                raw_row.get(total_key),
                label=f"history[{expected_cycle}].{total_key}",
            )
            if total != previous[field] + delta:
                raise RuntimeError(
                    "RESUME_HISTORY_CALL_ACCOUNTING_DRIFT="
                    f"cycle={expected_cycle}, field={field}"
                )
            previous[field] = total
        oom_events = _strict_nonnegative_counter(
            raw_row.get("oom_fallback_events"),
            label=f"history[{expected_cycle}].oom_fallback_events",
        )
        oom_active = raw_row.get("oom_fallback_active")
        if not isinstance(oom_active, bool):
            raise RuntimeError(
                f"RESUME_HISTORY_OOM_ACTIVE_INVALID={expected_cycle}:{oom_active!r}"
            )
        if oom_events < previous_oom_events or (previous_oom_active and not oom_active):
            raise RuntimeError(f"RESUME_HISTORY_OOM_MONOTONICITY_DRIFT={expected_cycle}")
        previous_oom_events = oom_events
        previous_oom_active = oom_active
        if schedule_mode == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER:
            expected = {
                "optimizer_steps": optimizer_group_count,
                "encoder_forward_calls": optimizer_group_count,
                "decoder_forward_calls": candidate_batch_count,
                "global_loss_calls": optimizer_group_count,
                "backward_calls": optimizer_group_count,
            }
            observed = {
                "optimizer_steps": optimizer_delta,
                **{
                    field: _strict_nonnegative_counter(
                        raw_row.get(field),
                        label=f"history[{expected_cycle}].{field}",
                    )
                    for field in _RESUME_TRAINING_CALL_FIELDS
                },
            }
            if observed != expected:
                raise RuntimeError(
                    "RESUME_SHARED_CYCLE_CALL_BUDGET_DRIFT="
                    f"cycle={expected_cycle}, observed={observed}, expected={expected}"
                )
            if oom_active or oom_events:
                raise RuntimeError(
                    f"RESUME_SHARED_OOM_FALLBACK_STATE_FORBIDDEN={expected_cycle}"
                )
    if previous != state_totals:
        raise RuntimeError(
            "RESUME_STATE_HISTORY_COUNTER_DRIFT="
            f"history={previous}, state={state_totals}"
        )
    state_oom_events = _strict_nonnegative_counter(
        state.get("oom_fallback_events"), label="state.oom_fallback_events"
    )
    state_oom_active = state.get("oom_fallback_active")
    if not isinstance(state_oom_active, bool):
        raise RuntimeError(f"RESUME_STATE_OOM_ACTIVE_INVALID={state_oom_active!r}")
    if (state_oom_events, state_oom_active) != (
        previous_oom_events,
        previous_oom_active,
    ):
        raise RuntimeError("RESUME_STATE_HISTORY_OOM_DRIFT")
    if schedule_mode == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER:
        completed_cycles = checkpoint_cycle + 1
        expected_shared = {
            "optimizer_steps": completed_cycles * optimizer_group_count,
            "encoder_forward_calls": completed_cycles * optimizer_group_count,
            "decoder_forward_calls": completed_cycles * candidate_batch_count,
            "global_loss_calls": completed_cycles * optimizer_group_count,
            "backward_calls": completed_cycles * optimizer_group_count,
        }
        if state_totals != expected_shared:
            raise RuntimeError(
                "RESUME_SHARED_COMPLETED_CALL_BUDGET_DRIFT="
                f"observed={state_totals}, expected={expected_shared}"
            )


def _kahan_weighted_tensor_sum(values, weights, *, torch):
    if len(values) != len(weights) or not values:
        raise RuntimeError("Chunk aggregation requires aligned non-empty values")
    total = torch.zeros_like(values[0], dtype=torch.float64, device="cpu")
    compensation = torch.zeros_like(total)
    for value, weight in zip(values, weights, strict=True):
        current = value.detach().to(device="cpu", dtype=torch.float64)
        if current.shape != total.shape:
            raise RuntimeError("Chunk output shapes differ")
        adjusted = current * float(weight) - compensation
        updated = total + adjusted
        compensation = (updated - total) - adjusted
        total = updated
    return total


def _aggregate_chunk_outputs(
    outputs_by_chunk: Mapping[int, list[Mapping[str, Any]]],
    *,
    chunks: tuple[int, ...],
    weights: Mapping[int, float],
    torch,
) -> list[dict[str, Any]]:
    """Aggregate logits in canonical chunk order before any nonlinear loss."""

    if tuple(sorted(outputs_by_chunk)) != chunks:
        raise RuntimeError("Validation did not produce every registered runtime chunk")
    batch_counts = {len(outputs_by_chunk[chunk]) for chunk in chunks}
    if len(batch_counts) != 1 or not batch_counts or next(iter(batch_counts)) < 1:
        raise RuntimeError("Runtime chunks produced unequal/empty candidate coverage")
    keys = ("final_logit", "direction_logit", "raw_graph_residual")
    result = []
    for batch_index in range(next(iter(batch_counts))):
        aggregate = {}
        for key in keys:
            aggregate[key] = _kahan_weighted_tensor_sum(
                [outputs_by_chunk[chunk][batch_index][key] for chunk in chunks],
                [weights[chunk] for chunk in chunks],
                torch=torch,
            )
        result.append(aggregate)
    return result


def _runtime_graph_for_chunk(bundle: Any, chunk: int, *, device: str):
    from ..gnn import move_graph, runtime_bundle_for_step

    runtime_bundle = runtime_bundle_for_step(bundle, int(chunk))
    if int(runtime_bundle.active_runtime_chunk) != int(chunk):
        raise RuntimeError("Runtime bundle returned the wrong active chunk")
    return move_graph(runtime_bundle, device, "cc_hhgt")


def _evaluate_full_runtime_coverage_core(
    model,
    bundle,
    batches,
    *,
    chunks: tuple[int, ...],
    weights: Mapping[int, float],
    torch,
    device: str,
    direction_loss_weight: float,
    shrinkage: float,
    precision: Mapping[str, Any] | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    include_candidate_detail: bool,
) -> float | RuntimeCoverageEvaluation:
    """Single production implementation of the exact all-chunk estimand.

    Formal training requests only the scalar and therefore retains its prior
    memory behaviour.  Comparison-only callers explicitly request canonical
    logits/labels from the already materialized aggregate.
    """

    model.eval()
    outputs_by_chunk: dict[int, list[dict[str, Any]]] = {}
    moved_labels = None
    resolved_precision = precision or {
        "active": "fp32",
        "autocast_enabled": False,
    }
    with torch.no_grad():
        for chunk in chunks:
            graph = _runtime_graph_for_chunk(bundle, chunk, device=device)
            with _autocast_context(torch, device, resolved_precision):
                encoded = model.encoder.encode(graph)
            chunk_outputs = []
            labels = []
            total_batches = len(batches)
            for batch_index, raw in enumerate(batches):
                batch = _move(raw, device)
                with _autocast_context(torch, device, resolved_precision):
                    output = model(
                        graph,
                        batch["candidate_batch"],
                        batch["base_logit"],
                        batch["conservation_context"],
                        batch.get("graph_available"),
                        admitted=True,
                        encoded=encoded,
                    )
                chunk_outputs.append(
                    {
                        key: output[key].detach().to(device="cpu", dtype=torch.float64)
                        for key in ("final_logit", "direction_logit", "raw_graph_residual")
                    }
                )
                labels.append(
                    {
                        key: value.detach().to("cpu")
                        for key, value in batch.items()
                        if key in {
                            "proxy_label",
                            "weak_positive",
                            "association_available",
                            "direction_label",
                            "direction_available",
                        }
                    }
                )
                if progress_callback is not None:
                    progress_callback(int(chunk), batch_index + 1, total_batches)
            outputs_by_chunk[int(chunk)] = chunk_outputs
            if moved_labels is None:
                moved_labels = labels
            elif len(labels) != len(moved_labels):
                raise RuntimeError("Validation label coverage changed across chunks")
    aggregate = _aggregate_chunk_outputs(
        outputs_by_chunk, chunks=chunks, weights=weights, torch=torch
    )
    loss = _global_loss_from_outputs(
        aggregate,
        moved_labels,
        torch=torch,
        direction_loss_weight=direction_loss_weight,
        shrinkage=shrinkage,
    )
    objective_value = float(loss.detach().cpu())
    if not include_candidate_detail:
        return objective_value
    mean_final_logit = torch.cat(
        [item["final_logit"].reshape(-1) for item in aggregate]
    ).detach().to(device="cpu", dtype=torch.float64)
    proxy_label = torch.cat(
        [item["proxy_label"].reshape(-1) for item in moved_labels]
    ).detach().to(device="cpu", dtype=torch.float64)
    if len(mean_final_logit) != len(proxy_label) or not len(mean_final_logit):
        raise RuntimeError("Validation detailed rows are empty or misaligned")
    return RuntimeCoverageEvaluation(
        objective_value=objective_value,
        mean_final_logit=mean_final_logit,
        proxy_label=proxy_label,
        candidate_rows=int(len(mean_final_logit)),
        encoder_forward_calls=int(len(chunks)),
        decoder_forward_calls=int(len(chunks) * len(batches)),
        global_loss_calls=1,
        runtime_chunks_evaluated=int(len(chunks)),
    )


def evaluate_full_runtime_coverage_detailed(
    model,
    bundle,
    batches,
    *,
    chunks: tuple[int, ...],
    weights: Mapping[int, float],
    torch,
    device: str,
    direction_loss_weight: float,
    shrinkage: float,
    precision: Mapping[str, Any] | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> RuntimeCoverageEvaluation:
    """Return exact validation objective plus canonical mean logits/labels."""

    result = _evaluate_full_runtime_coverage_core(
        model,
        bundle,
        batches,
        chunks=chunks,
        weights=weights,
        torch=torch,
        device=device,
        direction_loss_weight=direction_loss_weight,
        shrinkage=shrinkage,
        precision=precision,
        progress_callback=progress_callback,
        include_candidate_detail=True,
    )
    if not isinstance(result, RuntimeCoverageEvaluation):
        raise AssertionError("Detailed validation core returned a scalar")
    return result


def _evaluate_full_runtime_coverage(
    model,
    bundle,
    batches,
    *,
    chunks: tuple[int, ...],
    weights: Mapping[int, float],
    torch,
    device: str,
    direction_loss_weight: float,
    shrinkage: float,
    precision: Mapping[str, Any] | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> float:
    """Compatibility scalar API for formal training."""

    result = _evaluate_full_runtime_coverage_core(
        model,
        bundle,
        batches,
        chunks=chunks,
        weights=weights,
        torch=torch,
        device=device,
        direction_loss_weight=direction_loss_weight,
        shrinkage=shrinkage,
        precision=precision,
        progress_callback=progress_callback,
        include_candidate_detail=False,
    )
    if not isinstance(result, float):
        raise AssertionError("Scalar validation core returned detailed output")
    return result


def _atomic_jsonl_write(rows: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_authorized_task(context: Any) -> int:
    """Run one approved local patient-fold task.

    The current repository intentionally has no approval file, so this
    function is unreachable during the delivered CODE_ONLY phase.
    """

    # The public callable is itself a security boundary: reject fabricated
    # mappings/direct calls and re-run the complete no-Torch authorization
    # guard before importing the training framework.
    from .training_guard import (
        TrainingAuthorizationContext,
        TrainingAuthorizationError,
        guard_training_entry,
    )

    if type(context) is not TrainingAuthorizationContext:
        raise TrainingAuthorizationError(
            "run_authorized_task requires a guarded TrainingAuthorizationContext"
        )
    if context.authorized_trainer != AUTHORIZED_FORMAL_TRAINER_SPECIFICATION:
        raise TrainingAuthorizationError(
            "FORMAL_TRAINER_AUTHORIZATION_DRIFT: expected="
            f"{AUTHORIZED_FORMAL_TRAINER_SPECIFICATION!r}, observed="
            f"{context.authorized_trainer!r}"
        )
    revalidated = guard_training_entry(
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
    if revalidated != context:
        raise TrainingAuthorizationError(
            "TrainingAuthorizationContext changed since the CLI guard"
        )

    import torch

    from ..gnn import build_model, move_graph
    from ..training_resume import capture_rng_state, restore_rng_state
    from .model import build_v32_cc_hhgt_residual, build_v32_hierarchical_evidence_hhgt

    ctx = _context_mapping(context)
    config_path = Path(ctx["config_path"])
    root = Path(ctx["repo_root"])
    config = _load_mapping(config_path)
    task = _task_row(Path(ctx["task_manifest_path"]), ctx["task_id"])
    fold = int(task["patient_fold"])
    seed = int(task["seed"])
    prepared_path = _resolve_prepared_path(config, fold, root)
    payload, prepared_artifact_sha256 = load_prepared_artifact_from_authorized_handle(
        torch,
        ctx["input_manifest_path"],
        fold=fold,
        prepared_path=prepared_path,
    )
    if not isinstance(payload, Mapping):
        raise RuntimeError("Prepared fold artifact must contain a mapping")
    _validate_prepared(payload, fold=fold, artifact_hashes=ctx["artifact_hashes"])
    graph_variant = validate_formal_graph_variant_binding(
        run_id=str(ctx["run_id"]),
        config=config,
        payload=payload,
        prepared_path=prepared_path,
    )
    input_authority_hashes = _prepared_input_authority_hashes(payload)

    runtime = config.get("runtime_profile", {})
    model_config = dict(payload["legacy_model_config"])
    feature_dim = int(payload["feature_dim"])
    bundle = payload["bundle"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Authorized CC-HHGT training requires CUDA; CPU fallback is forbidden")
    precision = resolve_mixed_precision_contract(
        torch, runtime, device=device
    )
    runtime_plan = resolve_runtime_training_plan(runtime)
    _seed_everything(torch, seed)
    encoder = build_model("cc_hhgt", bundle, feature_dim, model_config)
    primary = config.get("primary_model", {})
    integration = primary.get("evidence_integration", {})
    integration_mode = str(integration.get("mode", "external_router")).lower()
    if integration_mode == "hierarchical_end_to_end":
        modality_names = tuple(integration.get("modalities", ("mutation", "cnv", "atac")))
        validate_hierarchical_modality_contract(payload, modality_names)
        model = build_v32_hierarchical_evidence_hhgt(
            encoder,
            hidden_channels=int(primary.get("hidden_channels", 96)),
            context_features=int(payload.get("conservation_context_features", 4)),
            modality_names=modality_names,
            dropout=float(primary.get("dropout", 0.20)),
        ).to(device)
        architecture_id = "HHGT_HIERARCHICAL_EVIDENCE_GATE_CANDIDATE"
    elif integration_mode == "external_router":
        model = build_v32_cc_hhgt_residual(
            encoder,
            hidden_channels=int(primary.get("hidden_channels", 96)),
            context_features=int(payload.get("conservation_context_features", 4)),
            dropout=float(primary.get("dropout", 0.20)),
        ).to(device)
        architecture_id = "HHGT_FORMAL_CORE_EXTERNAL_ROUTER"
    else:
        raise RuntimeError(f"Unsupported evidence integration mode: {integration_mode}")
    runtime_chunks, runtime_chunk_weights = _runtime_chunk_contract(bundle)
    runtime_permutation = _frozen_chunk_permutation(
        runtime_chunks, seed=seed, fold=fold
    )
    has_runtime_schedule = getattr(bundle, "runtime_schedule", None) is not None
    graph = payload.get("graph")
    if has_runtime_schedule and graph is not None:
        raise RuntimeError(
            "A fixed prepared graph cannot override the registered runtime schedule"
        )
    graph = (
        None
        if has_runtime_schedule
        else (_move(graph, device) if graph is not None else move_graph(bundle, device, "cc_hhgt"))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(primary.get("learning_rate", 1e-3)),
        weight_decay=float(primary.get("weight_decay", 1e-5)),
    )
    max_cycles = int(runtime.get("max_coverage_cycles", 40))
    patience = int(runtime.get("patience_coverage_cycles", 6))
    minimum_delta = float(runtime.get("validation_min_delta", 1e-4))
    require_early_stop = bool(runtime.get("require_early_stopping_before_cap", False))
    direction_weight = float(primary.get("direction_loss_weight", 0.25))
    shrinkage = float(primary.get("residual_shrinkage", 1e-4))
    if min(max_cycles, patience) < 1 or minimum_delta < 0:
        raise RuntimeError("Training cycle, accumulation and patience values must be positive")
    candidate_chunk_schedule_mode = _resolve_candidate_chunk_schedule_mode(runtime)
    allow_partial_candidate_chunk_rotation = _resolve_partial_rotation_authorization(
        runtime, schedule_mode=candidate_chunk_schedule_mode
    )
    gradient_algorithm = _gradient_algorithm_for_schedule_mode(
        candidate_chunk_schedule_mode
    )
    candidate_batch_row_counts = tuple(
        _batch_row_count(batch) for batch in payload["train_batches"]
    )
    optimizer_batch_groups = _canonical_optimizer_batch_groups(
        len(candidate_batch_row_counts), runtime_plan.gradient_accumulation
    )
    optimizer_group_row_counts = _optimizer_group_row_counts(
        candidate_batch_row_counts, optimizer_batch_groups
    )
    candidate_chunk_assignment_unit = (
        "optimizer_group"
        if candidate_chunk_schedule_mode
        == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
        else "candidate_batch"
    )
    optimizer_group_count = len(optimizer_batch_groups)
    full_optimizer_group_rows = max(optimizer_group_row_counts)
    final_optimizer_group_rows = optimizer_group_row_counts[-1]
    if (
        candidate_chunk_schedule_mode
        == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
        and any(
            rows > runtime_plan.candidate_microbatch_size
            for rows in candidate_batch_row_counts
        )
    ):
        raise RuntimeError(
            "Shared-encoder group mode forbids candidate-batch micro-splitting"
        )
    pass_offset_permutation = _frozen_pass_offset_permutation(
        runtime_chunks, seed=seed, fold=fold
    )
    train_cycle_schedules = _candidate_chunk_cycle_schedules(
        batch_count=len(payload["train_batches"]),
        chunks=runtime_chunks,
        permutation=runtime_permutation,
        pass_offset_permutation=pass_offset_permutation,
        max_cycles=max_cycles,
        mode=candidate_chunk_schedule_mode,
        gradient_accumulation=runtime_plan.gradient_accumulation,
    )
    schedule_payload = _candidate_chunk_schedule_payload(
        mode=candidate_chunk_schedule_mode,
        chunks=runtime_chunks,
        weights=runtime_chunk_weights,
        permutation=runtime_permutation,
        pass_offset_permutation=pass_offset_permutation,
        batch_count=len(payload["train_batches"]),
        cycle_schedules=train_cycle_schedules,
        gradient_accumulation=runtime_plan.gradient_accumulation,
        batch_row_counts=candidate_batch_row_counts,
    )
    candidate_chunk_schedule_sha256 = _candidate_chunk_schedule_sha256(
        schedule_payload
    )
    if candidate_chunk_schedule_mode == CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN:
        checkpoint_scope = "COMPLETE_COVERAGE_CYCLE"
    elif (
        candidate_chunk_schedule_mode
        == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
    ):
        checkpoint_scope = "COMPLETE_BALANCED_GROUP_SHARED_ENCODER_CYCLE"
    else:
        checkpoint_scope = "COMPLETE_BALANCED_SINGLE_PASS_ESTIMATOR_CYCLE"
    runtime_training_contract = {
        "graph_variant": graph_variant,
        "mixed_precision": precision,
        "candidate_microbatch_size": runtime_plan.candidate_microbatch_size,
        "gradient_accumulation": runtime_plan.gradient_accumulation,
        "oom_fallback_microbatch_size": runtime_plan.fallback_microbatch_size,
        "oom_fallback_gradient_accumulation": (
            runtime_plan.fallback_gradient_accumulation
        ),
        "gradient_algorithm": gradient_algorithm,
        "candidate_chunk_assignment_unit": candidate_chunk_assignment_unit,
        "optimizer_group_count": optimizer_group_count,
        "optimizer_group_row_counts": list(optimizer_group_row_counts),
        "full_optimizer_group_rows": full_optimizer_group_rows,
        "final_optimizer_group_rows": final_optimizer_group_rows,
        "maximum_live_full_graph_encoder_activations": 1,
        "global_loss_calls_per_optimizer_group": (
            1
            if candidate_chunk_schedule_mode
            == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
            else 2
        ),
        "oom_behavior": (
            "SHARED_ENCODER_GROUP_FAIL_CLOSED_NO_FALLBACK"
            if candidate_chunk_schedule_mode
            == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
            else "TWO_PASS_MICROBATCH_RETRY_ONLY"
        ),
        "oom_fallback_edge_chunking": "UNSUPPORTED_SEALED_GRAPH_FAIL_CLOSED",
        "candidate_chunk_schedule_mode": candidate_chunk_schedule_mode,
        "candidate_chunk_schedule_sha256": candidate_chunk_schedule_sha256,
        "checkpoint_scope": checkpoint_scope,
        "allow_partial_candidate_chunk_rotation": (
            allow_partial_candidate_chunk_rotation
        ),
        "training_call_telemetry_scope": (
            "OPTIMIZER_GROUPS_ONLY_EXCLUDES_VALIDATION"
        ),
        "validation_calls_included_in_training_telemetry": False,
    }
    runtime_training_contract_sha256 = hashlib.sha256(
        json.dumps(
            runtime_training_contract,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    output_root = _resolve_output_root(config, root) / ctx["task_id"].replace("|", "__")
    checkpoint_path = output_root / "last_training_state.pt"
    best_path = output_root / "best_model_state.pt"
    log_path = output_root / "training_log.jsonl"
    fingerprint = _runtime_fingerprint(
        torch,
        ctx["hardware_class"],
        device,
        seed=seed,
        mixed_precision=precision,
    )
    start_cycle = 0
    best_loss = float("inf")
    patience_anchor = float("inf")
    stale = 0
    optimizer_steps = 0
    encoder_forward_calls = 0
    decoder_forward_calls = 0
    global_loss_calls = 0
    backward_calls = 0
    oom_fallback_active = False
    oom_fallback_events = 0
    last_optimizer_guard: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        observed_checkpoint_format = state.get("checkpoint_format")
        if observed_checkpoint_format == LEGACY_CHECKPOINT_FORMAT_V3:
            raise RuntimeError(
                "RESUME_CHECKPOINT_V3_TYPED_REJECT_REQUIRES_V4_GROUP_SCHEDULE"
            )
        if observed_checkpoint_format != CHECKPOINT_FORMAT:
            raise RuntimeError(
                "Resume checkpoint format mismatch: "
                f"{observed_checkpoint_format!r}!={CHECKPOINT_FORMAT!r}"
            )
        if state.get("runtime_fingerprint") != fingerprint:
            raise RuntimeError("Cross-hardware/runtime resume is forbidden")
        if state.get("artifact_hashes") != ctx["artifact_hashes"]:
            raise RuntimeError("Resume checkpoint artifact hashes changed")
        if state.get("architecture_id") != architecture_id:
            raise RuntimeError("Resume checkpoint evidence-integration architecture changed")
        required_resume = {
            "rng_state",
            "history",
            "task_id",
            "patient_fold",
            "seed",
            "graph_variant",
            "checkpoint_scope",
            "candidate_chunk_schedule_mode",
            "candidate_chunk_schedule_sha256",
            "candidate_chunk_schedule_configured_cycles",
            "candidate_chunk_pass_offset_permutation",
            "runtime_chunk_permutation",
            "runtime_chunk_weights",
            "runtime_training_contract_sha256",
            "optimizer_steps",
            "gradient_algorithm",
            "candidate_chunk_assignment_unit",
            "optimizer_group_count",
            "full_optimizer_group_rows",
            "final_optimizer_group_rows",
            "encoder_forward_calls",
            "decoder_forward_calls",
            "global_loss_calls",
            "backward_calls",
            "completed_selected_pass_offsets",
            "completed_rotation_fraction",
            "full_pair_rotation_completed",
            "oom_fallback_active",
            "oom_fallback_events",
            "prepared_artifact_sha256",
            "last_optimizer_guard",
            "allow_partial_candidate_chunk_rotation",
            "training_call_telemetry_scope",
            "validation_calls_included_in_training_telemetry",
        }
        if missing := sorted(required_resume - set(state)):
            raise RuntimeError(f"Resume checkpoint is not exact; missing fields: {missing}")
        checkpoint_cycle = int(state.get("cycle", -1))
        if not (0 <= checkpoint_cycle < max_cycles):
            raise RuntimeError(f"Resume checkpoint cycle is invalid: {checkpoint_cycle}")
        expected_rotation_receipt = _completed_rotation_receipt(
            train_cycle_schedules[: checkpoint_cycle + 1], runtime_permutation
        )
        identity = {
            "task_id": ctx["task_id"],
            "patient_fold": fold,
            "seed": seed,
            "graph_variant": graph_variant,
            "checkpoint_scope": checkpoint_scope,
            "candidate_chunk_schedule_mode": candidate_chunk_schedule_mode,
            "candidate_chunk_schedule_sha256": candidate_chunk_schedule_sha256,
            "candidate_chunk_schedule_configured_cycles": max_cycles,
            "candidate_chunk_pass_offset_permutation": list(
                pass_offset_permutation
            ),
            "runtime_chunk_permutation": list(runtime_permutation),
            "runtime_chunk_weights": {
                str(key): value for key, value in runtime_chunk_weights.items()
            },
            "runtime_training_contract_sha256": runtime_training_contract_sha256,
            "gradient_algorithm": gradient_algorithm,
            "candidate_chunk_assignment_unit": candidate_chunk_assignment_unit,
            "optimizer_group_count": optimizer_group_count,
            "full_optimizer_group_rows": full_optimizer_group_rows,
            "final_optimizer_group_rows": final_optimizer_group_rows,
            "completed_selected_pass_offsets": expected_rotation_receipt[
                "completed_selected_pass_offsets"
            ],
            "completed_rotation_fraction": expected_rotation_receipt[
                "completed_rotation_fraction"
            ],
            "full_pair_rotation_completed": expected_rotation_receipt[
                "full_pair_rotation_completed"
            ],
            "prepared_artifact_sha256": prepared_artifact_sha256,
            "allow_partial_candidate_chunk_rotation": (
                allow_partial_candidate_chunk_rotation
            ),
            "training_call_telemetry_scope": (
                "OPTIMIZER_GROUPS_ONLY_EXCLUDES_VALIDATION"
            ),
            "validation_calls_included_in_training_telemetry": False,
        }
        drift = {
            key: {"checkpoint": state.get(key), "current": value}
            for key, value in identity.items()
            if state.get(key) != value
        }
        if drift:
            raise RuntimeError(f"Resume checkpoint task identity changed: {drift}")
        validate_resume_history_and_counters(
            state,
            checkpoint_cycle=checkpoint_cycle,
            schedule_mode=candidate_chunk_schedule_mode,
            optimizer_group_count=optimizer_group_count,
            candidate_batch_count=len(candidate_batch_row_counts),
        )
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_cycle = int(state["cycle"]) + 1
        best_loss = float(state["best_validation_loss"])
        patience_anchor = float(state.get("patience_anchor_validation_loss", best_loss))
        stale = int(state["stale_cycles"])
        optimizer_steps = state["optimizer_steps"]
        encoder_forward_calls = state["encoder_forward_calls"]
        decoder_forward_calls = state["decoder_forward_calls"]
        global_loss_calls = state["global_loss_calls"]
        backward_calls = state["backward_calls"]
        oom_fallback_active = bool(state["oom_fallback_active"])
        oom_fallback_events = int(state["oom_fallback_events"])
        if (
            candidate_chunk_schedule_mode
            == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
            and (oom_fallback_active or oom_fallback_events)
        ):
            raise RuntimeError(
                "Shared-encoder group mode cannot resume an OOM-fallback state"
            )
        if not isinstance(state["last_optimizer_guard"], Mapping):
            raise RuntimeError("Resume checkpoint optimizer guard is malformed")
        last_optimizer_guard = dict(state["last_optimizer_guard"])
        for required_true in (
            "grad_finite",
            "parameters_finite",
            "parameter_delta_positive",
        ):
            if last_optimizer_guard.get(required_true) is not True:
                raise RuntimeError(
                    f"Resume checkpoint optimizer guard failed: {required_true}"
                )
        history = list(state["history"])
        if len(history) != start_cycle:
            raise RuntimeError(
                "Resume checkpoint cycle/history mismatch: "
                f"cycle={state['cycle']}, history_rows={len(history)}"
            )
        restore_rng_state(torch, state["rng_state"])
        _atomic_jsonl_write(history, log_path)

    stop_reason = "HARD_CAP"
    completed_cycles = start_cycle

    def graph_for_chunk(chunk: int):
        return (
            _runtime_graph_for_chunk(bundle, chunk, device=device)
            if has_runtime_schedule
            else graph
        )

    for cycle in range(start_cycle, max_cycles):
        model.train()
        train_batches = payload["train_batches"]
        cycle_schedule = train_cycle_schedules[cycle]
        cycle_pair_count = sum(len(pass_pairs) for pass_pairs in cycle_schedule)
        cycle_chunk_exposures = {chunk: 0 for chunk in runtime_chunks}
        cycle_chunk_row_exposures = {chunk: 0 for chunk in runtime_chunks}
        for scheduled_pass in cycle_schedule:
            for batch_index, chunk in scheduled_pass:
                cycle_chunk_exposures[int(chunk)] += 1
                cycle_chunk_row_exposures[int(chunk)] += candidate_batch_row_counts[
                    int(batch_index)
                ]
        cycle_selected_pass_offsets = [
            runtime_permutation.index(pass_pairs[0][1])
            for pass_pairs in cycle_schedule
            if pass_pairs
        ]
        cycle_optimizer_steps_before = optimizer_steps
        cycle_encoder_forward_calls_before = encoder_forward_calls
        cycle_decoder_forward_calls_before = decoder_forward_calls
        cycle_global_loss_calls_before = global_loss_calls
        cycle_backward_calls_before = backward_calls
        # Exact mode executes the legacy B x K oracle.  Balanced modes execute
        # one frozen estimator cycle.  Validation and checkpoint/early-stop
        # decisions cannot observe a partial mode-specific cycle.
        for pass_index, pass_pairs in enumerate(cycle_schedule):
            selected_pass_offset = runtime_permutation.index(pass_pairs[0][1])
            primary_work_items = materialize_pass_work_items(
                pass_pairs,
                train_batches,
                microbatch_size=runtime_plan.candidate_microbatch_size,
            )
            for group_index, group_start in enumerate(
                range(0, len(primary_work_items), runtime_plan.gradient_accumulation)
            ):
                primary_group = primary_work_items[
                    group_start : group_start + runtime_plan.gradient_accumulation
                ]
                shared_encoder_group_mode = (
                    candidate_chunk_schedule_mode
                    == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
                )
                if shared_encoder_group_mode and oom_fallback_active:
                    raise RuntimeError(
                        "Shared-encoder group mode forbids OOM fallback semantics"
                    )
                group = primary_group
                group_used_fallback = False
                group_start_rng = _capture_torch_rng_state(torch)
                group_start_buffers = _capture_model_buffers(model)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                if shared_encoder_group_mode:
                    try:
                        loss_plan, call_telemetry = (
                            shared_encoder_conventional_group_backward(
                                model,
                                group,
                                graph_for_chunk,
                                torch=torch,
                                device=device,
                                precision=precision,
                                direction_loss_weight=direction_weight,
                                shrinkage=shrinkage,
                            )
                        )
                    except BaseException as exc:
                        if not _is_cuda_oom(torch, exc):
                            raise
                        optimizer.zero_grad(set_to_none=True)
                        _restore_torch_rng_state(torch, group_start_rng)
                        _restore_model_buffers(model, group_start_buffers)
                        torch.cuda.empty_cache()
                        raise RuntimeError(
                            "CUDA_OOM_SHARED_ENCODER_GROUP_FAIL_CLOSED; "
                            "fallback would change the registered algorithm"
                        ) from exc
                    expected_group_rows = optimizer_group_row_counts[group_index]
                    if call_telemetry.group_rows != expected_group_rows:
                        raise RuntimeError(
                            "Shared-encoder optimizer-group row authority changed: "
                            f"{call_telemetry.group_rows}!={expected_group_rows}"
                        )
                else:
                    group = (
                        split_work_items_for_oom_fallback(
                            primary_group,
                            microbatch_size=runtime_plan.fallback_microbatch_size,
                            maximum_fragments=runtime_plan.fallback_gradient_accumulation,
                        )
                        if oom_fallback_active
                        else primary_group
                    )
                    group_used_fallback = bool(oom_fallback_active)
                    try:
                        loss_plan = streaming_group_backward(
                            model,
                            group,
                            graph_for_chunk,
                            torch=torch,
                            device=device,
                            precision=precision,
                            direction_loss_weight=direction_weight,
                            shrinkage=shrinkage,
                        )
                    except BaseException as exc:
                        if not _is_cuda_oom(torch, exc):
                            raise
                        optimizer.zero_grad(set_to_none=True)
                        _restore_torch_rng_state(torch, group_start_rng)
                        _restore_model_buffers(model, group_start_buffers)
                        torch.cuda.empty_cache()
                        if oom_fallback_active:
                            raise RuntimeError(
                                "CUDA_OOM_FALLBACK_EXHAUSTED; sealed graph edge "
                                "chunking cannot be changed by the trainer"
                            ) from exc
                        oom_fallback_active = True
                        oom_fallback_events += 1
                        group_used_fallback = True
                        group = split_work_items_for_oom_fallback(
                            primary_group,
                            microbatch_size=runtime_plan.fallback_microbatch_size,
                            maximum_fragments=runtime_plan.fallback_gradient_accumulation,
                        )
                        try:
                            loss_plan = streaming_group_backward(
                                model,
                                group,
                                graph_for_chunk,
                                torch=torch,
                                device=device,
                                precision=precision,
                                direction_loss_weight=direction_weight,
                                shrinkage=shrinkage,
                            )
                        except BaseException as fallback_exc:
                            if _is_cuda_oom(torch, fallback_exc):
                                optimizer.zero_grad(set_to_none=True)
                                _restore_torch_rng_state(torch, group_start_rng)
                                _restore_model_buffers(model, group_start_buffers)
                                torch.cuda.empty_cache()
                                raise RuntimeError(
                                    "CUDA_OOM_FALLBACK_EXHAUSTED; sealed graph edge "
                                    "chunking cannot be changed by the trainer"
                                ) from fallback_exc
                            raise
                    call_telemetry = BackwardCallTelemetry(
                        unique_chunks=len({int(chunk) for _, chunk in group}),
                        group_rows=sum(_batch_row_count(batch) for batch, _ in group),
                        encoder_forward_calls=2 * len(group),
                        decoder_forward_calls=2 * len(group),
                        global_loss_calls=2,
                        backward_calls=len(group),
                    )
                try:
                    update_probe = optimizer_step_with_guards(
                        model,
                        optimizer,
                        torch=torch,
                        objective_value=loss_plan.objective_value,
                    )
                except BaseException as exc:
                    if _is_cuda_oom(torch, exc):
                        raise RuntimeError(
                            "CUDA_OOM_DURING_OPTIMIZER_STEP_IS_NOT_SAFE_TO_RETRY"
                        ) from exc
                    raise
                optimizer_steps += 1
                encoder_forward_calls += call_telemetry.encoder_forward_calls
                decoder_forward_calls += call_telemetry.decoder_forward_calls
                global_loss_calls += call_telemetry.global_loss_calls
                backward_calls += call_telemetry.backward_calls
                last_optimizer_guard = dict(update_probe)
                peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
                optimizer.zero_grad(set_to_none=True)
                print(
                    json.dumps(
                        {
                            "status": "OPTIMIZER_STEP",
                            "task_id": ctx["task_id"],
                            "graph_variant": graph_variant,
                            "cycle": cycle,
                            "pass": pass_index,
                            "selected_pass_offset": selected_pass_offset,
                            "group": group_index,
                            "optimizer_step": optimizer_steps,
                            "loss": loss_plan.objective_value,
                            "mixed_precision": precision["active"],
                            "oom_fallback_used": group_used_fallback,
                            "gradient_algorithm": gradient_algorithm,
                            "candidate_chunk_assignment_unit": schedule_payload[
                                "assignment_unit"
                            ],
                            "unique_chunks": call_telemetry.unique_chunks,
                            "group_rows": call_telemetry.group_rows,
                            "encoder_forward_calls": (
                                call_telemetry.encoder_forward_calls
                            ),
                            "decoder_forward_calls": (
                                call_telemetry.decoder_forward_calls
                            ),
                            "global_loss_calls": call_telemetry.global_loss_calls,
                            "backward_calls": call_telemetry.backward_calls,
                            "encoder_forward_calls_total": encoder_forward_calls,
                            "decoder_forward_calls_total": decoder_forward_calls,
                            "global_loss_calls_total": global_loss_calls,
                            "backward_calls_total": backward_calls,
                            "peak_reserved_bytes": peak_reserved_bytes,
                            **update_probe,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        validation_microbatch_size = (
            runtime_plan.fallback_microbatch_size
            if oom_fallback_active
            else runtime_plan.candidate_microbatch_size
        )

        def evaluate_at_microbatch_size(microbatch_size: int) -> float:
            validation_batches = split_batches_for_evaluation(
                payload["validation_batches"], microbatch_size=microbatch_size
            )

            def validation_progress(
                runtime_chunk: int,
                completed_batches: int,
                total_batches: int,
            ) -> None:
                if (
                    completed_batches != 1
                    and completed_batches != total_batches
                    and completed_batches % 16 != 0
                ):
                    return
                print(
                    json.dumps(
                        {
                            "status": "VALIDATION_HEARTBEAT",
                            "task_id": ctx["task_id"],
                            "graph_variant": graph_variant,
                            "cycle": cycle,
                            "runtime_chunk": runtime_chunk,
                            "completed_batches": completed_batches,
                            "total_batches": total_batches,
                            "microbatch_size": microbatch_size,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            return (
                _evaluate_full_runtime_coverage(
                    model,
                    bundle,
                    validation_batches,
                    chunks=runtime_chunks,
                    weights=runtime_chunk_weights,
                    torch=torch,
                    device=device,
                    direction_loss_weight=direction_weight,
                    shrinkage=shrinkage,
                    precision=precision,
                    progress_callback=validation_progress,
                )
                if has_runtime_schedule
                else _evaluate(
                    model,
                    graph,
                    validation_batches,
                    torch=torch,
                    device=device,
                    direction_loss_weight=direction_weight,
                    shrinkage=shrinkage,
                    precision=precision,
                    progress_callback=validation_progress,
                )
            )

        print(
            json.dumps(
                {
                    "status": "VALIDATION_HEARTBEAT",
                    "task_id": ctx["task_id"],
                    "graph_variant": graph_variant,
                    "cycle": cycle,
                    "phase": "START",
                    "microbatch_size": validation_microbatch_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            validation_loss = evaluate_at_microbatch_size(
                validation_microbatch_size
            )
        except BaseException as exc:
            if (
                candidate_chunk_schedule_mode
                == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
                and _is_cuda_oom(torch, exc)
            ):
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA_OOM_SHARED_ENCODER_MODE_VALIDATION_FAIL_CLOSED; "
                    "runtime fallback is forbidden for this mode"
                ) from exc
            if not _is_cuda_oom(torch, exc) or oom_fallback_active:
                raise
            torch.cuda.empty_cache()
            oom_fallback_active = True
            oom_fallback_events += 1
            try:
                validation_loss = evaluate_at_microbatch_size(
                    runtime_plan.fallback_microbatch_size
                )
            except BaseException as fallback_exc:
                if _is_cuda_oom(torch, fallback_exc):
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA_OOM_VALIDATION_FALLBACK_EXHAUSTED; sealed graph edge "
                        "chunking cannot be changed by the trainer"
                    ) from fallback_exc
                raise
        if not math.isfinite(validation_loss):
            raise RuntimeError(f"NONFINITE_VALIDATION_LOSS={validation_loss}")
        rotation_receipt_through_cycle = _completed_rotation_receipt(
            train_cycle_schedules[: cycle + 1], runtime_permutation
        )
        exact_improved = validation_loss < best_loss - 1e-12
        if exact_improved:
            best_loss = validation_loss
            _atomic_torch_save(
                torch,
                {
                    "checkpoint_format": CHECKPOINT_FORMAT,
                    "model_state": model.state_dict(),
                    "validation_loss": validation_loss,
                    "cycle": cycle,
                    "task_id": ctx["task_id"],
                    "patient_fold": fold,
                    "seed": seed,
                    "graph_variant": graph_variant,
                    "checkpoint_scope": checkpoint_scope,
                    "artifact_hashes": ctx["artifact_hashes"],
                    "input_authority_hashes": input_authority_hashes,
                    "runtime_fingerprint": fingerprint,
                    "architecture_id": architecture_id,
                    "prepared_artifact_sha256": prepared_artifact_sha256,
                    "runtime_training_contract": runtime_training_contract,
                    "runtime_training_contract_sha256": runtime_training_contract_sha256,
                    "gradient_algorithm": gradient_algorithm,
                    "candidate_chunk_assignment_unit": candidate_chunk_assignment_unit,
                    "optimizer_group_count": optimizer_group_count,
                    "full_optimizer_group_rows": full_optimizer_group_rows,
                    "final_optimizer_group_rows": final_optimizer_group_rows,
                    "encoder_forward_calls": encoder_forward_calls,
                    "decoder_forward_calls": decoder_forward_calls,
                    "global_loss_calls": global_loss_calls,
                    "backward_calls": backward_calls,
                    "completed_selected_pass_offsets": (
                        rotation_receipt_through_cycle[
                            "completed_selected_pass_offsets"
                        ]
                    ),
                    "completed_rotation_fraction": rotation_receipt_through_cycle[
                        "completed_rotation_fraction"
                    ],
                    "full_pair_rotation_completed": rotation_receipt_through_cycle[
                        "full_pair_rotation_completed"
                    ],
                    "oom_fallback_active": oom_fallback_active,
                    "oom_fallback_events": oom_fallback_events,
                    "candidate_chunk_schedule_mode": candidate_chunk_schedule_mode,
                    "candidate_chunk_schedule_sha256": candidate_chunk_schedule_sha256,
                    "candidate_chunk_schedule_configured_cycles": max_cycles,
                    "allow_partial_candidate_chunk_rotation": (
                        allow_partial_candidate_chunk_rotation
                    ),
                    "training_call_telemetry_scope": (
                        "OPTIMIZER_GROUPS_ONLY_EXCLUDES_VALIDATION"
                    ),
                    "validation_calls_included_in_training_telemetry": False,
                    "candidate_chunk_pass_offset_permutation": list(
                        pass_offset_permutation
                    ),
                    "runtime_chunk_permutation": list(runtime_permutation),
                    "runtime_chunk_weights": {
                        str(key): value for key, value in runtime_chunk_weights.items()
                    },
                },
                best_path,
            )
        patience_improved = validation_loss < patience_anchor - minimum_delta
        if patience_improved:
            patience_anchor = validation_loss
            stale = 0
        else:
            stale += 1
        cycle_record = {
            "cycle": cycle,
            "graph_variant": graph_variant,
            "validation_loss": validation_loss,
            "best_validation_loss": best_loss,
            "patience_anchor_validation_loss": patience_anchor,
            "stale_cycles": stale,
            "validation_min_delta": minimum_delta,
            "task_id": ctx["task_id"],
            "runtime_num_chunks": len(runtime_chunks),
            "candidate_chunk_schedule_mode": candidate_chunk_schedule_mode,
            "candidate_passes": len(cycle_schedule),
            "candidate_chunk_pairs": cycle_pair_count,
            "candidate_chunk_exposures": {
                str(key): value for key, value in cycle_chunk_exposures.items()
            },
            "candidate_chunk_row_exposures": {
                str(key): value for key, value in cycle_chunk_row_exposures.items()
            },
            "selected_pass_offsets": cycle_selected_pass_offsets,
            "gradient_algorithm": gradient_algorithm,
            "candidate_chunk_assignment_unit": candidate_chunk_assignment_unit,
            "optimizer_group_count": optimizer_group_count,
            "optimizer_group_row_counts": list(optimizer_group_row_counts),
            "full_optimizer_group_rows": full_optimizer_group_rows,
            "final_optimizer_group_rows": final_optimizer_group_rows,
            "encoder_forward_calls": (
                encoder_forward_calls - cycle_encoder_forward_calls_before
            ),
            "decoder_forward_calls": (
                decoder_forward_calls - cycle_decoder_forward_calls_before
            ),
            "global_loss_calls": global_loss_calls - cycle_global_loss_calls_before,
            "backward_calls": backward_calls - cycle_backward_calls_before,
            "encoder_forward_calls_total": encoder_forward_calls,
            "decoder_forward_calls_total": decoder_forward_calls,
            "global_loss_calls_total": global_loss_calls,
            "backward_calls_total": backward_calls,
            "completed_rotation_fraction": rotation_receipt_through_cycle[
                "completed_rotation_fraction"
            ],
            "full_pair_rotation_completed": rotation_receipt_through_cycle[
                "full_pair_rotation_completed"
            ],
            "validation_full_runtime_coverage": True,
            "candidate_chunk_schedule_sha256": candidate_chunk_schedule_sha256,
            "allow_partial_candidate_chunk_rotation": (
                allow_partial_candidate_chunk_rotation
            ),
            "runtime_training_contract_sha256": runtime_training_contract_sha256,
            "mixed_precision": precision["active"],
            "optimizer_steps": optimizer_steps,
            "optimizer_steps_this_cycle": (
                optimizer_steps - cycle_optimizer_steps_before
            ),
            "oom_fallback_active": oom_fallback_active,
            "oom_fallback_events": oom_fallback_events,
        }
        history.append(cycle_record)
        _atomic_torch_save(
            torch,
            {
                "checkpoint_format": CHECKPOINT_FORMAT,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cycle": cycle,
                "best_validation_loss": best_loss,
                "patience_anchor_validation_loss": patience_anchor,
                "stale_cycles": stale,
                "artifact_hashes": ctx["artifact_hashes"],
                "input_authority_hashes": input_authority_hashes,
                "runtime_fingerprint": fingerprint,
                "architecture_id": architecture_id,
                "task_id": ctx["task_id"],
                "patient_fold": fold,
                "seed": seed,
                "graph_variant": graph_variant,
                "checkpoint_scope": checkpoint_scope,
                "gradient_algorithm": gradient_algorithm,
                "candidate_chunk_assignment_unit": candidate_chunk_assignment_unit,
                "optimizer_group_count": optimizer_group_count,
                "full_optimizer_group_rows": full_optimizer_group_rows,
                "final_optimizer_group_rows": final_optimizer_group_rows,
                "candidate_chunk_schedule_mode": candidate_chunk_schedule_mode,
                "candidate_chunk_schedule_sha256": candidate_chunk_schedule_sha256,
                "candidate_chunk_schedule_configured_cycles": max_cycles,
                "allow_partial_candidate_chunk_rotation": (
                    allow_partial_candidate_chunk_rotation
                ),
                "training_call_telemetry_scope": (
                    "OPTIMIZER_GROUPS_ONLY_EXCLUDES_VALIDATION"
                ),
                "validation_calls_included_in_training_telemetry": False,
                "candidate_chunk_pass_offset_permutation": list(
                    pass_offset_permutation
                ),
                "runtime_chunk_permutation": list(runtime_permutation),
                "runtime_chunk_weights": {
                    str(key): value for key, value in runtime_chunk_weights.items()
                },
                "prepared_artifact_sha256": prepared_artifact_sha256,
                "runtime_training_contract": runtime_training_contract,
                "runtime_training_contract_sha256": runtime_training_contract_sha256,
                "optimizer_steps": optimizer_steps,
                "encoder_forward_calls": encoder_forward_calls,
                "decoder_forward_calls": decoder_forward_calls,
                "global_loss_calls": global_loss_calls,
                "backward_calls": backward_calls,
                "completed_selected_pass_offsets": rotation_receipt_through_cycle[
                    "completed_selected_pass_offsets"
                ],
                "completed_rotation_fraction": rotation_receipt_through_cycle[
                    "completed_rotation_fraction"
                ],
                "full_pair_rotation_completed": rotation_receipt_through_cycle[
                    "full_pair_rotation_completed"
                ],
                "oom_fallback_active": oom_fallback_active,
                "oom_fallback_events": oom_fallback_events,
                "last_optimizer_guard": last_optimizer_guard,
                "rng_state": capture_rng_state(torch),
                "history": history,
            },
            checkpoint_path,
        )
        _atomic_jsonl_write(history, log_path)
        print(json.dumps({
            "status": "CYCLE",
            "task_id": ctx["task_id"],
            "graph_variant": graph_variant,
            "cycle": cycle,
            "validation_loss": validation_loss,
            "best_validation_loss": best_loss,
            "patience_anchor_validation_loss": patience_anchor,
            "stale_cycles": stale,
            "optimizer_steps": optimizer_steps,
            "optimizer_steps_this_cycle": (
                optimizer_steps - cycle_optimizer_steps_before
            ),
            "candidate_chunk_schedule_mode": candidate_chunk_schedule_mode,
            "gradient_algorithm": gradient_algorithm,
            "candidate_chunk_assignment_unit": candidate_chunk_assignment_unit,
            "candidate_passes": len(cycle_schedule),
            "candidate_chunk_pairs": cycle_pair_count,
            "selected_pass_offsets": cycle_selected_pass_offsets,
            "encoder_forward_calls": (
                encoder_forward_calls - cycle_encoder_forward_calls_before
            ),
            "decoder_forward_calls": (
                decoder_forward_calls - cycle_decoder_forward_calls_before
            ),
            "global_loss_calls": global_loss_calls - cycle_global_loss_calls_before,
            "backward_calls": backward_calls - cycle_backward_calls_before,
            "mixed_precision": precision["active"],
            "oom_fallback_active": oom_fallback_active,
            "oom_fallback_events": oom_fallback_events,
        }), flush=True)
        completed_cycles = cycle + 1
        if stale >= patience:
            stop_reason = "EARLY_STOPPING"
            break

    if require_early_stop and stop_reason == "HARD_CAP":
        failure = {
            "status": "FAILED_HARD_CAP",
            "run_id": ctx["run_id"],
            "task_id": ctx["task_id"],
            "patient_fold": fold,
            "seed": seed,
            "graph_variant": graph_variant,
            "completed_cycles": completed_cycles,
            "configured_max_cycles": max_cycles,
            "best_validation_loss": best_loss,
            "artifact_hashes": ctx["artifact_hashes"],
            "runtime_fingerprint": fingerprint,
            "candidate_chunk_schedule_mode": candidate_chunk_schedule_mode,
            "gradient_algorithm": gradient_algorithm,
            "allow_partial_candidate_chunk_rotation": (
                allow_partial_candidate_chunk_rotation
            ),
        }
        output_root.mkdir(parents=True, exist_ok=True)
        temporary = output_root / ".FAILURE.json.tmp"
        temporary.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_root / "FAILURE.json")
        raise RuntimeError(
            f"Formal task reached hard cap {max_cycles} before early stopping"
        )

    if (
        optimizer_steps < 1
        or last_optimizer_guard is None
        or last_optimizer_guard.get("grad_finite") is not True
        or last_optimizer_guard.get("parameters_finite") is not True
        or last_optimizer_guard.get("parameter_delta_positive") is not True
    ):
        raise RuntimeError(
            "TRAINING_CANNOT_SUCCEED_WITHOUT_VERIFIED_OPTIMIZER_STEP="
            f"steps={optimizer_steps}, guard={last_optimizer_guard}"
        )
    if not math.isfinite(best_loss) or not best_path.is_file():
        raise RuntimeError(
            "TRAINING_CANNOT_SUCCEED_WITHOUT_FINITE_BEST_CHECKPOINT="
            f"best_loss={best_loss}, best_path={best_path}"
        )
    best_model_state_size_bytes = int(best_path.stat().st_size)
    if best_model_state_size_bytes <= 0:
        raise RuntimeError(f"BEST_CHECKPOINT_EMPTY={best_path}")
    best_model_state_sha256 = _file_sha256(best_path)

    completed_rotation_receipt = _completed_rotation_receipt(
        train_cycle_schedules[:completed_cycles], runtime_permutation
    )
    if (
        completed_rotation_receipt["full_pair_rotation_completed"] is not True
        and allow_partial_candidate_chunk_rotation is not True
    ):
        raise RuntimeError(
            "PARTIAL_CANDIDATE_CHUNK_ROTATION_COMPLETED_WITHOUT_AUTHORIZATION"
        )
    completed_chunk_exposures: list[dict[str, int]] = []
    completed_chunk_row_exposures: list[dict[str, int]] = []
    total_chunk_exposures = {chunk: 0 for chunk in runtime_chunks}
    total_chunk_row_exposures = {chunk: 0 for chunk in runtime_chunks}
    for cycle_schedule in train_cycle_schedules[:completed_cycles]:
        exposures = {chunk: 0 for chunk in runtime_chunks}
        row_exposures = {chunk: 0 for chunk in runtime_chunks}
        for pass_pairs in cycle_schedule:
            for batch_index, chunk in pass_pairs:
                chunk = int(chunk)
                exposures[chunk] += 1
                row_exposures[chunk] += candidate_batch_row_counts[int(batch_index)]
                total_chunk_exposures[chunk] += 1
                total_chunk_row_exposures[chunk] += candidate_batch_row_counts[
                    int(batch_index)
                ]
        completed_chunk_exposures.append(
            {str(key): value for key, value in exposures.items()}
        )
        completed_chunk_row_exposures.append(
            {str(key): value for key, value in row_exposures.items()}
        )
    if (
        candidate_chunk_schedule_mode
        == CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
    ):
        expected_shared_optimizer_steps = completed_cycles * optimizer_group_count
        expected_shared_decoder_calls = completed_cycles * len(candidate_batch_row_counts)
        observed_shared_counts = (
            optimizer_steps,
            encoder_forward_calls,
            decoder_forward_calls,
            global_loss_calls,
            backward_calls,
        )
        expected_shared_counts = (
            expected_shared_optimizer_steps,
            expected_shared_optimizer_steps,
            expected_shared_decoder_calls,
            expected_shared_optimizer_steps,
            expected_shared_optimizer_steps,
        )
        if observed_shared_counts != expected_shared_counts:
            raise RuntimeError(
                "Shared-encoder completed-call receipt mismatch: "
                f"observed={observed_shared_counts}, expected={expected_shared_counts}"
            )
        if oom_fallback_active or oom_fallback_events:
            raise RuntimeError(
                "Shared-encoder mode cannot produce SUCCESS after OOM fallback"
            )

    success = {
        "status": "SUCCESS",
        "run_id": ctx["run_id"],
        "task_id": ctx["task_id"],
        "patient_fold": fold,
        "seed": seed,
        "graph_variant": graph_variant,
        "stop_reason": stop_reason,
        "completed_cycles": completed_cycles,
        "configured_max_cycles": max_cycles,
        "validation_min_delta": minimum_delta,
        "did_not_hit_hard_cap": stop_reason != "HARD_CAP",
        "best_validation_loss": best_loss,
        "best_model_state_sha256": best_model_state_sha256,
        "best_model_state_size_bytes": best_model_state_size_bytes,
        "artifact_hashes": ctx["artifact_hashes"],
        "input_authority_hashes": input_authority_hashes,
        "prepared_artifact_sha256": prepared_artifact_sha256,
        "runtime_fingerprint": fingerprint,
        "runtime_training_contract": runtime_training_contract,
        "runtime_training_contract_sha256": runtime_training_contract_sha256,
        "mixed_precision": precision["active"],
        "optimizer_steps": optimizer_steps,
        "training_optimizer_steps": optimizer_steps,
        "gradient_algorithm": gradient_algorithm,
        "candidate_chunk_assignment_unit": candidate_chunk_assignment_unit,
        "candidate_batch_count": len(candidate_batch_row_counts),
        "candidate_batch_row_counts": list(candidate_batch_row_counts),
        "optimizer_group_count": optimizer_group_count,
        "optimizer_group_row_counts": list(optimizer_group_row_counts),
        "full_optimizer_group_rows": full_optimizer_group_rows,
        "final_optimizer_group_rows": final_optimizer_group_rows,
        "encoder_forward_calls": encoder_forward_calls,
        "decoder_forward_calls": decoder_forward_calls,
        "global_loss_calls": global_loss_calls,
        "backward_calls": backward_calls,
        "training_encoder_forward_calls": encoder_forward_calls,
        "training_decoder_forward_calls": decoder_forward_calls,
        "training_global_loss_calls": global_loss_calls,
        "training_backward_calls": backward_calls,
        "training_call_telemetry_scope": (
            "OPTIMIZER_GROUPS_ONLY_EXCLUDES_VALIDATION"
        ),
        "validation_calls_included_in_training_telemetry": False,
        "grad_finite": last_optimizer_guard["grad_finite"],
        "parameters_finite": last_optimizer_guard["parameters_finite"],
        "parameter_delta_positive": last_optimizer_guard[
            "parameter_delta_positive"
        ],
        "last_optimizer_guard": last_optimizer_guard,
        "oom_fallback_active": oom_fallback_active,
        "oom_fallback_events": oom_fallback_events,
        "architecture_id": architecture_id,
        "evidence_integration_mode": integration_mode,
        "runtime_num_chunks": len(runtime_chunks),
        "checkpoint_scope": checkpoint_scope,
        "candidate_chunk_schedule_mode": candidate_chunk_schedule_mode,
        "allow_partial_candidate_chunk_rotation": (
            allow_partial_candidate_chunk_rotation
        ),
        "candidate_passes_per_cycle": len(train_cycle_schedules[0]),
        "candidate_pairs_per_configured_cycle": [
            sum(len(pass_pairs) for pass_pairs in cycle_schedule)
            for cycle_schedule in train_cycle_schedules
        ],
        "candidate_pairs_per_completed_cycle": [
            sum(len(pass_pairs) for pass_pairs in cycle_schedule)
            for cycle_schedule in train_cycle_schedules[:completed_cycles]
        ],
        "candidate_pairs_completed_total": sum(
            len(pass_pairs)
            for cycle_schedule in train_cycle_schedules[:completed_cycles]
            for pass_pairs in cycle_schedule
        ),
        "candidate_chunk_pass_offset_permutation": list(
            pass_offset_permutation
        ),
        "completed_selected_pass_offsets_by_cycle": (
            completed_rotation_receipt["completed_selected_pass_offsets_by_cycle"]
        ),
        "completed_selected_pass_offsets": completed_rotation_receipt[
            "completed_selected_pass_offsets"
        ],
        "completed_unique_pass_offsets": completed_rotation_receipt[
            "completed_unique_pass_offsets"
        ],
        "completed_rotation_fraction": completed_rotation_receipt[
            "completed_rotation_fraction"
        ],
        "full_pair_rotation_completed": completed_rotation_receipt[
            "full_pair_rotation_completed"
        ],
        "partial_candidate_chunk_rotation_used": (
            completed_rotation_receipt["full_pair_rotation_completed"] is not True
        ),
        "candidate_full_pair_rotation_cycles": (
            1
            if candidate_chunk_schedule_mode
            == CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN
            else len(runtime_chunks)
        ),
        "candidate_passes_per_coverage_epoch": (
            len(train_cycle_schedules[0])
            if candidate_chunk_schedule_mode
            == CANDIDATE_CHUNK_SCHEDULE_EXACT_CARTESIAN
            else None
        ),
        "candidate_chunk_schedule_sha256": candidate_chunk_schedule_sha256,
        "candidate_chunk_exposures_per_completed_cycle": completed_chunk_exposures,
        "candidate_chunk_row_exposures_per_completed_cycle": (
            completed_chunk_row_exposures
        ),
        "candidate_chunk_exposures_completed_total": {
            str(key): value for key, value in total_chunk_exposures.items()
        },
        "candidate_chunk_row_exposures_completed_total": {
            str(key): value for key, value in total_chunk_row_exposures.items()
        },
        "runtime_chunk_permutation": list(runtime_permutation),
        "runtime_chunk_weights": {
            str(key): value for key, value in runtime_chunk_weights.items()
        },
        "validation_full_runtime_coverage": True,
        "test_labels_read": False,
        "test_metrics_computed": False,
        "formal_v32_primary_overwritten": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / ".SUCCESS.json.tmp"
    temporary.write_text(json.dumps(success, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_root / "SUCCESS.json")
    return 0
