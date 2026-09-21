"""One-step, hash-bound GPU backward probe for formal V3.2 training.

The callable in this module is intentionally reachable only through
``cc_hhgt.v32.cli run-shard``.  That CLI validates the approval and hashes the
code, config, input manifest and task manifest before importing this module.

The probe loads the exact prepared fold named by the authorized task, builds
the same formal model and runtime graph used by :mod:`cc_hhgt.v32.training`,
and delegates loss/backward/step semantics to that module's production
helpers.  It performs exactly one optimizer step and writes no checkpoint,
``SUCCESS.json`` or other formal-training artifact.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_ENVIRONMENT_GATE = (4096, 1)
_STRESS_GATE = (8192, 4)
_PEAK_RESERVED_LIMIT_BYTES = 22 * 1024**3
_REQUIRED_CUDA_DEVICE_TOKEN = "RTX 4090"
_AUTHORIZED_TRAINER = "cc_hhgt.v32.gpu_backward_probe:run_authorized_probe"
_ALLOWED_PROFILES = {
    _ENVIRONMENT_GATE: "ENVIRONMENT_GATE_4096_X_1",
    _STRESS_GATE: "PRIMARY_GROUP_STRESS_GATE_8192_X_4",
}


@dataclass(frozen=True)
class ProbeProfile:
    microbatch: int
    accumulation: int
    name: str

    @property
    def group_row_capacity(self) -> int:
        return self.microbatch * self.accumulation


def resolve_probe_profile(environ: Mapping[str, str] | None = None) -> ProbeProfile:
    """Resolve one of the two deliberately small, immutable probe profiles."""

    source = os.environ if environ is None else environ
    try:
        microbatch = int(source.get("V32_PROBE_MICROBATCH", "4096"))
        accumulation = int(source.get("V32_PROBE_ACCUMULATION", "1"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("V32 probe microbatch/accumulation must be integers") from exc
    pair = (microbatch, accumulation)
    if pair not in _ALLOWED_PROFILES:
        raise RuntimeError(
            "Unsupported V32 backward-probe profile; allowed profiles are "
            "4096x1 and 8192x4"
        )
    return ProbeProfile(microbatch, accumulation, _ALLOWED_PROFILES[pair])


def _require_authorized_pending_task(task: Mapping[str, str]) -> None:
    required = {
        "task_type": "CC_HHGT_PATIENT_FOLD",
        "model": "CC-HHGT",
        "status": "PENDING",
        "blocked_reason": "",
        "paid_task": "true",
    }
    drift = {
        key: {"observed": str(task.get(key, "")), "required": value}
        for key, value in required.items()
        if str(task.get(key, "")).strip().lower()
        != value.lower()
    }
    if drift:
        raise RuntimeError(f"Authorized probe task contract drift: {drift}")


def _reverify_authorization_context(context):
    """Re-run the complete guard so a merely constructed dataclass is denied."""

    from .training_guard import guard_training_entry

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
        raise RuntimeError("REAL_BACKWARD_PROBE_REVERIFIED_CONTEXT_DRIFT")
    return verified


def _validate_graph_variant_binding(
    *,
    run_id: str,
    config: Mapping[str, Any],
    payload: Mapping[str, Any],
    prepared_path: Path,
) -> str:
    """Bind the G0/G1/G2 task label, config, path and loaded payload."""

    expected = str(config.get("task_contract", {}).get("graph_variant", ""))
    if expected not in {"G0", "G1", "G2"}:
        raise RuntimeError(f"PROBE_CONFIG_GRAPH_VARIANT_INVALID={expected!r}")
    observed = {
        "run_id": str(run_id),
        "prepared_parent": prepared_path.resolve().parent.name,
        "payload": str(payload.get("formal_graph_variant", "")),
    }
    drift = {}
    if not str(run_id).startswith("v32-g012-") or f"-{expected.lower()}-" not in str(
        run_id
    ):
        drift["run_id"] = observed["run_id"]
    if observed["prepared_parent"] != expected:
        drift["prepared_parent"] = observed["prepared_parent"]
    if observed["payload"] != expected:
        drift["payload"] = observed["payload"]
    if drift:
        raise RuntimeError(
            "PROBE_GRAPH_VARIANT_BINDING_DRIFT: "
            f"expected={expected}, observed={drift}"
        )
    return expected


def _enforce_peak_reserved_limit(peak_reserved_bytes: int) -> dict[str, int | bool]:
    """Reserve about 2 GiB of a 24 GiB card and fail at the boundary."""

    peak = int(peak_reserved_bytes)
    limit = _PEAK_RESERVED_LIMIT_BYTES
    headroom = limit - peak
    if peak <= 0:
        raise RuntimeError(f"NONPOSITIVE_PEAK_RESERVED_BYTES={peak}")
    if not peak < limit:
        raise RuntimeError(
            "REAL_BACKWARD_PROBE_PEAK_RESERVED_LIMIT_EXCEEDED: "
            f"peak={peak}, limit={limit}, headroom={headroom}"
        )
    return {
        "peak_reserved_within_limit": True,
        "peak_reserved_limit_bytes": limit,
        "peak_reserved_headroom_bytes": headroom,
    }


def _resolve_gpu_identity(torch) -> dict[str, int | str]:
    """Fail closed unless the paid endpoint is the authorized 24 GiB 4090."""

    device_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device_index)
    name = str(properties.name)
    total_memory = int(properties.total_memory)
    if _REQUIRED_CUDA_DEVICE_TOKEN not in name:
        raise RuntimeError(
            "REAL_BACKWARD_PROBE_GPU_IDENTITY_MISMATCH: "
            f"required={_REQUIRED_CUDA_DEVICE_TOKEN!r}, observed={name!r}"
        )
    if total_memory <= _PEAK_RESERVED_LIMIT_BYTES:
        raise RuntimeError(
            "REAL_BACKWARD_PROBE_GPU_MEMORY_CAPACITY_TOO_SMALL: "
            f"total={total_memory}, reserved_limit={_PEAK_RESERVED_LIMIT_BYTES}"
        )
    return {
        "cuda_device_index": device_index,
        "cuda_device_name": name,
        "cuda_total_memory_bytes": total_memory,
    }


def _select_exact_probe_group(
    training,
    *,
    train_batches,
    pass_pairs,
    profile: ProbeProfile,
):
    """Select exactly one full-capacity optimizer group, or fail closed."""

    work_items = training.materialize_pass_work_items(
        pass_pairs,
        train_batches,
        microbatch_size=profile.microbatch,
    )
    group = work_items[: profile.accumulation]
    row_counts = [training._batch_row_count(batch) for batch, _ in group]
    if len(group) != profile.accumulation:
        raise RuntimeError(
            "Probe input cannot fill the requested optimizer group: "
            f"fragments={len(group)}, required={profile.accumulation}"
        )
    if any(rows != profile.microbatch for rows in row_counts):
        raise RuntimeError(
            "Probe refuses a partial optimizer group: "
            f"rows={row_counts}, required_each={profile.microbatch}"
        )
    observed_rows = sum(row_counts)
    if observed_rows != profile.group_row_capacity:
        raise RuntimeError(
            "Probe optimizer-group row capacity drift: "
            f"observed={observed_rows}, required={profile.group_row_capacity}"
        )
    return group, row_counts


def _build_formal_model(training, payload, config, *, device: str):
    """Mirror the formal trainer's model admission without redefining training."""

    from ..gnn import build_model
    from .model import build_v32_cc_hhgt_residual, build_v32_hierarchical_evidence_hhgt

    bundle = payload["bundle"]
    encoder = build_model(
        "cc_hhgt",
        bundle,
        int(payload["feature_dim"]),
        dict(payload["legacy_model_config"]),
    )
    primary = config.get("primary_model", {})
    integration = primary.get("evidence_integration", {})
    integration_mode = str(integration.get("mode", "external_router")).lower()
    if integration_mode == "hierarchical_end_to_end":
        modality_names = tuple(
            integration.get("modalities", ("mutation", "cnv", "atac"))
        )
        training.validate_hierarchical_modality_contract(payload, modality_names)
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
        raise RuntimeError(
            f"Unsupported evidence integration mode: {integration_mode}"
        )
    return model, architecture_id


def run_authorized_probe(context: Any) -> int:
    """Run one real formal GPU backward/step group after CLI authorization."""

    from .training_guard import TrainingAuthorizationContext

    if type(context) is not TrainingAuthorizationContext:
        raise RuntimeError(
            "REAL_BACKWARD_PROBE_REQUIRES_GUARDED_RUN_SHARD_CONTEXT"
        )
    if context.authorized_trainer != _AUTHORIZED_TRAINER:
        raise RuntimeError(
            "REAL_BACKWARD_PROBE_AUTHORIZED_TRAINER_MISMATCH"
        )
    context = _reverify_authorization_context(context)

    # The CLI imports this module only after its training guard succeeds.  Keep
    # Torch and all production-training imports on this side of that boundary.
    import torch

    from ..gnn import move_graph
    from . import training

    ctx = training._context_mapping(context)
    root = Path(ctx["repo_root"])
    config = training._load_mapping(Path(ctx["config_path"]))
    task = training._task_row(Path(ctx["task_manifest_path"]), ctx["task_id"])
    _require_authorized_pending_task(task)
    fold = int(task["patient_fold"])
    seed = int(task["seed"])
    profile = resolve_probe_profile()

    prepared_path = training._resolve_prepared_path(config, fold, root)
    payload, prepared_sha256 = (
        training.load_prepared_artifact_from_authorized_handle(
            torch,
            ctx["input_manifest_path"],
            fold=fold,
            prepared_path=prepared_path,
        )
    )
    if not isinstance(payload, Mapping):
        raise RuntimeError("Prepared fold artifact must contain a mapping")
    training._validate_prepared(
        payload, fold=fold, artifact_hashes=ctx["artifact_hashes"]
    )
    graph_variant = _validate_graph_variant_binding(
        run_id=str(ctx["run_id"]),
        config=config,
        payload=payload,
        prepared_path=prepared_path,
    )
    input_authority_hashes = training._prepared_input_authority_hashes(payload)

    if not torch.cuda.is_available():
        raise RuntimeError("REAL_BACKWARD_PROBE_REQUIRES_CUDA")
    device = "cuda"
    gpu_identity = _resolve_gpu_identity(torch)
    runtime = config.get("runtime_profile", {})
    precision = training.resolve_mixed_precision_contract(
        torch, runtime, device=device
    )
    if not (
        precision.get("requested") == "bf16"
        and precision.get("active") == "bf16"
        and precision.get("autocast_enabled") is True
    ):
        raise RuntimeError(f"REAL_BACKWARD_PROBE_REQUIRES_ACTIVE_BF16={precision}")
    runtime_plan = training.resolve_runtime_training_plan(runtime)
    if profile.name == _ALLOWED_PROFILES[_STRESS_GATE] and (
        profile.microbatch != runtime_plan.candidate_microbatch_size
        or profile.accumulation != runtime_plan.gradient_accumulation
    ):
        raise RuntimeError(
            "Stress probe must equal the formal primary optimizer-group contract: "
            f"probe={profile.microbatch}x{profile.accumulation}, "
            f"formal={runtime_plan.candidate_microbatch_size}x"
            f"{runtime_plan.gradient_accumulation}"
        )
    if profile.name == _ALLOWED_PROFILES[_ENVIRONMENT_GATE] and (
        profile.microbatch > runtime_plan.candidate_microbatch_size
    ):
        raise RuntimeError("Environment probe exceeds the formal primary microbatch")

    training._seed_everything(torch, seed)
    model, architecture_id = _build_formal_model(
        training, payload, config, device=device
    )
    bundle = payload["bundle"]
    runtime_chunks, _ = training._runtime_chunk_contract(bundle)
    runtime_permutation = training._frozen_chunk_permutation(
        runtime_chunks, seed=seed, fold=fold
    )
    pass_offset_permutation = training._frozen_pass_offset_permutation(
        runtime_chunks, seed=seed, fold=fold
    )
    candidate_chunk_schedule_mode = training._resolve_candidate_chunk_schedule_mode(
        runtime
    )
    gradient_algorithm = training._gradient_algorithm_for_schedule_mode(
        candidate_chunk_schedule_mode
    )
    probe_cycle_schedules = training._candidate_chunk_cycle_schedules(
        batch_count=len(payload["train_batches"]),
        chunks=runtime_chunks,
        permutation=runtime_permutation,
        pass_offset_permutation=pass_offset_permutation,
        max_cycles=1,
        mode=candidate_chunk_schedule_mode,
        gradient_accumulation=runtime_plan.gradient_accumulation,
    )
    group, row_counts = _select_exact_probe_group(
        training,
        train_batches=payload["train_batches"],
        pass_pairs=probe_cycle_schedules[0][0],
        profile=profile,
    )

    has_runtime_schedule = getattr(bundle, "runtime_schedule", None) is not None
    fixed_graph = payload.get("graph")
    if has_runtime_schedule and fixed_graph is not None:
        raise RuntimeError(
            "A fixed prepared graph cannot override the registered runtime schedule"
        )
    if not has_runtime_schedule:
        fixed_graph = (
            training._move(fixed_graph, device)
            if fixed_graph is not None
            else move_graph(bundle, device, "cc_hhgt")
        )

    def graph_for_chunk(chunk: int):
        return (
            training._runtime_graph_for_chunk(bundle, chunk, device=device)
            if has_runtime_schedule
            else fixed_graph
        )

    primary = config.get("primary_model", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(primary.get("learning_rate", 1e-3)),
        weight_decay=float(primary.get("weight_decay", 1e-5)),
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if (
        gradient_algorithm
        == training.SHARED_ENCODER_GRADIENT_ALGORITHM
    ):
        loss_plan, call_telemetry = (
            training.shared_encoder_conventional_group_backward(
                model,
                group,
                graph_for_chunk,
                torch=torch,
                device=device,
                precision=precision,
                direction_loss_weight=float(
                    primary.get("direction_loss_weight", 0.25)
                ),
                shrinkage=float(primary.get("residual_shrinkage", 1e-4)),
            )
        )
    elif gradient_algorithm == training.TWO_PASS_STREAMING_GRADIENT_ALGORITHM:
        loss_plan = training.streaming_group_backward(
            model,
            group,
            graph_for_chunk,
            torch=torch,
            device=device,
            precision=precision,
            direction_loss_weight=float(primary.get("direction_loss_weight", 0.25)),
            shrinkage=float(primary.get("residual_shrinkage", 1e-4)),
        )
        call_telemetry = training.BackwardCallTelemetry(
            unique_chunks=len({int(chunk) for _, chunk in group}),
            group_rows=sum(row_counts),
            encoder_forward_calls=2 * len(group),
            decoder_forward_calls=2 * len(group),
            global_loss_calls=2,
            backward_calls=len(group),
        )
    else:  # pragma: no cover - the resolver is already fail-closed
        raise RuntimeError(f"UNKNOWN_FORMAL_GRADIENT_ALGORITHM={gradient_algorithm}")
    observed_call_contract = (
        call_telemetry.unique_chunks,
        call_telemetry.encoder_forward_calls,
        call_telemetry.decoder_forward_calls,
        call_telemetry.global_loss_calls,
        call_telemetry.backward_calls,
    )
    expected_call_contract = (
        (1, 1, len(group), 1, 1)
        if gradient_algorithm == training.SHARED_ENCODER_GRADIENT_ALGORITHM
        else (
            len({int(chunk) for _, chunk in group}),
            2 * len(group),
            2 * len(group),
            2,
            len(group),
        )
    )
    if observed_call_contract != expected_call_contract:
        raise RuntimeError(
            "REAL_BACKWARD_PROBE_CALL_CONTRACT_DRIFT="
            f"{observed_call_contract}!={expected_call_contract}"
        )
    update = training.optimizer_step_with_guards(
        model,
        optimizer,
        torch=torch,
        objective_value=loss_plan.objective_value,
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
    memory_gate = _enforce_peak_reserved_limit(peak_reserved_bytes)

    loss_finite = math.isfinite(float(loss_plan.objective_value))
    result = {
        "status": "PASS_REAL_BACKWARD_PROBE",
        "run_id": str(ctx["run_id"]),
        "task_id": str(ctx["task_id"]),
        "patient_fold": fold,
        "graph_variant": graph_variant,
        "architecture_id": architecture_id,
        "prepared_artifact_sha256": prepared_sha256,
        "authorization_artifact_hashes": dict(ctx["artifact_hashes"]),
        "input_authority_hash_count": len(input_authority_hashes),
        "optimizer_steps": 1,
        "loss": float(loss_plan.objective_value),
        "loss_finite": loss_finite,
        "grad_finite": bool(update["grad_finite"]),
        "parameters_finite": bool(update["parameters_finite"]),
        "parameter_delta_positive": bool(update["parameter_delta_positive"]),
        "gradient_tensors": int(update["gradient_tensors"]),
        "trainable_parameter_tensors": int(
            update["trainable_parameter_tensors"]
        ),
        "update_probe_parameter": str(update["update_probe_parameter"]),
        "update_probe_max_abs_delta": float(
            update["update_probe_max_abs_delta"]
        ),
        "peak_reserved_bytes": peak_reserved_bytes,
        **memory_gate,
        **gpu_identity,
        "cuda_physical_headroom_bytes": int(
            gpu_identity["cuda_total_memory_bytes"]
        )
        - peak_reserved_bytes,
        "seconds": seconds,
        "microbatch": profile.microbatch,
        "accumulation": profile.accumulation,
        "group_rows": sum(row_counts),
        "group_row_capacity": profile.group_row_capacity,
        "group_capacity_preserved": sum(row_counts)
        == profile.group_row_capacity,
        "profile": profile.name,
        "precision": str(precision["active"]),
        "precision_contract": precision,
        "candidate_chunk_schedule_mode": candidate_chunk_schedule_mode,
        "candidate_chunk_assignment_unit": (
            "optimizer_group"
            if candidate_chunk_schedule_mode
            == training.CANDIDATE_CHUNK_SCHEDULE_BALANCED_GROUP_SHARED_ENCODER
            else "candidate_batch"
        ),
        "gradient_algorithm": gradient_algorithm,
        "unique_chunks": call_telemetry.unique_chunks,
        "encoder_forward_calls": call_telemetry.encoder_forward_calls,
        "decoder_forward_calls": call_telemetry.decoder_forward_calls,
        "global_loss_calls": call_telemetry.global_loss_calls,
        "backward_calls": call_telemetry.backward_calls,
        "call_contract_verified": True,
        "formal_artifacts_written": 0,
    }
    if not (
        result["loss_finite"]
        and result["grad_finite"]
        and result["parameter_delta_positive"]
        and result["call_contract_verified"]
        and result["group_capacity_preserved"]
        and result["peak_reserved_within_limit"]
    ):
        raise RuntimeError(f"REAL_BACKWARD_PROBE_POSTCONDITION_FAILED={result}")
    print(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


__all__ = ["ProbeProfile", "resolve_probe_profile", "run_authorized_probe"]
