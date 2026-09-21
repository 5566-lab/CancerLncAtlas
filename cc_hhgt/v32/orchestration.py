"""Fail-closed, zero-cost orchestration primitives for V3.2.

This module is deliberately free of ``torch`` and remote-execution imports.
It describes work; it never starts work.  The current code-review phase owns
five local RTX 4070 fold tasks and blocks every one of them until the separate
training authorization guard succeeds.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ANALYSIS_VERSION = "3.2.0"
DEFAULT_RUN_ID = "V3_2_ONESEED_CODE_ONLY"
DEFAULT_SEED = 20260726
N_PATIENT_FOLDS = 5
TASK_MANIFEST_COLUMNS = (
    "task_id",
    "run_id",
    "task_type",
    "model",
    "patient_fold",
    "seed",
    "owner",
    "hardware_class",
    "status",
    "blocked_reason",
    "paid_task",
)
VIRTUAL_CHECKPOINT_FORMAT = "CC_HHGT_V3_2_VIRTUAL_CHECKPOINT_V1"


class OrchestrationContractError(ValueError):
    """Raised when a task or endpoint violates the frozen V3.2 contract."""


@dataclass(frozen=True)
class ExecutionPolicy:
    """Execution controls that default to a safe, non-training state."""

    execution_mode: str = "CODE_ONLY"
    training_authorized: bool = False
    paid_enabled: bool = False
    max_paid_hours: float = 0.0
    max_cost_cny: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ExecutionPolicy":
        raw = dict(value or {})
        policy = cls(
            execution_mode=str(raw.get("execution_mode", "CODE_ONLY")).upper(),
            training_authorized=_strict_bool(
                raw.get("training_authorized", False), "training_authorized"
            ),
            paid_enabled=_strict_bool(raw.get("paid_enabled", False), "paid_enabled"),
            max_paid_hours=float(raw.get("max_paid_hours", 0.0)),
            max_cost_cny=float(raw.get("max_cost_cny", 0.0)),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.execution_mode not in {"CODE_ONLY", "TRAINING"}:
            raise OrchestrationContractError(
                "execution_mode must be CODE_ONLY or TRAINING"
            )
        if self.max_paid_hours < 0 or self.max_cost_cny < 0:
            raise OrchestrationContractError("Paid limits cannot be negative")
        if not self.paid_enabled and (
            self.max_paid_hours != 0 or self.max_cost_cny != 0
        ):
            raise OrchestrationContractError(
                "paid_enabled=false requires max_paid_hours=max_cost_cny=0"
            )
        if self.paid_enabled and (
            self.max_paid_hours <= 0 or self.max_cost_cny <= 0
        ):
            raise OrchestrationContractError(
                "Paid execution requires positive hour and CNY caps"
            )

    @property
    def blocks_training(self) -> bool:
        return self.execution_mode != "TRAINING" or not self.training_authorized


def _strict_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise OrchestrationContractError(f"{field} must be a JSON/YAML boolean")


def extract_execution_policy(config: Mapping[str, Any] | None) -> ExecutionPolicy:
    """Read policy from a supported config section without permissive coercion.

    New V3.2 configs use ``execution_control``.  Root-level keys are accepted
    for the compact configuration shown in the frozen implementation plan.
    """

    raw = dict(config or {})
    for section in ("execution_control", "execution", "orchestration"):
        nested = raw.get(section)
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise OrchestrationContractError(f"{section} must be a mapping")
            raw = dict(nested)
            break
    return ExecutionPolicy.from_mapping(raw)


def endpoint_registry(policy: ExecutionPolicy | None = None) -> list[dict[str, Any]]:
    """Return endpoint responsibilities without contacting an endpoint."""

    resolved = policy or ExecutionPolicy()
    return [
        {
            "endpoint_id": "server149",
            "hardware_class": "SERVER149_CPU",
            "roles": ["input_audit", "fold_assets", "linear_baselines", "final_audit"],
            "may_train_cc_hhgt": False,
            "remote_contacted": False,
            "paid": False,
        },
        {
            "endpoint_id": "local4070",
            "hardware_class": "LOCAL_RTX_4070_TI_SUPER_16GB",
            "roles": ["cc_hhgt_patient_fold_training"],
            "may_train_cc_hhgt": not resolved.blocks_training,
            "remote_contacted": False,
            "paid": False,
        },
        {
            "endpoint_id": "local_gpu",
            "hardware_class": "COMPUTE_HOST_CUDA_GPU",
            "roles": ["cc_hhgt_patient_fold_training"],
            "may_train_cc_hhgt": not resolved.blocks_training,
            "remote_contacted": False,
            "paid": False,
            "runtime_gpu_preflight_required": True,
        },
        {
            "endpoint_id": "paid_gpu",
            "hardware_class": "PAID_PREEMPTIBLE_GPU",
            "roles": ["cc_hhgt_patient_fold_training"],
            "may_train_cc_hhgt": bool(
                resolved.paid_enabled and not resolved.blocks_training
            ),
            "remote_contacted": False,
            "paid": True,
            "enabled": resolved.paid_enabled,
            "max_paid_hours": resolved.max_paid_hours,
            "max_cost_cny": resolved.max_cost_cny,
        },
    ]


def _blocked_reason(policy: ExecutionPolicy) -> str:
    reasons: list[str] = []
    if policy.execution_mode != "TRAINING":
        reasons.append(f"execution_mode={policy.execution_mode}")
    if not policy.training_authorized:
        reasons.append("training_authorized=false")
    return ";".join(reasons) if reasons else ""


def build_task_manifest(
    *,
    run_id: str = DEFAULT_RUN_ID,
    seed: int = DEFAULT_SEED,
    policy: ExecutionPolicy | None = None,
    owner: str = "local4070",
    hardware_class: str = "LOCAL_RTX_4070_TI_SUPER_16GB",
    paid_task: bool | None = None,
) -> list[dict[str, Any]]:
    """Build exactly five one-seed patient-fold GPU tasks."""

    resolved = policy or ExecutionPolicy()
    resolved.validate()
    if not str(run_id).strip():
        raise OrchestrationContractError("run_id cannot be empty")
    if int(seed) < 0:
        raise OrchestrationContractError("seed must be non-negative")
    reason = _blocked_reason(resolved)
    status = "BLOCKED" if reason else "PENDING"
    inferred_paid = owner == "paid_gpu"
    resolved_paid = inferred_paid if paid_task is None else bool(paid_task)
    if resolved_paid != inferred_paid:
        raise OrchestrationContractError(
            "paid_task must agree with the paid_gpu endpoint identity"
        )
    if resolved_paid and not resolved.paid_enabled:
        raise OrchestrationContractError(
            "paid_gpu tasks require an explicitly capped paid execution policy"
        )
    tasks = []
    for patient_fold in range(N_PATIENT_FOLDS):
        task_id = f"{run_id}|PATIENT_FOLD_{patient_fold}|CC-HHGT|{int(seed)}"
        tasks.append(
            {
                "task_id": task_id,
                "run_id": str(run_id),
                "task_type": "CC_HHGT_PATIENT_FOLD",
                "model": "CC-HHGT",
                "patient_fold": patient_fold,
                "seed": int(seed),
                "owner": str(owner),
                "hardware_class": str(hardware_class),
                "status": status,
                "blocked_reason": reason,
                "paid_task": resolved_paid,
            }
        )
    validate_task_manifest(tasks, policy=resolved)
    return tasks


def validate_task_manifest(
    tasks: Sequence[Mapping[str, Any]], *, policy: ExecutionPolicy
) -> None:
    if len(tasks) != N_PATIENT_FOLDS:
        raise OrchestrationContractError(
            f"Expected {N_PATIENT_FOLDS} CC-HHGT tasks, observed {len(tasks)}"
        )
    ids = [str(task.get("task_id", "")) for task in tasks]
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise OrchestrationContractError("Task IDs must be unique and non-empty")
    folds = sorted(int(task.get("patient_fold", -1)) for task in tasks)
    if folds != list(range(N_PATIENT_FOLDS)):
        raise OrchestrationContractError(f"Patient folds must be 0..4, got {folds}")
    for task in tasks:
        endpoint = (str(task.get("owner")), str(task.get("hardware_class")))
        allowed_endpoints = {
            ("local4070", "LOCAL_RTX_4070_TI_SUPER_16GB"),
            ("local_gpu", "COMPUTE_HOST_CUDA_GPU"),
            ("paid_gpu", "PAID_PREEMPTIBLE_GPU"),
        }
        if endpoint not in allowed_endpoints:
            raise OrchestrationContractError(f"Unsupported V3.2 CC-HHGT endpoint: {endpoint}")
        if task.get("model") != "CC-HHGT":
            raise OrchestrationContractError("Only CC-HHGT belongs in the GPU task manifest")
        expected_paid = task.get("owner") == "paid_gpu"
        if task.get("paid_task") is not expected_paid:
            raise OrchestrationContractError("Task paid flag differs from its endpoint")
        if expected_paid and not policy.paid_enabled:
            raise OrchestrationContractError("Paid task is disabled by execution policy")
        if policy.blocks_training and task.get("status") != "BLOCKED":
            raise OrchestrationContractError("CODE_ONLY tasks must be BLOCKED")


def task_manifest_tsv(tasks: Sequence[Mapping[str, Any]]) -> str:
    """Serialize the task manifest deterministically for hashing and review."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(TASK_MANIFEST_COLUMNS),
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for task in tasks:
        row = {key: task[key] for key in TASK_MANIFEST_COLUMNS}
        row["paid_task"] = "false" if row["paid_task"] is False else "true"
        writer.writerow(row)
    return output.getvalue()


def task_manifest_sha256(tasks: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(task_manifest_tsv(tasks).encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def write_task_manifest(path: str | Path, tasks: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    _atomic_write_text(destination, task_manifest_tsv(tasks))
    return destination


def build_virtual_checkpoint(
    task: Mapping[str, Any],
    *,
    artifact_hashes: Mapping[str, str],
    hardware_class: str | None = None,
    last_completed_cycle: int = 0,
) -> dict[str, Any]:
    """Create metadata-only checkpoint state for resume contract tests.

    It contains no parameters, optimizer state, tensors, or RNG state and is
    never accepted by the real training loader.
    """

    if int(last_completed_cycle) < 0:
        raise OrchestrationContractError("last_completed_cycle cannot be negative")
    required_hashes = {
        "code_sha256",
        "config_sha256",
        "input_manifest_sha256",
        "task_manifest_sha256",
    }
    missing = sorted(required_hashes - set(artifact_hashes))
    if missing:
        raise OrchestrationContractError(f"Virtual checkpoint hashes missing: {missing}")
    return {
        "checkpoint_format": VIRTUAL_CHECKPOINT_FORMAT,
        "virtual_only": True,
        "contains_model_state": False,
        "resume_authorized": False,
        "task_id": str(task["task_id"]),
        "hardware_class": str(hardware_class or task["hardware_class"]),
        "last_completed_cycle": int(last_completed_cycle),
        "optimizer_steps": 0,
        "artifact_hashes": {key: str(artifact_hashes[key]) for key in sorted(required_hashes)},
    }


def validate_virtual_resume(
    checkpoint: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    artifact_hashes: Mapping[str, str],
    hardware_class: str,
) -> None:
    """Validate metadata resume semantics without loading a real checkpoint."""

    if checkpoint.get("checkpoint_format") != VIRTUAL_CHECKPOINT_FORMAT:
        raise OrchestrationContractError("Not a V3.2 virtual checkpoint")
    if checkpoint.get("virtual_only") is not True:
        raise OrchestrationContractError("Virtual checkpoint marker is missing")
    if checkpoint.get("contains_model_state") is not False:
        raise OrchestrationContractError("Virtual checkpoint cannot contain model state")
    if checkpoint.get("task_id") != task.get("task_id"):
        raise OrchestrationContractError("Virtual checkpoint task mismatch")
    if checkpoint.get("hardware_class") != hardware_class:
        raise OrchestrationContractError(
            "Cross-hardware resume is forbidden; restart the task from cycle 0"
        )
    expected = {key: str(value) for key, value in sorted(artifact_hashes.items())}
    if checkpoint.get("artifact_hashes") != expected:
        raise OrchestrationContractError("Virtual checkpoint artifact hash mismatch")


def build_dry_run_report(
    *,
    run_id: str = DEFAULT_RUN_ID,
    seed: int = DEFAULT_SEED,
    policy: ExecutionPolicy | None = None,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Return the complete plan without importing CUDA or starting processes."""

    resolved = policy or ExecutionPolicy()
    tasks = build_task_manifest(run_id=run_id, seed=seed, policy=resolved)
    path_audit = {
        key: {"path": str(Path(value)), "exists": Path(value).exists()}
        for key, value in sorted((artifact_paths or {}).items())
    }
    return {
        "report_format": "CC_HHGT_V3_2_DRY_RUN_V1",
        "analysis_version": ANALYSIS_VERSION,
        "run_id": run_id,
        "execution_policy": asdict(resolved),
        "endpoints": endpoint_registry(resolved),
        "tasks": tasks,
        "task_summary": {
            "total": len(tasks),
            "blocked": sum(task["status"] == "BLOCKED" for task in tasks),
            "local4070": sum(task["owner"] == "local4070" for task in tasks),
            "server149_training": 0,
            "paid": sum(bool(task["paid_task"]) for task in tasks),
        },
        "task_manifest_content_sha256": task_manifest_sha256(tasks),
        "artifact_path_audit": path_audit,
        "safety_attestation": {
            "training_started": False,
            "optimizer_step_executed": False,
            "cuda_initialized": False,
            "remote_endpoint_contacted": False,
            "paid_instance_started": False,
            "estimated_paid_cost_cny": 0.0,
            "real_checkpoint_created": False,
        },
    }


def write_dry_run_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = Path(path)
    _atomic_write_text(
        destination,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return destination


def read_task_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a task manifest without executing or claiming any task."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    missing = set(TASK_MANIFEST_COLUMNS) - set(rows[0] if rows else [])
    if missing:
        raise OrchestrationContractError(f"Task manifest columns missing: {sorted(missing)}")
    return rows


def find_task(tasks: Iterable[Mapping[str, Any]], task_id: str) -> Mapping[str, Any]:
    matches = [task for task in tasks if str(task.get("task_id")) == str(task_id)]
    if len(matches) != 1:
        raise OrchestrationContractError(
            f"Expected exactly one task_id={task_id!r}, observed {len(matches)}"
        )
    return matches[0]
