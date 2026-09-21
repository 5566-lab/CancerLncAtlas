"""Training authorization guard that must run before any Torch/CUDA import."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .orchestration import ExecutionPolicy, extract_execution_policy, read_task_manifest


APPROVAL_FORMAT = "CC_HHGT_V3_2_TRAINING_APPROVAL_V1"
HASH_FIELDS = (
    "code_sha256",
    "config_sha256",
    "input_manifest_sha256",
    "task_manifest_sha256",
)
_CODE_EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git"}
_CODE_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


class TrainingAuthorizationError(RuntimeError):
    """Fail-closed denial raised before a training framework is imported."""


@dataclass(frozen=True)
class TrainingAuthorizationContext:
    run_id: str
    task_id: str
    endpoint_id: str
    hardware_class: str
    repo_root: str
    config_path: str
    input_manifest_path: str
    task_manifest_path: str
    approval_path: str
    artifact_hashes: dict[str, str]
    policy: dict[str, Any]
    # The CLI-selected callable is part of the execution authority.  Keeping
    # it in the frozen context lets every public trainer re-run the same guard
    # without allowing a caller to swap ``--trainer`` after approval.
    authorized_trainer: str | None = None


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise TrainingAuthorizationError(f"Required artifact is missing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _code_files(repo_root: Path) -> list[tuple[str, Path]]:
    # The authorized trainer imports production helpers outside cc_hhgt.v32
    # (notably gnn.py, relation_sampling.py and training_resume.py).  Hash the
    # complete Python package so an approval cannot silently authorize one
    # dependency closure and execute another.  The small CLI wrapper remains
    # explicitly bound as well.
    roots = [repo_root / "cc_hhgt", repo_root / "scripts" / "v32_pipeline.py"]
    found: list[tuple[str, Path]] = []
    for root in roots:
        if root.is_file():
            found.append((root.relative_to(repo_root).as_posix(), root))
            continue
        if not root.is_dir():
            raise TrainingAuthorizationError(f"V3.2 code path is missing: {root}")
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(repo_root)
            if any(part in _CODE_EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.suffix.lower() in _CODE_EXCLUDED_SUFFIXES:
                continue
            found.append((relative.as_posix(), path))
    if not found:
        raise TrainingAuthorizationError("No V3.2 source files were found for hashing")
    return sorted(found)


def code_tree_sha256(repo_root: str | Path) -> str:
    """Hash V3.2 production code with relative paths and file boundaries."""

    root = Path(repo_root).resolve()
    digest = hashlib.sha256()
    for relative, path in _code_files(root):
        content_hash = file_sha256(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def compute_artifact_hashes(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    input_manifest_path: str | Path,
    task_manifest_path: str | Path,
) -> dict[str, str]:
    return {
        "code_sha256": code_tree_sha256(repo_root),
        "config_sha256": file_sha256(config_path),
        "input_manifest_sha256": file_sha256(input_manifest_path),
        "task_manifest_sha256": file_sha256(task_manifest_path),
    }


def load_structured_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise TrainingAuthorizationError(f"Structured file is missing: {source}")
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml  # Safe parser dependency; deliberately not a training framework.
        except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
            raise TrainingAuthorizationError(
                "PyYAML is required to read a non-JSON V3.2 config"
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, Mapping):
        raise TrainingAuthorizationError(f"Expected a mapping in {source}")
    return dict(payload)


def _find_task_row(task_manifest_path: str | Path, task_id: str) -> dict[str, str]:
    rows = read_task_manifest(task_manifest_path)
    matches = [row for row in rows if row.get("task_id") == task_id]
    if len(matches) != 1:
        raise TrainingAuthorizationError(
            f"Approval target must identify one manifest task; {task_id!r} matched {len(matches)}"
        )
    return matches[0]


def _require_training_policy(config: Mapping[str, Any], *, paid_task: bool) -> ExecutionPolicy:
    policy = extract_execution_policy(config)
    if policy.execution_mode != "TRAINING":
        raise TrainingAuthorizationError(
            f"Config execution_mode={policy.execution_mode}; expected TRAINING"
        )
    if policy.training_authorized is not True:
        raise TrainingAuthorizationError("Config training_authorized is not true")
    if paid_task and not (
        policy.paid_enabled
        and policy.max_paid_hours > 0
        and policy.max_cost_cny > 0
    ):
        raise TrainingAuthorizationError(
            "Paid task requires paid_enabled=true and positive time/cost caps"
        )
    return policy


def _load_approval(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingAuthorizationError(
            f"Training approval is absent (expected {path}); CODE_ONLY remains enforced"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainingAuthorizationError(f"Training approval is invalid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise TrainingAuthorizationError("Training approval must be a JSON object")
    return dict(payload)


def _validate_approval(
    approval: Mapping[str, Any],
    *,
    run_id: str,
    task_id: str,
    endpoint_id: str,
    hardware_class: str,
    artifact_hashes: Mapping[str, str],
    paid_task: bool,
    policy: ExecutionPolicy,
    trainer_specification: str | None = None,
) -> None:
    if approval.get("approval_format") != APPROVAL_FORMAT:
        raise TrainingAuthorizationError("Approval format mismatch")
    if approval.get("training_authorized") is not True:
        raise TrainingAuthorizationError("Approval training_authorized is not true")
    expected_scalars = {
        "run_id": run_id,
        "endpoint_id": endpoint_id,
        "hardware_class": hardware_class,
    }
    mismatched = {
        key: {"expected": value, "observed": approval.get(key)}
        for key, value in expected_scalars.items()
        if approval.get(key) != value
    }
    if mismatched:
        raise TrainingAuthorizationError(f"Approval execution scope mismatch: {mismatched}")
    approved_ids = approval.get("approved_task_ids")
    if not isinstance(approved_ids, Sequence) or isinstance(approved_ids, (str, bytes)):
        raise TrainingAuthorizationError("approved_task_ids must be an explicit JSON array")
    if task_id not in approved_ids:
        raise TrainingAuthorizationError(f"Task is not explicitly approved: {task_id}")
    if trainer_specification is not None:
        if not isinstance(trainer_specification, str) or not trainer_specification:
            raise TrainingAuthorizationError("Trainer specification must be non-empty")
        if approval.get("authorized_trainer") != trainer_specification:
            raise TrainingAuthorizationError(
                "Approval trainer mismatch: "
                f"expected={trainer_specification!r}, "
                f"observed={approval.get('authorized_trainer')!r}"
            )
    observed_hashes = approval.get("artifact_hashes")
    if not isinstance(observed_hashes, Mapping):
        raise TrainingAuthorizationError("Approval artifact_hashes is missing")
    hash_drift = {
        key: {"expected": artifact_hashes[key], "observed": observed_hashes.get(key)}
        for key in HASH_FIELDS
        if observed_hashes.get(key) != artifact_hashes[key]
    }
    if hash_drift:
        raise TrainingAuthorizationError(f"Approval artifact hash mismatch: {hash_drift}")
    if bool(approval.get("paid_enabled", False)) != bool(policy.paid_enabled):
        raise TrainingAuthorizationError("Approval/config paid_enabled mismatch")
    if paid_task:
        if float(approval.get("max_paid_hours", 0)) != policy.max_paid_hours:
            raise TrainingAuthorizationError("Approval/config max_paid_hours mismatch")
        if float(approval.get("max_cost_cny", 0)) != policy.max_cost_cny:
            raise TrainingAuthorizationError("Approval/config max_cost_cny mismatch")


def guard_training_entry(
    *,
    allow_training: bool,
    repo_root: str | Path,
    config_path: str | Path,
    input_manifest_path: str | Path,
    task_manifest_path: str | Path,
    approval_path: str | Path,
    run_id: str,
    task_id: str,
    endpoint_id: str,
    hardware_class: str,
    trainer_specification: str | None = None,
) -> TrainingAuthorizationContext:
    """Validate all training authority before a caller may import Torch.

    Callers must invoke this function directly from their CLI branch.  It has
    no side effects: no file is written, no task is claimed, and no hardware or
    remote endpoint is queried.
    """

    if allow_training is not True:
        raise TrainingAuthorizationError("Missing explicit --allow-training flag")
    config = load_structured_mapping(config_path)
    row = _find_task_row(task_manifest_path, task_id)
    if row.get("run_id") != run_id:
        raise TrainingAuthorizationError("Task manifest run_id mismatch")
    if row.get("endpoint_id") not in (None, "", endpoint_id) and row.get("owner") != endpoint_id:
        raise TrainingAuthorizationError("Task manifest endpoint mismatch")
    if row.get("owner") != endpoint_id:
        raise TrainingAuthorizationError(
            f"Task is owned by {row.get('owner')!r}, not endpoint {endpoint_id!r}"
        )
    if row.get("hardware_class") != hardware_class:
        raise TrainingAuthorizationError("Task manifest hardware class mismatch")
    paid_task = str(row.get("paid_task", "")).strip().lower() == "true"
    policy = _require_training_policy(config, paid_task=paid_task)
    hashes = compute_artifact_hashes(
        repo_root=repo_root,
        config_path=config_path,
        input_manifest_path=input_manifest_path,
        task_manifest_path=task_manifest_path,
    )
    approval = _load_approval(Path(approval_path))
    _validate_approval(
        approval,
        run_id=run_id,
        task_id=task_id,
        endpoint_id=endpoint_id,
        hardware_class=hardware_class,
        artifact_hashes=hashes,
        paid_task=paid_task,
        policy=policy,
        trainer_specification=trainer_specification,
    )
    return TrainingAuthorizationContext(
        run_id=run_id,
        task_id=task_id,
        endpoint_id=endpoint_id,
        hardware_class=hardware_class,
        repo_root=str(Path(repo_root).resolve()),
        config_path=str(Path(config_path).resolve()),
        input_manifest_path=str(Path(input_manifest_path).resolve()),
        task_manifest_path=str(Path(task_manifest_path).resolve()),
        approval_path=str(Path(approval_path).resolve()),
        artifact_hashes=dict(hashes),
        policy=asdict(policy),
        authorized_trainer=(
            str(approval["authorized_trainer"])
            if trainer_specification is not None
            else None
        ),
    )
