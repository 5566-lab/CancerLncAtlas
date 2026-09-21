#!/usr/bin/env python3
"""Independent CPU-only trust gate for the r2 group-shared oracle.

This verifier is deployed under ``runtime/oracle_gates`` outside the staged
code overlay.  It never imports or executes staged Python.  All staged files,
receipts and manifests are treated as untrusted data and opened with
``O_NOFOLLOW`` through a single descriptor whose identity is checked before
and after reading.  The 55 GB PT files are not re-hashed here; their 15
hash/size rows must match the independently anchored r1 r6 static authority,
and every current file is opened only to verify its same-fd regular-file size.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


NAMESPACE = "v32_group_shared_oracle_paid_gpu_20260901_r2"
FORMAL_R2_NAMESPACE = "v32_g012_paid_gpu_20260901_r2"
R1_NAMESPACE = "v32_g012_paid_gpu_20260831_r1"
R1_ARCHIVE_NAMESPACE = "archive_aborted_runtime_infeasible_20260901_r1"
RUN_ID = "v32-g012-g2-group-shared-oracle-paid-gpu-20260901-r2"
TASK_ID = f"{RUN_ID}|PATIENT_FOLD_0|CC-HHGT|20260726"
TRAINER = (
    "cc_hhgt.v32.group_shared_encoder_oracle:"
    "run_authorized_oracle_comparison"
)
EXPECTED_VARIANT = "G2"
EXPECTED_FOLD = 0
EXPECTED_SEED = 20260726
ENDPOINT_ID = "paid_gpu"
HARDWARE_CLASS = "PAID_PREEMPTIBLE_GPU"
MAX_HOURS = 3.0
MAX_COST_CNY = 8.0
PROJECT_BUDGET_CAP_CNY = 210.0
COMPUTE_HOURLY_CNY = 2.05
# Backward-compatible field name; it now means the compute/GPU component,
# never the all-in total.
GPU_HOURLY_CNY = COMPUTE_HOURLY_CNY
DISK_HOURLY_CNY = 0.04
TOTAL_HOURLY_CNY = 2.09
MAX_JIT_VALIDITY = timedelta(minutes=45)
# The local dispatcher/supervisor reserve the first 30 minutes before any
# paid provider operation.  Once the instance has started, this independent
# remote gate requires 15 minutes immediately before CUDA discovery.
LOCAL_PAID_API_MINIMUM_TTL_SECONDS = 1800
REMOTE_PAID_LAUNCH_MINIMUM_TTL_SECONDS = 900
MIN_PAID_LAUNCH_TTL = timedelta(
    seconds=REMOTE_PAID_LAUNCH_MINIMUM_TTL_SECONDS
)
JIT_CLOCK_SKEW = timedelta(minutes=1)
EXPECTED_PROFILE_NAME_SHA256 = (
    "37a8eec1ce19687d132fe29051dca629d164e2c4958ba141d5f4133a33f0688f"
)
EXPECTED_PROJECT_SCOPE_SHA256 = (
    "726631660639f60035b7774f8be7e090618bf083bc400a1580f4050f02a861fd"
)

STATIC_FORMAT = "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_R2_STATIC_AUTH_V1"
STATIC_STATUS = "ORACLE_STATIC_AUTH_READY_CPU_ONLY"
APPROVAL_FORMAT = "CC_HHGT_V3_2_TRAINING_APPROVAL_V1"
INPUT_FORMAT = "CC_HHGT_V3_2_G012_R2_INPUT_REUSE_READY_V1"
INPUT_STATUS = "INPUT_REUSE_READY_HASH_VERIFIED"
SOURCE_INPUT_ARCHIVE_SHA256 = (
    "1c17b7be89621e5125c39e87f05ce14beb1f2227afdb481d2e9a20049b56bab0"
)
ABORT_FORMAT = "CC_HHGT_V3_2_G012_ABORTED_RUN_V1"
ABORT_STATUS = "ABORTED_RUNTIME_INFEASIBLE"
CODE_ARCHIVE_FORMAT = "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_CODE_ARCHIVE_R2_V1"
CODE_ARCHIVE_STATUS = "ORACLE_CODE_ARCHIVE_READY_CPU_ONLY"
JIT_FORMAT = "CANCERLNCATLAS_COMPSHARE_JIT_BUDGET_V1"
JIT_STATUS = "JIT_BUDGET_READY_CONSERVATIVE"
DECISION_FORMAT = "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_DECISION_V1"
DECISION_STATUS = "ORACLE_COMPARISON_PAID_GATE_AUTHORIZED"
TERMINAL_RECEIPT_FORMAT = "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_COMPARISON_V1"
PASS_STATUS = (
    "PASS_REAL_DATA_ORACLE_COMPARISON_ONLY_NOT_AUTHORIZED_FOR_FORMAL_TRAINING"
)
FAIL_STATUSES = frozenset(
    {
        "SCIENTIFIC_FAIL_REAL_DATA_ORACLE_COMPARISON_ONLY",
        "RUNTIME_FAIL_REAL_DATA_ORACLE_COMPARISON_ONLY",
    }
)

# Immutable root-of-trust anchor copied from the independently returned r1 r6
# receipt.  The archived server copy must be byte-identical.
EXPECTED_R1_STATIC_AUTH_R6_SHA256 = (
    "1f5bc54b2a8acc037373bfa380c95c8a717e76fde956852ec481b16a0d8716f0"
)
EXPECTED_G2_F0_SHA256 = (
    "4ebba1d4060f3efaa8c080824386802ca732bedf07681ebd0283deb087b0ca39"
)
EXPECTED_G2_F0_SIZE_BYTES = 4_482_708_589

CONFIG_RELATIVE = "config/model_v3_2_group_shared_oracle_paid_gpu_20260901_r2.yaml"
TASK_TEMPLATE_RELATIVE = (
    "config/v32_group_shared_oracle_paid_gpu_20260901_r2.TASK_MANIFEST.tsv"
)
RUNNER_RELATIVE = "cc_hhgt/v32/group_shared_encoder_oracle.py"
PREREG_RELATIVE = (
    "docs/v32_group_shared_encoder_estimator_preregistration_20260901.md"
)
DECISION_RELATIVE = (
    "docs/v32_group_shared_encoder_oracle_decision_20260901_r2.json"
)
VALIDATOR_NAME = "validate_v32_group_shared_oracle_static_auth_ready_r2.py"
LAUNCHER_NAME = "server_launch_v32_group_shared_oracle_paid_gpu_20260901_r2.sh"
AUTHORIZER_NAME = "cloud_authorize_v32_group_shared_oracle_no_gpu_20260901_r2.sh"
SUPERVISOR_NAME = (
    "local_supervise_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1"
)
DISPATCH_CONTROLLER_NAME = (
    "local_dispatch_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1"
)
LOG_GUARD_NAME = "v32_group_shared_oracle_log_guard_r2.psm1"
EXTERNAL_DEPLOYMENT_MANIFEST_NAME = "EXTERNAL_DEPLOYMENT_MANIFEST.json"
EXTERNAL_DEPLOYMENT_FORMAT = (
    "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_EXTERNAL_DEPLOYMENT_R2_V1"
)
EXTERNAL_DEPLOYMENT_STATUS = "EXTERNAL_CONTROL_ARTIFACTS_FROZEN"
EXTERNAL_CONTROL_NAMES = (
    VALIDATOR_NAME,
    LAUNCHER_NAME,
    AUTHORIZER_NAME,
    SUPERVISOR_NAME,
    DISPATCH_CONTROLLER_NAME,
    LOG_GUARD_NAME,
)
TASK_COLUMNS = (
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


class OracleGateError(RuntimeError):
    """The isolated comparison gate is incomplete, stale, or inconsistent."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OracleGateError(f"ORACLE_{label}_MAPPING_REQUIRED")
    return dict(value)


def _identity(observed: os.stat_result) -> tuple[int, ...]:
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_mode),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
    )


def _open_regular(path: Path, label: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OracleGateError(f"ORACLE_{label}_OPEN_FAILED={path}") from exc
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode) or observed.st_size <= 0:
        os.close(descriptor)
        raise OracleGateError(f"ORACLE_{label}_NOT_NONEMPTY_REGULAR={path}")
    # A second directory entry would let an untrusted overlay mutate the same
    # inode through a path that is outside the checked namespace.  Stable-fd
    # checks catch concurrent writes, but they cannot prove path ownership;
    # therefore every authority input must be a single-link regular file.
    if int(observed.st_nlink) != 1:
        os.close(descriptor)
        raise OracleGateError(
            f"ORACLE_{label}_HARDLINK_FORBIDDEN_NLINK={int(observed.st_nlink)}:{path}"
        )
    return descriptor


def _read_regular(path: Path, label: str) -> tuple[bytes, str, int]:
    """Read/hash one file through one no-follow fd and verify stable identity."""

    descriptor = _open_regular(path, label)
    try:
        before = _identity(os.fstat(descriptor))
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = _identity(os.fstat(descriptor))
        if before != after:
            raise OracleGateError(f"ORACLE_{label}_CHANGED_DURING_READ={path}")
        return b"".join(chunks), digest.hexdigest(), int(before[3])
    finally:
        os.close(descriptor)


def _stat_regular_size(path: Path, label: str) -> int:
    """Size-check a 55 GB fold without reading or hashing its payload."""

    descriptor = _open_regular(path, label)
    try:
        before = _identity(os.fstat(descriptor))
        after = _identity(os.fstat(descriptor))
        if before != after:
            raise OracleGateError(f"ORACLE_{label}_CHANGED_DURING_STAT={path}")
        return int(before[3])
    finally:
        os.close(descriptor)


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str, int]:
    raw, digest, size = _read_regular(path, label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleGateError(f"ORACLE_{label}_JSON_INVALID={path}") from exc
    return _mapping(value, label), digest, size


def _load_yaml(path: Path, label: str) -> tuple[dict[str, Any], str]:
    raw, digest, _ = _read_regular(path, label)
    try:
        value = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise OracleGateError(f"ORACLE_{label}_YAML_INVALID={path}") from exc
    return _mapping(value, label), digest


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise OracleGateError(f"ORACLE_{label}_TIMESTAMP_REQUIRED")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OracleGateError(f"ORACLE_{label}_TIMESTAMP_INVALID") from exc
    if parsed.tzinfo is None:
        raise OracleGateError(f"ORACLE_{label}_TIMEZONE_REQUIRED")
    return parsed.astimezone(timezone.utc)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OracleGateError(f"ORACLE_{label}_NUMBER_REQUIRED")
    result = float(value)
    if not math.isfinite(result):
        raise OracleGateError(f"ORACLE_{label}_FINITE_REQUIRED")
    return result


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed, _, _ = _read_regular(path, "EXISTING_TARGET")
        if observed != payload:
            raise OracleGateError(f"ORACLE_EXISTING_TARGET_DRIFT={path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise OracleGateError(f"ORACLE_PARTIAL_TARGET_PRESENT={temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _within(path: Path, root: Path, label: str) -> Path:
    # Do not call Path.resolve() on the candidate: doing so would follow a
    # symlink before O_NOFOLLOW sees it.  Check the lexical absolute path and
    # every extant component from the trusted boundary instead.
    boundary = Path(os.path.abspath(root))
    candidate = Path(os.path.abspath(path))
    try:
        relative = candidate.relative_to(boundary)
    except ValueError as exc:
        raise OracleGateError(f"ORACLE_{label}_ESCAPES_ROOT={candidate}") from exc
    current = boundary
    if current.is_symlink():
        raise OracleGateError(f"ORACLE_{label}_SYMLINK_BOUNDARY={current}")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise OracleGateError(f"ORACLE_{label}_SYMLINK_FORBIDDEN={current}")
    return candidate


def _code_tree_sha256(code_root: Path) -> str:
    """Mirror training_guard without importing any staged module."""

    root = code_root.resolve()
    sources = (root / "cc_hhgt", root / "scripts/v32_pipeline.py")
    found: list[tuple[str, Path]] = []
    for source in sources:
        _within(source, root, "CODE_TREE")
        if source.is_file() and not source.is_symlink():
            found.append((source.relative_to(root).as_posix(), source))
            continue
        if not source.is_dir() or source.is_symlink():
            raise OracleGateError(f"ORACLE_CODE_TREE_SOURCE_INVALID={source}")
        for candidate in source.rglob("*"):
            relative = candidate.relative_to(root)
            if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in relative.parts):
                continue
            if candidate.suffix.lower() in {".pyc", ".pyo"}:
                continue
            if candidate.is_symlink():
                raise OracleGateError(f"ORACLE_CODE_TREE_SYMLINK_FORBIDDEN={candidate}")
            if candidate.is_file():
                found.append((relative.as_posix(), candidate))
    if not found:
        raise OracleGateError("ORACLE_CODE_TREE_EMPTY")
    digest = hashlib.sha256()
    for relative, candidate in sorted(found):
        _, content_sha, _ = _read_regular(candidate, "CODE_TREE_FILE")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_sha.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _deployment_records(code_root: Path) -> tuple[list[dict[str, Any]], str]:
    relatives = (
        "cc_hhgt/v32/cli.py",
        "cc_hhgt/v32/training_guard.py",
        "cc_hhgt/v32/gpu_backward_probe.py",
        "cc_hhgt/v32/training.py",
        RUNNER_RELATIVE,
        PREREG_RELATIVE,
        DECISION_RELATIVE,
        CONFIG_RELATIVE,
        TASK_TEMPLATE_RELATIVE,
        "scripts/v32_pipeline.py",
    )
    records = []
    for relative in relatives:
        path = _within(code_root / relative, code_root, "DEPLOYMENT_ARTIFACT")
        _, digest, size = _read_regular(path, "DEPLOYMENT_ARTIFACT")
        records.append({"relative_path": relative, "sha256": digest, "size_bytes": size})
    contract_sha = hashlib.sha256(
        json.dumps(records, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return records, contract_sha


def _runner_constants(runner_path: Path) -> dict[str, Any]:
    raw, _, _ = _read_regular(runner_path, "ORACLE_RUNNER")
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(runner_path))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise OracleGateError("ORACLE_RUNNER_AST_INVALID") from exc
    wanted = {
        "SCALAR_ABSOLUTE_FLOOR",
        "SCALAR_RELATIVE_TOLERANCE",
        "GRADIENT_RELATIVE_L2_MAX",
        "GRADIENT_COSINE_MIN",
        "GRADIENT_NORM_RATIO_MIN",
        "GRADIENT_NORM_RATIO_MAX",
        "MODULE_GRADIENT_COSINE_MIN",
        "NONZERO_RETURN_CODE",
        "EXPECTED_VARIANT",
        "EXPECTED_FOLD",
        "EXPECTED_SEED",
    }
    observed: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                try:
                    observed[name] = ast.literal_eval(node.value)
                except (ValueError, TypeError) as exc:
                    raise OracleGateError(f"ORACLE_RUNNER_CONSTANT_NOT_LITERAL={name}") from exc
    if set(observed) != wanted:
        raise OracleGateError(f"ORACLE_RUNNER_CONSTANTS_MISSING={sorted(wanted - set(observed))}")
    return observed


def validate_config(
    config_path: Path,
    *,
    code_root: Path | None = None,
    prepared_parent: Path | None = None,
    externally_observed_verifier_sha256: str | None = None,
) -> dict[str, Any]:
    config, config_sha = _load_yaml(config_path, "CONFIG")
    if config.get("contract_version") != "3.2.0-group-shared-oracle-paid-gpu-20260901-r2":
        raise OracleGateError("ORACLE_CONFIG_VERSION_DRIFT")
    control = _mapping(config.get("execution_control"), "EXECUTION_CONTROL")
    required_control = {
        "execution_mode": "TRAINING",
        "training_authorized": True,
        "paid_enabled": True,
        "comparison_only": True,
        "formal_training_authorized": False,
        "overwrite_formal_v32": False,
    }
    for key, expected in required_control.items():
        if control.get(key) != expected:
            raise OracleGateError(f"ORACLE_EXECUTION_CONTROL_DRIFT={key}")
    if _number(control.get("max_paid_hours"), "MAX_PAID_HOURS") != MAX_HOURS:
        raise OracleGateError("ORACLE_MAX_PAID_HOURS_DRIFT")
    if _number(control.get("max_cost_cny"), "MAX_COST_CNY") != MAX_COST_CNY:
        raise OracleGateError("ORACLE_MAX_COST_CNY_DRIFT")

    oracle = _mapping(config.get("oracle_contract"), "CONTRACT")
    required_oracle = {
        "namespace": NAMESPACE,
        "run_id": RUN_ID,
        "task_id": TASK_ID,
        "trainer": TRAINER,
        "graph_variant": EXPECTED_VARIANT,
        "patient_fold": EXPECTED_FOLD,
        "seed": EXPECTED_SEED,
        "artifact_class": "TIMING_AND_ORACLE_COMPARISON_ONLY",
        "terminal_pass_status": PASS_STATUS,
        "scientific_nonzero_exit_code": 42,
        "receipt_channel": "STDOUT_JSON_LINES_ONLY",
        "formal_artifacts_allowed": False,
        "checkpoint_allowed": False,
        "prediction_allowed": False,
        "winner_selection_allowed": False,
        "equal_step_g0_g1_g2_pilot_still_required": True,
    }
    for key, expected in required_oracle.items():
        if oracle.get(key) != expected:
            raise OracleGateError(f"ORACLE_CONTRACT_DRIFT={key}")
    expected_verifier_relative = (
        f"runtime/oracle_gates/{NAMESPACE}/{VALIDATOR_NAME}"
    )
    if oracle.get("external_verifier_path") != expected_verifier_relative:
        raise OracleGateError("ORACLE_EXTERNAL_VERIFIER_PATH_DRIFT")
    verifier_sha = oracle.get("external_verifier_sha256")
    if not _is_sha256(verifier_sha):
        raise OracleGateError("ORACLE_EXTERNAL_VERIFIER_SHA_BINDING_INVALID")
    if externally_observed_verifier_sha256 is not None:
        if not _is_sha256(externally_observed_verifier_sha256):
            raise OracleGateError("ORACLE_EXTERNAL_VERIFIER_OBSERVED_SHA_INVALID")
        if externally_observed_verifier_sha256 != verifier_sha:
            raise OracleGateError("ORACLE_EXTERNAL_VERIFIER_SHA_DRIFT")
    expected_deployment_relative = (
        f"runtime/oracle_gates/{NAMESPACE}/{EXTERNAL_DEPLOYMENT_MANIFEST_NAME}"
    )
    if oracle.get("external_deployment_manifest_path") != expected_deployment_relative:
        raise OracleGateError("ORACLE_EXTERNAL_DEPLOYMENT_MANIFEST_PATH_DRIFT")
    if not _is_sha256(oracle.get("external_deployment_manifest_sha256")):
        raise OracleGateError("ORACLE_EXTERNAL_DEPLOYMENT_MANIFEST_SHA_INVALID")
    path_bindings = {
        "runner": (oracle.get("runner_path"), RUNNER_RELATIVE, oracle.get("runner_sha256")),
        "preregistration": (
            oracle.get("preregistration_path"),
            PREREG_RELATIVE,
            oracle.get("preregistration_sha256"),
        ),
        "decision": (
            oracle.get("decision_path"),
            DECISION_RELATIVE,
            oracle.get("decision_sha256"),
        ),
    }
    for label, (observed_path, expected_path, expected_sha) in path_bindings.items():
        if observed_path != expected_path or not _is_sha256(expected_sha):
            raise OracleGateError(f"ORACLE_{label.upper()}_CONFIG_BINDING_INVALID")

    task_contract = _mapping(config.get("task_contract"), "TASK_CONTRACT")
    if task_contract.get("graph_variant") != EXPECTED_VARIANT:
        raise OracleGateError("ORACLE_GRAPH_VARIANT_DRIFT")
    crossfit = _mapping(config.get("crossfit"), "CROSSFIT")
    if crossfit.get("seed") != EXPECTED_SEED:
        raise OracleGateError("ORACLE_CROSSFIT_SEED_DRIFT")
    runtime = _mapping(config.get("runtime_profile"), "RUNTIME_PROFILE")
    runtime_required = {
        "owner": ENDPOINT_ID,
        "mixed_precision": "bf16",
        "candidate_microbatch_size": 8192,
        "gradient_accumulation": 4,
        "candidate_chunk_schedule_mode": "balanced_group_latin_shared_encoder_v1",
        "supervisor_static_auth_line_seconds": 180,
        "supervisor_initial_heartbeat_minutes": 15,
        "supervisor_assignment_stall_minutes": 5,
        "supervisor_pre_submit_status_grace_seconds": 720,
        "supervisor_post_submit_status_grace_seconds": 180,
    }
    for key, expected in runtime_required.items():
        if runtime.get(key) != expected:
            raise OracleGateError(f"ORACLE_RUNTIME_PROFILE_DRIFT={key}")
    training_io = _mapping(config.get("training_io"), "TRAINING_IO")
    if "output_root" in training_io or training_io.get("artifact_sink") != "STDOUT_JSON_LINES_ONLY":
        raise OracleGateError("ORACLE_FORMAL_OUTPUT_SINK_FORBIDDEN")
    if any(str(key).lower() == "output_root" for key in _walk_keys(config)):
        raise OracleGateError("ORACLE_OUTPUT_ROOT_KEY_FORBIDDEN")
    encoded = json.dumps(config, sort_keys=True)
    if "/results/" in encoded or "v32_g012_paid_gpu_training_" in encoded:
        raise OracleGateError("ORACLE_FORMAL_RESULT_REFERENCE_FORBIDDEN")
    if prepared_parent is not None:
        expected_pattern = str(prepared_parent.resolve() / "G2" / "PATIENT_FOLD_{fold}.pt")
        if training_io.get("prepared_fold_pattern") != expected_pattern:
            raise OracleGateError("ORACLE_PREPARED_PATTERN_DRIFT")

    if code_root is not None:
        for label, (relative, _, expected_sha) in {
            "RUNNER": (RUNNER_RELATIVE, None, oracle["runner_sha256"]),
            "PREREG": (PREREG_RELATIVE, None, oracle["preregistration_sha256"]),
            "DECISION": (DECISION_RELATIVE, None, oracle["decision_sha256"]),
        }.items():
            _, observed_sha, _ = _read_regular(code_root / relative, label)
            if observed_sha != expected_sha:
                raise OracleGateError(f"ORACLE_{label}_SHA_DRIFT")
        _validate_decision_and_thresholds(code_root, oracle)
    return {"config": config, "config_sha256": config_sha, "oracle": oracle}


def _validate_external_deployment_manifest(
    path: Path,
    *,
    control_root: Path,
    expected_manifest_sha256: str,
    externally_observed_verifier_sha256: str | None,
) -> dict[str, Any]:
    payload, digest, _ = _load_json(path, "EXTERNAL_DEPLOYMENT_MANIFEST")
    if digest != expected_manifest_sha256:
        raise OracleGateError("ORACLE_EXTERNAL_DEPLOYMENT_MANIFEST_SHA_DRIFT")
    required = {
        "format": EXTERNAL_DEPLOYMENT_FORMAT,
        "status": EXTERNAL_DEPLOYMENT_STATUS,
        "namespace": NAMESPACE,
        "formal_training_authorized": False,
        "comparison_only": True,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise OracleGateError(f"ORACLE_EXTERNAL_DEPLOYMENT_DRIFT={key}")
    records = payload.get("artifacts")
    if not isinstance(records, list) or len(records) != len(EXTERNAL_CONTROL_NAMES):
        raise OracleGateError("ORACLE_EXTERNAL_DEPLOYMENT_ARTIFACT_CARDINALITY_DRIFT")
    by_name: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        record = _mapping(raw_record, "EXTERNAL_DEPLOYMENT_ARTIFACT")
        name = record.get("name")
        if name not in EXTERNAL_CONTROL_NAMES or name in by_name:
            raise OracleGateError("ORACLE_EXTERNAL_DEPLOYMENT_ARTIFACT_NAME_DRIFT")
        if record.get("server_relative_path") != name:
            raise OracleGateError(
                f"ORACLE_EXTERNAL_DEPLOYMENT_SERVER_PATH_DRIFT={name}"
            )
        if not _is_sha256(record.get("sha256")):
            raise OracleGateError(f"ORACLE_EXTERNAL_DEPLOYMENT_SHA_INVALID={name}")
        if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] <= 0:
            raise OracleGateError(f"ORACLE_EXTERNAL_DEPLOYMENT_SIZE_INVALID={name}")
        candidate = _within(control_root / name, control_root, "EXTERNAL_CONTROL")
        _, observed_sha, observed_size = _read_regular(candidate, "EXTERNAL_CONTROL")
        if observed_sha != record["sha256"] or observed_size != record["size_bytes"]:
            raise OracleGateError(f"ORACLE_EXTERNAL_CONTROL_ARTIFACT_DRIFT={name}")
        by_name[name] = record
    if set(by_name) != set(EXTERNAL_CONTROL_NAMES):
        raise OracleGateError("ORACLE_EXTERNAL_DEPLOYMENT_ARTIFACT_SET_DRIFT")
    verifier_record_sha = by_name[VALIDATOR_NAME]["sha256"]
    if verifier_record_sha != externally_observed_verifier_sha256:
        raise OracleGateError("ORACLE_EXTERNAL_VERIFIER_MANIFEST_OBSERVATION_DRIFT")
    return {
        "sha256": digest,
        "artifacts": [by_name[name] for name in EXTERNAL_CONTROL_NAMES],
    }


def _walk_keys(value: object):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield key
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def _validate_decision_and_thresholds(code_root: Path, oracle: Mapping[str, Any]) -> str:
    decision, decision_sha, _ = _load_json(code_root / DECISION_RELATIVE, "DECISION")
    if decision_sha != oracle.get("decision_sha256"):
        raise OracleGateError("ORACLE_DECISION_SHA_DRIFT")
    expected = {
        "format": DECISION_FORMAT,
        "decision": DECISION_STATUS,
        "authorized_callable": TRAINER,
        "artifact_class": "TIMING_AND_ORACLE_COMPARISON_ONLY",
        "formal_training_authorized": False,
        "terminal_pass_does_not_authorize_formal_training": True,
        "equal_step_g0_g1_g2_pilot_still_required": True,
        "hour_cap": MAX_HOURS,
        "cost_cap_cny": MAX_COST_CNY,
        "runner_sha256": oracle.get("runner_sha256"),
        "preregistration_sha256": oracle.get("preregistration_sha256"),
    }
    for key, required in expected.items():
        if decision.get(key) != required:
            raise OracleGateError(f"ORACLE_DECISION_DRIFT={key}")
    scope = _mapping(decision.get("authorized_scope"), "DECISION_SCOPE")
    if scope != {"graph_variant": EXPECTED_VARIANT, "patient_fold": 0, "seed": EXPECTED_SEED}:
        raise OracleGateError("ORACLE_DECISION_SCOPE_DRIFT")
    thresholds = _mapping(decision.get("scientific_thresholds"), "THRESHOLDS")
    expected_thresholds = {
        "branch_constant_across_all_58_required": True,
        "paired_branch_disagreement_max": 0,
        "scalar_absolute_floor": 1e-7,
        "scalar_relative_tolerance": 1e-6,
        "complete_gradient_relative_l2_max": 1e-3,
        "complete_gradient_cosine_min": 0.9999,
        "complete_gradient_norm_ratio_min": 0.999,
        "complete_gradient_norm_ratio_max": 1.001,
        "module_gradient_cosine_min": 0.999,
    }
    if thresholds != expected_thresholds:
        raise OracleGateError("ORACLE_DECISION_THRESHOLD_DRIFT")
    constants = _runner_constants(code_root / RUNNER_RELATIVE)
    expected_constants = {
        "SCALAR_ABSOLUTE_FLOOR": 1e-7,
        "SCALAR_RELATIVE_TOLERANCE": 1e-6,
        "GRADIENT_RELATIVE_L2_MAX": 1e-3,
        "GRADIENT_COSINE_MIN": 0.9999,
        "GRADIENT_NORM_RATIO_MIN": 0.999,
        "GRADIENT_NORM_RATIO_MAX": 1.001,
        "MODULE_GRADIENT_COSINE_MIN": 0.999,
        "NONZERO_RETURN_CODE": 42,
        "EXPECTED_VARIANT": EXPECTED_VARIANT,
        "EXPECTED_FOLD": EXPECTED_FOLD,
        "EXPECTED_SEED": EXPECTED_SEED,
    }
    if constants != expected_constants:
        raise OracleGateError(f"ORACLE_RUNNER_THRESHOLD_OR_SCOPE_DRIFT={constants}")
    prereg_raw, prereg_sha, _ = _read_regular(code_root / PREREG_RELATIVE, "PREREG")
    if prereg_sha != oracle.get("preregistration_sha256"):
        raise OracleGateError("ORACLE_PREREG_SHA_DRIFT")
    text = prereg_raw.decode("utf-8")
    required_tokens = (
        "constant across all 58 evaluations",
        "max(1e-7, 1e-6 * abs(reference_mean))",
        "cosine similarity is at least `0.9999`",
        "within `[0.999, 1.001]`",
        "similarity is at least `0.999`",
        "formal_artifacts_written=0",
    )
    missing = [token for token in required_tokens if token not in text]
    if missing:
        raise OracleGateError(f"ORACLE_PREREG_THRESHOLD_TEXT_DRIFT={missing}")
    return decision_sha


def _task_rows(raw: bytes) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OracleGateError("ORACLE_TASK_TEMPLATE_UTF8_INVALID") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t", strict=True)
    if tuple(reader.fieldnames or ()) != TASK_COLUMNS:
        raise OracleGateError("ORACLE_TASK_HEADER_DRIFT")
    rows = list(reader)
    if len(rows) != 1 or None in rows[0]:
        raise OracleGateError("ORACLE_TASK_CARDINALITY_DRIFT")
    required = {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "task_type": "CC_HHGT_PATIENT_FOLD",
        "model": "CC-HHGT",
        "patient_fold": "0",
        "seed": str(EXPECTED_SEED),
        "owner": ENDPOINT_ID,
        "hardware_class": HARDWARE_CLASS,
        "status": "PENDING",
        "blocked_reason": "",
        "paid_task": "true",
    }
    if rows[0] != required:
        raise OracleGateError(f"ORACLE_TASK_SCOPE_DRIFT={rows[0]}")
    return rows


def _fold_records(payload: Mapping[str, Any], label: str) -> dict[tuple[str, int], dict[str, Any]]:
    variants = payload.get("variants")
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if isinstance(variants, Mapping):
        for variant, section in variants.items():
            section_map = _mapping(section, f"{label}_{variant}")
            for raw in section_map.get("fold_inputs", []):
                row = _mapping(raw, f"{label}_FOLD")
                records[(str(variant), int(row.get("fold", -1)))] = row
    else:
        for raw in payload.get("fold_artifacts", []):
            row = _mapping(raw, f"{label}_FOLD")
            records[(str(row.get("variant")), int(row.get("fold", -1)))] = row
    expected = {(variant, fold) for variant in ("G0", "G1", "G2") for fold in range(5)}
    if set(records) != expected or len(records) != 15:
        raise OracleGateError(f"ORACLE_{label}_FOLD_SET_DRIFT")
    return records


def _validate_r1_and_inputs(
    *,
    project_root: Path,
    prepared_parent: Path,
    source_input_receipt: Path,
    returned_r1_static_receipt: Path,
) -> dict[str, Any]:
    r1_bootstrap = project_root / "runtime/bootstrap" / R1_NAMESPACE
    aborted_path = r1_bootstrap / "ABORTED.json"
    aborted, aborted_sha, _ = _load_json(aborted_path, "R1_ABORTED")
    expected_abort = {
        "format": ABORT_FORMAT,
        "status": ABORT_STATUS,
        "resume_authorized": False,
        "live_authorizations_removed": True,
        "gpu_training_complete_present": False,
        "formal_result_root_examined": False,
        "formal_result_root_modified": False,
        "formal_result_artifacts_reused": False,
        "deletion_performed": False,
    }
    for key, expected in expected_abort.items():
        if aborted.get(key) != expected:
            raise OracleGateError(f"ORACLE_R1_ABORTED_DRIFT={key}")
    archived = _mapping(aborted.get("archived_authorizations"), "R1_ARCHIVED")
    static_record = _mapping(archived.get("static_auth_ready"), "R1_STATIC_RECORD")
    if static_record.get("sha256") != EXPECTED_R1_STATIC_AUTH_R6_SHA256:
        raise OracleGateError("ORACLE_R1_ABORT_STATIC_SHA_ANCHOR_DRIFT")
    expected_static_path = r1_bootstrap / R1_ARCHIVE_NAMESPACE / "STATIC_AUTH_READY.live_at_abort.json"
    static_path = Path(str(static_record.get("archive_path", "")))
    if static_path.resolve(strict=False) != expected_static_path.resolve(strict=False):
        raise OracleGateError("ORACLE_R1_ABORT_STATIC_PATH_DRIFT")
    returned_static, returned_static_sha, returned_static_size = _load_json(
        returned_r1_static_receipt, "RETURNED_R1_STATIC_R6"
    )
    if returned_static_sha != EXPECTED_R1_STATIC_AUTH_R6_SHA256:
        raise OracleGateError("ORACLE_RETURNED_R1_STATIC_R6_SHA_DRIFT")
    r1_static, r1_static_sha, r1_static_size = _load_json(static_path, "R1_STATIC_R6")
    if r1_static_sha != EXPECTED_R1_STATIC_AUTH_R6_SHA256:
        raise OracleGateError("ORACLE_R1_STATIC_R6_SHA_DRIFT")
    if static_record.get("size_bytes") != r1_static_size:
        raise OracleGateError("ORACLE_R1_STATIC_R6_SIZE_DRIFT")
    if returned_static_size != r1_static_size or returned_static != r1_static:
        raise OracleGateError("ORACLE_RETURNED_AND_ARCHIVED_R1_STATIC_R6_DRIFT")
    if r1_static.get("status") != "STATIC_AUTH_READY" or r1_static.get("gpu_visible") is not False:
        raise OracleGateError("ORACLE_R1_STATIC_R6_STATUS_DRIFT")
    r1_records = _fold_records(r1_static, "R1_STATIC")

    reuse, reuse_sha, _ = _load_json(source_input_receipt, "INPUT_REUSE")
    required_reuse = {
        "format": INPUT_FORMAT,
        "status": INPUT_STATUS,
        "source_input_archive_sha256": SOURCE_INPUT_ARCHIVE_SHA256,
        "fold_artifact_count": 15,
        "reused_existing_prepared_inputs": True,
        "retransfer_performed": False,
        "copy_performed": False,
        "extraction_performed": False,
        "source_files_modified": False,
        "formal_result_artifacts_used": False,
    }
    for key, expected in required_reuse.items():
        if reuse.get(key) != expected:
            raise OracleGateError(f"ORACLE_INPUT_REUSE_DRIFT={key}")
    if reuse.get("prepared_parent") != str(prepared_parent.resolve()):
        raise OracleGateError("ORACLE_INPUT_REUSE_PARENT_DRIFT")
    reuse_records = _fold_records(reuse, "INPUT_REUSE")
    total = 0
    for key in sorted(r1_records):
        anchored = r1_records[key]
        candidate = reuse_records[key]
        variant, fold = key
        expected_path = prepared_parent / variant / f"PATIENT_FOLD_{fold}.pt"
        if anchored.get("path") != str(expected_path.resolve()) or candidate.get("path") != str(expected_path.resolve()):
            raise OracleGateError(f"ORACLE_INPUT_PATH_DRIFT={variant}:{fold}")
        if candidate.get("sha256") != anchored.get("sha256"):
            raise OracleGateError(f"ORACLE_INPUT_SHA_CROSSCHECK_DRIFT={variant}:{fold}")
        if candidate.get("size_bytes") != anchored.get("size_bytes"):
            raise OracleGateError(f"ORACLE_INPUT_SIZE_CROSSCHECK_DRIFT={variant}:{fold}")
        current_size = _stat_regular_size(expected_path, f"PT_{variant}_{fold}")
        if current_size != int(anchored.get("size_bytes", -1)):
            raise OracleGateError(f"ORACLE_INPUT_CURRENT_SIZE_DRIFT={variant}:{fold}")
        total += current_size
    if reuse.get("fold_artifact_total_bytes") != total:
        raise OracleGateError("ORACLE_INPUT_REUSE_TOTAL_SIZE_DRIFT")
    g2f0 = r1_records[(EXPECTED_VARIANT, EXPECTED_FOLD)]
    if g2f0.get("sha256") != EXPECTED_G2_F0_SHA256 or g2f0.get("size_bytes") != EXPECTED_G2_F0_SIZE_BYTES:
        raise OracleGateError("ORACLE_G2_F0_IMMUTABLE_ANCHOR_DRIFT")
    return {
        "r1_aborted_sha256": aborted_sha,
        "r1_static_auth_r6_sha256": r1_static_sha,
        "returned_r1_static_auth_r6_sha256": returned_static_sha,
        "source_input_receipt_sha256": reuse_sha,
        "fold_records": reuse_records,
        "g2f0": g2f0,
    }


def _validate_jit_budget(path: Path, *, now: datetime | None = None) -> dict[str, Any]:
    payload, digest, _ = _load_json(path, "JIT_BUDGET")
    expected = {
        "format": JIT_FORMAT,
        "status": JIT_STATUS,
        "currency": "CNY",
        "source": "COMPSHARE_API_BILLING_SNAPSHOT",
        "project_budget_cap_cny": PROJECT_BUDGET_CAP_CNY,
        "compute_hourly_cny": COMPUTE_HOURLY_CNY,
        "gpu_hourly_cny": GPU_HOURLY_CNY,
        "disk_hourly_cny": DISK_HOURLY_CNY,
        "total_hourly_cny": TOTAL_HOURLY_CNY,
        "profile_name_sha256": EXPECTED_PROFILE_NAME_SHA256,
        "project_scope_sha256": EXPECTED_PROJECT_SCOPE_SHA256,
    }
    for key, required in expected.items():
        if payload.get(key) != required:
            raise OracleGateError(f"ORACLE_JIT_BUDGET_DRIFT={key}")
    instance_id = payload.get("instance_id")
    if not isinstance(instance_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", instance_id) is None:
        raise OracleGateError("ORACLE_JIT_INSTANCE_ID_INVALID")
    queried = _parse_utc(payload.get("queried_at"), "JIT_QUERIED_AT")
    valid_until = _parse_utc(payload.get("valid_until"), "JIT_VALID_UNTIL")
    clock = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if queried > clock + JIT_CLOCK_SKEW or valid_until <= clock:
        raise OracleGateError("ORACLE_JIT_BUDGET_STALE")
    if valid_until <= queried or valid_until - queried > MAX_JIT_VALIDITY:
        raise OracleGateError("ORACLE_JIT_VALIDITY_WINDOW_INVALID")
    if valid_until - clock < MIN_PAID_LAUNCH_TTL:
        raise OracleGateError("ORACLE_JIT_MINIMUM_PAID_LAUNCH_TTL_NOT_AVAILABLE")
    remaining = _number(payload.get("conservative_remaining_cny"), "JIT_REMAINING")
    if not 0 < remaining <= PROJECT_BUDGET_CAP_CNY:
        raise OracleGateError("ORACLE_JIT_REMAINING_OUT_OF_RANGE")
    effective = min(MAX_COST_CNY, remaining)
    return {
        "payload": payload,
        "sha256": digest,
        "instance_id": instance_id,
        "remaining_cny": remaining,
        "effective_cost_cap_cny": effective,
        "valid_until": valid_until.isoformat(),
        "launch_ttl_seconds": (valid_until - clock).total_seconds(),
    }


def validate_paid_launch_readiness(
    *,
    jit_budget_path: Path,
    static_auth_path: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Repeat the dynamic launch gate immediately before GPU discovery."""

    jit = _validate_jit_budget(jit_budget_path, now=now)
    static, static_sha, _ = _load_json(static_auth_path, "STATIC_AUTH_LAUNCH")
    expected = {
        "format": STATIC_FORMAT,
        "status": STATIC_STATUS,
        "namespace": NAMESPACE,
        "comparison_only": True,
        "formal_training_authorized": False,
        "instance_id": jit["instance_id"],
        "jit_budget_receipt_sha256": jit["sha256"],
        "jit_valid_until": jit["valid_until"],
        "local_paid_api_minimum_ttl_seconds": (
            LOCAL_PAID_API_MINIMUM_TTL_SECONDS
        ),
        "remote_paid_launch_minimum_ttl_seconds": (
            REMOTE_PAID_LAUNCH_MINIMUM_TTL_SECONDS
        ),
    }
    for key, required in expected.items():
        if static.get(key) != required:
            raise OracleGateError(f"ORACLE_PAID_LAUNCH_STATIC_JIT_BINDING_DRIFT={key}")
    return {
        "status": "ORACLE_PAID_LAUNCH_READINESS_PASS",
        "static_auth_sha256": static_sha,
        "jit_budget_receipt_sha256": jit["sha256"],
        "jit_valid_until": jit["valid_until"],
        "launch_ttl_seconds": jit["launch_ttl_seconds"],
        "minimum_launch_ttl_seconds": int(MIN_PAID_LAUNCH_TTL.total_seconds()),
        "local_paid_api_minimum_ttl_seconds": (
            LOCAL_PAID_API_MINIMUM_TTL_SECONDS
        ),
        "remote_paid_launch_minimum_ttl_seconds": (
            REMOTE_PAID_LAUNCH_MINIMUM_TTL_SECONDS
        ),
    }


def materialize_code_archive_receipt(
    *,
    code_root: Path,
    bootstrap_root: Path,
    code_archive: Path,
    output: Path,
) -> dict[str, Any]:
    root = bootstrap_root.resolve()
    if root.name != NAMESPACE:
        raise OracleGateError("ORACLE_BOOTSTRAP_NAMESPACE_DRIFT")
    archive = _within(code_archive, root, "CODE_ARCHIVE")
    _, archive_sha, archive_size = _read_regular(archive, "CODE_ARCHIVE")
    records, deployment_sha = _deployment_records(code_root)
    tree_sha = _code_tree_sha256(code_root)
    payload = {
        "format": CODE_ARCHIVE_FORMAT,
        "status": CODE_ARCHIVE_STATUS,
        "namespace": NAMESPACE,
        "cpu_only": True,
        "gpu_visible": False,
        "archive_path": str(archive),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "code_root": str(code_root.resolve()),
        "code_tree_sha256": tree_sha,
        "deployment_contract_sha256": deployment_sha,
        "deployment_artifacts": records,
    }
    _atomic_write(output, _canonical_json(payload))
    return payload


def _validate_code_archive(
    path: Path, *, code_root: Path, bootstrap_root: Path
) -> dict[str, Any]:
    payload, receipt_sha, _ = _load_json(path, "CODE_ARCHIVE_RECEIPT")
    expected = {
        "format": CODE_ARCHIVE_FORMAT,
        "status": CODE_ARCHIVE_STATUS,
        "namespace": NAMESPACE,
        "cpu_only": True,
        "gpu_visible": False,
        "code_root": str(code_root.resolve()),
    }
    for key, required in expected.items():
        if payload.get(key) != required:
            raise OracleGateError(f"ORACLE_CODE_ARCHIVE_RECEIPT_DRIFT={key}")
    archive = _within(Path(str(payload.get("archive_path", ""))), bootstrap_root, "CODE_ARCHIVE")
    _, archive_sha, archive_size = _read_regular(archive, "CODE_ARCHIVE")
    if payload.get("archive_sha256") != archive_sha or payload.get("archive_size_bytes") != archive_size:
        raise OracleGateError("ORACLE_CODE_ARCHIVE_BYTES_DRIFT")
    records, deployment_sha = _deployment_records(code_root)
    tree_sha = _code_tree_sha256(code_root)
    if payload.get("deployment_artifacts") != records:
        raise OracleGateError("ORACLE_DEPLOYMENT_ARTIFACTS_DRIFT")
    if payload.get("deployment_contract_sha256") != deployment_sha:
        raise OracleGateError("ORACLE_DEPLOYMENT_CONTRACT_SHA_DRIFT")
    if payload.get("code_tree_sha256") != tree_sha:
        raise OracleGateError("ORACLE_CODE_TREE_SHA_DRIFT")
    return {
        "receipt_sha256": receipt_sha,
        "archive_sha256": archive_sha,
        "code_tree_sha256": tree_sha,
        "deployment_contract_sha256": deployment_sha,
        "deployment_artifacts": records,
    }


def _artifact_hashes(
    *, code_root: Path, config: Path, input_manifest: Path, task_manifest: Path
) -> dict[str, str]:
    return {
        "code_sha256": _code_tree_sha256(code_root),
        "config_sha256": _read_regular(config, "CONFIG_HASH")[1],
        "input_manifest_sha256": _read_regular(input_manifest, "INPUT_MANIFEST_HASH")[1],
        "task_manifest_sha256": _read_regular(task_manifest, "TASK_MANIFEST_HASH")[1],
    }


def _expected_approval(hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "approval_format": APPROVAL_FORMAT,
        "training_authorized": True,
        "formal_training_authorized": False,
        "comparison_only": True,
        "artifact_class": "TIMING_AND_ORACLE_COMPARISON_ONLY",
        "authorized_trainer": TRAINER,
        "run_id": RUN_ID,
        "endpoint_id": ENDPOINT_ID,
        "hardware_class": HARDWARE_CLASS,
        "approved_task_ids": [TASK_ID],
        "artifact_hashes": dict(hashes),
        "paid_enabled": True,
        "max_paid_hours": MAX_HOURS,
        "max_cost_cny": MAX_COST_CNY,
        "terminal_pass_does_not_authorize_formal_training": True,
        "equal_step_g0_g1_g2_pilot_still_required": True,
    }


def materialize_static_authorization(
    *,
    project_root: Path,
    code_root: Path,
    bootstrap_root: Path,
    prepared_parent: Path,
    config_path: Path,
    task_template_path: Path,
    source_input_receipt: Path,
    returned_r1_static_receipt: Path,
    jit_budget_path: Path,
    code_archive_receipt: Path,
    external_deployment_manifest: Path | None = None,
    now: datetime | None = None,
    externally_observed_verifier_sha256: str | None = None,
) -> dict[str, Any]:
    bootstrap = bootstrap_root.resolve()
    if bootstrap.name != NAMESPACE:
        raise OracleGateError("ORACLE_BOOTSTRAP_NAMESPACE_DRIFT")
    static_path = bootstrap / "STATIC_AUTH_READY.json"
    if static_path.exists():
        return validate_static_authorization(
            project_root=project_root,
            code_root=code_root,
            bootstrap_root=bootstrap,
            prepared_parent=prepared_parent,
            config_path=config_path,
            task_template_path=task_template_path,
            source_input_receipt=source_input_receipt,
            returned_r1_static_receipt=returned_r1_static_receipt,
            jit_budget_path=jit_budget_path,
            code_archive_receipt=code_archive_receipt,
            external_deployment_manifest=external_deployment_manifest,
            now=now,
            externally_observed_verifier_sha256=externally_observed_verifier_sha256,
        )
    config_gate = validate_config(
        config_path,
        code_root=code_root,
        prepared_parent=prepared_parent,
        externally_observed_verifier_sha256=externally_observed_verifier_sha256,
    )
    control_root = project_root.resolve() / "runtime/oracle_gates" / NAMESPACE
    deployment_path = external_deployment_manifest or (
        control_root / EXTERNAL_DEPLOYMENT_MANIFEST_NAME
    )
    external_deployment = _validate_external_deployment_manifest(
        deployment_path,
        control_root=control_root,
        expected_manifest_sha256=config_gate["oracle"][
            "external_deployment_manifest_sha256"
        ],
        externally_observed_verifier_sha256=externally_observed_verifier_sha256,
    )
    task_raw, task_template_sha, _ = _read_regular(task_template_path, "TASK_TEMPLATE")
    _task_rows(task_raw)
    lineage = _validate_r1_and_inputs(
        project_root=project_root,
        prepared_parent=prepared_parent,
        source_input_receipt=source_input_receipt,
        returned_r1_static_receipt=returned_r1_static_receipt,
    )
    jit = _validate_jit_budget(jit_budget_path, now=now)
    code = _validate_code_archive(
        code_archive_receipt, code_root=code_root, bootstrap_root=bootstrap
    )
    authorization = bootstrap / "authorization"
    task_path = authorization / "ORACLE_TASK_MANIFEST.tsv"
    input_path = authorization / "ORACLE_INPUT_MANIFEST.json"
    approval_path = authorization / "ORACLE_APPROVAL.json"
    input_payload = {
        "format": "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_INPUT_MANIFEST_V1",
        "comparison_only": True,
        "source_input_reuse_receipt_sha256": lineage["source_input_receipt_sha256"],
        "source_r1_static_auth_r6_sha256": lineage["r1_static_auth_r6_sha256"],
        "fold_inputs": [
            {
                "fold": EXPECTED_FOLD,
                "path": lineage["g2f0"]["path"],
                "sha256": lineage["g2f0"]["sha256"],
                "size_bytes": lineage["g2f0"]["size_bytes"],
            }
        ],
    }
    _atomic_write(task_path, task_raw)
    _atomic_write(input_path, _canonical_json(input_payload))
    hashes = _artifact_hashes(
        code_root=code_root,
        config=config_path,
        input_manifest=input_path,
        task_manifest=task_path,
    )
    approval = _expected_approval(hashes)
    _atomic_write(approval_path, _canonical_json(approval))
    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    receipt = {
        "format": STATIC_FORMAT,
        "status": STATIC_STATUS,
        "namespace": NAMESPACE,
        "created_at": created.isoformat(),
        "cpu_only_materialized": True,
        "gpu_visible_during_materialization": False,
        "comparison_only": True,
        "formal_training_authorized": False,
        "run_id": RUN_ID,
        "task_id": TASK_ID,
        "trainer": TRAINER,
        "graph_variant": EXPECTED_VARIANT,
        "patient_fold": EXPECTED_FOLD,
        "seed": EXPECTED_SEED,
        "max_paid_hours": MAX_HOURS,
        "configured_cost_cap_cny": MAX_COST_CNY,
        "jit_effective_cost_cap_cny": jit["effective_cost_cap_cny"],
        "jit_conservative_remaining_cny": jit["remaining_cny"],
        "jit_valid_until": jit["valid_until"],
        "local_paid_api_minimum_ttl_seconds": (
            LOCAL_PAID_API_MINIMUM_TTL_SECONDS
        ),
        "remote_paid_launch_minimum_ttl_seconds": (
            REMOTE_PAID_LAUNCH_MINIMUM_TTL_SECONDS
        ),
        "instance_id": jit["instance_id"],
        "artifact_hashes": hashes,
        "config_sha256": config_gate["config_sha256"],
        "task_template_sha256": task_template_sha,
        "task_manifest_sha256": hashes["task_manifest_sha256"],
        "input_manifest_sha256": hashes["input_manifest_sha256"],
        "approval_sha256": _read_regular(approval_path, "APPROVAL")[1],
        "oracle_decision_sha256": config_gate["oracle"]["decision_sha256"],
        "oracle_runner_sha256": config_gate["oracle"]["runner_sha256"],
        "preregistration_sha256": config_gate["oracle"]["preregistration_sha256"],
        "external_verifier_sha256": config_gate["oracle"][
            "external_verifier_sha256"
        ],
        "external_control_deployment_manifest_sha256": external_deployment[
            "sha256"
        ],
        "external_control_artifacts": external_deployment["artifacts"],
        "r1_aborted_sha256": lineage["r1_aborted_sha256"],
        "r1_static_auth_r6_sha256": lineage["r1_static_auth_r6_sha256"],
        "returned_r1_static_auth_r6_sha256": lineage[
            "returned_r1_static_auth_r6_sha256"
        ],
        "source_input_reuse_receipt_sha256": lineage["source_input_receipt_sha256"],
        "code_archive_receipt_sha256": code["receipt_sha256"],
        "code_archive_sha256": code["archive_sha256"],
        "deployment_contract_sha256": code["deployment_contract_sha256"],
        "jit_budget_receipt_sha256": jit["sha256"],
        "authorization_paths": {
            "task_manifest": str(task_path),
            "input_manifest": str(input_path),
            "approval": str(approval_path),
        },
        "formal_artifacts_allowed": False,
        "checkpoint_allowed": False,
        "prediction_allowed": False,
        "terminal_pass_does_not_authorize_formal_training": True,
        "equal_step_g0_g1_g2_pilot_still_required": True,
    }
    encoded = json.dumps(receipt, sort_keys=True)
    if "/results/" in encoded or "v32_g012_paid_gpu_training_" in encoded:
        raise OracleGateError("ORACLE_STATIC_AUTH_FORMAL_RESULT_REFERENCE_FORBIDDEN")
    _atomic_write(static_path, _canonical_json(receipt))
    return receipt


def validate_static_authorization(
    *,
    project_root: Path,
    code_root: Path,
    bootstrap_root: Path,
    prepared_parent: Path,
    config_path: Path,
    task_template_path: Path,
    source_input_receipt: Path,
    returned_r1_static_receipt: Path,
    jit_budget_path: Path,
    code_archive_receipt: Path,
    external_deployment_manifest: Path | None = None,
    now: datetime | None = None,
    externally_observed_verifier_sha256: str | None = None,
) -> dict[str, Any]:
    bootstrap = bootstrap_root.resolve()
    static_path = bootstrap / "STATIC_AUTH_READY.json"
    receipt, _, _ = _load_json(static_path, "STATIC_AUTH")
    config_gate = validate_config(
        config_path,
        code_root=code_root,
        prepared_parent=prepared_parent,
        externally_observed_verifier_sha256=externally_observed_verifier_sha256,
    )
    control_root = project_root.resolve() / "runtime/oracle_gates" / NAMESPACE
    deployment_path = external_deployment_manifest or (
        control_root / EXTERNAL_DEPLOYMENT_MANIFEST_NAME
    )
    external_deployment = _validate_external_deployment_manifest(
        deployment_path,
        control_root=control_root,
        expected_manifest_sha256=config_gate["oracle"][
            "external_deployment_manifest_sha256"
        ],
        externally_observed_verifier_sha256=externally_observed_verifier_sha256,
    )
    task_raw, task_template_sha, _ = _read_regular(task_template_path, "TASK_TEMPLATE")
    _task_rows(task_raw)
    lineage = _validate_r1_and_inputs(
        project_root=project_root,
        prepared_parent=prepared_parent,
        source_input_receipt=source_input_receipt,
        returned_r1_static_receipt=returned_r1_static_receipt,
    )
    jit = _validate_jit_budget(jit_budget_path, now=now)
    code = _validate_code_archive(
        code_archive_receipt, code_root=code_root, bootstrap_root=bootstrap
    )
    authorization = bootstrap / "authorization"
    task_path = authorization / "ORACLE_TASK_MANIFEST.tsv"
    input_path = authorization / "ORACLE_INPUT_MANIFEST.json"
    approval_path = authorization / "ORACLE_APPROVAL.json"
    observed_task, _, _ = _read_regular(task_path, "TASK_MANIFEST")
    if observed_task != task_raw:
        raise OracleGateError("ORACLE_TASK_MANIFEST_TEMPLATE_DRIFT")
    _task_rows(observed_task)
    input_payload, input_sha, _ = _load_json(input_path, "INPUT_MANIFEST")
    fold_inputs = input_payload.get("fold_inputs")
    expected_input_row = {
        "fold": EXPECTED_FOLD,
        "path": lineage["g2f0"]["path"],
        "sha256": lineage["g2f0"]["sha256"],
        "size_bytes": lineage["g2f0"]["size_bytes"],
    }
    if fold_inputs != [expected_input_row]:
        raise OracleGateError("ORACLE_INPUT_MANIFEST_SCOPE_DRIFT")
    if input_payload.get("source_input_reuse_receipt_sha256") != lineage["source_input_receipt_sha256"]:
        raise OracleGateError("ORACLE_INPUT_MANIFEST_REUSE_BINDING_DRIFT")
    if input_payload.get("source_r1_static_auth_r6_sha256") != EXPECTED_R1_STATIC_AUTH_R6_SHA256:
        raise OracleGateError("ORACLE_INPUT_MANIFEST_R1_BINDING_DRIFT")
    hashes = _artifact_hashes(
        code_root=code_root,
        config=config_path,
        input_manifest=input_path,
        task_manifest=task_path,
    )
    approval, approval_sha, _ = _load_json(approval_path, "APPROVAL")
    if approval != _expected_approval(hashes):
        raise OracleGateError("ORACLE_APPROVAL_DRIFT")
    required_receipt = {
        "format": STATIC_FORMAT,
        "status": STATIC_STATUS,
        "namespace": NAMESPACE,
        "cpu_only_materialized": True,
        "gpu_visible_during_materialization": False,
        "comparison_only": True,
        "formal_training_authorized": False,
        "run_id": RUN_ID,
        "task_id": TASK_ID,
        "trainer": TRAINER,
        "graph_variant": EXPECTED_VARIANT,
        "patient_fold": EXPECTED_FOLD,
        "seed": EXPECTED_SEED,
        "max_paid_hours": MAX_HOURS,
        "configured_cost_cap_cny": MAX_COST_CNY,
        "jit_effective_cost_cap_cny": jit["effective_cost_cap_cny"],
        "jit_conservative_remaining_cny": jit["remaining_cny"],
        "jit_valid_until": jit["valid_until"],
        "local_paid_api_minimum_ttl_seconds": (
            LOCAL_PAID_API_MINIMUM_TTL_SECONDS
        ),
        "remote_paid_launch_minimum_ttl_seconds": (
            REMOTE_PAID_LAUNCH_MINIMUM_TTL_SECONDS
        ),
        "instance_id": jit["instance_id"],
        "artifact_hashes": hashes,
        "config_sha256": config_gate["config_sha256"],
        "task_template_sha256": task_template_sha,
        "task_manifest_sha256": hashes["task_manifest_sha256"],
        "input_manifest_sha256": input_sha,
        "approval_sha256": approval_sha,
        "oracle_decision_sha256": config_gate["oracle"]["decision_sha256"],
        "oracle_runner_sha256": config_gate["oracle"]["runner_sha256"],
        "preregistration_sha256": config_gate["oracle"]["preregistration_sha256"],
        "external_verifier_sha256": config_gate["oracle"][
            "external_verifier_sha256"
        ],
        "external_control_deployment_manifest_sha256": external_deployment[
            "sha256"
        ],
        "external_control_artifacts": external_deployment["artifacts"],
        "r1_aborted_sha256": lineage["r1_aborted_sha256"],
        "r1_static_auth_r6_sha256": lineage["r1_static_auth_r6_sha256"],
        "returned_r1_static_auth_r6_sha256": lineage[
            "returned_r1_static_auth_r6_sha256"
        ],
        "source_input_reuse_receipt_sha256": lineage["source_input_receipt_sha256"],
        "code_archive_receipt_sha256": code["receipt_sha256"],
        "code_archive_sha256": code["archive_sha256"],
        "deployment_contract_sha256": code["deployment_contract_sha256"],
        "jit_budget_receipt_sha256": jit["sha256"],
        "formal_artifacts_allowed": False,
        "checkpoint_allowed": False,
        "prediction_allowed": False,
        "terminal_pass_does_not_authorize_formal_training": True,
        "equal_step_g0_g1_g2_pilot_still_required": True,
    }
    for key, expected in required_receipt.items():
        if receipt.get(key) != expected:
            raise OracleGateError(f"ORACLE_STATIC_AUTH_DRIFT={key}")
    paths = _mapping(receipt.get("authorization_paths"), "AUTHORIZATION_PATHS")
    expected_paths = {
        "task_manifest": str(task_path),
        "input_manifest": str(input_path),
        "approval": str(approval_path),
    }
    if paths != expected_paths:
        raise OracleGateError("ORACLE_STATIC_AUTH_PATH_DRIFT")
    encoded = json.dumps(receipt, sort_keys=True)
    if "/results/" in encoded or "v32_g012_paid_gpu_training_" in encoded:
        raise OracleGateError("ORACLE_STATIC_AUTH_FORMAL_RESULT_REFERENCE_FORBIDDEN")
    return receipt


def validate_terminal_log(path: Path, expected_exit_code: int) -> dict[str, Any]:
    raw, _, _ = _read_regular(path, "TERMINAL_LOG")
    receipts = []
    for line in raw.decode("utf-8", errors="strict").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(item, Mapping)
            and item.get("format") == TERMINAL_RECEIPT_FORMAT
            and item.get("status") in ({PASS_STATUS} | FAIL_STATUSES)
        ):
            receipts.append(dict(item))
    if len(receipts) != 1:
        raise OracleGateError(f"ORACLE_TERMINAL_RECEIPT_CARDINALITY={len(receipts)}")
    receipt = receipts[0]
    status = receipt.get("status")
    expected_identity = {
        "run_id": RUN_ID,
        "task_id": TASK_ID,
        "graph_variant": EXPECTED_VARIANT,
        "patient_fold": EXPECTED_FOLD,
        "seed": EXPECTED_SEED,
    }
    for key, expected in expected_identity.items():
        if receipt.get(key) != expected:
            raise OracleGateError(f"ORACLE_TERMINAL_IDENTITY_DRIFT={key}")
    expected_statuses = {PASS_STATUS} if expected_exit_code == 0 else FAIL_STATUSES
    if expected_exit_code not in {0, 42} or status not in expected_statuses:
        raise OracleGateError("ORACLE_TERMINAL_STATUS_EXIT_DRIFT")
    required_false = {
        "formal_training_authorized",
        "checkpoint_written",
        "success_json_written",
        "failure_json_written",
        "prediction_written",
        "winner_selection_input",
    }
    for key in required_false:
        if receipt.get(key) is not False:
            raise OracleGateError(f"ORACLE_TERMINAL_FORMAL_FLAG_DRIFT={key}")
    if receipt.get("formal_artifacts_written") != 0:
        raise OracleGateError("ORACLE_TERMINAL_FORMAL_ARTIFACT_COUNT_DRIFT")
    if expected_exit_code == 0 and receipt.get("scientific_pass") is not True:
        raise OracleGateError("ORACLE_TERMINAL_PASS_FLAG_DRIFT")
    if expected_exit_code == 42 and receipt.get("scientific_pass") is not False:
        raise OracleGateError("ORACLE_TERMINAL_FAIL_FLAG_DRIFT")
    return receipt


def _defaults(project_root: Path) -> dict[str, Path]:
    project = project_root.resolve()
    code = project / "runtime/tools" / NAMESPACE / "code"
    bootstrap = project / "runtime/bootstrap" / NAMESPACE
    prepared = (
        project
        / "inputs/v32_g012_patient_first_20260830_r1"
        / "formal_prepared_20260830_r3_affine_pyg280_localtorch"
    )
    return {
        "code_root": code,
        "bootstrap_root": bootstrap,
        "prepared_parent": prepared,
        "config": code / CONFIG_RELATIVE,
        "task_template": code / TASK_TEMPLATE_RELATIVE,
        "source_input_receipt": (
            project / "runtime/bootstrap" / FORMAL_R2_NAMESPACE / "INPUT_REUSE_READY.json"
        ),
        "returned_r1_static_receipt": (
            project
            / "runtime/oracle_gates"
            / NAMESPACE
            / "authority/STATIC_AUTH_READY.v32_g012_20260901_r6.json"
        ),
        "jit_budget": bootstrap / "JIT_BUDGET_READY.json",
        "code_archive_receipt": bootstrap / "CODE_ARCHIVE_READY.json",
        "external_deployment_manifest": (
            project
            / "runtime/oracle_gates"
            / NAMESPACE
            / EXTERNAL_DEPLOYMENT_MANIFEST_NAME
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("./data/CancerLncAtlas"))
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--bootstrap-root", type=Path)
    parser.add_argument("--prepared-parent", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task-template", type=Path)
    parser.add_argument("--source-input-receipt", type=Path)
    parser.add_argument("--returned-r1-static-receipt", type=Path)
    parser.add_argument("--jit-budget", type=Path)
    parser.add_argument("--code-archive-receipt", type=Path)
    parser.add_argument("--code-archive", type=Path)
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--materialize-code-ready", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--paid-launch-readiness", action="store_true")
    parser.add_argument("--terminal-log", type=Path)
    parser.add_argument("--expected-exit-code", type=int)
    parser.add_argument("--external-verifier-sha256")
    parser.add_argument("--external-deployment-manifest", type=Path)
    args = parser.parse_args(argv)
    defaults = _defaults(args.project_root)
    code_root = args.code_root or defaults["code_root"]
    bootstrap = args.bootstrap_root or defaults["bootstrap_root"]
    prepared = args.prepared_parent or defaults["prepared_parent"]
    config = args.config or defaults["config"]
    task_template = args.task_template or defaults["task_template"]
    source_input = args.source_input_receipt or defaults["source_input_receipt"]
    returned_r1_static = (
        args.returned_r1_static_receipt
        or defaults["returned_r1_static_receipt"]
    )
    jit = args.jit_budget or defaults["jit_budget"]
    code_receipt = args.code_archive_receipt or defaults["code_archive_receipt"]
    external_deployment = (
        args.external_deployment_manifest
        or defaults["external_deployment_manifest"]
    )
    try:
        if args.terminal_log is not None:
            if args.expected_exit_code is None:
                raise OracleGateError("ORACLE_EXPECTED_EXIT_CODE_REQUIRED")
            result = validate_terminal_log(args.terminal_log, args.expected_exit_code)
        elif args.paid_launch_readiness:
            result = validate_paid_launch_readiness(
                jit_budget_path=jit,
                static_auth_path=bootstrap / "STATIC_AUTH_READY.json",
            )
        elif args.config_only:
            if args.external_verifier_sha256 is None:
                raise OracleGateError("ORACLE_EXTERNAL_VERIFIER_OBSERVED_SHA_REQUIRED")
            result = validate_config(
                config,
                code_root=code_root,
                prepared_parent=prepared,
                externally_observed_verifier_sha256=args.external_verifier_sha256,
            )
            deployment = _validate_external_deployment_manifest(
                external_deployment,
                control_root=args.project_root.resolve()
                / "runtime/oracle_gates"
                / NAMESPACE,
                expected_manifest_sha256=result["oracle"][
                    "external_deployment_manifest_sha256"
                ],
                externally_observed_verifier_sha256=args.external_verifier_sha256,
            )
            result = {
                "status": "ORACLE_CONFIG_GATE_PASS_CPU_ONLY",
                "config_sha256": result["config_sha256"],
                "external_verifier_sha256": result["oracle"][
                    "external_verifier_sha256"
                ],
                "external_control_deployment_manifest_sha256": deployment[
                    "sha256"
                ],
            }
        elif args.materialize_code_ready:
            if args.code_archive is None:
                raise OracleGateError("ORACLE_CODE_ARCHIVE_REQUIRED")
            result = materialize_code_archive_receipt(
                code_root=code_root,
                bootstrap_root=bootstrap,
                code_archive=args.code_archive,
                output=code_receipt,
            )
        elif args.materialize:
            if args.external_verifier_sha256 is None:
                raise OracleGateError("ORACLE_EXTERNAL_VERIFIER_OBSERVED_SHA_REQUIRED")
            result = materialize_static_authorization(
                project_root=args.project_root,
                code_root=code_root,
                bootstrap_root=bootstrap,
                prepared_parent=prepared,
                config_path=config,
                task_template_path=task_template,
                source_input_receipt=source_input,
                returned_r1_static_receipt=returned_r1_static,
                jit_budget_path=jit,
                code_archive_receipt=code_receipt,
                external_deployment_manifest=external_deployment,
                externally_observed_verifier_sha256=args.external_verifier_sha256,
            )
        else:
            if args.external_verifier_sha256 is None:
                raise OracleGateError("ORACLE_EXTERNAL_VERIFIER_OBSERVED_SHA_REQUIRED")
            result = validate_static_authorization(
                project_root=args.project_root,
                code_root=code_root,
                bootstrap_root=bootstrap,
                prepared_parent=prepared,
                config_path=config,
                task_template_path=task_template,
                source_input_receipt=source_input,
                returned_r1_static_receipt=returned_r1_static,
                jit_budget_path=jit,
                code_archive_receipt=code_receipt,
                external_deployment_manifest=external_deployment,
                externally_observed_verifier_sha256=args.external_verifier_sha256,
            )
    except OracleGateError as exc:
        print(str(exc), file=sys.stderr)
        return 42
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
