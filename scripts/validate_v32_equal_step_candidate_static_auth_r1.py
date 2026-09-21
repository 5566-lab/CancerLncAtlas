#!/usr/bin/env python3
"""External, fail-closed verifier for the G0/G1/G2 equal-step paid gate.

Deploy this file outside the staged code overlay.  It never imports staged
Python and never imports Torch.  Overlay source is parsed or hashed as data;
authority files are opened with O_NOFOLLOW, require nlink=1, and are checked
through one stable descriptor before and after every read/stat operation.
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


NAMESPACE = "v32_equal_step_candidate_paid_gpu_20260901_r1"
PILOT_ID = "v32-g012-equal-step-candidate-paid-gpu-20260901-r1"
JOB_ID = "v32-equal-step-g012f0-s20260726-r1"
VARIANTS = ("G0", "G1", "G2")
FOLD = 0
SEED = 20260726
TRAINER = (
    "cc_hhgt.v32.equal_step_candidate_pilot:"
    "run_authorized_equal_step_candidate_pilot"
)
ENDPOINT = "paid_gpu"
HARDWARE = "PAID_PREEMPTIBLE_GPU"
MAX_HOURS = 6.0
MAX_COST_CNY = 15.0
COMPUTE_HOURLY_CNY = 2.05
DISK_HOURLY_CNY = 0.04
TOTAL_HOURLY_CNY = 2.09
PROJECT_BUDGET_CAP_CNY = 210.0
MAX_JIT_VALIDITY = timedelta(minutes=30)
JIT_SKEW = timedelta(minutes=1)

STATIC_FORMAT = "CC_HHGT_V3_2_EQUAL_STEP_CANDIDATE_STATIC_AUTH_R1_V1"
STATIC_STATUS = "EQUAL_STEP_STATIC_AUTH_READY_CPU_ONLY"
APPROVAL_FORMAT = "CC_HHGT_V3_2_TRAINING_APPROVAL_V1"
INPUT_FORMAT = "CC_HHGT_V3_2_G012_R2_INPUT_REUSE_READY_V1"
INPUT_STATUS = "INPUT_REUSE_READY_HASH_VERIFIED"
ABORT_FORMAT = "CC_HHGT_V3_2_G012_ABORTED_RUN_V1"
ABORT_STATUS = "ABORTED_RUNTIME_INFEASIBLE"
JIT_FORMAT = "CANCERLNCATLAS_COMPSHARE_JIT_BUDGET_V1"
JIT_STATUS = "JIT_BUDGET_READY_CONSERVATIVE"
CODE_FORMAT = "CC_HHGT_V3_2_EQUAL_STEP_CODE_ARCHIVE_R1_V1"
CODE_STATUS = "EQUAL_STEP_CODE_ARCHIVE_READY_CPU_ONLY"
DEPLOYMENT_FORMAT = "CC_HHGT_V3_2_EQUAL_STEP_EXTERNAL_DEPLOYMENT_R1_V1"
DEPLOYMENT_STATUS = "EXTERNAL_CONTROL_ARTIFACTS_FROZEN"
TERMINAL_SCHEMA = "CC_HHGT_V3_2_EQUAL_STEP_CANDIDATE_PILOT_V1"
PASS_STATUS = "PASS_EQUAL_STEP_CANDIDATE_ONLY_NOT_AUTHORIZED_FOR_FORMAL_TRAINING"
FAIL_STATUSES = frozenset(
    {
        "SCIENTIFIC_FAIL_EQUAL_STEP_CANDIDATE_ONLY",
        "RUNTIME_FAIL_EQUAL_STEP_CANDIDATE_ONLY",
    }
)
ORACLE_FORMAT = "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_COMPARISON_V1"
ORACLE_PASS = (
    "PASS_REAL_DATA_ORACLE_COMPARISON_ONLY_NOT_AUTHORIZED_FOR_FORMAL_TRAINING"
)
ORACLE_RUN_ID = "v32-g012-g2-group-shared-oracle-paid-gpu-20260901-r2"
ORACLE_TASK_ID = f"{ORACLE_RUN_ID}|PATIENT_FOLD_0|CC-HHGT|20260726"

FORMAL_R2_NAMESPACE = "v32_g012_paid_gpu_20260901_r2"
R1_NAMESPACE = "v32_g012_paid_gpu_20260831_r1"
R1_ARCHIVE_NAMESPACE = "archive_aborted_runtime_infeasible_20260901_r1"
EXPECTED_R1_STATIC_R6_SHA256 = (
    "1f5bc54b2a8acc037373bfa380c95c8a717e76fde956852ec481b16a0d8716f0"
)
SOURCE_INPUT_ARCHIVE_SHA256 = (
    "1c17b7be89621e5125c39e87f05ce14beb1f2227afdb481d2e9a20049b56bab0"
)
# G0/G1 Fold0 hashes are transitively hard-anchored by the immutable r1-r6
# receipt SHA above.  Their independently observed byte sizes are fixed here;
# G2 additionally had a separately returned direct hash anchor.
FOLD0_SIZE_ANCHORS = {
    "G0": 3_748_393_077,
    "G1": 4_346_233_053,
    "G2": 4_482_708_589,
}
G2_F0_SHA256 = (
    "4ebba1d4060f3efaa8c080824386802ca732bedf07681ebd0283deb087b0ca39"
)

TEMPLATE_RELATIVE = "config/model_v3_2_equal_step_candidate_paid_gpu_20260901_r1.json"
PILOT_RELATIVE = "cc_hhgt/v32/equal_step_candidate_pilot.py"
BOOTSTRAP_RELATIVE = "cc_hhgt/v32/equal_step_candidate_bootstrap.py"
PREREG_RELATIVE = "docs/v32_group_shared_encoder_estimator_preregistration_20260901.md"
DECISION_RELATIVE = "docs/v32_equal_step_candidate_paid_decision_20260901_r1.json"
TRAINING_RELATIVE = "cc_hhgt/v32/training.py"
ORACLE_RUNNER_RELATIVE = "cc_hhgt/v32/group_shared_encoder_oracle.py"
VALIDATOR_NAME = "validate_v32_equal_step_candidate_static_auth_r1.py"
LAUNCHER_NAME = "server_launch_v32_equal_step_candidate_paid_gpu_20260901_r1.sh"
AUTHORIZER_NAME = "cloud_authorize_v32_equal_step_candidate_no_gpu_20260901_r1.sh"
SUPERVISOR_NAME = "local_supervise_compshare_equal_step_candidate_paid_gpu_20260901_r1.ps1"
DISPATCH_NAME = "local_dispatch_compshare_equal_step_candidate_paid_gpu_20260901_r1.ps1"
LOG_GUARD_NAME = "v32_equal_step_candidate_log_guard_r1.psm1"
DEPLOYMENT_NAME = "EXTERNAL_DEPLOYMENT_MANIFEST.json"
CONTROL_NAMES = (
    VALIDATOR_NAME,
    LAUNCHER_NAME,
    AUTHORIZER_NAME,
    SUPERVISOR_NAME,
    DISPATCH_NAME,
    LOG_GUARD_NAME,
)


class EqualStepGateError(RuntimeError):
    pass


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EqualStepGateError(f"EQUAL_STEP_{label}_MAPPING_REQUIRED")
    return dict(value)


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _within(path: Path, root: Path, label: str) -> Path:
    boundary = Path(os.path.abspath(root))
    candidate = Path(os.path.abspath(path))
    try:
        relative = candidate.relative_to(boundary)
    except ValueError as exc:
        raise EqualStepGateError(
            f"EQUAL_STEP_{label}_ESCAPES_COMPONENT_ROOT={candidate}"
        ) from exc
    current = boundary
    if current.is_symlink():
        raise EqualStepGateError(f"EQUAL_STEP_{label}_SYMLINK_BOUNDARY={current}")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise EqualStepGateError(f"EQUAL_STEP_{label}_SYMLINK_COMPONENT={current}")
    return candidate


def _open_regular(path: Path, label: str) -> int:
    flags = (
        os.O_RDONLY | getattr(os, "O_BINARY", 0) |
        getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise EqualStepGateError(f"EQUAL_STEP_{label}_OPEN_FAILED={path}") from exc
    observed = os.fstat(fd)
    if not stat.S_ISREG(observed.st_mode) or observed.st_size <= 0:
        os.close(fd)
        raise EqualStepGateError(f"EQUAL_STEP_{label}_NOT_NONEMPTY_REGULAR={path}")
    if int(observed.st_nlink) != 1:
        os.close(fd)
        raise EqualStepGateError(
            f"EQUAL_STEP_{label}_HARDLINK_FORBIDDEN_NLINK={observed.st_nlink}:{path}"
        )
    return fd


def _read(path: Path, label: str) -> tuple[bytes, str, int]:
    fd = _open_regular(path, label)
    try:
        before = _identity(os.fstat(fd))
        digest = hashlib.sha256()
        pieces: list[bytes] = []
        while True:
            piece = os.read(fd, 1024 * 1024)
            if not piece:
                break
            digest.update(piece)
            pieces.append(piece)
        after = _identity(os.fstat(fd))
        if before != after:
            raise EqualStepGateError(f"EQUAL_STEP_{label}_CHANGED_DURING_READ={path}")
        return b"".join(pieces), digest.hexdigest(), int(before[3])
    finally:
        os.close(fd)


def _stat_size(path: Path, label: str) -> int:
    fd = _open_regular(path, label)
    try:
        before = _identity(os.fstat(fd))
        after = _identity(os.fstat(fd))
        if before != after:
            raise EqualStepGateError(f"EQUAL_STEP_{label}_CHANGED_DURING_STAT={path}")
        return int(before[3])
    finally:
        os.close(fd)


def _json(path: Path, label: str) -> tuple[dict[str, Any], str, int]:
    raw, digest, size = _read(path, label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EqualStepGateError(f"EQUAL_STEP_{label}_JSON_INVALID={path}") from exc
    return _mapping(value, label), digest, size


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current, _, _ = _read(path, "EXISTING_TARGET")
        if current != payload:
            raise EqualStepGateError(f"EQUAL_STEP_EXISTING_TARGET_DRIFT={path}")
        return
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(partial, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(partial, path)


def _parse_time(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise EqualStepGateError(f"EQUAL_STEP_{label}_TIMESTAMP_REQUIRED")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EqualStepGateError(f"EQUAL_STEP_{label}_TIMESTAMP_INVALID") from exc
    if parsed.tzinfo is None:
        raise EqualStepGateError(f"EQUAL_STEP_{label}_TIMEZONE_REQUIRED")
    return parsed.astimezone(timezone.utc)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EqualStepGateError(f"EQUAL_STEP_{label}_NUMBER_REQUIRED")
    result = float(value)
    if not math.isfinite(result):
        raise EqualStepGateError(f"EQUAL_STEP_{label}_FINITE_REQUIRED")
    return result


def _code_files(code_root: Path) -> list[tuple[str, Path]]:
    root = Path(os.path.abspath(code_root))
    found: list[tuple[str, Path]] = []
    for source in (root / "cc_hhgt", root / "scripts/v32_pipeline.py"):
        _within(source, root, "CODE_TREE")
        if source.is_file() and not source.is_symlink():
            found.append((source.relative_to(root).as_posix(), source))
            continue
        if not source.is_dir() or source.is_symlink():
            raise EqualStepGateError(f"EQUAL_STEP_CODE_SOURCE_INVALID={source}")
        for candidate in source.rglob("*"):
            relative = candidate.relative_to(root)
            if any(p in {"__pycache__", ".pytest_cache", ".git"} for p in relative.parts):
                continue
            if candidate.suffix.lower() in {".pyc", ".pyo"}:
                continue
            _within(candidate, root, "CODE_TREE_COMPONENT")
            if candidate.is_symlink():
                raise EqualStepGateError(f"EQUAL_STEP_CODE_SYMLINK={candidate}")
            if candidate.is_file():
                found.append((relative.as_posix(), candidate))
    if not found:
        raise EqualStepGateError("EQUAL_STEP_CODE_TREE_EMPTY")
    return sorted(found)


def _code_sha(code_root: Path) -> str:
    digest = hashlib.sha256()
    for relative, path in _code_files(code_root):
        content_sha = _read(path, "CODE_FILE")[1]
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_sha.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _ast_constants(path: Path, names: set[str], label: str) -> dict[str, Any]:
    raw, _, _ = _read(path, label)
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise EqualStepGateError(f"EQUAL_STEP_{label}_AST_INVALID") from exc
    values: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                try:
                    values[name] = ast.literal_eval(node.value)
                except (TypeError, ValueError) as exc:
                    raise EqualStepGateError(f"EQUAL_STEP_{label}_NONLITERAL={name}") from exc
    if set(values) != names:
        raise EqualStepGateError(f"EQUAL_STEP_{label}_CONSTANTS_MISSING={sorted(names-set(values))}")
    return values


def validate_overlay(code_root: Path, template_path: Path) -> dict[str, Any]:
    template, template_sha, _ = _json(template_path, "CONFIG_TEMPLATE")
    required_template = {
        "contract_version": "3.2.0-equal-step-candidate-paid-gpu-20260901-r1",
        "graph_variant": "__GRAPH_VARIANT__",
        "pilot_id": PILOT_ID,
        "trainer": TRAINER,
        "formal_training_authorized": False,
    }
    contract = _mapping(template.get("equal_step_contract"), "CONTRACT")
    for key, expected in required_template.items():
        observed = template.get(key) if key == "contract_version" else contract.get(key)
        if observed != expected:
            raise EqualStepGateError(f"EQUAL_STEP_CONFIG_TEMPLATE_DRIFT={key}")
    if not _sha(contract.get("external_deployment_manifest_sha256")):
        raise EqualStepGateError("EQUAL_STEP_CONFIG_DEPLOYMENT_MANIFEST_SHA_REQUIRED")
    if (
        contract.get("decision_path") != DECISION_RELATIVE or
        not _sha(contract.get("decision_sha256"))
    ):
        raise EqualStepGateError("EQUAL_STEP_CONFIG_DECISION_BINDING_DRIFT")
    control = _mapping(template.get("execution_control"), "EXECUTION_CONTROL")
    expected_control = {
        "execution_mode": "TRAINING", "training_authorized": True,
        "paid_enabled": True, "max_paid_hours": MAX_HOURS,
        "max_cost_cny": MAX_COST_CNY, "candidate_only": True,
        "comparison_only": True, "formal_training_authorized": False,
        "overwrite_formal_v32": False,
    }
    for key, expected in expected_control.items():
        if control.get(key) != expected:
            raise EqualStepGateError(f"EQUAL_STEP_EXECUTION_CONTROL_DRIFT={key}")
    pilot_path = _within(code_root / PILOT_RELATIVE, code_root, "PILOT")
    bootstrap_path = _within(code_root / BOOTSTRAP_RELATIVE, code_root, "BOOTSTRAP")
    prereg_path = _within(code_root / PREREG_RELATIVE, code_root, "PREREG")
    decision_path = _within(code_root / DECISION_RELATIVE, code_root, "DECISION")
    training_path = _within(code_root / TRAINING_RELATIVE, code_root, "TRAINING")
    oracle_runner_path = _within(
        code_root / ORACLE_RUNNER_RELATIVE, code_root, "ORACLE_RUNNER"
    )
    pilot_constants = _ast_constants(
        pilot_path,
        {"RECEIPT_SCHEMA", "PASS_STATUS", "SCIENTIFIC_FAIL_STATUS", "RUNTIME_FAIL_STATUS",
         "EXPECTED_VARIANTS", "EXPECTED_FOLD", "EXPECTED_SEED", "EXPECTED_BATCH_COUNT",
         "EXPECTED_OPTIMIZER_STEPS", "EXPECTED_RUNTIME_CHUNKS", "AUTHORIZED_TRAINER_SPECIFICATION"},
        "PILOT",
    )
    if (
        pilot_constants["RECEIPT_SCHEMA"] != TERMINAL_SCHEMA or
        pilot_constants["PASS_STATUS"] != PASS_STATUS or
        frozenset({pilot_constants["SCIENTIFIC_FAIL_STATUS"], pilot_constants["RUNTIME_FAIL_STATUS"]}) != FAIL_STATUSES or
        tuple(pilot_constants["EXPECTED_VARIANTS"]) != VARIANTS or
        pilot_constants["EXPECTED_FOLD"] != FOLD or
        pilot_constants["EXPECTED_SEED"] != SEED or
        pilot_constants["EXPECTED_BATCH_COUNT"] != 403 or
        pilot_constants["EXPECTED_OPTIMIZER_STEPS"] != 101 or
        pilot_constants["EXPECTED_RUNTIME_CHUNKS"] != 29 or
        pilot_constants["AUTHORIZED_TRAINER_SPECIFICATION"] != TRAINER
    ):
        raise EqualStepGateError("EQUAL_STEP_PILOT_SEMANTIC_CONSTANT_DRIFT")
    bootstrap_constants = _ast_constants(
        bootstrap_path, {"PILOT_ID", "TRAINER", "VARIANTS", "SEED", "FOLD", "ENDPOINT_ID", "HARDWARE_CLASS"}, "BOOTSTRAP"
    )
    expected_bootstrap = {
        "PILOT_ID": PILOT_ID, "TRAINER": TRAINER, "VARIANTS": VARIANTS,
        "SEED": SEED, "FOLD": FOLD, "ENDPOINT_ID": ENDPOINT,
        "HARDWARE_CLASS": HARDWARE,
    }
    if bootstrap_constants != expected_bootstrap:
        raise EqualStepGateError("EQUAL_STEP_BOOTSTRAP_CONSTANT_DRIFT")
    decision, decision_sha, _ = _json(decision_path, "DECISION")
    if decision_sha != contract["decision_sha256"]:
        raise EqualStepGateError("EQUAL_STEP_DECISION_SHA_DRIFT")
    decision_expected = {
        "format": "CC_HHGT_V3_2_EQUAL_STEP_CANDIDATE_PAID_DECISION_R1_V1",
        "decision": "EQUAL_STEP_CANDIDATE_COMPARISON_PAID_GATE_AUTHORIZED",
        "authorized_callable": TRAINER, "formal_training_authorized": False,
        "budget_authorized_by_this_decision": False,
        "hour_cap": MAX_HOURS, "comparison_cost_cap_cny": MAX_COST_CNY,
        "terminal_schema": TERMINAL_SCHEMA, "terminal_pass_status": PASS_STATUS,
        "terminal_pass_does_not_authorize_formal_training": True,
        "pilot_sha256": _read(pilot_path, "PILOT_DECISION_HASH")[1],
        "bootstrap_sha256": _read(bootstrap_path, "BOOTSTRAP_DECISION_HASH")[1],
        "preregistration_sha256": _read(prereg_path, "PREREG_DECISION_HASH")[1],
    }
    for key, expected in decision_expected.items():
        if decision.get(key) != expected:
            raise EqualStepGateError(f"EQUAL_STEP_DECISION_SEMANTIC_DRIFT={key}")
    records = []
    for relative, path in (
        (TEMPLATE_RELATIVE, template_path), (PILOT_RELATIVE, pilot_path),
        (BOOTSTRAP_RELATIVE, bootstrap_path), (PREREG_RELATIVE, prereg_path),
        (DECISION_RELATIVE, decision_path),
        (TRAINING_RELATIVE, training_path),
        ("cc_hhgt/v32/training_guard.py", code_root / "cc_hhgt/v32/training_guard.py"),
        ("scripts/v32_pipeline.py", code_root / "scripts/v32_pipeline.py"),
    ):
        _, digest, size = _read(path, "DEPLOYMENT_ARTIFACT")
        records.append({"relative_path": relative, "sha256": digest, "size_bytes": size})
    return {
        "template": template,
        "template_sha256": template_sha,
        "pilot_sha256": _read(pilot_path, "PILOT_HASH")[1],
        "bootstrap_sha256": _read(bootstrap_path, "BOOTSTRAP_HASH")[1],
        "preregistration_sha256": _read(prereg_path, "PREREG_HASH")[1],
        "decision_sha256": decision_sha,
        "production_training_sha256": _read(training_path, "TRAINING_HASH")[1],
        "oracle_runner_sha256": _read(oracle_runner_path, "ORACLE_RUNNER_HASH")[1],
        "code_tree_sha256": _code_sha(code_root),
        "deployment_artifacts": records,
    }


def _fold_records(payload: Mapping[str, Any], label: str) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    if isinstance(payload.get("fold_artifacts"), list):
        rows = payload["fold_artifacts"]
    else:
        rows = []
        variants = _mapping(payload.get("variants"), f"{label}_VARIANTS")
        for variant, value in variants.items():
            entry = _mapping(value, f"{label}_{variant}")
            for raw in entry.get("fold_inputs", []):
                row = dict(_mapping(raw, f"{label}_ROW"))
                row["variant"] = variant
                rows.append(row)
    for raw in rows:
        row = _mapping(raw, f"{label}_ROW")
        variant = str(row.get("variant", ""))
        try:
            fold = int(row.get("fold"))
        except (TypeError, ValueError) as exc:
            raise EqualStepGateError(f"EQUAL_STEP_{label}_FOLD_INVALID") from exc
        key = (variant, fold)
        if variant not in VARIANTS or fold not in range(5) or key in result:
            raise EqualStepGateError(f"EQUAL_STEP_{label}_KEY_INVALID={key}")
        if not _sha(row.get("sha256")) or not isinstance(row.get("size_bytes"), int) or row["size_bytes"] <= 0:
            raise EqualStepGateError(f"EQUAL_STEP_{label}_HASH_OR_SIZE_INVALID={key}")
        result[key] = {
            "variant": variant, "fold": fold, "path": str(row.get("path", "")),
            "sha256": row["sha256"], "size_bytes": row["size_bytes"],
        }
    expected = {(variant, fold) for variant in VARIANTS for fold in range(5)}
    if set(result) != expected:
        raise EqualStepGateError(f"EQUAL_STEP_{label}_REQUIRES_EXACT_15")
    return result


def validate_inputs(
    *, project_root: Path, prepared_parent: Path, source_reuse: Path,
    returned_r1: Path,
) -> dict[str, Any]:
    r1_bootstrap = project_root / "runtime/bootstrap" / R1_NAMESPACE
    aborted, aborted_sha, _ = _json(r1_bootstrap / "ABORTED.json", "R1_ABORTED")
    abort_expected = {
        "format": ABORT_FORMAT, "status": ABORT_STATUS,
        "resume_authorized": False, "live_authorizations_removed": True,
        "gpu_training_complete_present": False,
        "formal_result_root_examined": False,
        "formal_result_root_modified": False,
        "formal_result_artifacts_reused": False, "deletion_performed": False,
    }
    for key, expected in abort_expected.items():
        if aborted.get(key) != expected:
            raise EqualStepGateError(f"EQUAL_STEP_R1_ABORTED_DRIFT={key}")
    archived_authorizations = _mapping(
        aborted.get("archived_authorizations"), "R1_ABORTED_AUTHORIZATIONS"
    )
    archived_record = _mapping(
        archived_authorizations.get("static_auth_ready"), "R1_ABORTED_STATIC_RECORD"
    )
    archived = (
        r1_bootstrap /
        R1_ARCHIVE_NAMESPACE / "STATIC_AUTH_READY.live_at_abort.json"
    )
    archived_raw, archived_sha, _ = _read(archived, "R1_STATIC_ARCHIVED")
    returned_raw, returned_sha, _ = _read(returned_r1, "R1_STATIC_RETURNED")
    if archived_sha != EXPECTED_R1_STATIC_R6_SHA256 or returned_sha != archived_sha or returned_raw != archived_raw:
        raise EqualStepGateError("EQUAL_STEP_R1_R6_STATIC_BYTE_AUTHORITY_DRIFT")
    if (
        archived_record.get("sha256") != archived_sha or
        archived_record.get("size_bytes") != len(archived_raw) or
        Path(str(archived_record.get("archive_path", ""))) != archived
    ):
        raise EqualStepGateError("EQUAL_STEP_R1_ABORTED_STATIC_ARCHIVE_BINDING_DRIFT")
    r1 = _mapping(json.loads(archived_raw.decode("utf-8")), "R1_STATIC")
    if r1.get("status") != "STATIC_AUTH_READY" or r1.get("gpu_visible") is not False:
        raise EqualStepGateError("EQUAL_STEP_R1_STATIC_SEMANTIC_DRIFT")
    r1_rows = _fold_records(r1, "R1_STATIC")
    reuse, reuse_sha, _ = _json(source_reuse, "INPUT_REUSE")
    expected_reuse = {
        "format": INPUT_FORMAT, "status": INPUT_STATUS,
        "fold_artifact_count": 15, "reused_existing_prepared_inputs": True,
        "retransfer_performed": False, "copy_performed": False,
        "extraction_performed": False, "formal_result_artifacts_used": False,
        "source_files_modified": False,
        "source_input_archive_sha256": SOURCE_INPUT_ARCHIVE_SHA256,
    }
    for key, expected in expected_reuse.items():
        if reuse.get(key) != expected:
            raise EqualStepGateError(f"EQUAL_STEP_INPUT_REUSE_DRIFT={key}")
    if Path(str(reuse.get("prepared_parent", ""))) != prepared_parent:
        raise EqualStepGateError("EQUAL_STEP_INPUT_REUSE_PARENT_DRIFT")
    reuse_rows = _fold_records(reuse, "INPUT_REUSE")
    fold0 = {}
    total_size = 0
    for key in sorted(r1_rows):
        expected = r1_rows[key]
        observed = reuse_rows[key]
        if observed != expected:
            raise EqualStepGateError(f"EQUAL_STEP_INPUT_REUSE_R1_ROW_DRIFT={key}")
        path = _within(Path(expected["path"]), prepared_parent, "PREPARED_FOLD")
        if path != prepared_parent / expected["variant"] / f"PATIENT_FOLD_{expected['fold']}.pt":
            raise EqualStepGateError(f"EQUAL_STEP_PREPARED_PATH_DRIFT={key}")
        if _stat_size(path, "PREPARED_FOLD") != expected["size_bytes"]:
            raise EqualStepGateError(f"EQUAL_STEP_PREPARED_SIZE_DRIFT={key}")
        total_size += int(expected["size_bytes"])
        if expected["fold"] == FOLD:
            if expected["size_bytes"] != FOLD0_SIZE_ANCHORS[expected["variant"]]:
                raise EqualStepGateError(f"EQUAL_STEP_FOLD0_SIZE_ANCHOR_DRIFT={expected['variant']}")
            fold0[expected["variant"]] = expected
    if fold0["G2"]["sha256"] != G2_F0_SHA256:
        raise EqualStepGateError("EQUAL_STEP_G2_FOLD0_DIRECT_SHA_ANCHOR_DRIFT")
    if reuse.get("fold_artifact_total_bytes") != total_size:
        raise EqualStepGateError("EQUAL_STEP_INPUT_REUSE_TOTAL_SIZE_DRIFT")
    fold0_anchor_sha = hashlib.sha256(
        json.dumps(fold0, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "source_input_reuse_sha256": reuse_sha,
        "r1_aborted_sha256": aborted_sha,
        "r1_static_r6_sha256": archived_sha,
        "fold0": fold0,
        "fold0_triplet_sha256": fold0_anchor_sha,
        "all_15_rows_sha256": hashlib.sha256(
            json.dumps(
                [reuse_rows[key] for key in sorted(reuse_rows)],
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def validate_oracle_pass(receipt_path: Path, execution_log: Path) -> dict[str, Any]:
    receipt_raw, receipt_file_sha, _ = _read(receipt_path, "RETURNED_ORACLE_PASS")
    log_raw, log_sha, _ = _read(execution_log, "SERVER_ORACLE_EXECUTION_LOG")
    terminal_lines: list[bytes] = []
    for line in log_raw.splitlines():
        try:
            item = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(item, Mapping) and item.get("format") == ORACLE_FORMAT and item.get("status") == ORACLE_PASS:
            terminal_lines.append(line)
    if len(terminal_lines) != 1:
        raise EqualStepGateError(f"EQUAL_STEP_ORACLE_PASS_CARDINALITY={len(terminal_lines)}")
    standalone = receipt_raw.strip()
    if standalone != terminal_lines[0].strip():
        raise EqualStepGateError("EQUAL_STEP_RETURNED_ORACLE_PASS_NOT_EXACT_LOG_LINE")
    try:
        receipt = _mapping(json.loads(standalone.decode("utf-8")), "ORACLE_PASS")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EqualStepGateError("EQUAL_STEP_ORACLE_PASS_JSON_INVALID") from exc
    expected = {
        "format": ORACLE_FORMAT, "status": ORACLE_PASS,
        "run_id": ORACLE_RUN_ID, "task_id": ORACLE_TASK_ID,
        "graph_variant": "G2", "patient_fold": FOLD, "seed": SEED,
        "scientific_pass": True, "formal_artifacts_written": 0,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise EqualStepGateError(f"EQUAL_STEP_ORACLE_PASS_SEMANTIC_DRIFT={key}")
    for key in (
        "formal_training_authorized", "checkpoint_written", "success_json_written",
        "failure_json_written", "prediction_written", "winner_selection_input",
    ):
        if receipt.get(key) is not False:
            raise EqualStepGateError(f"EQUAL_STEP_ORACLE_PASS_UNSAFE_FLAG={key}")
    real_execution_expected = {
        "validation_payload_schema_validated": True,
        "architecture_id": "HHGT_FORMAL_CORE_EXTERNAL_ROUTER",
        "prepared_artifact_sha256": G2_F0_SHA256,
        "candidate_batch_count": 403,
        "comparison_batch_indices": [0, 1, 2, 3],
        "comparison_batch_row_counts": [8192, 8192, 8192, 8192],
        "comparison_group_rows": 32768,
        "canonical_offset_order": list(range(29)),
        "training_batches_used": 4,
        "training_batch_indices_used": [0, 1, 2, 3],
        "oof_modality_lineage_preserved": True,
        "patient_fold_authority_preserved": True,
        "optimizer_boundary_preserved": True,
        "immutable_batch_composition_preserved": True,
    }
    for key, value in real_execution_expected.items():
        if receipt.get(key) != value:
            raise EqualStepGateError(
                f"EQUAL_STEP_ORACLE_REAL_EXECUTION_DRIFT={key}"
            )
    runtime_chunks = receipt.get("runtime_chunks")
    permutation = receipt.get("runtime_chunk_permutation")
    if (
        not isinstance(runtime_chunks, list)
        or len(runtime_chunks) != 29
        or len(set(runtime_chunks)) != 29
        or not isinstance(permutation, list)
        or len(permutation) != 29
        or set(permutation) != set(runtime_chunks)
    ):
        raise EqualStepGateError("EQUAL_STEP_ORACLE_REAL_K29_DRIFT")
    for key in (
        "preregistration_sha256", "oracle_runner_sha256",
        "production_training_sha256", "comparison_group_input_sha256",
        "assignment_sha256", "theta_0_model_state_sha256",
        "theta_0_rng_sha256", "theta_1_model_state_sha256",
        "theta_1_rng_sha256",
    ):
        if not _sha(receipt.get(key)):
            raise EqualStepGateError(f"EQUAL_STEP_ORACLE_REAL_SHA_INVALID={key}")
    theta_1 = _mapping(receipt.get("theta_1_construction"), "ORACLE_THETA_1")
    if (
        theta_1.get("optimizer_steps") != 1
        or theta_1.get("grad_finite") is not True
        or theta_1.get("parameters_finite") is not True
        or theta_1.get("parameter_delta_positive") is not True
    ):
        raise EqualStepGateError("EQUAL_STEP_ORACLE_THETA_1_UPDATE_DRIFT")
    theta_results = _mapping(receipt.get("theta_results"), "ORACLE_THETA_RESULTS")
    if set(theta_results) != {"theta_0", "theta_1"}:
        raise EqualStepGateError("EQUAL_STEP_ORACLE_THETA_RESULT_SET_DRIFT")
    for theta_name in ("theta_0", "theta_1"):
        theta = _mapping(theta_results.get(theta_name), f"ORACLE_{theta_name}")
        branch = _mapping(theta.get("branch_gate"), f"ORACLE_{theta_name}_BRANCH")
        components = _mapping(theta.get("components"), f"ORACLE_{theta_name}_COMPONENTS")
        complete_gradient = _mapping(
            theta.get("complete_mean_gradient"),
            f"ORACLE_{theta_name}_COMPLETE_GRADIENT",
        )
        modules = _mapping(
            theta.get("module_mean_gradients"), f"ORACLE_{theta_name}_MODULES"
        )
        if (
            theta.get("theta") != theta_name
            or theta.get("pass") is not True
            or theta.get("scientific_fail_reasons") != []
            or branch.get("evaluations") != 58
            or branch.get("paired_disagreement_count") != 0
            or branch.get("constant_across_all_58") is not True
            or branch.get("pass") is not True
            or set(components) != {
                "membership_risk", "direction_loss", "shrinkage_penalty",
                "total_objective",
            }
            or any(
                not isinstance(value, Mapping) or value.get("pass") is not True
                for value in components.values()
            )
            or complete_gradient.get("pass") is not True
            or not modules
            or any(
                not isinstance(value, Mapping) or value.get("pass") is not True
                for value in modules.values()
            )
        ):
            raise EqualStepGateError(
                f"EQUAL_STEP_ORACLE_SCIENTIFIC_GATE_DRIFT={theta_name}"
            )
    gpu_identity = _mapping(receipt.get("gpu_identity"), "ORACLE_GPU_IDENTITY")
    if not gpu_identity or _number(receipt.get("peak_reserved_bytes"), "ORACLE_PEAK_RESERVED") <= 0:
        raise EqualStepGateError("EQUAL_STEP_ORACLE_REAL_GPU_EVIDENCE_MISSING")
    if _number(receipt.get("elapsed_seconds"), "ORACLE_ELAPSED") <= 0:
        raise EqualStepGateError("EQUAL_STEP_ORACLE_REAL_ELAPSED_INVALID")
    return {
        "returned_receipt_file_sha256": receipt_file_sha,
        "server_execution_log_sha256": log_sha,
        "terminal_line_sha256": hashlib.sha256(terminal_lines[0]).hexdigest(),
        "preregistration_sha256": receipt["preregistration_sha256"],
        "oracle_runner_sha256": receipt["oracle_runner_sha256"],
        "production_training_sha256": receipt["production_training_sha256"],
        "prepared_artifact_sha256": receipt["prepared_artifact_sha256"],
    }


def validate_jit(path: Path, now: datetime | None = None) -> dict[str, Any]:
    payload, digest, _ = _json(path, "JIT_BUDGET")
    expected = {
        "format": JIT_FORMAT, "status": JIT_STATUS, "currency": "CNY",
        "source": "COMPSHARE_API_BILLING_SNAPSHOT",
        "project_budget_cap_cny": PROJECT_BUDGET_CAP_CNY,
        "compute_hourly_cny": COMPUTE_HOURLY_CNY,
        "gpu_hourly_cny": COMPUTE_HOURLY_CNY,
        "disk_hourly_cny": DISK_HOURLY_CNY,
        "total_hourly_cny": TOTAL_HOURLY_CNY,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise EqualStepGateError(f"EQUAL_STEP_JIT_DRIFT={key}")
    instance = payload.get("instance_id")
    if not isinstance(instance, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", instance) is None:
        raise EqualStepGateError("EQUAL_STEP_JIT_INSTANCE_INVALID")
    queried = _parse_time(payload.get("queried_at"), "JIT_QUERIED")
    valid_until = _parse_time(payload.get("valid_until"), "JIT_VALID_UNTIL")
    clock = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if queried > clock + JIT_SKEW or valid_until <= clock:
        raise EqualStepGateError("EQUAL_STEP_JIT_STALE")
    if valid_until <= queried or valid_until - queried > MAX_JIT_VALIDITY:
        raise EqualStepGateError("EQUAL_STEP_JIT_VALIDITY_WINDOW")
    remaining = _number(payload.get("conservative_remaining_cny"), "JIT_REMAINING")
    if not 0 < remaining <= PROJECT_BUDGET_CAP_CNY:
        raise EqualStepGateError("EQUAL_STEP_JIT_REMAINING_RANGE")
    if payload.get("budget_method") != (
        "FULL_2_09_CNY_RATE_FOR_EVERY_SECOND_SINCE_INSTANCE_CREATE_"
        "MINUS_FUTURE_RESERVE"
    ):
        raise EqualStepGateError("EQUAL_STEP_JIT_BUDGET_METHOD_DRIFT")
    age = _number(payload.get("instance_age_seconds"), "JIT_INSTANCE_AGE")
    worst = _number(payload.get("worst_case_full_rate_spend_cny"), "JIT_WORST_SPEND")
    reserve = _number(payload.get("future_cleanup_reserve_cny"), "JIT_RESERVE")
    if age < 0 or reserve < 5.0 or reserve >= PROJECT_BUDGET_CAP_CNY:
        raise EqualStepGateError("EQUAL_STEP_JIT_CONSERVATIVE_COMPONENT_RANGE")
    expected_worst = round(age * TOTAL_HOURLY_CNY / 3600.0, 4)
    expected_remaining = round(
        PROJECT_BUDGET_CAP_CNY - expected_worst - reserve, 4
    )
    if (
        abs(worst - expected_worst) > 0.0001
        or abs(remaining - expected_remaining) > 0.0001
    ):
        raise EqualStepGateError("EQUAL_STEP_JIT_CONSERVATIVE_FORMULA_DRIFT")
    if not _sha(payload.get("provider_snapshot_sha256")) or not _sha(
        payload.get("generator_sha256")
    ):
        raise EqualStepGateError("EQUAL_STEP_JIT_PROVENANCE_SHA_INVALID")
    return {
        "sha256": digest, "instance_id": instance, "queried_at": queried.isoformat(),
        "valid_until": valid_until.isoformat(), "remaining_cny": remaining,
        "effective_cost_cap_cny": min(MAX_COST_CNY, remaining),
        "budget_method": payload["budget_method"],
        "future_cleanup_reserve_cny": reserve,
    }


def validate_deployment(
    path: Path, *, control_root: Path, expected_sha256: str,
    external_verifier_sha256: str,
) -> dict[str, Any]:
    manifest, digest, _ = _json(_within(path, control_root, "DEPLOYMENT_MANIFEST"), "DEPLOYMENT_MANIFEST")
    if digest != expected_sha256:
        raise EqualStepGateError("EQUAL_STEP_DEPLOYMENT_MANIFEST_SHA_DRIFT")
    if manifest.get("format") != DEPLOYMENT_FORMAT or manifest.get("status") != DEPLOYMENT_STATUS or manifest.get("namespace") != NAMESPACE:
        raise EqualStepGateError("EQUAL_STEP_DEPLOYMENT_MANIFEST_SEMANTIC_DRIFT")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or [item.get("name") for item in artifacts if isinstance(item, Mapping)] != list(CONTROL_NAMES):
        raise EqualStepGateError("EQUAL_STEP_DEPLOYMENT_ARTIFACT_SET_DRIFT")
    records = []
    for item in artifacts:
        row = _mapping(item, "DEPLOYMENT_ROW")
        name = str(row.get("name", ""))
        artifact = _within(control_root / name, control_root, "CONTROL_ARTIFACT")
        _, observed_sha, observed_size = _read(artifact, "CONTROL_ARTIFACT")
        if row.get("sha256") != observed_sha or row.get("size_bytes") != observed_size or row.get("server_relative_path") != name:
            raise EqualStepGateError(f"EQUAL_STEP_CONTROL_ARTIFACT_DRIFT={name}")
        records.append(dict(row))
    verifier_row = next(row for row in records if row["name"] == VALIDATOR_NAME)
    if verifier_row["sha256"] != external_verifier_sha256:
        raise EqualStepGateError("EQUAL_STEP_EXTERNAL_VERIFIER_INDEPENDENT_SHA_DRIFT")
    return {"sha256": digest, "artifacts": records}


def validate_code_receipt(path: Path, *, code_root: Path, bootstrap_root: Path, overlay: Mapping[str, Any]) -> dict[str, Any]:
    payload, receipt_sha, _ = _json(path, "CODE_ARCHIVE_RECEIPT")
    expected = {
        "format": CODE_FORMAT, "status": CODE_STATUS, "namespace": NAMESPACE,
        "cpu_only": True, "gpu_visible": False,
        "code_root": str(Path(os.path.abspath(code_root))),
        "code_tree_sha256": overlay["code_tree_sha256"],
        "pilot_sha256": overlay["pilot_sha256"],
        "bootstrap_sha256": overlay["bootstrap_sha256"],
        "preregistration_sha256": overlay["preregistration_sha256"],
        "decision_sha256": overlay["decision_sha256"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise EqualStepGateError(f"EQUAL_STEP_CODE_RECEIPT_DRIFT={key}")
    archive = _within(Path(str(payload.get("archive_path", ""))), bootstrap_root, "CODE_ARCHIVE")
    _, archive_sha, archive_size = _read(archive, "CODE_ARCHIVE")
    if payload.get("archive_sha256") != archive_sha or payload.get("archive_size_bytes") != archive_size:
        raise EqualStepGateError("EQUAL_STEP_CODE_ARCHIVE_BYTE_DRIFT")
    return {"receipt_sha256": receipt_sha, "archive_sha256": archive_sha}


def materialize_code_receipt(
    *, code_root: Path, bootstrap_root: Path, template_path: Path,
    code_archive: Path, output: Path,
) -> dict[str, Any]:
    overlay = validate_overlay(code_root, template_path)
    archive = _within(code_archive, bootstrap_root, "CODE_ARCHIVE")
    _, archive_sha, archive_size = _read(archive, "CODE_ARCHIVE")
    payload = {
        "format": CODE_FORMAT, "status": CODE_STATUS, "namespace": NAMESPACE,
        "cpu_only": True, "gpu_visible": False,
        "archive_path": str(archive), "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "code_root": str(Path(os.path.abspath(code_root))),
        "code_tree_sha256": overlay["code_tree_sha256"],
        "pilot_sha256": overlay["pilot_sha256"],
        "bootstrap_sha256": overlay["bootstrap_sha256"],
        "preregistration_sha256": overlay["preregistration_sha256"],
        "decision_sha256": overlay["decision_sha256"],
    }
    _atomic(output, _canonical(payload))
    return payload


def _render_config(template: Mapping[str, Any], variant: str, prepared_parent: Path) -> dict[str, Any]:
    rendered = json.loads(json.dumps(template))
    rendered["equal_step_contract"]["graph_variant"] = variant
    rendered["task_contract"]["graph_variant"] = variant
    rendered["analysis_version"] = f"CancerLncAtlas_V3.2_{variant}_EQUAL_STEP_CANDIDATE_R1"
    rendered["training_io"]["prepared_fold_pattern"] = str(
        prepared_parent / variant / "PATIENT_FOLD_{fold}.pt"
    )
    return rendered


def _task_bytes(variant: str) -> bytes:
    run_id = f"{PILOT_ID}-{variant.lower()}"
    task_id = f"{run_id}|PATIENT_FOLD_0|CC-HHGT|{SEED}"
    columns = ("task_id", "run_id", "task_type", "model", "patient_fold", "seed", "owner", "hardware_class", "status", "blocked_reason", "paid_task")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerow({
        "task_id": task_id, "run_id": run_id, "task_type": "CC_HHGT_PATIENT_FOLD",
        "model": "CC-HHGT", "patient_fold": FOLD, "seed": SEED,
        "owner": ENDPOINT, "hardware_class": HARDWARE, "status": "PENDING",
        "blocked_reason": "", "paid_task": "true",
    })
    return output.getvalue().encode("utf-8")


def _approval(*, code_sha: str, config_sha: str, input_sha: str, task_sha: str, variant: str) -> dict[str, Any]:
    run_id = f"{PILOT_ID}-{variant.lower()}"
    task_id = f"{run_id}|PATIENT_FOLD_0|CC-HHGT|{SEED}"
    return {
        "approval_format": APPROVAL_FORMAT, "training_authorized": True,
        "formal_training_authorized": False, "comparison_only": True,
        "candidate_only": True, "artifact_class": "CANDIDATE_ONLY_EQUAL_STEP_COMPARISON",
        "authorized_trainer": TRAINER, "run_id": run_id,
        "endpoint_id": ENDPOINT, "hardware_class": HARDWARE,
        "approved_task_ids": [task_id],
        "artifact_hashes": {
            "code_sha256": code_sha, "config_sha256": config_sha,
            "input_manifest_sha256": input_sha, "task_manifest_sha256": task_sha,
        },
        "paid_enabled": True, "max_paid_hours": MAX_HOURS,
        "max_cost_cny": MAX_COST_CNY,
        "this_receipt_does_not_authorize_formal_training": True,
        "this_receipt_does_not_authorize_budget_spend_beyond_this_pilot": True,
    }


def materialize(
    *, project_root: Path, code_root: Path, bootstrap_root: Path,
    prepared_parent: Path, template_path: Path, source_reuse: Path,
    returned_r1: Path, oracle_pass: Path, oracle_log: Path, jit_path: Path,
    code_receipt: Path, deployment_manifest: Path,
    deployment_manifest_sha256: str, external_verifier_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    if bootstrap_root.name != NAMESPACE:
        raise EqualStepGateError("EQUAL_STEP_BOOTSTRAP_NAMESPACE_DRIFT")
    existing_created: datetime | None = None
    static_path = bootstrap_root / "STATIC_AUTH_READY.json"
    if static_path.exists():
        existing, _, _ = _json(static_path, "EXISTING_STATIC_AUTH")
        existing_created = _parse_time(existing.get("created_at"), "STATIC_CREATED_AT")
    overlay = validate_overlay(code_root, template_path)
    inputs = validate_inputs(project_root=project_root, prepared_parent=prepared_parent, source_reuse=source_reuse, returned_r1=returned_r1)
    oracle = validate_oracle_pass(oracle_pass, oracle_log)
    oracle_code_expected = {
        "preregistration_sha256": overlay["preregistration_sha256"],
        "oracle_runner_sha256": overlay["oracle_runner_sha256"],
        "production_training_sha256": overlay["production_training_sha256"],
        "prepared_artifact_sha256": inputs["fold0"]["G2"]["sha256"],
    }
    for key, expected in oracle_code_expected.items():
        if oracle.get(key) != expected:
            raise EqualStepGateError(
                f"EQUAL_STEP_ORACLE_CURRENT_OVERLAY_BINDING_DRIFT={key}"
            )
    jit = validate_jit(jit_path, now=now)
    control_root = project_root / "runtime/equal_step_gates" / NAMESPACE
    deployment = validate_deployment(
        deployment_manifest, control_root=control_root,
        expected_sha256=deployment_manifest_sha256,
        external_verifier_sha256=external_verifier_sha256,
    )
    code = validate_code_receipt(code_receipt, code_root=code_root, bootstrap_root=bootstrap_root, overlay=overlay)
    auth_root = bootstrap_root / "authorization"
    component_records = []
    for variant in VARIANTS:
        root = auth_root / variant.lower()
        config_path = root / "config.json"
        input_path = root / "INPUT_MANIFEST.json"
        task_path = root / "TASK_MANIFEST.tsv"
        approval_path = root / "TRAINER_APPROVAL.json"
        config_bytes = _canonical(_render_config(overlay["template"], variant, prepared_parent))
        input_bytes = _canonical({
            "format": "CC_HHGT_V3_2_EQUAL_STEP_COMPONENT_INPUT_V1",
            "comparison_only": True, "candidate_only": True,
            "source_input_reuse_receipt_sha256": inputs["source_input_reuse_sha256"],
            "source_r1_static_auth_r6_sha256": inputs["r1_static_r6_sha256"],
            "fold_inputs": [inputs["fold0"][variant]],
        })
        task_bytes = _task_bytes(variant)
        hashes = {
            "code_sha256": overlay["code_tree_sha256"],
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "input_manifest_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "task_manifest_sha256": hashlib.sha256(task_bytes).hexdigest(),
        }
        approval_bytes = _canonical(_approval(
            code_sha=hashes["code_sha256"], config_sha=hashes["config_sha256"],
            input_sha=hashes["input_manifest_sha256"], task_sha=hashes["task_manifest_sha256"], variant=variant,
        ))
        _atomic(config_path, config_bytes)
        _atomic(input_path, input_bytes)
        _atomic(task_path, task_bytes)
        _atomic(approval_path, approval_bytes)
        component_records.append({
            "variant": variant, "run_id": f"{PILOT_ID}-{variant.lower()}",
            "task_id": f"{PILOT_ID}-{variant.lower()}|PATIENT_FOLD_0|CC-HHGT|{SEED}",
            "config_path": str(config_path), "input_manifest_path": str(input_path),
            "task_manifest_path": str(task_path), "approval_path": str(approval_path),
            "artifact_hashes": hashes,
            "approval_sha256": hashlib.sha256(approval_bytes).hexdigest(),
        })
    created = (
        existing_created or now or datetime.now(timezone.utc)
    ).astimezone(timezone.utc)
    receipt = {
        "format": STATIC_FORMAT, "status": STATIC_STATUS, "namespace": NAMESPACE,
        "pilot_id": PILOT_ID, "job_id": JOB_ID, "created_at": created.isoformat(),
        "cpu_only_materialized": True, "gpu_visible_during_materialization": False,
        "comparison_only": True, "candidate_only": True,
        "formal_training_authorized": False, "budget_authorized_by_this_receipt": False,
        "variants": list(VARIANTS), "patient_fold": FOLD, "seed": SEED,
        "trainer": TRAINER, "component_authorizations": component_records,
        "r1_static_auth_r6_sha256": inputs["r1_static_r6_sha256"],
        "r1_aborted_sha256": inputs["r1_aborted_sha256"],
        "source_input_reuse_receipt_sha256": inputs["source_input_reuse_sha256"],
        "all_15_input_rows_sha256": inputs["all_15_rows_sha256"],
        "fold0_triplet_sha256": inputs["fold0_triplet_sha256"],
        "fold0_anchors": inputs["fold0"],
        "oracle_pass_authority": oracle,
        "pilot_sha256": overlay["pilot_sha256"],
        "bootstrap_sha256": overlay["bootstrap_sha256"],
        "preregistration_sha256": overlay["preregistration_sha256"],
        "decision_sha256": overlay["decision_sha256"],
        "production_training_sha256": overlay["production_training_sha256"],
        "code_tree_sha256": overlay["code_tree_sha256"],
        "code_archive_receipt_sha256": code["receipt_sha256"],
        "code_archive_sha256": code["archive_sha256"],
        "external_verifier_sha256": external_verifier_sha256,
        "external_deployment_manifest_sha256": deployment["sha256"],
        "external_control_artifacts": deployment["artifacts"],
        "jit_budget_receipt_sha256": jit["sha256"], "instance_id": jit["instance_id"],
        "max_paid_hours": MAX_HOURS, "configured_cost_cap_cny": MAX_COST_CNY,
        "jit_conservative_remaining_cny": jit["remaining_cny"],
        "jit_effective_cost_cap_cny": jit["effective_cost_cap_cny"],
        "jit_budget_method": jit["budget_method"],
        "jit_future_cleanup_reserve_cny": jit["future_cleanup_reserve_cny"],
        "compute_hourly_cny": COMPUTE_HOURLY_CNY,
        "disk_hourly_cny": DISK_HOURLY_CNY, "total_hourly_cny": TOTAL_HOURLY_CNY,
        "formal_artifacts_allowed": False, "checkpoint_allowed": False,
        "prediction_allowed": False, "winner_selection_allowed": False,
        "formal_gate_authorized_by_terminal_receipt": False,
    }
    encoded = json.dumps(receipt, sort_keys=True)
    if "/results/" in encoded or "v32_g012_paid_gpu_training_" in encoded:
        raise EqualStepGateError("EQUAL_STEP_FORMAL_RESULT_REFERENCE_FORBIDDEN")
    _atomic(static_path, _canonical(receipt))
    return receipt


def validate_terminal_log(path: Path, expected_exit_code: int) -> dict[str, Any]:
    raw, _, _ = _read(path, "TERMINAL_LOG")
    receipts = []
    for line in raw.decode("utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, Mapping) and item.get("schema") == TERMINAL_SCHEMA and item.get("status") in ({PASS_STATUS} | FAIL_STATUSES):
            receipts.append(dict(item))
    if len(receipts) != 1:
        raise EqualStepGateError(f"EQUAL_STEP_TERMINAL_CARDINALITY={len(receipts)}")
    receipt = receipts[0]
    expected_statuses = {PASS_STATUS} if expected_exit_code == 0 else FAIL_STATUSES
    if expected_exit_code not in {0, 42} or receipt.get("status") not in expected_statuses:
        raise EqualStepGateError("EQUAL_STEP_TERMINAL_EXIT_STATUS_DRIFT")
    expected = {
        "pilot_id": PILOT_ID, "patient_fold": FOLD, "seed": SEED,
        "comparison_only": True, "candidate_only": True,
        "formal_training_authorized_by_this_receipt": False,
        "formal_artifacts_written": 0, "formal_checkpoint_written": False,
        "formal_success_marker_written": False, "formal_failure_marker_written": False,
        "formal_prediction_written": False, "test_or_outer_artifact_written": False,
        "winner_selection_input": False, "output_channel": "STDOUT_JSON_ONLY",
        "test_rows_read": 0, "test_labels_read": False,
        "outer_metrics_read": False, "outer_predictions_read": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise EqualStepGateError(f"EQUAL_STEP_TERMINAL_SAFETY_DRIFT={key}")
    if expected_exit_code == 0:
        if receipt.get("scientific_pass") is not True or receipt.get("graph_variants") != list(VARIANTS):
            raise EqualStepGateError("EQUAL_STEP_TERMINAL_PASS_SEMANTIC_DRIFT")
    elif receipt.get("scientific_pass") is not False:
        raise EqualStepGateError("EQUAL_STEP_TERMINAL_FAIL_SEMANTIC_DRIFT")
    return receipt


def _scientific_pass_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if receipt.get("status") != PASS_STATUS or receipt.get("scientific_pass") is not True:
        raise EqualStepGateError("EQUAL_STEP_HANDOFF_REQUIRES_TERMINAL_PASS")
    if receipt.get("per_variant_pass") != {variant: True for variant in VARIANTS}:
        raise EqualStepGateError("EQUAL_STEP_HANDOFF_PER_VARIANT_PASS_DRIFT")
    if receipt.get("cross_variant_pass") is not True:
        raise EqualStepGateError("EQUAL_STEP_HANDOFF_CROSS_VARIANT_PASS_DRIFT")
    cross = _mapping(receipt.get("cross_variant_gates"), "HANDOFF_CROSS_VARIANT_GATES")
    if not cross or any(value is not True for value in cross.values()):
        raise EqualStepGateError("EQUAL_STEP_HANDOFF_CROSS_VARIANT_GATE_FALSE")
    variants = _mapping(receipt.get("variants"), "HANDOFF_VARIANTS")
    if tuple(sorted(variants)) != VARIANTS:
        raise EqualStepGateError("EQUAL_STEP_HANDOFF_VARIANT_SET_DRIFT")
    exact_cross_fields = (
        "architecture_id", "initial_model_sha256", "initial_rng_sha256",
        "candidate_batch_row_counts_sha256", "candidate_rows",
        "ordered_train_candidate_key_sha256", "ordered_train_label_sha256",
        "ordered_train_model_input_sha256", "ordered_train_mask_weight_sha256",
        "training_loss_payload_sha256", "ordered_train_identity_sha256",
        "runtime_chunks", "runtime_chunk_permutation_sha256",
        "precision_contract_sha256",
        "ordered_validation_candidate_key_sha256", "ordered_validation_label_sha256",
        "ordered_validation_model_input_sha256", "ordered_validation_mask_weight_sha256",
        "validation_loss_payload_sha256", "validation_loss_contract_sha256",
        "ordered_validation_identity_sha256", "validation_fold_role",
        "validation_candidate_rows", "optimization_contract_sha256",
    )
    expected_cross_gate_names = set(exact_cross_fields) | {
        "reference_schedule_byte_identical",
        "reference_chunk_weights_byte_identical",
        "proposed_schedule_byte_identical",
        "proposed_chunk_weights_byte_identical",
    }
    if set(cross) != expected_cross_gate_names:
        raise EqualStepGateError(
            "EQUAL_STEP_HANDOFF_CROSS_VARIANT_GATE_SET_DRIFT"
        )
    first_values: dict[str, Any] | None = None
    identities = []
    for variant in VARIANTS:
        item = _mapping(variants.get(variant), f"HANDOFF_{variant}")
        run_id = f"{PILOT_ID}-{variant.lower()}"
        task_id = f"{run_id}|PATIENT_FOLD_0|CC-HHGT|{SEED}"
        expected = {
            "graph_variant": variant, "authorization_run_id": run_id,
            "authorization_task_id": task_id, "authorization_endpoint_id": ENDPOINT,
            "authorization_hardware_class": HARDWARE, "patient_fold": FOLD,
            "seed": SEED, "candidate_batch_count": 403,
            "optimizer_group_count": 101, "variant_scientific_pass": True,
        }
        for key, value in expected.items():
            if item.get(key) != value:
                raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_{variant}_IDENTITY_DRIFT={key}")
        if not isinstance(item.get("runtime_chunks"), list) or len(item["runtime_chunks"]) != 29 or len(set(item["runtime_chunks"])) != 29:
            raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_{variant}_K29_DRIFT")
        for key in ("prepared_artifact_sha256", "initial_model_sha256", "initial_rng_sha256") + exact_cross_fields[3:]:
            if key.endswith("sha256") and not _sha(item.get(key)):
                raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_{variant}_SHA_INVALID={key}")
        artifact_hashes = _mapping(
            item.get("authorization_artifact_hashes"),
            f"HANDOFF_{variant}_AUTHORIZATION_HASHES",
        )
        if set(artifact_hashes) != {
            "code_sha256", "config_sha256", "input_manifest_sha256",
            "task_manifest_sha256",
        } or any(not _sha(value) for value in artifact_hashes.values()):
            raise EqualStepGateError(
                f"EQUAL_STEP_HANDOFF_{variant}_AUTHORIZATION_HASH_DRIFT"
            )
        execution = _mapping(item.get("execution_gates"), f"HANDOFF_{variant}_EXECUTION")
        required_execution = {
            "reference_optimizer_steps_exact_101", "proposed_optimizer_steps_exact_101",
            "reference_exact_29_chunk_validation", "proposed_exact_29_chunk_validation",
            "same_initial_model", "same_initial_rng",
        }
        if set(execution) != required_execution or any(execution[key] is not True for key in required_execution):
            raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_{variant}_EXECUTION_GATE_DRIFT")
        arms = {}
        for arm_name in ("reference", "proposed"):
            arm = _mapping(item.get(arm_name), f"HANDOFF_{variant}_{arm_name}")
            telemetry = _mapping(arm.get("training_call_telemetry"), f"HANDOFF_{variant}_{arm_name}_TELEMETRY")
            if telemetry.get("optimizer_steps") != 101 or arm.get("validation_runtime_chunks") != 29:
                raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_{variant}_{arm_name}_101_K29_DRIFT")
            if arm.get("initial_model_sha256") != item.get("initial_model_sha256") or arm.get("initial_rng_sha256") != item.get("initial_rng_sha256"):
                raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_{variant}_{arm_name}_INIT_RNG_DRIFT")
            if not _sha(arm.get("final_model_sha256")) or not _sha(arm.get("final_rng_sha256")):
                raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_{variant}_{arm_name}_FINAL_SHA_INVALID")
            if not _sha(arm.get("schedule_sha256")) or not _sha(
                arm.get("runtime_chunk_weights_sha256")
            ):
                raise EqualStepGateError(
                    f"EQUAL_STEP_HANDOFF_{variant}_{arm_name}_SCHEDULE_SHA_INVALID"
                )
            arms[arm_name] = {
                "optimizer_steps": telemetry["optimizer_steps"],
                "validation_runtime_chunks": arm["validation_runtime_chunks"],
                "initial_model_sha256": arm["initial_model_sha256"],
                "initial_rng_sha256": arm["initial_rng_sha256"],
                "final_model_sha256": arm.get("final_model_sha256"),
                "final_rng_sha256": arm.get("final_rng_sha256"),
            }
        comparison = _mapping(item.get("comparison"), f"HANDOFF_{variant}_COMPARISON")
        gates = _mapping(comparison.get("gates"), f"HANDOFF_{variant}_SCIENTIFIC_GATES")
        thresholds = _mapping(comparison.get("thresholds"), f"HANDOFF_{variant}_THRESHOLDS")
        required_gates = {
            "validation_logloss_absolute_difference",
            "mean_logit_pearson", "mean_logit_spearman",
            "ordered_candidate_keys_labels_fold_role_chunk_weights_exact",
        }
        if comparison.get("pass") is not True or set(gates) != required_gates or any(gates[key] is not True for key in required_gates):
            raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_{variant}_SCIENTIFIC_GATE_DRIFT")
        exact_field_gates = _mapping(
            comparison.get("exact_identity_field_gates"),
            f"HANDOFF_{variant}_EXACT_IDENTITY_FIELD_GATES",
        )
        expected_exact_field_gates = {
            "ordered_candidate_key_sha256", "ordered_label_sha256",
            "ordered_identity_sha256", "ordered_model_input_sha256",
            "ordered_mask_weight_sha256", "validation_loss_payload_sha256",
            "validation_loss_contract_sha256", "precision_contract_sha256",
            "fold_role", "runtime_chunk_weights",
            "runtime_chunk_weights_sha256", "candidate_rows",
        }
        if set(exact_field_gates) != expected_exact_field_gates or any(value is not True for value in exact_field_gates.values()) or comparison.get("proxy_label_tensor_exact") is not True:
            raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_{variant}_EXACT_IDENTITY_GATE_DRIFT")
        if thresholds != {
            "validation_logloss_absolute_difference_max": 0.002,
            "mean_logit_pearson_min": 0.995,
            "mean_logit_spearman_min": 0.995,
        }:
            raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_{variant}_THRESHOLD_DRIFT")
        difference = _number(comparison.get("validation_logloss_absolute_difference"), f"{variant}_LOGLOSS_DIFF")
        reference_logloss = _number(
            comparison.get("reference_validation_logloss"),
            f"{variant}_REFERENCE_LOGLOSS",
        )
        proposed_logloss = _number(
            comparison.get("proposed_validation_logloss"),
            f"{variant}_PROPOSED_LOGLOSS",
        )
        pearson = _number(comparison.get("mean_logit_pearson"), f"{variant}_PEARSON")
        spearman = _number(comparison.get("mean_logit_spearman"), f"{variant}_SPEARMAN")
        if (
            difference != abs(reference_logloss - proposed_logloss)
            or difference > 0.002
            or pearson < 0.995
            or spearman < 0.995
        ):
            raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_{variant}_NUMERIC_GATE_DRIFT")
        values = {key: item.get(key) for key in exact_cross_fields}
        if first_values is None:
            first_values = values
        elif values != first_values:
            raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_CROSS_VARIANT_EXACT_IDENTITY_DRIFT={variant}")
        identities.append({
            "variant": variant, "run_id": run_id, "task_id": task_id,
            "prepared_artifact_sha256": item["prepared_artifact_sha256"],
            "initial_model_sha256": item["initial_model_sha256"],
            "initial_rng_sha256": item["initial_rng_sha256"],
            "ordered_train_identity_sha256": item["ordered_train_identity_sha256"],
            "ordered_validation_identity_sha256": item["ordered_validation_identity_sha256"],
            "runtime_chunk_permutation_sha256": item["runtime_chunk_permutation_sha256"],
            "authorization_artifact_hashes": item.get("authorization_artifact_hashes"),
            "arms": arms,
            "validation_logloss_absolute_difference": difference,
            "mean_logit_pearson": pearson, "mean_logit_spearman": spearman,
        })
    return {
        "variants": identities,
        "same_initial_model_cross_variant": len({item["initial_model_sha256"] for item in identities}) == 1,
        "same_initial_rng_cross_variant": len({item["initial_rng_sha256"] for item in identities}) == 1,
        "cross_variant_gates": cross,
    }


def materialize_formal_gate_handoff(
    *, static_path: Path, jit_path: Path, terminal_log: Path,
    autostop_path: Path, decision_path: Path, deployment_path: Path,
    output: Path,
) -> dict[str, Any]:
    static, static_sha, _ = _json(static_path, "HANDOFF_STATIC")
    if static.get("format") != STATIC_FORMAT or static.get("status") != STATIC_STATUS or static.get("formal_training_authorized") is not False:
        raise EqualStepGateError("EQUAL_STEP_HANDOFF_STATIC_DRIFT")
    jit, jit_sha, _ = _json(jit_path, "HANDOFF_JIT")
    if jit.get("format") != JIT_FORMAT or jit.get("status") != JIT_STATUS or static.get("jit_budget_receipt_sha256") != jit_sha:
        raise EqualStepGateError("EQUAL_STEP_HANDOFF_JIT_DRIFT")
    terminal = validate_terminal_log(terminal_log, 0)
    log_raw, log_sha, _ = _read(terminal_log, "HANDOFF_TERMINAL_LOG")
    terminal_lines = []
    for line in log_raw.splitlines():
        try:
            item = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(item, Mapping) and item.get("schema") == TERMINAL_SCHEMA and item.get("status") == PASS_STATUS:
            terminal_lines.append(line)
    if len(terminal_lines) != 1:
        raise EqualStepGateError("EQUAL_STEP_HANDOFF_TERMINAL_LINE_CARDINALITY")
    terminal_line_sha = hashlib.sha256(terminal_lines[0]).hexdigest()
    autostop, autostop_sha, _ = _json(autostop_path, "HANDOFF_AUTOSTOP")
    autostop_expected = {
        "format": "CANCERLNCATLAS_EQUAL_STEP_CANDIDATE_AUTOSTOP_V1",
        "status": "INSTANCE_STOPPED_STATE_CONFIRMED", "namespace": NAMESPACE,
        "stop_trigger": "EQUAL_STEP_PASS_OBSERVED_IMMEDIATE_STOP",
        "instance_id": static.get("instance_id"), "job_id": JOB_ID,
        "provider_state": "Stopped", "gpu_billing_active": False,
        "static_auth_sha256": static_sha, "jit_budget_receipt_sha256": jit_sha,
        "terminal_line_sha256": terminal_line_sha,
        "terminal_receipt_validated_after_stop": True,
    }
    for key, expected in autostop_expected.items():
        if autostop.get(key) != expected:
            raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_AUTOSTOP_DRIFT={key}")
    dispatch_not_before = autostop.get("dispatch_attempt_not_before_unix")
    job_created = autostop.get("job_created_time")
    if (
        isinstance(dispatch_not_before, bool)
        or not isinstance(dispatch_not_before, int)
        or dispatch_not_before <= 0
        or isinstance(job_created, bool)
        or not isinstance(job_created, int)
        or job_created < dispatch_not_before
    ):
        raise EqualStepGateError("EQUAL_STEP_HANDOFF_FRESH_JOB_TIME_DRIFT")
    _parse_time(autostop.get("stopped_at"), "HANDOFF_STOPPED_AT")
    decision, decision_sha, _ = _json(decision_path, "HANDOFF_DECISION")
    if static.get("decision_sha256") != decision_sha or decision.get("formal_training_authorized") is not False:
        raise EqualStepGateError("EQUAL_STEP_HANDOFF_DECISION_DRIFT")
    _, deployment_sha, _ = _read(deployment_path, "HANDOFF_DEPLOYMENT")
    if static.get("external_deployment_manifest_sha256") != deployment_sha:
        raise EqualStepGateError("EQUAL_STEP_HANDOFF_DEPLOYMENT_DRIFT")
    scientific = _scientific_pass_identity(terminal)
    component_by_variant = {
        item.get("variant"): item for item in static.get("component_authorizations", [])
        if isinstance(item, Mapping)
    }
    if tuple(sorted(component_by_variant)) != VARIANTS:
        raise EqualStepGateError("EQUAL_STEP_HANDOFF_STATIC_COMPONENT_SET_DRIFT")
    for identity in scientific["variants"]:
        component = component_by_variant[identity["variant"]]
        if component.get("run_id") != identity["run_id"] or component.get("task_id") != identity["task_id"]:
            raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_STATIC_TERMINAL_IDENTITY_DRIFT={identity['variant']}")
        if component.get("artifact_hashes") != identity.get("authorization_artifact_hashes"):
            raise EqualStepGateError(f"EQUAL_STEP_HANDOFF_STATIC_TERMINAL_ARTIFACT_HASH_DRIFT={identity['variant']}")
    handoff = {
        "format": "CC_HHGT_V3_2_EQUAL_STEP_FORMAL_GATE_HANDOFF_R1_V1",
        "status": "EQUAL_STEP_GATE_PASS_HANDOFF_NOT_FORMAL_AUTHORIZATION",
        "namespace": NAMESPACE, "pilot_id": PILOT_ID, "job_id": JOB_ID,
        "comparison_only": True, "formal_training_authorized": False,
        "budget_authorized_by_this_receipt": False,
        "static_auth_sha256": static_sha, "jit_budget_receipt_sha256": jit_sha,
        "terminal_execution_log_sha256": log_sha,
        "terminal_line_sha256": terminal_line_sha,
        "autostop_receipt_sha256": autostop_sha,
        "decision_sha256": decision_sha,
        "external_deployment_manifest_sha256": deployment_sha,
        "pilot_sha256": static.get("pilot_sha256"),
        "preregistration_sha256": static.get("preregistration_sha256"),
        "code_tree_sha256": static.get("code_tree_sha256"),
        "r1_static_auth_r6_sha256": static.get("r1_static_auth_r6_sha256"),
        "source_input_reuse_receipt_sha256": static.get("source_input_reuse_receipt_sha256"),
        "fold0_triplet_sha256": static.get("fold0_triplet_sha256"),
        "scientific_identity": scientific,
        "required_downstream_validator_action": (
            "RECOMPUTE_ALL_HASHES_AND_SCIENTIFIC_GATES;DO_NOT_TRUST_STATUS_ALONE"
        ),
    }
    if not scientific["same_initial_model_cross_variant"] or not scientific["same_initial_rng_cross_variant"]:
        raise EqualStepGateError("EQUAL_STEP_HANDOFF_CROSS_VARIANT_INIT_RNG_FALSE")
    _atomic(output, _canonical(handoff))
    return handoff


def _defaults(project: Path) -> dict[str, Path]:
    code = project / "runtime/tools" / NAMESPACE / "code"
    bootstrap = project / "runtime/bootstrap" / NAMESPACE
    prepared = project / "inputs/v32_g012_patient_first_20260830_r1/formal_prepared_20260830_r3_affine_pyg280_localtorch"
    oracle_bootstrap = project / "runtime/bootstrap/v32_group_shared_oracle_paid_gpu_20260901_r2"
    control = project / "runtime/equal_step_gates" / NAMESPACE
    return {
        "code_root": code, "bootstrap_root": bootstrap, "prepared_parent": prepared,
        "template": code / TEMPLATE_RELATIVE,
        "source_reuse": project / f"runtime/bootstrap/{FORMAL_R2_NAMESPACE}/INPUT_REUSE_READY.json",
        "returned_r1": control / "authority/STATIC_AUTH_READY.v32_g012_20260901_r6.json",
        "oracle_pass": control / "authority/ORACLE_PASS_RECEIPT.returned.json",
        "oracle_log": oracle_bootstrap / "ORACLE_EXECUTION.stdout.jsonl",
        "jit": bootstrap / "JIT_BUDGET_READY.json",
        "code_receipt": bootstrap / "CODE_ARCHIVE_READY.json",
        "deployment": control / DEPLOYMENT_NAME,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--materialize-code-ready", action="store_true")
    parser.add_argument("--materialize-handoff", action="store_true")
    parser.add_argument("--code-archive", type=Path)
    parser.add_argument("--autostop", type=Path)
    parser.add_argument("--handoff-output", type=Path)
    parser.add_argument("--terminal-log", type=Path)
    parser.add_argument("--expected-exit-code", type=int)
    parser.add_argument("--external-verifier-sha256")
    parser.add_argument("--deployment-manifest-sha256")
    for name in ("code-root", "bootstrap-root", "prepared-parent", "template", "source-reuse", "returned-r1", "oracle-pass", "oracle-log", "jit", "code-receipt", "deployment"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args(argv)
    defaults = _defaults(args.project_root)
    get = lambda name: getattr(args, name) or defaults[name]
    try:
        if args.materialize_handoff:
            if args.terminal_log is None or args.autostop is None or args.handoff_output is None:
                raise EqualStepGateError("EQUAL_STEP_HANDOFF_ARGUMENTS_REQUIRED")
            result = materialize_formal_gate_handoff(
                static_path=get("bootstrap_root") / "STATIC_AUTH_READY.json",
                jit_path=get("jit"), terminal_log=args.terminal_log,
                autostop_path=args.autostop,
                decision_path=get("code_root") / DECISION_RELATIVE,
                deployment_path=get("deployment"), output=args.handoff_output,
            )
        elif args.terminal_log:
            result = validate_terminal_log(args.terminal_log, int(args.expected_exit_code))
        elif args.config_only:
            result = validate_overlay(get("code_root"), get("template"))
        elif args.materialize_code_ready:
            if args.code_archive is None:
                raise EqualStepGateError("EQUAL_STEP_CODE_ARCHIVE_ARGUMENT_REQUIRED")
            result = materialize_code_receipt(
                code_root=get("code_root"), bootstrap_root=get("bootstrap_root"),
                template_path=get("template"), code_archive=args.code_archive,
                output=get("code_receipt"),
            )
        else:
            template_payload, _, _ = _json(get("template"), "CONFIG_TEMPLATE_CLI")
            template_contract = _mapping(template_payload.get("equal_step_contract"), "CONFIG_TEMPLATE_CLI_CONTRACT")
            deployment_sha = args.deployment_manifest_sha256 or template_contract.get("external_deployment_manifest_sha256")
            if not _sha(args.external_verifier_sha256) or not _sha(deployment_sha):
                raise EqualStepGateError("EQUAL_STEP_EXTERNAL_HASH_ARGUMENTS_REQUIRED")
            kwargs = dict(
                project_root=args.project_root, code_root=get("code_root"),
                bootstrap_root=get("bootstrap_root"), prepared_parent=get("prepared_parent"),
                template_path=get("template"), source_reuse=get("source_reuse"),
                returned_r1=get("returned_r1"), oracle_pass=get("oracle_pass"),
                oracle_log=get("oracle_log"), jit_path=get("jit"),
                code_receipt=get("code_receipt"), deployment_manifest=get("deployment"),
                deployment_manifest_sha256=deployment_sha,
                external_verifier_sha256=args.external_verifier_sha256,
            )
            result = materialize(**kwargs) if args.materialize else materialize(**kwargs)
        print(json.dumps(result, sort_keys=True))
        return 0
    except EqualStepGateError as exc:
        print(str(exc), file=sys.stderr)
        return 42


if __name__ == "__main__":
    raise SystemExit(main())
