#!/usr/bin/env python3
"""Materialize or validate the CPU-only V3.2 G012 r2 static authorization.

The estimator decision is the first gate.  The checked-in r2 config deliberately
contains ``ESTIMATOR_NOT_YET_AUTHORIZED``; while that blocker remains, every CLI
mode exits 42 before inspecting CUDA or creating any authorization/output file.

The 55 GB prepared input tree is never copied or re-hashed here.  Its hashes are
produced once by ``materialize_v32_g012_r2_input_reuse_ready.py`` on a no-GPU
host.  This validator binds that receipt and performs only path/size checks on
the 15 immutable fold files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import stat
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


NAMESPACE = "v32_g012_paid_gpu_20260901_r2"
RUN_NAMESPACE = "v32_g012_paid_gpu_training_20260901_r2"
R1_NAMESPACE = "v32_g012_paid_gpu_20260831_r1"
R1_RESULT_NAMESPACE = "v32_g012_paid_gpu_training_20260831_r1"
R1_ARCHIVE_NAMESPACE = "archive_aborted_runtime_infeasible_20260901_r1"
BLOCKER = "ESTIMATOR_NOT_YET_AUTHORIZED"
ORACLE_PREREQUISITE_SCHEMA = (
    "CC_HHGT_V3_2_G012_R2_ORACLE_PAID_GATE_BINDINGS_V1"
)
FORMAL_GATE_SCHEMA = "CC_HHGT_V3_2_G012_R2_FORMAL_PAID_GATE_BINDINGS_V2"
FORMAL_GATE_BLOCKER = "FORMAL_EQUAL_STEP_RUNTIME_BUDGET_GATE_NOT_YET_LOCKED"
LEGACY_RECEIPT_SCHEMA_BLOCKER = "RECEIPT_SCHEMA_NOT_YET_LOCKED"
RECEIPT_SCHEMA_BLOCKER = FORMAL_GATE_BLOCKER
FORMAL_V2_HASH_FIELDS = (
    "oracle_decision_sha256",
    "oracle_static_auth_sha256",
    "oracle_terminal_sha256",
    "oracle_jit_budget_sha256",
    "oracle_provider_auto_stop_sha256",
    "equal_step_decision_sha256",
    "equal_step_static_auth_sha256",
    "equal_step_terminal_sha256",
    "equal_step_jit_budget_sha256",
    "equal_step_provider_auto_stop_sha256",
    "formal_training_decision_sha256",
    "formal_runtime_projection_sha256",
    "formal_jit_budget_sha256",
    "formal_provider_auto_stop_sha256",
)
FORMAL_V2_PASS_FIELDS = (
    "formal_training_authorized",
    "real_data_oracle_pass",
    "equal_step_g0_g1_g2_pilot_pass",
    "formal_runtime_within_50h_pass",
    "formal_budget_projection_pass",
    "all_future_cpu_nogpu_plus_paid_components_included",
    "fresh_jit_budget_semantics_recomputed",
    "completed_component_receipt_timelines_recomputed",
    "formal_runtime_projection_hours_recomputed",
    "formal_provider_auto_stop_margin_pass",
)
FORMAL_V2_BUDGET_FORMAT = (
    "CC_HHGT_V3_2_G012_R2_ALL_FUTURE_INSTANCE_TIME_BUDGET_V1"
)
FORMAL_V2_BUDGET_COMPONENTS = (
    "cpu_nogpu_prepare_r2",
    "cpu_nogpu_oracle_authorization_transport",
    "paid_oracle",
    "cpu_nogpu_equal_step_authorization_transport",
    "paid_equal_step",
    "cpu_nogpu_formal_authorization_transport",
    "paid_formal_training",
    "provider_stop_result_return_margin",
)
FORMAL_V2_MINIMUM_FUTURE_HOURS = {
    "cpu_nogpu_prepare_r2": Decimal("0.75"),
    "cpu_nogpu_oracle_authorization_transport": Decimal("0.4167"),
    "paid_oracle": Decimal("3"),
    "cpu_nogpu_equal_step_authorization_transport": Decimal("0.4167"),
    "paid_equal_step": Decimal("6"),
    "cpu_nogpu_formal_authorization_transport": Decimal("0.4167"),
    "paid_formal_training": Decimal("0.0001"),
    "provider_stop_result_return_margin": Decimal("0.25"),
}
ESTIMATOR_READY = "ESTIMATOR_AUTHORIZED"
STATIC_FORMAT = "CC_HHGT_V3_2_G012_R2_STATIC_AUTH_READY_V2"
PATCH_FORMAT = "CC_HHGT_V3_2_G012_R2_PATCH_READY_V3"
EXTERNAL_ARCHIVE_LOCK_FORMAT = (
    "CC_HHGT_V3_2_G012_R2_EXTERNAL_CODE_ARCHIVE_LOCK_V1"
)
EXTERNAL_ARCHIVE_LOCK_STATUS = "CODE_ARCHIVE_HASHES_LOCKED_EXTERNAL_BOOTSTRAP"
EXTERNAL_ARCHIVE_LOCK_REQUIRED = "EXTERNAL_CODE_ARCHIVE_LOCK_REQUIRED"
EXTERNAL_ARCHIVE_LOCK_BLOCKER = "EXTERNAL_CODE_ARCHIVE_LOCK_NOT_YET_MATERIALIZED"
INPUT_FORMAT = "CC_HHGT_V3_2_G012_R2_INPUT_REUSE_READY_V1"
INPUT_STATUS = "INPUT_REUSE_READY_HASH_VERIFIED"
ABORT_FORMAT = "CC_HHGT_V3_2_G012_ABORTED_RUN_V2"
ABORT_STATUS = "ABORTED_RUNTIME_INFEASIBLE"
R1_STATIC_AUTH_R6_SHA256 = (
    "1f5bc54b2a8acc037373bfa380c95c8a717e76fde956852ec481b16a0d8716f0"
)
R1_PATCH_READY_SHA256 = (
    "100713824da87b8eb4a369ef118861a322ca4f93d8c1e4846a29f769c24548ca"
)
R1_BASE_CODE_ARCHIVE_SHA256 = (
    "10fb4fa3c9e55a22b0cefdca4dfbf6ee8ffcca60192222e82a00b3cac702c520"
)
INSTANCE_ID = "uhost-1up504geeqzj"
FORMAL_TRAINER = "cc_hhgt.v32.training:run_authorized_task"
ORACLE_NAMESPACE = "v32_group_shared_oracle_paid_gpu_20260901_r2"
ORACLE_RUN_ID = "v32-g012-g2-group-shared-oracle-paid-gpu-20260901-r2"
ORACLE_TASK_ID = (
    f"{ORACLE_RUN_ID}|PATIENT_FOLD_0|CC-HHGT|20260726"
)
ORACLE_TRAINER = (
    "cc_hhgt.v32.group_shared_encoder_oracle:"
    "run_authorized_oracle_comparison"
)
ORACLE_DECISION_FORMAT = "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_DECISION_V1"
ORACLE_DECISION_STATUS = "ORACLE_COMPARISON_PAID_GATE_AUTHORIZED"
ORACLE_DECISION_SHA256 = (
    "a5e60343489dfcad918557a9c268b10dfa258fcb4af40ba24d448570a2d79534"
)
ORACLE_RUNNER_SHA256 = (
    "2aa813081be331c3d6f58f950ba36698f27d71d33cb3b0a73d1a4c0f9325b771"
)
ORACLE_PREREG_SHA256 = (
    "c2d5085f18674431f8e02e5d95c7c3506b6e8d99b18131fa32f5c8def12fdec9"
)
ORACLE_STATIC_FORMAT = "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_R2_STATIC_AUTH_V1"
ORACLE_STATIC_STATUS = "ORACLE_STATIC_AUTH_READY_CPU_ONLY"
ORACLE_TERMINAL_FORMAT = "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_COMPARISON_V1"
ORACLE_PASS_STATUS = (
    "PASS_REAL_DATA_ORACLE_COMPARISON_ONLY_NOT_AUTHORIZED_FOR_FORMAL_TRAINING"
)
JIT_FORMAT = "CANCERLNCATLAS_COMPSHARE_JIT_BUDGET_V1"
JIT_STATUS = "JIT_BUDGET_READY_CONSERVATIVE"
AUTOSTOP_FORMAT = "CANCERLNCATLAS_GROUP_SHARED_ORACLE_AUTOSTOP_V1"
AUTOSTOP_STATUS = "INSTANCE_STOPPED_STATE_CONFIRMED"
ORACLE_MAX_HOURS = 3.0
ORACLE_MAX_COST_CNY = 8.0
PROJECT_BUDGET_CAP_CNY = 210.0
COMPUTE_HOURLY_CNY = 2.05
GPU_HOURLY_CNY = COMPUTE_HOURLY_CNY
DISK_HOURLY_CNY = 0.04
TOTAL_HOURLY_CNY = 2.09
JIT_PROJECT_SCOPE_FORMAT = "CANCERLNCATLAS_FIXED_INSTANCE_PLUS_200GB_BOOT_DISK_V1"
JIT_PROJECT_SCOPE_SHA256 = (
    "726631660639f60035b7774f8be7e090618bf083bc400a1580f4050f02a861fd"
)
JIT_PROFILE_NAME_SHA256 = (
    "37a8eec1ce19687d132fe29051dca629d164e2c4958ba141d5f4133a33f0688f"
)
ORACLE_VERIFIER_SHA256 = (
    "d0761345eea1ebe371f60ce74ce7920f6d1795c0df174dca3fb779069011f553"
)
ORACLE_EXTERNAL_MANIFEST_SHA256 = (
    "8b7e79536bf05c9eb6d9b11c02998eae07b080664319b583bbb074839c43709b"
)
ORACLE_CONFIG_SHA256 = (
    "207a521b46d6ce01cf5f586bc0c7575501133240939132bdd25977435dced3e3"
)
EXPECTED_G2_F0_SHA256 = (
    "4ebba1d4060f3efaa8c080824386802ca732bedf07681ebd0283deb087b0ca39"
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
FRESH_AUTHORITY_FORMAT = "CC_HHGT_V3_2_G012_R2_FRESH_AUTHORITY_V1"
INPUT_MANIFEST_FORMAT = "CC_HHGT_V3_2_G012_R2_INPUT_MANIFEST_V1"
APPROVAL_FORMAT = "CC_HHGT_V3_2_TRAINING_APPROVAL_V1"
RUNTIME_READY_FORMAT = "CC_HHGT_V3_2_G012_R2_RUNTIME_READY_V1"
RUNTIME_READY_STATUS = "R2_RUNTIME_READY_CPU_ONLY"
FORMAL_SEED = 20260726
PAID_ENDPOINT = "paid_gpu"
PAID_HARDWARE = "PAID_PREEMPTIBLE_GPU"
SOURCE_INPUT_ARCHIVE_SHA256 = (
    "1c17b7be89621e5125c39e87f05ce14beb1f2227afdb481d2e9a20049b56bab0"
)
VARIANTS = ("G0", "G1", "G2")
FOLDS = tuple(range(5))
ALLOWED_CODE_PREFIXES = frozenset({"cc_hhgt", "config", "scripts", "tests", "docs"})
SUPPORTED_SCHEDULES = frozenset(
    {
        "exact_cartesian_v1",
        "balanced_cyclic_single_pass_v1",
        "balanced_group_latin_shared_encoder_v1",
    }
)
REQUIRED_CODE_ARTIFACTS = (
    "cc_hhgt/v32/cli.py",
    "cc_hhgt/v32/gpu_backward_probe.py",
    "cc_hhgt/v32/training.py",
    "cc_hhgt/v32/training_guard.py",
    "config/model_v3_2_g012_paid_gpu_20260901_r2.yaml",
    "scripts/cloud_apply_v32_g012_code_patch_no_gpu_20260901_r2.sh",
    "scripts/cloud_authorize_v32_g012_no_gpu_20260901_r2.sh",
    "scripts/cloud_finalize_v32_g012_no_gpu_20260901_r2.sh",
    "scripts/local_materialize_compshare_jit_budget_20260901_r2.ps1",
    "scripts/local_supervise_compshare_paid_gpu_20260901_r2.ps1",
    "scripts/local_supervise_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1",
    "scripts/materialize_v32_g012_external_archive_lock_no_gpu_r1.py",
    "scripts/server_launch_v32_g012_paid_gpu_20260901_r2.sh",
    "scripts/validate_v32_g012_static_auth_ready_r2.py",
    "scripts/verify_v32_g012_r2_overlay_manifest.py",
    "scripts/v32_pipeline.py",
)


class R2AuthorizationError(RuntimeError):
    """A lineage, estimator, path, or immutable-artifact gate failed."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R2AuthorizationError(f"R2_{label}_MAPPING_REQUIRED")
    return dict(value)


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path, label: str) -> None:
    """Reject a symlink/reparse point in every currently existing component.

    Linux reads additionally keep a no-follow descriptor chain open while the
    leaf is consumed.  The explicit lstat walk is retained for Windows, where
    Python does not expose the required ``openat`` primitives.
    """

    target = _absolute_lexical(path)
    chain = [target, *target.parents]
    for component in reversed(chain):
        try:
            observed = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise R2AuthorizationError(
                f"R2_{label}_COMPONENT_LSTAT_FAILED={component}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode) or getattr(
            os.path, "isjunction", lambda _: False
        )(component):
            raise R2AuthorizationError(
                f"R2_{label}_SYMLINK_COMPONENT_FORBIDDEN={component}"
            )


def _open_regular_no_follow(path: Path, label: str) -> tuple[int, list[int], Path]:
    """Open a leaf through a held no-follow directory chain where supported."""

    target = _absolute_lexical(path)
    _assert_no_symlink_components(target, label)
    leaf_flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    held_directories: list[int] = []
    supports_openat = os.name != "nt" and os.open in getattr(os, "supports_dir_fd", set())
    try:
        if supports_openat:
            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            current = os.open(target.anchor, directory_flags)
            held_directories.append(current)
            for component in target.parts[1:-1]:
                current = os.open(component, directory_flags, dir_fd=current)
                observed = os.fstat(current)
                if not stat.S_ISDIR(observed.st_mode):
                    raise R2AuthorizationError(
                        f"R2_{label}_PARENT_NOT_DIRECTORY={component}"
                    )
                held_directories.append(current)
            descriptor = os.open(target.name, leaf_flags, dir_fd=current)
        else:
            descriptor = os.open(target, leaf_flags)
    except (OSError, R2AuthorizationError) as exc:
        for directory in reversed(held_directories):
            try:
                os.close(directory)
            except OSError:
                pass
        if isinstance(exc, R2AuthorizationError):
            raise
        raise R2AuthorizationError(f"R2_{label}_OPEN_FAILED={target}") from exc
    return descriptor, held_directories, target


def _read_regular(path: Path, label: str, *, allow_empty: bool = False) -> tuple[bytes, str, int]:
    descriptor, held_directories, target = _open_regular_no_follow(path, label)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (before.st_size <= 0 and not allow_empty):
            raise R2AuthorizationError(f"R2_{label}_NOT_NONEMPTY_REGULAR={target}")
        if before.st_nlink != 1:
            raise R2AuthorizationError(f"R2_{label}_HARDLINK_FORBIDDEN={target}")
        digest = hashlib.sha256()
        raw = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            raw.extend(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise R2AuthorizationError(f"R2_{label}_CHANGED_DURING_READ={target}")
        try:
            bound_path = os.stat(target, follow_symlinks=False)
        except OSError as exc:
            raise R2AuthorizationError(f"R2_{label}_PATH_LOST_DURING_READ={target}") from exc
        if not stat.S_ISREG(bound_path.st_mode) or (
            bound_path.st_dev,
            bound_path.st_ino,
            bound_path.st_size,
            bound_path.st_mtime_ns,
        ) != identity_after[:4]:
            raise R2AuthorizationError(f"R2_{label}_PATH_REPLACED_DURING_READ={target}")
        return bytes(raw), digest.hexdigest(), int(before.st_size)
    finally:
        os.close(descriptor)
        for directory in reversed(held_directories):
            os.close(directory)


def _sha256_regular(path: Path, label: str, *, allow_empty: bool = False) -> tuple[str, int]:
    _, digest, size = _read_regular(path, label, allow_empty=allow_empty)
    return digest, size


def _stat_regular_no_follow(path: Path, label: str) -> os.stat_result:
    """Bind a regular file without reading its potentially large payload."""

    descriptor, held_directories, target = _open_regular_no_follow(path, label)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_size <= 0:
            raise R2AuthorizationError(f"R2_{label}_NOT_NONEMPTY_REGULAR={target}")
        if observed.st_nlink != 1:
            raise R2AuthorizationError(f"R2_{label}_HARDLINK_FORBIDDEN={target}")
        try:
            rebound = os.stat(target, follow_symlinks=False)
        except OSError as exc:
            raise R2AuthorizationError(f"R2_{label}_PATH_LOST_DURING_STAT={target}") from exc
        if not stat.S_ISREG(rebound.st_mode) or (
            rebound.st_dev,
            rebound.st_ino,
            rebound.st_size,
            rebound.st_mtime_ns,
            rebound.st_ctime_ns,
        ) != (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        ):
            raise R2AuthorizationError(f"R2_{label}_PATH_REPLACED_DURING_STAT={target}")
        return observed
    finally:
        os.close(descriptor)
        for directory in reversed(held_directories):
            os.close(directory)


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    raw, digest, _ = _read_regular(path, label)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R2AuthorizationError(f"R2_{label}_JSON_INVALID={path}") from exc
    return _require_mapping(payload, label), digest


def _load_yaml(path: Path) -> tuple[dict[str, Any], str]:
    raw, digest, _ = _read_regular(path, "CONFIG")
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise R2AuthorizationError(f"R2_CONFIG_YAML_INVALID={path}") from exc
    return _require_mapping(payload, "CONFIG"), digest


def _validate_external_archive_lock(
    path: Path, *, bootstrap_root: Path
) -> tuple[dict[str, Any], str]:
    payload, digest = _load_json(path, "EXTERNAL_CODE_ARCHIVE_LOCK")
    _require_exact_keys(
        payload,
        {
            "format",
            "status",
            "namespace",
            "created_at",
            "archive_path",
            "archive_sha256",
            "manifest_path",
            "manifest_sha256",
            "bootstrap_verifier_path",
            "bootstrap_verifier_sha256",
            "code_archive_contains_lock",
            "hashes_embedded_in_archive",
            "formal_training_authorized",
        },
        "EXTERNAL_CODE_ARCHIVE_LOCK",
    )
    bootstrap = _absolute_lexical(bootstrap_root)
    expected = {
        "format": EXTERNAL_ARCHIVE_LOCK_FORMAT,
        "status": EXTERNAL_ARCHIVE_LOCK_STATUS,
        "namespace": NAMESPACE,
        "archive_path": "/tmp/v32_g012_r2_code_archive_20260901_r1.tar.gz",
        "manifest_path": str(bootstrap / "CODE_ARCHIVE.MANIFEST.json"),
        "bootstrap_verifier_path": str(
            bootstrap / "verify_v32_g012_r2_overlay_manifest.py"
        ),
        "code_archive_contains_lock": False,
        "hashes_embedded_in_archive": False,
        "formal_training_authorized": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise R2AuthorizationError(f"R2_EXTERNAL_ARCHIVE_LOCK_DRIFT={key}")
    for key in ("archive_sha256", "manifest_sha256", "bootstrap_verifier_sha256"):
        if not _is_sha256(payload.get(key)):
            raise R2AuthorizationError(f"R2_EXTERNAL_ARCHIVE_LOCK_HASH_INVALID={key}")
    _parse_utc(payload.get("created_at"), "EXTERNAL_ARCHIVE_LOCK_CREATED_AT")
    return payload, digest


def _strict_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise R2AuthorizationError(f"R2_{label}_POSITIVE_INTEGER_REQUIRED")
    return value


def validate_estimator_config(
    config_path: Path,
    *,
    prepared_parent: Path | None = None,
    run_parent: Path | None = None,
) -> dict[str, Any]:
    """Validate the explicit science/runtime decision without importing Torch."""

    config, config_sha = _load_yaml(config_path)
    encoded = json.dumps(config, sort_keys=True)
    estimator = _require_mapping(config.get("estimator_authorization"), "ESTIMATOR")
    runtime = _require_mapping(config.get("runtime_profile"), "RUNTIME_PROFILE")
    control = _require_mapping(config.get("execution_control"), "EXECUTION_CONTROL")
    if BLOCKER in encoded or estimator.get("status") == BLOCKER:
        raise R2AuthorizationError(BLOCKER)
    if estimator.get("status") != ESTIMATOR_READY:
        raise R2AuthorizationError("R2_ESTIMATOR_STATUS_NOT_AUTHORIZED")
    if estimator.get("runtime_estimate_status") != "RUNTIME_ESTIMATE_ACCEPTED":
        raise R2AuthorizationError("R2_RUNTIME_ESTIMATE_NOT_ACCEPTED")
    if not _is_sha256(estimator.get("decision_receipt_sha256")):
        raise R2AuthorizationError("R2_ESTIMATOR_DECISION_SHA256_INVALID")

    selected = estimator.get("selected_schedule_mode")
    configured = runtime.get("candidate_chunk_schedule_mode")
    if selected not in SUPPORTED_SCHEDULES or configured != selected:
        raise R2AuthorizationError("R2_ESTIMATOR_SCHEDULE_BINDING_DRIFT")
    cycles = _strict_positive_int(runtime.get("max_coverage_cycles"), "MAX_CYCLES")
    patience = _strict_positive_int(
        runtime.get("patience_coverage_cycles"), "PATIENCE_CYCLES"
    )
    if patience > cycles:
        raise R2AuthorizationError("R2_PATIENCE_EXCEEDS_MAX_CYCLES")

    expected_control = {
        "execution_mode": "TRAINING",
        "training_authorized": True,
        "paid_enabled": True,
        "candidate_only": True,
        "overwrite_formal_v32": False,
    }
    for key, expected in expected_control.items():
        if control.get(key) != expected:
            raise R2AuthorizationError(f"R2_EXECUTION_CONTROL_DRIFT={key}")
    hours = control.get("max_paid_hours")
    cost = control.get("max_cost_cny")
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not 0 < hours <= 96:
        raise R2AuthorizationError("R2_PAID_HOURS_CAP_INVALID")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not 0 < cost <= 210:
        raise R2AuthorizationError("R2_PAID_COST_CAP_INVALID")

    if not str(config.get("contract_version", "")).startswith(
        "3.2.0-g012-paid-gpu-20260901-r2"
    ):
        raise R2AuthorizationError("R2_CONFIG_CONTRACT_VERSION_DRIFT")
    fresh_contract = _require_mapping(
        config.get("fresh_launch_contract"), "FRESH_LAUNCH_CONTRACT"
    )
    if fresh_contract != {
        "first_launch_requires_absent_output_root": True,
        "preexisting_checkpoint_forbidden": True,
        "resume_requires_same_r2_lineage": True,
        "lineage_format": "CC_HHGT_V3_2_G012_R2_RUN_LINEAGE_V1",
    }:
        raise R2AuthorizationError("R2_FRESH_LAUNCH_CONTRACT_DRIFT")
    task_contract = _require_mapping(config.get("task_contract"), "TASK_CONTRACT")
    if task_contract.get("graph_variant") != "__GRAPH_VARIANT__":
        raise R2AuthorizationError("R2_CONFIG_TEMPLATE_GRAPH_VARIANT_DRIFT")
    training_io = _require_mapping(config.get("training_io"), "TRAINING_IO")
    output_root = str(training_io.get("output_root", ""))
    if RUN_NAMESPACE not in output_root or R1_RESULT_NAMESPACE in output_root:
        raise R2AuthorizationError("R2_OUTPUT_NAMESPACE_DRIFT")
    if prepared_parent is not None:
        expected_pattern = str(_absolute_lexical(prepared_parent)) + "/__GRAPH_VARIANT__/PATIENT_FOLD_{fold}.pt"
        if training_io.get("prepared_fold_pattern") != expected_pattern:
            raise R2AuthorizationError("R2_PREPARED_INPUT_PATTERN_DRIFT")
    if run_parent is not None:
        expected_output = str(_absolute_lexical(run_parent)) + "/__GRAPH_VARIANT__/training"
        if output_root != expected_output:
            raise R2AuthorizationError("R2_OUTPUT_ROOT_DRIFT")
    return {
        "config": config,
        "config_sha256": config_sha,
        "schedule_mode": selected,
        "max_coverage_cycles": cycles,
        "patience_coverage_cycles": patience,
    }


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R2AuthorizationError(f"R2_{label}_NUMBER_REQUIRED")
    observed = float(value)
    if not math.isfinite(observed):
        raise R2AuthorizationError(f"R2_{label}_FINITE_REQUIRED")
    return observed


def _decimal_number(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R2AuthorizationError(f"R2_{label}_NUMBER_REQUIRED")
    try:
        observed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise R2AuthorizationError(f"R2_{label}_DECIMAL_INVALID") from exc
    if not observed.is_finite():
        raise R2AuthorizationError(f"R2_{label}_FINITE_REQUIRED")
    return observed


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise R2AuthorizationError(f"R2_{label}_TIMESTAMP_REQUIRED")
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise R2AuthorizationError(f"R2_{label}_TIMESTAMP_INVALID") from exc
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise R2AuthorizationError(f"R2_{label}_TIMEZONE_REQUIRED")
    return observed.astimezone(timezone.utc)


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise R2AuthorizationError(
            f"R2_{label}_SCHEMA_DRIFT=extra:{sorted(set(value) - keys)}:"
            f"missing:{sorted(keys - set(value))}"
        )


def _receipt_binding(
    value: object, *, label: str, path_key: str = "path", sha_key: str = "sha256"
) -> tuple[dict[str, Any], Path, str]:
    payload = _require_mapping(value, label)
    raw_path = payload.get(path_key)
    digest = payload.get(sha_key)
    if not isinstance(raw_path, str) or not raw_path:
        raise R2AuthorizationError(f"R2_{label}_PATH_INVALID")
    if not _is_sha256(digest):
        raise R2AuthorizationError(f"R2_{label}_SHA256_INVALID")
    return payload, _absolute_lexical(Path(raw_path)), str(digest)


def _validate_oracle_decision(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload, digest = _load_json(path, "ORACLE_DECISION")
    if digest != ORACLE_DECISION_SHA256 or expected_sha256 != ORACLE_DECISION_SHA256:
        raise R2AuthorizationError("R2_ORACLE_DECISION_SHA256_DRIFT")
    expected = {
        "format": ORACLE_DECISION_FORMAT,
        "decision": ORACLE_DECISION_STATUS,
        "authorized_callable": ORACLE_TRAINER,
        "artifact_class": "TIMING_AND_ORACLE_COMPARISON_ONLY",
        "formal_training_authorized": False,
        "terminal_pass_does_not_authorize_formal_training": True,
        "equal_step_g0_g1_g2_pilot_still_required": True,
        "hour_cap": ORACLE_MAX_HOURS,
        "cost_cap_cny": ORACLE_MAX_COST_CNY,
        "runner_sha256": ORACLE_RUNNER_SHA256,
        "preregistration_sha256": ORACLE_PREREG_SHA256,
    }
    for key, required in expected.items():
        if payload.get(key) != required:
            raise R2AuthorizationError(f"R2_ORACLE_DECISION_DRIFT={key}")
    if payload.get("authorized_scope") != {
        "graph_variant": "G2",
        "patient_fold": 0,
        "seed": 20260726,
    }:
        raise R2AuthorizationError("R2_ORACLE_DECISION_SCOPE_DRIFT")
    thresholds = _require_mapping(payload.get("scientific_thresholds"), "ORACLE_THRESHOLDS")
    if thresholds != {
        "branch_constant_across_all_58_required": True,
        "paired_branch_disagreement_max": 0,
        "scalar_absolute_floor": 1e-7,
        "scalar_relative_tolerance": 1e-6,
        "complete_gradient_relative_l2_max": 1e-3,
        "complete_gradient_cosine_min": 0.9999,
        "complete_gradient_norm_ratio_min": 0.999,
        "complete_gradient_norm_ratio_max": 1.001,
        "module_gradient_cosine_min": 0.999,
    }:
        raise R2AuthorizationError("R2_ORACLE_DECISION_THRESHOLD_DRIFT")
    return payload


def _validate_jit_budget(
    path: Path,
    expected_sha256: str,
    *,
    require_hardened_generator: bool = False,
    generator_path: Path | None = None,
) -> dict[str, Any]:
    payload, digest = _load_json(path, "ORACLE_JIT_BUDGET")
    if digest != expected_sha256:
        raise R2AuthorizationError("R2_ORACLE_JIT_BUDGET_SHA256_DRIFT")
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
        "instance_id": INSTANCE_ID,
    }
    for key, required in expected.items():
        if payload.get(key) != required:
            raise R2AuthorizationError(f"R2_ORACLE_JIT_BUDGET_DRIFT={key}")
    queried = _parse_utc(payload.get("queried_at"), "ORACLE_JIT_QUERIED_AT")
    valid_until = _parse_utc(payload.get("valid_until"), "ORACLE_JIT_VALID_UNTIL")
    if valid_until <= queried or (valid_until - queried).total_seconds() > 45 * 60:
        raise R2AuthorizationError("R2_ORACLE_JIT_VALIDITY_WINDOW_DRIFT")
    remaining = _number(
        payload.get("conservative_remaining_cny"), "ORACLE_JIT_REMAINING_CNY"
    )
    if not 0 < remaining <= PROJECT_BUDGET_CAP_CNY:
        raise R2AuthorizationError("R2_ORACLE_JIT_REMAINING_OUT_OF_RANGE")
    base_fields = set(expected) | {
        "queried_at",
        "valid_until",
        "conservative_remaining_cny",
    }
    legacy_generator_fields = {
        "budget_method",
        "instance_age_seconds",
        "worst_case_full_rate_spend_cny",
        "future_cleanup_reserve_cny",
        "provider_snapshot_sha256",
        "generator_sha256",
    }
    hardened_generator_fields = legacy_generator_fields | {
        "project_scope_format",
        "project_scope_sha256",
        "profile_name_sha256",
        "provider_stop_time_unix",
        "provider_query_duration_seconds",
        "worst_case_spend_rounding",
        "remaining_rounding",
        "unallocated_rounding_reserve_cny",
    }
    extension_fields = set(payload) - base_fields
    is_hardened = extension_fields == hardened_generator_fields
    is_legacy = extension_fields == legacy_generator_fields
    if require_hardened_generator and not is_hardened:
        raise R2AuthorizationError("R2_ORACLE_JIT_HARDENED_GENERATOR_FIELDS_REQUIRED")
    if require_hardened_generator and generator_path is None:
        raise R2AuthorizationError("R2_ORACLE_JIT_GENERATOR_SOURCE_BINDING_REQUIRED")
    if extension_fields and not is_hardened and not is_legacy:
        raise R2AuthorizationError("R2_ORACLE_JIT_GENERATOR_FIELDS_PARTIAL")
    if extension_fields:
        if (
            payload.get("budget_method")
            != "FULL_2_09_CNY_RATE_FOR_EVERY_SECOND_SINCE_INSTANCE_CREATE_MINUS_FUTURE_RESERVE"
        ):
            raise R2AuthorizationError("R2_ORACLE_JIT_BUDGET_METHOD_DRIFT")
        age_seconds = payload.get("instance_age_seconds")
        if (
            isinstance(age_seconds, bool)
            or not isinstance(age_seconds, int)
            or age_seconds < 0
        ):
            raise R2AuthorizationError("R2_ORACLE_JIT_INSTANCE_AGE_INVALID")
        reserve = _number(
            payload.get("future_cleanup_reserve_cny"),
            "ORACLE_JIT_FUTURE_CLEANUP_RESERVE_CNY",
        )
        if reserve < 5.0:
            raise R2AuthorizationError("R2_ORACLE_JIT_CLEANUP_RESERVE_TOO_SMALL")
        for key in ("provider_snapshot_sha256", "generator_sha256"):
            if not _is_sha256(payload.get(key)):
                raise R2AuthorizationError(f"R2_ORACLE_JIT_{key.upper()}_INVALID")
        if is_hardened and require_hardened_generator:
            observed_generator_sha, _ = _sha256_regular(
                generator_path, "FORMAL_JIT_GENERATOR_SOURCE"
            )
            if payload.get("generator_sha256") != observed_generator_sha:
                raise R2AuthorizationError(
                    "R2_ORACLE_JIT_GENERATOR_SOURCE_SHA256_DRIFT"
                )
        try:
            remaining_decimal = Decimal(str(payload["conservative_remaining_cny"]))
            reserve_decimal = Decimal(str(payload["future_cleanup_reserve_cny"]))
            reported_spend_decimal = Decimal(
                str(payload["worst_case_full_rate_spend_cny"])
            )
        except (InvalidOperation, ValueError) as exc:
            raise R2AuthorizationError("R2_ORACLE_JIT_DECIMAL_FIELD_INVALID") from exc
        raw_spend_decimal = (
            Decimal("2.09") * Decimal(age_seconds) / Decimal(3600)
        )
        if is_hardened:
            validity_seconds = Decimal(str((valid_until - queried).total_seconds()))
            minimum_transport_reserve = (
                Decimal("2.09") * validity_seconds / Decimal(3600)
            ).quantize(Decimal("0.0001"), rounding=ROUND_CEILING)
            if reserve_decimal < Decimal("5.0") + minimum_transport_reserve:
                raise R2AuthorizationError(
                    "R2_ORACLE_JIT_VALIDITY_TRANSPORT_RESERVE_TOO_SMALL"
                )
            expected_spend = raw_spend_decimal.quantize(
                Decimal("0.0001"), rounding=ROUND_CEILING
            )
            expected_remaining_raw = (
                Decimal("210.0") - expected_spend - reserve_decimal
            )
            expected_remaining = expected_remaining_raw.quantize(
                Decimal("0.0001"), rounding=ROUND_FLOOR
            )
            expected_unallocated = expected_remaining_raw - expected_remaining
            if reported_spend_decimal != expected_spend:
                raise R2AuthorizationError("R2_ORACLE_JIT_WORST_CASE_SPEND_DRIFT")
            if remaining_decimal != expected_remaining:
                raise R2AuthorizationError(
                    "R2_ORACLE_JIT_CONSERVATIVE_REMAINING_DRIFT"
                )
            try:
                unallocated = Decimal(
                    str(payload["unallocated_rounding_reserve_cny"])
                )
            except (InvalidOperation, ValueError) as exc:
                raise R2AuthorizationError(
                    "R2_ORACLE_JIT_ROUNDING_RESERVE_INVALID"
                ) from exc
            if unallocated != expected_unallocated or not (
                Decimal(0) <= unallocated < Decimal("0.0001")
            ):
                raise R2AuthorizationError("R2_ORACLE_JIT_ROUNDING_RESERVE_DRIFT")
            exact_hardened = {
                "project_scope_format": JIT_PROJECT_SCOPE_FORMAT,
                "project_scope_sha256": JIT_PROJECT_SCOPE_SHA256,
                "profile_name_sha256": JIT_PROFILE_NAME_SHA256,
                "worst_case_spend_rounding": "CEILING_0_0001_CNY",
                "remaining_rounding": "FLOOR_0_0001_CNY",
            }
            for key, required in exact_hardened.items():
                if payload.get(key) != required:
                    raise R2AuthorizationError(f"R2_ORACLE_JIT_{key.upper()}_DRIFT")
            stop_unix = payload.get("provider_stop_time_unix")
            if isinstance(stop_unix, bool) or not isinstance(stop_unix, int):
                raise R2AuthorizationError("R2_ORACLE_JIT_PROVIDER_STOP_TIME_INVALID")
            queried_unix = int(queried.timestamp())
            if not queried_unix - age_seconds <= stop_unix <= queried_unix:
                raise R2AuthorizationError("R2_ORACLE_JIT_PROVIDER_STOP_TIME_DRIFT")
            query_duration = _number(
                payload.get("provider_query_duration_seconds"),
                "ORACLE_JIT_PROVIDER_QUERY_DURATION_SECONDS",
            )
            if not 0 <= query_duration <= 185:
                raise R2AuthorizationError("R2_ORACLE_JIT_PROVIDER_QUERY_DURATION_INVALID")
        else:
            # Historical oracle receipts used the unrounded spend in the
            # remaining calculation.  Preserve retrospective validation only;
            # future formal V2 callers must set require_hardened_generator.
            reported_spend = float(reported_spend_decimal)
            raw_spend = float(raw_spend_decimal)
            if reported_spend + 1e-9 < raw_spend or reported_spend - raw_spend >= 0.0001001:
                raise R2AuthorizationError("R2_ORACLE_JIT_WORST_CASE_SPEND_DRIFT")
            raw_remaining = PROJECT_BUDGET_CAP_CNY - raw_spend - reserve
            if remaining > raw_remaining + 1e-9 or raw_remaining - remaining >= 0.0001001:
                raise R2AuthorizationError(
                    "R2_ORACLE_JIT_CONSERVATIVE_REMAINING_DRIFT"
                )
    return {
        "payload": payload,
        "sha256": digest,
        "queried_at": queried,
        "valid_until": valid_until,
        "remaining_cny": remaining,
        "effective_cost_cap_cny": min(ORACLE_MAX_COST_CNY, remaining),
    }


def _validate_oracle_static(
    path: Path,
    expected_sha256: str,
    *,
    jit: Mapping[str, Any],
) -> dict[str, Any]:
    payload, digest = _load_json(path, "ORACLE_STATIC_AUTH")
    if digest != expected_sha256:
        raise R2AuthorizationError("R2_ORACLE_STATIC_AUTH_SHA256_DRIFT")
    expected = {
        "format": ORACLE_STATIC_FORMAT,
        "status": ORACLE_STATIC_STATUS,
        "namespace": ORACLE_NAMESPACE,
        "cpu_only_materialized": True,
        "gpu_visible_during_materialization": False,
        "comparison_only": True,
        "formal_training_authorized": False,
        "run_id": ORACLE_RUN_ID,
        "task_id": ORACLE_TASK_ID,
        "trainer": ORACLE_TRAINER,
        "graph_variant": "G2",
        "patient_fold": 0,
        "seed": 20260726,
        "max_paid_hours": ORACLE_MAX_HOURS,
        "configured_cost_cap_cny": ORACLE_MAX_COST_CNY,
        "jit_effective_cost_cap_cny": jit["effective_cost_cap_cny"],
        "jit_budget_receipt_sha256": jit["sha256"],
        "instance_id": INSTANCE_ID,
        "oracle_decision_sha256": ORACLE_DECISION_SHA256,
        "oracle_runner_sha256": ORACLE_RUNNER_SHA256,
        "preregistration_sha256": ORACLE_PREREG_SHA256,
        "config_sha256": ORACLE_CONFIG_SHA256,
        "external_verifier_sha256": ORACLE_VERIFIER_SHA256,
        "external_control_deployment_manifest_sha256": (
            ORACLE_EXTERNAL_MANIFEST_SHA256
        ),
        "r1_static_auth_r6_sha256": R1_STATIC_AUTH_R6_SHA256,
        "formal_artifacts_allowed": False,
        "checkpoint_allowed": False,
        "prediction_allowed": False,
        "terminal_pass_does_not_authorize_formal_training": True,
        "equal_step_g0_g1_g2_pilot_still_required": True,
    }
    for key, required in expected.items():
        if payload.get(key) != required:
            raise R2AuthorizationError(f"R2_ORACLE_STATIC_AUTH_DRIFT={key}")
    created = _parse_utc(payload.get("created_at"), "ORACLE_STATIC_CREATED_AT")
    if not jit["queried_at"] <= created <= jit["valid_until"]:
        raise R2AuthorizationError("R2_ORACLE_STATIC_OUTSIDE_JIT_WINDOW")
    hashes = _require_mapping(payload.get("artifact_hashes"), "ORACLE_ARTIFACT_HASHES")
    if set(hashes) != {
        "code_sha256",
        "config_sha256",
        "input_manifest_sha256",
        "task_manifest_sha256",
    } or not all(_is_sha256(value) for value in hashes.values()):
        raise R2AuthorizationError("R2_ORACLE_STATIC_ARTIFACT_HASHES_INVALID")
    for key in (
        "config_sha256",
        "task_manifest_sha256",
        "input_manifest_sha256",
        "approval_sha256",
        "external_verifier_sha256",
        "r1_aborted_sha256",
        "returned_r1_static_auth_r6_sha256",
        "source_input_reuse_receipt_sha256",
        "code_archive_receipt_sha256",
        "code_archive_sha256",
        "deployment_contract_sha256",
    ):
        if not _is_sha256(payload.get(key)):
            raise R2AuthorizationError(f"R2_ORACLE_STATIC_HASH_INVALID={key}")
    if payload.get("returned_r1_static_auth_r6_sha256") != R1_STATIC_AUTH_R6_SHA256:
        raise R2AuthorizationError("R2_ORACLE_RETURNED_R1_STATIC_AUTH_DRIFT")
    return {"payload": payload, "sha256": digest, "created_at": created}


def _close_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _validate_implementation_conformance(value: object, theta: str) -> None:
    payload = _require_mapping(value, f"ORACLE_{theta}_IMPLEMENTATION")
    loss_reference = _number(payload.get("loss_reference"), f"ORACLE_{theta}_LOSS_REFERENCE")
    loss_production = _number(payload.get("loss_production"), f"ORACLE_{theta}_LOSS_PRODUCTION")
    difference = _number(
        payload.get("loss_absolute_difference"), f"ORACLE_{theta}_LOSS_DIFFERENCE"
    )
    tolerance = _number(payload.get("loss_tolerance"), f"ORACLE_{theta}_LOSS_TOLERANCE")
    expected_difference = abs(loss_production - loss_reference)
    expected_tolerance = 1e-6 + 1e-6 * abs(loss_reference)
    if not _close_number(difference, expected_difference) or not _close_number(
        tolerance, expected_tolerance
    ) or difference > tolerance:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_IMPLEMENTATION_LOSS_GATE_FAILED")
    flattened = _require_mapping(
        payload.get("flattened_gradient"), f"ORACLE_{theta}_IMPLEMENTATION_GRADIENT"
    )
    if (
        flattened.get("status") != "COMPARABLE_NONZERO"
        or _number(flattened.get("cosine"), f"ORACLE_{theta}_IMPLEMENTATION_COSINE")
        < 0.99999
        or _number(
            flattened.get("relative_l2"), f"ORACLE_{theta}_IMPLEMENTATION_RELATIVE_L2"
        )
        > 1e-4
    ):
        raise R2AuthorizationError(
            f"R2_ORACLE_{theta}_IMPLEMENTATION_FLATTENED_GRADIENT_FAILED"
        )
    telemetry = _require_mapping(
        payload.get("production_call_telemetry"), f"ORACLE_{theta}_TELEMETRY"
    )
    if telemetry != {
        "unique_chunks": 1,
        "group_rows": 32768,
        "encoder_forward_calls": 1,
        "decoder_forward_calls": 4,
        "global_loss_calls": 1,
        "backward_calls": 1,
    }:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_IMPLEMENTATION_TELEMETRY_DRIFT")
    required_true = {
        "pass",
        "nnpu_branch_identical",
        "all_parameter_gradient_tolerances_pass",
        "flattened_gradient_cosine_pass",
        "production_call_telemetry_pass",
    }
    for key in required_true:
        if payload.get(key) is not True:
            raise R2AuthorizationError(f"R2_ORACLE_{theta}_IMPLEMENTATION_FLAG_DRIFT={key}")
    if _number(
        payload.get("maximum_parameter_gradient_abs_difference"),
        f"ORACLE_{theta}_IMPLEMENTATION_MAX_ABS",
    ) > 1e-5:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_IMPLEMENTATION_MAX_ABS_FAILED")
    checked = payload.get("parameter_tensors_checked")
    if isinstance(checked, bool) or not isinstance(checked, int) or checked <= 0:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_IMPLEMENTATION_PARAMETER_COUNT_INVALID")
    if payload.get("failing_parameters") != []:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_IMPLEMENTATION_FAILURES_PRESENT")


def _validate_theta_result(value: object, theta: str) -> dict[str, bool]:
    payload = _require_mapping(value, f"ORACLE_{theta}")
    branch = _require_mapping(payload.get("branch_gate"), f"ORACLE_{theta}_BRANCH")
    active = branch.get("active_count")
    inactive = branch.get("inactive_count")
    branch_pass = bool(
        branch.get("evaluations") == 58
        and isinstance(active, int)
        and not isinstance(active, bool)
        and isinstance(inactive, int)
        and not isinstance(inactive, bool)
        and active + inactive == 58
        and active in {0, 58}
        and branch.get("constant_across_all_58") is True
        and branch.get("paired_disagreement_count") == 0
    )
    if branch.get("pass") is not branch_pass or not branch_pass:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_BRANCH_GATE_FAILED")

    components = _require_mapping(payload.get("components"), f"ORACLE_{theta}_COMPONENTS")
    expected_components = {
        "membership_risk",
        "direction_loss",
        "shrinkage_penalty",
        "total_objective",
    }
    if set(components) != expected_components:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_COMPONENT_SCHEMA_DRIFT")
    scalar_pass = True
    bootstrap_pass = True
    for name in sorted(expected_components):
        component = _require_mapping(
            components[name], f"ORACLE_{theta}_{name.upper()}"
        )
        reference = _number(component.get("reference_mean"), f"ORACLE_{theta}_{name}_REFERENCE")
        proposed = _number(component.get("proposed_mean"), f"ORACLE_{theta}_{name}_PROPOSED")
        difference = _number(
            component.get("absolute_mean_difference"), f"ORACLE_{theta}_{name}_DIFFERENCE"
        )
        tolerance = _number(
            component.get("identity_tolerance"), f"ORACLE_{theta}_{name}_TOLERANCE"
        )
        expected_difference = abs(proposed - reference)
        expected_tolerance = max(1e-7, 1e-6 * abs(reference))
        identity = bool(difference <= tolerance)
        if (
            not _close_number(difference, expected_difference)
            or not _close_number(tolerance, expected_tolerance)
            or component.get("identity_pass") is not identity
        ):
            raise R2AuthorizationError(f"R2_ORACLE_{theta}_{name}_SCALAR_RECOMPUTE_DRIFT")
        ci = component.get("paired_bootstrap_95_ci")
        if not isinstance(ci, list) or len(ci) != 2:
            raise R2AuthorizationError(f"R2_ORACLE_{theta}_{name}_BOOTSTRAP_CI_INVALID")
        lower = _number(ci[0], f"ORACLE_{theta}_{name}_BOOTSTRAP_LOWER")
        upper = _number(ci[1], f"ORACLE_{theta}_{name}_BOOTSTRAP_UPPER")
        if lower > upper:
            raise R2AuthorizationError(f"R2_ORACLE_{theta}_{name}_BOOTSTRAP_CI_REVERSED")
        margin = _number(
            component.get("bootstrap_equivalence_margin"),
            f"ORACLE_{theta}_{name}_BOOTSTRAP_MARGIN",
        )
        expected_margin = 0.01 * abs(reference)
        bootstrap = bool(lower >= -margin and upper <= margin)
        passed = bool(identity and bootstrap)
        if (
            not _close_number(margin, expected_margin)
            or component.get("bootstrap_pass") is not bootstrap
            or component.get("pass") is not passed
        ):
            raise R2AuthorizationError(f"R2_ORACLE_{theta}_{name}_BOOTSTRAP_RECOMPUTE_DRIFT")
        scalar_pass = scalar_pass and identity
        bootstrap_pass = bootstrap_pass and bootstrap
    if not scalar_pass:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_SCALAR_GATE_FAILED")
    if not bootstrap_pass:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_BOOTSTRAP_GATE_FAILED")

    complete = _require_mapping(
        payload.get("complete_mean_gradient"), f"ORACLE_{theta}_COMPLETE_GRADIENT"
    )
    complete_pass = bool(
        complete.get("status") == "COMPARABLE_NONZERO"
        and _number(complete.get("relative_l2"), f"ORACLE_{theta}_COMPLETE_RELATIVE_L2")
        <= 1e-3
        and _number(complete.get("cosine"), f"ORACLE_{theta}_COMPLETE_COSINE") >= 0.9999
        and 0.999
        <= _number(complete.get("norm_ratio"), f"ORACLE_{theta}_COMPLETE_NORM_RATIO")
        <= 1.001
    )
    if complete.get("pass") is not complete_pass or not complete_pass:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_COMPLETE_GRADIENT_GATE_FAILED")

    modules = _require_mapping(
        payload.get("module_mean_gradients"), f"ORACLE_{theta}_MODULE_GRADIENTS"
    )
    expected_modules = {"encoder", "residual_map_output", "gate", "direction_head"}
    if set(modules) != expected_modules:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_MODULE_SCHEMA_DRIFT")
    module_pass = True
    for name in sorted(expected_modules):
        module = _require_mapping(modules[name], f"ORACLE_{theta}_MODULE_{name}")
        tensors = module.get("parameter_tensors")
        if isinstance(tensors, bool) or not isinstance(tensors, int) or tensors <= 0:
            raise R2AuthorizationError(f"R2_ORACLE_{theta}_MODULE_PARAMETER_COUNT_INVALID={name}")
        status_value = module.get("status")
        if status_value == "TYPED_NA_BOTH_ZERO":
            observed_pass = bool(
                module.get("cosine") is None
                and module.get("norm_ratio") is None
                and _number(module.get("reference_norm"), f"ORACLE_{theta}_{name}_REF_NORM") == 0
                and _number(module.get("proposed_norm"), f"ORACLE_{theta}_{name}_PROP_NORM") == 0
            )
        elif status_value == "COMPARABLE_NONZERO":
            observed_pass = bool(
                _number(module.get("cosine"), f"ORACLE_{theta}_{name}_COSINE") >= 0.999
            )
        else:
            observed_pass = False
        if module.get("pass") is not observed_pass or not observed_pass:
            raise R2AuthorizationError(f"R2_ORACLE_{theta}_MODULE_GRADIENT_GATE_FAILED={name}")
        module_pass = module_pass and observed_pass

    if payload.get("theta") != theta or payload.get("pass") is not True:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_PASS_FLAG_DRIFT")
    if payload.get("scientific_fail_reasons") != []:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_FAIL_REASONS_PRESENT")
    for key in ("reference_mean_gradient_sha256", "proposed_mean_gradient_sha256", "model_state_sha256"):
        if not _is_sha256(payload.get(key)):
            raise R2AuthorizationError(f"R2_ORACLE_{theta}_HASH_INVALID={key}")
    if payload.get("dropout_disabled") is not True or payload.get("optimizer_steps") != 0:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_EXECUTION_CONTRACT_DRIFT")
    if payload.get("reference_call_counts") != {
        "encoder_forward": 116,
        "decoder_forward": 116,
        "global_loss": 29,
        "backward": 29,
    } or payload.get("proposed_call_counts") != {
        "encoder_forward": 29,
        "decoder_forward": 116,
        "global_loss": 29,
        "backward": 29,
    }:
        raise R2AuthorizationError(f"R2_ORACLE_{theta}_CALL_COUNT_DRIFT")
    _validate_implementation_conformance(payload.get("production_shared_api_conformance"), theta)
    return {
        "branch": branch_pass,
        "scalar_identity": scalar_pass,
        "bootstrap_equivalence": bootstrap_pass,
        "complete_gradient": complete_pass,
        "module_gradient": module_pass,
        "implementation_conformance": True,
    }


def _validate_oracle_terminal(
    path: Path,
    expected_sha256: str,
    *,
    expected_exit_code: int,
    oracle_static: Mapping[str, Any],
) -> dict[str, Any]:
    raw, digest, _ = _read_regular(path, "ORACLE_TERMINAL_LOG")
    if digest != expected_sha256:
        raise R2AuthorizationError("R2_ORACLE_TERMINAL_SHA256_DRIFT")
    terminal: list[dict[str, Any]] = []
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise R2AuthorizationError("R2_ORACLE_TERMINAL_UTF8_INVALID") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise R2AuthorizationError(
                f"R2_ORACLE_TERMINAL_JSONL_INVALID_LINE={line_number}"
            ) from exc
        if not isinstance(item, Mapping):
            raise R2AuthorizationError(
                f"R2_ORACLE_TERMINAL_LINE_NOT_MAPPING={line_number}"
            )
        if item.get("format") == ORACLE_TERMINAL_FORMAT and item.get("status") == ORACLE_PASS_STATUS:
            terminal.append(dict(item))
    if len(terminal) != 1:
        raise R2AuthorizationError(f"R2_ORACLE_TERMINAL_CARDINALITY={len(terminal)}")
    receipt = terminal[0]
    if expected_exit_code != 0:
        raise R2AuthorizationError("R2_ORACLE_TERMINAL_PASS_REQUIRES_EXIT_ZERO")
    expected = {
        "format": ORACLE_TERMINAL_FORMAT,
        "status": ORACLE_PASS_STATUS,
        "artifact_class": "TIMING_AND_ORACLE_COMPARISON_ONLY",
        "comparison_only": True,
        "formal_training_authorized": False,
        "formal_artifacts_written": 0,
        "checkpoint_written": False,
        "formal_log_written": False,
        "success_json_written": False,
        "failure_json_written": False,
        "prediction_written": False,
        "winner_selection_input": False,
        "scientific_pass": True,
        "run_id": ORACLE_RUN_ID,
        "task_id": ORACLE_TASK_ID,
        "patient_fold": 0,
        "seed": 20260726,
        "graph_variant": "G2",
        "prepared_artifact_sha256": EXPECTED_G2_F0_SHA256,
        "oof_modality_lineage_preserved": True,
        "patient_fold_authority_preserved": True,
        "optimizer_boundary_preserved": True,
        "immutable_batch_composition_preserved": True,
    }
    for key, required in expected.items():
        if receipt.get(key) != required:
            raise R2AuthorizationError(f"R2_ORACLE_TERMINAL_DRIFT={key}")
    static_payload = _require_mapping(oracle_static.get("payload"), "ORACLE_STATIC_PAYLOAD")
    if receipt.get("authorization_artifact_hashes") != static_payload.get("artifact_hashes"):
        raise R2AuthorizationError("R2_ORACLE_TERMINAL_AUTHORIZATION_HASH_DRIFT")
    if not isinstance(receipt.get("input_authority_hashes"), Mapping) or not receipt[
        "input_authority_hashes"
    ]:
        raise R2AuthorizationError("R2_ORACLE_TERMINAL_INPUT_AUTHORITY_MISSING")
    thresholds = receipt.get("thresholds")
    if thresholds != {
        "branch_constant_across_all_58_required": True,
        "paired_branch_disagreement_max": 0,
        "scalar_absolute_floor": 1e-7,
        "scalar_relative_tolerance": 1e-6,
        "bootstrap_replicates": 20000,
        "bootstrap_seed": 20260901,
        "bootstrap_equivalence_fraction": 0.01,
        "complete_gradient_relative_l2_max": 1e-3,
        "complete_gradient_cosine_min": 0.9999,
        "complete_gradient_norm_ratio": [0.999, 1.001],
        "module_gradient_cosine_min": 0.999,
        "implementation_loss_atol": 1e-6,
        "implementation_loss_rtol": 1e-6,
        "implementation_gradient_max_abs": 1e-5,
        "implementation_gradient_relative_l2": 1e-4,
        "implementation_gradient_cosine_min": 0.99999,
    }:
        raise R2AuthorizationError("R2_ORACLE_TERMINAL_THRESHOLD_DRIFT")
    elapsed = _number(receipt.get("elapsed_seconds"), "ORACLE_ELAPSED_SECONDS")
    peak = receipt.get("peak_reserved_bytes")
    if elapsed <= 0 or elapsed > ORACLE_MAX_HOURS * 3600:
        raise R2AuthorizationError("R2_ORACLE_TIMING_GATE_FAILED")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
        raise R2AuthorizationError("R2_ORACLE_PEAK_RESERVED_BYTES_INVALID")
    theta_results = _require_mapping(receipt.get("theta_results"), "ORACLE_THETA_RESULTS")
    if set(theta_results) != {"theta_0", "theta_1"}:
        raise R2AuthorizationError("R2_ORACLE_THETA_SET_DRIFT")
    recomputed = {
        theta: _validate_theta_result(theta_results[theta], theta)
        for theta in ("theta_0", "theta_1")
    }
    return {
        "payload": receipt,
        "sha256": digest,
        "elapsed_seconds": elapsed,
        "peak_reserved_bytes": peak,
        "recomputed_paid_gates": recomputed,
    }


def _validate_oracle_autostop(
    path: Path,
    expected_sha256: str,
    *,
    oracle_static: Mapping[str, Any],
    jit: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    payload, digest = _load_json(path, "ORACLE_AUTOSTOP")
    if digest != expected_sha256:
        raise R2AuthorizationError("R2_ORACLE_AUTOSTOP_SHA256_DRIFT")
    expected = {
        "format": AUTOSTOP_FORMAT,
        "status": AUTOSTOP_STATUS,
        "namespace": ORACLE_NAMESPACE,
        "stop_reason": "ORACLE_PASS_OBSERVED_IMMEDIATE_STOP",
        "observed_state": "Stopped",
        "gpu_billing_active": False,
        "instance_id": INSTANCE_ID,
        "static_auth_sha256": oracle_static["sha256"],
        "jit_budget_receipt_sha256": jit["sha256"],
        "comparison_only": True,
        "formal_training_authorized": False,
        "terminal_receipt_status": ORACLE_PASS_STATUS,
        "terminal_receipt_sha256": terminal["sha256"],
        "terminal_receipt_validated_after_stop": True,
        "stop_receipt_written_after_provider_confirmation": True,
    }
    for key, required in expected.items():
        if payload.get(key) != required:
            raise R2AuthorizationError(f"R2_ORACLE_AUTOSTOP_DRIFT={key}")
    if not isinstance(payload.get("job_id"), str) or not payload["job_id"].strip():
        raise R2AuthorizationError("R2_ORACLE_AUTOSTOP_JOB_ID_MISSING")
    elapsed = _number(
        payload.get("supervisor_elapsed_seconds"), "ORACLE_AUTOSTOP_ELAPSED_SECONDS"
    )
    if elapsed <= 0 or elapsed > ORACLE_MAX_HOURS * 3600 + 300:
        raise R2AuthorizationError("R2_ORACLE_AUTOSTOP_ELAPSED_INVALID")
    stopped_at = _parse_utc(payload.get("stopped_at"), "ORACLE_AUTOSTOP_STOPPED_AT")
    if stopped_at < oracle_static["created_at"]:
        raise R2AuthorizationError("R2_ORACLE_AUTOSTOP_PRECEDES_STATIC_AUTH")
    return {"payload": payload, "sha256": digest, "stopped_at": stopped_at}


def _oracle_prerequisite_receipt_specs(
    config: Mapping[str, Any],
    *,
    code_root: Path | None = None,
    bootstrap_root: Path | None = None,
) -> dict[str, Any]:
    specs = _require_mapping(config.get("formal_gate_receipts"), "FORMAL_GATE_RECEIPTS")
    status = specs.get("schema_status")
    if status != ORACLE_PREREQUISITE_SCHEMA:
        raise R2AuthorizationError(f"R2_FORMAL_GATE_RECEIPT_SCHEMA_UNSUPPORTED={status}")
    if code_root is None or bootstrap_root is None:
        raise R2AuthorizationError("R2_FORMAL_GATE_CONTEXT_REQUIRED")
    _require_exact_keys(
        specs,
        {
            "schema_status",
            "decision",
            "oracle",
            "timing",
            "jit_budget",
            "provider_auto_stop",
        },
        "FORMAL_GATE_RECEIPTS",
    )
    code = _absolute_lexical(code_root)
    bootstrap = _absolute_lexical(bootstrap_root)
    if bootstrap.name != NAMESPACE:
        raise R2AuthorizationError("R2_BOOTSTRAP_NAMESPACE_DRIFT")
    try:
        project = bootstrap.parents[2]
    except IndexError as exc:
        raise R2AuthorizationError("R2_BOOTSTRAP_PROJECT_ROOT_UNRESOLVED") from exc
    oracle_bootstrap = project / "runtime/bootstrap" / ORACLE_NAMESPACE

    decision_spec, decision_path, decision_sha = _receipt_binding(
        specs.get("decision"), label="FORMAL_DECISION"
    )
    _require_exact_keys(decision_spec, {"path", "sha256"}, "FORMAL_DECISION")
    if decision_path != code / "docs/v32_group_shared_encoder_oracle_decision_20260901_r2.json":
        raise R2AuthorizationError("R2_ORACLE_DECISION_PATH_DRIFT")

    oracle_spec = _require_mapping(specs.get("oracle"), "FORMAL_ORACLE")
    _require_exact_keys(
        oracle_spec,
        {
            "terminal_path",
            "terminal_sha256",
            "exit_code",
            "static_auth_path",
            "static_auth_sha256",
        },
        "FORMAL_ORACLE",
    )
    terminal_path_raw = oracle_spec.get("terminal_path")
    static_path_raw = oracle_spec.get("static_auth_path")
    if not isinstance(terminal_path_raw, str) or not isinstance(static_path_raw, str):
        raise R2AuthorizationError("R2_ORACLE_PATH_BINDING_INVALID")
    terminal_path = _absolute_lexical(Path(terminal_path_raw))
    static_path = _absolute_lexical(Path(static_path_raw))
    terminal_sha = oracle_spec.get("terminal_sha256")
    static_sha = oracle_spec.get("static_auth_sha256")
    if not _is_sha256(terminal_sha) or not _is_sha256(static_sha):
        raise R2AuthorizationError("R2_ORACLE_HASH_BINDING_INVALID")
    if terminal_path != oracle_bootstrap / "ORACLE_EXECUTION.stdout.jsonl":
        raise R2AuthorizationError("R2_ORACLE_TERMINAL_PATH_DRIFT")
    if static_path != oracle_bootstrap / "STATIC_AUTH_READY.json":
        raise R2AuthorizationError("R2_ORACLE_STATIC_AUTH_PATH_DRIFT")

    timing_spec = _require_mapping(specs.get("timing"), "FORMAL_TIMING")
    _require_exact_keys(timing_spec, {"source", "path", "sha256"}, "FORMAL_TIMING")
    if (
        timing_spec.get("source") != "ORACLE_TERMINAL_RECEIPT_FIELDS"
        or timing_spec.get("path") != str(terminal_path)
        or timing_spec.get("sha256") != terminal_sha
    ):
        raise R2AuthorizationError("R2_ORACLE_TIMING_MUST_REUSE_TERMINAL_RECEIPT")

    jit_spec, jit_path, jit_sha = _receipt_binding(
        specs.get("jit_budget"), label="FORMAL_JIT_BUDGET"
    )
    _require_exact_keys(jit_spec, {"path", "sha256"}, "FORMAL_JIT_BUDGET")
    if jit_path != oracle_bootstrap / "JIT_BUDGET_READY.json":
        raise R2AuthorizationError("R2_ORACLE_JIT_BUDGET_PATH_DRIFT")

    stop_spec = _require_mapping(specs.get("provider_auto_stop"), "FORMAL_AUTOSTOP")
    _require_exact_keys(
        stop_spec,
        {"path", "sha256", "supervisor_path", "supervisor_sha256"},
        "FORMAL_AUTOSTOP",
    )
    stop_path_raw = stop_spec.get("path")
    supervisor_path_raw = stop_spec.get("supervisor_path")
    if not isinstance(stop_path_raw, str) or not isinstance(supervisor_path_raw, str):
        raise R2AuthorizationError("R2_ORACLE_AUTOSTOP_PATH_BINDING_INVALID")
    stop_path = _absolute_lexical(Path(stop_path_raw))
    supervisor_path = _absolute_lexical(Path(supervisor_path_raw))
    stop_sha = stop_spec.get("sha256")
    supervisor_sha = stop_spec.get("supervisor_sha256")
    if not _is_sha256(stop_sha) or not _is_sha256(supervisor_sha):
        raise R2AuthorizationError("R2_ORACLE_AUTOSTOP_HASH_BINDING_INVALID")
    if stop_path != bootstrap / "formal_gate_receipts/ORACLE_AUTOSTOP.json":
        raise R2AuthorizationError("R2_ORACLE_AUTOSTOP_PATH_DRIFT")
    expected_supervisor = code / "scripts/local_supervise_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1"
    if supervisor_path != expected_supervisor:
        raise R2AuthorizationError("R2_ORACLE_SUPERVISOR_PATH_DRIFT")
    observed_supervisor_sha, _ = _sha256_regular(supervisor_path, "ORACLE_SUPERVISOR")
    if observed_supervisor_sha != supervisor_sha:
        raise R2AuthorizationError("R2_ORACLE_SUPERVISOR_SHA256_DRIFT")

    decision = _validate_oracle_decision(decision_path, decision_sha)
    observed_runner_sha, _ = _sha256_regular(
        code / "cc_hhgt/v32/group_shared_encoder_oracle.py",
        "ORACLE_RUNNER_SOURCE",
    )
    observed_prereg_sha, _ = _sha256_regular(
        code / "docs/v32_group_shared_encoder_estimator_preregistration_20260901.md",
        "ORACLE_PREREGISTRATION",
    )
    observed_config_sha, _ = _sha256_regular(
        code / "config/model_v3_2_group_shared_oracle_paid_gpu_20260901_r2.yaml",
        "ORACLE_CONFIG_SOURCE",
    )
    observed_verifier_sha, _ = _sha256_regular(
        code / "scripts/validate_v32_group_shared_oracle_static_auth_ready_r2.py",
        "ORACLE_VERIFIER_SOURCE",
    )
    observed_manifest_sha, _ = _sha256_regular(
        code / "docs/v32_group_shared_oracle_external_deployment_manifest_20260901_r2.json",
        "ORACLE_EXTERNAL_DEPLOYMENT_MANIFEST_SOURCE",
    )
    if observed_runner_sha != decision.get("runner_sha256"):
        raise R2AuthorizationError("R2_ORACLE_RUNNER_SOURCE_SHA256_DRIFT")
    if observed_prereg_sha != decision.get("preregistration_sha256"):
        raise R2AuthorizationError("R2_ORACLE_PREREGISTRATION_SOURCE_SHA256_DRIFT")
    if observed_config_sha != ORACLE_CONFIG_SHA256:
        raise R2AuthorizationError("R2_ORACLE_CONFIG_SOURCE_SHA256_DRIFT")
    if observed_verifier_sha != ORACLE_VERIFIER_SHA256:
        raise R2AuthorizationError("R2_ORACLE_VERIFIER_SOURCE_SHA256_DRIFT")
    if observed_manifest_sha != ORACLE_EXTERNAL_MANIFEST_SHA256:
        raise R2AuthorizationError(
            "R2_ORACLE_EXTERNAL_DEPLOYMENT_MANIFEST_SOURCE_SHA256_DRIFT"
        )
    jit = _validate_jit_budget(jit_path, jit_sha)
    oracle_static = _validate_oracle_static(
        static_path, str(static_sha), jit=jit
    )
    terminal = _validate_oracle_terminal(
        terminal_path,
        str(terminal_sha),
        expected_exit_code=oracle_spec.get("exit_code"),
        oracle_static=oracle_static,
    )
    autostop = _validate_oracle_autostop(
        stop_path,
        str(stop_sha),
        oracle_static=oracle_static,
        jit=jit,
        terminal=terminal,
    )
    return {
        "schema": ORACLE_PREREQUISITE_SCHEMA,
        "decision_sha256": decision_sha,
        "oracle_static_auth_sha256": oracle_static["sha256"],
        "oracle_terminal_sha256": terminal["sha256"],
        "oracle_elapsed_seconds": terminal["elapsed_seconds"],
        "oracle_peak_reserved_bytes": terminal["peak_reserved_bytes"],
        "jit_budget_sha256": jit["sha256"],
        "provider_auto_stop_sha256": autostop["sha256"],
        "oracle_supervisor_sha256": supervisor_sha,
        "recomputed_paid_gates": terminal["recomputed_paid_gates"],
        "formal_training_authorized_by_oracle": False,
        "oracle_pass_is_prerequisite_only": True,
    }


def _formal_receipt_specs(
    config: Mapping[str, Any],
    *,
    code_root: Path | None = None,
    bootstrap_root: Path | None = None,
) -> dict[str, Any]:
    """Fail closed until the complete formal V2 evidence bridge exists.

    The V1 bridge proves only the real-data oracle prerequisite.  Its own
    decision and terminal receipts explicitly say that an equal-step G0/G1/G2
    pilot is still required and that oracle PASS does not authorize formal
    training.  Formal V2 must additionally bind and recompute the equal-step
    pilot, a <=50-hour runtime/cost projection, a fresh formal-training JIT
    budget receipt, and the formal provider auto-stop margin.  None of those
    real receipt hashes exists yet, so no placeholder can authorize r2.

    Before this blocker is replaced, the bridge must open the hash-bound fresh
    JIT receipt with ``_validate_jit_budget(require_hardened_generator=True)``
    and source ``conservative_remaining_cny``/timestamps from that receipt; it
    must also open every completed component receipt, prove its timestamp is no
    later than the fresh JIT query, and prove the paid-formal hours equal the
    independently recomputed runtime-projection receipt.  A mapping that merely
    self-reports those values is not a valid implementation of Formal V2.
    """

    del code_root, bootstrap_root
    specs = _require_mapping(config.get("formal_gate_receipts"), "FORMAL_GATE_RECEIPTS")
    status = specs.get("schema_status")
    if status in {
        FORMAL_GATE_BLOCKER,
        LEGACY_RECEIPT_SCHEMA_BLOCKER,
        ORACLE_PREREQUISITE_SCHEMA,
        FORMAL_GATE_SCHEMA,
    }:
        raise R2AuthorizationError(FORMAL_GATE_BLOCKER)
    raise R2AuthorizationError(f"R2_FORMAL_GATE_RECEIPT_SCHEMA_UNSUPPORTED={status}")


def _validate_formal_budget_projection(
    value: object,
    *,
    formal_runtime_projection_sha256: str,
    formal_jit_budget_sha256: str,
) -> None:
    """Recompute the complete future instance-time envelope.

    The historical 123.31 CNY arithmetic covered only the 3 h oracle, 6 h
    equal-step pilot and 50 h formal paid caps.  It omitted CPU/no-GPU
    bootstrap, authorization/transport and post-run stop/return time.  Formal
    V2 therefore carries every phase explicitly and must fit under the fresh
    JIT receipt's already-reserved conservative remainder.
    """

    projection = _require_mapping(value, "FORMAL_V2_BUDGET_PROJECTION")
    _require_exact_keys(
        projection,
        {
            "format",
            "projection_receipt_sha256",
            "fresh_jit_receipt_sha256",
            "project_budget_cap_cny",
            "full_rate_cny_per_hour",
            "fresh_jit_conservative_remaining_cny",
            "fresh_jit_queried_at",
            "fresh_jit_valid_until",
            "formal_runtime_projection_hours",
            "components",
            "total_projected_future_hours",
            "total_projected_future_cost_cny",
            "budget_headroom_cny",
            "cleanup_and_validity_reserve_already_excluded",
            "all_future_cpu_nogpu_plus_paid_components_included",
            "stop_result_return_margin_included",
        },
        "FORMAL_V2_BUDGET_PROJECTION",
    )
    if projection.get("format") != FORMAL_V2_BUDGET_FORMAT:
        raise R2AuthorizationError("R2_FORMAL_V2_BUDGET_FORMAT_DRIFT")
    if projection.get("projection_receipt_sha256") != formal_runtime_projection_sha256:
        raise R2AuthorizationError("R2_FORMAL_V2_BUDGET_PROJECTION_HASH_DRIFT")
    if projection.get("fresh_jit_receipt_sha256") != formal_jit_budget_sha256:
        raise R2AuthorizationError("R2_FORMAL_V2_BUDGET_JIT_HASH_DRIFT")
    if (
        projection.get("cleanup_and_validity_reserve_already_excluded") is not True
        or projection.get("all_future_cpu_nogpu_plus_paid_components_included") is not True
        or projection.get("stop_result_return_margin_included") is not True
    ):
        raise R2AuthorizationError("R2_FORMAL_V2_BUDGET_COVERAGE_FLAG_FAILED")

    project_cap = _decimal_number(
        projection.get("project_budget_cap_cny"), "FORMAL_V2_PROJECT_CAP_CNY"
    )
    rate = _decimal_number(
        projection.get("full_rate_cny_per_hour"), "FORMAL_V2_FULL_RATE_CNY"
    )
    remaining = _decimal_number(
        projection.get("fresh_jit_conservative_remaining_cny"),
        "FORMAL_V2_FRESH_JIT_REMAINING_CNY",
    )
    if project_cap != Decimal("210") or rate != Decimal("2.09"):
        raise R2AuthorizationError("R2_FORMAL_V2_BUDGET_PRICE_OR_CAP_DRIFT")
    if not Decimal(0) < remaining <= project_cap:
        raise R2AuthorizationError("R2_FORMAL_V2_FRESH_JIT_REMAINING_INVALID")
    fresh_jit_queried_at = _parse_utc(
        projection.get("fresh_jit_queried_at"), "FORMAL_V2_FRESH_JIT_QUERIED_AT"
    )
    fresh_jit_valid_until = _parse_utc(
        projection.get("fresh_jit_valid_until"), "FORMAL_V2_FRESH_JIT_VALID_UNTIL"
    )
    if (
        fresh_jit_valid_until <= fresh_jit_queried_at
        or (fresh_jit_valid_until - fresh_jit_queried_at).total_seconds() > 45 * 60
    ):
        raise R2AuthorizationError("R2_FORMAL_V2_FRESH_JIT_WINDOW_DRIFT")

    components = _require_mapping(
        projection.get("components"), "FORMAL_V2_BUDGET_COMPONENTS"
    )
    _require_exact_keys(
        components,
        set(FORMAL_V2_BUDGET_COMPONENTS),
        "FORMAL_V2_BUDGET_COMPONENTS",
    )
    total_hours = Decimal(0)
    total_cost = Decimal(0)
    for name in FORMAL_V2_BUDGET_COMPONENTS:
        component = _require_mapping(
            components.get(name), f"FORMAL_V2_BUDGET_COMPONENT_{name.upper()}"
        )
        _require_exact_keys(
            component,
            {
                "state",
                "hours_cap",
                "cost_cap_cny",
                "completion_receipt_sha256",
                "completed_at",
            },
            f"FORMAL_V2_BUDGET_COMPONENT_{name.upper()}",
        )
        state = component.get("state")
        hours = _decimal_number(
            component.get("hours_cap"),
            f"FORMAL_V2_BUDGET_{name.upper()}_HOURS",
        )
        cost = _decimal_number(
            component.get("cost_cap_cny"),
            f"FORMAL_V2_BUDGET_{name.upper()}_COST",
        )
        completion_hash = component.get("completion_receipt_sha256")
        if state == "FUTURE_CAP_INCLUDED":
            if (
                hours < FORMAL_V2_MINIMUM_FUTURE_HOURS[name]
                or completion_hash is not None
                or component.get("completed_at") is not None
            ):
                raise R2AuthorizationError(
                    f"R2_FORMAL_V2_BUDGET_FUTURE_COMPONENT_DRIFT={name}"
                )
            expected_cost = (hours * rate).quantize(
                Decimal("0.0001"), rounding=ROUND_CEILING
            )
            if cost != expected_cost:
                raise R2AuthorizationError(
                    f"R2_FORMAL_V2_BUDGET_COMPONENT_COST_DRIFT={name}"
                )
        elif state == "COMPLETED_BEFORE_FRESH_JIT":
            try:
                completed_at = _parse_utc(
                    component.get("completed_at"),
                    f"FORMAL_V2_BUDGET_{name.upper()}_COMPLETED_AT",
                )
            except R2AuthorizationError as exc:
                raise R2AuthorizationError(
                    f"R2_FORMAL_V2_BUDGET_COMPLETED_COMPONENT_DRIFT={name}"
                ) from exc
            if (
                hours != 0
                or cost != 0
                or not _is_sha256(completion_hash)
                or completed_at > fresh_jit_queried_at
            ):
                raise R2AuthorizationError(
                    f"R2_FORMAL_V2_BUDGET_COMPLETED_COMPONENT_DRIFT={name}"
                )
        else:
            raise R2AuthorizationError(
                f"R2_FORMAL_V2_BUDGET_COMPONENT_STATE_DRIFT={name}"
            )
        total_hours += hours
        total_cost += cost

    formal_training = _require_mapping(
        components["paid_formal_training"],
        "FORMAL_V2_BUDGET_PAID_FORMAL_TRAINING",
    )
    formal_hours = _decimal_number(
        formal_training.get("hours_cap"), "FORMAL_V2_PAID_FORMAL_HOURS"
    )
    runtime_projection_hours = _decimal_number(
        projection.get("formal_runtime_projection_hours"),
        "FORMAL_V2_RUNTIME_PROJECTION_HOURS",
    )
    if formal_training.get("state") != "FUTURE_CAP_INCLUDED" or not (
        Decimal(0) < formal_hours <= Decimal(50)
    ) or runtime_projection_hours != formal_hours:
        raise R2AuthorizationError("R2_FORMAL_V2_PAID_FORMAL_HOURS_INVALID")
    formal_cpu = _require_mapping(
        components["cpu_nogpu_formal_authorization_transport"],
        "FORMAL_V2_BUDGET_FORMAL_CPU_AUTHORIZATION",
    )
    if formal_cpu.get("state") != "FUTURE_CAP_INCLUDED":
        raise R2AuthorizationError("R2_FORMAL_V2_FORMAL_CPU_AUTH_NOT_FUTURE")
    stop_margin = _require_mapping(
        components["provider_stop_result_return_margin"],
        "FORMAL_V2_BUDGET_STOP_RETURN_MARGIN",
    )
    if stop_margin.get("state") != "FUTURE_CAP_INCLUDED":
        raise R2AuthorizationError("R2_FORMAL_V2_STOP_RETURN_MARGIN_NOT_FUTURE")

    reported_hours = _decimal_number(
        projection.get("total_projected_future_hours"),
        "FORMAL_V2_TOTAL_FUTURE_HOURS",
    )
    reported_cost = _decimal_number(
        projection.get("total_projected_future_cost_cny"),
        "FORMAL_V2_TOTAL_FUTURE_COST_CNY",
    )
    headroom = _decimal_number(
        projection.get("budget_headroom_cny"), "FORMAL_V2_BUDGET_HEADROOM_CNY"
    )
    if reported_hours != total_hours or reported_cost != total_cost:
        raise R2AuthorizationError("R2_FORMAL_V2_BUDGET_TOTAL_RECOMPUTE_DRIFT")
    expected_headroom = remaining - total_cost
    if total_cost > remaining or headroom != expected_headroom or headroom < 0:
        raise R2AuthorizationError("R2_FORMAL_V2_BUDGET_EXCEEDS_FRESH_JIT")


def _formal_prerequisite_hashes(formal_receipts: object) -> dict[str, str]:
    """Validate the exact future V2 hand-off and preserve every receipt hash.

    This downstream check is deliberately independent of the receipt bridge.
    It prevents a future implementation from validating equal-step/runtime
    evidence and then accidentally dropping those bindings from the three
    training approvals, fresh-authority index, or static authorization.
    """

    try:
        formal = _require_mapping(formal_receipts, "FORMAL_V2_RESULT")
    except R2AuthorizationError as exc:
        raise R2AuthorizationError(FORMAL_GATE_BLOCKER) from exc
    expected_keys = {
        "schema",
        "budget_projection",
        *FORMAL_V2_HASH_FIELDS,
        *FORMAL_V2_PASS_FIELDS,
    }
    if set(formal) != expected_keys or formal.get("schema") != FORMAL_GATE_SCHEMA:
        raise R2AuthorizationError(FORMAL_GATE_BLOCKER)
    if any(formal.get(field) is not True for field in FORMAL_V2_PASS_FIELDS):
        raise R2AuthorizationError(FORMAL_GATE_BLOCKER)
    if any(not _is_sha256(formal.get(field)) for field in FORMAL_V2_HASH_FIELDS):
        raise R2AuthorizationError(FORMAL_GATE_BLOCKER)
    try:
        _validate_formal_budget_projection(
            formal.get("budget_projection"),
            formal_runtime_projection_sha256=str(
                formal["formal_runtime_projection_sha256"]
            ),
            formal_jit_budget_sha256=str(formal["formal_jit_budget_sha256"]),
        )
    except R2AuthorizationError as exc:
        raise R2AuthorizationError(FORMAL_GATE_BLOCKER) from exc
    return {field: str(formal[field]) for field in FORMAL_V2_HASH_FIELDS}


def _validate_cpu_ready(path: Path) -> tuple[dict[str, Any], str]:
    raw, digest, _ = _read_regular(path, "CPU_READY")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise R2AuthorizationError("R2_CPU_READY_NON_ASCII") from exc
    lines = text.splitlines()
    if len(lines) != 1:
        raise R2AuthorizationError("R2_CPU_READY_LINE_COUNT_DRIFT")
    fields = lines[0].split("\t")
    if fields != [SOURCE_INPUT_ARCHIVE_SHA256, R1_BASE_CODE_ARCHIVE_SHA256, "15"]:
        raise R2AuthorizationError("R2_CPU_READY_FIELDS_DRIFT")
    return {
        "source_input_archive_sha256": fields[0],
        "base_code_archive_sha256": fields[1],
        "fold_artifact_count": 15,
    }, digest


def _assert_fresh_output_absent(run_parent: Path) -> None:
    target = _absolute_lexical(run_parent)
    # lstat catches broken links, unlike Path.exists().
    try:
        os.lstat(target)
    except FileNotFoundError:
        pass
    else:
        if target.is_symlink() or getattr(
            os.path, "isjunction", lambda _: False
        )(target):
            raise R2AuthorizationError(f"R2_OUTPUT_ROOT_SYMLINK_FORBIDDEN={target}")
        raise R2AuthorizationError(f"R2_OUTPUT_ROOT_PREEXISTS_FRESH_ONLY={target}")
    parent = target.parent
    while parent != parent.parent:
        try:
            observed = os.lstat(parent)
        except FileNotFoundError:
            parent = parent.parent
            continue
        if stat.S_ISLNK(observed.st_mode) or getattr(
            os.path, "isjunction", lambda _: False
        )(parent):
            raise R2AuthorizationError(f"R2_OUTPUT_PARENT_SYMLINK_FORBIDDEN={parent}")
        parent = parent.parent


def _validate_fresh_or_resume_state(
    run_parent: Path,
    *,
    allow_resume: bool,
    static_auth_sha256: str | None = None,
    static_payload: Mapping[str, Any] | None = None,
) -> str:
    target = _absolute_lexical(run_parent)
    try:
        observed = os.lstat(target)
    except FileNotFoundError:
        return "FRESH_FIRST_LAUNCH_OUTPUT_ABSENT"
    if stat.S_ISLNK(observed.st_mode) or getattr(
        os.path, "isjunction", lambda _: False
    )(target):
        raise R2AuthorizationError(f"R2_OUTPUT_ROOT_SYMLINK_FORBIDDEN={target}")
    if not stat.S_ISDIR(observed.st_mode):
        raise R2AuthorizationError(f"R2_OUTPUT_ROOT_NOT_DIRECTORY={target}")
    if not allow_resume:
        raise R2AuthorizationError(f"R2_OUTPUT_ROOT_PREEXISTS_FRESH_ONLY={target}")
    if static_payload is None or not _is_sha256(static_auth_sha256):
        raise R2AuthorizationError("R2_RESUME_STATIC_AUTH_BINDING_REQUIRED")
    lineage_path = target / "RUN_LINEAGE.json"
    lineage, _ = _load_json(lineage_path, "RUN_LINEAGE")
    expected = {
        "format": "CC_HHGT_V3_2_G012_R2_RUN_LINEAGE_V1",
        "status": "R2_RUN_LINEAGE_ACTIVE",
        "namespace": NAMESPACE,
        "run_parent": str(target),
        "resume_authorized": True,
        "static_auth_ready_sha256": static_auth_sha256,
        "config_sha256": static_payload.get("config_sha256"),
        "code_tree_sha256": static_payload.get("code_tree_sha256"),
        "input_reuse_ready_sha256": static_payload.get("input_reuse_ready_sha256"),
        "launcher_sha256": static_payload.get("launcher_sha256"),
        "supervisor_sha256": static_payload.get("supervisor_sha256"),
        "finalizer_sha256": static_payload.get("finalizer_sha256"),
        "fresh_authority_sha256": static_payload.get("fresh_authority_sha256"),
        "authorized_trainer": FORMAL_TRAINER,
        "formal_result_artifacts_reused": False,
    }
    _require_exact_keys(lineage, set(expected), "RESUME_LINEAGE")
    for key, value in expected.items():
        if lineage.get(key) != value:
            raise R2AuthorizationError(f"R2_RESUME_LINEAGE_DRIFT={key}")
    return "SAME_R2_LINEAGE_RESUME_READY"


def _validate_r1_aborted(path: Path) -> str:
    payload, digest = _load_json(path, "R1_ABORTED")
    expected = {
        "format": ABORT_FORMAT,
        "status": ABORT_STATUS,
        "resume_authorized": False,
        "superseded_by_namespace": NAMESPACE,
        "instance_id": INSTANCE_ID,
        "r1_patch_ready_sha256": R1_PATCH_READY_SHA256,
        "r1_static_auth_r6_sha256": R1_STATIC_AUTH_R6_SHA256,
        "formal_result_root_examined": False,
        "formal_result_root_modified": False,
        "formal_result_artifacts_reused": False,
        "deletion_performed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise R2AuthorizationError(f"R2_R1_ABORTED_DRIFT={key}")
    if payload.get("live_authorizations_removed") is not True:
        raise R2AuthorizationError("R2_R1_LIVE_AUTH_NOT_REMOVED")
    if payload.get("gpu_training_complete_present") is not False:
        raise R2AuthorizationError("R2_R1_GPU_COMPLETE_FLAG_DRIFT")
    stop = _require_mapping(payload.get("instance_stop_evidence"), "R1_STOP_EVIDENCE")
    if (
        stop.get("instance_id") != INSTANCE_ID
        or stop.get("observed_state") != "Stopped"
        or stop.get("gpu_billing_active") is not False
    ):
        raise R2AuthorizationError("R2_R1_STOP_EVIDENCE_DRIFT")
    age = stop.get("age_seconds_at_seal")
    if isinstance(age, bool) or not isinstance(age, (int, float)) or not 0 <= age <= 900:
        raise R2AuthorizationError("R2_R1_STOP_EVIDENCE_WAS_NOT_JIT")
    return digest


def _validate_input_reuse(
    path: Path, prepared_parent: Path
) -> tuple[str, dict[tuple[str, int], dict[str, Any]]]:
    payload, digest = _load_json(path, "INPUT_REUSE_READY")
    expected = {
        "format": INPUT_FORMAT,
        "status": INPUT_STATUS,
        "source_input_archive_sha256": SOURCE_INPUT_ARCHIVE_SHA256,
        "r1_static_auth_r6_sha256": R1_STATIC_AUTH_R6_SHA256,
        "fold_artifact_count": 15,
        "reused_existing_prepared_inputs": True,
        "retransfer_performed": False,
        "copy_performed": False,
        "extraction_performed": False,
        "source_files_modified": False,
        "formal_result_artifacts_used": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise R2AuthorizationError(f"R2_INPUT_REUSE_DRIFT={key}")
    _assert_no_symlink_components(prepared_parent, "PREPARED_PARENT")
    parent = _absolute_lexical(prepared_parent)
    if payload.get("prepared_parent") != str(parent):
        raise R2AuthorizationError("R2_INPUT_REUSE_PREPARED_PARENT_DRIFT")
    records = payload.get("fold_artifacts")
    if not isinstance(records, list) or len(records) != 15:
        raise R2AuthorizationError("R2_INPUT_REUSE_RECORD_COUNT_DRIFT")
    observed: set[tuple[str, int]] = set()
    observed_bindings: dict[tuple[str, int], dict[str, Any]] = {}
    total = 0
    for raw in records:
        record = _require_mapping(raw, "INPUT_RECORD")
        variant, fold = record.get("variant"), record.get("fold")
        if variant not in VARIANTS or fold not in FOLDS:
            raise R2AuthorizationError("R2_INPUT_REUSE_RECORD_KEY_INVALID")
        key = (str(variant), int(fold))
        if key in observed:
            raise R2AuthorizationError("R2_INPUT_REUSE_RECORD_DUPLICATE")
        observed.add(key)
        expected_path = parent / str(variant) / f"PATIENT_FOLD_{fold}.pt"
        if record.get("path") != str(expected_path):
            raise R2AuthorizationError(f"R2_INPUT_REUSE_PATH_DRIFT={variant}:{fold}")
        observed_stat = _stat_regular_no_follow(
            expected_path, f"INPUT_REUSE_FOLD_{variant}_{fold}"
        )
        size = observed_stat.st_size
        if size <= 0 or record.get("size_bytes") != size:
            raise R2AuthorizationError(f"R2_INPUT_REUSE_SIZE_DRIFT={variant}:{fold}")
        if not _is_sha256(record.get("sha256")):
            raise R2AuthorizationError(f"R2_INPUT_REUSE_SHA_INVALID={variant}:{fold}")
        total += size
        observed_bindings[key] = {
            "fold": int(fold),
            "path": str(expected_path),
            "sha256": record["sha256"],
            "size_bytes": size,
        }
    if observed != {(variant, fold) for variant in VARIANTS for fold in FOLDS}:
        raise R2AuthorizationError("R2_INPUT_REUSE_FOLD_SET_DRIFT")
    if payload.get("fold_artifact_total_bytes") != total:
        raise R2AuthorizationError("R2_INPUT_REUSE_TOTAL_SIZE_DRIFT")
    authority_path_raw = payload.get("r1_static_auth_r6_path")
    if not isinstance(authority_path_raw, str) or not authority_path_raw:
        raise R2AuthorizationError("R2_INPUT_REUSE_R1_STATIC_PATH_MISSING")
    authority_path = _absolute_lexical(Path(authority_path_raw))
    expected_authority_path = _absolute_lexical(
        path.parent.parent
        / R1_NAMESPACE
        / R1_ARCHIVE_NAMESPACE
        / "STATIC_AUTH_READY.live_at_abort.json"
    )
    if authority_path != expected_authority_path:
        raise R2AuthorizationError("R2_INPUT_REUSE_R1_STATIC_PATH_DRIFT")
    authority, authority_sha = _load_json(authority_path, "R1_STATIC_AUTH_R6")
    if authority_sha != R1_STATIC_AUTH_R6_SHA256 or authority.get("status") != "STATIC_AUTH_READY":
        raise R2AuthorizationError("R2_INPUT_REUSE_R1_STATIC_AUTH_DRIFT")
    variants = authority.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(VARIANTS):
        raise R2AuthorizationError("R2_INPUT_REUSE_R1_STATIC_VARIANTS_DRIFT")
    authority_bindings: dict[tuple[str, int], dict[str, Any]] = {}
    for variant in VARIANTS:
        variant_payload = variants.get(variant)
        if not isinstance(variant_payload, Mapping):
            raise R2AuthorizationError(f"R2_INPUT_REUSE_R1_STATIC_VARIANT_INVALID={variant}")
        fold_inputs = variant_payload.get("fold_inputs")
        if not isinstance(fold_inputs, list) or len(fold_inputs) != 5:
            raise R2AuthorizationError(f"R2_INPUT_REUSE_R1_STATIC_FOLD_COUNT={variant}")
        for raw in fold_inputs:
            item = _require_mapping(raw, "R1_STATIC_FOLD_INPUT")
            fold = item.get("fold")
            if fold not in FOLDS or (variant, int(fold)) in authority_bindings:
                raise R2AuthorizationError("R2_INPUT_REUSE_R1_STATIC_FOLD_KEY_DRIFT")
            authority_bindings[(variant, int(fold))] = item
    if authority_bindings != observed_bindings:
        raise R2AuthorizationError("R2_INPUT_REUSE_VS_R1_STATIC_BINDING_DRIFT")
    return digest, observed_bindings


def _code_artifacts(code_root: Path) -> tuple[list[dict[str, Any]], str]:
    _assert_no_symlink_components(code_root, "CODE_ROOT")
    root = _absolute_lexical(code_root)
    records: list[dict[str, Any]] = []
    for relative in REQUIRED_CODE_ARTIFACTS:
        path = root / relative
        resolved = path.resolve(strict=False)
        if root not in resolved.parents:
            raise R2AuthorizationError(f"R2_CODE_PATH_ESCAPES_ROOT={relative}")
        digest, size = _sha256_regular(path, "CODE_ARTIFACT")
        records.append({"relative_path": relative, "sha256": digest, "size_bytes": size})
    encoded = json.dumps(records, separators=(",", ":"), sort_keys=True).encode()
    return records, hashlib.sha256(encoded).hexdigest()


def _code_tree_sha256(code_root: Path) -> str:
    """Mirror training_guard.code_tree_sha256 without importing the package."""

    _assert_no_symlink_components(code_root, "CODE_ROOT")
    root = _absolute_lexical(code_root)
    excluded_parts = {"__pycache__", ".pytest_cache", ".git"}
    excluded_suffixes = {".pyc", ".pyo"}
    sources = (root / "cc_hhgt", root / "scripts/v32_pipeline.py")
    found: list[tuple[str, Path]] = []
    for source in sources:
        if source.is_symlink():
            raise R2AuthorizationError(f"R2_CODE_TREE_SYMLINK_FORBIDDEN={source}")
        if source.is_file():
            found.append((source.relative_to(root).as_posix(), source))
            continue
        if not source.is_dir():
            raise R2AuthorizationError(f"R2_CODE_TREE_PATH_MISSING={source}")
        for path in source.rglob("*"):
            if path.is_symlink():
                raise R2AuthorizationError(f"R2_CODE_TREE_SYMLINK_FORBIDDEN={path}")
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in excluded_parts for part in relative.parts):
                continue
            if path.suffix.lower() in excluded_suffixes:
                continue
            found.append((relative.as_posix(), path))
    if not found:
        raise R2AuthorizationError("R2_CODE_TREE_EMPTY")
    digest = hashlib.sha256()
    for relative, path in sorted(found):
        content, _ = _sha256_regular(path, "CODE_TREE_FILE", allow_empty=True)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_patch_ready(
    path: Path, code_root: Path, config: Mapping[str, Any]
) -> tuple[str, str, str]:
    payload, receipt_sha = _load_json(path, "PATCH_READY")
    _require_exact_keys(
        payload,
        {
            "format",
            "status",
            "namespace",
            "created_at",
            "cpu_only",
            "gpu_visible",
            "code_root",
            "baseline_root",
            "archive_path",
            "archive_sha256",
            "manifest_path",
            "manifest_sha256",
            "bootstrap_verifier_path",
            "bootstrap_verifier_sha256",
            "external_code_archive_lock_path",
            "external_code_archive_lock_sha256",
            "code_tree_sha256",
            "deployment_contract_sha256",
            "overlay_code_imported",
            "overlay_code_executed",
            "prepared_inputs_touched",
            "formal_result_artifacts_used",
        },
        "PATCH_READY",
    )
    expected = {
        "format": PATCH_FORMAT,
        "status": "PATCH_READY",
        "namespace": NAMESPACE,
        "cpu_only": True,
        "gpu_visible": False,
        "code_root": str(code_root.resolve()),
        "overlay_code_imported": False,
        "overlay_code_executed": False,
        "prepared_inputs_touched": False,
        "formal_result_artifacts_used": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise R2AuthorizationError(f"R2_PATCH_READY_DRIFT={key}")
    _parse_utc(payload.get("created_at"), "PATCH_READY_CREATED_AT")
    expected_paths = {
        "baseline_root": str(
            _absolute_lexical(code_root.parent / "code.baseline_before_archive_r1")
        ),
        "archive_path": "/tmp/v32_g012_r2_code_archive_20260901_r1.tar.gz",
        "manifest_path": str(
            _absolute_lexical(path.parent / "CODE_ARCHIVE.MANIFEST.json")
        ),
        "bootstrap_verifier_path": str(
            _absolute_lexical(
                path.parent / "verify_v32_g012_r2_overlay_manifest.py"
            )
        ),
        "external_code_archive_lock_path": str(
            _absolute_lexical(path.parent / "CODE_ARCHIVE.LOCK.json")
        ),
    }
    for key, value in expected_paths.items():
        if payload.get(key) != value:
            raise R2AuthorizationError(f"R2_PATCH_READY_DRIFT={key}")
    bootstrap_verifier_sha, _ = _sha256_regular(
        Path(expected_paths["bootstrap_verifier_path"]),
        "EXTERNAL_BOOTSTRAP_VERIFIER",
    )
    archive_contract = _require_mapping(
        config.get("code_archive_contract"), "CODE_ARCHIVE_CONTRACT"
    )
    expected_archive_contract = {
        "status": EXTERNAL_ARCHIVE_LOCK_REQUIRED,
        "required_status_after_lock": EXTERNAL_ARCHIVE_LOCK_REQUIRED,
        "lock_format": EXTERNAL_ARCHIVE_LOCK_FORMAT,
        "lock_path": expected_paths["external_code_archive_lock_path"],
        "root_of_trust": "ARCHIVE_EXTERNAL_BOOTSTRAP_LOCK",
        "lock_sha256_authority": (
            "EXTERNAL_BOOTSTRAP_CONTROLLER_ENV_V32_R2_EXTERNAL_LOCK_SHA256"
        ),
        "lock_is_inside_code_archive": False,
        "archive_hash_embedded_in_archive": False,
        "manifest_hash_embedded_in_archive": False,
    }
    if archive_contract != expected_archive_contract:
        if archive_contract.get("status") == EXTERNAL_ARCHIVE_LOCK_BLOCKER:
            raise R2AuthorizationError(EXTERNAL_ARCHIVE_LOCK_BLOCKER)
        raise R2AuthorizationError("R2_CODE_ARCHIVE_EXTERNAL_LOCK_CONTRACT_DRIFT")
    external_lock, external_lock_sha = _validate_external_archive_lock(
        Path(expected_paths["external_code_archive_lock_path"]),
        bootstrap_root=path.parent,
    )
    if payload.get("external_code_archive_lock_sha256") != external_lock_sha:
        raise R2AuthorizationError("R2_EXTERNAL_ARCHIVE_LOCK_SHA256_DRIFT")
    if bootstrap_verifier_sha != external_lock["bootstrap_verifier_sha256"]:
        raise R2AuthorizationError("R2_EXTERNAL_BOOTSTRAP_VERIFIER_SHA256_DRIFT")
    patch_hash_fields = {
        "archive_sha256": external_lock["archive_sha256"],
        "manifest_sha256": external_lock["manifest_sha256"],
        "bootstrap_verifier_sha256": external_lock["bootstrap_verifier_sha256"],
    }
    for key, expected_value in patch_hash_fields.items():
        if payload.get(key) != expected_value:
            raise R2AuthorizationError(f"R2_PATCH_READY_DRIFT={key}")
    manifest_path = payload.get("manifest_path")
    if not isinstance(manifest_path, str):
        raise R2AuthorizationError("R2_PATCH_MANIFEST_PATH_MISSING")
    manifest, manifest_sha = _load_json(Path(manifest_path), "CODE_ARCHIVE_MANIFEST")
    if manifest_sha != external_lock["manifest_sha256"]:
        raise R2AuthorizationError("R2_CODE_ARCHIVE_MANIFEST_SHA_DRIFT")
    if (
        manifest.get("format") != "CC_HHGT_V3_2_G012_R2_CODE_ARCHIVE_MANIFEST_V1"
        or manifest.get("namespace") != NAMESPACE
        or manifest.get("archive_sha256") != external_lock["archive_sha256"]
    ):
        raise R2AuthorizationError("R2_CODE_ARCHIVE_MANIFEST_SCHEMA_DRIFT")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise R2AuthorizationError("R2_CODE_ARCHIVE_MANIFEST_FILES_INVALID")
    installed: set[str] = set()
    for raw in files:
        item = _require_mapping(raw, "CODE_ARCHIVE_FILE")
        if set(item) != {"path", "sha256", "size_bytes", "mode"}:
            raise R2AuthorizationError("R2_CODE_ARCHIVE_FILE_SCHEMA_DRIFT")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or "\\" in relative
            or ".." in Path(relative).parts
            or Path(relative).parts[0] not in ALLOWED_CODE_PREFIXES
            or relative in installed
        ):
            raise R2AuthorizationError(f"R2_CODE_ARCHIVE_FILE_PATH_INVALID={relative}")
        installed.add(relative)
        digest, size = _sha256_regular(
            code_root / Path(relative), "INSTALLED_CODE_FILE", allow_empty=True
        )
        if digest != item.get("sha256") or size != item.get("size_bytes"):
            raise R2AuthorizationError(f"R2_INSTALLED_CODE_FILE_DRIFT={relative}")
    try:
        candidates = list(_absolute_lexical(code_root).rglob("*"))
    except OSError as exc:
        raise R2AuthorizationError("R2_INSTALLED_CODE_TREE_ENUMERATION_FAILED") from exc
    actual_installed: set[str] = set()
    for candidate in candidates:
        relative_path = candidate.relative_to(_absolute_lexical(code_root))
        relative = relative_path.as_posix()
        try:
            candidate_stat = os.lstat(candidate)
        except OSError as exc:
            raise R2AuthorizationError(
                f"R2_INSTALLED_CODE_TREE_LSTAT_FAILED={relative}"
            ) from exc
        if stat.S_ISLNK(candidate_stat.st_mode) or getattr(
            os.path, "isjunction", lambda _: False
        )(candidate):
            raise R2AuthorizationError(
                f"R2_INSTALLED_CODE_TREE_SYMLINK_FORBIDDEN={relative}"
            )
        if relative_path.parts[0] not in ALLOWED_CODE_PREFIXES:
            raise R2AuthorizationError(
                f"R2_INSTALLED_CODE_TREE_PREFIX_FORBIDDEN={relative_path.parts[0]}"
            )
        if stat.S_ISREG(candidate_stat.st_mode):
            actual_installed.add(relative)
        elif not stat.S_ISDIR(candidate_stat.st_mode):
            raise R2AuthorizationError(
                f"R2_INSTALLED_CODE_TREE_SPECIAL_FILE={relative}"
            )
    if actual_installed != installed:
        raise R2AuthorizationError("R2_INSTALLED_CODE_TREE_FILE_SET_DRIFT")
    _, observed_deployment_sha = _code_artifacts(code_root)
    observed_tree_sha = _code_tree_sha256(code_root)
    if payload.get("deployment_contract_sha256") != observed_deployment_sha:
        raise R2AuthorizationError("R2_PATCH_DEPLOYMENT_CONTRACT_SHA_DRIFT")
    if payload.get("code_tree_sha256") != observed_tree_sha:
        raise R2AuthorizationError("R2_PATCH_CODE_TREE_SHA_DRIFT")
    return receipt_sha, observed_tree_sha, observed_deployment_sha


def _encoded_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_new_bytes(path: Path, content: bytes, label: str) -> str:
    """Create one transaction-owned artifact without following existing paths."""

    target = _absolute_lexical(path)
    _assert_no_symlink_components(target.parent, label)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(target, flags, 0o600)
    except OSError as exc:
        raise R2AuthorizationError(f"R2_{label}_CREATE_FAILED={target}") from exc
    try:
        offset = 0
        while offset < len(content):
            offset += os.write(descriptor, content[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(content).hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_id(variant: str) -> str:
    return f"v32-g012-{variant.lower()}-paid-gpu-20260901-r2"


def _task_rows(variant: str) -> list[dict[str, str]]:
    run_id = _run_id(variant)
    return [
        {
            "task_id": f"{run_id}|PATIENT_FOLD_{fold}|CC-HHGT|{FORMAL_SEED}",
            "run_id": run_id,
            "task_type": "CC_HHGT_PATIENT_FOLD",
            "model": "CC-HHGT",
            "patient_fold": str(fold),
            "seed": str(FORMAL_SEED),
            "owner": PAID_ENDPOINT,
            "hardware_class": PAID_HARDWARE,
            "status": "PENDING",
            "blocked_reason": "",
            "paid_task": "true",
        }
        for fold in FOLDS
    ]


def _task_manifest_bytes(variant: str) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(TASK_COLUMNS),
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(_task_rows(variant))
    return stream.getvalue().encode("utf-8")


def _render_variant_config(
    template_raw: bytes,
    *,
    variant: str,
    prepared_parent: Path,
    run_parent: Path,
) -> tuple[bytes, dict[str, Any]]:
    try:
        text = template_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise R2AuthorizationError("R2_CONFIG_TEMPLATE_UTF8_INVALID") from exc
    if "__GRAPH_VARIANT__" not in text:
        raise R2AuthorizationError("R2_CONFIG_TEMPLATE_VARIANT_MARKER_MISSING")
    rendered = text.replace("__GRAPH_VARIANT__", variant)
    if "__GRAPH_VARIANT__" in rendered:
        raise R2AuthorizationError("R2_CONFIG_VARIANT_MARKER_REMAINS")
    try:
        payload = yaml.safe_load(rendered)
    except yaml.YAMLError as exc:
        raise R2AuthorizationError(f"R2_VARIANT_CONFIG_YAML_INVALID={variant}") from exc
    config = _require_mapping(payload, "VARIANT_CONFIG")
    task = _require_mapping(config.get("task_contract"), "VARIANT_TASK_CONTRACT")
    training_io = _require_mapping(config.get("training_io"), "VARIANT_TRAINING_IO")
    control = _require_mapping(config.get("execution_control"), "VARIANT_EXECUTION_CONTROL")
    expected = {
        "graph_variant": variant,
        "prepared_fold_pattern": str(_absolute_lexical(prepared_parent))
        + f"/{variant}/PATIENT_FOLD_{{fold}}.pt",
        "output_root": str(_absolute_lexical(run_parent)) + f"/{variant}/training",
    }
    if task.get("graph_variant") != expected["graph_variant"]:
        raise R2AuthorizationError(f"R2_VARIANT_CONFIG_GRAPH_DRIFT={variant}")
    for key in ("prepared_fold_pattern", "output_root"):
        if training_io.get(key) != expected[key]:
            raise R2AuthorizationError(f"R2_VARIANT_CONFIG_IO_DRIFT={variant}:{key}")
    for key, value in {
        "execution_mode": "TRAINING",
        "training_authorized": True,
        "paid_enabled": True,
        "candidate_only": True,
        "overwrite_formal_v32": False,
    }.items():
        if control.get(key) != value:
            raise R2AuthorizationError(f"R2_VARIANT_CONFIG_CONTROL_DRIFT={variant}:{key}")
    return rendered.encode("utf-8"), config


def _input_manifest_payload(
    *,
    variant: str,
    input_reuse_sha256: str,
    input_bindings: Mapping[tuple[str, int], Mapping[str, Any]],
    prepared_parent: Path,
) -> dict[str, Any]:
    folds = [dict(input_bindings[(variant, fold)]) for fold in FOLDS]
    return {
        "format": INPUT_MANIFEST_FORMAT,
        "status": "R2_INPUT_MANIFEST_READY",
        "namespace": NAMESPACE,
        "run_id": _run_id(variant),
        "graph_variant": variant,
        "prepared_parent": str(_absolute_lexical(prepared_parent)),
        "source_input_reuse_ready_sha256": input_reuse_sha256,
        "r1_static_auth_r6_sha256": R1_STATIC_AUTH_R6_SHA256,
        "fold_inputs": folds,
    }


def _approval_payload(
    *,
    variant: str,
    artifact_hashes: Mapping[str, str],
    config: Mapping[str, Any],
    prerequisite_hashes: Mapping[str, str],
) -> dict[str, Any]:
    control = _require_mapping(config.get("execution_control"), "APPROVAL_CONTROL")
    return {
        "approval_format": APPROVAL_FORMAT,
        "training_authorized": True,
        "namespace": NAMESPACE,
        "run_id": _run_id(variant),
        "endpoint_id": PAID_ENDPOINT,
        "hardware_class": PAID_HARDWARE,
        "approved_task_ids": [row["task_id"] for row in _task_rows(variant)],
        "authorized_trainer": FORMAL_TRAINER,
        "paid_enabled": True,
        "max_paid_hours": control.get("max_paid_hours"),
        "max_cost_cny": control.get("max_cost_cny"),
        "artifact_hashes": dict(artifact_hashes),
        "prerequisite_hashes": dict(prerequisite_hashes),
    }


def _runtime_snapshot(*, code_tree_sha256: str) -> dict[str, Any]:
    distributions: dict[str, str] = {}
    missing: list[str] = []
    for distribution in ("torch", "torch-geometric", "PyYAML"):
        try:
            distributions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing:
        raise R2AuthorizationError(
            "R2_RUNTIME_DISTRIBUTIONS_MISSING=" + ",".join(sorted(missing))
        )
    return {
        "format": RUNTIME_READY_FORMAT,
        "status": RUNTIME_READY_STATUS,
        "namespace": NAMESPACE,
        "cpu_only": True,
        "gpu_visible": False,
        "cuda_api_inspected": False,
        "torch_imported": False,
        "metadata_api": "importlib.metadata",
        "python_executable": str(_absolute_lexical(Path(sys.executable))),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "distributions": distributions,
        "required_distributions_present": True,
        "code_tree_sha256": code_tree_sha256,
    }


def _control_bindings(code_root: Path) -> dict[str, dict[str, str]]:
    relative_paths = {
        "launcher": "scripts/server_launch_v32_g012_paid_gpu_20260901_r2.sh",
        "supervisor": "scripts/local_supervise_compshare_paid_gpu_20260901_r2.ps1",
        "finalizer": "scripts/cloud_finalize_v32_g012_no_gpu_20260901_r2.sh",
    }
    bindings: dict[str, dict[str, str]] = {}
    for name, relative in relative_paths.items():
        path = _absolute_lexical(code_root / relative)
        digest, _ = _sha256_regular(path, f"{name.upper()}_CONTROL")
        bindings[name] = {"path": str(path), "sha256": digest}
    return bindings


def _fresh_artifact_paths(root: Path, variant: str) -> dict[str, Path]:
    variant_root = root / variant
    return {
        "config": variant_root / "CONFIG.yaml",
        "task_manifest": variant_root / "TASK_MANIFEST.tsv",
        "input_manifest": variant_root / "INPUT_MANIFEST.json",
        "approval": variant_root / "TRAINING_APPROVAL.json",
    }


def _materialize_fresh_authority(
    *,
    bootstrap: Path,
    config_path: Path,
    code_root: Path,
    prepared_parent: Path,
    run_parent: Path,
    input_reuse_sha256: str,
    input_bindings: Mapping[tuple[str, int], Mapping[str, Any]],
    code_tree_sha256: str,
    deployment_contract_sha256: str,
    formal_receipts: Mapping[str, Any],
    prerequisite_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    root = _absolute_lexical(bootstrap / "fresh_authority")
    staging = _absolute_lexical(bootstrap / ".fresh_authority.partial")
    for candidate, label in ((root, "FRESH_AUTHORITY"), (staging, "FRESH_STAGING")):
        try:
            os.lstat(candidate)
        except FileNotFoundError:
            pass
        else:
            raise R2AuthorizationError(f"R2_{label}_ALREADY_EXISTS={candidate}")
    _assert_no_symlink_components(bootstrap, "FRESH_BOOTSTRAP")
    try:
        os.mkdir(staging, 0o700)
    except OSError as exc:
        raise R2AuthorizationError(f"R2_FRESH_STAGING_CREATE_FAILED={staging}") from exc

    template_raw, template_sha, _ = _read_regular(config_path, "CONFIG_TEMPLATE")
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        paths = _fresh_artifact_paths(staging, variant)
        os.mkdir(paths["config"].parent, 0o700)
        config_raw, rendered_config = _render_variant_config(
            template_raw,
            variant=variant,
            prepared_parent=prepared_parent,
            run_parent=run_parent,
        )
        config_sha = _write_new_bytes(paths["config"], config_raw, "VARIANT_CONFIG")
        task_raw = _task_manifest_bytes(variant)
        task_sha = _write_new_bytes(
            paths["task_manifest"], task_raw, "TASK_MANIFEST"
        )
        input_payload = _input_manifest_payload(
            variant=variant,
            input_reuse_sha256=input_reuse_sha256,
            input_bindings=input_bindings,
            prepared_parent=prepared_parent,
        )
        input_sha = _write_new_bytes(
            paths["input_manifest"],
            _encoded_json(input_payload),
            "INPUT_MANIFEST",
        )
        artifact_hashes = {
            "code_sha256": code_tree_sha256,
            "config_sha256": config_sha,
            "input_manifest_sha256": input_sha,
            "task_manifest_sha256": task_sha,
        }
        approval = _approval_payload(
            variant=variant,
            artifact_hashes=artifact_hashes,
            config=rendered_config,
            prerequisite_hashes=prerequisite_hashes,
        )
        approval_sha = _write_new_bytes(
            paths["approval"], _encoded_json(approval), "TRAINING_APPROVAL"
        )
        _fsync_directory(paths["config"].parent)
        variants[variant] = {
            "run_id": _run_id(variant),
            "task_count": 5,
            "task_ids": [row["task_id"] for row in _task_rows(variant)],
            "config": {"path": str(root / variant / "CONFIG.yaml"), "sha256": config_sha},
            "task_manifest": {
                "path": str(root / variant / "TASK_MANIFEST.tsv"),
                "sha256": task_sha,
            },
            "input_manifest": {
                "path": str(root / variant / "INPUT_MANIFEST.json"),
                "sha256": input_sha,
            },
            "approval": {
                "path": str(root / variant / "TRAINING_APPROVAL.json"),
                "sha256": approval_sha,
            },
        }

    runtime_payload = _runtime_snapshot(code_tree_sha256=code_tree_sha256)
    runtime_path = staging / "RUNTIME_READY.json"
    runtime_sha = _write_new_bytes(
        runtime_path, _encoded_json(runtime_payload), "RUNTIME_READY"
    )
    controls = _control_bindings(code_root)
    index = {
        "format": FRESH_AUTHORITY_FORMAT,
        "status": "R2_FRESH_AUTHORITY_READY_CPU_ONLY",
        "namespace": NAMESPACE,
        "cpu_only": True,
        "gpu_visible": False,
        "authorized_trainer": FORMAL_TRAINER,
        "formal_seed": FORMAL_SEED,
        "config_template": {"path": str(_absolute_lexical(config_path)), "sha256": template_sha},
        "prepared_parent": str(_absolute_lexical(prepared_parent)),
        "run_parent": str(_absolute_lexical(run_parent)),
        "source_input_reuse_ready_sha256": input_reuse_sha256,
        "r1_static_auth_r6_sha256": R1_STATIC_AUTH_R6_SHA256,
        "code_tree_sha256": code_tree_sha256,
        "deployment_contract_sha256": deployment_contract_sha256,
        "formal_gate_receipts": dict(formal_receipts),
        "prerequisite_hashes": dict(prerequisite_hashes),
        "runtime_ready": {
            "path": str(root / "RUNTIME_READY.json"),
            "sha256": runtime_sha,
        },
        "controls": controls,
        "variants": variants,
    }
    index_path = staging / "FRESH_AUTHORITY.json"
    _write_new_bytes(index_path, _encoded_json(index), "FRESH_AUTHORITY_INDEX")
    _fsync_directory(staging)
    try:
        os.replace(staging, root)
    except OSError as exc:
        raise R2AuthorizationError("R2_FRESH_AUTHORITY_COMMIT_FAILED") from exc
    _fsync_directory(bootstrap)
    index_sha, _ = _sha256_regular(root / "FRESH_AUTHORITY.json", "FRESH_AUTHORITY_INDEX")
    return index, index_sha


def _parse_task_manifest(raw: bytes, variant: str) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t", strict=True)
        if tuple(reader.fieldnames or ()) != TASK_COLUMNS:
            raise R2AuthorizationError(f"R2_TASK_MANIFEST_HEADER_DRIFT={variant}")
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise R2AuthorizationError(f"R2_TASK_MANIFEST_INVALID={variant}") from exc
    if rows != _task_rows(variant):
        raise R2AuthorizationError(f"R2_TASK_MANIFEST_SEMANTIC_DRIFT={variant}")
    return rows


def _validate_fresh_authority(
    *,
    bootstrap: Path,
    config_path: Path,
    code_root: Path,
    prepared_parent: Path,
    run_parent: Path,
    input_reuse_sha256: str,
    input_bindings: Mapping[tuple[str, int], Mapping[str, Any]],
    code_tree_sha256: str,
    deployment_contract_sha256: str,
    formal_receipts: Mapping[str, Any],
    prerequisite_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    root = _absolute_lexical(bootstrap / "fresh_authority")
    index_path = root / "FRESH_AUTHORITY.json"
    index, index_sha = _load_json(index_path, "FRESH_AUTHORITY_INDEX")
    template_raw, template_sha, _ = _read_regular(config_path, "CONFIG_TEMPLATE")
    controls = _control_bindings(code_root)
    expected_top = {
        "format": FRESH_AUTHORITY_FORMAT,
        "status": "R2_FRESH_AUTHORITY_READY_CPU_ONLY",
        "namespace": NAMESPACE,
        "cpu_only": True,
        "gpu_visible": False,
        "authorized_trainer": FORMAL_TRAINER,
        "formal_seed": FORMAL_SEED,
        "config_template": {"path": str(_absolute_lexical(config_path)), "sha256": template_sha},
        "prepared_parent": str(_absolute_lexical(prepared_parent)),
        "run_parent": str(_absolute_lexical(run_parent)),
        "source_input_reuse_ready_sha256": input_reuse_sha256,
        "r1_static_auth_r6_sha256": R1_STATIC_AUTH_R6_SHA256,
        "code_tree_sha256": code_tree_sha256,
        "deployment_contract_sha256": deployment_contract_sha256,
        "formal_gate_receipts": dict(formal_receipts),
        "prerequisite_hashes": dict(prerequisite_hashes),
        "runtime_ready": index.get("runtime_ready"),
        "controls": controls,
    }
    _require_exact_keys(
        index,
        set(expected_top) | {"variants"},
        "FRESH_AUTHORITY_INDEX",
    )
    for key, value in expected_top.items():
        if index.get(key) != value:
            raise R2AuthorizationError(f"R2_FRESH_AUTHORITY_DRIFT={key}")
    runtime_binding = _require_mapping(index.get("runtime_ready"), "RUNTIME_BINDING")
    _require_exact_keys(runtime_binding, {"path", "sha256"}, "RUNTIME_BINDING")
    expected_runtime_path = root / "RUNTIME_READY.json"
    if runtime_binding.get("path") != str(expected_runtime_path):
        raise R2AuthorizationError("R2_RUNTIME_READY_PATH_DRIFT")
    runtime, runtime_sha = _load_json(expected_runtime_path, "RUNTIME_READY")
    if runtime_sha != runtime_binding.get("sha256"):
        raise R2AuthorizationError("R2_RUNTIME_READY_SHA256_DRIFT")
    if runtime != _runtime_snapshot(code_tree_sha256=code_tree_sha256):
        raise R2AuthorizationError("R2_RUNTIME_READY_SEMANTIC_DRIFT")

    variants = _require_mapping(index.get("variants"), "FRESH_VARIANTS")
    if set(variants) != set(VARIANTS):
        raise R2AuthorizationError("R2_FRESH_VARIANT_SET_DRIFT")
    expected_files = {"FRESH_AUTHORITY.json", "RUNTIME_READY.json"}
    for variant in VARIANTS:
        variant_binding = _require_mapping(variants.get(variant), "FRESH_VARIANT")
        paths = _fresh_artifact_paths(root, variant)
        config_raw, config_sha, _ = _read_regular(paths["config"], "VARIANT_CONFIG")
        expected_config_raw, rendered_config = _render_variant_config(
            template_raw,
            variant=variant,
            prepared_parent=prepared_parent,
            run_parent=run_parent,
        )
        if config_raw != expected_config_raw:
            raise R2AuthorizationError(f"R2_VARIANT_CONFIG_BYTES_DRIFT={variant}")
        task_raw, task_sha, _ = _read_regular(paths["task_manifest"], "TASK_MANIFEST")
        rows = _parse_task_manifest(task_raw, variant)
        input_payload, input_sha = _load_json(paths["input_manifest"], "INPUT_MANIFEST")
        expected_input = _input_manifest_payload(
            variant=variant,
            input_reuse_sha256=input_reuse_sha256,
            input_bindings=input_bindings,
            prepared_parent=prepared_parent,
        )
        if input_payload != expected_input:
            raise R2AuthorizationError(f"R2_INPUT_MANIFEST_DRIFT={variant}")
        approval, approval_sha = _load_json(paths["approval"], "TRAINING_APPROVAL")
        artifact_hashes = {
            "code_sha256": code_tree_sha256,
            "config_sha256": config_sha,
            "input_manifest_sha256": input_sha,
            "task_manifest_sha256": task_sha,
        }
        expected_approval = _approval_payload(
            variant=variant,
            artifact_hashes=artifact_hashes,
            config=rendered_config,
            prerequisite_hashes=prerequisite_hashes,
        )
        if approval != expected_approval:
            if approval.get("authorized_trainer") != FORMAL_TRAINER:
                raise R2AuthorizationError(f"R2_AUTHORIZED_TRAINER_DRIFT={variant}")
            raise R2AuthorizationError(f"R2_TRAINING_APPROVAL_DRIFT={variant}")
        expected_variant = {
            "run_id": _run_id(variant),
            "task_count": 5,
            "task_ids": [row["task_id"] for row in rows],
            "config": {"path": str(paths["config"]), "sha256": config_sha},
            "task_manifest": {"path": str(paths["task_manifest"]), "sha256": task_sha},
            "input_manifest": {"path": str(paths["input_manifest"]), "sha256": input_sha},
            "approval": {"path": str(paths["approval"]), "sha256": approval_sha},
        }
        if variant_binding != expected_variant:
            raise R2AuthorizationError(f"R2_FRESH_VARIANT_BINDING_DRIFT={variant}")
        expected_files.update(
            {
                f"{variant}/CONFIG.yaml",
                f"{variant}/TASK_MANIFEST.tsv",
                f"{variant}/INPUT_MANIFEST.json",
                f"{variant}/TRAINING_APPROVAL.json",
            }
        )
    observed_files: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        observed = os.lstat(candidate)
        if stat.S_ISLNK(observed.st_mode) or getattr(
            os.path, "isjunction", lambda _: False
        )(candidate):
            raise R2AuthorizationError(f"R2_FRESH_AUTHORITY_SYMLINK_FORBIDDEN={relative}")
        if stat.S_ISREG(observed.st_mode):
            observed_files.add(relative)
        elif not stat.S_ISDIR(observed.st_mode):
            raise R2AuthorizationError(f"R2_FRESH_AUTHORITY_SPECIAL_FILE={relative}")
    if observed_files != expected_files:
        raise R2AuthorizationError("R2_FRESH_AUTHORITY_FILE_SET_DRIFT")
    return index, index_sha


def materialize_static_authorization(
    *,
    receipt_path: Path,
    config_path: Path,
    code_root: Path,
    prepared_parent: Path,
    run_parent: Path,
    bootstrap_root: Path,
) -> dict[str, Any]:
    gate = validate_estimator_config(
        config_path, prepared_parent=prepared_parent, run_parent=run_parent
    )
    formal_receipts = _formal_receipt_specs(
        gate["config"], code_root=code_root, bootstrap_root=bootstrap_root
    )
    formal_prerequisite_hashes = _formal_prerequisite_hashes(formal_receipts)
    bootstrap = _absolute_lexical(bootstrap_root)
    _assert_no_symlink_components(bootstrap, "BOOTSTRAP_ROOT")
    if bootstrap.name != NAMESPACE:
        raise R2AuthorizationError("R2_BOOTSTRAP_NAMESPACE_DRIFT")
    target = _absolute_lexical(receipt_path)
    if target != bootstrap / "STATIC_AUTH_READY.json":
        raise R2AuthorizationError("R2_STATIC_AUTH_TARGET_DRIFT")
    try:
        os.lstat(target)
    except FileNotFoundError:
        pass
    else:
        raise R2AuthorizationError("R2_STATIC_AUTH_ALREADY_EXISTS")

    _assert_fresh_output_absent(run_parent)

    aborted = bootstrap.parent / R1_NAMESPACE / "ABORTED.json"
    cpu_ready = bootstrap.parent / R1_NAMESPACE / "CPU_READY"
    input_reuse = bootstrap / "INPUT_REUSE_READY.json"
    patch_ready = bootstrap / "PATCH_READY.json"
    aborted_sha = _validate_r1_aborted(aborted)
    cpu_ready_payload, cpu_ready_sha = _validate_cpu_ready(cpu_ready)
    input_sha, input_bindings = _validate_input_reuse(input_reuse, prepared_parent)
    patch_sha, code_tree_sha, deployment_sha = _validate_patch_ready(
        patch_ready, code_root, gate["config"]
    )
    code_records, observed_deployment_sha = _code_artifacts(code_root)
    observed_tree_sha = _code_tree_sha256(code_root)
    if observed_deployment_sha != deployment_sha or observed_tree_sha != code_tree_sha:
        raise R2AuthorizationError("R2_CODE_CHANGED_DURING_AUTH")

    prerequisite_hashes = {
        "r1_aborted_sha256": aborted_sha,
        "cpu_ready_sha256": cpu_ready_sha,
        "input_reuse_ready_sha256": input_sha,
        "patch_ready_sha256": patch_sha,
        **formal_prerequisite_hashes,
    }
    fresh_root = bootstrap / "fresh_authority"
    try:
        observed_fresh = os.lstat(fresh_root)
    except FileNotFoundError:
        fresh_authority, fresh_authority_sha = _materialize_fresh_authority(
            bootstrap=bootstrap,
            config_path=config_path,
            code_root=code_root,
            prepared_parent=prepared_parent,
            run_parent=run_parent,
            input_reuse_sha256=input_sha,
            input_bindings=input_bindings,
            code_tree_sha256=code_tree_sha,
            deployment_contract_sha256=deployment_sha,
            formal_receipts=formal_receipts,
            prerequisite_hashes=prerequisite_hashes,
        )
    else:
        if (
            not stat.S_ISDIR(observed_fresh.st_mode)
            or stat.S_ISLNK(observed_fresh.st_mode)
            or getattr(os.path, "isjunction", lambda _: False)(fresh_root)
        ):
            raise R2AuthorizationError("R2_FRESH_AUTHORITY_ROOT_INVALID")
        fresh_authority, fresh_authority_sha = _validate_fresh_authority(
            bootstrap=bootstrap,
            config_path=config_path,
            code_root=code_root,
            prepared_parent=prepared_parent,
            run_parent=run_parent,
            input_reuse_sha256=input_sha,
            input_bindings=input_bindings,
            code_tree_sha256=code_tree_sha,
            deployment_contract_sha256=deployment_sha,
            formal_receipts=formal_receipts,
            prerequisite_hashes=prerequisite_hashes,
        )

    receipt = {
        "format": STATIC_FORMAT,
        "status": "STATIC_AUTH_READY",
        "namespace": NAMESPACE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "gpu_visible": False,
        "training_authorized": True,
        "authorized_trainer": FORMAL_TRAINER,
        "estimator_status": ESTIMATOR_READY,
        "schedule_mode": gate["schedule_mode"],
        "config_path": str(config_path.resolve()),
        "config_sha256": gate["config_sha256"],
        "code_root": str(code_root.resolve()),
        "code_tree_sha256": code_tree_sha,
        "deployment_contract_sha256": deployment_sha,
        "code_artifacts": code_records,
        "prepared_parent": str(prepared_parent.resolve()),
        "run_parent": str(run_parent.resolve()),
        "r1_aborted_path": str(aborted),
        "r1_aborted_sha256": aborted_sha,
        "cpu_ready_path": str(cpu_ready),
        "cpu_ready_sha256": cpu_ready_sha,
        "cpu_ready": cpu_ready_payload,
        "input_reuse_ready_path": str(input_reuse),
        "input_reuse_ready_sha256": input_sha,
        "patch_ready_path": str(patch_ready),
        "patch_ready_sha256": patch_sha,
        "formal_gate_receipts": formal_receipts,
        "prerequisite_hashes": prerequisite_hashes,
        "fresh_authority_path": str(fresh_root / "FRESH_AUTHORITY.json"),
        "fresh_authority_sha256": fresh_authority_sha,
        "fresh_authority": fresh_authority,
        "launcher_path": fresh_authority["controls"]["launcher"]["path"],
        "launcher_sha256": fresh_authority["controls"]["launcher"]["sha256"],
        "supervisor_path": fresh_authority["controls"]["supervisor"]["path"],
        "supervisor_sha256": fresh_authority["controls"]["supervisor"]["sha256"],
        "finalizer_path": fresh_authority["controls"]["finalizer"]["path"],
        "finalizer_sha256": fresh_authority["controls"]["finalizer"]["sha256"],
        "source_input_archive_sha256": SOURCE_INPUT_ARCHIVE_SHA256,
        "variants": list(VARIANTS),
        "folds": list(FOLDS),
        "fold_artifact_count": 15,
        "formal_config_count": 3,
        "formal_task_count": 15,
        "formal_input_manifest_count": 3,
        "formal_approval_count": 3,
        "prepared_inputs_retransferred": False,
        "formal_result_artifacts_used": False,
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if R1_RESULT_NAMESPACE in encoded:
        raise R2AuthorizationError("R2_STATIC_AUTH_REFERENCES_R1_FORMAL_RESULTS")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
    try:
        os.lstat(partial)
    except FileNotFoundError:
        pass
    else:
        raise R2AuthorizationError(f"R2_STATIC_AUTH_PARTIAL_PRESENT={partial}")
    _write_new_bytes(partial, encoded.encode("utf-8"), "STATIC_AUTH_PARTIAL")
    try:
        os.link(partial, target, follow_symlinks=False)
    except OSError as exc:
        raise R2AuthorizationError("R2_STATIC_AUTH_EXCLUSIVE_COMMIT_FAILED") from exc
    os.unlink(partial)
    _fsync_directory(target.parent)
    return receipt


def validate_static_authorization(
    *,
    receipt_path: Path,
    config_path: Path,
    code_root: Path,
    prepared_parent: Path,
    run_parent: Path,
    bootstrap_root: Path,
    allow_resume: bool = False,
) -> dict[str, Any]:
    gate = validate_estimator_config(
        config_path, prepared_parent=prepared_parent, run_parent=run_parent
    )
    formal_receipts = _formal_receipt_specs(
        gate["config"], code_root=code_root, bootstrap_root=bootstrap_root
    )
    formal_prerequisite_hashes = _formal_prerequisite_hashes(formal_receipts)
    bootstrap = _absolute_lexical(bootstrap_root)
    _assert_no_symlink_components(bootstrap, "BOOTSTRAP_ROOT")
    if bootstrap.name != NAMESPACE:
        raise R2AuthorizationError("R2_BOOTSTRAP_NAMESPACE_DRIFT")
    if _absolute_lexical(receipt_path) != bootstrap / "STATIC_AUTH_READY.json":
        raise R2AuthorizationError("R2_STATIC_AUTH_TARGET_DRIFT")
    payload, static_auth_sha = _load_json(receipt_path, "STATIC_AUTH_READY")
    if R1_RESULT_NAMESPACE in json.dumps(payload, sort_keys=True):
        raise R2AuthorizationError("R2_STATIC_AUTH_REFERENCES_R1_FORMAL_RESULTS")

    aborted = bootstrap.parent / R1_NAMESPACE / "ABORTED.json"
    cpu_ready = bootstrap.parent / R1_NAMESPACE / "CPU_READY"
    input_reuse = bootstrap / "INPUT_REUSE_READY.json"
    patch_ready = bootstrap / "PATCH_READY.json"
    aborted_sha = _validate_r1_aborted(aborted)
    cpu_ready_payload, cpu_ready_sha = _validate_cpu_ready(cpu_ready)
    input_sha, input_bindings = _validate_input_reuse(input_reuse, prepared_parent)
    patch_sha, code_tree_sha, deployment_sha = _validate_patch_ready(
        patch_ready, code_root, gate["config"]
    )
    records, observed_deployment_sha = _code_artifacts(code_root)
    observed_tree_sha = _code_tree_sha256(code_root)
    if deployment_sha != observed_deployment_sha or code_tree_sha != observed_tree_sha:
        raise R2AuthorizationError("R2_CODE_CHANGED_DURING_VALIDATION")

    prerequisite_hashes = {
        "r1_aborted_sha256": aborted_sha,
        "cpu_ready_sha256": cpu_ready_sha,
        "input_reuse_ready_sha256": input_sha,
        "patch_ready_sha256": patch_sha,
        **formal_prerequisite_hashes,
    }
    fresh_authority, fresh_authority_sha = _validate_fresh_authority(
        bootstrap=bootstrap,
        config_path=config_path,
        code_root=code_root,
        prepared_parent=prepared_parent,
        run_parent=run_parent,
        input_reuse_sha256=input_sha,
        input_bindings=input_bindings,
        code_tree_sha256=code_tree_sha,
        deployment_contract_sha256=deployment_sha,
        formal_receipts=formal_receipts,
        prerequisite_hashes=prerequisite_hashes,
    )

    expected = {
        "format": STATIC_FORMAT,
        "status": "STATIC_AUTH_READY",
        "namespace": NAMESPACE,
        "cpu_only": True,
        "gpu_visible": False,
        "training_authorized": True,
        "authorized_trainer": FORMAL_TRAINER,
        "estimator_status": ESTIMATOR_READY,
        "schedule_mode": gate["schedule_mode"],
        "config_path": str(config_path.resolve()),
        "config_sha256": gate["config_sha256"],
        "code_root": str(code_root.resolve()),
        "code_tree_sha256": code_tree_sha,
        "deployment_contract_sha256": deployment_sha,
        "code_artifacts": records,
        "prepared_parent": str(prepared_parent.resolve()),
        "run_parent": str(run_parent.resolve()),
        "r1_aborted_path": str(aborted),
        "r1_aborted_sha256": aborted_sha,
        "cpu_ready_path": str(cpu_ready),
        "cpu_ready_sha256": cpu_ready_sha,
        "cpu_ready": cpu_ready_payload,
        "input_reuse_ready_path": str(input_reuse),
        "input_reuse_ready_sha256": input_sha,
        "patch_ready_path": str(patch_ready),
        "patch_ready_sha256": patch_sha,
        "formal_gate_receipts": formal_receipts,
        "prerequisite_hashes": prerequisite_hashes,
        "fresh_authority_path": str(
            bootstrap / "fresh_authority/FRESH_AUTHORITY.json"
        ),
        "fresh_authority_sha256": fresh_authority_sha,
        "fresh_authority": fresh_authority,
        "launcher_path": fresh_authority["controls"]["launcher"]["path"],
        "launcher_sha256": fresh_authority["controls"]["launcher"]["sha256"],
        "supervisor_path": fresh_authority["controls"]["supervisor"]["path"],
        "supervisor_sha256": fresh_authority["controls"]["supervisor"]["sha256"],
        "finalizer_path": fresh_authority["controls"]["finalizer"]["path"],
        "finalizer_sha256": fresh_authority["controls"]["finalizer"]["sha256"],
        "source_input_archive_sha256": SOURCE_INPUT_ARCHIVE_SHA256,
        "variants": list(VARIANTS),
        "folds": list(FOLDS),
        "fold_artifact_count": 15,
        "formal_config_count": 3,
        "formal_task_count": 15,
        "formal_input_manifest_count": 3,
        "formal_approval_count": 3,
        "prepared_inputs_retransferred": False,
        "formal_result_artifacts_used": False,
    }
    _require_exact_keys(payload, set(expected) | {"created_at"}, "STATIC_AUTH_READY")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise R2AuthorizationError(f"R2_STATIC_AUTH_DRIFT={key}")
    _parse_utc(payload.get("created_at"), "STATIC_AUTH_CREATED_AT")
    _validate_fresh_or_resume_state(
        run_parent,
        allow_resume=allow_resume,
        static_auth_sha256=static_auth_sha,
        static_payload=payload,
    )
    return payload


def _defaults() -> dict[str, Path]:
    project = Path("./data/CancerLncAtlas")
    code = project / "runtime/tools" / NAMESPACE / "code"
    bootstrap = project / "runtime/bootstrap" / NAMESPACE
    return {
        "config": code / "config/model_v3_2_g012_paid_gpu_20260901_r2.yaml",
        "code_root": code,
        "prepared_parent": project
        / "inputs/v32_g012_patient_first_20260830_r1/formal_prepared_20260830_r3_affine_pyg280_localtorch",
        "run_parent": project / "results" / RUN_NAMESPACE,
        "bootstrap_root": bootstrap,
        "receipt": bootstrap / "STATIC_AUTH_READY.json",
    }


def main(argv: Sequence[str] | None = None) -> int:
    defaults = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=defaults["config"])
    parser.add_argument("--code-root", type=Path, default=defaults["code_root"])
    parser.add_argument(
        "--prepared-parent", type=Path, default=defaults["prepared_parent"]
    )
    parser.add_argument("--run-parent", type=Path, default=defaults["run_parent"])
    parser.add_argument(
        "--bootstrap-root", type=Path, default=defaults["bootstrap_root"]
    )
    parser.add_argument("--receipt", type=Path, default=defaults["receipt"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--config-only", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    parser.add_argument("--allow-resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.config_only:
            payload = validate_estimator_config(
                args.config,
                prepared_parent=args.prepared_parent,
                run_parent=args.run_parent,
            )
            _formal_receipt_specs(
                payload["config"],
                code_root=args.code_root,
                bootstrap_root=args.bootstrap_root,
            )
            print(f"PASS_R2_ESTIMATOR_AUTHORIZED={payload['schedule_mode']}")
        elif args.materialize:
            materialize_static_authorization(
                receipt_path=args.receipt,
                config_path=args.config,
                code_root=args.code_root,
                prepared_parent=args.prepared_parent,
                run_parent=args.run_parent,
                bootstrap_root=args.bootstrap_root,
            )
            print(f"PASS_R2_STATIC_AUTH_MATERIALIZED={args.receipt.resolve()}")
        else:
            validate_static_authorization(
                receipt_path=args.receipt,
                config_path=args.config,
                code_root=args.code_root,
                prepared_parent=args.prepared_parent,
                run_parent=args.run_parent,
                bootstrap_root=args.bootstrap_root,
                allow_resume=args.allow_resume,
            )
            print(f"PASS_R2_STATIC_AUTH_READY={args.receipt.resolve()}")
        return 0
    except R2AuthorizationError as exc:
        print(str(exc), file=sys.stderr)
        return (
            42
            if BLOCKER in str(exc)
            or RECEIPT_SCHEMA_BLOCKER in str(exc)
            or FORMAL_GATE_BLOCKER in str(exc)
            or EXTERNAL_ARCHIVE_LOCK_BLOCKER in str(exc)
            else 22
        )


if __name__ == "__main__":
    raise SystemExit(main())
