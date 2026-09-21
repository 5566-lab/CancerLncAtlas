#!/usr/bin/env python3
"""Validate/materialize the corrected V3.2 G0/G1/G2 launch contract.

This module is intentionally CPU/no-GPU only.  It closes the lineage from the
completed r5 rebind to the fifteen corrected local-CNV fold manifests, hashes
only small authority/code/config files, and (when an independently recorded
training decision is supplied) materializes hash-bound paid-task manifests.
It never opens a ``.pt`` payload, imports torch/CUDA, calls a provider API, or
writes a training ``SUCCESS.json``.  A real trainer is the only component that
may create a fold SUCCESS receipt.

The old pre-local-CNV materialization is deliberately rejected everywhere in
this contract.  The canonical input root is fixed to the corrected root named
below; callers cannot redirect this validator to a historical baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HOST_EXPECTED = "149"
PROJECT_ROOT = Path("./data/CancerLncAtlas")
PREPARED_ROOT = PROJECT_ROOT / "inputs/v32_g012_local_cnv_formal_prepared_20260903_r1"
REBIN_ROOT = Path(
    "${PRIVATE_WORK_ROOT}/CancerLncAtlas_v32_staging_20260904_r3/"
    "g012_rebind_20260904_r5"
)
NAMESPACE = "v32_g012_corrected_paid_gpu_20260905_r1"
RUN_NAMESPACE = "v32_g012_corrected_paid_gpu_training_20260905_r1"
SEED = 20260726
VARIANTS = ("G0", "G1", "G2")
FOLDS = tuple(range(5))
ENDPOINT = "paid_gpu"
HARDWARE = "PAID_PREEMPTIBLE_GPU"
TRAINER = "cc_hhgt.v32.training:run_authorized_task"
OLD_BASELINE_ROOT = (
    "./data/CancerLncAtlas/inputs/"
    "v32_g012_patient_first_20260830_r1/formal_prepared_20260830_r3_affine_pyg280_localtorch"
)
REBIN_SUCCESS_STATUS = "PASS_INPUT_REBIND_MANIFEST_AND_PREFLIGHT_ONLY"
REBIN_MANIFEST_STATUS = "PASS_INPUT_HASHES_BOUND_NO_TRAINING"
REBIN_PLAN_STATUS = "INPUT_REBIND_READY_LAUNCH_BLOCKED"
REBIN_PREFLIGHT_STATUS = "BLOCKED_INPUT_REBIND_ONLY_NO_LAUNCH"
AUTH_FORMAT = "CC_HHGT_V3_2_TRAINING_APPROVAL_V1"
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


class StaticAuthError(RuntimeError):
    """A corrected static authorization contract is not safe to use."""


def _host() -> str:
    return platform.node().split(".", 1)[0]


def _sha(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


def _canonical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_old(value: Any, label: str) -> None:
    """Reject a superseded pre-local-CNV path in a current authority."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    if OLD_BASELINE_ROOT in encoded:
        raise StaticAuthError(f"OLD_PRE_LOCAL_CNV_PATH_FORBIDDEN={label}")


def _walk_no_symlink(path: Path, label: str) -> None:
    target = _canonical(path)
    # Existing components must all be ordinary directories/files.  This is a
    # lexical walk rather than resolve(), so a deleted/moved target fails
    # closed instead of silently resolving somewhere else.
    current = Path(target.anchor)
    for part in target.parts[1:]:
        current /= part
        try:
            item = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise StaticAuthError(f"LSTAT_FAILED={label}:{current}") from exc
        if stat.S_ISLNK(item.st_mode):
            raise StaticAuthError(f"SYMLINK_COMPONENT_FORBIDDEN={label}:{current}")


def _regular(path: Path, label: str, *, allow_empty: bool = False) -> os.stat_result:
    _walk_no_symlink(path, label)
    try:
        item = os.lstat(path)
    except OSError as exc:
        raise StaticAuthError(f"MISSING={label}:{path}") from exc
    if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode):
        raise StaticAuthError(f"NOT_REGULAR={label}:{path}")
    if item.st_nlink != 1:
        raise StaticAuthError(f"HARDLINK_FORBIDDEN={label}:{path}")
    if not allow_empty and item.st_size <= 0:
        raise StaticAuthError(f"EMPTY={label}:{path}")
    return item


def _read_small(path: Path, label: str, *, allow_empty: bool = False) -> bytes:
    if path.suffix.lower() == ".pt":
        raise StaticAuthError(f"PAYLOAD_READ_FORBIDDEN={path}")
    before = _regular(path, label, allow_empty=allow_empty)
    try:
        with path.open("rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise StaticAuthError(f"READ_FAILED={label}:{path}") from exc
    after = os.stat(path, follow_symlinks=False)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise StaticAuthError(f"CHANGED_DURING_READ={label}:{path}")
    _reject_old(raw.decode("utf-8", errors="replace"), label)
    return raw


def _sha_file(path: Path, label: str) -> str:
    return hashlib.sha256(_read_small(path, label)).hexdigest()


def _json(path: Path, label: str, expected_sha: str | None = None) -> dict[str, Any]:
    raw = _read_small(path, label)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise StaticAuthError(f"SHA_DRIFT={label}:{digest}!={expected_sha}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticAuthError(f"INVALID_JSON={label}:{path}") from exc
    if not isinstance(value, Mapping):
        raise StaticAuthError(f"JSON_OBJECT_REQUIRED={label}")
    _reject_old(value, label)
    return dict(value)


def _write_new(path: Path, raw: bytes) -> str:
    _reject_old(raw.decode("utf-8", errors="replace"), str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise StaticAuthError(f"REFUSING_OVERWRITE={path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise StaticAuthError(f"PARTIAL_OUTPUT_PRESENT={temporary}")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _load_config(path: Path) -> tuple[dict[str, Any], str, bytes]:
    raw = _read_small(path, "CONFIG")
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(raw.decode("utf-8"))
    except ModuleNotFoundError as exc:
        raise StaticAuthError("PYYAML_REQUIRED_FOR_CORRECTED_CONFIG") from exc
    except Exception as exc:
        raise StaticAuthError(f"CONFIG_PARSE_FAILED={path}") from exc
    if not isinstance(value, Mapping):
        raise StaticAuthError("CONFIG_MAPPING_REQUIRED")
    result = dict(value)
    _reject_old(result, "CONFIG")
    return result, hashlib.sha256(raw).hexdigest(), raw


def _code_files(root: Path) -> list[tuple[str, Path]]:
    roots = [root / "cc_hhgt", root / "scripts" / "v32_pipeline.py"]
    result: list[tuple[str, Path]] = []
    for source in roots:
        if source.is_file():
            result.append((source.relative_to(root).as_posix(), source))
            continue
        if not source.is_dir() or source.is_symlink():
            raise StaticAuthError(f"CODE_ROOT_MISSING={source}")
        for candidate in source.rglob("*"):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            rel = candidate.relative_to(root)
            if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in rel.parts):
                continue
            if candidate.suffix.lower() in {".pyc", ".pyo"}:
                continue
            result.append((rel.as_posix(), candidate))
    if not result:
        raise StaticAuthError("CODE_TREE_EMPTY")
    return sorted(result)


def code_tree_sha256(root: Path) -> str:
    root = _canonical(root)
    _walk_no_symlink(root, "CODE_ROOT")
    digest = hashlib.sha256()
    for relative, path in _code_files(root):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha_file(path, f"CODE:{relative}").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _run_id(variant: str) -> str:
    return f"v32-g012-{variant.lower()}-corrected-local-cnv-paid-gpu-20260905-r1"


def _payload_stat(path: Path, label: str) -> os.stat_result:
    """Metadata-only payload check; this function never opens a .pt file."""

    if path.suffix.lower() != ".pt":
        raise StaticAuthError(f"EXPECTED_PT_PAYLOAD={path}")
    _walk_no_symlink(path, label)
    try:
        item = os.lstat(path)
    except OSError as exc:
        raise StaticAuthError(f"MISSING_PAYLOAD={label}:{path}") from exc
    if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode):
        raise StaticAuthError(f"INVALID_PAYLOAD={label}:{path}")
    if item.st_nlink != 1 or item.st_size <= 0:
        raise StaticAuthError(f"PAYLOAD_IDENTITY_INVALID={label}:{path}")
    return item


def _validate_rebind(
    rebind_root: Path, prepared_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    rebind_root = _canonical(rebind_root)
    prepared_root = _canonical(prepared_root)
    success_path = rebind_root / "SUCCESS.json"
    manifest_path = rebind_root / "REBOUND_INPUT_MANIFEST.json"
    plan_path = rebind_root / "REBIND_PLAN.json"
    preflight_path = rebind_root / "LAUNCH_PREFLIGHT.json"
    success = _json(success_path, "REBIND_SUCCESS")
    manifest = _json(manifest_path, "REBIND_MANIFEST")
    plan = _json(plan_path, "REBIND_PLAN")
    preflight = _json(preflight_path, "REBIND_PREFLIGHT")
    if success.get("format") != "CANCERLNCATLAS_V32_G012_CORRECTED_LOCAL_CNV_REBIND_SUCCESS_V1":
        raise StaticAuthError("REBIND_SUCCESS_FORMAT_DRIFT")
    if success.get("status") != REBIN_SUCCESS_STATUS:
        raise StaticAuthError(f"REBIND_SUCCESS_STATUS={success.get('status')!r}")
    if success.get("host") != HOST_EXPECTED or success.get("output_root") != str(rebind_root):
        raise StaticAuthError("REBIND_SUCCESS_HOST_OR_ROOT_DRIFT")
    if success.get("fold_artifact_count") != 15:
        raise StaticAuthError("REBIND_FOLD_COUNT_DRIFT")
    # r5 keeps the two provenance-exclusion flags on the byte-bound input
    # manifest (rather than duplicating them in SUCCESS.json).  Read those
    # flags from the manifest as a compatibility normalization, while still
    # requiring an explicit false value in at least one of the two immutable
    # receipts.  This does not weaken the gate: a missing flag in both files
    # remains a hard failure.
    manifest_flags = manifest.get("safety_flags", {})
    if not isinstance(manifest_flags, Mapping):
        manifest_flags = {}
    for key in (
        "training_started",
        "gpu_started",
        "sealed_test_read",
        "old_checkpoint_loaded",
        "old_predictions_used_as_features",
        "training_approval_emitted",
        "production_result_written",
    ):
        value = success.get(key)
        if key in {"old_checkpoint_loaded", "old_predictions_used_as_features"} and key not in success:
            value = manifest.get(key, manifest_flags.get(key))
        if value is not False:
            raise StaticAuthError(f"REBIND_UNSAFE_FLAG={key}")
    if manifest.get("format") != "CANCERLNCATLAS_V32_G012_CORRECTED_LOCAL_CNV_INPUT_REBIND_V1":
        raise StaticAuthError("REBIND_MANIFEST_FORMAT_DRIFT")
    if manifest.get("status") != REBIN_MANIFEST_STATUS or manifest.get("prepared_root") != str(prepared_root):
        raise StaticAuthError("REBIND_MANIFEST_STATUS_OR_ROOT_DRIFT")
    if manifest.get("fold_artifact_count") != 15 or manifest.get("training_started") is not False or manifest.get("gpu_started") is not False:
        raise StaticAuthError("REBIND_MANIFEST_UNSAFE_FLAGS")
    if plan.get("format") != "CANCERLNCATLAS_V32_G012_CORRECTED_LOCAL_CNV_REBIND_PLAN_V1" or plan.get("status") != REBIN_PLAN_STATUS:
        raise StaticAuthError("REBIND_PLAN_STATUS_DRIFT")
    if plan.get("prepared_root") != str(prepared_root) or plan.get("fold_artifact_count") != 15:
        raise StaticAuthError("REBIND_PLAN_ROOT_OR_COUNT_DRIFT")
    # The r5 plan predates the explicit top-level launch_permitted field; its
    # launch-blocked status plus training_approval_emitted=false is the
    # immutable equivalent.  Accept the omitted field only for this exact
    # blocked status and normalize it to false for the current bundle.
    plan_launch_permitted = plan.get("launch_permitted", False)
    if plan.get("training_approval_emitted") is not False or plan_launch_permitted is not False:
        raise StaticAuthError("REBIND_PLAN_UNSAFE_FLAGS")
    if preflight.get("format") != "CANCERLNCATLAS_V32_G012_CORRECTED_LOCAL_CNV_LAUNCH_PREFLIGHT_V1" or preflight.get("status") != REBIN_PREFLIGHT_STATUS:
        raise StaticAuthError("REBIND_PREFLIGHT_STATUS_DRIFT")
    for key in (
        "training_started", "gpu_started", "optimizer_progress_observed",
        "gpu_telemetry_observed", "sealed_test_read", "training_approval_emitted",
        "paid_compute_requested", "launch_permitted",
    ):
        if preflight.get(key) is not False:
            raise StaticAuthError(f"REBIND_PREFLIGHT_UNSAFE_FLAG={key}")

    rows = manifest.get("fold_inputs")
    if not isinstance(rows, list) or len(rows) != 15:
        raise StaticAuthError("REBIND_FOLD_ROWS_DRIFT")
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise StaticAuthError("REBIND_ROW_MAPPING_REQUIRED")
        variant = row.get("variant")
        try:
            fold = int(row.get("fold", -1))
        except (TypeError, ValueError) as exc:
            raise StaticAuthError("REBIND_ROW_FOLD_INVALID") from exc
        if variant not in VARIANTS or fold not in FOLDS or (variant, fold) in seen:
            raise StaticAuthError(f"REBIND_ROW_KEY_INVALID={variant}:{fold}")
        expected = prepared_root / str(variant) / f"PATIENT_FOLD_{fold}.pt"
        if row.get("path") != str(expected) or row.get("relative_path") != f"{variant}/PATIENT_FOLD_{fold}.pt":
            raise StaticAuthError(f"REBIND_ROW_PATH_DRIFT={variant}:{fold}")
        if row.get("corrected_local_cnv") is not True or not _sha(row.get("sha256")):
            raise StaticAuthError(f"REBIND_ROW_DECLARATION_DRIFT={variant}:{fold}")
        item = _payload_stat(expected, f"REBIND_PAYLOAD_{variant}_{fold}")
        if row.get("bytes") != item.st_size:
            raise StaticAuthError(f"REBIND_ROW_SIZE_DRIFT={variant}:{fold}")
        seen.add((str(variant), fold))
    if seen != {(variant, fold) for variant in VARIANTS for fold in FOLDS}:
        raise StaticAuthError("REBIND_ROW_SET_DRIFT")
    declared = success.get("rebound_input_manifest")
    if not isinstance(declared, Mapping) or declared.get("path") != str(manifest_path) or declared.get("sha256") != _sha_file(manifest_path, "REBIND_MANIFEST"):
        raise StaticAuthError("REBIND_SUCCESS_MANIFEST_BINDING_DRIFT")
    return success, manifest, plan, preflight, _sha_file(success_path, "REBIND_SUCCESS")


def _validate_config(config: dict[str, Any], config_path: Path, variant: str, prepared_root: Path) -> None:
    policy = config.get("execution_control")
    if not isinstance(policy, Mapping):
        raise StaticAuthError("CONFIG_EXECUTION_CONTROL_MISSING")
    expected_policy = {
        "execution_mode": "TRAINING",
        "training_authorized": True,
        "paid_enabled": True,
        "max_paid_hours": 96,
        "max_cost_cny": 210,
    }
    for key, value in expected_policy.items():
        if policy.get(key) != value:
            raise StaticAuthError(f"CONFIG_POLICY_DRIFT={key}:{policy.get(key)!r}")
    task = config.get("task_contract")
    if not isinstance(task, Mapping) or task.get("graph_variant") != variant:
        raise StaticAuthError(f"CONFIG_GRAPH_VARIANT_DRIFT={variant}")
    crossfit = config.get("crossfit")
    if not isinstance(crossfit, Mapping) or crossfit.get("cancers") != 33 or crossfit.get("patient_folds") != 5 or crossfit.get("seed") != SEED:
        raise StaticAuthError("CONFIG_CROSSFIT_DRIFT")
    runtime = config.get("runtime_profile")
    if not isinstance(runtime, Mapping) or runtime.get("owner") != ENDPOINT or runtime.get("mixed_precision") != "bf16":
        raise StaticAuthError("CONFIG_RUNTIME_DRIFT")
    io = config.get("training_io")
    if not isinstance(io, Mapping):
        raise StaticAuthError("CONFIG_TRAINING_IO_MISSING")
    expected_pattern = str(prepared_root / variant / "PATIENT_FOLD_{fold}.pt")
    if io.get("prepared_fold_pattern") != expected_pattern:
        raise StaticAuthError(f"CONFIG_INPUT_PATTERN_DRIFT={variant}")
    _reject_old(config, f"CONFIG_{variant}")


def _load_rebind_variant_rows(rebind_root: Path, variant: str) -> list[dict[str, Any]]:
    path = rebind_root / variant / "INPUT_MANIFEST.json"
    payload = _json(path, f"REBIND_VARIANT_INPUT_{variant}")
    rows = payload.get("fold_inputs")
    if not isinstance(rows, list) or len(rows) != 5:
        raise StaticAuthError(f"REBIND_VARIANT_INPUT_COUNT={variant}")
    result: list[dict[str, Any]] = []
    folds: set[int] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise StaticAuthError(f"REBIND_VARIANT_INPUT_ROW={variant}")
        fold = int(raw.get("fold", -1))
        expected = PREPARED_ROOT / variant / f"PATIENT_FOLD_{fold}.pt"
        if fold not in FOLDS or raw.get("path") != str(expected) or not _sha(raw.get("sha256")):
            raise StaticAuthError(f"REBIND_VARIANT_INPUT_DRIFT={variant}:{fold}")
        item = _payload_stat(expected, f"VARIANT_PAYLOAD_{variant}_{fold}")
        if raw.get("bytes") != item.st_size:
            raise StaticAuthError(f"REBIND_VARIANT_INPUT_SIZE={variant}:{fold}")
        result.append({"fold": fold, "path": str(expected), "bytes": item.st_size, "sha256": raw["sha256"], "corrected_local_cnv": True})
        folds.add(fold)
    if folds != set(FOLDS):
        raise StaticAuthError(f"REBIND_VARIANT_INPUT_FOLDS={variant}")
    return sorted(result, key=lambda row: row["fold"])


def _task_rows(variant: str) -> list[dict[str, Any]]:
    run_id = _run_id(variant)
    return [
        {
            "task_id": f"{run_id}|PATIENT_FOLD_{fold}|CC-HHGT|{SEED}",
            "run_id": run_id,
            "task_type": "CC_HHGT_PATIENT_FOLD",
            "model": "CC-HHGT",
            "patient_fold": fold,
            "seed": SEED,
            "owner": ENDPOINT,
            "hardware_class": HARDWARE,
            "status": "PENDING",
            "blocked_reason": "",
            "paid_task": True,
        }
        for fold in FOLDS
    ]


def _task_tsv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=TASK_COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "paid_task": str(row["paid_task"]).lower()})
    return stream.getvalue().encode("utf-8")


def _validate_decision(path: Path, prepared_root: Path, rebind_sha: str, manifest_sha: str) -> dict[str, Any]:
    decision = _json(path, "TRAINING_DECISION")
    if decision.get("format") != "CANCERLNCATLAS_V32_G012_CORRECTED_TRAINING_DECISION_V1" or decision.get("status") != "TRAINING_AUTHORIZED":
        raise StaticAuthError("TRAINING_DECISION_NOT_AUTHORIZED")
    if decision.get("host") != HOST_EXPECTED or decision.get("prepared_root") != str(prepared_root):
        raise StaticAuthError("TRAINING_DECISION_SCOPE_DRIFT")
    if decision.get("rebind_success_sha256") != rebind_sha or decision.get("rebound_input_manifest_sha256") != manifest_sha:
        raise StaticAuthError("TRAINING_DECISION_REBIND_DRIFT")
    for key in ("sealed_test_read", "old_baseline_used", "gpu_started", "training_started"):
        if decision.get(key) is not False:
            raise StaticAuthError(f"TRAINING_DECISION_UNSAFE_FLAG={key}")
    return decision


def _materialize(
    *,
    output_root: Path,
    run_parent: Path,
    bootstrap_root: Path,
    code_root: Path,
    config_template: Path,
    rebind_root: Path,
    prepared_root: Path,
    decision_path: Path,
) -> dict[str, Any]:
    if output_root.exists() or run_parent.exists():
        raise StaticAuthError("STATIC_AUTH_OUTPUT_MUST_BE_ABSENT")
    code_sha = code_tree_sha256(code_root)
    template_config, _, template_raw = _load_config(config_template)
    success, manifest, plan, preflight, rebind_sha = _validate_rebind(rebind_root, prepared_root)
    manifest_sha = _sha_file(rebind_root / "REBOUND_INPUT_MANIFEST.json", "REBIND_MANIFEST")
    _validate_decision(decision_path, prepared_root, rebind_sha, manifest_sha)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    run_parent.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=False)
    run_parent.mkdir(parents=True, exist_ok=False)
    variant_records: dict[str, Any] = {}
    try:
        for variant in VARIANTS:
            rendered = template_raw.replace(b"__GRAPH_VARIANT__", variant.encode("ascii"))
            if rendered == template_raw or b"__GRAPH_VARIANT__" in rendered:
                raise StaticAuthError(f"CONFIG_RENDER_FAILED={variant}")
            config_path = run_parent / variant / "config.yaml"
            authorization = run_parent / variant / "authorization"
            authorization.mkdir(parents=True, exist_ok=False)
            _write_new(config_path, rendered)
            config, _, _ = _load_config(config_path)
            _validate_config(config, config_path, variant, prepared_root)
            fold_inputs = _load_rebind_variant_rows(rebind_root, variant)
            input_payload = {
                "format": "CANCERLNCATLAS_V32_G012_CORRECTED_LOCAL_CNV_INPUT_MANIFEST_V1",
                "analysis_version": f"CancerLncAtlas_V3.2_G012_{variant}_CORRECTED_LOCAL_CNV",
                "graph_variant": variant,
                "corrected_local_cnv": True,
                "formal_v32_primary_unchanged": True,
                "sealed_test_read": False,
                "rebind_success_sha256": rebind_sha,
                "rebound_input_manifest_sha256": manifest_sha,
                "fold_inputs": fold_inputs,
            }
            input_path = authorization / "INPUT_MANIFEST.json"
            input_sha = _write_new(input_path, _json_bytes(input_payload))
            tasks = _task_rows(variant)
            task_path = authorization / "TASK_MANIFEST.tsv"
            task_sha = _write_new(task_path, _task_tsv(tasks))
            config_sha = hashlib.sha256(rendered).hexdigest()
            approval = {
                "approval_format": AUTH_FORMAT,
                "training_authorized": True,
                "authorization_basis": "CANCERLNCATLAS_V32_G012_CORRECTED_TRAINING_DECISION",
                "run_id": _run_id(variant),
                "endpoint_id": ENDPOINT,
                "hardware_class": HARDWARE,
                "authorized_trainer": TRAINER,
                "approved_task_ids": [row["task_id"] for row in tasks],
                "artifact_hashes": {
                    "code_sha256": code_sha,
                    "config_sha256": config_sha,
                    "input_manifest_sha256": input_sha,
                    "task_manifest_sha256": task_sha,
                },
                "paid_enabled": True,
                "max_paid_hours": 96,
                "max_cost_cny": 210,
                "prepared_root": str(prepared_root),
                "rebind_success_sha256": rebind_sha,
                "rebound_input_manifest_sha256": manifest_sha,
                "sealed_test_read": False,
            }
            approval_path = authorization / "TRAINING_APPROVAL.json"
            approval_sha = _write_new(approval_path, _json_bytes(approval))
            marker = {
                "format": "CANCERLNCATLAS_V32_G012_CORRECTED_STATIC_AUTHORIZATION_V1",
                "status": "STATIC_AUTHORIZATION_READY",
                "variant": variant,
                "run_id": _run_id(variant),
                "host": HOST_EXPECTED,
                "prepared_root": str(prepared_root),
                "rebind_success_sha256": rebind_sha,
                "rebound_input_manifest_sha256": manifest_sha,
                "artifact_hashes": approval["artifact_hashes"],
                "training_started": False,
                "gpu_started": False,
                "sealed_test_read": False,
                "training_approval_emitted": True,
                "optimizer_progress_observed": False,
            }
            marker_path = authorization / "STATIC_AUTHORIZATION.json"
            marker_sha = _write_new(marker_path, _json_bytes(marker))
            variant_records[variant] = {
                "run_id": _run_id(variant),
                "config": {"path": str(config_path), "sha256": config_sha},
                "input_manifest": {"path": str(input_path), "sha256": input_sha},
                "task_manifest": {"path": str(task_path), "sha256": task_sha},
                "approval": {"path": str(approval_path), "sha256": approval_sha},
                "static_authorization": {"path": str(marker_path), "sha256": marker_sha},
                "fold_inputs": fold_inputs,
                "tasks": 5,
            }
    except Exception:
        # A failed materialization must not leave an apparently usable bundle.
        # The caller may remove this uniquely named incomplete tree after
        # recording the exception; no SUCCESS marker is ever written here.
        raise
    top = {
        "format": "CANCERLNCATLAS_V32_G012_CORRECTED_STATIC_AUTH_READY_V1",
        "status": "STATIC_AUTH_READY_CORRECTED_G012",
        "namespace": NAMESPACE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": HOST_EXPECTED,
        "cpu_only": True,
        "gpu_visible": False,
        "training_authorized": True,
        "paid_enabled": True,
        "launch_permitted": True,
        "formal_seed": SEED,
        "prepared_root": str(prepared_root),
        "rebind_root": str(rebind_root),
        "rebind_success_sha256": rebind_sha,
        "rebound_input_manifest_sha256": manifest_sha,
        "rebind_plan_sha256": _sha_file(rebind_root / "REBIND_PLAN.json", "REBIND_PLAN"),
        "code_root": str(_canonical(code_root)),
        "code_tree_sha256": code_sha,
        "config_template": str(_canonical(config_template)),
        "config_template_sha256": hashlib.sha256(template_raw).hexdigest(),
        "run_parent": str(run_parent),
        "variants": variant_records,
        "fold_artifact_count": 15,
        "sealed_test_read": False,
        "old_baseline_used": False,
        "gpu_started": False,
        "training_started": False,
        "training_complete_written": False,
        "launch_preflight_required": True,
        "launch_preflight_path": str(bootstrap_root / "LAUNCH_PREFLIGHT.json"),
        "launch_preflight_present": False,
        "training_decision_path": str(_canonical(decision_path)),
        "no_payload_bytes_read_by_materializer": True,
    }
    top_path = output_root / "STATIC_AUTH_READY.json"
    top_sha = _write_new(top_path, _json_bytes(top))
    sums_lines = []
    for candidate in sorted(output_root.rglob("*")):
        if candidate.is_file() and candidate.name != "SHA256SUMS.tsv":
            sums_lines.append(f"{_sha_file(candidate, candidate.name)}\t{candidate.stat().st_size}\t{candidate.relative_to(output_root).as_posix()}")
    sums_sha = _write_new(output_root / "SHA256SUMS.tsv", ("\n".join(sums_lines) + "\n").encode())
    return {"status": top["status"], "path": str(top_path), "sha256": top_sha, "sha256sums": sums_sha, "variants": variant_records}


def validate_static_auth(
    *,
    receipt: Path,
    code_root: Path,
    config_template: Path,
    prepared_root: Path,
    rebind_root: Path,
    run_parent: Path,
    bootstrap_root: Path,
) -> dict[str, Any]:
    """Validate an existing bundle and all hashes; no writes occur."""

    top = _json(receipt, "STATIC_AUTH_READY")
    if top.get("status") != "STATIC_AUTH_READY_CORRECTED_G012" or top.get("host") != HOST_EXPECTED:
        raise StaticAuthError("STATIC_AUTH_STATUS_OR_HOST_DRIFT")
    if top.get("prepared_root") != str(_canonical(prepared_root)) or top.get("rebind_root") != str(_canonical(rebind_root)):
        raise StaticAuthError("STATIC_AUTH_ROOT_DRIFT")
    if (
        top.get("gpu_visible") is not False
        or top.get("gpu_started") is not False
        or top.get("training_started") is not False
        or top.get("sealed_test_read") is not False
        or top.get("old_baseline_used") is not False
        or top.get("training_complete_written") is not False
    ):
        raise StaticAuthError("STATIC_AUTH_UNSAFE_FLAGS")
    if top.get("training_authorized") is not True or top.get("paid_enabled") is not True or top.get("launch_permitted") is not True:
        raise StaticAuthError("STATIC_AUTH_NOT_PAID_TRAINING_AUTHORIZED")
    if top.get("run_parent") != str(_canonical(run_parent)):
        raise StaticAuthError("STATIC_AUTH_RUN_ROOT_DRIFT")
    if top.get("formal_seed") != SEED or top.get("fold_artifact_count") != 15:
        raise StaticAuthError("STATIC_AUTH_SEED_OR_COUNT_DRIFT")
    success, manifest, plan, preflight, rebind_sha = _validate_rebind(_canonical(rebind_root), _canonical(prepared_root))
    manifest_sha = _sha_file(_canonical(rebind_root) / "REBOUND_INPUT_MANIFEST.json", "REBIND_MANIFEST")
    if top.get("rebind_success_sha256") != rebind_sha or top.get("rebound_input_manifest_sha256") != manifest_sha:
        raise StaticAuthError("STATIC_AUTH_REBIND_HASH_DRIFT")
    observed_code_sha = code_tree_sha256(_canonical(code_root))
    if top.get("code_tree_sha256") != observed_code_sha:
        raise StaticAuthError("STATIC_AUTH_CODE_TREE_DRIFT")
    template_config, template_sha, template_raw = _load_config(_canonical(config_template))
    if top.get("config_template_sha256") != template_sha:
        raise StaticAuthError("STATIC_AUTH_TEMPLATE_SHA_DRIFT")
    variants = top.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(VARIANTS):
        raise StaticAuthError("STATIC_AUTH_VARIANTS_DRIFT")
    for variant in VARIANTS:
        item = variants[variant]
        if not isinstance(item, Mapping) or item.get("run_id") != _run_id(variant):
            raise StaticAuthError(f"STATIC_AUTH_VARIANT_DRIFT={variant}")
        config_path = Path(str(item.get("config", {}).get("path", "")))
        input_path = Path(str(item.get("input_manifest", {}).get("path", "")))
        task_path = Path(str(item.get("task_manifest", {}).get("path", "")))
        approval_path = Path(str(item.get("approval", {}).get("path", "")))
        marker_path = Path(str(item.get("static_authorization", {}).get("path", "")))
        for path, label, declared in (
            (config_path, f"CONFIG_{variant}", item.get("config", {}).get("sha256")),
            (input_path, f"INPUT_{variant}", item.get("input_manifest", {}).get("sha256")),
            (task_path, f"TASK_{variant}", item.get("task_manifest", {}).get("sha256")),
            (approval_path, f"APPROVAL_{variant}", item.get("approval", {}).get("sha256")),
            (marker_path, f"MARKER_{variant}", item.get("static_authorization", {}).get("sha256")),
        ):
            if not _sha(declared) or _sha_file(path, label) != declared:
                raise StaticAuthError(f"STATIC_AUTH_ARTIFACT_SHA_DRIFT={variant}:{label}")
        config, config_sha, _ = _load_config(config_path)
        _validate_config(config, config_path, variant, _canonical(prepared_root))
        if config_sha != item["config"]["sha256"]:
            raise StaticAuthError(f"STATIC_AUTH_CONFIG_HASH_DRIFT={variant}")
        # Reuse the parser already used by the training CLI for exact TSV
        # semantics; importing this pure-Python helper does not import torch.
        script_dir = Path(__file__).resolve().parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from normalize_v32_task_manifest import validate_task_manifest  # type: ignore

        rows = validate_task_manifest(task_path, run_id=_run_id(variant), owner=ENDPOINT, hardware_class=HARDWARE, paid_task="true", expected_folds=FOLDS)
        input_doc = _json(input_path, f"INPUT_{variant}")
        if input_doc.get("corrected_local_cnv") is not True or input_doc.get("rebound_input_manifest_sha256") != manifest_sha:
            raise StaticAuthError(f"STATIC_AUTH_INPUT_SCOPE_DRIFT={variant}")
        fold_rows = input_doc.get("fold_inputs")
        if not isinstance(fold_rows, list) or len(fold_rows) != 5:
            raise StaticAuthError(f"STATIC_AUTH_INPUT_COUNT_DRIFT={variant}")
        fold_set: set[int] = set()
        for fold_row in fold_rows:
            if not isinstance(fold_row, Mapping):
                raise StaticAuthError(f"STATIC_AUTH_INPUT_ROW_DRIFT={variant}")
            fold = int(fold_row.get("fold", -1))
            expected_payload = _canonical(prepared_root) / variant / f"PATIENT_FOLD_{fold}.pt"
            if fold not in FOLDS or fold in fold_set or fold_row.get("path") != str(expected_payload):
                raise StaticAuthError(f"STATIC_AUTH_INPUT_PATH_DRIFT={variant}:{fold}")
            if not _sha(fold_row.get("sha256")) or fold_row.get("corrected_local_cnv") is not True:
                raise StaticAuthError(f"STATIC_AUTH_INPUT_DECLARATION_DRIFT={variant}:{fold}")
            observed_payload = _payload_stat(expected_payload, f"AUTH_PAYLOAD_{variant}_{fold}")
            if fold_row.get("bytes") != observed_payload.st_size:
                raise StaticAuthError(f"STATIC_AUTH_INPUT_SIZE_DRIFT={variant}:{fold}")
            fold_set.add(fold)
        if fold_set != set(FOLDS):
            raise StaticAuthError(f"STATIC_AUTH_INPUT_FOLD_SET_DRIFT={variant}")
        approval = _json(approval_path, f"APPROVAL_{variant}")
        if approval.get("approval_format") != AUTH_FORMAT or approval.get("training_authorized") is not True or approval.get("run_id") != _run_id(variant) or approval.get("approved_task_ids") != [row["task_id"] for row in rows]:
            raise StaticAuthError(f"STATIC_AUTH_APPROVAL_SCOPE_DRIFT={variant}")
        hashes = approval.get("artifact_hashes")
        expected_hashes = {"code_sha256": observed_code_sha, "config_sha256": item["config"]["sha256"], "input_manifest_sha256": item["input_manifest"]["sha256"], "task_manifest_sha256": item["task_manifest"]["sha256"]}
        if hashes != expected_hashes:
            raise StaticAuthError(f"STATIC_AUTH_APPROVAL_HASH_DRIFT={variant}")
        marker = _json(marker_path, f"MARKER_{variant}")
        if marker.get("status") != "STATIC_AUTHORIZATION_READY" or marker.get("training_started") is not False or marker.get("gpu_started") is not False:
            raise StaticAuthError(f"STATIC_AUTH_MARKER_DRIFT={variant}")
    return top


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "materialize"), default="validate")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--config-template", "--config", dest="config_template", type=Path)
    parser.add_argument(
        "--prepared-root", "--prepared-parent", dest="prepared_root", type=Path,
        default=PREPARED_ROOT,
    )
    parser.add_argument("--rebind-root", type=Path)
    parser.add_argument("--rebind-success", type=Path)
    parser.add_argument("--rebound-manifest", type=Path)
    parser.add_argument("--launch-preflight", type=Path)
    parser.add_argument("--code-archive", type=Path)
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--run-parent", type=Path)
    parser.add_argument("--bootstrap-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--decision", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if _host() != HOST_EXPECTED:
            raise StaticAuthError(f"BLOCKED_WRONG_HOST expected={HOST_EXPECTED} observed={_host()}")
        prepared = _canonical(args.prepared_root)
        code_root = _canonical(args.code_root)
        config_template = _canonical(
            args.config_template
            or code_root / "config/model_v3_2_g012_corrected_paid_gpu_20260905_r1.yaml"
        )
        receipt_path = _canonical(args.receipt) if args.receipt is not None else None
        bootstrap_root = _canonical(
            args.bootstrap_root
            or (receipt_path.parent if receipt_path is not None else code_root.parent / "bootstrap" / NAMESPACE)
        )
        run_parent = _canonical(
            args.run_parent
            or (PROJECT_ROOT / "results" / RUN_NAMESPACE)
        )
        rebind = _canonical(
            args.rebind_root
            or (args.rebind_success.parent if args.rebind_success is not None else REBIN_ROOT)
        )
        if prepared != PREPARED_ROOT or OLD_BASELINE_ROOT in str(prepared):
            raise StaticAuthError(f"PREPARED_ROOT_NOT_CURRENT_CORRECTED={prepared}")
        if OLD_BASELINE_ROOT in str(rebind) or OLD_BASELINE_ROOT in str(code_root) or OLD_BASELINE_ROOT in str(config_template):
            raise StaticAuthError("OLD_PRE_LOCAL_CNV_PATH_IN_ARGUMENTS")
        if args.mode == "materialize":
            if args.output_root is None or args.decision is None:
                raise StaticAuthError("MATERIALIZE_REQUIRES_OUTPUT_RUN_PARENT_DECISION")
            result = _materialize(
                output_root=_canonical(args.output_root),
                run_parent=run_parent,
                bootstrap_root=bootstrap_root,
                code_root=code_root,
                config_template=config_template,
                rebind_root=rebind,
                prepared_root=prepared,
                decision_path=_canonical(args.decision),
            )
        else:
            if receipt_path is None:
                raise StaticAuthError("VALIDATE_REQUIRES_RECEIPT_AND_RUN_PARENT")
            result = validate_static_auth(
                receipt=receipt_path,
                code_root=code_root,
                config_template=config_template,
                prepared_root=prepared,
                rebind_root=rebind,
                run_parent=run_parent,
                bootstrap_root=bootstrap_root,
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except StaticAuthError as exc:
        print(str(exc), file=sys.stderr)
        return 42


if __name__ == "__main__":
    raise SystemExit(main())
