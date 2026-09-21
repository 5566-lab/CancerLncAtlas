#!/usr/bin/env python3
"""Fail-closed, CPU-only sealing of the infeasible V3.2 G012 r1 run.

The production CLI has no project-root or instance-id override.  It is fixed to
the CancerLncAtlas server root and the stopped CompShare instance.  Its default
mode is read-only.  ``--apply`` is admitted only with a fresh provider receipt
and the exact reviewed r6 live authorization pair.  No r1 result path is ever
resolved, listed, opened, moved, deleted, or reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path("./data/CancerLncAtlas")
INSTANCE_ID = "uhost-1up504geeqzj"
STOPPED_RECEIPT_FORMAT = "CANCERLNCATLAS_COMPSHARE_STOPPED_RECEIPT_V1"
ABORTED_RECEIPT_FORMAT = "CC_HHGT_V3_2_G012_ABORTED_RUN_V2"
ABORTED_STATUS = "ABORTED_RUNTIME_INFEASIBLE"
R1_NAMESPACE = "v32_g012_paid_gpu_20260831_r1"
R1_RESULT_NAMESPACE = "v32_g012_paid_gpu_training_20260831_r1"
ARCHIVE_NAMESPACE = "archive_aborted_runtime_infeasible_20260901_r1"
R1_PATCH_READY_SHA256 = (
    "100713824da87b8eb4a369ef118861a322ca4f93d8c1e4846a29f769c24548ca"
)
R1_STATIC_AUTH_R6_SHA256 = (
    "1f5bc54b2a8acc037373bfa380c95c8a717e76fde956852ec481b16a0d8716f0"
)
R1_CODE_TREE_SHA256 = (
    "7764f9d0cc192db546034ed9232aa338fb0616dd837a85806dd75069f632527d"
)
R1_OVERLAY_SHA256 = (
    "badf90e3863e9b7d25d6844387c92c37b49d596aa00342de7a562401ee08c1e1"
)
MAX_STOPPED_RECEIPT_AGE_SECONDS = 15 * 60
MAX_FUTURE_CLOCK_SKEW_SECONDS = 60


class AbortSealError(RuntimeError):
    """The r1 run cannot be safely sealed under the supplied evidence."""


@dataclass(frozen=True)
class SealPaths:
    project_root: Path
    bootstrap_root: Path
    result_root: Path
    live_patch_ready: Path
    live_static_auth_ready: Path
    gpu_training_complete: Path
    archive_root: Path
    archived_patch_ready: Path
    archived_static_auth_ready: Path
    aborted_receipt: Path
    partial_receipt: Path
    temporary_final_receipt: Path


def _identity(observed: os.stat_result, *, include_ctime: bool = True) -> tuple[int, ...]:
    values = (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
    )
    return values + ((observed.st_ctime_ns,) if include_ctime else ())


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _path_present(path: Path, label: str) -> bool:
    """Return lstat-style presence, including broken symbolic links."""

    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise AbortSealError(f"{label}_LSTAT_FAILED={path}") from exc
    return True


def _assert_no_symlink_components(path: Path, label: str) -> None:
    target = _absolute_lexical(path)
    for component in reversed([target, *target.parents]):
        try:
            observed = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise AbortSealError(f"{label}_COMPONENT_LSTAT_FAILED={component}") from exc
        if stat.S_ISLNK(observed.st_mode) or getattr(
            os.path, "isjunction", lambda _: False
        )(component):
            raise AbortSealError(f"{label}_SYMLINK_COMPONENT_FORBIDDEN={component}")


def _open_regular_no_follow(path: Path, label: str) -> tuple[int, list[int], Path]:
    target = _absolute_lexical(path)
    _assert_no_symlink_components(target, label)
    flags = (
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
                if not stat.S_ISDIR(os.fstat(current).st_mode):
                    raise AbortSealError(f"{label}_PARENT_NOT_DIRECTORY={component}")
                held_directories.append(current)
            descriptor = os.open(target.name, flags, dir_fd=current)
        else:
            descriptor = os.open(target, flags)
    except (OSError, AbortSealError) as exc:
        for directory in reversed(held_directories):
            try:
                os.close(directory)
            except OSError:
                pass
        if isinstance(exc, AbortSealError):
            raise
        raise AbortSealError(f"{label}_OPEN_FAILED={target}") from exc
    return descriptor, held_directories, target


def _read_regular_bytes(
    path: Path, label: str, *, allow_empty: bool = False
) -> tuple[bytes, str, int]:
    descriptor, held_directories, target = _open_regular_no_follow(path, label)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (
            before.st_size <= 0 and not allow_empty
        ):
            raise AbortSealError(f"{label}_NOT_NONEMPTY_REGULAR={target}")
        if before.st_nlink != 1:
            raise AbortSealError(f"{label}_HARDLINK_FORBIDDEN={target}")
        digest = hashlib.sha256()
        content = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            content.extend(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise AbortSealError(f"{label}_CHANGED_DURING_READ={target}")
        try:
            bound_path = os.stat(target, follow_symlinks=False)
        except OSError as exc:
            raise AbortSealError(f"{label}_PATH_LOST_DURING_READ={target}") from exc
        if not stat.S_ISREG(bound_path.st_mode) or _identity(
            bound_path, include_ctime=False
        ) != _identity(after, include_ctime=False):
            raise AbortSealError(f"{label}_PATH_REPLACED_DURING_READ={target}")
        return bytes(content), digest.hexdigest(), int(before.st_size)
    finally:
        os.close(descriptor)
        for directory in reversed(held_directories):
            os.close(directory)


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str, int]:
    raw, digest, size = _read_regular_bytes(path, label)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AbortSealError(f"{label}_JSON_INVALID={path}") from exc
    if not isinstance(payload, Mapping):
        raise AbortSealError(f"{label}_MAPPING_REQUIRED={path}")
    return dict(payload), digest, size


def _safe_project_root(path: Path) -> Path:
    root = _absolute_lexical(path)
    if root.name != "CancerLncAtlas" or root.parent.name != "DSC":
        raise AbortSealError(f"PROJECT_ROOT_SCOPE_INVALID={root}")
    _assert_no_symlink_components(root, "PROJECT_ROOT")
    return root


def resolve_paths(project_root: Path = PROJECT_ROOT) -> SealPaths:
    root = _safe_project_root(project_root)
    bootstrap = root / "runtime" / "bootstrap" / R1_NAMESPACE
    archive = bootstrap / ARCHIVE_NAMESPACE
    paths = SealPaths(
        project_root=root,
        bootstrap_root=bootstrap,
        # Lexical declaration only.  Never resolve this path.
        result_root=root / "results" / R1_RESULT_NAMESPACE,
        live_patch_ready=bootstrap / "PATCH_READY.json",
        live_static_auth_ready=bootstrap / "STATIC_AUTH_READY.json",
        gpu_training_complete=bootstrap / "GPU_TRAINING_COMPLETE.json",
        archive_root=archive,
        archived_patch_ready=archive / "PATCH_READY.live_at_abort.json",
        archived_static_auth_ready=archive / "STATIC_AUTH_READY.live_at_abort.json",
        aborted_receipt=bootstrap / "ABORTED.json",
        partial_receipt=bootstrap / ".ABORTED.json.partial",
        temporary_final_receipt=archive / ".ABORTED.final.json.partial",
    )
    for key, candidate in paths.__dict__.items():
        if key == "result_root" or not isinstance(candidate, Path):
            continue
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise AbortSealError(f"SEAL_TARGET_ESCAPES_PROJECT_ROOT={candidate}") from exc
    return paths


def _parse_fresh_checked_at(
    value: object, *, now: datetime, max_age_seconds: int
) -> tuple[datetime, float]:
    if not isinstance(value, str) or not value.strip():
        raise AbortSealError("STOPPED_RECEIPT_CHECKED_AT_MISSING")
    try:
        checked = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AbortSealError("STOPPED_RECEIPT_CHECKED_AT_INVALID") from exc
    if checked.tzinfo is None or checked.utcoffset() is None:
        raise AbortSealError("STOPPED_RECEIPT_CHECKED_AT_TIMEZONE_REQUIRED")
    current = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    age = (current.astimezone(timezone.utc) - checked.astimezone(timezone.utc)).total_seconds()
    if age < -MAX_FUTURE_CLOCK_SKEW_SECONDS:
        raise AbortSealError(f"STOPPED_RECEIPT_CHECKED_AT_IN_FUTURE={age}")
    if age > max_age_seconds:
        raise AbortSealError(f"STOPPED_RECEIPT_STALE_SECONDS={age:.3f}")
    return checked, max(0.0, age)


def validate_stopped_receipt(
    path: Path,
    *,
    now: datetime | None = None,
    max_age_seconds: int = MAX_STOPPED_RECEIPT_AGE_SECONDS,
) -> tuple[dict[str, Any], str, float]:
    payload, digest, _ = _load_json(path, "STOPPED_RECEIPT")
    expected = {
        "format": STOPPED_RECEIPT_FORMAT,
        "status": "INSTANCE_STOPPED_STATE_CONFIRMED",
        "instance_id": INSTANCE_ID,
        "observed_state": "Stopped",
        "provider_domain_state": "DOMAIN_SHUT_OFF",
        "gpu_billing_active": False,
        "source": "DIRECT_COMPSHARE_INSTANCE_SHOW_QUERY",
        "raw_provider_response_embedded": False,
        "credentials_embedded": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AbortSealError(f"STOPPED_RECEIPT_CONTRACT_DRIFT={key}")
    current = now or datetime.now(timezone.utc)
    _, age = _parse_fresh_checked_at(
        payload.get("checked_at"), now=current, max_age_seconds=max_age_seconds
    )
    return payload, digest, age


def _validate_patch_payload(payload: Mapping[str, Any], digest: str) -> None:
    if digest != R1_PATCH_READY_SHA256:
        raise AbortSealError(f"R1_PATCH_READY_SHA256_DRIFT={digest}")
    expected = {
        "status": "PATCH_READY",
        "gpu_visible": False,
        "baseline_retained": True,
        "code_root": "./data/CancerLncAtlas/runtime/tools/"
        "v32_g012_paid_gpu_20260831_r1/code",
        "code_tree_sha256": R1_CODE_TREE_SHA256,
        "overlay_sha256": R1_OVERLAY_SHA256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AbortSealError(f"R1_PATCH_READY_SCHEMA_DRIFT={key}")


def _validate_static_payload(payload: Mapping[str, Any], digest: str) -> None:
    if digest != R1_STATIC_AUTH_R6_SHA256:
        raise AbortSealError(f"R1_STATIC_AUTH_R6_SHA256_DRIFT={digest}")
    expected = {
        "status": "STATIC_AUTH_READY",
        "gpu_visible": False,
        "formal_seed": 20260726,
        "patch_ready_sha256": R1_PATCH_READY_SHA256,
        "code_tree_sha256": R1_CODE_TREE_SHA256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AbortSealError(f"R1_STATIC_AUTH_R6_SCHEMA_DRIFT={key}")
    variants = payload.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != {"G0", "G1", "G2"}:
        raise AbortSealError("R1_STATIC_AUTH_R6_VARIANTS_DRIFT")
    for variant in ("G0", "G1", "G2"):
        record = variants.get(variant)
        if not isinstance(record, Mapping):
            raise AbortSealError(f"R1_STATIC_AUTH_R6_VARIANT_INVALID={variant}")
        folds = record.get("fold_inputs")
        if record.get("folds") != list(range(5)) or record.get("tasks") != 5:
            raise AbortSealError(f"R1_STATIC_AUTH_R6_FOLD_DECLARATION_DRIFT={variant}")
        if not isinstance(folds, list) or len(folds) != 5:
            raise AbortSealError(f"R1_STATIC_AUTH_R6_FOLD_COUNT_DRIFT={variant}")


def _validate_live_authorizations(paths: SealPaths) -> dict[str, dict[str, Any]]:
    # Existence check only in bootstrap; no result-root operation occurs.
    if _path_present(paths.gpu_training_complete, "GPU_TRAINING_COMPLETE"):
        raise AbortSealError(
            f"R1_GPU_TRAINING_COMPLETE_PRESENT_REFUSING_ABORT={paths.gpu_training_complete}"
        )
    patch, patch_sha, patch_size = _load_json(paths.live_patch_ready, "PATCH_READY")
    static_payload, static_sha, static_size = _load_json(
        paths.live_static_auth_ready, "STATIC_AUTH_READY"
    )
    _validate_patch_payload(patch, patch_sha)
    _validate_static_payload(static_payload, static_sha)
    return {
        "patch_ready": {
            "source_path": str(paths.live_patch_ready),
            "archive_path": str(paths.archived_patch_ready),
            "sha256": patch_sha,
            "size_bytes": patch_size,
        },
        "static_auth_ready": {
            "source_path": str(paths.live_static_auth_ready),
            "archive_path": str(paths.archived_static_auth_ready),
            "sha256": static_sha,
            "size_bytes": static_size,
        },
    }


def _validate_archived_pair(paths: SealPaths) -> None:
    patch, patch_sha, _ = _load_json(paths.archived_patch_ready, "ARCHIVED_PATCH_READY")
    static_payload, static_sha, _ = _load_json(
        paths.archived_static_auth_ready, "ARCHIVED_STATIC_AUTH_READY"
    )
    _validate_patch_payload(patch, patch_sha)
    _validate_static_payload(static_payload, static_sha)


def _existing_seal(paths: SealPaths) -> dict[str, Any] | None:
    if not _path_present(paths.aborted_receipt, "ABORTED_RECEIPT"):
        return None
    receipt, _, _ = _load_json(paths.aborted_receipt, "ABORTED_RECEIPT")
    expected = {
        "format": ABORTED_RECEIPT_FORMAT,
        "status": ABORTED_STATUS,
        "resume_authorized": False,
        "instance_id": INSTANCE_ID,
        "r1_patch_ready_sha256": R1_PATCH_READY_SHA256,
        "r1_static_auth_r6_sha256": R1_STATIC_AUTH_R6_SHA256,
        "formal_result_root_examined": False,
        "formal_result_root_modified": False,
        "formal_result_artifacts_reused": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise AbortSealError(f"EXISTING_ABORTED_RECEIPT_CONTRACT_DRIFT={key}")
    if _path_present(paths.live_patch_ready, "LIVE_PATCH_READY") or _path_present(
        paths.live_static_auth_ready, "LIVE_STATIC_AUTH_READY"
    ):
        raise AbortSealError("LIVE_AUTH_PRESENT_WITH_EXISTING_ABORTED_RECEIPT")
    _validate_archived_pair(paths)
    return receipt


def _normalize_committed_receipt_link(paths: SealPaths) -> None:
    """Finish the exclusive-link commit after a process crash.

    ``os.link`` is the no-overwrite commit primitive.  A crash between the
    link and removal of the private staging name legitimately leaves two names
    for one inode.  Admit only that exact two-link shape, remove the private
    name, and thereby restore the public receipt's required nlink==1 contract.
    """

    public = os.lstat(paths.aborted_receipt)
    if not stat.S_ISREG(public.st_mode):
        raise AbortSealError("PARTIAL_ABORT_PUBLIC_RECEIPT_NOT_REGULAR")
    temporary_present = _path_present(
        paths.temporary_final_receipt, "PARTIAL_ABORT_FINAL_TEMP"
    )
    if not temporary_present:
        if public.st_nlink != 1:
            raise AbortSealError(
                f"PARTIAL_ABORT_PUBLIC_RECEIPT_NLINK_INVALID={public.st_nlink}"
            )
        return
    temporary = os.lstat(paths.temporary_final_receipt)
    same_inode = (temporary.st_dev, temporary.st_ino) == (
        public.st_dev,
        public.st_ino,
    )
    if (
        not stat.S_ISREG(temporary.st_mode)
        or not same_inode
        or public.st_nlink != 2
        or temporary.st_nlink != 2
    ):
        raise AbortSealError("PARTIAL_ABORT_COMMITTED_LINK_IDENTITY_DRIFT")
    paths.temporary_final_receipt.unlink()


def _remove_uncommitted_final_temp(paths: SealPaths) -> None:
    if not _path_present(paths.temporary_final_receipt, "ROLLBACK_FINAL_TEMP"):
        return
    observed = os.lstat(paths.temporary_final_receipt)
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise AbortSealError("ROLLBACK_FINAL_TEMP_IDENTITY_INVALID")
    paths.temporary_final_receipt.unlink()


def _recover_partial_apply(paths: SealPaths) -> str | None:
    if not _path_present(paths.partial_receipt, "PARTIAL_ABORT_RECEIPT"):
        return None
    raw, _, _ = _read_regular_bytes(
        paths.partial_receipt, "PARTIAL_ABORT_RECEIPT", allow_empty=True
    )
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        decoded = None
    if not isinstance(decoded, Mapping):
        # The journal is created and fsynced before the archive directory or any
        # authorization move.  Therefore an interrupted private journal write is
        # recoverable only while the exact live authorization pair is intact and
        # no later transaction artifact exists.  Any other shape remains a
        # fail-closed integrity error.
        later_artifacts = (
            paths.archive_root,
            paths.aborted_receipt,
            paths.temporary_final_receipt,
        )
        if any(
            _path_present(path, "PARTIAL_ABORT_LATER_ARTIFACT")
            for path in later_artifacts
        ):
            raise AbortSealError("PARTIAL_ABORT_RECEIPT_JSON_INVALID_AFTER_PROGRESS")
        _validate_live_authorizations(paths)
        paths.partial_receipt.unlink()
        return "ROLLED_BACK"
    payload = dict(decoded)
    expected = {
        "format": ABORTED_RECEIPT_FORMAT,
        "status": "ABORT_SEAL_APPLY_IN_PROGRESS_FAIL_CLOSED",
        "resume_authorized": False,
        "instance_id": INSTANCE_ID,
        "r1_patch_ready_sha256": R1_PATCH_READY_SHA256,
        "r1_static_auth_r6_sha256": R1_STATIC_AUTH_R6_SHA256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AbortSealError(f"PARTIAL_ABORT_RECEIPT_SCHEMA_DRIFT={key}")

    if _path_present(paths.aborted_receipt, "PARTIAL_ABORT_PUBLIC_RECEIPT"):
        if _path_present(paths.live_patch_ready, "PARTIAL_LIVE_PATCH") or _path_present(
            paths.live_static_auth_ready, "PARTIAL_LIVE_STATIC"
        ):
            raise AbortSealError("PARTIAL_ABORT_COMMITTED_WITH_LIVE_AUTH")
        _normalize_committed_receipt_link(paths)
        _validate_archived_pair(paths)
        receipt = _existing_seal(paths)
        if receipt is None:  # pragma: no cover - guarded by lstat above
            raise AbortSealError("PARTIAL_ABORT_COMMITTED_RECEIPT_LOST")
        paths.partial_receipt.unlink()
        return "COMMITTED"

    _remove_uncommitted_final_temp(paths)
    pairs = (
        (paths.archived_static_auth_ready, paths.live_static_auth_ready),
        (paths.archived_patch_ready, paths.live_patch_ready),
    )
    for archived, live in pairs:
        archived_present = _path_present(archived, "PARTIAL_ARCHIVED_AUTH")
        live_present = _path_present(live, "PARTIAL_LIVE_AUTH")
        if archived_present and live_present:
            raise AbortSealError(f"PARTIAL_ABORT_BOTH_LIVE_AND_ARCHIVED={live}")
        if archived_present:
            os.replace(archived, live)
    paths.partial_receipt.unlink()
    try:
        paths.archive_root.rmdir()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise AbortSealError("PARTIAL_ABORT_ARCHIVE_NOT_EMPTY_AFTER_RECOVERY") from exc
    return "ROLLED_BACK"


def _recover_orphan_empty_archive(paths: SealPaths) -> None:
    """Recover the sole crash shape produced before the journal-first fix."""

    if not _path_present(paths.archive_root, "ORPHAN_ABORT_ARCHIVE_ROOT"):
        return
    if _path_present(paths.aborted_receipt, "ORPHAN_ABORT_PUBLIC_RECEIPT"):
        return
    try:
        paths.archive_root.rmdir()
    except OSError as exc:
        raise AbortSealError("ORPHAN_ABORT_ARCHIVE_NOT_EMPTY") from exc


def build_dry_run_plan(
    *,
    project_root: Path,
    stopped_receipt_path: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    paths = resolve_paths(project_root)
    _assert_no_symlink_components(paths.bootstrap_root, "BOOTSTRAP_ROOT")
    stopped, stopped_sha, age = validate_stopped_receipt(stopped_receipt_path, now=now)
    existing = _existing_seal(paths)
    if existing is not None:
        return {
            "format": ABORTED_RECEIPT_FORMAT,
            "status": "DRY_RUN_EXISTING_ABORTED_SEAL_VALID",
            "would_modify_filesystem": False,
            "aborted_receipt": str(paths.aborted_receipt),
        }
    if _path_present(paths.partial_receipt, "PARTIAL_ABORT_RECEIPT"):
        raise AbortSealError("PARTIAL_ABORT_RECEIPT_REQUIRES_APPLY_RECOVERY")
    authorizations = _validate_live_authorizations(paths)
    return {
        "format": ABORTED_RECEIPT_FORMAT,
        "status": "DRY_RUN_R1_ABORT_SEAL_READY",
        "would_modify_filesystem": False,
        "apply_required_for_mutation": True,
        "reason": ABORTED_STATUS,
        "instance_id": stopped["instance_id"],
        "stopped_receipt_path": str(stopped_receipt_path.resolve()),
        "stopped_receipt_sha256": stopped_sha,
        "stopped_receipt_age_seconds": age,
        "authorizations": authorizations,
        "would_write": str(paths.aborted_receipt),
        "formal_result_root": str(paths.result_root),
        "formal_result_root_examined": False,
        "formal_result_root_modified": False,
        "formal_result_artifacts_reused": False,
    }


def _rollback_moves(paths: SealPaths) -> None:
    _remove_uncommitted_final_temp(paths)
    for archived, live in (
        (paths.archived_static_auth_ready, paths.live_static_auth_ready),
        (paths.archived_patch_ready, paths.live_patch_ready),
    ):
        if _path_present(archived, "ROLLBACK_ARCHIVED_AUTH") and not _path_present(
            live, "ROLLBACK_LIVE_AUTH"
        ):
            os.replace(archived, live)
    if _path_present(paths.partial_receipt, "ROLLBACK_PARTIAL_RECEIPT"):
        paths.partial_receipt.unlink()
    try:
        paths.archive_root.rmdir()
    except (FileNotFoundError, OSError):
        pass


def apply_seal(
    *,
    project_root: Path,
    stopped_receipt_path: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    paths = resolve_paths(project_root)
    _assert_no_symlink_components(paths.bootstrap_root, "BOOTSTRAP_ROOT")
    timestamp = now or datetime.now(timezone.utc)
    stopped, stopped_sha, age = validate_stopped_receipt(
        stopped_receipt_path, now=timestamp
    )
    _recover_partial_apply(paths)
    existing = _existing_seal(paths)
    if existing is not None:
        return existing
    _recover_orphan_empty_archive(paths)
    authorizations = _validate_live_authorizations(paths)
    for target in (paths.archived_patch_ready, paths.archived_static_auth_ready):
        if _path_present(target, "ABORT_ARCHIVE_TARGET"):
            raise AbortSealError(f"ABORT_ARCHIVE_TARGET_ALREADY_EXISTS={target}")
    if _path_present(paths.archive_root, "ABORT_ARCHIVE_ROOT"):
        raise AbortSealError(f"ABORT_ARCHIVE_ROOT_ALREADY_EXISTS={paths.archive_root}")

    pending = {
        "format": ABORTED_RECEIPT_FORMAT,
        "status": "ABORT_SEAL_APPLY_IN_PROGRESS_FAIL_CLOSED",
        "resume_authorized": False,
        "instance_id": INSTANCE_ID,
        "stopped_receipt_sha256": stopped_sha,
        "r1_patch_ready_sha256": R1_PATCH_READY_SHA256,
        "r1_static_auth_r6_sha256": R1_STATIC_AUTH_R6_SHA256,
    }
    commit_created = False
    try:
        with paths.partial_receipt.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(pending, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        paths.archive_root.mkdir(parents=False, exist_ok=False)
        os.replace(paths.live_patch_ready, paths.archived_patch_ready)
        os.replace(paths.live_static_auth_ready, paths.archived_static_auth_ready)
        _validate_archived_pair(paths)

        receipt = {
            "format": ABORTED_RECEIPT_FORMAT,
            "status": ABORTED_STATUS,
            "sealed_at": timestamp.astimezone(timezone.utc).isoformat(),
            "resume_authorized": False,
            "superseded_by_namespace": "v32_g012_paid_gpu_20260901_r2",
            "instance_id": INSTANCE_ID,
            "instance_stop_evidence": {
                "instance_id": stopped["instance_id"],
                "observed_state": stopped["observed_state"],
                "gpu_billing_active": stopped["gpu_billing_active"],
                "checked_at": stopped["checked_at"],
                "age_seconds_at_seal": age,
                "receipt_path": str(stopped_receipt_path.resolve()),
                "receipt_sha256": stopped_sha,
            },
            "r1_patch_ready_sha256": R1_PATCH_READY_SHA256,
            "r1_static_auth_r6_sha256": R1_STATIC_AUTH_R6_SHA256,
            "archived_authorizations": authorizations,
            "live_authorizations_removed": True,
            "gpu_training_complete_present": False,
            "formal_result_root": str(paths.result_root),
            "formal_result_root_examined": False,
            "formal_result_root_modified": False,
            "formal_result_artifacts_reused": False,
            "deletion_performed": False,
        }
        with paths.temporary_final_receipt.open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                paths.temporary_final_receipt,
                paths.aborted_receipt,
                follow_symlinks=False,
            )
            commit_created = True
        except OSError as exc:
            raise AbortSealError("ABORTED_RECEIPT_EXCLUSIVE_COMMIT_FAILED") from exc
        os.unlink(paths.temporary_final_receipt)
    except BaseException as exc:
        if commit_created:
            # The public receipt is already committed.  Preserve the journal
            # and both names so the next invocation can verify and normalize
            # the exact same-inode nlink==2 crash state.
            if isinstance(exc, AbortSealError):
                raise
            raise AbortSealError(f"ABORT_SEAL_COMMIT_CLEANUP_FAILED={exc}") from exc
        try:
            _rollback_moves(paths)
        except BaseException as rollback_exc:
            raise AbortSealError(
                f"ABORT_SEAL_ROLLBACK_FAILED={rollback_exc}"
            ) from exc
        if isinstance(exc, AbortSealError):
            raise
        raise AbortSealError(f"ABORT_SEAL_TRANSACTION_FAILED={exc}") from exc
    paths.partial_receipt.unlink()
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stopped-receipt", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move exact live r1 auth receipts and atomically write ABORTED.json.",
    )
    args = parser.parse_args(argv)
    try:
        payload = (
            apply_seal(
                project_root=PROJECT_ROOT,
                stopped_receipt_path=args.stopped_receipt,
            )
            if args.apply
            else build_dry_run_plan(
                project_root=PROJECT_ROOT,
                stopped_receipt_path=args.stopped_receipt,
            )
        )
    except (AbortSealError, FileNotFoundError, NotADirectoryError) as exc:
        print(str(exc), file=sys.stderr)
        return 22
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
