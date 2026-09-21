#!/usr/bin/env python3
"""Verify and install the fixed CompShare CPU/no-GPU bootstrap artifacts.

The manifest is an external trust object.  This tool never discovers files from
the source tree: it accepts only manifest rows with fixed, safe relative paths,
opens every file without following links, requires a single regular-file link,
and hashes through the same descriptor before checking that the pathname still
names the same inode.  ``--install`` copies verified bytes into fixed
CancerLncAtlas bootstrap/control locations using create-exclusive temporary
files and a no-replace hard-link commit.

It does not contact CompShare and it never starts an instance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


FORMAT = "CANCERLNCATLAS_COMPSHARE_CPU_NOGPU_BOOTSTRAP_MANIFEST_V1"
READY_STATUS = "BOOTSTRAP_ARTIFACTS_FROZEN"
DRAFT_STATUS = "DRAFT_NOT_LAUNCHABLE"
INSTANCE_ID = "uhost-1up504geeqzj"
PROJECT_ROOT = PurePosixPath("./data/CancerLncAtlas")
NAMESPACE = "v32_compshare_cpu_nogpu_bootstrap_20260901_r1"
PHASES = frozenset({"prepare_r2", "authorize_oracle", "both"})
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_ARTIFACTS = 64
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 768 * 1024 * 1024

ALLOWED_REMOTE_EXACT = frozenset(
    {
        "/tmp/v32_g012_r2_code_archive_20260901_r1.tar.gz",
    }
)
ALLOWED_REMOTE_PREFIXES = (
    "./data/CancerLncAtlas/runtime/bootstrap/",
    "./data/CancerLncAtlas/runtime/oracle_gates/",
)
CPU_ROOT = f"./data/CancerLncAtlas/runtime/bootstrap/{NAMESPACE}"
FORMAL_ROOT = "./data/CancerLncAtlas/runtime/bootstrap/v32_g012_paid_gpu_20260901_r2"
ORACLE_NAME = "v32_group_shared_oracle_paid_gpu_20260901_r2"
ORACLE_BOOTSTRAP = f"./data/CancerLncAtlas/runtime/bootstrap/{ORACLE_NAME}"
ORACLE_CONTROL = f"./data/CancerLncAtlas/runtime/oracle_gates/{ORACLE_NAME}"
EXPECTED_ARTIFACT_SPECS: dict[str, tuple[str, str, str, str, int]] = {
    "bootstrap_manifest_verifier": (
        "both", "scripts/verify_v32_compshare_cpu_nogpu_bootstrap_manifest.py",
        "artifacts/bootstrap_manifest_verifier/verify_v32_compshare_cpu_nogpu_bootstrap_manifest.py",
        f"{CPU_ROOT}/verify_v32_compshare_cpu_nogpu_bootstrap_manifest.py", 0o700,
    ),
    "dynamic_receipt_verifier": (
        "both", "scripts/verify_v32_compshare_cpu_nogpu_dynamic_receipt.py",
        "artifacts/dynamic_receipt_verifier/verify_v32_compshare_cpu_nogpu_dynamic_receipt.py",
        f"{CPU_ROOT}/verify_v32_compshare_cpu_nogpu_dynamic_receipt.py", 0o700,
    ),
    "remote_cpu_bootstrap_driver": (
        "both", "scripts/cloud_bootstrap_v32_compshare_cpu_nogpu_20260901_r1.sh",
        "artifacts/remote_cpu_bootstrap_driver/cloud_bootstrap_v32_compshare_cpu_nogpu_20260901_r1.sh",
        f"{CPU_ROOT}/cloud_bootstrap_v32_compshare_cpu_nogpu_20260901_r1.sh", 0o700,
    ),
    "oracle_overlay_verifier": (
        "prepare_r2", "scripts/verify_v32_group_shared_oracle_overlay_bundle.py",
        "artifacts/oracle_overlay_verifier/verify_v32_group_shared_oracle_overlay_bundle.py",
        f"{CPU_ROOT}/verify_v32_group_shared_oracle_overlay_bundle.py", 0o700,
    ),
    "r1_abort_sealer": (
        "prepare_r2", "scripts/seal_v32_g012_r1_aborted_no_gpu.py",
        "artifacts/r1_abort_sealer/seal_v32_g012_r1_aborted_no_gpu.py",
        f"{FORMAL_ROOT}/seal_v32_g012_r1_aborted_no_gpu.py", 0o700,
    ),
    "r2_input_reuse_materializer": (
        "prepare_r2", "scripts/materialize_v32_g012_r2_input_reuse_ready.py",
        "artifacts/r2_input_reuse_materializer/materialize_v32_g012_r2_input_reuse_ready.py",
        f"{FORMAL_ROOT}/materialize_v32_g012_r2_input_reuse_ready.py", 0o700,
    ),
    "shared_jit_budget_generator": (
        "both", "scripts/local_materialize_compshare_jit_budget_20260901_r2.ps1",
        "artifacts/shared_jit_budget_generator/local_materialize_compshare_jit_budget_20260901_r2.ps1",
        f"{CPU_ROOT}/local_materialize_compshare_jit_budget_20260901_r2.ps1", 0o600,
    ),
    "oracle_code_bundle": (
        "prepare_r2", f"artifacts/{NAMESPACE}/ORACLE_CODE_BUNDLE.tar.gz",
        "artifacts/oracle_code_bundle/ORACLE_CODE_BUNDLE.tar.gz",
        f"{ORACLE_BOOTSTRAP}/ORACLE_CODE_BUNDLE.tar.gz", 0o600,
    ),
    "oracle_code_bundle_manifest": (
        "prepare_r2", f"artifacts/{NAMESPACE}/ORACLE_CODE_BUNDLE.MANIFEST.json",
        "artifacts/oracle_code_bundle_manifest/ORACLE_CODE_BUNDLE.MANIFEST.json",
        f"{ORACLE_BOOTSTRAP}/ORACLE_CODE_BUNDLE.MANIFEST.json", 0o600,
    ),
    "oracle_external_deployment_manifest": (
        "prepare_r2", "docs/v32_group_shared_oracle_external_deployment_manifest_20260901_r2.json",
        "artifacts/oracle_external_deployment_manifest/v32_group_shared_oracle_external_deployment_manifest_20260901_r2.json",
        f"{ORACLE_CONTROL}/EXTERNAL_DEPLOYMENT_MANIFEST.json", 0o600,
    ),
    "oracle_static_verifier": (
        "prepare_r2", "scripts/validate_v32_group_shared_oracle_static_auth_ready_r2.py",
        "artifacts/oracle_static_verifier/validate_v32_group_shared_oracle_static_auth_ready_r2.py",
        f"{ORACLE_CONTROL}/validate_v32_group_shared_oracle_static_auth_ready_r2.py", 0o700,
    ),
    "oracle_cpu_authorizer": (
        "prepare_r2", "scripts/cloud_authorize_v32_group_shared_oracle_no_gpu_20260901_r2.sh",
        "artifacts/oracle_cpu_authorizer/cloud_authorize_v32_group_shared_oracle_no_gpu_20260901_r2.sh",
        f"{ORACLE_CONTROL}/cloud_authorize_v32_group_shared_oracle_no_gpu_20260901_r2.sh", 0o700,
    ),
    "oracle_paid_launcher": (
        "prepare_r2", "scripts/server_launch_v32_group_shared_oracle_paid_gpu_20260901_r2.sh",
        "artifacts/oracle_paid_launcher/server_launch_v32_group_shared_oracle_paid_gpu_20260901_r2.sh",
        f"{ORACLE_CONTROL}/server_launch_v32_group_shared_oracle_paid_gpu_20260901_r2.sh", 0o700,
    ),
    "oracle_local_supervisor": (
        "prepare_r2", "scripts/local_supervise_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1",
        "artifacts/oracle_local_supervisor/local_supervise_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1",
        f"{ORACLE_CONTROL}/local_supervise_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1", 0o600,
    ),
    "oracle_local_dispatcher": (
        "prepare_r2", "scripts/local_dispatch_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1",
        "artifacts/oracle_local_dispatcher/local_dispatch_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1",
        f"{ORACLE_CONTROL}/local_dispatch_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1", 0o600,
    ),
    "oracle_log_guard": (
        "prepare_r2", "scripts/v32_group_shared_oracle_log_guard_r2.psm1",
        "artifacts/oracle_log_guard/v32_group_shared_oracle_log_guard_r2.psm1",
        f"{ORACLE_CONTROL}/v32_group_shared_oracle_log_guard_r2.psm1", 0o600,
    ),
}


class BootstrapManifestError(RuntimeError):
    """The deployment manifest or one of its artifacts is unsafe or drifted."""


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _identity(value: os.stat_result, *, include_ctime: bool = True) -> tuple[int, ...]:
    base = (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    return base + ((value.st_ctime_ns,) if include_ctime else ())


def _assert_no_link_components(path: Path, label: str) -> None:
    target = _absolute_lexical(path)
    for component in reversed([target, *target.parents]):
        try:
            observed = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise BootstrapManifestError(
                f"{label}_COMPONENT_LSTAT_FAILED={component}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode) or getattr(
            os.path, "isjunction", lambda _: False
        )(component):
            raise BootstrapManifestError(f"{label}_LINK_COMPONENT_FORBIDDEN={component}")


def _open_bound_regular(
    path: Path,
    label: str,
    *,
    allowed_nlinks: frozenset[int] = frozenset({1}),
) -> tuple[int, os.stat_result, list[int], Path]:
    target = _absolute_lexical(path)
    _assert_no_link_components(target, label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    held: list[int] = []
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
            held.append(current)
            for component in target.parts[1:-1]:
                current = os.open(component, directory_flags, dir_fd=current)
                if not stat.S_ISDIR(os.fstat(current).st_mode):
                    raise BootstrapManifestError(f"{label}_PARENT_NOT_DIRECTORY={component}")
                held.append(current)
            descriptor = os.open(target.name, flags, dir_fd=current)
        else:
            descriptor = os.open(target, flags)
    except (OSError, BootstrapManifestError) as exc:
        for directory in reversed(held):
            try:
                os.close(directory)
            except OSError:
                pass
        if isinstance(exc, BootstrapManifestError):
            raise
        raise BootstrapManifestError(f"{label}_OPEN_FAILED={target}") from exc
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode) or observed.st_size <= 0:
        os.close(descriptor)
        for directory in reversed(held):
            os.close(directory)
        raise BootstrapManifestError(f"{label}_NOT_NONEMPTY_REGULAR={target}")
    if observed.st_nlink not in allowed_nlinks:
        os.close(descriptor)
        for directory in reversed(held):
            os.close(directory)
        raise BootstrapManifestError(f"{label}_NLINK_NOT_ONE={target}")
    return descriptor, observed, held, target


def _finish_bound_regular(
    descriptor: int,
    before: os.stat_result,
    held: Sequence[int],
    path: Path,
    label: str,
) -> None:
    try:
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise BootstrapManifestError(f"{label}_CHANGED_DURING_READ={path}")
        rebound = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(rebound.st_mode) or _identity(
            rebound, include_ctime=False
        ) != _identity(after, include_ctime=False):
            raise BootstrapManifestError(f"{label}_PATH_REPLACED_DURING_READ={path}")
    finally:
        os.close(descriptor)
        for directory in reversed(held):
            os.close(directory)


def _read_bound(
    path: Path,
    label: str,
    *,
    capture: bool = False,
    allowed_nlinks: frozenset[int] = frozenset({1}),
) -> tuple[str, int, bytes | None]:
    descriptor, before, held, target = _open_bound_regular(
        path, label, allowed_nlinks=allowed_nlinks
    )
    digest = hashlib.sha256()
    content = bytearray() if capture else None
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if content is not None:
                content.extend(chunk)
        return digest.hexdigest(), int(before.st_size), bytes(content) if content is not None else None
    finally:
        _finish_bound_regular(descriptor, before, held, target, label)


def _safe_relative(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise BootstrapManifestError(f"{label}_INVALID={value!r}")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or not candidate.parts or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise BootstrapManifestError(f"{label}_UNSAFE={value}")
    return candidate


def _safe_remote(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value.startswith("/") or "\\" in value or "\x00" in value:
        raise BootstrapManifestError(f"REMOTE_INSTALL_PATH_INVALID={value!r}")
    normalized = PurePosixPath(value)
    if str(normalized) != value or any(part in {"", ".", ".."} for part in normalized.parts[1:]):
        raise BootstrapManifestError(f"REMOTE_INSTALL_PATH_UNSAFE={value}")
    if value not in ALLOWED_REMOTE_EXACT and not any(
        value.startswith(prefix) for prefix in ALLOWED_REMOTE_PREFIXES
    ):
        raise BootstrapManifestError(f"REMOTE_INSTALL_PATH_OUTSIDE_SCOPE={value}")
    return normalized


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BootstrapManifestError(f"{label}_MAPPING_REQUIRED")
    return value


def _parse_manifest_bytes(
    raw: bytes,
    digest: str,
    expected_sha256: str,
    *,
    allow_draft: bool,
) -> tuple[dict[str, Any], str]:
    if not SHA256_RE.fullmatch(expected_sha256):
        raise BootstrapManifestError("EXPECTED_MANIFEST_SHA256_INVALID")
    if digest != expected_sha256:
        raise BootstrapManifestError(f"BOOTSTRAP_MANIFEST_SHA256_DRIFT={digest}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BootstrapManifestError(f"BOOTSTRAP_MANIFEST_DUPLICATE_JSON_KEY={key}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapManifestError("BOOTSTRAP_MANIFEST_JSON_INVALID") from exc
    manifest = dict(_mapping(payload, "BOOTSTRAP_MANIFEST"))
    required_manifest_keys = {
        "format",
        "status",
        "namespace",
        "instance_id",
        "remote_project_root",
        "gpu_start_permitted",
        "without_gpu_spec",
        "formal_r2_code_promotion_permitted",
        "formal_r2_patch_ready_permitted",
        "artifacts",
    }
    if set(manifest) != required_manifest_keys:
        raise BootstrapManifestError(
            f"BOOTSTRAP_MANIFEST_KEY_SET_DRIFT="
            f"{sorted(required_manifest_keys - set(manifest))}:"
            f"{sorted(set(manifest) - required_manifest_keys)}"
        )
    expected = {
        "format": FORMAT,
        "namespace": NAMESPACE,
        "instance_id": INSTANCE_ID,
        "remote_project_root": str(PROJECT_ROOT),
        "gpu_start_permitted": False,
        "without_gpu_spec": "A",
        "formal_r2_code_promotion_permitted": False,
        "formal_r2_patch_ready_permitted": False,
    }
    for key, required in expected.items():
        if manifest.get(key) != required:
            raise BootstrapManifestError(f"BOOTSTRAP_MANIFEST_CONTRACT_DRIFT={key}")
    status = manifest.get("status")
    if status != READY_STATUS and not (allow_draft and status == DRAFT_STATUS):
        raise BootstrapManifestError(f"BOOTSTRAP_MANIFEST_NOT_FROZEN={status}")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_ARTIFACTS:
        raise BootstrapManifestError("BOOTSTRAP_ARTIFACT_CARDINALITY_INVALID")
    return manifest, digest


def load_manifest(path: Path, expected_sha256: str, *, allow_draft: bool = False) -> tuple[dict[str, Any], str]:
    digest, _, raw = _read_bound(path, "BOOTSTRAP_MANIFEST", capture=True)
    assert raw is not None
    return _parse_manifest_bytes(
        raw,
        digest,
        expected_sha256,
        allow_draft=allow_draft,
    )


def load_manifest_fd(
    descriptor: int,
    expected_sha256: str,
    *,
    allow_draft: bool = False,
) -> tuple[dict[str, Any], str]:
    if isinstance(descriptor, bool) or not isinstance(descriptor, int) or descriptor < 0:
        raise BootstrapManifestError("BOOTSTRAP_MANIFEST_FD_INVALID")
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        raise BootstrapManifestError("BOOTSTRAP_MANIFEST_FD_UNAVAILABLE") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_nlink != 1:
        raise BootstrapManifestError("BOOTSTRAP_MANIFEST_FD_TYPE_OR_NLINK_INVALID")
    digest = hashlib.sha256()
    raw = bytearray()
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            raw.extend(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise BootstrapManifestError("BOOTSTRAP_MANIFEST_FD_READ_FAILED") from exc
    if _identity(before) != _identity(after):
        raise BootstrapManifestError("BOOTSTRAP_MANIFEST_FD_CHANGED_DURING_READ")
    return _parse_manifest_bytes(
        bytes(raw),
        digest.hexdigest(),
        expected_sha256,
        allow_draft=allow_draft,
    )


def validate_records(
    manifest: Mapping[str, Any],
    *,
    allow_relaxed_spec_for_local_stub: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    locals_seen: set[str] = set()
    stages: set[str] = set()
    targets: set[str] = set()
    total = 0
    for raw in manifest["artifacts"]:
        row = dict(_mapping(raw, "BOOTSTRAP_ARTIFACT"))
        required_keys = {
            "name",
            "phase",
            "local_relative_path",
            "stage_relative_path",
            "remote_install_path",
            "sha256",
            "size_bytes",
            "mode",
        }
        if set(row) != required_keys:
            raise BootstrapManifestError(f"BOOTSTRAP_ARTIFACT_SCHEMA_DRIFT={sorted(set(row))}")
        name = row["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
            raise BootstrapManifestError(f"BOOTSTRAP_ARTIFACT_NAME_INVALID={name!r}")
        phase = row["phase"]
        if phase not in PHASES:
            raise BootstrapManifestError(f"BOOTSTRAP_ARTIFACT_PHASE_INVALID={name}")
        local = _safe_relative(row["local_relative_path"], "LOCAL_RELATIVE_PATH").as_posix()
        stage = _safe_relative(row["stage_relative_path"], "STAGE_RELATIVE_PATH").as_posix()
        target = str(_safe_remote(row["remote_install_path"]))
        digest = row["sha256"]
        size = row["size_bytes"]
        mode = row["mode"]
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise BootstrapManifestError(f"BOOTSTRAP_ARTIFACT_SHA256_INVALID={name}")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_ARTIFACT_BYTES:
            raise BootstrapManifestError(f"BOOTSTRAP_ARTIFACT_SIZE_INVALID={name}")
        if mode not in {0o600, 0o700}:
            raise BootstrapManifestError(f"BOOTSTRAP_ARTIFACT_MODE_INVALID={name}")
        spec = EXPECTED_ARTIFACT_SPECS.get(name)
        if spec is None or (
            not allow_relaxed_spec_for_local_stub
            and (phase, local, stage, target, mode) != spec
        ):
            raise BootstrapManifestError(f"BOOTSTRAP_ARTIFACT_EXACT_SPEC_DRIFT={name}")
        if name in names or local in locals_seen or stage in stages or target in targets:
            raise BootstrapManifestError(f"BOOTSTRAP_ARTIFACT_DUPLICATE={name}")
        names.add(name)
        locals_seen.add(local)
        stages.add(stage)
        targets.add(target)
        total += size
        records.append(row)
    if total > MAX_TOTAL_BYTES:
        raise BootstrapManifestError("BOOTSTRAP_ARTIFACT_TOTAL_BYTES_EXCEEDS_CAP")
    required_names = set(EXPECTED_ARTIFACT_SPECS)
    if names != required_names:
        raise BootstrapManifestError(
            f"BOOTSTRAP_ARTIFACT_NAME_SET_DRIFT={sorted(required_names - names)}:{sorted(names - required_names)}"
        )
    return records


def verify_artifacts(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    location_field: str,
    phase: str,
    allow_relaxed_spec_for_local_stub: bool = False,
) -> list[dict[str, Any]]:
    bound_root = _absolute_lexical(root)
    _assert_no_link_components(bound_root, "ARTIFACT_ROOT")
    records = validate_records(
        manifest,
        allow_relaxed_spec_for_local_stub=allow_relaxed_spec_for_local_stub,
    )
    verified: list[dict[str, Any]] = []
    for row in records:
        if phase != "both" and row["phase"] not in {phase, "both"}:
            continue
        relative = _safe_relative(row[location_field], location_field.upper())
        path = bound_root.joinpath(*relative.parts)
        lexical = _absolute_lexical(path)
        if lexical != bound_root and bound_root not in lexical.parents:
            raise BootstrapManifestError(f"ARTIFACT_PATH_ESCAPES_ROOT={relative}")
        digest, size, _ = _read_bound(lexical, f"ARTIFACT_{row['name']}")
        if digest != row["sha256"] or size != row["size_bytes"]:
            raise BootstrapManifestError(f"BOOTSTRAP_ARTIFACT_BYTES_DRIFT={row['name']}")
        verified.append(
            {
                "name": row["name"],
                "sha256": digest,
                "size_bytes": size,
                "path": str(lexical),
            }
        )
    return verified


def _mkdir_scoped(path: Path) -> None:
    target = _absolute_lexical(path)
    _safe_remote(str(target / ".scope_probe"))
    missing: list[Path] = []
    cursor = target
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    _assert_no_link_components(cursor, "INSTALL_EXISTING_PARENT")
    if not cursor.is_dir():
        raise BootstrapManifestError(f"INSTALL_PARENT_NOT_DIRECTORY={cursor}")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
        _assert_no_link_components(directory, "INSTALL_CREATED_PARENT")


def _copy_bound_to_target(source: Path, target: Path, row: Mapping[str, Any]) -> None:
    _safe_remote(str(target))
    _mkdir_scoped(target.parent)
    temporary = target.with_name(f".{target.name}.{row['sha256']}.install.partial")

    def candidate(path: Path, label: str, links: frozenset[int]) -> tuple[str, int, os.stat_result]:
        digest, size, _ = _read_bound(path, label, allowed_nlinks=links)
        observed = os.stat(path, follow_symlinks=False)
        if (
            digest != row["sha256"]
            or size != row["size_bytes"]
            or stat.S_IMODE(observed.st_mode) != row["mode"]
        ):
            raise BootstrapManifestError(f"{label}_BYTES_OR_MODE_DRIFT={row['name']}")
        return digest, size, observed

    target_present = os.path.lexists(target)
    partial_present = os.path.lexists(temporary)
    if target_present:
        _, _, target_stat = candidate(
            target, "EXISTING_INSTALL_TARGET", frozenset({1, 2})
        )
        if partial_present:
            _, _, partial_stat = candidate(
                temporary, "RECOVERY_INSTALL_PARTIAL", frozenset({1, 2})
            )
            same_inode = (target_stat.st_dev, target_stat.st_ino) == (
                partial_stat.st_dev,
                partial_stat.st_ino,
            )
            if same_inode and target_stat.st_nlink == partial_stat.st_nlink == 2:
                temporary.unlink()
            elif not same_inode and target_stat.st_nlink == partial_stat.st_nlink == 1:
                # The target is already committed and exact; an independently
                # exact stable partial is a recoverable pre-link crash.
                temporary.unlink()
            else:
                raise BootstrapManifestError(f"INSTALL_RECOVERY_INODE_DRIFT={row['name']}")
        digest, size, _ = _read_bound(target, f"RECOVERED_EXISTING_{row['name']}")
        if digest != row["sha256"] or size != row["size_bytes"]:
            raise BootstrapManifestError(f"EXISTING_INSTALL_TARGET_DRIFT={row['name']}")
        return

    if partial_present:
        partial_stat = os.lstat(temporary)
        if (
            not stat.S_ISREG(partial_stat.st_mode)
            or partial_stat.st_nlink != 1
            or stat.S_IMODE(partial_stat.st_mode) != row["mode"]
        ):
            raise BootstrapManifestError(f"INSTALL_PARTIAL_TYPE_DRIFT={row['name']}")
        try:
            candidate(temporary, "RESUMABLE_INSTALL_PARTIAL", frozenset({1}))
        except BootstrapManifestError:
            # An interrupted short write is safe to discard only while the
            # no-replace target is absent and the stable partial is a private
            # regular single-link file in the already-validated parent.
            temporary.unlink()
            partial_present = False
        if partial_present:
            os.link(temporary, target, follow_symlinks=False)
            temporary.unlink()
            digest, size, _ = _read_bound(target, f"RESUMED_{row['name']}")
            if digest != row["sha256"] or size != row["size_bytes"]:
                raise BootstrapManifestError(f"RESUMED_INSTALL_TARGET_DRIFT={row['name']}")
            return

    source_fd, source_before, held, source_path = _open_bound_regular(source, f"INSTALL_SOURCE_{row['name']}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    target_fd: int | None = None
    committed = False
    try:
        target_fd = os.open(temporary, flags, int(row["mode"]))
        os.fchmod(target_fd, int(row["mode"]))
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            offset = 0
            while offset < len(chunk):
                offset += os.write(target_fd, chunk[offset:])
        os.fsync(target_fd)
        os.close(target_fd)
        target_fd = None
        if digest.hexdigest() != row["sha256"] or size != row["size_bytes"]:
            raise BootstrapManifestError(f"INSTALL_COPY_BYTES_DRIFT={row['name']}")
        os.link(temporary, target, follow_symlinks=False)
        committed = True
    finally:
        if target_fd is not None:
            os.close(target_fd)
        _finish_bound_regular(source_fd, source_before, held, source_path, f"INSTALL_SOURCE_{row['name']}")
        if os.path.lexists(temporary) and committed:
            temporary.unlink()
    if not committed:
        raise BootstrapManifestError(f"INSTALL_COMMIT_FAILED={row['name']}")
    digest, size, _ = _read_bound(target, f"INSTALLED_{row['name']}")
    mode = stat.S_IMODE(os.stat(target, follow_symlinks=False).st_mode)
    if digest != row["sha256"] or size != row["size_bytes"] or mode != row["mode"]:
        raise BootstrapManifestError(f"INSTALLED_TARGET_RECHECK_DRIFT={row['name']}")


def install_artifacts(
    manifest: Mapping[str, Any],
    *,
    staging_root: Path,
    phase: str,
    allow_relaxed_spec_for_local_stub: bool = False,
) -> list[dict[str, Any]]:
    verified = verify_artifacts(
        manifest,
        root=staging_root,
        location_field="stage_relative_path",
        phase=phase,
        allow_relaxed_spec_for_local_stub=allow_relaxed_spec_for_local_stub,
    )
    by_name = {
        row["name"]: row
        for row in validate_records(
            manifest,
            allow_relaxed_spec_for_local_stub=allow_relaxed_spec_for_local_stub,
        )
    }
    for observed in verified:
        row = by_name[observed["name"]]
        _copy_bound_to_target(Path(observed["path"]), Path(row["remote_install_path"]), row)
    return verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    manifest_source = parser.add_mutually_exclusive_group(required=True)
    manifest_source.add_argument("--manifest", type=Path)
    manifest_source.add_argument("--manifest-fd", type=int)
    parser.add_argument("--expected-manifest-sha256", required=True)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument("--source-root", type=Path)
    roots.add_argument("--staging-root", type=Path)
    parser.add_argument("--phase", choices=sorted(PHASES), default="both")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--allow-draft-for-local-test", action="store_true")
    parser.add_argument("--allow-relaxed-spec-for-local-stub", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.allow_relaxed_spec_for_local_stub and (
            not args.allow_draft_for_local_test
            or os.environ.get("CANCERLNCATLAS_COMPSHARE_STUB_E2E") != "1"
        ):
            raise BootstrapManifestError("RELAXED_SPEC_LOCAL_STUB_GUARD_MISSING")
        if args.manifest_fd is not None:
            manifest, manifest_sha = load_manifest_fd(
                args.manifest_fd,
                args.expected_manifest_sha256,
                allow_draft=args.allow_draft_for_local_test,
            )
        else:
            assert args.manifest is not None
            manifest, manifest_sha = load_manifest(
                args.manifest,
                args.expected_manifest_sha256,
                allow_draft=args.allow_draft_for_local_test,
            )
        if args.install:
            if args.staging_root is None:
                raise BootstrapManifestError("INSTALL_REQUIRES_STAGING_ROOT")
            if manifest.get("status") != READY_STATUS:
                raise BootstrapManifestError("DRAFT_MANIFEST_INSTALL_FORBIDDEN")
            verified = install_artifacts(
                manifest,
                staging_root=args.staging_root,
                phase=args.phase,
                allow_relaxed_spec_for_local_stub=args.allow_relaxed_spec_for_local_stub,
            )
            status_value = "BOOTSTRAP_ARTIFACTS_INSTALLED_AND_RECHECKED"
        else:
            field = "local_relative_path" if args.source_root is not None else "stage_relative_path"
            verified = verify_artifacts(
                manifest,
                root=args.source_root or args.staging_root,
                location_field=field,
                phase=args.phase,
                allow_relaxed_spec_for_local_stub=args.allow_relaxed_spec_for_local_stub,
            )
            status_value = "BOOTSTRAP_ARTIFACTS_HASH_SIZE_NLINK_PATH_VERIFIED"
        print(
            json.dumps(
                {
                    "format": FORMAT,
                    "status": status_value,
                    "manifest_sha256": manifest_sha,
                    "phase": args.phase,
                    "artifact_count": len(verified),
                    "artifacts": verified,
                    "gpu_start_permitted": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (BootstrapManifestError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 22


if __name__ == "__main__":
    raise SystemExit(main())
