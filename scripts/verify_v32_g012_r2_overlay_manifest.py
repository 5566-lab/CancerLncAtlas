#!/usr/bin/env python3
"""Independently verify and extract the fixed r2 code archive.

This verifier must be installed in the r2 bootstrap directory separately from
the archive it verifies.  It never imports or executes extracted content.  The
manifest and gzip archive are opened with O_NOFOLLOW where available, hashed
and consumed through the same file descriptors, and rejected when hardlinked,
replaced, truncated, or changed during use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Sequence


FORMAT = "CC_HHGT_V3_2_G012_R2_CODE_ARCHIVE_MANIFEST_V1"
NAMESPACE = "v32_g012_paid_gpu_20260901_r2"
ALLOWED_PREFIXES = frozenset({"cc_hhgt", "config", "scripts", "tests", "docs"})
MAX_FILE_COUNT = 10_000
MAX_TOTAL_BYTES = 256 * 1024 * 1024
DEPLOYMENT_ARTIFACTS = (
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


class OverlayVerificationError(RuntimeError):
    """The external manifest/archive pair is unsafe, stale, or inconsistent."""


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path, label: str) -> None:
    target = _absolute_lexical(path)
    for component in reversed([target, *target.parents]):
        try:
            observed = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise OverlayVerificationError(
                f"{label}_COMPONENT_LSTAT_FAILED={component}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode) or getattr(
            os.path, "isjunction", lambda _: False
        )(component):
            raise OverlayVerificationError(
                f"{label}_SYMLINK_COMPONENT_FORBIDDEN={component}"
            )


def _identity(value: os.stat_result, *, include_ctime: bool = True) -> tuple[int, ...]:
    base = (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    return base + ((value.st_ctime_ns,) if include_ctime else ())


def _open_bound_regular(
    path: Path, label: str
) -> tuple[int, os.stat_result, list[int], Path]:
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
                    raise OverlayVerificationError(
                        f"{label}_PARENT_NOT_DIRECTORY={component}"
                    )
                held_directories.append(current)
            descriptor = os.open(target.name, flags, dir_fd=current)
        else:
            descriptor = os.open(target, flags)
    except (OSError, OverlayVerificationError) as exc:
        for directory in reversed(held_directories):
            try:
                os.close(directory)
            except OSError:
                pass
        if isinstance(exc, OverlayVerificationError):
            raise
        raise OverlayVerificationError(f"{label}_OPEN_FAILED={target}") from exc
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode) or observed.st_size <= 0:
        os.close(descriptor)
        for directory in reversed(held_directories):
            os.close(directory)
        raise OverlayVerificationError(f"{label}_NOT_NONEMPTY_REGULAR={target}")
    if observed.st_nlink != 1:
        os.close(descriptor)
        for directory in reversed(held_directories):
            os.close(directory)
        raise OverlayVerificationError(f"{label}_HARDLINK_FORBIDDEN={target}")
    return descriptor, observed, held_directories, target


def _hash_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _close_bound_regular(
    descriptor: int,
    before: os.stat_result,
    held_directories: Sequence[int],
    path: Path,
    label: str,
) -> None:
    try:
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise OverlayVerificationError(f"{label}_CHANGED_DURING_USE={path}")
        try:
            bound_path = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise OverlayVerificationError(f"{label}_PATH_LOST_DURING_USE={path}") from exc
        if not stat.S_ISREG(bound_path.st_mode) or _identity(
            bound_path, include_ctime=False
        ) != _identity(after, include_ctime=False):
            raise OverlayVerificationError(f"{label}_PATH_REPLACED_DURING_USE={path}")
    finally:
        os.close(descriptor)
        for directory in reversed(held_directories):
            os.close(directory)


def _safe_relative(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise OverlayVerificationError(f"MANIFEST_PATH_INVALID={value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise OverlayVerificationError(f"MANIFEST_PATH_UNSAFE={value}")
    if path.parts[0] not in ALLOWED_PREFIXES:
        raise OverlayVerificationError(f"MANIFEST_PREFIX_FORBIDDEN={value}")
    return path


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _load_manifest_from_descriptor(
    descriptor: int, *, expected_manifest_sha256: str, expected_archive_sha256: str
) -> tuple[dict[PurePosixPath, dict[str, Any]], dict[str, Any]]:
    observed_sha = _hash_descriptor(descriptor)
    if observed_sha != expected_manifest_sha256:
        raise OverlayVerificationError(f"MANIFEST_SHA256_DRIFT={observed_sha}")
    raw = bytearray()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        raw.extend(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    try:
        payload = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OverlayVerificationError("MANIFEST_JSON_INVALID") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "format",
        "namespace",
        "archive_sha256",
        "files",
    }:
        raise OverlayVerificationError("MANIFEST_TOP_LEVEL_SCHEMA_DRIFT")
    if payload.get("format") != FORMAT or payload.get("namespace") != NAMESPACE:
        raise OverlayVerificationError("MANIFEST_IDENTITY_DRIFT")
    if payload.get("archive_sha256") != expected_archive_sha256:
        raise OverlayVerificationError("MANIFEST_ARCHIVE_SHA_BINDING_DRIFT")
    files = payload.get("files")
    if not isinstance(files, list) or not 0 < len(files) <= MAX_FILE_COUNT:
        raise OverlayVerificationError("MANIFEST_FILE_COUNT_INVALID")
    records: dict[PurePosixPath, dict[str, Any]] = {}
    casefolded: set[str] = set()
    total = 0
    for raw_record in files:
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "path",
            "sha256",
            "size_bytes",
            "mode",
        }:
            raise OverlayVerificationError("MANIFEST_FILE_SCHEMA_DRIFT")
        relative = _safe_relative(raw_record.get("path"))
        folded = relative.as_posix().casefold()
        if relative in records or folded in casefolded:
            raise OverlayVerificationError(f"MANIFEST_FILE_DUPLICATE={relative}")
        if not _is_sha(raw_record.get("sha256")):
            raise OverlayVerificationError(f"MANIFEST_FILE_SHA_INVALID={relative}")
        size = raw_record.get("size_bytes")
        mode = raw_record.get("mode")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise OverlayVerificationError(f"MANIFEST_FILE_SIZE_INVALID={relative}")
        if mode not in (0o644, 0o755):
            raise OverlayVerificationError(f"MANIFEST_FILE_MODE_INVALID={relative}")
        total += size
        if total > MAX_TOTAL_BYTES:
            raise OverlayVerificationError("MANIFEST_TOTAL_BYTES_EXCEEDS_CAP")
        records[relative] = dict(raw_record)
        casefolded.add(folded)
    return records, dict(payload)


def _contract_hashes(
    records: Mapping[PurePosixPath, Mapping[str, Any]],
) -> tuple[str, str]:
    code_records = [
        (relative.as_posix(), record)
        for relative, record in records.items()
        if (
            relative.parts[0] == "cc_hhgt"
            or relative.as_posix() == "scripts/v32_pipeline.py"
        )
        and not any(part in {"__pycache__", ".pytest_cache", ".git"} for part in relative.parts)
        and relative.suffix.lower() not in {".pyc", ".pyo"}
    ]
    if not code_records:
        raise OverlayVerificationError("MANIFEST_CODE_TREE_EMPTY")
    code_digest = hashlib.sha256()
    for relative, record in sorted(code_records):
        code_digest.update(relative.encode("utf-8"))
        code_digest.update(b"\0")
        code_digest.update(str(record["sha256"]).encode("ascii"))
        code_digest.update(b"\n")
    deployment_records: list[dict[str, Any]] = []
    for relative in DEPLOYMENT_ARTIFACTS:
        record = records.get(PurePosixPath(relative))
        if record is None:
            raise OverlayVerificationError(
                f"MANIFEST_DEPLOYMENT_ARTIFACT_MISSING={relative}"
            )
        deployment_records.append(
            {
                "relative_path": relative,
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
        )
    deployment_raw = json.dumps(
        deployment_records, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return code_digest.hexdigest(), hashlib.sha256(deployment_raw).hexdigest()


def _copy_member(
    source: BinaryIO, target: Path, *, expected_size: int, expected_sha256: str, mode: int
) -> None:
    digest = hashlib.sha256()
    observed = 0
    with target.open("xb") as output:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > expected_size:
                raise OverlayVerificationError(f"ARCHIVE_MEMBER_SIZE_OVERFLOW={target}")
            digest.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    if observed != expected_size or digest.hexdigest() != expected_sha256:
        raise OverlayVerificationError(f"ARCHIVE_MEMBER_CONTENT_DRIFT={target}")
    os.chmod(target, mode)


def verify_and_extract(
    *,
    archive_path: Path,
    manifest_path: Path,
    staging_root: Path,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    if not _is_sha(expected_archive_sha256) or not _is_sha(expected_manifest_sha256):
        raise OverlayVerificationError("EXPECTED_ARCHIVE_OR_MANIFEST_SHA_INVALID")
    staging = _absolute_lexical(staging_root)
    _assert_no_symlink_components(staging.parent, "STAGING_PARENT")
    try:
        os.lstat(staging)
    except FileNotFoundError:
        pass
    else:
        raise OverlayVerificationError(f"STAGING_ROOT_ALREADY_EXISTS={staging}")

    manifest_fd, manifest_before, manifest_dirs, manifest_target = _open_bound_regular(
        manifest_path, "MANIFEST"
    )
    try:
        records, _ = _load_manifest_from_descriptor(
            manifest_fd,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_archive_sha256=expected_archive_sha256,
        )
    finally:
        _close_bound_regular(
            manifest_fd, manifest_before, manifest_dirs, manifest_target, "MANIFEST"
        )

    archive_fd, archive_before, archive_dirs, archive_target = _open_bound_regular(
        archive_path, "ARCHIVE"
    )
    staging_created = False
    try:
        staging.mkdir(parents=False, exist_ok=False)
        staging_created = True
        observed_files: set[PurePosixPath] = set()
        observed_dirs: set[PurePosixPath] = set()
        try:
            observed_archive_sha = _hash_descriptor(archive_fd)
            if observed_archive_sha != expected_archive_sha256:
                raise OverlayVerificationError(
                    f"ARCHIVE_SHA256_DRIFT={observed_archive_sha}"
                )
            with os.fdopen(os.dup(archive_fd), "rb", closefd=True) as archive_stream:
                with tarfile.open(fileobj=archive_stream, mode="r:gz") as archive:
                    for member in archive:
                        relative = _safe_relative(member.name.rstrip("/"))
                        if member.isdir():
                            if relative in observed_dirs:
                                raise OverlayVerificationError(
                                    f"ARCHIVE_DIRECTORY_DUPLICATE={relative}"
                                )
                            observed_dirs.add(relative)
                            (staging / Path(*relative.parts)).mkdir(
                                parents=True, exist_ok=True
                            )
                            continue
                        if not member.isreg():
                            raise OverlayVerificationError(
                                f"ARCHIVE_MEMBER_TYPE_FORBIDDEN={relative}"
                            )
                        if relative in observed_files:
                            raise OverlayVerificationError(
                                f"ARCHIVE_FILE_DUPLICATE={relative}"
                            )
                        record = records.get(relative)
                        if record is None:
                            raise OverlayVerificationError(
                                f"ARCHIVE_FILE_NOT_IN_MANIFEST={relative}"
                            )
                        if member.size != record["size_bytes"]:
                            raise OverlayVerificationError(
                                f"ARCHIVE_MEMBER_DECLARED_SIZE_DRIFT={relative}"
                            )
                        target = staging / Path(*relative.parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        resolved = target.resolve(strict=False)
                        if staging not in resolved.parents:
                            raise OverlayVerificationError(
                                f"ARCHIVE_MEMBER_ESCAPES_STAGING={relative}"
                            )
                        source = archive.extractfile(member)
                        if source is None:
                            raise OverlayVerificationError(
                                f"ARCHIVE_MEMBER_READ_FAILED={relative}"
                            )
                        with source:
                            _copy_member(
                                source,
                                target,
                                expected_size=record["size_bytes"],
                                expected_sha256=record["sha256"],
                                mode=record["mode"],
                            )
                        observed_files.add(relative)
            if observed_files != set(records):
                missing = sorted(
                    path.as_posix() for path in set(records) - observed_files
                )
                raise OverlayVerificationError(
                    f"ARCHIVE_MANIFEST_FILES_MISSING={missing}"
                )
        finally:
            _close_bound_regular(
                archive_fd, archive_before, archive_dirs, archive_target, "ARCHIVE"
            )
    except BaseException:
        # A create failure occurs before the nested archive-descriptor finally
        # is entered.  Close the already-bound archive in that narrow path too.
        if not staging_created:
            _close_bound_regular(
                archive_fd, archive_before, archive_dirs, archive_target, "ARCHIVE"
            )
        if staging_created:
            try:
                observed_staging = os.lstat(staging)
                if stat.S_ISDIR(observed_staging.st_mode) and not stat.S_ISLNK(
                    observed_staging.st_mode
                ):
                    shutil.rmtree(staging)
                else:
                    raise OverlayVerificationError(
                        f"STAGING_RECOVERY_ROOT_INVALID={staging}"
                    )
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                raise OverlayVerificationError(
                    f"STAGING_RECOVERY_CLEANUP_FAILED={staging}"
                ) from cleanup_error
        raise
    return {
        "format": "CC_HHGT_V3_2_G012_R2_CODE_ARCHIVE_VERIFIED_V1",
        "status": "CODE_ARCHIVE_VERIFIED_AND_EXTRACTED",
        "namespace": NAMESPACE,
        "archive_path": str(archive_target),
        "archive_sha256": expected_archive_sha256,
        "manifest_path": str(manifest_target),
        "manifest_sha256": expected_manifest_sha256,
        "staging_root": str(staging),
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records.values()),
        "overlay_code_imported": False,
        "overlay_code_executed": False,
    }


def verify_installed_tree(
    *,
    manifest_path: Path,
    installed_root: Path,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Re-hash every installed file against the external locked manifest."""

    root = _absolute_lexical(installed_root)
    _assert_no_symlink_components(root, "INSTALLED_ROOT")
    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise OverlayVerificationError(f"INSTALLED_ROOT_INVALID={root}") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise OverlayVerificationError(f"INSTALLED_ROOT_INVALID={root}")
    manifest_fd, manifest_before, manifest_dirs, manifest_target = _open_bound_regular(
        manifest_path, "MANIFEST"
    )
    try:
        records, _ = _load_manifest_from_descriptor(
            manifest_fd,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_archive_sha256=expected_archive_sha256,
        )
    finally:
        _close_bound_regular(
            manifest_fd, manifest_before, manifest_dirs, manifest_target, "MANIFEST"
        )

    observed: set[PurePosixPath] = set()
    for relative, record in records.items():
        target = root / Path(*relative.parts)
        resolved = target.resolve(strict=False)
        if root not in resolved.parents:
            raise OverlayVerificationError(f"INSTALLED_FILE_ESCAPES_ROOT={relative}")
        descriptor, before, target_dirs, target_bound = _open_bound_regular(
            target, "INSTALLED_FILE"
        )
        try:
            digest = _hash_descriptor(descriptor)
            if digest != record["sha256"] or before.st_size != record["size_bytes"]:
                raise OverlayVerificationError(f"INSTALLED_FILE_CONTENT_DRIFT={relative}")
            if os.name != "nt" and stat.S_IMODE(before.st_mode) != record["mode"]:
                raise OverlayVerificationError(f"INSTALLED_FILE_MODE_DRIFT={relative}")
        finally:
            _close_bound_regular(
                descriptor, before, target_dirs, target_bound, "INSTALLED_FILE"
            )
        observed.add(relative)
    actual: set[PurePosixPath] = set()
    for path in root.rglob("*"):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        try:
            observed_path = os.lstat(path)
        except OSError as exc:
            raise OverlayVerificationError(
                f"INSTALLED_TREE_LSTAT_FAILED={relative}"
            ) from exc
        if stat.S_ISLNK(observed_path.st_mode) or getattr(
            os.path, "isjunction", lambda _: False
        )(path):
            raise OverlayVerificationError(f"INSTALLED_SYMLINK_FORBIDDEN={relative}")
        if relative.parts[0] not in ALLOWED_PREFIXES:
            raise OverlayVerificationError(
                f"INSTALLED_PREFIX_FORBIDDEN={relative.parts[0]}"
            )
        if stat.S_ISREG(observed_path.st_mode):
            actual.add(relative)
        elif not stat.S_ISDIR(observed_path.st_mode):
            raise OverlayVerificationError(
                f"INSTALLED_SPECIAL_FILE_FORBIDDEN={relative}"
            )
    if actual != observed:
        extra = sorted(path.as_posix() for path in actual - observed)
        missing = sorted(path.as_posix() for path in observed - actual)
        raise OverlayVerificationError(
            f"INSTALLED_MANIFEST_SET_DRIFT=extra:{extra}:missing:{missing}"
        )
    code_tree_sha256, deployment_contract_sha256 = _contract_hashes(records)
    return {
        "format": "CC_HHGT_V3_2_G012_R2_INSTALLED_CODE_VERIFIED_V1",
        "status": "INSTALLED_CODE_MATCHES_EXTERNAL_MANIFEST",
        "namespace": NAMESPACE,
        "installed_root": str(root),
        "archive_sha256": expected_archive_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "file_count": len(records),
        "code_tree_sha256": code_tree_sha256,
        "deployment_contract_sha256": deployment_contract_sha256,
        "overlay_code_imported": False,
        "overlay_code_executed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--staging-root", type=Path)
    action.add_argument("--installed-root", type=Path)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.installed_root is not None:
            payload = verify_installed_tree(
                manifest_path=args.manifest,
                installed_root=args.installed_root,
                expected_archive_sha256=args.expected_archive_sha256,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
        else:
            if args.archive is None:
                raise OverlayVerificationError("ARCHIVE_REQUIRED_FOR_EXTRACTION")
            payload = verify_and_extract(
                archive_path=args.archive,
                manifest_path=args.manifest,
                staging_root=args.staging_root,
                expected_archive_sha256=args.expected_archive_sha256,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
    except (OverlayVerificationError, OSError, tarfile.TarError) as exc:
        print(str(exc), file=sys.stderr)
        return 22
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
