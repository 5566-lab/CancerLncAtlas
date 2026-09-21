#!/usr/bin/env python3
"""Independently verify/extract the group-shared oracle-only code bundle.

This is a CPU-bootstrap transport verifier, not one of the six paid-gate
controls.  It never imports or executes archive content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Sequence


FORMAT = "CC_HHGT_V3_2_GROUP_SHARED_ORACLE_CODE_BUNDLE_MANIFEST_V1"
STATUS = "ORACLE_CODE_BUNDLE_FROZEN"
NAMESPACE = "v32_group_shared_oracle_paid_gpu_20260901_r2"
ALLOWED_PREFIXES = frozenset({"cc_hhgt", "config", "scripts", "docs"})
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_FILES = 10_000
MAX_TOTAL_BYTES = 256 * 1024 * 1024
_IS_WINDOWS = os.name == "nt"


class OracleOverlayError(RuntimeError):
    pass


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _identity(value: os.stat_result, ctime: bool = True) -> tuple[int, ...]:
    base = (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    return base + ((value.st_ctime_ns,) if ctime else ())


def _assert_components(path: Path, label: str) -> None:
    target = _absolute(path)
    for component in reversed([target, *target.parents]):
        try:
            observed = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(observed.st_mode):
            raise OracleOverlayError(f"{label}_SYMLINK_COMPONENT={component}")


def _open(path: Path, label: str) -> tuple[int, os.stat_result, Path]:
    target = _absolute(path)
    _assert_components(target, label)
    fd = os.open(
        target,
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    observed = os.fstat(fd)
    if not stat.S_ISREG(observed.st_mode) or observed.st_size <= 0 or observed.st_nlink != 1:
        os.close(fd)
        raise OracleOverlayError(f"{label}_FILE_TYPE_OR_NLINK_INVALID={target}")
    return fd, observed, target


def _finish(fd: int, before: os.stat_result, path: Path, label: str) -> None:
    try:
        after = os.fstat(fd)
        if _identity(before) != _identity(after):
            raise OracleOverlayError(f"{label}_CHANGED_DURING_USE={path}")
        rebound = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(rebound.st_mode) or _identity(rebound, False) != _identity(after, False):
            raise OracleOverlayError(f"{label}_PATH_REPLACED={path}")
    finally:
        os.close(fd)


def _hash_fd(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str, int]:
    fd, before, target = _open(path, label)
    try:
        digest = _hash_fd(fd)
        raw = bytearray()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            raw.extend(chunk)
        try:
            payload = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OracleOverlayError(f"{label}_JSON_INVALID") from exc
        if not isinstance(payload, Mapping):
            raise OracleOverlayError(f"{label}_MAPPING_REQUIRED")
        return dict(payload), digest, int(before.st_size)
    finally:
        _finish(fd, before, target, label)


def _relative(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise OracleOverlayError(f"ORACLE_BUNDLE_PATH_INVALID={value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise OracleOverlayError(f"ORACLE_BUNDLE_PATH_UNSAFE={value}")
    if path.parts[0] not in ALLOWED_PREFIXES:
        raise OracleOverlayError(f"ORACLE_BUNDLE_PREFIX_FORBIDDEN={value}")
    return path


def _manifest(path: Path, expected_sha: str, expected_archive_sha: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not SHA_RE.fullmatch(expected_sha) or not SHA_RE.fullmatch(expected_archive_sha):
        raise OracleOverlayError("ORACLE_BUNDLE_EXPECTED_SHA_INVALID")
    payload, digest, _ = _read_json(path, "ORACLE_BUNDLE_MANIFEST")
    if digest != expected_sha:
        raise OracleOverlayError(f"ORACLE_BUNDLE_MANIFEST_SHA_DRIFT={digest}")
    expected = {
        "format": FORMAT,
        "status": STATUS,
        "namespace": NAMESPACE,
        "archive_sha256": expected_archive_sha,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise OracleOverlayError(f"ORACLE_BUNDLE_MANIFEST_CONTRACT_DRIFT={key}")
    rows = payload.get("files")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_FILES:
        raise OracleOverlayError("ORACLE_BUNDLE_FILE_COUNT_INVALID")
    by_path: dict[str, dict[str, Any]] = {}
    total = 0
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != {"relative_path", "sha256", "size_bytes", "mode"}:
            raise OracleOverlayError("ORACLE_BUNDLE_FILE_SCHEMA_DRIFT")
        row = dict(raw)
        relative = _relative(row["relative_path"]).as_posix()
        digest_value = row["sha256"]
        size = row["size_bytes"]
        mode = row["mode"]
        if relative in by_path:
            raise OracleOverlayError(f"ORACLE_BUNDLE_FILE_DUPLICATE={relative}")
        if not isinstance(digest_value, str) or not SHA_RE.fullmatch(digest_value):
            raise OracleOverlayError(f"ORACLE_BUNDLE_FILE_SHA_INVALID={relative}")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise OracleOverlayError(f"ORACLE_BUNDLE_FILE_SIZE_INVALID={relative}")
        if mode not in {0o600, 0o700}:
            raise OracleOverlayError(f"ORACLE_BUNDLE_FILE_MODE_INVALID={relative}")
        total += size
        if total > MAX_TOTAL_BYTES:
            raise OracleOverlayError("ORACLE_BUNDLE_TOTAL_BYTES_EXCEEDS_CAP")
        by_path[relative] = row
    if payload.get("file_count") != len(by_path) or payload.get("total_bytes") != total:
        raise OracleOverlayError("ORACLE_BUNDLE_MANIFEST_TOTAL_DRIFT")
    required = {
        "cc_hhgt/v32/cli.py",
        "cc_hhgt/v32/training_guard.py",
        "cc_hhgt/v32/gpu_backward_probe.py",
        "cc_hhgt/v32/training.py",
        "cc_hhgt/v32/group_shared_encoder_oracle.py",
        "config/model_v3_2_group_shared_oracle_paid_gpu_20260901_r2.yaml",
        "config/v32_group_shared_oracle_paid_gpu_20260901_r2.TASK_MANIFEST.tsv",
        "scripts/v32_pipeline.py",
        "docs/v32_group_shared_encoder_estimator_preregistration_20260901.md",
        "docs/v32_group_shared_encoder_oracle_decision_20260901_r2.json",
    }
    if not required.issubset(by_path):
        raise OracleOverlayError(f"ORACLE_BUNDLE_REQUIRED_FILES_MISSING={sorted(required - set(by_path))}")
    return payload, by_path


def _copy_member_mode_for_platform(descriptor: int, mode: int) -> None:
    """Bind a manifest mode through the open fd where POSIX supports it."""

    if not _IS_WINDOWS:
        os.fchmod(descriptor, mode)


def _copy_member(source: BinaryIO, target: Path, row: Mapping[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _assert_components(target.parent, "ORACLE_BUNDLE_TARGET_PARENT")
    fd = os.open(
        target,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        int(row["mode"]),
    )
    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            offset = 0
            while offset < len(chunk):
                offset += os.write(fd, chunk[offset:])
        # Windows has no os.fchmod. Exact hash/size auditing still binds the
        # create-exclusive output bytes; POSIX additionally binds its mode to
        # the same descriptor before close.
        _copy_member_mode_for_platform(fd, int(row["mode"]))
        os.fsync(fd)
    finally:
        os.close(fd)
    if digest.hexdigest() != row["sha256"] or size != row["size_bytes"]:
        raise OracleOverlayError(f"ORACLE_BUNDLE_MEMBER_BYTES_DRIFT={row['relative_path']}")


def extract(archive: Path, manifest_path: Path, staging: Path, expected_archive: str, expected_manifest: str) -> dict[str, Any]:
    _, rows = _manifest(manifest_path, expected_manifest, expected_archive)
    staging = _absolute(staging)
    _assert_components(staging.parent, "ORACLE_BUNDLE_STAGING_PARENT")
    if staging.exists() or staging.is_symlink():
        raise OracleOverlayError(f"ORACLE_BUNDLE_STAGING_EXISTS={staging}")
    archive_fd, before, archive_path = _open(archive, "ORACLE_BUNDLE_ARCHIVE")
    created = False
    try:
        observed_archive = _hash_fd(archive_fd)
        if observed_archive != expected_archive:
            raise OracleOverlayError(f"ORACLE_BUNDLE_ARCHIVE_SHA_DRIFT={observed_archive}")
        staging.mkdir(mode=0o700)
        created = True
        found: set[str] = set()
        with os.fdopen(os.dup(archive_fd), "rb", closefd=True) as stream:
            with tarfile.open(fileobj=stream, mode="r:gz") as bundle:
                for member in bundle:
                    relative = _relative(member.name).as_posix()
                    if member.isdir():
                        continue
                    if not member.isfile() or member.issym() or member.islnk():
                        raise OracleOverlayError(f"ORACLE_BUNDLE_MEMBER_TYPE_FORBIDDEN={relative}")
                    if relative in found or relative not in rows:
                        raise OracleOverlayError(f"ORACLE_BUNDLE_MEMBER_SET_DRIFT={relative}")
                    row = rows[relative]
                    if member.size != row["size_bytes"]:
                        raise OracleOverlayError(f"ORACLE_BUNDLE_MEMBER_SIZE_DRIFT={relative}")
                    source = bundle.extractfile(member)
                    if source is None:
                        raise OracleOverlayError(f"ORACLE_BUNDLE_MEMBER_READ_FAILED={relative}")
                    target = staging.joinpath(*PurePosixPath(relative).parts)
                    _copy_member(source, target, row)
                    found.add(relative)
        if found != set(rows):
            raise OracleOverlayError(f"ORACLE_BUNDLE_MEMBERS_MISSING={sorted(set(rows) - found)}")
    except BaseException:
        # Deliberately retain a newly created invalid staging directory for
        # transaction forensics.  Each transaction uses a unique path.
        if created:
            try:
                (staging / ".INVALID_RETAINED_FOR_FORENSICS").touch(exist_ok=True)
            except OSError:
                pass
        raise
    finally:
        _finish(archive_fd, before, archive_path, "ORACLE_BUNDLE_ARCHIVE")
    return audit(manifest_path, staging, expected_archive, expected_manifest)


def audit(manifest_path: Path, root: Path, expected_archive: str, expected_manifest: str) -> dict[str, Any]:
    _, rows = _manifest(manifest_path, expected_manifest, expected_archive)
    root = _absolute(root)
    _assert_components(root, "ORACLE_BUNDLE_INSTALLED_ROOT")
    if not root.is_dir():
        raise OracleOverlayError(f"ORACLE_BUNDLE_INSTALLED_ROOT_INVALID={root}")
    observed: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise OracleOverlayError(f"ORACLE_BUNDLE_INSTALLED_SYMLINK={relative}")
        if candidate.is_dir():
            continue
        if relative not in rows:
            raise OracleOverlayError(f"ORACLE_BUNDLE_INSTALLED_EXTRA_FILE={relative}")
        fd, before, path = _open(candidate, "ORACLE_BUNDLE_INSTALLED_FILE")
        try:
            digest = _hash_fd(fd)
            mode = stat.S_IMODE(before.st_mode)
            row = rows[relative]
            # Windows reports synthesized POSIX modes (normally 0666) even
            # after fchmod.  Byte/hash/size checks remain exact locally; the
            # deployed Linux verifier continues to require the frozen mode.
            mode_matches = os.name == "nt" or mode == row["mode"]
            if digest != row["sha256"] or before.st_size != row["size_bytes"] or not mode_matches:
                raise OracleOverlayError(f"ORACLE_BUNDLE_INSTALLED_FILE_DRIFT={relative}")
        finally:
            _finish(fd, before, path, "ORACLE_BUNDLE_INSTALLED_FILE")
        observed.add(relative)
    if observed != set(rows):
        raise OracleOverlayError(f"ORACLE_BUNDLE_INSTALLED_FILES_MISSING={sorted(set(rows) - observed)}")
    tree = hashlib.sha256()
    for relative in sorted(rows):
        tree.update(relative.encode("utf-8") + b"\0" + rows[relative]["sha256"].encode("ascii") + b"\n")
    return {
        "format": FORMAT,
        "status": "ORACLE_CODE_BUNDLE_INSTALLED_VERIFIED",
        "namespace": NAMESPACE,
        "code_root": str(root),
        "file_count": len(rows),
        "code_tree_sha256": tree.hexdigest(),
        "archive_sha256": expected_archive,
        "manifest_sha256": expected_manifest,
        "overlay_code_imported": False,
        "overlay_code_executed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--archive", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--staging-root", type=Path)
    group.add_argument("--installed-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.staging_root is not None:
            if args.archive is None:
                raise OracleOverlayError("ORACLE_BUNDLE_ARCHIVE_REQUIRED")
            result = extract(
                args.archive,
                args.manifest,
                args.staging_root,
                args.expected_archive_sha256,
                args.expected_manifest_sha256,
            )
        else:
            result = audit(
                args.manifest,
                args.installed_root,
                args.expected_archive_sha256,
                args.expected_manifest_sha256,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OracleOverlayError, OSError, tarfile.TarError) as exc:
        print(str(exc), file=sys.stderr)
        return 22


if __name__ == "__main__":
    raise SystemExit(main())
