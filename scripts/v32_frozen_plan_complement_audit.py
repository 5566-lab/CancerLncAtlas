#!/usr/bin/env python3
"""Audit only the frozen complement of two immutable V3.2 transfer plans.

The full plan describes every required website payload.  The remaining plan is
the exact subset that still needs transfer.  This helper validates both plans,
proves their overlap is expectation-identical, and hashes only
``full - remaining``.  It never opens a remaining-plan target for hashing.

The audit is published with link(2), so an existing result is never replaced
and readers can never observe a partially written final JSON file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PLAN_FORMAT = "CANCERLNCATLAS_V32_REMOTE_TRANSFER_PLAN_V1"
AUDIT_FORMAT = "CANCERLNCATLAS_V32_FROZEN_PLAN_COMPLEMENT_AUDIT_V1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_BYTES = 4 * 1024 * 1024


class AuditSafetyError(RuntimeError):
    """A fail-closed plan, path, or no-overwrite invariant was violated."""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _absolute_normal_path(raw: str, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise AuditSafetyError(f"{label} must be a non-empty path")
    path = Path(raw)
    if not path.is_absolute():
        raise AuditSafetyError(f"{label} must be absolute: {raw}")
    if ".." in path.parts:
        raise AuditSafetyError(f"{label} contains '..': {raw}")
    normalized = Path(os.path.normpath(raw))
    if str(normalized) != raw:
        raise AuditSafetyError(f"{label} is not lexically normalized: {raw}")
    return normalized


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _require_safe_existing_directory(path: Path, *, label: str) -> None:
    current = _lstat_or_none(path)
    if current is None:
        raise AuditSafetyError(f"{label} does not exist: {path}")
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
        raise AuditSafetyError(f"{label} is not a non-symlink directory: {path}")


def _safe_under_allowed(path: Path, allowed_roots: Iterable[Path], *, label: str) -> None:
    if not any(_inside(path, root) for root in allowed_roots):
        raise AuditSafetyError(f"{label} is outside every allowed root: {path}")


def _target_path(raw: str, *, target_root: Path) -> Path:
    target = _absolute_normal_path(raw, label="target_path")
    if not _inside(target, target_root):
        raise AuditSafetyError(f"Target escapes target_root: {target}")
    resolved = target.resolve(strict=False)
    resolved_root = target_root.resolve(strict=False)
    if not _inside(resolved, resolved_root):
        raise AuditSafetyError(f"Resolved target escapes target_root: {target}")
    return target


def _open_regular_no_follow(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AuditSafetyError(f"Cannot safely open regular file {path}: {exc}") from exc
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode):
            raise AuditSafetyError(f"Expected a regular file: {path}")
        return descriptor, current
    except Exception:
        os.close(descriptor)
        raise


def _read_regular_bytes(path: Path) -> bytes:
    direct = _lstat_or_none(path)
    if direct is None or stat.S_ISLNK(direct.st_mode) or not stat.S_ISREG(direct.st_mode):
        raise AuditSafetyError(f"Expected a regular non-symlink file: {path}")
    descriptor, before = _open_regular_no_follow(path)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except Exception:
        # fdopen owns the descriptor after it succeeds.  If it did not succeed,
        # close here; an already-closed descriptor is harmlessly ignored.
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise AuditSafetyError(f"File changed while reading: {path}")
    if len(payload) != before.st_size:
        raise AuditSafetyError(f"Short read while reading: {path}")
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_expected_sha(value: str, *, label: str) -> str:
    normalized = str(value).lower()
    if not _SHA256.fullmatch(normalized):
        raise AuditSafetyError(f"{label} is not a SHA-256 digest")
    return normalized


def _load_plan(
    path: Path,
    *,
    expected_sha256: str,
    expected_count: int,
    expected_bytes: int,
    target_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    encoded = _read_regular_bytes(path)
    observed_sha256 = _sha256_bytes(encoded)
    if observed_sha256 != expected_sha256:
        raise AuditSafetyError(
            f"Plan SHA-256 mismatch for {path}: {observed_sha256} != {expected_sha256}"
        )
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditSafetyError(f"Plan is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != PLAN_FORMAT:
        raise AuditSafetyError(f"Transfer plan format mismatch: {path}")
    declared_root = _absolute_normal_path(
        str(payload.get("target_root", "")), label="plan target_root"
    )
    if declared_root != target_root:
        raise AuditSafetyError(f"Transfer plan target_root mismatch: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise AuditSafetyError(f"Transfer plan entries are not a list: {path}")
    declared_count = payload.get("entry_count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
        raise AuditSafetyError(f"Transfer plan entry_count is not an integer: {path}")
    declared_bytes = payload.get("total_bytes")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int):
        raise AuditSafetyError(f"Transfer plan total_bytes is not an integer: {path}")

    indexed: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for offset, record in enumerate(entries):
        if not isinstance(record, dict):
            raise AuditSafetyError(f"Plan entry {offset} is not an object: {path}")
        target = _target_path(str(record.get("target_path", "")), target_root=target_root)
        key = str(target)
        if key in indexed:
            raise AuditSafetyError(f"Duplicate target in plan: {target}")
        value_bytes = record.get("bytes")
        if isinstance(value_bytes, bool) or not isinstance(value_bytes, int) or value_bytes < 0:
            raise AuditSafetyError(f"Invalid bytes for target: {target}")
        value_sha256 = _validate_expected_sha(
            str(record.get("sha256", "")), label=f"sha256 for {target}"
        )
        indexed[key] = {
            "target_path": key,
            "bytes": value_bytes,
            "sha256": value_sha256,
        }
        total_bytes += value_bytes

    actual_count = len(indexed)
    if declared_count != actual_count or declared_count != expected_count:
        raise AuditSafetyError(
            f"Plan count mismatch for {path}: declared={declared_count}, "
            f"actual={actual_count}, expected={expected_count}"
        )
    if declared_bytes != total_bytes or declared_bytes != expected_bytes:
        raise AuditSafetyError(
            f"Plan bytes mismatch for {path}: declared={declared_bytes}, "
            f"actual={total_bytes}, expected={expected_bytes}"
        )
    return indexed, {
        "path": str(path),
        "sha256": observed_sha256,
        "file_bytes": len(encoded),
        "entry_count": actual_count,
        "total_bytes": total_bytes,
    }


def _symlink_component(path: Path, *, target_root: Path) -> Path | None:
    root_status = _lstat_or_none(target_root)
    if root_status is None or stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode):
        return target_root
    current = target_root
    for part in path.relative_to(target_root).parts:
        current = current / part
        status_value = _lstat_or_none(current)
        if status_value is None:
            return None
        if stat.S_ISLNK(status_value.st_mode):
            return current
    return None


def _hash_target(record: dict[str, Any], *, target_root: Path) -> dict[str, Any]:
    path = Path(record["target_path"])
    link = _symlink_component(path, target_root=target_root)
    if link is not None:
        return {
            "target_path": str(path),
            "status": "symlink_or_unsafe_component",
            "unsafe_component": str(link),
        }
    direct = _lstat_or_none(path)
    if direct is None:
        return {"target_path": str(path), "status": "missing"}
    if stat.S_ISLNK(direct.st_mode):
        return {"target_path": str(path), "status": "symlink_target"}
    if not stat.S_ISREG(direct.st_mode):
        return {
            "target_path": str(path),
            "status": "non_regular",
            "observed_mode": stat.filemode(direct.st_mode),
        }
    try:
        descriptor, before = _open_regular_no_follow(path)
    except AuditSafetyError as exc:
        return {"target_path": str(path), "status": "unsafe_open", "detail": str(exc)}
    digest = hashlib.sha256()
    observed_bytes = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            while True:
                block = stream.read(_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
                observed_bytes += len(block)
            after = os.fstat(stream.fileno())
    except Exception as exc:  # convert all per-target I/O failures into audit evidence
        try:
            os.close(descriptor)
        except OSError:
            pass
        return {"target_path": str(path), "status": "read_error", "detail": str(exc)}
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or observed_bytes != after.st_size:
        return {
            "target_path": str(path),
            "status": "changed_during_hash",
            "observed_bytes": observed_bytes,
        }
    observed_sha256 = digest.hexdigest()
    failures: list[str] = []
    if observed_bytes != record["bytes"]:
        failures.append("size_mismatch")
    if observed_sha256 != record["sha256"]:
        failures.append("sha256_mismatch")
    if failures:
        return {
            "target_path": str(path),
            "status": "+".join(failures),
            "expected_bytes": record["bytes"],
            "observed_bytes": observed_bytes,
            "expected_sha256": record["sha256"],
            "observed_sha256": observed_sha256,
        }
    return {
        "target_path": str(path),
        "status": "verified",
        "bytes": observed_bytes,
        "sha256": observed_sha256,
    }


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_no_replace_write(path: Path, payload: bytes) -> str:
    if _lstat_or_none(path) is not None:
        raise AuditSafetyError(f"Audit output already exists; refusing overwrite: {path}")
    _require_safe_existing_directory(path.parent, label="audit output parent")
    digest = _sha256_bytes(payload)
    partial = path.with_name(f"{path.name}.partial.{digest}.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(partial, flags, 0o600)
    published = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(partial, 0o444)
        os.link(partial, path, follow_symlinks=False)
        published = True
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise AuditSafetyError(f"Audit output appeared concurrently: {path}") from exc
    finally:
        if _lstat_or_none(partial) is not None:
            try:
                partial.unlink()
            except OSError:
                if not published:
                    raise
        if published:
            _fsync_directory(path.parent)
    observed = _sha256_bytes(_read_regular_bytes(path))
    if observed != digest:
        raise AuditSafetyError(f"Published audit failed self-verification: {path}")
    return digest


def _relationship_conflicts(
    full: dict[str, dict[str, Any]],
    remaining: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for target in sorted(set(remaining) - set(full)):
        conflicts.append({"type": "remaining_target_absent_from_full", "target_path": target})
    for target in sorted(set(full) & set(remaining)):
        full_expectation = full[target]
        remaining_expectation = remaining[target]
        if (
            full_expectation["bytes"],
            full_expectation["sha256"],
        ) != (
            remaining_expectation["bytes"],
            remaining_expectation["sha256"],
        ):
            conflicts.append(
                {
                    "type": "overlap_expectation_mismatch",
                    "target_path": target,
                    "full": {
                        "bytes": full_expectation["bytes"],
                        "sha256": full_expectation["sha256"],
                    },
                    "remaining": {
                        "bytes": remaining_expectation["bytes"],
                        "sha256": remaining_expectation["sha256"],
                    },
                }
            )
    return conflicts


def run_audit(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    started = time.monotonic()
    allowed_roots = tuple(
        _absolute_normal_path(item, label="allowed_root").resolve(strict=False)
        for item in args.allowed_root
    )
    if not allowed_roots:
        raise AuditSafetyError("At least one --allowed-root is required")
    for root in allowed_roots:
        _require_safe_existing_directory(root, label="allowed_root")
    target_root = _absolute_normal_path(args.target_root, label="target_root").resolve(
        strict=False
    )
    _safe_under_allowed(target_root, allowed_roots, label="target_root")
    _require_safe_existing_directory(target_root, label="target_root")
    output = _absolute_normal_path(args.output, label="output").resolve(strict=False)
    _safe_under_allowed(output, allowed_roots, label="output")

    full_sha256 = _validate_expected_sha(args.full_plan_sha256, label="full plan SHA-256")
    remaining_sha256 = _validate_expected_sha(
        args.remaining_plan_sha256, label="remaining plan SHA-256"
    )
    full, full_meta = _load_plan(
        Path(args.full_plan),
        expected_sha256=full_sha256,
        expected_count=args.expected_full_count,
        expected_bytes=args.expected_full_bytes,
        target_root=target_root,
    )
    remaining, remaining_meta = _load_plan(
        Path(args.remaining_plan),
        expected_sha256=remaining_sha256,
        expected_count=args.expected_remaining_count,
        expected_bytes=args.expected_remaining_bytes,
        target_root=target_root,
    )

    relationship = _relationship_conflicts(full, remaining)
    complement_keys = sorted(set(full) - set(remaining))
    complement_bytes = sum(full[target]["bytes"] for target in complement_keys)
    if len(complement_keys) != args.expected_complement_count:
        relationship.append(
            {
                "type": "complement_count_mismatch",
                "expected": args.expected_complement_count,
                "observed": len(complement_keys),
            }
        )
    if complement_bytes != args.expected_complement_bytes:
        relationship.append(
            {
                "type": "complement_bytes_mismatch",
                "expected": args.expected_complement_bytes,
                "observed": complement_bytes,
            }
        )

    target_results: list[dict[str, Any]] = []
    if not relationship:
        for target in complement_keys:
            target_results.append(_hash_target(full[target], target_root=target_root))
    target_conflicts = [row for row in target_results if row["status"] != "verified"]
    verified = [row for row in target_results if row["status"] == "verified"]
    verified_bytes = sum(int(row["bytes"]) for row in verified)
    audit_status = "PASS" if not relationship and not target_conflicts else "FAIL"
    payload: dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": audit_status,
        "target_root": str(target_root),
        "plans": {"full": full_meta, "remaining": remaining_meta},
        "plan_relationship": {
            "full_count": len(full),
            "remaining_count": len(remaining),
            "overlap_count": len(set(full) & set(remaining)),
            "overlap_expectations_identical": not any(
                row["type"] == "overlap_expectation_mismatch" for row in relationship
            ),
            "remaining_is_subset_of_full": not any(
                row["type"] == "remaining_target_absent_from_full" for row in relationship
            ),
            "complement_count": len(complement_keys),
            "complement_bytes": complement_bytes,
        },
        "hash_scope": {
            "policy": "full_plan_minus_remaining_plan_only",
            "remaining_targets_skipped_count": len(remaining),
            "remaining_targets_rehashed_count": 0,
            "complement_targets_considered_count": len(complement_keys),
            "complement_targets_hashed_count": len(target_results),
            "complement_bytes_hashed": sum(
                int(row.get("bytes", row.get("observed_bytes", 0))) for row in target_results
            ),
            "verified_count": len(verified),
            "verified_bytes": verified_bytes,
        },
        "conflict_counts": {
            "plan_relationship": len(relationship),
            "targets": len(target_conflicts),
            "total": len(relationship) + len(target_conflicts),
        },
        "conflicts": {
            "plan_relationship": relationship,
            "targets": target_conflicts,
        },
        "safety": {
            "regular_non_symlink_required": True,
            "path_escape_forbidden": True,
            "audit_write": "atomic_link_no_replace",
            "remaining_targets_opened_for_hashing": False,
            "production_port_8260_touched": False,
            "services_started_or_restarted": False,
        },
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    encoded = _canonical_json_bytes(payload)
    output_sha256 = _atomic_no_replace_write(output, encoded)
    return payload, output_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-plan", required=True)
    parser.add_argument("--full-plan-sha256", required=True)
    parser.add_argument("--remaining-plan", required=True)
    parser.add_argument("--remaining-plan-sha256", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--allowed-root", action="append", required=True)
    parser.add_argument("--expected-full-count", required=True, type=int)
    parser.add_argument("--expected-full-bytes", required=True, type=int)
    parser.add_argument("--expected-remaining-count", required=True, type=int)
    parser.add_argument("--expected-remaining-bytes", required=True, type=int)
    parser.add_argument("--expected-complement-count", required=True, type=int)
    parser.add_argument("--expected-complement-bytes", required=True, type=int)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload, output_sha256 = run_audit(args)
    except (AuditSafetyError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    receipt = {
        "status": payload["status"],
        "output": args.output,
        "output_sha256": output_sha256,
        "complement_count": payload["plan_relationship"]["complement_count"],
        "complement_bytes": payload["plan_relationship"]["complement_bytes"],
        "verified_count": payload["hash_scope"]["verified_count"],
        "verified_bytes": payload["hash_scope"]["verified_bytes"],
        "remaining_targets_rehashed_count": 0,
        "conflicts": payload["conflict_counts"]["total"],
    }
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
