#!/usr/bin/env python3
"""Verify and atomically install one transaction-scoped CPU bootstrap receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


INSTANCE_ID = "uhost-1up504geeqzj"
PROJECT_ROOT = Path("./data/CancerLncAtlas")
CPU_NAMESPACE = "v32_compshare_cpu_nogpu_bootstrap_20260901_r1"
STOPPED_FORMAT = "CANCERLNCATLAS_COMPSHARE_STOPPED_RECEIPT_V1"
JIT_FORMAT = "CANCERLNCATLAS_COMPSHARE_JIT_BUDGET_V1"
PROJECT_SCOPE_FORMAT = "CANCERLNCATLAS_FIXED_INSTANCE_PLUS_200GB_BOOT_DISK_V1"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
TX_RE = re.compile(r"cpu-(?:prepare|oracle)-[0-9]{10}-[0-9a-f]{12}\Z")
STOPPED_KEYS = frozenset(
    {
        "format",
        "status",
        "instance_id",
        "observed_state",
        "provider_domain_state",
        "gpu_billing_active",
        "source",
        "checked_at",
        "valid_until",
        "provider_stop_time_unix",
        "provider_stopped_profile",
        "provider_gpu_count",
        "provider_gpu_type",
        "provider_cpu_count",
        "provider_memory_mib",
        "provider_machine_type",
        "provider_compute_hourly_cny",
        "provider_disk_hourly_cny",
        "support_without_gpu_start",
        "transaction_id",
        "project_scope_format",
        "project_scope_sha256",
        "profile_name_sha256",
        "raw_provider_response_embedded",
        "credentials_embedded",
    }
)
JIT_KEYS = frozenset(
    {
        "format",
        "status",
        "currency",
        "source",
        "project_budget_cap_cny",
        "compute_hourly_cny",
        "gpu_hourly_cny",
        "disk_hourly_cny",
        "total_hourly_cny",
        "instance_id",
        "project_scope_format",
        "project_scope_sha256",
        "profile_name_sha256",
        "queried_at",
        "valid_until",
        "conservative_remaining_cny",
        "budget_method",
        "instance_age_seconds",
        "provider_stop_time_unix",
        "provider_query_duration_seconds",
        "worst_case_full_rate_spend_cny",
        "future_cleanup_reserve_cny",
        "worst_case_spend_rounding",
        "remaining_rounding",
        "unallocated_rounding_reserve_cny",
        "provider_snapshot_sha256",
        "generator_sha256",
    }
)


class DynamicReceiptError(RuntimeError):
    pass


def _identity(value: os.stat_result, include_ctime: bool = True) -> tuple[int, ...]:
    base = (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    return base + ((value.st_ctime_ns,) if include_ctime else ())


def _assert_components(path: Path, label: str) -> None:
    target = Path(os.path.abspath(os.fspath(path)))
    for component in reversed([target, *target.parents]):
        try:
            observed = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(observed.st_mode):
            raise DynamicReceiptError(f"{label}_SYMLINK_COMPONENT={component}")


def _read(
    path: Path,
    label: str,
    *,
    allowed_nlinks: frozenset[int] = frozenset({1}),
) -> tuple[bytes, str, int]:
    target = Path(os.path.abspath(os.fspath(path)))
    _assert_components(target, label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(target, flags)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_nlink not in allowed_nlinks
        ):
            raise DynamicReceiptError(f"{label}_FILE_TYPE_OR_NLINK_INVALID={target}")
        raw = bytearray()
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            raw.extend(chunk)
            digest.update(chunk)
        after = os.fstat(fd)
        if _identity(before) != _identity(after):
            raise DynamicReceiptError(f"{label}_CHANGED_DURING_READ={target}")
        rebound = os.stat(target, follow_symlinks=False)
        if not stat.S_ISREG(rebound.st_mode) or _identity(rebound, False) != _identity(after, False):
            raise DynamicReceiptError(f"{label}_PATH_REPLACED={target}")
        return bytes(raw), digest.hexdigest(), int(before.st_size)
    finally:
        os.close(fd)


def _load(path: Path, label: str) -> tuple[dict[str, Any], bytes, str, int]:
    raw, digest, size = _read(path, label)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DynamicReceiptError(f"{label}_JSON_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise DynamicReceiptError(f"{label}_MAPPING_REQUIRED")
    return dict(payload), raw, digest, size


def _aware(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise DynamicReceiptError(f"{label}_MISSING")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DynamicReceiptError(f"{label}_INVALID") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DynamicReceiptError(f"{label}_TIMEZONE_REQUIRED")
    return parsed.astimezone(timezone.utc)


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DynamicReceiptError(f"{label}_NUMBER_REQUIRED")
    try:
        observed = Decimal(str(value))
    except InvalidOperation as exc:
        raise DynamicReceiptError(f"{label}_NUMBER_INVALID") from exc
    if not observed.is_finite():
        raise DynamicReceiptError(f"{label}_NUMBER_NONFINITE")
    return observed


def _validate_stopped(
    payload: Mapping[str, Any],
    transaction: str,
    project_scope_sha: str,
    profile_sha: str,
) -> Path:
    if set(payload) != STOPPED_KEYS:
        raise DynamicReceiptError(
            f"STOPPED_RECEIPT_KEY_SET_DRIFT="
            f"{sorted(STOPPED_KEYS - set(payload))}:{sorted(set(payload) - STOPPED_KEYS)}"
        )
    expected = {
        "format": STOPPED_FORMAT,
        "status": "INSTANCE_STOPPED_STATE_CONFIRMED",
        "instance_id": INSTANCE_ID,
        "observed_state": "Stopped",
        "provider_domain_state": "DOMAIN_SHUT_OFF",
        "gpu_billing_active": False,
        "source": "DIRECT_COMPSHARE_INSTANCE_SHOW_QUERY",
        "raw_provider_response_embedded": False,
        "credentials_embedded": False,
        "transaction_id": transaction,
        "project_scope_format": PROJECT_SCOPE_FORMAT,
        "project_scope_sha256": project_scope_sha,
        "profile_name_sha256": profile_sha,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise DynamicReceiptError(f"STOPPED_RECEIPT_CONTRACT_DRIFT={key}")
    checked = _aware(payload.get("checked_at"), "STOPPED_CHECKED_AT")
    valid_until = _aware(payload.get("valid_until"), "STOPPED_VALID_UNTIL")
    now = datetime.now(timezone.utc)
    age = (now - checked).total_seconds()
    if age < -60 or age > 15 * 60:
        raise DynamicReceiptError(f"STOPPED_RECEIPT_NOT_FRESH={age:.3f}")
    if valid_until <= now or valid_until <= checked or (valid_until - checked).total_seconds() > 15 * 60:
        raise DynamicReceiptError("STOPPED_RECEIPT_VALIDITY_INVALID")
    stop_unix = payload.get("provider_stop_time_unix")
    if isinstance(stop_unix, bool) or not isinstance(stop_unix, int) or stop_unix <= 0:
        raise DynamicReceiptError("STOPPED_PROVIDER_STOP_TIME_INVALID")
    if stop_unix > int(checked.timestamp()):
        raise DynamicReceiptError("STOPPED_PROVIDER_LIFECYCLE_ORDER_INVALID")
    stopped_profile = payload.get("provider_stopped_profile")
    exact_provider_profiles = {
        "NOMINAL_G_GPU_ATTACHED_STOPPED": {
            "provider_gpu_count": 1,
            "provider_gpu_type": "4090",
            "provider_cpu_count": 16,
            "provider_memory_mib": 65536,
            "provider_machine_type": "G",
            "provider_compute_hourly_cny": Decimal("2.05"),
        },
        "NO_GPU_A_POST_START_STOPPED": {
            "provider_gpu_count": 0,
            "provider_gpu_type": "4090",
            "provider_cpu_count": 2,
            "provider_memory_mib": 4096,
            "provider_machine_type": "O",
            "provider_compute_hourly_cny": Decimal("0.14"),
        },
    }
    if stopped_profile not in exact_provider_profiles:
        raise DynamicReceiptError("STOPPED_PROVIDER_PROFILE_UNKNOWN")
    exact_profile = exact_provider_profiles[stopped_profile]
    for key in (
        "provider_gpu_count",
        "provider_gpu_type",
        "provider_cpu_count",
        "provider_memory_mib",
        "provider_machine_type",
    ):
        if payload.get(key) != exact_profile[key]:
            raise DynamicReceiptError(f"STOPPED_PROVIDER_PROFILE_DRIFT={key}")
    if (
        _decimal(
            payload.get("provider_compute_hourly_cny"),
            "STOPPED_PROVIDER_COMPUTE_HOURLY",
        )
        != exact_profile["provider_compute_hourly_cny"]
        or _decimal(
            payload.get("provider_disk_hourly_cny"),
            "STOPPED_PROVIDER_DISK_HOURLY",
        )
        != Decimal("0.04")
        or payload.get("support_without_gpu_start") is not True
    ):
        raise DynamicReceiptError("STOPPED_PROVIDER_BILLING_OR_CAPABILITY_DRIFT")
    return (
        PROJECT_ROOT
        / "runtime/bootstrap"
        / CPU_NAMESPACE
        / "transactions"
        / transaction
        / "PROVIDER_STOPPED.json"
    )


def _validate_jit(
    payload: Mapping[str, Any],
    transaction: str,
    project_scope_sha: str,
    profile_sha: str,
    expected_generator_sha: str,
) -> Path:
    if set(payload) != JIT_KEYS:
        raise DynamicReceiptError(
            f"JIT_RECEIPT_KEY_SET_DRIFT="
            f"{sorted(JIT_KEYS - set(payload))}:{sorted(set(payload) - JIT_KEYS)}"
        )
    expected = {
        "format": JIT_FORMAT,
        "status": "JIT_BUDGET_READY_CONSERVATIVE",
        "currency": "CNY",
        "source": "COMPSHARE_API_BILLING_SNAPSHOT",
        "instance_id": INSTANCE_ID,
        "project_scope_format": PROJECT_SCOPE_FORMAT,
        "project_scope_sha256": project_scope_sha,
        "profile_name_sha256": profile_sha,
        "generator_sha256": expected_generator_sha,
        "budget_method": "FULL_2_09_CNY_RATE_FOR_EVERY_SECOND_SINCE_INSTANCE_CREATE_MINUS_FUTURE_RESERVE",
        "worst_case_spend_rounding": "CEILING_0_0001_CNY",
        "remaining_rounding": "FLOOR_0_0001_CNY",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise DynamicReceiptError(f"JIT_RECEIPT_CONTRACT_DRIFT={key}")
    queried = _aware(payload.get("queried_at"), "JIT_QUERIED_AT")
    valid_until = _aware(payload.get("valid_until"), "JIT_VALID_UNTIL")
    now = datetime.now(timezone.utc)
    if valid_until <= now or valid_until <= queried or queried > now + timedelta(seconds=60):
        raise DynamicReceiptError("JIT_RECEIPT_NOT_CURRENT")
    if (valid_until - queried).total_seconds() > 45 * 60:
        raise DynamicReceiptError("JIT_RECEIPT_VALIDITY_TOO_WIDE")
    exact_numbers = {
        "project_budget_cap_cny": Decimal("210"),
        "compute_hourly_cny": Decimal("2.05"),
        "gpu_hourly_cny": Decimal("2.05"),
        "disk_hourly_cny": Decimal("0.04"),
        "total_hourly_cny": Decimal("2.09"),
    }
    for key, expected_number in exact_numbers.items():
        if _decimal(payload.get(key), f"JIT_{key.upper()}") != expected_number:
            raise DynamicReceiptError(f"JIT_RECEIPT_NUMERIC_POLICY_DRIFT={key}")
    age = payload.get("instance_age_seconds")
    stop_unix = payload.get("provider_stop_time_unix")
    if isinstance(age, bool) or not isinstance(age, int) or age < 0:
        raise DynamicReceiptError("JIT_INSTANCE_AGE_INVALID")
    if isinstance(stop_unix, bool) or not isinstance(stop_unix, int) or stop_unix <= 0:
        raise DynamicReceiptError("JIT_PROVIDER_STOP_TIME_INVALID")
    queried_unix = int(queried.timestamp())
    inferred_create = queried_unix - age
    if inferred_create <= 0 or not inferred_create <= stop_unix <= queried_unix:
        raise DynamicReceiptError("JIT_PROVIDER_LIFECYCLE_ORDER_INVALID")
    duration = _decimal(payload.get("provider_query_duration_seconds"), "JIT_QUERY_DURATION")
    if duration < 0 or duration > Decimal("185"):
        raise DynamicReceiptError("JIT_QUERY_DURATION_INVALID")
    reserve = _decimal(payload.get("future_cleanup_reserve_cny"), "JIT_CLEANUP_RESERVE")
    if reserve < Decimal("5") or reserve >= Decimal("210"):
        raise DynamicReceiptError("JIT_CLEANUP_RESERVE_INVALID")
    quantum = Decimal("0.0001")
    spend = (Decimal("2.09") * Decimal(age) / Decimal(3600)).quantize(
        quantum, rounding=ROUND_CEILING
    )
    remaining = (Decimal("210") - spend - reserve).quantize(
        quantum, rounding=ROUND_FLOOR
    )
    unallocated = Decimal("210") - spend - reserve - remaining
    if _decimal(payload.get("worst_case_full_rate_spend_cny"), "JIT_WORST_CASE_SPEND") != spend:
        raise DynamicReceiptError("JIT_WORST_CASE_SPEND_EQUATION_DRIFT")
    if _decimal(payload.get("conservative_remaining_cny"), "JIT_REMAINING") != remaining:
        raise DynamicReceiptError("JIT_REMAINING_EQUATION_DRIFT")
    if _decimal(payload.get("unallocated_rounding_reserve_cny"), "JIT_UNALLOCATED") != unallocated:
        raise DynamicReceiptError("JIT_UNALLOCATED_EQUATION_DRIFT")
    if not Decimal("0") <= unallocated < quantum:
        raise DynamicReceiptError("JIT_UNALLOCATED_RANGE_INVALID")
    if not Decimal("0") < remaining <= Decimal("210"):
        raise DynamicReceiptError("JIT_RECEIPT_REMAINING_INVALID")
    snapshot_sha = payload.get("provider_snapshot_sha256")
    if not isinstance(snapshot_sha, str) or not SHA_RE.fullmatch(snapshot_sha):
        raise DynamicReceiptError("JIT_PROVIDER_SNAPSHOT_SHA256_INVALID")
    # The transport verifier installs every dynamic receipt into the CPU
    # transaction namespace first.  The remote driver subsequently promotes a
    # verified JIT receipt to the oracle's fixed consumer path, archiving a
    # prior unconsumed JIT when a failed attempt is retried.
    return (
        PROJECT_ROOT
        / "runtime/bootstrap"
        / CPU_NAMESPACE
        / "transactions"
        / transaction
        / "JIT_BUDGET_READY.json"
    )


def _atomic_install(raw: bytes, target: Path, digest: str, size: int) -> None:
    _assert_components(target.parent, "DYNAMIC_TARGET_PARENT")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _assert_components(target.parent, "DYNAMIC_TARGET_PARENT")
    partial = target.with_name(f".{target.name}.{digest}.dynamic.partial")
    if os.path.lexists(target):
        _, current_sha, current_size = _read(
            target,
            "EXISTING_DYNAMIC_RECEIPT",
            allowed_nlinks=frozenset({1, 2}),
        )
        if current_sha != digest or current_size != size:
            raise DynamicReceiptError(f"EXISTING_DYNAMIC_RECEIPT_DRIFT={target}")
        target_stat = os.stat(target, follow_symlinks=False)
        if os.path.lexists(partial):
            _, partial_sha, partial_size = _read(
                partial,
                "DYNAMIC_RECOVERY_PARTIAL",
                allowed_nlinks=frozenset({1, 2}),
            )
            partial_stat = os.stat(partial, follow_symlinks=False)
            if partial_sha != digest or partial_size != size:
                raise DynamicReceiptError("DYNAMIC_RECOVERY_PARTIAL_BYTES_DRIFT")
            same = (target_stat.st_dev, target_stat.st_ino) == (
                partial_stat.st_dev,
                partial_stat.st_ino,
            )
            if same and target_stat.st_nlink == partial_stat.st_nlink == 2:
                partial.unlink()
            elif not same and target_stat.st_nlink == partial_stat.st_nlink == 1:
                partial.unlink()
            else:
                raise DynamicReceiptError("DYNAMIC_RECOVERY_PARTIAL_INODE_DRIFT")
        _, final_sha, final_size = _read(target, "EXISTING_DYNAMIC_RECEIPT_FINAL")
        if final_sha != digest or final_size != size:
            raise DynamicReceiptError("EXISTING_DYNAMIC_RECEIPT_FINAL_DRIFT")
        return
    if os.path.lexists(partial):
        partial_stat = os.lstat(partial)
        if not stat.S_ISREG(partial_stat.st_mode) or partial_stat.st_nlink != 1:
            raise DynamicReceiptError("DYNAMIC_PARTIAL_TYPE_OR_NLINK_DRIFT")
        try:
            _, partial_sha, partial_size = _read(partial, "DYNAMIC_RESUMABLE_PARTIAL")
        except (DynamicReceiptError, OSError):
            partial.unlink()
        else:
            if partial_sha != digest or partial_size != size:
                partial.unlink()
            else:
                os.link(partial, target, follow_symlinks=False)
                partial.unlink()
                _, final_sha, final_size = _read(target, "RESUMED_DYNAMIC_RECEIPT")
                if final_sha != digest or final_size != size:
                    raise DynamicReceiptError("RESUMED_DYNAMIC_RECEIPT_DRIFT")
                return
    fd = os.open(
        partial,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(partial, target, follow_symlinks=False)
    except OSError as exc:
        raise DynamicReceiptError(f"DYNAMIC_RECEIPT_COMMIT_FAILED={target}") from exc
    finally:
        if os.path.lexists(partial):
            partial.unlink()
    _, final_sha, final_size = _read(target, "INSTALLED_DYNAMIC_RECEIPT")
    if final_sha != digest or final_size != size:
        raise DynamicReceiptError("INSTALLED_DYNAMIC_RECEIPT_DRIFT")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("stopped", "jit", "binding"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--transaction-id")
    parser.add_argument("--project-scope-sha256")
    parser.add_argument("--profile-name-sha256")
    parser.add_argument("--expected-generator-sha256")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--install-target", type=Path)
    args = parser.parse_args(argv)
    try:
        if not SHA_RE.fullmatch(args.expected_sha256):
            raise DynamicReceiptError("EXPECTED_SHA256_INVALID")
        payload, raw, digest, size = _load(args.source, "DYNAMIC_RECEIPT_SOURCE")
        if digest != args.expected_sha256 or size != args.expected_size:
            raise DynamicReceiptError("DYNAMIC_RECEIPT_TRANSFER_BINDING_DRIFT")
        if args.kind == "binding":
            if args.audit_only and args.install_target is not None:
                raise DynamicReceiptError("BINDING_INSTALL_AND_AUDIT_ONLY_CONFLICT")
            if args.install_target is not None:
                _atomic_install(raw, args.install_target, digest, size)
            print(
                json.dumps(
                    {
                        "status": "REGULAR_NLINK_HASH_SIZE_PATH_BINDING_VERIFIED",
                        "sha256": digest,
                        "size_bytes": size,
                        "path": str(Path(os.path.abspath(os.fspath(args.source)))),
                        "installed_target": (
                            str(Path(os.path.abspath(os.fspath(args.install_target))))
                            if args.install_target is not None
                            else None
                        ),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.transaction_id is None or not TX_RE.fullmatch(args.transaction_id):
            raise DynamicReceiptError("TRANSACTION_ID_INVALID")
        if args.project_scope_sha256 is None or not SHA_RE.fullmatch(args.project_scope_sha256):
            raise DynamicReceiptError("PROJECT_SCOPE_SHA256_INVALID")
        if args.profile_name_sha256 is None or not SHA_RE.fullmatch(args.profile_name_sha256):
            raise DynamicReceiptError("PROFILE_NAME_SHA256_INVALID")
        if args.kind == "jit" and (
            args.expected_generator_sha256 is None
            or not SHA_RE.fullmatch(args.expected_generator_sha256)
        ):
            raise DynamicReceiptError("EXPECTED_GENERATOR_SHA256_INVALID")
        target = (
            _validate_stopped(
                payload,
                args.transaction_id,
                args.project_scope_sha256,
                args.profile_name_sha256,
            )
            if args.kind == "stopped"
            else _validate_jit(
                payload,
                args.transaction_id,
                args.project_scope_sha256,
                args.profile_name_sha256,
                args.expected_generator_sha256,
            )
        )
        if not args.audit_only:
            _atomic_install(raw, target, digest, size)
        print(
            json.dumps(
                {
                    "status": (
                        "DYNAMIC_RECEIPT_AUDITED_NO_WRITE"
                        if args.audit_only
                        else "DYNAMIC_RECEIPT_INSTALLED_AND_RECHECKED"
                    ),
                    "kind": args.kind,
                    "target": str(target),
                    "sha256": digest,
                    "size_bytes": size,
                    "transaction_id": args.transaction_id,
                },
                sort_keys=True,
            )
        )
        return 0
    except (DynamicReceiptError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 22


if __name__ == "__main__":
    raise SystemExit(main())
