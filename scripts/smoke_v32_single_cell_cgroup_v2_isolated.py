#!/usr/bin/env python3
"""One-shot harmless cgroup-v2 memory.current/memory.peak smoke.

This script never imports or starts a scientific runner.  It creates exactly
one fresh, randomly named child below the explicitly supplied delegated
parent, fixes memory.max at 512 MiB, moves one short Python allocation process
into it, observes aggregate memory files, waits for an empty cgroup, and uses
an exact non-recursive rmdir.  Its JSON receipt is exclusive and immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from typing import Any


FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R10_CGROUP_V2_SMOKE_V1"
MEMORY_LIMIT_BYTES = 512 * 1024**2
SAFE_ALLOCATION_BYTES = 8 * 1024**2
NAME_PREFIX = "cc_hhgt_r10_memory_smoke_"


class SmokeError(RuntimeError):
    """Raised when the isolated cgroup smoke cannot be proved safe."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SmokeError(f"cannot hash absent/unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SmokeError(f"receipt reuse is forbidden: {path}")
    if path.parent.exists() or path.parent.is_symlink():
        raise SmokeError(f"fresh receipt directory already exists: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def parse_events(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise SmokeError("malformed memory.events")
        values[fields[0]] = int(fields[1])
    return values


def read_integer(path: Path) -> int:
    value = int(path.read_text(encoding="ascii").strip())
    if value < 0:
        raise SmokeError(f"negative cgroup value: {path.name}")
    return value


def read_pids(path: Path) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(value)
                for value in path.read_text(encoding="ascii").splitlines()
                if value.strip()
            }
        )
    )


def current_cgroup_v2_path() -> str:
    rows = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    matches = [row.split("::", 1)[1] for row in rows if "::" in row]
    if len(matches) != 1:
        raise SmokeError("current process does not have one cgroup-v2 identity")
    return matches[0]


def join_cgroup(cgroup: Path):
    procs = cgroup / "cgroup.procs"

    def join() -> None:
        with procs.open("w", encoding="ascii") as stream:
            stream.write(str(os.getpid()))

    return join


def terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)


def execute(parent: Path, receipt: Path, expected_self_sha256: str) -> int:
    self_path = Path(__file__).resolve()
    self_sha = sha256_file(self_path)
    if self_sha != expected_self_sha256:
        raise SmokeError("smoke script SHA drift")
    raw_parent = parent
    if raw_parent.is_symlink():
        raise SmokeError("delegated parent is a symlink")
    parent = raw_parent.resolve(strict=True)
    if not parent.is_dir() or parent.stat().st_uid != os.geteuid():
        raise SmokeError("delegated parent is absent or not owned by effective UID")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise SmokeError("delegated parent is not writable/searchable")
    controllers = set(
        (parent / "cgroup.controllers").read_text(encoding="ascii").split()
    )
    subtree = set(
        (parent / "cgroup.subtree_control").read_text(encoding="ascii").split()
    )
    if "memory" not in controllers or "memory" not in subtree:
        raise SmokeError("memory controller is not delegated and enabled")
    current_scope = current_cgroup_v2_path()
    expected_scope_prefix = (
        "/user.slice/user-1001.slice/user@1001.service/app.slice/"
    )
    if not current_scope.startswith(expected_scope_prefix):
        raise SmokeError("smoke controller was not launched inside delegated app.slice")

    name = f"{NAME_PREFIX}{int(time.time())}_{os.getpid()}_{secrets.token_hex(6)}"
    cgroup = parent / name
    if cgroup.exists() or cgroup.is_symlink():
        raise SmokeError("random cgroup collision")
    child: subprocess.Popen[bytes] | None = None
    created = False
    removed = False
    cleanup_error: str | None = None
    failure: str | None = None
    samples: list[dict[str, Any]] = []
    child_pid: int | None = None
    return_code: int | None = None
    try:
        cgroup.mkdir(mode=0o700)
        created = True
        if cgroup.parent != parent or not cgroup.name.startswith(NAME_PREFIX):
            raise SmokeError("created cgroup topology drift")
        (cgroup / "memory.max").write_text(
            str(MEMORY_LIMIT_BYTES), encoding="ascii", newline="\n"
        )
        if (cgroup / "memory.max").read_text(encoding="ascii").strip() != str(
            MEMORY_LIMIT_BYTES
        ):
            raise SmokeError("memory.max did not bind exact 512 MiB")
        if read_integer(cgroup / "memory.peak") != 0:
            raise SmokeError("fresh cgroup memory.peak is nonzero")
        if read_pids(cgroup / "cgroup.procs"):
            raise SmokeError("fresh cgroup is not empty")
        command = [
            sys.executable,
            "-c",
            (
                "import time; x=bytearray(8*1024*1024); "
                "x[0]=1; time.sleep(0.75)"
            ),
        ]
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            preexec_fn=join_cgroup(cgroup),
        )
        child_pid = child.pid
        deadline = time.monotonic() + 5
        while child.poll() is None and time.monotonic() < deadline:
            samples.append(
                {
                    "memory_current": read_integer(cgroup / "memory.current"),
                    "memory_peak": read_integer(cgroup / "memory.peak"),
                    "process_ids": list(read_pids(cgroup / "cgroup.procs")),
                    "events": parse_events(cgroup / "memory.events"),
                }
            )
            time.sleep(0.02)
        if child.poll() is None:
            raise SmokeError("harmless child exceeded its five-second deadline")
        return_code = child.wait()
        if return_code != 0:
            raise SmokeError(f"harmless child exited nonzero: {return_code}")
        empty_deadline = time.monotonic() + 5
        while read_pids(cgroup / "cgroup.procs") and time.monotonic() < empty_deadline:
            time.sleep(0.02)
        final_pids = read_pids(cgroup / "cgroup.procs")
        final_current = read_integer(cgroup / "memory.current")
        final_peak = read_integer(cgroup / "memory.peak")
        final_events = parse_events(cgroup / "memory.events")
        if final_pids:
            raise SmokeError("exclusive cgroup did not become empty")
        if not any(child_pid in row["process_ids"] for row in samples):
            raise SmokeError("harmless child was never observed in exclusive cgroup")
        if final_peak < SAFE_ALLOCATION_BYTES or final_peak > MEMORY_LIMIT_BYTES:
            raise SmokeError("aggregate memory.peak is outside the safe expected range")
        if final_events.get("oom", 0) or final_events.get("oom_kill", 0):
            raise SmokeError("harmless cgroup recorded an OOM event")
        cgroup.rmdir()
        removed = not cgroup.exists() and not cgroup.is_symlink()
        if not removed:
            raise SmokeError("exact empty cgroup rmdir did not complete")
    except Exception as exc:
        failure = f"{type(exc).__name__}:{exc}"
    finally:
        if child is not None:
            try:
                terminate(child)
            except Exception as exc:
                cleanup_error = f"CHILD_TERMINATION:{type(exc).__name__}:{exc}"
        if created and not removed:
            try:
                if read_pids(cgroup / "cgroup.procs"):
                    raise SmokeError("refusing to rmdir non-empty cgroup during cleanup")
                if cgroup.parent != parent or not cgroup.name.startswith(NAME_PREFIX):
                    raise SmokeError("cleanup target topology drift")
                cgroup.rmdir()
                removed = not cgroup.exists() and not cgroup.is_symlink()
                if not removed:
                    raise SmokeError("cleanup rmdir did not remove exact cgroup")
            except Exception as exc:
                extra = f"CGROUP_CLEANUP:{type(exc).__name__}:{exc}"
                cleanup_error = f"{cleanup_error};{extra}" if cleanup_error else extra

    peak = max((int(row["memory_peak"]) for row in samples), default=0)
    current = max((int(row["memory_current"]) for row in samples), default=0)
    max_members = max((len(row["process_ids"]) for row in samples), default=0)
    payload: dict[str, Any] = {
        "format": FORMAT,
        "status": "PASS" if failure is None and cleanup_error is None and removed else "FAIL_CLOSED",
        "smoke_script_path": str(self_path),
        "smoke_script_sha256": self_sha,
        "delegated_parent": str(parent),
        "controller_cgroup": current_scope,
        "exclusive_child_cgroup": str(cgroup),
        "exclusive_child_created": created,
        "exclusive_child_removed": removed,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "memory_max_exactly_bound": True if created else False,
        "harmless_allocation_bytes": SAFE_ALLOCATION_BYTES,
        "harmless_child_pid": child_pid,
        "harmless_child_return_code": return_code,
        "sample_count": len(samples),
        "maximum_memory_current_bytes_observed": current,
        "maximum_memory_peak_bytes_observed": peak,
        "maximum_cgroup_process_count_observed": max_members,
        "child_observed_in_cgroup": child_pid is not None
        and any(child_pid in row["process_ids"] for row in samples),
        "aggregate_peak_persists_between_polls": peak >= SAFE_ALLOCATION_BYTES,
        "final_memory_events": samples[-1]["events"] if samples else None,
        "failure": failure,
        "cleanup_error": cleanup_error,
        "scientific_runner_started": False,
        "production_deployed": False,
        "port_8260_touched": False,
        "timestamp_unix": time.time(),
    }
    payload["receipt_contract_sha256"] = canonical_sha256(payload)
    exclusive_json(receipt, payload)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delegated-parent", required=True, type=Path)
    parser.add_argument("--receipt-json", required=True, type=Path)
    parser.add_argument("--expected-self-sha256", required=True)
    args = parser.parse_args()
    return execute(
        args.delegated_parent,
        args.receipt_json.resolve(),
        args.expected_self_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
