#!/usr/bin/env python3
"""Pure-stdlib, per-cancer watchdog for the frozen R9/R10 runners.

The coordinator never imports the scientific stack.  It starts a fresh
watchdog interpreter for each cancer; that watchdog starts every scientific
job in a fresh process group and exits when the cancer is complete.  The
512 MiB gate preferentially uses a new, exclusive cgroup-v2 job scope:

    max(cgroup memory.current, cgroup memory.peak)

This catches a short aggregate peak between polling instants.  If a delegated
cgroup-v2 parent is unavailable, the only permitted fallback is a SHA-bound
runner proved not to overlap helper processes.  That fallback uses:

    max(sum(live member VmRSS), max(live member VmHWM))

Member high-water marks are deliberately not summed because they need not
have occurred at the same time.  More than one live group member makes the
fallback fail closed.  The watchdog's own RSS/HWM is recorded as monitoring
overhead but is never added to either runner metric.

This is a launch candidate.  Cohort mode is fail-closed behind an exact code
SHA, launch-spec SHA, immutable R9 receipts, fresh roots, and an explicit
post-review confirmation token.  It does not publish the final 17-cancer
binding; successful supervision remains pending a separate scientific audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence


FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R10_ISOLATED_SUPERVISOR_V1"
LAUNCH_SPEC_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R10_ISOLATED_LAUNCH_SPEC_V1"
WATCH_REQUEST_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R10_CANCER_WATCH_REQUEST_V1"
WATCH_RECEIPT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R10_CANCER_WATCH_RECEIPT_V1"
FAILURE_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R10_ISOLATED_TYPED_FAILURE_V1"
SUCCESS_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R10_ISOLATED_SUCCESS_V1"

MEMORY_LIMIT_BYTES = 512 * 1024**2
POLL_SECONDS = 0.25
ROOT_CONFIRMATION_TOKEN = "ROOT_CONFIRMED_AFTER_ISOLATED_SUPERVISOR_REVIEW"
REMAINING_EXECUTION_ORDER = ("UCEC", "READ", "GBM", "PCPG")
AUTHORIZED_ROOT = Path("./data/CancerLncAtlas")
SCIENTIFIC_TOP_LEVEL_MODULES = frozenset(
    {
        "anndata",
        "duckdb",
        "h5py",
        "numpy",
        "pandas",
        "polars",
        "pyarrow",
        "scanpy",
        "scipy",
        "sklearn",
        "torch",
    }
)

REQUIRED_IMMUTABLE_BINDINGS = {
    "R9_ACC_SUCCESS": {
        "path": (
            "./data/CancerLncAtlas/results/model/"
            "v32_single_cell_r7_fresh_streaming_r9_remaining5_20260829_r1/"
            "cancer_id=ACC/SUCCESS.json"
        ),
        "sha256": "b3dd5d51c55fc79d39bd4dafa21dd682e2651a4b5d2a3b820ab9abdb41b1923d",
    },
    "R9_TYPED_FAILURE": {
        "path": (
            "./data/CancerLncAtlas/runtime/audits/"
            "single_cell_r7_r9_remaining5_20260829_r1/TYPED_FAILURE.json"
        ),
        "sha256": "24163d6555ada72eab2eea76024cee15e4dfd291a3f4d7b0dd0322bc19a2dc69",
    },
    "R9_UCEC_PLAN": {
        "path": (
            "./data/CancerLncAtlas/runtime/audits/"
            "single_cell_r7_r9_remaining5_20260829_r1/plans/02_UCEC.json"
        ),
        "sha256": "3c03b7fbceb0270eca73272d0c54356d09fe8b45e37e2e278b9a7062efc01e01",
    },
}

THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
ALLOWED_JOB_ENVIRONMENT_KEYS = frozenset({"PYTHONPATH", *THREAD_ENVIRONMENT})


class IsolatedSupervisorError(RuntimeError):
    """Raised when a launch or observation invariant cannot be proved."""


class ProcessIdentity(NamedTuple):
    process_group_id: int
    start_time_ticks: int


class ProcessMemory(NamedTuple):
    process_id: int
    rss_bytes: int
    high_water_bytes: int


class GroupMemorySnapshot(NamedTuple):
    process_group_id: int
    members: tuple[ProcessMemory, ...]
    unreadable_live_member_pids: tuple[int, ...]
    observation_errors: tuple[str, ...]
    current_group_rss_sum_bytes: int
    maximum_live_member_hwm_bytes: int


class CgroupMemorySnapshot(NamedTuple):
    cgroup_path: str
    current_bytes: int
    peak_bytes: int
    process_ids: tuple[int, ...]
    memory_events: tuple[tuple[str, int], ...]
    observation_errors: tuple[str, ...]


class MemoryDecision(NamedTuple):
    stop: bool
    reason: str | None
    formal_gate_value_bytes: int
    memory_limit_bytes: int


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise IsolatedSupervisorError(f"cannot hash absent/unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise IsolatedSupervisorError(f"JSON is absent/unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IsolatedSupervisorError(f"JSON root is not an object: {path}")
    return value


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise IsolatedSupervisorError(f"immutable JSON reuse is forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise IsolatedSupervisorError(f"unsafe JSON temporary exists: {temporary}")
    exclusive_json(temporary, payload)
    os.replace(temporary, path)


def loaded_scientific_modules() -> list[str]:
    loaded = {name.partition(".")[0] for name in sys.modules}
    return sorted(loaded & SCIENTIFIC_TOP_LEVEL_MODULES)


def assert_stdlib_control_plane() -> None:
    loaded = loaded_scientific_modules()
    if loaded:
        raise IsolatedSupervisorError(
            "scientific modules are loaded in the supervisor process: "
            + ",".join(loaded)
        )


def parse_proc_stat_identity(text: str) -> ProcessIdentity:
    """Parse pgrp and starttime from Linux /proc/PID/stat.

    The command name is parenthesized and may contain spaces or right
    parentheses, so splitting the entire line is unsafe; the final ')' is the
    kernel field boundary.
    """

    close = text.rfind(")")
    open_ = text.find("(")
    if open_ <= 0 or close <= open_:
        raise IsolatedSupervisorError("malformed /proc stat command field")
    fields = text[close + 1 :].strip().split()
    # fields[0] is field 3 (state), fields[2] is field 5 (pgrp), and
    # fields[19] is field 22 (starttime).
    if len(fields) <= 19:
        raise IsolatedSupervisorError("truncated /proc stat record")
    try:
        return ProcessIdentity(int(fields[2]), int(fields[19]))
    except ValueError as exc:
        raise IsolatedSupervisorError("non-integer /proc stat identity") from exc


def parse_proc_status_memory(text: str, process_id: int) -> ProcessMemory:
    values: dict[str, int] = {}
    for line in text.splitlines():
        if not (line.startswith("VmRSS:") or line.startswith("VmHWM:")):
            continue
        fields = line.split()
        if len(fields) != 3 or fields[2] != "kB":
            raise IsolatedSupervisorError(
                f"unexpected /proc status memory unit for PID {process_id}"
            )
        try:
            values[fields[0][:-1]] = int(fields[1]) * 1024
        except ValueError as exc:
            raise IsolatedSupervisorError(
                f"non-integer /proc status memory for PID {process_id}"
            ) from exc
    if set(values) != {"VmRSS", "VmHWM"}:
        raise IsolatedSupervisorError(
            f"VmRSS/VmHWM absent for live process PID {process_id}"
        )
    return ProcessMemory(process_id, values["VmRSS"], values["VmHWM"])


def _default_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def collect_process_group_snapshot(
    process_group_id: int,
    *,
    proc_root: Path = Path("/proc"),
    read_text: Callable[[Path], str] = _default_read_text,
) -> GroupMemorySnapshot:
    """Return a race-aware snapshot of only the currently live group members."""

    members: list[ProcessMemory] = []
    unreadable: list[int] = []
    errors: list[str] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        entries = []
        errors.append(f"PROC_SCAN_FAILED:{type(exc).__name__}:{exc}")
    for entry in entries:
        if not entry.name.isdigit():
            continue
        process_id = int(entry.name)
        stat_path = entry / "stat"
        status_path = entry / "status"
        try:
            identity_before = parse_proc_stat_identity(read_text(stat_path))
        except (OSError, IsolatedSupervisorError):
            continue
        if identity_before.process_group_id != int(process_group_id):
            continue
        try:
            status_text = read_text(status_path)
            memory = parse_proc_status_memory(status_text, process_id)
        except (OSError, IsolatedSupervisorError) as exc:
            try:
                identity_after_error = parse_proc_stat_identity(read_text(stat_path))
            except (OSError, IsolatedSupervisorError):
                continue
            if identity_after_error == identity_before:
                unreadable.append(process_id)
                errors.append(
                    f"LIVE_MEMBER_STATUS_UNREADABLE:{process_id}:"
                    f"{type(exc).__name__}:{exc}"
                )
            continue
        try:
            identity_after = parse_proc_stat_identity(read_text(stat_path))
        except (OSError, IsolatedSupervisorError):
            continue
        if identity_after != identity_before:
            # The process exited or the PID was reused between reads.  Its old
            # HWM is not a live-group observation and must not enter the gate.
            continue
        members.append(memory)
    members.sort(key=lambda row: row.process_id)
    return GroupMemorySnapshot(
        int(process_group_id),
        tuple(members),
        tuple(sorted(unreadable)),
        tuple(errors),
        sum(row.rss_bytes for row in members),
        max((row.high_water_bytes for row in members), default=0),
    )


def _parse_nonnegative_integer(text: str, *, field: str) -> int:
    try:
        value = int(text.strip())
    except ValueError as exc:
        raise IsolatedSupervisorError(f"non-integer cgroup field: {field}") from exc
    if value < 0:
        raise IsolatedSupervisorError(f"negative cgroup field: {field}")
    return value


def _parse_memory_events(text: str) -> tuple[tuple[str, int], ...]:
    values: list[tuple[str, int]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise IsolatedSupervisorError("malformed cgroup memory.events")
        values.append(
            (fields[0], _parse_nonnegative_integer(fields[1], field=fields[0]))
        )
    return tuple(sorted(values))


def read_cgroup_v2_snapshot(cgroup_path: Path) -> CgroupMemorySnapshot:
    """Read aggregate current/peak memory from one exclusive cgroup-v2 job."""

    errors: list[str] = []
    current = peak = 0
    process_ids: tuple[int, ...] = ()
    events: tuple[tuple[str, int], ...] = ()
    try:
        if cgroup_path.is_symlink() or not cgroup_path.is_dir():
            raise IsolatedSupervisorError("exclusive cgroup path is absent/unsafe")
        current = _parse_nonnegative_integer(
            (cgroup_path / "memory.current").read_text(encoding="ascii"),
            field="memory.current",
        )
        peak = _parse_nonnegative_integer(
            (cgroup_path / "memory.peak").read_text(encoding="ascii"),
            field="memory.peak",
        )
        maximum = (cgroup_path / "memory.max").read_text(
            encoding="ascii"
        ).strip()
        if maximum != str(MEMORY_LIMIT_BYTES):
            raise IsolatedSupervisorError("exclusive cgroup memory.max is not 512 MiB")
        raw_pids = (cgroup_path / "cgroup.procs").read_text(
            encoding="ascii"
        ).splitlines()
        process_ids = tuple(
            sorted(
                {
                    _parse_nonnegative_integer(value, field="cgroup.procs")
                    for value in raw_pids
                    if value.strip()
                }
            )
        )
        events = _parse_memory_events(
            (cgroup_path / "memory.events").read_text(encoding="ascii")
        )
    except (OSError, IsolatedSupervisorError) as exc:
        errors.append(f"CGROUP_V2_MEMORY_UNREADABLE:{type(exc).__name__}:{exc}")
    return CgroupMemorySnapshot(
        str(cgroup_path), current, peak, process_ids, events, tuple(errors)
    )


def evaluate_cgroup_v2_snapshot(
    snapshot: CgroupMemorySnapshot,
    *,
    expected_leader_pid: int | None = None,
    leader_is_running: bool = True,
) -> MemoryDecision:
    formal_value = max(snapshot.current_bytes, snapshot.peak_bytes)
    if snapshot.observation_errors:
        return MemoryDecision(
            True,
            "CGROUP_V2_MEMORY_UNOBSERVABLE",
            int(formal_value),
            MEMORY_LIMIT_BYTES,
        )
    if (
        leader_is_running
        and expected_leader_pid is not None
        and int(expected_leader_pid) not in snapshot.process_ids
    ):
        return MemoryDecision(
            True,
            "CGROUP_V2_LEADER_NOT_BOUND",
            int(formal_value),
            MEMORY_LIMIT_BYTES,
        )
    events = dict(snapshot.memory_events)
    if events.get("oom", 0) or events.get("oom_kill", 0):
        return MemoryDecision(
            True,
            "CGROUP_V2_MEMORY_OOM_EVENT",
            int(formal_value),
            MEMORY_LIMIT_BYTES,
        )
    if formal_value > MEMORY_LIMIT_BYTES:
        return MemoryDecision(
            True,
            "OBSERVED_CGROUP_V2_MEMORY_PEAK_EXCEEDED_512_MIB",
            int(formal_value),
            MEMORY_LIMIT_BYTES,
        )
    return MemoryDecision(False, None, int(formal_value), MEMORY_LIMIT_BYTES)


def _decode_mountinfo_path(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def cgroup_v2_mount_for(
    path: Path, *, mountinfo_path: Path = Path("/proc/self/mountinfo")
) -> Path | None:
    resolved = path.resolve()
    matches: list[Path] = []
    try:
        lines = mountinfo_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return None
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        pre_fields = before.split()
        post_fields = after.split()
        if len(pre_fields) < 5 or not post_fields or post_fields[0] != "cgroup2":
            continue
        mount = Path(_decode_mountinfo_path(pre_fields[4])).resolve()
        if resolved == mount or _is_within(resolved, mount):
            matches.append(mount)
    return max(matches, key=lambda item: len(item.parts), default=None)


def current_cgroup_v2_absolute_path(mount: Path) -> Path:
    rows = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    identities = [row.split("::", 1)[1] for row in rows if "::" in row]
    if len(identities) != 1 or not identities[0].startswith("/"):
        raise IsolatedSupervisorError("current process lacks one cgroup-v2 identity")
    return (mount / identities[0].lstrip("/")).resolve(strict=True)


def create_exclusive_job_cgroup(parent: Path, name: str) -> Path:
    """Create one fresh delegated cgroup and bind its kernel 512 MiB ceiling."""

    if (
        not name
        or name in {".", ".."}
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
               for character in name)
    ):
        raise IsolatedSupervisorError("unsafe exclusive cgroup job name")
    if parent.is_symlink():
        raise IsolatedSupervisorError("delegated cgroup parent is a symlink")
    parent = parent.resolve(strict=True)
    mount = cgroup_v2_mount_for(parent)
    if mount is None:
        raise IsolatedSupervisorError("delegated parent is not on cgroup v2")
    controller_cgroup = current_cgroup_v2_absolute_path(mount)
    if controller_cgroup != parent and not _is_within(controller_cgroup, parent):
        raise IsolatedSupervisorError(
            "watchdog is outside delegated parent; safe child migration is unproved"
        )
    if not parent.is_dir():
        raise IsolatedSupervisorError("delegated cgroup parent is absent/unsafe")
    if parent.stat().st_uid != os.geteuid() or not os.access(
        parent, os.W_OK | os.X_OK
    ):
        raise IsolatedSupervisorError("delegated cgroup parent is not user-writable")
    controllers = set(
        (parent / "cgroup.controllers").read_text(encoding="ascii").split()
    )
    subtree = set(
        (parent / "cgroup.subtree_control").read_text(encoding="ascii").split()
    )
    if "memory" not in controllers or "memory" not in subtree:
        raise IsolatedSupervisorError(
            "delegated cgroup parent has not enabled the memory controller"
        )
    cgroup = parent / name
    if cgroup.exists() or cgroup.is_symlink():
        raise IsolatedSupervisorError(f"exclusive job cgroup already exists: {cgroup}")
    cgroup.mkdir(mode=0o700)
    try:
        (cgroup / "memory.max").write_text(
            str(MEMORY_LIMIT_BYTES), encoding="ascii", newline="\n"
        )
        if (cgroup / "memory.peak").read_text(encoding="ascii").strip() != "0":
            raise IsolatedSupervisorError("fresh cgroup memory.peak is not zero")
        if (cgroup / "cgroup.procs").read_text(encoding="ascii").strip():
            raise IsolatedSupervisorError("fresh cgroup unexpectedly contains a process")
        snapshot = read_cgroup_v2_snapshot(cgroup)
        if snapshot.observation_errors:
            raise IsolatedSupervisorError(snapshot.observation_errors[0])
    except Exception:
        try:
            cgroup.rmdir()
        except OSError:
            pass
        raise
    return cgroup


def preflight_cgroup_v2_observation(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    if validate_memory_observation_contract(observation) != "CGROUP_V2_EXCLUSIVE":
        raise IsolatedSupervisorError("not a cgroup-v2 observation contract")
    raw_parent = Path(str(observation["delegated_parent"]))
    if raw_parent.is_symlink():
        raise IsolatedSupervisorError("delegated cgroup parent is a symlink")
    parent = raw_parent.resolve(strict=True)
    mount = cgroup_v2_mount_for(parent)
    if mount is None:
        raise IsolatedSupervisorError("delegated parent is not on cgroup v2")
    controller_cgroup = current_cgroup_v2_absolute_path(mount)
    if controller_cgroup != parent and not _is_within(controller_cgroup, parent):
        raise IsolatedSupervisorError(
            "supervisor must be launched inside the delegated cgroup parent"
        )
    observed = parent.stat()
    controllers = set(
        (parent / "cgroup.controllers").read_text(encoding="ascii").split()
    )
    subtree = set(
        (parent / "cgroup.subtree_control").read_text(encoding="ascii").split()
    )
    if "memory" not in controllers or "memory" not in subtree:
        raise IsolatedSupervisorError(
            "delegated cgroup parent has not enabled the memory controller"
        )
    if observed.st_uid != os.geteuid() or not os.access(
        parent, os.W_OK | os.X_OK
    ):
        raise IsolatedSupervisorError("delegated cgroup parent is not user-writable")
    job = parent / str(observation["job_cgroup_name"])
    if job.exists() or job.is_symlink():
        raise IsolatedSupervisorError(f"exclusive job cgroup already exists: {job}")
    required_parent_files = (
        "cgroup.controllers",
        "cgroup.subtree_control",
        "cgroup.procs",
        "memory.current",
        "memory.peak",
    )
    if any(not (parent / name).is_file() for name in required_parent_files):
        raise IsolatedSupervisorError("delegated parent lacks cgroup-v2 memory files")
    return {
        "mode": "CGROUP_V2_EXCLUSIVE",
        "delegated_parent": str(parent),
        "mount": str(mount),
        "controller_cgroup": str(controller_cgroup),
        "controller_inside_delegated_parent": True,
        "parent_uid": int(observed.st_uid),
        "effective_uid": int(os.geteuid()),
        "memory_controller_available": True,
        "memory_controller_enabled_for_children": True,
        "parent_user_writable": True,
        "job_cgroup_path": str(job),
        "job_cgroup_fresh": True,
        "memory_max_bytes_to_bind": MEMORY_LIMIT_BYTES,
        "memory_current_available": True,
        "memory_peak_available": True,
    }


def remove_empty_job_cgroup(cgroup: Path) -> None:
    snapshot = read_cgroup_v2_snapshot(cgroup)
    if snapshot.observation_errors:
        raise IsolatedSupervisorError(snapshot.observation_errors[0])
    if snapshot.process_ids:
        raise IsolatedSupervisorError("cannot remove a non-empty exclusive job cgroup")
    cgroup.rmdir()


def _join_cgroup_before_exec(cgroup: Path) -> Callable[[], None]:
    procs = cgroup / "cgroup.procs"

    def join() -> None:
        # The watchdog is deliberately single-threaded.  This hook runs in the
        # forked child before exec, so every later helper inherits the scope.
        with procs.open("w", encoding="ascii") as stream:
            stream.write(str(os.getpid()))

    return join


def evaluate_memory_snapshot(
    snapshot: GroupMemorySnapshot,
    *,
    memory_limit_bytes: int = MEMORY_LIMIT_BYTES,
    require_live_member: bool = True,
    require_single_live_member: bool = False,
) -> MemoryDecision:
    if int(memory_limit_bytes) != MEMORY_LIMIT_BYTES:
        raise IsolatedSupervisorError("the formal 512 MiB memory limit is immutable")
    formal_value = max(
        snapshot.current_group_rss_sum_bytes,
        snapshot.maximum_live_member_hwm_bytes,
    )
    if snapshot.observation_errors or snapshot.unreadable_live_member_pids:
        return MemoryDecision(
            True,
            "PROCESS_GROUP_MEMORY_UNOBSERVABLE",
            int(formal_value),
            MEMORY_LIMIT_BYTES,
        )
    if require_live_member and not snapshot.members:
        return MemoryDecision(
            True,
            "PROCESS_GROUP_HAS_NO_OBSERVABLE_LIVE_MEMBER",
            int(formal_value),
            MEMORY_LIMIT_BYTES,
        )
    if require_single_live_member and len(snapshot.members) != 1:
        return MemoryDecision(
            True,
            "HELPER_PROCESS_OVERLAP_FORBIDDEN_WITHOUT_CGROUP_PEAK",
            int(formal_value),
            MEMORY_LIMIT_BYTES,
        )
    if formal_value > MEMORY_LIMIT_BYTES:
        return MemoryDecision(
            True,
            "OBSERVED_PROCESS_GROUP_MEMORY_EXCEEDED_512_MIB",
            int(formal_value),
            MEMORY_LIMIT_BYTES,
        )
    return MemoryDecision(False, None, int(formal_value), MEMORY_LIMIT_BYTES)


def process_memory_bytes(process_id: int) -> ProcessMemory | None:
    path = Path(f"/proc/{int(process_id)}/status")
    try:
        return parse_proc_status_memory(_default_read_text(path), int(process_id))
    except (OSError, IsolatedSupervisorError):
        return None


def current_process_start_time_ticks() -> int:
    try:
        return parse_proc_stat_identity(
            Path("/proc/self/stat").read_text(encoding="utf-8", errors="replace")
        ).start_time_ticks
    except (OSError, IsolatedSupervisorError) as exc:
        raise IsolatedSupervisorError(
            f"cannot bind watchdog process start identity: {exc}"
        ) from exc


def terminate_process_group(
    process: subprocess.Popen[bytes], *, grace_seconds: float = 10.0
) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=grace_seconds)


def _snapshot_payload(snapshot: GroupMemorySnapshot) -> dict[str, Any]:
    return {
        "process_group_id": snapshot.process_group_id,
        "live_member_count": len(snapshot.members),
        "live_member_pids": [row.process_id for row in snapshot.members],
        "current_group_rss_sum_bytes": snapshot.current_group_rss_sum_bytes,
        "maximum_live_member_hwm_bytes": snapshot.maximum_live_member_hwm_bytes,
        "unreadable_live_member_pids": list(snapshot.unreadable_live_member_pids),
        "observation_errors": list(snapshot.observation_errors),
    }


def _cgroup_snapshot_payload(snapshot: CgroupMemorySnapshot) -> dict[str, Any]:
    return {
        "cgroup_path": snapshot.cgroup_path,
        "memory_current_bytes": snapshot.current_bytes,
        "memory_peak_bytes": snapshot.peak_bytes,
        "process_ids": list(snapshot.process_ids),
        "memory_events": dict(snapshot.memory_events),
        "observation_errors": list(snapshot.observation_errors),
    }


def validate_memory_observation_contract(value: Mapping[str, Any]) -> str:
    mode = str(value.get("mode", ""))
    if mode == "CGROUP_V2_EXCLUSIVE":
        parent = Path(str(value.get("delegated_parent", "")))
        name = str(value.get("job_cgroup_name", ""))
        if not parent.is_absolute() or not name:
            raise IsolatedSupervisorError("cgroup-v2 observation contract is incomplete")
        if value.get("exclusive_fresh_job_cgroup") is not True:
            raise IsolatedSupervisorError("cgroup-v2 job scope is not declared fresh")
        if int(value.get("memory_max_bytes", -1)) != MEMORY_LIMIT_BYTES:
            raise IsolatedSupervisorError("cgroup-v2 memory.max is not fixed at 512 MiB")
        return mode
    if mode == "PROC_SINGLE_MEMBER_FALLBACK":
        proof = value.get("single_member_proof")
        if value.get("helper_overlap_permitted") is not False:
            raise IsolatedSupervisorError("helper overlap is forbidden without cgroup peak")
        if not isinstance(proof, dict):
            raise IsolatedSupervisorError("single-member fallback proof is absent")
        if (
            proof.get("claim")
            != "FROZEN_RUNNER_USES_EXEC_REPLACEMENT_AND_SPAWNS_NO_HELPERS"
            or proof.get("helper_overlap_permitted") is not False
            or not Path(str(proof.get("runner_path", ""))).is_absolute()
            or not isinstance(proof.get("runner_sha256"), str)
            or len(str(proof.get("runner_sha256"))) != 64
            or not Path(str(proof.get("proof_receipt_path", ""))).is_absolute()
            or not isinstance(proof.get("proof_receipt_sha256"), str)
            or len(str(proof.get("proof_receipt_sha256"))) != 64
        ):
            raise IsolatedSupervisorError("single-member fallback proof semantics drift")
        return mode
    raise IsolatedSupervisorError("memory observation mode must be cgroup-v2 or proved fallback")


def run_monitored(
    argv: Sequence[str],
    *,
    log_path: Path,
    environment_additions: Mapping[str, str] | None = None,
    memory_observation: Mapping[str, Any],
    memory_limit_bytes: int = MEMORY_LIMIT_BYTES,
    poll_seconds: float = POLL_SECONDS,
) -> dict[str, Any]:
    """Run one job with cgroup-v2 peak or a proved single-member fallback."""

    assert_stdlib_control_plane()
    if int(memory_limit_bytes) != MEMORY_LIMIT_BYTES:
        raise IsolatedSupervisorError("the formal 512 MiB limit cannot be changed")
    if float(poll_seconds) <= 0 or float(poll_seconds) > POLL_SECONDS:
        raise IsolatedSupervisorError("memory polling may not be slower than 0.25 s")
    if not argv or any(not isinstance(value, str) or not value for value in argv):
        raise IsolatedSupervisorError("argv must be a non-empty string array")
    observation_mode = validate_memory_observation_contract(memory_observation)
    if log_path.exists() or log_path.is_symlink():
        raise IsolatedSupervisorError(f"log reuse is forbidden: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    additions = dict(environment_additions or {})
    forbidden = set(additions) - ALLOWED_JOB_ENVIRONMENT_KEYS
    if forbidden:
        raise IsolatedSupervisorError(
            "job environment contains forbidden keys: " + ",".join(sorted(forbidden))
        )
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    environment.update(additions)
    started = time.time()
    peaks = {
        "process_group_rss_sum_bytes": 0,
        "maximum_live_member_hwm_bytes": 0,
        "cgroup_memory_current_bytes": 0,
        "cgroup_memory_peak_bytes": 0,
        "formal_gate_value_bytes": 0,
        "live_member_count": 0,
        "watchdog_rss_bytes": 0,
        "watchdog_hwm_bytes": 0,
    }
    sample_count = 0
    stop_reason: str | None = None
    last_snapshot: GroupMemorySnapshot | None = None
    last_cgroup_snapshot: CgroupMemorySnapshot | None = None
    cgroup: Path | None = None
    cgroup_cleanup_error: str | None = None
    if observation_mode == "CGROUP_V2_EXCLUSIVE":
        cgroup = create_exclusive_job_cgroup(
            Path(str(memory_observation["delegated_parent"])),
            str(memory_observation["job_cgroup_name"]),
        )
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                shell=False,
                start_new_session=True,
                preexec_fn=_join_cgroup_before_exec(cgroup)
                if cgroup is not None
                else None,
            )
            while process.poll() is None:
                snapshot = collect_process_group_snapshot(process.pid)
                last_snapshot = snapshot
                if observation_mode == "CGROUP_V2_EXCLUSIVE":
                    assert cgroup is not None
                    cgroup_snapshot = read_cgroup_v2_snapshot(cgroup)
                    last_cgroup_snapshot = cgroup_snapshot
                    leader_running = process.poll() is None
                    if not leader_running:
                        break
                    decision = evaluate_cgroup_v2_snapshot(
                        cgroup_snapshot,
                        expected_leader_pid=process.pid,
                        leader_is_running=True,
                    )
                    peaks["cgroup_memory_current_bytes"] = max(
                        peaks["cgroup_memory_current_bytes"],
                        cgroup_snapshot.current_bytes,
                    )
                    peaks["cgroup_memory_peak_bytes"] = max(
                        peaks["cgroup_memory_peak_bytes"], cgroup_snapshot.peak_bytes
                    )
                else:
                    # A process may exit between poll() and /proc enumeration.
                    # Re-poll before treating an empty group as unobservable.
                    if not snapshot.members and process.poll() is not None:
                        break
                    decision = evaluate_memory_snapshot(
                        snapshot, require_single_live_member=True
                    )
                sample_count += 1
                peaks["process_group_rss_sum_bytes"] = max(
                    peaks["process_group_rss_sum_bytes"],
                    snapshot.current_group_rss_sum_bytes,
                )
                peaks["maximum_live_member_hwm_bytes"] = max(
                    peaks["maximum_live_member_hwm_bytes"],
                    snapshot.maximum_live_member_hwm_bytes,
                )
                peaks["formal_gate_value_bytes"] = max(
                    peaks["formal_gate_value_bytes"], decision.formal_gate_value_bytes
                )
                peaks["live_member_count"] = max(
                    peaks["live_member_count"], len(snapshot.members)
                )
                watchdog_memory = process_memory_bytes(os.getpid())
                if watchdog_memory is not None:
                    peaks["watchdog_rss_bytes"] = max(
                        peaks["watchdog_rss_bytes"], watchdog_memory.rss_bytes
                    )
                    peaks["watchdog_hwm_bytes"] = max(
                        peaks["watchdog_hwm_bytes"],
                        watchdog_memory.high_water_bytes,
                    )
                if decision.stop:
                    stop_reason = decision.reason
                    terminate_process_group(process)
                    break
                time.sleep(float(poll_seconds))
            return_code = process.wait()
            if cgroup is not None:
                final_cgroup = read_cgroup_v2_snapshot(cgroup)
                last_cgroup_snapshot = final_cgroup
                final_decision = evaluate_cgroup_v2_snapshot(
                    final_cgroup,
                    expected_leader_pid=None,
                    leader_is_running=False,
                )
                peaks["cgroup_memory_current_bytes"] = max(
                    peaks["cgroup_memory_current_bytes"], final_cgroup.current_bytes
                )
                peaks["cgroup_memory_peak_bytes"] = max(
                    peaks["cgroup_memory_peak_bytes"], final_cgroup.peak_bytes
                )
                peaks["formal_gate_value_bytes"] = max(
                    peaks["formal_gate_value_bytes"],
                    final_decision.formal_gate_value_bytes,
                )
                if stop_reason is None and final_decision.stop:
                    stop_reason = final_decision.reason
            log.flush()
            os.fsync(log.fileno())
    finally:
        if cgroup is not None:
            try:
                remove_empty_job_cgroup(cgroup)
            except (OSError, IsolatedSupervisorError) as exc:
                cgroup_cleanup_error = f"{type(exc).__name__}:{exc}"
                if stop_reason is None:
                    stop_reason = "CGROUP_V2_CLEANUP_FAILED"
    metric = (
        "MAX_OF_EXCLUSIVE_CGROUP_V2_MEMORY_CURRENT_AND_MEMORY_PEAK"
        if observation_mode == "CGROUP_V2_EXCLUSIVE"
        else "MAX_OF_SINGLE_LIVE_PROCESS_RSS_AND_ITS_VMHWM"
    )
    return {
        "argv": list(argv),
        "return_code": int(return_code),
        "elapsed_seconds": round(time.time() - started, 3),
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "memory_exceeded": stop_reason in {
            "OBSERVED_PROCESS_GROUP_MEMORY_EXCEEDED_512_MIB",
            "OBSERVED_CGROUP_V2_MEMORY_PEAK_EXCEEDED_512_MIB",
            "CGROUP_V2_MEMORY_OOM_EVENT",
        },
        "memory_observation_failed": stop_reason in {
            "PROCESS_GROUP_MEMORY_UNOBSERVABLE",
            "PROCESS_GROUP_HAS_NO_OBSERVABLE_LIVE_MEMBER",
            "CGROUP_V2_MEMORY_UNOBSERVABLE",
            "CGROUP_V2_LEADER_NOT_BOUND",
            "CGROUP_V2_CLEANUP_FAILED",
        },
        "stop_reason": stop_reason,
        "memory_observation_mode": observation_mode,
        "memory_metric": metric,
        "cgroup_v2_aggregate_peak_authoritative": observation_mode
        == "CGROUP_V2_EXCLUSIVE",
        "between_poll_aggregate_peak_observable": observation_mode
        == "CGROUP_V2_EXCLUSIVE",
        "sampling_limitation": None
        if observation_mode == "CGROUP_V2_EXCLUSIVE"
        else (
            "PROC_SAMPLING_CANNOT_OBSERVE_SHORT_MULTI_PROCESS_AGGREGATE_PEAKS;"
            "FALLBACK_ALLOWED_ONLY_WITH_BOUND_NO_HELPER_OVERLAP_PROOF"
        ),
        "helper_overlap_permitted": False,
        "single_member_proof": memory_observation.get("single_member_proof")
        if observation_mode == "PROC_SINGLE_MEMBER_FALLBACK"
        else None,
        "member_hwm_values_summed": False,
        "exited_member_hwm_retained_in_gate": False,
        "watchdog_memory_included_in_formal_gate": False,
        "process_group_rss_sum_measured": True,
        "sample_count": sample_count,
        "poll_seconds": float(poll_seconds),
        "peaks": peaks,
        "last_snapshot": _snapshot_payload(last_snapshot)
        if last_snapshot is not None
        else None,
        "last_cgroup_v2_snapshot": _cgroup_snapshot_payload(last_cgroup_snapshot)
        if last_cgroup_snapshot is not None
        else None,
        "exclusive_cgroup_removed_after_job": cgroup is not None
        and cgroup_cleanup_error is None,
        "cgroup_cleanup_error": cgroup_cleanup_error,
        "thread_pins": dict(THREAD_ENVIRONMENT),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def verify_postcondition(row: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(row.get("path", "")))
    if not path.is_absolute() or path.is_symlink():
        raise IsolatedSupervisorError(f"unsafe postcondition path: {path}")
    kind = str(row.get("kind", "file"))
    if kind == "file":
        if not path.is_file():
            raise IsolatedSupervisorError(f"required file absent: {path}")
    elif kind == "directory":
        if not path.is_dir():
            raise IsolatedSupervisorError(f"required directory absent: {path}")
    else:
        raise IsolatedSupervisorError(f"unknown postcondition kind: {kind}")
    result: dict[str, Any] = {"path": str(path), "kind": kind}
    if kind == "file":
        result["bytes"] = path.stat().st_size
        result["sha256"] = sha256_file(path)
        expected_sha = row.get("sha256")
        if expected_sha is not None and result["sha256"] != expected_sha:
            raise IsolatedSupervisorError(f"postcondition SHA drift: {path}")
    expected_json = row.get("json_equals")
    if expected_json is not None:
        if kind != "file" or not isinstance(expected_json, dict):
            raise IsolatedSupervisorError("json_equals requires a file and object")
        payload = load_json(path)
        for key, expected in expected_json.items():
            if payload.get(key) != expected:
                raise IsolatedSupervisorError(
                    f"postcondition JSON field drift: {path}:{key}"
                )
        result["json_equals_verified"] = dict(expected_json)
    return result


def validate_watch_request(request: Mapping[str, Any]) -> None:
    if request.get("format") != WATCH_REQUEST_FORMAT:
        raise IsolatedSupervisorError("watch request format drift")
    cancer = str(request.get("cancer_id", ""))
    if cancer not in REMAINING_EXECUTION_ORDER:
        raise IsolatedSupervisorError(f"unexpected remaining cancer: {cancer}")
    if int(request.get("memory_limit_bytes", -1)) != MEMORY_LIMIT_BYTES:
        raise IsolatedSupervisorError("watch request changed the 512 MiB limit")
    jobs = request.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise IsolatedSupervisorError("watch request has no jobs")
    for job in jobs:
        if not isinstance(job, dict):
            raise IsolatedSupervisorError("watch job is not an object")
        stage = str(job.get("stage", ""))
        if stage not in {"PLAN", "RUN", "AUDIT"}:
            raise IsolatedSupervisorError(f"unsupported watch stage: {stage}")
        argv = job.get("argv")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(value, str) and value for value in argv
        ):
            raise IsolatedSupervisorError(f"invalid argv for {cancer} {stage}")
        log_path = Path(str(job.get("log_path", "")))
        if not log_path.is_absolute():
            raise IsolatedSupervisorError(f"non-absolute log for {cancer} {stage}")
        environment = job.get("environment", {})
        if not isinstance(environment, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        ):
            raise IsolatedSupervisorError("job environment is not string-to-string")
        if set(environment) - ALLOWED_JOB_ENVIRONMENT_KEYS:
            raise IsolatedSupervisorError("job environment contains forbidden keys")
        observation = job.get("memory_observation")
        if not isinstance(observation, dict):
            raise IsolatedSupervisorError("job memory observation contract is absent")
        validate_memory_observation_contract(observation)
        postconditions = job.get("postconditions", [])
        if not isinstance(postconditions, list) or any(
            not isinstance(row, dict) for row in postconditions
        ):
            raise IsolatedSupervisorError("postconditions must be objects")


def execute_watch_request(
    request: Mapping[str, Any], *, receipt_path: Path
) -> tuple[int, dict[str, Any]]:
    assert_stdlib_control_plane()
    validate_watch_request(request)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise IsolatedSupervisorError(f"watch receipt reuse forbidden: {receipt_path}")
    cancer = str(request["cancer_id"])
    started = time.time()
    watchdog_start_time_ticks = current_process_start_time_ticks()
    records: list[dict[str, Any]] = []
    failed_stage: str | None = None
    for job in request["jobs"]:
        stage = str(job["stage"])
        pre_absent = [Path(str(value)) for value in job.get("must_be_absent_before", [])]
        for path in pre_absent:
            if path.exists() or path.is_symlink():
                raise IsolatedSupervisorError(
                    f"{cancer} {stage} would reuse pre-existing path: {path}"
                )
        monitored = run_monitored(
            job["argv"],
            log_path=Path(str(job["log_path"])),
            environment_additions=job.get("environment", {}),
            memory_observation=job["memory_observation"],
            memory_limit_bytes=MEMORY_LIMIT_BYTES,
            poll_seconds=float(request.get("poll_seconds", POLL_SECONDS)),
        )
        postconditions: list[dict[str, Any]] = []
        if monitored["stop_reason"] is None and monitored["return_code"] == 0:
            try:
                postconditions = [
                    verify_postcondition(row) for row in job.get("postconditions", [])
                ]
            except IsolatedSupervisorError as exc:
                monitored["postcondition_error"] = str(exc)
        succeeded = (
            monitored["stop_reason"] is None
            and monitored["return_code"] == 0
            and "postcondition_error" not in monitored
        )
        records.append(
            {
                "stage": stage,
                "status": "SUCCESS" if succeeded else "FAILED",
                "process": monitored,
                "postconditions": postconditions,
            }
        )
        if not succeeded:
            failed_stage = stage
            break
    payload: dict[str, Any] = {
        "format": WATCH_RECEIPT_FORMAT,
        "status": "SUCCESS" if failed_stage is None else "FAILED_STOPPED",
        "cancer_id": cancer,
        "watchdog_pid": os.getpid(),
        "watchdog_start_time_ticks": watchdog_start_time_ticks,
        "watchdog_process_identity": {
            "pid": os.getpid(),
            "start_time_ticks": watchdog_start_time_ticks,
        },
        "watchdog_hostname": socket.gethostname(),
        "watchdog_scientific_modules_loaded_at_start": [],
        "watchdog_short_lifecycle_per_cancer": True,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "memory_policy": (
            "EXCLUSIVE_CGROUP_V2_CURRENT_AND_PEAK_FIRST;"
            "PROVED_SINGLE_MEMBER_PROC_FALLBACK_ONLY"
        ),
        "cgroup_v2_aggregate_peak_preferred": True,
        "between_poll_multi_helper_peak_must_not_be_sampling_only": True,
        "watchdog_memory_included_in_formal_gate": False,
        "member_hwm_values_summed": False,
        "exited_member_hwm_retained_in_gate": False,
        "jobs": records,
        "failed_stage": failed_stage,
        "elapsed_seconds": round(time.time() - started, 3),
        "production_deployed": False,
        "port_8260_touched": False,
        "timestamp_unix": time.time(),
    }
    payload["receipt_contract_sha256"] = canonical_sha256(payload)
    exclusive_json(receipt_path, payload)
    return (0 if failed_stage is None else 2), payload


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _require_authorized(path: Path) -> None:
    resolved = path.resolve()
    root = AUTHORIZED_ROOT.resolve()
    if resolved == root or not _is_within(resolved, root):
        raise IsolatedSupervisorError(f"path leaves authorized dell_2 tree: {path}")


def verify_immutable_bindings() -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for role, binding in REQUIRED_IMMUTABLE_BINDINGS.items():
        path = Path(binding["path"])
        digest = sha256_file(path)
        if digest != binding["sha256"]:
            raise IsolatedSupervisorError(f"immutable binding SHA drift: {role}")
        observed[role] = {
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    acc = load_json(Path(REQUIRED_IMMUTABLE_BINDINGS["R9_ACC_SUCCESS"]["path"]))
    if acc.get("status") != "SUCCESS" or acc.get("cancer_id") != "ACC":
        raise IsolatedSupervisorError("immutable ACC SUCCESS semantics drift")
    failure = load_json(
        Path(REQUIRED_IMMUTABLE_BINDINGS["R9_TYPED_FAILURE"]["path"])
    )
    if (
        failure.get("status") != "TYPED_FAILURE_STOPPED"
        or failure.get("cancer_id") != "UCEC"
        or failure.get("reason") != "OBSERVED_PROCESS_MEMORY_EXCEEDED_512_MIB"
    ):
        raise IsolatedSupervisorError("immutable R9 typed failure semantics drift")
    plan = load_json(Path(REQUIRED_IMMUTABLE_BINDINGS["R9_UCEC_PLAN"]["path"]))
    if (
        plan.get("cancer_id") != "UCEC"
        or plan.get("raw_h5_full_sha256_verified") is not True
        or plan.get("safe_to_start_pilot") is not True
        or int(plan.get("resource_estimate", {}).get("estimated_peak_ram_bytes", -1))
        != 495_494_384
        or int(plan.get("pathways_total", -1)) != 2_135
    ):
        raise IsolatedSupervisorError("immutable UCEC plan semantics drift")
    return observed


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def verify_launch_code_and_memory_bindings(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    python = spec.get("python_identity")
    if not isinstance(python, dict):
        raise IsolatedSupervisorError("launch Python identity is absent")
    python_path = Path(str(python.get("path", "")))
    expected_resolved = Path(str(python.get("resolved_path", "")))
    if not python_path.is_absolute() or not expected_resolved.is_absolute():
        raise IsolatedSupervisorError("launch Python paths are not absolute")
    observed_resolved = python_path.resolve(strict=True)
    if observed_resolved != expected_resolved.resolve(strict=True):
        raise IsolatedSupervisorError("launch Python resolved path drift")
    if not _is_sha256(python.get("sha256")) or sha256_file(observed_resolved) != python[
        "sha256"
    ]:
        raise IsolatedSupervisorError("launch Python SHA drift")

    rows = spec.get("code_bindings")
    if not isinstance(rows, list) or not rows:
        raise IsolatedSupervisorError("launch code bindings are absent")
    bindings: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise IsolatedSupervisorError("launch code binding is not an object")
        role = str(row.get("role", ""))
        path = Path(str(row.get("path", "")))
        digest = row.get("sha256")
        if not role or role in bindings or not path.is_absolute() or not _is_sha256(
            digest
        ):
            raise IsolatedSupervisorError("launch code binding identity is invalid")
        if path.is_symlink() or not path.is_file():
            raise IsolatedSupervisorError(f"launch code is absent/unsafe: {role}")
        observed_sha = sha256_file(path)
        observed_bytes = path.stat().st_size
        if observed_sha != digest:
            raise IsolatedSupervisorError(f"launch code SHA drift: {role}")
        if row.get("bytes") is not None and int(row["bytes"]) != observed_bytes:
            raise IsolatedSupervisorError(f"launch code byte count drift: {role}")
        bindings[role] = {
            "role": role,
            "path": str(path),
            "sha256": observed_sha,
            "bytes": observed_bytes,
        }

    cgroup_preflights: list[dict[str, Any]] = []
    fallback_proofs: list[dict[str, Any]] = []
    cgroup_names: set[tuple[str, str]] = set()
    for cancer_row in spec["cancers"]:
        cancer = str(cancer_row["cancer_id"])
        for job in cancer_row["jobs"]:
            stage = str(job["stage"])
            argv = job["argv"]
            role = str(job.get("code_role", ""))
            if role not in bindings:
                raise IsolatedSupervisorError(
                    f"{cancer} {stage} refers to an unbound code role"
                )
            if len(argv) < 2 or Path(argv[0]) != python_path or Path(argv[1]) != Path(
                bindings[role]["path"]
            ):
                raise IsolatedSupervisorError(
                    f"{cancer} {stage} argv is not bound to Python/code identity"
                )
            if stage in {"PLAN", "RUN"}:
                if "--cancer-id" not in argv:
                    raise IsolatedSupervisorError(f"{cancer} {stage} lacks --cancer-id")
                index = argv.index("--cancer-id")
                if index + 1 >= len(argv) or argv[index + 1] != cancer:
                    raise IsolatedSupervisorError(
                        f"{cancer} {stage} cancer argument drift"
                    )
            observation = job["memory_observation"]
            mode = validate_memory_observation_contract(observation)
            if mode == "CGROUP_V2_EXCLUSIVE":
                key = (
                    str(Path(str(observation["delegated_parent"])).resolve()),
                    str(observation["job_cgroup_name"]),
                )
                if key in cgroup_names:
                    raise IsolatedSupervisorError("exclusive job cgroup name was reused")
                cgroup_names.add(key)
                value = preflight_cgroup_v2_observation(observation)
                value.update({"cancer_id": cancer, "stage": stage})
                cgroup_preflights.append(value)
            else:
                proof = observation["single_member_proof"]
                runner_path = Path(str(proof["runner_path"]))
                proof_path = Path(str(proof["proof_receipt_path"]))
                if (
                    runner_path != Path(argv[1])
                    or sha256_file(runner_path) != proof["runner_sha256"]
                    or proof["runner_sha256"] != bindings[role]["sha256"]
                    or sha256_file(proof_path) != proof["proof_receipt_sha256"]
                ):
                    raise IsolatedSupervisorError(
                        f"{cancer} {stage} single-member proof binding drift"
                    )
                proof_receipt = load_json(proof_path)
                if (
                    proof_receipt.get("status")
                    != "PASS_FROZEN_RUNNER_SINGLE_PROCESS_ONLY"
                    or proof_receipt.get("runner_sha256") != proof["runner_sha256"]
                    or proof_receipt.get("helper_process_spawn_api_found") is not False
                    or proof_receipt.get("exec_replacement_without_overlap") is not True
                ):
                    raise IsolatedSupervisorError(
                        f"{cancer} {stage} fallback proof semantics drift"
                    )
                fallback_proofs.append(
                    {
                        "cancer_id": cancer,
                        "stage": stage,
                        "runner_path": str(runner_path),
                        "runner_sha256": proof["runner_sha256"],
                        "proof_receipt_path": str(proof_path),
                        "proof_receipt_sha256": proof["proof_receipt_sha256"],
                    }
                )
    return {
        "python_identity": {
            "path": str(python_path),
            "resolved_path": str(observed_resolved),
            "sha256": python["sha256"],
        },
        "code_bindings": [bindings[role] for role in sorted(bindings)],
        "cgroup_v2_job_preflights": cgroup_preflights,
        "single_member_fallback_proofs": fallback_proofs,
        "cgroup_v2_preferred": True,
        "proc_fallback_requires_no_helper_overlap_proof": True,
    }


def _validate_postcondition_path(path: Path, output_root: Path, audit_root: Path) -> None:
    resolved = path.resolve()
    if not (_is_within(resolved, output_root) or _is_within(resolved, audit_root)):
        raise IsolatedSupervisorError(
            f"postcondition leaves fresh output/audit roots: {path}"
        )


def validate_launch_spec(spec: Mapping[str, Any]) -> tuple[Path, Path]:
    if spec.get("format") != LAUNCH_SPEC_FORMAT:
        raise IsolatedSupervisorError("launch spec format drift")
    if spec.get("status") != "READY_FOR_EXPLICIT_ROOT_CONFIRMATION":
        raise IsolatedSupervisorError("launch spec is not in review-ready state")
    if int(spec.get("memory_limit_bytes", -1)) != MEMORY_LIMIT_BYTES:
        raise IsolatedSupervisorError("launch spec changed the 512 MiB gate")
    if tuple(spec.get("cancer_order", [])) != REMAINING_EXECUTION_ORDER:
        raise IsolatedSupervisorError("launch spec remaining-cancer order drift")
    output_root = Path(str(spec.get("output_root", ""))).resolve()
    audit_root = Path(str(spec.get("audit_root", ""))).resolve()
    _require_authorized(output_root)
    _require_authorized(audit_root)
    if output_root == audit_root or _is_within(output_root, audit_root) or _is_within(
        audit_root, output_root
    ):
        raise IsolatedSupervisorError("fresh output and audit roots overlap")
    if output_root.exists() or output_root.is_symlink():
        raise IsolatedSupervisorError(f"fresh output root already exists: {output_root}")
    if audit_root.exists() or audit_root.is_symlink():
        raise IsolatedSupervisorError(f"fresh audit root already exists: {audit_root}")
    cancers = spec.get("cancers")
    if not isinstance(cancers, list) or tuple(
        str(row.get("cancer_id", "")) for row in cancers if isinstance(row, dict)
    ) != REMAINING_EXECUTION_ORDER or len(cancers) != len(REMAINING_EXECUTION_ORDER):
        raise IsolatedSupervisorError("launch spec cancer request set/order drift")
    for row in cancers:
        if not isinstance(row, dict):
            raise IsolatedSupervisorError("launch cancer request is not an object")
        request = {
            "format": WATCH_REQUEST_FORMAT,
            "cancer_id": row.get("cancer_id"),
            "memory_limit_bytes": spec.get("memory_limit_bytes"),
            "poll_seconds": spec.get("poll_seconds", POLL_SECONDS),
            "jobs": row.get("jobs"),
        }
        validate_watch_request(request)
        for job in request["jobs"]:
            log_path = Path(str(job["log_path"])).resolve()
            if not _is_within(log_path, audit_root):
                raise IsolatedSupervisorError("job log leaves fresh audit root")
            for postcondition in job.get("postconditions", []):
                _validate_postcondition_path(
                    Path(str(postcondition.get("path", ""))),
                    output_root,
                    audit_root,
                )
            for raw in job.get("must_be_absent_before", []):
                _validate_postcondition_path(Path(str(raw)), output_root, audit_root)
    generation = str(spec.get("runner_generation", ""))
    if generation not in {"R9", "R10"}:
        raise IsolatedSupervisorError("runner generation must be R9 or R10")
    if generation == "R10":
        for row in cancers:
            for job in row["jobs"]:
                if job["stage"] != "RUN":
                    continue
                argv = job["argv"]
                if "--association-scratch-root" not in argv:
                    raise IsolatedSupervisorError(
                        "R10 RUN lacks explicit association scratch root"
                    )
                index = argv.index("--association-scratch-root")
                if index + 1 >= len(argv):
                    raise IsolatedSupervisorError("R10 scratch argument has no value")
                scratch = Path(argv[index + 1])
                if scratch.exists() or scratch.is_symlink():
                    raise IsolatedSupervisorError(
                        f"R10 exact scratch root already exists: {scratch}"
                    )
    return output_root, audit_root


def _active_processes(markers: Iterable[str]) -> list[dict[str, Any]]:
    marker_values = tuple(value.encode("utf-8") for value in markers if value)
    if not marker_values:
        return []
    found: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if any(marker in raw for marker in marker_values):
            found.append(
                {
                    "pid": int(entry.name),
                    "cmdline": raw.replace(b"\x00", b" ").decode(
                        "utf-8", errors="replace"
                    )[:1000],
                }
            )
    return sorted(found, key=lambda row: row["pid"])


def preflight_launch_spec(
    spec_path: Path,
    *,
    expected_spec_sha256: str,
    expected_supervisor_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert_stdlib_control_plane()
    self_path = Path(__file__).resolve()
    if sha256_file(self_path) != expected_supervisor_sha256:
        raise IsolatedSupervisorError("isolated supervisor SHA drift")
    if sha256_file(spec_path) != expected_spec_sha256:
        raise IsolatedSupervisorError("launch spec SHA drift")
    spec = load_json(spec_path)
    output_root, audit_root = validate_launch_spec(spec)
    bindings = verify_immutable_bindings()
    launch_bindings = verify_launch_code_and_memory_bindings(spec)
    active = _active_processes(spec.get("forbidden_active_cmdline_markers", []))
    if active:
        raise IsolatedSupervisorError(
            "an old or competing formal runner is active: " + json.dumps(active)
        )
    receipt = {
        "format": FORMAT,
        "status": "PASS_READY_BUT_NOT_STARTED",
        "supervisor_path": str(self_path),
        "supervisor_sha256": expected_supervisor_sha256,
        "launch_spec_path": str(spec_path.resolve()),
        "launch_spec_sha256": expected_spec_sha256,
        "runner_generation": spec["runner_generation"],
        "output_root": str(output_root),
        "audit_root": str(audit_root),
        "cancer_order": list(REMAINING_EXECUTION_ORDER),
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "memory_policy": (
            "EXCLUSIVE_CGROUP_V2_CURRENT_AND_PEAK_FIRST;"
            "PROVED_SINGLE_MEMBER_PROC_FALLBACK_ONLY"
        ),
        "cgroup_v2_aggregate_peak_preferred": True,
        "watchdog_memory_included_in_formal_gate": False,
        "member_hwm_values_summed": False,
        "immutable_bindings": bindings,
        "launch_code_and_memory_bindings": launch_bindings,
        "active_competing_processes": [],
        "explicit_root_confirmation_still_required": True,
        "formal_task_started": False,
        "production_deployed": False,
        "port_8260_touched": False,
    }
    receipt["preflight_contract_sha256"] = canonical_sha256(receipt)
    return spec, receipt


def _build_watch_request(spec: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    request = {
        "format": WATCH_REQUEST_FORMAT,
        "cancer_id": row["cancer_id"],
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "poll_seconds": float(spec.get("poll_seconds", POLL_SECONDS)),
        "jobs": row["jobs"],
    }
    request["request_contract_sha256"] = canonical_sha256(request)
    return request


def execute_cohort(
    spec: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    expected_supervisor_sha256: str,
    root_confirmation_token: str,
) -> int:
    if root_confirmation_token != ROOT_CONFIRMATION_TOKEN:
        raise IsolatedSupervisorError(
            "formal start requires the exact post-review root confirmation token"
        )
    output_root = Path(str(spec["output_root"])).resolve()
    audit_root = Path(str(spec["audit_root"])).resolve()
    # Re-run all freshness and immutable checks immediately before mutation.
    validate_launch_spec(spec)
    verify_immutable_bindings()
    output_root.mkdir(parents=True, exist_ok=False)
    audit_root.mkdir(parents=True, exist_ok=False)
    running = {
        **dict(preflight),
        "status": "RUNNING_ISOLATED_PER_CANCER",
        "coordinator_pid": os.getpid(),
        "coordinator_hostname": socket.gethostname(),
        "explicit_root_confirmation_received": True,
        "formal_task_started": True,
        "timestamp_unix": time.time(),
    }
    running.pop("preflight_contract_sha256", None)
    running["running_contract_sha256"] = canonical_sha256(running)
    exclusive_json(audit_root / "COHORT_RUNNING.json", running)
    requests_root = audit_root / "watch_requests"
    receipts_root = audit_root / "watch_receipts"
    requests_root.mkdir()
    receipts_root.mkdir()
    completed: list[dict[str, Any]] = []
    script = Path(__file__).resolve()
    for ordinal, row in enumerate(spec["cancers"], start=1):
        cancer = str(row["cancer_id"])
        request = _build_watch_request(spec, row)
        request_path = requests_root / f"{ordinal:02d}_{cancer}.json"
        receipt_path = receipts_root / f"{ordinal:02d}_{cancer}.json"
        exclusive_json(request_path, request)
        request_sha = sha256_file(request_path)
        command = [
            sys.executable,
            str(script),
            "watch-cancer",
            "--request-json",
            str(request_path),
            "--expected-request-sha256",
            request_sha,
            "--receipt-json",
            str(receipt_path),
            "--expected-supervisor-sha256",
            expected_supervisor_sha256,
        ]
        watchdog = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            shell=False,
        )
        if not receipt_path.is_file() or receipt_path.is_symlink():
            receipt = None
        else:
            receipt = load_json(receipt_path)
        if (
            watchdog.returncode != 0
            or receipt is None
            or receipt.get("status") != "SUCCESS"
            or receipt.get("cancer_id") != cancer
        ):
            failure: dict[str, Any] = {
                "format": FAILURE_FORMAT,
                "status": "TYPED_FAILURE_STOPPED",
                "cancer_id": cancer,
                "reason": "PER_CANCER_WATCHDOG_FAILED",
                "watchdog_return_code": int(watchdog.returncode),
                "watch_request_path": str(request_path),
                "watch_request_sha256": request_sha,
                "watch_receipt_path": str(receipt_path),
                "watch_receipt_sha256": sha256_file(receipt_path)
                if receipt is not None
                else None,
                "watch_receipt": receipt,
                "completed_before_failure": [item["cancer_id"] for item in completed],
                "subsequent_cancers_not_started": True,
                "memory_limit_bytes": MEMORY_LIMIT_BYTES,
                "old_r9_typed_failure_preserved": verify_immutable_bindings()[
                    "R9_TYPED_FAILURE"
                ],
                "production_deployed": False,
                "port_8260_touched": False,
                "timestamp_unix": time.time(),
            }
            failure["failure_contract_sha256"] = canonical_sha256(failure)
            exclusive_json(audit_root / "TYPED_FAILURE.json", failure)
            return 2
        completed.append(
            {
                "cancer_id": cancer,
                "watchdog_pid": int(receipt["watchdog_pid"]),
                "watchdog_start_time_ticks": int(
                    receipt["watchdog_start_time_ticks"]
                ),
                "watch_request_sha256": request_sha,
                "watch_receipt_path": str(receipt_path),
                "watch_receipt_sha256": sha256_file(receipt_path),
            }
        )
        # The old R9 terminal receipt is immutable and is re-hashed after each
        # cancer so the recovery run cannot silently replace its provenance.
        verify_immutable_bindings()
        status = {
            "format": FORMAT,
            "status": "RUNNING_ISOLATED_PER_CANCER",
            "completed": completed,
            "remaining": list(REMAINING_EXECUTION_ORDER[len(completed) :]),
            "timestamp_unix": time.time(),
        }
        status["status_contract_sha256"] = canonical_sha256(status)
        atomic_json(audit_root / "COHORT_STATUS.json", status)
    watchdog_identities = [
        (row["watchdog_pid"], row["watchdog_start_time_ticks"])
        for row in completed
    ]
    if len(set(watchdog_identities)) != len(watchdog_identities):
        raise IsolatedSupervisorError(
            "the same watchdog process identity served more than one cancer"
        )
    success: dict[str, Any] = {
        "format": SUCCESS_FORMAT,
        "status": "SUPERVISION_COMPLETE_PENDING_SCIENTIFIC_COHORT_BINDING",
        "runner_generation": spec["runner_generation"],
        "cancer_order": list(REMAINING_EXECUTION_ORDER),
        "per_cancer_watchdogs": completed,
        "distinct_watchdog_process_identity_per_cancer": True,
        "coordinator_imported_scientific_stack": False,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "memory_policy": (
            "EXCLUSIVE_CGROUP_V2_CURRENT_AND_PEAK_FIRST;"
            "PROVED_SINGLE_MEMBER_PROC_FALLBACK_ONLY"
        ),
        "cgroup_v2_aggregate_peak_preferred": True,
        "watchdog_memory_included_in_formal_gate": False,
        "member_hwm_values_summed": False,
        "old_r9_typed_failure_preserved": verify_immutable_bindings()[
            "R9_TYPED_FAILURE"
        ],
        "final_17_cancer_binding_published": False,
        "separate_scientific_audit_required": True,
        "production_deployed": False,
        "port_8260_touched": False,
        "timestamp_unix": time.time(),
    }
    success["success_contract_sha256"] = canonical_sha256(success)
    exclusive_json(audit_root / "ISOLATED_SUPERVISOR_SUCCESS.json", success)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prove = subparsers.add_parser("prove-stdlib")
    prove.add_argument("--output-json", type=Path)

    watch = subparsers.add_parser("watch-cancer")
    watch.add_argument("--request-json", required=True, type=Path)
    watch.add_argument("--expected-request-sha256", required=True)
    watch.add_argument("--receipt-json", required=True, type=Path)
    watch.add_argument("--expected-supervisor-sha256", required=True)

    cohort = subparsers.add_parser("cohort")
    cohort.add_argument("--launch-spec-json", required=True, type=Path)
    cohort.add_argument("--expected-launch-spec-sha256", required=True)
    cohort.add_argument("--expected-supervisor-sha256", required=True)
    cohort.add_argument("--preflight-only", action="store_true")
    cohort.add_argument("--root-confirmation-token")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    assert_stdlib_control_plane()
    if args.command == "prove-stdlib":
        payload = {
            "format": FORMAT,
            "status": "PASS_STDLIB_ONLY_CONTROL_PROCESS",
            "process_id": os.getpid(),
            "scientific_modules_loaded": loaded_scientific_modules(),
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        }
        if args.output_json is not None:
            exclusive_json(args.output_json.resolve(), payload)
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        return 0
    if args.command == "watch-cancer":
        self_path = Path(__file__).resolve()
        if sha256_file(self_path) != args.expected_supervisor_sha256:
            raise IsolatedSupervisorError("isolated supervisor SHA drift in watchdog")
        if sha256_file(args.request_json) != args.expected_request_sha256:
            raise IsolatedSupervisorError("watch request SHA drift")
        request = load_json(args.request_json)
        observed_contract = request.pop("request_contract_sha256", None)
        if observed_contract != canonical_sha256(request):
            raise IsolatedSupervisorError("watch request contract SHA drift")
        return execute_watch_request(request, receipt_path=args.receipt_json.resolve())[0]
    spec, preflight = preflight_launch_spec(
        args.launch_spec_json.resolve(),
        expected_spec_sha256=args.expected_launch_spec_sha256,
        expected_supervisor_sha256=args.expected_supervisor_sha256,
    )
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return execute_cohort(
        spec,
        preflight,
        expected_supervisor_sha256=args.expected_supervisor_sha256,
        root_confirmation_token=str(args.root_confirmation_token or ""),
    )


if __name__ == "__main__":
    raise SystemExit(main())
