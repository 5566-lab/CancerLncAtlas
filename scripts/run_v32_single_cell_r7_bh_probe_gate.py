#!/usr/bin/env python3
"""Enforce the 512 MiB process gate around the r2 BH large-spool probe."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


MEMORY_LIMIT_BYTES = 512 * 1024**2
POLL_SECONDS = 0.25
PYTHON = Path("${PRIVATE_WORK_ROOT}/miniconda3/bin/python")
CANDIDATE_ROOT = Path(
    "./data/CancerLncAtlas/runtime/tools/"
    "single_cell_r7_streaming_bh_candidate_20260829_r3"
)
PROBE = CANDIDATE_ROOT / "scripts/smoke_v32_single_cell_r7_bh_compartment_remote.py"
PROBE_RESULT = Path(
    "./data/CancerLncAtlas/runtime/audits/"
    "single_cell_r7_bh_compartment_probe_20260829_r3/RESULT.json"
)
AUDIT_ROOT = Path(
    "./data/CancerLncAtlas/runtime/audits/"
    "single_cell_r7_bh_compartment_probe_gate_20260829_r3"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def memory_bytes(pid: int) -> tuple[int, int]:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return 0, 0
    rss = 0
    hwm = 0
    for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            hwm = int(line.split()[1]) * 1024
    return rss, hwm


def terminate_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def exclusive_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    assert not AUDIT_ROOT.exists() and not AUDIT_ROOT.is_symlink()
    assert not PROBE_RESULT.exists() and not PROBE_RESULT.is_symlink()
    AUDIT_ROOT.mkdir(parents=True)
    log_path = AUDIT_ROOT / "probe.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        "./data/CancerLncAtlas/runtime/tools/"
        "single_cell_r7_streaming_20260829_r7"
    )
    maximum_rss = 0
    maximum_hwm = 0
    memory_exceeded = False
    started = time.time()
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            [str(PYTHON), str(PROBE)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        while process.poll() is None:
            rss, hwm = memory_bytes(process.pid)
            maximum_rss = max(maximum_rss, rss)
            maximum_hwm = max(maximum_hwm, hwm)
            if max(rss, hwm) > MEMORY_LIMIT_BYTES:
                memory_exceeded = True
                terminate_group(process)
                break
            time.sleep(POLL_SECONDS)
        return_code = int(process.wait())
        log.flush()
        os.fsync(log.fileno())
    probe_result = (
        json.loads(PROBE_RESULT.read_text(encoding="utf-8"))
        if PROBE_RESULT.is_file() and not PROBE_RESULT.is_symlink()
        else None
    )
    passed = (
        return_code == 0
        and not memory_exceeded
        and isinstance(probe_result, dict)
        and probe_result.get("status") == "PASS"
        and probe_result.get("association_duckdb_memory_limit") == "64MB"
    )
    payload = {
        "candidate_runner_sha256": sha256_file(
            CANDIDATE_ROOT / "scripts/run_v32_single_cell_r7_streaming.py"
        ),
        "elapsed_seconds": round(time.time() - started, 3),
        "formal_gate_value_bytes": max(maximum_rss, maximum_hwm),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "maximum_high_water_bytes_observed": maximum_hwm,
        "maximum_rss_bytes_observed": maximum_rss,
        "memory_exceeded": memory_exceeded,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "memory_metric": "MAX_CHILD_PID_VMRSS_OR_VMHWM_NOT_PROCESS_GROUP_SUM",
        "poll_seconds": POLL_SECONDS,
        "probe_result": probe_result,
        "process_group_rss_sum_measured": False,
        "return_code": return_code,
        "status": "PASS" if passed else "FAIL",
    }
    exclusive_json(AUDIT_ROOT / "GATE_RESULT.json", payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
