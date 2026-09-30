#!/usr/bin/env python3
"""Independently stop one authorized CompShare job when training makes no progress.

Run detached on host 149 after the platform shutdown schedule has been set.
Only the exact instance and job IDs supplied at launch are ever inspected.
Credentials remain in the protected local CompShare CLI profile on 149.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def cli(*args: str, timeout: int = 90) -> Any:
    env = os.environ.copy()
    deps = Path(__file__).resolve().parent / "monitor_deps"
    if not (deps / "compshare_cli").is_dir():
        raise RuntimeError(f"Pinned CompShare CLI is missing: {deps}")
    env["PYTHONPATH"] = str(deps) + os.pathsep + env.get("PYTHONPATH", "")
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        env.pop(name, None)
    command = [sys.executable, "-m", "compshare_cli", "--json", *args]
    done = subprocess.run(command, text=True, capture_output=True, timeout=timeout, env=env)
    if done.returncode:
        raise RuntimeError(f"CompShare CLI failed ({done.returncode}): {done.stderr[-1000:]}")
    try:
        result = json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("CompShare CLI did not return JSON") from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError(f"CompShare CLI returned an unsuccessful response: {str(result)[:1000]}")
    return result


def log_lines(payload: Any) -> list[str]:
    """Extract text from CLI log envelopes without trusting one SDK layout."""
    if isinstance(payload, str):
        return payload.splitlines()
    if isinstance(payload, list):
        return [line for value in payload for line in log_lines(value)]
    if isinstance(payload, dict):
        return [line for key, value in payload.items()
                if key.lower() in {"stdout", "stderr", "logs", "lines", "content", "text", "data"}
                for line in log_lines(value)]
    return []


def progress(lines: list[str]) -> tuple[int, int, bool]:
    step, telemetry, failed = 0, 0, False
    for line in lines:
        try:
            row = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict):
            continue
        status = row.get("status")
        if status == "OPTIMIZER_STEP":
            step = max(step, int(row.get("optimizer_step", 0)))
        elif status == "GPU_TELEMETRY":
            telemetry = max(telemetry, int(row.get("sequence", 0)))
        elif status in {"TRAINING_FAILED", "LAUNCH_FAILED"}:
            failed = True
    return step, telemetry, failed


def stop_one(instance_id: str, reason: str, receipt: Path) -> None:
    # Record intent before the API call so a crash cannot erase the stop reason.
    receipt.write_text(json.dumps({"instance_id": instance_id, "reason": reason,
                                   "stop_requested_unix": time.time()}, sort_keys=True) + "\n")
    errors = []
    for _ in range(3):
        try:
            cli("instance", "stop", instance_id, "--yes", "--wait", "--timeout", "180", timeout=210)
            observed = cli("instance", "show", instance_id, "--status")
            state_text = json.dumps(observed, ensure_ascii=False).lower()
            if "stopped" in state_text or "已关机" in state_text:
                receipt.write_text(json.dumps({"instance_id": instance_id, "reason": reason,
                                               "stopped": True, "observed": observed},
                                              ensure_ascii=False, sort_keys=True) + "\n")
                return
            errors.append(f"Unconfirmed stop state: {state_text[:500]}")
        except Exception as exc:
            errors.append(str(exc))
        time.sleep(10)
    receipt.write_text(json.dumps({"instance_id": instance_id, "reason": reason,
                                   "stopped": False, "errors": errors}, sort_keys=True) + "\n")
    raise RuntimeError("CompShare stop could not be confirmed; inspect the platform schedule")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance-id", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--first-step-minutes", type=int, default=45)
    p.add_argument("--progress-stall-minutes", type=int, default=30)
    p.add_argument("--hard-hours", type=float, required=True)
    args = p.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("CompShare independent training monitor must run on host 149")
    if args.first_step_minutes not in range(10, 91) or args.progress_stall_minutes not in range(10, 91):
        raise RuntimeError("Monitor windows must be 10..90 minutes")
    if not 0 < args.hard_hours <= 96:
        raise RuntimeError("Monitor hard deadline must be positive and at most 96 hours")
    if args.receipt.exists():
        raise FileExistsError("Monitor receipt already exists; refusing duplicate supervisor")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    start = last_step = last_gpu = time.monotonic()
    seen_step = seen_gpu = 0
    consecutive_errors = 0
    while True:
        now = time.monotonic()
        reason = None
        if now - start >= args.hard_hours * 3600:
            reason = "HARD_DEADLINE"
        elif seen_step == 0 and now - start >= args.first_step_minutes * 60:
            reason = "FIRST_OPTIMIZER_STEP_TIMEOUT"
        elif seen_step > 0 and now - last_step >= args.progress_stall_minutes * 60:
            reason = "OPTIMIZER_PROGRESS_STALLED"
        elif seen_step > 0 and now - last_gpu >= args.progress_stall_minutes * 60:
            reason = "GPU_TELEMETRY_STALLED"
        if reason:
            stop_one(args.instance_id, reason, args.receipt)
            return 1
        try:
            output = cli("instance", "job", "logs", args.instance_id, args.job_id, "--tail", "300")
            step, gpu, failed = progress(log_lines(output))
            job_state = str(output.get("data", {}).get("logs", {}).get("State", ""))
            consecutive_errors = 0
            if failed or job_state in {"Failed", "Cancelled"}:
                stop_one(args.instance_id, "TRAINING_LAUNCH_FAILED", args.receipt)
                return 1
            if step > seen_step:
                seen_step, last_step = step, now
            if gpu > seen_gpu:
                seen_gpu, last_gpu = gpu, now
            args.receipt.write_text(json.dumps({
                "instance_id": args.instance_id, "job_id": args.job_id,
                "optimizer_step": seen_step, "gpu_telemetry_sequence": seen_gpu,
                "last_checked_unix": time.time(), "stopped": False,
            }, sort_keys=True) + "\n")
        except Exception as exc:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                stop_one(args.instance_id, f"MONITOR_API_FAILURE: {exc}", args.receipt)
                return 1
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
