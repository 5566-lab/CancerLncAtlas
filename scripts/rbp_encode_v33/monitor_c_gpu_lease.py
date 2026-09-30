#!/usr/bin/env python3
"""Monitor one CompShare job with a provider-enforced shutdown lease.

The initial scheduled shutdown must be set and verified before the GPU job is
submitted.  This controller only extends that shutdown while optimizer or
validation progress and GPU telemetry both advance.  If this controller or its network disappears,
the provider's existing schedule remains in force without copying credentials
to host 149 or the GPU instance.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

if __package__:
    from .monitor_c_gpu_149 import log_lines, progress
else:
    from monitor_c_gpu_149 import log_lines, progress


def cli(executable: Path, *args: str, timeout: int = 90) -> dict[str, Any]:
    done = subprocess.run(
        [str(executable), "--json", *args],
        capture_output=True, text=True, timeout=timeout,
    )
    if done.returncode:
        raise RuntimeError(f"CompShare CLI failed ({done.returncode})")
    try:
        result = json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("CompShare CLI did not return JSON") from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError("CompShare CLI returned an unsuccessful response")
    return result


def schedule_stop_time(result: dict[str, Any], instance_id: str) -> int:
    data = result.get("data")
    if (not isinstance(data, dict) or data.get("instance") != instance_id
            or data.get("scheduled") is not True
            or type(data.get("scheduler_stop_time")) is not int):
        raise RuntimeError("Provider shutdown schedule is missing or has the wrong instance")
    return data["scheduler_stop_time"]


def current_schedule(executable: Path, instance_id: str) -> int:
    return schedule_stop_time(
        cli(executable, "instance", "schedule", "show", instance_id), instance_id,
    )


def extend_lease(executable: Path, instance_id: str, *, now: int,
                 hard_deadline: int, renew_seconds: int, margin_seconds: int) -> int:
    prior = current_schedule(executable, instance_id)
    if prior <= now + 300:
        raise RuntimeError("Shutdown lease is too close to expiry to renew safely")
    if prior > hard_deadline:
        raise RuntimeError("Shutdown schedule exceeds the authorized hard deadline")
    if prior - now > margin_seconds:
        return prior
    extension = min(renew_seconds, hard_deadline - prior)
    if extension < 300:
        return prior
    cli(executable, "instance", "schedule", "extend", instance_id,
        "--by", f"{extension}s")
    observed = current_schedule(executable, instance_id)
    if observed != prior + extension or observed > hard_deadline:
        raise RuntimeError("CompShare shutdown extension did not read back exactly")
    return observed


def job_succeeded(lines: list[str]) -> bool:
    for line in lines:
        try:
            row = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(row, dict) and row.get("status") == "TRAINING_SUCCEEDED":
            return True
    return False


def validation_marker(lines: list[str]) -> tuple[int, int, int] | None:
    marker = None
    for line in lines:
        try:
            row = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(row, dict) and row.get("status") == "VALIDATION_HEARTBEAT":
            if all(key in row for key in ("cycle", "runtime_chunk", "completed_batches")):
                marker = (int(row["cycle"]), int(row["runtime_chunk"]),
                          int(row["completed_batches"]))
    return marker


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stop_one(executable: Path, instance_id: str, reason: str, receipt: Path) -> None:
    write_receipt(receipt, {"instance_id": instance_id, "reason": reason,
                            "stop_requested_unix": time.time(), "stopped": False})
    errors = []
    for attempt in range(3):
        try:
            cli(executable, "instance", "stop", instance_id,
                "--yes", "--wait", "--timeout", "180", timeout=210)
            observed = cli(executable, "instance", "show", instance_id, "--status")
            hosts = observed.get("data", {}).get("UHostSet", [])
            if (len(hosts) == 1 and hosts[0].get("UHostId") == instance_id
                    and str(hosts[0].get("State", "")).lower() == "stopped"):
                write_receipt(receipt, {"instance_id": instance_id, "reason": reason,
                                        "stopped": True, "verified_unix": time.time()})
                return
            errors.append("CompShare did not report this instance as Stopped")
        except Exception as exc:
            errors.append(str(exc))
        if attempt < 2:
            time.sleep(10)
    write_receipt(receipt, {"instance_id": instance_id, "reason": reason,
                            "stopped": False, "errors": errors})
    raise RuntimeError("Could not confirm CompShare stop; provider schedule remains armed")


def run(args: argparse.Namespace) -> int:
    if socket.gethostname() == "149":
        raise RuntimeError("Lease controller requires the authorized local CLI profile")
    executable = args.cli.resolve(strict=True)
    if not executable.is_file():
        raise RuntimeError("CompShare CLI executable is missing")
    if args.receipt.exists():
        raise FileExistsError("Lease monitor receipt already exists; duplicate monitor refused")
    if not 0 < args.hard_hours <= 96:
        raise RuntimeError("Hard deadline must be within 96 hours")
    if not 10 <= args.first_step_minutes <= 90 or not 10 <= args.stall_minutes <= 90:
        raise RuntimeError("Progress windows must be 10..90 minutes")
    if (not 10 <= args.poll_seconds <= 120
            or not 10 <= args.initial_lease_minutes <= 120
            or not 10 <= args.renew_minutes <= 60
            or not 5 <= args.renew_margin_minutes < args.renew_minutes):
        raise RuntimeError("Shutdown lease or polling interval is outside the safe range")
    hard_deadline = int(args.instance_created_unix + args.hard_hours * 3600)
    now = int(time.time())
    scheduled = current_schedule(executable, args.instance_id)
    if (scheduled <= now + 300 or scheduled > hard_deadline
            or scheduled > now + args.initial_lease_minutes * 60):
        raise RuntimeError("Initial provider shutdown schedule is absent or outside the approved window")
    if args.initial_optimizer_step < 0 or args.initial_gpu_telemetry_sequence < 0:
        raise RuntimeError("Initial progress evidence must be non-negative")
    started = last_work = last_gpu = time.monotonic()
    seen_step = args.initial_optimizer_step
    seen_gpu = args.initial_gpu_telemetry_sequence
    seen_validation = None
    errors = 0
    while True:
        now_mono = time.monotonic()
        reason = None
        if time.time() >= hard_deadline:
            reason = "HARD_DEADLINE"
        elif seen_step == 0 and now_mono - started >= args.first_step_minutes * 60:
            reason = "FIRST_OPTIMIZER_STEP_TIMEOUT"
        elif seen_step > 0 and now_mono - last_work >= args.stall_minutes * 60:
            reason = "TRAINING_OR_VALIDATION_PROGRESS_STALLED"
        elif seen_step > 0 and now_mono - last_gpu >= args.stall_minutes * 60:
            reason = "GPU_TELEMETRY_STALLED"
        if reason:
            stop_one(executable, args.instance_id, reason, args.receipt)
            return 1
        try:
            output = cli(executable, "instance", "job", "logs", args.instance_id,
                         args.job_id, "--tail", "300")
            lines = log_lines(output)
            step, gpu, failed = progress(lines)
            validation = validation_marker(lines)
            job_state = str(output.get("data", {}).get("logs", {}).get("State", ""))
            if failed or job_state in {"Failed", "Cancelled", "Interrupted"}:
                stop_one(executable, args.instance_id, "TRAINING_LAUNCH_FAILED", args.receipt)
                return 1
            if step > seen_step:
                seen_step, last_work = step, now_mono
            if validation is not None and validation != seen_validation:
                seen_validation, last_work = validation, now_mono
            if gpu > seen_gpu:
                seen_gpu, last_gpu = gpu, now_mono
            if job_state == "Succeeded":
                if not job_succeeded(lines) or seen_step == 0 or seen_gpu == 0:
                    stop_one(executable, args.instance_id,
                             "JOB_SUCCEEDED_WITHOUT_TRAINING_EVIDENCE", args.receipt)
                    return 1
                write_receipt(args.receipt, {"instance_id": args.instance_id,
                                            "job_id": args.job_id,
                                            "status": "TRAINING_SUCCEEDED_RESULT_RETURN_PENDING",
                                            "optimizer_step": seen_step,
                                            "gpu_telemetry_sequence": seen_gpu,
                                            "scheduler_stop_time": current_schedule(executable, args.instance_id)})
                return 0
            if seen_step > 0 and seen_gpu > 0:
                scheduled = extend_lease(
                    executable, args.instance_id, now=int(time.time()),
                    hard_deadline=hard_deadline,
                    renew_seconds=args.renew_minutes * 60,
                    margin_seconds=args.renew_margin_minutes * 60,
                )
            else:
                scheduled = current_schedule(executable, args.instance_id)
            errors = 0
            write_receipt(args.receipt, {"instance_id": args.instance_id,
                                        "job_id": args.job_id,
                                        "status": "MONITORING",
                                        "optimizer_step": seen_step,
                                        "gpu_telemetry_sequence": seen_gpu,
                                        "validation_marker": seen_validation,
                                        "scheduler_stop_time": scheduled,
                                        "last_checked_unix": time.time()})
        except Exception as exc:
            errors += 1
            if errors >= args.max_api_errors:
                stop_one(executable, args.instance_id,
                         f"MONITOR_API_FAILURE: {exc}", args.receipt)
                return 1
        time.sleep(args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--instance-created-unix", type=float, required=True)
    parser.add_argument("--hard-hours", type=float, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--first-step-minutes", type=int, default=45)
    parser.add_argument("--stall-minutes", type=int, default=30)
    parser.add_argument("--initial-optimizer-step", type=int, default=0)
    parser.add_argument("--initial-gpu-telemetry-sequence", type=int, default=0)
    parser.add_argument("--initial-lease-minutes", type=int, default=90)
    parser.add_argument("--renew-minutes", type=int, default=30)
    parser.add_argument("--renew-margin-minutes", type=int, default=15)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-api-errors", type=int, default=3)
    args = parser.parse_args()
    if not 3 <= args.max_api_errors <= 30:
        parser.error("--max-api-errors must be between 3 and 30")
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "MONITOR_FAILED", "error": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
