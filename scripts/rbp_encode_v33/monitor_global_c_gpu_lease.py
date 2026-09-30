#!/usr/bin/env python3
"""Guard several final G2 fold jobs on one CompShare instance with one lease."""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rbp_encode_v33.monitor_c_gpu_149 import log_lines, progress
from scripts.rbp_encode_v33.monitor_c_gpu_lease import (
    cli, current_schedule, extend_lease, job_succeeded, stop_one, write_receipt,
)


def validation_marker(lines: list[str]) -> tuple[int, int, int] | None:
    """Return the latest completed validation batch, if one is in the log tail."""
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


def run(args: argparse.Namespace) -> int:
    if socket.gethostname() == "149":
        raise RuntimeError("CompShare lease controller requires the local CLI profile")
    executable = args.cli.resolve(strict=True)
    ids = tuple(args.job_id)
    if not ids or len(ids) > 4 or len(set(ids)) != len(ids):
        raise RuntimeError("Expected one to four distinct fold job IDs")
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    if not 0 < args.hard_hours <= 96:
        raise RuntimeError("Hard deadline must be within 96 hours")
    if not 10 <= args.first_step_minutes <= 90 or not 10 <= args.stall_minutes <= 90:
        raise RuntimeError("Progress windows must be 10..90 minutes")
    if not 10 <= args.poll_seconds <= 120 or not 3 <= args.max_api_errors <= 30:
        raise RuntimeError("Monitor polling/error limits are invalid")
    hard_deadline = int(args.paid_start_unix + args.hard_hours * 3600)
    now = int(time.time())
    scheduled = current_schedule(executable, args.instance_id)
    if (scheduled <= now + 300 or scheduled > hard_deadline
            or scheduled > now + args.initial_lease_minutes * 60):
        raise RuntimeError("Initial provider shutdown schedule is absent or outside the approved window")
    started = time.monotonic()
    states = {job: {"step": 0, "gpu": 0, "last_work": started,
                    "last_gpu": started, "validation": None,
                    "done": False} for job in ids}
    if args.resume_receipt is not None:
        prior = json.loads(args.resume_receipt.read_text(encoding="utf-8"))
        if (prior.get("instance_id") != args.instance_id
                or prior.get("status") != "MONITORING_GLOBAL_G2_JOBS"
                or set(prior.get("jobs", {})) != set(ids)):
            raise RuntimeError("Prior monitor receipt does not match active jobs")
        for job in ids:
            row = prior["jobs"][job]
            states[job]["step"] = int(row["optimizer_step"])
            states[job]["gpu"] = int(row["gpu_telemetry_sequence"])
            states[job]["done"] = bool(row.get("done", False))
            if states[job]["step"] <= 0 or states[job]["gpu"] <= 0:
                raise RuntimeError("Prior monitor receipt lacks training evidence")
    errors = 0
    while True:
        now_mono = time.monotonic()
        reason = None
        if time.time() >= hard_deadline:
            reason = "HARD_DEADLINE"
        for job, state in states.items():
            if state["done"]:
                continue
            if not state["step"] and now_mono - started >= args.first_step_minutes * 60:
                reason = f"FIRST_OPTIMIZER_STEP_TIMEOUT:{job}"
                break
            if state["step"] and now_mono - state["last_work"] >= args.stall_minutes * 60:
                reason = f"TRAINING_OR_VALIDATION_PROGRESS_STALLED:{job}"
                break
            if state["step"] and now_mono - state["last_gpu"] >= args.stall_minutes * 60:
                reason = f"GPU_TELEMETRY_STALLED:{job}"
                break
        if reason:
            stop_one(executable, args.instance_id, reason, args.receipt)
            return 1
        try:
            for job, state in states.items():
                if state["done"]:
                    continue
                output = cli(executable, "instance", "job", "logs", args.instance_id,
                             job, "--tail", "300")
                lines = log_lines(output)
                step, gpu, failed = progress(lines)
                validation = validation_marker(lines)
                job_state = str(output.get("data", {}).get("logs", {}).get("State", ""))
                if failed or job_state in {"Failed", "Cancelled", "Interrupted"}:
                    stop_one(executable, args.instance_id,
                             f"TRAINING_LAUNCH_FAILED:{job}", args.receipt)
                    return 1
                if step > state["step"]:
                    state["step"], state["last_work"] = step, now_mono
                if validation is not None and validation != state["validation"]:
                    state["validation"], state["last_work"] = validation, now_mono
                if gpu > state["gpu"]:
                    state["gpu"], state["last_gpu"] = gpu, now_mono
                if job_state == "Succeeded":
                    if not job_succeeded(lines) or not state["step"] or not state["gpu"]:
                        stop_one(executable, args.instance_id,
                                 f"JOB_SUCCEEDED_WITHOUT_TRAINING_EVIDENCE:{job}", args.receipt)
                        return 1
                    state["done"] = True
            if all(state["done"] for state in states.values()):
                write_receipt(args.receipt, {
                    "status": "ALL_GLOBAL_G2_JOBS_SUCCEEDED_RESULT_RETURN_PENDING",
                    "instance_id": args.instance_id,
                    "jobs": {job: {"optimizer_step": state["step"],
                                   "gpu_telemetry_sequence": state["gpu"]}
                             for job, state in states.items()},
                    "scheduler_stop_time": current_schedule(executable, args.instance_id),
                })
                return 0
            if all(state["done"] or (state["step"] and state["gpu"])
                   for state in states.values()):
                scheduled = extend_lease(
                    executable, args.instance_id, now=int(time.time()),
                    hard_deadline=hard_deadline,
                    renew_seconds=args.renew_minutes * 60,
                    margin_seconds=args.renew_margin_minutes * 60,
                )
            else:
                scheduled = current_schedule(executable, args.instance_id)
            errors = 0
            write_receipt(args.receipt, {
                "status": "MONITORING_GLOBAL_G2_JOBS",
                "instance_id": args.instance_id,
                "jobs": {job: {"optimizer_step": state["step"],
                               "gpu_telemetry_sequence": state["gpu"],
                               "validation_marker": state["validation"],
                               "done": state["done"]} for job, state in states.items()},
                "scheduler_stop_time": scheduled,
                "last_checked_unix": time.time(),
            })
        except Exception as exc:
            errors += 1
            if errors >= args.max_api_errors:
                stop_one(executable, args.instance_id,
                         f"MONITOR_API_FAILURE:{exc}", args.receipt)
                return 1
        time.sleep(args.poll_seconds)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cli", type=Path, required=True)
    p.add_argument("--instance-id", required=True)
    p.add_argument("--job-id", action="append", required=True)
    p.add_argument("--paid-start-unix", type=float, required=True)
    p.add_argument("--hard-hours", type=float, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--resume-receipt", type=Path)
    p.add_argument("--first-step-minutes", type=int, default=45)
    p.add_argument("--stall-minutes", type=int, default=30)
    p.add_argument("--initial-lease-minutes", type=int, default=90)
    p.add_argument("--renew-minutes", type=int, default=30)
    p.add_argument("--renew-margin-minutes", type=int, default=15)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--max-api-errors", type=int, default=3)
    args = p.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "GLOBAL_G2_MONITOR_FAILED", "error": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
