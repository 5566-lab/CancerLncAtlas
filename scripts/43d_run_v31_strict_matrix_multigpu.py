#!/usr/bin/env python3
"""Fail-closed V3.1 matrix runner with one isolated subprocess per GPU."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from cc_hhgt.common import load_config, read_table
from cc_hhgt.formal_gate import validate_formal_training_gate
from cc_hhgt.v30_integrity import atomic_write_json


def _load_shared_runner(repo_root: Path):
    path = repo_root / "scripts" / "43c_run_v29_strict_matrix_parallel.py"
    spec = importlib.util.spec_from_file_location("v31_shared_matrix_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared task audit: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _execution_lanes(devices: list[str], slots_per_device: int) -> list[dict[str, str]]:
    if slots_per_device < 1:
        raise RuntimeError("--slots-per-device must be positive")
    return [
        {"lane_id": f"gpu_{device}_slot_{slot}", "device": device}
        for device in devices
        for slot in range(slots_per_device)
    ]


def _write_status(
    run_control: Path,
    lock: threading.Lock,
    results: list[dict],
    n_tasks: int,
    run_id: str,
    gate_sha256: str,
) -> None:
    with lock:
        snapshot = list(results)
    atomic_write_json(
        run_control / "run_status_parallel.json",
        {
            "status": "RUNNING",
            "mode": f"FULL_{n_tasks}_MULTIGPU",
            "run_id": run_id,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "tasks_done": len(snapshot),
            "n_tasks": n_tasks,
            "completed": sum(row["status"] == "COMPLETED" for row in snapshot),
            "failed": sum(row["status"] == "FAILED" for row in snapshot),
            "formal_training_gate_sha256": gate_sha256,
            "failures": [row for row in snapshot if row["status"] == "FAILED"],
        },
    )


def _run_one(
    shared: dict,
    shared_runner,
    task: dict,
    device: str,
) -> dict:
    output_root = Path(shared["output_root"])
    root = (
        output_root
        / task["model"]
        / task["fold_id"]
        / f"seed_{task['seed']}"
    )
    resume_training = bool(task.get("resume_training", False))
    resume_checkpoint = root / "last_training_state.pt"
    if root.exists() and not resume_training:
        return {
            **task,
            "status": "FAILED",
            "device": device,
            "error": "task output existed before launch without an authorized exact checkpoint resume",
        }
    if resume_training and (
        not root.is_dir()
        or not resume_checkpoint.is_file()
        or (root / "SUCCESS.json").exists()
    ):
        return {
            **task,
            "status": "FAILED",
            "device": device,
            "error": "authorized exact checkpoint resume is missing, invalid, or already complete",
        }
    command = [
        shared["python"],
        "scripts/42_train_v29_strict_multitask.py",
        "--config",
        shared["config"],
        "--run-id",
        shared["run_id"],
        "--input-root",
        shared["input_root"],
        "--input-manifest",
        shared["input_manifest"],
        "--formal-gate",
        shared["formal_gate"],
        "--output-root",
        shared["output_root"],
        "--model",
        task["model"],
        "--fold",
        task["fold_id"],
        "--seed",
        str(task["seed"]),
    ]
    if resume_training:
        command.append("--resume-training")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    started = time.time()
    result = subprocess.run(
        command,
        cwd=shared["repo_root"],
        capture_output=True,
        text=True,
        env=environment,
    )
    duration = time.time() - started
    log_root = output_root / "run_control" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    attempt_id = f"{time.time_ns()}"
    log_path = (
        log_root
        / f"{task['fold_id']}__{task['model']}__seed_{task['seed']}__gpu_{device}__attempt_{attempt_id}.log"
    )
    log_path.write_text(
        "\n".join(
            [
                f"CUDA_VISIBLE_DEVICES={device}",
                f"COMMAND: {subprocess.list2cmdline(command)}",
                f"RETURN_CODE: {result.returncode}",
                f"DURATION_SECONDS: {duration:.3f}",
                "",
                "STDOUT",
                result.stdout,
                "",
                "STDERR",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if result.returncode:
        return {
            **task,
            "status": "FAILED",
            "device": device,
            "duration_seconds": duration,
            "error": result.stderr[-3000:],
        }
    valid, reason = shared_runner.audit_task(
        root, task, shared["gate_sha256"], shared["run_id"]
    )
    if not valid:
        return {
            **task,
            "status": "FAILED",
            "device": device,
            "duration_seconds": duration,
            "error": f"post-task audit: {reason}",
        }
    return {
        **task,
        "status": "COMPLETED",
        "device": device,
        "duration_seconds": duration,
        "quality_gate": reason,
    }


def _partition_resume_tasks(
    tasks: list[dict],
    output_root: Path,
    shared_runner,
    shared: dict,
    *,
    quarantine_incomplete: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Audit completed tasks and recover interrupted task roots fail-closed.

    A task is skipped only when the same post-training audit used by a fresh
    run passes against the current run ID and formal-gate hash.  An interrupted
    task with a task-local exact training checkpoint and no SUCCESS marker is
    resumed in place; the guarded task CLI validates its complete contract.
    Other incomplete or mismatched roots are never deleted.  With explicit
    authorization they are atomically moved below ``run_control/quarantine``
    before being scheduled again; otherwise resume stops and reports the task.
    """

    completed: list[dict] = []
    pending: list[dict] = []
    quarantined: list[dict] = []
    for task in tasks:
        root = (
            output_root
            / task["model"]
            / task["fold_id"]
            / f"seed_{task['seed']}"
        )
        if not root.exists():
            pending.append(task)
            continue
        valid, reason = shared_runner.audit_task(
            root, task, shared["gate_sha256"], shared["run_id"]
        )
        if valid:
            completed.append(
                {
                    **task,
                    "status": "COMPLETED",
                    "resumed": True,
                    "quality_gate": reason,
                }
            )
            continue
        exact_checkpoint = root / "last_training_state.pt"
        if exact_checkpoint.is_file() and not (root / "SUCCESS.json").exists():
            pending.append(
                {
                    **task,
                    "resume_training": True,
                    "resume_checkpoint": str(exact_checkpoint),
                    "pre_resume_audit_failure": reason,
                }
            )
            continue
        if not quarantine_incomplete:
            raise RuntimeError(
                "Resume found an incomplete or lineage-invalid task root; "
                f"rerun with --quarantine-incomplete to preserve and retry it: "
                f"{root}: {reason}"
            )
        quarantine = (
            output_root
            / "run_control"
            / "quarantine"
            / f"{task['fold_id']}__{task['model']}__seed_{task['seed']}__{time.time_ns()}"
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        os.replace(root, quarantine)
        quarantined.append(
            {
                **task,
                "original_root": str(root),
                "quarantine_root": str(quarantine),
                "audit_failure": reason,
            }
        )
        pending.append(task)
    return completed, pending, quarantined


def _lane(
    lane: dict[str, str],
    work: queue.Queue,
    shared: dict,
    shared_runner,
    failure: threading.Event,
    results: list[dict],
    results_lock: threading.Lock,
    status_lock: threading.Lock,
    run_control: Path,
    n_tasks: int,
) -> dict:
    device = lane["device"]
    completed = 0
    while not failure.is_set():
        try:
            task = work.get_nowait()
        except queue.Empty:
            break
        result = _run_one(shared, shared_runner, task, device)
        result["lane_id"] = lane["lane_id"]
        with results_lock:
            results.append(result)
            done = len(results)
        with status_lock:
            _write_status(
                run_control,
                results_lock,
                results,
                n_tasks,
                shared["run_id"],
                shared["gate_sha256"],
            )
        print(
            f"[{done}/{n_tasks}] gpu={device} {result['fold_id']} "
            f"{result['model']} {result['seed']} -> {result['status']}",
            flush=True,
        )
        completed += int(result["status"] == "COMPLETED")
        work.task_done()
        if result["status"] != "COMPLETED":
            failure.set()
            break
    return {**lane, "completed": completed}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the locked V3.1 matrix on isolated GPU lanes"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--formal-gate", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument(
        "--slots-per-device",
        type=int,
        default=1,
        help="Number of isolated task processes sharing each physical GPU",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only audit-valid completed tasks from the same formal run",
    )
    parser.add_argument(
        "--quarantine-incomplete",
        action="store_true",
        help="With --resume, preserve incomplete task roots under run_control/quarantine and retry",
    )
    args = parser.parse_args()

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise RuntimeError("--devices must contain unique GPU indices")
    lanes = _execution_lanes(devices, args.slots_per_device)
    repo_root = Path(__file__).resolve().parents[1]
    shared_runner = _load_shared_runner(repo_root)
    cfg = load_config(
        args.config,
        project_root_override=Path(args.input_root).resolve(),
        create_dirs=False,
    )
    contract = cfg["formal_contract"]
    execution = cfg.get("formal_execution", {})
    expected_folds = int(contract["expected_folds"])
    expected_models = int(contract["expected_models"])
    expected_seeds = int(contract["expected_seeds"])
    expected_tasks = int(contract["expected_tasks"])
    if (
        len(shared_runner.MODELS) != expected_models
        or len(shared_runner.SEEDS) != expected_seeds
    ):
        raise RuntimeError("Shared runner model/seed constants disagree with config")
    required_gpus = int(execution.get("required_physical_gpus", len(devices)))
    configured_slots = int(execution.get("slots_per_device", args.slots_per_device))
    if len(devices) != required_gpus or args.slots_per_device != configured_slots:
        raise RuntimeError(
            "Execution lanes disagree with the frozen formal configuration: "
            f"gpus={len(devices)}/{required_gpus}, "
            f"slots={args.slots_per_device}/{configured_slots}"
        )

    output_root = Path(args.output_root).resolve()
    if output_root.exists() and not args.resume:
        raise RuntimeError(f"V3 multi-GPU runner refuses resume/reuse: {output_root}")
    if not output_root.exists() and args.resume:
        raise RuntimeError(f"V3 multi-GPU resume requires an existing output root: {output_root}")
    if args.quarantine_incomplete and not args.resume:
        raise RuntimeError("--quarantine-incomplete requires --resume")
    gate, gate_sha256 = validate_formal_training_gate(
        args.formal_gate,
        cfg,
        verify_assets=True,
        run_id=args.run_id,
        input_root=args.input_root,
        input_manifest=args.input_manifest,
        output_root=output_root,
    )
    folds = read_table(
        Path(gate["paths"]["asset_results"]) / "tables" / "fold_manifest.tsv"
    )
    fold_ids = sorted(folds.fold_id.astype(str).unique())
    if len(fold_ids) != expected_folds:
        raise RuntimeError(
            f"Fold matrix mismatch: observed={len(fold_ids)}, expected={expected_folds}"
        )
    tasks = [
        {"fold_id": fold, "model": model, "seed": seed}
        for fold in fold_ids
        for model in shared_runner.MODELS
        for seed in shared_runner.SEEDS
    ]
    if len(tasks) != expected_tasks:
        raise RuntimeError(
            f"Task matrix mismatch: observed={len(tasks)}, expected={expected_tasks}"
        )

    output_root.mkdir(parents=True, exist_ok=args.resume)
    run_control = output_root / "run_control"
    run_control.mkdir(exist_ok=args.resume)
    gate_snapshot = run_control / "FORMAL_TRAINING_GATE_SNAPSHOT.json"
    dispatch_path = run_control / "MULTIGPU_DISPATCH.json"
    if args.resume:
        if not gate_snapshot.exists() or not dispatch_path.exists():
            raise RuntimeError("Resume requires the original gate snapshot and dispatch lock")
        if json.loads(gate_snapshot.read_text(encoding="utf-8")) != gate:
            raise RuntimeError("Resume gate snapshot differs from the currently validated gate")
        dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
        if (
            dispatch.get("run_id") != args.run_id
            or dispatch.get("formal_training_gate_sha256") != gate_sha256
            or int(dispatch.get("task_count", -1)) != len(tasks)
        ):
            raise RuntimeError("Resume dispatch lock disagrees with run, gate, or task matrix")
    else:
        atomic_write_json(gate_snapshot, gate)
        atomic_write_json(
            dispatch_path,
            {
                "status": "LOCKED",
                "run_id": args.run_id,
                "formal_training_gate_sha256": gate_sha256,
                "devices": devices,
                "slots_per_device": args.slots_per_device,
                "lanes": lanes,
                "isolation": "CUDA_VISIBLE_DEVICES one visible physical GPU per task process",
                "task_count": len(tasks),
                "resume_used": False,
            },
        )
    shared = {
        "config": str(Path(args.config).resolve()),
        "run_id": args.run_id,
        "input_root": str(Path(args.input_root).resolve()),
        "input_manifest": str(Path(args.input_manifest).resolve()),
        "formal_gate": str(Path(args.formal_gate).resolve()),
        "output_root": str(output_root),
        "python": str(Path(args.python).resolve()),
        "repo_root": str(repo_root),
        "gate_sha256": gate_sha256,
    }

    if args.resume:
        resumed, tasks_to_run, quarantined = _partition_resume_tasks(
            tasks,
            output_root,
            shared_runner,
            shared,
            quarantine_incomplete=args.quarantine_incomplete,
        )
    else:
        resumed, tasks_to_run, quarantined = [], tasks, []

    work: queue.Queue = queue.Queue()
    for task in tasks_to_run:
        work.put(task)
    failure = threading.Event()
    results: list[dict] = list(resumed)
    results_lock = threading.Lock()
    status_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = [
            pool.submit(
                _lane,
                lane,
                work,
                shared,
                shared_runner,
                failure,
                results,
                results_lock,
                status_lock,
                run_control,
                len(tasks),
            )
            for lane in lanes
        ]
        for future in as_completed(futures):
            future.result()

    failed = [row for row in results if row["status"] == "FAILED"]
    if not failed and len(results) == len(tasks):
        validate_formal_training_gate(
            args.formal_gate,
            cfg,
            verify_assets=True,
            run_id=args.run_id,
            input_root=args.input_root,
            input_manifest=args.input_manifest,
            output_root=output_root,
        )
        status = "PASS"
        asset_status = "PASS"
    else:
        status = "FAIL"
        asset_status = "NOT_RUN_AFTER_INCOMPLETE_MATRIX"
    summary = {
        "status": status,
        "mode": f"FULL_{expected_tasks}_MULTIGPU",
        "run_id": args.run_id,
        "n_tasks": len(tasks),
        "tasks_done": len(results),
        "completed": sum(row["status"] == "COMPLETED" for row in results),
        "failed": len(failed),
        "not_started": len(tasks) - len(results),
        "failures": failed,
        "devices": devices,
        "slots_per_device": args.slots_per_device,
        "lanes": lanes,
        "formal_training_gate_sha256": gate_sha256,
        "asset_verification_after_matrix": asset_status,
        "resume_used": args.resume,
        "resumed_completed": len(resumed),
        "quarantined_incomplete": quarantined,
    }
    atomic_write_json(run_control / "PARALLEL_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
