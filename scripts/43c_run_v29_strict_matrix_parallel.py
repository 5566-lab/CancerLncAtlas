#!/usr/bin/env python3
"""Fail-closed isolated V3 matrix runner (fixed pilot or configured formal matrix)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cc_hhgt.common import file_sha256, load_config, read_table
from cc_hhgt.formal_gate import validate_formal_training_gate
from cc_hhgt.prediction_contract import (
    PredictionScale,
    validate_probability_aliases,
)
from cc_hhgt.v30_integrity import atomic_write_json


MODELS = ("rgcn", "hgt", "cc_hhgt")
SEEDS = (20260726, 20261726, 20262726)
PILOT_FOLDS = ("LOCO_BRCA", "LOCO_KIRC", "LOCO_OV")


def audit_task(root: Path, task: dict, gate_sha256: str, run_id: str) -> tuple[bool, str]:
    required = [
        "TRAINING_SUCCESS.json",
        "CALIBRATION_SUCCESS.json",
        "SUCCESS.json",
        "state_training_sample_audit.json",
        "TRAINING_PROGRESS.json",
        "metrics.tsv",
        "training_history.tsv",
        "best_pathway.pt",
        "best_state.pt",
        "last_training_state.pt",
        "prediction_pathway_calibrated.parquet",
        "prediction_state_calibrated.parquet",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        return False, f"missing outputs: {missing}"
    try:
        success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
        audit = json.loads((root / "state_training_sample_audit.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(root / "metrics.tsv", sep="\t")
        history = pd.read_csv(root / "training_history.tsv", sep="\t")
        pathway = pd.read_parquet(root / "prediction_pathway_calibrated.parquet")
        state = pd.read_parquet(root / "prediction_state_calibrated.parquet")
    except Exception as exc:
        return False, f"unreadable outputs: {exc}"
    expected = {
        "status": "COMPLETED",
        "run_id": run_id,
        "model_name": task["model"],
        "fold_id": task["fold_id"],
        "formal_training_gate_sha256": gate_sha256,
    }
    disagreements = {key: (value, success.get(key)) for key, value in expected.items() if success.get(key) != value}
    if disagreements:
        return False, f"SUCCESS lineage mismatch: {disagreements}"
    if success.get("pathway_checkpoint_sha256") != file_sha256(root / "best_pathway.pt"):
        return False, "pathway checkpoint SHA256 mismatch"
    if success.get("state_checkpoint_sha256") != file_sha256(root / "best_state.pt"):
        return False, "state checkpoint SHA256 mismatch"
    if (
        not success.get("exact_interruption_resume_supported")
        or success.get("last_training_state_sha256")
        != file_sha256(root / "last_training_state.pt")
    ):
        return False, "exact interruption-resume checkpoint missing or SHA256 mismatch"
    if success.get("training_progress_sha256") != file_sha256(
        root / "TRAINING_PROGRESS.json"
    ):
        return False, "training progress SHA256 mismatch"
    if not success.get("success_written_last") or success.get("asset_verification_at_end") != "PASS":
        return False, "task SUCCESS ordering or end-asset verification missing"
    if success.get("hit_hard_epoch_cap") or not success.get(
        "stopped_by_joint_patience"
    ):
        return False, "task did not stop by joint patience before the hard epoch cap"
    gpu_contract = success.get("formal_gpu_contract") or {}
    if (
        int(gpu_contract.get("visible_device_count", 0)) != 1
        or float(gpu_contract.get("total_memory_gib", 0.0))
        < float(gpu_contract.get("minimum_gpu_memory_gib", float("inf")))
    ):
        return False, f"formal GPU execution contract failed: {gpu_contract}"
    positive = int(audit.get("state_training_positive", 0))
    unlabeled = int(audit.get("state_training_unlabeled", 0))
    if positive <= 0 or unlabeled <= 0:
        return False, f"state training lacks both PU classes: positive={positive}, unlabeled={unlabeled}"
    if int(audit.get("state_unlabeled_direction_used_in_loss", -1)) != 0:
        return False, "unlabeled state direction supervision is nonzero"
    if history.empty or int(success.get("best_pathway_epoch", 0)) <= 0 or int(success.get("best_state_epoch", 0)) <= 0:
        return False, "missing training history or independently selected checkpoints"
    if int(success.get("optimizer_steps", 0)) != int(history.optimizer_steps.iloc[-1]):
        return False, "recorded optimizer step count disagrees with history"
    for task_name, frame in (("pathway", pathway), ("state", state)):
        try:
            validate_probability_aliases(
                frame, PredictionScale.CALIBRATED_PROBABILITY
            )
        except RuntimeError as exc:
            return False, f"{task_name} probability semantics invalid: {exc}"
        if frame.duplicated(["candidate_id", "split"]).any():
            return False, f"{task_name} candidate keys are duplicated"
    state_test = metrics.loc[metrics.task.eq("state") & metrics.split.eq("test")]
    pathway_test = metrics.loc[metrics.task.eq("pathway") & metrics.split.eq("test")]
    if pathway_test.shape[0] != 1 or state_test.empty or state_test.state_id.isna().any():
        return False, "test metrics are not one pathway row plus state-specific rows"
    return True, "PASS"


def run_task(shared: dict, task: dict) -> dict:
    output_root = Path(shared["output_root"])
    root = output_root / task["model"] / task["fold_id"] / f"seed_{task['seed']}"
    if root.exists():
        return {**task, "status": "FAILED", "error": "task output existed before launch; resume is forbidden"}
    command = [
        shared["python"],
        "scripts/42_train_v29_strict_multitask.py",
        "--config", shared["config"],
        "--run-id", shared["run_id"],
        "--input-root", shared["input_root"],
        "--input-manifest", shared["input_manifest"],
        "--formal-gate", shared["formal_gate"],
        "--output-root", shared["output_root"],
        "--model", task["model"],
        "--fold", task["fold_id"],
        "--seed", str(task["seed"]),
    ]
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONUNBUFFERED"] = "1"
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
    log_path = log_root / f"{task['fold_id']}__{task['model']}__seed_{task['seed']}.log"
    log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\nRETURN_CODE: {result.returncode}\nDURATION_SECONDS: {duration:.3f}\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode:
        return {**task, "status": "FAILED", "duration_seconds": duration, "error": result.stderr[-3000:]}
    valid, reason = audit_task(root, task, shared["gate_sha256"], shared["run_id"])
    return (
        {**task, "status": "COMPLETED", "duration_seconds": duration, "quality_gate": reason}
        if valid
        else {**task, "status": "FAILED", "duration_seconds": duration, "error": f"post-task audit: {reason}"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed V3 formal pilot or full matrix")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--formal-gate", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--pilot", action="store_true", help="Exactly BRCA/KIRC/OV x three models x primary seed")
    args = parser.parse_args()
    if args.jobs < 1:
        raise RuntimeError("--jobs must be positive")
    cfg = load_config(
        args.config,
        project_root_override=Path(args.input_root).resolve(),
        create_dirs=False,
    )
    contract = cfg["formal_contract"]
    configured_tasks = int(contract["expected_tasks"])
    configured_folds = int(contract["expected_folds"])
    configured_models = int(contract["expected_models"])
    configured_seeds = int(contract["expected_seeds"])
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"V3 runner refuses every resume/reuse path: {output_root}")
    gate_payload, gate_sha256 = validate_formal_training_gate(
        args.formal_gate,
        cfg,
        verify_assets=True,
        run_id=args.run_id,
        input_root=args.input_root,
        input_manifest=args.input_manifest,
        output_root=output_root,
    )
    folds = read_table(Path(gate_payload["paths"]["asset_results"]) / "tables" / "fold_manifest.tsv")
    available = set(folds.fold_id.astype(str))
    fold_ids = list(PILOT_FOLDS) if args.pilot else sorted(available)
    if not set(fold_ids).issubset(available):
        raise RuntimeError(f"Requested fixed pilot folds are absent: {sorted(set(fold_ids) - available)}")
    seeds = (SEEDS[0],) if args.pilot else SEEDS
    tasks = [
        {"fold_id": fold_id, "model": model, "seed": seed}
        for fold_id in fold_ids
        for model in MODELS
        for seed in seeds
    ]
    if not args.pilot and len(fold_ids) != configured_folds:
        raise RuntimeError(f"Runner fold matrix is not the locked size {configured_folds}: {len(fold_ids)}")
    if not args.pilot and (len(MODELS) != configured_models or len(SEEDS) != configured_seeds):
        raise RuntimeError(
            "Runner model/seed constants disagree with formal config: "
            f"models={len(MODELS)}/{configured_models}, seeds={len(SEEDS)}/{configured_seeds}"
        )
    expected = 9 if args.pilot else configured_tasks
    if len(tasks) != expected:
        raise RuntimeError(f"Runner task matrix is not the locked size {expected}: {len(tasks)}")

    output_root.mkdir(parents=True)
    run_control = output_root / "run_control"
    run_control.mkdir()
    atomic_write_json(run_control / "FORMAL_TRAINING_GATE_SNAPSHOT.json", gate_payload)
    shared = {
        "config": str(Path(args.config).resolve()),
        "run_id": args.run_id,
        "input_root": str(Path(args.input_root).resolve()),
        "input_manifest": str(Path(args.input_manifest).resolve()),
        "formal_gate": str(Path(args.formal_gate).resolve()),
        "output_root": str(output_root),
        "python": str(Path(args.python).resolve()),
        "repo_root": str(Path(__file__).resolve().parents[1]),
        "gate_sha256": gate_sha256,
    }
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_task, shared, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {**task, "status": "FAILED", "error": repr(exc)}
            results.append(result)
            status = {
                "status": "RUNNING",
                "mode": "PILOT" if args.pilot else f"FULL_{configured_tasks}",
                "run_id": args.run_id,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "tasks_done": done,
                "n_tasks": len(tasks),
                "completed": sum(row["status"] == "COMPLETED" for row in results),
                "failed": sum(row["status"] == "FAILED" for row in results),
                "formal_training_gate_sha256": gate_sha256,
                "failures": [row for row in results if row["status"] == "FAILED"],
            }
            atomic_write_json(run_control / "run_status_parallel.json", status)
            print(f"[{done}/{len(tasks)}] {result['fold_id']} {result['model']} {result['seed']} -> {result['status']}", flush=True)

    # Revalidate every frozen asset after the last task.
    validate_formal_training_gate(
        args.formal_gate,
        cfg,
        verify_assets=True,
        run_id=args.run_id,
        input_root=args.input_root,
        input_manifest=args.input_manifest,
        output_root=output_root,
    )
    failed = [row for row in results if row["status"] == "FAILED"]
    summary = {
        "status": "PASS" if not failed else "FAIL",
        "mode": "PILOT" if args.pilot else f"FULL_{configured_tasks}",
        "run_id": args.run_id,
        "n_tasks": len(tasks),
        "completed": sum(row["status"] == "COMPLETED" for row in results),
        "failed": len(failed),
        "failures": failed,
        "formal_training_gate_sha256": gate_sha256,
        "asset_verification_after_matrix": "PASS",
        "resume_used": False,
    }
    atomic_write_json(run_control / "PARALLEL_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
