#!/usr/bin/env python3
"""Parallel V2.9 strict matrix runner with bundle reuse.

Each worker handles ONE (fold, seed) unit: loads the graph bundle ONCE
(resident in memory) and trains rgcn + hgt + cc_hhgt sequentially, so the
expensive graph construction is done 3x less often than the serial runner.
GPU utilization is also higher because multiple folds train concurrently.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cc_hhgt.common import load_config, read_table, write_json
from cc_hhgt.v29_multitask import train_multitask_fold
from cc_hhgt.v29_multitask import calibrate_multitask_fold

MODELS = ["rgcn", "hgt", "cc_hhgt"]
DEFAULT_SEEDS = [20260726, 20261726, 20262726]


def run_unit(args, fold_id: str, seed: int) -> dict:
    cfg, config = args
    folds = read_table(cfg["_results"] / "tables" / "fold_manifest.tsv")
    fold_row = folds.loc[folds.fold_id.astype(str).eq(fold_id)].iloc[0]
    fold_row["split_seed"] = seed
    completed = []
    failed = []
    bundle = None
    for model in MODELS:
        output_name = "cc_hhgt_strict" if model == "cc_hhgt" else model
        root = cfg["_results"] / "v2_9_strict" / output_name / fold_id / f"seed_{seed}"
        if (root / "SUCCESS.json").exists() and (root / "CALIBRATION_SUCCESS.json").exists():
            completed.append(f"{fold_id} {model} {seed}")
            continue
        try:
            bundle = train_multitask_fold(cfg, fold_row, model, seed, _bundle=bundle)
            result = calibrate_multitask_fold(root)
            (root / "CALIBRATION_SUCCESS.json").write_text(
                json.dumps({"status": "COMPLETED", "calibration": result}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            completed.append(f"{fold_id} {model} {seed}")
        except Exception as exc:
            import traceback
            failed.append({"task": f"{fold_id} {model} {seed}", "error": traceback.format_exc()[-2000:]})
    return {"fold_id": fold_id, "seed": seed, "completed": completed, "failed": failed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_v2_9_state_graph_local_run.yaml")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--folds", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    folds = read_table(cfg["_results"] / "tables" / "fold_manifest.tsv")
    fold_ids = [f for f in sorted(folds.fold_id.astype(str).unique()) if not args.folds or f in set(args.folds)]
    units = [(fold, seed) for fold in fold_ids for seed in args.seeds]

    run_root = cfg["_results"] / "v2_9_strict" / "run_control"
    run_root.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_unit, (cfg, args.config), fold, seed): (fold, seed) for fold, seed in units}
        done = 0
        for future in as_completed(futures):
            fold, seed = futures[future]
            try:
                res = future.result()
            except Exception as exc:
                import traceback
                res = {"fold_id": fold, "seed": seed, "completed": [], "failed": [{"task": f"{fold} {seed}", "error": traceback.format_exc()[-2000:]}]}
            results.append(res)
            done += 1
            n_ok = sum(len(r["completed"]) for r in results)
            n_fail = sum(len(r["failed"]) for r in results)
            write_json({
                "updated_at": datetime.now().isoformat(),
                "units_finished": done,
                "n_units": len(units),
                "tasks_completed": n_ok,
                "tasks_failed": n_fail,
                "failures": [f for r in results for f in r["failed"]],
            }, run_root / "run_status_parallel.json")
            print(f"[done {done}/{len(units)}] {fold} seed={seed} completed={n_ok} failed={n_fail}", flush=True)

    all_failed = [f for r in results for f in r["failed"]]
    summary = {
        "status": "COMPLETED" if not all_failed else "FAILED",
        "n_units": len(units),
        "n_tasks": len(units) * 3,
        "tasks_completed": sum(len(r["completed"]) for r in results),
        "tasks_failed": len(all_failed),
        "failures": all_failed[:20],
        "strict_graph_retrained": True,
        "state_nodes_required": True,
    }
    write_json(summary, run_root / "PARALLEL_SUMMARY.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if all_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
