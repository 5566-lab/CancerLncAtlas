#!/usr/bin/env python3
"""Run the fixed three-cancer Graph residual matrix, never a full-cancer run."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cc_hhgt.common import file_sha256, load_config, read_table
from cc_hhgt.gnn import load_graph_bundle
from cc_hhgt.v29_multitask import calibrate_multitask_fold, train_multitask_fold
from cc_hhgt.v30_integrity import atomic_write_json
from cc_hhgt.v31_graph_residual_stage import (
    GRAPH_CONTRACT,
    audit_residual_task,
    expected_graph_residual_matrix,
    lambda_token,
    load_fold_residual_base,
)
from cc_hhgt.v31_pilot_gate import PILOT_CANCERS, PILOT_SEEDS, validate_v31_pilot_gate
from cc_hhgt.v31_residual_gate import validate_residual_stage_gate


def _require_pass(path: Path, stage: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("failures"):
        raise RuntimeError(f"{stage} gate is not PASS: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot-gate", required=True)
    parser.add_argument("--b1-gate", required=True)
    parser.add_argument("--best-simple-gate", required=True)
    parser.add_argument("--stage-gate", required=True)
    parser.add_argument("--residual-base", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["smoke", "b2"], required=True)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    gate_path = Path(args.pilot_gate).resolve()
    gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    fixed_cfg = load_config(gate_payload["paths"]["config"])
    registered_output_root = Path(gate_payload["paths"]["output_root"]).resolve()
    gate, gate_sha256 = validate_v31_pilot_gate(
        gate_path,
        fixed_cfg,
        output_root=registered_output_root,
        verify_assets=True,
    )
    b1_gate_path = Path(args.b1_gate).resolve()
    simple_gate_path = Path(args.best_simple_gate).resolve()
    b1_gate = _require_pass(b1_gate_path, "B1")
    simple_gate = _require_pass(simple_gate_path, "BestSimple")
    if int(b1_gate.get("tasks_completed", 0)) != 12:
        raise RuntimeError("B1 does not contain the exact 12-task matrix")
    if simple_gate.get("selection_used_test_labels") is not False:
        raise RuntimeError("BestSimple gate does not prove test-independent selection")

    pilot = cfg.get("pilot_contract", {})
    if tuple(pilot.get("cancers", [])) != PILOT_CANCERS:
        raise RuntimeError("Graph residual cancer scope drift")
    if tuple(map(int, pilot.get("seeds", []))) != PILOT_SEEDS:
        raise RuntimeError("Graph residual seed scope drift")
    if pilot.get("forbid_full_cancer_training") is not True:
        raise RuntimeError("Graph residual config does not forbid full-cancer training")
    if cfg.get("graph_contract", {}).get("primary") != GRAPH_CONTRACT:
        raise RuntimeError("R1 Graph residual must use the stable Contract-S Core Graph")
    residual_cfg = cfg.get("residual_learning", {})
    if residual_cfg.get("enabled") is not True:
        raise RuntimeError("Graph residual training is disabled")
    lambdas = list(map(float, residual_cfg.get("shrinkage_grid", [])))
    full_matrix = expected_graph_residual_matrix(lambdas)
    if args.mode == "smoke":
        if args.smoke_epochs < 1:
            raise ValueError("--smoke-epochs must be positive")
        matrix = [full_matrix[0]]
        stage_root = Path(args.output_root).resolve() / "GRAPH_RESIDUAL_SMOKE"
        cfg["training"]["epochs"] = int(args.smoke_epochs)
        cfg["training"]["patience"] = int(args.smoke_epochs)
    else:
        matrix = full_matrix
        stage_root = Path(args.output_root).resolve() / "GRAPH_RESIDUAL" / "runs"
    if stage_root.exists():
        raise RuntimeError(f"Graph residual runner has no resume/reuse path: {stage_root}")
    stage_root.mkdir(parents=True)
    control = stage_root / "run_control"
    control.mkdir()

    residual_base_path = Path(args.residual_base).resolve()
    if not residual_base_path.is_file():
        raise RuntimeError(f"Frozen residual base is missing: {residual_base_path}")
    stage_gate_path = Path(args.stage_gate).resolve()
    stage_gate, stage_gate_sha256 = validate_residual_stage_gate(
        stage_gate_path,
        cfg,
        output_root=Path(args.output_root).resolve(),
        verify_files=True,
    )
    if Path(stage_gate["paths"]["residual_base"]).resolve() != residual_base_path:
        raise RuntimeError("Residual base differs from the registered stage gate")
    asset_results = Path(gate["paths"]["asset_results"]).resolve()
    folds = read_table(asset_results / "tables/fold_manifest.tsv")
    fold_rows: dict[str, pd.Series] = {}
    for cancer in PILOT_CANCERS:
        hit = folds.loc[folds.test_cancer.astype(str).eq(cancer)]
        if len(hit) != 1:
            raise RuntimeError(f"Expected one fold for {cancer}, observed {len(hit)}")
        fold_rows[cancer] = pd.Series(hit.iloc[0].to_dict())

    cfg["_results"] = asset_results
    cfg["_standardized"] = asset_results.parent / "standardized"
    cfg["_cache"] = asset_results.parent / "cache"
    cfg["_formal_training_gate_path"] = str(gate_path)
    cfg["_formal_training_gate_sha256"] = gate_sha256
    cfg["_run_id"] = f"V31-GRAPH-RESIDUAL-{args.mode.upper()}"
    atomic_write_json(
        control / "STAGE_INPUTS.json",
        {
            "status": "PASS",
            "mode": args.mode,
            "tasks_expected": len(matrix),
            "pilot_gate_sha256": gate_sha256,
            "b1_gate_sha256": file_sha256(b1_gate_path),
            "best_simple_gate_sha256": file_sha256(simple_gate_path),
            "residual_base_sha256": file_sha256(residual_base_path),
            "residual_stage_gate_sha256": stage_gate_sha256,
            "full_cancer_training_authorized": False,
        },
    )

    results: list[dict[str, object]] = []
    by_cancer: dict[str, list[tuple[int, float]]] = {}
    for cancer, seed, value in matrix:
        by_cancer.setdefault(cancer, []).append((seed, value))
    for cancer, jobs in by_cancer.items():
        row = fold_rows[cancer]
        base = load_fold_residual_base(residual_base_path, heldout_cancer=cancer)
        bundle_started = time.perf_counter()
        bundle = load_graph_bundle(
            cfg,
            PILOT_SEEDS[0],
            excluded_cancers={str(row.test_cancer), str(row.validation_cancer)},
            contract=GRAPH_CONTRACT,
        )
        bundle_seconds = time.perf_counter() - bundle_started
        for seed, value in jobs:
            validate_v31_pilot_gate(
                gate_path,
                fixed_cfg,
                output_root=registered_output_root,
                verify_assets=True,
            )
            validate_residual_stage_gate(
                stage_gate_path,
                cfg,
                output_root=Path(args.output_root).resolve(),
                verify_files=True,
            )
            cfg["residual_learning"]["shrinkage_lambda"] = float(value)
            cfg["_strict_output_root"] = stage_root / f"lambda_{lambda_token(value)}"
            task_root = (
                Path(cfg["_strict_output_root"])
                / "cc_hhgt"
                / str(row.fold_id)
                / f"seed_{seed}"
            )
            if task_root.exists():
                raise RuntimeError(f"Graph residual task output already exists: {task_root}")
            started = time.perf_counter()
            seed_row = row.copy()
            seed_row["split_seed"] = int(seed)
            training = train_multitask_fold(
                cfg,
                seed_row,
                "cc_hhgt",
                int(seed),
                _bundle=bundle,
                _residual_base=base,
            )
            calibration = calibrate_multitask_fold(task_root)
            audit = audit_residual_task(
                task_root,
                heldout_cancer=cancer,
                seed=int(seed),
                shrinkage_lambda=float(value),
            )
            validate_v31_pilot_gate(
                gate_path,
                fixed_cfg,
                output_root=registered_output_root,
                verify_assets=True,
            )
            validate_residual_stage_gate(
                stage_gate_path,
                cfg,
                output_root=Path(args.output_root).resolve(),
                verify_files=True,
            )
            success = {
                **training,
                "status": "COMPLETED" if audit["status"] == "PASS" else "FAILED",
                "stage": "PHASE_B2_GRAPH_RESIDUAL",
                "test_cancer": cancer,
                "seed": int(seed),
                "contract": GRAPH_CONTRACT,
                "residual_shrinkage_lambda": float(value),
                "fresh_initialization": True,
                "schedule_build_seconds": bundle_seconds,
                "task_seconds": time.perf_counter() - started,
                "pilot_gate_sha256": gate_sha256,
                "b1_gate_sha256": file_sha256(b1_gate_path),
                "best_simple_gate_sha256": file_sha256(simple_gate_path),
                "residual_base_sha256": file_sha256(residual_base_path),
                "residual_stage_gate_sha256": stage_gate_sha256,
                "calibration": calibration,
                "post_task_audit": audit,
                "success_written_last": True,
            }
            atomic_write_json(task_root / "SUCCESS.json", success)
            result = {
                "test_cancer": cancer,
                "fold_id": str(row.fold_id),
                "seed": int(seed),
                "residual_shrinkage_lambda": float(value),
                "status": success["status"],
                "task_root": str(task_root),
                "task_seconds": success["task_seconds"],
            }
            results.append(result)
            atomic_write_json(
                control / "RUN_STATUS.json",
                {
                    "status": "RUNNING",
                    "mode": args.mode,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "tasks_completed": len(results),
                    "tasks_expected": len(matrix),
                    "results": results,
                    "full_cancer_training_started": False,
                },
            )
            if success["status"] != "COMPLETED":
                raise RuntimeError(f"Graph residual task audit failed: {result}")

    summary = {
        "status": "PASS",
        "mode": args.mode,
        "tasks_completed": len(results),
        "tasks_expected": len(matrix),
        "selection_used_test_labels": False,
        "full_cancer_training_started": False,
        "results": results,
    }
    atomic_write_json(control / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
