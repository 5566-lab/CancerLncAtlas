#!/usr/bin/env python3
"""Run only the A2-approved V3.1 Fixed CC-HHGT smoke/B1 matrix."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cc_hhgt.common import file_sha256, load_config, read_table
from cc_hhgt.gnn import load_graph_bundle
from cc_hhgt.prediction_contract import candidate_universe_sha256
from cc_hhgt.v29_multitask import calibrate_multitask_fold, train_multitask_fold
from cc_hhgt.v30_integrity import atomic_write_json
from cc_hhgt.v31_pilot_gate import (
    DIAGNOSTIC_CONTRACT,
    PILOT_CANCERS,
    PILOT_SEEDS,
    PRIMARY_CONTRACT,
    validate_v31_pilot_gate,
)


def audit_task(root: Path, contract: str, gate_sha256: str) -> dict[str, object]:
    required = [
        "TRAINING_SUCCESS.json",
        "CALIBRATION_SUCCESS.json",
        "best_pathway.pt",
        "best_state.pt",
        "training_history.tsv",
        "loss_output_diagnostics.tsv",
        "prediction_pathway_raw.parquet",
        "prediction_state_raw.parquet",
        "prediction_pathway_calibrated.parquet",
        "prediction_state_calibrated.parquet",
        "metrics_raw.tsv",
        "metrics_pathway_calibrated.tsv",
        "metrics_state_calibrated.tsv",
        "node_embeddings.pt",
        "embedding_metadata.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    failures: list[str] = []
    if missing:
        failures.append(f"missing={missing}")
    training = json.loads((root / "TRAINING_SUCCESS.json").read_text(encoding="utf-8"))
    if training.get("graph_contract") != contract:
        failures.append("graph_contract mismatch")
    if training.get("episodic_pseudoheldout") is not True:
        failures.append("episodic pseudoheldout disabled")
    if int(training.get("embedding_runtime_ensemble_chunks", 0)) <= 1:
        failures.append("embedding is not a runtime coverage-cycle ensemble")
    for task in ("pathway", "state"):
        raw = pd.read_parquet(root / f"prediction_{task}_raw.parquet")
        calibrated = pd.read_parquet(root / f"prediction_{task}_calibrated.parquet")
        if not raw.prediction_scale.astype(str).eq("raw_probability").all():
            failures.append(f"{task} raw scale mislabeled")
        if not calibrated.prediction_scale.astype(str).eq("calibrated_probability").all():
            failures.append(f"{task} calibrated scale mislabeled")
        if candidate_universe_sha256(raw) != candidate_universe_sha256(calibrated):
            failures.append(f"{task} candidate universe changed during calibration")
        if raw.proxy_positive_probability.isna().any() or calibrated.proxy_positive_probability.isna().any():
            failures.append(f"{task} prediction contains NaN")
    diagnostics = pd.read_csv(root / "loss_output_diagnostics.tsv", sep="\t")
    collapse = diagnostics.loc[
        diagnostics.split.astype(str).eq("test") & diagnostics.collapse_flag.astype(bool)
    ]
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "test_collapse_rows": int(len(collapse)),
        "pilot_gate_sha256": gate_sha256,
        "pathway_checkpoint_sha256": file_sha256(root / "best_pathway.pt"),
        "state_checkpoint_sha256": file_sha256(root / "best_state.pt"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot-gate", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["smoke", "b1"], required=True)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(args.output_root).resolve()
    gate, gate_sha256 = validate_v31_pilot_gate(
        args.pilot_gate, cfg, output_root=output_root, verify_assets=True
    )
    asset_results = Path(gate["paths"]["asset_results"]).resolve()
    folds = read_table(asset_results / "tables/fold_manifest.tsv")
    fold_rows = {}
    expected_validation = cfg["pilot_contract"]["validation_cancers"]
    for cancer in PILOT_CANCERS:
        hit = folds.loc[folds.test_cancer.astype(str).eq(cancer)]
        if len(hit) != 1:
            raise RuntimeError(f"Expected one fold for {cancer}, observed {len(hit)}")
        row = pd.Series(hit.iloc[0].to_dict())
        if str(row.validation_cancer) != str(expected_validation[cancer]):
            raise RuntimeError(f"Validation cancer drift for {cancer}")
        fold_rows[cancer] = row

    if args.mode == "smoke":
        if args.smoke_epochs < 1:
            raise ValueError("--smoke-epochs must be positive")
        matrix = [(PRIMARY_CONTRACT, "BRCA", PILOT_SEEDS[0])]
        stage_root = output_root / "SMOKE"
        cfg["training"]["epochs"] = args.smoke_epochs
        cfg["training"]["patience"] = max(args.smoke_epochs, 1)
    else:
        matrix = [
            (PRIMARY_CONTRACT, cancer, seed)
            for cancer in PILOT_CANCERS
            for seed in PILOT_SEEDS
        ] + [
            (DIAGNOSTIC_CONTRACT, cancer, PILOT_SEEDS[0])
            for cancer in PILOT_CANCERS
        ]
        if len(matrix) != 12:
            raise RuntimeError("B1 matrix must contain exactly 12 tasks")
        stage_root = output_root / "FIXED_GRAPH"
    if stage_root.exists():
        raise RuntimeError(f"V3.1 runner has no resume/reuse path: {stage_root}")
    stage_root.mkdir(parents=True)
    control = stage_root / "run_control"
    control.mkdir()
    atomic_write_json(control / "PILOT_GATE_SNAPSHOT.json", gate)

    cfg["_results"] = asset_results
    cfg["_standardized"] = asset_results.parent / "standardized"
    cfg["_cache"] = asset_results.parent / "cache"
    cfg["_formal_training_gate_path"] = str(Path(args.pilot_gate).resolve())
    cfg["_formal_training_gate_sha256"] = gate_sha256
    cfg["_run_id"] = f"V31-FIXED-GRAPH-{args.mode.upper()}"
    results: list[dict[str, object]] = []

    # One canonical schedule is shared by all seeds of a fold/contract.  It is
    # deterministic and seed-independent; every model/optimizer is still
    # freshly initialized by train_multitask_fold(seed).
    grouped: dict[tuple[str, str], list[int]] = {}
    for contract, cancer, seed in matrix:
        grouped.setdefault((contract, cancer), []).append(seed)
    for (contract, cancer), seeds in grouped.items():
        row = fold_rows[cancer]
        contract_root = stage_root / contract
        cfg["_strict_output_root"] = contract_root
        cfg["graph_contract"]["primary"] = contract
        bundle_started = time.perf_counter()
        bundle = load_graph_bundle(
            cfg,
            PILOT_SEEDS[0],
            excluded_cancers={str(row.test_cancer), str(row.validation_cancer)},
            contract=contract,
        )
        bundle_seconds = time.perf_counter() - bundle_started
        for seed in seeds:
            task_root = contract_root / "cc_hhgt" / str(row.fold_id) / f"seed_{seed}"
            if task_root.exists():
                raise RuntimeError(f"Task output already exists: {task_root}")
            validate_v31_pilot_gate(
                args.pilot_gate, cfg, output_root=output_root, verify_assets=True
            )
            started = time.perf_counter()
            seed_row = row.copy()
            seed_row["split_seed"] = seed
            training = train_multitask_fold(
                cfg, seed_row, "cc_hhgt", seed, _bundle=bundle
            )
            calibration = calibrate_multitask_fold(task_root)
            validate_v31_pilot_gate(
                args.pilot_gate, cfg, output_root=output_root, verify_assets=True
            )
            audit = audit_task(task_root, contract, gate_sha256)
            success = {
                **training,
                "status": "COMPLETED" if audit["status"] == "PASS" else "FAILED",
                "contract": contract,
                "test_cancer": cancer,
                "seed": seed,
                "fresh_initialization": True,
                "shared_schedule_across_seeds": True,
                "schedule_build_seconds": bundle_seconds,
                "task_seconds": time.perf_counter() - started,
                "pilot_gate_sha256": gate_sha256,
                "phase_a2_gate_sha256": gate["phase_a2_gate_sha256"],
                "git_commit": gate["git_commit"],
                "calibration": calibration,
                "post_task_audit": audit,
                "asset_verification_at_start": "PASS",
                "asset_verification_at_end": "PASS",
                "success_written_last": True,
            }
            atomic_write_json(task_root / "SUCCESS.json", success)
            result = {
                "contract": contract,
                "test_cancer": cancer,
                "fold_id": str(row.fold_id),
                "seed": seed,
                "status": success["status"],
                "task_root": str(task_root),
                "task_seconds": success["task_seconds"],
                "test_collapse_rows": audit["test_collapse_rows"],
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
                },
            )
            if success["status"] != "COMPLETED":
                raise RuntimeError(f"B1 task audit failed: {result}")

    summary = {
        "status": "PASS",
        "mode": args.mode,
        "tasks_completed": len(results),
        "tasks_expected": len(matrix),
        "contracts_reported_separately": True,
        "full_cancer_training_started": False,
        "results": results,
    }
    atomic_write_json(control / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
