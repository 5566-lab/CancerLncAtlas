#!/usr/bin/env python3
"""Run the fail-closed V3.1 matrix audit, sparse baselines, and comparison on 149."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from cc_hhgt.common import file_sha256
from cc_hhgt.v30_integrity import atomic_write_json


SEEDS = (20260726, 20261726, 20262726)
EXPECTED_TASKS = 99


def _run(command: list[str], log_path: Path | None = None) -> None:
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    if log_path is None:
        subprocess.run(command, check=True, env=environment)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        subprocess.run(command, check=True, env=environment, stdout=handle, stderr=subprocess.STDOUT)


def _passing_json(path: Path, run_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "PASS" and payload.get("run_id") == run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execution-gate", required=True)
    parser.add_argument("--training-lineage-gate", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    python = sys.executable
    repo_root = Path(args.repo_root).resolve()
    training_root = Path(args.training_root).resolve()
    output_base = Path(args.output_base).resolve()
    matrix_audit_root = output_base / "matrix_audit_297"
    selection_root = output_base / "website_model_selection"
    baseline_root = output_base / "sparse_baselines_exact_99"
    baseline_audit_root = output_base / "sparse_baseline_audit"
    comparison_root = output_base / "selected_deep_vs_sparse_baselines"
    logs = output_base / "postmatrix_logs"

    handoff = training_root / "run_control" / "MATRIX_HANDOFF_SUCCESS.json"
    backups = training_root / "run_control" / "backup_audit"
    if not _passing_json(handoff, args.run_id) or len(list(backups.glob("*.json"))) != 297:
        raise RuntimeError("149 post-matrix workflow requires the verified 297-task matrix handoff")
    if output_base.exists() and not output_base.is_dir():
        raise RuntimeError(f"Post-matrix output base is not a directory: {output_base}")
    output_base.mkdir(parents=True, exist_ok=True)

    matrix_audit_path = matrix_audit_root / "AUDIT.json"
    if not _passing_json(matrix_audit_path, args.run_id):
        if matrix_audit_root.exists():
            raise RuntimeError(f"Refusing incomplete matrix-audit root: {matrix_audit_root}")
        _run(
            [
                python,
                str(repo_root / "scripts" / "72_audit_v30_state_matrix.py"),
                "--config", args.config,
                "--run-id", args.run_id,
                "--input-root", args.input_root,
                "--input-manifest", args.input_manifest,
                "--formal-gate", args.execution_gate,
                "--training-lineage-gate", args.training_lineage_gate,
                "--training-root", args.training_root,
                "--output-root", str(matrix_audit_root),
            ],
            logs / "matrix_audit.log",
        )

    selection_path = selection_root / "WEBSITE_MODEL_SELECTION.json"
    if not _passing_json(selection_path, args.run_id):
        if selection_root.exists():
            raise RuntimeError(f"Refusing incomplete website-selection root: {selection_root}")
        _run(
            [
                python,
                str(repo_root / "scripts" / "76_select_v31_exact_pathway_website_model.py"),
                "--run-id", args.run_id,
                "--training-root", args.training_root,
                "--matrix-audit", str(matrix_audit_path),
                "--repo-root", str(repo_root),
                "--output-root", str(selection_root),
            ],
            logs / "website_selection.log",
        )

    folds = pd.read_csv(
        Path(json.loads(Path(args.execution_gate).read_text(encoding="utf-8"))["paths"]["asset_results"])
        / "tables"
        / "fold_manifest.tsv",
        sep="\t",
    ).fold_id.astype(str).tolist()
    if len(folds) != 33 or len(set(folds)) != 33:
        raise RuntimeError(f"Expected 33 unique formal folds, observed {len(set(folds))}")
    pending: list[tuple[str, int]] = []
    for fold in folds:
        for seed in SEEDS:
            success = baseline_root / fold / f"seed_{seed}" / "SUCCESS.json"
            if not _passing_json(success, args.run_id):
                task_root = success.parent
                if task_root.exists():
                    raise RuntimeError(f"Refusing incomplete sparse-baseline task root: {task_root}")
                pending.append((fold, seed))

    def run_baseline(task: tuple[str, int]) -> tuple[str, int]:
        fold, seed = task
        _run(
            [
                python,
                str(repo_root / "scripts" / "77_train_v31_exact_pathway_sparse_baseline.py"),
                "--config", args.config,
                "--run-id", args.run_id,
                "--formal-gate", args.execution_gate,
                "--training-lineage-gate", args.training_lineage_gate,
                "--training-root", args.training_root,
                "--matrix-audit", str(matrix_audit_path),
                "--repo-root", str(repo_root),
                "--output-root", str(baseline_root),
                "--fold", fold,
                "--seed", str(seed),
            ],
            logs / "sparse_baselines" / f"{fold}__seed_{seed}.log",
        )
        return task

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_baseline, task): task for task in pending}
        for future in as_completed(futures):
            fold, seed = futures[future]
            try:
                future.result()
            except Exception as exc:  # subprocess failure is recorded in its task log
                failures.append(f"{fold}/seed_{seed}: {exc}")
    if failures:
        raise RuntimeError(f"Sparse baseline matrix failed: {failures[:20]}")

    success_count = len(list(baseline_root.glob("LOCO_*/seed_*/SUCCESS.json")))
    if success_count != EXPECTED_TASKS:
        raise RuntimeError(f"Expected {EXPECTED_TASKS} sparse baseline tasks, observed {success_count}")
    if not _passing_json(baseline_audit_root / "AUDIT.json", args.run_id):
        if baseline_audit_root.exists():
            raise RuntimeError(f"Refusing incomplete sparse-baseline audit root: {baseline_audit_root}")
        _run(
            [
                python,
                str(repo_root / "scripts" / "81_audit_v31_exact_pathway_sparse_baselines.py"),
                "--run-id", args.run_id,
                "--training-root", args.training_root,
                "--matrix-audit", str(matrix_audit_path),
                "--baseline-root", str(baseline_root),
                "--repo-root", str(repo_root),
                "--output-root", str(baseline_audit_root),
            ],
            logs / "sparse_baseline_audit.log",
        )

    comparison_success = comparison_root / "SUCCESS.json"
    if not _passing_json(comparison_success, args.run_id):
        if comparison_root.exists():
            raise RuntimeError(f"Refusing incomplete comparison root: {comparison_root}")
        _run(
            [
                python,
                str(repo_root / "scripts" / "82_compare_v31_selected_deep_to_sparse_baselines.py"),
                "--run-id", args.run_id,
                "--training-root", args.training_root,
                "--matrix-audit", str(matrix_audit_path),
                "--selection", str(selection_path),
                "--baseline-audit-root", str(baseline_audit_root),
                "--repo-root", str(repo_root),
                "--output-root", str(comparison_root),
            ],
            logs / "selected_deep_comparison.log",
        )

    payload = {
        "status": "PASS",
        "run_id": args.run_id,
        "matrix_audit_sha256": file_sha256(matrix_audit_path),
        "website_selection_sha256": file_sha256(selection_path),
        "sparse_baseline_audit_sha256": file_sha256(baseline_audit_root / "AUDIT.json"),
        "selected_deep_comparison_sha256": file_sha256(comparison_success),
        "workers": args.workers,
        "sparse_baseline_tasks": success_count,
    }
    atomic_write_json(output_base / "POSTMATRIX_149_SUCCESS.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
