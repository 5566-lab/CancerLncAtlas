#!/usr/bin/env python3
"""Run 33-cancer selected-model full-universe inference on isolated GPU lanes."""
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
from cc_hhgt.v30_integrity import atomic_write_json, canonical_json_sha256


MODELS = ("rgcn", "hgt", "cc_hhgt")
EXPECTED_FOLDS = 33
EXPECTED_SEEDS = 3


def _valid_success(
    path: Path,
    *,
    run_id: str,
    selected_model: str,
    selection_sha256: str,
    formal_gate_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    prediction = path.parent / "prediction_exact_pathway_three_seed_ensemble.parquet"
    return bool(
        payload.get("status") == "PASS"
        and payload.get("run_id") == run_id
        and payload.get("selected_model") == selected_model
        and payload.get("selection_sha256") == selection_sha256
        and payload.get("formal_gate_sha256") == formal_gate_sha256
        and payload.get("pathway_target_level") == "exact_pathway"
        and int(payload.get("seed_count", -1)) == EXPECTED_SEEDS
        and not bool(payload.get("heldout_label_columns_in_output"))
        and prediction.is_file()
        and payload.get("prediction_file_sha256") == file_sha256(prediction)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--formal-gate", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--slots-per-device", type=int, default=2)
    parser.add_argument("--microbatch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.slots_per_device < 1 or args.microbatch_size < 1:
        raise ValueError("slots-per-device and microbatch-size must be positive")

    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise RuntimeError("Formal selected-model inference requires exactly two physical GPU devices")
    gate_path = Path(args.formal_gate).resolve()
    gate_sha256 = file_sha256(gate_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("run_id") != args.run_id:
        raise RuntimeError("Selected-model inference requires the passing cloud formal gate")
    selection_path = Path(args.selection).resolve()
    selection_sha256 = file_sha256(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_model = str(selection.get("selected_model"))
    if (
        selection.get("status") != "PASS"
        or selection.get("run_id") != args.run_id
        or selected_model not in MODELS
        or selection.get("selection_split") != "val_only"
        or bool(selection.get("test_metrics_used"))
        or int(selection.get("folds", -1)) != EXPECTED_FOLDS
        or int(selection.get("seed_ensemble_size", -1)) != EXPECTED_SEEDS
        or not bool(selection.get("matrix_audit_release_eligible"))
    ):
        raise RuntimeError("Invalid validation-only website architecture selection")

    folds = pd.read_csv(
        Path(gate["paths"]["asset_results"]) / "tables" / "fold_manifest.tsv",
        sep="\t",
    )
    if len(folds) != EXPECTED_FOLDS or folds.fold_id.astype(str).nunique() != EXPECTED_FOLDS:
        raise RuntimeError("Selected-model inference requires all 33 formal LOCO folds")
    output_root = Path(args.output_root).resolve()
    control = output_root / "run_control"
    logs = control / "logs"
    control.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root).resolve()
    pending: list[str] = []
    for fold in folds.fold_id.astype(str):
        task_root = output_root / selected_model / fold
        success = task_root / "SUCCESS.json"
        if not _valid_success(
            success,
            run_id=args.run_id,
            selected_model=selected_model,
            selection_sha256=selection_sha256,
            formal_gate_sha256=gate_sha256,
        ):
            if task_root.exists():
                raise RuntimeError(f"Refusing incomplete selected-inference task root: {task_root}")
            pending.append(fold)

    lanes = [device for device in devices for _ in range(args.slots_per_device)]
    failures: list[dict] = []

    def run_fold(fold: str, device: str) -> dict:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = device
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        command = [
            sys.executable,
            str(repo_root / "scripts" / "78_v31_selected_model_seed_ensemble_inference.py"),
            "--config", args.config,
            "--run-id", args.run_id,
            "--formal-gate", args.formal_gate,
            "--training-root", args.training_root,
            "--selection", args.selection,
            "--fold", fold,
            "--output-root", str(output_root),
            "--microbatch-size", str(args.microbatch_size),
        ]
        log_path = logs / f"{fold}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as handle:
            completed = subprocess.run(
                command,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"exit={completed.returncode}; log={log_path}")
        success = output_root / selected_model / fold / "SUCCESS.json"
        if not _valid_success(
            success,
            run_id=args.run_id,
            selected_model=selected_model,
            selection_sha256=selection_sha256,
            formal_gate_sha256=gate_sha256,
        ):
            raise RuntimeError(f"Inference child returned without a valid SUCCESS: {fold}")
        payload = json.loads(success.read_text(encoding="utf-8"))
        return {
            "fold_id": fold,
            "device": device,
            "rows": int(payload["rows"]),
            "success_sha256": file_sha256(success),
            "prediction_sha256": payload["prediction_file_sha256"],
        }

    lane_tasks = [pending[index::len(lanes)] for index in range(len(lanes))]

    def run_lane(device: str, assigned: list[str]) -> list[dict]:
        return [run_fold(fold, device) for fold in assigned]

    with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        futures = {
            executor.submit(run_lane, device, assigned): (device, assigned)
            for device, assigned in zip(lanes, lane_tasks)
            if assigned
        }
        for future in as_completed(futures):
            device, assigned = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(
                    {"device": device, "assigned_folds": assigned, "error": str(exc)}
                )
    if failures:
        atomic_write_json(control / "INFERENCE_FAILURES.json", {"status": "FAIL", "failures": failures})
        raise RuntimeError(f"Selected-model inference failed: {failures[:10]}")

    audits: list[dict] = []
    for fold in sorted(folds.fold_id.astype(str)):
        success = output_root / selected_model / fold / "SUCCESS.json"
        if not _valid_success(
            success,
            run_id=args.run_id,
            selected_model=selected_model,
            selection_sha256=selection_sha256,
            formal_gate_sha256=gate_sha256,
        ):
            raise RuntimeError(f"Final selected-model inference coverage is incomplete: {fold}")
        payload = json.loads(success.read_text(encoding="utf-8"))
        audits.append(
            {
                "fold_id": fold,
                "test_cancer": payload["test_cancer"],
                "rows": int(payload["rows"]),
                "success_sha256": file_sha256(success),
                "prediction_sha256": payload["prediction_file_sha256"],
            }
        )
    payload = {
        "status": "PASS",
        "mode": "FULL_33_SELECTED_MODEL_MULTIGPU",
        "run_id": args.run_id,
        "selected_model": selected_model,
        "selection_sha256": selection_sha256,
        "selection_split": "val_only",
        "test_metrics_used_for_selection": False,
        "formal_gate_sha256": gate_sha256,
        "folds": len(audits),
        "seed_ensemble_size": EXPECTED_SEEDS,
        "devices": devices,
        "slots_per_device": args.slots_per_device,
        "microbatch_size": args.microbatch_size,
        "total_rows": int(sum(row["rows"] for row in audits)),
        "fold_audits": audits,
        "fold_audit_merkle_sha256": canonical_json_sha256(audits),
    }
    atomic_write_json(control / "PARALLEL_INFERENCE_SUMMARY.json", payload)
    atomic_write_json(control / "SUCCESS.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
