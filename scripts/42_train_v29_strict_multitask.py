#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from cc_hhgt.common import configure_logging, file_sha256, load_config, read_table
from cc_hhgt.formal_gate import validate_formal_training_gate
from cc_hhgt.v29_multitask import calibrate_multitask_fold, train_multitask_fold
from cc_hhgt.v30_integrity import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one guarded V3.0 graph multitask job")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--formal-gate", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=["rgcn", "hgt", "cc_hhgt"], required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--state-max-training-pairs", type=int)
    parser.add_argument("--state-unlabeled-to-positive-ratio", type=float)
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="Resume an interrupted task from its exact task-local training state",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    task_started = time.time()
    task_started_at_utc = datetime.now(timezone.utc).isoformat()
    configure_logging(args.verbose)

    cfg = load_config(
        args.config,
        project_root_override=Path(args.input_root).resolve(),
        create_dirs=False,
    )
    import torch

    execution = cfg.get("formal_execution", {})
    minimum_gpu_memory_gib = float(execution.get("minimum_gpu_memory_gib", 0.0))
    if not torch.cuda.is_available() or torch.cuda.device_count() != int(
        execution.get("task_visible_gpu_count", 1)
    ):
        raise RuntimeError(
            "Formal task requires exactly one CUDA-visible GPU per process"
        )
    gpu_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    gpu_memory_gib = float(gpu_properties.total_memory) / (1024**3)
    if gpu_memory_gib < minimum_gpu_memory_gib:
        raise RuntimeError(
            f"Formal GPU has {gpu_memory_gib:.2f} GiB; "
            f"configuration requires at least {minimum_gpu_memory_gib:.2f} GiB"
        )
    cfg["_formal_gpu_contract"] = {
        "device_name": str(gpu_properties.name),
        "total_memory_gib": gpu_memory_gib,
        "visible_device_count": int(torch.cuda.device_count()),
        "minimum_gpu_memory_gib": minimum_gpu_memory_gib,
    }
    output_root = Path(args.output_root).resolve()
    task_root = output_root / args.model / args.fold / f"seed_{args.seed}"
    resume_checkpoint = task_root / "last_training_state.pt"
    if args.resume_training:
        if not task_root.is_dir() or not resume_checkpoint.is_file():
            raise RuntimeError(
                "Exact task resume requires an existing last_training_state.pt: "
                f"{resume_checkpoint}"
            )
        if (task_root / "SUCCESS.json").exists():
            raise RuntimeError(f"Completed tasks cannot be resumed: {task_root}")
    elif task_root.exists():
        raise RuntimeError(
            f"Task output already exists; pass --resume-training only for an exact checkpoint: {task_root}"
        )
    gate_payload, gate_sha256 = validate_formal_training_gate(
        args.formal_gate,
        cfg,
        verify_assets=True,
        run_id=args.run_id,
        input_root=args.input_root,
        input_manifest=args.input_manifest,
        output_root=output_root,
    )
    cfg["_results"] = Path(gate_payload["paths"]["asset_results"]).resolve()
    cfg["_standardized"] = cfg["_results"].parent / "standardized"
    cfg["_cache"] = cfg["_results"].parent / "cache"
    cfg["_strict_output_root"] = output_root
    cfg["_formal_training_gate_path"] = str(Path(args.formal_gate).resolve())
    cfg["_formal_training_gate_sha256"] = gate_sha256
    cfg["_run_id"] = args.run_id
    cfg["_sample_universe_sha256"] = gate_payload["sample_universe_sha256"]
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.patience is not None:
        cfg["training"]["patience"] = args.patience
    if args.state_max_training_pairs is not None:
        cfg.setdefault("state_training", {})["max_training_pairs_per_fold"] = args.state_max_training_pairs
    if args.state_unlabeled_to_positive_ratio is not None:
        cfg.setdefault("state_training", {})["unlabeled_to_positive_ratio"] = args.state_unlabeled_to_positive_ratio
    if args.resume_training:
        cfg["_resume_training_checkpoint"] = resume_checkpoint

    folds = read_table(cfg["_results"] / "tables" / "fold_manifest.tsv")
    hit = folds.loc[folds.fold_id.astype(str).eq(args.fold)]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one fold {args.fold}, observed {len(hit)}")
    row = pd.Series(hit.iloc[0].to_dict())
    row["split_seed"] = args.seed
    training = train_multitask_fold(cfg, row, args.model, args.seed)
    calibration = calibrate_multitask_fold(task_root)

    # Full code/input/asset validation is repeated at task end. SUCCESS is
    # deliberately the final task output and is never written by the trainer.
    _, end_gate_sha256 = validate_formal_training_gate(
        args.formal_gate,
        cfg,
        verify_assets=True,
        run_id=args.run_id,
        input_root=args.input_root,
        input_manifest=args.input_manifest,
        output_root=output_root,
    )
    if end_gate_sha256 != gate_sha256:
        raise RuntimeError("Formal gate changed while the task was running")
    success = {
        **training,
        "status": "COMPLETED",
        "run_id": args.run_id,
        "formal_training_gate_sha256": gate_sha256,
        "sample_universe_sha256": gate_payload["sample_universe_sha256"],
        "input_merkle_sha256": gate_payload["manifest_contracts"][0]["merkle_sha256"],
        "code_merkle_sha256": gate_payload["manifest_contracts"][1]["merkle_sha256"],
        "asset_merkle_sha256": gate_payload["asset_merkle_sha256"],
        "config_merkle_sha256": json.loads((Path(gate_payload["paths"]["run_root"]) / "provenance" / "PROVENANCE.json").read_text(encoding="utf-8"))["config_merkle_sha256"],
        "calibration": calibration,
        "calibration_success_sha256": file_sha256(task_root / "CALIBRATION_SUCCESS.json"),
        "asset_verification_at_start": "PASS",
        "asset_verification_at_end": "PASS",
        "formal_gpu_contract": cfg["_formal_gpu_contract"],
        "resume_training_invocation": bool(args.resume_training),
        "task_started_at_utc": task_started_at_utc,
        "task_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_duration_seconds": time.time() - task_started,
        "success_written_last": True,
    }
    atomic_write_json(task_root / "SUCCESS.json", success)
    print(json.dumps(success, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
