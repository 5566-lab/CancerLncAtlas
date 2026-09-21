#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from cc_hhgt.common import file_sha256, load_config, read_table, write_json


MODEL_DIRS = {"rgcn": "rgcn", "hgt": "hgt", "cc_hhgt": "cc_hhgt_strict"}
DEFAULT_SEEDS = [20260726, 20261726, 20262726]


def safe_metrics(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(labels) & np.isfinite(probability)
    labels = labels[valid]; probability = probability[valid]
    auroc = roc_auc_score(labels, probability) if np.unique(labels).size > 1 else np.nan
    auprc = average_precision_score(labels, probability) if labels.sum() > 0 else np.nan
    return float(auroc), float(auprc)


def expected_jobs(cfg: dict) -> set[tuple[str, str, int]]:
    folds = read_table(cfg["_results"] / "tables" / "fold_manifest.tsv")
    return {
        (str(fold), output_model, seed)
        for fold in folds.fold_id.astype(str).unique()
        for output_model in MODEL_DIRS.values()
        for seed in DEFAULT_SEEDS
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit fixed-sampler V2.9 state retraining")
    parser.add_argument("--config", default="config/model_v2_9_state_graph_local_run.yaml")
    parser.add_argument("--strict-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    strict_root = Path(args.strict_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_status_path = strict_root / "run_control" / "run_status_parallel.json"
    runner_status = (
        json.loads(runner_status_path.read_text(encoding="utf-8"))
        if runner_status_path.exists()
        else {}
    )
    expected_gate_sha256 = str(runner_status.get("formal_training_gate_sha256", ""))

    task_rows: list[dict] = []
    state_rows: list[dict] = []
    observed: set[tuple[str, str, int]] = set()
    invalid: list[dict] = []
    for success_path in sorted(strict_root.glob("*/LOCO_*/seed_*/SUCCESS.json")):
        model_dir = success_path.parent
        try:
            success = json.loads(success_path.read_text(encoding="utf-8"))
            audit = json.loads((model_dir / "state_training_sample_audit.json").read_text(encoding="utf-8"))
            metrics = pd.read_csv(model_dir / "metrics.tsv", sep="\t")
            history = pd.read_csv(model_dir / "training_history.tsv", sep="\t")
            prediction = pd.read_parquet(model_dir / "prediction_state_raw.parquet")
            calibration_exists = (model_dir / "CALIBRATION_SUCCESS.json").exists()
        except Exception as exc:
            invalid.append({"model_dir": str(model_dir), "error": str(exc)})
            continue
        key = (str(success["fold_id"]), str(success["model_name"]), int(success["seed"]))
        training_rows = int(audit["state_training_rows"])
        training_positive = int(audit["state_training_positive"])
        training_unlabeled = int(audit["state_training_unlabeled"])
        training_positive_rate = float(audit["state_training_positive_rate"])
        quality_errors = []
        if training_rows != 150_000:
            quality_errors.append(f"state_training_rows={training_rows}, expected 150000")
        if training_positive <= 0 or training_unlabeled <= 0:
            quality_errors.append(f"P/U classes absent: P={training_positive}, U={training_unlabeled}")
        if not 0.15 <= training_positive_rate <= 0.25:
            quality_errors.append(f"positive_rate={training_positive_rate:.6f}, expected 0.15-0.25")
        if not calibration_exists:
            quality_errors.append("CALIBRATION_SUCCESS.json missing")
        if len(expected_gate_sha256) != 64:
            quality_errors.append("runner formal gate SHA256 is missing")
        elif str(success.get("formal_training_gate_sha256", "")) != expected_gate_sha256:
            quality_errors.append("formal training gate provenance mismatch")
        checkpoint = model_dir / "best.pt"
        if not checkpoint.exists():
            quality_errors.append("best.pt missing")
        elif str(success.get("checkpoint_sha256", "")) != file_sha256(checkpoint):
            quality_errors.append("best.pt SHA256 mismatch")
        if int(audit.get("state_unlabeled_direction_used_in_loss", -1)) != 0:
            quality_errors.append("unlabeled state direction supervision is nonzero")
        for split in ("val", "test"):
            if audit.get(f"state_{split}_evaluation_policy") != ["full_detectable_lncrna_state_universe"]:
                quality_errors.append(f"invalid state {split} evaluation policy")
            pathway_policy = set(map(str, audit.get(f"pathway_{split}_evaluation_policy", [])))
            if not pathway_policy or not pathway_policy.issubset(
                {"full_universe", "label_independent_deterministic"}
            ):
                quality_errors.append(f"invalid pathway {split} evaluation policy")
        if quality_errors:
            invalid.append({"model_dir": str(model_dir), "error": "; ".join(quality_errors)})
            continue
        state_test = metrics.loc[metrics.task.eq("state") & metrics.split.eq("test")]
        pathway_test = metrics.loc[metrics.task.eq("pathway") & metrics.split.eq("test")]
        if len(state_test) != 1 or len(pathway_test) != 1:
            invalid.append({"model_dir": str(model_dir), "error": "missing unique task metrics"})
            continue
        observed.add(key)
        state_test = state_test.iloc[0]; pathway_test = pathway_test.iloc[0]
        task_rows.append(
            {
                "fold_id": key[0],
                "test_cancer": key[0].removeprefix("LOCO_"),
                "model_name": key[1],
                "seed": key[2],
                "epochs_run": len(history),
                "best_epoch": int(success["best_epoch"]),
                "state_training_rows": training_rows,
                "state_training_positive": training_positive,
                "state_training_unlabeled": training_unlabeled,
                "state_training_positive_rate": training_positive_rate,
                "state_test_auroc_pooled": float(state_test.auroc),
                "state_test_auprc_pooled": float(state_test.auprc),
                "pathway_test_auroc": float(pathway_test.auroc),
                "pathway_test_auprc": float(pathway_test.auprc),
                "calibrated": calibration_exists,
                "formal_training_gate_sha256": str(success.get("formal_training_gate_sha256", "")),
                "checkpoint_sha256_verified": True,
                "model_dir": str(model_dir),
            }
        )
        test = prediction.loc[prediction.split.astype(str).eq("test")]
        for state_id, group in test.groupby("state_id", observed=True):
            labels = group.proxy_label.to_numpy(float)
            probability = group.raw_probability.to_numpy(float)
            auroc, auprc = safe_metrics(labels, probability)
            state_rows.append(
                {
                    "fold_id": key[0],
                    "test_cancer": key[0].removeprefix("LOCO_"),
                    "model_name": key[1],
                    "seed": key[2],
                    "state_id": str(state_id),
                    "n": len(group),
                    "n_positive": int(labels.sum()),
                    "positive_rate": float(labels.mean()),
                    "auroc": auroc,
                    "auprc": auprc,
                }
            )

    tasks = pd.DataFrame(task_rows)
    states = pd.DataFrame(state_rows)
    tasks.to_csv(output_dir / "retrained_task_metrics.tsv", sep="\t", index=False)
    states.to_csv(output_dir / "retrained_per_state_metrics.tsv", sep="\t", index=False)
    if not states.empty:
        summary_table = (
            states.groupby(["model_name", "state_id"], observed=True)
            .agg(
                n_runs=("auroc", "size"),
                n_valid=("auroc", "count"),
                median_auroc=("auroc", "median"),
                mean_auroc=("auroc", "mean"),
                std_auroc=("auroc", "std"),
                median_auprc=("auprc", "median"),
            )
            .reset_index()
        )
        summary_table.to_csv(output_dir / "retrained_state_summary.tsv", sep="\t", index=False)

    expected = expected_jobs(cfg)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    payload = {
        "status": "PASS" if not missing and not unexpected and not invalid and len(tasks) == len(expected) else "INCOMPLETE",
        "expected_tasks": len(expected),
        "observed_success_files": len(observed),
        "valid_task_metrics": len(tasks),
        "missing_tasks": [list(item) for item in missing],
        "unexpected_tasks": [list(item) for item in unexpected],
        "invalid_tasks": invalid,
        "strict_root": str(strict_root),
        "formal_training_gate_sha256": expected_gate_sha256,
        "formal_training_gate_provenance_verified": bool(expected_gate_sha256) and not any(
            "formal training gate" in str(item.get("error", ""))
            for item in invalid
        ),
    }
    write_json(payload, output_dir / "retrained_matrix_audit.json")
    console_payload = {
        **{key: value for key, value in payload.items() if key not in {"missing_tasks", "unexpected_tasks", "invalid_tasks"}},
        "missing_task_count": len(missing),
        "missing_tasks_preview": [list(item) for item in missing[:20]],
        "unexpected_task_count": len(unexpected),
        "unexpected_tasks_preview": [list(item) for item in unexpected[:20]],
        "invalid_task_count": len(invalid),
        "invalid_tasks_preview": invalid[:20],
    }
    print(json.dumps(console_payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
