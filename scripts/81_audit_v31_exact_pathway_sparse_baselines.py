#!/usr/bin/env python3
"""Audit and ensemble all 99 matched V3.1 LASSO/Ridge baseline tasks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.common import file_sha256, read_table, write_table
from cc_hhgt.prediction_contract import candidate_universe_sha256
from cc_hhgt.v30_integrity import atomic_write_json, canonical_json_sha256
from cc_hhgt.v31_postmatrix import (
    EXACT_IDENTITY,
    align_exact_frames,
    assert_metric_close,
    binary_metrics,
)


SEEDS = (20260726, 20261726, 20262726)
MODELS = ("lasso", "ridge")
EXPECTED_FOLDS = 33
EXPECTED_TASKS = EXPECTED_FOLDS * len(SEEDS)
PREDICTION = "prediction_exact_pathway_sparse_baselines.parquet"
PROTOCOL = "V3_1_EXACT_PATHWAY_BASELINE_PROTOCOL.md"


def _require_matrix_audit(path: Path, run_id: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "PASS"
        or payload.get("run_id") != run_id
        or int(payload.get("verified_tasks", -1)) != 297
        or not bool(payload.get("release_eligible"))
        or not bool(payload.get("all_tasks_converged_before_hard_epoch_cap"))
        or not bool(payload.get("candidate_keys_and_labels_identical_across_model_seed"))
    ):
        raise RuntimeError("Baseline audit requires the release-eligible 297-task matrix audit")
    return payload


def _prediction(task_root: Path) -> pd.DataFrame:
    required = set(EXACT_IDENTITY + [
        "label_class", "direction", "split", "lasso_probability", "ridge_probability"
    ])
    frame = pd.read_parquet(task_root / PREDICTION)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Sparse baseline prediction lacks columns {missing}: {task_root}")
    if set(frame.split.astype(str)) != {"val", "test"}:
        raise RuntimeError(f"Sparse baseline task lacks exact validation/test splits: {task_root}")
    for model in MODELS:
        probability = pd.to_numeric(frame[f"{model}_probability"], errors="coerce").to_numpy(float)
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise RuntimeError(f"Invalid {model} probabilities: {task_root}")
    return frame


def _verify_task(
    task_root: Path,
    graph_task: Path,
    run_id: str,
    fold: str,
    seed: int,
    matrix_audit_sha256: str,
    protocol_sha256: str,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    success_path = task_root / "SUCCESS.json"
    prediction_path = task_root / PREDICTION
    metrics_path = task_root / "metrics.tsv"
    for path in (success_path, prediction_path, metrics_path):
        if not path.is_file():
            raise RuntimeError(f"Missing sparse baseline task artifact: {path}")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if (
        success.get("status") != "PASS"
        or success.get("run_id") != run_id
        or success.get("fold_id") != fold
        or int(success.get("seed", -1)) != seed
        or success.get("target_level") != "exact_pathway"
        or success.get("models") != list(MODELS)
        or success.get("matrix_audit_sha256") != matrix_audit_sha256
        or success.get("baseline_protocol_sha256") != protocol_sha256
        or bool(success.get("test_used_for_tuning"))
        or bool(success.get("pair_evidence_features_used"))
    ):
        raise RuntimeError(f"Invalid sparse baseline SUCCESS contract: {task_root}")
    if success.get("prediction_sha256") != file_sha256(prediction_path):
        raise RuntimeError(f"Sparse baseline prediction hash mismatch: {task_root}")

    graph_prediction = graph_task / "prediction_pathway_calibrated.parquet"
    if success.get("graph_reference_prediction_sha256") != file_sha256(graph_prediction):
        raise RuntimeError(f"Sparse baseline graph-reference hash mismatch: {task_root}")
    frame = _prediction(task_root)
    metric_frame = read_table(metrics_path)
    verified_metrics: list[dict] = []
    for split in ("val", "test"):
        current = frame.loc[frame.split.astype(str).eq(split)].copy()
        reference = pd.read_parquet(
            graph_prediction,
            columns=EXACT_IDENTITY + ["split"],
            filters=[("split", "==", split)],
        )
        align_exact_frames([current, reference])
        digest = candidate_universe_sha256(current, EXACT_IDENTITY)
        registered = success[
            "validation_candidate_label_sha256" if split == "val" else "test_candidate_label_sha256"
        ]
        if digest != registered:
            raise RuntimeError(f"Sparse baseline {split} identity hash mismatch: {task_root}")
        for model in MODELS:
            recomputed = binary_metrics(current.proxy_label, current[f"{model}_probability"])
            hit = metric_frame.loc[
                metric_frame.model.astype(str).eq(model)
                & metric_frame.split.astype(str).eq(split)
            ]
            if len(hit) != 1:
                raise RuntimeError(f"Expected one registered metric row for {model}/{split}: {task_root}")
            row = hit.iloc[0]
            for metric in ("n", "positive_prevalence", "auroc", "auprc", "auprc_lift"):
                assert_metric_close(row[metric], recomputed[metric], f"{fold}/{seed}/{model}/{split}/{metric}")
            verified_metrics.append(
                {"fold_id": fold, "seed": seed, "model": model, "split": split, **recomputed}
            )
    audit = {
        "fold_id": fold,
        "seed": seed,
        "status": "PASS",
        "rows": int(len(frame)),
        "prediction_sha256": file_sha256(prediction_path),
        "training_lineage_gate_sha256": success.get("training_lineage_gate_sha256"),
        "execution_site_gate_sha256": success.get("execution_site_gate_sha256"),
        "relocated_execution": bool(success.get("relocated_execution")),
    }
    return audit, frame, verified_metrics


def _ensemble(frames: list[pd.DataFrame], split: str) -> pd.DataFrame:
    selected = [frame.loc[frame.split.astype(str).eq(split)].copy() for frame in frames]
    identity, ordered = align_exact_frames(selected)
    out = identity.copy()
    out["split"] = split
    for model in MODELS:
        values = np.vstack(
            [pd.to_numeric(frame[f"{model}_probability"], errors="raise").to_numpy(float) for frame in ordered]
        )
        out[f"{model}_probability"] = values.mean(axis=0)
        out[f"{model}_seed_std"] = values.std(axis=0)
    out["seed_count"] = len(frames)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--matrix-audit", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    training_root = Path(args.training_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    output_root = Path(args.output_root).resolve()
    matrix_audit_path = Path(args.matrix_audit).resolve()
    matrix_audit = _require_matrix_audit(matrix_audit_path, args.run_id)
    matrix_audit_sha256 = file_sha256(matrix_audit_path)
    protocol_path = Path(args.repo_root).resolve() / PROTOCOL
    protocol_sha256 = file_sha256(protocol_path)
    if output_root.exists():
        raise RuntimeError(f"Sparse baseline audit refuses reuse: {output_root}")

    folds = sorted(path.name for path in (training_root / "cc_hhgt").glob("LOCO_*") if path.is_dir())
    if len(folds) != EXPECTED_FOLDS:
        raise RuntimeError(f"Expected {EXPECTED_FOLDS} formal folds, observed {len(folds)}")
    temporary = output_root.parent / f".{output_root.name}.tmp"
    if temporary.exists():
        raise RuntimeError(f"Stale sparse baseline audit temporary root: {temporary}")
    temporary.mkdir(parents=True)

    task_rows: list[dict] = []
    metric_rows: list[dict] = []
    ensemble_metric_rows: list[dict] = []
    ensemble_files: list[dict] = []
    lineage_hashes: set[str] = set()
    for fold in folds:
        seed_frames: list[pd.DataFrame] = []
        for seed in SEEDS:
            task_root = baseline_root / fold / f"seed_{seed}"
            graph_task = training_root / "cc_hhgt" / fold / f"seed_{seed}"
            task_audit, frame, verified = _verify_task(
                task_root,
                graph_task,
                args.run_id,
                fold,
                seed,
                matrix_audit_sha256,
                protocol_sha256,
            )
            task_rows.append(task_audit)
            metric_rows.extend(verified)
            seed_frames.append(frame)
            lineage_hashes.add(str(task_audit["training_lineage_gate_sha256"]))
        for split in ("val", "test"):
            ensemble = _ensemble(seed_frames, split)
            for model in MODELS:
                ensemble_metric_rows.append(
                    {
                        "fold_id": fold,
                        "model": model,
                        "split": split,
                        **binary_metrics(ensemble.proxy_label, ensemble[f"{model}_probability"]),
                    }
                )
            if split == "test":
                path = temporary / "test_seed_ensemble" / f"{fold}.parquet"
                write_table(ensemble, path)
                ensemble_files.append(
                    {"relative_path": str(path.relative_to(temporary)).replace("\\", "/"), "sha256": file_sha256(path), "rows": len(ensemble)}
                )

    if len(task_rows) != EXPECTED_TASKS or len(lineage_hashes) != 1 or "None" in lineage_hashes:
        raise RuntimeError(
            f"Incomplete or mixed sparse baseline lineage: tasks={len(task_rows)}, lineage={sorted(lineage_hashes)}"
        )
    write_table(pd.DataFrame(task_rows), temporary / "task_audit.tsv")
    write_table(pd.DataFrame(metric_rows), temporary / "task_metrics_recomputed.tsv")
    write_table(pd.DataFrame(ensemble_metric_rows), temporary / "seed_ensemble_fold_metrics.tsv")
    payload = {
        "status": "PASS",
        "run_id": args.run_id,
        "target_level": "exact_pathway",
        "models": list(MODELS),
        "folds": len(folds),
        "seeds": list(SEEDS),
        "expected_tasks": EXPECTED_TASKS,
        "verified_tasks": len(task_rows),
        "test_used_for_tuning": False,
        "pair_evidence_features_used": False,
        "candidate_rows_matched_to_graph": True,
        "training_lineage_gate_sha256": next(iter(lineage_hashes)),
        "matrix_audit_sha256": matrix_audit_sha256,
        "matrix_audit_release_eligible": bool(matrix_audit["release_eligible"]),
        "baseline_protocol_sha256": protocol_sha256,
        "test_seed_ensemble_files": ensemble_files,
        "test_seed_ensemble_merkle_sha256": canonical_json_sha256(ensemble_files),
        "release_eligible": True,
    }
    atomic_write_json(temporary / "AUDIT.json", payload)
    atomic_write_json(temporary / "SUCCESS.json", payload)
    os.replace(temporary, output_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
