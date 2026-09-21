#!/usr/bin/env python3
"""Compare the validation-selected deep model with matched LASSO and Ridge."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.common import file_sha256, write_table
from cc_hhgt.v30_integrity import atomic_write_json, canonical_json_sha256
from cc_hhgt.v31_postmatrix import EXACT_IDENTITY, align_exact_frames, binary_metrics


DEEP_MODELS = ("rgcn", "hgt", "cc_hhgt")
BASELINES = ("lasso", "ridge")
SEEDS = (20260726, 20261726, 20262726)
EXPECTED_FOLDS = 33
PROTOCOL = "V3_1_EXACT_PATHWAY_POSTMATRIX_COMPARISON_PROTOCOL.md"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_inputs(
    run_id: str,
    matrix_audit_path: Path,
    selection_path: Path,
    baseline_audit_path: Path,
) -> tuple[dict, dict, dict]:
    matrix = _load_json(matrix_audit_path)
    selection = _load_json(selection_path)
    baseline = _load_json(baseline_audit_path)
    matrix_sha = file_sha256(matrix_audit_path)
    if (
        matrix.get("status") != "PASS"
        or matrix.get("run_id") != run_id
        or int(matrix.get("verified_tasks", -1)) != 297
        or not bool(matrix.get("release_eligible"))
        or not bool(matrix.get("all_tasks_converged_before_hard_epoch_cap"))
    ):
        raise RuntimeError("Comparison requires the release-eligible 297-task matrix audit")
    if (
        selection.get("status") != "PASS"
        or selection.get("run_id") != run_id
        or selection.get("selected_model") not in DEEP_MODELS
        or selection.get("selection_split") != "val_only"
        or bool(selection.get("test_metrics_used"))
        or selection.get("matrix_audit_sha256") != matrix_sha
    ):
        raise RuntimeError("Comparison requires a validation-only website architecture selection")
    if (
        baseline.get("status") != "PASS"
        or baseline.get("run_id") != run_id
        or int(baseline.get("verified_tasks", -1)) != EXPECTED_FOLDS * len(SEEDS)
        or not bool(baseline.get("release_eligible"))
        or baseline.get("matrix_audit_sha256") != matrix_sha
        or not bool(baseline.get("candidate_rows_matched_to_graph"))
        or bool(baseline.get("test_used_for_tuning"))
    ):
        raise RuntimeError("Comparison requires the release-eligible matched sparse-baseline audit")
    return matrix, selection, baseline


def _deep_ensemble(
    training_root: Path,
    run_id: str,
    model: str,
    fold: str,
    training_gate_sha256: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        task = training_root / model / fold / f"seed_{seed}"
        success = _load_json(task / "SUCCESS.json")
        if (
            success.get("status") != "COMPLETED"
            or success.get("run_id") != run_id
            or success.get("formal_training_gate_sha256") != training_gate_sha256
            or bool(success.get("hit_hard_epoch_cap"))
            or not bool(success.get("stopped_by_joint_patience"))
        ):
            raise RuntimeError(f"Ineligible selected-deep task: {task}")
        frame = pd.read_parquet(
            task / "prediction_pathway_calibrated.parquet",
            columns=EXACT_IDENTITY + ["split", "proxy_positive_probability"],
            filters=[("split", "==", "test")],
        )
        if frame.empty or set(frame.split.astype(str)) != {"test"}:
            raise RuntimeError(f"Selected-deep task lacks test rows: {task}")
        frames.append(frame)
    identity, ordered = align_exact_frames(frames)
    scores = np.vstack(
        [pd.to_numeric(frame.proxy_positive_probability, errors="raise").to_numpy(float) for frame in ordered]
    )
    out = identity.copy()
    out["deep_probability"] = scores.mean(axis=0)
    out["deep_seed_std"] = scores.std(axis=0)
    return out


def _family_metrics(matched: pd.DataFrame, fold: str, model: str, score: str) -> list[dict]:
    rows: list[dict] = []
    for family, group in matched.groupby("pathway_family_id", observed=True, sort=True):
        metrics = binary_metrics(group.proxy_label, group[score], allow_single_class=True)
        rows.append({"fold_id": fold, "pathway_family_id": str(family), "model": model, **metrics})
    return rows


def _delta_summary(fold_metrics: pd.DataFrame, key_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_columns = ["auroc", "auprc", "auprc_lift"]
    eligible = fold_metrics.loc[fold_metrics.both_classes.astype(bool)].copy()
    pivot = eligible.pivot(index=key_columns, columns="model", values=value_columns)
    delta_rows: list[dict] = []
    for baseline in BASELINES:
        for index, row in pivot.iterrows():
            index_values = index if isinstance(index, tuple) else (index,)
            record = dict(zip(key_columns, index_values))
            record["baseline"] = baseline
            for metric in value_columns:
                record[f"delta_{metric}"] = float(row[(metric, "selected_deep")] - row[(metric, baseline)])
            delta_rows.append(record)
    deltas = pd.DataFrame(delta_rows)

    group_columns = [column for column in key_columns if column != "fold_id"] + ["baseline"]
    summary_rows: list[dict] = []
    grouped = deltas.groupby(group_columns, observed=True, sort=True)
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_columns, keys))
        record["eligible_folds"] = int(group.fold_id.nunique())
        for metric in value_columns:
            values = pd.to_numeric(group[f"delta_{metric}"], errors="coerce").dropna()
            record[f"median_delta_{metric}"] = float(values.median())
            record[f"mean_delta_{metric}"] = float(values.mean())
            record[f"fraction_positive_delta_{metric}"] = float(values.gt(0).mean())
        eligible_count = int(record["eligible_folds"])
        median_lift = float(record["median_delta_auprc_lift"])
        fraction_lift = float(record["fraction_positive_delta_auprc_lift"])
        record["stable_deep_benefit"] = bool(
            eligible_count >= 5 and median_lift > 0 and fraction_lift >= 2 / 3
        )
        record["stable_baseline_benefit"] = bool(
            eligible_count >= 5 and median_lift < 0 and fraction_lift <= 1 / 3
        )
        summary_rows.append(record)
    return deltas, pd.DataFrame(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--matrix-audit", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--baseline-audit-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    training_root = Path(args.training_root).resolve()
    matrix_audit_path = Path(args.matrix_audit).resolve()
    selection_path = Path(args.selection).resolve()
    baseline_audit_root = Path(args.baseline_audit_root).resolve()
    baseline_audit_path = baseline_audit_root / "AUDIT.json"
    _, selection, baseline_audit = _require_inputs(
        args.run_id, matrix_audit_path, selection_path, baseline_audit_path
    )
    selected = str(selection["selected_model"])
    training_gate_sha256 = str(baseline_audit["training_lineage_gate_sha256"])
    protocol_path = Path(args.repo_root).resolve() / PROTOCOL
    protocol_sha256 = file_sha256(protocol_path)
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"Selected-model comparison refuses reuse: {output_root}")
    temporary = output_root.parent / f".{output_root.name}.tmp"
    if temporary.exists():
        raise RuntimeError(f"Stale selected-model comparison temporary root: {temporary}")
    temporary.mkdir(parents=True)

    folds = sorted(path.name for path in (training_root / selected).glob("LOCO_*") if path.is_dir())
    if len(folds) != EXPECTED_FOLDS:
        raise RuntimeError(f"Expected {EXPECTED_FOLDS} selected-model folds, observed {len(folds)}")
    fold_metric_rows: list[dict] = []
    family_metric_rows: list[dict] = []
    matched_files: list[dict] = []
    for fold in folds:
        deep = _deep_ensemble(
            training_root, args.run_id, selected, fold, training_gate_sha256
        )
        baseline_path = baseline_audit_root / "test_seed_ensemble" / f"{fold}.parquet"
        sparse = pd.read_parquet(baseline_path)
        identity, aligned = align_exact_frames([deep, sparse])
        deep_aligned, sparse_aligned = aligned
        matched = identity.copy()
        matched["selected_deep_probability"] = deep_aligned.deep_probability.to_numpy(float)
        matched["selected_deep_seed_std"] = deep_aligned.deep_seed_std.to_numpy(float)
        for baseline in BASELINES:
            matched[f"{baseline}_probability"] = sparse_aligned[f"{baseline}_probability"].to_numpy(float)
            matched[f"{baseline}_seed_std"] = sparse_aligned[f"{baseline}_seed_std"].to_numpy(float)
        score_columns = {
            "selected_deep": "selected_deep_probability",
            "lasso": "lasso_probability",
            "ridge": "ridge_probability",
        }
        for model, score in score_columns.items():
            fold_metric_rows.append(
                {"fold_id": fold, "model": model, **binary_metrics(matched.proxy_label, matched[score])}
            )
            family_metric_rows.extend(_family_metrics(matched, fold, model, score))
        path = temporary / "matched_test_predictions" / f"{fold}.parquet"
        write_table(matched, path)
        matched_files.append(
            {"relative_path": str(path.relative_to(temporary)).replace("\\", "/"), "sha256": file_sha256(path), "rows": len(matched)}
        )

    fold_metrics = pd.DataFrame(fold_metric_rows)
    family_metrics = pd.DataFrame(family_metric_rows)
    overall_deltas, overall_summary = _delta_summary(fold_metrics, ["fold_id"])
    family_deltas, family_summary = _delta_summary(
        family_metrics, ["fold_id", "pathway_family_id"]
    )
    model_summary = (
        fold_metrics.groupby("model", observed=True)
        .agg(
            folds=("fold_id", "nunique"),
            median_auroc=("auroc", "median"),
            mean_auroc=("auroc", "mean"),
            median_auprc=("auprc", "median"),
            mean_auprc=("auprc", "mean"),
            median_auprc_lift=("auprc_lift", "median"),
            mean_auprc_lift=("auprc_lift", "mean"),
        )
        .reset_index()
    )
    write_table(fold_metrics, temporary / "heldout_test_fold_metrics.tsv")
    write_table(model_summary, temporary / "heldout_test_model_summary.tsv")
    write_table(overall_deltas, temporary / "heldout_test_fold_deltas.tsv")
    write_table(overall_summary, temporary / "heldout_test_delta_summary.tsv")
    write_table(family_metrics, temporary / "pathway_family_fold_metrics.tsv")
    write_table(family_deltas, temporary / "pathway_family_fold_deltas.tsv")
    write_table(family_summary, temporary / "pathway_family_delta_summary.tsv")
    payload = {
        "status": "PASS",
        "run_id": args.run_id,
        "selected_deep_model": selected,
        "selection_split": "val_only",
        "test_metrics_used_for_selection": False,
        "baseline_models": list(BASELINES),
        "primary_article_baseline": "lasso",
        "folds": len(folds),
        "seeds": list(SEEDS),
        "pathway_target_level": "exact_pathway",
        "pathway_family_role": "stratified_interpretation_only",
        "stable_family_minimum_eligible_folds": 5,
        "stable_family_positive_fraction_threshold": 2 / 3,
        "matrix_audit_sha256": file_sha256(matrix_audit_path),
        "selection_sha256": file_sha256(selection_path),
        "baseline_audit_sha256": file_sha256(baseline_audit_path),
        "comparison_protocol_sha256": protocol_sha256,
        "matched_test_files": matched_files,
        "matched_test_merkle_sha256": canonical_json_sha256(matched_files),
        "release_eligible": True,
    }
    atomic_write_json(temporary / "REPORT_AUDIT.json", payload)
    atomic_write_json(temporary / "SUCCESS.json", payload)
    os.replace(temporary, output_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
