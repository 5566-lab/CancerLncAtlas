#!/usr/bin/env python3
"""Select the V3.1 website architecture from validation predictions only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from cc_hhgt.common import file_sha256
from cc_hhgt.v30_integrity import atomic_write_json


MODELS = ("rgcn", "hgt", "cc_hhgt")
SEEDS = (20260726, 20261726, 20262726)
EXPECTED_FOLDS = 33
EXPECTED_TASKS = EXPECTED_FOLDS * len(MODELS) * len(SEEDS)
TIE_ORDER = {"cc_hhgt": 0, "hgt": 1, "rgcn": 2}
PROTOCOL = "V3_1_EXACT_PATHWAY_WEBSITE_SELECTION_PROTOCOL.md"


def _load_runner_summary(training_root: Path, run_id: str) -> dict:
    path = training_root / "run_control" / "PARALLEL_SUMMARY.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_modes = {f"FULL_{EXPECTED_TASKS}", f"FULL_{EXPECTED_TASKS}_MULTIGPU"}
    if (
        payload.get("status") != "PASS"
        or payload.get("run_id") != run_id
        or int(payload.get("n_tasks", -1)) != EXPECTED_TASKS
        or int(payload.get("completed", -1)) != EXPECTED_TASKS
        or payload.get("mode") not in expected_modes
    ):
        raise RuntimeError("Website selection requires the completed formal 297-task matrix")
    return payload


def _load_matrix_audit(path: Path, run_id: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "PASS"
        or payload.get("run_id") != run_id
        or int(payload.get("expected_tasks", -1)) != EXPECTED_TASKS
        or int(payload.get("verified_tasks", -1)) != EXPECTED_TASKS
        or not bool(payload.get("release_eligible"))
        or not bool(payload.get("all_tasks_converged_before_hard_epoch_cap"))
        or not bool(payload.get("candidate_keys_and_labels_identical_across_model_seed"))
    ):
        raise RuntimeError("Website selection requires the release-eligible 297-task matrix audit")
    return payload


def _validation_frame(task_root: Path, run_id: str, model: str, fold: str, seed: int) -> pd.DataFrame:
    success = json.loads((task_root / "SUCCESS.json").read_text(encoding="utf-8"))
    if (
        success.get("status") != "COMPLETED"
        or success.get("run_id") != run_id
        or bool(success.get("hit_hard_epoch_cap"))
    ):
        raise RuntimeError(f"Ineligible formal task: {task_root}")
    required = {
        "candidate_id", "cancer_id", "lncrna_id", "pathway_id",
        "pathway_family_id", "proxy_label", "split",
        "proxy_positive_probability",
    }
    prediction_path = task_root / "prediction_pathway_calibrated.parquet"
    frame = pd.read_parquet(
        prediction_path,
        columns=sorted(required),
        filters=[("split", "==", "val")],
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Validation prediction lacks exact-pathway columns: {missing}")
    frame = frame[list(required)].copy()
    if frame.empty or set(frame.split.astype(str)) != {"val"}:
        raise RuntimeError(f"Task has no registered validation rows: {task_root}")
    if frame.candidate_id.astype(str).duplicated().any():
        raise RuntimeError(f"Duplicate validation candidate: {task_root}")
    frame["fold_id"] = fold
    frame["model"] = model
    frame["seed"] = seed
    return frame


def _fold_metric(group: pd.DataFrame) -> dict:
    labels = pd.to_numeric(group.proxy_label, errors="coerce").to_numpy(float)
    scores = pd.to_numeric(group.ensemble_probability, errors="coerce").to_numpy(float)
    valid = np.isfinite(labels) & np.isfinite(scores)
    labels = labels[valid].astype(int)
    scores = scores[valid]
    if len(labels) == 0 or np.unique(labels).size != 2:
        raise RuntimeError("Validation fold does not contain both proxy classes")
    if np.unique(np.round(scores, 12)).size <= 1 or float(np.std(scores)) < 1e-8:
        raise RuntimeError("Validation seed ensemble is collapsed")
    prevalence = float(labels.mean())
    auprc = float(average_precision_score(labels, scores))
    return {
        "n_candidates": int(len(labels)),
        "n_positive": int(labels.sum()),
        "positive_prevalence": prevalence,
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": auprc,
        "auprc_lift": auprc - prevalence,
    }


def select_architecture(validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ensemble_rows: list[pd.DataFrame] = []
    expected_fold_ids = sorted(validation.fold_id.astype(str).unique())
    if len(expected_fold_ids) != EXPECTED_FOLDS:
        raise RuntimeError(f"Expected {EXPECTED_FOLDS} validation folds, observed {len(expected_fold_ids)}")
    for fold in expected_fold_ids:
        fold_frame = validation.loc[validation.fold_id.astype(str).eq(fold)].copy()
        reference: pd.DataFrame | None = None
        for model in MODELS:
            model_frame = fold_frame.loc[fold_frame.model.astype(str).eq(model)].copy()
            if set(model_frame.seed.astype(int)) != set(SEEDS):
                raise RuntimeError(f"Incomplete seed coverage for {fold}/{model}")
            pivot = model_frame.pivot(
                index=["candidate_id", "cancer_id", "lncrna_id", "pathway_id", "pathway_family_id", "proxy_label"],
                columns="seed",
                values="proxy_positive_probability",
            ).reset_index()
            if len(pivot) * len(SEEDS) != len(model_frame) or any(seed not in pivot for seed in SEEDS):
                raise RuntimeError(f"Candidate/seed matrix is incomplete for {fold}/{model}")
            current_identity = pivot[
                ["candidate_id", "cancer_id", "lncrna_id", "pathway_id", "pathway_family_id", "proxy_label"]
            ].sort_values("candidate_id").reset_index(drop=True)
            if reference is None:
                reference = current_identity
            elif not current_identity.equals(reference):
                raise RuntimeError(f"Candidate identities or labels drift across models in {fold}")
            out = current_identity.copy()
            out["fold_id"] = fold
            out["model"] = model
            out["seed_count"] = len(SEEDS)
            out["ensemble_probability"] = pivot[list(SEEDS)].mean(axis=1).to_numpy(float)
            ensemble_rows.append(out)
    ensemble = pd.concat(ensemble_rows, ignore_index=True)

    metric_rows = []
    for (fold, model), group in ensemble.groupby(["fold_id", "model"], observed=True):
        metric_rows.append({"fold_id": fold, "model": model, **_fold_metric(group)})
    metrics = pd.DataFrame(metric_rows)
    coverage = metrics.groupby("model", observed=True).fold_id.nunique().to_dict()
    if coverage != {model: EXPECTED_FOLDS for model in MODELS}:
        raise RuntimeError(f"Architecture validation coverage is incomplete: {coverage}")

    summary = (
        metrics.groupby("model", observed=True)
        .agg(
            folds=("fold_id", "nunique"),
            median_validation_auprc_lift=("auprc_lift", "median"),
            mean_validation_auprc_lift=("auprc_lift", "mean"),
            median_validation_auroc=("auroc", "median"),
            mean_validation_auroc=("auroc", "mean"),
        )
        .reset_index()
    )
    summary["tie_order"] = summary.model.map(TIE_ORDER)
    order = [
        "median_validation_auprc_lift", "mean_validation_auprc_lift",
        "median_validation_auroc", "mean_validation_auroc",
    ]
    summary = summary.sort_values(order + ["tie_order"], ascending=[False] * 4 + [True]).reset_index(drop=True)
    selected = str(summary.iloc[0].model)
    exact_tie = bool(
        len(summary) > 1
        and all(float(summary.iloc[0][column]) == float(summary.iloc[1][column]) for column in order)
    )
    selection = {
        "selected_model": selected,
        "selection_split": "val_only",
        "test_metrics_used": False,
        "seed_ensemble_size": len(SEEDS),
        "folds": EXPECTED_FOLDS,
        "primary_criterion": "median_validation_auprc_lift",
        "criterion_order": order,
        "exact_tie_resolved": exact_tie,
        "tie_order": list(TIE_ORDER),
    }
    return metrics, summary.drop(columns="tie_order"), selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--matrix-audit", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    training_root = Path(args.training_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"Website architecture selection refuses reuse: {output_root}")
    protocol_path = repo_root / PROTOCOL
    if not protocol_path.is_file():
        raise RuntimeError(f"Missing pre-registered selection protocol: {protocol_path}")
    runner = _load_runner_summary(training_root, args.run_id)
    matrix_audit_path = Path(args.matrix_audit).resolve()
    matrix_audit = _load_matrix_audit(matrix_audit_path, args.run_id)

    frames = []
    for model in MODELS:
        for task_root in sorted((training_root / model).glob("LOCO_*/seed_*")):
            fold = task_root.parent.name
            seed = int(task_root.name.removeprefix("seed_"))
            if seed in SEEDS:
                frames.append(_validation_frame(task_root, args.run_id, model, fold, seed))
    if len(frames) != EXPECTED_TASKS:
        raise RuntimeError(f"Expected {EXPECTED_TASKS} task validation files, observed {len(frames)}")
    validation = pd.concat(frames, ignore_index=True)
    metrics, summary, selection = select_architecture(validation)

    temporary = output_root.parent / f".{output_root.name}.tmp"
    if temporary.exists():
        raise RuntimeError(f"Stale website-selection temporary root: {temporary}")
    temporary.mkdir(parents=True)
    metrics.to_csv(temporary / "validation_seed_ensemble_fold_metrics.tsv", sep="\t", index=False)
    summary.to_csv(temporary / "architecture_validation_summary.tsv", sep="\t", index=False)
    payload = {
        "status": "PASS",
        "run_id": args.run_id,
        **selection,
        "formal_runner_summary_sha256": file_sha256(training_root / "run_control" / "PARALLEL_SUMMARY.json"),
        "matrix_audit_sha256": file_sha256(matrix_audit_path),
        "selection_protocol": str(protocol_path),
        "selection_protocol_sha256": file_sha256(protocol_path),
        "architecture_summary": summary.to_dict("records"),
        "formal_runner_mode": runner["mode"],
        "matrix_audit_release_eligible": bool(matrix_audit["release_eligible"]),
    }
    atomic_write_json(temporary / "WEBSITE_MODEL_SELECTION.json", payload)
    atomic_write_json(temporary / "SUCCESS.json", payload)
    os.replace(temporary, output_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
