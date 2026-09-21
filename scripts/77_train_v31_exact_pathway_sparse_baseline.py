#!/usr/bin/env python3
"""Fit one matched LOCO LASSO/Ridge exact-pathway baseline task."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from cc_hhgt.common import file_sha256, load_config, read_table, write_table
from cc_hhgt.formal_gate import validate_formal_training_gate
from cc_hhgt.pathway_target import require_exact_pathway_contract
from cc_hhgt.prediction_contract import candidate_universe_sha256
from cc_hhgt.relocated_lineage import validate_relocated_training_lineage
from cc_hhgt.training_data import split_for_fold
from cc_hhgt.v30_integrity import atomic_write_json
from cc_hhgt.v31_exact_baselines import (
    _binary_metrics,
    fit_sparse_logistic_baseline,
)


SEEDS = (20260726, 20261726, 20262726)
EXPECTED_TASKS = 297
C_GRID = (0.01, 0.1, 1.0, 10.0)
PROTOCOL = "V3_1_EXACT_PATHWAY_BASELINE_PROTOCOL.md"
IDENTITY = [
    "candidate_id",
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
    "proxy_label",
]


def _require_matrix_audit(path: Path, run_id: str) -> dict:
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
        raise RuntimeError("Sparse baselines require the release-eligible audited 297-task matrix")
    return payload


def _identity_sha256(frame: pd.DataFrame) -> str:
    return candidate_universe_sha256(frame, IDENTITY)


def _reference_split(task_root: Path, split: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        task_root / "prediction_pathway_calibrated.parquet",
        columns=IDENTITY + ["split"],
        filters=[("split", "==", split)],
    )
    if frame.empty or set(frame.split.astype(str)) != {split}:
        raise RuntimeError(f"Graph reference lacks {split} rows: {task_root}")
    if frame.candidate_id.astype(str).duplicated().any():
        raise RuntimeError(f"Graph reference has duplicate {split} candidates")
    return frame


def _assert_matched(frame: pd.DataFrame, reference: pd.DataFrame, split: str) -> str:
    observed = _identity_sha256(frame)
    expected = _identity_sha256(reference)
    if observed != expected or len(frame) != len(reference):
        raise RuntimeError(
            f"Sparse baseline {split} candidate/label identity differs from graph task: "
            f"rows={len(frame)}/{len(reference)}, sha={observed}/{expected}"
        )
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--formal-gate", required=True)
    parser.add_argument(
        "--training-lineage-gate",
        help="Original training-site gate when baselines run on a verified relocation.",
    )
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--matrix-audit", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()

    training_root = Path(args.training_root).resolve()
    output_root = Path(args.output_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    protocol_path = repo_root / PROTOCOL
    if not protocol_path.is_file():
        raise RuntimeError(f"Missing frozen baseline protocol: {protocol_path}")
    matrix_audit_path = Path(args.matrix_audit).resolve()
    _require_matrix_audit(matrix_audit_path, args.run_id)

    gate_path = Path(args.formal_gate).resolve()
    gate_document = json.loads(gate_path.read_text(encoding="utf-8"))
    cfg = load_config(
        args.config,
        project_root_override=Path(gate_document["paths"]["run_root"]).resolve(),
        create_dirs=False,
    )
    require_exact_pathway_contract(cfg)
    gate, execution_gate_sha256 = validate_formal_training_gate(
        gate_path,
        cfg,
        verify_assets=True,
        run_id=args.run_id,
        input_root=gate_document["paths"]["input_root"],
        input_manifest=gate_document["paths"]["input_manifest"],
        output_root=training_root,
    )
    lineage_gate_path = args.training_lineage_gate or args.formal_gate
    _, training_gate_sha256, normalized_gate_contract_sha256 = (
        validate_relocated_training_lineage(lineage_gate_path, gate)
    )
    cfg["_results"] = Path(gate["paths"]["asset_results"]).resolve()
    cfg["_standardized"] = cfg["_results"].parent / "standardized"
    cfg["_cache"] = cfg["_results"].parent / "cache"
    if not bool(
        cfg.get("state_training", {}).get(
            "mask_pair_evidence_for_all_pathway_models", False
        )
    ):
        raise RuntimeError("Matched sparse baselines require the primary complete pair-evidence mask")

    folds = read_table(cfg["_results"] / "tables" / "fold_manifest.tsv")
    hit = folds.loc[folds.fold_id.astype(str).eq(args.fold)]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one registered fold {args.fold}, observed {len(hit)}")
    fold_row = pd.Series(hit.iloc[0].to_dict())
    fold_row["split_seed"] = int(args.seed)
    train, validation, test = split_for_fold(cfg, fold_row)

    graph_task = training_root / "cc_hhgt" / args.fold / f"seed_{args.seed}"
    graph_success = json.loads((graph_task / "SUCCESS.json").read_text(encoding="utf-8"))
    if (
        graph_success.get("status") != "COMPLETED"
        or graph_success.get("run_id") != args.run_id
        or graph_success.get("formal_training_gate_sha256") != training_gate_sha256
    ):
        raise RuntimeError("Graph identity reference has invalid formal lineage")
    validation_reference = _reference_split(graph_task, "val")
    test_reference = _reference_split(graph_task, "test")
    validation_sha = _assert_matched(validation, validation_reference, "validation")
    test_sha = _assert_matched(test, test_reference, "test")

    fits = {
        "lasso": fit_sparse_logistic_baseline(
            train,
            validation,
            test,
            penalty="l1",
            c_grid=C_GRID,
            seed=args.seed,
            weak_positive_weight=float(cfg["training"]["weak_positive_weight"]),
        ),
        "ridge": fit_sparse_logistic_baseline(
            train,
            validation,
            test,
            penalty="l2",
            c_grid=C_GRID,
            seed=args.seed,
            weak_positive_weight=float(cfg["training"]["weak_positive_weight"]),
        ),
    }

    prediction_parts = []
    metric_rows = []
    for split, frame in (("val", validation), ("test", test)):
        part = frame[IDENTITY + ["label_class", "direction"]].copy()
        part["split"] = split
        for name, fit in fits.items():
            probability = (
                fit.validation_probability if split == "val" else fit.test_probability
            )
            part[f"{name}_probability"] = probability
            metric_rows.append(
                {
                    "fold_id": args.fold,
                    "seed": args.seed,
                    "model": name,
                    "penalty": fit.penalty,
                    "selected_C": fit.selected_c,
                    "split": split,
                    "metric_scope": "validation" if split == "val" else "LOCO_test",
                    **_binary_metrics(frame.proxy_label, probability),
                }
            )
        prediction_parts.append(part)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)

    task_root = output_root / args.fold / f"seed_{args.seed}"
    temporary = output_root / f".{args.fold}__seed_{args.seed}.tmp"
    if task_root.exists() or temporary.exists():
        raise RuntimeError(f"Sparse baseline task refuses reuse: {task_root}")
    temporary.mkdir(parents=True)
    write_table(predictions, temporary / "prediction_exact_pathway_sparse_baselines.parquet")
    write_table(metrics, temporary / "metrics.tsv")
    for name, fit in fits.items():
        write_table(fit.validation_metrics, temporary / f"{name}_validation_grid.tsv")
        joblib.dump(fit.model, temporary / f"{name}_model.joblib")

    prediction_path = temporary / "prediction_exact_pathway_sparse_baselines.parquet"
    payload = {
        "status": "PASS",
        "run_id": args.run_id,
        "fold_id": args.fold,
        "seed": args.seed,
        "target_level": "exact_pathway",
        "models": ["lasso", "ridge"],
        "selected_C": {name: fit.selected_c for name, fit in fits.items()},
        "feature_contract": {name: fit.feature_contract for name, fit in fits.items()},
        "validation_candidate_label_sha256": validation_sha,
        "test_candidate_label_sha256": test_sha,
        "graph_reference_task": str(graph_task),
        "graph_reference_prediction_sha256": file_sha256(
            graph_task / "prediction_pathway_calibrated.parquet"
        ),
        "prediction_sha256": file_sha256(prediction_path),
        "formal_gate_sha256": execution_gate_sha256,
        "execution_site_gate_sha256": execution_gate_sha256,
        "training_lineage_gate_sha256": training_gate_sha256,
        "normalized_gate_contract_sha256": normalized_gate_contract_sha256,
        "relocated_execution": training_gate_sha256 != execution_gate_sha256,
        "matrix_audit_sha256": file_sha256(matrix_audit_path),
        "baseline_protocol": str(protocol_path),
        "baseline_protocol_sha256": file_sha256(protocol_path),
        "test_used_for_tuning": False,
        "pair_evidence_features_used": False,
    }
    atomic_write_json(temporary / "SUCCESS.json", payload)
    task_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, task_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
