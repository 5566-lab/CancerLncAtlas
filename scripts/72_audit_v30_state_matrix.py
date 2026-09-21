#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.common import load_config, read_table
from cc_hhgt.formal_gate import validate_formal_training_gate
from cc_hhgt.prediction_contract import (
    PredictionScale,
    validate_probability_aliases,
)
from cc_hhgt.pathway_target import EXACT_PATHWAY_TARGET, pathway_target_level
from cc_hhgt.relocated_lineage import validate_relocated_training_lineage
from cc_hhgt.v30_integrity import atomic_write_bytes, atomic_write_json, canonical_json_sha256, file_sha256


MODELS = ("rgcn", "hgt", "cc_hhgt")
SEEDS = (20260726, 20261726, 20262726)
PILOT_FOLDS = ("LOCO_BRCA", "LOCO_KIRC", "LOCO_OV")


def hard_epoch_cap_violation(success: dict, configured_epoch_cap: int) -> bool:
    """Formal tasks must demonstrate convergence before the hard ceiling."""
    completed = int(success.get("epochs_completed", 0))
    recorded_cap = int(success.get("configured_epoch_cap", configured_epoch_cap))
    return (
        completed >= configured_epoch_cap
        or recorded_cap != configured_epoch_cap
        or bool(success.get("hit_hard_epoch_cap", False))
        or not bool(success.get("stopped_by_joint_patience", False))
    )


def _key_hash(frame: pd.DataFrame) -> str:
    columns = [
        column
        for column in [
            "candidate_id",
            "cancer_id",
            "lncrna_id",
            "pathway_id",
            "pathway_family_id",
            "proxy_label",
            "label_class",
            "state_id",
        ]
        if column in frame
    ]
    records = frame[columns].sort_values(columns, kind="stable").astype(str).to_dict("records")
    return canonical_json_sha256(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit V3 pilot or configured formal multitask matrix")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--formal-gate", required=True)
    parser.add_argument(
        "--training-lineage-gate",
        help=(
            "Original training-site gate. Required only when auditing a verified "
            "relocation whose execution-site gate has different absolute paths."
        ),
    )
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    cfg = load_config(
        args.config,
        project_root_override=Path(args.input_root).resolve(),
        create_dirs=False,
    )
    target_level = pathway_target_level(cfg)
    configured_tasks = int(cfg["formal_contract"]["expected_tasks"])
    training_root = Path(args.training_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"Refusing to reuse matrix audit root: {output_root}")
    gate, execution_gate_sha256 = validate_formal_training_gate(
        args.formal_gate,
        cfg,
        verify_assets=True,
        run_id=args.run_id,
        input_root=args.input_root,
        input_manifest=args.input_manifest,
        output_root=training_root,
    )
    lineage_gate_path = args.training_lineage_gate or args.formal_gate
    _, training_gate_sha256, normalized_gate_contract_sha256 = (
        validate_relocated_training_lineage(lineage_gate_path, gate)
    )
    summary_path = training_root / "run_control" / "PARALLEL_SUMMARY.json"
    runner = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_modes = (
        {"PILOT"}
        if args.pilot
        else {f"FULL_{configured_tasks}", f"FULL_{configured_tasks}_MULTIGPU"}
    )
    expected_count = 9 if args.pilot else configured_tasks
    errors: list[dict] = []
    if runner.get("status") != "PASS" or runner.get("mode") not in expected_modes or int(runner.get("n_tasks", 0)) != expected_count:
        errors.append({"scope": "runner", "error": "runner summary is not the expected completed matrix", "detail": runner})

    results = Path(gate["paths"]["asset_results"])
    folds = read_table(results / "tables" / "fold_manifest.tsv")
    fold_rows = folds.loc[folds.fold_id.astype(str).isin(PILOT_FOLDS)] if args.pilot else folds
    seeds = (SEEDS[0],) if args.pilot else SEEDS
    task_rows = []
    key_rows = []
    metric_rows = []
    configured_epoch_cap = int(cfg["training"]["epochs"])
    convergence_cap_hits = []
    for fold in fold_rows.itertuples(index=False):
        for model in MODELS:
            for seed in seeds:
                task_root = training_root / model / str(fold.fold_id) / f"seed_{seed}"
                required = [
                    "SUCCESS.json", "TRAINING_PROGRESS.json", "best_pathway.pt", "best_state.pt", "last_training_state.pt",
                    "prediction_pathway_calibrated.parquet", "prediction_state_calibrated.parquet",
                    "calibration_pathway.json", "calibration_state.json", "metrics.tsv",
                ]
                missing = [name for name in required if not (task_root / name).exists()]
                if missing:
                    errors.append({"scope": str(task_root), "error": "missing", "detail": missing})
                    continue
                success = json.loads((task_root / "SUCCESS.json").read_text(encoding="utf-8"))
                if success.get("run_id") != args.run_id or success.get("formal_training_gate_sha256") != training_gate_sha256:
                    errors.append({"scope": str(task_root), "error": "provenance mismatch"})
                if success.get("pathway_checkpoint_sha256") != file_sha256(task_root / "best_pathway.pt") or success.get("state_checkpoint_sha256") != file_sha256(task_root / "best_state.pt"):
                    errors.append({"scope": str(task_root), "error": "checkpoint hash mismatch"})
                if (
                    not success.get("exact_interruption_resume_supported")
                    or success.get("last_training_state_sha256")
                    != file_sha256(task_root / "last_training_state.pt")
                ):
                    errors.append(
                        {
                            "scope": str(task_root),
                            "error": "exact interruption-resume checkpoint mismatch",
                        }
                    )
                if success.get("training_progress_sha256") != file_sha256(
                    task_root / "TRAINING_PROGRESS.json"
                ):
                    errors.append(
                        {
                            "scope": str(task_root),
                            "error": "training progress hash mismatch",
                        }
                    )
                gpu_contract = success.get("formal_gpu_contract") or {}
                if (
                    int(gpu_contract.get("visible_device_count", 0)) != 1
                    or float(gpu_contract.get("total_memory_gib", 0.0))
                    < float(gpu_contract.get("minimum_gpu_memory_gib", float("inf")))
                ):
                    errors.append(
                        {
                            "scope": str(task_root),
                            "error": "formal GPU execution contract mismatch",
                            "detail": gpu_contract,
                        }
                    )
                if hard_epoch_cap_violation(success, configured_epoch_cap):
                    detail = {
                        "fold_id": str(fold.fold_id),
                        "model": model,
                        "seed": seed,
                        "epochs_completed": success.get("epochs_completed"),
                        "configured_epoch_cap": success.get("configured_epoch_cap"),
                        "best_pathway_epoch": success.get("best_pathway_epoch"),
                        "best_state_epoch": success.get("best_state_epoch"),
                        "last_pathway_patience_reset_epoch": success.get("last_pathway_patience_reset_epoch"),
                        "last_state_patience_reset_epoch": success.get("last_state_patience_reset_epoch"),
                        "stopped_by_joint_patience": success.get("stopped_by_joint_patience"),
                    }
                    convergence_cap_hits.append(detail)
                    errors.append({"scope": str(task_root), "error": "hard epoch cap reached before joint convergence", "detail": detail})
                state_calibration = json.loads((task_root / "calibration_state.json").read_text(encoding="utf-8"))
                if not state_calibration.get("independent_by_state"):
                    errors.append({"scope": str(task_root), "error": "state calibration is not independent"})
                pathway = pd.read_parquet(task_root / "prediction_pathway_calibrated.parquet")
                state = pd.read_parquet(task_root / "prediction_state_calibrated.parquet")
                if target_level == EXACT_PATHWAY_TARGET:
                    required_exact = {"cancer_id", "lncrna_id", "pathway_id", "pathway_family_id"}
                    missing_exact = sorted(required_exact - set(pathway.columns))
                    duplicate_exact = bool(
                        not missing_exact
                        and pathway.duplicated(["split", "cancer_id", "lncrna_id", "pathway_id"]).any()
                    )
                    if missing_exact or duplicate_exact:
                        errors.append(
                            {
                                "scope": str(task_root),
                                "error": "exact-pathway prediction identity invalid",
                                "detail": {"missing": missing_exact, "duplicates": duplicate_exact},
                            }
                        )
                for task_name, frame in (("pathway", pathway), ("state", state)):
                    try:
                        validate_probability_aliases(
                            frame, PredictionScale.CALIBRATED_PROBABILITY
                        )
                    except RuntimeError as exc:
                        errors.append(
                            {
                                "scope": str(task_root),
                                "error": f"{task_name} probability contract invalid",
                                "detail": str(exc),
                            }
                        )
                    test = frame.loc[frame.split.astype(str).eq("test")].copy()
                    key_rows.append(
                        {
                            "fold_id": str(fold.fold_id),
                            "model": model,
                            "seed": seed,
                            "task": task_name,
                            "rows": len(test),
                            "key_label_sha256": _key_hash(test),
                        }
                    )
                metrics = pd.read_csv(task_root / "metrics.tsv", sep="\t")
                test_metrics = metrics.loc[metrics.split.astype(str).eq("test")].copy()
                test_metrics["fold_id"] = str(fold.fold_id)
                test_metrics["model"] = model
                test_metrics["seed"] = seed
                metric_rows.extend(test_metrics.to_dict("records"))
                task_rows.append(
                    {
                        "fold_id": str(fold.fold_id), "model": model, "seed": seed,
                        "best_pathway_epoch": success.get("best_pathway_epoch"),
                        "best_state_epoch": success.get("best_state_epoch"),
                        "optimizer_steps": success.get("optimizer_steps"),
                        "epochs_completed": success.get("epochs_completed"),
                        "configured_epoch_cap": success.get("configured_epoch_cap"),
                        "hit_hard_epoch_cap": success.get("hit_hard_epoch_cap"),
                        "status": "PASS",
                    }
                )

    task_frame = pd.DataFrame(task_rows)
    key_frame = pd.DataFrame(key_rows)
    metrics = pd.DataFrame(metric_rows)
    if len(task_frame) != expected_count:
        errors.append({"scope": "matrix", "error": f"valid tasks={len(task_frame)}, expected={expected_count}"})
    if not key_frame.empty:
        inconsistent = (
            key_frame.groupby(["fold_id", "task"], observed=True).key_label_sha256.nunique().reset_index(name="n_hashes")
        )
        inconsistent = inconsistent.loc[inconsistent.n_hashes.ne(1)]
        if len(inconsistent):
            errors.append({"scope": "candidate_keys", "error": "candidate keys/labels differ across model or seed", "detail": inconsistent.to_dict("records")})

    state_metrics = metrics.loc[metrics.task.astype(str).eq("state")].copy() if not metrics.empty else pd.DataFrame()
    reversal = []
    if not state_metrics.empty:
        for state_id, group in state_metrics.groupby("state_id", observed=True):
            values = pd.to_numeric(group.auroc, errors="coerce").dropna()
            reversal.append(
                {
                    "state_id": str(state_id),
                    "runs": len(values),
                    "median_auroc": float(values.median()) if len(values) else np.nan,
                    "fraction_below_0_4": float(values.lt(0.4).mean()) if len(values) else np.nan,
                    "systematic_reverse": bool(len(values) and values.median() < 0.45 and values.lt(0.4).mean() >= 2 / 3),
                }
            )
    if args.pilot and (not reversal or any(row["systematic_reverse"] for row in reversal)):
        errors.append({"scope": "pilot", "error": "systematic reverse signal", "detail": reversal})

    ov_dnass = None
    eligibility = read_table(Path(gate["paths"]["run_root"]) / "canonical" / "state_complete_case_eligibility.tsv")
    hit = eligibility.loc[eligibility.cancer_id.astype(str).eq("OV") & eligibility.state_id.astype(str).eq("stemness_dna::DNAss")]
    if len(hit) == 1:
        ov_dnass = hit.iloc[0].to_dict()
    if not ov_dnass or str(ov_dnass.get("eligibility")) != "UNAVAILABLE":
        errors.append({"scope": "OV/DNAss", "error": "must be explicit UNAVAILABLE", "detail": ov_dnass})

    status = "PASS" if not errors else "FAIL"
    output_root.mkdir(parents=True)
    atomic_write_bytes(output_root / "task_audit.tsv", task_frame.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8"))
    atomic_write_bytes(output_root / "candidate_key_audit.tsv", key_frame.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8"))
    atomic_write_bytes(output_root / "test_metrics.tsv", metrics.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8"))
    payload = {
        "status": status,
        "mode": str(runner.get("mode")),
        "run_id": args.run_id,
        "training_lineage_gate_sha256": training_gate_sha256,
        "execution_site_gate_sha256": execution_gate_sha256,
        "normalized_gate_contract_sha256": normalized_gate_contract_sha256,
        "relocated_execution": training_gate_sha256 != execution_gate_sha256,
        "pathway_target_level": target_level,
        "expected_tasks": expected_count,
        "verified_tasks": len(task_frame),
        "candidate_keys_and_labels_identical_across_model_seed": not any(error["scope"] == "candidate_keys" for error in errors),
        "probabilities_non_null": not any("probability" in error["error"] for error in errors),
        "state_calibration_independent": not any("calibration" in error["error"] for error in errors),
        "configured_epoch_cap": configured_epoch_cap,
        "hard_epoch_cap_hits": convergence_cap_hits,
        "all_tasks_converged_before_hard_epoch_cap": not convergence_cap_hits,
        "systematic_reverse_audit": reversal,
        "ov_dnass": ov_dnass,
        "errors": errors,
        "release_eligible": status == "PASS" and not args.pilot,
    }
    atomic_write_json(output_root / "AUDIT.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError(f"V3 matrix audit failed with {len(errors)} errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
