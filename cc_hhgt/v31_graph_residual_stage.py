"""Fail-closed orchestration helpers for the V3.1 Graph residual pilot."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import file_sha256, require_columns, write_table
from .metrics import binary_metrics
from .prediction_contract import candidate_universe_sha256
from .v30_integrity import atomic_write_json, merkle_sha256
from .v31_admission import select_residual_lambda
from .v31_pilot_gate import PILOT_CANCERS, PILOT_SEEDS
from .v31_residual import FALLBACK_ATOL, assert_exact_fallback


GRAPH_CONTRACT = "CONTRACT-S"


def lambda_token(value: float) -> str:
    """Stable path token for a residual shrinkage coefficient."""

    token = np.format_float_positional(float(value), trim="-")
    return token.replace("-", "m").replace(".", "p")


def expected_graph_residual_matrix(
    lambdas: list[float] | tuple[float, ...],
) -> list[tuple[str, int, float]]:
    values = tuple(float(value) for value in lambdas)
    if not values or any(value < 0 or not np.isfinite(value) for value in values):
        raise ValueError("Graph residual shrinkage grid must be finite and nonnegative")
    if len(set(values)) != len(values):
        raise ValueError("Graph residual shrinkage grid contains duplicates")
    return [
        (cancer, seed, value)
        for cancer in PILOT_CANCERS
        for seed in PILOT_SEEDS
        for value in values
    ]


def load_fold_residual_base(
    path: Path,
    *,
    heldout_cancer: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load one outer fold of the frozen cross-fit BestSimple offset."""

    frame = pd.read_parquet(path)
    require_columns(
        frame,
        [
            "candidate_id",
            "cancer_id",
            "task",
            "split",
            "loco_cancer",
            "best_simple_score",
            "z_base",
            "metric_scope",
            "simple_base_fit_cancers",
        ],
        "BestSimple residual base",
    )
    frame = frame.loc[frame.loco_cancer.astype(str).eq(heldout_cancer)].copy()
    if frame.empty:
        raise RuntimeError(f"Residual base lacks outer fold {heldout_cancer}")
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for task in ("pathway", "state"):
        task_frame = frame.loc[frame.task.astype(str).eq(task)].copy()
        result[task] = {}
        for split in ("train", "val", "test"):
            current = task_frame.loc[task_frame.split.astype(str).eq(split)].copy()
            if current.empty:
                raise RuntimeError(
                    f"Residual base is empty for {heldout_cancer}/{task}/{split}"
                )
            if current.candidate_id.astype(str).duplicated().any():
                raise RuntimeError(
                    f"Residual base duplicates candidates for {heldout_cancer}/{task}/{split}"
                )
            result[task][split] = current
    return result


def audit_residual_task(
    task_root: Path,
    *,
    heldout_cancer: str,
    seed: int,
    shrinkage_lambda: float,
) -> dict[str, Any]:
    """Verify additive-logit and prediction contracts after one GPU task."""

    required = [
        "TRAINING_SUCCESS.json",
        "CALIBRATION_SUCCESS.json",
        "best_pathway.pt",
        "best_state.pt",
        "prediction_pathway_raw.parquet",
        "prediction_state_raw.parquet",
        "prediction_pathway_calibrated.parquet",
        "prediction_state_calibrated.parquet",
        "metrics_raw.tsv",
    ]
    missing = [name for name in required if not (task_root / name).is_file()]
    if missing:
        raise RuntimeError(f"Graph residual task is incomplete: {missing}")
    training = json.loads(
        (task_root / "TRAINING_SUCCESS.json").read_text(encoding="utf-8")
    )
    failures: list[str] = []
    if training.get("residual_learning") is not True:
        failures.append("residual_learning is disabled")
    if training.get("residual_structure") != "z_final=z_base+delta_graph":
        failures.append("residual structure mismatch")
    if training.get("graph_contract") != GRAPH_CONTRACT:
        failures.append("graph contract mismatch")
    if not np.isclose(
        float(training.get("residual_shrinkage_lambda", np.nan)),
        float(shrinkage_lambda),
        rtol=0,
        atol=0,
    ):
        failures.append("shrinkage lambda mismatch")
    initialization_error = float(
        training.get("residual_initialization_max_abs", np.inf)
    )
    if not np.isfinite(initialization_error) or initialization_error > FALLBACK_ATOL:
        failures.append("epoch-zero exact base invariant failed")

    row_audits: list[dict[str, Any]] = []
    for task in ("pathway", "state"):
        raw = pd.read_parquet(task_root / f"prediction_{task}_raw.parquet")
        calibrated = pd.read_parquet(
            task_root / f"prediction_{task}_calibrated.parquet"
        )
        require_columns(
            raw,
            [
                "candidate_id",
                "cancer_id",
                "proxy_label",
                "split",
                "raw_logit",
                "base_logit",
                "base_probability",
                "graph_residual_logit",
                "proxy_positive_probability",
                "prediction_scale",
                "residual_shrinkage_lambda",
            ],
            f"Graph residual {task} raw prediction",
        )
        if not raw.prediction_scale.astype(str).eq("raw_probability").all():
            failures.append(f"{task} raw prediction scale mismatch")
        if not calibrated.prediction_scale.astype(str).eq(
            "calibrated_probability"
        ).all():
            failures.append(f"{task} calibrated prediction scale mismatch")
        raw_signature = raw[["candidate_id", "proxy_label", "split"]].sort_values(
            "candidate_id"
        ).reset_index(drop=True)
        calibrated_signature = calibrated[
            ["candidate_id", "proxy_label", "split"]
        ].sort_values("candidate_id").reset_index(drop=True)
        if not raw_signature.equals(calibrated_signature):
            failures.append(f"{task} calibration changed candidates or labels")
        additive_error = float(
            np.max(
                np.abs(
                    pd.to_numeric(raw.raw_logit).to_numpy(dtype=float)
                    - pd.to_numeric(raw.base_logit).to_numpy(dtype=float)
                    - pd.to_numeric(raw.graph_residual_logit).to_numpy(dtype=float)
                )
            )
        )
        if not np.isfinite(additive_error) or additive_error > 2e-6:
            failures.append(f"{task} additive-logit identity failed: {additive_error}")
        if raw.proxy_positive_probability.isna().any():
            failures.append(f"{task} contains NaN probabilities")
        test = raw.loc[raw.split.astype(str).eq("test")]
        if test.empty or set(test.cancer_id.astype(str)) != {heldout_cancer}:
            failures.append(f"{task} test cancer mismatch")
        row_audits.append(
            {
                "task": task,
                "candidate_universe_sha256": candidate_universe_sha256(raw),
                "additive_logit_max_abs_error": additive_error,
                "prediction_rows": int(len(raw)),
            }
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "heldout_cancer": heldout_cancer,
        "seed": int(seed),
        "residual_shrinkage_lambda": float(shrinkage_lambda),
        "residual_initialization_max_abs": initialization_error,
        "row_audits": row_audits,
    }


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _target_subtype(frame: pd.DataFrame, task: str) -> pd.Series:
    if task == "pathway":
        return pd.Series("Pathway", index=frame.index, dtype="object")
    require_columns(frame, ["state_id"], "state residual prediction")
    state = frame.state_id.astype(str)
    return pd.Series(
        np.where(state.str.contains("RNAss", case=False, regex=False), "RNAss", "DNAss"),
        index=frame.index,
    )


def _read_prediction(
    root: Path,
    *,
    cancer: str,
    seed: int,
    shrinkage_lambda: float,
    task: str,
    scale: str,
) -> pd.DataFrame:
    task_root = (
        root
        / f"lambda_{lambda_token(shrinkage_lambda)}"
        / "cc_hhgt"
        / f"LOCO_{cancer}"
        / f"seed_{seed}"
    )
    suffix = "raw" if scale == "raw_probability" else "calibrated"
    path = task_root / f"prediction_{task}_{suffix}.parquet"
    frame = pd.read_parquet(path)
    frame = frame.loc[frame.split.astype(str).isin(["val", "validation", "test"])].copy()
    frame["split"] = frame.split.astype(str).replace({"validation": "val"})
    frame["loco_cancer"] = cancer
    frame["seed"] = int(seed)
    frame["task"] = task
    frame["target_subtype"] = _target_subtype(frame, task)
    frame["residual_shrinkage_lambda"] = float(shrinkage_lambda)
    frame["prediction_scale"] = scale
    frame["source_prediction_file_sha256"] = file_sha256(path)
    return frame


def _metric_rows(predictions: pd.DataFrame, variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = [
        "loco_cancer",
        "seed",
        "target_subtype",
        "split",
        "prediction_scale",
    ]
    if "residual_shrinkage_lambda" in predictions:
        groups.append("residual_shrinkage_lambda")
    for keys, group in predictions.groupby(groups, observed=True, sort=True):
        values = binary_metrics(group.proxy_label, group.proxy_positive_probability)
        prevalence = float(values["positive_rate"])
        auprc = float(values["auprc"])
        rows.append(
            {
                **dict(zip(groups, keys)),
                "model_variant": variant,
                "metric_scope": "LOCO_test" if keys[3] == "test" else "validation",
                **values,
                "auprc_over_prevalence": auprc / prevalence if prevalence > 0 else np.nan,
                "auprc_minus_prevalence": auprc - prevalence,
                "candidate_universe_sha256": candidate_universe_sha256(group),
            }
        )
    return rows


def finalize_graph_residual(
    run_root: Path,
    output_dir: Path,
    *,
    lambdas: list[float] | tuple[float, ...],
    best_simple_gate_path: Path,
    aggregation_git_commit: str,
) -> dict[str, Any]:
    """Select lambda on pooled validation rows and publish test-only results."""

    matrix = expected_graph_residual_matrix(lambdas)
    simple_gate = json.loads(best_simple_gate_path.read_text(encoding="utf-8"))
    if simple_gate.get("status") != "PASS":
        raise RuntimeError("BestSimple gate is not PASS")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output_dir / "GRAPH_RESIDUAL_PREDICTIONS.parquet",
        "metrics": output_dir / "GRAPH_RESIDUAL_METRICS.tsv",
        "selection": output_dir / "GRAPH_RESIDUAL_ADMISSION.tsv",
        "fallback": output_dir / "GRAPH_RESIDUAL_FALLBACK_AUDIT.tsv",
        "gate": output_dir / "GRAPH_RESIDUAL_GATE.json",
        "manifest": output_dir / "GRAPH_RESIDUAL_SHA256.tsv",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise RuntimeError(f"Graph residual finalizer refuses overwrite: {existing}")

    expected_roots = {
        (
            run_root
            / f"lambda_{lambda_token(value)}"
            / "cc_hhgt"
            / f"LOCO_{cancer}"
            / f"seed_{seed}"
        ).resolve()
        for cancer, seed, value in matrix
    }
    observed_roots = {
        path.parent.resolve()
        for path in run_root.glob("lambda_*/cc_hhgt/LOCO_*/seed_*/SUCCESS.json")
    }
    if observed_roots != expected_roots:
        raise RuntimeError(
            "Graph residual task set mismatch: "
            f"missing={sorted(map(str, expected_roots-observed_roots))}, "
            f"extra={sorted(map(str, observed_roots-expected_roots))}"
        )

    raw_parts: list[pd.DataFrame] = []
    calibrated_parts: list[pd.DataFrame] = []
    for cancer, seed, value in matrix:
        task_root = (
            run_root
            / f"lambda_{lambda_token(value)}"
            / "cc_hhgt"
            / f"LOCO_{cancer}"
            / f"seed_{seed}"
        )
        success = json.loads((task_root / "SUCCESS.json").read_text(encoding="utf-8"))
        if success.get("status") != "COMPLETED":
            raise RuntimeError(f"Graph residual task did not complete: {task_root}")
        for task in ("pathway", "state"):
            raw_parts.append(
                _read_prediction(
                    run_root,
                    cancer=cancer,
                    seed=seed,
                    shrinkage_lambda=value,
                    task=task,
                    scale="raw_probability",
                )
            )
            calibrated_parts.append(
                _read_prediction(
                    run_root,
                    cancer=cancer,
                    seed=seed,
                    shrinkage_lambda=value,
                    task=task,
                    scale="calibrated_probability",
                )
            )
    raw = pd.concat(raw_parts, ignore_index=True, sort=False)
    calibrated = pd.concat(calibrated_parts, ignore_index=True, sort=False)

    # Candidate sampling is frozen across model seeds and lambdas.  A composite
    # validation ID permits one global, test-independent choice per task subtype.
    validation = raw.loc[raw.split.eq("val")].copy()
    validation["source_candidate_id"] = validation.candidate_id.astype(str)
    validation["candidate_id"] = (
        validation.loco_cancer.astype(str)
        + "|"
        + validation.seed.astype(str)
        + "|"
        + validation.source_candidate_id
    )
    validation["metric_scope"] = "validation"
    lambda_metrics, selection = select_residual_lambda(
        validation,
        group_columns=["target_subtype"],
    )
    selection["global_across_pilot_cancers"] = True
    selection["global_across_model_seeds"] = True
    selection["test_metric_used_for_selection"] = False

    selected_parts: list[pd.DataFrame] = []
    fallback_rows: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        subtype = str(row.target_subtype)
        selected_lambda = float(row.selected_lambda)
        admitted = bool(row.admitted)
        for scale, source in (
            ("raw_probability", raw),
            ("calibrated_probability", calibrated),
        ):
            current = source.loc[
                source.target_subtype.astype(str).eq(subtype)
                & np.isclose(
                    source.residual_shrinkage_lambda.astype(float),
                    selected_lambda,
                    rtol=0,
                    atol=0,
                )
            ].copy()
            current["selected_lambda"] = selected_lambda
            current["module_admitted"] = admitted
            current["selection_scope"] = "pooled_outer_validation"
            current["test_metric_used_for_selection"] = False
            if not admitted:
                # Exact fallback is defined on the raw additive score.  The
                # calibrated candidate is intentionally replaced by the same
                # frozen base so post-processing cannot silently overwrite it.
                current["proxy_positive_probability"] = current.base_probability.astype(float)
                current["raw_logit"] = current.base_logit.astype(float)
                current["graph_residual_logit"] = 0.0
                current["prediction_scale"] = "raw_probability"
            current["model_variant"] = "R1_SELECTED" if admitted else "B0_FALLBACK"
            selected_parts.append(current)
            if not admitted:
                assert_exact_fallback(
                    current.base_logit.to_numpy(dtype=float),
                    current.raw_logit.to_numpy(dtype=float),
                )
                error = float(
                    np.max(
                        np.abs(
                            current.base_probability.to_numpy(dtype=float)
                            - current.proxy_positive_probability.to_numpy(dtype=float)
                        )
                    )
                )
                fallback_rows.append(
                    {
                        "target_subtype": subtype,
                        "requested_scale": scale,
                        "candidate_rows": int(len(current)),
                        "max_abs_probability_error": error,
                        "tolerance": FALLBACK_ATOL,
                        "status": "PASS" if error <= FALLBACK_ATOL else "FAIL",
                    }
                )
    selected = pd.concat(selected_parts, ignore_index=True, sort=False)
    # OFF produces identical raw rows for both source scales; retain one exact
    # fallback copy while preserving both scales for admitted modules.
    selected = selected.drop_duplicates(
        ["loco_cancer", "seed", "target_subtype", "split", "candidate_id", "prediction_scale"],
        keep="first",
    ).reset_index(drop=True)

    metric_rows = _metric_rows(raw, "R1_LAMBDA_CANDIDATE")
    metric_rows.extend(_metric_rows(calibrated, "R1_LAMBDA_CANDIDATE"))
    selected_for_metrics = selected.drop(columns=["residual_shrinkage_lambda"], errors="ignore")
    metric_rows.extend(_metric_rows(selected_for_metrics, "R1_SELECTED_OR_BASE"))
    metrics = pd.DataFrame(metric_rows)
    fallback = pd.DataFrame(fallback_rows)
    if not fallback.empty and not fallback.status.eq("PASS").all():
        raise RuntimeError("Graph residual OFF did not return the exact frozen base")

    _atomic_table(selected, paths["predictions"])
    _atomic_table(metrics, paths["metrics"])
    _atomic_table(selection, paths["selection"])
    _atomic_table(fallback, paths["fallback"])
    manifest_rows = [
        {
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in (
            paths["predictions"],
            paths["metrics"],
            paths["selection"],
            paths["fallback"],
        )
    ]
    _atomic_table(pd.DataFrame(manifest_rows), paths["manifest"])
    payload = {
        "status": "PASS",
        "stage": "PHASE_B2_GRAPH_RESIDUAL",
        "tasks_expected": len(matrix),
        "tasks_completed": len(matrix),
        "pilot_cancers": list(PILOT_CANCERS),
        "model_seeds": list(PILOT_SEEDS),
        "shrinkage_grid": list(map(float, lambdas)),
        "graph_contract": GRAPH_CONTRACT,
        "selection_scope": "pooled_outer_validation",
        "selection_used_test_labels": False,
        "module_decision_is_global_across_test_cancers": True,
        "aggregation_git_commit": aggregation_git_commit,
        "best_simple_gate_sha256": file_sha256(best_simple_gate_path),
        "predictions_sha256": file_sha256(paths["predictions"]),
        "metrics_sha256": file_sha256(paths["metrics"]),
        "admission_sha256": file_sha256(paths["selection"]),
        "manifest_sha256": file_sha256(paths["manifest"]),
        "manifest_merkle_sha256": merkle_sha256(manifest_rows),
        "full_cancer_training_started": False,
        "failures": [],
    }
    atomic_write_json(paths["gate"], payload)
    return payload
