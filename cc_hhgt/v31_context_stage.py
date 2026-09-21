"""Orchestration for the V3.1 multimodal CPU residual pilot."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import file_sha256, require_columns, write_table
from .metrics import binary_metrics
from .prediction_contract import candidate_universe_sha256
from .v30_integrity import atomic_write_json, merkle_sha256
from .v31_context_residual import (
    ContextTransform,
    attach_context_features,
    exact_module_fallback,
    fit_offset_logistic,
    predict_offset_logistic,
    select_context_residual,
)
from .v31_pilot_gate import PILOT_CANCERS, PILOT_SEEDS
from .v31_residual import FALLBACK_ATOL, assert_exact_fallback, logit_to_probability


TARGET_SUBTYPES = ("Pathway", "RNAss", "DNAss")


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pass_json(path: Path, stage: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("failures"):
        raise RuntimeError(f"{stage} gate is not a clean PASS: {path}")
    return payload


def load_context_asset(
    table_path: Path,
    success_path: Path,
    *,
    expected_table_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load an immutable context table and verify its declared SHA256."""

    table_path = Path(table_path).resolve()
    success_path = Path(success_path).resolve()
    success = _pass_json(success_path, "context asset")
    observed = file_sha256(table_path)
    declared = success.get("table_sha256")
    if declared is None:
        manifest_sha = str(success.get("manifest_sha256", ""))
        declarations: list[str] = []
        for manifest_path in sorted(success_path.parent.glob("*SHA256.tsv")):
            if manifest_sha and file_sha256(manifest_path) != manifest_sha:
                continue
            manifest = pd.read_csv(manifest_path, sep="\t")
            if not {"relative_path", "sha256"}.issubset(manifest.columns):
                continue
            hits = manifest.loc[manifest.relative_path.astype(str).eq(table_path.name)]
            declarations.extend(hits.sha256.astype(str).tolist())
        if len(set(declarations)) != 1:
            raise RuntimeError(f"Context table has no unique manifest declaration: {table_path}")
        declared = declarations[0]
    declared = str(declared)
    if declared != observed:
        raise RuntimeError(f"Context asset SHA256 mismatch: {table_path}")
    if expected_table_name and table_path.name != expected_table_name:
        raise RuntimeError(f"Unexpected context table: {table_path.name}")
    return pd.read_parquet(table_path), {
        "table_path": str(table_path),
        "table_sha256": observed,
        "success_path": str(success_path),
        "success_sha256": file_sha256(success_path),
    }


def _normalize_training_base(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame,
        [
            "candidate_id", "cancer_id", "lncrna_id", "loco_cancer", "task",
            "target_subtype", "split", "label", "z_base", "best_simple_score",
            "metric_scope",
        ],
        "BestSimple residual base",
    )
    train = frame.loc[frame.split.astype(str).eq("train")].copy()
    train["proxy_label"] = pd.to_numeric(train.label, errors="raise").astype(np.int8)
    train["training_offset_logit"] = pd.to_numeric(train.z_base, errors="raise")
    train["training_offset_probability"] = pd.to_numeric(
        train.best_simple_score, errors="raise"
    )
    if not train.metric_scope.astype(str).eq("cancer_crossfit_OOF").all():
        raise RuntimeError("Context learner training offset is not cancer-crossfit OOF")
    return train


def _normalize_graph_base(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame,
        [
            "candidate_id", "cancer_id", "lncrna_id", "loco_cancer", "seed",
            "task", "target_subtype", "split", "proxy_label", "raw_logit",
            "proxy_positive_probability", "prediction_scale",
        ],
        "accepted Graph predictions",
    )
    result = frame.loc[
        frame.prediction_scale.astype(str).eq("raw_probability")
        & frame.split.astype(str).isin(["val", "validation", "test"])
    ].copy()
    result["split"] = result.split.astype(str).replace({"validation": "val"})
    result["current_base_logit"] = pd.to_numeric(result.raw_logit, errors="raise")
    result["current_base_probability"] = pd.to_numeric(
        result.proxy_positive_probability, errors="raise"
    )
    unique = ["loco_cancer", "seed", "target_subtype", "split", "candidate_id"]
    if result.duplicated(unique).any():
        raise RuntimeError("Accepted Graph base contains duplicate raw prediction rows")
    if set(result.loco_cancer.astype(str)) != set(PILOT_CANCERS):
        raise RuntimeError("Accepted Graph base cancer scope drift")
    if set(result.seed.astype(int)) != set(PILOT_SEEDS):
        raise RuntimeError("Accepted Graph base seed scope drift")
    return result


def _module_predictions(
    module: str,
    training_base: pd.DataFrame,
    graph_base: pd.DataFrame,
    lnc_context: pd.DataFrame,
    *,
    target_context: pd.DataFrame | None,
    shrinkage_grid: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    for heldout_cancer in PILOT_CANCERS:
        outer_train = training_base.loc[
            training_base.loco_cancer.astype(str).eq(heldout_cancer)
        ].copy()
        query = graph_base.loc[
            graph_base.loco_cancer.astype(str).eq(heldout_cancer)
        ].copy()
        if outer_train.empty or query.empty:
            raise RuntimeError(f"Context residual fold is empty: {module}/{heldout_cancer}")
        for subtype in TARGET_SUBTYPES:
            train_subtype = outer_train.loc[
                outer_train.target_subtype.astype(str).eq(subtype)
            ].copy()
            query_subtype = query.loc[query.target_subtype.astype(str).eq(subtype)].copy()
            if train_subtype.empty or query_subtype.empty:
                raise RuntimeError(f"Context residual subtype is empty: {module}/{heldout_cancer}/{subtype}")
            pathway_context = target_context if subtype == "Pathway" else None
            train_attached, feature_columns = attach_context_features(
                train_subtype, lnc_context, target_context=pathway_context
            )
            query_attached, query_features = attach_context_features(
                query_subtype, lnc_context, target_context=pathway_context
            )
            if feature_columns != query_features:
                raise RuntimeError("Train/query context feature contract drift")
            transform = ContextTransform.fit(train_attached, feature_columns)
            train_matrix, train_available = transform.transform(train_attached)
            query_matrix, query_available = transform.transform(query_attached)
            for shrinkage in map(float, shrinkage_grid):
                fit = fit_offset_logistic(
                    train_matrix,
                    train_attached.proxy_label,
                    train_attached.training_offset_logit,
                    shrinkage_lambda=shrinkage,
                )
                if not fit.converged:
                    raise RuntimeError(
                        f"Context residual did not converge: {module}/{heldout_cancer}/{subtype}/{shrinkage}"
                    )
                delta, final_logit, probability = predict_offset_logistic(
                    fit,
                    query_matrix,
                    query_attached.current_base_logit,
                    query_available,
                )
                current = query_attached.copy()
                current["module"] = module
                current["residual_shrinkage_lambda"] = shrinkage
                current["context_residual_logit"] = delta
                current["final_logit"] = final_logit
                current["proxy_positive_probability"] = probability
                current["context_available"] = query_available
                current["metric_scope"] = np.where(
                    current.split.astype(str).eq("test"), "LOCO_test", "validation"
                )
                current["prediction_scale"] = "raw_probability"
                current["context_training_offset"] = "BestSimple_cancer_crossfit_OOF"
                current["evaluation_offset"] = "accepted_Graph_or_exact_BestSimple_fallback"
                prediction_parts.append(current)
                fit_rows.append(
                    {
                        "module": module,
                        "loco_cancer": heldout_cancer,
                        "target_subtype": subtype,
                        "residual_shrinkage_lambda": shrinkage,
                        "training_rows": int(len(train_attached)),
                        "training_available_rows": int(train_available.sum()),
                        "query_rows_including_seeds": int(len(query_attached)),
                        "query_available_rows_including_seeds": int(query_available.sum()),
                        "input_features": int(len(feature_columns)),
                        "design_columns_with_masks": int(train_matrix.shape[1]),
                        "optimizer_converged": fit.converged,
                        "optimizer_iterations": fit.iterations,
                        "objective": fit.objective,
                        "zero_initialization_max_abs_error": fit.initialization_max_abs_error,
                        "training_offset_is_crossfit": True,
                        "test_labels_used_for_fit": False,
                    }
                )
    return pd.concat(prediction_parts, ignore_index=True, sort=False), pd.DataFrame(fit_rows)


def _unavailable_module(graph_base: pd.DataFrame, module: str) -> pd.DataFrame:
    result = graph_base.copy()
    result["module"] = module
    result["residual_shrinkage_lambda"] = 1.0
    result["context_residual_logit"] = 0.0
    result["final_logit"] = result.current_base_logit.astype(float)
    result["proxy_positive_probability"] = result.current_base_probability.astype(float)
    result["context_available"] = False
    result["metric_scope"] = np.where(result.split.eq("test"), "LOCO_test", "validation")
    result["prediction_scale"] = "raw_probability"
    result["context_training_offset"] = "UNAVAILABLE"
    result["evaluation_offset"] = "accepted_Graph_or_exact_BestSimple_fallback"
    assert_exact_fallback(result.current_base_logit, result.final_logit)
    return result


def _metric_rows(predictions: pd.DataFrame, model_variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = ["module", "loco_cancer", "seed", "target_subtype", "split"]
    for keys, group in predictions.groupby(groups, observed=True, sort=True):
        base = binary_metrics(group.proxy_label, group.current_base_probability)
        final = binary_metrics(group.proxy_label, group.proxy_positive_probability)
        prevalence = float(final["positive_rate"])
        rows.append(
            {
                **dict(zip(groups, keys)),
                "model_variant": model_variant,
                "metric_scope": "LOCO_test" if keys[-1] == "test" else "validation",
                "n": int(final["n"]),
                "positive_rate": prevalence,
                "auprc": float(final["auprc"]),
                "auroc": float(final["auroc"]),
                "brier": float(final["brier"]),
                "ece": float(final["ece"]),
                "auprc_over_prevalence": float(final["auprc"] / prevalence) if prevalence > 0 else np.nan,
                "auprc_minus_prevalence": float(final["auprc"] - prevalence),
                "base_auprc": float(base["auprc"]),
                "delta_auprc": float(final["auprc"] - base["auprc"]),
                "delta_auroc": float(final["auroc"] - base["auroc"]),
                "delta_brier": float(final["brier"] - base["brier"]),
                "delta_ece": float(final["ece"] - base["ece"]),
                "candidate_universe_sha256": candidate_universe_sha256(group),
            }
        )
    return rows


def _select_and_compose(
    candidates: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    validation = candidates.loc[candidates.metric_scope.eq("validation")].copy()
    # Candidate IDs repeat across outer folds and graph seeds by design.  The
    # composite key makes global validation selection explicit and auditable.
    validation["source_candidate_id"] = validation.candidate_id.astype(str)
    validation["candidate_id"] = (
        validation.loco_cancer.astype(str)
        + "|" + validation.seed.astype(str)
        + "|" + validation.source_candidate_id
    )
    pooled, per_cancer, admission = select_context_residual(validation)

    selected_parts: list[pd.DataFrame] = []
    fallback_rows: list[dict[str, Any]] = []
    for row in admission.itertuples(index=False):
        current = candidates.loc[
            candidates.module.astype(str).eq(str(row.module))
            & candidates.target_subtype.astype(str).eq(str(row.target_subtype))
            & np.isclose(
                candidates.residual_shrinkage_lambda.astype(float),
                float(row.selected_lambda), rtol=0, atol=0,
            )
        ].copy()
        current["selected_lambda"] = float(row.selected_lambda)
        current["module_admitted"] = bool(row.admitted)
        current["selection_scope"] = "pooled_outer_validation"
        current["test_metric_used_for_selection"] = False
        if not bool(row.admitted):
            current = exact_module_fallback(current)
        error = float(
            np.max(np.abs(current.final_logit - current.current_base_logit))
        ) if not bool(row.admitted) and len(current) else 0.0
        fallback_rows.append(
            {
                "module": str(row.module),
                "target_subtype": str(row.target_subtype),
                "admitted": bool(row.admitted),
                "candidate_rows": int(len(current)),
                "off_max_abs_logit_error": error if not bool(row.admitted) else np.nan,
                "tolerance": FALLBACK_ATOL,
                "status": "PASS" if bool(row.admitted) or error <= FALLBACK_ATOL else "FAIL",
            }
        )
        selected_parts.append(current)
    selected = pd.concat(selected_parts, ignore_index=True, sort=False)
    fallback = pd.DataFrame(fallback_rows)
    if not fallback.status.eq("PASS").all():
        raise RuntimeError("A rejected context module failed exact fallback")

    identity = ["loco_cancer", "seed", "target_subtype", "split", "candidate_id"]
    base_columns = [
        *identity, "cancer_id", "lncrna_id", "task", "proxy_label",
        "current_base_logit", "current_base_probability",
    ]
    if "pathway_family_id" in selected:
        base_columns.append("pathway_family_id")
    if "state_id" in selected:
        base_columns.append("state_id")
    base = selected.loc[:, base_columns].drop_duplicates(identity)
    expected_rows = len(base)
    for module, module_frame in selected.groupby("module", observed=True, sort=True):
        contribution = module_frame.loc[:, [*identity, "context_residual_logit"]].copy()
        contribution = contribution.rename(
            columns={"context_residual_logit": f"delta_{str(module).lower()}"}
        )
        base = base.merge(contribution, on=identity, how="left", validate="one_to_one")
        if len(base) != expected_rows:
            raise RuntimeError("Context module composition changed candidate rows")
    delta_columns = [column for column in base if column.startswith("delta_")]
    base[delta_columns] = base[delta_columns].fillna(0.0)
    base["final_logit"] = base.current_base_logit + base[delta_columns].sum(axis=1)
    base["proxy_positive_probability"] = logit_to_probability(base.final_logit)
    base["prediction_scale"] = "raw_probability"
    base["metric_scope"] = np.where(base.split.eq("test"), "LOCO_test", "validation")
    base["model_variant"] = "R_SELECTED"
    if not delta_columns or all(np.max(np.abs(base[column])) == 0 for column in delta_columns):
        assert_exact_fallback(base.current_base_logit, base.final_logit)
    return selected, pooled, per_cancer, admission, fallback, base


def run_multimodal_context_stage(
    *,
    residual_base_path: Path,
    best_simple_gate_path: Path,
    graph_predictions_path: Path,
    graph_gate_path: Path,
    b2_hard_report_path: Path,
    modules: Mapping[str, tuple[pd.DataFrame, pd.DataFrame | None] | None],
    module_lineage: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    shrinkage_grid: Sequence[float],
    aggregation_git_commit: str,
) -> dict[str, Any]:
    """Fit RNA/Genomic/ATAC residuals, admit on validation, and compose."""

    best_gate = _pass_json(Path(best_simple_gate_path), "BestSimple")
    graph_gate = _pass_json(Path(graph_gate_path), "Graph residual")
    if graph_gate.get("selection_used_test_labels") is not False:
        raise RuntimeError("Graph residual selection provenance is unsafe")
    hard_report = _pass_json(Path(b2_hard_report_path), "B2 HARD REPORT")
    if hard_report.get("stage") != "B2_HARD_REPORT":
        raise RuntimeError("B3 requires the canonical B2 HARD REPORT stage")
    if hard_report.get("decision") != "HARD_GO_B3":
        raise RuntimeError("B2 HARD REPORT did not issue HARD_GO_B3")
    if hard_report.get("b3_authorized") is not True:
        raise RuntimeError("B2 HARD REPORT does not authorize B3")
    if hard_report.get("full_cancer_training_authorized") is not False:
        raise RuntimeError("B2 HARD REPORT has an unsafe full-cancer authorization")
    hard_gates = hard_report.get("gates", {})
    required_hard_gates = {
        "delta_auprc",
        "cluster_aware_ci",
        "cancer_direction",
        "calibration",
        "exact_fallback",
        "integrity",
    }
    if set(hard_gates) != required_hard_gates or not all(
        hard_gates.get(key) is True for key in required_hard_gates
    ):
        raise RuntimeError("B2 HARD REPORT does not pass every registered hard gate")
    if hard_report.get("b2_gate_sha256") != file_sha256(graph_gate_path):
        raise RuntimeError("B2 HARD REPORT is not bound to the supplied graph gate")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RuntimeError(f"Context residual stage refuses reuse: {output_dir}")
    output_dir.mkdir(parents=True)
    training_base = _normalize_training_base(pd.read_parquet(residual_base_path))
    graph_base = _normalize_graph_base(pd.read_parquet(graph_predictions_path))
    grids = tuple(map(float, shrinkage_grid))
    if not grids or any(value < 0 or not np.isfinite(value) for value in grids):
        raise ValueError("Context shrinkage grid must be finite and nonnegative")

    prediction_parts: list[pd.DataFrame] = []
    fit_parts: list[pd.DataFrame] = []
    for module in ("RNA", "GENOMIC", "ATAC", "SINGLECELL"):
        asset = modules.get(module)
        if asset is None:
            prediction_parts.append(_unavailable_module(graph_base, module))
            continue
        predictions, fit_audit = _module_predictions(
            module,
            training_base,
            graph_base,
            asset[0],
            target_context=asset[1],
            shrinkage_grid=grids,
        )
        prediction_parts.append(predictions)
        fit_parts.append(fit_audit)
    candidates = pd.concat(prediction_parts, ignore_index=True, sort=False)
    selected, lambda_metrics, cancer_metrics, admission, fallback, final = _select_and_compose(candidates)
    fit_audit = pd.concat(fit_parts, ignore_index=True) if fit_parts else pd.DataFrame()

    metric_rows = _metric_rows(selected, "M2_MODULE_SELECTED_OR_OFF")
    final_for_metrics = final.copy()
    final_for_metrics["module"] = "SELECTED"
    metric_rows.extend(_metric_rows(final_for_metrics, "R_SELECTED"))
    metrics = pd.DataFrame(metric_rows)
    paths = {
        "candidates": output_dir / "MULTIMODAL_RESIDUAL_CANDIDATES.parquet",
        "selected": output_dir / "MULTIMODAL_RESIDUAL_PREDICTIONS.parquet",
        "final": output_dir / "SELECTED_RESIDUAL_PREDICTIONS.parquet",
        "metrics": output_dir / "MULTIMODAL_RESIDUAL_METRICS.tsv",
        "lambda_metrics": output_dir / "CONTEXT_LAMBDA_VALIDATION_METRICS.tsv",
        "cancer_metrics": output_dir / "CONTEXT_CANCER_VALIDATION_METRICS.tsv",
        "admission": output_dir / "MODULE_ADMISSION_MATRIX.tsv",
        "fit": output_dir / "CONTEXT_RESIDUAL_FIT_AUDIT.tsv",
        "fallback": output_dir / "CONTEXT_FALLBACK_AUDIT.tsv",
        "lineage": output_dir / "CONTEXT_INPUT_LINEAGE.json",
        "manifest": output_dir / "CONTEXT_RESIDUAL_SHA256.tsv",
        "gate": output_dir / "CONTEXT_RESIDUAL_GATE.json",
    }
    for frame, key in (
        (candidates, "candidates"), (selected, "selected"), (final, "final"),
        (metrics, "metrics"), (lambda_metrics, "lambda_metrics"),
        (cancer_metrics, "cancer_metrics"), (admission, "admission"),
        (fit_audit, "fit"), (fallback, "fallback"),
    ):
        _atomic_table(frame, paths[key])
    lineage = {
        "residual_base_path": str(Path(residual_base_path).resolve()),
        "residual_base_sha256": file_sha256(residual_base_path),
        "best_simple_gate_path": str(Path(best_simple_gate_path).resolve()),
        "best_simple_gate_sha256": file_sha256(best_simple_gate_path),
        "graph_predictions_path": str(Path(graph_predictions_path).resolve()),
        "graph_predictions_sha256": file_sha256(graph_predictions_path),
        "graph_gate_path": str(Path(graph_gate_path).resolve()),
        "graph_gate_sha256": file_sha256(graph_gate_path),
        "b2_hard_report_path": str(Path(b2_hard_report_path).resolve()),
        "b2_hard_report_sha256": file_sha256(b2_hard_report_path),
        "modules": {key: dict(value) for key, value in module_lineage.items()},
        "test_labels_used_for_fit_or_selection": False,
    }
    atomic_write_json(paths["lineage"], lineage)
    output_files = [path for key, path in paths.items() if key not in {"manifest", "gate"}]
    manifest_rows = [
        {"relative_path": path.name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in output_files
    ]
    _atomic_table(pd.DataFrame(manifest_rows), paths["manifest"])
    payload = {
        "status": "PASS",
        "stage": "PHASE_B3_MULTIMODAL_CONTEXT_RESIDUAL",
        "pilot_cancers": list(PILOT_CANCERS),
        "model_seeds": list(PILOT_SEEDS),
        "shrinkage_grid": list(grids),
        "selection_scope": "pooled_outer_validation",
        "selection_used_test_labels": False,
        "minimum_validation_delta_auprc": 0.01,
        "minimum_positive_validation_cancers": 2,
        "singlecell_policy": "OFF_UNAVAILABLE_NO_PATIENT_LEVEL_PSEUDOBULK",
        "aggregation_git_commit": aggregation_git_commit,
        "best_simple_gate_sha256": file_sha256(best_simple_gate_path),
        "graph_gate_sha256": file_sha256(graph_gate_path),
        "b2_hard_report_sha256": file_sha256(b2_hard_report_path),
        "b2_hard_report_decision": hard_report["decision"],
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(final)),
        "manifest_sha256": file_sha256(paths["manifest"]),
        "manifest_merkle_sha256": merkle_sha256(manifest_rows),
        "full_cancer_training_started": False,
        "failures": [],
    }
    atomic_write_json(paths["gate"], payload)
    return payload
