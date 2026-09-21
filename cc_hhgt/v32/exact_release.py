"""Build the sanitized five-fold V3.2 exact-pathway ensemble release."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .full_model_contract import validate_module_lineage, validate_public_module_frame
from .input_lineage import artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
RUN_ID = "v32-exact-pathway-20260825"
KEYS = ["cancer_id", "lncrna_id", "pathway_id"]
ENSEMBLE_COLUMNS = [
    "association_membership_probability",
    "association_direction_probability",
    "l1_probability",
    "ridge_probability",
    "graph_residual",
    "graph_gate",
]


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def ensemble_fold_predictions(paths: list[str | Path], *, run_id: str = RUN_ID) -> pd.DataFrame:
    if len(paths) != 5:
        raise ValueError("Exactly five V3.2 fold prediction files are required")
    base: pd.DataFrame | None = None
    sums: dict[str, np.ndarray] = {}
    for expected_fold, value in enumerate(paths):
        path = Path(value)
        columns = KEYS + [
            "pathway_family_id",
            "shared_or_local_scope",
            "patient_fold_id",
            *ENSEMBLE_COLUMNS,
        ]
        frame = pd.read_parquet(path, columns=columns)
        if frame.patient_fold_id.nunique() != 1 or int(frame.patient_fold_id.iloc[0]) != expected_fold:
            raise ValueError(f"Fold file {path} does not declare patient_fold_id={expected_fold}")
        if frame[KEYS].duplicated().any():
            raise ValueError(f"Fold {expected_fold} duplicates exact candidates")
        frame = frame.sort_values(KEYS, kind="stable").reset_index(drop=True)
        if base is None:
            base = frame[KEYS + ["pathway_family_id", "shared_or_local_scope"]].copy()
            sums = {
                # Arrow-backed pandas columns may expose a read-only NumPy
                # view.  These arrays are accumulated in place below, so
                # own a writable copy rather than relying on backend-
                # specific mutability.
                column: pd.to_numeric(frame[column], errors="raise").to_numpy(
                    dtype=np.float64, copy=True
                )
                for column in ENSEMBLE_COLUMNS
            }
        else:
            if not frame[KEYS].equals(base[KEYS]):
                raise ValueError(f"Fold {expected_fold} exact candidate keys drift")
            if not frame[["pathway_family_id", "shared_or_local_scope"]].equals(
                base[["pathway_family_id", "shared_or_local_scope"]]
            ):
                raise ValueError(f"Fold {expected_fold} static candidate annotations drift")
            for column in ENSEMBLE_COLUMNS:
                sums[column] += pd.to_numeric(frame[column], errors="raise").to_numpy(np.float64)
    assert base is not None
    for column, values in sums.items():
        base[column] = (values / 5.0).astype(np.float32)
    base["association_direction"] = np.where(
        base.association_direction_probability.ge(0.5), "positive", "negative"
    )
    base["pathway_target_level"] = "exact_pathway"
    base["n_folds_available"] = 5
    base["analysis_version"] = ANALYSIS_VERSION
    base["training_run_id"] = str(run_id)
    base["changes_primary_ranking"] = True
    if not base.association_membership_probability.between(0, 1).all():
        raise ValueError("Exact ensemble probability is outside [0,1]")
    return base


def materialize_exact_release(
    *,
    fold_prediction_paths: list[str | Path],
    fold_metrics_paths: list[str | Path],
    training_success_paths: list[str | Path],
    checkpoint_paths: list[str | Path],
    candidate_path: str | Path,
    fold_manifest_path: str | Path,
    prep_summary_path: str | Path,
    output_root: str | Path,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    if (
        len(fold_metrics_paths) != 5
        or len(training_success_paths) != 5
        or len(checkpoint_paths) != 5
    ):
        raise ValueError(
            "Exactly five fold metrics, training SUCCESS files, and checkpoints are required"
        )
    successes = [json.loads(Path(path).read_text(encoding="utf-8")) for path in training_success_paths]
    metrics = [json.loads(Path(path).read_text(encoding="utf-8")) for path in fold_metrics_paths]
    for fold, success in enumerate(successes):
        if success.get("status") != "SUCCESS" or int(success.get("patient_fold", -1)) != fold:
            raise ValueError(f"Training fold {fold} lacks a matching SUCCESS attestation")
        if success.get("did_not_hit_hard_cap") is not True:
            raise ValueError(f"Training fold {fold} hit its hard cap")
        metric = metrics[fold]
        if int(metric.get("fold", -1)) != fold:
            raise ValueError(f"Metrics file for fold {fold} declares a different fold")
        if metric.get("candidate_alignment") != "FULL_ROW_EXACT":
            raise ValueError(f"Metrics fold {fold} lacks FULL_ROW_EXACT candidate alignment")
        checkpoint_cycle = int(metric.get("checkpoint_cycle", -1))
        completed_cycles = int(success.get("completed_cycles", -1))
        if checkpoint_cycle < 0 or checkpoint_cycle >= completed_cycles:
            raise ValueError(
                f"Metrics checkpoint cycle is outside the completed training span for fold {fold}"
            )
        for field in (
            "code_sha256",
            "config_sha256",
            "input_manifest_sha256",
            "task_manifest_sha256",
        ):
            if str(metric.get("artifact_hashes", {}).get(field, "")) != str(
                success.get("artifact_hashes", {}).get(field, "")
            ):
                raise ValueError(f"Metrics and SUCCESS {field} disagree for fold {fold}")
    hash_fields = (
        "code_sha256",
        "config_sha256",
        "input_manifest_sha256",
        "task_manifest_sha256",
    )
    for field in hash_fields:
        values = {str(item["artifact_hashes"][field]) for item in successes}
        if len(values) != 1:
            raise ValueError(f"Training {field} differs across folds")

    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    public = ensemble_fold_predictions(fold_prediction_paths, run_id=run_id)
    for fold, metric in enumerate(metrics):
        if int(metric.get("rows", -1)) != len(public):
            raise ValueError(f"Metrics fold {fold} row count differs from the exact universe")
        for model_key in ("cc_hhgt", "l1", "ridge"):
            if int(metric.get(model_key, {}).get("rows", -1)) != len(public):
                raise ValueError(f"Metrics fold {fold} {model_key} row count is incomplete")
    prediction_path = output / "exact_pathway_five_fold_ensemble.parquet"
    _atomic_parquet(public, prediction_path)
    checkpoint_records = []
    ensemble_source_records = []
    for fold, (success_path, checkpoint_path) in enumerate(
        zip(training_success_paths, checkpoint_paths, strict=True)
    ):
        prediction_source_path = Path(fold_prediction_paths[fold]).resolve()
        metrics_path = Path(fold_metrics_paths[fold]).resolve()
        success_source_path = Path(success_path).resolve()
        checkpoint_source_path = Path(checkpoint_path).resolve()
        checkpoint_record = {
            "patient_fold": fold,
            "seed": int(successes[fold]["seed"]),
            "checkpoint_path": str(checkpoint_source_path),
            "checkpoint_sha256": artifact_sha256(checkpoint_source_path),
            "success_path": str(success_source_path),
            "success_sha256": artifact_sha256(success_source_path),
            "source_checkpoint_sha256": None,
            "trained_from_random_initialization": True,
        }
        checkpoint_records.append(checkpoint_record)
        ensemble_source_records.append(
            {
                "patient_fold": fold,
                "seed": int(successes[fold]["seed"]),
                "prediction_path": str(prediction_source_path),
                "prediction_sha256": artifact_sha256(prediction_source_path),
                "prediction_rows": int(metrics[fold]["rows"]),
                "metrics_path": str(metrics_path),
                "metrics_sha256": artifact_sha256(metrics_path),
                "checkpoint_path": str(checkpoint_source_path),
                "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
                "success_path": str(success_source_path),
                "success_sha256": checkpoint_record["success_sha256"],
                "candidate_alignment": "FULL_ROW_EXACT",
                "checkpoint_cycle": int(metrics[fold]["checkpoint_cycle"]),
                "completed_cycles": int(successes[fold]["completed_cycles"]),
                "artifact_hashes": {
                    field: str(successes[fold]["artifact_hashes"][field])
                    for field in hash_fields
                },
                "analysis_version": ANALYSIS_VERSION,
                "trained_from_random_initialization": True,
                "old_checkpoint_loaded": False,
                "old_predictions_used_as_features": False,
                "old_rankings_used_as_outputs": False,
            }
        )
    checkpoint_manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "exact_pathway",
        "records": checkpoint_records,
        "all_five_folds_trained_from_random_initialization": True,
        "old_checkpoint_loaded": False,
    }
    checkpoint_manifest_path = output / "CHECKPOINT_MANIFEST.json"
    _atomic_json(checkpoint_manifest, checkpoint_manifest_path)
    ensemble_source_manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "exact_pathway",
        "training_run_id": str(run_id),
        "ensemble_formula": "arithmetic_mean_of_five_patient_fold_predictions",
        "ensemble_columns": ENSEMBLE_COLUMNS,
        "records": ensemble_source_records,
        "all_sources_current_v32": True,
        "all_five_folds_trained_from_random_initialization": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
    }
    ensemble_source_manifest_path = output / "ENSEMBLE_SOURCE_MANIFEST.json"
    _atomic_json(ensemble_source_manifest, ensemble_source_manifest_path)
    inputs = [
        {
            "path": str(Path(candidate_path).resolve()),
            "sha256": artifact_sha256(candidate_path),
            "artifact_kind": "standardized_input",
        },
        {
            "path": str(Path(fold_manifest_path).resolve()),
            "sha256": artifact_sha256(fold_manifest_path),
            "artifact_kind": "split_manifest",
        },
        {
            "path": str(Path(prep_summary_path).resolve()),
            "sha256": artifact_sha256(prep_summary_path),
            "artifact_kind": "standardized_input",
        },
    ]
    first_hashes = successes[0]["artifact_hashes"]
    lineage: dict[str, Any] = {
        "module_id": "exact_pathway",
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": str(run_id),
        "training_status": "SUCCESS",
        "initialization_policy": "TRAIN_FROM_RANDOM_INITIALIZATION",
        "folds": 5,
        "seeds": sorted({int(item["seed"]) for item in successes}),
        "code_sha256": first_hashes["code_sha256"],
        "config_sha256": first_hashes["config_sha256"],
        "input_manifest_sha256": first_hashes["input_manifest_sha256"],
        "checkpoint_manifest_sha256": artifact_sha256(checkpoint_manifest_path),
        "ensemble_source_manifest_path": str(ensemble_source_manifest_path),
        "ensemble_source_manifest_sha256": artifact_sha256(ensemble_source_manifest_path),
        "ensemble_source_records": ensemble_source_records,
        "ensemble_columns": ENSEMBLE_COLUMNS,
        "ensemble_formula": "arithmetic_mean_of_five_patient_fold_predictions",
        "input_artifacts": inputs,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "trained_from_scratch": True,
        "five_fold_ensemble": True,
        "seed_count": 1,
        "prediction_path": str(prediction_path),
        "prediction_sha256": artifact_sha256(prediction_path),
        "prediction_rows": int(len(public)),
        "held_out_labels_in_public_output": False,
    }
    validate_module_lineage("exact_pathway", lineage)
    validate_public_module_frame("exact_pathway", public)
    lineage_path = output / "MODULE_LINEAGE.json"
    _atomic_json(lineage, lineage_path)
    return {
        "status": "SUCCESS",
        "prediction_path": str(prediction_path),
        "prediction_rows": int(len(public)),
        "lineage_path": str(lineage_path),
        "contract_validation": "PASS",
    }


__all__ = ["ensemble_fold_predictions", "materialize_exact_release"]
