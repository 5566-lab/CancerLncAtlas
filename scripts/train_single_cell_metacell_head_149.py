#!/usr/bin/env python3
"""Train five donor-blocked signed metacell heads on host 149 CPU."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import torch
from torch import nn
import torch.nn.functional as F


FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_SIGNED_HEAD_V1"
RUN_CONFIG_FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_RUN_CONFIG_V1"
INPUT_LINEAGE_FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_INPUT_LINEAGE_V1"
CHECKPOINT_MANIFEST_FORMAT = (
    "CANCERLNCATLAS_V32_SC_METACELL_CHECKPOINT_MANIFEST_V1"
)
TRAINING_LINEAGE_FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_TRAINING_LINEAGE_V1"
FULL_CLOSURE_STATUS = "PASS_FULL_28_CANCER_MATERIALIZATION_CLOSURE"
FULL_CANCER_COUNT = 28
DISCOVERY_MIN_DONORS = 3
DISCOVERY_MIN_OUTER_FOLDS = 2
DISCOVERY_MAX_FDR = 0.05
DISCOVERY_MIN_ABS_RHO = 0.2
DISCOVERY_MIN_DIRECTION_CONSISTENCY = 0.75
ROW_KEY = ["cancer_id", "cell_key", "query_donor_id", "lncrna_id", "pathway_id"]
PAIR_KEY = ["dataset_id", "cancer_id", "cell_key", "lncrna_id", "pathway_id"]
FEATURE_COLUMNS = [
    "source_meta_rho",
    "source_direction_consistency",
    "source_donor_count",
    "n_metacells",
    "lnc_detection_rate",
    "is_bulk_candidate",
    "is_sc_discovery_candidate",
    "measurement_scale_log_normalized",
]
TARGET_METHOD = "WITHIN_DONOR_METACELL_SPEARMAN"
TARGET_OBSERVATION_UNIT = "within_donor_metacell"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_full_training_inputs(materialization_root: Path) -> dict[str, object]:
    """Require exact full-materialization and preprocessing closure."""
    validation_path = materialization_root / "MATERIALIZATION_VALIDATION.json"
    if not validation_path.is_file() or validation_path.is_symlink():
        raise RuntimeError("complete materialization validation is missing or unsafe")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("materialization validation status is not PASS")
    if validation.get("closure_status") != FULL_CLOSURE_STATUS:
        raise RuntimeError("materialization validation is not a full closure")
    if validation.get("validation_scope") != "FULL":
        raise RuntimeError("materialization validation scope is not FULL")
    if (
        validation.get("expected_cancer_count") != FULL_CANCER_COUNT
        or validation.get("validated_cancer_count") != FULL_CANCER_COUNT
    ):
        raise RuntimeError("materialization validation does not cover 28 cancers")
    expected = validation.get("expected_cancers")
    validated = validation.get("validated_cancers")
    if not isinstance(expected, list) or not isinstance(validated, list):
        raise RuntimeError("materialization validation cancer lists are absent")
    expected = [str(value).upper() for value in expected]
    validated = [str(value).upper() for value in validated]
    if (
        len(expected) != FULL_CANCER_COUNT
        or len(set(expected)) != FULL_CANCER_COUNT
        or sorted(expected) != sorted(validated)
    ):
        raise RuntimeError("expected and validated cancer lists differ")
    complete = sorted(
        path.name.upper()
        for path in (materialization_root / "cancers").glob("*")
        if path.is_dir() and (path / "SUCCESS.json").is_file()
    )
    if complete != sorted(expected):
        raise RuntimeError("complete materialization directories differ from validation")

    audit_path = materialization_root / "SINGLE_CELL_PREPROCESSING_AUDIT.json"
    if not audit_path.is_file() or audit_path.is_symlink():
        raise RuntimeError("single-cell preprocessing audit is missing or unsafe")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_cancers = [str(value).upper() for value in audit.get("active_cancers", [])]
    if (
        audit.get("status") != "PASS"
        or audit.get("active_cancer_count") != FULL_CANCER_COUNT
        or sorted(audit_cancers) != sorted(expected)
        or audit.get("eligible_cells_failing_preprocessing_pass_qc") != 0
        or audit.get("eligible_cells_failing_batch_correction_applied") != 0
        or audit.get("eligible_cells_failing_annotation_completed") != 0
    ):
        raise RuntimeError("single-cell preprocessing audit does not close all 28 cancers")
    return {
        "materialization_validation_path": str(validation_path.resolve()),
        "materialization_validation_sha256": sha256(validation_path),
        "materialization_closure_status": validation["closure_status"],
        "materialization_expected_cancers": sorted(expected),
        "preprocessing_audit_path": str(audit_path.resolve()),
        "preprocessing_audit_sha256": sha256(audit_path),
        "preprocessing_audit_eligible_cells": int(audit["eligible_training_cells"]),
    }


def file_record(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def build_run_config(settings) -> dict[str, object]:
    seeds = [int(settings.seed + fold * 101) for fold in range(5)]
    return {
        "format": RUN_CONFIG_FORMAT,
        "target_host": "149",
        "device": "cpu",
        "cuda_used": False,
        "seed": int(settings.seed),
        "base_seed": int(settings.seed),
        "fold_seeds": seeds,
        "outer_folds": 5,
        "epochs": int(settings.epochs),
        "patience": int(settings.patience),
        "batch_size": int(settings.batch_size),
        "max_train_rows": int(settings.max_train_rows),
        "learning_rate": float(settings.learning_rate),
        "optimizer": {"name": "AdamW", "weight_decay": 1e-4},
        "loss": {
            "name": "SmoothL1Loss",
            "beta": 0.1,
            "target": "signed_within_donor_metacell_spearman_rho",
            "row_weight": "sqrt(clip(n_metacells,20,200))_normalized_on_fit",
        },
        "numeric_feature_columns": FEATURE_COLUMNS,
        "architecture": {
            "frozen_lnc_core_dimensions": 96,
            "frozen_pathway_core_dimensions": 96,
            "frozen_cancer_core_dimensions": 96,
            "private_lnc_embedding_dimensions": 32,
            "celltype_embedding_dimensions": 16,
            "core_availability_flag_dimensions": 1,
            "input_dimensions": 96 * 3 + 32 + 16 + len(FEATURE_COLUMNS) + 1,
            "network": [
                "Linear(input,128)", "LayerNorm(128)", "GELU",
                "Dropout(0.10)", "Linear(128,64)", "GELU",
                "Linear(64,1)", "Tanh",
            ],
        },
        "discovery_thresholds": {
            "min_donors": DISCOVERY_MIN_DONORS,
            "min_outer_folds": DISCOVERY_MIN_OUTER_FOLDS,
            "max_fdr": DISCOVERY_MAX_FDR,
            "min_abs_meta_rho": DISCOVERY_MIN_ABS_RHO,
            "min_direction_consistency": DISCOVERY_MIN_DIRECTION_CONSISTENCY,
            "minimum_donors": DISCOVERY_MIN_DONORS,
            "minimum_outer_folds": DISCOVERY_MIN_OUTER_FOLDS,
            "maximum_meta_fdr": DISCOVERY_MAX_FDR,
            "minimum_absolute_observed_meta_rho": DISCOVERY_MIN_ABS_RHO,
            "minimum_direction_consistency": DISCOVERY_MIN_DIRECTION_CONSISTENCY,
            "bulk_candidate_must_be_false": True,
            "single_cell_discovery_candidate_must_be_true": True,
        },
        "fresh_random_initialization": True,
        "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None,
        "allow_partial_materialization": bool(
            settings.allow_partial_materialization
        ),
    }


def build_input_lineage(settings, full_contract) -> dict[str, object]:
    core_manifest_path = settings.core_root / "CORE_EMBEDDING_MANIFEST.json"
    if not core_manifest_path.is_file() or core_manifest_path.is_symlink():
        raise RuntimeError("core embedding manifest is missing or unsafe")
    core_manifest = json.loads(core_manifest_path.read_text(encoding="utf-8"))
    if (
        core_manifest.get("all_embeddings_from_newly_trained_v32_core") is not True
        or core_manifest.get("historical_checkpoint_loaded") is not False
        or core_manifest.get("historical_prediction_loaded") is not False
    ):
        raise RuntimeError("core embedding manifest is not a fresh frozen-core export")
    if full_contract is None:
        validation_path = settings.materialization_root / "MATERIALIZATION_VALIDATION.json"
        preprocessing_path = (
            settings.materialization_root / "SINGLE_CELL_PREPROCESSING_AUDIT.json"
        )
        if validation_path.is_file():
            validation_payload = json.loads(
                validation_path.read_text(encoding="utf-8")
            )
            materialization = {
                **file_record(validation_path),
                "status": validation_payload.get("status"),
                "closure_status": validation_payload.get("closure_status"),
                "validation_scope": validation_payload.get("validation_scope"),
            }
        else:
            materialization = None
        if preprocessing_path.is_file():
            preprocessing_payload = json.loads(
                preprocessing_path.read_text(encoding="utf-8")
            )
            preprocessing = {
                **file_record(preprocessing_path),
                "status": preprocessing_payload.get("status"),
                "active_cancer_count": preprocessing_payload.get(
                    "active_cancer_count"
                ),
            }
        else:
            preprocessing = None
        status = "PASS_PARTIAL_PILOT_INPUT_LINEAGE"
    else:
        validation_path = Path(full_contract["materialization_validation_path"])
        preprocessing_path = Path(full_contract["preprocessing_audit_path"])
        validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
        preprocessing_payload = json.loads(
            preprocessing_path.read_text(encoding="utf-8")
        )
        materialization = {
            **file_record(validation_path),
            "status": validation_payload.get("status"),
            "closure_status": validation_payload.get("closure_status"),
            "validation_scope": validation_payload.get("validation_scope"),
            "expected_cancer_count": validation_payload.get(
                "expected_cancer_count"
            ),
            "validated_cancer_count": validation_payload.get(
                "validated_cancer_count"
            ),
        }
        preprocessing = {
            **file_record(preprocessing_path),
            "status": preprocessing_payload.get("status"),
            "active_cancer_count": preprocessing_payload.get(
                "active_cancer_count"
            ),
            "eligible_training_cells": preprocessing_payload.get(
                "eligible_training_cells"
            ),
        }
        status = "PASS_FULL_28_CANCER_INPUT_LINEAGE"
    return {
        "format": INPUT_LINEAGE_FORMAT,
        "target_host": "149",
        "status": "PASS",
        "scope": status,
        "materialization_validation_path": (
            materialization["path"] if materialization is not None else None
        ),
        "materialization_validation_sha256": (
            materialization["sha256"] if materialization is not None else None
        ),
        "preprocessing_audit_path": (
            preprocessing["path"] if preprocessing is not None else None
        ),
        "preprocessing_audit_sha256": (
            preprocessing["sha256"] if preprocessing is not None else None
        ),
        "core_manifest_path": str(core_manifest_path.resolve()),
        "core_manifest_sha256": sha256(core_manifest_path),
        "materialization_validation": materialization,
        "preprocessing_audit": preprocessing,
        "core_manifest": {
            **file_record(core_manifest_path),
            "status": "PASS_FRESH_FROZEN_CORE_EMBEDDINGS",
            "all_embeddings_from_newly_trained_v32_core": True,
            "historical_checkpoint_loaded": False,
            "historical_prediction_loaded": False,
            "training_generation": core_manifest.get("training_generation"),
            "analysis_version": core_manifest.get("analysis_version"),
        },
        "full_input_contract": full_contract,
        "old_checkpoint_loaded": False,
        "old_predictions_loaded": False,
        "old_rankings_loaded": False,
        "old_training_outputs_reused": False,
    }


def build_checkpoint_manifest(reports: list[dict]) -> dict[str, object]:
    if len(reports) != 5 or sorted(item["outer_fold_id"] for item in reports) != list(range(5)):
        raise RuntimeError("checkpoint manifest requires exactly five outer folds")
    records = []
    for report in sorted(reports, key=lambda item: item["outer_fold_id"]):
        if (
            report.get("fresh_random_initialization") is not True
            or report.get("old_checkpoint_loaded") is not False
            or report.get("source_checkpoint_sha256") is not None
            or report.get("device") != "cpu"
            or report.get("cuda_used") is not False
            or report["initial_parameter_sha256"]
            == report["trained_parameter_sha256"]
            or report["frozen_core_sha256_before"]
            != report["frozen_core_sha256_after"]
            or not report.get("history")
        ):
            raise RuntimeError("fold checkpoint lineage contract failed")
        fold_success = Path(report["files"]["checkpoint"]["path"]).parent / "SUCCESS.json"
        if not fold_success.is_file():
            raise RuntimeError("fold SUCCESS receipt is absent")
        records.append({
            "outer_fold_id": int(report["outer_fold_id"]),
            "seed": int(report["seed"]),
            "fold_success": file_record(fold_success),
            "checkpoint": report["files"]["checkpoint"],
            "predictions": report["files"]["predictions"],
            "initial_parameter_sha256": report["initial_parameter_sha256"],
            "trained_parameter_sha256": report["trained_parameter_sha256"],
            "frozen_core_sha256_before": report["frozen_core_sha256_before"],
            "frozen_core_sha256_after": report["frozen_core_sha256_after"],
            "core_files": report["core_files"],
            "fresh_random_initialization": True,
            "old_checkpoint_loaded": False,
            "source_checkpoint_sha256": None,
            "device": "cpu",
            "cuda_used": False,
            "epochs_completed": int(report["epochs_completed"]),
            "best_validation_loss": float(report["best_validation_loss"]),
            "history": report["history"],
        })
    return {
        "format": CHECKPOINT_MANIFEST_FORMAT,
        "target_host": "149",
        "status": "SUCCESS",
        "outer_folds": 5,
        "models_trained": 5,
        "fold_seeds": [item["seed"] for item in records],
        "fresh_random_initialization": True,
        "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None,
        "device": "cpu",
        "cuda_used": False,
        "records": records,
    }


def collect_artifacts(root: Path, excluded_names=()) -> dict[str, dict[str, object]]:
    excluded = set(excluded_names)
    result = {}
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and not path.is_symlink()
            and path.name not in excluded
            and not path.name.endswith(".partial")
        ):
            result[str(path.relative_to(root)).replace("\\", "/")] = file_record(path)
    return result


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def normalize_lnc(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    for prefix in ("LNCRNA:", "LNC:"):
        if upper.startswith(prefix):
            return text[len(prefix):]
    return text


def load_core_table(path: Path, kind: str) -> tuple[list[str], np.ndarray, str]:
    frame = pd.read_parquet(path)
    features = sorted(column for column in frame if column.startswith("core_feature_"))
    if len(features) != 96 or frame.node_id.astype(str).duplicated().any():
        raise RuntimeError(f"invalid {kind} core embedding table")
    ids = frame.node_id.astype(str).tolist()
    if kind == "lncRNA":
        ids = [normalize_lnc(value) for value in ids]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate normalized {kind} core IDs")
    values = frame[features].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite {kind} core embeddings")
    return ids, values, sha256(path)


class SignedMetacellHead(nn.Module):
    def __init__(
        self,
        lnc_core: np.ndarray,
        pathway_core: np.ndarray,
        cancer_core: np.ndarray,
        private_lncs: int,
        celltypes: int,
        numeric_features: int,
    ) -> None:
        super().__init__()
        self.register_buffer("lnc_core", torch.as_tensor(lnc_core, dtype=torch.float32))
        self.register_buffer("pathway_core", torch.as_tensor(pathway_core, dtype=torch.float32))
        self.register_buffer("cancer_core", torch.as_tensor(cancer_core, dtype=torch.float32))
        # Index zero is an immutable all-zero fallback for identities absent
        # from the outer-training rows.  Query identities never create random,
        # untrained embedding rows.
        self.private_lnc = nn.Embedding(private_lncs, 32, padding_idx=0)
        self.celltype = nn.Embedding(celltypes, 16, padding_idx=0)
        width = 96 * 3 + 32 + 16 + numeric_features + 1
        self.network = nn.Sequential(
            nn.Linear(width, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        lnc_core_index,
        pathway_core_index,
        cancer_core_index,
        private_lnc_index,
        celltype_index,
        numeric,
        core_available,
    ):
        features = torch.cat(
            [
                self.lnc_core[lnc_core_index],
                self.pathway_core[pathway_core_index],
                self.cancer_core[cancer_core_index],
                self.private_lnc(private_lnc_index),
                self.celltype(celltype_index),
                numeric,
                core_available[:, None],
            ],
            dim=1,
        )
        return torch.tanh(self.network(features).squeeze(1))


def _read_fold_rows(materialization: Path, fold: int, name: str) -> pd.DataFrame:
    columns = [
        "lncrna_id", "pathway_id", "query_donor_id", "rho", "n_metacells",
        "lnc_detection_rate", "is_bulk_candidate", "is_sc_discovery_candidate",
        "source_donor_count", "source_meta_rho", "source_direction_consistency",
        "dataset_id", "cancer_id", "cell_key", "outer_fold_id", "measurement_scale",
        "candidate_selection_query_donor_used", "association_method",
        "association_observation_unit", "donor_resolved_target",
    ]
    parts = []
    for cancer in sorted((materialization / "cancers").glob("*")):
        if not (cancer / "SUCCESS.json").is_file():
            continue
        path = cancer / f"outer_fold={fold}" / name
        if path.is_file():
            frame = pd.read_parquet(path)
            if len(frame):
                missing = sorted(set(columns) - set(frame.columns))
                if missing:
                    raise RuntimeError(f"{path} lacks {missing}")
                parts.append(frame[columns])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


def _sample_training(
    frame: pd.DataFrame,
    maximum: int,
    seed: int,
    anchor_columns=("lncrna_id",),
) -> pd.DataFrame:
    """Deterministically sample rows while retaining one row per anchor group."""
    if maximum < 1:
        raise RuntimeError("sample maximum must be positive")
    if len(frame) <= maximum:
        return frame.reset_index(drop=True)
    missing = sorted(set(anchor_columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"sampling anchor columns are missing: {missing}")
    ordered = frame.copy()
    ordered["_sample_hash"] = [
        _stable_hash(
            f"{seed}|{row.cancer_id}|{row.cell_key}|{row.query_donor_id}|"
            f"{row.lncrna_id}|{row.pathway_id}"
        )
        for row in ordered.itertuples(index=False)
    ]
    anchors = (
        ordered.sort_values("_sample_hash", kind="mergesort")
        .groupby(list(anchor_columns), observed=True, sort=False, dropna=False)
        .head(1)
    )
    if len(anchors) > maximum:
        raise RuntimeError(
            f"sample maximum {maximum} cannot retain {len(anchors)} anchor groups"
        )
    remaining = ordered.drop(index=anchors.index).nsmallest(
        maximum - len(anchors), "_sample_hash"
    )
    return pd.concat([anchors, remaining], ignore_index=True).drop(columns="_sample_hash")


def _sample_donor_split_training(
    frame: pd.DataFrame,
    tuning_mask,
    maximum: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    """Sample after donor splitting, preserving fit lnc and tuning donor anchors."""
    tuning_mask = np.asarray(tuning_mask, dtype=bool)
    if len(tuning_mask) != len(frame) or not tuning_mask.any() or tuning_mask.all():
        raise RuntimeError("invalid donor tuning mask for split-aware sampling")
    if maximum < 2:
        raise RuntimeError("sample maximum must retain both fit and tuning rows")

    fit = frame.loc[~tuning_mask].copy()
    tuning = frame.loc[tuning_mask].copy()
    target_total = min(len(frame), maximum)
    fit_minimum = int(fit.lncrna_id.astype(str).nunique(dropna=False))
    tuning_minimum = int(
        tuning[["cancer_id", "query_donor_id"]].drop_duplicates().shape[0]
    )
    lower_tuning = max(tuning_minimum, target_total - len(fit))
    upper_tuning = min(len(tuning), target_total - fit_minimum)
    if lower_tuning > upper_tuning:
        raise RuntimeError(
            "sample maximum cannot retain every fit lncRNA and tuning donor anchor"
        )
    desired_tuning = int(round(target_total * len(tuning) / len(frame)))
    target_tuning = min(max(desired_tuning, lower_tuning), upper_tuning)
    target_fit = target_total - target_tuning

    sampled_fit = _sample_training(
        fit, target_fit, seed, anchor_columns=("lncrna_id",)
    )
    sampled_tuning = _sample_training(
        tuning,
        target_tuning,
        seed + 1_000_003,
        anchor_columns=("cancer_id", "query_donor_id"),
    )
    sampled = pd.concat([sampled_fit, sampled_tuning], ignore_index=True)
    sampled_mask = np.concatenate([
        np.zeros(len(sampled_fit), dtype=bool),
        np.ones(len(sampled_tuning), dtype=bool),
    ])
    sampling = {
        "fit_rows_available": len(fit),
        "tuning_rows_available": len(tuning),
        "fit_rows_sampled": len(sampled_fit),
        "tuning_rows_sampled": len(sampled_tuning),
        "fit_lncrna_anchors_required": fit_minimum,
        "fit_lncrna_anchors_retained": int(
            sampled_fit.lncrna_id.astype(str).nunique(dropna=False)
        ),
        "tuning_donor_anchors_required": tuning_minimum,
        "tuning_donor_anchors_retained": int(
            sampled_tuning[["cancer_id", "query_donor_id"]]
            .drop_duplicates()
            .shape[0]
        ),
    }
    return sampled, sampled_mask, sampling


def _prepare_numeric(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame.source_meta_rho.astype(float),
            frame.source_direction_consistency.astype(float),
            np.log1p(frame.source_donor_count.astype(float)),
            np.log1p(frame.n_metacells.astype(float)),
            frame.lnc_detection_rate.astype(float),
            frame.is_bulk_candidate.astype(float),
            frame.is_sc_discovery_candidate.astype(float),
            frame.measurement_scale.astype(str).eq("log_normalized").astype(float),
        ]
    ).astype(np.float32)


def _indices(frame, lnc_ids, pathway_ids, cancer_ids, private_index, cell_index):
    lnc_pos = pd.Index(lnc_ids).get_indexer(frame.lncrna_id.astype(str))
    lnc_available = lnc_pos >= 0
    lnc_pos = np.where(lnc_available, lnc_pos, len(lnc_ids)).astype(np.int64)
    pathway_pos = pd.Index(pathway_ids).get_indexer(frame.pathway_id.astype(str))
    cancer_pos = pd.Index(cancer_ids).get_indexer(frame.cancer_id.astype(str))
    if (pathway_pos < 0).any() or (cancer_pos < 0).any():
        raise RuntimeError("pathway/cancer is absent from frozen core")
    private_values = frame.lncrna_id.astype(str).map(private_index)
    cell_values = frame.cell_key.astype(str).map(cell_index)
    private_available = private_values.notna().to_numpy()
    cell_available = cell_values.notna().to_numpy()
    private_pos = private_values.fillna(0).to_numpy(np.int64)
    cell_pos = cell_values.fillna(0).to_numpy(np.int64)
    return {
        "lnc": lnc_pos,
        "pathway": pathway_pos.astype(np.int64),
        "cancer": cancer_pos.astype(np.int64),
        "private": private_pos,
        "cell": cell_pos,
        "core_available": lnc_available.astype(np.float32),
        "private_available": private_available,
        "celltype_available": cell_available,
    }


def _embedding_routes(core_available, private_available) -> np.ndarray:
    core = np.asarray(core_available, dtype=bool)
    private = np.asarray(private_available, dtype=bool)
    return np.select(
        [core & private, core, private],
        [
            "FROZEN_CORE_PLUS_PRIVATE_ID_ADAPTER",
            "FROZEN_CORE_WITH_ZERO_PRIVATE_FALLBACK",
            "SINGLE_CELL_PRIVATE_OOV_ID_EMBEDDING",
        ],
        default="SINGLE_CELL_OOV_ZERO_ID_FALLBACK",
    )


def _validate_fold_rows(frame: pd.DataFrame, fold: int, role: str) -> None:
    if frame.empty:
        raise RuntimeError(f"fold {fold} lacks {role} rows")
    if not frame.outer_fold_id.astype(int).eq(fold).all():
        raise RuntimeError(f"{role} row fold identity drift")
    if frame.duplicated(ROW_KEY).any():
        raise RuntimeError(f"duplicate {role} donor target")
    numeric = frame[[
        "rho", "n_metacells", "lnc_detection_rate", "source_donor_count",
        "source_meta_rho", "source_direction_consistency",
    ]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"non-finite {role} target/features")
    if frame.rho.astype(float).abs().gt(1.000001).any():
        raise RuntimeError(f"{role} rho is outside [-1, 1]")
    if frame.source_meta_rho.astype(float).abs().gt(1.000001).any():
        raise RuntimeError(f"{role} source meta-rho is outside [-1, 1]")
    if frame.candidate_selection_query_donor_used.astype(bool).any():
        raise RuntimeError(f"outer query donor was used for {role} candidate selection")
    if not frame.association_method.astype(str).eq(TARGET_METHOD).all():
        raise RuntimeError(f"{role} association method drift")
    if not frame.association_observation_unit.astype(str).eq(
        TARGET_OBSERVATION_UNIT
    ).all():
        raise RuntimeError(f"{role} observation unit drift")
    if not frame.donor_resolved_target.eq(True).all():
        raise RuntimeError(f"{role} target is not donor resolved")


def _donor_tuning_mask(frame: pd.DataFrame, fold: int) -> np.ndarray:
    """Choose an exact donor-blocked tuning split without row-level fallback."""
    keys = list(zip(frame.cancer_id.astype(str), frame.query_donor_id.astype(str)))
    groups = sorted(
        set(keys),
        key=lambda item: _stable_hash(f"tuning|{fold}|{item[0]}|{item[1]}"),
    )
    if len(groups) < 2:
        raise RuntimeError("at least two outer-training donor groups are required")
    tuning_count = min(max(1, int(round(len(groups) * 0.10))), len(groups) - 1)
    tuning = set(groups[:tuning_count])
    mask = np.fromiter((key in tuning for key in keys), dtype=bool, count=len(keys))
    if not mask.any() or mask.all():
        raise RuntimeError("failed to construct donor-blocked tuning split")
    return mask


def _tensor(values, dtype=None):
    return torch.as_tensor(values, dtype=dtype)


def _forward(model, indices, numeric, rows):
    return model(
        _tensor(indices["lnc"][rows], torch.long),
        _tensor(indices["pathway"][rows], torch.long),
        _tensor(indices["cancer"][rows], torch.long),
        _tensor(indices["private"][rows], torch.long),
        _tensor(indices["cell"][rows], torch.long),
        _tensor(numeric[rows], torch.float32),
        _tensor(indices["core_available"][rows], torch.float32),
    )


def _predict(model, indices, numeric, batch_size: int) -> np.ndarray:
    model.eval()
    output = np.empty(len(numeric), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(numeric), batch_size):
            rows = np.arange(start, min(start + batch_size, len(numeric)))
            output[rows] = _forward(model, indices, numeric, rows).cpu().numpy()
    return output


def train_fold(settings, fold: int, output: Path) -> tuple[dict, pd.DataFrame]:
    fold_output = output / f"outer_fold={fold}"
    if fold_output.exists():
        raise RuntimeError(f"training output already exists: {fold_output}")
    fold_output.mkdir(parents=True)
    train = _read_fold_rows(settings.materialization_root, fold, "TRAIN_ROWS.parquet")
    query = _read_fold_rows(settings.materialization_root, fold, "QUERY_ROWS.parquet")
    _validate_fold_rows(train, fold, "train")
    _validate_fold_rows(query, fold, "query")
    query_keys = set(zip(query.cancer_id.astype(str), query.query_donor_id.astype(str)))
    train_keys = set(zip(train.cancer_id.astype(str), train.query_donor_id.astype(str)))
    if query_keys & train_keys:
        raise RuntimeError("outer query donor leaked into training rows")
    train_rows_available = len(train)
    # Fix the donor split on the complete outer-training data before row
    # sampling. This prevents sampling from moving a sole lncRNA anchor into
    # tuning and keeps all learned identity vocabularies strictly fit-only.
    tuning_mask = _donor_tuning_mask(train, fold)
    train, tuning_mask, sampling = _sample_donor_split_training(
        train, tuning_mask, settings.max_train_rows, settings.seed + fold
    )
    fit_rows = np.flatnonzero(~tuning_mask)
    tuning_rows = np.flatnonzero(tuning_mask)
    fit_donor_keys = set(zip(
        train.loc[fit_rows, "cancer_id"].astype(str),
        train.loc[fit_rows, "query_donor_id"].astype(str),
    ))
    tuning_donor_keys = set(zip(
        train.loc[tuning_rows, "cancer_id"].astype(str),
        train.loc[tuning_rows, "query_donor_id"].astype(str),
    ))
    if fit_donor_keys & tuning_donor_keys:
        raise RuntimeError("donor leaked between fit and tuning rows")
    core_dir = settings.core_root / f"patient_fold={fold}"
    lnc_ids, lnc_core, lnc_sha = load_core_table(core_dir / "lncRNA.parquet", "lncRNA")
    pathway_ids, pathway_core, pathway_sha = load_core_table(
        core_dir / "pathway.parquet", "pathway"
    )
    cancer_ids, cancer_core, cancer_sha = load_core_table(core_dir / "cancer.parquet", "cancer")
    lnc_core = np.vstack([lnc_core, np.zeros((1, lnc_core.shape[1]), np.float32)])
    # Build learned identity vocabularies only from rows used by the optimizer.
    # Index zero is the frozen fallback reserved by the embedding modules.
    training_lncs = sorted(set(train.loc[fit_rows, "lncrna_id"].astype(str)))
    training_cells = sorted(set(train.loc[fit_rows, "cell_key"].astype(str)))
    private_index = {value: index + 1 for index, value in enumerate(training_lncs)}
    cell_index = {value: index + 1 for index, value in enumerate(training_cells)}
    train_indices = _indices(
        train, lnc_ids, pathway_ids, cancer_ids, private_index, cell_index
    )
    query_indices = _indices(
        query, lnc_ids, pathway_ids, cancer_ids, private_index, cell_index
    )
    unresolved_tuning_oov = (
        ~train_indices["core_available"][tuning_rows].astype(bool)
        & ~train_indices["private_available"][tuning_rows]
    )
    unresolved_query_oov = (
        ~query_indices["core_available"].astype(bool)
        & ~query_indices["private_available"]
    )
    train_numeric = _prepare_numeric(train)
    query_numeric = _prepare_numeric(query)
    mean = train_numeric[fit_rows].mean(axis=0)
    sd = train_numeric[fit_rows].std(axis=0)
    sd[sd < 1e-6] = 1.0
    train_numeric = (train_numeric - mean) / sd
    query_numeric = (query_numeric - mean) / sd
    seed = settings.seed + fold * 101
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = SignedMetacellHead(
        lnc_core, pathway_core, cancer_core,
        len(training_lncs) + 1, len(training_cells) + 1, len(FEATURE_COLUMNS)
    )
    if any(value.device.type != "cpu" for value in model.state_dict().values()):
        raise RuntimeError("single-cell head must remain entirely on CPU")
    initial_hash = state_hash(model.state_dict())
    frozen_before = state_hash(
        {key: value for key, value in model.state_dict().items() if key.endswith("_core")}
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=1e-4)
    target = train.rho.to_numpy(np.float32)
    weight = np.sqrt(np.clip(train.n_metacells.to_numpy(np.float32), 20, 200))
    weight /= weight[fit_rows].mean()
    rng = np.random.default_rng(seed)
    best = None
    best_loss = float("inf")
    stale = 0
    history = []
    for epoch in range(settings.epochs):
        model.train()
        shuffled = rng.permutation(fit_rows)
        losses = []
        for start in range(0, len(shuffled), settings.batch_size):
            rows = shuffled[start:start + settings.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, train_indices, train_numeric, rows)
            loss = (
                F.smooth_l1_loss(
                    prediction, _tensor(target[rows], torch.float32), reduction="none", beta=0.1
                ) * _tensor(weight[rows], torch.float32)
            ).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation = []
            for start in range(0, len(tuning_rows), settings.batch_size):
                rows = tuning_rows[start:start + settings.batch_size]
                prediction = _forward(model, train_indices, train_numeric, rows)
                loss = F.smooth_l1_loss(
                    prediction, _tensor(target[rows], torch.float32), reduction="mean", beta=0.1
                )
                validation.append(float(loss))
        validation_loss = float(np.mean(validation))
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if best is None:
        raise RuntimeError("no best model state")
    model.load_state_dict(best)
    frozen_after = state_hash(
        {key: value for key, value in model.state_dict().items() if key.endswith("_core")}
    )
    if frozen_before != frozen_after:
        raise RuntimeError("frozen core embeddings changed during private-head training")
    trained_hash = state_hash(model.state_dict())
    if trained_hash == initial_hash:
        raise RuntimeError("optimizer completed without changing private-head parameters")
    predicted = _predict(model, query_indices, query_numeric, settings.batch_size)
    predictions = query.copy()
    predictions["predicted_rho"] = predicted
    predictions["predicted_association_strength"] = np.abs(predicted)
    predictions["lncrna_core_available"] = query_indices["core_available"].astype(bool)
    predictions["private_lnc_embedding_available"] = query_indices["private_available"]
    predictions["celltype_embedding_available"] = query_indices["celltype_available"]
    predictions["embedding_route"] = _embedding_routes(
        predictions.lncrna_core_available,
        predictions.private_lnc_embedding_available,
    )
    predictions["residual"] = predictions.rho - predictions.predicted_rho
    predictions.to_parquet(fold_output / "QUERY_PREDICTIONS.parquet", index=False)
    checkpoint = {
        "format": FORMAT,
        "outer_fold_id": fold,
        "seed": seed,
        "state_dict": model.state_dict(),
        "lncrna_vocabulary": training_lncs,
        "celltype_vocabulary": training_cells,
        "unknown_identity_index": 0,
        "numeric_mean": mean,
        "numeric_sd": sd,
        "numeric_columns": FEATURE_COLUMNS,
        "history": history,
        "initial_parameter_sha256": initial_hash,
        "trained_parameter_sha256": trained_hash,
        "frozen_core_sha256_before": frozen_before,
        "frozen_core_sha256_after": frozen_after,
        "fresh_random_initialization": True,
        "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None,
        "device": "cpu",
        "cuda_used": False,
    }
    torch.save(checkpoint, fold_output / "best_model_state.pt")
    observed = predictions.rho.to_numpy(float)
    baseline = predictions.source_meta_rho.to_numpy(float)
    report = {
        "format": FORMAT,
        "target_host": "149",
        "state": "FOLD_TRAINED_AND_OOF_PREDICTED",
        "outer_fold_id": fold,
        "seed": seed,
        "train_rows_available": train_rows_available,
        "train_rows_sampled": len(train),
        "donor_split_before_sampling": True,
        **sampling,
        "fit_rows": len(fit_rows),
        "tuning_rows": len(tuning_rows),
        "fit_donor_groups": len(fit_donor_keys),
        "tuning_donor_groups": len(tuning_donor_keys),
        "fit_tuning_donor_overlap": 0,
        "query_rows": len(query),
        "query_donors": int(query.query_donor_id.nunique()),
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "query_mae": float(np.mean(np.abs(predicted - observed))),
        "source_meta_baseline_mae": float(np.mean(np.abs(baseline - observed))),
        "initial_parameter_sha256": initial_hash,
        "trained_parameter_sha256": trained_hash,
        "frozen_core_sha256_before": frozen_before,
        "frozen_core_sha256_after": frozen_after,
        "fresh_random_initialization": True,
        "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None,
        "device": "cpu",
        "cuda_used": False,
        "core_files": {
            "lncRNA": lnc_sha, "pathway": pathway_sha, "cancer": cancer_sha,
        },
        "oov_query_rows": int((~predictions.lncrna_core_available).sum()),
        "query_rows_without_private_lnc_embedding": int(
            (~predictions.private_lnc_embedding_available).sum()
        ),
        "tuning_oov_zero_id_fallback_rows": int(unresolved_tuning_oov.sum()),
        "query_oov_zero_id_fallback_rows": int(unresolved_query_oov.sum()),
        "query_rows_without_celltype_embedding": int(
            (~predictions.celltype_embedding_available).sum()
        ),
        "files": {
            "checkpoint": file_record(fold_output / "best_model_state.pt"),
            "predictions": file_record(fold_output / "QUERY_PREDICTIONS.parquet"),
        },
        "history": history,
        "finished_unix": time.time(),
    }
    atomic_json(fold_output / "SUCCESS.json", report)
    return report, predictions


def _bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    if np.isfinite(p).any() and (
        np.nanmin(p) < -1e-12 or np.nanmax(p) > 1.0 + 1e-12
    ):
        raise RuntimeError("p-value outside [0, 1]")
    result = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return result
    order = finite[np.argsort(p[finite], kind="mergesort")]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def aggregate_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.duplicated(ROW_KEY).any():
        raise RuntimeError("OOF donor target was predicted more than once")
    if predictions.empty:
        raise RuntimeError("no OOF donor predictions")
    observed = predictions.rho.to_numpy(float)
    predicted = predictions.predicted_rho.to_numpy(float)
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise RuntimeError("non-finite OOF observed/predicted rho")
    if (np.abs(observed) > 1.000001).any() or (np.abs(predicted) > 1.000001).any():
        raise RuntimeError("OOF observed/predicted rho outside [-1, 1]")
    frame = predictions.copy()
    frame["_fisher"] = np.arctanh(np.clip(frame.rho.astype(float), -0.999999, 0.999999))
    frame["_fisher_sq"] = frame._fisher ** 2
    frame["_pred_fisher"] = np.arctanh(
        np.clip(frame.predicted_rho.astype(float), -0.999999, 0.999999)
    )
    frame["_positive"] = frame.rho.gt(0).astype(int)
    frame["_negative"] = frame.rho.lt(0).astype(int)
    grouped = frame.groupby(PAIR_KEY, observed=True).agg(
        n_donors=("query_donor_id", "nunique"),
        observed_rows=("query_donor_id", "size"),
        outer_fold_count=("outer_fold_id", "nunique"),
        fisher_sum=("_fisher", "sum"),
        fisher_sq_sum=("_fisher_sq", "sum"),
        predicted_fisher_sum=("_pred_fisher", "sum"),
        positive_donors=("_positive", "sum"),
        negative_donors=("_negative", "sum"),
        is_bulk_candidate=("is_bulk_candidate", "max"),
        is_sc_discovery_candidate=("is_sc_discovery_candidate", "max"),
        lnc_core_available=("lncrna_core_available", "max"),
        mean_n_metacells=("n_metacells", "mean"),
    ).reset_index()
    if not grouped.observed_rows.eq(grouped.n_donors).all():
        raise RuntimeError("a donor contributes multiple effects to one association")
    n = grouped.n_donors.to_numpy(float)
    mean = grouped.fisher_sum.to_numpy(float) / n
    variance = np.divide(
        grouped.fisher_sq_sum.to_numpy(float) - grouped.fisher_sum.to_numpy(float) ** 2 / n,
        n - 1,
        out=np.full(len(grouped), np.nan),
        where=n > 1,
    )
    standard_error = np.sqrt(np.maximum(variance, 0.0) / n)
    statistic = np.divide(
        mean,
        standard_error,
        out=np.full(len(grouped), np.nan),
        where=standard_error > 1e-12,
    )
    p_value = 2.0 * student_t.sf(np.abs(statistic), np.maximum(n - 1, 1))
    exact = (standard_error <= 1e-12) & (n > 1)
    p_value[exact & (np.abs(mean) > 1e-12)] = 0.0
    p_value[exact & (np.abs(mean) <= 1e-12)] = 1.0
    grouped["observed_meta_rho"] = np.tanh(mean)
    grouped["predicted_meta_rho"] = np.tanh(
        grouped.predicted_fisher_sum.to_numpy(float) / n
    )
    signed_donors = grouped.positive_donors + grouped.negative_donors
    grouped["direction_consistency"] = np.divide(
        np.maximum(grouped.positive_donors, grouped.negative_donors),
        signed_donors,
        out=np.full(len(grouped), np.nan),
        where=signed_donors > 0,
    )
    grouped["meta_p_value"] = p_value
    grouped["meta_fdr"] = np.nan
    for _, rows in grouped.groupby(["cancer_id", "cell_key"], observed=True).groups.items():
        grouped.loc[rows, "meta_fdr"] = _bh(grouped.loc[rows, "meta_p_value"])
    grouped["multiple_testing_family"] = "CANCER_X_CELLTYPE_CROSSFIT_SELECTED_PAIRS"
    grouped["new_relationship_discovered"] = (
        ~grouped.is_bulk_candidate.astype(bool)
        & grouped.is_sc_discovery_candidate.astype(bool)
        & grouped.n_donors.ge(DISCOVERY_MIN_DONORS)
        & grouped.outer_fold_count.ge(DISCOVERY_MIN_OUTER_FOLDS)
        & grouped.meta_fdr.le(DISCOVERY_MAX_FDR)
        & grouped.observed_meta_rho.abs().ge(DISCOVERY_MIN_ABS_RHO)
        & grouped.direction_consistency.ge(DISCOVERY_MIN_DIRECTION_CONSISTENCY)
    )
    grouped["discovery_semantics"] = (
        "CROSSFIT_SOURCE_SCREEN_HELDOUT_DONOR_EFFECT_META_ANALYSIS"
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-root", required=True, type=Path)
    parser.add_argument("--core-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--max-train-rows", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--allow-partial-materialization", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != "149":
        raise SystemExit("WRONG_HOST")
    if args.output.exists():
        raise SystemExit("OUTPUT_ALREADY_EXISTS")
    full_input_contract = None
    if not args.allow_partial_materialization:
        try:
            full_input_contract = validate_full_training_inputs(
                args.materialization_root
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise SystemExit(
                f"COMPLETE_MATERIALIZATION_VALIDATION_REQUIRED: {exc}"
            ) from exc
    run_config = build_run_config(args)
    input_lineage = build_input_lineage(args, full_input_contract)
    args.output.mkdir(parents=True)
    run_config_path = args.output / "RUN_CONFIG.json"
    input_lineage_path = args.output / "INPUT_LINEAGE_AUDIT.json"
    atomic_json(run_config_path, run_config)
    atomic_json(input_lineage_path, input_lineage)
    trainer_script = Path(__file__).resolve()
    atomic_json(
        args.output / "LAUNCH_PREFLIGHT.json",
        {
            "format": FORMAT,
            "target_host": "149",
            "cpu_only": True,
            "paid_compute": False,
            "materialization_root": str(args.materialization_root.resolve()),
            "core_root": str(args.core_root.resolve()),
            "allow_partial_materialization": args.allow_partial_materialization,
            "full_input_contract": full_input_contract,
            "trainer_script": file_record(trainer_script),
            "run_config": file_record(run_config_path),
            "input_lineage_audit": file_record(input_lineage_path),
            "fresh_random_initialization": True,
            "old_checkpoint_loaded": False,
            "source_checkpoint_sha256": None,
            "device": "cpu",
            "cuda_used": False,
            "started_unix": time.time(),
        },
    )
    reports = []
    prediction_parts = []
    for fold in range(5):
        report, predictions = train_fold(args, fold, args.output)
        reports.append(report)
        prediction_parts.append(predictions)
        print(json.dumps({"outer_fold_id": fold, "state": "TRAINED", "query_rows": len(predictions)}), flush=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_parquet(args.output / "OOF_DONOR_PREDICTIONS.parquet", index=False)
    associations = aggregate_oof(predictions)
    associations.to_parquet(args.output / "CROSSFIT_ASSOCIATIONS.parquet", index=False)
    discoveries = associations.loc[associations.new_relationship_discovered].copy()
    discoveries.to_parquet(args.output / "DISCOVERED_NEW_RELATIONSHIPS.parquet", index=False)
    checkpoint_manifest_path = args.output / "CHECKPOINT_MANIFEST.json"
    atomic_json(checkpoint_manifest_path, build_checkpoint_manifest(reports))
    validation_payload = {
        "format": FORMAT,
        "target_host": "149",
        "status": "PASS_TECHNICAL_BASELINE",
        "outer_folds": 5,
        "oof_rows": len(predictions),
        "oof_duplicate_keys": int(predictions.duplicated(ROW_KEY).sum()),
        "association_rows": len(associations),
        "new_relationships": len(discoveries),
        "signed_target": True,
        "donor_blocked": True,
        "metacell_resolved": True,
        "core_detached_and_frozen": all(
            item["frozen_core_sha256_before"] == item["frozen_core_sha256_after"]
            for item in reports
        ),
        "oov_private_embedding_enabled": True,
        "fresh_random_initialization": True,
        "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None,
        "device": "cpu",
        "cuda_used": False,
        "run_config": file_record(run_config_path),
        "input_lineage_audit": file_record(input_lineage_path),
        "checkpoint_manifest": file_record(checkpoint_manifest_path),
        "folds": reports,
    }
    contract_validation_path = args.output / "FULL_MODEL_CONTRACT_VALIDATION.json"
    atomic_json(contract_validation_path, validation_payload)
    lineage_path = args.output / "LINEAGE.json"
    launch_preflight_path = args.output / "LAUNCH_PREFLIGHT.json"
    pre_lineage_artifacts = collect_artifacts(
        args.output, excluded_names={"LINEAGE.json", "SUCCESS.json"}
    )
    result_outputs = {
        name: file_record(args.output / name)
        for name in (
            "OOF_DONOR_PREDICTIONS.parquet",
            "CROSSFIT_ASSOCIATIONS.parquet",
            "DISCOVERED_NEW_RELATIONSHIPS.parquet",
            "FULL_MODEL_CONTRACT_VALIDATION.json",
        )
    }
    lineage = {
        "format": TRAINING_LINEAGE_FORMAT,
        "target_host": "149",
        "status": "SUCCESS",
        "scope": (
            "PASS_PARTIAL_PILOT_LINEAGE"
            if args.allow_partial_materialization
            else "PASS_FULL_28_CANCER_TRAINING_LINEAGE"
        ),
        "device": "cpu",
        "cuda_used": False,
        "cpu_only": True,
        "paid_compute": False,
        "models_trained": 5,
        "training_run_id": args.output.name,
        "seeds": [int(item["seed"]) for item in reports],
        "fold_seeds": [int(item["seed"]) for item in reports],
        "trainer_path": str(trainer_script),
        "trainer_sha256": sha256(trainer_script),
        "trainer_script": file_record(trainer_script),
        "trainer_script_sha256": sha256(trainer_script),
        "run_config": file_record(run_config_path),
        "launch_preflight": file_record(launch_preflight_path),
        "input_lineage_audit": file_record(input_lineage_path),
        "checkpoint_manifest": file_record(checkpoint_manifest_path),
        "contract_validation": file_record(contract_validation_path),
        "launch_preflight_sha256": sha256(launch_preflight_path),
        "run_config_sha256": sha256(run_config_path),
        "input_lineage_audit_sha256": sha256(input_lineage_path),
        "checkpoint_manifest_sha256": sha256(checkpoint_manifest_path),
        "outputs": result_outputs,
        "output_artifacts_before_lineage": pre_lineage_artifacts,
        "fresh_random_initialization": True,
        "core_parameters_frozen": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "old_family_support_used": False,
        "old_predictions_loaded": False,
        "old_rankings_loaded": False,
        "old_training_outputs_reused": False,
        "source_checkpoint_sha256": None,
        "finished_unix": time.time(),
    }
    atomic_json(lineage_path, lineage)
    top_level_files = (
        "RUN_CONFIG.json",
        "LAUNCH_PREFLIGHT.json",
        "INPUT_LINEAGE_AUDIT.json",
        "CHECKPOINT_MANIFEST.json",
        "OOF_DONOR_PREDICTIONS.parquet",
        "CROSSFIT_ASSOCIATIONS.parquet",
        "DISCOVERED_NEW_RELATIONSHIPS.parquet",
        "FULL_MODEL_CONTRACT_VALIDATION.json",
        "LINEAGE.json",
    )
    files = {name: file_record(args.output / name) for name in top_level_files}
    required_closure_files = {
        name: files[name]
        for name in (
            "RUN_CONFIG.json",
            "INPUT_LINEAGE_AUDIT.json",
            "CHECKPOINT_MANIFEST.json",
            "LINEAGE.json",
        )
    }
    success = {
        "format": FORMAT,
        "target_host": "149",
        "state": "FIVE_FOLD_DONOR_BLOCKED_METACELL_TRAINING_COMPLETE",
        "status": "SUCCESS",
        "scope": (
            "PASS_PARTIAL_PILOT_TRAINING"
            if args.allow_partial_materialization
            else "PASS_FULL_28_CANCER_TRAINING"
        ),
        "folds": 5,
        "models_trained": 5,
        "fold_seeds": [int(item["seed"]) for item in reports],
        "oof_rows": len(predictions),
        "association_rows": len(associations),
        "new_relationships": len(discoveries),
        "device": "cpu",
        "cuda_used": False,
        "fresh_random_initialization": True,
        "old_checkpoint_loaded": False,
        "source_checkpoint_sha256": None,
        "lineage_sha256": files["LINEAGE.json"]["sha256"],
        "required_closure_files": required_closure_files,
        "files": files,
        "finished_unix": time.time(),
    }
    atomic_json(args.output / "SUCCESS.json", success)
    print(json.dumps(success, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
