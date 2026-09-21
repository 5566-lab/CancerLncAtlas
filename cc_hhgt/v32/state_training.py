"""Fresh V3.2 training for the seven historical tumour-state programs.

This module consumes measurements, current V3.2 patient folds, current V3.2
fold core embeddings, and lncRNA expression.  It has no code path for loading
historical State checkpoints or predictions.  For each outer fold, patient-
local association features and labels are computed from the training split,
a private head is randomly initialised, and the held-out patient split is used
only for the released OOF effect/availability assessment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import canonical_json_sha256, dataframe_sha256
from .integrated_model import (
    build_private_auxiliary_head,
    module_state_sha256,
    validate_private_head_checkpoint,
)
from .patient_folds import assign_outer_split


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_STATE_FRESH"
STATE_TRAINING_FORMAT = "CC_HHGT_V3_2_STATE_FRESH_TRAINING_V1"
STATE_CHECKPOINT_FORMAT = "CC_HHGT_V3_2_PRIVATE_STATE_HEAD_V1"
CORE_EXPORT_FORMAT = "CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1"
CORE_CHECKPOINT_FORMAT = "CC_HHGT_V3_2_FULL_TRAINING_STATE_V1"
N_FOLDS = 5

HISTORICAL_STATE_IDS = (
    "stemness_rna::RNAss",
    "stemness_dna::DNAss",
    "stemness_dna::DMPss",
    "stemness_dna::ENHss",
    "stemness_dna::EREG-METHss",
    "stemness_rna::EREG.EXPss",
    "EXTEND::published_score",
)
MAINLINE_REQUIRED_STATE_IDS = frozenset(
    {
        "stemness_rna::RNAss",
        "stemness_dna::DNAss",
        "stemness_rna::EREG.EXPss",
        "EXTEND::published_score",
    }
)

STATE_REQUIRED_COLUMNS = frozenset(
    {"sample_id", "cancer_id", "patient_id", "state_id", "state_value"}
)
FOLD_REQUIRED_COLUMNS = frozenset(
    {"sample_id", "cancer_id", "patient_id", "patient_fold_id"}
)
EXPRESSION_REQUIRED_COLUMNS = frozenset({"sample_id", "cancer_id", "lncrna_id"})
FORBIDDEN_INPUT_NAME_TOKENS = (
    "strict_state_oof",
    "lncrna_state_final",
    "checkpoint",
    "prediction",
    "probability",
    "state_oof_prediction",
    "historical_prediction",
    "old_prediction",
    "old_checkpoint",
    "legacy_checkpoint",
    "final_probability",
    "ranking",
)
FORBIDDEN_INPUT_COLUMN_TOKENS = (
    "strict_state_oof",
    "lncrna_state_final",
    "checkpoint",
    "prediction",
    "probability",
    "ranking",
    "historical_prediction",
    "old_prediction",
    "legacy_probability",
    "old_probability",
    "old_ranking",
    "source_checkpoint",
)
TORCH_INPUT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt", ".bin"})
OUTPUT_PROBABILITY_COLUMNS = ("state_membership_probability", "state_effect")
_TCGA_PATIENT_RE = re.compile(r"^TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}")


class StateTrainingError(RuntimeError):
    """Raised when fresh V3.2 State training cannot be proven."""


@dataclass(frozen=True)
class StateTrainingConfig:
    seed: int = 20260726
    n_folds: int = N_FOLDS
    validation_offset: int = 1
    min_pairs: int = 10
    effect_threshold: float = 0.20
    max_epochs: int = 50
    patience: int = 8
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    batch_size: int = 4096
    hidden_features: int = 64
    dropout: float = 0.10
    min_available_folds: int = 3
    expression_value_column: str = "logcpm"
    device: str = "auto"

    def validate(self) -> None:
        if int(self.n_folds) != N_FOLDS:
            raise StateTrainingError("V3.2 State training requires exactly five patient folds")
        if not 0 < int(self.validation_offset) < N_FOLDS:
            raise StateTrainingError("validation_offset must be in 1..4")
        if int(self.min_pairs) < 2:
            raise StateTrainingError("min_pairs must be at least two")
        if not 0.0 < float(self.effect_threshold) < 1.0:
            raise StateTrainingError("effect_threshold must be in (0, 1)")
        if int(self.max_epochs) < 1 or int(self.patience) < 1:
            raise StateTrainingError("max_epochs and patience must be positive")
        if float(self.learning_rate) <= 0 or float(self.weight_decay) < 0:
            raise StateTrainingError("learning rate/weight decay are invalid")
        if int(self.batch_size) < 1 or int(self.hidden_features) < 1:
            raise StateTrainingError("batch_size and hidden_features must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise StateTrainingError("dropout must be in [0, 1)")
        if not 1 <= int(self.min_available_folds) <= N_FOLDS:
            raise StateTrainingError("min_available_folds must be in 1..5")
        if not str(self.expression_value_column).strip():
            raise StateTrainingError("expression_value_column is empty")
        if str(self.device) not in {"auto", "cpu", "cuda"}:
            raise StateTrainingError("device must be auto, cpu, or cuda")


@dataclass
class StateTrainingResult:
    oof_predictions: pd.DataFrame
    release_predictions: pd.DataFrame
    fold_heads: dict[int, Any]
    fold_lineage: dict[int, dict[str, Any]]
    observed_state_ids: tuple[str, ...]
    missing_state_ids: tuple[str, ...]


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    if not source.is_file():
        raise StateTrainingError(f"Required file is missing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def path_sha256(path: str | Path) -> str:
    """Hash either one file or a partitioned input dataset deterministically."""

    source = Path(path)
    if source.is_file():
        return file_sha256(source)
    if not source.is_dir():
        raise StateTrainingError(f"Required artifact is missing: {source}")
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise StateTrainingError(f"Input dataset directory is empty: {source}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def reject_forbidden_input_path(path: str | Path, role: str) -> Path:
    """Reject historical result/checkpoint inputs before opening them."""

    source = Path(path)
    normalized = source.as_posix().lower()
    if source.suffix.lower() in TORCH_INPUT_SUFFIXES:
        raise StateTrainingError(f"{role} may not be a checkpoint/model file: {source}")
    matched = [token for token in FORBIDDEN_INPUT_NAME_TOKENS if token in normalized]
    if matched:
        raise StateTrainingError(
            f"{role} path names a forbidden historical result ({matched[0]}): {source}"
        )
    return source


def _reject_forbidden_columns(frame: pd.DataFrame, role: str) -> None:
    bad = sorted(
        column
        for column in map(str, frame.columns)
        if any(token in column.lower() for token in FORBIDDEN_INPUT_COLUMN_TOKENS)
    )
    if bad:
        raise StateTrainingError(f"{role} contains forbidden old-result columns: {bad}")


def read_table(path: str | Path, role: str) -> pd.DataFrame:
    source = reject_forbidden_input_path(path, role)
    if source.is_dir():
        parts = sorted(source.rglob("*.parquet"))
        if not parts:
            raise StateTrainingError(f"{role} dataset has no Parquet partitions: {source}")
        for part in parts:
            reject_forbidden_input_path(part, role)
        return pd.concat((pd.read_parquet(part) for part in parts), ignore_index=True)
    if not source.is_file():
        raise StateTrainingError(f"{role} file/dataset is missing: {source}")
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(source, sep="\t")
    raise StateTrainingError(f"Unsupported {role} table format: {source}")


def _require_columns(frame: pd.DataFrame, required: set[str] | frozenset[str], role: str) -> None:
    missing = sorted(set(required) - set(map(str, frame.columns)))
    if missing:
        raise StateTrainingError(f"{role} lacks required columns: {missing}")


def _require_nonempty_ids(frame: pd.DataFrame, columns: Sequence[str], role: str) -> None:
    for column in columns:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise StateTrainingError(f"{role}.{column} contains missing/empty identifiers")


def validate_state_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    _reject_forbidden_columns(frame, "tumor_state_long measurement")
    _require_columns(frame, STATE_REQUIRED_COLUMNS, "tumor_state_long measurement")
    result = frame.loc[:, list(STATE_REQUIRED_COLUMNS)].copy()
    # The historical standardized table also contains non-State metadata rows
    # such as ``sample_type_code``; a small number predate a cancer mapping.
    # They are outside this module's declared seven-program estimand.  Filter
    # the declared State IDs first, then retain the strict no-missing-ID rule
    # for every row that can affect V3.2 training.
    _require_nonempty_ids(result, ["state_id"], "state")
    result["state_id"] = result["state_id"].astype(str)
    result = result.loc[result.state_id.isin(HISTORICAL_STATE_IDS)].copy()
    if result.empty:
        raise StateTrainingError("No historical State measurements were found")
    _require_nonempty_ids(result, ["sample_id", "cancer_id", "patient_id"], "state")
    for column in ("sample_id", "cancer_id", "patient_id", "state_id"):
        result[column] = result[column].astype(str)
    result["state_value"] = pd.to_numeric(result["state_value"], errors="coerce")
    if not np.isfinite(result["state_value"].dropna().to_numpy(dtype=float)).all():
        raise StateTrainingError("state_value contains infinity")
    keys = ["cancer_id", "sample_id", "state_id"]
    if result[keys].duplicated().any():
        raise StateTrainingError("tumor_state_long has duplicate cancer/sample/state measurements")
    cancer_per_sample = result.groupby("sample_id", observed=True).cancer_id.nunique()
    if (cancer_per_sample > 1).any():
        raise StateTrainingError("A State sample appears in multiple cancers")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def validate_fold_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    _reject_forbidden_columns(frame, "V3.2 patient fold manifest")
    _require_columns(frame, FOLD_REQUIRED_COLUMNS, "V3.2 patient fold manifest")
    columns = ["cancer_id", "sample_id", "patient_fold_id"]
    columns.append("patient_id")
    if "fold_seed" in frame.columns:
        columns.append("fold_seed")
    result = frame.loc[:, columns].copy()
    _require_nonempty_ids(result, ["sample_id", "cancer_id"], "fold manifest")
    result["sample_id"] = result["sample_id"].astype(str)
    result["cancer_id"] = result["cancer_id"].astype(str)
    _require_nonempty_ids(result, ["patient_id"], "fold manifest")
    result["patient_id"] = result.patient_id.astype(str).str.strip()
    sample_mapping = result.groupby(["cancer_id", "sample_id"], observed=True).patient_id.nunique()
    if sample_mapping.gt(1).any():
        raise StateTrainingError("A fold sample maps to multiple explicit patients")
    result["patient_fold_id"] = pd.to_numeric(
        result["patient_fold_id"], errors="coerce"
    ).astype("Int64")
    if result.patient_fold_id.isna().any():
        raise StateTrainingError("patient_fold_id is not an integer")
    observed = set(result.patient_fold_id.astype(int))
    if observed != set(range(N_FOLDS)):
        raise StateTrainingError(f"Patient fold manifest must contain folds 0..4, observed {observed}")
    if result.sample_id.duplicated().any():
        raise StateTrainingError("Patient fold manifest duplicates sample_id")
    patient_folds = result.groupby(
        ["cancer_id", "patient_id"], observed=True
    ).patient_fold_id.nunique()
    if (patient_folds > 1).any():
        raise StateTrainingError("Samples from one patient cross V3.2 patient folds")
    if "fold_seed" in result:
        seeds = pd.to_numeric(result.fold_seed, errors="coerce").dropna().astype(int).unique()
        if len(seeds) > 1:
            raise StateTrainingError("Patient fold manifest contains multiple fold seeds")
    return result.sort_values(
        ["cancer_id", "patient_fold_id", "sample_id"], kind="stable"
    ).reset_index(drop=True)


def canonical_patient_id(sample_id: Any) -> str:
    """Return a patient key without collapsing non-TCGA sample identifiers."""

    value = str(sample_id).strip()
    match = _TCGA_PATIENT_RE.match(value)
    return match.group(0).upper() if match else value


def validate_lncrna_expression(
    frame: pd.DataFrame, value_column: str = "logcpm"
) -> pd.DataFrame:
    _reject_forbidden_columns(frame, "lncRNA expression")
    required = set(EXPRESSION_REQUIRED_COLUMNS) | {str(value_column)}
    _require_columns(frame, required, "lncRNA expression")
    result = frame.loc[:, ["cancer_id", "sample_id", "lncrna_id", value_column]].copy()
    _require_nonempty_ids(result, ["cancer_id", "sample_id", "lncrna_id"], "expression")
    for column in ("cancer_id", "sample_id", "lncrna_id"):
        result[column] = result[column].astype(str)
    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")
    if not np.isfinite(result[value_column].dropna().to_numpy(dtype=float)).all():
        raise StateTrainingError(f"{value_column} contains infinity")
    keys = ["cancer_id", "sample_id", "lncrna_id"]
    if result[keys].duplicated().any():
        raise StateTrainingError("lncRNA expression has duplicate cancer/sample/lncRNA rows")
    if result[value_column].notna().sum() == 0:
        raise StateTrainingError("lncRNA expression has no measured values")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def validate_core_embedding_frame(frame: pd.DataFrame, fold: int) -> pd.DataFrame:
    _reject_forbidden_columns(frame, f"V3.2 fold-{fold} core embeddings")
    _require_columns(frame, {"node_id"}, f"V3.2 fold-{fold} core embeddings")
    feature_columns = sorted(
        column for column in map(str, frame.columns) if column.startswith("core_feature_")
    )
    if not feature_columns:
        raise StateTrainingError(f"Fold {fold} core embedding has no core_feature columns")
    result = frame.loc[:, ["node_id", *feature_columns]].copy()
    _require_nonempty_ids(result, ["node_id"], f"fold-{fold} core embeddings")
    result["node_id"] = result.node_id.astype(str)
    if result.node_id.duplicated().any():
        raise StateTrainingError(f"Fold {fold} core embedding duplicates lncRNA node_id")
    for column in feature_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(result[feature_columns].to_numpy(dtype=float)).all():
        raise StateTrainingError(f"Fold {fold} core embeddings contain non-finite values")
    return result.sort_values("node_id", kind="stable").reset_index(drop=True)


def _resolve_inside(root: Path, value: str, role: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StateTrainingError(f"{role} escapes repo root: {value}") from exc
    return resolved


def load_v32_core_embeddings(
    manifest_path: str | Path, *, repo_root: str | Path
) -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, Any]], dict[str, Any]]:
    """Load only exported V3.2 lncRNA embeddings, never a Torch checkpoint."""

    source = reject_forbidden_input_path(manifest_path, "V3.2 core embedding manifest")
    if not source.is_file():
        raise StateTrainingError(f"Core embedding manifest is missing: {source}")
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StateTrainingError("Core embedding manifest is invalid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise StateTrainingError("Core embedding manifest must be an object")
    if manifest.get("export_format") != CORE_EXPORT_FORMAT:
        raise StateTrainingError("Core embeddings are not a registered V3.2 export")
    if str(manifest.get("training_generation", "")).upper() != "V3.2":
        raise StateTrainingError("Core embeddings do not declare V3.2 training")
    if manifest.get("all_embeddings_from_newly_trained_v32_core") is not True:
        raise StateTrainingError("Core embeddings lack fresh-V3.2 attestation")
    if manifest.get("historical_checkpoint_loaded") is not False:
        raise StateTrainingError("Core embedding export loaded a historical checkpoint")
    if manifest.get("historical_prediction_loaded") is not False:
        raise StateTrainingError("Core embedding export loaded historical predictions")
    folds = manifest.get("folds")
    if not isinstance(folds, Mapping) or set(map(str, folds)) != set(map(str, range(N_FOLDS))):
        raise StateTrainingError("Core embedding manifest must contain folds 0..4")

    root = Path(repo_root).resolve()
    embeddings: dict[int, pd.DataFrame] = {}
    fold_lineage: dict[int, dict[str, Any]] = {}
    width: int | None = None
    for fold in range(N_FOLDS):
        row = folds[str(fold)]
        if not isinstance(row, Mapping):
            raise StateTrainingError(f"Core fold {fold} lineage is not an object")
        if int(row.get("patient_fold", -1)) != fold:
            raise StateTrainingError(f"Core embedding fold identity drift for fold {fold}")
        if row.get("checkpoint_format") != CORE_CHECKPOINT_FORMAT:
            raise StateTrainingError(f"Fold {fold} core parent is not a V3.2 checkpoint")
        if row.get("old_checkpoint_loaded") is not False:
            raise StateTrainingError(f"Fold {fold} declares an old checkpoint")
        if row.get("trained_from_scratch") is not True:
            raise StateTrainingError(f"Fold {fold} core was not trained from scratch")
        core_hash = str(row.get("core_parameter_sha256", ""))
        if len(core_hash) != 64:
            raise StateTrainingError(f"Fold {fold} lacks core parameter SHA-256")
        checkpoint_hash = str(row.get("checkpoint_sha256", ""))
        if len(checkpoint_hash) != 64:
            raise StateTrainingError(f"Fold {fold} lacks its V3.2 core checkpoint SHA-256")
        exports = row.get("exports")
        if not isinstance(exports, Mapping):
            raise StateTrainingError(f"Fold {fold} lacks exported embeddings")
        lnc_key = next((key for key in exports if str(key).lower() == "lncrna"), None)
        if lnc_key is None or not isinstance(exports[lnc_key], Mapping):
            raise StateTrainingError(f"Fold {fold} lacks lncRNA core embeddings")
        export = exports[lnc_key]
        export_path = _resolve_inside(root, str(export.get("path", "")), "core embedding")
        reject_forbidden_input_path(export_path, f"fold-{fold} core embedding")
        expected_hash = str(export.get("sha256", ""))
        if len(expected_hash) != 64 or file_sha256(export_path) != expected_hash:
            raise StateTrainingError(f"Fold {fold} core embedding SHA-256 mismatch")
        frame = validate_core_embedding_frame(read_table(export_path, "core embedding"), fold)
        feature_count = sum(column.startswith("core_feature_") for column in frame.columns)
        if width is None:
            width = feature_count
        elif width != feature_count:
            raise StateTrainingError("Core embedding width changes across folds")
        if int(export.get("rows", -1)) != len(frame) or int(export.get("features", -1)) != feature_count:
            raise StateTrainingError(f"Fold {fold} core embedding shape/manifest mismatch")
        embeddings[fold] = frame
        fold_lineage[fold] = {
            "core_embedding_path": export_path.as_posix(),
            "core_embedding_sha256": expected_hash,
            "core_parameter_sha256": core_hash,
            "v32_core_checkpoint_sha256": checkpoint_hash,
            "core_checkpoint_format": row.get("checkpoint_format"),
        }
    return embeddings, fold_lineage, dict(manifest)


def _paired_correlations(
    matrix: np.ndarray, state_values: np.ndarray, min_pairs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(state_values, dtype=float).reshape(-1, 1)
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum(axis=0).astype(np.int64)
    x0 = np.where(mask, x, 0.0)
    y0 = np.where(mask, y, 0.0)
    safe_n = np.maximum(n, 1)
    sum_x = x0.sum(axis=0)
    sum_y = y0.sum(axis=0)
    cross = (x0 * y0).sum(axis=0) - sum_x * sum_y / safe_n
    var_x = (x0 * x0).sum(axis=0) - sum_x * sum_x / safe_n
    var_y = (y0 * y0).sum(axis=0) - sum_y * sum_y / safe_n
    denominator = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0))
    effect = np.full(x.shape[1], np.nan, dtype=float)
    available = (n >= int(min_pairs)) & (var_x > 1e-12) & (var_y > 1e-12)
    effect[available] = np.clip(cross[available] / denominator[available], -1.0, 1.0)
    reason = np.full(x.shape[1], None, dtype=object)
    reason[n < int(min_pairs)] = "insufficient_pairs"
    reason[(n >= int(min_pairs)) & (var_x <= 1e-12)] = "lncrna_expression_zero_variance"
    reason[(n >= int(min_pairs)) & (var_x > 1e-12) & (var_y <= 1e-12)] = (
        "state_measurement_zero_variance"
    )
    return n, effect, reason


def compute_fold_local_associations(
    state_measurements: pd.DataFrame,
    lncrna_expression: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    split: str,
    min_pairs: int,
    effect_threshold: float,
    expression_value_column: str = "logcpm",
) -> pd.DataFrame:
    """Compute association features/labels using one registered split only."""

    if split not in {"train", "validation", "test"}:
        raise StateTrainingError(f"Unregistered split: {split}")
    _require_columns(split_manifest, {"cancer_id", "sample_id", "split"}, "outer split")
    assignments = split_manifest.loc[
        split_manifest.split.astype(str).eq(split), ["cancer_id", "sample_id"]
    ].astype(str)
    expression = lncrna_expression.merge(
        assignments.assign(_selected=True),
        on=["cancer_id", "sample_id"],
        how="inner",
        validate="many_to_one",
    )
    states = state_measurements.merge(
        assignments.assign(_selected=True),
        on=["cancer_id", "sample_id"],
        how="inner",
        validate="many_to_one",
    )
    candidate_universe = (
        lncrna_expression[["cancer_id", "lncrna_id"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["cancer_id", "lncrna_id"], kind="stable")
    )
    frames: list[pd.DataFrame] = []
    for cancer_id, candidate_group in candidate_universe.groupby("cancer_id", observed=True):
        lncrnas = candidate_group.lncrna_id.astype(str).tolist()
        cancer_expression = expression.loc[expression.cancer_id.eq(cancer_id)]
        if cancer_expression.empty:
            expression_wide = pd.DataFrame(columns=lncrnas, dtype=float)
        else:
            expression_wide = cancer_expression.pivot(
                index="sample_id", columns="lncrna_id", values=expression_value_column
            ).reindex(columns=lncrnas)
        cancer_states = states.loc[states.cancer_id.eq(cancer_id)]
        for state_id in HISTORICAL_STATE_IDS:
            state_rows = cancer_states.loc[
                cancer_states.state_id.eq(state_id), ["sample_id", "state_value"]
            ].dropna(subset=["state_value"])
            if state_rows.empty:
                n = np.zeros(len(lncrnas), dtype=np.int64)
                effect = np.full(len(lncrnas), np.nan, dtype=float)
                reason = np.full(len(lncrnas), "state_measurement_missing", dtype=object)
            else:
                state_series = state_rows.set_index("sample_id").state_value.astype(float)
                aligned = expression_wide.reindex(state_series.index)
                n, effect, reason = _paired_correlations(
                    aligned.to_numpy(dtype=float), state_series.to_numpy(dtype=float), min_pairs
                )
            available = pd.isna(reason)
            label = np.full(len(lncrnas), np.nan, dtype=float)
            label[available] = (
                np.abs(effect[available]) >= float(effect_threshold)
            ).astype(float)
            frames.append(
                pd.DataFrame(
                    {
                        "cancer_id": str(cancer_id),
                        "lncrna_id": lncrnas,
                        "state_id": state_id,
                        "split": split,
                        "n_pairs": n,
                        "effect": effect,
                        "association_label": label,
                        "available": available,
                        "unavailability_reason": reason,
                    }
                )
            )
    if not frames:
        raise StateTrainingError("No cancer/lncRNA candidates were available")
    result = pd.concat(frames, ignore_index=True)
    keys = ["cancer_id", "lncrna_id", "state_id"]
    if result[keys].duplicated().any():
        raise StateTrainingError("Fold-local association table has duplicate candidates")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _state_domain_matrix(
    frame: pd.DataFrame, *, max_train_pairs: int
) -> np.ndarray:
    effect = pd.to_numeric(frame.effect, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n_pairs = pd.to_numeric(frame.n_pairs, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    denominator = max(math.log1p(max(int(max_train_pairs), 1)), 1.0)
    columns = [
        effect,
        np.abs(effect),
        effect * effect,
        np.log1p(np.maximum(n_pairs, 0.0)) / denominator,
    ]
    for state_id in HISTORICAL_STATE_IDS:
        columns.append(frame.state_id.astype(str).eq(state_id).to_numpy(dtype=float))
    return np.column_stack(columns).astype(np.float32, copy=False)


def _attach_core(
    statistics: pd.DataFrame, core_embeddings: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = sorted(
        column for column in core_embeddings.columns if str(column).startswith("core_feature_")
    )
    core = core_embeddings.rename(columns={"node_id": "lncrna_id"})
    merged = statistics.merge(
        core[["lncrna_id", *feature_columns]],
        on="lncrna_id",
        how="left",
        validate="many_to_one",
    )
    merged["core_embedding_available"] = merged[feature_columns].notna().all(axis=1)
    return merged, feature_columns


def _model_arrays(
    statistics: pd.DataFrame,
    core_embeddings: pd.DataFrame,
    *,
    max_train_pairs: int,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray]:
    merged, core_columns = _attach_core(statistics, core_embeddings)
    core = merged[core_columns].fillna(0.0).to_numpy(dtype=np.float32)
    domain = _state_domain_matrix(merged, max_train_pairs=max_train_pairs)
    labels = pd.to_numeric(merged.association_label, errors="coerce").to_numpy(dtype=np.float32)
    return merged, core_columns, core, domain, labels


def _resolve_device(requested: str):
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise StateTrainingError("CUDA was requested but is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _evaluate_bce(
    head: Any,
    core: np.ndarray,
    domain: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: Any,
    positive_weight: float,
) -> float:
    import torch

    if not len(indices):
        return float("nan")
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(positive_weight), dtype=torch.float32, device=device),
        reduction="sum",
    )
    total = 0.0
    head.eval()
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            batch = indices[start : start + int(batch_size)]
            logits = head(
                torch.from_numpy(core[batch]).to(device),
                torch.from_numpy(domain[batch]).to(device),
            ).squeeze(-1)
            target = torch.from_numpy(labels[batch]).to(device)
            total += float(criterion(logits, target).detach().cpu())
    return total / len(indices)


def train_fold_private_head(
    train_statistics: pd.DataFrame,
    validation_statistics: pd.DataFrame,
    core_embeddings: pd.DataFrame,
    *,
    fold: int,
    config: StateTrainingConfig,
) -> tuple[Any, dict[str, Any], int]:
    """Train one freshly initialised head; no checkpoint argument exists."""

    import torch

    max_train_pairs = int(pd.to_numeric(train_statistics.n_pairs, errors="coerce").max())
    train_frame, core_columns, train_core, train_domain, train_labels = _model_arrays(
        train_statistics, core_embeddings, max_train_pairs=max_train_pairs
    )
    validation_frame, validation_core_columns, val_core, val_domain, val_labels = _model_arrays(
        validation_statistics, core_embeddings, max_train_pairs=max_train_pairs
    )
    if core_columns != validation_core_columns:
        raise StateTrainingError("Core embedding feature order changed within a fold")
    train_mask = (
        train_frame.available.astype(bool).to_numpy()
        & train_frame.core_embedding_available.astype(bool).to_numpy()
        & np.isfinite(train_labels)
    )
    validation_mask = (
        validation_frame.available.astype(bool).to_numpy()
        & validation_frame.core_embedding_available.astype(bool).to_numpy()
        & np.isfinite(val_labels)
    )
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if len(train_indices) < 2:
        raise StateTrainingError(f"Fold {fold} has fewer than two trainable State candidates")
    validation_fallback = False
    if not len(validation_indices):
        validation_indices = train_indices
        val_core, val_domain, val_labels = train_core, train_domain, train_labels
        validation_fallback = True

    positives = float(train_labels[train_indices].sum())
    negatives = float(len(train_indices) - positives)
    positive_weight = 1.0 if positives <= 0 or negatives <= 0 else min(negatives / positives, 20.0)
    fold_seed = int(config.seed) + int(fold)
    head, initialization = build_private_auxiliary_head(
        f"state_fold_{fold}",
        core_features=len(core_columns),
        domain_features=train_domain.shape[1],
        hidden_features=int(config.hidden_features),
        dropout=float(config.dropout),
        seed=fold_seed,
    )
    device = _resolve_device(config.device)
    head.to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(int(config.max_epochs)):
        head.train()
        order = np.random.default_rng(fold_seed + epoch * 1009).permutation(train_indices)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, len(order), int(config.batch_size)):
            batch = order[start : start + int(config.batch_size)]
            core_tensor = torch.from_numpy(train_core[batch]).to(device)
            domain_tensor = torch.from_numpy(train_domain[batch]).to(device)
            target = torch.from_numpy(train_labels[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(core_tensor, domain_tensor).squeeze(-1)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(batch)
            seen += len(batch)
        validation_loss = _evaluate_bce(
            head,
            val_core,
            val_domain,
            val_labels,
            validation_indices,
            batch_size=int(config.batch_size),
            device=device,
            positive_weight=positive_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": epoch_loss / max(seen, 1),
                "validation_loss": validation_loss,
            }
        )
        if np.isfinite(validation_loss) and validation_loss < best_loss - 1e-8:
            best_loss = float(validation_loss)
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(config.patience):
            break
    if best_state is None:
        raise StateTrainingError(f"Fold {fold} private head never produced finite validation loss")
    head.load_state_dict(best_state, strict=True)
    head.to("cpu")
    validate_private_head_checkpoint(initialization.as_dict(), head)
    final_hash = module_state_sha256(head)
    if final_hash == initialization.initial_parameter_sha256:
        raise StateTrainingError(f"Fold {fold} private head parameters did not update")
    lineage = {
        "patient_fold": int(fold),
        "seed": fold_seed,
        "initialization": initialization.as_dict(),
        "final_parameter_sha256": final_hash,
        "private_head_trained_from_scratch": True,
        "source_checkpoint_sha256": None,
        "core_embeddings_detached": True,
        "core_parameters_frozen": True,
        "trainable_candidates": int(len(train_indices)),
        "validation_candidates": int(len(validation_indices)),
        "validation_fallback_to_training": validation_fallback,
        "positive_candidates": int(positives),
        "negative_candidates": int(negatives),
        "positive_weight": float(positive_weight),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "epochs_completed": len(history),
        "history": history,
        "core_features": len(core_columns),
        "domain_features": int(train_domain.shape[1]),
        "max_train_pairs": max_train_pairs,
    }
    return head, lineage, max_train_pairs


def _predict_probabilities(
    head: Any,
    statistics: pd.DataFrame,
    core_embeddings: pd.DataFrame,
    *,
    max_train_pairs: int,
    config: StateTrainingConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    import torch

    merged, _, core, domain, labels = _model_arrays(
        statistics, core_embeddings, max_train_pairs=max_train_pairs
    )
    eligible = (
        merged.available.astype(bool).to_numpy()
        & merged.core_embedding_available.astype(bool).to_numpy()
        & np.isfinite(labels)
    )
    probabilities = np.full(len(merged), np.nan, dtype=float)
    indices = np.flatnonzero(eligible)
    head.eval()
    with torch.no_grad():
        for start in range(0, len(indices), int(config.batch_size)):
            batch = indices[start : start + int(config.batch_size)]
            logits = head(
                torch.from_numpy(core[batch]), torch.from_numpy(domain[batch])
            ).squeeze(-1)
            probabilities[batch] = torch.sigmoid(logits).cpu().numpy().astype(float)
    return merged, probabilities


def build_fold_oof_predictions(
    head: Any,
    train_statistics: pd.DataFrame,
    test_statistics: pd.DataFrame,
    core_embeddings: pd.DataFrame,
    *,
    fold: int,
    max_train_pairs: int,
    config: StateTrainingConfig,
) -> pd.DataFrame:
    train_frame, probabilities = _predict_probabilities(
        head,
        train_statistics,
        core_embeddings,
        max_train_pairs=max_train_pairs,
        config=config,
    )
    keys = ["cancer_id", "lncrna_id", "state_id"]
    train_out = train_frame[
        [*keys, "n_pairs", "effect", "available", "unavailability_reason", "core_embedding_available"]
    ].rename(
        columns={
            "n_pairs": "train_n_pairs",
            "effect": "train_effect",
            "available": "train_available",
            "unavailability_reason": "train_unavailability_reason",
        }
    )
    train_out["_probability"] = probabilities
    test_out = test_statistics[
        [*keys, "n_pairs", "effect", "available", "unavailability_reason"]
    ].rename(
        columns={
            "n_pairs": "test_n_pairs",
            "effect": "test_effect",
            "available": "test_available",
            "unavailability_reason": "test_unavailability_reason",
        }
    )
    result = train_out.merge(test_out, on=keys, how="outer", validate="one_to_one")
    if len(result) != len(train_out) or len(result) != len(test_out):
        raise StateTrainingError(f"Fold {fold} train/test candidate universes differ")
    available = (
        result.train_available.fillna(False).astype(bool)
        & result.test_available.fillna(False).astype(bool)
        & result.core_embedding_available.fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(result._probability, errors="coerce"))
    )
    reason = np.full(len(result), None, dtype=object)
    reason[~result.core_embedding_available.fillna(False).astype(bool).to_numpy()] = (
        "core_embedding_missing"
    )
    train_bad = ~result.train_available.fillna(False).astype(bool).to_numpy()
    reason[train_bad] = [
        f"train_{value or 'association_unavailable'}"
        for value in result.loc[train_bad, "train_unavailability_reason"]
    ]
    test_bad = ~result.test_available.fillna(False).astype(bool).to_numpy()
    reason[test_bad] = [
        f"test_{value or 'association_unavailable'}"
        for value in result.loc[test_bad, "test_unavailability_reason"]
    ]
    result["patient_fold_id"] = int(fold)
    result["state_membership_probability"] = result._probability.where(available, pd.NA).astype(
        "Float64"
    )
    result["state_effect"] = pd.to_numeric(result.test_effect, errors="coerce").where(
        available, pd.NA
    ).astype("Float64")
    result["availability"] = available.astype(bool)
    result["availability_reason"] = pd.Series(reason).where(~available, None)
    effect_values = result.state_effect.astype("Float64")
    result["association_direction"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[available & effect_values.gt(0).fillna(False), "association_direction"] = "positive"
    result.loc[available & effect_values.lt(0).fillna(False), "association_direction"] = "negative"
    result.loc[available & effect_values.eq(0).fillna(False), "association_direction"] = "neutral"
    result["model_version"] = "V3.2"
    result["probability_source"] = "fresh_fold_private_state_head"
    result["effect_source"] = "held_out_patient_fold_measurement"
    output_columns = [
        "patient_fold_id",
        *keys,
        "state_membership_probability",
        "state_effect",
        "association_direction",
        "availability",
        "availability_reason",
        "test_n_pairs",
        "train_n_pairs",
        "model_version",
        "probability_source",
        "effect_source",
    ]
    output = result.loc[:, output_columns].sort_values(
        ["patient_fold_id", *keys], kind="stable"
    ).reset_index(drop=True)
    validate_nullable_predictions(output)
    return output


def validate_nullable_predictions(frame: pd.DataFrame) -> None:
    required = {
        "state_membership_probability",
        "state_effect",
        "availability",
        "availability_reason",
    }
    _require_columns(frame, required, "State prediction output")
    available = frame.availability.fillna(False).astype(bool)
    for column in OUTPUT_PROBABILITY_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.loc[~available].notna().any():
            raise StateTrainingError(f"Unavailable {column} must be null, never 0/0.5")
        if numeric.loc[available].isna().any() or not np.isfinite(
            numeric.loc[available].to_numpy(dtype=float)
        ).all():
            raise StateTrainingError(f"Available {column} must be finite")
    probability = pd.to_numeric(frame.state_membership_probability, errors="coerce")
    if ((probability.loc[available] < 0) | (probability.loc[available] > 1)).any():
        raise StateTrainingError("Available State probability is outside [0, 1]")
    if frame.loc[~available, "availability_reason"].fillna("").astype(str).str.strip().eq("").any():
        raise StateTrainingError("Every unavailable State row needs an explicit reason")
    if frame.loc[available, "availability_reason"].notna().any():
        raise StateTrainingError("Available State rows may not carry an unavailability reason")


def aggregate_oof_predictions(
    oof: pd.DataFrame, *, min_available_folds: int
) -> pd.DataFrame:
    validate_nullable_predictions(oof)
    rows: list[dict[str, Any]] = []
    keys = ["cancer_id", "lncrna_id", "state_id"]
    for key, group in oof.groupby(keys, observed=True, sort=True):
        available = group.loc[group.availability.astype(bool)].copy()
        enough = len(available) >= int(min_available_folds)
        if enough:
            weights = np.maximum(
                pd.to_numeric(available.test_n_pairs, errors="coerce").to_numpy(dtype=float), 1.0
            )
            probability = float(
                np.average(
                    pd.to_numeric(
                        available.state_membership_probability, errors="coerce"
                    ).to_numpy(dtype=float),
                    weights=weights,
                )
            )
            effect = float(
                np.average(
                    pd.to_numeric(available.state_effect, errors="coerce").to_numpy(dtype=float),
                    weights=weights,
                )
            )
            reason = None
            direction = "positive" if effect > 0 else "negative" if effect < 0 else "neutral"
        else:
            probability = pd.NA
            effect = pd.NA
            reasons = sorted(
                set(
                    group.loc[~group.availability.astype(bool), "availability_reason"]
                    .dropna()
                    .astype(str)
                )
            )
            reason = (
                f"fewer_than_{int(min_available_folds)}_available_folds:"
                + ("|".join(reasons) if reasons else "unknown")
            )
            direction = pd.NA
        rows.append(
            {
                "cancer_id": key[0],
                "lncrna_id": key[1],
                "state_id": key[2],
                "state_membership_probability": probability,
                "state_effect": effect,
                "association_direction": direction,
                "availability": bool(enough),
                "availability_reason": reason,
                "folds_available": int(len(available)),
                "folds_expected": N_FOLDS,
                "model_version": "V3.2",
                "probability_source": "five_fold_oof_fresh_private_state_heads",
                "effect_source": "held_out_patient_fold_measurement",
            }
        )
    result = pd.DataFrame(rows)
    for column in OUTPUT_PROBABILITY_COLUMNS:
        result[column] = pd.array(result[column], dtype="Float64")
    result["association_direction"] = result.association_direction.astype("string")
    validate_nullable_predictions(result)
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _align_source_tables(
    state: pd.DataFrame,
    expression: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    expression_value_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Align sample-resolution inputs to leakage-safe patient fold units.

    The historical State table uses TCGA sample barcodes such as
    ``TCGA-02-0047-01`` while the V3.2 expression manifest uses aliquot
    barcodes.  The State table already supplies ``patient_id``; the fold table
    derives the same TCGA patient key.  Multiple aliquots, if present, are
    averaged *inside* a patient before any association statistic is computed.
    """

    assignments = folds[["cancer_id", "sample_id", "patient_id", "patient_fold_id"]].copy()
    patient_fold_counts = assignments.groupby(
        ["cancer_id", "patient_id"], observed=True
    ).patient_fold_id.nunique()
    if (patient_fold_counts > 1).any():
        raise StateTrainingError("Patient-level alignment would cross V3.2 folds")
    patient_folds = (
        assignments.groupby(["cancer_id", "patient_id"], observed=True, as_index=False)
        .patient_fold_id.first()
        .rename(columns={"patient_id": "sample_id"})
    )
    patient_folds["patient_id"] = patient_folds.sample_id
    if "fold_seed" in folds:
        seeds = folds.groupby(["cancer_id", "patient_id"], observed=True).fold_seed.first()
        patient_folds = patient_folds.merge(
            seeds.rename("fold_seed").reset_index().rename(columns={"patient_id": "sample_id"}),
            on=["cancer_id", "sample_id"],
            how="left",
            validate="one_to_one",
        )

    state_source = state.copy()
    state_source["patient_id"] = state_source.patient_id.astype(str).map(canonical_patient_id)
    state_joined = state_source.merge(
        patient_folds[["cancer_id", "sample_id"]].rename(columns={"sample_id": "patient_id"}),
        on=["cancer_id", "patient_id"],
        how="inner",
        validate="many_to_one",
    )
    state_patient = (
        state_joined.groupby(
            ["cancer_id", "patient_id", "state_id"], observed=True, as_index=False
        )
        .state_value.mean()
        .rename(columns={"patient_id": "sample_id"})
    )
    state_patient["patient_id"] = state_patient.sample_id

    expression_joined = expression.merge(
        assignments.assign(_in_fold=True),
        on=["cancer_id", "sample_id"],
        how="inner",
        validate="many_to_one",
    )
    expression_patient = (
        expression_joined.groupby(
            ["cancer_id", "patient_id", "lncrna_id"], observed=True, as_index=False
        )[expression_value_column]
        .mean()
        .rename(columns={"patient_id": "sample_id"})
    )
    if state_patient.empty or expression_patient.empty:
        raise StateTrainingError("State/expression inputs do not overlap the V3.2 fold manifest")
    counts = {
        "state_rows_outside_fold_manifest": int(len(state) - len(state_joined)),
        "expression_rows_outside_fold_manifest": int(len(expression) - len(expression_joined)),
        "state_rows_collapsed_to_patient": int(len(state_joined) - len(state_patient)),
        "expression_rows_collapsed_to_patient": int(len(expression_joined) - len(expression_patient)),
        "analysis_patients": int(patient_folds.sample_id.nunique()),
    }
    return state_patient, expression_patient, patient_folds, counts


def train_state_models(
    state_measurements: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    core_embeddings_by_fold: Mapping[int, pd.DataFrame],
    lncrna_expression: pd.DataFrame,
    *,
    config: StateTrainingConfig | None = None,
    core_lineage_by_fold: Mapping[int, Mapping[str, Any]] | None = None,
) -> StateTrainingResult:
    """Train all five private heads from frames, suitable for tests or orchestration."""

    cfg = config or StateTrainingConfig()
    cfg.validate()
    state = validate_state_measurements(state_measurements)
    folds = validate_fold_manifest(fold_manifest)
    expression = validate_lncrna_expression(
        lncrna_expression, value_column=cfg.expression_value_column
    )
    if set(map(int, core_embeddings_by_fold)) != set(range(N_FOLDS)):
        raise StateTrainingError("Exactly five V3.2 fold core-embedding tables are required")
    cores = {
        fold: validate_core_embedding_frame(core_embeddings_by_fold[fold], fold)
        for fold in range(N_FOLDS)
    }
    widths = {
        sum(str(column).startswith("core_feature_") for column in frame.columns)
        for frame in cores.values()
    }
    if len(widths) != 1:
        raise StateTrainingError("Core embedding width changes across folds")
    state, expression, folds, alignment_counts = _align_source_tables(
        state,
        expression,
        folds,
        expression_value_column=cfg.expression_value_column,
    )
    observed = tuple(state_id for state_id in HISTORICAL_STATE_IDS if state.state_id.eq(state_id).any())
    missing = tuple(state_id for state_id in HISTORICAL_STATE_IDS if state_id not in observed)

    predictions: list[pd.DataFrame] = []
    heads: dict[int, Any] = {}
    lineage: dict[int, dict[str, Any]] = {}
    for fold in range(N_FOLDS):
        split_manifest = assign_outer_split(
            folds,
            fold,
            n_folds=N_FOLDS,
            validation_offset=int(cfg.validation_offset),
        )
        statistics = {
            split: compute_fold_local_associations(
                state,
                expression,
                split_manifest,
                split=split,
                min_pairs=int(cfg.min_pairs),
                effect_threshold=float(cfg.effect_threshold),
                expression_value_column=cfg.expression_value_column,
            )
            for split in ("train", "validation", "test")
        }
        head, fold_lineage, max_train_pairs = train_fold_private_head(
            statistics["train"],
            statistics["validation"],
            cores[fold],
            fold=fold,
            config=cfg,
        )
        fold_predictions = build_fold_oof_predictions(
            head,
            statistics["train"],
            statistics["test"],
            cores[fold],
            fold=fold,
            max_train_pairs=max_train_pairs,
            config=cfg,
        )
        fold_lineage.update(
            {
                "split_manifest_sha256": dataframe_sha256(
                    split_manifest, ["cancer_id", "sample_id", "outer_fold"]
                ),
                "fold_local_features_and_labels": True,
                "test_measurements_used_for_training": False,
                "alignment_counts": dict(alignment_counts),
            }
        )
        if core_lineage_by_fold is not None:
            fold_lineage.update(dict(core_lineage_by_fold[fold]))
        heads[fold] = head
        lineage[fold] = fold_lineage
        predictions.append(fold_predictions)
    oof = pd.concat(predictions, ignore_index=True).sort_values(
        ["patient_fold_id", "cancer_id", "lncrna_id", "state_id"], kind="stable"
    ).reset_index(drop=True)
    release = aggregate_oof_predictions(
        oof, min_available_folds=int(cfg.min_available_folds)
    )
    return StateTrainingResult(
        oof_predictions=oof,
        release_predictions=release,
        fold_heads=heads,
        fold_lineage=lineage,
        observed_state_ids=observed,
        missing_state_ids=missing,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _save_new_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def run_state_training_from_paths(
    *,
    state_measurements_path: str | Path,
    fold_manifest_path: str | Path,
    core_embedding_manifest_path: str | Path,
    lncrna_expression_path: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    run_id: str,
    config: StateTrainingConfig | None = None,
) -> dict[str, Any]:
    cfg = config or StateTrainingConfig()
    cfg.validate()
    if not re.fullmatch(r"v32-[a-z0-9][a-z0-9._-]*", str(run_id).strip()):
        raise StateTrainingError("run_id must be a non-empty lowercase v32-* identifier")
    input_paths = {
        "tumor_state_long_measurement": reject_forbidden_input_path(
            state_measurements_path, "tumor_state_long measurement"
        ),
        "v32_patient_fold_manifest": reject_forbidden_input_path(
            fold_manifest_path, "V3.2 patient fold manifest"
        ),
        "v32_core_embedding_manifest": reject_forbidden_input_path(
            core_embedding_manifest_path, "V3.2 core embedding manifest"
        ),
        "lncrna_expression": reject_forbidden_input_path(
            lncrna_expression_path, "lncRNA expression"
        ),
    }
    state = read_table(input_paths["tumor_state_long_measurement"], "tumor_state_long measurement")
    historical_mask = state["state_id"].astype(str).isin(HISTORICAL_STATE_IDS)
    state_input_audit = {
        "raw_rows": int(len(state)),
        "historical_state_rows": int(historical_mask.sum()),
        "excluded_non_historical_rows": int((~historical_mask).sum()),
        "excluded_non_historical_rows_missing_cancer_id": int(
            ((~historical_mask) & state["cancer_id"].isna()).sum()
        ),
        "historical_rows_missing_required_identifier": int(
            state.loc[historical_mask, ["sample_id", "cancer_id", "patient_id", "state_id"]]
            .isna()
            .any(axis=1)
            .sum()
        ),
        "exclusion_policy": "ONLY_NON_HISTORICAL_METADATA_ROWS_ARE_OUT_OF_SCOPE",
    }
    folds = read_table(input_paths["v32_patient_fold_manifest"], "V3.2 patient fold manifest")
    expression = read_table(input_paths["lncrna_expression"], "lncRNA expression")
    cores, core_lineage, _ = load_v32_core_embeddings(
        input_paths["v32_core_embedding_manifest"], repo_root=repo_root
    )
    core_hashes_before = {
        fold: file_sha256(core_lineage[fold]["core_embedding_path"])
        for fold in range(N_FOLDS)
    }
    result = train_state_models(
        state,
        folds,
        cores,
        expression,
        config=cfg,
        core_lineage_by_fold=core_lineage,
    )
    core_hashes_after = {
        fold: file_sha256(core_lineage[fold]["core_embedding_path"])
        for fold in range(N_FOLDS)
    }
    if core_hashes_before != core_hashes_after:
        raise StateTrainingError("A supposedly frozen V3.2 core embedding artifact changed")
    for fold in range(N_FOLDS):
        result.fold_lineage[fold].update(
            {
                "core_embedding_sha256_before": core_hashes_before[fold],
                "core_embedding_sha256_after": core_hashes_after[fold],
                "core_embedding_artifact_unchanged": True,
                "core_parameters_before_sha256": core_lineage[fold][
                    "core_parameter_sha256"
                ],
                "core_parameters_after_sha256": core_lineage[fold][
                    "core_parameter_sha256"
                ],
            }
        )

    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    oof_path = output / "v32_state_oof_predictions.parquet"
    release_path = output / "v32_lncrna_state_release.parquet"
    _atomic_parquet(oof_path, result.oof_predictions)
    _atomic_parquet(release_path, result.release_predictions)

    checkpoint_rows: dict[str, Any] = {}
    for fold in range(N_FOLDS):
        checkpoint_path = output / f"patient_fold={fold}" / "v32_private_state_head.pt"
        head = result.fold_heads[fold]
        fold_row = result.fold_lineage[fold]
        _save_new_checkpoint(
            checkpoint_path,
            {
                "checkpoint_format": STATE_CHECKPOINT_FORMAT,
                "analysis_version": ANALYSIS_VERSION,
                "module_id": "state",
                "patient_fold": fold,
                "run_id": str(run_id),
                "model_state": {
                    name: value.detach().cpu() for name, value in head.state_dict().items()
                },
                "initialization": fold_row["initialization"],
                "final_parameter_sha256": fold_row["final_parameter_sha256"],
                "source_checkpoint_sha256": None,
                "private_head_trained_from_scratch": True,
                "core_parameters_frozen": True,
                "historical_predictions_used": False,
                "state_ids": list(HISTORICAL_STATE_IDS),
                "config": asdict(cfg),
            },
        )
        checkpoint_rows[str(fold)] = {
            "path": checkpoint_path.as_posix(),
            "sha256": file_sha256(checkpoint_path),
            "format": STATE_CHECKPOINT_FORMAT,
            "source_checkpoint_sha256": None,
        }

    input_frames = {
        "tumor_state_long_measurement": state,
        "v32_patient_fold_manifest": folds,
        "lncrna_expression": expression,
    }
    input_artifacts = []
    for role, path in input_paths.items():
        artifact = {
            "role": role,
            "path": path.resolve().as_posix(),
            "sha256": path_sha256(path),
        }
        if role in input_frames:
            artifact.update(
                {
                    "rows": int(len(input_frames[role])),
                    "columns": list(map(str, input_frames[role].columns)),
                }
            )
        input_artifacts.append(artifact)
    availability = (
        result.release_predictions.groupby("state_id", observed=True)
        .availability.agg(rows="size", available_rows="sum")
        .reset_index()
    )
    availability_rows = {
        row.state_id: {"rows": int(row.rows), "available_rows": int(row.available_rows)}
        for row in availability.itertuples(index=False)
    }
    lineage: dict[str, Any] = {
        "training_format": STATE_TRAINING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "state",
        "training_run_id": str(run_id),
        "training_status": "SUCCESS",
        "model_version": "V3.2",
        "folds": N_FOLDS,
        "seeds": [int(cfg.seed) + fold for fold in range(N_FOLDS)],
        "initialization_policy": "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        "private_head_trained_from_scratch": True,
        "core_parameters_frozen": True,
        "core_embeddings_detached": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "strict_state_oof_used": False,
        "lncrna_state_final_used": False,
        "historical_state_ids": list(HISTORICAL_STATE_IDS),
        "mainline_required_state_ids": sorted(MAINLINE_REQUIRED_STATE_IDS),
        "observed_state_ids": list(result.observed_state_ids),
        "missing_state_ids": list(result.missing_state_ids),
        "full_historical_state_measurement_coverage": not result.missing_state_ids,
        "state_input_audit": state_input_audit,
        "unavailable_encoding": "null_with_reason",
        "unavailable_sentinel_values_used": False,
        "test_measurements_used_for_training": False,
        "fold_local_association_features_and_labels": True,
        "config": asdict(cfg),
        "config_sha256": canonical_json_sha256(asdict(cfg)),
        "input_artifacts": input_artifacts,
        "fold_lineage": {str(key): value for key, value in result.fold_lineage.items()},
        "new_private_head_checkpoints": checkpoint_rows,
        "availability_by_state": availability_rows,
        "outputs": {
            "oof_predictions": {
                "path": oof_path.as_posix(),
                "sha256": file_sha256(oof_path),
                "rows": int(len(result.oof_predictions)),
            },
            "lncrna_state_release": {
                "path": release_path.as_posix(),
                "sha256": file_sha256(release_path),
                "rows": int(len(result.release_predictions)),
            },
        },
    }
    lineage["lineage_sha256"] = canonical_json_sha256(lineage)
    lineage_path = output / "STATE_TRAINING_LINEAGE.json"
    _atomic_json(lineage_path, lineage)
    return {
        "status": "SUCCESS",
        "lineage": lineage_path.as_posix(),
        "oof_predictions": oof_path.as_posix(),
        "lncrna_state_release": release_path.as_posix(),
        "missing_state_ids": list(result.missing_state_ids),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train fresh V3.2 seven-State private heads from measurements"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--state-measurements", required=True)
    parser.add_argument("--patient-fold-manifest", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--core-embedding-manifest", required=True)
    parser.add_argument("--lncrna-expression", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument("--effect-threshold", type=float, default=0.20)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--min-available-folds", type=int, default=3)
    parser.add_argument("--expression-value-column", default="logcpm")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    from .patient_fold_authority import validate_frozen_v32_patient_fold_binding

    validate_frozen_v32_patient_fold_binding(
        args.patient_fold_manifest,
        args.patient_fold_authority_receipt,
    )
    config = StateTrainingConfig(
        seed=args.seed,
        min_pairs=args.min_pairs,
        effect_threshold=args.effect_threshold,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        hidden_features=args.hidden_features,
        dropout=args.dropout,
        min_available_folds=args.min_available_folds,
        expression_value_column=args.expression_value_column,
        device=args.device,
    )
    result = run_state_training_from_paths(
        state_measurements_path=args.state_measurements,
        fold_manifest_path=args.patient_fold_manifest,
        core_embedding_manifest_path=args.core_embedding_manifest,
        lncrna_expression_path=args.lncrna_expression,
        output_root=args.output_root,
        repo_root=args.repo_root,
        run_id=args.run_id,
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "ANALYSIS_VERSION",
    "STATE_TRAINING_FORMAT",
    "STATE_CHECKPOINT_FORMAT",
    "HISTORICAL_STATE_IDS",
    "MAINLINE_REQUIRED_STATE_IDS",
    "StateTrainingError",
    "StateTrainingConfig",
    "StateTrainingResult",
    "file_sha256",
    "path_sha256",
    "canonical_patient_id",
    "reject_forbidden_input_path",
    "validate_state_measurements",
    "validate_fold_manifest",
    "validate_lncrna_expression",
    "validate_core_embedding_frame",
    "load_v32_core_embeddings",
    "compute_fold_local_associations",
    "train_fold_private_head",
    "build_fold_oof_predictions",
    "validate_nullable_predictions",
    "aggregate_oof_predictions",
    "train_state_models",
    "run_state_training_from_paths",
    "build_arg_parser",
    "main",
]
