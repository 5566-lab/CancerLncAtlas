"""Shared tumour-state measurements and joint-core input validation.

This module contains data preparation only.  It does not construct a State
head, run an optimizer, load a private State checkpoint, or publish results.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .patient_folds import assign_outer_split


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
STATE_REQUIRED_COLUMNS = frozenset(
    {"sample_id", "cancer_id", "patient_id", "state_id", "state_value"}
)
FOLD_REQUIRED_COLUMNS = frozenset(
    {"sample_id", "cancer_id", "patient_id", "patient_fold_id"}
)
EXPRESSION_REQUIRED_COLUMNS = frozenset({"sample_id", "cancer_id", "lncrna_id"})
FORBIDDEN_INPUT_NAME_TOKENS = (
    "strict_state_oof", "lncrna_state_final", "checkpoint", "prediction",
    "probability", "state_oof_prediction", "historical_prediction",
    "old_prediction", "old_checkpoint", "legacy_checkpoint", "final_probability",
    "ranking",
)
FORBIDDEN_INPUT_COLUMN_TOKENS = (
    "strict_state_oof", "lncrna_state_final", "checkpoint", "prediction",
    "probability", "ranking", "historical_prediction", "old_prediction",
    "legacy_probability", "old_probability", "old_ranking", "source_checkpoint",
)
TORCH_INPUT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt", ".bin"})
_TCGA_PATIENT_RE = re.compile(r"^TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}")


class StateInputError(RuntimeError):
    """Raised when a tumour-state measurement or joint-core input is invalid."""


@dataclass(frozen=True)
class StateInputConfig:
    validation_offset: int = 1
    min_pairs: int = 10
    effect_threshold: float = 0.20
    expression_value_column: str = "logcpm"

    def validate(self) -> None:
        if type(self.validation_offset) is not int or not 0 < self.validation_offset < N_FOLDS:
            raise StateInputError("validation_offset must be in 1..4")
        if type(self.min_pairs) is not int or self.min_pairs < 2:
            raise StateInputError("min_pairs must be at least two")
        if (type(self.effect_threshold) not in (int, float)
                or not math.isfinite(self.effect_threshold)
                or not 0 < self.effect_threshold < 1):
            raise StateInputError("effect_threshold must be finite and in (0, 1)")
        if not str(self.expression_value_column).strip():
            raise StateInputError("expression_value_column is empty")


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    if not source.is_file():
        raise StateInputError(f"Required file is missing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def path_sha256(path: str | Path) -> str:
    source = Path(path)
    if source.is_file():
        return file_sha256(source)
    if not source.is_dir():
        raise StateInputError(f"Required artifact is missing: {source}")
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise StateInputError(f"Input dataset directory is empty: {source}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def reject_forbidden_input_path(path: str | Path, role: str) -> Path:
    source = Path(path)
    normalized = source.as_posix().lower()
    if source.suffix.lower() in TORCH_INPUT_SUFFIXES:
        raise StateInputError(f"{role} may not be a checkpoint/model file: {source}")
    matched = [token for token in FORBIDDEN_INPUT_NAME_TOKENS if token in normalized]
    if matched:
        raise StateInputError(
            f"{role} path names a forbidden model result ({matched[0]}): {source}"
        )
    return source


def _reject_forbidden_columns(frame: pd.DataFrame, role: str) -> None:
    bad = sorted(
        column for column in map(str, frame.columns)
        if any(token in column.lower() for token in FORBIDDEN_INPUT_COLUMN_TOKENS)
    )
    if bad:
        raise StateInputError(f"{role} contains forbidden model-result columns: {bad}")


def read_table(path: str | Path, role: str = "input") -> pd.DataFrame:
    source = reject_forbidden_input_path(path, role)
    if source.is_dir():
        parts = sorted(source.rglob("*.parquet"))
        if not parts:
            raise StateInputError(f"{role} dataset has no Parquet partitions: {source}")
        return pd.concat((pd.read_parquet(part) for part in parts), ignore_index=True)
    if not source.is_file():
        raise StateInputError(f"{role} file/dataset is missing: {source}")
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    dtype = {column: str for column in
             ("sample_id", "patient_id", "cancer_id", "lncrna_id", "state_id", "node_id")}
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source, dtype=dtype)
    if source.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(source, sep="\t", dtype=dtype)
    raise StateInputError(f"Unsupported {role} table format: {source}")


def _require_columns(frame: pd.DataFrame, required: set[str] | frozenset[str], role: str) -> None:
    missing = sorted(set(required) - set(map(str, frame.columns)))
    if missing:
        raise StateInputError(f"{role} lacks required columns: {missing}")


def _require_nonempty_ids(frame: pd.DataFrame, columns: Sequence[str], role: str) -> None:
    for column in columns:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise StateInputError(f"{role}.{column} contains missing/empty identifiers")


def validate_state_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    _reject_forbidden_columns(frame, "tumour-state measurement")
    _require_columns(frame, STATE_REQUIRED_COLUMNS, "tumour-state measurement")
    result = frame.loc[:, list(STATE_REQUIRED_COLUMNS)].copy()
    _require_nonempty_ids(result, ["state_id"], "state")
    result["state_id"] = result.state_id.astype(str)
    result = result.loc[result.state_id.isin(HISTORICAL_STATE_IDS)].copy()
    if result.empty:
        raise StateInputError("No registered State measurements were found")
    _require_nonempty_ids(result, ["sample_id", "cancer_id", "patient_id"], "state")
    for column in ("sample_id", "cancer_id", "patient_id", "state_id"):
        result[column] = result[column].astype(str)
    result["state_value"] = pd.to_numeric(result.state_value, errors="coerce")
    if not np.isfinite(result.state_value.dropna().to_numpy(float)).all():
        raise StateInputError("state_value contains infinity")
    keys = ["cancer_id", "sample_id", "state_id"]
    if result[keys].duplicated().any():
        raise StateInputError("State measurements contain duplicate cancer/sample/state rows")
    if (result.groupby("sample_id", observed=True).cancer_id.nunique() > 1).any():
        raise StateInputError("A State sample appears in multiple cancers")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def validate_fold_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    _reject_forbidden_columns(frame, "patient fold manifest")
    _require_columns(frame, FOLD_REQUIRED_COLUMNS, "patient fold manifest")
    columns = ["cancer_id", "sample_id", "patient_fold_id", "patient_id"]
    if "fold_seed" in frame.columns:
        columns.append("fold_seed")
    result = frame.loc[:, columns].copy()
    _require_nonempty_ids(result, ["sample_id", "cancer_id", "patient_id"], "fold manifest")
    for column in ("sample_id", "cancer_id", "patient_id"):
        result[column] = result[column].astype(str).str.strip()
    if (result.groupby(["cancer_id", "sample_id"], observed=True).patient_id.nunique() > 1).any():
        raise StateInputError("A fold sample maps to multiple explicit patients")
    result["patient_fold_id"] = pd.to_numeric(result.patient_fold_id, errors="coerce").astype("Int64")
    if result.patient_fold_id.isna().any():
        raise StateInputError("patient_fold_id is not an integer")
    observed = set(result.patient_fold_id.astype(int))
    if observed != set(range(N_FOLDS)):
        raise StateInputError(f"Patient fold manifest must contain folds 0..4, observed {observed}")
    if result.sample_id.duplicated().any():
        raise StateInputError("Patient fold manifest duplicates sample_id")
    if (result.groupby(["cancer_id", "patient_id"], observed=True).patient_fold_id.nunique() > 1).any():
        raise StateInputError("Samples from one patient cross patient folds")
    if "fold_seed" in result and len(pd.to_numeric(result.fold_seed, errors="coerce").dropna().astype(int).unique()) > 1:
        raise StateInputError("Patient fold manifest contains multiple fold seeds")
    return result.sort_values(["cancer_id", "patient_fold_id", "sample_id"], kind="stable").reset_index(drop=True)


def canonical_patient_id(sample_id: Any) -> str:
    value = str(sample_id).strip()
    match = _TCGA_PATIENT_RE.match(value)
    return match.group(0).upper() if match else value


def validate_lncrna_expression(frame: pd.DataFrame, value_column: str = "logcpm") -> pd.DataFrame:
    _reject_forbidden_columns(frame, "lncRNA expression")
    _require_columns(frame, set(EXPRESSION_REQUIRED_COLUMNS) | {str(value_column)}, "lncRNA expression")
    result = frame.loc[:, ["cancer_id", "sample_id", "lncrna_id", value_column]].copy()
    _require_nonempty_ids(result, ["cancer_id", "sample_id", "lncrna_id"], "expression")
    for column in ("cancer_id", "sample_id", "lncrna_id"):
        result[column] = result[column].astype(str)
    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")
    if not np.isfinite(result[value_column].dropna().to_numpy(float)).all():
        raise StateInputError(f"{value_column} contains infinity")
    keys = ["cancer_id", "sample_id", "lncrna_id"]
    if result[keys].duplicated().any():
        raise StateInputError("lncRNA expression has duplicate cancer/sample/lncRNA rows")
    if result[value_column].notna().sum() == 0:
        raise StateInputError("lncRNA expression has no measured values")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def validate_core_embedding_frame(frame: pd.DataFrame, fold: int) -> pd.DataFrame:
    _require_columns(frame, {"node_id"}, f"fold-{fold} core embeddings")
    feature_columns = sorted(c for c in map(str, frame.columns) if c.startswith("core_feature_"))
    if not feature_columns:
        raise StateInputError(f"Fold {fold} core embedding has no core_feature columns")
    result = frame.loc[:, ["node_id", *feature_columns]].copy()
    _require_nonempty_ids(result, ["node_id"], f"fold-{fold} core embeddings")
    result["node_id"] = result.node_id.astype(str)
    if result.node_id.duplicated().any():
        raise StateInputError(f"Fold {fold} core embedding duplicates node_id")
    result[feature_columns] = result[feature_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(result[feature_columns].to_numpy(float)).all():
        raise StateInputError(f"Fold {fold} core embeddings contain non-finite values")
    return result.sort_values("node_id", kind="stable").reset_index(drop=True)


def _resolve_export(manifest_path: Path, value: str) -> Path:
    path = (manifest_path.parent / value).resolve()
    if not path.is_relative_to(manifest_path.parent):
        raise StateInputError("Core embedding export escapes its manifest directory")
    return path


def load_v32_core_embeddings(
    manifest_path: str | Path, *, repo_root: str | Path | None = None
) -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, Any]], dict[str, Any]]:
    """Load five validated lncRNA exports from a current joint-core manifest.

    ``repo_root`` remains accepted for callers that already pass it, but export
    paths are resolved against and confined to the manifest directory.
    """
    del repo_root
    source = Path(manifest_path).resolve()
    if not source.is_file():
        raise StateInputError(f"Core embedding manifest is missing: {source}")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise StateInputError("Core embedding manifest must be an object")
    from .core_export_contract_r4 import EXPORT_FORMAT as JOINT_EXPORT_FORMAT, validate_export
    from .current_core_export import EXPORT_FORMAT as CURRENT_EXPORT_FORMAT, validate_current_export
    if manifest.get("export_format") == JOINT_EXPORT_FORMAT:
        validate_export(source)
    elif manifest.get("export_format") == CURRENT_EXPORT_FORMAT:
        validate_current_export(source)
    else:
        raise StateInputError("Core embeddings must use a validated current joint export")
    folds = manifest.get("folds")
    if not isinstance(folds, Mapping) or set(map(str, folds)) != set(map(str, range(N_FOLDS))):
        raise StateInputError("Core embedding manifest must contain folds 0..4")
    embeddings: dict[int, pd.DataFrame] = {}
    lineage: dict[int, dict[str, Any]] = {}
    width: int | None = None
    for fold in range(N_FOLDS):
        row = folds[str(fold)]
        if int(row.get("patient_fold", -1)) != fold:
            raise StateInputError(f"Core embedding fold identity drift for fold {fold}")
        exports = row.get("exports", {})
        key = next((name for name in exports if str(name).lower() == "lncrna"), None)
        if key is None:
            raise StateInputError(f"Fold {fold} lacks lncRNA core embeddings")
        declaration = exports[key]
        path = _resolve_export(source, str(declaration.get("path", "")))
        if file_sha256(path) != declaration.get("sha256"):
            raise StateInputError(f"Fold {fold} core embedding SHA-256 mismatch")
        frame = validate_core_embedding_frame(pd.read_parquet(path), fold)
        features = sum(str(c).startswith("core_feature_") for c in frame.columns)
        if int(declaration.get("rows", -1)) != len(frame) or int(declaration.get("features", -1)) != features:
            raise StateInputError(f"Fold {fold} core embedding shape differs from manifest")
        if width is None:
            width = features
        elif width != features:
            raise StateInputError("Core embedding width changes across folds")
        embeddings[fold] = frame
        lineage[fold] = {
            "core_run_id": manifest.get("core_run_id"),
            "core_embedding_path": path.as_posix(),
            "core_embedding_sha256": declaration["sha256"],
            "core_parameter_sha256": row["core_parameter_sha256"],
            "v32_core_checkpoint_sha256": row["checkpoint_sha256"],
            "core_checkpoint_format": row["checkpoint_format"],
        }
    return embeddings, lineage, dict(manifest)


def _paired_correlations(matrix: np.ndarray, state_values: np.ndarray, min_pairs: int,
                         *, patient_observer=None):
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(state_values, dtype=float).reshape(-1, 1)
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum(axis=0).astype(np.int64)
    x0, y0 = np.where(mask, x, 0.0), np.where(mask, y, 0.0)
    safe_n = np.maximum(n, 1)
    sx, sy = x0.sum(axis=0), y0.sum(axis=0)
    cross = (x0 * y0).sum(axis=0) - sx * sy / safe_n
    vx = (x0 * x0).sum(axis=0) - sx * sx / safe_n
    vy = (y0 * y0).sum(axis=0) - sy * sy / safe_n
    denominator = np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
    effect = np.full(x.shape[1], np.nan)
    available = (n >= min_pairs) & (vx > 1e-12) & (vy > 1e-12)
    effect[available] = np.clip(cross[available] / denominator[available], -1, 1)
    reason = np.full(x.shape[1], None, dtype=object)
    reason[n < min_pairs] = "insufficient_pairs"
    reason[(n >= min_pairs) & (vx <= 1e-12)] = "lncrna_expression_zero_variance"
    reason[(n >= min_pairs) & (vx > 1e-12) & (vy <= 1e-12)] = "state_measurement_zero_variance"
    if patient_observer is not None:
        patient_observer(mask, n, pd.isna(reason))
    return n, effect, reason


def compute_fold_local_associations(state_measurements, lncrna_expression, split_manifest,
                                    *, split, min_pairs, effect_threshold,
                                    expression_value_column="logcpm", patient_observer=None):
    if split not in {"train", "validation", "test"}:
        raise StateInputError(f"Unregistered split: {split}")
    _require_columns(split_manifest, {"cancer_id", "sample_id", "split"}, "outer split")
    assignments = split_manifest.loc[split_manifest.split.astype(str).eq(split),
                                     ["cancer_id", "sample_id"]].astype(str)
    expression = lncrna_expression.merge(assignments.assign(_selected=True),
        on=["cancer_id", "sample_id"], how="inner", validate="many_to_one")
    states = state_measurements.merge(assignments.assign(_selected=True),
        on=["cancer_id", "sample_id"], how="inner", validate="many_to_one")
    universe = lncrna_expression[["cancer_id", "lncrna_id"]].astype(str).drop_duplicates()
    frames = []
    for cancer, candidates in universe.groupby("cancer_id", observed=True):
        lncrnas = sorted(candidates.lncrna_id.astype(str))
        subset = expression.loc[expression.cancer_id.eq(cancer)]
        wide = (pd.DataFrame(columns=lncrnas, dtype=float) if subset.empty else
                subset.pivot(index="sample_id", columns="lncrna_id",
                             values=expression_value_column).reindex(columns=lncrnas))
        cancer_states = states.loc[states.cancer_id.eq(cancer)]
        for state_id in HISTORICAL_STATE_IDS:
            rows = cancer_states.loc[cancer_states.state_id.eq(state_id),
                                     ["sample_id", "state_value"]].dropna()
            if rows.empty:
                n = np.zeros(len(lncrnas), dtype=np.int64)
                effect = np.full(len(lncrnas), np.nan)
                reason = np.full(len(lncrnas), "state_measurement_missing", dtype=object)
            else:
                values = rows.set_index("sample_id").state_value.astype(float)
                observer = None if patient_observer is None else lambda mask, counts, available: patient_observer(
                    str(cancer), state_id, split, lncrnas, values.index.tolist(), mask, counts, available)
                n, effect, reason = _paired_correlations(wide.reindex(values.index).to_numpy(float),
                                                         values.to_numpy(float), min_pairs,
                                                         patient_observer=observer)
            available = pd.isna(reason)
            label = np.full(len(lncrnas), np.nan)
            label[available] = (np.abs(effect[available]) >= effect_threshold).astype(float)
            frames.append(pd.DataFrame({"cancer_id": str(cancer), "lncrna_id": lncrnas,
                "state_id": state_id, "split": split, "n_pairs": n, "effect": effect,
                "association_label": label, "available": available,
                "unavailability_reason": reason}))
    if not frames:
        raise StateInputError("No cancer/lncRNA candidates were available")
    result = pd.concat(frames, ignore_index=True)
    keys = ["cancer_id", "lncrna_id", "state_id"]
    if result[keys].duplicated().any():
        raise StateInputError("Fold-local association table has duplicate candidates")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _align_source_tables(state, expression, folds, *, expression_value_column,
                         alignment_observer=None):
    assignments = folds[["cancer_id", "sample_id", "patient_id", "patient_fold_id"]].copy()
    if (assignments.groupby(["cancer_id", "patient_id"], observed=True).patient_fold_id.nunique() > 1).any():
        raise StateInputError("Patient-level alignment would cross folds")
    patient_folds = (assignments.groupby(["cancer_id", "patient_id"], observed=True,
        as_index=False).patient_fold_id.first().rename(columns={"patient_id": "sample_id"}))
    patient_folds["patient_id"] = patient_folds.sample_id
    state_source = state.copy()
    state_source["patient_id"] = state_source.patient_id.map(canonical_patient_id)
    state_joined = state_source.merge(patient_folds[["cancer_id", "sample_id"]].rename(
        columns={"sample_id": "patient_id"}), on=["cancer_id", "patient_id"],
        how="inner", validate="many_to_one")
    state_patient = (state_joined.groupby(["cancer_id", "patient_id", "state_id"],
        observed=True, as_index=False).state_value.mean().rename(columns={"patient_id": "sample_id"}))
    state_patient["patient_id"] = state_patient.sample_id
    expression_joined = expression.merge(assignments.assign(_in_fold=True),
        on=["cancer_id", "sample_id"], how="inner", validate="many_to_one")
    expression_patient = (expression_joined.groupby(["cancer_id", "patient_id", "lncrna_id"],
        observed=True, as_index=False)[expression_value_column].mean().rename(columns={"patient_id": "sample_id"}))
    if alignment_observer is not None:
        alignment_observer("tumor_state_long_measurement", state_joined, state_patient, "state_id", "state_value")
        alignment_observer("lncrna_expression", expression_joined, expression_patient, "lncrna_id", expression_value_column)
    if state_patient.empty or expression_patient.empty:
        raise StateInputError("State/expression inputs do not overlap the fold manifest")
    counts = {"state_rows_outside_fold_manifest": len(state)-len(state_joined),
        "expression_rows_outside_fold_manifest": len(expression)-len(expression_joined),
        "state_rows_collapsed_to_patient": len(state_joined)-len(state_patient),
        "expression_rows_collapsed_to_patient": len(expression_joined)-len(expression_patient),
        "analysis_patients": patient_folds.sample_id.nunique()}
    return state_patient, expression_patient, patient_folds, {k: int(v) for k, v in counts.items()}


def prepare_state_inputs(state_measurements, fold_manifest, core_embeddings_by_fold,
                         lncrna_expression, cfg: StateInputConfig, *, alignment_observer=None):
    cfg.validate()
    state = validate_state_measurements(state_measurements)
    folds = validate_fold_manifest(fold_manifest)
    expression = validate_lncrna_expression(lncrna_expression, cfg.expression_value_column)
    if set(core_embeddings_by_fold) != set(range(N_FOLDS)):
        raise StateInputError("Exactly five fold core-embedding tables are required")
    cores = {fold: validate_core_embedding_frame(core_embeddings_by_fold[fold], fold)
             for fold in range(N_FOLDS)}
    state, expression, folds, counts = _align_source_tables(state, expression, folds,
        expression_value_column=cfg.expression_value_column, alignment_observer=alignment_observer)
    return state, expression, folds, cores, counts


def state_fold_statistics(state, expression, folds, fold, cfg: StateInputConfig,
                          *, splits=("train", "validation", "test"), patient_observer=None):
    split_manifest = assign_outer_split(folds, fold, n_folds=N_FOLDS,
                                        validation_offset=cfg.validation_offset)
    statistics = {split: compute_fold_local_associations(state, expression, split_manifest,
        split=split, min_pairs=cfg.min_pairs, effect_threshold=cfg.effect_threshold,
        expression_value_column=cfg.expression_value_column,
        patient_observer=patient_observer) for split in splits}
    return split_manifest, statistics


__all__ = ["HISTORICAL_STATE_IDS", "N_FOLDS", "StateInputConfig", "StateInputError",
    "canonical_patient_id", "compute_fold_local_associations", "file_sha256",
    "load_v32_core_embeddings", "path_sha256", "prepare_state_inputs", "read_table",
    "state_fold_statistics", "validate_core_embedding_frame", "validate_fold_manifest",
    "validate_lncrna_expression", "validate_state_measurements"]
