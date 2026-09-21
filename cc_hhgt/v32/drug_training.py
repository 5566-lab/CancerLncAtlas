"""Fresh V3.2 cell-line drug-association private-head training.

This module intentionally starts from cell-line level response and lncRNA
expression measurements.  Earlier PRISM/GDSC association tables, model
checkpoints, probabilities and rankings are rejected.  The public estimand is
an association replicated across held-out cell lines; it is never presented as
a TCGA patient treatment response.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .input_lineage import artifact_sha256, audit_input_lineage


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
MODULE_ID = "drug"
TARGET_LEVEL = "cancer_x_lncrna_x_drug_response"
TARGET_KEYS = ("cancer_id", "lncrna_id", "drug_id")
N_FOLDS = 5
CORE_EXPORT_FORMAT = "CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1"
PRIVATE_CHECKPOINT_FORMAT = "CC_HHGT_V3_2_DRUG_PRIVATE_HEAD_V1"
PREDICTION_FORMAT = "CC_HHGT_V3_2_DRUG_CELL_LINE_ASSOCIATION_V1"
CROSS_DATASET_ASSOCIATION_POLICY = (
    "NO_OVERLAP_FISHER_Z_N_MINUS_3__OVERLAP_UNIQUE_CANONICAL_MODEL_"
    "MIDRANK_PERCENTILE_CONSENSUS_SPEARMAN_V1"
)
EXPRESSION_MOMENT_POLICY = (
    "UNIQUE_CANONICAL_MODEL__DATASET_SPECIFIC_ARITHMETIC_MEAN_CONSENSUS_V1"
)

_DOMAIN_FEATURES = (
    "log1p_cell_lines",
    "dataset_count",
    "lncrna_expression_mean",
    "lncrna_expression_sd",
    "log1p_drug_target_count",
    "drug_target_embedding_available",
)
_AGGREGATED_ASSOCIATION_COLUMNS = frozenset(
    {
        "rho",
        "beta",
        "se",
        "p_value",
        "fdr",
        "association_id",
        "observed_or_predicted",
        "replication_status",
        "best_gdsc_fdr",
        "best_gdsc_p_value",
    }
)
_FORBIDDEN_RESULT_COLUMNS = frozenset(
    {
        "prediction",
        "probability",
        "ranking",
        "rank",
        "score",
        "sample_weight",
        "oof_prediction",
        "historical_probability",
        "legacy_probability",
        "association_membership_probability",
    }
)
_FORBIDDEN_INPUT_PATH_TOKENS = (
    "checkpoint",
    "oof_prediction",
    "probability",
    "ranking",
    "ranked",
    "web_table",
    "release_table",
)


class DrugTrainingError(RuntimeError):
    """Raised when a drug input or fresh-training invariant fails."""


@dataclass(frozen=True)
class DrugTrainingConfig:
    seed: int = 20260726
    epochs: int = 40
    batch_size: int = 1024
    hidden_features: int = 64
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    min_cell_lines_per_dataset: int = 5
    min_total_cell_lines: int = 8
    positive_abs_rho: float = 0.30
    negative_abs_rho: float = 0.10
    max_train_rows: int = 200_000
    max_validation_rows: int = 100_000
    prediction_batch_size: int = 16_384
    require_all_folds: bool = True

    def validate(self) -> None:
        positive = (
            self.epochs,
            self.batch_size,
            self.hidden_features,
            self.patience,
            self.min_cell_lines_per_dataset,
            self.min_total_cell_lines,
            self.max_train_rows,
            self.max_validation_rows,
            self.prediction_batch_size,
        )
        if any(int(value) < 1 for value in positive):
            raise ValueError("Drug training integer controls must be positive")
        if not 0 <= float(self.dropout) < 1:
            raise ValueError("dropout must be in [0, 1)")
        if float(self.learning_rate) <= 0 or float(self.weight_decay) < 0:
            raise ValueError("learning rate/weight decay are invalid")
        if not 0 <= float(self.negative_abs_rho) < float(self.positive_abs_rho) <= 1:
            raise ValueError("Require 0 <= negative_abs_rho < positive_abs_rho <= 1")


@dataclass(frozen=True)
class EmbeddingLookup:
    values: np.ndarray
    index: Mapping[str, int]


@dataclass(frozen=True)
class FoldCoreEmbeddings:
    fold: int
    lncrna: EmbeddingLookup
    gene: EmbeddingLookup
    cancer: EmbeddingLookup
    checkpoint_sha256: str
    core_parameter_sha256: str
    input_hashes: Mapping[str, str]


@dataclass(frozen=True)
class AssociationStatistics:
    domain: np.ndarray
    labels: np.ndarray
    assay_available: np.ndarray
    reasons: np.ndarray
    rho: np.ndarray
    n_cell_lines: np.ndarray
    dataset_count: np.ndarray


@dataclass(frozen=True)
class DatasetAssociationArrays:
    """One assay's candidate-aligned values, keyed by canonical model ID."""

    dataset_id: str
    canonical_model_ids: np.ndarray
    expression: np.ndarray
    sensitivity: np.ndarray


@dataclass(frozen=True)
class CrossDatasetAssociationBatch:
    """Cross-screen statistics for candidate-aligned assay arrays."""

    rho: np.ndarray
    n_cell_lines: np.ndarray
    dataset_count: np.ndarray
    expression_mean: np.ndarray
    expression_sd: np.ndarray
    has_cross_dataset_overlap: np.ndarray


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_table(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    suffixes = [suffix.lower() for suffix in source.suffixes]
    logical = suffixes[-2] if suffixes and suffixes[-1] in {".gz", ".bz2", ".xz"} else source.suffix.lower()
    if source.is_dir() or logical == ".parquet":
        return pd.read_parquet(source, columns=list(columns) if columns else None)
    separator = "\t" if logical in {".tsv", ".txt"} else ","
    return pd.read_csv(source, sep=separator, usecols=list(columns) if columns else None)


def _first_column(frame: pd.DataFrame, candidates: Sequence[str], context: str) -> str:
    lookup = {_normalise_name(column): str(column) for column in frame.columns}
    for candidate in candidates:
        if _normalise_name(candidate) in lookup:
            return lookup[_normalise_name(candidate)]
    raise DrugTrainingError(f"{context} lacks any of columns {list(candidates)}")


def _bool_series(values: pd.Series, context: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        if values.isna().any():
            raise DrugTrainingError(f"{context} contains missing direction flags")
        return values.astype(bool)
    lowered = values.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no"}
    if not set(lowered.unique()).issubset(allowed):
        raise DrugTrainingError(f"{context} must be an explicit boolean")
    return lowered.isin({"true", "1", "yes"})


def _assert_no_result_columns(frame: pd.DataFrame, context: str, *, allow_curated_direction: bool = False) -> None:
    normalised = {_normalise_name(column): str(column) for column in frame.columns}
    bad = sorted(
        original
        for token, original in normalised.items()
        if token in _FORBIDDEN_RESULT_COLUMNS
        or token.endswith(("_probability", "_prediction", "_ranking", "_rank"))
    )
    if context in {"raw response", "raw lncRNA expression"}:
        bad.extend(
            original for token, original in normalised.items() if token in _AGGREGATED_ASSOCIATION_COLUMNS
        )
    if not allow_curated_direction and "direction" in normalised and context != "drug candidates":
        # Raw assay direction is represented by higher_is_sensitive, never by
        # an old aggregate association direction.
        bad.append(normalised["direction"])
    if bad:
        raise DrugTrainingError(f"{context} contains old/result-like columns: {sorted(set(bad))}")


def _assert_source_path(path: str | Path, *, static_annotation: bool = False) -> None:
    if static_annotation:
        return
    token = _normalise_name(Path(path).name)
    found = [item for item in _FORBIDDEN_INPUT_PATH_TOKENS if item in token]
    if found:
        raise DrugTrainingError(f"Forbidden historical/result-like input path: {path}; tokens={found}")


def _canonical_lncrna(value: Any) -> str:
    text = str(value).strip()
    if re.fullmatch(r"ENSG\d+(?:\.\d+)?", text, flags=re.IGNORECASE):
        text = text.split(".", 1)[0].upper()
        return f"LNC:{text}"
    return text


def _canonical_gene(value: Any) -> str:
    text = str(value).strip()
    if re.fullmatch(r"ENSG\d+(?:\.\d+)?", text, flags=re.IGNORECASE):
        text = text.split(".", 1)[0].upper()
        return f"GENE:{text}"
    return text


def normalise_exact_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_no_result_columns(frame, "exact candidates")
    cancer = _first_column(frame, ("cancer_id",), "exact candidates")
    lncrna = _first_column(frame, ("lncrna_id",), "exact candidates")
    result = frame[[cancer, lncrna]].rename(columns={cancer: "cancer_id", lncrna: "lncrna_id"})
    result["cancer_id"] = result.cancer_id.astype(str).str.strip().str.upper()
    result["lncrna_id"] = result.lncrna_id.map(_canonical_lncrna)
    if result[["cancer_id", "lncrna_id"]].eq("").any().any():
        raise DrugTrainingError("Exact candidate IDs must be non-empty")
    return result.drop_duplicates().sort_values(["cancer_id", "lncrna_id"], kind="stable").reset_index(drop=True)


def normalise_drug_candidates(frame: pd.DataFrame, exact_candidates: pd.DataFrame) -> pd.DataFrame:
    _assert_no_result_columns(frame, "drug candidates")
    missing = sorted(set(TARGET_KEYS) - set(frame.columns))
    if missing:
        raise DrugTrainingError(f"Drug candidate table lacks keys: {missing}")
    result = frame[list(TARGET_KEYS)].copy()
    result["cancer_id"] = result.cancer_id.astype(str).str.strip().str.upper()
    result["lncrna_id"] = result.lncrna_id.map(_canonical_lncrna)
    result["drug_id"] = result.drug_id.astype(str).str.strip()
    if result[list(TARGET_KEYS)].eq("").any().any():
        raise DrugTrainingError("Drug candidate keys must be non-empty")
    if result.duplicated(list(TARGET_KEYS)).any():
        raise DrugTrainingError("Drug candidate keys are duplicated")
    allowed = set(map(tuple, exact_candidates[["cancer_id", "lncrna_id"]].itertuples(index=False, name=None)))
    observed = set(map(tuple, result[["cancer_id", "lncrna_id"]].itertuples(index=False, name=None)))
    if extra := sorted(observed - allowed)[:5]:
        raise DrugTrainingError(f"Drug candidates are outside the V3.2 exact candidate universe: {extra}")
    return result.sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)


def normalise_cell_line_map(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_no_result_columns(frame, "cell-line map")
    dataset = _first_column(frame, ("dataset_id", "source_dataset", "resource"), "cell-line map")
    cell = _first_column(frame, ("cell_line_id", "model_id", "depmap_id"), "cell-line map")
    cancer = _first_column(frame, ("cancer_id",), "cell-line map")
    lookup = {_normalise_name(column): str(column) for column in frame.columns}
    model = next((lookup[key] for key in ("canonical_model_id", "model_id", "depmap_id") if key in lookup and lookup[key] != cell), None)
    result = frame[[dataset, cell, cancer] + ([model] if model else [])].copy()
    result.columns = ["dataset_id", "cell_line_id", "cancer_id"] + (["canonical_model_id"] if model else [])
    for column in ("dataset_id", "cell_line_id"):
        result[column] = result[column].astype(str).str.strip()
    result["cancer_id"] = result.cancer_id.astype(str).str.strip().str.upper()
    if model:
        result["canonical_model_id"] = result.canonical_model_id.astype(str).str.strip()
    else:
        # Dataset-independent IDs prevent a shared model measured in both
        # GDSC and PRISM from crossing folds when the source IDs agree.
        result["canonical_model_id"] = result.cell_line_id
    if result[["dataset_id", "cell_line_id", "cancer_id", "canonical_model_id"]].eq("").any().any():
        raise DrugTrainingError("Cell-line mapping contains empty identifiers")
    conflicts = result.groupby(["dataset_id", "cell_line_id"], observed=True).canonical_model_id.nunique()
    if (conflicts > 1).any():
        raise DrugTrainingError("A dataset cell line maps to conflicting canonical model IDs")
    # Tissue-compatible cell lines may support more than one TCGA code (for
    # example kidney models for KICH/KIRC/KIRP).  Keeping those rows is valid;
    # the canonical model still receives exactly one fold across all cancers.
    result = result.drop_duplicates(["dataset_id", "cell_line_id", "cancer_id"])
    return result.reset_index(drop=True)


def normalise_raw_response(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_no_result_columns(frame, "raw response")
    dataset = _first_column(frame, ("dataset_id", "source_dataset", "resource"), "raw response")
    cell = _first_column(frame, ("cell_line_id", "model_id", "depmap_id"), "raw response")
    drug = _first_column(frame, ("drug_id",), "raw response")
    value = _first_column(frame, ("response_value", "auc", "ln_ic50", "logfold_change"), "raw response")
    direction = _first_column(frame, ("higher_is_sensitive",), "raw response")
    result = frame[[dataset, cell, drug, value, direction]].rename(
        columns={dataset: "dataset_id", cell: "cell_line_id", drug: "drug_id", value: "response_value", direction: "higher_is_sensitive"}
    )
    for column in ("dataset_id", "cell_line_id", "drug_id"):
        result[column] = result[column].astype(str).str.strip()
    result["response_value"] = pd.to_numeric(result.response_value, errors="coerce")
    result["higher_is_sensitive"] = _bool_series(result.higher_is_sensitive, "higher_is_sensitive")
    result = result.loc[np.isfinite(result.response_value)].copy()
    if result.empty:
        raise DrugTrainingError("Raw drug response has no finite cell-line measurements")
    sign = np.where(result.higher_is_sensitive, 1.0, -1.0)
    result["sensitivity_value"] = result.response_value.to_numpy(float) * sign
    grouped = result.groupby(["dataset_id", "cell_line_id", "drug_id"], observed=True, sort=False)
    conflict = grouped.higher_is_sensitive.nunique()
    if (conflict > 1).any():
        raise DrugTrainingError("Replicate responses disagree on higher_is_sensitive")
    return grouped.agg(
        sensitivity_value=("sensitivity_value", "mean"),
        raw_replicates=("sensitivity_value", "size"),
    ).reset_index()


def normalise_raw_expression(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_no_result_columns(frame, "raw lncRNA expression")
    lncrna = _first_column(frame, ("lncrna_id",), "raw lncRNA expression")
    value = _first_column(frame, ("expression_value", "tpm", "log2_tpm", "expression"), "raw lncRNA expression")
    lookup = {_normalise_name(column): str(column) for column in frame.columns}
    canonical = lookup.get("canonical_model_id")
    if canonical is not None:
        result = frame[[canonical, lncrna, value]].rename(
            columns={canonical: "canonical_model_id", lncrna: "lncrna_id", value: "expression_value"}
        )
        result["canonical_model_id"] = result.canonical_model_id.astype(str).str.strip()
        keys = ["canonical_model_id", "lncrna_id"]
    else:
        dataset = _first_column(frame, ("dataset_id", "source_dataset", "resource"), "raw lncRNA expression")
        cell = _first_column(frame, ("cell_line_id", "model_id", "depmap_id"), "raw lncRNA expression")
        result = frame[[dataset, cell, lncrna, value]].rename(
            columns={dataset: "dataset_id", cell: "cell_line_id", lncrna: "lncrna_id", value: "expression_value"}
        )
        result["dataset_id"] = result.dataset_id.astype(str).str.strip()
        result["cell_line_id"] = result.cell_line_id.astype(str).str.strip()
        keys = ["dataset_id", "cell_line_id", "lncrna_id"]
    result["lncrna_id"] = result.lncrna_id.map(_canonical_lncrna)
    result["expression_value"] = pd.to_numeric(result.expression_value, errors="coerce")
    result = result.loc[np.isfinite(result.expression_value)].copy()
    if result.empty:
        raise DrugTrainingError("Raw lncRNA expression has no finite cell-line measurements")
    return result.groupby(keys, observed=True, sort=False).expression_value.mean().reset_index()


def normalise_drug_targets(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_no_result_columns(frame, "drug-target annotation", allow_curated_direction=True)
    drug = _first_column(frame, ("drug_id",), "drug-target annotation")
    gene = _first_column(frame, ("gene_id",), "drug-target annotation")
    result = frame[[drug, gene]].rename(columns={drug: "drug_id", gene: "gene_id"})
    result["drug_id"] = result.drug_id.astype(str).str.strip()
    result["gene_id"] = result.gene_id.map(_canonical_gene)
    result = result.loc[result.drug_id.ne("") & result.gene_id.ne("")]
    return result.drop_duplicates().reset_index(drop=True)


def normalise_curated_response(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_no_result_columns(frame, "curated drug response", allow_curated_direction=True)
    cancer = _first_column(frame, ("cancer_id",), "curated drug response")
    lncrna = _first_column(frame, ("lncrna_id",), "curated drug response")
    drug = _first_column(frame, ("drug_id",), "curated drug response")
    lookup = {_normalise_name(column): str(column) for column in frame.columns}
    pmid = lookup.get("pmid")
    source = lookup.get("source_database")
    columns = [cancer, lncrna, drug] + ([pmid] if pmid else []) + ([source] if source else [])
    result = frame[columns].copy()
    result.columns = ["cancer_id", "lncrna_id", "drug_id"] + (["pmid"] if pmid else []) + (["source_database"] if source else [])
    result["cancer_id"] = result.cancer_id.astype(str).str.strip().str.upper()
    result["lncrna_id"] = result.lncrna_id.map(_canonical_lncrna)
    result["drug_id"] = result.drug_id.astype(str).str.strip()
    if "pmid" not in result:
        result["pmid"] = pd.NA
    if "source_database" not in result:
        result["source_database"] = "curated"
    return result.drop_duplicates().reset_index(drop=True)


def assign_cell_line_folds(mapping: pd.DataFrame, seed: int) -> pd.DataFrame:
    models = sorted(mapping.canonical_model_id.astype(str).unique())
    if len(models) < N_FOLDS:
        raise DrugTrainingError("At least five independent cell-line models are required")
    ordered = sorted(models, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    fold_map = {model: index % N_FOLDS for index, model in enumerate(ordered)}
    result = mapping.copy()
    result["cell_line_fold_id"] = result.canonical_model_id.map(fold_map).astype(int)
    if set(result.cell_line_fold_id.unique()) != set(range(N_FOLDS)):
        raise DrugTrainingError("Cell-line split failed to create folds 0..4")
    return result


def build_drug_candidate_universe(
    exact_candidates: pd.DataFrame,
    response: pd.DataFrame,
    mapping: pd.DataFrame,
    curated: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cross each V3.2 cancer-lncRNA candidate with actually assayed drugs."""

    linked = response.merge(
        mapping[["dataset_id", "cell_line_id", "cancer_id"]],
        on=["dataset_id", "cell_line_id"],
        how="inner",
        validate="many_to_many",
    )
    parts: list[pd.DataFrame] = []
    drugs_by_cancer = linked.groupby("cancer_id", observed=True).drug_id.unique().to_dict()
    for cancer, local in exact_candidates.groupby("cancer_id", observed=True, sort=False):
        drugs = np.asarray(drugs_by_cancer.get(cancer, ()), dtype=object)
        if not len(drugs):
            continue
        parts.append(
            pd.DataFrame(
                {
                    "cancer_id": str(cancer),
                    "lncrna_id": np.repeat(local.lncrna_id.to_numpy(object), len(drugs)),
                    "drug_id": np.tile(drugs, len(local)),
                }
            )
        )
    if curated is not None and not curated.empty:
        allowed = exact_candidates.merge(curated, on=["cancer_id", "lncrna_id"], how="inner")
        parts.append(allowed[list(TARGET_KEYS)])
    if not parts:
        raise DrugTrainingError("No cancer-lncRNA-drug candidates can be constructed")
    result = pd.concat(parts, ignore_index=True).drop_duplicates(list(TARGET_KEYS))
    return result.sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(float)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(float)
    if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _rank_correlation_columns(matrix: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Column-wise Spearman correlation with a dense fast path."""

    result = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    dense = np.isfinite(matrix).all(axis=0) & np.isfinite(response).all()
    if dense.any():
        ranked_x = pd.DataFrame(matrix[:, dense]).rank(method="average").to_numpy(float)
        ranked_y = pd.Series(response).rank(method="average").to_numpy(float)
        ranked_x -= ranked_x.mean(axis=0)
        ranked_y -= ranked_y.mean()
        denominator = np.sqrt(
            np.sum(ranked_x * ranked_x, axis=0) * np.sum(ranked_y * ranked_y)
        )
        result[dense] = np.divide(
            ranked_x.T @ ranked_y,
            denominator,
            out=np.full(int(dense.sum()), np.nan),
            where=denominator > 1e-12,
        )
    for column in np.flatnonzero(~dense):
        result[column] = _safe_spearman(matrix[:, column], response)
    return result


def _rank_correlation_pairs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Column-wise Spearman correlations for candidate-specific paired vectors."""

    if left.shape != right.shape:
        raise DrugTrainingError("Paired association matrices have different shapes")
    result = np.full(left.shape[1], np.nan, dtype=np.float64)
    dense = np.isfinite(left).all(axis=0) & np.isfinite(right).all(axis=0)
    if dense.any():
        ranked_left = (
            pd.DataFrame(left[:, dense]).rank(method="average").to_numpy(float)
        )
        ranked_right = (
            pd.DataFrame(right[:, dense]).rank(method="average").to_numpy(float)
        )
        ranked_left -= ranked_left.mean(axis=0)
        ranked_right -= ranked_right.mean(axis=0)
        denominator = np.sqrt(
            np.sum(ranked_left * ranked_left, axis=0)
            * np.sum(ranked_right * ranked_right, axis=0)
        )
        result[dense] = np.divide(
            np.sum(ranked_left * ranked_right, axis=0),
            denominator,
            out=np.full(int(dense.sum()), np.nan),
            where=denominator > 1e-12,
        )
    for column in np.flatnonzero(~dense):
        result[column] = _safe_spearman(left[:, column], right[:, column])
    return result


def _midrank_percentile(values: np.ndarray) -> np.ndarray:
    """Return tie-aware midrank percentiles in the open unit interval."""

    vector = np.asarray(values, dtype=float)
    if not len(vector) or not np.isfinite(vector).all():
        raise DrugTrainingError("Midrank percentiles require finite values")
    ranks = pd.Series(vector).rank(method="average").to_numpy(float)
    return (ranks - 0.5) / float(len(vector))


def combine_cross_dataset_associations(
    records: Sequence[DatasetAssociationArrays],
    *,
    candidate_count: int,
    min_cell_lines_per_dataset: int,
) -> CrossDatasetAssociationBatch:
    """Combine screen associations without treating shared models as independent.

    Disjoint screens retain the historical Fisher-z estimator exactly.  When a
    candidate has a canonical model in more than one valid screen, sensitivity
    is converted to a within-screen midrank percentile and averaged per model;
    dataset-specific expression is averaged per model as an explicit consensus.
    A single Spearman correlation and expression moments are then computed over
    the unique canonical-model union.
    """

    size = int(candidate_count)
    if size < 0:
        raise DrugTrainingError("candidate_count cannot be negative")
    if int(min_cell_lines_per_dataset) < 1:
        raise DrugTrainingError("min_cell_lines_per_dataset must be positive")
    rho = np.zeros(size, dtype=np.float64)
    n_values = np.zeros(size, dtype=np.int32)
    dataset_values = np.zeros(size, dtype=np.int16)
    expression_sum = np.zeros(size, dtype=np.float64)
    expression_sum_sq = np.zeros(size, dtype=np.float64)
    expression_n = np.zeros(size, dtype=np.int32)
    if not size or not records:
        return CrossDatasetAssociationBatch(
            rho,
            n_values,
            dataset_values,
            expression_sum,
            expression_sum,
            np.zeros(size, dtype=bool),
        )

    seen_datasets: set[str] = set()
    union_models = sorted(
        {
            str(model)
            for record in records
            for model in np.asarray(record.canonical_model_ids, dtype=object)
        }
    )
    model_position = {model: index for index, model in enumerate(union_models)}
    model_use_count = np.zeros((size, len(union_models)), dtype=np.uint8)
    fisher_sum = np.zeros(size, dtype=np.float64)
    fisher_weight = np.zeros(size, dtype=np.float64)
    processed: list[
        tuple[DatasetAssociationArrays, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for record in records:
        dataset = str(record.dataset_id)
        if dataset in seen_datasets:
            raise DrugTrainingError(f"Duplicate dataset association record: {dataset}")
        seen_datasets.add(dataset)
        models = np.asarray(record.canonical_model_ids, dtype=object).astype(str)
        values = np.asarray(record.expression, dtype=float)
        sensitivity = np.asarray(record.sensitivity, dtype=float)
        if values.shape != (len(models), size) or sensitivity.shape != (len(models),):
            raise DrugTrainingError(
                f"Dataset association array shape drifted for {dataset}"
            )
        if len(models) != len(set(models)):
            raise DrugTrainingError(
                f"Dataset association models are duplicated for {dataset}"
            )
        finite = np.isfinite(values) & np.isfinite(sensitivity)[:, None]
        counts = finite.sum(axis=0)
        eligible = counts >= int(min_cell_lines_per_dataset)
        correlations = np.full(size, np.nan, dtype=np.float64)
        if eligible.any():
            eligible_positions = np.flatnonzero(eligible)
            correlations[eligible_positions] = _rank_correlation_columns(
                values[:, eligible_positions], sensitivity
            )
        valid = eligible & np.isfinite(correlations)
        if valid.any():
            positions = np.flatnonzero(valid)
            local_counts = counts[positions].astype(np.int32)
            clipped = np.clip(correlations[positions], -0.999999, 0.999999)
            weights = np.maximum(local_counts - 3, 1).astype(float)
            fisher_sum[positions] += np.arctanh(clipped) * weights
            fisher_weight[positions] += weights
            dataset_values[positions] += 1
            selected_finite = finite[:, positions]
            model_columns = np.asarray(
                [model_position[str(model)] for model in models], dtype=int
            )
            model_use_count[np.ix_(positions, model_columns)] += (
                selected_finite.T.astype(np.uint8)
            )
            selected_values = values[:, positions]
            safe_values = np.where(selected_finite, selected_values, 0.0)
            expression_sum[positions] += safe_values.sum(axis=0)
            expression_sum_sq[positions] += (safe_values * safe_values).sum(axis=0)
            expression_n[positions] += selected_finite.sum(axis=0).astype(np.int32)
        processed.append((record, models, finite, valid))

    n_values[:] = (model_use_count > 0).sum(axis=1).astype(np.int32)
    rho[:] = np.tanh(
        np.divide(
            fisher_sum,
            fisher_weight,
            out=np.zeros_like(fisher_sum),
            where=fisher_weight > 0,
        )
    )
    populated = expression_n > 0
    expression_mean = np.divide(
        expression_sum,
        expression_n,
        out=np.zeros_like(expression_sum),
        where=populated,
    )
    expression_variance = np.divide(
        expression_sum_sq,
        expression_n,
        out=np.zeros_like(expression_sum_sq),
        where=populated,
    ) - expression_mean * expression_mean
    expression_sd = np.sqrt(np.maximum(expression_variance, 0.0))

    overlap = (model_use_count > 1).any(axis=1)
    overlap_positions = np.flatnonzero(overlap)
    if len(overlap_positions):
        width = len(overlap_positions)
        consensus_expression_sum = np.zeros(
            (len(union_models), width), dtype=np.float64
        )
        consensus_expression_n = np.zeros(
            (len(union_models), width), dtype=np.uint8
        )
        consensus_sensitivity_sum = np.zeros(
            (len(union_models), width), dtype=np.float64
        )
        consensus_sensitivity_n = np.zeros(
            (len(union_models), width), dtype=np.uint8
        )
        for record, models, finite, valid in processed:
            selected_valid = valid[overlap_positions]
            if not selected_valid.any():
                continue
            values = np.asarray(record.expression, dtype=float)[:, overlap_positions]
            sensitivity = np.asarray(record.sensitivity, dtype=float)
            selected_finite = finite[:, overlap_positions].copy()
            selected_finite[:, ~selected_valid] = False
            percentiles = np.zeros_like(values, dtype=np.float64)
            dense_columns = selected_valid & selected_finite.all(axis=0)
            if dense_columns.any():
                percentiles[:, dense_columns] = _midrank_percentile(
                    sensitivity
                )[:, None]
            for column in np.flatnonzero(selected_valid & ~dense_columns):
                mask = selected_finite[:, column]
                percentiles[mask, column] = _midrank_percentile(
                    sensitivity[mask]
                )
            model_columns = np.asarray(
                [model_position[str(model)] for model in models], dtype=int
            )
            consensus_expression_sum[model_columns, :] += np.where(
                selected_finite, values, 0.0
            )
            consensus_expression_n[model_columns, :] += selected_finite.astype(
                np.uint8
            )
            consensus_sensitivity_sum[model_columns, :] += np.where(
                selected_finite, percentiles, 0.0
            )
            consensus_sensitivity_n[model_columns, :] += selected_finite.astype(
                np.uint8
            )
        consensus_pair = (
            (consensus_expression_n > 0) & (consensus_sensitivity_n > 0)
        )
        consensus_expression = np.divide(
            consensus_expression_sum,
            consensus_expression_n,
            out=np.full_like(consensus_expression_sum, np.nan),
            where=consensus_expression_n > 0,
        )
        consensus_sensitivity = np.divide(
            consensus_sensitivity_sum,
            consensus_sensitivity_n,
            out=np.full_like(consensus_sensitivity_sum, np.nan),
            where=consensus_sensitivity_n > 0,
        )
        consensus_expression[~consensus_pair] = np.nan
        consensus_sensitivity[~consensus_pair] = np.nan
        rho[overlap_positions] = _rank_correlation_pairs(
            consensus_expression, consensus_sensitivity
        )
        consensus_n = consensus_pair.sum(axis=0).astype(np.int32)
        n_values[overlap_positions] = consensus_n
        safe_consensus_expression = np.where(
            consensus_pair, consensus_expression, 0.0
        )
        consensus_mean = np.divide(
            safe_consensus_expression.sum(axis=0),
            consensus_n,
            out=np.zeros(width, dtype=np.float64),
            where=consensus_n > 0,
        )
        consensus_variance = np.divide(
            (safe_consensus_expression * safe_consensus_expression).sum(axis=0),
            consensus_n,
            out=np.zeros(width, dtype=np.float64),
            where=consensus_n > 0,
        ) - consensus_mean * consensus_mean
        expression_mean[overlap_positions] = consensus_mean
        expression_sd[overlap_positions] = np.sqrt(
            np.maximum(consensus_variance, 0.0)
        )

    return CrossDatasetAssociationBatch(
        rho,
        n_values,
        dataset_values,
        expression_mean,
        expression_sd,
        overlap,
    )


def association_statistics(
    candidates: pd.DataFrame,
    expression: pd.DataFrame,
    response: pd.DataFrame,
    mapping: pd.DataFrame,
    selected_models: Sequence[str],
    drug_target_counts: Mapping[str, int],
    drug_target_embedding_available: Mapping[str, bool],
    *,
    min_cell_lines_per_dataset: int,
    min_total_cell_lines: int,
    positive_abs_rho: float,
    negative_abs_rho: float,
) -> AssociationStatistics:
    """Recompute association labels from selected raw cell-line measurements."""

    size = len(candidates)
    domain = np.zeros((size, len(_DOMAIN_FEATURES)), dtype=np.float32)
    labels = np.full(size, np.nan, dtype=np.float32)
    assay_available = np.zeros(size, dtype=bool)
    reasons = np.full(size, "NO_MATCHED_CELL_LINE_ASSAY", dtype=object)
    rho_values = np.full(size, np.nan, dtype=np.float32)
    n_values = np.zeros(size, dtype=np.int32)
    dataset_values = np.zeros(size, dtype=np.int16)
    selected = set(map(str, selected_models))
    local_map = mapping.loc[mapping.canonical_model_id.astype(str).isin(selected)]
    if local_map.empty or candidates.empty:
        return AssociationStatistics(domain, labels, assay_available, reasons, rho_values, n_values, dataset_values)
    canonical_expression = "canonical_model_id" in expression
    if canonical_expression:
        expression_linked = expression.merge(
            local_map[["canonical_model_id", "cancer_id"]].drop_duplicates(),
            on="canonical_model_id", how="inner", validate="many_to_many",
        )
    else:
        expression_linked = expression.merge(
            local_map[["dataset_id", "cell_line_id", "canonical_model_id", "cancer_id"]],
            on=["dataset_id", "cell_line_id"], how="inner", validate="many_to_many",
        )
    response_linked = response.merge(
        local_map[["dataset_id", "cell_line_id", "canonical_model_id", "cancer_id"]],
        on=["dataset_id", "cell_line_id"], how="inner", validate="many_to_many",
    )
    if canonical_expression:
        expression_groups = {
            (str(cancer), str(lncrna)): group[["canonical_model_id", "expression_value"]]
            for (cancer, lncrna), group in expression_linked.groupby(
                ["cancer_id", "lncrna_id"], observed=True, sort=False
            )
        }
    else:
        expression_groups = {
            (str(cancer), str(dataset), str(lncrna)): group[["canonical_model_id", "expression_value"]]
            for (cancer, dataset, lncrna), group in expression_linked.groupby(
                ["cancer_id", "dataset_id", "lncrna_id"], observed=True, sort=False
            )
        }
    response_groups = {
        (str(cancer), str(dataset), str(drug)): group[["canonical_model_id", "sensitivity_value"]]
        for (cancer, dataset, drug), group in response_linked.groupby(
            ["cancer_id", "dataset_id", "drug_id"], observed=True, sort=False
        )
    }
    datasets_by_cancer = local_map.groupby("cancer_id", observed=True).dataset_id.unique().to_dict()
    for row_index, row in enumerate(candidates.itertuples(index=False)):
        records: list[DatasetAssociationArrays] = []
        for dataset in datasets_by_cancer.get(str(row.cancer_id), ()):
            expression_key = (
                (str(row.cancer_id), str(row.lncrna_id))
                if canonical_expression
                else (str(row.cancer_id), str(dataset), str(row.lncrna_id))
            )
            e = expression_groups.get(expression_key)
            r = response_groups.get((str(row.cancer_id), str(dataset), str(row.drug_id)))
            if e is None or r is None:
                continue
            joined = e.merge(r, on="canonical_model_id", how="inner", validate="one_to_one")
            records.append(
                DatasetAssociationArrays(
                    dataset_id=str(dataset),
                    canonical_model_ids=joined.canonical_model_id.astype(str).to_numpy(
                        object
                    ),
                    expression=joined.expression_value.to_numpy(float)[:, None],
                    sensitivity=joined.sensitivity_value.to_numpy(float),
                )
            )
        target_count = int(drug_target_counts.get(str(row.drug_id), 0))
        target_available = bool(drug_target_embedding_available.get(str(row.drug_id), False))
        domain[row_index, 4] = np.log1p(target_count)
        domain[row_index, 5] = float(target_available)
        summary = combine_cross_dataset_associations(
            records,
            candidate_count=1,
            min_cell_lines_per_dataset=int(min_cell_lines_per_dataset),
        )
        if int(summary.dataset_count[0]) == 0:
            reasons[row_index] = "NO_DATASET_WITH_SUFFICIENT_MATCHED_CELL_LINES"
            continue
        total = int(summary.n_cell_lines[0])
        combined = float(summary.rho[0])
        dataset_values[row_index] = int(summary.dataset_count[0])
        n_values[row_index] = total
        domain[row_index, 0] = np.log1p(total)
        domain[row_index, 1] = int(summary.dataset_count[0])
        domain[row_index, 2] = float(summary.expression_mean[0])
        domain[row_index, 3] = float(summary.expression_sd[0])
        if total < int(min_total_cell_lines):
            reasons[row_index] = "INSUFFICIENT_TOTAL_MATCHED_CELL_LINES"
            continue
        if not np.isfinite(combined):
            reasons[row_index] = "CROSS_DATASET_CONSENSUS_UNDEFINED"
            continue
        rho_values[row_index] = combined
        assay_available[row_index] = True
        reasons[row_index] = ""
        magnitude = abs(combined)
        if magnitude >= float(positive_abs_rho):
            labels[row_index] = 1.0
        elif magnitude <= float(negative_abs_rho):
            labels[row_index] = 0.0
    return AssociationStatistics(domain, labels, assay_available, reasons, rho_values, n_values, dataset_values)


def _embedding_lookup(frame: pd.DataFrame, kind: str) -> EmbeddingLookup:
    if "node_id" not in frame:
        raise DrugTrainingError(f"{kind} core embedding lacks node_id")
    features = sorted(column for column in frame.columns if str(column).startswith("core_feature_"))
    if not features:
        raise DrugTrainingError(f"{kind} core embedding lacks core_feature_* columns")
    ids = frame.node_id.astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise DrugTrainingError(f"{kind} core embedding IDs are duplicated")
    values = frame[features].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise DrugTrainingError(f"{kind} core embeddings are not finite")
    return EmbeddingLookup(values=values, index={value: index for index, value in enumerate(ids)})


def _resolve_manifest_path(manifest_path: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() and value.is_file():
        return value.resolve()
    for ancestor in (manifest_path.parent, *manifest_path.parents):
        candidate = ancestor / value
        if candidate.is_file():
            return candidate.resolve()
    raise DrugTrainingError(f"Core embedding export cannot be resolved: {relative}")


def _validate_core_manifest(path: str | Path) -> tuple[dict[str, Any], str, str]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("export_format") != CORE_EXPORT_FORMAT:
        raise DrugTrainingError("Core embeddings are not the registered V3.2 export format")
    if not str(payload.get("analysis_version", "")).startswith("CancerLncAtlas_V3.2"):
        raise DrugTrainingError("Drug parent core is not V3.2")
    if payload.get("all_embeddings_from_newly_trained_v32_core") is not True:
        raise DrugTrainingError("Core lacks the newly-trained V3.2 attestation")
    if payload.get("historical_checkpoint_loaded") is not False:
        raise DrugTrainingError("Core lineage loaded a historical checkpoint")
    if payload.get("historical_prediction_loaded") is not False:
        raise DrugTrainingError("Core lineage loaded historical predictions")
    folds = payload.get("folds")
    if not isinstance(folds, Mapping) or set(map(str, folds)) != set(map(str, range(N_FOLDS))):
        raise DrugTrainingError("Core embedding manifest requires exactly five folds")
    parameter_hashes: list[str] = []
    for fold in range(N_FOLDS):
        item = folds[str(fold)]
        if int(item.get("patient_fold", -1)) != fold:
            raise DrugTrainingError(f"Core embedding fold ID drift: {fold}")
        if item.get("old_checkpoint_loaded") is not False or item.get("trained_from_scratch") is not True:
            raise DrugTrainingError(f"Fold {fold} core is not newly trained from scratch")
        value = str(item.get("core_parameter_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise DrugTrainingError(f"Fold {fold} lacks a core parameter SHA256")
        parameter_hashes.append(value)
    return payload, artifact_sha256(source), _canonical_json_sha256(parameter_hashes)


def load_fold_core_embeddings(
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
    fold: int,
) -> FoldCoreEmbeddings:
    source = Path(manifest_path).resolve()
    item = manifest["folds"][str(fold)]
    exports = item.get("exports", {})
    required = {"lncRNA": "lncrna", "gene": "gene", "cancer": "cancer"}
    if not set(required).issubset(exports):
        raise DrugTrainingError(f"Fold {fold} lacks lncRNA/gene/cancer core exports")
    lookups: dict[str, EmbeddingLookup] = {}
    hashes: dict[str, str] = {}
    width: int | None = None
    for node_type, kind in required.items():
        declaration = exports[node_type]
        path = _resolve_manifest_path(source, str(declaration["path"]))
        observed = artifact_sha256(path)
        if observed != str(declaration.get("sha256", "")):
            raise DrugTrainingError(f"Fold {fold} {kind} core embedding SHA256 mismatch")
        lookup = _embedding_lookup(pd.read_parquet(path), kind)
        width = lookup.values.shape[1] if width is None else width
        if lookup.values.shape[1] != width:
            raise DrugTrainingError("V3.2 core embedding widths differ across node types")
        lookups[kind] = lookup
        hashes[str(path)] = observed
    return FoldCoreEmbeddings(
        fold=fold,
        lncrna=lookups["lncrna"],
        gene=lookups["gene"],
        cancer=lookups["cancer"],
        checkpoint_sha256=str(item["checkpoint_sha256"]),
        core_parameter_sha256=str(item["core_parameter_sha256"]),
        input_hashes=hashes,
    )


def _drug_target_vectors(
    targets: pd.DataFrame, core: FoldCoreEmbeddings
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, bool]]:
    width = core.gene.values.shape[1]
    vectors: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    available: dict[str, bool] = {}
    for drug, local in targets.groupby("drug_id", observed=True, sort=False):
        genes = local.gene_id.astype(str).unique()
        positions = [core.gene.index[gene] for gene in genes if gene in core.gene.index]
        counts[str(drug)] = int(len(genes))
        available[str(drug)] = bool(positions)
        vectors[str(drug)] = (
            core.gene.values[np.asarray(positions, int)].mean(axis=0).astype(np.float32)
            if positions
            else np.zeros(width, dtype=np.float32)
        )
    return vectors, counts, available


def _candidate_core(
    candidates: pd.DataFrame,
    core: FoldCoreEmbeddings,
    drug_vectors: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    width = core.lncrna.values.shape[1]
    result = np.zeros((len(candidates), width * 3), dtype=np.float32)
    available = np.ones(len(candidates), dtype=bool)
    for index, row in enumerate(candidates.itertuples(index=False)):
        lpos = core.lncrna.index.get(str(row.lncrna_id))
        cpos = core.cancer.index.get(str(row.cancer_id))
        if lpos is None or cpos is None:
            available[index] = False
            continue
        result[index, :width] = core.lncrna.values[lpos]
        result[index, width : 2 * width] = core.cancer.values[cpos]
        target = drug_vectors.get(str(row.drug_id))
        if target is not None:
            result[index, 2 * width :] = target
    return result, available


def _balanced_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(labels == value) for value in (0.0, 1.0)]
    if any(len(index) == 0 for index in classes):
        return np.empty(0, dtype=int)
    per_class = max(1, int(maximum) // 2)
    chosen = [rng.choice(index, size=min(len(index), per_class), replace=False) for index in classes]
    result = np.concatenate(chosen)
    rng.shuffle(result)
    return result


def _fit_head(
    train_core: np.ndarray,
    train_domain: np.ndarray,
    train_labels: np.ndarray,
    validation_core: np.ndarray,
    validation_domain: np.ndarray,
    validation_labels: np.ndarray,
    *,
    fold: int,
    config: DrugTrainingConfig,
) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray, list[dict[str, float]]]:
    import torch
    from torch.nn import functional as F

    from .integrated_model import build_private_auxiliary_head

    if set(np.unique(train_labels)) != {0.0, 1.0}:
        raise DrugTrainingError(f"Drug fold {fold} lacks two explicit training classes")
    mean = train_domain.mean(axis=0).astype(np.float32)
    scale = train_domain.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    train_scaled = (train_domain - mean) / scale
    validation_scaled = (validation_domain - mean) / scale if len(validation_domain) else validation_domain
    head_seed = int(config.seed + fold * 101)
    head, initialization = build_private_auxiliary_head(
        MODULE_ID,
        core_features=int(train_core.shape[1]),
        domain_features=int(train_domain.shape[1]),
        hidden_features=int(config.hidden_features),
        dropout=float(config.dropout),
        seed=head_seed,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay)
    )
    generator = torch.Generator().manual_seed(head_seed + 1)
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(int(config.epochs)):
        head.train()
        order = torch.randperm(len(train_labels), generator=generator).numpy()
        losses: list[float] = []
        for start in range(0, len(order), int(config.batch_size)):
            batch = order[start : start + int(config.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = head(
                torch.from_numpy(train_core[batch]), torch.from_numpy(train_scaled[batch])
            ).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(train_labels[batch]))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.no_grad():
            if len(validation_labels):
                validation_logits = head(
                    torch.from_numpy(validation_core), torch.from_numpy(validation_scaled)
                ).squeeze(-1)
                validation_loss = float(
                    F.binary_cross_entropy_with_logits(
                        validation_logits, torch.from_numpy(validation_labels)
                    )
                )
            else:
                validation_loss = float(np.mean(losses))
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(config.patience):
                break
    if best_state is None:
        raise DrugTrainingError(f"Drug fold {fold} produced no trainable checkpoint")
    head.load_state_dict(best_state)
    return head, initialization.as_dict(), mean, scale, history


def _predict_head(
    head: Any,
    core: np.ndarray,
    domain: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    import torch

    scaled = (domain - mean) / scale
    result = np.empty(len(core), dtype=np.float32)
    head.eval()
    with torch.no_grad():
        for start in range(0, len(core), int(batch_size)):
            stop = min(start + int(batch_size), len(core))
            logits = head(torch.from_numpy(core[start:stop]), torch.from_numpy(scaled[start:stop]))
            result[start:stop] = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
    return result


def _split_models(mapping: pd.DataFrame, fold: int, split: str) -> list[str]:
    validation = (fold + 1) % N_FOLDS
    if split == "test":
        selected = mapping.cell_line_fold_id.eq(fold)
    elif split == "validation":
        selected = mapping.cell_line_fold_id.eq(validation)
    elif split == "train":
        selected = ~mapping.cell_line_fold_id.isin([fold, validation])
    else:
        raise ValueError(split)
    return mapping.loc[selected, "canonical_model_id"].astype(str).drop_duplicates().tolist()


def _statistics_for_split(
    candidates: pd.DataFrame,
    expression: pd.DataFrame,
    response: pd.DataFrame,
    mapping: pd.DataFrame,
    models: Sequence[str],
    target_counts: Mapping[str, int],
    target_available: Mapping[str, bool],
    config: DrugTrainingConfig,
) -> AssociationStatistics:
    return association_statistics(
        candidates,
        expression,
        response,
        mapping,
        models,
        target_counts,
        target_available,
        min_cell_lines_per_dataset=config.min_cell_lines_per_dataset,
        min_total_cell_lines=config.min_total_cell_lines,
        positive_abs_rho=config.positive_abs_rho,
        negative_abs_rho=config.negative_abs_rho,
    )


def run_drug_training(
    *,
    exact_candidates_path: str | Path,
    raw_drug_response_path: str | Path,
    raw_lncrna_expression_path: str | Path,
    cell_line_map_path: str | Path,
    drug_gene_target_path: str | Path,
    core_embedding_manifest_path: str | Path,
    output_root: str | Path,
    training_run_id: str,
    drug_candidates_path: str | Path | None = None,
    curated_drug_response_path: str | Path | None = None,
    config: DrugTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train five fresh heads and emit a complete, availability-aware universe."""

    settings = config or DrugTrainingConfig()
    settings.validate()
    if not re.fullmatch(r"v32-[a-z0-9][a-z0-9._-]*", str(training_run_id).strip()):
        raise DrugTrainingError(
            "training_run_id must be lowercase and start with 'v32-'"
        )
    paths = {
        "exact": Path(exact_candidates_path).resolve(),
        "response": Path(raw_drug_response_path).resolve(),
        "expression": Path(raw_lncrna_expression_path).resolve(),
        "mapping": Path(cell_line_map_path).resolve(),
        "targets": Path(drug_gene_target_path).resolve(),
        "core_manifest": Path(core_embedding_manifest_path).resolve(),
    }
    if drug_candidates_path is not None:
        paths["drug_candidates"] = Path(drug_candidates_path).resolve()
    if curated_drug_response_path is not None:
        paths["curated"] = Path(curated_drug_response_path).resolve()
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        if key != "core_manifest":
            _assert_source_path(path, static_annotation=key in {"mapping", "targets", "curated"})

    exact = normalise_exact_candidates(_read_table(paths["exact"]))
    response = normalise_raw_response(_read_table(paths["response"]))
    expression = normalise_raw_expression(_read_table(paths["expression"]))
    mapping = assign_cell_line_folds(normalise_cell_line_map(_read_table(paths["mapping"])), settings.seed)
    targets = normalise_drug_targets(_read_table(paths["targets"]))
    curated = (
        normalise_curated_response(_read_table(paths["curated"]))
        if "curated" in paths else pd.DataFrame(columns=[*TARGET_KEYS, "pmid", "source_database"])
    )
    candidates = (
        normalise_drug_candidates(_read_table(paths["drug_candidates"]), exact)
        if "drug_candidates" in paths
        else build_drug_candidate_universe(exact, response, mapping, curated)
    )
    if candidates.empty:
        raise DrugTrainingError("The V3.2 drug candidate universe is empty")

    source_specs: list[dict[str, Any]] = [
        {"path": paths["exact"], "generation": "V3.2", "source_role": "standardized_input", "outcome_derived": False, "fold_fitted": False, "use_role": "aux_input", "artifact_id": "v32_exact_candidate_universe"},
        {"path": paths["response"], "generation": "raw_source", "source_role": "raw_data", "outcome_derived": True, "fold_fitted": False, "use_role": "training_target", "artifact_id": "raw_cell_line_drug_response"},
        {"path": paths["expression"], "generation": "raw_source", "source_role": "raw_data", "outcome_derived": False, "fold_fitted": False, "use_role": "aux_input", "artifact_id": "raw_cell_line_lncrna_expression"},
        {"path": paths["mapping"], "generation": "historical_static", "source_role": "static_annotation", "outcome_derived": False, "fold_fitted": False, "use_role": "split_control", "artifact_id": "cell_line_cancer_mapping"},
        {"path": paths["targets"], "generation": "historical_static", "source_role": "static_annotation", "outcome_derived": False, "fold_fitted": False, "use_role": "aux_input", "artifact_id": "curated_drug_gene_targets"},
    ]
    if "drug_candidates" in paths:
        source_specs.append({"path": paths["drug_candidates"], "generation": "V3.2", "source_role": "standardized_input", "outcome_derived": False, "fold_fitted": False, "use_role": "aux_input", "artifact_id": "v32_drug_candidate_universe"})
    if "curated" in paths:
        source_specs.append({"path": paths["curated"], "generation": "historical_curated_observation", "source_role": "raw_data", "outcome_derived": True, "fold_fitted": False, "use_role": "training_target", "artifact_id": "curated_lncrna_drug_response_observations"})
    source_audit = audit_input_lineage(source_specs)
    if source_audit["status"] != "PASS":
        failures = {row["artifact_id"]: row["reasons"] for row in source_audit["artifacts"] if row["status"] != "PASS"}
        raise DrugTrainingError(f"V3.2 drug input lineage rejected: {failures}")

    core_manifest, core_manifest_sha, core_parameter_composite = _validate_core_manifest(paths["core_manifest"])
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mapping[["dataset_id", "cell_line_id", "canonical_model_id", "cancer_id", "cell_line_fold_id"]].to_parquet(
        output / "CELL_LINE_FOLD_MANIFEST.parquet", index=False
    )
    probability_sum = np.zeros(len(candidates), dtype=np.float64)
    probability_count = np.zeros(len(candidates), dtype=np.int16)
    checkpoint_rows: list[dict[str, Any]] = []
    fold_status: list[dict[str, Any]] = []
    evaluation_parts: list[pd.DataFrame] = []
    frozen_hashes: dict[str, str] = {}

    for fold in range(N_FOLDS):
        core = load_fold_core_embeddings(paths["core_manifest"], core_manifest, fold)
        for path_text, digest in core.input_hashes.items():
            if path_text in frozen_hashes and frozen_hashes[path_text] != digest:
                raise DrugTrainingError("Frozen core embedding hash drifted between folds")
            frozen_hashes[path_text] = digest
        drug_vectors, target_counts, target_available = _drug_target_vectors(targets, core)
        candidate_core, core_available = _candidate_core(candidates, core, drug_vectors)
        split_stats = {
            split: _statistics_for_split(
                candidates, expression, response, mapping, _split_models(mapping, fold, split),
                target_counts, target_available, settings,
            )
            for split in ("train", "validation", "test")
        }
        train_mask = np.isfinite(split_stats["train"].labels) & core_available
        validation_mask = np.isfinite(split_stats["validation"].labels) & core_available
        train_positions = np.flatnonzero(train_mask)
        validation_positions = np.flatnonzero(validation_mask)
        if len(train_positions):
            local = _balanced_indices(
                split_stats["train"].labels[train_positions], settings.max_train_rows, settings.seed + fold
            )
            train_positions = train_positions[local]
        if len(validation_positions) > settings.max_validation_rows:
            rng = np.random.default_rng(settings.seed + fold + 50_000)
            validation_positions = rng.choice(validation_positions, settings.max_validation_rows, replace=False)
        if not len(train_positions) or set(np.unique(split_stats["train"].labels[train_positions])) != {0.0, 1.0}:
            fold_status.append({"cell_line_fold": fold, "status": "UNAVAILABLE", "reason": "TWO_TRAINING_CLASSES_NOT_AVAILABLE"})
            continue
        head, initialization, mean, scale, history = _fit_head(
            candidate_core[train_positions], split_stats["train"].domain[train_positions], split_stats["train"].labels[train_positions],
            candidate_core[validation_positions], split_stats["validation"].domain[validation_positions], split_stats["validation"].labels[validation_positions],
            fold=fold, config=settings,
        )
        import torch

        checkpoint_path = output / f"cell_line_fold={fold}" / "private_head.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_format": PRIVATE_CHECKPOINT_FORMAT,
                "analysis_version": ANALYSIS_VERSION,
                "module_id": MODULE_ID,
                "cell_line_fold": fold,
                "model_state": head.state_dict(),
                "initialization": initialization,
                "domain_features": list(_DOMAIN_FEATURES),
                "domain_mean": mean,
                "domain_scale": scale,
                "history": history,
                "core_checkpoint_sha256": core.checkpoint_sha256,
                "core_parameter_sha256": core.core_parameter_sha256,
                "old_checkpoint_loaded": False,
                "old_predictions_used": False,
            },
            checkpoint_path,
        )
        checkpoint_sha = artifact_sha256(checkpoint_path)
        checkpoint_rows.append(
            {
                "cell_line_fold": fold,
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "initial_parameter_sha256": initialization["initial_parameter_sha256"],
                "source_checkpoint_sha256": None,
                "core_checkpoint_sha256": core.checkpoint_sha256,
                "core_parameter_sha256": core.core_parameter_sha256,
                "train_rows": int(len(train_positions)),
                "validation_rows": int(len(validation_positions)),
                "status": "SUCCESS",
            }
        )
        fold_status.append({"cell_line_fold": fold, "status": "SUCCESS", "train_rows": int(len(train_positions)), "validation_rows": int(len(validation_positions))})
        test_stats = split_stats["test"]
        predict_mask = test_stats.assay_available & core_available
        if predict_mask.any():
            probability = _predict_head(
                head, candidate_core[predict_mask], test_stats.domain[predict_mask], mean, scale,
                settings.prediction_batch_size,
            )
            positions = np.flatnonzero(predict_mask)
            probability_sum[positions] += probability
            probability_count[positions] += 1
            evaluation_parts.append(
                candidates.iloc[positions].assign(
                    cell_line_fold=fold,
                    association_proxy_label=test_stats.labels[positions],
                    held_out_rho=test_stats.rho[positions],
                    held_out_n_cell_lines=test_stats.n_cell_lines[positions],
                    held_out_probability=probability,
                )
            )

    if settings.require_all_folds and len(checkpoint_rows) != N_FOLDS:
        raise DrugTrainingError(f"Fresh V3.2 drug training requires five successful heads; observed={len(checkpoint_rows)}")
    if not checkpoint_rows:
        raise DrugTrainingError("No fresh V3.2 drug private head was trainable")
    for path_text, before in frozen_hashes.items():
        if artifact_sha256(path_text) != before:
            raise DrugTrainingError(f"Frozen V3.2 core embedding changed during drug training: {path_text}")
    if artifact_sha256(paths["core_manifest"]) != core_manifest_sha:
        raise DrugTrainingError("Core embedding manifest changed during drug training")

    probability = np.divide(
        probability_sum, probability_count,
        out=np.full(len(candidates), np.nan, dtype=float), where=probability_count > 0,
    )
    # Full-data statistics are used only to explain unavailable rows.  Their
    # correlation labels/rho never enter the public table or a trained head.
    reference_core = load_fold_core_embeddings(paths["core_manifest"], core_manifest, 0)
    reference_vectors, reference_counts, reference_available = _drug_target_vectors(targets, reference_core)
    _, reference_core_available = _candidate_core(candidates, reference_core, reference_vectors)
    full_stats = _statistics_for_split(
        candidates, expression, response, mapping, mapping.canonical_model_id.unique(),
        reference_counts, reference_available, settings,
    )
    reason = full_stats.reasons.astype(object).copy()
    reason[~reference_core_available] = "V32_CORE_LNCRNA_OR_CANCER_EMBEDDING_UNAVAILABLE"
    reason[(probability_count == 0) & (reason == "")] = "NO_HELD_OUT_FOLD_PREDICTION"
    reason[probability_count > 0] = ""
    public = candidates.copy()
    public["drug_response_association_probability"] = probability
    public["availability"] = probability_count > 0
    public["failure_reason"] = pd.Series(reason).replace("", pd.NA)
    public["cell_line_folds_with_prediction"] = probability_count.astype(int)
    public["evidence_scope"] = "CELL_LINE_ASSOCIATION_NOT_TCGA_PATIENT_RESPONSE"
    public["cell_line_response_only"] = True
    public["tcga_patient_response_claimed"] = False
    public["scientific_status"] = "diagnostic_only"
    public["changes_primary_ranking"] = False
    public["analysis_version"] = ANALYSIS_VERSION
    public["training_run_id"] = str(training_run_id)
    public["module_id"] = MODULE_ID
    public["target_level"] = TARGET_LEVEL
    public["prediction_format"] = PREDICTION_FORMAT
    if not curated.empty:
        curated_count = curated.groupby(list(TARGET_KEYS), observed=True).size().rename("curated_evidence_count").reset_index()
        public = public.merge(curated_count, on=list(TARGET_KEYS), how="left", validate="one_to_one")
        public["curated_evidence_count"] = public.curated_evidence_count.fillna(0).astype(int)
    else:
        public["curated_evidence_count"] = 0
    public["curated_evidence_used_as_model_feature"] = False
    if len(public) != len(candidates) or public[list(TARGET_KEYS)].duplicated().any():
        raise DrugTrainingError("Public drug output changed its complete candidate universe")
    finite = pd.to_numeric(public.drug_response_association_probability, errors="coerce").dropna()
    if not finite.between(0, 1).all():
        raise DrugTrainingError("Drug association probabilities are outside [0, 1]")
    if public.loc[~public.availability, "drug_response_association_probability"].notna().any():
        raise DrugTrainingError("Unavailable drug rows must have null probability")

    prediction_path = output / "drug_response_association.parquet"
    _atomic_parquet(public, prediction_path)
    private_evaluation_path = output / "drug_oof_evaluation.PRIVATE.parquet"
    private_evaluation = pd.concat(evaluation_parts, ignore_index=True) if evaluation_parts else pd.DataFrame(
        columns=[*TARGET_KEYS, "cell_line_fold", "association_proxy_label", "held_out_rho", "held_out_n_cell_lines", "held_out_probability"]
    )
    _atomic_parquet(private_evaluation, private_evaluation_path)
    checkpoint_manifest = {
        "checkpoint_format": PRIVATE_CHECKPOINT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "folds": N_FOLDS,
        "records": checkpoint_rows,
        "fold_status": fold_status,
        "all_private_heads_random_initialization": all(row["source_checkpoint_sha256"] is None for row in checkpoint_rows),
        "core_detached_and_frozen": True,
    }
    checkpoint_manifest_path = output / "CHECKPOINT_MANIFEST.json"
    _atomic_json(checkpoint_manifest_path, checkpoint_manifest)
    config_path = output / "RUN_CONFIG.json"
    _atomic_json(
        config_path,
        {
            "analysis_version": ANALYSIS_VERSION,
            "training_run_id": training_run_id,
            "training": asdict(settings),
            "estimand": "held_out_cell_line_lncrna_expression_x_drug_sensitivity_association",
            "patient_response_estimand": False,
            "old_association_tables_allowed": False,
            "curated_evidence_used_as_model_feature": False,
        },
    )
    _atomic_json(output / "INPUT_LINEAGE_AUDIT.json", source_audit)
    input_artifacts = [
        {
            "path": row["path"],
            "sha256": row["sha256"],
            "artifact_kind": (
                "raw_data" if row["source_role"] == "raw_data"
                else "annotation" if row["source_role"] == "static_annotation"
                else "split_manifest" if row["source_role"] == "split_manifest"
                else "standardized_input"
            ),
            "generation": row["generation"],
            "source_role": row["source_role"],
            "outcome_derived": row["outcome_derived"],
            "fold_fitted": row["fold_fitted"],
        }
        for row in source_audit["artifacts"]
    ]
    input_artifacts.append(
        {
            "path": str(paths["core_manifest"]),
            "sha256": core_manifest_sha,
            "artifact_kind": "v32_core_checkpoint",
            "generation": "V3.2",
            "source_role": "v32_core_checkpoint",
            "outcome_derived": True,
            "fold_fitted": True,
            "use_role": "aux_parent",
        }
    )
    runner_path = Path(__file__).resolve().parents[2] / "scripts" / "run_v32_drug_training.py"
    code_hashes = {"drug_training.py": artifact_sha256(Path(__file__).resolve())}
    if runner_path.is_file():
        code_hashes["run_v32_drug_training.py"] = artifact_sha256(runner_path)
    lineage = {
        "module_id": MODULE_ID,
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": str(training_run_id),
        "training_status": "SUCCESS",
        "initialization_policy": "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        "folds": N_FOLDS,
        "seeds": [int(settings.seed + fold * 101) for fold in range(N_FOLDS)],
        "code_sha256": _canonical_json_sha256(code_hashes),
        "config_sha256": artifact_sha256(config_path),
        "input_manifest_sha256": source_audit["lineage_sha256"],
        "checkpoint_manifest_sha256": artifact_sha256(checkpoint_manifest_path),
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "old_gdsc_prism_association_tables_used": False,
        "private_head_trained_from_scratch": len(checkpoint_rows) == N_FOLDS,
        "core_parameters_frozen": True,
        "v32_core_checkpoint_sha256": core_manifest_sha,
        "core_parameters_before_sha256": core_parameter_composite,
        "core_parameters_after_sha256": core_parameter_composite,
        "input_artifacts": input_artifacts,
        "fold_status": fold_status,
        "split_unit": "canonical_cell_line_model",
        "cell_line_response_never_claimed_as_patient_response": True,
        "scientific_status": "diagnostic_only",
        "changes_primary_ranking": False,
        "formal_release_eligible": False,
        "dense_development_only": True,
        "release_ready": False,
        "partial_not_publishable": True,
        "nonreleaseable_reason": (
            "LEGACY_DENSE_RUNNER_LACKS_HASH_BOUND_TYPED_SPARSE_ABSENCE_RESOLVER"
        ),
        "prediction_path": str(prediction_path),
        "prediction_sha256": artifact_sha256(prediction_path),
        "prediction_rows": int(len(public)),
        "available_rows": int(public.availability.sum()),
        "unavailable_rows": int((~public.availability).sum()),
        "private_evaluation_path": str(private_evaluation_path),
        "private_evaluation_sha256": artifact_sha256(private_evaluation_path),
    }
    from .full_model_contract import validate_module_lineage, validate_public_module_frame

    validate_module_lineage(MODULE_ID, lineage)
    validate_public_module_frame(MODULE_ID, public)
    lineage_path = output / "MODULE_LINEAGE.json"
    _atomic_json(lineage_path, lineage)
    success = {
        "status": "DEVELOPMENT_SUCCESS_NOT_RELEASEABLE",
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "training_run_id": str(training_run_id),
        "prediction_path": str(prediction_path),
        "prediction_sha256": lineage["prediction_sha256"],
        "lineage_path": str(lineage_path),
        "lineage_sha256": artifact_sha256(lineage_path),
        "candidate_rows_preserved": int(len(public)),
        "available_rows": lineage["available_rows"],
        "scientific_status": "diagnostic_only",
        "tcga_patient_response_claimed": False,
        "formal_release_eligible": False,
        "release_ready": False,
        "partial_not_publishable": True,
    }
    _atomic_json(output / "DEVELOPMENT_SUCCESS.json", success)
    return success


__all__ = [
    "ANALYSIS_VERSION",
    "CORE_EXPORT_FORMAT",
    "DrugTrainingConfig",
    "DrugTrainingError",
    "association_statistics",
    "assign_cell_line_folds",
    "build_drug_candidate_universe",
    "normalise_cell_line_map",
    "normalise_drug_candidates",
    "normalise_drug_targets",
    "normalise_raw_expression",
    "normalise_raw_response",
    "run_drug_training",
]
