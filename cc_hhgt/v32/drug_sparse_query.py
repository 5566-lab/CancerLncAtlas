"""Fail-closed resolver for sparse V3.2 Drug public predictions.

The Drug conceptual universe is a factored relation::

    exact(cancer_id, lncRNA_id, pathway_id)
      x pathway_drug(pathway_id, drug_id)

It is deliberately *not* materialised.  Public predictions contain only keys
for which a newly trained V3.2 head had a valid held-out cell-line assay,
matched lncRNA expression and current V3.2 core entities.  This module binds
the sparse result and the compact factors into one hash-registered bundle and
resolves an absent key to a precise machine-readable reason; absence never
means probability zero.

Training integration is intentionally a post-training call and does not make
this query layer depend on private labels or checkpoints.  The streaming
trainer already owns ``fold_eligible``, ``expression``, ``mapping``, ``cores``
and ``drug_features``.  It can call, in order::

    expression_coverage = factor_expression_fold_coverage(...)
    core_availability = factor_core_entity_availability(...)
    write_drug_sparse_query_sidecars(
        bundle_root=output,
        exact_candidates=paths["exact"],
        pathway_drug_edges=edge_path,
        assay_fold_eligibility=fold_eligible,
        expression_fold_coverage=expression_coverage,
        core_entity_availability=core_availability,
        available_predictions_path=prediction_root,
        conceptual_candidate_rows=conceptual_rows,
        training_run_id=training_run_id,
    )

Only the two input factors and compact eligibility tables are copied into the
bundle.  The potentially tens-of-millions-row candidate join is never built.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Iterator, Literal, TypedDict

import numpy as np
import pandas as pd

from .input_lineage import artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
MODULE_ID = "drug"
MANIFEST_FORMAT = "CC_HHGT_V3_2_DRUG_SPARSE_QUERY_BUNDLE_V1"
PROBABILITY_COLUMN = "drug_response_association_probability"
MANIFEST_NAME = "DRUG_SPARSE_QUERY_MANIFEST.json"
VALIDATION_AUDIT_NAME = "SPARSE_QUERY_VALIDATION_AUDIT.json"
FACTOR_DIRECTORY = "drug_sparse_query_factors"
FOLDS = tuple(range(5))
FOLD_MASK_MIN = 1
FOLD_MASK_MAX = sum(1 << fold for fold in FOLDS)

# Every DuckDB connection in this module is bounded.  Formal runners can tune
# the bounds without weakening semantic validation by setting these three
# environment variables.  A unique spill directory is created below the
# configured base and removed after the connection closes.
DUCKDB_MEMORY_ENV = "CC_HHGT_DRUG_SPARSE_DUCKDB_MEMORY_LIMIT"
DUCKDB_THREADS_ENV = "CC_HHGT_DRUG_SPARSE_DUCKDB_THREADS"
DUCKDB_TEMP_ENV = "CC_HHGT_DRUG_SPARSE_DUCKDB_TEMP_DIRECTORY"
DUCKDB_MAX_TEMP_ENV = (
    "CC_HHGT_DRUG_SPARSE_DUCKDB_MAX_TEMP_DIRECTORY_SIZE"
)
DEFAULT_DUCKDB_MEMORY_LIMIT = "1GB"
DEFAULT_DUCKDB_THREADS = 2
DEFAULT_DUCKDB_MAX_TEMP_SIZE = "4294967296B"
MIN_DUCKDB_MAX_TEMP_BYTES = 256 * 1024 * 1024
STRICT_VALIDATION_STRATEGY = (
    "FULL_CANCER_AND_PHYSICAL_PARTITION_STREAMING_NO_SAMPLE"
)

OUTSIDE_CONCEPTUAL_UNIVERSE = "OUTSIDE_CONCEPTUAL_UNIVERSE"
NO_HELD_OUT_NATIVE_ASSAY = "NO_HELD_OUT_NATIVE_ASSAY"
NO_MATCHED_LNCRNA_EXPRESSION = "NO_MATCHED_LNCRNA_EXPRESSION"
CURRENT_V32_CORE_UNAVAILABLE = "CURRENT_V32_CORE_UNAVAILABLE"
MODEL_RUN_AUDITED_UNAVAILABLE = "MODEL_RUN_AUDITED_UNAVAILABLE"
MODEL_OUTPUT_MISSING_FAIL_CLOSED = "MODEL_OUTPUT_MISSING_FAIL_CLOSED"

REASON_PRECEDENCE = (
    OUTSIDE_CONCEPTUAL_UNIVERSE,
    MODEL_RUN_AUDITED_UNAVAILABLE,
    NO_HELD_OUT_NATIVE_ASSAY,
    NO_MATCHED_LNCRNA_EXPRESSION,
    CURRENT_V32_CORE_UNAVAILABLE,
    MODEL_OUTPUT_MISSING_FAIL_CLOSED,
)

DrugFailureReason = Literal[
    "OUTSIDE_CONCEPTUAL_UNIVERSE",
    "NO_HELD_OUT_NATIVE_ASSAY",
    "NO_MATCHED_LNCRNA_EXPRESSION",
    "CURRENT_V32_CORE_UNAVAILABLE",
    "MODEL_RUN_AUDITED_UNAVAILABLE",
    "MODEL_OUTPUT_MISSING_FAIL_CLOSED",
]


class DrugSparseResolution(TypedDict):
    """Stable public result shape returned for both available and null keys."""

    cancer_id: str
    lncrna_id: str
    drug_id: str
    drug_response_association_probability: float | None
    availability: bool
    failure_reason: DrugFailureReason | None
    failure_detail: str | None
    analysis_version: str
    module_id: str
    scientific_status: str
    evidence_scope: str
    tcga_patient_response_claimed: bool
    manifest_sha256: str

_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "exact_candidates": ("cancer_id", "lncrna_id", "pathway_id"),
    "pathway_drug_edges": ("pathway_id", "drug_id"),
    "assay_fold_eligibility": ("cancer_id", "drug_id", "fold_id"),
    "expression_fold_coverage": ("cancer_id", "lncrna_id", "fold_id"),
    "core_entity_availability": ("fold_id", "entity_type", "entity_id"),
    "available_predictions": (
        "cancer_id",
        "lncrna_id",
        "drug_id",
        PROBABILITY_COLUMN,
        "availability",
        "prediction_fold_mask",
        "cell_line_folds_with_prediction",
    ),
}
_REQUIRED_ARTIFACTS = frozenset(_REQUIRED_COLUMNS)
_CORE_TYPES = frozenset({"cancer", "lncRNA", "drug_target"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


class DrugSparseQueryError(RuntimeError):
    """Base error for a sparse Drug query or bundle."""


class DrugSparseAssetError(DrugSparseQueryError):
    """A bundle is missing, mutable, escaped or semantically invalid."""


class DrugSparseInputError(DrugSparseQueryError):
    """A query key is empty or otherwise invalid."""


def _is_null_scalar(value: Any) -> bool:
    """Return a strict scalar-null decision before any string coercion."""

    try:
        observed = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(observed) if isinstance(observed, (bool, np.bool_)) else False


def _require_non_null_series(frame: pd.DataFrame, columns: Iterable[str], role: str) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise DrugSparseAssetError(f"{role} contains null key: {column}")


@contextmanager
def _bounded_duckdb() -> Iterator[Any]:
    """Yield an in-memory DuckDB connection with bounded RAM, CPU and spill."""

    import duckdb

    memory_limit = str(os.environ.get(DUCKDB_MEMORY_ENV, DEFAULT_DUCKDB_MEMORY_LIMIT)).strip()
    if not memory_limit:
        raise DrugSparseAssetError(f"{DUCKDB_MEMORY_ENV} cannot be empty")
    try:
        threads = int(os.environ.get(DUCKDB_THREADS_ENV, DEFAULT_DUCKDB_THREADS))
    except (TypeError, ValueError) as exc:
        raise DrugSparseAssetError(f"{DUCKDB_THREADS_ENV} must be an integer") from exc
    if not 1 <= threads <= 32:
        raise DrugSparseAssetError(f"{DUCKDB_THREADS_ENV} must be within 1..32")
    max_temp_size = str(
        os.environ.get(DUCKDB_MAX_TEMP_ENV, DEFAULT_DUCKDB_MAX_TEMP_SIZE)
    ).strip()
    if not max_temp_size:
        raise DrugSparseAssetError(f"{DUCKDB_MAX_TEMP_ENV} cannot be empty")
    size_match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB|GB|TB)",
        max_temp_size,
        flags=re.IGNORECASE,
    )
    if size_match is None:
        raise DrugSparseAssetError(
            f"{DUCKDB_MAX_TEMP_ENV} must be a byte size such as 4GB"
        )
    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }
    max_temp_bytes = float(size_match.group(1)) * units[size_match.group(2).upper()]
    if not math.isfinite(max_temp_bytes) or max_temp_bytes < MIN_DUCKDB_MAX_TEMP_BYTES:
        raise DrugSparseAssetError(
            f"{DUCKDB_MAX_TEMP_ENV} must be at least 256MB"
        )
    configured_base = os.environ.get(DUCKDB_TEMP_ENV)
    if configured_base:
        spill_base = Path(configured_base).resolve()
        spill_base.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="cc_hhgt_drug_sparse_", dir=str(spill_base)
        )
    else:
        temporary = tempfile.TemporaryDirectory(prefix="cc_hhgt_drug_sparse_")
    try:
        connection = duckdb.connect(
            database=":memory:",
            config={
                "memory_limit": memory_limit,
                "threads": str(threads),
                "temp_directory": temporary.name,
                "max_temp_directory_size": max_temp_size,
                "preserve_insertion_order": "false",
            },
        )
        try:
            yield connection
        finally:
            connection.close()
    finally:
        temporary.cleanup()


def _canonical_lnc(value: Any) -> str:
    if _is_null_scalar(value):
        raise DrugSparseAssetError("lncRNA identifier cannot be null")
    text = str(value).strip().upper()
    if text.startswith("GENE:"):
        text = text[5:]
    if not text.startswith("LNC:"):
        text = "LNC:" + text
    prefix, identifier = text.split(":", 1)
    if re.fullmatch(r"ENSG\d+\.\d+", identifier):
        identifier = identifier.split(".", 1)[0]
    return prefix + ":" + identifier


def _canonical_cancer(value: Any) -> str:
    if _is_null_scalar(value):
        raise DrugSparseAssetError("cancer identifier cannot be null")
    return str(value).strip().upper()


def _canonical_drug(value: Any) -> str:
    # Canonical hashed Drug IDs intentionally contain lower-case hex.  Do not
    # case-fold them here; staging is the authority for the exact identifier.
    if _is_null_scalar(value):
        raise DrugSparseAssetError("drug identifier cannot be null")
    return str(value).strip()


def _nonempty(value: Any, label: str) -> str:
    if _is_null_scalar(value):
        raise DrugSparseInputError(f"{label} must be a non-empty canonical identifier")
    text = str(value).strip()
    if not text or text.casefold() in {"none", "nan", "<na>"}:
        raise DrugSparseInputError(f"{label} must be a non-empty canonical identifier")
    return text


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


def _sql_literal(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".parquet", ".pq"}:
            raise DrugSparseAssetError(f"Sparse query artifact is not Parquet: {path}")
        return [path]
    if not path.is_dir():
        raise DrugSparseAssetError(f"Sparse query artifact is missing: {path}")
    files = sorted(item for item in path.rglob("*.parquet") if item.is_file())
    other = [item for item in path.rglob("*") if item.is_file() and item.suffix.lower() != ".parquet"]
    if other:
        raise DrugSparseAssetError(
            f"Sparse Parquet dataset contains non-Parquet files: {other[0]}"
        )
    if not files:
        raise DrugSparseAssetError(f"Sparse Parquet dataset is empty: {path}")
    return files


def _parquet_relation(path: Path) -> str:
    if path.is_file():
        source = path.resolve().as_posix()
    else:
        source = (path.resolve().as_posix().rstrip("/") + "/**/*.parquet")
    return f"read_parquet('{_sql_literal(source)}', union_by_name=true)"


def _parquet_file_relation(paths: Iterable[Path]) -> str:
    """Build one relation from an explicit, already-audited file list."""

    files = [Path(path).resolve() for path in paths]
    if not files:
        raise DrugSparseAssetError("Sparse Parquet file relation cannot be empty")
    literals = ",".join(
        "'" + _sql_literal(path.as_posix()) + "'" for path in files
    )
    return f"read_parquet([{literals}], union_by_name=true)"


def _available_partition_groups(path: Path) -> dict[str, list[Path]]:
    """Return every available file grouped by every cancer it actually holds."""

    if path.is_file():
        # Writer fixtures may use a single file.  The caller can retain the
        # legacy all-file proof for that bounded case.
        return {}
    groups: dict[str, list[Path]] = {}
    pattern = re.compile(r"^cancer-([A-Z0-9_-]{2,16})/part-[0-9]+\.parquet$")
    import pyarrow.parquet as pq

    for item in _parquet_files(path):
        relative = item.relative_to(path).as_posix()
        match = pattern.fullmatch(relative)
        if match is None:
            raise DrugSparseAssetError(
                f"Available Drug partition path is non-canonical: {relative}"
            )
        # Some valid writer fixtures co-locate more than one cancer in one
        # physical part.  Group by the data, not merely the directory label,
        # while still requiring the canonical physical layout.  Reading this
        # one low-cardinality column is a complete partition census.
        cancers = {
            str(value)
            for value in pq.read_table(item, columns=["cancer_id"])
            .column("cancer_id")
            .unique()
            .to_pylist()
            if value is not None
        }
        if not cancers:
            cancers = {match.group(1)}
        for cancer_id in cancers:
            groups.setdefault(cancer_id, []).append(item)
    return {cancer: sorted(files) for cancer, files in sorted(groups.items())}


def _parquet_rows_and_columns(path: Path) -> tuple[int, set[str]]:
    import pyarrow.parquet as pq

    files = _parquet_files(path)
    rows = 0
    columns: set[str] | None = None
    for item in files:
        metadata = pq.ParquetFile(item).metadata
        rows += int(metadata.num_rows)
        observed = set(pq.read_schema(item).names)
        columns = observed if columns is None else columns.intersection(observed)
    return rows, columns or set()


def _assert_no_symlink(path: Path, root: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise DrugSparseAssetError(f"Symlink is forbidden in sparse query bundle: {current}")
        if current == root:
            break
        if root not in current.parents:
            raise DrugSparseAssetError(f"Artifact escaped sparse query bundle: {path}")
        current = current.parent
    if path.is_dir():
        for item in path.rglob("*"):
            if item.is_symlink():
                raise DrugSparseAssetError(
                    f"Symlink is forbidden in sparse query bundle: {item}"
                )


def _normalise_fold(frame: pd.DataFrame, role: str) -> pd.Series:
    source = next(
        (name for name in ("fold_id", "cell_line_fold_id", "patient_fold") if name in frame),
        None,
    )
    if source is None:
        raise DrugSparseAssetError(f"{role} lacks fold_id/cell_line_fold_id")
    raw = frame[source]
    if raw.isna().any():
        raise DrugSparseAssetError(f"{role} contains null fold IDs")
    if pd.api.types.is_bool_dtype(raw.dtype) or raw.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).any():
        raise DrugSparseAssetError(f"{role} contains non-integer fold IDs")
    try:
        values = pd.to_numeric(raw, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DrugSparseAssetError(f"{role} contains non-integer fold IDs") from exc
    # Validate in a wide numeric representation before narrowing.  In
    # particular, 256 must never wrap to zero through int8 conversion.
    wide = values.astype("float64")
    if not np.isfinite(wide).all() or not np.equal(wide, np.floor(wide)).all():
        raise DrugSparseAssetError(f"{role} contains non-integer fold IDs")
    if not wide.isin(FOLDS).all():
        raise DrugSparseAssetError(f"{role} contains folds outside 0..4")
    return wide.astype("int64")


def _eligible_rows(frame: pd.DataFrame, aliases: tuple[str, ...], role: str) -> pd.DataFrame:
    flag = next((name for name in aliases if name in frame), None)
    if flag is None:
        return frame
    values = frame[flag]
    if values.isna().any():
        raise DrugSparseAssetError(f"{role} eligibility contains null")
    if values.dtype != bool:
        normal = values.astype(str).str.strip().str.casefold()
        if not normal.isin({"true", "false", "1", "0"}).all():
            raise DrugSparseAssetError(f"{role} eligibility is not boolean")
        values = normal.isin({"true", "1"})
    return frame.loc[np.asarray(values, dtype=bool)].copy()


def _normalise_factor_frame(role: str, source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    required = _REQUIRED_COLUMNS[role]
    if role == "exact_candidates":
        if not set(required).issubset(frame):
            raise DrugSparseAssetError(f"{role} lacks {sorted(set(required) - set(frame))}")
        _require_non_null_series(frame, required, role)
        result = pd.DataFrame(
            {
                "cancer_id": frame.cancer_id.map(_canonical_cancer),
                "lncrna_id": frame.lncrna_id.map(_canonical_lnc),
                "pathway_id": frame.pathway_id.astype(str).str.strip(),
            }
        )
    elif role == "pathway_drug_edges":
        if not set(required).issubset(frame):
            raise DrugSparseAssetError(f"{role} lacks {sorted(set(required) - set(frame))}")
        _require_non_null_series(frame, required, role)
        result = pd.DataFrame(
            {
                "pathway_id": frame.pathway_id.astype(str).str.strip(),
                "drug_id": frame.drug_id.map(_canonical_drug),
            }
        )
    elif role == "assay_fold_eligibility":
        frame = _eligible_rows(frame, ("assay_eligible", "availability", "eligible"), role)
        if not {"cancer_id", "drug_id"}.issubset(frame):
            raise DrugSparseAssetError(f"{role} lacks cancer_id/drug_id")
        _require_non_null_series(frame, ("cancer_id", "drug_id"), role)
        result = pd.DataFrame(
            {
                "cancer_id": frame.cancer_id.map(_canonical_cancer),
                "drug_id": frame.drug_id.map(_canonical_drug),
                "fold_id": _normalise_fold(frame, role),
            }
        )
    elif role == "expression_fold_coverage":
        frame = _eligible_rows(
            frame, ("expression_available", "availability", "eligible"), role
        )
        if not {"cancer_id", "lncrna_id"}.issubset(frame):
            raise DrugSparseAssetError(f"{role} lacks cancer_id/lncrna_id")
        _require_non_null_series(frame, ("cancer_id", "lncrna_id"), role)
        result = pd.DataFrame(
            {
                "cancer_id": frame.cancer_id.map(_canonical_cancer),
                "lncrna_id": frame.lncrna_id.map(_canonical_lnc),
                "fold_id": _normalise_fold(frame, role),
            }
        )
    elif role == "core_entity_availability":
        frame = _eligible_rows(frame, ("core_available", "availability", "eligible"), role)
        if not {"entity_type", "entity_id"}.issubset(frame):
            raise DrugSparseAssetError(f"{role} lacks entity_type/entity_id")
        _require_non_null_series(frame, ("entity_type", "entity_id"), role)
        aliases = {
            "cancer": "cancer",
            "lncrna": "lncRNA",
            "lnc_rna": "lncRNA",
            "drug": "drug_target",
            "drug_target": "drug_target",
        }
        types = (
            frame.entity_type.astype(str).str.strip().str.replace("-", "_", regex=False)
            .str.casefold().map(aliases)
        )
        if types.isna().any():
            raise DrugSparseAssetError(f"{role} contains unsupported entity_type")
        ids = []
        for kind, value in zip(types, frame.entity_id, strict=True):
            if kind == "cancer":
                ids.append(_canonical_cancer(value))
            elif kind == "lncRNA":
                ids.append(_canonical_lnc(value))
            else:
                ids.append(_canonical_drug(value))
        result = pd.DataFrame(
            {
                "fold_id": _normalise_fold(frame, role),
                "entity_type": types,
                "entity_id": ids,
            }
        )
    else:  # pragma: no cover - internal programming error
        raise AssertionError(role)
    if result.isna().any().any():
        raise DrugSparseAssetError(f"{role} contains null keys")
    text_columns = [name for name in result if name != "fold_id"]
    if text_columns and result[text_columns].astype(str).apply(lambda x: x.str.len().eq(0)).any().any():
        raise DrugSparseAssetError(f"{role} contains empty keys")
    if result.duplicated(list(result.columns), keep=False).any():
        raise DrugSparseAssetError(f"{role} contains duplicate keys")
    return result.sort_values(list(result.columns), kind="stable").reset_index(drop=True)


def _read_factor_source(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if path.suffix.lower() in {".parquet", ".pq"} or path.is_dir():
        return pd.read_parquet(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise DrugSparseAssetError(f"Unsupported factor source: {path}")


def _write_factor(
    role: str,
    source: pd.DataFrame | str | Path,
    destination: Path,
) -> pd.DataFrame:
    # Eligibility/core factors are compact.  The exact factor can be several
    # million rows, still far smaller than the prohibited candidate join.  A
    # DataFrame source is useful to the trainer (which already has it in RAM),
    # while a Parquet path supports a post-training integration call.
    normal = _normalise_factor_frame(role, _read_factor_source(source))
    _atomic_parquet(normal, destination)
    return normal


def factor_expression_fold_coverage(
    expression: pd.DataFrame,
    cell_line_map: pd.DataFrame,
    *,
    min_cell_lines_per_dataset: int,
    min_total_cell_lines: int,
) -> pd.DataFrame:
    """Create the compact cancer × lncRNA × held-out-fold coverage factor.

    The result contains only eligible triples.  For dataset-specific expression
    a dataset contributes only after meeting ``min_cell_lines_per_dataset``;
    the sum across contributing datasets must meet ``min_total_cell_lines``.
    Canonical-model expression has no dataset axis and is counted once per
    canonical model.
    """

    if min_cell_lines_per_dataset < 1 or min_total_cell_lines < 1:
        raise ValueError("Expression coverage thresholds must be positive")
    required_map = {
        "canonical_model_id", "cancer_id", "cell_line_fold_id"
    }
    if not required_map.issubset(cell_line_map):
        raise DrugSparseAssetError(
            f"cell_line_map lacks {sorted(required_map - set(cell_line_map))}"
        )
    if not {"lncrna_id", "expression_value"}.issubset(expression):
        raise DrugSparseAssetError("expression lacks lncRNA_id/expression_value")
    mapping = cell_line_map.copy()
    mapping["cancer_id"] = mapping.cancer_id.map(_canonical_cancer)
    mapping["fold_id"] = _normalise_fold(mapping, "cell_line_map")
    values = expression.copy()
    values["lncrna_id"] = values.lncrna_id.map(_canonical_lnc)
    values["expression_value"] = pd.to_numeric(values.expression_value, errors="coerce")
    values = values.loc[np.isfinite(values.expression_value)].copy()
    if "canonical_model_id" in values:
        joined = values.merge(
            mapping[["canonical_model_id", "cancer_id", "fold_id"]].drop_duplicates(),
            on="canonical_model_id",
            how="inner",
            validate="many_to_many",
        )
        joined = joined.drop_duplicates(
            ["canonical_model_id", "cancer_id", "lncrna_id", "fold_id"]
        )
        counts = (
            joined.groupby(["cancer_id", "lncrna_id", "fold_id"], observed=True)
            .canonical_model_id.nunique().rename("matched_expression_models").reset_index()
        )
    else:
        required = {"dataset_id", "cell_line_id"}
        if not required.issubset(values) or not required.issubset(mapping):
            raise DrugSparseAssetError(
                "dataset-specific expression requires dataset_id/cell_line_id"
            )
        joined = values.merge(
            mapping[[
                "dataset_id", "cell_line_id", "canonical_model_id", "cancer_id", "fold_id"
            ]],
            on=["dataset_id", "cell_line_id"],
            how="inner",
            validate="many_to_many",
        ).drop_duplicates(
            ["dataset_id", "canonical_model_id", "cancer_id", "lncrna_id", "fold_id"]
        )
        by_dataset = (
            joined.groupby(
                ["cancer_id", "lncrna_id", "fold_id", "dataset_id"], observed=True
            ).canonical_model_id.nunique().rename("dataset_models").reset_index()
        )
        eligible_datasets = by_dataset.loc[
            by_dataset.dataset_models >= int(min_cell_lines_per_dataset)
        , ["cancer_id", "lncrna_id", "fold_id", "dataset_id"]]
        # The same canonical model can occur in GDSC and PRISM.  Dataset
        # thresholds decide which datasets contribute, but total expression
        # coverage is the union of canonical models, never a sum of per-dataset
        # counts that double-counts the same biological cell-line model.
        eligible_models = joined.merge(
            eligible_datasets,
            on=["cancer_id", "lncrna_id", "fold_id", "dataset_id"],
            how="inner",
            validate="many_to_one",
        )
        counts = (
            eligible_models.groupby(
                ["cancer_id", "lncrna_id", "fold_id"], observed=True
            ).canonical_model_id.nunique().rename("matched_expression_models").reset_index()
        )
    counts = counts.loc[
        counts.matched_expression_models >= int(min_total_cell_lines)
    ].copy()
    counts["expression_available"] = True
    return counts.sort_values(
        ["cancer_id", "lncrna_id", "fold_id"], kind="stable"
    ).reset_index(drop=True)


def factor_core_entity_availability(
    *,
    lncrna_ids_by_fold: Mapping[int, Iterable[str]],
    cancer_ids_by_fold: Mapping[int, Iterable[str]],
    drug_target_ids_by_fold: Mapping[int, Iterable[str]],
) -> pd.DataFrame:
    """Create a compact fold × entity availability factor from loaded cores.

    ``drug_target_ids_by_fold`` must contain only drugs whose target aggregate
    is available in that fold.  In the streaming trainer this is the set of
    keys whose value is true in ``drug_features[fold][2]``.
    """

    mappings = {
        "lncRNA": lncrna_ids_by_fold,
        "cancer": cancer_ids_by_fold,
        "drug_target": drug_target_ids_by_fold,
    }
    if any(set(map(int, values)) != set(FOLDS) for values in mappings.values()):
        raise DrugSparseAssetError("Core availability requires exactly folds 0..4")
    rows: list[dict[str, Any]] = []
    for entity_type, by_fold in mappings.items():
        for fold in FOLDS:
            for entity_id in sorted(set(map(str, by_fold[fold]))):
                rows.append(
                    {
                        "fold_id": fold,
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "core_available": True,
                    }
                )
    return _normalise_factor_frame(
        "core_entity_availability",
        pd.DataFrame(
            rows,
            columns=["fold_id", "entity_type", "entity_id", "core_available"],
        ),
    )


def _artifact_declaration(path: Path, root: Path, role: str) -> dict[str, Any]:
    rows, columns = _parquet_rows_and_columns(path)
    required = _REQUIRED_COLUMNS[role]
    if not set(required).issubset(columns):
        raise DrugSparseAssetError(
            f"{role} lacks required columns: {sorted(set(required) - columns)}"
        )
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": artifact_sha256(path),
        "rows": int(rows),
        "required_columns": list(required),
        "storage": "parquet_dataset" if path.is_dir() else "parquet_file",
    }


_FACTOR_KEYS: dict[str, tuple[str, ...]] = {
    "exact_candidates": ("cancer_id", "lncrna_id", "pathway_id"),
    "pathway_drug_edges": ("pathway_id", "drug_id"),
    "assay_fold_eligibility": ("cancer_id", "drug_id", "fold_id"),
    "expression_fold_coverage": ("cancer_id", "lncrna_id", "fold_id"),
    "core_entity_availability": ("fold_id", "entity_type", "entity_id"),
}


def _validate_factor_semantics(role: str, path: Path) -> dict[str, int]:
    """Independently validate one stored factor; never trust writer cleanup."""

    relation = _parquet_relation(path)
    keys = _FACTOR_KEYS[role]
    text_keys = tuple(key for key in keys if key != "fold_id")
    null_predicate = " OR ".join(f"{key} IS NULL" for key in keys)
    empty_predicate = " OR ".join(
        f"trim(CAST({key} AS VARCHAR)) = ''" for key in text_keys
    ) or "FALSE"
    fold_predicate = "FALSE"
    if "fold_id" in keys:
        fold_double = "TRY_CAST(fold_id AS DOUBLE)"
        fold_predicate = (
            "fold_id IS NULL OR "
            f"{fold_double} IS NULL OR NOT isfinite({fold_double}) OR "
            f"{fold_double} != floor({fold_double}) OR "
            f"{fold_double} < 0 OR {fold_double} > 4"
        )
    domain_predicates = []
    if "cancer_id" in keys:
        domain_predicates.append(
            "CAST(cancer_id AS VARCHAR) != upper(trim(CAST(cancer_id AS VARCHAR)))"
        )
    if "lncrna_id" in keys:
        domain_predicates.append(
            "NOT starts_with(CAST(lncrna_id AS VARCHAR), 'LNC:') "
            "OR length(CAST(lncrna_id AS VARCHAR)) <= 4 "
            "OR CAST(lncrna_id AS VARCHAR) != upper(CAST(lncrna_id AS VARCHAR))"
        )
    if "drug_id" in keys:
        domain_predicates.append(
            "NOT starts_with(CAST(drug_id AS VARCHAR), 'DRUG:') "
            "OR length(CAST(drug_id AS VARCHAR)) <= 5"
        )
    if role == "core_entity_availability":
        domain_predicates.append(
            "CAST(entity_type AS VARCHAR) NOT IN ('cancer', 'lncRNA', 'drug_target')"
        )
    domain_predicate = " OR ".join(f"({item})" for item in domain_predicates) or "FALSE"
    distinct_tuple = ", ".join(keys)
    with _bounded_duckdb() as con:
        row = con.execute(
            f"""
            SELECT
              count(*) AS rows,
              count(DISTINCT ({distinct_tuple})) AS unique_keys,
              count(*) FILTER (WHERE {null_predicate}) AS null_keys,
              count(*) FILTER (WHERE {empty_predicate}) AS empty_keys,
              count(*) FILTER (WHERE {fold_predicate}) AS invalid_folds,
              count(*) FILTER (WHERE {domain_predicate}) AS invalid_domains
            FROM {relation}
            """
        ).fetchone()
    rows, unique_keys, null_keys, empty_keys, invalid_folds, invalid_domains = map(
        int, row
    )
    if null_keys:
        raise DrugSparseAssetError(f"{role} contains null keys")
    if empty_keys:
        raise DrugSparseAssetError(f"{role} contains empty keys")
    if invalid_folds:
        raise DrugSparseAssetError(f"{role} contains folds outside integer domain 0..4")
    if invalid_domains:
        raise DrugSparseAssetError(f"{role} contains invalid identifier/entity domains")
    if unique_keys != rows:
        raise DrugSparseAssetError(f"{role} contains duplicate keys")
    return {"rows": rows, "unique_keys": unique_keys}


def _validate_available_values(path: Path, *, allow_empty: bool) -> dict[str, int]:
    mask_double = "TRY_CAST(prediction_fold_mask AS DOUBLE)"
    count_double = "TRY_CAST(cell_line_folds_with_prediction AS DOUBLE)"
    valid_mask = (
        f"prediction_fold_mask IS NOT NULL AND {mask_double} IS NOT NULL "
        f"AND isfinite({mask_double}) AND {mask_double} = floor({mask_double}) "
        f"AND {mask_double} BETWEEN {FOLD_MASK_MIN} AND {FOLD_MASK_MAX}"
    )
    valid_count = (
        f"cell_line_folds_with_prediction IS NOT NULL AND {count_double} IS NOT NULL "
        f"AND isfinite({count_double}) AND {count_double} = floor({count_double}) "
        f"AND {count_double} BETWEEN 1 AND {len(FOLDS)}"
    )
    names = (
        "rows",
        "unique_keys",
        "null_keys",
        "empty_keys",
        "invalid_domains",
        "unavailable",
        "bad_probability",
        "bad_mask",
        "bad_fold_count",
        "bad_popcount",
        "fold_contributions",
        "declared_fold_contributions",
        "partition_cancer_mismatch",
    )

    def inspect(relation: str, expected_cancer: str | None = None) -> dict[str, int]:
        mismatch = (
            "FALSE"
            if expected_cancer is None
            else f"cancer_id <> '{_sql_literal(expected_cancer)}'"
        )
        with _bounded_duckdb() as con:
            row = con.execute(
                f"""
                SELECT
                  count(*) AS n,
                  count(DISTINCT (cancer_id, lncrna_id, drug_id)) AS unique_keys,
                  count(*) FILTER (
                    WHERE cancer_id IS NULL OR lncrna_id IS NULL OR drug_id IS NULL
                  ) AS null_keys,
                  count(*) FILTER (
                    WHERE trim(CAST(cancer_id AS VARCHAR)) = ''
                       OR trim(CAST(lncrna_id AS VARCHAR)) = ''
                       OR trim(CAST(drug_id AS VARCHAR)) = ''
                  ) AS empty_keys,
                  count(*) FILTER (
                    WHERE CAST(cancer_id AS VARCHAR) != upper(trim(CAST(cancer_id AS VARCHAR)))
                       OR NOT starts_with(CAST(lncrna_id AS VARCHAR), 'LNC:')
                       OR length(CAST(lncrna_id AS VARCHAR)) <= 4
                       OR CAST(lncrna_id AS VARCHAR) != upper(CAST(lncrna_id AS VARCHAR))
                       OR NOT starts_with(CAST(drug_id AS VARCHAR), 'DRUG:')
                       OR length(CAST(drug_id AS VARCHAR)) <= 5
                  ) AS invalid_domains,
                  count(*) FILTER (WHERE availability IS NOT TRUE) AS unavailable,
                  count(*) FILTER (
                    WHERE {PROBABILITY_COLUMN} IS NULL
                       OR TRY_CAST({PROBABILITY_COLUMN} AS DOUBLE) IS NULL
                       OR NOT isfinite(TRY_CAST({PROBABILITY_COLUMN} AS DOUBLE))
                       OR TRY_CAST({PROBABILITY_COLUMN} AS DOUBLE) < 0
                       OR TRY_CAST({PROBABILITY_COLUMN} AS DOUBLE) > 1
                  ) AS bad_probability,
                  count(*) FILTER (WHERE NOT ({valid_mask})) AS bad_mask,
                  count(*) FILTER (WHERE NOT ({valid_count})) AS bad_fold_count,
                  count(*) FILTER (
                    WHERE ({valid_mask}) AND ({valid_count})
                      AND bit_count(CAST({mask_double} AS INTEGER))
                          != CAST({count_double} AS INTEGER)
                  ) AS bad_popcount,
                  coalesce(sum(
                    CASE WHEN ({valid_mask})
                      THEN bit_count(CAST({mask_double} AS INTEGER)) ELSE 0 END
                  ), 0) AS fold_contributions,
                  coalesce(sum(
                    CASE WHEN ({valid_count}) THEN CAST({count_double} AS INTEGER) ELSE 0 END
                  ), 0) AS declared_fold_contributions,
                  count(*) FILTER (WHERE {mismatch}) AS partition_cancer_mismatch
                FROM {relation}
                """
            ).fetchone()
        return dict(zip(names, map(int, row), strict=True))

    def assert_partition(audit: Mapping[str, int]) -> None:
        if audit["null_keys"]:
            raise DrugSparseAssetError("Available Drug partitions contain null keys")
        if audit["empty_keys"]:
            raise DrugSparseAssetError("Available Drug partitions contain empty keys")
        if audit["invalid_domains"]:
            raise DrugSparseAssetError(
                "Available Drug partitions contain invalid identifier domains"
            )
        if audit["unavailable"] or audit["bad_probability"]:
            raise DrugSparseAssetError(
                "Available Drug partitions contain unavailable or invalid-probability rows"
            )
        if audit["unique_keys"] != audit["rows"]:
            raise DrugSparseAssetError("Available Drug partitions contain duplicate keys")
        if audit["bad_mask"]:
            raise DrugSparseAssetError(
                "Available Drug partitions contain invalid prediction_fold_mask"
            )
        if audit["bad_fold_count"] or audit["bad_popcount"]:
            raise DrugSparseAssetError(
                "Available Drug prediction fold count does not equal fold-mask popcount"
            )
        if audit["fold_contributions"] != audit["declared_fold_contributions"]:
            raise DrugSparseAssetError("Available Drug fold contribution identity failed")

    groups = _available_partition_groups(path)
    if not groups:
        audit = inspect(_parquet_relation(path))
        assert_partition(audit)
        audit["partition_files"] = 1
        audit["cancers"] = 0
    else:
        audit = {name: 0 for name in names}
        unique_files = sorted({item for files in groups.values() for item in files})
        for item in unique_files:
            partition = inspect(_parquet_relation(item))
            assert_partition(partition)
            for name in names:
                audit[name] += partition[name]
        covered_rows = 0
        for cancer_id, files in groups.items():
            relation = _parquet_file_relation(files)
            with _bounded_duckdb() as con:
                rows, unique_keys = map(
                    int,
                    con.execute(
                        f"""
                        SELECT count(*),
                               count(DISTINCT (cancer_id, lncrna_id, drug_id))
                        FROM {relation}
                        WHERE cancer_id = ?
                        """,
                        [cancer_id],
                    ).fetchone(),
                )
            covered_rows += rows
            if unique_keys != rows:
                raise DrugSparseAssetError(
                    f"Available Drug cancer partition contains cross-file duplicates: {cancer_id}"
                )
        if covered_rows != audit["rows"]:
            raise DrugSparseAssetError("Available Drug cancer partition census is incomplete")
        # Cross-cancer duplicates are impossible because cancer_id is part of
        # the key; every observed cancer was obtained from every file above.
        audit["unique_keys"] = audit["rows"]
        audit["partition_files"] = len(unique_files)
        audit["cancers"] = len(groups)
    if not allow_empty and audit["rows"] == 0:
        raise DrugSparseAssetError("Successful Drug bundle has no available predictions")
    return audit


def _conceptual_key_count(exact_path: Path, edge_path: Path) -> int:
    """Recompute the DISTINCT conceptual universe one complete cancer at a time."""

    exact = _parquet_relation(exact_path)
    edges = _parquet_relation(edge_path)
    with _bounded_duckdb() as con:
        cancers = [
            str(row[0])
            for row in con.execute(
                f"SELECT DISTINCT cancer_id FROM {exact} ORDER BY cancer_id"
            ).fetchall()
        ]
    total = 0
    # Cancer is part of the conceptual key, so these disjoint exact partitions
    # sum to precisely the same DISTINCT relation as the former all-cancer
    # query.  A fresh bounded connection releases every partition's hash table
    # and spill before the next cancer starts.
    for cancer_id in cancers:
        with _bounded_duckdb() as con:
            total += int(
                con.execute(
                    f"""
                    SELECT count(*)
                    FROM (
                      SELECT DISTINCT c.lncrna_id, d.drug_id
                      FROM {exact} c
                      JOIN {edges} d USING (pathway_id)
                      WHERE c.cancer_id = ?
                    ) conceptual
                    """,
                    [cancer_id],
                ).fetchone()[0]
            )
    return total


def _validate_available_bindings(paths: Mapping[str, Path]) -> dict[str, int]:
    """Prove every fold named by every available key has full support.

    The formal available relation contains tens of millions of conceptual
    candidates across 33 cancers.  Validate one observed cancer partition at
    a time so DuckDB never has to retain the all-cancer joins concurrently.
    This is an exact partition of the same proof, not a sample: every public
    row belongs to exactly one ``cancer_id`` and every named fold is expanded
    and checked inside that partition.
    """

    exact = _parquet_relation(paths["exact_candidates"])
    edges = _parquet_relation(paths["pathway_drug_edges"])
    assay = _parquet_relation(paths["assay_fold_eligibility"])
    expression = _parquet_relation(paths["expression_fold_coverage"])
    core = _parquet_relation(paths["core_entity_availability"])
    totals = [0, 0, 0, 0, 0]
    groups = _available_partition_groups(paths["available_predictions"])
    if not groups:
        available = _parquet_relation(paths["available_predictions"])
        with _bounded_duckdb() as con:
            cancers = [
                str(row[0])
                for row in con.execute(
                    f"SELECT DISTINCT cancer_id FROM {available} ORDER BY cancer_id"
                ).fetchall()
            ]
        groups = {cancer_id: [paths["available_predictions"]] for cancer_id in cancers}
    physical_files = {item for files in groups.values() for item in files}
    logical_partitions = 0
    for cancer_id, files in groups.items():
        # Materialize only one cancer's bounded support relations.  The
        # conceptual relation can still contain millions of keys, so DuckDB's
        # configured memory/spill contract remains active, but it is released
        # before the next cancer begins.
        with _bounded_duckdb() as con:
            con.execute(
                f"CREATE TEMP TABLE exact_cancer AS "
                f"SELECT lncrna_id, pathway_id FROM {exact} WHERE cancer_id = ?",
                [cancer_id],
            )
            con.execute(
                f"CREATE TEMP TABLE edge_support AS "
                f"SELECT pathway_id, drug_id FROM {edges}"
            )
            con.execute(
                """
                CREATE TEMP TABLE conceptual_cancer AS
                SELECT DISTINCT c.lncrna_id, d.drug_id
                FROM exact_cancer c
                JOIN edge_support d USING (pathway_id)
                """
            )
            con.execute(
                f"CREATE TEMP TABLE assay_cancer AS "
                f"SELECT drug_id, fold_id FROM {assay} WHERE cancer_id = ?",
                [cancer_id],
            )
            con.execute(
                f"CREATE TEMP TABLE expression_cancer AS "
                f"SELECT lncrna_id, fold_id FROM {expression} WHERE cancer_id = ?",
                [cancer_id],
            )
            con.execute(
                f"CREATE TEMP TABLE core_support AS "
                f"SELECT fold_id, entity_type, entity_id FROM {core}"
            )
            for item in files:
                available = _parquet_relation(item)
                logical_partitions += 1
                counts = tuple(
                    map(
                        int,
                        con.execute(
                            f"""
                            WITH
                            available_keys AS (
                              SELECT cancer_id, lncrna_id, drug_id,
                                     CAST(prediction_fold_mask AS INTEGER)
                                       AS prediction_fold_mask,
                                     CAST(cell_line_folds_with_prediction AS INTEGER)
                                       AS cell_line_folds_with_prediction
                              FROM {available}
                              WHERE cancer_id = ?
                            ),
                            conceptual_available AS (
                              SELECT a.cancer_id, a.lncrna_id, a.drug_id
                              FROM available_keys a
                              WHERE EXISTS (
                                SELECT 1 FROM conceptual_cancer c
                                WHERE c.lncrna_id = a.lncrna_id
                                  AND c.drug_id = a.drug_id
                              )
                            ),
                            fold_contributions AS (
                              SELECT a.cancer_id, a.lncrna_id, a.drug_id, f.fold_id
                              FROM available_keys a
                              CROSS JOIN range(0, 5) AS f(fold_id)
                              WHERE (a.prediction_fold_mask & (1 << f.fold_id)) != 0
                            ),
                            same_fold_eligible AS (
                              SELECT a.cancer_id, a.lncrna_id, a.drug_id, a.fold_id
                              FROM fold_contributions a
                              WHERE EXISTS (
                                SELECT 1 FROM assay_cancer s
                                WHERE s.drug_id = a.drug_id
                                  AND s.fold_id = a.fold_id
                              )
                              AND EXISTS (
                                SELECT 1 FROM expression_cancer x
                                WHERE x.lncrna_id = a.lncrna_id
                                  AND x.fold_id = a.fold_id
                              )
                              AND EXISTS (
                                SELECT 1 FROM core_support cc
                                WHERE cc.fold_id = a.fold_id
                                  AND cc.entity_type = 'cancer'
                                  AND cc.entity_id = a.cancer_id
                              )
                              AND EXISTS (
                                SELECT 1 FROM core_support cl
                                WHERE cl.fold_id = a.fold_id
                                  AND cl.entity_type = 'lncRNA'
                                  AND cl.entity_id = a.lncrna_id
                              )
                              AND EXISTS (
                                SELECT 1 FROM core_support cd
                                WHERE cd.fold_id = a.fold_id
                                  AND cd.entity_type = 'drug_target'
                                  AND cd.entity_id = a.drug_id
                              )
                            )
                            SELECT
                              (SELECT count(*) FROM available_keys),
                              (SELECT count(*) FROM conceptual_available),
                              (SELECT coalesce(sum(cell_line_folds_with_prediction), 0)
                                 FROM available_keys),
                              (SELECT count(*) FROM fold_contributions),
                              (SELECT count(*) FROM same_fold_eligible)
                            """,
                            [cancer_id],
                        ).fetchone(),
                    )
                )
                for index, count in enumerate(counts):
                    totals[index] += count
    (
        available_rows,
        conceptual_rows,
        declared_contributions,
        expanded_contributions,
        supported_contributions,
    ) = totals
    if conceptual_rows != available_rows:
        raise DrugSparseAssetError(
            "Available Drug key is outside the factored conceptual universe"
        )
    if not (
        declared_contributions == expanded_contributions == supported_contributions
    ):
        raise DrugSparseAssetError(
            "Available Drug key lacks per-mask-fold assay/expression/current-core eligibility"
        )
    return {
        "available_rows": available_rows,
        "conceptual_available_rows": conceptual_rows,
        "declared_fold_contributions": declared_contributions,
        "expanded_fold_contributions": expanded_contributions,
        "supported_fold_contributions": supported_contributions,
    }


def write_drug_sparse_query_sidecars(
    *,
    bundle_root: str | Path,
    exact_candidates: pd.DataFrame | str | Path,
    pathway_drug_edges: pd.DataFrame | str | Path,
    assay_fold_eligibility: pd.DataFrame | str | Path,
    expression_fold_coverage: pd.DataFrame | str | Path,
    core_entity_availability: pd.DataFrame | str | Path,
    available_predictions_path: str | Path,
    conceptual_candidate_rows: int,
    training_run_id: str,
    model_run_status: str = "SUCCESS",
    model_run_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Write and hash-bind compact sidecars to a training output directory.

    ``available_predictions_path`` must already live under ``bundle_root``;
    this avoids copying a potentially large sparse public result.  Every other
    input is normalised into ``drug_sparse_query_factors`` under that root.
    The function refuses to overwrite an existing query manifest or sidecars.
    """

    root = Path(bundle_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_NAME
    validation_audit_path = root / VALIDATION_AUDIT_NAME
    factor_root = root / FACTOR_DIRECTORY
    if (
        manifest_path.exists()
        or validation_audit_path.exists()
        or (factor_root.exists() and any(factor_root.iterdir()))
    ):
        raise DrugSparseAssetError("Refusing to overwrite existing Drug sparse query bundle")
    status = str(model_run_status).strip().upper()
    if status not in {"SUCCESS", "AUDITED_UNAVAILABLE"}:
        raise DrugSparseAssetError("model_run_status must be SUCCESS or AUDITED_UNAVAILABLE")
    if status == "AUDITED_UNAVAILABLE" and not str(model_run_unavailable_reason or "").strip():
        raise DrugSparseAssetError("Audited-unavailable Drug bundle requires a reason")
    if int(conceptual_candidate_rows) < 0:
        raise DrugSparseAssetError("conceptual_candidate_rows must be non-negative")

    available = Path(available_predictions_path).resolve()
    try:
        available.relative_to(root)
    except ValueError as exc:
        raise DrugSparseAssetError(
            "Available prediction partitions must be confined to bundle_root"
        ) from exc
    _assert_no_symlink(available, root)
    allow_empty = status == "AUDITED_UNAVAILABLE"
    _, available_columns = _parquet_rows_and_columns(available)
    missing_available = set(_REQUIRED_COLUMNS["available_predictions"]) - available_columns
    if missing_available:
        raise DrugSparseAssetError(
            "available_predictions lacks required columns: "
            f"{sorted(missing_available)}"
        )
    available_audit = _validate_available_values(available, allow_empty=allow_empty)
    available_rows = available_audit["rows"]
    if status == "AUDITED_UNAVAILABLE" and available_rows:
        raise DrugSparseAssetError(
            "Audited-unavailable Drug run cannot publish non-null predictions"
        )

    factor_root.mkdir(parents=True, exist_ok=True)
    sources = {
        "exact_candidates": exact_candidates,
        "pathway_drug_edges": pathway_drug_edges,
        "assay_fold_eligibility": assay_fold_eligibility,
        "expression_fold_coverage": expression_fold_coverage,
        "core_entity_availability": core_entity_availability,
    }
    paths: dict[str, Path] = {}
    for role, source in sources.items():
        path = factor_root / f"{role}.parquet"
        _write_factor(role, source, path)
        paths[role] = path
    paths["available_predictions"] = available

    factor_audits = {
        role: _validate_factor_semantics(role, paths[role]) for role in sources
    }

    observed_conceptual_rows = _conceptual_key_count(
        paths["exact_candidates"], paths["pathway_drug_edges"]
    )
    if observed_conceptual_rows != int(conceptual_candidate_rows):
        raise DrugSparseAssetError(
            "Declared conceptual_candidate_rows does not match DISTINCT "
            f"exact×pathway-drug keys: declared={int(conceptual_candidate_rows)}, "
            f"observed={observed_conceptual_rows}"
        )
    if available_rows > observed_conceptual_rows:
        raise DrugSparseAssetError(
            "available_rows exceeds the recomputed conceptual candidate universe"
        )
    if status == "SUCCESS":
        binding_audit = _validate_available_bindings(paths)
    else:
        binding_audit = {
            "available_rows": 0,
            "conceptual_available_rows": 0,
            "declared_fold_contributions": 0,
            "expanded_fold_contributions": 0,
            "supported_fold_contributions": 0,
        }

    artifacts = {
        role: _artifact_declaration(path, root, role)
        for role, path in paths.items()
    }
    validation_audit = {
        "audit_format": "CC_HHGT_V3_2_DRUG_SPARSE_QUERY_VALIDATION_AUDIT_V1",
        "status": "PASS",
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "training_run_id": _nonempty(training_run_id, "training_run_id"),
        "model_run_status": status,
        "typed_absence_resolver": True,
        "fold_provenance_validated": True,
        "every_prediction_fold_has_assay_expression_and_current_core": True,
        "unique_keys_validated_for_every_artifact": True,
        "null_empty_and_domain_validation_passed": True,
        "count_identities_validated": True,
        "counts": {
            "conceptual_candidate_rows": int(observed_conceptual_rows),
            "available_rows": int(available_rows),
            "available_unique_keys": int(available_audit["unique_keys"]),
            "available_fold_contributions": int(
                available_audit["fold_contributions"]
            ),
            "available_declared_fold_contributions": int(
                available_audit["declared_fold_contributions"]
            ),
            "supported_fold_contributions": int(
                binding_audit["supported_fold_contributions"]
            ),
        },
        "factor_counts": factor_audits,
        "artifact_sha256": {
            role: declaration["sha256"] for role, declaration in artifacts.items()
        },
        "identities": {
            "available_rows_equals_unique_available_keys": (
                available_rows == available_audit["unique_keys"]
            ),
            "available_rows_lte_conceptual_candidate_rows": (
                available_rows <= observed_conceptual_rows
            ),
            "fold_mask_popcount_equals_declared_fold_count": (
                available_audit["fold_contributions"]
                == available_audit["declared_fold_contributions"]
            ),
            "expanded_fold_contributions_equal_supported_fold_contributions": (
                binding_audit["expanded_fold_contributions"]
                == binding_audit["supported_fold_contributions"]
            ),
        },
    }
    if not all(validation_audit["identities"].values()):
        raise DrugSparseAssetError("Sparse query validation count identity failed")
    _atomic_json(validation_audit_path, validation_audit)
    validation_audit_declaration = {
        "path": validation_audit_path.relative_to(root).as_posix(),
        "sha256": artifact_sha256(validation_audit_path),
        "status": "PASS",
        "audit_format": validation_audit["audit_format"],
    }
    manifest = {
        "manifest_format": MANIFEST_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "training_run_id": _nonempty(training_run_id, "training_run_id"),
        "model_run_status": status,
        "model_run_unavailable_reason": (
            str(model_run_unavailable_reason).strip()
            if status == "AUDITED_UNAVAILABLE"
            else None
        ),
        "conceptual_candidate_representation": "FACTORED_EXACT_X_PATHWAY_DRUG",
        "conceptual_candidate_rows": int(observed_conceptual_rows),
        "dense_candidate_table_materialized": False,
        "available_rows": int(available_rows),
        "absent_key_means_unavailable_not_zero": True,
        "available_probability_column": PROBABILITY_COLUMN,
        "typed_absence_resolver": True,
        "available_fold_provenance_required": True,
        "available_prediction_fold_mask_column": "prediction_fold_mask",
        "available_prediction_fold_count_column": "cell_line_folds_with_prediction",
        "fold_support_validation": (
            "EVERY_MASK_FOLD_ASSAY_EXPRESSION_AND_CANCER_LNCRNA_DRUG_TARGET_CORE"
        ),
        "count_identities_validated": True,
        "folds": list(FOLDS),
        "reason_precedence": list(REASON_PRECEDENCE),
        "artifacts": artifacts,
        "validation_audit": validation_audit_declaration,
        "old_association_tables_read": False,
        "old_predictions_used": False,
        "old_checkpoints_used": False,
        "tcga_patient_response_claimed": False,
        "scientific_status": "diagnostic_only",
    }
    _atomic_json(manifest_path, manifest)
    return {
        "status": status,
        "manifest_path": str(manifest_path),
        "manifest_sha256": artifact_sha256(manifest_path),
        "validation_audit_path": str(validation_audit_path),
        "validation_audit_sha256": validation_audit_declaration["sha256"],
        "available_rows": int(available_rows),
        "conceptual_candidate_rows": int(observed_conceptual_rows),
        "dense_candidate_table_materialized": False,
    }


def _validate_manifest(
    manifest_path: Path,
    expected_manifest_sha256: str | None,
    artifact_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path], str]:
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise DrugSparseAssetError(f"Drug sparse query manifest is missing: {manifest_path}")
    observed_manifest_sha = artifact_sha256(manifest_path)
    if expected_manifest_sha256 is None:
        raise DrugSparseAssetError(
            "Formal Drug sparse bundle loading requires expected_manifest_sha256"
        )
    if expected_manifest_sha256 is not None and not _SHA256.fullmatch(
        str(expected_manifest_sha256)
    ):
        raise DrugSparseAssetError("Expected Drug sparse manifest SHA256 is invalid")
    if expected_manifest_sha256 is not None and observed_manifest_sha != expected_manifest_sha256:
        raise DrugSparseAssetError("Drug sparse query manifest SHA256 mismatch")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DrugSparseAssetError("Drug sparse query manifest is invalid JSON") from exc
    required_scalars = {
        "manifest_format": MANIFEST_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "conceptual_candidate_representation": "FACTORED_EXACT_X_PATHWAY_DRUG",
        "dense_candidate_table_materialized": False,
        "absent_key_means_unavailable_not_zero": True,
        "available_probability_column": PROBABILITY_COLUMN,
        "typed_absence_resolver": True,
        "available_fold_provenance_required": True,
        "available_prediction_fold_mask_column": "prediction_fold_mask",
        "available_prediction_fold_count_column": "cell_line_folds_with_prediction",
        "fold_support_validation": (
            "EVERY_MASK_FOLD_ASSAY_EXPRESSION_AND_CANCER_LNCRNA_DRUG_TARGET_CORE"
        ),
        "count_identities_validated": True,
        "old_association_tables_read": False,
        "old_predictions_used": False,
        "old_checkpoints_used": False,
        "tcga_patient_response_claimed": False,
        "scientific_status": "diagnostic_only",
    }
    for key, expected in required_scalars.items():
        if payload.get(key) != expected:
            raise DrugSparseAssetError(f"Drug sparse manifest rejected {key}")
    if payload.get("folds") != list(FOLDS):
        raise DrugSparseAssetError("Drug sparse manifest requires folds 0..4")
    if payload.get("reason_precedence") != list(REASON_PRECEDENCE):
        raise DrugSparseAssetError("Drug sparse reason precedence drifted")
    status = payload.get("model_run_status")
    if status not in {"SUCCESS", "AUDITED_UNAVAILABLE"}:
        raise DrugSparseAssetError("Drug sparse manifest has invalid model_run_status")
    if status == "AUDITED_UNAVAILABLE" and not str(
        payload.get("model_run_unavailable_reason") or ""
    ).strip():
        raise DrugSparseAssetError("Audited-unavailable Drug manifest lacks a reason")
    if not isinstance(payload.get("conceptual_candidate_rows"), int) or payload[
        "conceptual_candidate_rows"
    ] < 0:
        raise DrugSparseAssetError("Invalid conceptual_candidate_rows")

    declarations = payload.get("artifacts")
    if not isinstance(declarations, Mapping) or set(declarations) != _REQUIRED_ARTIFACTS:
        raise DrugSparseAssetError("Drug sparse manifest artifact set is incomplete")
    manifest_root = manifest_path.parent.resolve()
    if artifact_root is None:
        root = manifest_root
    else:
        requested_root = Path(artifact_root)
        if not requested_root.is_absolute() or ".." in requested_root.parts:
            raise DrugSparseAssetError(
                "Drug sparse artifact_root must be an absolute normalized path"
            )
        absolute_root = requested_root.absolute()
        try:
            root = requested_root.resolve(strict=True)
        except OSError as exc:
            raise DrugSparseAssetError(
                "Drug sparse artifact_root is missing"
            ) from exc
        if root != absolute_root or requested_root.is_symlink() or not root.is_dir():
            raise DrugSparseAssetError(
                "Drug sparse artifact_root is symlinked or not a directory"
            )
    paths: dict[str, Path] = {}
    for role in sorted(_REQUIRED_ARTIFACTS):
        declaration = declarations[role]
        if not isinstance(declaration, Mapping):
            raise DrugSparseAssetError(f"Invalid artifact declaration: {role}")
        raw_path = str(declaration.get("path", ""))
        relative = Path(raw_path)
        if (
            not raw_path
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise DrugSparseAssetError(f"Artifact path must be relative: {role}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise DrugSparseAssetError(f"Artifact escaped bundle root: {role}") from exc
        _assert_no_symlink(path, root)
        expected_sha = str(declaration.get("sha256", ""))
        if not _SHA256.fullmatch(expected_sha):
            raise DrugSparseAssetError(f"Artifact lacks SHA256: {role}")
        try:
            observed_sha = artifact_sha256(path)
        except Exception as exc:
            raise DrugSparseAssetError(f"Artifact is missing or unhashable: {role}") from exc
        if observed_sha != expected_sha:
            raise DrugSparseAssetError(f"Artifact SHA256 mismatch: {role}")
        rows, columns = _parquet_rows_and_columns(path)
        required = _REQUIRED_COLUMNS[role]
        if declaration.get("required_columns") != list(required):
            raise DrugSparseAssetError(f"Artifact schema declaration drifted: {role}")
        if not set(required).issubset(columns):
            raise DrugSparseAssetError(f"Artifact schema is incomplete: {role}")
        if declaration.get("rows") != rows:
            raise DrugSparseAssetError(f"Artifact row count mismatch: {role}")
        paths[role] = path

    factor_audits = {
        role: _validate_factor_semantics(role, paths[role])
        for role in _FACTOR_KEYS
    }
    available_audit = _validate_available_values(
        paths["available_predictions"], allow_empty=status == "AUDITED_UNAVAILABLE"
    )
    available_rows = available_audit["rows"]
    if payload.get("available_rows") != available_rows:
        raise DrugSparseAssetError("Available prediction row count drifted")
    observed_conceptual_rows = _conceptual_key_count(
        paths["exact_candidates"], paths["pathway_drug_edges"]
    )
    if payload["conceptual_candidate_rows"] != observed_conceptual_rows:
        raise DrugSparseAssetError(
            "Manifest conceptual_candidate_rows does not match DISTINCT "
            "exact×pathway-drug keys"
        )
    if available_rows > observed_conceptual_rows:
        raise DrugSparseAssetError(
            "available_rows exceeds the recomputed conceptual candidate universe"
        )
    if status == "AUDITED_UNAVAILABLE" and available_rows:
        raise DrugSparseAssetError("Audited-unavailable Drug bundle contains predictions")
    if status == "SUCCESS":
        binding_audit = _validate_available_bindings(paths)
    else:
        binding_audit = {
            "available_rows": 0,
            "conceptual_available_rows": 0,
            "declared_fold_contributions": 0,
            "expanded_fold_contributions": 0,
            "supported_fold_contributions": 0,
        }

    audit_declaration = payload.get("validation_audit")
    if not isinstance(audit_declaration, Mapping):
        raise DrugSparseAssetError("Drug sparse manifest lacks validation_audit")
    raw_audit_path = str(audit_declaration.get("path", ""))
    relative_audit_path = Path(raw_audit_path)
    if not raw_audit_path or relative_audit_path.is_absolute():
        raise DrugSparseAssetError("Sparse query validation audit path must be relative")
    audit_path = (manifest_root / relative_audit_path).resolve()
    try:
        audit_path.relative_to(manifest_root)
    except ValueError as exc:
        raise DrugSparseAssetError("Sparse query validation audit escaped bundle root") from exc
    _assert_no_symlink(audit_path, manifest_root)
    expected_audit_sha = str(audit_declaration.get("sha256", ""))
    if not _SHA256.fullmatch(expected_audit_sha):
        raise DrugSparseAssetError("Sparse query validation audit lacks SHA256")
    try:
        observed_audit_sha = artifact_sha256(audit_path)
    except Exception as exc:
        raise DrugSparseAssetError("Sparse query validation audit is missing") from exc
    if observed_audit_sha != expected_audit_sha:
        raise DrugSparseAssetError("Sparse query validation audit SHA256 mismatch")
    try:
        validation_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DrugSparseAssetError("Sparse query validation audit is invalid JSON") from exc
    audit_format = "CC_HHGT_V3_2_DRUG_SPARSE_QUERY_VALIDATION_AUDIT_V1"
    if audit_declaration.get("status") != "PASS" or validation_audit.get("status") != "PASS":
        raise DrugSparseAssetError("Sparse query validation audit did not PASS")
    if (
        audit_declaration.get("audit_format") != audit_format
        or validation_audit.get("audit_format") != audit_format
    ):
        raise DrugSparseAssetError("Sparse query validation audit format drifted")
    audit_required = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "training_run_id": payload.get("training_run_id"),
        "model_run_status": status,
        "typed_absence_resolver": True,
        "fold_provenance_validated": True,
        "every_prediction_fold_has_assay_expression_and_current_core": True,
        "unique_keys_validated_for_every_artifact": True,
        "null_empty_and_domain_validation_passed": True,
        "count_identities_validated": True,
    }
    for key, expected in audit_required.items():
        if validation_audit.get(key) != expected:
            raise DrugSparseAssetError(f"Sparse query validation audit rejected {key}")
    expected_counts = {
        "conceptual_candidate_rows": observed_conceptual_rows,
        "available_rows": available_rows,
        "available_unique_keys": available_audit["unique_keys"],
        "available_fold_contributions": available_audit["fold_contributions"],
        "available_declared_fold_contributions": available_audit[
            "declared_fold_contributions"
        ],
        "supported_fold_contributions": binding_audit[
            "supported_fold_contributions"
        ],
    }
    if validation_audit.get("counts") != expected_counts:
        raise DrugSparseAssetError("Sparse query validation audit counts drifted")
    if validation_audit.get("factor_counts") != factor_audits:
        raise DrugSparseAssetError("Sparse query validation audit factor counts drifted")
    expected_artifact_hashes = {
        role: declarations[role]["sha256"] for role in sorted(_REQUIRED_ARTIFACTS)
    }
    if validation_audit.get("artifact_sha256") != expected_artifact_hashes:
        raise DrugSparseAssetError("Sparse query validation audit artifact hashes drifted")
    identities = validation_audit.get("identities")
    if not isinstance(identities, Mapping) or not identities or not all(
        value is True for value in identities.values()
    ):
        raise DrugSparseAssetError("Sparse query validation audit count identity failed")
    return payload, paths, observed_manifest_sha


class DrugSparseQueryBundle:
    """Validated, immutable view of a factored sparse Drug release."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_manifest_sha256: str | None = None,
        artifact_root: str | Path | None = None,
    ) -> None:
        source = Path(manifest_path).resolve()
        explicit_artifact_root = (
            None if artifact_root is None else Path(artifact_root)
        )
        self.manifest, self.paths, self.manifest_sha256 = _validate_manifest(
            source,
            expected_manifest_sha256,
            explicit_artifact_root,
        )
        self.manifest_path = source
        self.artifact_root = (
            source.parent.resolve()
            if explicit_artifact_root is None
            else explicit_artifact_root.resolve(strict=True)
        )

    def _exists(self, sql: str, parameters: list[Any]) -> bool:
        with _bounded_duckdb() as con:
            return con.execute(sql, parameters).fetchone() is not None

    def _folds(self, role: str, where: str, parameters: list[Any]) -> set[int]:
        relation = _parquet_relation(self.paths[role])
        with _bounded_duckdb() as con:
            rows = con.execute(
                f"SELECT DISTINCT fold_id FROM {relation} WHERE {where}", parameters
            ).fetchall()
        return {int(row[0]) for row in rows}

    def _available_prediction(
        self, cancer_id: str, lncrna_id: str, drug_id: str
    ) -> tuple[float, int, set[int]] | None:
        relation = _parquet_relation(self.paths["available_predictions"])
        with _bounded_duckdb() as con:
            rows = con.execute(
                f"""
                SELECT {PROBABILITY_COLUMN}, prediction_fold_mask,
                       cell_line_folds_with_prediction
                FROM {relation}
                WHERE cancer_id = ? AND lncrna_id = ? AND drug_id = ?
                LIMIT 2
                """,
                [cancer_id, lncrna_id, drug_id],
            ).fetchall()
        if len(rows) > 1:
            raise DrugSparseAssetError("Available Drug key is duplicated")
        if not rows:
            return None
        value = float(rows[0][0])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise DrugSparseAssetError("Available Drug probability is invalid")
        mask = int(rows[0][1])
        declared_count = int(rows[0][2])
        folds = {fold for fold in FOLDS if mask & (1 << fold)}
        if not FOLD_MASK_MIN <= mask <= FOLD_MASK_MAX or len(folds) != declared_count:
            raise DrugSparseAssetError("Available Drug fold provenance is invalid")
        return value, mask, folds

    def _result(
        self,
        *,
        cancer_id: str,
        lncrna_id: str,
        drug_id: str,
        probability: float | None,
        reason: DrugFailureReason | None,
    ) -> DrugSparseResolution:
        return {
            "cancer_id": cancer_id,
            "lncrna_id": lncrna_id,
            "drug_id": drug_id,
            PROBABILITY_COLUMN: probability,
            "availability": probability is not None,
            "failure_reason": reason,
            "failure_detail": (
                self.manifest.get("model_run_unavailable_reason")
                if reason == MODEL_RUN_AUDITED_UNAVAILABLE
                else None
            ),
            "analysis_version": ANALYSIS_VERSION,
            "module_id": MODULE_ID,
            "scientific_status": "diagnostic_only",
            "evidence_scope": "CELL_LINE_ASSOCIATION_NOT_TCGA_PATIENT_RESPONSE",
            "tcga_patient_response_claimed": False,
            "manifest_sha256": self.manifest_sha256,
        }

    def resolve(
        self, cancer_id: Any, lncrna_id: Any, drug_id: Any
    ) -> DrugSparseResolution:
        """Resolve one key without expanding the conceptual candidate join."""

        cancer = _canonical_cancer(_nonempty(cancer_id, "cancer_id"))
        lncrna = _canonical_lnc(_nonempty(lncrna_id, "lncrna_id"))
        drug = _canonical_drug(_nonempty(drug_id, "drug_id"))
        exact = _parquet_relation(self.paths["exact_candidates"])
        edges = _parquet_relation(self.paths["pathway_drug_edges"])
        conceptual = self._exists(
            f"""
            SELECT 1
            FROM {exact} c
            JOIN {edges} d USING (pathway_id)
            WHERE c.cancer_id = ? AND c.lncrna_id = ? AND d.drug_id = ?
            LIMIT 1
            """,
            [cancer, lncrna, drug],
        )
        if not conceptual:
            return self._result(
                cancer_id=cancer,
                lncrna_id=lncrna,
                drug_id=drug,
                probability=None,
                reason=OUTSIDE_CONCEPTUAL_UNIVERSE,
            )

        if self.manifest["model_run_status"] == "AUDITED_UNAVAILABLE":
            return self._result(
                cancer_id=cancer,
                lncrna_id=lncrna,
                drug_id=drug,
                probability=None,
                reason=MODEL_RUN_AUDITED_UNAVAILABLE,
            )

        available_prediction = self._available_prediction(cancer, lncrna, drug)
        probability = available_prediction[0] if available_prediction is not None else None
        prediction_folds = (
            available_prediction[2] if available_prediction is not None else set()
        )
        assay_folds = self._folds(
            "assay_fold_eligibility",
            "cancer_id = ? AND drug_id = ?",
            [cancer, drug],
        )
        if not assay_folds:
            if probability is not None:
                raise DrugSparseAssetError(
                    "Available Drug key contradicts assay-fold sidecar"
                )
            return self._result(
                cancer_id=cancer,
                lncrna_id=lncrna,
                drug_id=drug,
                probability=None,
                reason=NO_HELD_OUT_NATIVE_ASSAY,
            )
        if probability is not None and not prediction_folds.issubset(assay_folds):
            raise DrugSparseAssetError(
                "Available Drug key has a prediction fold without held-out assay support"
            )
        expression_folds = self._folds(
            "expression_fold_coverage",
            "cancer_id = ? AND lncrna_id = ?",
            [cancer, lncrna],
        )
        eligible_folds = assay_folds.intersection(expression_folds)
        if not eligible_folds:
            if probability is not None:
                raise DrugSparseAssetError(
                    "Available Drug key contradicts expression-fold sidecar"
                )
            return self._result(
                cancer_id=cancer,
                lncrna_id=lncrna,
                drug_id=drug,
                probability=None,
                reason=NO_MATCHED_LNCRNA_EXPRESSION,
            )
        if probability is not None and not prediction_folds.issubset(expression_folds):
            raise DrugSparseAssetError(
                "Available Drug key has a prediction fold without expression support"
            )

        core_relation = _parquet_relation(self.paths["core_entity_availability"])
        with _bounded_duckdb() as con:
            rows = con.execute(
                f"""
                SELECT fold_id, entity_type
                FROM {core_relation}
                WHERE (entity_type = 'cancer' AND entity_id = ?)
                   OR (entity_type = 'lncRNA' AND entity_id = ?)
                   OR (entity_type = 'drug_target' AND entity_id = ?)
                """,
                [cancer, lncrna, drug],
            ).fetchall()
        core_types_by_fold: dict[int, set[str]] = {}
        for fold, entity_type in rows:
            core_types_by_fold.setdefault(int(fold), set()).add(str(entity_type))
        core_folds = {
            fold
            for fold in eligible_folds
            if core_types_by_fold.get(fold, set()) == set(_CORE_TYPES)
        }
        if not core_folds:
            if probability is not None:
                raise DrugSparseAssetError(
                    "Available Drug key contradicts current-core sidecar"
                )
            return self._result(
                cancer_id=cancer,
                lncrna_id=lncrna,
                drug_id=drug,
                probability=None,
                reason=CURRENT_V32_CORE_UNAVAILABLE,
            )
        if probability is not None and not prediction_folds.issubset(core_folds):
            raise DrugSparseAssetError(
                "Available Drug key has a prediction fold without all current-core entities"
            )
        if probability is not None:
            return self._result(
                cancer_id=cancer,
                lncrna_id=lncrna,
                drug_id=drug,
                probability=probability,
                reason=None,
            )
        return self._result(
            cancer_id=cancer,
            lncrna_id=lncrna,
            drug_id=drug,
            probability=None,
            reason=MODEL_OUTPUT_MISSING_FAIL_CLOSED,
        )


def load_drug_sparse_query_bundle(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    artifact_root: str | Path | None = None,
) -> DrugSparseQueryBundle:
    """Load a bundle only after caller-pinned hash and content validation."""

    return DrugSparseQueryBundle(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        artifact_root=artifact_root,
    )


__all__ = [
    "ANALYSIS_VERSION",
    "CURRENT_V32_CORE_UNAVAILABLE",
    "DrugSparseAssetError",
    "DrugSparseInputError",
    "DrugSparseQueryBundle",
    "DrugSparseQueryError",
    "DrugSparseResolution",
    "MANIFEST_FORMAT",
    "MANIFEST_NAME",
    "MODEL_OUTPUT_MISSING_FAIL_CLOSED",
    "MODEL_RUN_AUDITED_UNAVAILABLE",
    "NO_HELD_OUT_NATIVE_ASSAY",
    "NO_MATCHED_LNCRNA_EXPRESSION",
    "OUTSIDE_CONCEPTUAL_UNIVERSE",
    "PROBABILITY_COLUMN",
    "STRICT_VALIDATION_STRATEGY",
    "VALIDATION_AUDIT_NAME",
    "factor_core_entity_availability",
    "factor_expression_fold_coverage",
    "load_drug_sparse_query_bundle",
    "write_drug_sparse_query_sidecars",
]
