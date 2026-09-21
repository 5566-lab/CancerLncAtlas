"""Strict, read-only query binding for the fresh HNSC cell-level UCell pilot.

This module is deliberately separate from the production/staging API.  It
binds one audited, immutable HNSC pilot and exposes pathway-signature scores at
cell and donor/cell-type level.  It is not an lncRNA-level result, does not
make the single-cell module complete, and does not make a release deployable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CC_HHGT_V3_2_HNSC_UCELL_PILOT_QUERY_BINDING_V1"
BINDING_STATUS = "SUCCESS_HNSC_UCELL_PILOT_HASH_BOUND"
PAGE_FORMAT = "CC_HHGT_V3_2_HNSC_UCELL_PILOT_QUERY_PAGE_V1"
RESULT_ROLE = "HNSC_FRESH_CELL_LEVEL_UCELL_PILOT_NOT_RELEASE"
DATASET_ID = "SC_GSE103322_HNSC"
CANCER_ID = "HNSC"
PSEUDOTIME_REASON = "NO_EXPLICIT_TRAJECTORY_ROOT_OR_ORDERED_SOURCE_STATE"
UCell_UNAVAILABLE_REASONS = {
    "FULL_SIGNATURE_SIZE_AT_OR_EXCEEDS_UCELL_MAX_RANK_DOMAIN": 14,
    "NO_MEMBER_PROTEIN_IN_MATRIX": 1,
}

EXPECTED_CELLS = 5_902
EXPECTED_PATHWAYS = 2_135
EXPECTED_AVAILABLE_PATHWAYS = 2_120
EXPECTED_UNAVAILABLE_PATHWAYS = 15
EXPECTED_PARTITIONS = 93
EXPECTED_MANIFEST_ENTRIES = 98
EXPECTED_CELL_LEVEL_ROWS = 12_600_770
EXPECTED_NUMERIC_ROWS = 12_512_240
EXPECTED_TYPED_UNAVAILABLE_ROWS = 88_530
EXPECTED_DONOR_CELLTYPE_GROUPS = 101
EXPECTED_DONOR_CELLTYPE_ROWS = 215_635

MAX_PAGE_LIMIT = 500
MAX_TEXT_FILTER_LENGTH = 512
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

PINNED_SHA256: Mapping[str, str] = {
    "SUCCESS.json": "a49a118056e76bf256f5f572f28f73138600416cd1b981428ddfcea2230cd28d",
    "STATUS.json": "a14f425ba9abd510b0eaf4ea6f64a937a89dcd95816c5f4d5518b4966b61a296",
    "LINEAGE.json": "437a06f44c15d8831ea8366fb91d8f151ee0dd5cacec293e2d490931048f1548",
    "CELL_LEVEL_HANDOFF.json": "8dd6c222a36015f30054127e49d8f0056cd81c4bd9d8b797fca06974466fae3a",
    "FILE_MANIFEST.parquet": "d685535e1b7b8dac07bd8ed9b097251aaf0d09df980532ec87401921e0965591",
    "AUDIT.json": "88d84545a17bdac0ebe2d8bf6a79df78ecd1978f69dcd39d408d2a511b3fc196",
}

_CELL_COLUMNS = {
    "dataset_id",
    "cancer_id",
    "cell_id",
    "patient_id",
    "cell_type_major",
    "pathway_id",
    "ucell_available",
    "unavailable_reason",
    "ucell_score",
}
_AGGREGATE_COLUMNS = {
    "dataset_id",
    "cancer_id",
    "pathway_id",
    "patient_id",
    "cell_type_major",
    "cell_count",
    "ucell_available",
    "unavailable_reason",
    "ucell_score_mean",
}
_PATHWAY_COLUMNS = {
    "pathway_id",
    "signature_gene_count",
    "present_gene_count",
    "missing_gene_count",
    "max_rank",
    "ucell_available",
    "unavailable_reason",
    "complete_signature_length_used_in_denominator",
    "matrix_missing_member_policy",
    "matrix_missing_member_is_expression_data_imputation",
    "smoothing_used",
}
_PSEUDO_CELL_COLUMNS = {
    "dataset_id",
    "cancer_id",
    "cell_id",
    "patient_id",
    "cell_type_major",
    "pseudotime_available",
    "pseudotime_value",
    "unavailable_reason",
}
_PSEUDO_PATHWAY_COLUMNS = {
    "pathway_id",
    "pseudotime_available",
    "pseudotime_effect",
    "unavailable_reason",
}


class HNSCUCellQueryError(RuntimeError):
    """Base HNSC UCell query error."""


class HNSCUCellAssetError(HNSCUCellQueryError):
    """Raised when a hash binding or result artifact is invalid."""


class HNSCUCellInputError(HNSCUCellQueryError):
    """Raised when a query filter is invalid or inapplicable."""


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HNSCUCellAssetError(message)


def _safe_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise HNSCUCellAssetError(f"{label} is missing or unsafe: {resolved}")
    return resolved


def _safe_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    if not root.is_dir() or root.is_symlink():
        raise HNSCUCellAssetError(f"HNSC UCell result root is missing or unsafe: {root}")
    return root


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HNSCUCellAssetError(f"{label} is missing or invalid JSON") from exc
    if not isinstance(value, dict):
        raise HNSCUCellAssetError(f"{label} must be a JSON object")
    return value


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _read_schema(path: Path, required: set[str], label: str) -> None:
    try:
        observed = set(pq.read_schema(path).names)
    except Exception as exc:
        raise HNSCUCellAssetError(f"{label} is unreadable Parquet: {path}") from exc
    missing = sorted(required - observed)
    if missing:
        raise HNSCUCellAssetError(f"{label} lacks columns: {missing}")
    forbidden = sorted(
        name for name in observed if name.casefold() in {"lncrna_id", "lncrna_symbol"}
    )
    if forbidden:
        raise HNSCUCellAssetError(
            f"{label} falsely exposes lncRNA-level UCell columns: {forbidden}"
        )


def _relation(path: Path, *, glob: bool = False) -> str:
    source = path.resolve().as_posix()
    if glob:
        source = source.rstrip("/") + "/part-*.parquet"
    source = source.replace("'", "''")
    return f"read_parquet('{source}', union_by_name=true)"


def _duckdb_connect():
    try:
        import duckdb
    except ImportError as exc:
        raise HNSCUCellAssetError("duckdb is required for bounded HNSC UCell queries") from exc
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET threads=2")
    connection.execute("SET memory_limit='512MB'")
    return connection


def _expected_manifest_paths() -> set[str]:
    paths = {
        f"cell_level_ucell/part-{index:05d}.parquet"
        for index in range(EXPECTED_PARTITIONS)
    }
    paths.update(
        {
            "PRESENT_MISSING_COUNT_AUDIT.json",
            "pseudotime_cell_availability.parquet",
            "pseudotime_pathway_availability.parquet",
            "ucell_donor_celltype.parquet",
            "ucell_pathway_availability.parquet",
        }
    )
    return paths


def _validate_manifest(root: Path, manifest_path: Path) -> pd.DataFrame:
    try:
        manifest = pd.read_parquet(manifest_path)
    except Exception as exc:
        raise HNSCUCellAssetError("HNSC UCell file manifest is unreadable") from exc
    required = {"relative_path", "size_bytes", "sha256"}
    _require(required.issubset(manifest.columns), "HNSC UCell manifest schema drift")
    _require(len(manifest) == EXPECTED_MANIFEST_ENTRIES, "Manifest must have 98 entries")
    _require(not manifest[list(required)].isna().any().any(), "Manifest contains null fields")
    relative = manifest["relative_path"].astype(str)
    _require(not relative.duplicated().any(), "Manifest contains duplicate paths")
    _require(set(relative) == _expected_manifest_paths(), "Manifest artifact universe drift")

    for row in manifest.itertuples(index=False):
        text = str(row.relative_path)
        pure = PurePosixPath(text)
        _require(
            text == pure.as_posix()
            and not pure.is_absolute()
            and ".." not in pure.parts
            and "\\" not in text,
            f"Unsafe manifest path: {text}",
        )
        expected_sha = str(row.sha256).lower()
        _require(bool(_SHA256.fullmatch(expected_sha)), f"Invalid manifest SHA: {text}")
        try:
            expected_size = int(row.size_bytes)
        except (TypeError, ValueError) as exc:
            raise HNSCUCellAssetError(f"Invalid manifest size: {text}") from exc
        _require(expected_size > 0, f"Invalid manifest size: {text}")
        target = (root / Path(*pure.parts)).resolve()
        _require(root == target.parent or root in target.parents, f"Manifest path escaped root: {text}")
        _require(target.is_file() and not target.is_symlink(), f"Manifest file missing: {text}")
        _require(target.stat().st_size == expected_size, f"Manifest size drift: {text}")
        _require(artifact_sha256(target) == expected_sha, f"Manifest SHA drift: {text}")
    return manifest


def _require_values(record: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, value in expected.items():
        if record.get(key) != value:
            raise HNSCUCellAssetError(
                f"{label} has invalid {key}: {record.get(key)!r} != {value!r}"
            )


def _validate_control_semantics(
    controls: Mapping[str, dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    success = controls["SUCCESS.json"]
    status = controls["STATUS.json"]
    lineage = controls["LINEAGE.json"]
    handoff = controls["CELL_LEVEL_HANDOFF.json"]

    _require_values(
        success,
        {
            "status": "SUCCESS",
            "cancer_id": CANCER_ID,
            "cells": EXPECTED_CELLS,
            "pathways_total_coverage": EXPECTED_PATHWAYS,
            "pathways_available": EXPECTED_AVAILABLE_PATHWAYS,
            "pathways_typed_unavailable": EXPECTED_UNAVAILABLE_PATHWAYS,
            "cell_level_coverage_rows": EXPECTED_CELL_LEVEL_ROWS,
            "cell_level_numeric_score_rows": EXPECTED_NUMERIC_ROWS,
            "cell_level_typed_unavailable_rows": EXPECTED_TYPED_UNAVAILABLE_ROWS,
            "full_hnsc_started": True,
            "full_hnsc_completed": True,
            "pseudotime_numeric_output": False,
            "unavailable_values_filled_with_zero_or_half": False,
            "single_cell_module_complete": False,
            "release_ready": False,
        },
        "SUCCESS",
    )
    for label, record in {
        "STATUS": status,
        "LINEAGE": lineage,
        "HANDOFF": handoff,
    }.items():
        _require_values(
            record,
            {
                "status": "PASS",
                "analysis_version": ANALYSIS_VERSION,
                "cancer_id": CANCER_ID,
                "dataset_id": DATASET_ID,
                "historical_sc_trajectory_used": False,
                "historical_predictions_used": False,
                "historical_rankings_used": False,
                "historical_checkpoints_used": False,
                "pseudotime_numeric_output": False,
                "single_cell_module_complete": False,
                "release_ready": False,
            },
            label,
        )
    _require_values(
        status,
        {
            "cell_level_coverage_rows": EXPECTED_CELL_LEVEL_ROWS,
            "cell_level_numeric_score_rows": EXPECTED_NUMERIC_ROWS,
            "cell_level_typed_unavailable_rows": EXPECTED_TYPED_UNAVAILABLE_ROWS,
            "pathways_available": EXPECTED_AVAILABLE_PATHWAYS,
            "pathways_unavailable": EXPECTED_UNAVAILABLE_PATHWAYS,
            "cell_level_partitions": EXPECTED_PARTITIONS,
            "donor_celltype_groups": EXPECTED_DONOR_CELLTYPE_GROUPS,
            "donor_celltype_rows": EXPECTED_DONOR_CELLTYPE_ROWS,
            "training_started": False,
            "production_deployed": False,
        },
        "STATUS",
    )
    for record in (status, lineage, handoff):
        _require(
            record.get("file_manifest_sha256") == PINNED_SHA256["FILE_MANIFEST.parquet"],
            "Control does not bind the pinned 98-entry manifest",
        )
        _require(
            record.get("complete_exact_signature_length_used") is True,
            "Control lost complete-signature UCell semantics",
        )
        _require(
            record.get("matrix_missing_member_is_expression_data_imputation") is False,
            "Control falsely claims expression imputation",
        )
    _require(success.get("status_sha256") == PINNED_SHA256["STATUS.json"], "SUCCESS status SHA drift")
    _require(success.get("lineage_sha256") == PINNED_SHA256["LINEAGE.json"], "SUCCESS lineage SHA drift")
    _require(
        success.get("handoff_sha256") == PINNED_SHA256["CELL_LEVEL_HANDOFF.json"],
        "SUCCESS handoff SHA drift",
    )
    _require(
        success.get("file_manifest_sha256") == PINNED_SHA256["FILE_MANIFEST.parquet"],
        "SUCCESS manifest SHA drift",
    )

    _require_values(
        audit,
        {
            "format": "CC_HHGT_V3_2_HNSC_UCELL_FULL_INDEPENDENT_AUDIT_V1",
            "status": "PASS",
            "cells": EXPECTED_CELLS,
            "pathways_total": EXPECTED_PATHWAYS,
            "pathways_available": EXPECTED_AVAILABLE_PATHWAYS,
            "pathways_typed_unavailable": EXPECTED_UNAVAILABLE_PATHWAYS,
            "partitions": EXPECTED_PARTITIONS,
            "manifest_entries": EXPECTED_MANIFEST_ENTRIES,
            "manifest_mismatches": 0,
            "cell_level_coverage_rows": EXPECTED_CELL_LEVEL_ROWS,
            "cell_level_numeric_rows": EXPECTED_NUMERIC_ROWS,
            "cell_level_typed_unavailable_rows": EXPECTED_TYPED_UNAVAILABLE_ROWS,
            "donor_celltype_groups": EXPECTED_DONOR_CELLTYPE_GROUPS,
            "donor_celltype_rows": EXPECTED_DONOR_CELLTYPE_ROWS,
            "pseudotime_cell_rows": EXPECTED_CELLS,
            "pseudotime_pathway_rows": EXPECTED_PATHWAYS,
            "pseudotime_numeric_values": 0,
            "historical_outputs_used": False,
            "unavailable_values_filled_with_zero_or_half": False,
            "single_cell_module_complete": False,
            "release_ready": False,
        },
        "Independent AUDIT",
    )
    expected_reason_rows = {
        reason: EXPECTED_CELLS * count
        for reason, count in UCell_UNAVAILABLE_REASONS.items()
    }
    _require(
        audit.get("unavailable_reason_row_counts") == expected_reason_rows,
        "Independent AUDIT unavailable reason counts drift",
    )
    expected_control_shas = {
        "success_sha256": PINNED_SHA256["SUCCESS.json"],
        "status_sha256": PINNED_SHA256["STATUS.json"],
        "lineage_sha256": PINNED_SHA256["LINEAGE.json"],
        "handoff_sha256": PINNED_SHA256["CELL_LEVEL_HANDOFF.json"],
        "file_manifest_sha256": PINNED_SHA256["FILE_MANIFEST.parquet"],
    }
    _require(
        audit.get("control_shas") == expected_control_shas,
        "Independent AUDIT control SHA binding drift",
    )


def _validate_cell_level(root: Path) -> dict[str, Any]:
    parts = sorted((root / "cell_level_ucell").glob("part-*.parquet"))
    _require(len(parts) == EXPECTED_PARTITIONS, "Cell-level partition count drift")
    for part in parts:
        _require(not part.is_symlink(), f"Unsafe cell-level partition: {part}")
        _read_schema(part, _CELL_COLUMNS, "Cell-level UCell partition")
    relation = _relation(root / "cell_level_ucell", glob=True)
    connection = _duckdb_connect()
    try:
        summary = connection.execute(
            f"""
            SELECT
                count(*)::BIGINT AS coverage_rows,
                count(ucell_score)::BIGINT AS numeric_rows,
                count(*) FILTER (WHERE NOT ucell_available)::BIGINT AS unavailable_rows,
                count(DISTINCT cell_id)::BIGINT AS cells,
                count(DISTINCT pathway_id)::BIGINT AS pathways,
                count(*) FILTER (WHERE dataset_id <> ? OR cancer_id <> ?)::BIGINT AS scope_errors,
                count(*) FILTER (
                    WHERE ucell_available IS NULL
                       OR (ucell_available AND (ucell_score IS NULL OR unavailable_reason IS NOT NULL))
                       OR (NOT ucell_available AND (ucell_score IS NOT NULL OR unavailable_reason IS NULL))
                )::BIGINT AS typed_errors,
                count(*) FILTER (
                    WHERE ucell_score IS NOT NULL
                      AND (NOT isfinite(ucell_score) OR ucell_score < 0 OR ucell_score > 1)
                )::BIGINT AS numeric_errors,
                min(ucell_score)::DOUBLE AS score_min,
                max(ucell_score)::DOUBLE AS score_max
            FROM {relation}
            """,
            [DATASET_ID, CANCER_ID],
        ).fetchone()
        grouped = connection.execute(
            f"""
            SELECT
                count(*)::BIGINT,
                min(rows_per_cell)::BIGINT,
                max(rows_per_cell)::BIGINT,
                min(pathways_per_cell)::BIGINT,
                max(pathways_per_cell)::BIGINT,
                min(available_per_cell)::BIGINT,
                max(available_per_cell)::BIGINT
            FROM (
                SELECT cell_id,
                       count(*) AS rows_per_cell,
                       count(DISTINCT pathway_id) AS pathways_per_cell,
                       count(*) FILTER (WHERE ucell_available) AS available_per_cell
                FROM {relation}
                GROUP BY cell_id
            )
            """
        ).fetchone()
        reason_rows = connection.execute(
            f"""
            SELECT unavailable_reason, count(*)::BIGINT AS rows
            FROM {relation}
            WHERE NOT ucell_available
            GROUP BY unavailable_reason
            ORDER BY unavailable_reason
            """
        ).fetchall()
    finally:
        connection.close()
    _require(summary is not None and grouped is not None, "Cell-level UCell scan returned no facts")
    expected_summary = (
        EXPECTED_CELL_LEVEL_ROWS,
        EXPECTED_NUMERIC_ROWS,
        EXPECTED_TYPED_UNAVAILABLE_ROWS,
        EXPECTED_CELLS,
        EXPECTED_PATHWAYS,
        0,
        0,
        0,
    )
    _require(tuple(int(value) for value in summary[:8]) == expected_summary, "Cell-level UCell counts drift")
    _require(
        tuple(int(value) for value in grouped)
        == (
            EXPECTED_CELLS,
            EXPECTED_PATHWAYS,
            EXPECTED_PATHWAYS,
            EXPECTED_PATHWAYS,
            EXPECTED_PATHWAYS,
            EXPECTED_AVAILABLE_PATHWAYS,
            EXPECTED_AVAILABLE_PATHWAYS,
        ),
        "Per-cell UCell coverage drift",
    )
    expected_reason_rows = {
        reason: EXPECTED_CELLS * count
        for reason, count in UCell_UNAVAILABLE_REASONS.items()
    }
    observed_reason_rows = {str(reason): int(rows) for reason, rows in reason_rows}
    _require(observed_reason_rows == expected_reason_rows, "Cell-level typed-unavailable reasons drift")
    return {
        "coverage_rows": int(summary[0]),
        "numeric_rows": int(summary[1]),
        "typed_unavailable_rows": int(summary[2]),
        "cells": int(summary[3]),
        "pathways": int(summary[4]),
        "score_min": float(summary[8]),
        "score_max": float(summary[9]),
    }


def _validate_sidecars(root: Path) -> dict[str, Any]:
    pathway_path = root / "ucell_pathway_availability.parquet"
    aggregate_path = root / "ucell_donor_celltype.parquet"
    pseudo_cell_path = root / "pseudotime_cell_availability.parquet"
    pseudo_pathway_path = root / "pseudotime_pathway_availability.parquet"
    _read_schema(pathway_path, _PATHWAY_COLUMNS, "Pathway availability")
    _read_schema(aggregate_path, _AGGREGATE_COLUMNS, "Donor/cell-type UCell")
    _read_schema(pseudo_cell_path, _PSEUDO_CELL_COLUMNS, "Cell pseudotime availability")
    _read_schema(pseudo_pathway_path, _PSEUDO_PATHWAY_COLUMNS, "Pathway pseudotime availability")

    pathways = pd.read_parquet(pathway_path)
    _require(len(pathways) == EXPECTED_PATHWAYS, "Pathway availability row count drift")
    _require(pathways["pathway_id"].nunique(dropna=False) == EXPECTED_PATHWAYS, "Pathway IDs are not unique")
    available = pathways["ucell_available"].astype(bool)
    _require(int(available.sum()) == EXPECTED_AVAILABLE_PATHWAYS, "Available pathway count drift")
    _require(
        pathways.loc[available, "unavailable_reason"].isna().all()
        and pathways.loc[~available, "unavailable_reason"].notna().all(),
        "Pathway availability is not typed",
    )
    _require(
        pathways.loc[~available, "unavailable_reason"].value_counts().astype(int).to_dict()
        == UCell_UNAVAILABLE_REASONS,
        "Pathway unavailable reason counts drift",
    )
    _require(
        pathways["signature_gene_count"].eq(
            pathways["present_gene_count"] + pathways["missing_gene_count"]
        ).all(),
        "Signature gene count invariant failed",
    )
    _require(pathways["max_rank"].eq(1500).all(), "UCell maxRank drift")
    _require(
        pathways["complete_signature_length_used_in_denominator"].eq(True).all(),
        "Complete signature denominator flag drift",
    )
    _require(
        pathways["matrix_missing_member_is_expression_data_imputation"].eq(False).all()
        and pathways["smoothing_used"].eq(False).all(),
        "UCell semantics drift",
    )

    aggregate_relation = _relation(aggregate_path)
    connection = _duckdb_connect()
    try:
        aggregate = connection.execute(
            f"""
            SELECT
                count(*)::BIGINT,
                count(DISTINCT patient_id || chr(31) || cell_type_major)::BIGINT,
                count(DISTINCT pathway_id)::BIGINT,
                count(ucell_score_mean)::BIGINT,
                count(*) FILTER (WHERE NOT ucell_available)::BIGINT,
                count(*) FILTER (
                    WHERE dataset_id <> ? OR cancer_id <> ? OR cell_count <= 0
                       OR ucell_available IS NULL
                       OR (ucell_available AND (ucell_score_mean IS NULL OR unavailable_reason IS NOT NULL))
                       OR (NOT ucell_available AND (ucell_score_mean IS NOT NULL OR unavailable_reason IS NULL))
                )::BIGINT
            FROM {aggregate_relation}
            """,
            [DATASET_ID, CANCER_ID],
        ).fetchone()
        aggregate_groups = connection.execute(
            f"""
            SELECT count(*)::BIGINT,
                   min(rows_per_group)::BIGINT, max(rows_per_group)::BIGINT,
                   min(pathways_per_group)::BIGINT, max(pathways_per_group)::BIGINT,
                   min(available_per_group)::BIGINT, max(available_per_group)::BIGINT
            FROM (
                SELECT patient_id, cell_type_major,
                       count(*) AS rows_per_group,
                       count(DISTINCT pathway_id) AS pathways_per_group,
                       count(*) FILTER (WHERE ucell_available) AS available_per_group
                FROM {aggregate_relation}
                GROUP BY patient_id, cell_type_major
            )
            """
        ).fetchone()
    finally:
        connection.close()
    _require(
        tuple(int(value) for value in aggregate)
        == (
            EXPECTED_DONOR_CELLTYPE_ROWS,
            EXPECTED_DONOR_CELLTYPE_GROUPS,
            EXPECTED_PATHWAYS,
            EXPECTED_DONOR_CELLTYPE_GROUPS * EXPECTED_AVAILABLE_PATHWAYS,
            EXPECTED_DONOR_CELLTYPE_GROUPS * EXPECTED_UNAVAILABLE_PATHWAYS,
            0,
        ),
        "Donor/cell-type aggregate counts drift",
    )
    _require(
        tuple(int(value) for value in aggregate_groups)
        == (
            EXPECTED_DONOR_CELLTYPE_GROUPS,
            EXPECTED_PATHWAYS,
            EXPECTED_PATHWAYS,
            EXPECTED_PATHWAYS,
            EXPECTED_PATHWAYS,
            EXPECTED_AVAILABLE_PATHWAYS,
            EXPECTED_AVAILABLE_PATHWAYS,
        ),
        "Donor/cell-type aggregate coverage drift",
    )

    pseudo_cells = pd.read_parquet(pseudo_cell_path)
    _require(len(pseudo_cells) == EXPECTED_CELLS, "Pseudotime cell row count drift")
    _require(pseudo_cells["cell_id"].nunique(dropna=False) == EXPECTED_CELLS, "Pseudotime cell IDs drift")
    _require(
        pseudo_cells["pseudotime_available"].eq(False).all()
        and pseudo_cells["pseudotime_value"].isna().all()
        and pseudo_cells["unavailable_reason"].eq(PSEUDOTIME_REASON).all(),
        "Pseudotime cell typed-unavailable semantics drift",
    )
    pseudo_pathways = pd.read_parquet(pseudo_pathway_path)
    _require(len(pseudo_pathways) == EXPECTED_PATHWAYS, "Pseudotime pathway row count drift")
    _require(
        pseudo_pathways["pathway_id"].nunique(dropna=False) == EXPECTED_PATHWAYS,
        "Pseudotime pathway IDs drift",
    )
    _require(
        pseudo_pathways["pseudotime_available"].eq(False).all()
        and pseudo_pathways["pseudotime_effect"].isna().all()
        and pseudo_pathways["unavailable_reason"].eq(PSEUDOTIME_REASON).all(),
        "Pseudotime pathway typed-unavailable semantics drift",
    )
    return {
        "pathways_available": int(available.sum()),
        "pathways_typed_unavailable": int((~available).sum()),
        "donor_celltype_groups": int(aggregate[1]),
        "donor_celltype_rows": int(aggregate[0]),
        "pseudotime_numeric_values": 0,
    }


def validate_hnsc_ucell_pilot(
    *,
    result_root: str | Path,
    independent_audit_path: str | Path,
) -> dict[str, Any]:
    """Re-hash and independently re-check the complete immutable pilot."""
    root = _safe_root(result_root)
    audit_path = _safe_file(independent_audit_path, "Independent HNSC UCell AUDIT")
    _require(
        artifact_sha256(audit_path) == PINNED_SHA256["AUDIT.json"],
        "Independent HNSC UCell AUDIT SHA drift",
    )
    controls: dict[str, dict[str, Any]] = {}
    for name in ("SUCCESS.json", "STATUS.json", "LINEAGE.json", "CELL_LEVEL_HANDOFF.json"):
        path = _safe_file(root / name, name)
        _require(artifact_sha256(path) == PINNED_SHA256[name], f"Pinned {name} SHA drift")
        controls[name] = _load_json(path, name)
    manifest_path = _safe_file(root / "FILE_MANIFEST.parquet", "FILE_MANIFEST.parquet")
    _require(
        artifact_sha256(manifest_path) == PINNED_SHA256["FILE_MANIFEST.parquet"],
        "Pinned FILE_MANIFEST.parquet SHA drift",
    )
    audit = _load_json(audit_path, "Independent HNSC UCell AUDIT")
    _validate_control_semantics(controls, audit)
    manifest = _validate_manifest(root, manifest_path)
    cell = _validate_cell_level(root)
    sidecars = _validate_sidecars(root)
    return {
        "result_root": str(root),
        "independent_audit_path": str(audit_path),
        "manifest_entries": len(manifest),
        "cells": cell["cells"],
        "pathways_total": cell["pathways"],
        "pathways_available": sidecars["pathways_available"],
        "pathways_typed_unavailable": sidecars["pathways_typed_unavailable"],
        "cell_level_coverage_rows": cell["coverage_rows"],
        "cell_level_numeric_rows": cell["numeric_rows"],
        "cell_level_typed_unavailable_rows": cell["typed_unavailable_rows"],
        "donor_celltype_groups": sidecars["donor_celltype_groups"],
        "donor_celltype_rows": sidecars["donor_celltype_rows"],
        "pseudotime_numeric_values": sidecars["pseudotime_numeric_values"],
        "score_min": cell["score_min"],
        "score_max": cell["score_max"],
        "historical_inputs_used": False,
        "lncrna_query_applicable": False,
        "single_cell_module_complete": False,
        "release_ready": False,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_hnsc_ucell_query_binding(
    *,
    result_root: str | Path,
    independent_audit_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create a new, non-overwriting binding for the one audited HNSC pilot."""
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise HNSCUCellAssetError(f"Refusing to overwrite HNSC UCell binding: {destination}")
    validated = validate_hnsc_ucell_pilot(
        result_root=result_root,
        independent_audit_path=independent_audit_path,
    )
    destination.mkdir(parents=True, exist_ok=False)
    binding_path = destination / "HNSC_UCELL_QUERY_BINDING.json"
    success_path = destination / "SUCCESS.json"
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": BINDING_STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_role": RESULT_ROLE,
        "scope": "HNSC_ONLY_PILOT",
        "cancer_id": CANCER_ID,
        "dataset_id": DATASET_ID,
        "query_axes": ["pathway_id", "cell_id", "cell_type_major", "patient_id"],
        "lncrna_query_applicable": False,
        "lncrna_query_unavailable_reason": "UCELL_IS_PATHWAY_SIGNATURE_LEVEL_NOT_LNCRNA_LEVEL",
        "typed_unavailable_preserved": True,
        "historical_sc_trajectory_used": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "pseudotime_numeric_output": False,
        "single_cell_module_complete": False,
        "release_ready": False,
        "production_deployed": False,
        "validated_counts": {
            key: validated[key]
            for key in (
                "manifest_entries",
                "cells",
                "pathways_total",
                "pathways_available",
                "pathways_typed_unavailable",
                "cell_level_coverage_rows",
                "cell_level_numeric_rows",
                "cell_level_typed_unavailable_rows",
                "donor_celltype_groups",
                "donor_celltype_rows",
                "pseudotime_numeric_values",
            )
        },
        "sources": {
            "result_root": validated["result_root"],
            "independent_audit": {
                "path": validated["independent_audit_path"],
                "sha256": PINNED_SHA256["AUDIT.json"],
            },
            "controls": {
                name: {"relative_path": name, "sha256": digest}
                for name, digest in PINNED_SHA256.items()
                if name != "AUDIT.json"
            },
        },
    }
    _atomic_json(binding_path, binding)
    binding_sha = artifact_sha256(binding_path)
    success = {
        "format": "CC_HHGT_V3_2_HNSC_UCELL_PILOT_QUERY_BINDING_SUCCESS_V1",
        "status": BINDING_STATUS,
        "binding": binding_path.name,
        "binding_sha256": binding_sha,
        "scope": "HNSC_ONLY_PILOT",
        "single_cell_module_complete": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _atomic_json(success_path, success)
    return {
        "binding_path": str(binding_path),
        "binding_sha256": binding_sha,
        "success_path": str(success_path),
        "binding": binding,
    }


def _normalise_filter(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HNSCUCellInputError(f"{label} must be a string")
    text = value.strip()
    if not text or len(text) > MAX_TEXT_FILTER_LENGTH:
        raise HNSCUCellInputError(
            f"{label} must be non-empty and at most {MAX_TEXT_FILTER_LENGTH} characters"
        )
    return text


def _normalise_availability(value: str) -> str:
    if not isinstance(value, str):
        raise HNSCUCellInputError("availability must be ALL, AVAILABLE, or TYPED_UNAVAILABLE")
    normal = value.strip().upper()
    if normal not in {"ALL", "AVAILABLE", "TYPED_UNAVAILABLE"}:
        raise HNSCUCellInputError("availability must be ALL, AVAILABLE, or TYPED_UNAVAILABLE")
    return normal


def _validate_page(limit: int, offset: int, maximum_offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_LIMIT:
        raise HNSCUCellInputError(f"limit must be an integer in 1..{MAX_PAGE_LIMIT}")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or not 0 <= offset <= maximum_offset
    ):
        raise HNSCUCellInputError(f"offset must be an integer in 0..{maximum_offset}")
    return limit, offset


class HNSCUCellPilotQuery:
    """Strictly hash-pinned, read-only query view of the HNSC UCell pilot."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
    ) -> None:
        source = _safe_file(binding_path, "HNSC UCell query binding")
        expected = str(expected_binding_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise HNSCUCellAssetError("HNSC UCell query requires an expected binding SHA256")
        observed = artifact_sha256(source)
        if observed != expected:
            raise HNSCUCellAssetError(
                f"HNSC UCell binding SHA mismatch: {observed} != {expected}"
            )
        binding = _load_json(source, "HNSC UCell query binding")
        _require_values(
            binding,
            {
                "format": BINDING_FORMAT,
                "analysis_version": ANALYSIS_VERSION,
                "status": BINDING_STATUS,
                "result_role": RESULT_ROLE,
                "scope": "HNSC_ONLY_PILOT",
                "cancer_id": CANCER_ID,
                "dataset_id": DATASET_ID,
                "lncrna_query_applicable": False,
                "lncrna_query_unavailable_reason": "UCELL_IS_PATHWAY_SIGNATURE_LEVEL_NOT_LNCRNA_LEVEL",
                "typed_unavailable_preserved": True,
                "historical_sc_trajectory_used": False,
                "historical_predictions_used": False,
                "historical_rankings_used": False,
                "historical_checkpoints_used": False,
                "pseudotime_numeric_output": False,
                "single_cell_module_complete": False,
                "release_ready": False,
                "production_deployed": False,
            },
            "HNSC UCell binding",
        )
        _require(
            binding.get("query_axes")
            == ["pathway_id", "cell_id", "cell_type_major", "patient_id"],
            "HNSC UCell query axes drift",
        )
        success = _load_json(source.parent / "SUCCESS.json", "HNSC UCell binding SUCCESS")
        _require_values(
            success,
            {
                "format": "CC_HHGT_V3_2_HNSC_UCELL_PILOT_QUERY_BINDING_SUCCESS_V1",
                "status": BINDING_STATUS,
                "binding": source.name,
                "binding_sha256": observed,
                "scope": "HNSC_ONLY_PILOT",
                "single_cell_module_complete": False,
                "release_ready": False,
                "production_deployed": False,
            },
            "HNSC UCell binding SUCCESS",
        )
        sources = binding.get("sources")
        _require(isinstance(sources, dict), "HNSC UCell binding lacks sources")
        controls = sources.get("controls")
        _require(isinstance(controls, dict), "HNSC UCell binding lacks control pins")
        for name, digest in PINNED_SHA256.items():
            if name == "AUDIT.json":
                continue
            _require(
                controls.get(name) == {"relative_path": name, "sha256": digest},
                f"HNSC UCell binding control pin drift: {name}",
            )
        audit_decl = sources.get("independent_audit")
        _require(
            isinstance(audit_decl, dict)
            and audit_decl.get("sha256") == PINNED_SHA256["AUDIT.json"],
            "HNSC UCell binding AUDIT pin drift",
        )
        validated = validate_hnsc_ucell_pilot(
            result_root=sources.get("result_root", ""),
            independent_audit_path=audit_decl.get("path", ""),
        )
        expected_counts = {
            key: validated[key]
            for key in (
                "manifest_entries",
                "cells",
                "pathways_total",
                "pathways_available",
                "pathways_typed_unavailable",
                "cell_level_coverage_rows",
                "cell_level_numeric_rows",
                "cell_level_typed_unavailable_rows",
                "donor_celltype_groups",
                "donor_celltype_rows",
                "pseudotime_numeric_values",
            )
        }
        _require(binding.get("validated_counts") == expected_counts, "Binding validated counts drift")
        self.binding = binding
        self.binding_sha256 = observed
        self.result_root = Path(validated["result_root"])

    def capability_status(self) -> dict[str, Any]:
        return {
            "result_role": RESULT_ROLE,
            "scope": "HNSC_ONLY_PILOT",
            "cancer_id": CANCER_ID,
            "dataset_id": DATASET_ID,
            "query_axes": list(self.binding["query_axes"]),
            "lncrna_query_applicable": False,
            "lncrna_query_unavailable_reason": self.binding[
                "lncrna_query_unavailable_reason"
            ],
            "typed_unavailable_preserved": True,
            "cells": EXPECTED_CELLS,
            "pathways_total": EXPECTED_PATHWAYS,
            "pathways_available": EXPECTED_AVAILABLE_PATHWAYS,
            "pathways_typed_unavailable": EXPECTED_UNAVAILABLE_PATHWAYS,
            "pseudotime_numeric_values": 0,
            "historical_inputs_used": False,
            "single_cell_module_complete": False,
            "release_ready": False,
            "production_deployed": False,
        }

    def query_lncrna(self, lncrna_id: str, **_: Any) -> dict[str, Any]:
        _normalise_filter(lncrna_id, "lncrna_id")
        raise HNSCUCellInputError(
            "lncRNA query is not applicable: UCell scores pathway signatures, not lncRNAs"
        )

    def _page(
        self,
        *,
        relation: str,
        columns: list[str],
        clauses: list[str],
        parameters: list[Any],
        order_by: str,
        query_level: str,
        filters: Mapping[str, Any],
        limit: int,
        offset: int,
        maximum_offset: int,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        limit, offset = _validate_page(limit, offset, maximum_offset)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        connection = _duckdb_connect()
        try:
            total = int(
                connection.execute(
                    f"SELECT count(*)::BIGINT FROM {relation}{where}", parameters
                ).fetchone()[0]
            )
            frame = connection.execute(
                f"SELECT {', '.join(columns)} FROM {relation}{where} "
                f"ORDER BY {order_by} LIMIT ? OFFSET ?",
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            connection.close()
        next_offset = offset + len(frame)
        payload = {
            "format": PAGE_FORMAT,
            "result_role": RESULT_ROLE,
            "scope": "HNSC_ONLY_PILOT",
            "cancer_id": CANCER_ID,
            "dataset_id": DATASET_ID,
            "query_level": query_level,
            "filters": dict(filters),
            "total_matching_rows": total,
            "returned_rows": len(frame),
            "limit": limit,
            "offset": offset,
            "next_offset": next_offset if next_offset < total else None,
            "rows": _records(frame),
            "typed_unavailable_preserved": True,
            "lncrna_query_applicable": False,
            "single_cell_module_complete": False,
            "release_ready": False,
        }
        if extra:
            payload.update(extra)
        return payload

    @staticmethod
    def _availability_clause(availability: str, clauses: list[str]) -> str:
        normal = _normalise_availability(availability)
        if normal == "AVAILABLE":
            clauses.append("ucell_available = TRUE")
        elif normal == "TYPED_UNAVAILABLE":
            clauses.append("ucell_available = FALSE")
        return normal

    def query_cell_scores(
        self,
        *,
        pathway_id: str | None = None,
        cell_id: str | None = None,
        cell_type_major: str | None = None,
        patient_id: str | None = None,
        availability: str = "ALL",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        values = {
            "pathway_id": _normalise_filter(pathway_id, "pathway_id"),
            "cell_id": _normalise_filter(cell_id, "cell_id"),
            "cell_type_major": _normalise_filter(cell_type_major, "cell_type_major"),
            "patient_id": _normalise_filter(patient_id, "patient_id"),
        }
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in values.items():
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        normal_availability = self._availability_clause(availability, clauses)
        filters = {**values, "availability": normal_availability}
        return self._page(
            relation=_relation(self.result_root / "cell_level_ucell", glob=True),
            columns=[
                "dataset_id",
                "cancer_id",
                "cell_id",
                "patient_id",
                "cell_type_major",
                "pathway_id",
                "ucell_available",
                "unavailable_reason",
                "ucell_score",
            ],
            clauses=clauses,
            parameters=parameters,
            order_by="pathway_id, cell_id, patient_id, cell_type_major",
            query_level="CELL_PATHWAY",
            filters=filters,
            limit=limit,
            offset=offset,
            maximum_offset=EXPECTED_CELL_LEVEL_ROWS,
        )

    def query_donor_celltype_scores(
        self,
        *,
        pathway_id: str | None = None,
        cell_type_major: str | None = None,
        patient_id: str | None = None,
        availability: str = "ALL",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        values = {
            "pathway_id": _normalise_filter(pathway_id, "pathway_id"),
            "cell_type_major": _normalise_filter(cell_type_major, "cell_type_major"),
            "patient_id": _normalise_filter(patient_id, "patient_id"),
        }
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in values.items():
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        normal_availability = self._availability_clause(availability, clauses)
        return self._page(
            relation=_relation(self.result_root / "ucell_donor_celltype.parquet"),
            columns=[
                "dataset_id",
                "cancer_id",
                "pathway_id",
                "patient_id",
                "cell_type_major",
                "cell_count",
                "ucell_available",
                "unavailable_reason",
                "ucell_score_mean",
            ],
            clauses=clauses,
            parameters=parameters,
            order_by="pathway_id, patient_id, cell_type_major",
            query_level="DONOR_CELLTYPE_PATHWAY",
            filters={**values, "availability": normal_availability},
            limit=limit,
            offset=offset,
            maximum_offset=EXPECTED_DONOR_CELLTYPE_ROWS,
        )

    def query_pathway_availability(
        self,
        *,
        pathway_id: str | None = None,
        availability: str = "ALL",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        pathway = _normalise_filter(pathway_id, "pathway_id")
        clauses: list[str] = []
        parameters: list[Any] = []
        if pathway is not None:
            clauses.append("pathway_id = ?")
            parameters.append(pathway)
        normal_availability = self._availability_clause(availability, clauses)
        return self._page(
            relation=_relation(self.result_root / "ucell_pathway_availability.parquet"),
            columns=[
                "pathway_id",
                "signature_gene_count",
                "present_gene_count",
                "missing_gene_count",
                "max_rank",
                "ucell_available",
                "unavailable_reason",
                "complete_signature_length_used_in_denominator",
                "matrix_missing_member_policy",
                "matrix_missing_member_is_expression_data_imputation",
                "smoothing_used",
            ],
            clauses=clauses,
            parameters=parameters,
            order_by="pathway_id",
            query_level="PATHWAY_AVAILABILITY",
            filters={"pathway_id": pathway, "availability": normal_availability},
            limit=limit,
            offset=offset,
            maximum_offset=EXPECTED_PATHWAYS,
        )

    def query_pseudotime_availability(
        self,
        *,
        level: str,
        pathway_id: str | None = None,
        cell_id: str | None = None,
        cell_type_major: str | None = None,
        patient_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        if not isinstance(level, str) or level.strip().upper() not in {"CELL", "PATHWAY"}:
            raise HNSCUCellInputError("pseudotime level must be CELL or PATHWAY")
        normal_level = level.strip().upper()
        if normal_level == "PATHWAY":
            if any(value is not None for value in (cell_id, cell_type_major, patient_id)):
                raise HNSCUCellInputError("cell/donor filters are invalid for pathway pseudotime")
            pathway = _normalise_filter(pathway_id, "pathway_id")
            clauses = ["pseudotime_available = FALSE"]
            parameters: list[Any] = []
            if pathway is not None:
                clauses.append("pathway_id = ?")
                parameters.append(pathway)
            return self._page(
                relation=_relation(
                    self.result_root / "pseudotime_pathway_availability.parquet"
                ),
                columns=[
                    "pathway_id",
                    "pseudotime_available",
                    "pseudotime_effect",
                    "unavailable_reason",
                ],
                clauses=clauses,
                parameters=parameters,
                order_by="pathway_id",
                query_level="PSEUDOTIME_PATHWAY_TYPED_UNAVAILABLE",
                filters={"pathway_id": pathway},
                limit=limit,
                offset=offset,
                maximum_offset=EXPECTED_PATHWAYS,
                extra={"pseudotime_numeric_values": 0},
            )
        if pathway_id is not None:
            raise HNSCUCellInputError("pathway_id is invalid for cell pseudotime")
        values = {
            "cell_id": _normalise_filter(cell_id, "cell_id"),
            "cell_type_major": _normalise_filter(cell_type_major, "cell_type_major"),
            "patient_id": _normalise_filter(patient_id, "patient_id"),
        }
        clauses = ["pseudotime_available = FALSE"]
        parameters = []
        for column, value in values.items():
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        return self._page(
            relation=_relation(self.result_root / "pseudotime_cell_availability.parquet"),
            columns=[
                "dataset_id",
                "cancer_id",
                "cell_id",
                "patient_id",
                "cell_type_major",
                "pseudotime_available",
                "pseudotime_value",
                "unavailable_reason",
            ],
            clauses=clauses,
            parameters=parameters,
            order_by="cell_id, patient_id, cell_type_major",
            query_level="PSEUDOTIME_CELL_TYPED_UNAVAILABLE",
            filters=values,
            limit=limit,
            offset=offset,
            maximum_offset=EXPECTED_CELLS,
            extra={"pseudotime_numeric_values": 0},
        )


__all__ = [
    "HNSCUCellAssetError",
    "HNSCUCellInputError",
    "HNSCUCellPilotQuery",
    "HNSCUCellQueryError",
    "PINNED_SHA256",
    "artifact_sha256",
    "build_hnsc_ucell_query_binding",
    "validate_hnsc_ucell_pilot",
]
