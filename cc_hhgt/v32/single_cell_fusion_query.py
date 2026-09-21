"""Hash-pinned read-only queries for the formal V3.2 single-cell expert.

The queried table is the full cancer/lncRNA/exact-pathway fusion adapter, not
the older HNSC pilot and not the raw expression-coverage audit.  Missing
single-cell support remains a typed null and is never interpreted as zero.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .single_cell_fusion_adapter import (
    ADAPTER_FORMAT,
    ANALYSIS_VERSION,
    AVAILABILITY_COLUMN,
    PROBABILITY_COLUMN,
    REASON_COLUMN,
    TARGET_KEYS,
)
from .single_cell_fusion_binding import (
    BINDING_FORMAT,
    FORMAL_CANDIDATE_SHA256,
    artifact_sha256,
)


MAX_QUERY_LIMIT = 1_000
MAX_QUERY_OFFSET = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANCER = re.compile(r"^[A-Z0-9_-]{2,16}$")


class SingleCellFusionQueryError(RuntimeError):
    """Base single-cell exact-pathway query error."""


class SingleCellFusionQueryAssetError(SingleCellFusionQueryError):
    """Raised when the bound adapter is missing, stale or semantically invalid."""


class SingleCellFusionQueryInputError(SingleCellFusionQueryError):
    """Raised when a query filter is invalid."""


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise SingleCellFusionQueryAssetError(f"{label} is missing or unsafe: {source}")
    return source


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleCellFusionQueryAssetError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise SingleCellFusionQueryAssetError(f"{label} must be a JSON object")
    return value


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _canonical_lnc(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    text = re.sub(r"^(?:LNC|LNCRNA|GENE):", "", text)
    text = re.sub(r"\.\d+$", "", text)
    if not text or len(text) > 128 or any(ord(character) < 32 for character in text):
        raise SingleCellFusionQueryInputError("lncrna_id is invalid")
    return "LNC:" + text


def _canonical_cancer(value: Any | None) -> str | None:
    if value is None:
        return None
    cancer = str(value).strip().upper()
    if not _CANCER.fullmatch(cancer):
        raise SingleCellFusionQueryInputError("cancer_id is invalid")
    return cancer


def _canonical_pathway(value: Any | None) -> str | None:
    if value is None:
        return None
    pathway = str(value).strip()
    if not pathway or len(pathway) > 256 or any(ord(character) < 32 for character in pathway):
        raise SingleCellFusionQueryInputError("pathway_id is invalid")
    return pathway


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise SingleCellFusionQueryInputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise SingleCellFusionQueryInputError(f"offset must be 0..{MAX_QUERY_OFFSET}")
    return int(limit), int(offset)


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


class SingleCellFusionReleaseQuery:
    """Validated immutable view of the formal exact-pathway adapter."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
    ) -> None:
        source = _safe_file(binding_path, "Single-cell fusion binding")
        expected = str(expected_binding_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise SingleCellFusionQueryAssetError(
                "Single-cell fusion query requires an expected binding SHA256"
            )
        observed = artifact_sha256(source)
        if observed != expected:
            raise SingleCellFusionQueryAssetError(
                f"Single-cell fusion binding SHA mismatch: {observed} != {expected}"
            )
        binding = _load_json(source, "Single-cell fusion binding")
        required = {
            "format": BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS_HASH_BOUND_FUSION_INPUT",
            "adapter_format": ADAPTER_FORMAT,
            "target_level": "cancer_x_lncrna_x_exact_pathway",
            "fusion_input_eligible": True,
            "direct_target_evidence": False,
            "affects_discovery": True,
            "affects_confidence": True,
            "changes_primary_ranking": False,
            "family_to_exact_broadcast": False,
            "unavailable_encoding": "null_with_reason",
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "release_ready": False,
            "production_deployed": False,
        }
        for key, value in required.items():
            if binding.get(key) != value:
                raise SingleCellFusionQueryAssetError(
                    f"Single-cell fusion binding has invalid {key}: {binding.get(key)!r}"
                )
        authority = binding.get("candidate_authority")
        if (
            not isinstance(authority, dict)
            or authority.get("sha256") != FORMAL_CANDIDATE_SHA256
        ):
            raise SingleCellFusionQueryAssetError(
                "Single-cell fusion binding is not tied to the formal candidate authority"
            )
        candidate_rows = binding.get("candidate_rows")
        available_rows = binding.get("available_rows")
        unavailable_rows = binding.get("unavailable_rows")
        if (
            isinstance(candidate_rows, bool)
            or not isinstance(candidate_rows, int)
            or candidate_rows < 1
            or isinstance(available_rows, bool)
            or not isinstance(available_rows, int)
            or available_rows < 0
            or isinstance(unavailable_rows, bool)
            or not isinstance(unavailable_rows, int)
            or unavailable_rows < 0
            or candidate_rows != available_rows + unavailable_rows
        ):
            raise SingleCellFusionQueryAssetError(
                "Single-cell fusion binding row counts are invalid"
            )
        prediction = binding.get("prediction")
        if not isinstance(prediction, dict) or prediction.get("rows") != candidate_rows:
            raise SingleCellFusionQueryAssetError(
                "Single-cell fusion binding lacks the full prediction declaration"
            )
        digest = str(prediction.get("sha256", "")).lower()
        if not _SHA256.fullmatch(digest):
            raise SingleCellFusionQueryAssetError("Single-cell prediction SHA256 is invalid")
        prediction_path = _safe_file(
            prediction.get("path", ""), "Single-cell exact-pathway adapter"
        )
        try:
            prediction_path.relative_to(source.parent.resolve())
        except ValueError as exc:
            raise SingleCellFusionQueryAssetError(
                "Single-cell prediction escaped the binding root"
            ) from exc
        if artifact_sha256(prediction_path) != digest:
            raise SingleCellFusionQueryAssetError("Single-cell prediction SHA drift")

        self.binding = binding
        self.binding_path = source
        self.binding_sha256 = observed
        self.prediction_path = prediction_path
        self.training_run_id = str(binding.get("training_run_id", ""))
        if not self.training_run_id.startswith("V32-SINGLE-CELL-FRESH-"):
            raise SingleCellFusionQueryAssetError(
                "Single-cell fusion binding lacks a fresh V3.2 training run"
            )
        self._validate_prediction(
            candidate_rows=candidate_rows,
            available_rows=available_rows,
            unavailable_rows=unavailable_rows,
        )

    @staticmethod
    def _connect():
        return duckdb.connect(database=":memory:")

    def _validate_prediction(
        self, *, candidate_rows: int, available_rows: int, unavailable_rows: int
    ) -> None:
        relation = f"read_parquet({_sql_path(self.prediction_path)})"
        connection = self._connect()
        try:
            columns = {
                row[0]
                for row in connection.execute(
                    f"DESCRIBE SELECT * FROM {relation}"
                ).fetchall()
            }
            required = set(TARGET_KEYS) | {
                PROBABILITY_COLUMN,
                AVAILABILITY_COLUMN,
                REASON_COLUMN,
                "analysis_version",
                "module_id",
                "adapter_format",
                "target_level",
                "direct_target_evidence",
                "family_to_exact_broadcast",
                "changes_primary_ranking",
                "availability_encoding",
            }
            missing = sorted(required - columns)
            if missing:
                raise SingleCellFusionQueryAssetError(
                    f"Single-cell adapter lacks columns: {missing}"
                )
            row = connection.execute(
                f"""
                SELECT count(*),
                       count(DISTINCT (cancer_id, lncrna_id, pathway_id)),
                       count_if({AVAILABILITY_COLUMN}),
                       count_if(NOT {AVAILABILITY_COLUMN}),
                       count_if({AVAILABILITY_COLUMN} AND
                                ({PROBABILITY_COLUMN} IS NULL OR
                                 NOT isfinite({PROBABILITY_COLUMN}) OR
                                 {PROBABILITY_COLUMN} < 0 OR {PROBABILITY_COLUMN} > 1)),
                       count_if(NOT {AVAILABILITY_COLUMN} AND
                                {PROBABILITY_COLUMN} IS NOT NULL),
                       count_if(NOT {AVAILABILITY_COLUMN} AND
                                ({REASON_COLUMN} IS NULL OR trim({REASON_COLUMN}) = '')),
                       count_if(analysis_version <> ? OR module_id <> 'single_cell'
                                OR adapter_format <> ?
                                OR target_level <> 'cancer_x_lncrna_x_exact_pathway'),
                       count_if(direct_target_evidence OR family_to_exact_broadcast
                                OR changes_primary_ranking
                                OR availability_encoding <> 'null_with_reason')
                FROM {relation}
                """,
                [ANALYSIS_VERSION, ADAPTER_FORMAT],
            ).fetchone()
        finally:
            connection.close()
        expected = (
            candidate_rows,
            candidate_rows,
            available_rows,
            unavailable_rows,
            0,
            0,
            0,
            0,
            0,
        )
        if tuple(map(int, row)) != expected:
            raise SingleCellFusionQueryAssetError(
                f"Single-cell adapter semantic validation failed: {row}"
            )

    def capability_status(self) -> dict[str, Any]:
        return {
            "module_id": "single_cell",
            "status": self.binding["status"],
            "target_level": "cancer_x_lncrna_x_exact_pathway",
            "formal_full_universe_adapter": True,
            "candidate_rows": self.binding["candidate_rows"],
            "available_rows": self.binding["available_rows"],
            "unavailable_rows": self.binding["unavailable_rows"],
            "affects_discovery": True,
            "affects_confidence": True,
            "changes_primary_ranking": False,
            "typed_unavailable_nulls": True,
            "release_ready": False,
            "production_deployed": False,
            "binding_sha256": self.binding_sha256,
        }

    def query_associations(
        self,
        *,
        cancer_id: Any | None = None,
        lncrna_id: Any | None = None,
        pathway_id: Any | None = None,
        availability: bool | None = None,
        min_probability: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        cancer = _canonical_cancer(cancer_id)
        lnc = _canonical_lnc(lncrna_id)
        pathway = _canonical_pathway(pathway_id)
        limit, offset = _bounds(limit, offset)
        minimum = None
        if min_probability is not None:
            minimum = float(min_probability)
            if not math.isfinite(minimum) or not 0 <= minimum <= 1:
                raise SingleCellFusionQueryInputError(
                    "min_probability must be within 0..1"
                )
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("cancer_id", cancer),
            ("lncrna_id", lnc),
            ("pathway_id", pathway),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if availability is not None:
            clauses.append(f"{AVAILABILITY_COLUMN} = ?")
            parameters.append(bool(availability))
        if minimum is not None:
            clauses.append(f"{AVAILABILITY_COLUMN} AND {PROBABILITY_COLUMN} >= ?")
            parameters.append(minimum)
        where = " AND ".join(clauses) if clauses else "TRUE"
        relation = f"read_parquet({_sql_path(self.prediction_path)})"
        connection = self._connect()
        try:
            frame = connection.execute(
                f"""
                SELECT * FROM {relation}
                WHERE {where}
                ORDER BY {AVAILABILITY_COLUMN} DESC,
                         {PROBABILITY_COLUMN} DESC NULLS LAST,
                         cancer_id, lncrna_id, pathway_id
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            connection.close()
        rows = [
            {str(key): _json_value(value) for key, value in record.items()}
            for record in frame.to_dict("records")
        ]
        return {
            "module": "single_cell",
            "query_kind": "cancer_lncrna_exact_pathway_association",
            "target_level": "cancer_x_lncrna_x_exact_pathway",
            "affects_discovery": True,
            "affects_confidence": True,
            "changes_primary_ranking": False,
            "typed_unavailable_nulls": True,
            "filters": {
                "cancer_id": cancer,
                "lncrna_id": lnc,
                "pathway_id": pathway,
                "availability": availability,
                "min_probability": minimum,
                "limit": limit,
                "offset": offset,
            },
            "returned_rows": len(rows),
            "rows": rows,
            "provenance": {
                "analysis_version": ANALYSIS_VERSION,
                "training_run_id": self.training_run_id,
                "binding_path": str(self.binding_path),
                "binding_sha256": self.binding_sha256,
                "release_ready": False,
                "production_deployed": False,
            },
        }


__all__ = [
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
    "SingleCellFusionQueryAssetError",
    "SingleCellFusionQueryError",
    "SingleCellFusionQueryInputError",
    "SingleCellFusionReleaseQuery",
]
