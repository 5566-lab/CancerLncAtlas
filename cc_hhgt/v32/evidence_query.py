"""Hash-pinned read-only Evidence confidence and event-lineage queries."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .evidence_output_binding import (
    ANALYSIS_VERSION,
    BINDING_FORMAT,
    FORMAL_CANDIDATE_SHA256,
    FORMAL_EVIDENCE_CODE_SHA256,
    FORMAL_EVIDENCE_RUNNER_SHA256,
    FORMAL_MEMBERSHIP_SHA256,
)
from .release_registry import artifact_sha256


MAX_QUERY_LIMIT = 1_000
MAX_QUERY_OFFSET = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvidenceQueryError(RuntimeError):
    """Base Evidence query error."""


class EvidenceQueryAssetError(EvidenceQueryError):
    """Raised when the Evidence binding or an artifact is invalid."""


class EvidenceQueryInputError(EvidenceQueryError):
    """Raised when a query filter is invalid."""


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _canonical_lnc(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"^(?:LNC|LNCRNA):", "", text)
    text = re.sub(r"\.\d+$", "", text)
    if not text:
        raise EvidenceQueryInputError("lncrna_id is required")
    return "LNC:" + text


def _lnc_storage_aliases(value: Any) -> tuple[str, str]:
    """Return canonical and bare storage forms used by V3.2 Evidence assets."""

    canonical = _canonical_lnc(value)
    return canonical, canonical.removeprefix("LNC:")


def _clean(value: Any, label: str, *, uppercase: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        raise EvidenceQueryInputError(f"{label} is required")
    if len(text) > 512 or any(ord(character) < 32 for character in text):
        raise EvidenceQueryInputError(f"{label} is invalid")
    return text.upper() if uppercase else text


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise EvidenceQueryInputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise EvidenceQueryInputError(f"offset must be 0..{MAX_QUERY_OFFSET}")
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


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


class EvidenceBindingQuery:
    """Validated immutable view of auxiliary fresh V3.2 Evidence outputs."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
    ) -> None:
        source = Path(binding_path).resolve()
        if not source.is_file() or source.is_symlink():
            raise EvidenceQueryAssetError(f"Evidence binding is missing/unsafe: {source}")
        expected = str(expected_binding_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise EvidenceQueryAssetError("Evidence query requires an expected binding SHA256")
        observed = artifact_sha256(source)
        if observed != expected:
            raise EvidenceQueryAssetError(
                f"Evidence binding SHA256 mismatch: {observed} != {expected}"
            )
        try:
            binding = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvidenceQueryAssetError("Evidence binding is invalid JSON") from exc
        required = {
            "format": BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND",
            "release_ready": False,
            "production_deployed": False,
            "evidence_module_still_auxiliary_partial": True,
            "interaction_materialization_input_eligible": True,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "family_to_exact_broadcast": False,
            "five_fresh_private_heads_verified": True,
        }
        for key, expected_value in required.items():
            if binding.get(key) != expected_value:
                raise EvidenceQueryAssetError(
                    f"Evidence binding has invalid {key}: {binding.get(key)!r}"
                )
        run_id = str(binding.get("evidence_training_run_id", ""))
        if not run_id.startswith("V32-EVIDENCE-TRAIN-"):
            raise EvidenceQueryAssetError("Evidence binding lacks fresh training_run_id")
        if int(binding.get("optimizer_steps_total", 0)) <= 0:
            raise EvidenceQueryAssetError("Evidence binding has no optimizer updates")
        authorities = binding.get("authorities")
        if not isinstance(authorities, dict):
            raise EvidenceQueryAssetError("Evidence binding lacks authorities")
        expected_authorities = {
            "evidence_code": FORMAL_EVIDENCE_CODE_SHA256,
            "evidence_runner": FORMAL_EVIDENCE_RUNNER_SHA256,
            "exact_membership": FORMAL_MEMBERSHIP_SHA256,
            "exact_candidates": FORMAL_CANDIDATE_SHA256,
        }
        for role, digest in expected_authorities.items():
            declaration = authorities.get(role)
            if not isinstance(declaration, dict) or declaration.get("sha256") != digest:
                raise EvidenceQueryAssetError(f"Evidence {role} is not formal authority")
        success_path = source.parent / "SUCCESS.json"
        try:
            success = json.loads(success_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvidenceQueryAssetError("Evidence binding success marker is missing/invalid") from exc
        if (
            success.get("status") != binding["status"]
            or success.get("release_ready") is not False
            or success.get("binding") != source.name
            or success.get("binding_sha256") != observed
        ):
            raise EvidenceQueryAssetError("Evidence binding success marker is stale")

        declarations = binding.get("artifacts")
        if not isinstance(declarations, dict):
            raise EvidenceQueryAssetError("Evidence binding lacks artifacts")
        paths: dict[str, Path] = {}
        for role, declaration in declarations.items():
            if not isinstance(declaration, dict):
                raise EvidenceQueryAssetError(f"Evidence artifact declaration is invalid: {role}")
            path = Path(str(declaration.get("path", ""))).resolve()
            if not path.is_file() or path.is_symlink():
                raise EvidenceQueryAssetError(f"Evidence artifact is missing/unsafe: {role}")
            if artifact_sha256(path) != declaration.get("sha256"):
                raise EvidenceQueryAssetError(f"Evidence artifact SHA drift: {role}")
            paths[str(role)] = path
        for role in ("evidence_predictions", "event_lineage"):
            if role not in paths:
                raise EvidenceQueryAssetError(f"Evidence binding lacks query artifact: {role}")
        self.binding = binding
        self.binding_path = source
        self.binding_sha256 = observed
        self.training_run_id = run_id
        self.paths = paths
        self._validate_tables()

    def _connect(self):
        return duckdb.connect(":memory:")

    def _validate_tables(self) -> None:
        predictions = f"read_parquet({_sql_path(self.paths['evidence_predictions'])})"
        events = f"read_parquet({_sql_path(self.paths['event_lineage'])})"
        required_predictions = {
            "cancer_id", "lncrna_id", "pathway_id",
            "evidence_confidence_probability", "uncertainty", "availability",
            "failure_reason", "analysis_version", "training_run_id",
            "changes_primary_ranking", "main_ranking_modified",
        }
        required_events = {
            "cancer_id", "lncrna_id", "pathway_id", "family_broadcast_used"
        }
        con = self._connect()
        try:
            prediction_columns = {
                row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {predictions}").fetchall()
            }
            event_columns = {
                row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {events}").fetchall()
            }
            if missing := sorted(required_predictions - prediction_columns):
                raise EvidenceQueryAssetError(f"Evidence predictions lack columns: {missing}")
            if missing := sorted(required_events - event_columns):
                raise EvidenceQueryAssetError(f"Evidence event lineage lacks columns: {missing}")
            declared_predictions = int(
                self.binding["artifacts"]["evidence_predictions"]["rows"]
            )
            declared_events = int(self.binding["artifacts"]["event_lineage"]["rows"])
            audit = con.execute(
                f"""
                SELECT count(*),
                       count(DISTINCT (cancer_id, lncrna_id, pathway_id)),
                       count_if(analysis_version <> ? OR training_run_id <> ?),
                       count_if(changes_primary_ranking IS DISTINCT FROM false
                                OR main_ranking_modified IS DISTINCT FROM false),
                       count_if(evidence_confidence_probability IS NOT NULL AND
                                (NOT isfinite(evidence_confidence_probability)
                                 OR evidence_confidence_probability NOT BETWEEN 0 AND 1)),
                       count_if(uncertainty IS NOT NULL AND
                                (NOT isfinite(uncertainty) OR uncertainty NOT BETWEEN 0 AND 1)),
                       count_if(availability AND evidence_confidence_probability IS NULL),
                       count_if(NOT availability AND evidence_confidence_probability IS NOT NULL),
                       count_if(NOT availability AND
                                (failure_reason IS NULL OR trim(failure_reason) = ''))
                FROM {predictions}
                """,
                [ANALYSIS_VERSION, self.training_run_id],
            ).fetchone()
            event_audit = con.execute(
                f"""
                SELECT count(*),
                       count_if(family_broadcast_used IS DISTINCT FROM false),
                       count_if(cancer_id IS NULL OR lncrna_id IS NULL OR pathway_id IS NULL)
                FROM {events}
                """
            ).fetchone()
        finally:
            con.close()
        if (
            int(audit[0]) != declared_predictions
            or int(audit[1]) != declared_predictions
            or any(int(value or 0) for value in audit[2:])
            or int(event_audit[0]) != declared_events
            or int(event_audit[1] or 0)
            or int(event_audit[2] or 0)
        ):
            raise EvidenceQueryAssetError(
                f"Evidence artifact semantic validation failed: predictions={audit}, events={event_audit}"
            )

    def _result(
        self,
        kind: str,
        frame: pd.DataFrame,
        filters: dict[str, Any],
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        return {
            "module": "evidence",
            "query_kind": kind,
            "prediction_role": "AUXILIARY_CONFIDENCE_ONLY",
            "changes_primary_ranking": False,
            "filters": filters,
            "limit": limit,
            "offset": offset,
            "returned_rows": len(frame),
            "rows": _records(frame),
            "provenance": {
                "analysis_version": ANALYSIS_VERSION,
                "training_run_id": self.training_run_id,
                "binding_path": str(self.binding_path),
                "binding_sha256": self.binding_sha256,
                "historical_results_used": False,
            },
        }

    def query_confidence(
        self,
        *,
        lncrna_id: Any,
        cancer_id: Any | None = None,
        pathway_id: Any | None = None,
        availability: bool | None = None,
        min_confidence: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        lnc, bare_lnc = _lnc_storage_aliases(lncrna_id)
        clauses = ["lncrna_id IN (?, ?)"]
        parameters: list[Any] = [lnc, bare_lnc]
        cancer = None if cancer_id is None else _clean(cancer_id, "cancer_id", uppercase=True)
        pathway = None if pathway_id is None else _clean(pathway_id, "pathway_id")
        if cancer is not None:
            clauses.append("cancer_id = ?")
            parameters.append(cancer)
        if pathway is not None:
            clauses.append("pathway_id = ?")
            parameters.append(pathway)
        if availability is not None:
            if not isinstance(availability, bool):
                raise EvidenceQueryInputError("availability must be boolean")
            clauses.append("availability = ?")
            parameters.append(availability)
        confidence = None
        if min_confidence is not None:
            confidence = float(min_confidence)
            if not 0 <= confidence <= 1:
                raise EvidenceQueryInputError("min_confidence must be within 0..1")
            clauses.append("evidence_confidence_probability >= ?")
            parameters.append(confidence)
        relation = f"read_parquet({_sql_path(self.paths['evidence_predictions'])})"
        parameters.extend([limit, offset])
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT * FROM {relation}
                WHERE {' AND '.join(clauses)}
                ORDER BY availability DESC, evidence_confidence_probability DESC NULLS LAST,
                         uncertainty, cancer_id, pathway_id
                LIMIT ? OFFSET ?
                """,
                parameters,
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            "confidence",
            frame,
            {
                "lncrna_id": lnc,
                "cancer_id": cancer,
                "pathway_id": pathway,
                "availability": availability,
                "min_confidence": confidence,
            },
            limit,
            offset,
        )

    def query_events(
        self,
        *,
        cancer_id: Any,
        lncrna_id: Any,
        pathway_id: Any,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        cancer = _clean(cancer_id, "cancer_id", uppercase=True)
        lnc, bare_lnc = _lnc_storage_aliases(lncrna_id)
        pathway = _clean(pathway_id, "pathway_id")
        relation = f"read_parquet({_sql_path(self.paths['event_lineage'])})"
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT * FROM {relation}
                WHERE cancer_id = ? AND lncrna_id IN (?, ?) AND pathway_id = ?
                ORDER BY event_id
                LIMIT ? OFFSET ?
                """,
                [cancer, lnc, bare_lnc, pathway, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            "event_lineage",
            frame,
            {"cancer_id": cancer, "lncrna_id": lnc, "pathway_id": pathway},
            limit,
            offset,
        )


__all__ = [
    "EvidenceBindingQuery",
    "EvidenceQueryAssetError",
    "EvidenceQueryError",
    "EvidenceQueryInputError",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
]
