"""Fail-closed website binding for the streaming V3.2 Evidence candidate.

This module deliberately separates a technically queryable candidate from a
publishable Evidence release.  In particular, an independently audited model
whose frozen core lineage is superseded may be mounted on an isolated
candidate API, but it can never be described as release-ready.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import duckdb
import numpy as np
import pandas as pd

from .release_registry import artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_QUERY_BINDING_V2"
PREDICTION_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_PREDICTION_V1"
TRAINING_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_TRAINING_V1"
STAGE_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_STAGE_V1"
INDEPENDENT_AUDIT_FORMAT = (
    "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_INDEPENDENT_AUDIT_V1"
)
SUCCESS_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_QUERY_SUCCESS_V2"
FORMAL_UNIVERSE_ROWS = 3_300_000
FORMAL_AVAILABLE_ROWS = 1_409_003
FORMAL_UNAVAILABLE_ROWS = 1_890_997
FORMAL_PREDICTION_SHA256 = (
    "ba6bf2abb66af9b398bfe43fca9b207811a0b5b37e1f0cd637956a75e8e74435"
)
NO_EVENT_REASON = "NO_EXACT_PATHWAY_EVENT"
EVENT_COUNT_CAP = 64
MAX_QUERY_LIMIT = 1_000
MAX_QUERY_OFFSET = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AUDIT_BLOCKED_STATUS = "BLOCKED_INDEPENDENT_AUDIT_NOT_PUBLISHABLE"
_ELIGIBLE_STATUS = "PASS_QUERY_CANDIDATE_AUDIT_ELIGIBLE"


class StreamingEvidenceWebsiteError(RuntimeError):
    """Base error for candidate binding and query validation."""


class StreamingEvidenceAssetError(StreamingEvidenceWebsiteError):
    """Raised when a hash-pinned Evidence website asset is invalid."""


class StreamingEvidenceInputError(StreamingEvidenceWebsiteError):
    """Raised when an Evidence query filter is invalid."""


def _read_json(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise StreamingEvidenceAssetError(f"{role} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StreamingEvidenceAssetError(f"{role} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise StreamingEvidenceAssetError(f"{role} must be a JSON object")
    return value


def _file(path: str | Path, role: str) -> Path:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise StreamingEvidenceAssetError(f"{role} is missing or unsafe: {unresolved}")
    try:
        value = unresolved.resolve(strict=True)
    except OSError as exc:
        raise StreamingEvidenceAssetError(
            f"{role} is missing or unsafe: {unresolved}"
        ) from exc
    if not value.is_file():
        raise StreamingEvidenceAssetError(f"{role} is missing or unsafe: {value}")
    return value


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _nested_values(value: Any, keys: Iterable[str]) -> list[Any]:
    wanted = set(keys)
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in wanted:
                found.append(child)
            found.extend(_nested_values(child, wanted))
    elif isinstance(value, list):
        for child in value:
            found.extend(_nested_values(child, wanted))
    return found


def _audit_scalar(
    audit: Mapping[str, Any],
    keys: Iterable[str],
    *,
    predicate,
    role: str,
) -> Any:
    values = [value for value in _nested_values(audit, keys) if predicate(value)]
    if not values:
        raise StreamingEvidenceAssetError(f"independent audit lacks {role}")
    unique = {json.dumps(value, sort_keys=True) for value in values}
    if len(unique) != 1:
        raise StreamingEvidenceAssetError(
            f"independent audit has conflicting {role}: {sorted(unique)}"
        )
    return values[0]


def _columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    relation = f"read_parquet({_sql_path(path)})"
    return {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()}


def _validate_prediction_semantics(
    prediction_path: Path,
    *,
    expected_rows: int,
    expected_available: int,
    expected_unavailable: int,
) -> dict[str, Any]:
    required = {
        "cancer_id",
        "lncrna_id",
        "pathway_id",
        "evidence_confidence_probability",
        "direction",
        "uncertainty",
        "availability",
        "unavailable_reason",
        "event_count",
        "evidence_fold",
        "failure_reason",
        "analysis_version",
        "training_run_id",
        "changes_primary_ranking",
        "main_ranking_modified",
    }
    relation = f"read_parquet({_sql_path(prediction_path)})"
    con = duckdb.connect(":memory:")
    try:
        missing = sorted(required - _columns(con, prediction_path))
        if missing:
            raise StreamingEvidenceAssetError(
                f"streaming Evidence predictions lack columns: {missing}"
            )
        row = con.execute(
            f"""
            SELECT
              count(*) AS rows,
              count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS unique_rows,
              count_if(availability) AS available_rows,
              count_if(NOT availability) AS unavailable_rows,
              count_if(availability AND
                       (evidence_confidence_probability IS NULL OR
                        NOT isfinite(evidence_confidence_probability) OR
                        evidence_confidence_probability NOT BETWEEN 0 AND 1 OR
                        uncertainty IS NULL OR NOT isfinite(uncertainty) OR
                        uncertainty NOT BETWEEN 0 AND 1 OR
                        direction NOT IN ('negative','neutral','positive') OR
                        event_count <= 0 OR event_count > ? OR
                        evidence_fold NOT BETWEEN 0 AND 4 OR
                        coalesce(failure_reason, '') <> '')) AS bad_available,
              count_if(NOT availability AND
                       (evidence_confidence_probability IS NOT NULL OR
                        uncertainty IS NOT NULL OR direction IS NOT NULL OR
                        event_count <> 0 OR evidence_fold IS NOT NULL OR
                        failure_reason <> ? OR unavailable_reason <> ?)) AS bad_unavailable,
              count_if(analysis_version <> ? OR
                       training_run_id <> 'V32-EVIDENCE-STREAMING-R1' OR
                       changes_primary_ranking IS DISTINCT FROM false OR
                       main_ranking_modified IS DISTINCT FROM false) AS bad_contract,
              count(DISTINCT cancer_id) AS cancers
            FROM {relation}
            """,
            [EVENT_COUNT_CAP, NO_EVENT_REASON, NO_EVENT_REASON, ANALYSIS_VERSION],
        ).fetchone()
    finally:
        con.close()
    observed = {
        "rows": int(row[0]),
        "unique_rows": int(row[1]),
        "available_rows": int(row[2]),
        "unavailable_rows": int(row[3]),
        "bad_available_rows": int(row[4]),
        "bad_unavailable_rows": int(row[5]),
        "bad_contract_rows": int(row[6]),
        "cancers": int(row[7]),
    }
    expected = {
        "rows": int(expected_rows),
        "unique_rows": int(expected_rows),
        "available_rows": int(expected_available),
        "unavailable_rows": int(expected_unavailable),
        "bad_available_rows": 0,
        "bad_unavailable_rows": 0,
        "bad_contract_rows": 0,
    }
    drift = {
        key: {"observed": observed[key], "expected": value}
        for key, value in expected.items()
        if observed[key] != value
    }
    if drift:
        raise StreamingEvidenceAssetError(
            f"streaming Evidence prediction semantic drift: {drift}"
        )
    return observed


def _validate_event_semantics(
    candidate_events_path: Path,
    event_lineage_path: Path,
    *,
    expected_event_rows: int,
    expected_lineage_rows: int,
) -> dict[str, Any]:
    event_required = {
        "event_id",
        "source_event_id",
        "cancer_id",
        "lncrna_id",
        "pathway_id",
        "source_database",
        "source_dataset",
        "source_record_id",
        "pmid",
        "direction_raw",
        "direction_target",
        "experiment_type",
        "relation_type",
        "is_experimental",
        "is_computational",
        "is_model_prediction",
    }
    lineage_required = {
        "lineage_id",
        "event_id",
        "source_kind",
        "source_row_sha256",
        "mapping_route",
        "family_broadcast_used",
    }
    con = duckdb.connect(":memory:")
    try:
        missing_events = sorted(event_required - _columns(con, candidate_events_path))
        missing_lineage = sorted(lineage_required - _columns(con, event_lineage_path))
        if missing_events or missing_lineage:
            raise StreamingEvidenceAssetError(
                "streaming Evidence event schema drift: "
                f"events={missing_events}, lineage={missing_lineage}"
            )
        events = f"read_parquet({_sql_path(candidate_events_path)})"
        lineage = f"read_parquet({_sql_path(event_lineage_path)})"
        event_audit = con.execute(
            f"""
            SELECT count(*), count(DISTINCT event_id),
                   count_if(cancer_id IS NULL OR trim(cancer_id)='' OR
                            lncrna_id IS NULL OR trim(lncrna_id)='' OR
                            pathway_id IS NULL OR trim(pathway_id)='' OR
                            is_model_prediction IS DISTINCT FROM false)
            FROM {events}
            """
        ).fetchone()
        lineage_audit = con.execute(
            f"""
            SELECT count(*), count_if(family_broadcast_used IS DISTINCT FROM false),
                   count_if(event_id IS NULL OR trim(event_id)='' OR
                            source_row_sha256 IS NULL OR
                            length(source_row_sha256) <> 64)
            FROM {lineage}
            """
        ).fetchone()
    finally:
        con.close()
    observed = {
        "candidate_event_rows": int(event_audit[0]),
        "candidate_unique_event_ids": int(event_audit[1]),
        "candidate_bad_rows": int(event_audit[2]),
        "event_lineage_rows": int(lineage_audit[0]),
        "family_broadcast_rows": int(lineage_audit[1]),
        "bad_lineage_rows": int(lineage_audit[2]),
    }
    if (
        observed["candidate_event_rows"] != int(expected_event_rows)
        or observed["candidate_unique_event_ids"] != int(expected_event_rows)
        or observed["candidate_bad_rows"] != 0
        or observed["event_lineage_rows"] != int(expected_lineage_rows)
        or observed["family_broadcast_rows"] != 0
        or observed["bad_lineage_rows"] != 0
    ):
        raise StreamingEvidenceAssetError(
            f"streaming Evidence event semantic drift: {observed}"
        )
    return observed


def _materialize_exact_event_count_index(
    candidate_events_path: Path,
    prediction_path: Path,
    output_path: Path,
) -> dict[str, int]:
    """Materialize and validate an immutable exact-key count index from events."""

    if output_path.exists():
        raise FileExistsError(f"Evidence event-count index already exists: {output_path}")
    events = f"read_parquet({_sql_path(candidate_events_path)})"
    predictions = f"read_parquet({_sql_path(prediction_path)})"
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            f"""
            COPY (
              SELECT
                upper(cancer_id) AS cancer_id,
                regexp_replace(upper(lncrna_id), '^(LNC|LNCRNA):', '') AS lncrna_id,
                pathway_id,
                count(*)::BIGINT AS total_exact_event_count
              FROM {events}
              GROUP BY 1,2,3
            ) TO {_sql_path(output_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        index_relation = f"read_parquet({_sql_path(output_path)})"
        summary = con.execute(
            f"""
            SELECT
              count(*) AS exact_keys,
              sum(total_exact_event_count)::BIGINT AS event_rows,
              count_if(total_exact_event_count > {EVENT_COUNT_CAP}) AS capped_keys,
              max(total_exact_event_count) AS max_total_exact_event_count,
              count_if(total_exact_event_count <= 0) AS bad_count_rows
            FROM {index_relation}
            """
        ).fetchone()
        mismatch = int(
            con.execute(
                f"""
                SELECT count(*)
                FROM {predictions} p
                FULL JOIN {index_relation} e
                  ON upper(p.cancer_id)=e.cancer_id
                 AND regexp_replace(upper(p.lncrna_id), '^(LNC|LNCRNA):', '')=e.lncrna_id
                 AND p.pathway_id=e.pathway_id
                WHERE
                  p.cancer_id IS NULL OR
                  (p.availability IS DISTINCT FROM (e.cancer_id IS NOT NULL)) OR
                  p.event_count IS DISTINCT FROM
                    CASE WHEN e.cancer_id IS NULL THEN 0
                         ELSE least(e.total_exact_event_count, {EVENT_COUNT_CAP}) END
                """
            ).fetchone()[0]
        )
    finally:
        con.close()
    observed = {
        "exact_keys": int(summary[0]),
        "event_rows": int(summary[1]),
        "capped_keys": int(summary[2]),
        "max_total_exact_event_count": int(summary[3]),
        "bad_count_rows": int(summary[4]),
        "prediction_count_mismatch_rows": mismatch,
    }
    if observed["bad_count_rows"] != 0 or mismatch != 0:
        raise StreamingEvidenceAssetError(
            f"exact event-count index semantic drift: {observed}"
        )
    return observed


def materialize_streaming_evidence_website_binding(
    *,
    prediction_manifest_path: str | Path,
    predictions_path: str | Path,
    training_manifest_path: str | Path,
    stage_manifest_path: str | Path,
    candidate_events_path: str | Path,
    event_lineage_path: str | Path,
    independent_audit_path: str | Path,
    expected_independent_audit_sha256: str,
    output_root: str | Path,
    strict_formal: bool = True,
) -> dict[str, Any]:
    """Create a no-overwrite, hash-pinned isolated website candidate binding."""

    paths = {
        "prediction_manifest": _file(prediction_manifest_path, "prediction manifest"),
        "evidence_predictions": _file(predictions_path, "Evidence predictions"),
        "training_manifest": _file(training_manifest_path, "training manifest"),
        "stage_manifest": _file(stage_manifest_path, "stage manifest"),
        "candidate_exact_events": _file(candidate_events_path, "candidate exact events"),
        "event_lineage": _file(event_lineage_path, "event lineage"),
        "independent_audit": _file(independent_audit_path, "independent audit"),
    }
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"Evidence website binding refuses output reuse: {destination}")

    prediction_manifest = _read_json(paths["prediction_manifest"], "prediction manifest")
    training_manifest = _read_json(paths["training_manifest"], "training manifest")
    stage_manifest = _read_json(paths["stage_manifest"], "stage manifest")
    audit = _read_json(paths["independent_audit"], "independent audit")
    hashes = {role: artifact_sha256(path) for role, path in paths.items()}

    expected_audit_sha = str(expected_independent_audit_sha256 or "").lower()
    if not _SHA256.fullmatch(expected_audit_sha):
        raise StreamingEvidenceAssetError(
            "expected independent audit SHA256 is required"
        )
    if hashes["independent_audit"] != expected_audit_sha:
        raise StreamingEvidenceAssetError(
            "independent audit SHA mismatch: "
            f"{hashes['independent_audit']} != {expected_audit_sha}"
        )

    if (
        prediction_manifest.get("format") != PREDICTION_FORMAT
        or prediction_manifest.get("status")
        != "PASS_FRESH_V32_EVIDENCE_STREAMING_PREDICTION"
        or prediction_manifest.get("analysis_version") != ANALYSIS_VERSION
        or prediction_manifest.get("production_deployed") is not False
        or prediction_manifest.get("port_8260_touched") is not False
    ):
        raise StreamingEvidenceAssetError("prediction manifest is not the isolated V3.2 candidate")
    if (
        training_manifest.get("format") != TRAINING_FORMAT
        or training_manifest.get("status")
        != "PASS_FRESH_V32_EVIDENCE_STREAMING_TRAINING"
        or training_manifest.get("analysis_version") != ANALYSIS_VERSION
        or training_manifest.get("production_deployed") is not False
        or training_manifest.get("port_8260_touched") is not False
    ):
        raise StreamingEvidenceAssetError("training manifest contract drift")
    if (
        stage_manifest.get("format") != STAGE_FORMAT
        or stage_manifest.get("strict_formal_3_3m_contract") is not True
        or stage_manifest.get("production_deployed") is not False
        or stage_manifest.get("production_port_8260_touched") is not False
    ):
        raise StreamingEvidenceAssetError("stage manifest contract drift")

    declared_prediction_sha = str(prediction_manifest.get("predictions_sha256", ""))
    if declared_prediction_sha != hashes["evidence_predictions"]:
        raise StreamingEvidenceAssetError("prediction manifest/output SHA mismatch")
    if prediction_manifest.get("training_manifest_sha256") != hashes["training_manifest"]:
        raise StreamingEvidenceAssetError("prediction/training manifest SHA mismatch")
    stage_artifacts = stage_manifest.get("artifacts")
    if not isinstance(stage_artifacts, Mapping):
        raise StreamingEvidenceAssetError("stage manifest lacks artifacts")
    for role, filename in (
        ("candidate_exact_events", "candidate_exact_events.parquet"),
        ("event_lineage", "event_lineage.parquet"),
    ):
        declaration = stage_artifacts.get(filename)
        if not isinstance(declaration, Mapping) or declaration.get("sha256") != hashes[role]:
            raise StreamingEvidenceAssetError(f"stage manifest SHA drift: {filename}")

    rows = int(prediction_manifest.get("prediction_rows", -1))
    available = int(prediction_manifest.get("available_rows", -1))
    unavailable = int(prediction_manifest.get("unavailable_rows", -1))
    if rows <= 0 or available < 0 or unavailable < 0 or available + unavailable != rows:
        raise StreamingEvidenceAssetError("prediction coverage counts are invalid")
    if strict_formal and (
        rows != FORMAL_UNIVERSE_ROWS
        or available != FORMAL_AVAILABLE_ROWS
        or unavailable != FORMAL_UNAVAILABLE_ROWS
        or hashes["evidence_predictions"] != FORMAL_PREDICTION_SHA256
    ):
        raise StreamingEvidenceAssetError("formal 3.3M Evidence authority drift")

    if (
        audit.get("format") != INDEPENDENT_AUDIT_FORMAT
        or audit.get("analysis_version") != ANALYSIS_VERSION
        or audit.get("auditor_independent_of_training_implementation") is not True
        or audit.get("training_module_imported") is not False
        or audit.get("sealed_test_opened") is not False
        or audit.get("production_deployed") is not False
        or audit.get("port_8260_touched") is not False
    ):
        raise StreamingEvidenceAssetError("independent audit identity/containment drift")
    audit_prediction = audit.get("prediction")
    if not isinstance(audit_prediction, Mapping):
        raise StreamingEvidenceAssetError("independent audit lacks canonical prediction block")
    audit_prediction_sha = str(audit_prediction.get("predictions_sha256", "")).lower()
    if not _SHA256.fullmatch(audit_prediction_sha):
        raise StreamingEvidenceAssetError(
            "independent audit lacks canonical prediction.predictions_sha256"
        )
    canonical_counts: dict[str, int] = {}
    for name in ("universe_rows", "prediction_rows", "available_rows", "unavailable_rows"):
        value = audit.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise StreamingEvidenceAssetError(
                f"independent audit lacks canonical integer {name}"
            )
        canonical_counts[name] = int(value)
    if (
        audit_prediction_sha != hashes["evidence_predictions"]
        or canonical_counts["universe_rows"] != rows
        or canonical_counts["prediction_rows"] != rows
        or canonical_counts["available_rows"] != available
        or canonical_counts["unavailable_rows"] != unavailable
    ):
        raise StreamingEvidenceAssetError("independent audit is not bound to this prediction")
    audit_status = audit.get("status")
    publication_allowed = audit.get("publication_allowed")
    closure_allowed = audit.get("closure_allowed")
    technical_acceptance = audit.get("technical_acceptance")
    technical_prediction_pass = audit.get("technical_prediction_pass")
    technical_failures = audit.get("technical_failures")
    publication_blockers = audit.get("publication_blockers")
    if not isinstance(publication_allowed, bool):
        raise StreamingEvidenceAssetError(
            "independent audit must declare canonical boolean publication_allowed"
        )
    if closure_allowed is not publication_allowed:
        raise StreamingEvidenceAssetError(
            "independent audit closure/publication verdict conflict"
        )
    if (
        technical_acceptance is not True
        or technical_prediction_pass is not True
        or not isinstance(technical_failures, list)
        or any(not isinstance(value, str) or not value for value in technical_failures)
        or technical_failures
    ):
        raise StreamingEvidenceAssetError(
            f"independent audit has technical failures: {technical_failures!r}"
        )
    if (
        not isinstance(publication_blockers, list)
        or any(not isinstance(value, str) or not value for value in publication_blockers)
    ):
        raise StreamingEvidenceAssetError("independent audit publication blockers are invalid")
    if publication_allowed:
        if audit_status != "PASS_INDEPENDENT_AUDIT_PUBLICATION_ALLOWED" or publication_blockers:
            raise StreamingEvidenceAssetError(
                "independent audit PASS/publication verdict is inconsistent"
            )
    elif (
        audit_status != _AUDIT_BLOCKED_STATUS
        or not publication_blockers
    ):
        raise StreamingEvidenceAssetError(
            "independent audit BLOCKED/publication verdict is inconsistent"
        )
    core_blocker_id = "fresh_current_g2_core_lineage"
    core_values = _nested_values(
        audit,
        ("core_lineage_verdict", "core_lineage_status", "core_lineage"),
    )
    declared_core_lineage_verdict = next(
        (str(value) for value in core_values if isinstance(value, str) and value.strip()),
        None,
    )
    publication_blocked_by_core_lineage = core_blocker_id in publication_blockers
    audit_checks = audit.get("checks")
    core_lineage_check = (
        audit_checks.get(core_blocker_id)
        if isinstance(audit_checks, Mapping)
        else None
    )
    if publication_blocked_by_core_lineage:
        if (
            isinstance(core_lineage_check, Mapping)
            and core_lineage_check.get("passed") is not False
        ):
            raise StreamingEvidenceAssetError(
                "independent audit core-lineage blocker/check verdict conflict"
            )
        core_lineage_verdict = (
            declared_core_lineage_verdict
            or "BLOCKED_FRESH_CURRENT_G2_CORE_LINEAGE"
        )
    else:
        if (
            isinstance(core_lineage_check, Mapping)
            and core_lineage_check.get("passed") is False
        ):
            raise StreamingEvidenceAssetError(
                "independent audit failed core-lineage check lacks publication blocker"
            )
        core_lineage_verdict = (
            declared_core_lineage_verdict
            or (
                "PASS_FRESH_CURRENT_G2_CORE_LINEAGE"
                if isinstance(core_lineage_check, Mapping)
                and core_lineage_check.get("passed") is True
                else "NOT_DECLARED"
            )
        )

    prediction_semantics = _validate_prediction_semantics(
        paths["evidence_predictions"],
        expected_rows=rows,
        expected_available=available,
        expected_unavailable=unavailable,
    )
    stage_counts = stage_manifest.get("counts")
    if not isinstance(stage_counts, Mapping):
        raise StreamingEvidenceAssetError("stage manifest lacks counts")
    event_semantics = _validate_event_semantics(
        paths["candidate_exact_events"],
        paths["event_lineage"],
        expected_event_rows=int(stage_counts.get("candidate_exact_events", -1)),
        expected_lineage_rows=int(stage_counts.get("event_lineage", -1)),
    )

    destination.mkdir(parents=True)
    event_count_index_path = destination / "EXACT_EVENT_COUNTS.parquet"
    event_count_index_semantics = _materialize_exact_event_count_index(
        paths["candidate_exact_events"],
        paths["evidence_predictions"],
        event_count_index_path,
    )
    if event_count_index_semantics["event_rows"] != event_semantics["candidate_event_rows"]:
        raise StreamingEvidenceAssetError(
            "exact event-count index does not sum to the candidate event authority"
        )
    audit_checks = audit.get("checks")
    cap_audit = (
        audit_checks.get("event_cap_quality_priority")
        if isinstance(audit_checks, Mapping)
        else None
    )
    if not isinstance(cap_audit, Mapping) or (
        cap_audit.get("capped_bags") != event_count_index_semantics["capped_keys"]
        or cap_audit.get("max_raw_event_count")
        != event_count_index_semantics["max_total_exact_event_count"]
    ):
        raise StreamingEvidenceAssetError(
            "independent audit event-cap counts do not bind the exact event index"
        )

    status = _ELIGIBLE_STATUS if publication_allowed else _AUDIT_BLOCKED_STATUS
    artifacts = {
        role: {
            "path": str(path),
            "sha256": hashes[role],
            "bytes": path.stat().st_size,
        }
        for role, path in paths.items()
    }
    artifacts["exact_event_counts"] = {
        "path": str(event_count_index_path),
        "sha256": artifact_sha256(event_count_index_path),
        "bytes": event_count_index_path.stat().st_size,
        "rows": event_count_index_semantics["exact_keys"],
        "source_candidate_exact_events_sha256": hashes["candidate_exact_events"],
        "aggregation": "exact COUNT(*) by cancer_id, normalized lncrna_id, pathway_id",
    }
    artifacts["evidence_predictions"]["rows"] = rows
    artifacts["candidate_exact_events"]["rows"] = event_semantics["candidate_event_rows"]
    artifacts["event_lineage"]["rows"] = event_semantics["event_lineage_rows"]
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": status,
        "candidate_only": True,
        "mountable_on_isolated_candidate_api": True,
        "release_ready": False,
        "independent_audit_status": audit_status,
        "technical_acceptance": True,
        "technical_prediction_pass": True,
        "technical_failures": [],
        "publication_blockers": list(publication_blockers),
        "publication_allowed_by_independent_audit": publication_allowed,
        "publication_blocked_by_core_lineage": publication_blocked_by_core_lineage,
        "core_lineage_verdict": core_lineage_verdict,
        "production_deployed": False,
        "port_8260_touched": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "changes_primary_ranking": False,
        "prediction_role": "AUXILIARY_CONFIDENCE_ONLY",
        "training_run_id": "V32-EVIDENCE-STREAMING-R1",
        "coverage": {
            "universe_rows": rows,
            "available_rows": available,
            "typed_unavailable_rows": unavailable,
            "typed_unavailable_reason": NO_EVENT_REASON,
            "cancers": prediction_semantics["cancers"],
        },
        "query_contract": {
            "exact_key": ["cancer_id", "lncrna_id", "pathway_id"],
            "lncrna_aliases": ["ENSG...", "LNC:ENSG...", "LNCRNA:ENSG..."],
            "available_has_numeric_probability": True,
            "typed_unavailable_is_null_never_zero": True,
            "event_sources_and_directions_exposed": True,
            "no_event_query_returns_empty_event_rows": True,
            "model_consumed_event_count_field": "model_consumed_event_count",
            "event_count_cap": EVENT_COUNT_CAP,
            "event_count_capped_field": "event_count_capped",
            "total_exact_event_count_field": "total_exact_event_count",
            "total_exact_event_count_source": "candidate_exact_events.parquet exact-key COUNT(*)",
            "download_artifacts": ["evidence_predictions", "candidate_exact_events"],
        },
        "semantic_validation": {
            "predictions": prediction_semantics,
            "events": event_semantics,
            "exact_event_count_index": event_count_index_semantics,
        },
        "artifacts": artifacts,
        "path_drift_note": (
            "Binding uses the real resolved files and their hashes; it does not trust "
            "the older COMPLETION.json direct-result-root path claims."
        ),
    }
    binding_path = destination / "EVIDENCE_QUERY_BINDING.json"
    binding_path.write_text(
        json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    binding_sha = artifact_sha256(binding_path)
    success = {
        "format": SUCCESS_FORMAT,
        "status": status,
        "candidate_only": True,
        "release_ready": False,
        "independent_audit_status": audit_status,
        "technical_acceptance": True,
        "technical_prediction_pass": True,
        "publication_allowed_by_independent_audit": publication_allowed,
        "publication_blocked_by_core_lineage": publication_blocked_by_core_lineage,
        "production_deployed": False,
        "port_8260_touched": False,
        "binding": binding_path.name,
        "binding_sha256": binding_sha,
    }
    success_path = destination / "SUCCESS.json"
    success_path.write_text(
        json.dumps(success, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": status,
        "binding_path": str(binding_path),
        "binding_sha256": binding_sha,
        "success_path": str(success_path),
        "success_sha256": artifact_sha256(success_path),
        "candidate_only": True,
        "release_ready": False,
        "publication_blocked_by_core_lineage": publication_blocked_by_core_lineage,
    }


def _canonical_lnc(value: Any) -> tuple[str, str]:
    text = str(value or "").strip().upper()
    text = re.sub(r"^(?:LNC|LNCRNA):", "", text)
    text = re.sub(r"\.\d+$", "", text)
    if not text or len(text) > 64 or any(ord(character) < 32 for character in text):
        raise StreamingEvidenceInputError("lncrna_id is required and must be valid")
    return "LNC:" + text, text


def _clean(value: Any, role: str, *, uppercase: bool = False) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 512 or any(ord(character) < 32 for character in text):
        raise StreamingEvidenceInputError(f"{role} is required and must be valid")
    return text.upper() if uppercase else text


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise StreamingEvidenceInputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise StreamingEvidenceInputError(f"offset must be 0..{MAX_QUERY_OFFSET}")
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


class StreamingEvidenceWebsiteQuery:
    """Read-only exact-pair query over one immutable website candidate."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
    ) -> None:
        source = _file(binding_path, "Evidence streaming website binding")
        expected = str(expected_binding_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise StreamingEvidenceAssetError("expected binding SHA256 is required")
        observed = artifact_sha256(source)
        if observed != expected:
            raise StreamingEvidenceAssetError(
                f"Evidence website binding SHA mismatch: {observed} != {expected}"
            )
        binding = _read_json(source, "Evidence streaming website binding")
        if (
            binding.get("format") != BINDING_FORMAT
            or binding.get("analysis_version") != ANALYSIS_VERSION
            or binding.get("status") not in {_AUDIT_BLOCKED_STATUS, _ELIGIBLE_STATUS}
            or binding.get("candidate_only") is not True
            or binding.get("mountable_on_isolated_candidate_api") is not True
            or binding.get("release_ready") is not False
            or binding.get("production_deployed") is not False
            or binding.get("port_8260_touched") is not False
            or binding.get("changes_primary_ranking") is not False
            or binding.get("technical_acceptance") is not True
            or binding.get("technical_prediction_pass") is not True
            or binding.get("technical_failures") != []
        ):
            raise StreamingEvidenceAssetError("Evidence website candidate contract drift")
        publication_allowed = binding.get("publication_allowed_by_independent_audit")
        if (
            not isinstance(publication_allowed, bool)
            or (binding.get("status") == _ELIGIBLE_STATUS) != publication_allowed
            or binding.get("independent_audit_status")
            != (
                "PASS_INDEPENDENT_AUDIT_PUBLICATION_ALLOWED"
                if publication_allowed
                else _AUDIT_BLOCKED_STATUS
            )
        ):
            raise StreamingEvidenceAssetError(
                "Evidence website candidate audit verdict drift"
            )
        success_path = source.parent / "SUCCESS.json"
        success = _read_json(success_path, "Evidence website SUCCESS")
        if (
            success.get("format") != SUCCESS_FORMAT
            or success.get("status") != binding["status"]
            or success.get("binding") != source.name
            or success.get("binding_sha256") != observed
            or success.get("candidate_only") is not True
            or success.get("release_ready") is not False
            or success.get("production_deployed") is not False
            or success.get("port_8260_touched") is not False
            or success.get("independent_audit_status")
            != binding.get("independent_audit_status")
            or success.get("technical_acceptance") is not True
            or success.get("technical_prediction_pass") is not True
        ):
            raise StreamingEvidenceAssetError("Evidence website SUCCESS marker is stale")
        declarations = binding.get("artifacts")
        if not isinstance(declarations, Mapping):
            raise StreamingEvidenceAssetError("Evidence website binding lacks artifacts")
        paths: dict[str, Path] = {}
        for role in (
            "evidence_predictions",
            "candidate_exact_events",
            "event_lineage",
            "exact_event_counts",
            "independent_audit",
        ):
            declaration = declarations.get(role)
            if not isinstance(declaration, Mapping):
                raise StreamingEvidenceAssetError(f"Evidence website binding lacks {role}")
            path = _file(str(declaration.get("path", "")), role)
            if artifact_sha256(path) != declaration.get("sha256"):
                raise StreamingEvidenceAssetError(f"Evidence website artifact SHA drift: {role}")
            paths[role] = path
        self.binding = binding
        self.binding_path = source
        self.binding_sha256 = observed
        self.paths = paths

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(":memory:")

    def capability(self) -> dict[str, Any]:
        return {
            "module": "evidence",
            "analysis_version": ANALYSIS_VERSION,
            "status": self.binding["status"],
            "candidate_only": True,
            "release_ready": False,
            "publication_allowed_by_independent_audit": self.binding[
                "publication_allowed_by_independent_audit"
            ],
            "publication_blocked_by_core_lineage": self.binding[
                "publication_blocked_by_core_lineage"
            ],
            "core_lineage_verdict": self.binding["core_lineage_verdict"],
            "prediction_role": "AUXILIARY_CONFIDENCE_ONLY",
            "changes_primary_ranking": False,
            "coverage": self.binding["coverage"],
            "query_contract": self.binding["query_contract"],
            "provenance": {
                "binding_path": str(self.binding_path),
                "binding_sha256": self.binding_sha256,
                "independent_audit_sha256": self.binding["artifacts"][
                    "independent_audit"
                ]["sha256"],
            },
        }

    def _result(
        self,
        kind: str,
        frame: pd.DataFrame,
        filters: Mapping[str, Any],
        *,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        return {
            "module": "evidence",
            "query_kind": kind,
            "candidate_only": True,
            "release_ready": False,
            "publication_blocked_by_core_lineage": self.binding[
                "publication_blocked_by_core_lineage"
            ],
            "prediction_role": "AUXILIARY_CONFIDENCE_ONLY",
            "changes_primary_ranking": False,
            "filters": dict(filters),
            "limit": limit,
            "offset": offset,
            "returned_rows": len(frame),
            "rows": _records(frame),
            "provenance": {
                "analysis_version": ANALYSIS_VERSION,
                "training_run_id": self.binding["training_run_id"],
                "binding_sha256": self.binding_sha256,
                "independent_audit_sha256": self.binding["artifacts"][
                    "independent_audit"
                ]["sha256"],
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
        canonical_lnc, bare_lnc = _canonical_lnc(lncrna_id)
        cancer = None if cancer_id is None else _clean(cancer_id, "cancer_id", uppercase=True)
        pathway = None if pathway_id is None else _clean(pathway_id, "pathway_id")
        clauses = ["regexp_replace(upper(lncrna_id), '^(LNC|LNCRNA):', '') = ?"]
        parameters: list[Any] = [bare_lnc]
        if cancer is not None:
            clauses.append("upper(cancer_id) = ?")
            parameters.append(cancer)
        if pathway is not None:
            clauses.append("pathway_id = ?")
            parameters.append(pathway)
        if availability is not None:
            if not isinstance(availability, bool):
                raise StreamingEvidenceInputError("availability must be boolean")
            clauses.append("availability = ?")
            parameters.append(availability)
        confidence = None
        if min_confidence is not None:
            confidence = float(min_confidence)
            if not 0 <= confidence <= 1:
                raise StreamingEvidenceInputError("min_confidence must be within 0..1")
            clauses.append("evidence_confidence_probability >= ?")
            parameters.append(confidence)
        relation = f"read_parquet({_sql_path(self.paths['evidence_predictions'])})"
        event_counts = f"read_parquet({_sql_path(self.paths['exact_event_counts'])})"
        parameters.extend([limit, offset])
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                WITH selected_predictions AS (
                  SELECT *
                  FROM {relation}
                  WHERE {' AND '.join(clauses)}
                  ORDER BY availability DESC,
                           evidence_confidence_probability DESC NULLS LAST,
                           uncertainty NULLS LAST, cancer_id, pathway_id
                  LIMIT ? OFFSET ?
                ), event_totals AS (
                  SELECT
                    p.cancer_id,
                    p.lncrna_id,
                    p.pathway_id,
                    coalesce(max(e.total_exact_event_count), 0)::BIGINT
                      AS total_exact_event_count
                  FROM selected_predictions p
                  LEFT JOIN {event_counts} e
                    ON upper(p.cancer_id)=e.cancer_id
                   AND regexp_replace(upper(p.lncrna_id), '^(LNC|LNCRNA):', '')
                       = e.lncrna_id
                   AND p.pathway_id=e.pathway_id
                  GROUP BY p.cancer_id, p.lncrna_id, p.pathway_id
                )
                SELECT
                  p.cancer_id,
                  'LNC:' || regexp_replace(upper(p.lncrna_id),
                                             '^(LNC|LNCRNA):', '') AS lncrna_id,
                  p.pathway_id,
                  p.evidence_confidence_probability,
                  p.direction,
                  p.uncertainty,
                  p.availability,
                  p.unavailable_reason,
                  p.event_count::BIGINT AS model_consumed_event_count,
                  {EVENT_COUNT_CAP}::BIGINT AS event_count_cap,
                  (t.total_exact_event_count > p.event_count) AS event_count_capped,
                  t.total_exact_event_count,
                  p.evidence_fold,
                  p.failure_reason,
                  p.analysis_version,
                  p.training_run_id,
                  p.changes_primary_ranking,
                  p.main_ranking_modified
                FROM selected_predictions p
                JOIN event_totals t
                  USING(cancer_id, lncrna_id, pathway_id)
                ORDER BY p.availability DESC,
                         p.evidence_confidence_probability DESC NULLS LAST,
                         p.uncertainty NULLS LAST, p.cancer_id, p.pathway_id
                """,
                parameters,
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            "confidence",
            frame,
            {
                "lncrna_id": canonical_lnc,
                "cancer_id": cancer,
                "pathway_id": pathway,
                "availability": availability,
                "min_confidence": confidence,
            },
            limit=limit,
            offset=offset,
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
        canonical_lnc, bare_lnc = _canonical_lnc(lncrna_id)
        pathway = _clean(pathway_id, "pathway_id")
        events = f"read_parquet({_sql_path(self.paths['candidate_exact_events'])})"
        lineage = f"read_parquet({_sql_path(self.paths['event_lineage'])})"
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                WITH matching_events AS (
                  SELECT * FROM {events}
                  WHERE upper(cancer_id)=?
                    AND regexp_replace(upper(lncrna_id), '^(LNC|LNCRNA):', '')=?
                    AND pathway_id=?
                ), one_lineage AS (
                  SELECT * EXCLUDE(_rn) FROM (
                    SELECT *, row_number() OVER (
                      PARTITION BY event_id ORDER BY lineage_id
                    ) AS _rn
                    FROM {lineage}
                    WHERE event_id IN (SELECT source_event_id FROM matching_events)
                  ) WHERE _rn=1
                )
                SELECT
                  e.event_id,
                  e.source_event_id,
                  e.cancer_id,
                  'LNC:' || regexp_replace(upper(e.lncrna_id),
                                             '^(LNC|LNCRNA):', '') AS lncrna_id,
                  e.pathway_id,
                  e.partner_id,
                  e.member_type,
                  e.route_type,
                  e.source_database,
                  e.source_dataset,
                  e.source_record_id,
                  nullif(e.pmid, '') AS pmid,
                  e.experiment_type,
                  e.relation_type,
                  e.direction_raw,
                  e.direction_target,
                  e.confidence_target,
                  e.tissue,
                  e.cell_line,
                  e.species,
                  e.is_experimental,
                  e.is_computational,
                  e.is_physical,
                  e.is_model_prediction,
                  l.lineage_id,
                  l.source_kind,
                  l.mapping_route,
                  l.source_row_sha256,
                  coalesce(l.family_broadcast_used, false) AS family_broadcast_used
                FROM matching_events e
                LEFT JOIN one_lineage l ON l.event_id=e.source_event_id
                ORDER BY e.event_id, l.lineage_id
                LIMIT ? OFFSET ?
                """,
                [cancer, bare_lnc, pathway, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            "event_sources",
            frame,
            {
                "cancer_id": cancer,
                "lncrna_id": canonical_lnc,
                "pathway_id": pathway,
            },
            limit=limit,
            offset=offset,
        )

    def download(self, artifact_id: str) -> dict[str, Any]:
        role = str(artifact_id or "").strip()
        allowed = set(self.binding["query_contract"]["download_artifacts"])
        if role not in allowed:
            raise StreamingEvidenceInputError(
                f"unknown Evidence download artifact: {role}"
            )
        declaration = self.binding["artifacts"][role]
        return {
            "artifact_id": role,
            "path": str(self.paths[role]),
            "filename": self.paths[role].name,
            "media_type": "application/vnd.apache.parquet",
            "bytes": int(declaration["bytes"]),
            "sha256": declaration["sha256"],
            "rows": int(declaration["rows"]),
            "candidate_only": True,
            "release_ready": False,
            "publication_blocked_by_core_lineage": self.binding[
                "publication_blocked_by_core_lineage"
            ],
        }


__all__ = [
    "ANALYSIS_VERSION",
    "BINDING_FORMAT",
    "EVENT_COUNT_CAP",
    "INDEPENDENT_AUDIT_FORMAT",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
    "StreamingEvidenceAssetError",
    "StreamingEvidenceInputError",
    "StreamingEvidenceWebsiteError",
    "StreamingEvidenceWebsiteQuery",
    "materialize_streaming_evidence_website_binding",
]
