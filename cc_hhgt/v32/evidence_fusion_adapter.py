"""Canonical full-universe adapter for the fresh V3.2 Evidence expert.

The freshly trained Evidence head intentionally used the training identifiers
(``ENSG...`` and source pathway spelling).  The authoritative exact-pathway
candidate universe uses ``LNC:ENSG...`` and canonical pathway spelling.  Raw
Evidence predictions therefore must never be joined directly to the primary
release.  This module performs one fail-closed, hash-bound normalization and
proves a bijection with the complete candidate universe before producing a
confidence-only fusion expert.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import duckdb

from .evidence_output_binding import ANALYSIS_VERSION
from .evidence_semantic_wrapper import (
    SEMANTIC_WRAPPER_FORMAT,
    SEMANTIC_WRAPPER_STATUS,
)


ADAPTER_FORMAT = "CC_HHGT_V3_2_EVIDENCE_CANONICAL_FUSION_ADAPTER_V1"
BINDING_FORMAT = "CC_HHGT_V3_2_EVIDENCE_CANONICAL_FUSION_BINDING_V1"
AUDIT_FORMAT = "CC_HHGT_V3_2_EVIDENCE_CANONICAL_FUSION_AUDIT_V1"
R2_BINDING_FORMAT = "CC_HHGT_V3_2_EVIDENCE_OUTPUT_BINDING_V1"
FORMAL_CANDIDATE_ROWS = 3_300_000
FORMAL_AVAILABLE_ROWS = 825_753
FORMAL_UNAVAILABLE_ROWS = 2_474_247
FORMAL_CANDIDATE_SHA256 = (
    "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
)
FORMAL_RAW_PREDICTION_SHA256 = (
    "9595732bf91be5499d31c0b23d40c461df37c7821781687f8303d879a0ce3451"
)


class EvidenceFusionAdapterError(RuntimeError):
    """Raised when Evidence identifiers cannot be mapped bijectively."""


def artifact_sha256(path: str | Path) -> str:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise EvidenceFusionAdapterError(f"Missing or unsafe file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path, role: str) -> tuple[Path, dict[str, Any]]:
    requested = Path(path)
    if requested.is_symlink():
        raise EvidenceFusionAdapterError(f"{role} is unsafe: {requested}")
    source = requested.resolve()
    if not source.is_file():
        raise EvidenceFusionAdapterError(f"{role} is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceFusionAdapterError(f"{role} is invalid JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise EvidenceFusionAdapterError(f"{role} must contain a JSON object")
    return source, payload


def _require(payload: Mapping[str, Any], expected: Mapping[str, Any], role: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise EvidenceFusionAdapterError(
                f"{role}.{key} drift: observed={payload.get(key)!r}, expected={value!r}"
            )


def _bound_record(record: Any, role: str, *, rows: int | None = None) -> tuple[Path, str]:
    if not isinstance(record, Mapping):
        raise EvidenceFusionAdapterError(f"{role} record is missing")
    source = Path(str(record.get("path", ""))).resolve()
    expected_sha = str(record.get("sha256", "")).lower()
    if artifact_sha256(source) != expected_sha:
        raise EvidenceFusionAdapterError(f"{role} path/SHA256 drift")
    if rows is not None and int(record.get("rows", -1)) != int(rows):
        raise EvidenceFusionAdapterError(f"{role} row-count drift")
    return source, expected_sha


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _validate_semantic_wrapper(
    path: str | Path,
    expected_sha256: str,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path]:
    wrapper_path, wrapper = _read_json(path, "Evidence semantic wrapper")
    if artifact_sha256(wrapper_path) != str(expected_sha256).lower():
        raise EvidenceFusionAdapterError("Evidence semantic wrapper SHA256 drift")
    _require(
        wrapper,
        {
            "format": SEMANTIC_WRAPPER_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": SEMANTIC_WRAPPER_STATUS,
            "release_ready": False,
            "production_deployed": False,
            "direct_target_evidence": True,
            "direct_exact_pathway_assertion": False,
            "unavailable_encoding": "null_with_reason",
            "confidence_only": True,
            "changes_primary_ranking": False,
            "affects_discovery": False,
            "affects_primary_ranking": False,
            "raw_prediction_direct_fusion_allowed": False,
            "canonical_candidate_adapter_required": True,
            "lncrna_identifier_normalization": (
                "LNC:ENSG_TO_ENSG_FOR_TRAINING__RESTORE_LNC_PREFIX_FOR_FUSION"
            ),
            "pathway_identifier_normalization": (
                "TRIM_AND_UPPERCASE_TO_CANONICAL_PATHWAY_FOR_FUSION"
            ),
            "five_fresh_private_heads_verified": True,
        },
        "Evidence semantic wrapper",
    )
    r2_path, _ = _bound_record(wrapper.get("r2_evidence_binding"), "R2 Evidence binding")
    _, r2 = _read_json(r2_path, "R2 Evidence binding")
    _require(
        r2,
        {
            "format": R2_BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND",
            "release_ready": False,
            "production_deployed": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "family_to_exact_broadcast": False,
            "five_fresh_private_heads_verified": True,
        },
        "R2 Evidence binding",
    )
    if int(r2.get("optimizer_steps_total", 0)) <= 0:
        raise EvidenceFusionAdapterError("R2 Evidence binding has no optimizer updates")
    post_audit, _ = _bound_record(
        wrapper.get("independent_post_audit"), "Evidence independent post-audit"
    )
    return wrapper_path, wrapper, r2_path, r2, post_audit


def materialize_evidence_fusion_adapter(
    *,
    candidates_path: str | Path,
    semantic_wrapper_path: str | Path,
    expected_semantic_wrapper_sha256: str,
    output_root: str | Path,
    strict_formal: bool = True,
) -> dict[str, Any]:
    """Normalize Evidence keys and prove exact full-universe closure."""

    candidates = Path(candidates_path).resolve()
    if not candidates.is_file() or candidates.is_symlink():
        raise EvidenceFusionAdapterError(f"Candidate authority is missing or unsafe: {candidates}")
    candidate_sha = artifact_sha256(candidates)
    wrapper_path, wrapper, r2_path, r2, post_audit = _validate_semantic_wrapper(
        semantic_wrapper_path, expected_semantic_wrapper_sha256
    )
    artifacts = r2.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EvidenceFusionAdapterError("R2 Evidence binding lacks artifacts")
    raw_prediction, raw_prediction_sha = _bound_record(
        artifacts.get("evidence_predictions"),
        "raw Evidence prediction",
        rows=FORMAL_CANDIDATE_ROWS if strict_formal else None,
    )
    if strict_formal:
        if candidate_sha != FORMAL_CANDIDATE_SHA256:
            raise EvidenceFusionAdapterError("Formal candidate authority SHA256 drift")
        if raw_prediction_sha != FORMAL_RAW_PREDICTION_SHA256:
            raise EvidenceFusionAdapterError("Formal raw Evidence prediction SHA256 drift")

    output = Path(output_root).resolve()
    if output.exists():
        raise EvidenceFusionAdapterError(f"Refusing to reuse adapter output: {output}")
    output.mkdir(parents=True)
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            f"""
            CREATE TEMP TABLE candidate_norm AS
            SELECT trim(CAST(cancer_id AS VARCHAR)) AS cancer_id,
                   trim(CAST(lncrna_id AS VARCHAR)) AS lncrna_id,
                   trim(CAST(pathway_id AS VARCHAR)) AS pathway_id,
                   upper(trim(CAST(cancer_id AS VARCHAR))) AS cancer_norm,
                   upper(trim(CAST(lncrna_id AS VARCHAR))) AS lncrna_norm,
                   upper(trim(CAST(pathway_id AS VARCHAR))) AS pathway_norm
            FROM read_parquet({_sql_path(candidates)})
            """
        )
        candidate_summary = connection.execute(
            """
            SELECT count(*) AS rows,
                   count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS distinct_keys,
                   count(DISTINCT (cancer_norm, lncrna_norm, pathway_norm)) AS distinct_norm_keys,
                   count_if(cancer_id = '' OR lncrna_id = '' OR pathway_id = '') AS empty_keys,
                   count_if(NOT starts_with(lncrna_id, 'LNC:')) AS missing_lnc_prefix
            FROM candidate_norm
            """
        ).fetchone()
        candidate_rows = int(candidate_summary[0])
        if (
            candidate_rows < 1
            or int(candidate_summary[1]) != candidate_rows
            or int(candidate_summary[2]) != candidate_rows
            or int(candidate_summary[3]) != 0
            or int(candidate_summary[4]) != 0
        ):
            raise EvidenceFusionAdapterError(
                f"Candidate normalized-key authority is ambiguous: {candidate_summary}"
            )
        if strict_formal and candidate_rows != FORMAL_CANDIDATE_ROWS:
            raise EvidenceFusionAdapterError("Formal candidate row count drift")

        connection.execute(
            f"""
            CREATE TEMP TABLE evidence_norm AS
            SELECT upper(trim(CAST(cancer_id AS VARCHAR))) AS cancer_norm,
                   CASE
                     WHEN starts_with(upper(trim(CAST(lncrna_id AS VARCHAR))), 'LNC:')
                       THEN upper(trim(CAST(lncrna_id AS VARCHAR)))
                     ELSE 'LNC:' || upper(trim(CAST(lncrna_id AS VARCHAR)))
                   END AS lncrna_norm,
                   upper(trim(CAST(pathway_id AS VARCHAR))) AS pathway_norm,
                   trim(CAST(cancer_id AS VARCHAR)) AS raw_cancer_id,
                   trim(CAST(lncrna_id AS VARCHAR)) AS raw_lncrna_id,
                   trim(CAST(pathway_id AS VARCHAR)) AS raw_pathway_id,
                   TRY_CAST(evidence_confidence_probability AS DOUBLE)
                     AS evidence_confidence_probability,
                   TRY_CAST(availability AS BOOLEAN) AS availability,
                   CAST(direction AS VARCHAR) AS direction,
                   TRY_CAST(uncertainty AS DOUBLE) AS uncertainty,
                   CAST(unavailable_reason AS VARCHAR) AS unavailable_reason,
                   TRY_CAST(event_count AS BIGINT) AS event_count,
                   TRY_CAST(evidence_fold AS DOUBLE) AS evidence_fold,
                   CAST(failure_reason AS VARCHAR) AS failure_reason,
                   CAST(analysis_version AS VARCHAR) AS analysis_version,
                   CAST(training_run_id AS VARCHAR) AS training_run_id,
                   TRY_CAST(changes_primary_ranking AS BOOLEAN) AS changes_primary_ranking,
                   TRY_CAST(main_ranking_modified AS BOOLEAN) AS main_ranking_modified
            FROM read_parquet({_sql_path(raw_prediction)})
            """
        )
        raw_summary = connection.execute(
            """
            SELECT count(*) AS rows,
                   count(DISTINCT (raw_cancer_id, raw_lncrna_id, raw_pathway_id)) AS raw_distinct_keys,
                   count(DISTINCT (cancer_norm, lncrna_norm, pathway_norm)) AS norm_distinct_keys,
                   count_if(raw_cancer_id = '' OR raw_lncrna_id = '' OR raw_pathway_id = '') AS empty_keys,
                   count_if(availability IS NULL) AS null_availability,
                   count_if(availability AND
                            (evidence_confidence_probability IS NULL
                             OR NOT isfinite(evidence_confidence_probability)
                             OR evidence_confidence_probability < 0
                             OR evidence_confidence_probability > 1)) AS bad_available_probability,
                   count_if(NOT availability AND evidence_confidence_probability IS NOT NULL
                            AND isfinite(evidence_confidence_probability)) AS filled_unavailable_probability,
                   count_if(NOT availability AND
                            (unavailable_reason IS NULL OR trim(unavailable_reason) = ''))
                     AS missing_unavailable_reason,
                   count_if(analysis_version <> ?) AS version_mismatch,
                   count(DISTINCT training_run_id) AS training_run_ids,
                   count_if(changes_primary_ranking OR main_ranking_modified) AS primary_changes,
                   count_if(availability) AS available_rows
            FROM evidence_norm
            """,
            [ANALYSIS_VERSION],
        ).fetchone()
        raw_rows = int(raw_summary[0])
        violations = tuple(int(raw_summary[index]) for index in (3, 4, 5, 6, 7, 8, 10))
        if (
            raw_rows != candidate_rows
            or int(raw_summary[1]) != raw_rows
            or int(raw_summary[2]) != raw_rows
            or any(violations)
            or int(raw_summary[9]) != 1
        ):
            raise EvidenceFusionAdapterError(f"Raw Evidence semantic/key audit failed: {raw_summary}")
        available_rows = int(raw_summary[11])
        unavailable_rows = raw_rows - available_rows
        if strict_formal and (
            available_rows != FORMAL_AVAILABLE_ROWS
            or unavailable_rows != FORMAL_UNAVAILABLE_ROWS
        ):
            raise EvidenceFusionAdapterError("Formal Evidence availability counts drift")

        closure = connection.execute(
            """
            SELECT
              (SELECT count(*) FROM (
                 SELECT cancer_norm, lncrna_norm, pathway_norm FROM evidence_norm
                 EXCEPT
                 SELECT cancer_norm, lncrna_norm, pathway_norm FROM candidate_norm
               )) AS raw_only,
              (SELECT count(*) FROM (
                 SELECT cancer_norm, lncrna_norm, pathway_norm FROM candidate_norm
                 EXCEPT
                 SELECT cancer_norm, lncrna_norm, pathway_norm FROM evidence_norm
               )) AS candidate_only
            """
        ).fetchone()
        if tuple(map(int, closure)) != (0, 0):
            raise EvidenceFusionAdapterError(f"Normalized Evidence/candidate closure failed: {closure}")

        mapping = connection.execute(
            """
            SELECT count_if(NOT starts_with(upper(raw_lncrna_id), 'LNC:')) AS lnc_prefix_added,
                   count_if(raw_pathway_id <> c.pathway_id) AS pathway_spelling_normalized,
                   count_if(raw_cancer_id <> c.cancer_id) AS cancer_spelling_normalized
            FROM evidence_norm e
            JOIN candidate_norm c USING (cancer_norm, lncrna_norm, pathway_norm)
            """
        ).fetchone()

        prediction_path = output / "evidence_exact_fusion_expert.parquet"
        temporary_prediction = output / ".evidence_exact_fusion_expert.parquet.tmp"
        connection.execute(
            f"""
            COPY (
              SELECT c.cancer_id,
                     c.lncrna_id,
                     c.pathway_id,
                     CASE WHEN e.availability
                          THEN e.evidence_confidence_probability ELSE NULL END
                       AS evidence_confidence_probability,
                     e.availability,
                     e.direction,
                     e.uncertainty,
                     CASE WHEN e.availability THEN '' ELSE e.unavailable_reason END
                       AS unavailable_reason,
                     e.event_count,
                     e.evidence_fold,
                     CASE WHEN e.availability THEN '' ELSE e.failure_reason END
                       AS failure_reason,
                     e.analysis_version,
                     e.training_run_id,
                     FALSE AS changes_primary_ranking,
                     FALSE AS main_ranking_modified,
                     NOT starts_with(upper(e.raw_lncrna_id), 'LNC:') AS lncrna_prefix_restored,
                     e.raw_pathway_id <> c.pathway_id AS pathway_spelling_normalized,
                     TRUE AS canonical_candidate_key_verified,
                     TRUE AS direct_target_evidence,
                     TRUE AS confidence_only,
                     FALSE AS affects_discovery,
                     FALSE AS family_to_exact_broadcast
              FROM candidate_norm c
              JOIN evidence_norm e USING (cancer_norm, lncrna_norm, pathway_norm)
              ORDER BY c.cancer_id, c.lncrna_id, c.pathway_id
            ) TO {_sql_path(temporary_prediction)}
              (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
        os.replace(temporary_prediction, prediction_path)

        output_summary = connection.execute(
            f"""
            SELECT count(*) AS rows,
                   count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS distinct_keys,
                   count_if(availability) AS available_rows,
                   count_if(NOT availability) AS unavailable_rows,
                   count_if(availability AND
                            (evidence_confidence_probability IS NULL
                             OR NOT isfinite(evidence_confidence_probability)
                             OR evidence_confidence_probability < 0
                             OR evidence_confidence_probability > 1)) AS bad_available,
                   count_if(NOT availability AND evidence_confidence_probability IS NOT NULL)
                     AS filled_unavailable,
                   count_if(NOT availability AND
                            (unavailable_reason IS NULL OR trim(unavailable_reason) = ''))
                     AS missing_reason,
                   count_if(analysis_version <> ?) AS version_mismatch,
                   count_if(changes_primary_ranking OR main_ranking_modified) AS primary_changes,
                   count_if(NOT canonical_candidate_key_verified) AS unverified_keys,
                   count_if(family_to_exact_broadcast) AS family_broadcasts,
                   count_if(NOT confidence_only OR affects_discovery) AS routing_violations
            FROM read_parquet({_sql_path(prediction_path)})
            """,
            [ANALYSIS_VERSION],
        ).fetchone()
        observed = {
            "rows": int(output_summary[0]),
            "distinct_keys": int(output_summary[1]),
            "available_rows": int(output_summary[2]),
            "unavailable_rows": int(output_summary[3]),
            "bad_available_probability_rows": int(output_summary[4]),
            "filled_unavailable_probability_rows": int(output_summary[5]),
            "missing_unavailable_reason_rows": int(output_summary[6]),
            "analysis_version_mismatch_rows": int(output_summary[7]),
            "primary_change_rows": int(output_summary[8]),
            "unverified_key_rows": int(output_summary[9]),
            "family_broadcast_rows": int(output_summary[10]),
            "routing_violation_rows": int(output_summary[11]),
            "raw_only_normalized_keys": int(closure[0]),
            "candidate_only_normalized_keys": int(closure[1]),
            "lncrna_prefix_restored_rows": int(mapping[0]),
            "pathway_spelling_normalized_rows": int(mapping[1]),
            "cancer_spelling_normalized_rows": int(mapping[2]),
        }
        zero_fields = (
            "bad_available_probability_rows",
            "filled_unavailable_probability_rows",
            "missing_unavailable_reason_rows",
            "analysis_version_mismatch_rows",
            "primary_change_rows",
            "unverified_key_rows",
            "family_broadcast_rows",
            "routing_violation_rows",
            "raw_only_normalized_keys",
            "candidate_only_normalized_keys",
        )
        if (
            observed["rows"] != candidate_rows
            or observed["distinct_keys"] != candidate_rows
            or observed["available_rows"] != available_rows
            or observed["unavailable_rows"] != unavailable_rows
            or any(observed[field] != 0 for field in zero_fields)
        ):
            raise EvidenceFusionAdapterError(f"Materialized Evidence adapter failed: {observed}")

        audit_path = output / "ADAPTER_AUDIT.json"
        _atomic_json(
            audit_path,
            {
                "format": AUDIT_FORMAT,
                "analysis_version": ANALYSIS_VERSION,
                "status": "PASS",
                "fail_closed": True,
                "observed": observed,
                "normalization": {
                    "lncrna": "TRIM_UPPERCASE_AND_RESTORE_REQUIRED_LNC_PREFIX",
                    "pathway": "TRIM_UPPERCASE_LOOKUP_THEN_EMIT_CANDIDATE_CANONICAL_ID",
                    "cancer": "TRIM_UPPERCASE_LOOKUP_THEN_EMIT_CANDIDATE_CANONICAL_ID",
                },
                "normalized_bijection_proved": True,
                "raw_prediction_direct_fusion_allowed": False,
                "confidence_only": True,
                "primary_ranking_unchanged": True,
            },
        )
        binding_path = output / "EVIDENCE_FUSION_BINDING.json"
        binding = {
            "format": BINDING_FORMAT,
            "adapter_format": ADAPTER_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS_HASH_BOUND_CANONICAL_FUSION_INPUT",
            "training_run_id": wrapper["evidence_training_run_id"],
            "candidate_rows": candidate_rows,
            "available_rows": available_rows,
            "unavailable_rows": unavailable_rows,
            "prediction": {
                "path": str(prediction_path),
                "sha256": artifact_sha256(prediction_path),
                "rows": candidate_rows,
            },
            "adapter_audit": {
                "path": str(audit_path),
                "sha256": artifact_sha256(audit_path),
            },
            "candidate_authority": {
                "path": str(candidates),
                "sha256": candidate_sha,
                "rows": candidate_rows,
            },
            "raw_evidence_prediction": {
                "path": str(raw_prediction),
                "sha256": raw_prediction_sha,
                "rows": raw_rows,
            },
            "semantic_wrapper": {
                "path": str(wrapper_path),
                "sha256": artifact_sha256(wrapper_path),
            },
            "r2_evidence_binding": {
                "path": str(r2_path),
                "sha256": artifact_sha256(r2_path),
            },
            "independent_post_audit": {
                "path": str(post_audit),
                "sha256": artifact_sha256(post_audit),
            },
            "adapter_code_sha256": artifact_sha256(Path(__file__)),
            "normalized_bijection_proved": True,
            "fusion_input_eligible": True,
            "direct_target_evidence": True,
            "confidence_only": True,
            "affects_discovery": False,
            "affects_confidence": True,
            "changes_primary_ranking": False,
            "family_to_exact_broadcast": False,
            "unavailable_encoding": "null_with_reason",
            "raw_prediction_direct_fusion_allowed": False,
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "release_ready": False,
            "production_deployed": False,
        }
        _atomic_json(binding_path, binding)
        success_path = output / "SUCCESS.json"
        _atomic_json(
            success_path,
            {
                "status": binding["status"],
                "binding": binding_path.name,
                "binding_sha256": artifact_sha256(binding_path),
                "fusion_input_eligible": True,
                "raw_prediction_direct_fusion_allowed": False,
                "primary_ranking_unchanged": True,
                "release_ready": False,
                "production_deployed": False,
            },
        )
        return {
            "output_root": str(output),
            "binding_path": str(binding_path),
            "binding_sha256": artifact_sha256(binding_path),
            "prediction_path": str(prediction_path),
            "prediction_sha256": artifact_sha256(prediction_path),
            "audit_path": str(audit_path),
            "audit_sha256": artifact_sha256(audit_path),
            **observed,
        }
    except Exception:
        connection.close()
        shutil.rmtree(output, ignore_errors=True)
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass


__all__ = [
    "ADAPTER_FORMAT",
    "AUDIT_FORMAT",
    "BINDING_FORMAT",
    "EvidenceFusionAdapterError",
    "artifact_sha256",
    "materialize_evidence_fusion_adapter",
]
