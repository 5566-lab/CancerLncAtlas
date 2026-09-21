"""Hash-bound remediation for selected V3.2 historical artifact gaps.

This module does not declare the whole website or the 25-capability release
ready.  It materialises only logical artifacts that can be derived without
inventing data, and records typed gaps for outputs that do not exist.  In
    particular, single-cell pseudotime/figures are never inferred from
categorical or null outputs. Evidence direction probabilities are accepted
only through separately hash-pinned V3.2 checkpoint re-inference and
independent-audit bindings.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import duckdb

from .evidence_direction_query import EvidenceDirectionProbabilityQuery


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
FORMAT = "CC_HHGT_V3_2_HISTORICAL_ARTIFACT_REMEDIATION_V1"
FORMAL_EXACT_ROWS = 3_300_000
FORMAL_CANCERS = 33
FORMAL_ENDPOINTS = ("DFI", "DFS", "DSS", "OS", "PFI", "PFS")
EXACT_KEYS = ("cancer_id", "lncrna_id", "pathway_id")


class HistoricalArtifactRemediationError(RuntimeError):
    """Raised when an input or output would overstate V3.2 capability."""


@dataclass(frozen=True)
class RemediationInputs:
    clinical_risk: Path
    clinical_lineage: Path
    clinical_entity: Path
    clinical_entity_lineage: Path
    genomic_predictions: Path
    genomic_lineage: Path
    mixed_asset_root: Path
    mixed_query_code: Path
    single_cell_root: Path
    single_cell_fusion_binding: Path
    hnsc_ucell_binding: Path
    evidence_prediction: Path
    evidence_fusion_binding: Path
    evidence_model_binding: Path
    evidence_direction_binding: Path
    evidence_direction_audit_binding: Path
    multimodal_scores: Path
    multimodal_binding: Path

    def resolved(self) -> "RemediationInputs":
        return RemediationInputs(
            **{name: Path(value).resolve() for name, value in self.__dict__.items()}
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalArtifactRemediationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HistoricalArtifactRemediationError(f"Expected JSON object: {path}")
    return value


def _require_files(inputs: RemediationInputs) -> None:
    for name, path in inputs.__dict__.items():
        if name in {"mixed_asset_root", "single_cell_root"}:
            if not path.is_dir():
                raise HistoricalArtifactRemediationError(f"Required input directory is missing: {path}")
        elif not path.is_file() or path.stat().st_size <= 0:
            raise HistoricalArtifactRemediationError(f"Required input is missing/empty: {path}")


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''").replace("\\", "/")


def _connect() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    connection.execute("SET memory_limit='4GB'")
    connection.execute("SET preserve_insertion_order=false")
    return connection


def _parquet_columns(connection: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
    ]


def _require_columns(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    required: set[str],
) -> None:
    columns = set(_parquet_columns(connection, path))
    missing = sorted(required - columns)
    if missing:
        raise HistoricalArtifactRemediationError(f"{path} lacks columns {missing}")


def _artifact(path: Path, rows: int | None, *, status: str = "READY") -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "status": status,
        "model_version": "V3.2",
        "availability_encoding": "null_with_reason",
        "unavailable_fill_value": None,
    }


def _validate_v32_lineage(path: Path, *, expected_run_id: str | None = None) -> dict[str, Any]:
    value = _read_json(path)
    version = str(value.get("analysis_version", ""))
    if "V3.2" not in version:
        raise HistoricalArtifactRemediationError(f"Non-V3.2 lineage rejected: {path}")
    for key in (
        "old_checkpoint_loaded",
        "old_predictions_used_as_features",
        "old_rankings_used_as_outputs",
    ):
        if value.get(key) is True:
            raise HistoricalArtifactRemediationError(f"Historical derived asset flag set: {path}:{key}")
    if expected_run_id is not None and value.get("training_run_id") != expected_run_id:
        raise HistoricalArtifactRemediationError(f"Training run mismatch in {path}")
    return value


def _materialize_clinical(
    connection: duckdb.DuckDBPyConnection,
    inputs: RemediationInputs,
    output: Path,
    *,
    formal: bool,
) -> dict[str, Any]:
    _require_columns(
        connection,
        inputs.clinical_entity,
        {
            "cancer_id",
            "subject_type",
            "subject_id",
            "clinical_endpoint",
            "clinical_relevance_probability",
            "hazard_ratio",
            "fdr",
            "direction",
            "replication_tier",
            "n_folds_available",
            "n_folds_same_direction",
            "n_patients_test_total",
            "n_events_test_total",
            "availability",
            "failure_reason",
            "model_version",
            "analysis_version",
            "old_checkpoint_loaded",
            "old_predictions_used_as_features",
            "changes_primary_ranking",
        },
    )
    _require_columns(
        connection,
        inputs.clinical_risk,
        {
            "cancer_id",
            "subject_id",
            "clinical_endpoint",
            "clinical_relevance_probability",
            "availability",
            "failure_reason",
            "analysis_version",
            "training_run_id",
            "changes_primary_ranking",
        },
    )
    entity_lineage = _validate_v32_lineage(inputs.clinical_entity_lineage)
    clinical_lineage = _validate_v32_lineage(inputs.clinical_lineage)
    entity_path = _sql_path(inputs.clinical_entity)
    risk_path = _sql_path(inputs.clinical_risk)
    priority_path = output / "clinical_translational_priority.parquet"
    endpoint_path = output / "clinical_endpoint_summary.parquet"
    connection.execute(
        f"""
        COPY (
          SELECT
            cancer_id,
            subject_type,
            subject_id,
            clinical_endpoint,
            CASE WHEN availability THEN clinical_relevance_probability ELSE NULL END
              AS translational_priority_score,
            CASE WHEN availability THEN CAST(
              row_number() OVER (
                PARTITION BY cancer_id, clinical_endpoint, subject_type
                ORDER BY
                  CASE WHEN availability THEN 0 ELSE 1 END,
                  clinical_relevance_probability DESC NULLS LAST,
                  fdr ASC NULLS LAST,
                  subject_id ASC
              ) AS BIGINT) ELSE NULL END AS translational_priority_rank,
            CASE WHEN availability THEN clinical_relevance_probability ELSE NULL END
              AS clinical_relevance_probability,
            CASE WHEN availability AND isfinite(hazard_ratio) THEN hazard_ratio ELSE NULL END
              AS hazard_ratio,
            CASE WHEN availability AND isfinite(fdr) THEN fdr ELSE NULL END AS fdr,
            CASE WHEN availability THEN direction ELSE 'unavailable' END AS direction,
            CASE WHEN availability THEN replication_tier ELSE 'unavailable' END AS replication_tier,
            n_folds_available,
            CASE WHEN availability THEN n_folds_same_direction ELSE NULL END
              AS n_folds_same_direction,
            CASE WHEN availability THEN n_patients_test_total ELSE NULL END
              AS n_patients_test_total,
            CASE WHEN availability THEN n_events_test_total ELSE NULL END
              AS n_events_test_total,
            availability,
            CASE WHEN availability THEN '' ELSE failure_reason END AS unavailable_reason,
            model_version,
            analysis_version,
            'EXISTING_V32_CLINICAL_RELEVANCE_PROBABILITY_DESC_WITH_FDR_AND_ID_TIEBREAK'
              AS priority_definition,
            'DESCRIPTIVE_TRANSLATIONAL_TRIAGE_NOT_PRIMARY_MODEL_SCORE'
              AS priority_role,
            false AS changes_primary_ranking
          FROM read_parquet('{entity_path}')
        ) TO '{_sql_path(priority_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    connection.execute(
        f"""
        COPY (
          WITH entity AS (
            SELECT
              cancer_id,
              clinical_endpoint,
              count(*) AS entity_rows,
              count_if(availability) AS entity_available_rows,
              count_if(NOT availability) AS entity_unavailable_rows,
              count_if(subject_type='lncRNA' AND availability) AS lncrna_available_rows,
              count_if(subject_type='exact_pathway' AND availability)
                AS exact_pathway_available_rows,
              count_if(subject_type='state' AND availability) AS state_available_rows
            FROM read_parquet('{entity_path}')
            GROUP BY cancer_id, clinical_endpoint
          ), risk AS (
            SELECT
              cancer_id,
              clinical_endpoint,
              count(*) AS patient_risk_rows,
              count_if(availability) AS patient_risk_available_rows,
              count_if(NOT availability) AS patient_risk_unavailable_rows
            FROM read_parquet('{risk_path}')
            GROUP BY cancer_id, clinical_endpoint
          )
          SELECT
            entity.cancer_id,
            entity.clinical_endpoint,
            entity.entity_rows,
            entity.entity_available_rows,
            entity.entity_unavailable_rows,
            entity.lncrna_available_rows,
            entity.exact_pathway_available_rows,
            entity.state_available_rows,
            coalesce(risk.patient_risk_rows, 0) AS patient_risk_rows,
            coalesce(risk.patient_risk_available_rows, 0) AS patient_risk_available_rows,
            coalesce(risk.patient_risk_unavailable_rows, 0) AS patient_risk_unavailable_rows,
            CASE
              WHEN entity.clinical_endpoint='DFS' THEN 'NULL_WITH_NO_DISTINCT_DFS_SOURCE'
              ELSE ''
            END AS endpoint_unavailable_policy,
            'V3.2' AS model_version,
            '{ANALYSIS_VERSION}' AS analysis_version,
            false AS endpoint_substituted,
            false AS changes_primary_ranking
          FROM entity
          LEFT JOIN risk USING (cancer_id, clinical_endpoint)
          ORDER BY entity.cancer_id, entity.clinical_endpoint
        ) TO '{_sql_path(endpoint_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    priority_stats = connection.execute(
        """
        SELECT count(*), count_if(availability), count_if(NOT availability),
               count(DISTINCT cancer_id), count(DISTINCT clinical_endpoint),
               count_if(NOT availability AND translational_priority_score IS NOT NULL),
               count_if(availability AND (translational_priority_score IS NULL OR
                        NOT isfinite(translational_priority_score) OR
                        translational_priority_score < 0 OR translational_priority_score > 1)),
               count_if(changes_primary_ranking)
        FROM read_parquet(?)
        """,
        [str(priority_path)],
    ).fetchone()
    endpoint_rows = int(
        connection.execute("SELECT count(*) FROM read_parquet(?)", [str(endpoint_path)]).fetchone()[0]
    )
    if priority_stats[5:] != (0, 0, 0):
        raise HistoricalArtifactRemediationError("Clinical priority null/probability/ranking contract failed")
    if formal:
        if int(priority_stats[0]) != 884_520 or int(priority_stats[3]) != FORMAL_CANCERS:
            raise HistoricalArtifactRemediationError("Formal clinical entity universe mismatch")
        if int(priority_stats[4]) != len(FORMAL_ENDPOINTS) or endpoint_rows != 198:
            raise HistoricalArtifactRemediationError("Formal clinical endpoint universe mismatch")
    report = {
        "format": "CC_HHGT_V3_2_CLINICAL_LOGICAL_RELEASE_V1",
        "status": "SUCCESS_HASH_BOUND_CLINICAL_LOGICAL_RELEASE",
        "analysis_version": ANALYSIS_VERSION,
        "model_version": "V3.2",
        "training_run_ids": [
            str(clinical_lineage.get("training_run_id")),
            str(entity_lineage.get("training_run_id")),
        ],
        "all_non_null_results_generated_in_v32": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "changes_primary_ranking": False,
        "endpoints": list(FORMAL_ENDPOINTS),
        "dfs_policy": "NULL_WITH_NO_DISTINCT_DFS_SOURCE__NO_SUBSTITUTION",
        "priority_semantics": {
            "score": "existing V3.2 clinical_relevance_probability",
            "ranking": "descending within cancer x endpoint x subject_type; FDR then ID tie-break",
            "role": "descriptive translational triage; not a new probability and not primary ranking",
        },
        "sources": {
            "clinical_patient_risk": _artifact(inputs.clinical_risk, 62_592),
            "clinical_lineage": _artifact(inputs.clinical_lineage, None),
            "clinical_entity_associations": _artifact(inputs.clinical_entity, int(priority_stats[0])),
            "clinical_entity_lineage": _artifact(inputs.clinical_entity_lineage, None),
        },
        "artifacts": {
            "v32_clinical_endpoint_summary": _artifact(endpoint_path, endpoint_rows),
            "v32_survival_associations": _artifact(inputs.clinical_entity, int(priority_stats[0])),
            "v32_patient_risk_oof": _artifact(inputs.clinical_risk, 62_592),
            "v32_translational_priority": _artifact(priority_path, int(priority_stats[0])),
        },
        "counts": {
            "priority_rows": int(priority_stats[0]),
            "priority_available_rows": int(priority_stats[1]),
            "priority_unavailable_rows": int(priority_stats[2]),
            "endpoint_summary_rows": endpoint_rows,
        },
    }
    report_path = output / "CLINICAL_RELEASE_MANIFEST.json"
    _write_json(report_path, report)
    report["artifacts"]["v32_clinical_report_manifest"] = _artifact(report_path, None)
    return {"manifest": report, "manifest_path": report_path}


def _materialize_genomic(
    connection: duckdb.DuckDBPyConnection,
    inputs: RemediationInputs,
    output: Path,
    *,
    formal: bool,
) -> dict[str, Any]:
    _require_columns(
        connection,
        inputs.genomic_predictions,
        {
            *EXACT_KEYS,
            "mutation_context_probability",
            "mutation_available",
            "mutation_unavailable_reason",
            "mutation_patient_folds_with_prediction",
            "cnv_context_probability",
            "cnv_available",
            "cnv_unavailable_reason",
            "cnv_patient_folds_with_prediction",
            "analysis_version",
            "training_run_id",
            "target_level",
            "changes_primary_ranking",
        },
    )
    lineage = _validate_v32_lineage(inputs.genomic_lineage)
    source = _sql_path(inputs.genomic_predictions)
    mutation_path = output / "mutation_subgroup_predictions.parquet"
    cnv_coverage_path = output / "cnv_coverage.parquet"
    connection.execute(
        f"""
        COPY (
          SELECT
            cancer_id,
            lncrna_id,
            pathway_id,
            CASE WHEN mutation_available THEN mutation_context_probability ELSE NULL END
              AS mutation_subgroup_probability,
            mutation_available AS availability,
            CASE WHEN mutation_available THEN '' ELSE mutation_unavailable_reason END
              AS unavailable_reason,
            mutation_patient_folds_with_prediction,
            'EXPLICIT_CALLABLE_LNCRNA_MUTATED_VS_WILDTYPE_X_EXACT_PATHWAY_MUTATED_VS_WILDTYPE'
              AS subgroup_definition,
            'POSITIVE_COOCCURRENCE_CLASS_PROBABILITY_FROM_FRESH_V32_MUTATION_HEAD'
              AS probability_definition,
            analysis_version,
            training_run_id,
            'exact_pathway' AS target_level,
            'exact_pathway' AS source_target_level,
            false AS family_broadcast,
            false AS missing_assumed_wildtype,
            false AS changes_primary_ranking
          FROM read_parquet('{source}')
        ) TO '{_sql_path(mutation_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    connection.execute(
        f"""
        COPY (
          SELECT
            cancer_id,
            count(*) AS formal_exact_key_rows,
            count_if(cnv_available) AS cnv_available_rows,
            count_if(NOT cnv_available) AS cnv_unavailable_rows,
            count_if(cnv_available)::DOUBLE / count(*) AS cnv_coverage_fraction,
            min(cnv_patient_folds_with_prediction) FILTER (WHERE cnv_available)
              AS min_patient_folds_with_prediction,
            max(cnv_patient_folds_with_prediction) FILTER (WHERE cnv_available)
              AS max_patient_folds_with_prediction,
            count(DISTINCT pathway_id) AS exact_pathways,
            count(DISTINCT lncrna_id) AS lncrnas,
            count(DISTINCT CASE WHEN cnv_available THEN pathway_id END)
              AS exact_pathways_with_prediction,
            count(DISTINCT CASE WHEN cnv_available THEN lncrna_id END)
              AS lncrnas_with_prediction,
            count_if(cnv_unavailable_reason='CNV_NOT_AVAILABLE_FOR_CANCER')
              AS no_cancer_source_rows,
            count_if(cnv_unavailable_reason='CNV_NO_OOF_PREDICTION')
              AS no_oof_prediction_rows,
            count_if(cnv_unavailable_reason='CNV_NO_EXPLICIT_WT_EVENT_VARIATION')
              AS no_explicit_event_variation_rows,
            count_if(cnv_unavailable_reason='CNV_INSUFFICIENT_PAIR_CALLABILITY')
              AS insufficient_pair_callability_rows,
            count_if(cnv_available) > 0 AS coverage_available,
            CASE WHEN count_if(cnv_available)=0 THEN 'CNV_NOT_AVAILABLE_FOR_CANCER'
                 ELSE '' END AS coverage_unavailable_reason,
            'V3.2' AS model_version,
            '{ANALYSIS_VERSION}' AS analysis_version,
            'exact_pathway' AS target_level,
            'exact_pathway' AS source_target_level,
            false AS family_broadcast,
            false AS missing_assumed_neutral,
            NULL::DOUBLE AS unavailable_fill_value,
            'null_with_reason' AS availability_encoding,
            false AS changes_primary_ranking
          FROM read_parquet('{source}')
          GROUP BY cancer_id
          ORDER BY cancer_id
        ) TO '{_sql_path(cnv_coverage_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    source_stats = connection.execute(
        """
        SELECT count(*), count(DISTINCT cancer_id),
               count_if(mutation_available), count_if(cnv_available),
               count(*)-count(DISTINCT (cancer_id,lncrna_id,pathway_id)),
               count_if(changes_primary_ranking),
               count_if(mutation_available AND
                 (mutation_context_probability IS NULL OR
                  NOT isfinite(mutation_context_probability) OR
                  mutation_context_probability < 0 OR mutation_context_probability > 1)),
               count_if(NOT mutation_available AND
                 mutation_context_probability IS NOT NULL AND isfinite(mutation_context_probability)),
               count_if(cnv_available AND
                 (cnv_context_probability IS NULL OR NOT isfinite(cnv_context_probability) OR
                  cnv_context_probability < 0 OR cnv_context_probability > 1)),
               count_if(NOT cnv_available AND
                 cnv_context_probability IS NOT NULL AND isfinite(cnv_context_probability))
        FROM read_parquet(?)
        """,
        [str(inputs.genomic_predictions)],
    ).fetchone()
    mutation_stats = connection.execute(
        """
        SELECT count(*), count_if(availability),
               count_if(NOT availability AND mutation_subgroup_probability IS NOT NULL),
               count_if(family_broadcast), count_if(missing_assumed_wildtype),
               count_if(changes_primary_ranking)
        FROM read_parquet(?)
        """,
        [str(mutation_path)],
    ).fetchone()
    cnv_rows = int(
        connection.execute("SELECT count(*) FROM read_parquet(?)", [str(cnv_coverage_path)]).fetchone()[0]
    )
    if any(int(value) != 0 for value in source_stats[4:]):
        raise HistoricalArtifactRemediationError("Genomic source typed-availability contract failed")
    if any(int(value) != 0 for value in mutation_stats[2:]):
        raise HistoricalArtifactRemediationError("Mutation logical release contract failed")
    if formal and tuple(map(int, source_stats[:4])) != (
        FORMAL_EXACT_ROWS,
        FORMAL_CANCERS,
        698_302,
        86_461,
    ):
        raise HistoricalArtifactRemediationError("Formal genomic counts mismatch")
    if formal and (int(mutation_stats[0]) != FORMAL_EXACT_ROWS or cnv_rows != FORMAL_CANCERS):
        raise HistoricalArtifactRemediationError("Formal logical genomic artifact size mismatch")
    report = {
        "format": "CC_HHGT_V3_2_GENOMIC_LOGICAL_RELEASE_V1",
        "status": "SUCCESS_HASH_BOUND_GENOMIC_LOGICAL_RELEASE",
        "analysis_version": ANALYSIS_VERSION,
        "model_version": "V3.2",
        "training_run_id": str(lineage.get("training_run_id")),
        "all_non_null_results_generated_in_v32": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "changes_primary_ranking": False,
        "target_level": "exact_pathway",
        "source_target_level": "exact_pathway",
        "family_broadcast": False,
        "mutation_subgroup_semantics": (
            "Probability of the positive co-occurrence class learned from explicitly callable "
            "lncRNA mutated/wild-type and exact-pathway mutated/wild-type patient groups"
        ),
        "missing_mutation_assumed_wildtype": False,
        "missing_cnv_assumed_neutral": False,
        "sources": {
            "typed_predictions": _artifact(inputs.genomic_predictions, int(source_stats[0])),
            "lineage": _artifact(inputs.genomic_lineage, None),
        },
        "capability_artifacts": {
            "mutation": {
                "v32_mutation_context": _artifact(inputs.genomic_predictions, int(source_stats[0])),
                "v32_mutation_subgroup_predictions": _artifact(mutation_path, int(mutation_stats[0])),
                "v32_mutation_lncrna_exact_pathway": _artifact(
                    inputs.genomic_predictions, int(source_stats[0])
                ),
            },
            "cnv": {
                "v32_cnv_context": _artifact(inputs.genomic_predictions, int(source_stats[0])),
                "v32_cnv_lncrna_exact_pathway": _artifact(
                    inputs.genomic_predictions, int(source_stats[0])
                ),
                "v32_cnv_coverage": _artifact(cnv_coverage_path, cnv_rows),
            },
        },
        "counts": {
            "formal_exact_rows": int(source_stats[0]),
            "mutation_available_rows": int(source_stats[2]),
            "mutation_unavailable_rows": int(source_stats[0] - source_stats[2]),
            "cnv_available_rows": int(source_stats[3]),
            "cnv_unavailable_rows": int(source_stats[0] - source_stats[3]),
            "cnv_coverage_rows": cnv_rows,
        },
    }
    report_path = output / "GENOMIC_LOGICAL_RELEASE_MANIFEST.json"
    _write_json(report_path, report)
    return {"manifest": report, "manifest_path": report_path}


def _materialize_mixed(inputs: RemediationInputs, output: Path) -> dict[str, Any]:
    manifest_path = inputs.mixed_asset_root / "ASSET_MANIFEST.json"
    asset_manifest = _read_json(manifest_path)
    if asset_manifest.get("status") != "SUCCESS":
        raise HistoricalArtifactRemediationError("Mixed query asset manifest is not successful")
    if asset_manifest.get("analysis_version") != ANALYSIS_VERSION:
        raise HistoricalArtifactRemediationError("Mixed query assets are not V3.2")
    if asset_manifest.get("pathway_family_broadcast") is not False:
        raise HistoricalArtifactRemediationError("Mixed query asset may broadcast pathway families")
    if any(
        asset_manifest.get(key) is True
        for key in (
            "historical_checkpoints_used",
            "historical_predictions_used",
            "historical_rankings_used",
        )
    ):
        raise HistoricalArtifactRemediationError("Mixed query asset reuses historical derived output")
    resolved_artifacts: dict[str, dict[str, Any]] = {}
    for artifact_id, record in asset_manifest.get("artifacts", {}).items():
        if not isinstance(record, Mapping):
            raise HistoricalArtifactRemediationError("Malformed mixed query artifact record")
        path = (inputs.mixed_asset_root / str(record.get("filename", ""))).resolve()
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise HistoricalArtifactRemediationError(f"Mixed asset hash mismatch: {artifact_id}")
        resolved_artifacts[str(artifact_id)] = {
            "path": str(path),
            "sha256": str(record["sha256"]),
            "bytes": path.stat().st_size,
            "rows": asset_manifest.get("rows", {}).get(
                str(artifact_id).replace("mixed_query_", "")
            ),
        }
    code_sha = sha256_file(inputs.mixed_query_code)
    modes = {
        "v32_mixed_set_encoder": {
            "filename": "MIXED_SET_ENCODER_BINDING.json",
            "query_mode": "mixed_lncrna_and_protein_gene_set",
            "accepted_entity_types": ["lncRNA", "protein_coding_gene"],
            "downstream_channels": ["v32_lncrna_exact_model", "protein_exact_pathway_ora"],
        },
        "v32_custom_gene_set_encoder": {
            "filename": "CUSTOM_GENE_SET_ENCODER_BINDING.json",
            "query_mode": "custom_gene_set_identifier_encoder",
            "accepted_entity_types": ["lncRNA", "protein_coding_gene"],
            "downstream_channels": ["v32_lncrna_exact_model", "protein_exact_pathway_ora"],
        },
        "v32_protein_set_encoder": {
            "filename": "PROTEIN_SET_ENCODER_BINDING.json",
            "query_mode": "protein_coding_gene_set",
            "accepted_entity_types": ["protein_coding_gene"],
            "downstream_channels": ["protein_exact_pathway_ora"],
        },
    }
    bindings: dict[str, Any] = {}
    for artifact_id, mode in modes.items():
        binding = {
            "format": "CC_HHGT_V3_2_MIXED_QUERY_ENCODER_BINDING_V1",
            "status": "READY_HASH_BOUND_DETERMINISTIC_ENCODER",
            "artifact_id": artifact_id,
            "analysis_version": ANALYSIS_VERSION,
            "model_version": "V3.2",
            "training_run_id": asset_manifest.get("training_run_id"),
            "encoder_kind": "DETERMINISTIC_IDENTIFIER_AND_SET_ENCODER_NO_LEARNABLE_PARAMETERS",
            "learnable_encoder_parameters": False,
            "encoder_checkpoint_required": False,
            "v32_model_dependency": {
                "role": "lncRNA channel only",
                "source_exact_association_sha256": asset_manifest.get(
                    "source_exact_association_sha256"
                ),
                "training_run_id": asset_manifest.get("training_run_id"),
            },
            "query_implementation": {
                "path": str(inputs.mixed_query_code),
                "sha256": code_sha,
                "class": "cc_hhgt.v32.mixed_query.MixedExactPathwayQuery",
            },
            "query_mode": mode["query_mode"],
            "accepted_entity_types": mode["accepted_entity_types"],
            "downstream_channels": mode["downstream_channels"],
            "identifier_policy": "unique canonical mapping; ambiguous/unmapped/duplicate reported",
            "target_level": "exact_pathway",
            "source_target_level": "exact_pathway",
            "family_broadcast": False,
            "lncrna_is_classical_pathway_member": False,
            "combined_score_is_probability": False,
            "historical_checkpoints_used": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "static_annotations_only": True,
            "asset_manifest": _artifact(manifest_path, None),
            "assets": resolved_artifacts,
        }
        path = output / str(mode["filename"])
        _write_json(path, binding)
        bindings[artifact_id] = _artifact(path, None)
    result = {
        "status": "SUCCESS_THREE_ENCODER_BINDINGS",
        "artifact_ids": bindings,
        "identifier_crosswalk": _artifact(
            inputs.mixed_asset_root / "identifier_map.parquet",
            int(asset_manifest["rows"]["identifier_map"]),
        ),
        "ora_reference": _artifact(
            inputs.mixed_asset_root / "exact_pathway_membership.parquet",
            int(asset_manifest["rows"]["exact_pathway_membership"]),
        ),
    }
    report_path = output / "MIXED_ENCODER_RELEASE_MANIFEST.json"
    _write_json(
        report_path,
        {
            "format": "CC_HHGT_V3_2_MIXED_ENCODER_RELEASE_V1",
            "analysis_version": ANALYSIS_VERSION,
            "model_version": "V3.2",
            "training_run_id": asset_manifest.get("training_run_id"),
            "status": result["status"],
            "artifacts": {
                **bindings,
                "v32_identifier_crosswalk": result["identifier_crosswalk"],
                "v32_exact_pathway_ora_reference": result["ora_reference"],
            },
            "artifact_semantics": (
                "The three encoders are deterministic identifier/set contracts, not fabricated "
                "learned models. The lncRNA channel is bound to fresh V3.2 exact predictions."
            ),
            "family_broadcast": False,
            "historical_predictions_used": False,
        },
    )
    result["manifest_path"] = report_path
    return result


def _materialize_single_cell(
    connection: duckdb.DuckDBPyConnection,
    inputs: RemediationInputs,
    output: Path,
    *,
    formal: bool,
) -> dict[str, Any]:
    lnc_cell = inputs.single_cell_root / "lnc_celltype_summary.parquet"
    lnc_exact = inputs.single_cell_root / "lnc_exact_pathway.parquet"
    activity = inputs.single_cell_root / "activity.parquet"
    figure = inputs.single_cell_root / "FIGURE_MANIFEST.json"
    success = inputs.single_cell_root / "SUCCESS.json"
    for path in (lnc_cell, lnc_exact, activity, figure, success):
        if not path.is_file() or path.stat().st_size <= 0:
            raise HistoricalArtifactRemediationError(f"Single-cell source missing: {path}")
    _require_columns(
        connection,
        activity,
        {
            "cancer_id",
            "pathway_id",
            "activity_kind",
            "activity_value",
            "mean_pseudotime",
            "activity_available",
            "activity_unavailable_reason",
            "analysis_version",
        },
    )
    activity_stats = connection.execute(
        """
        SELECT count(*), count_if(activity_available), count(DISTINCT cancer_id),
               count_if(mean_pseudotime IS NOT NULL AND isfinite(mean_pseudotime)),
               count_if(activity_available AND
                 (activity_value IS NULL OR NOT isfinite(activity_value))),
               count_if(NOT activity_available AND activity_value IS NOT NULL AND
                 isfinite(activity_value))
        FROM read_parquet(?)
        """,
        [str(activity)],
    ).fetchone()
    exact_stats = connection.execute(
        """
        SELECT count(*), count_if(single_cell_available), count(DISTINCT cancer_id),
               count_if(NOT exact_pathway_only),
               count_if(single_cell_available AND
                 (single_cell_replication_probability IS NULL OR
                  NOT isfinite(single_cell_replication_probability))),
               count_if(NOT single_cell_available AND
                 single_cell_replication_probability IS NOT NULL AND
                 isfinite(single_cell_replication_probability))
        FROM read_parquet(?)
        """,
        [str(lnc_exact)],
    ).fetchone()
    cell_rows = int(
        connection.execute("SELECT count(*) FROM read_parquet(?)", [str(lnc_cell)]).fetchone()[0]
    )
    if any(int(value) != 0 for value in activity_stats[4:]) or any(
        int(value) != 0 for value in exact_stats[3:]
    ):
        raise HistoricalArtifactRemediationError("Single-cell source semantic contract failed")
    if formal and (
        int(activity_stats[0]) != 4_354_871
        or int(activity_stats[1]) != 4_320_711
        or int(activity_stats[2]) != FORMAL_CANCERS
        or int(activity_stats[3]) != 0
        or int(exact_stats[0]) != 2_554_541
        or int(exact_stats[1]) != 954_541
        or cell_rows != 229_104
    ):
        raise HistoricalArtifactRemediationError("Formal single-cell counts mismatch")
    figure_value = _read_json(figure)
    entries = figure_value.get("entries", [])
    if not isinstance(entries, list):
        raise HistoricalArtifactRemediationError("Single-cell figure manifest entries malformed")
    figure_available = sum(
        1
        for item in entries
        if isinstance(item, Mapping)
        and item.get("status") not in {"NULL_WITH_REASON", "UNAVAILABLE"}
        and item.get("path")
    )
    hnsc = _read_json(inputs.hnsc_ucell_binding)
    validated = hnsc.get("validated_counts", {})
    if hnsc.get("historical_sc_trajectory_used") is not False:
        raise HistoricalArtifactRemediationError("HNSC UCell binding may reuse historical trajectory")
    if hnsc.get("cancer_id") != "HNSC" or hnsc.get("scope") != "HNSC_ONLY_PILOT":
        raise HistoricalArtifactRemediationError("HNSC UCell scope is not explicit")
    binding = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_FUNCTIONAL_BINDING_V1",
        "status": "PARTIAL_WITH_TYPED_GAPS",
        "analysis_version": ANALYSIS_VERSION,
        "model_version": "V3.2",
        "production_deployed": False,
        "capability_artifact_gate_closed": False,
        "historical_checkpoints_used": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "family_broadcast": False,
        "artifacts": {
            "v32_sc_lncrna_celltype": _artifact(lnc_cell, cell_rows),
            "v32_sc_lncrna_exact_pathway": _artifact(lnc_exact, int(exact_stats[0])),
            "v32_sc_activity": _artifact(activity, int(activity_stats[0])),
            "v32_sc_ucell": {
                **_artifact(inputs.hnsc_ucell_binding, None, status="PARTIAL_SCOPE_HNSC_ONLY"),
                "scope": "HNSC_ONLY_PILOT",
                "cell_level_coverage_rows": int(validated.get("cell_level_coverage_rows", 0)),
                "cell_level_numeric_rows": int(validated.get("cell_level_numeric_rows", 0)),
                "formal_cancers_covered": 1,
                "formal_cancers_total": FORMAL_CANCERS,
            },
            "v32_sc_pseudotime": {
                "present": False,
                "path": None,
                "sha256": None,
                "rows": 0,
                "status": "GAP_TYPED_UNAVAILABLE",
                "numeric_rows": int(activity_stats[3]),
                "unavailable_reason": "NO_EXPLICIT_TRAJECTORY_ROOT_OR_ORDERED_SOURCE_STATE",
                "unavailable_fill_value": None,
            },
            "v32_sc_figure_manifest": {
                **_artifact(figure, len(entries), status="TYPED_UNAVAILABLE_NO_FIGURE_FILES"),
                "entries": len(entries),
                "available_figure_files": figure_available,
                "typed_unavailable_entries": len(entries) - figure_available,
            },
        },
        "counts": {
            "lncrna_celltype_rows": cell_rows,
            "lncrna_exact_pathway_rows": int(exact_stats[0]),
            "lncrna_exact_pathway_available_rows": int(exact_stats[1]),
            "activity_rows": int(activity_stats[0]),
            "activity_available_rows": int(activity_stats[1]),
            "pseudotime_numeric_rows": int(activity_stats[3]),
            "ucell_formal_cancers": 1,
            "figure_files": figure_available,
        },
        "blocking_gaps": [
            "v32_sc_ucell is HNSC-only, not all formal eligible cancers",
            "v32_sc_pseudotime has zero numeric values",
            "v32_sc_figure_manifest contains zero figure files",
        ],
        "source_success": _artifact(success, None),
        "single_cell_fusion_binding": _artifact(inputs.single_cell_fusion_binding, None),
    }
    path = output / "SINGLE_CELL_FUNCTIONAL_BINDING.json"
    _write_json(path, binding)
    return {"manifest": binding, "manifest_path": path}


def _materialize_evidence(
    connection: duckdb.DuckDBPyConnection,
    inputs: RemediationInputs,
    output: Path,
    *,
    formal: bool,
) -> dict[str, Any]:
    _require_columns(
        connection,
        inputs.evidence_prediction,
        {
            *EXACT_KEYS,
            "evidence_confidence_probability",
            "availability",
            "direction",
            "uncertainty",
            "unavailable_reason",
            "analysis_version",
            "training_run_id",
            "confidence_only",
            "affects_discovery",
            "family_to_exact_broadcast",
        },
    )
    _require_columns(
        connection,
        inputs.multimodal_scores,
        {
            *EXACT_KEYS,
            "primary_probability",
            "fused_confidence_probability",
            "confidence_evidence_transformer_available",
            "confidence_evidence_transformer_logit_contribution",
            "confidence_evidence_transformer_fusion_weight",
            "evidence_transformer_native_probability",
            "evidence_transformer_native_available",
            "primary_ranking_unchanged",
            "used_for_primary_release",
            "historical_predictions_used",
            "historical_rankings_used",
        },
    )
    evidence_stats = connection.execute(
        """
        SELECT count(*), count_if(availability), count_if(NOT availability),
               count(DISTINCT cancer_id),
               count_if(availability AND
                 (evidence_confidence_probability IS NULL OR
                  NOT isfinite(evidence_confidence_probability) OR
                  evidence_confidence_probability < 0 OR
                  evidence_confidence_probability > 1)),
               count_if(NOT availability AND
                 evidence_confidence_probability IS NOT NULL AND
                 isfinite(evidence_confidence_probability)),
               count_if(NOT confidence_only), count_if(affects_discovery),
               count_if(family_to_exact_broadcast),
               count_if(availability AND direction IS NULL),
               count_if(availability AND direction NOT IN ('negative','neutral','positive'))
        FROM read_parquet(?)
        """,
        [str(inputs.evidence_prediction)],
    ).fetchone()
    direction_counts = {
        str(direction): int(count)
        for direction, count in connection.execute(
            """
            SELECT direction, count(*) FROM read_parquet(?)
            WHERE availability GROUP BY direction ORDER BY direction
            """,
            [str(inputs.evidence_prediction)],
        ).fetchall()
    }
    fusion_stats = connection.execute(
        """
        SELECT count(*), count(DISTINCT cancer_id),
               count_if(fused_confidence_probability IS NULL OR
                 NOT isfinite(fused_confidence_probability) OR
                 fused_confidence_probability < 0 OR fused_confidence_probability > 1),
               count_if(abs(fused_confidence_probability-primary_probability) > 1e-12),
               count_if(confidence_evidence_transformer_available),
               count_if(abs(confidence_evidence_transformer_fusion_weight) > 1e-15),
               count_if(abs(coalesce(confidence_evidence_transformer_logit_contribution,0)) > 1e-15),
               count_if(NOT primary_ranking_unchanged),
               count_if(used_for_primary_release),
               count_if(historical_predictions_used),
               count_if(historical_rankings_used),
               count_if(evidence_transformer_native_available),
               count_if(evidence_transformer_native_available AND
                 evidence_transformer_native_probability IS NULL)
        FROM read_parquet(?)
        """,
        [str(inputs.multimodal_scores)],
    ).fetchone()
    if any(int(value) != 0 for value in evidence_stats[4:]):
        raise HistoricalArtifactRemediationError("Evidence native output semantic contract failed")
    if any(int(fusion_stats[index]) != 0 for index in (2, 5, 6, 7, 8, 9, 10, 12)):
        raise HistoricalArtifactRemediationError("Fused-confidence semantic contract failed")
    if formal and tuple(map(int, evidence_stats[:4])) != (
        FORMAL_EXACT_ROWS,
        825_753,
        2_474_247,
        FORMAL_CANCERS,
    ):
        raise HistoricalArtifactRemediationError("Formal Evidence counts mismatch")
    if formal and (int(fusion_stats[0]) != FORMAL_EXACT_ROWS or int(fusion_stats[1]) != FORMAL_CANCERS):
        raise HistoricalArtifactRemediationError("Formal fused confidence universe mismatch")
    model_binding = _read_json(inputs.evidence_model_binding)
    if model_binding.get("historical_checkpoints_used") is not False:
        raise HistoricalArtifactRemediationError("Evidence model binding lineage is not fresh")
    direction_query = EvidenceDirectionProbabilityQuery(
        inputs.evidence_direction_binding,
        expected_binding_sha256=sha256_file(inputs.evidence_direction_binding),
        audit_binding_path=inputs.evidence_direction_audit_binding,
        expected_audit_binding_sha256=sha256_file(
            inputs.evidence_direction_audit_binding
        ),
    )
    direction_capability = direction_query.capability()
    direction_stats = connection.execute(
        """
        SELECT count(*), count_if(direction_probability_available),
               count_if(NOT direction_probability_available),
               count(DISTINCT cancer_id),
               count_if(direction_probability_available AND
                 (direction_negative_probability IS NULL OR
                  direction_neutral_probability IS NULL OR
                  direction_positive_probability IS NULL OR
                  abs(direction_negative_probability+
                      direction_neutral_probability+
                      direction_positive_probability-1.0) > 1e-5)),
               count_if(NOT direction_probability_available AND
                 (direction_negative_probability IS NOT NULL OR
                  direction_neutral_probability IS NOT NULL OR
                  direction_positive_probability IS NOT NULL)),
               count_if(changes_primary_ranking),
               count_if(changes_discovery_ranking),
               count_if(used_for_fusion),
               count_if(historical_checkpoint_used),
               count_if(historical_prediction_used)
        FROM read_parquet(?)
        """,
        [str(direction_query.artifact_path)],
    ).fetchone()
    if any(int(direction_stats[index]) != 0 for index in range(4, 11)):
        raise HistoricalArtifactRemediationError(
            "Evidence direction probability semantic contract failed"
        )
    if formal and tuple(map(int, direction_stats[:4])) != (
        FORMAL_EXACT_ROWS,
        825_753,
        2_474_247,
        FORMAL_CANCERS,
    ):
        raise HistoricalArtifactRemediationError(
            "Formal Evidence direction probability counts mismatch"
        )
    binding = {
        "format": "CC_HHGT_V3_2_EVIDENCE_SEMANTIC_RELEASE_V1",
        "status": "SUCCESS_CONFIDENCE_AND_DIRECTION_PROBABILITIES_HASH_BOUND",
        "analysis_version": ANALYSIS_VERSION,
        "model_version": "V3.2",
        "capability_artifact_gate_closed": True,
        "confidence_separate_from_primary": True,
        "native_evidence_probability_affects_discovery": False,
        "family_broadcast": False,
        "historical_checkpoints_used": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "artifacts": {
            "v32_evidence_transformer_model": _artifact(inputs.evidence_model_binding, None),
            "v32_neural_evidence_probability": _artifact(
                inputs.evidence_prediction, int(evidence_stats[0])
            ),
            "v32_fused_confidence_probability": {
                **_artifact(inputs.multimodal_scores, int(fusion_stats[0])),
                "role": "SECONDARY_CONFIDENCE_ONLY_NOT_PRIMARY_RELEASE",
                "evidence_transformer_fusion_weight": 0.0,
                "evidence_transformer_nonzero_contribution_rows": int(fusion_stats[6]),
                "rows_different_from_primary_probability": int(fusion_stats[3]),
            },
            "v32_evidence_direction_probability": {
                **_artifact(
                    direction_query.artifact_path,
                    int(direction_stats[0]),
                    status="SUCCESS_THREE_CLASS_PROBABILITIES_HASH_PINNED",
                ),
                "present": True,
                "available_rows": int(direction_stats[1]),
                "typed_null_rows": int(direction_stats[2]),
                "classes": ["negative", "neutral", "positive"],
                "used_for_fusion": False,
                "changes_primary_ranking": False,
                "changes_discovery_ranking": False,
                "release_binding": _artifact(
                    inputs.evidence_direction_binding, None
                ),
                "independent_audit_binding": _artifact(
                    inputs.evidence_direction_audit_binding, None
                ),
            },
        },
        "direction_semantics": {
            "published_value_kind": "categorical_argmax_plus_three_class_probability",
            "classes": ["negative", "neutral", "positive"],
            "available_direction_counts": direction_counts,
            "probability_vector_published": True,
            "probability_vector_available_rows": int(direction_stats[1]),
            "probability_vector_typed_null_rows": int(direction_stats[2]),
            "checkpoint_reinference_status": direction_capability["status"],
            "independent_audit": {
                "status": direction_query.audit["status"],
                "binding_sha256": direction_query.audit_binding_sha256,
                "report_sha256": direction_query.audit_report_sha256,
            },
            "fixed_seed_recovered_argmax_mismatch_rows": direction_query.release[
                "counts"
            ]["published_recovered_argmax_mismatch_rows"],
            "mismatch_is_not_checkpoint_drift": direction_query.release[
                "argmax_mismatch_is_not_checkpoint_drift"
            ],
        },
        "fused_confidence_semantics": {
            "available_for_all_exact_keys_via_primary_fallback": True,
            "is_primary_probability": False,
            "used_for_primary_release": False,
            "evidence_transformer_weight": 0.0,
            "interpretation": (
                "The artifact exists, but the trained fusion assigned zero incremental weight "
                "to Evidence; native Evidence probabilities remain independently queryable."
            ),
        },
        "counts": {
            "native_rows": int(evidence_stats[0]),
            "native_available_rows": int(evidence_stats[1]),
            "native_unavailable_rows": int(evidence_stats[2]),
            "fused_confidence_rows": int(fusion_stats[0]),
            "fused_confidence_rows_different_from_primary": int(fusion_stats[3]),
            "evidence_available_to_fusion_rows": int(fusion_stats[4]),
            "evidence_nonzero_weight_rows": int(fusion_stats[5]),
        },
        "evidence_fusion_binding": _artifact(inputs.evidence_fusion_binding, None),
        "multimodal_binding": _artifact(inputs.multimodal_binding, None),
    }
    path = output / "EVIDENCE_SEMANTIC_RELEASE.json"
    _write_json(path, binding)
    return {"manifest": binding, "manifest_path": path}


def materialize_historical_artifact_remediation(
    *,
    inputs: RemediationInputs,
    output_root: str | Path,
    formal: bool = True,
) -> dict[str, Any]:
    """Materialise all truthfully closable artifacts and explicit gaps."""

    inputs = inputs.resolved()
    _require_files(inputs)
    output = Path(output_root).resolve()
    if output.exists():
        raise HistoricalArtifactRemediationError(f"Output reuse refused: {output}")
    output.mkdir(parents=True)
    connection = _connect()
    try:
        clinical = _materialize_clinical(connection, inputs, output, formal=formal)
        genomic = _materialize_genomic(connection, inputs, output, formal=formal)
        mixed = _materialize_mixed(inputs, output)
        single_cell = _materialize_single_cell(connection, inputs, output, formal=formal)
        evidence = _materialize_evidence(connection, inputs, output, formal=formal)
    finally:
        connection.close()

    files: dict[str, Any] = {}
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file():
            rows: int | None = None
            if path.suffix.lower() == ".parquet":
                con = _connect()
                try:
                    rows = int(
                        con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
                    )
                finally:
                    con.close()
            files[path.name] = _artifact(path, rows)
    binding = {
        "format": FORMAT,
        "status": "MATERIALIZATION_COMPLETE_WITH_EXPLICIT_GAPS",
        "analysis_version": ANALYSIS_VERSION,
        "model_version": "V3.2",
        "production_deployed": False,
        "all_non_null_results_generated_in_v32": True,
        "legacy_checkpoint_or_prediction_used": False,
        "target_level": "exact_pathway",
        "family_broadcast": False,
        "fully_closed_artifact_logic": [
            "clinical",
            "mutation",
            "cnv",
            "mixed_lncrna_protein_pathway_query",
            "evidence_transformer",
        ],
        "partial_artifact_logic": ["single_cell"],
        "explicit_gaps": [
            {
                "capability_id": "single_cell",
                "artifact_id": "v32_sc_ucell",
                "status": "PARTIAL_SCOPE_HNSC_ONLY",
            },
            {
                "capability_id": "single_cell",
                "artifact_id": "v32_sc_pseudotime",
                "status": "GAP_TYPED_UNAVAILABLE",
            },
            {
                "capability_id": "single_cell",
                "artifact_id": "v32_sc_figure_manifest",
                "status": "TYPED_UNAVAILABLE_NO_FIGURE_FILES",
            },
        ],
        "capabilities": {
            "clinical": {
                "status": "ARTIFACT_LOGIC_CLOSED",
                "manifest": _artifact(clinical["manifest_path"], None),
            },
            "mutation_cnv": {
                "status": "ARTIFACT_LOGIC_CLOSED",
                "manifest": _artifact(genomic["manifest_path"], None),
            },
            "mixed_lncrna_protein_pathway_query": {
                "status": "ARTIFACT_LOGIC_CLOSED",
                "manifest": _artifact(mixed["manifest_path"], None),
            },
            "single_cell": {
                "status": "PARTIAL_WITH_TYPED_GAPS",
                "manifest": _artifact(single_cell["manifest_path"], None),
            },
            "evidence_transformer": {
                "status": "ARTIFACT_LOGIC_CLOSED",
                "manifest": _artifact(evidence["manifest_path"], None),
            },
        },
        "logical_artifact_index": {
            "clinical": clinical["manifest"]["artifacts"],
            "mutation": genomic["manifest"]["capability_artifacts"]["mutation"],
            "cnv": genomic["manifest"]["capability_artifacts"]["cnv"],
            "mixed_lncrna_protein_pathway_query": {
                **mixed["artifact_ids"],
                "v32_identifier_crosswalk": mixed["identifier_crosswalk"],
                "v32_exact_pathway_ora_reference": mixed["ora_reference"],
            },
            "single_cell": single_cell["manifest"]["artifacts"],
            "evidence_transformer": evidence["manifest"]["artifacts"],
        },
        "files": files,
        "input_binding_sha256": _json_sha256(
            {
                name: (
                    sha256_file(path)
                    if path.is_file()
                    else sha256_file(
                        path
                        / (
                            "ASSET_MANIFEST.json"
                            if name == "mixed_asset_root"
                            else "SUCCESS.json"
                        )
                    )
                )
                for name, path in inputs.__dict__.items()
            }
        ),
    }
    binding_path = output / "HISTORICAL_ARTIFACT_REMEDIATION_BINDING.json"
    _write_json(binding_path, binding)
    success = {
        "status": "SUCCESS_MATERIALIZATION_WITH_GAPS_PRESERVED",
        "analysis_version": ANALYSIS_VERSION,
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "fully_closed_artifact_logic_count": len(binding["fully_closed_artifact_logic"]),
        "partial_artifact_logic_count": len(binding["partial_artifact_logic"]),
        "production_deployed": False,
        "parity_release_ready": False,
    }
    _write_json(output / "SUCCESS.json", success)
    return success


class HistoricalArtifactQuery:
    """Small fail-closed query loader for the remediation release."""

    def __init__(
        self,
        release_root: str | Path,
        *,
        expected_binding_sha256: str | None = None,
        independent_audit_binding_path: str | Path | None = None,
        expected_independent_audit_sha256: str | None = None,
    ) -> None:
        self.root = Path(release_root).resolve()
        success_path = self.root / "SUCCESS.json"
        binding_path = self.root / "HISTORICAL_ARTIFACT_REMEDIATION_BINDING.json"
        actual_binding_sha256 = sha256_file(binding_path)
        if (
            expected_binding_sha256 is not None
            and actual_binding_sha256 != str(expected_binding_sha256).lower()
        ):
            raise HistoricalArtifactRemediationError(
                "Remediation external binding hash mismatch"
            )
        self.success = _read_json(success_path)
        if self.success.get("status") != "SUCCESS_MATERIALIZATION_WITH_GAPS_PRESERVED":
            raise HistoricalArtifactRemediationError("Remediation SUCCESS status is invalid")
        if self.success.get("binding_sha256") != actual_binding_sha256:
            raise HistoricalArtifactRemediationError("Remediation binding hash drift")
        self.binding = _read_json(binding_path)
        if self.binding.get("format") != FORMAT:
            raise HistoricalArtifactRemediationError("Remediation binding format mismatch")
        for filename, record in self.binding.get("files", {}).items():
            path = self.root / str(filename)
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise HistoricalArtifactRemediationError(f"Bound artifact hash drift: {filename}")
        if (independent_audit_binding_path is None) != (
            expected_independent_audit_sha256 is None
        ):
            raise HistoricalArtifactRemediationError(
                "Independent audit path and expected SHA256 must be supplied together"
            )
        self.independent_audit: dict[str, Any] | None = None
        if independent_audit_binding_path is not None:
            audit_path = Path(independent_audit_binding_path).resolve()
            if sha256_file(audit_path) != str(expected_independent_audit_sha256).lower():
                raise HistoricalArtifactRemediationError(
                    "Independent audit external binding hash mismatch"
                )
            audit = _read_json(audit_path)
            if (
                audit.get("status") != "PASS_HASH_BOUND"
                or int(audit.get("failed_checks", -1)) != 0
                or int(audit.get("checks", 0)) < 1
                or audit.get("release_binding_sha256") != actual_binding_sha256
            ):
                raise HistoricalArtifactRemediationError(
                    "Independent audit does not approve this remediation binding"
                )
            report_path = Path(str(audit.get("report_path", ""))).resolve()
            if (
                not report_path.is_file()
                or sha256_file(report_path) != audit.get("report_sha256")
            ):
                raise HistoricalArtifactRemediationError(
                    "Independent audit report hash drift"
                )
            self.independent_audit = audit

    def clinical_priority(
        self,
        *,
        cancer_id: str,
        endpoint: str | None = None,
        subject_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be in 1..1000")
        predicates = ["cancer_id = ?", "availability"]
        params: list[Any] = [str(cancer_id).upper()]
        if endpoint:
            predicates.append("clinical_endpoint = ?")
            params.append(str(endpoint).upper())
        if subject_type:
            predicates.append("subject_type = ?")
            params.append(str(subject_type))
        params.append(int(limit))
        connection = _connect()
        try:
            frame = connection.execute(
                f"""
                SELECT * FROM read_parquet(?)
                WHERE {' AND '.join(predicates)}
                ORDER BY clinical_endpoint, subject_type, translational_priority_rank
                LIMIT ?
                """,
                [str(self.root / "clinical_translational_priority.parquet"), *params],
            ).fetchdf()
        finally:
            connection.close()
        return _records(frame)

    def mutation_subgroup(
        self,
        *,
        cancer_id: str,
        lncrna_id: str | None = None,
        pathway_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be in 1..1000")
        predicates = ["cancer_id = ?"]
        params: list[Any] = [str(cancer_id).upper()]
        if lncrna_id:
            predicates.append("lncrna_id = ?")
            params.append(str(lncrna_id))
        if pathway_id:
            predicates.append("pathway_id = ?")
            params.append(str(pathway_id))
        params.append(int(limit))
        connection = _connect()
        try:
            frame = connection.execute(
                f"""
                SELECT * FROM read_parquet(?)
                WHERE {' AND '.join(predicates)}
                ORDER BY availability DESC, mutation_subgroup_probability DESC NULLS LAST,
                         lncrna_id, pathway_id
                LIMIT ?
                """,
                [str(self.root / "mutation_subgroup_predictions.parquet"), *params],
            ).fetchdf()
        finally:
            connection.close()
        return _records(frame)

    def cnv_coverage(self, cancer_id: str | None = None) -> list[dict[str, Any]]:
        connection = _connect()
        try:
            if cancer_id:
                frame = connection.execute(
                    "SELECT * FROM read_parquet(?) WHERE cancer_id=?",
                    [str(self.root / "cnv_coverage.parquet"), str(cancer_id).upper()],
                ).fetchdf()
            else:
                frame = connection.execute(
                    "SELECT * FROM read_parquet(?) ORDER BY cancer_id",
                    [str(self.root / "cnv_coverage.parquet")],
                ).fetchdf()
        finally:
            connection.close()
        return _records(frame)

    def capability_status(self, capability_id: str) -> dict[str, Any]:
        key = str(capability_id)
        if key == "mutation" or key == "cnv":
            key = "mutation_cnv"
        record = self.binding.get("capabilities", {}).get(key)
        if not isinstance(record, Mapping):
            raise KeyError(capability_id)
        manifest_path = Path(str(record["manifest"]["path"]))
        if sha256_file(manifest_path) != record["manifest"]["sha256"]:
            raise HistoricalArtifactRemediationError(f"Capability manifest drift: {key}")
        return _read_json(manifest_path)


def _records(frame: Any) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                clean[str(key)] = None
            elif hasattr(value, "item"):
                clean[str(key)] = value.item()
            else:
                clean[str(key)] = value
        values.append(clean)
    return values


__all__ = [
    "ANALYSIS_VERSION",
    "FORMAT",
    "HistoricalArtifactQuery",
    "HistoricalArtifactRemediationError",
    "RemediationInputs",
    "materialize_historical_artifact_remediation",
    "sha256_file",
]
