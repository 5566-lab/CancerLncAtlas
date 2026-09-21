"""Hash-pinned, read-only queries for fresh V3.2 external validation."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .external_validation_release import (
    ANALYSIS_VERSION,
    BINDING_FORMAT,
    FORMAL_CANDIDATE_SHA256,
    FORMAL_PREDICTION_LINEAGE_SHA256,
    FORMAL_PREDICTION_SHA256,
    FORMAL_TASK_DEFINITION_SHA256,
    FORMAL_TRAINING_EVIDENCE_SHA256,
    FORMAL_TRAINING_INTERACTION_SHA256,
    HISTORICAL_DERIVED_COLUMNS_IGNORED,
    RANK_CUTOFFS,
    RAW_SOURCE_AUTHORITIES,
    RELEASE_STATUS,
    artifact_sha256,
    normalize_pmid,
)


MAX_QUERY_LIMIT = 1_000
MAX_QUERY_OFFSET = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANCER = re.compile(r"^[A-Z0-9_-]{2,16}$")
_SOURCE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_ROLES = ("primary_known_positive", "secondary_consistency")


class ExternalValidationQueryError(RuntimeError):
    """Base external-validation query error."""


class ExternalValidationQueryAssetError(ExternalValidationQueryError):
    """Raised when a binding, source authority, or artifact is stale."""


class ExternalValidationQueryInputError(ExternalValidationQueryError):
    """Raised when a query filter is invalid."""


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ExternalValidationQueryAssetError(
            f"{label} is missing or unsafe: {source}"
        )
    return source


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalValidationQueryAssetError(
            f"{label} is missing or invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ExternalValidationQueryAssetError(f"{label} must be a JSON object")
    return value


def _sql_text(value: str) -> str:
    quote = chr(39)
    return quote + value.replace(quote, quote * 2) + quote


def _relation(path: Path) -> str:
    return f"read_parquet({_sql_text(path.resolve().as_posix())})"


def _canonical_lnc(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"^(?:LNC|LNCRNA):", "", text)
    text = re.sub(r"\.\d+$", "", text)
    if not text or len(text) > 128 or any(ord(character) < 32 for character in text):
        raise ExternalValidationQueryInputError(
            "lncrna_id is required and must be valid"
        )
    return "LNC:" + text


def _canonical_cancer(value: Any | None) -> str | None:
    if value is None:
        return None
    cancer = str(value).strip().upper()
    if not _CANCER.fullmatch(cancer):
        raise ExternalValidationQueryInputError("cancer_id is invalid")
    return cancer


def _canonical_source(value: Any | None) -> str | None:
    if value is None:
        return None
    source = str(value).strip()
    if not _SOURCE.fullmatch(source):
        raise ExternalValidationQueryInputError("source_database is invalid")
    return source


def _canonical_role(value: Any | None) -> str | None:
    if value is None:
        return None
    role = str(value).strip().lower()
    if role not in _ROLES:
        raise ExternalValidationQueryInputError(
            f"validation_role must be one of {list(_ROLES)}"
        )
    return role


def _canonical_cutoff(value: Any | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ExternalValidationQueryInputError(
            f"k must be one of {list(RANK_CUTOFFS)}"
        )
    try:
        cutoff = int(value)
    except (TypeError, ValueError) as exc:
        raise ExternalValidationQueryInputError(
            f"k must be one of {list(RANK_CUTOFFS)}"
        ) from exc
    if cutoff not in RANK_CUTOFFS:
        raise ExternalValidationQueryInputError(
            f"k must be one of {list(RANK_CUTOFFS)}"
        )
    return cutoff


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise ExternalValidationQueryInputError(
            f"limit must be 1..{MAX_QUERY_LIMIT}"
        )
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise ExternalValidationQueryInputError(
            f"offset must be 0..{MAX_QUERY_OFFSET}"
        )
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


class ExternalValidationReleaseQuery:
    """Validated immutable view of one exact external-validation release."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
        require_formal_authority: bool = True,
    ) -> None:
        source = _safe_file(binding_path, "External-validation binding")
        expected = str(expected_binding_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise ExternalValidationQueryAssetError(
                "External-validation query requires an expected binding SHA256"
            )
        observed = artifact_sha256(source)
        if observed != expected:
            raise ExternalValidationQueryAssetError(
                f"External-validation binding SHA mismatch: {observed} != {expected}"
            )
        binding = _load_json(source, "External-validation binding")
        required = {
            "format": BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": RELEASE_STATUS,
            "module_id": "external_validation",
            "result_role": "SECONDARY_INDEPENDENT_KNOWN_POSITIVE_RANK_RECOVERY",
            "fresh_rank_recovery_calculation": True,
            "pmid_overlap_recomputed": True,
            "training_not_applicable": True,
            "historical_data_use_role": "RAW_OR_TASK_DEFINITION_ONLY",
            "historical_metrics_used": False,
            "historical_details_used": False,
            "historical_ranks_used": False,
            "historical_predictions_used": False,
            "current_v32_predictions_used_as_only_rank_source": True,
            "all_output_rows_generated_current_run": True,
            "changes_primary_ranking": False,
            "release_ready": False,
            "production_deployed": False,
        }
        for key, expected_value in required.items():
            if binding.get(key) != expected_value:
                raise ExternalValidationQueryAssetError(
                    f"External-validation binding has invalid {key}: "
                    f"{binding.get(key)!r}"
                )
        if binding.get("rank_cutoffs") != list(RANK_CUTOFFS):
            raise ExternalValidationQueryAssetError(
                "External-validation rank cutoffs drifted"
            )
        if require_formal_authority and binding.get("formal_authority") is not True:
            raise ExternalValidationQueryAssetError(
                "External-validation binding is not formal authority"
            )
        run_id = str(binding.get("computation_run_id", ""))
        if not re.fullmatch(r"V32-EXTERNAL-VALIDATION-[0-9A-F]{16}", run_id):
            raise ExternalValidationQueryAssetError(
                "External-validation computation_run_id is invalid"
            )

        counts = binding.get("counts")
        required_counts = {
            "task_records",
            "training_union_distinct_pmids",
            "external_overlap_records",
            "primary_eligible_records",
            "primary_unique_positives",
            "primary_rank_available",
            "secondary_unique_positives",
            "secondary_rank_available",
            "detail_rows",
            "metric_rows",
            "available_metric_rows",
            "overlap_audit_rows",
            "fresh_rank_rows",
            "source_summary_rows",
            "rank_cutoffs",
        }
        if not isinstance(counts, dict) or not required_counts.issubset(counts):
            raise ExternalValidationQueryAssetError(
                "External-validation binding counts are incomplete"
            )
        if (
            counts["detail_rows"]
            != counts["primary_unique_positives"]
            + counts["secondary_unique_positives"]
            or counts["rank_cutoffs"] != len(RANK_CUTOFFS)
            or counts["primary_rank_available"] > counts["primary_unique_positives"]
            or counts["secondary_rank_available"]
            > counts["secondary_unique_positives"]
        ):
            raise ExternalValidationQueryAssetError(
                "External-validation count relationships are invalid"
            )
        if require_formal_authority and (
            counts["task_records"] != 645_602
            or counts["training_union_distinct_pmids"] != 8_437
            or counts["primary_unique_positives"] != 7_193
            or counts["primary_rank_available"] != 4_327
            or counts["fresh_rank_rows"] != 76_734
        ):
            raise ExternalValidationQueryAssetError(
                "External-validation formal counts are invalid"
            )

        self._validate_source_inputs(
            binding.get("source_inputs"), require_formal_authority
        )
        success = _load_json(source.parent / "SUCCESS.json", "External-validation SUCCESS")
        if (
            success.get("status") != RELEASE_STATUS
            or success.get("binding") != source.name
            or success.get("binding_sha256") != observed
            or success.get("computation_run_id") != run_id
            or success.get("fresh_rank_recovery_calculation") is not True
            or success.get("pmid_overlap_recomputed") is not True
            or success.get("changes_primary_ranking") is not False
            or success.get("release_ready") is not False
            or success.get("production_deployed") is not False
        ):
            raise ExternalValidationQueryAssetError(
                "External-validation SUCCESS marker is stale"
            )

        artifacts = binding.get("artifacts")
        roles = {
            "metrics",
            "details",
            "overlap_audit",
            "fresh_ranks",
            "source_summary",
            "task_definition_audit",
            "training_pmid_audit",
            "source_inputs",
            "module_lineage",
        }
        if not isinstance(artifacts, dict) or not roles.issubset(artifacts):
            raise ExternalValidationQueryAssetError(
                "External-validation artifact declarations are incomplete"
            )
        paths: dict[str, Path] = {}
        for role in sorted(roles):
            declaration = artifacts[role]
            if (
                not isinstance(declaration, dict)
                or not _SHA256.fullmatch(str(declaration.get("sha256", "")))
            ):
                raise ExternalValidationQueryAssetError(
                    f"External-validation artifact declaration is invalid: {role}"
                )
            path = _safe_file(declaration.get("path", ""), role)
            if artifact_sha256(path) != declaration["sha256"]:
                raise ExternalValidationQueryAssetError(
                    f"External-validation artifact SHA drift: {role}"
                )
            paths[role] = path

        source_inputs = _load_json(paths["source_inputs"], "External-validation inputs")
        task_audit = _load_json(
            paths["task_definition_audit"], "External-validation task audit"
        )
        pmid_audit = _load_json(
            paths["training_pmid_audit"], "External-validation PMID audit"
        )
        lineage = _load_json(paths["module_lineage"], "External-validation lineage")
        if source_inputs != binding["source_inputs"]:
            raise ExternalValidationQueryAssetError(
                "External-validation source-input declaration is inconsistent"
            )
        if (
            task_audit.get("status") != "PASS_DEFINITION_ONLY"
            or task_audit.get("old_external_result_artifacts_opened") is not False
            or task_audit.get("historical_metrics_read") is not False
            or task_audit.get("historical_details_read") is not False
            or task_audit.get("historical_ranks_read") is not False
            or task_audit.get("historical_predictions_read") is not False
            or task_audit.get("historical_derived_columns_explicitly_ignored")
            != list(HISTORICAL_DERIVED_COLUMNS_IGNORED)
            or pmid_audit.get("status")
            != "PASS_RECOMPUTED_FROM_V32_TRAINING_RAW_SOURCES"
            or pmid_audit.get("training_union_distinct_pmids")
            != counts["training_union_distinct_pmids"]
            or pmid_audit.get("external_task_records_with_training_pmid_overlap")
            != counts["external_overlap_records"]
            or pmid_audit.get("primary_records_after_overlap_exclusion")
            != counts["primary_eligible_records"]
            or pmid_audit.get("pmid_values_not_written_as_training_list") is not True
            or lineage.get("status")
            != "SUCCESS_FRESH_V32_KNOWN_POSITIVE_RANK_RECOVERY"
            or lineage.get("computation_run_id") != run_id
        ):
            raise ExternalValidationQueryAssetError(
                "External-validation audit/lineage semantics are invalid"
            )
        for key, expected_value in required.items():
            # The lineage has its own more specific success status, checked
            # above; all shared policy flags must still agree with the binding.
            if key != "status" and key in lineage and lineage.get(key) != expected_value:
                raise ExternalValidationQueryAssetError(
                    f"External-validation lineage has invalid {key}"
                )

        self.binding = binding
        self.binding_path = source
        self.binding_sha256 = observed
        self.computation_run_id = run_id
        self.counts = counts
        self.paths = paths
        self.require_formal_authority = require_formal_authority
        self._validate_tables()

    @staticmethod
    def _validate_declaration(
        declaration: Any,
        label: str,
        formal_sha: str | None,
        require_formal_authority: bool,
    ) -> None:
        if (
            not isinstance(declaration, dict)
            or not _SHA256.fullmatch(str(declaration.get("sha256", "")))
        ):
            raise ExternalValidationQueryAssetError(
                f"External-validation source declaration is invalid: {label}"
            )
        if require_formal_authority and declaration["sha256"] != formal_sha:
            raise ExternalValidationQueryAssetError(
                f"External-validation source is not formal authority: {label}"
            )
        path = _safe_file(declaration.get("path", ""), label)
        if artifact_sha256(path) != declaration["sha256"]:
            raise ExternalValidationQueryAssetError(
                f"External-validation source SHA drift: {label}"
            )

    @classmethod
    def _validate_source_inputs(
        cls, value: Any, require_formal_authority: bool
    ) -> None:
        if not isinstance(value, dict):
            raise ExternalValidationQueryAssetError(
                "External-validation source inputs are missing"
            )
        if (
            value.get("analysis_version") != ANALYSIS_VERSION
            or value.get("input_policy")
            != "HISTORICAL_RAW_OR_TASK_DEFINITION_PLUS_CURRENT_V32_AUTHORITIES_ONLY"
            or any(
                value.get(key) is not False
                for key in (
                    "historical_metrics_used",
                    "historical_details_used",
                    "historical_ranks_used",
                    "historical_predictions_used",
                )
            )
        ):
            raise ExternalValidationQueryAssetError(
                "External-validation source policy is invalid"
            )
        cls._validate_declaration(
            value.get("historical_task_definition"),
            "historical_task_definition",
            FORMAL_TASK_DEFINITION_SHA256,
            require_formal_authority,
        )
        raw = value.get("historical_raw_sources")
        if not isinstance(raw, dict) or set(raw) != set(RAW_SOURCE_AUTHORITIES):
            raise ExternalValidationQueryAssetError(
                "External-validation raw-source declarations are incomplete"
            )
        for role, specification in RAW_SOURCE_AUTHORITIES.items():
            cls._validate_declaration(
                raw[role],
                f"historical_raw_sources.{role}",
                specification["sha256"],
                require_formal_authority,
            )
        training = value.get("v32_training_pmid_sources")
        current = value.get("current_v32_rank_source")
        if not isinstance(training, dict) or not isinstance(current, dict):
            raise ExternalValidationQueryAssetError(
                "External-validation current authority declarations are missing"
            )
        cls._validate_declaration(
            training.get("evidence_event"),
            "v32_training_pmid_sources.evidence_event",
            FORMAL_TRAINING_EVIDENCE_SHA256,
            require_formal_authority,
        )
        cls._validate_declaration(
            training.get("interaction_relation"),
            "v32_training_pmid_sources.interaction_relation",
            FORMAL_TRAINING_INTERACTION_SHA256,
            require_formal_authority,
        )
        cls._validate_declaration(
            current.get("prediction"),
            "current_v32_rank_source.prediction",
            FORMAL_PREDICTION_SHA256,
            require_formal_authority,
        )
        cls._validate_declaration(
            current.get("prediction_lineage"),
            "current_v32_rank_source.prediction_lineage",
            FORMAL_PREDICTION_LINEAGE_SHA256,
            require_formal_authority,
        )
        cls._validate_declaration(
            current.get("candidates"),
            "current_v32_rank_source.candidates",
            FORMAL_CANDIDATE_SHA256,
            require_formal_authority,
        )

    @staticmethod
    def _connect():
        return duckdb.connect(":memory:")

    def _validate_tables(self) -> None:
        relations = {
            key: _relation(self.paths[key])
            for key in (
                "metrics", "details", "overlap_audit", "fresh_ranks", "source_summary"
            )
        }
        required_columns = {
            "metrics": {
                "validation_role", "source_database", "cancer_id", "availability",
                "failure_reason", "task_records_before_exclusion",
                "missing_pmid_records_excluded", "overlapping_pmid_records_excluded",
                "eligible_records_after_exclusion", "n_external_positive_lncrnas",
                "n_external_positive_in_candidate_universe", "n_ranked_lncrnas",
                "median_positive_rank", "median_positive_percentile", "k",
                "effective_k", "hits_at_k", "annotation_hit_rate_at_k", "recall_at_k",
                "expected_hits_at_k", "fold_enrichment_at_k", "analysis_version",
                "computation_run_id",
            },
            "details": {
                "source_database", "cancer_id", "lncrna_id", "evidence_record_count",
                "independent_pmid_count", "independent_pmids", "validation_role",
                "known_positive_basis", "pmid_overlap_exclusion_applied",
                "training_pmid_overlap_in_used_records", "overlapping_records_excluded",
                "missing_pmid_records_excluded", "lncrna_rank_score", "rank",
                "n_ranked_lncrnas", "percentile", "availability", "failure_reason",
                *(f"recovered_at_{cutoff}" for cutoff in RANK_CUTOFFS),
                "analysis_version", "computation_run_id",
            },
            "overlap_audit": {
                "source_database", "pmid_normalized", "external_records",
                "training_evidence_event_present", "training_interaction_relation_present",
                "training_pmid_overlap", "pmid_available", "excluded_from_primary",
                "exclusion_reason", "analysis_version", "computation_run_id",
            },
            "fresh_ranks": {
                "cancer_id", "lncrna_id", "max_pathway_probability",
                "mean_top5_pathway_probability", "n_top_pathways", "top_pathway_id",
                "lncrna_rank_score", "rank", "n_ranked_lncrnas", "percentile",
                "ranking_policy", "analysis_version", "computation_run_id",
            },
            "source_summary": {
                "source_database", "task_records", "mapped_context_records",
                "pmid_available_records", "training_pmid_overlap_records",
                "primary_eligible_records", "secondary_eligible_records",
                "unique_positive_contexts", "rank_available_contexts",
                "analysis_version", "computation_run_id",
            },
        }
        con = self._connect()
        try:
            for role, required in required_columns.items():
                columns = {
                    row[0]
                    for row in con.execute(
                        f"DESCRIBE SELECT * FROM {relations[role]}"
                    ).fetchall()
                }
                if missing := sorted(required - columns):
                    raise ExternalValidationQueryAssetError(
                        f"External-validation {role} lacks columns: {missing}"
                    )
            rank_audit = con.execute(
                f"""
                SELECT count(*), count(DISTINCT (cancer_id, lncrna_id)),
                       count_if(analysis_version <> ? OR computation_run_id <> ?),
                       count_if(rank < 1 OR rank > n_ranked_lncrnas
                                OR n_ranked_lncrnas < 1 OR n_top_pathways < 1
                                OR n_top_pathways > 5
                                OR max_pathway_probability NOT BETWEEN 0 AND 1
                                OR mean_top5_pathway_probability NOT BETWEEN 0 AND 1
                                OR lncrna_rank_score NOT BETWEEN 0 AND 1
                                OR percentile NOT BETWEEN 0 AND 1)
                FROM {relations['fresh_ranks']}
                """,
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
            rank_partition_failures = con.execute(
                f"""
                SELECT count_if(rows <> n_ranked OR distinct_ranks <> n_ranked
                                OR min_rank <> 1 OR max_rank <> n_ranked)
                FROM (
                    SELECT cancer_id, count(*) AS rows, count(DISTINCT rank) AS distinct_ranks,
                           min(rank) AS min_rank, max(rank) AS max_rank,
                           max(n_ranked_lncrnas) AS n_ranked
                    FROM {relations['fresh_ranks']}
                    GROUP BY cancer_id
                )
                """
            ).fetchone()[0]
            hit_invalid = " OR ".join(
                f"recovered_at_{cutoff} IS DISTINCT FROM "
                f"(availability AND rank <= {cutoff})"
                for cutoff in RANK_CUTOFFS
            )
            detail_audit = con.execute(
                f"""
                SELECT count(*),
                       count(DISTINCT (validation_role, source_database, cancer_id, lncrna_id)),
                       count_if(validation_role = 'primary_known_positive'),
                       count_if(validation_role = 'primary_known_positive' AND availability),
                       count_if(analysis_version <> ? OR computation_run_id <> ?),
                       count_if(validation_role NOT IN {tuple(_ROLES)}
                                OR pmid_overlap_exclusion_applied IS DISTINCT FROM true
                                OR training_pmid_overlap_in_used_records IS DISTINCT FROM false
                                OR (validation_role = 'primary_known_positive'
                                    AND independent_pmid_count <= 0)
                                OR (availability AND (rank IS NULL OR failure_reason IS NOT NULL))
                                OR (NOT availability AND
                                    (rank IS NOT NULL OR failure_reason <>
                                     'NOT_IN_CURRENT_V32_CANDIDATE_UNIVERSE'))
                                OR {hit_invalid})
                FROM {relations['details']}
                """,
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
            metric_audit = con.execute(
                f"""
                SELECT count(*),
                       count(DISTINCT (validation_role, source_database, cancer_id, k)),
                       count_if(availability),
                       count_if(analysis_version <> ? OR computation_run_id <> ?),
                       count_if(validation_role NOT IN {tuple(_ROLES)}
                                OR k NOT IN {tuple(RANK_CUTOFFS)}
                                OR effective_k < 0 OR effective_k > k
                                OR n_external_positive_in_candidate_universe
                                   > n_external_positive_lncrnas
                                OR (availability AND
                                    (failure_reason IS NOT NULL OR hits_at_k IS NULL
                                     OR hits_at_k < 0
                                     OR hits_at_k > n_external_positive_in_candidate_universe
                                     OR recall_at_k NOT BETWEEN 0 AND 1
                                     OR annotation_hit_rate_at_k NOT BETWEEN 0 AND 1))
                                OR (NOT availability AND
                                    (failure_reason IS NULL OR hits_at_k IS NOT NULL)))
                FROM {relations['metrics']}
                """,
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
            metric_detail_mismatch = con.execute(
                f"""
                WITH detail AS (
                    SELECT validation_role, source_database, cancer_id,
                           count(*) AS positives,
                           count_if(availability) AS matched
                    FROM {relations['details']}
                    GROUP BY ALL
                )
                SELECT count_if(
                    metric.n_external_positive_lncrnas <> coalesce(detail.positives, 0)
                    OR metric.n_external_positive_in_candidate_universe
                       <> coalesce(detail.matched, 0)
                )
                FROM {relations['metrics']} AS metric
                LEFT JOIN detail USING (validation_role, source_database, cancer_id)
                """
            ).fetchone()[0]
            rank_detail_mismatch = con.execute(
                f"""
                SELECT count(*)
                FROM {relations['details']} AS detail
                LEFT JOIN {relations['fresh_ranks']} AS rank
                  USING (cancer_id, lncrna_id)
                WHERE detail.availability IS DISTINCT FROM (rank.rank IS NOT NULL)
                   OR (detail.availability AND
                       (detail.rank <> rank.rank
                        OR abs(detail.lncrna_rank_score-rank.lncrna_rank_score) > 1e-12))
                """
            ).fetchone()[0]
            overlap_audit = con.execute(
                f"""
                SELECT count(*),
                       count_if(analysis_version <> ? OR computation_run_id <> ?),
                       count_if(training_pmid_overlap IS DISTINCT FROM
                                (training_evidence_event_present
                                 OR training_interaction_relation_present)
                                OR pmid_available IS DISTINCT FROM
                                   (pmid_normalized IS NOT NULL)
                                OR excluded_from_primary IS DISTINCT FROM
                                   (NOT pmid_available OR training_pmid_overlap))
                FROM {relations['overlap_audit']}
                """,
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
            overlap_duplicates = con.execute(
                f"""
                SELECT count(*) FROM (
                    SELECT source_database, pmid_normalized, count(*) AS n
                    FROM {relations['overlap_audit']}
                    GROUP BY ALL HAVING n <> 1
                )
                """
            ).fetchone()[0]
            summary_audit = con.execute(
                f"""
                SELECT count(*), count(DISTINCT source_database),
                       count_if(analysis_version <> ? OR computation_run_id <> ?
                                OR task_records < mapped_context_records
                                OR task_records < pmid_available_records
                                OR rank_available_contexts > unique_positive_contexts)
                FROM {relations['source_summary']}
                """,
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
        finally:
            con.close()

        expected = self.counts
        artifact_rows = {
            role: self.binding["artifacts"][role].get("rows")
            for role in (
                "metrics", "details", "overlap_audit", "fresh_ranks", "source_summary"
            )
        }
        if (
            int(rank_audit[0]) != int(expected["fresh_rank_rows"])
            or int(rank_audit[1]) != int(expected["fresh_rank_rows"])
            or any(int(value or 0) for value in rank_audit[2:])
            or int(rank_partition_failures or 0)
            or int(detail_audit[0]) != int(expected["detail_rows"])
            or int(detail_audit[1]) != int(expected["detail_rows"])
            or int(detail_audit[2]) != int(expected["primary_unique_positives"])
            or int(detail_audit[3]) != int(expected["primary_rank_available"])
            or any(int(value or 0) for value in detail_audit[4:])
            or int(metric_audit[0]) != int(expected["metric_rows"])
            or int(metric_audit[1]) != int(expected["metric_rows"])
            or int(metric_audit[2]) != int(expected["available_metric_rows"])
            or any(int(value or 0) for value in metric_audit[3:])
            or int(metric_detail_mismatch or 0)
            or int(rank_detail_mismatch or 0)
            or int(overlap_audit[0]) != int(expected["overlap_audit_rows"])
            or any(int(value or 0) for value in overlap_audit[1:])
            or int(overlap_duplicates or 0)
            or int(summary_audit[0]) != int(expected["source_summary_rows"])
            or int(summary_audit[1]) != int(expected["source_summary_rows"])
            or int(summary_audit[2] or 0)
            or artifact_rows
            != {
                "metrics": expected["metric_rows"],
                "details": expected["detail_rows"],
                "overlap_audit": expected["overlap_audit_rows"],
                "fresh_ranks": expected["fresh_rank_rows"],
                "source_summary": expected["source_summary_rows"],
            }
        ):
            raise ExternalValidationQueryAssetError(
                "External-validation output semantic validation failed"
            )

    def _envelope(
        self, query_kind: str, filters: dict[str, Any], rows: pd.DataFrame
    ) -> dict[str, Any]:
        return {
            "module": "external_validation",
            "query_kind": query_kind,
            "result_role": self.binding["result_role"],
            "fresh_rank_recovery_calculation": True,
            "pmid_overlap_recomputed": True,
            "training_not_applicable": True,
            "changes_primary_ranking": False,
            "filters": filters,
            "returned_rows": len(rows),
            "rows": _records(rows),
            "provenance": {
                "analysis_version": ANALYSIS_VERSION,
                "computation_run_id": self.computation_run_id,
                "binding_path": str(self.binding_path),
                "binding_sha256": self.binding_sha256,
                "historical_data_use_role": "RAW_OR_TASK_DEFINITION_ONLY",
                "historical_metrics_used": False,
                "historical_details_used": False,
                "historical_ranks_used": False,
                "historical_predictions_used": False,
                "current_v32_predictions_used_as_only_rank_source": True,
                "release_ready": False,
                "production_deployed": False,
            },
        }

    def query_metrics(
        self,
        *,
        validation_role: Any | None = None,
        source_database: Any | None = None,
        cancer_id: Any | None = None,
        k: Any | None = None,
        availability: bool | None = None,
        limit: int = MAX_QUERY_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        role = _canonical_role(validation_role)
        source = _canonical_source(source_database)
        cancer = _canonical_cancer(cancer_id)
        cutoff = _canonical_cutoff(k)
        limit, offset = _bounds(limit, offset)
        if availability is not None and not isinstance(availability, bool):
            raise ExternalValidationQueryInputError("availability must be boolean")
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("validation_role", role),
            ("cancer_id", cancer),
            ("k", cutoff),
            ("availability", availability),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if source is not None:
            clauses.append("lower(source_database) = lower(?)")
            parameters.append(source)
        where = " AND ".join(clauses) if clauses else "true"
        con = self._connect()
        try:
            rows = con.execute(
                f"""
                SELECT * FROM {_relation(self.paths['metrics'])}
                WHERE {where}
                ORDER BY validation_role, source_database, cancer_id, k
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        response = self._envelope(
            "cohort_metrics",
            {
                "validation_role": role,
                "source_database": source,
                "cancer_id": cancer,
                "k": cutoff,
                "availability": availability,
                "limit": limit,
                "offset": offset,
            },
            rows,
        )
        return response

    def query_lncrna(
        self,
        *,
        lncrna_id: Any,
        cancer_id: Any | None = None,
        validation_role: Any | None = None,
        source_database: Any | None = None,
        limit: int = MAX_QUERY_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        lnc = _canonical_lnc(lncrna_id)
        cancer = _canonical_cancer(cancer_id)
        role = _canonical_role(validation_role)
        source = _canonical_source(source_database)
        limit, offset = _bounds(limit, offset)
        clauses = ["lncrna_id = ?"]
        parameters: list[Any] = [lnc]
        for column, value in (("cancer_id", cancer), ("validation_role", role)):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if source is not None:
            clauses.append("lower(source_database) = lower(?)")
            parameters.append(source)
        con = self._connect()
        try:
            rows = con.execute(
                f"""
                SELECT * FROM {_relation(self.paths['details'])}
                WHERE {' AND '.join(clauses)}
                ORDER BY validation_role, source_database, cancer_id, lncrna_id
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return self._envelope(
            "lncrna_known_positive",
            {
                "lncrna_id": lnc,
                "cancer_id": cancer,
                "validation_role": role,
                "source_database": source,
                "limit": limit,
                "offset": offset,
            },
            rows,
        )

    def query_overlap(
        self,
        *,
        pmid: Any | None = None,
        source_database: Any | None = None,
        training_pmid_overlap: bool | None = None,
        limit: int = MAX_QUERY_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        source = _canonical_source(source_database)
        normalized = None if pmid is None else normalize_pmid(pmid)
        if pmid is not None and not normalized:
            raise ExternalValidationQueryInputError("pmid is invalid")
        if training_pmid_overlap is not None and not isinstance(
            training_pmid_overlap, bool
        ):
            raise ExternalValidationQueryInputError(
                "training_pmid_overlap must be boolean"
            )
        limit, offset = _bounds(limit, offset)
        clauses: list[str] = []
        parameters: list[Any] = []
        if normalized is not None:
            clauses.append("pmid_normalized = ?")
            parameters.append(normalized)
        if source is not None:
            clauses.append("lower(source_database) = lower(?)")
            parameters.append(source)
        if training_pmid_overlap is not None:
            clauses.append("training_pmid_overlap = ?")
            parameters.append(training_pmid_overlap)
        where = " AND ".join(clauses) if clauses else "true"
        con = self._connect()
        try:
            rows = con.execute(
                f"""
                SELECT * FROM {_relation(self.paths['overlap_audit'])}
                WHERE {where}
                ORDER BY source_database, pmid_available DESC, pmid_normalized
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return self._envelope(
            "pmid_overlap_audit",
            {
                "pmid": normalized,
                "source_database": source,
                "training_pmid_overlap": training_pmid_overlap,
                "limit": limit,
                "offset": offset,
            },
            rows,
        )


__all__ = [
    "ExternalValidationQueryAssetError",
    "ExternalValidationQueryError",
    "ExternalValidationQueryInputError",
    "ExternalValidationReleaseQuery",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
]
