"""Hash-pinned, read-only queries for fresh V3.2 lncRNA survival facts."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

if __package__:
    from .clinical_km_release import (
        ANALYSIS_VERSION,
        BINDING_FORMAT,
        DFS_FAILURE_REASON,
        EXPRESSION_ROOT_SHA256,
        FORMAL_CANDIDATE_PAIRS,
        FORMAL_CANDIDATE_SHA256,
        FORMAL_STATISTICS_ROWS,
        HORIZON_YEARS,
        RELEASE_STATUS,
        WORKBOOK_SHA256,
        artifact_sha256,
    )
    from .clinical_training import CLINICAL_ENDPOINTS
else:  # pragma: no cover - standalone import support
    from clinical_km_release import (  # type: ignore
        ANALYSIS_VERSION,
        BINDING_FORMAT,
        DFS_FAILURE_REASON,
        EXPRESSION_ROOT_SHA256,
        FORMAL_CANDIDATE_PAIRS,
        FORMAL_CANDIDATE_SHA256,
        FORMAL_STATISTICS_ROWS,
        HORIZON_YEARS,
        RELEASE_STATUS,
        WORKBOOK_SHA256,
        artifact_sha256,
    )
    from clinical_training import CLINICAL_ENDPOINTS  # type: ignore


MAX_QUERY_LIMIT = len(CLINICAL_ENDPOINTS) * 33
MAX_QUERY_OFFSET = FORMAL_STATISTICS_ROWS - 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANCER = re.compile(r"^[A-Z0-9_-]{2,16}$")


class ClinicalKMQueryError(RuntimeError):
    """Base clinical-KM query error."""


class ClinicalKMQueryAssetError(ClinicalKMQueryError):
    """Raised when a binding, authority, or output artifact is stale."""


class ClinicalKMQueryInputError(ClinicalKMQueryError):
    """Raised when query filters are invalid."""


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ClinicalKMQueryAssetError(f"{label} is missing or unsafe: {source}")
    return source


def _safe_artifact(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if source.is_symlink() or (not source.is_file() and not source.is_dir()):
        raise ClinicalKMQueryAssetError(f"{label} is missing or unsafe: {source}")
    if source.is_dir():
        files = [item for item in source.rglob("*") if item.is_file()]
        if not files or any(item.is_symlink() for item in files):
            raise ClinicalKMQueryAssetError(f"{label} is empty or unsafe: {source}")
    return source


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClinicalKMQueryAssetError(f"{label} is missing or invalid JSON") from exc
    if not isinstance(value, dict):
        raise ClinicalKMQueryAssetError(f"{label} must be a JSON object")
    return value


def _sql_text(value: str) -> str:
    quote = chr(39)
    return quote + value.replace(quote, quote * 2) + quote


def _parquet_relation(path: Path) -> str:
    if path.is_dir():
        pattern = path.resolve().as_posix().rstrip("/") + "/**/*.parquet"
        return f"read_parquet({_sql_text(pattern)}, hive_partitioning=false)"
    return f"read_parquet({_sql_text(path.resolve().as_posix())})"


def _canonical_lnc(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"^(?:LNC|LNCRNA):", "", text)
    text = re.sub(r"\.\d+$", "", text)
    if not text or len(text) > 128 or any(ord(character) < 32 for character in text):
        raise ClinicalKMQueryInputError("lncrna_id is required and must be valid")
    return "LNC:" + text


def _canonical_cancer(value: Any | None) -> str | None:
    if value is None:
        return None
    cancer = str(value).strip().upper()
    if not _CANCER.fullmatch(cancer):
        raise ClinicalKMQueryInputError("cancer_id is invalid")
    return cancer


def _canonical_endpoint(value: Any | None) -> str | None:
    if value is None:
        return None
    endpoint = str(value).strip().upper()
    if endpoint not in CLINICAL_ENDPOINTS:
        raise ClinicalKMQueryInputError(
            f"clinical_endpoint must be one of {list(CLINICAL_ENDPOINTS)}"
        )
    return endpoint


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise ClinicalKMQueryInputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise ClinicalKMQueryInputError(f"offset must be 0..{MAX_QUERY_OFFSET}")
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


class ClinicalKMReleaseQuery:
    """Validated immutable view of one exact clinical-KM release binding."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
        require_formal_authority: bool = True,
    ) -> None:
        source = _safe_file(binding_path, "Clinical KM binding")
        expected = str(expected_binding_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise ClinicalKMQueryAssetError(
                "Clinical KM query requires an expected binding SHA256"
            )
        observed = artifact_sha256(source)
        if observed != expected:
            raise ClinicalKMQueryAssetError(
                f"Clinical KM binding SHA mismatch: {observed} != {expected}"
            )
        binding = _load_json(source, "Clinical KM binding")
        required = {
            "format": BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": RELEASE_STATUS,
            "module_id": "clinical_lncrna_kaplan_meier",
            "result_role": "SECONDARY_FRESH_LNCRNA_SURVIVAL_STATISTICS",
            "fresh_statistical_calculation": True,
            "training_not_applicable": True,
            "all_output_rows_generated_current_run": True,
            "historical_derived_outputs_used": False,
            "historical_predictions_used": False,
            "historical_checkpoints_used": False,
            "historical_rankings_used": False,
            "historical_web_tables_used": False,
            "changes_primary_ranking": False,
            "release_ready": False,
            "production_deployed": False,
            "dfs_policy": DFS_FAILURE_REASON,
        }
        for key, expected_value in required.items():
            if binding.get(key) != expected_value:
                raise ClinicalKMQueryAssetError(
                    f"Clinical KM binding has invalid {key}: {binding.get(key)!r}"
                )
        if binding.get("clinical_endpoints") != list(CLINICAL_ENDPOINTS):
            raise ClinicalKMQueryAssetError("Clinical KM binding lacks all six endpoints")
        if binding.get("horizon_years") != list(HORIZON_YEARS):
            raise ClinicalKMQueryAssetError("Clinical KM horizon policy drifted")
        if require_formal_authority and binding.get("formal_authority") is not True:
            raise ClinicalKMQueryAssetError("Clinical KM binding is not formal authority")
        run_id = str(binding.get("computation_run_id", ""))
        if not re.fullmatch(r"V32-CLINICAL-KM-[0-9A-F]{16}", run_id):
            raise ClinicalKMQueryAssetError("Clinical KM computation_run_id is invalid")

        counts = binding.get("counts")
        required_counts = {
            "candidate_pairs", "statistics_rows", "available_rows", "unavailable_rows",
            "curve_rows", "cancers", "lncrnas", "endpoints", "dfs_rows",
            "dfs_available_rows", "horizons",
        }
        if not isinstance(counts, dict) or not required_counts.issubset(counts):
            raise ClinicalKMQueryAssetError("Clinical KM binding counts are incomplete")
        if require_formal_authority and (
            counts.get("candidate_pairs") != FORMAL_CANDIDATE_PAIRS
            or counts.get("statistics_rows") != FORMAL_STATISTICS_ROWS
            or counts.get("cancers") != 33
            or counts.get("lncrnas") != 8_541
        ):
            raise ClinicalKMQueryAssetError("Clinical KM formal counts are invalid")
        if (
            counts.get("statistics_rows")
            != counts.get("available_rows") + counts.get("unavailable_rows")
            or counts.get("endpoints") != len(CLINICAL_ENDPOINTS)
            or counts.get("dfs_rows") != counts.get("candidate_pairs")
            or counts.get("dfs_available_rows") != 0
            or counts.get("horizons") != len(HORIZON_YEARS)
            or counts.get("curve_rows")
            != counts.get("available_rows") * 2 * len(HORIZON_YEARS)
        ):
            raise ClinicalKMQueryAssetError("Clinical KM count relationships are invalid")

        authorities = binding.get("authorities")
        if not isinstance(authorities, dict):
            raise ClinicalKMQueryAssetError("Clinical KM authorities are missing")
        formal_hashes = {
            "tcga_cdr_workbook": WORKBOOK_SHA256,
            "current_patient_logcpm": EXPRESSION_ROOT_SHA256,
            "current_exact_candidates": FORMAL_CANDIDATE_SHA256,
        }
        for role, expected_sha in formal_hashes.items():
            declaration = authorities.get(role)
            if (
                not isinstance(declaration, dict)
                or not _SHA256.fullmatch(str(declaration.get("sha256", "")))
            ):
                raise ClinicalKMQueryAssetError(
                    f"Clinical KM authority declaration is invalid: {role}"
                )
            if require_formal_authority and declaration["sha256"] != expected_sha:
                raise ClinicalKMQueryAssetError(
                    f"Clinical KM source is not formal authority: {role}"
                )
            authority_path = _safe_artifact(declaration.get("path", ""), role)
            if artifact_sha256(authority_path) != declaration["sha256"]:
                raise ClinicalKMQueryAssetError(f"Clinical KM source SHA drift: {role}")

        success = _load_json(source.parent / "SUCCESS.json", "Clinical KM SUCCESS")
        if (
            success.get("status") != RELEASE_STATUS
            or success.get("binding") != source.name
            or success.get("binding_sha256") != observed
            or success.get("computation_run_id") != run_id
            or success.get("fresh_statistical_calculation") is not True
            or success.get("training_not_applicable") is not True
            or success.get("changes_primary_ranking") is not False
            or success.get("release_ready") is not False
            or success.get("production_deployed") is not False
        ):
            raise ClinicalKMQueryAssetError("Clinical KM SUCCESS marker is stale")

        artifacts = binding.get("artifacts")
        roles = {"statistics", "curves", "source_inputs", "module_lineage"}
        if not isinstance(artifacts, dict) or not roles.issubset(artifacts):
            raise ClinicalKMQueryAssetError("Clinical KM artifact declarations are incomplete")
        paths: dict[str, Path] = {}
        for role in sorted(roles):
            declaration = artifacts[role]
            if (
                not isinstance(declaration, dict)
                or not _SHA256.fullmatch(str(declaration.get("sha256", "")))
            ):
                raise ClinicalKMQueryAssetError(
                    f"Clinical KM artifact declaration is invalid: {role}"
                )
            path = _safe_artifact(declaration.get("path", ""), role)
            if artifact_sha256(path) != declaration["sha256"]:
                raise ClinicalKMQueryAssetError(f"Clinical KM artifact SHA drift: {role}")
            paths[role] = path
        source_inputs = _load_json(paths["source_inputs"], "Clinical KM source inputs")
        lineage = _load_json(paths["module_lineage"], "Clinical KM lineage")
        if (
            source_inputs.get("fresh_statistical_calculation") is not True
            or source_inputs.get("training_not_applicable") is not True
            or source_inputs.get("historical_derived_outputs_used") is not False
            or lineage.get("status") != "SUCCESS_FRESH_V32_STATISTICAL_CALCULATION"
            or lineage.get("computation_run_id") != run_id
            or lineage.get("all_output_rows_generated_current_run") is not True
            or lineage.get("historical_derived_outputs_used") is not False
            or lineage.get("changes_primary_ranking") is not False
            or lineage.get("release_ready") is not False
        ):
            raise ClinicalKMQueryAssetError("Clinical KM source/lineage semantics are invalid")

        self.binding = binding
        self.binding_path = source
        self.binding_sha256 = observed
        self.computation_run_id = run_id
        self.counts = counts
        self.paths = paths
        self.require_formal_authority = require_formal_authority
        self._validate_tables()

    @staticmethod
    def _connect():
        return duckdb.connect(":memory:")

    def _validate_tables(self) -> None:
        statistics = _parquet_relation(self.paths["statistics"])
        curves = _parquet_relation(self.paths["curves"])
        required_statistics = {
            "cancer_id", "lncrna_id", "clinical_endpoint", "availability",
            "failure_reason", "endpoint_source", "n_expression_patients",
            "n_endpoint_patients", "n_events", "median_logcpm", "n_high", "n_low",
            "events_high", "events_low", "logrank_observed_minus_expected_high",
            "logrank_variance", "logrank_z_high_vs_low", "logrank_chi_square",
            "logrank_p_value", "grouping_method", "statistical_method",
            "analysis_version", "computation_run_id",
        }
        required_curves = {
            "cancer_id", "lncrna_id", "clinical_endpoint", "expression_group",
            "horizon_years", "horizon_days", "n_at_risk", "cumulative_events",
            "survival_probability", "analysis_version", "computation_run_id",
        }
        con = self._connect()
        try:
            statistics_columns = {
                row[0]
                for row in con.execute(
                    f"DESCRIBE SELECT * FROM {statistics}"
                ).fetchall()
            }
            curves_columns = {
                row[0]
                for row in con.execute(f"DESCRIBE SELECT * FROM {curves}").fetchall()
            }
            if missing := sorted(required_statistics - statistics_columns):
                raise ClinicalKMQueryAssetError(
                    f"Clinical KM statistics lack columns: {missing}"
                )
            if missing := sorted(required_curves - curves_columns):
                raise ClinicalKMQueryAssetError(
                    f"Clinical KM curves lack columns: {missing}"
                )
            audit = con.execute(
                f"""
                SELECT count(*) AS rows,
                       count(DISTINCT (cancer_id, lncrna_id, clinical_endpoint)) AS keys,
                       count(DISTINCT (cancer_id, lncrna_id)) AS pairs,
                       count(DISTINCT cancer_id) AS cancers,
                       count(DISTINCT lncrna_id) AS lncrnas,
                       count(DISTINCT clinical_endpoint) AS endpoints,
                       count_if(availability) AS available,
                       count_if(analysis_version <> ? OR computation_run_id <> ?) AS stale,
                       count_if(availability AND
                                (failure_reason IS NOT NULL
                                 OR n_endpoint_patients IS NULL OR n_events IS NULL
                                 OR n_endpoint_patients <= 0 OR n_events <= 0
                                 OR n_events > n_endpoint_patients
                                 OR n_high IS NULL OR n_low IS NULL
                                 OR events_high IS NULL OR events_low IS NULL
                                 OR n_high <= 0 OR n_low <= 0
                                 OR n_high + n_low <> n_endpoint_patients
                                 OR events_high + events_low <> n_events
                                 OR events_high > n_high OR events_low > n_low
                                 OR logrank_variance <= 0
                                 OR logrank_z_high_vs_low IS NULL
                                 OR logrank_chi_square IS NULL
                                 OR logrank_p_value NOT BETWEEN 0 AND 1)) AS bad_available,
                       count_if(NOT availability AND
                                (failure_reason IS NULL
                                 OR logrank_observed_minus_expected_high IS NOT NULL
                                 OR logrank_variance IS NOT NULL
                                 OR logrank_z_high_vs_low IS NOT NULL
                                 OR logrank_chi_square IS NOT NULL
                                 OR logrank_p_value IS NOT NULL)) AS bad_unavailable,
                       count_if(clinical_endpoint = 'DFS' AND
                                (availability OR failure_reason <> ?
                                 OR endpoint_source <> ?
                                 OR median_logcpm IS NOT NULL
                                 OR n_high IS NOT NULL OR n_low IS NOT NULL)) AS bad_dfs,
                       count_if(clinical_endpoint <> 'DFS'
                                AND failure_reason = ?) AS copied_dfi_to_dfs
                FROM {statistics}
                """,
                [
                    ANALYSIS_VERSION,
                    self.computation_run_id,
                    DFS_FAILURE_REASON,
                    DFS_FAILURE_REASON,
                    DFS_FAILURE_REASON,
                ],
            ).fetchone()
            endpoint_sizes = con.execute(
                f"""
                SELECT min(n), max(n)
                FROM (
                    SELECT cancer_id, lncrna_id,
                           count(DISTINCT clinical_endpoint) AS n
                    FROM {statistics}
                    GROUP BY cancer_id, lncrna_id
                )
                """
            ).fetchone()
            endpoint_values = {
                row[0]
                for row in con.execute(
                    f"SELECT DISTINCT clinical_endpoint FROM {statistics}"
                ).fetchall()
            }
            curve_audit = con.execute(
                f"""
                SELECT count(*) AS rows,
                       count(DISTINCT (cancer_id, lncrna_id, clinical_endpoint,
                                       expression_group, horizon_years)) AS keys,
                       count_if(expression_group NOT IN ('HIGH', 'LOW')
                                OR horizon_years NOT IN (0, 1, 2, 3, 5, 10)
                                OR abs(horizon_days - horizon_years * 365.25) > 1e-9
                                OR n_at_risk < 0 OR cumulative_events < 0
                                OR survival_probability NOT BETWEEN 0 AND 1
                                OR analysis_version <> ?
                                OR computation_run_id <> ?) AS invalid
                FROM {curves}
                """,
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
            curve_key_audit = con.execute(
                f"""
                SELECT count_if(n <> 12 OR groups <> 2 OR horizons <> 6)
                FROM (
                    SELECT cancer_id, lncrna_id, clinical_endpoint,
                           count(*) AS n,
                           count(DISTINCT expression_group) AS groups,
                           count(DISTINCT horizon_years) AS horizons
                    FROM {curves}
                    GROUP BY cancer_id, lncrna_id, clinical_endpoint
                )
                """
            ).fetchone()[0]
            curve_stat_mismatch = con.execute(
                f"""
                SELECT count(*) FROM (
                    (SELECT DISTINCT cancer_id, lncrna_id, clinical_endpoint
                     FROM {curves}
                     EXCEPT
                     SELECT cancer_id, lncrna_id, clinical_endpoint
                     FROM {statistics} WHERE availability)
                    UNION ALL
                    (SELECT cancer_id, lncrna_id, clinical_endpoint
                     FROM {statistics} WHERE availability
                     EXCEPT
                     SELECT DISTINCT cancer_id, lncrna_id, clinical_endpoint
                     FROM {curves})
                )
                """
            ).fetchone()[0]
            monotonic_failures = con.execute(
                f"""
                SELECT count_if(
                    (previous_risk IS NOT NULL AND n_at_risk > previous_risk)
                    OR (previous_survival IS NOT NULL
                        AND survival_probability > previous_survival + 1e-12)
                )
                FROM (
                    SELECT *,
                           lag(n_at_risk) OVER (
                               PARTITION BY cancer_id, lncrna_id, clinical_endpoint,
                                            expression_group
                               ORDER BY horizon_years
                           ) AS previous_risk,
                           lag(survival_probability) OVER (
                               PARTITION BY cancer_id, lncrna_id, clinical_endpoint,
                                            expression_group
                               ORDER BY horizon_years
                           ) AS previous_survival
                    FROM {curves}
                )
                """
            ).fetchone()[0]
        finally:
            con.close()

        expected = self.counts
        if (
            int(audit[0]) != int(expected["statistics_rows"])
            or int(audit[1]) != int(expected["statistics_rows"])
            or int(audit[2]) != int(expected["candidate_pairs"])
            or int(audit[3]) != int(expected["cancers"])
            or int(audit[4]) != int(expected["lncrnas"])
            or int(audit[5]) != len(CLINICAL_ENDPOINTS)
            or int(audit[6]) != int(expected["available_rows"])
            or any(int(value or 0) for value in audit[7:])
            or tuple(map(int, endpoint_sizes)) != (
                len(CLINICAL_ENDPOINTS), len(CLINICAL_ENDPOINTS)
            )
            or endpoint_values != set(CLINICAL_ENDPOINTS)
            or int(curve_audit[0]) != int(expected["curve_rows"])
            or int(curve_audit[1]) != int(expected["curve_rows"])
            or int(curve_audit[2] or 0)
            or int(curve_key_audit or 0)
            or int(curve_stat_mismatch or 0)
            or int(monotonic_failures or 0)
        ):
            raise ClinicalKMQueryAssetError(
                "Clinical KM output semantic validation failed"
            )

    def query_lncrna_survival(
        self,
        *,
        lncrna_id: Any,
        cancer_id: Any | None = None,
        clinical_endpoint: Any | None = None,
        availability: bool | None = None,
        include_curves: bool = True,
        limit: int = MAX_QUERY_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        lnc = _canonical_lnc(lncrna_id)
        cancer = _canonical_cancer(cancer_id)
        endpoint = _canonical_endpoint(clinical_endpoint)
        limit, offset = _bounds(limit, offset)
        if availability is not None and not isinstance(availability, bool):
            raise ClinicalKMQueryInputError("availability must be boolean")
        if not isinstance(include_curves, bool):
            raise ClinicalKMQueryInputError("include_curves must be boolean")
        clauses = ["lncrna_id = ?"]
        parameters: list[Any] = [lnc]
        if cancer is not None:
            clauses.append("cancer_id = ?")
            parameters.append(cancer)
        if endpoint is not None:
            clauses.append("clinical_endpoint = ?")
            parameters.append(endpoint)
        if availability is not None:
            clauses.append("availability = ?")
            parameters.append(availability)
        statistics_relation = _parquet_relation(self.paths["statistics"])
        curves_relation = _parquet_relation(self.paths["curves"])
        con = self._connect()
        try:
            statistics = con.execute(
                f"""
                SELECT * FROM {statistics_relation}
                WHERE {' AND '.join(clauses)}
                ORDER BY cancer_id,
                         CASE clinical_endpoint
                             WHEN 'OS' THEN 0 WHEN 'DSS' THEN 1 WHEN 'PFI' THEN 2
                             WHEN 'PFS' THEN 3 WHEN 'DFI' THEN 4 WHEN 'DFS' THEN 5
                         END
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
            curves = pd.DataFrame()
            if include_curves and not statistics.empty:
                selected = statistics[
                    ["cancer_id", "lncrna_id", "clinical_endpoint"]
                ].drop_duplicates()
                con.register("selected_clinical_km", selected)
                curves = con.execute(
                    f"""
                    SELECT curve.*
                    FROM {curves_relation} AS curve
                    INNER JOIN selected_clinical_km AS selected
                    USING (cancer_id, lncrna_id, clinical_endpoint)
                    ORDER BY cancer_id,
                             CASE clinical_endpoint
                                 WHEN 'OS' THEN 0 WHEN 'DSS' THEN 1 WHEN 'PFI' THEN 2
                                 WHEN 'PFS' THEN 3 WHEN 'DFI' THEN 4 WHEN 'DFS' THEN 5
                             END,
                             expression_group, horizon_years
                    """
                ).fetchdf()
        finally:
            con.close()
        return {
            "module": "clinical_lncrna_kaplan_meier",
            "query_kind": "lncrna_survival",
            "result_role": self.binding["result_role"],
            "fresh_statistical_calculation": True,
            "training_not_applicable": True,
            "changes_primary_ranking": False,
            "filters": {
                "lncrna_id": lnc,
                "cancer_id": cancer,
                "clinical_endpoint": endpoint,
                "availability": availability,
            },
            "limit": limit,
            "offset": offset,
            "returned_statistics_rows": len(statistics),
            "returned_curve_rows": len(curves),
            "statistics": _records(statistics),
            "curves": _records(curves),
            "provenance": {
                "analysis_version": ANALYSIS_VERSION,
                "computation_run_id": self.computation_run_id,
                "binding_path": str(self.binding_path),
                "binding_sha256": self.binding_sha256,
                "historical_derived_outputs_used": False,
                "historical_predictions_used": False,
                "historical_checkpoints_used": False,
                "historical_rankings_used": False,
                "historical_web_tables_used": False,
                "release_ready": False,
                "production_deployed": False,
            },
        }

    def query_pair(
        self,
        *,
        cancer_id: Any,
        lncrna_id: Any,
        clinical_endpoint: Any,
        include_curves: bool = True,
    ) -> dict[str, Any]:
        return self.query_lncrna_survival(
            cancer_id=cancer_id,
            lncrna_id=lncrna_id,
            clinical_endpoint=clinical_endpoint,
            include_curves=include_curves,
            limit=1,
        )


__all__ = [
    "ClinicalKMQueryAssetError",
    "ClinicalKMQueryError",
    "ClinicalKMQueryInputError",
    "ClinicalKMReleaseQuery",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
]
