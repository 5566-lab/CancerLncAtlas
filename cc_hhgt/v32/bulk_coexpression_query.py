"""Hash-pinned read-only queries for fresh V3.2 bulk coexpression."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .bulk_coexpression_release import (
    ANALYSIS_VERSION,
    BINDING_FORMAT,
    FORMAL_CANDIDATE_PAIRS,
    FORMAL_CANDIDATE_SHA256,
    FORMAL_CANCERS,
    FORMAL_COVARIATE_SHA256,
    FORMAL_GENE_EXPRESSION_TREE_SHA256,
    FORMAL_LNCRNA_EXPRESSION_TREE_SHA256,
    artifact_sha256,
)


MAX_QUERY_LIMIT = 500
MAX_QUERY_OFFSET = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BulkCoexpressionQueryError(RuntimeError):
    """Base bulk-coexpression query error."""


class BulkCoexpressionQueryAssetError(BulkCoexpressionQueryError):
    """Raised when a binding or bound artifact is invalid or stale."""


class BulkCoexpressionQueryInputError(BulkCoexpressionQueryError):
    """Raised when query filters are invalid."""


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise BulkCoexpressionQueryAssetError(f"{label} is missing or unsafe: {source}")
    return source


def _safe_dir(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_dir() or source.is_symlink():
        raise BulkCoexpressionQueryAssetError(f"{label} is missing or unsafe: {source}")
    return source


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BulkCoexpressionQueryAssetError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise BulkCoexpressionQueryAssetError(f"{label} must be an object")
    return value


def _sql_path(path: Path) -> str:
    quote = chr(39)
    return quote + path.resolve().as_posix().replace(quote, quote * 2) + quote


def _relation(root: Path) -> str:
    files = sorted(item for item in root.rglob("*.parquet") if item.is_file())
    if not files:
        raise BulkCoexpressionQueryAssetError(f"No parquet files in {root}")
    return (
        "read_parquet(["
        + ",".join(_sql_path(item) for item in files)
        + "], union_by_name=true)"
    )


def _canonical_lnc(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"^(?:LNC|LNCRNA):", "", text)
    text = re.sub(r"\.\d+$", "", text)
    if not text or len(text) > 128 or any(ord(character) < 32 for character in text):
        raise BulkCoexpressionQueryInputError("lncrna_id is invalid")
    return "LNC:" + text


def _canonical_gene(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"^(?:GENE):", "", text)
    text = re.sub(r"\.\d+$", "", text)
    if not text or len(text) > 128 or any(ord(character) < 32 for character in text):
        raise BulkCoexpressionQueryInputError("gene_id is invalid")
    return "GENE:" + text


def _cancer(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or len(text) > 32 or not re.fullmatch(r"[A-Z0-9_-]+", text):
        raise BulkCoexpressionQueryInputError("cancer_id is invalid")
    return text


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise BulkCoexpressionQueryInputError(
            f"limit must be 1..{MAX_QUERY_LIMIT}"
        )
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise BulkCoexpressionQueryInputError(
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


class BulkCoexpressionReleaseQuery:
    """Validated immutable view of a current V3.2 coexpression release."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
        require_formal_authority: bool = True,
    ) -> None:
        source = _safe_file(binding_path, "Bulk coexpression binding")
        expected = str(expected_binding_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise BulkCoexpressionQueryAssetError(
                "Bulk coexpression query requires an expected binding SHA256"
            )
        observed = artifact_sha256(source)
        if observed != expected:
            raise BulkCoexpressionQueryAssetError(
                f"Bulk coexpression binding SHA mismatch: {observed} != {expected}"
            )
        binding = _load_json(source, "Bulk coexpression binding")
        required = {
            "format": BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "SUCCESS_FRESH_V32_BULK_COEXPRESSION_HASH_BOUND",
            "result_role": "TUMOUR_LNCRNA_GENE_COEXPRESSION_AND_PROFILE_CLUSTERS",
            "fresh_statistical_calculation": True,
            "training_not_applicable": True,
            "all_output_rows_generated_current_run": True,
            "historical_derived_outputs_used": False,
            "historical_coexpression_edges_used": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "family_to_exact_broadcast": False,
            "changes_primary_exact_pathway_ranking": False,
            "physical_binding_claimed": False,
            "causal_effect_claimed": False,
            "release_ready": False,
            "production_deployed": False,
        }
        for key, value in required.items():
            if binding.get(key) != value:
                raise BulkCoexpressionQueryAssetError(
                    f"Bulk coexpression binding has invalid {key}: {binding.get(key)!r}"
                )
        if require_formal_authority and binding.get("formal_authority") is not True:
            raise BulkCoexpressionQueryAssetError(
                "Bulk coexpression binding is not formal authority"
            )
        run_id = str(binding.get("computation_run_id", ""))
        if not run_id.startswith("V32-BULK-COEXPRESSION-"):
            raise BulkCoexpressionQueryAssetError("Invalid coexpression run ID")
        counts = binding.get("counts")
        if not isinstance(counts, dict):
            raise BulkCoexpressionQueryAssetError("Binding lacks counts")
        for key in (
            "cancers", "candidate_pairs", "edge_rows", "cluster_membership_rows",
            "cluster_summary_rows",
        ):
            if not isinstance(counts.get(key), int) or int(counts[key]) <= 0:
                raise BulkCoexpressionQueryAssetError(f"Invalid count: {key}")
        if require_formal_authority and (
            counts["cancers"] != FORMAL_CANCERS
            or counts["candidate_pairs"] != FORMAL_CANDIDATE_PAIRS
        ):
            raise BulkCoexpressionQueryAssetError("Formal coexpression counts are invalid")

        authorities = binding.get("authorities")
        if not isinstance(authorities, dict):
            raise BulkCoexpressionQueryAssetError("Binding lacks authorities")
        if require_formal_authority:
            expected_authorities = {
                "lncrna_expression_tree": FORMAL_LNCRNA_EXPRESSION_TREE_SHA256,
                "gene_expression_tree": FORMAL_GENE_EXPRESSION_TREE_SHA256,
                "covariates": FORMAL_COVARIATE_SHA256,
                "exact_candidates": FORMAL_CANDIDATE_SHA256,
            }
            for role, digest in expected_authorities.items():
                declaration = authorities.get(role)
                if not isinstance(declaration, dict) or declaration.get("sha256") != digest:
                    raise BulkCoexpressionQueryAssetError(
                        f"Coexpression authority mismatch: {role}"
                    )

        success = _load_json(source.parent / "SUCCESS.json", "Coexpression SUCCESS")
        if (
            success.get("status") != binding["status"]
            or success.get("binding") != source.name
            or success.get("binding_sha256") != observed
            or success.get("computation_run_id") != run_id
            or success.get("release_ready") is not False
            or success.get("production_deployed") is not False
        ):
            raise BulkCoexpressionQueryAssetError("Coexpression SUCCESS is stale")
        artifacts = binding.get("artifacts")
        if not isinstance(artifacts, dict):
            raise BulkCoexpressionQueryAssetError("Binding lacks artifacts")
        paths: dict[str, Path] = {}
        directory_roles = {
            "tumor_lncrna_gene_coexpression",
            "coexpression_availability",
            "coexpression_cluster_membership",
            "coexpression_cluster_summary",
        }
        required_roles = directory_roles | {
            "coexpression_cancer_summary", "source_inputs", "module_lineage"
        }
        for role in required_roles:
            declaration = artifacts.get(role)
            if (
                not isinstance(declaration, dict)
                or not _SHA256.fullmatch(str(declaration.get("sha256", "")))
            ):
                raise BulkCoexpressionQueryAssetError(
                    f"Invalid coexpression artifact declaration: {role}"
                )
            path = (
                _safe_dir(declaration.get("path", ""), role)
                if role in directory_roles
                else _safe_file(declaration.get("path", ""), role)
            )
            if artifact_sha256(path) != declaration["sha256"]:
                raise BulkCoexpressionQueryAssetError(
                    f"Coexpression artifact SHA drift: {role}"
                )
            paths[role] = path
        lineage = _load_json(paths["module_lineage"], "Coexpression lineage")
        if (
            lineage.get("status") != "SUCCESS_FRESH_V32_BULK_COEXPRESSION"
            or lineage.get("computation_run_id") != run_id
            or lineage.get("fresh_statistical_calculation") is not True
            or lineage.get("historical_derived_outputs_used") is not False
            or lineage.get("physical_binding_claimed") is not False
        ):
            raise BulkCoexpressionQueryAssetError("Coexpression lineage is invalid")
        self.binding = binding
        self.binding_path = source
        self.binding_sha256 = observed
        self.computation_run_id = run_id
        self.paths = paths
        self.require_formal_authority = require_formal_authority
        self._validate_tables()

    def _connect(self):
        return duckdb.connect(":memory:")

    def _validate_tables(self) -> None:
        edges = _relation(self.paths["tumor_lncrna_gene_coexpression"])
        availability = _relation(self.paths["coexpression_availability"])
        membership = _relation(self.paths["coexpression_cluster_membership"])
        clusters = _relation(self.paths["coexpression_cluster_summary"])
        cancer_summary = (
            f"read_parquet({_sql_path(self.paths['coexpression_cancer_summary'])})"
        )
        con = self._connect()
        try:
            edge_columns = {
                row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {edges}").fetchall()
            }
            availability_columns = {
                row[0]
                for row in con.execute(
                    f"DESCRIBE SELECT * FROM {availability}"
                ).fetchall()
            }
            member_columns = {
                row[0]
                for row in con.execute(
                    f"DESCRIBE SELECT * FROM {membership}"
                ).fetchall()
            }
            cluster_columns = {
                row[0]
                for row in con.execute(f"DESCRIBE SELECT * FROM {clusters}").fetchall()
            }
            required_edges = {
                "edge_id", "cancer_id", "lncrna_id", "gene_id", "rho", "p_value",
                "fdr", "direction", "n_patients", "residual_design_rank",
                "correlation_df", "method", "analysis_version", "computation_run_id",
                "physical_binding_claimed", "causal_effect_claimed",
            }
            required_availability = {
                "cancer_id", "lncrna_id", "availability", "failure_reason",
                "result_status", "n_patients", "n_edges", "cluster_available",
                "cluster_id", "analysis_version", "computation_run_id",
                "historical_derived_outputs_used", "physical_binding_claimed",
                "causal_effect_claimed",
            }
            required_members = {
                "cancer_id", "cluster_id", "lncrna_id", "member_rank",
                "member_strength", "n_edges", "representative", "analysis_version",
                "computation_run_id",
            }
            required_clusters = {
                "cancer_id", "cluster_id", "n_lncrnas", "n_unique_genes", "n_edges",
                "representative_lncrna_id", "top_gene_ids_json",
                "clustering_method", "analysis_version", "computation_run_id",
            }
            for label, observed, required in (
                ("edges", edge_columns, required_edges),
                ("availability", availability_columns, required_availability),
                ("membership", member_columns, required_members),
                ("clusters", cluster_columns, required_clusters),
            ):
                if missing := sorted(required - observed):
                    raise BulkCoexpressionQueryAssetError(
                        f"Coexpression {label} lacks columns: {missing}"
                    )
            edge_audit = con.execute(
                f"""
                SELECT count(*),
                       count(DISTINCT edge_id),
                       count(DISTINCT (cancer_id, lncrna_id)),
                       count_if(rho IS NULL OR NOT isfinite(rho) OR abs(rho) < 0.2
                                OR p_value IS NULL OR p_value NOT BETWEEN 0 AND 1
                                OR fdr IS NULL OR fdr NOT BETWEEN 0 AND 0.05
                                OR direction NOT IN ('positive', 'negative')
                                OR (direction = 'positive' AND rho < 0)
                                OR (direction = 'negative' AND rho >= 0)
                                OR physical_binding_claimed
                                OR causal_effect_claimed
                                OR analysis_version <> ?
                                OR computation_run_id <> ?)
                FROM {edges}
                """,
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
            availability_audit = con.execute(
                f"""
                SELECT count(*),
                       count(DISTINCT (cancer_id, lncrna_id)),
                       count(DISTINCT cancer_id),
                       sum(n_edges),
                       count_if(
                           analysis_version <> ? OR computation_run_id <> ?
                           OR historical_derived_outputs_used
                           OR physical_binding_claimed OR causal_effect_claimed
                           OR (availability AND failure_reason IS NOT NULL)
                           OR (NOT availability AND failure_reason IS NULL)
                           OR (cluster_available AND cluster_id IS NULL)
                           OR (NOT cluster_available AND cluster_id IS NOT NULL)
                           OR n_edges < 0
                           OR (cluster_available AND n_edges <= 0)
                       ),
                       count_if(availability)
                FROM {availability}
                """,
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
            membership_audit = con.execute(
                f"""
                SELECT count(*),
                       count(DISTINCT (cancer_id, lncrna_id)),
                       count(DISTINCT cluster_id),
                       sum(n_edges),
                       count_if(member_rank <= 0 OR member_strength <= 0 OR n_edges <= 0
                                OR analysis_version <> ?
                                OR computation_run_id <> ?)
                FROM {membership}
                """,
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
            cluster_audit = con.execute(
                f"""
                SELECT count(*), count(DISTINCT cluster_id),
                       sum(n_lncrnas), sum(n_edges),
                       count_if(n_lncrnas <= 0 OR n_unique_genes <= 0 OR n_edges <= 0
                                OR analysis_version <> ?
                                OR computation_run_id <> ?)
                FROM {clusters}
                """,
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
            cross_audit = con.execute(
                f"""
                SELECT
                  (SELECT count(*) FROM (
                     SELECT DISTINCT cancer_id, lncrna_id FROM {edges}
                  )),
                  (SELECT count(*) FROM {membership} m
                     LEFT JOIN {clusters} c USING (cancer_id, cluster_id)
                     WHERE c.cluster_id IS NULL),
                  (SELECT count(*) FROM {membership} m
                     LEFT JOIN {availability} a USING (cancer_id, lncrna_id)
                     WHERE a.lncrna_id IS NULL OR NOT a.cluster_available
                           OR a.cluster_id <> m.cluster_id
                           OR a.n_edges <> m.n_edges),
                  (SELECT count(*) FROM {availability} a
                     LEFT JOIN (
                       SELECT cancer_id, lncrna_id, count(*) AS edge_n
                       FROM {edges} GROUP BY cancer_id, lncrna_id
                     ) e USING (cancer_id, lncrna_id)
                     WHERE a.n_edges <> coalesce(e.edge_n, 0))
                """
            ).fetchone()
            cancer_audit = con.execute(
                f"""
                SELECT count(*), count(DISTINCT cancer_id),
                       count_if(analysis_version <> ? OR computation_run_id <> ?)
                FROM {cancer_summary}
                """,
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
        finally:
            con.close()
        counts = self.binding["counts"]
        if (
            int(edge_audit[0]) != int(counts["edge_rows"])
            or int(edge_audit[1]) != int(edge_audit[0])
            or int(edge_audit[2]) != int(counts["cluster_membership_rows"])
            or int(edge_audit[3] or 0)
            or int(availability_audit[0]) != int(counts["candidate_pairs"])
            or int(availability_audit[1]) != int(availability_audit[0])
            or int(availability_audit[2]) != int(counts["cancers"])
            or int(availability_audit[3] or 0) != int(edge_audit[0])
            or int(availability_audit[4] or 0)
            or int(availability_audit[5] or 0) != int(counts["available_pairs"])
            or int(membership_audit[0]) != int(counts["cluster_membership_rows"])
            or int(membership_audit[1]) != int(membership_audit[0])
            or int(membership_audit[2]) != int(counts["cluster_summary_rows"])
            or int(membership_audit[3] or 0) != int(edge_audit[0])
            or int(membership_audit[4] or 0)
            or int(cluster_audit[0]) != int(counts["cluster_summary_rows"])
            or int(cluster_audit[1]) != int(cluster_audit[0])
            or int(cluster_audit[2] or 0) != int(membership_audit[0])
            or int(cluster_audit[3] or 0) != int(edge_audit[0])
            or int(cluster_audit[4] or 0)
            or int(cross_audit[0]) != int(membership_audit[0])
            or any(int(value or 0) for value in cross_audit[1:])
            or int(cancer_audit[0]) != int(counts["cancers"])
            or int(cancer_audit[1]) != int(cancer_audit[0])
            or int(cancer_audit[2] or 0)
        ):
            raise BulkCoexpressionQueryAssetError(
                "Bulk coexpression table semantic validation failed"
            )

    def _provenance(self) -> dict[str, Any]:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "computation_run_id": self.computation_run_id,
            "binding_path": str(self.binding_path),
            "binding_sha256": self.binding_sha256,
            "fresh_statistical_calculation": True,
            "historical_derived_outputs_used": False,
            "changes_primary_exact_pathway_ranking": False,
            "physical_binding_claimed": False,
            "causal_effect_claimed": False,
            "release_ready": False,
            "production_deployed": False,
        }

    def query_lncrna(
        self,
        *,
        lncrna_id: Any,
        cancer_id: Any | None = None,
        gene_id: Any | None = None,
        direction: str | None = None,
        min_abs_rho: float | None = None,
        max_fdr: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        lnc = _canonical_lnc(lncrna_id)
        cancer = _cancer(cancer_id)
        gene = _canonical_gene(gene_id) if gene_id is not None else None
        normalized_direction = None
        if direction is not None:
            normalized_direction = str(direction).strip().lower()
            if normalized_direction not in {"positive", "negative"}:
                raise BulkCoexpressionQueryInputError(
                    "direction must be positive or negative"
                )
        if min_abs_rho is not None and (
            isinstance(min_abs_rho, bool)
            or not 0 <= float(min_abs_rho) <= 1
        ):
            raise BulkCoexpressionQueryInputError("min_abs_rho must be 0..1")
        if max_fdr is not None and (
            isinstance(max_fdr, bool) or not 0 <= float(max_fdr) <= 1
        ):
            raise BulkCoexpressionQueryInputError("max_fdr must be 0..1")

        availability_relation = _relation(self.paths["coexpression_availability"])
        edge_relation = _relation(self.paths["tumor_lncrna_gene_coexpression"])
        member_relation = _relation(self.paths["coexpression_cluster_membership"])
        availability_clauses = ["lncrna_id = ?"]
        availability_parameters: list[Any] = [lnc]
        edge_clauses = ["e.lncrna_id = ?"]
        edge_parameters: list[Any] = [lnc]
        if cancer is not None:
            availability_clauses.append("cancer_id = ?")
            availability_parameters.append(cancer)
            edge_clauses.append("e.cancer_id = ?")
            edge_parameters.append(cancer)
        if gene is not None:
            edge_clauses.append("e.gene_id = ?")
            edge_parameters.append(gene)
        if normalized_direction is not None:
            edge_clauses.append("e.direction = ?")
            edge_parameters.append(normalized_direction)
        if min_abs_rho is not None:
            edge_clauses.append("abs(e.rho) >= ?")
            edge_parameters.append(float(min_abs_rho))
        if max_fdr is not None:
            edge_clauses.append("e.fdr <= ?")
            edge_parameters.append(float(max_fdr))
        edge_parameters.extend([limit, offset])
        con = self._connect()
        try:
            availability_frame = con.execute(
                f"""
                SELECT * FROM {availability_relation}
                WHERE {' AND '.join(availability_clauses)}
                ORDER BY cancer_id
                """,
                availability_parameters,
            ).fetchdf()
            edge_frame = con.execute(
                f"""
                SELECT e.*, m.cluster_id, m.member_rank AS cluster_member_rank,
                       m.representative AS cluster_representative
                FROM {edge_relation} e
                LEFT JOIN {member_relation} m
                  ON e.cancer_id = m.cancer_id AND e.lncrna_id = m.lncrna_id
                WHERE {' AND '.join(edge_clauses)}
                ORDER BY e.fdr, abs(e.rho) DESC, e.cancer_id, e.gene_id
                LIMIT ? OFFSET ?
                """,
                edge_parameters,
            ).fetchdf()
        finally:
            con.close()
        return {
            "module": "bulk_coexpression",
            "query_kind": "lncrna_gene_coexpression",
            "result_role": self.binding["result_role"],
            "association_not_physical_binding": True,
            "association_not_causality": True,
            "filters": {
                "lncrna_id": lnc,
                "cancer_id": cancer,
                "gene_id": gene,
                "direction": normalized_direction,
                "min_abs_rho": min_abs_rho,
                "max_fdr": max_fdr,
            },
            "limit": limit,
            "offset": offset,
            "returned_availability_rows": len(availability_frame),
            "returned_edge_rows": len(edge_frame),
            "availability_rows": _records(availability_frame),
            "edge_rows": _records(edge_frame),
            "provenance": self._provenance(),
        }

    def query_clusters(
        self,
        *,
        cancer_id: Any | None = None,
        cluster_id: Any | None = None,
        lncrna_id: Any | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        cancer = _cancer(cancer_id)
        cluster = None
        if cluster_id is not None:
            cluster = str(cluster_id).strip().upper()
            if (
                not cluster
                or len(cluster) > 128
                or not re.fullmatch(r"COEXC:[A-Z0-9_-]+:\d{3}", cluster)
            ):
                raise BulkCoexpressionQueryInputError("cluster_id is invalid")
        lnc = _canonical_lnc(lncrna_id) if lncrna_id is not None else None
        membership_relation = _relation(
            self.paths["coexpression_cluster_membership"]
        )
        cluster_relation = _relation(self.paths["coexpression_cluster_summary"])
        clauses: list[str] = []
        parameters: list[Any] = []
        if cancer is not None:
            clauses.append("m.cancer_id = ?")
            parameters.append(cancer)
        if cluster is not None:
            clauses.append("m.cluster_id = ?")
            parameters.append(cluster)
        if lnc is not None:
            clauses.append("m.lncrna_id = ?")
            parameters.append(lnc)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.extend([limit, offset])
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT m.*, c.n_lncrnas AS cluster_n_lncrnas,
                       c.n_unique_genes AS cluster_n_unique_genes,
                       c.n_edges AS cluster_n_edges,
                       c.representative_lncrna_id,
                       c.top_gene_ids_json,
                       c.clustering_method
                FROM {membership_relation} m
                JOIN {cluster_relation} c USING (cancer_id, cluster_id)
                {where}
                ORDER BY m.cancer_id, m.cluster_id, m.member_rank
                LIMIT ? OFFSET ?
                """,
                parameters,
            ).fetchdf()
        finally:
            con.close()
        return {
            "module": "bulk_coexpression",
            "query_kind": "coexpression_clusters",
            "result_role": self.binding["result_role"],
            "association_not_physical_binding": True,
            "filters": {
                "cancer_id": cancer,
                "cluster_id": cluster,
                "lncrna_id": lnc,
            },
            "limit": limit,
            "offset": offset,
            "returned_rows": len(frame),
            "rows": _records(frame),
            "provenance": self._provenance(),
        }


__all__ = [
    "BulkCoexpressionQueryAssetError",
    "BulkCoexpressionQueryError",
    "BulkCoexpressionQueryInputError",
    "BulkCoexpressionReleaseQuery",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
]
