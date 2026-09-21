"""Hash-pinned read-only queries for V3.2 Gene Sets and ranked subtypes."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import duckdb
import numpy as np
import pandas as pd

from .gene_set_subtype_release import (
    ANALYSIS_VERSION,
    RELEASE_FORMAT,
    RELEASE_STATUS,
    TYPED_UNAVAILABLE,
    artifact_sha256,
)


AUDIT_FORMAT = "CC_HHGT_V3_2_GENE_SET_RANKED_SUBTYPE_INDEPENDENT_AUDIT_V1"
AUDIT_BINDING_FORMAT = (
    "CC_HHGT_V3_2_GENE_SET_RANKED_SUBTYPE_INDEPENDENT_AUDIT_BINDING_V1"
)
MAX_QUERY_LIMIT = 500
MAX_MEMBER_LIMIT = 500
MAX_QUERY_OFFSET = 100_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANCER = re.compile(r"^[A-Z0-9_-]{2,16}$")


class GeneSetSubtypeQueryError(RuntimeError):
    """Base query error."""


class GeneSetSubtypeQueryAssetError(GeneSetSubtypeQueryError):
    """Raised when a bound artifact is missing, stale, or semantically invalid."""


class GeneSetSubtypeQueryInputError(GeneSetSubtypeQueryError):
    """Raised when a query filter is invalid."""


class GeneSetSubtypeQueryNotFoundError(GeneSetSubtypeQueryError):
    """Raised when a requested Gene Set does not exist."""


def _safe_file(path: str | Path, label: str) -> Path:
    requested = Path(path)
    if requested.is_symlink():
        raise GeneSetSubtypeQueryAssetError(f"{label} may not be a symlink")
    resolved = requested.resolve()
    if not resolved.is_file():
        raise GeneSetSubtypeQueryAssetError(f"{label} is missing: {resolved}")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeneSetSubtypeQueryAssetError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise GeneSetSubtypeQueryAssetError(f"{label} must be a JSON object")
    return value


def _resolve_record(record: Any, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise GeneSetSubtypeQueryAssetError(f"Missing artifact record: {label}")
    expected = str(record.get("sha256", "")).lower()
    if not _SHA256.fullmatch(expected):
        raise GeneSetSubtypeQueryAssetError(f"Invalid artifact SHA256: {label}")
    path = _safe_file(str(record.get("path", "")), label)
    if artifact_sha256(path) != expected:
        raise GeneSetSubtypeQueryAssetError(f"Artifact SHA256 drift: {label}")
    return path


def _sql_literal(path: str | Path) -> str:
    return "'" + str(Path(path).resolve()).replace("'", "''") + "'"


def _parquet(path: str | Path) -> str:
    return f"read_parquet({_sql_literal(path)}, hive_partitioning=false)"


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    def clean(value: Any) -> Any:
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

    return [
        {str(key): clean(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _bounds(limit: int, offset: int, maximum: int = MAX_QUERY_LIMIT) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= maximum:
        raise GeneSetSubtypeQueryInputError(f"limit must be 1..{maximum}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise GeneSetSubtypeQueryInputError(
            f"offset must be 0..{MAX_QUERY_OFFSET}"
        )
    return int(limit), int(offset)


def _clean_text(value: Any, name: str, *, maximum: int = 512) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum or any(ord(char) < 32 for char in text):
        raise GeneSetSubtypeQueryInputError(f"{name} is invalid")
    return text


def _cancer(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not _CANCER.fullmatch(text):
        raise GeneSetSubtypeQueryInputError("cancer_id is invalid")
    return text


def _direction(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text not in {"positive", "negative"}:
        raise GeneSetSubtypeQueryInputError(
            "direction must be positive or negative"
        )
    return text


class GeneSetSubtypeReleaseQuery:
    """Validated immutable view of the deterministic V3.2 derived head."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
        audit_binding_path: str | Path,
        expected_audit_binding_sha256: str | None,
        require_formal_authority: bool = True,
    ) -> None:
        source = _safe_file(binding_path, "Gene Set/ranked-subtype binding")
        expected = str(expected_binding_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise GeneSetSubtypeQueryAssetError(
                "Gene Set/ranked-subtype query requires an expected binding SHA256"
            )
        observed = artifact_sha256(source)
        if observed != expected:
            raise GeneSetSubtypeQueryAssetError(
                f"Gene Set/ranked-subtype binding SHA mismatch: {observed} != {expected}"
            )
        binding = _read_json(source, "Gene Set/ranked-subtype binding")
        required = {
            "format": RELEASE_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": RELEASE_STATUS,
            "module_ids": ["gene_set", "ranked_subtype"],
            "result_role": "DETERMINISTIC_V32_EXACT_PATHWAY_DERIVED_FUNCTIONAL_HEAD",
            "head_kind": "DETERMINISTIC_DERIVED_FUNCTIONAL_HEAD_NOT_PREDICTIVE_MODEL",
            "initialization": "v32_core_frozen_new_head",
            "initialization_semantics": "FUNCTIONAL_HEAD_BOUND_TO_FROZEN_CURRENT_V32_EXACT_PATHWAY_OUTPUTS;NO_PARAMETER_INITIALIZATION",
            "model_training_performed": False,
            "learned_parameters": False,
            "new_checkpoint_created": False,
            "deterministic_derivation_from_current_v32_exact_gene_sets": True,
            "source_core_model_version": "V3.2",
            "source_core_frozen": True,
            "primary_score_preserved": True,
            "subtype_feedback_to_model": False,
            "target_level": "exact_pathway",
            "family_to_exact_broadcast": False,
            "old_checkpoints_used": False,
            "old_predictions_used": False,
            "old_rankings_used": False,
            "missingness_encoding": "null_with_reason",
            "unavailable_fill_value": None,
            "release_ready": False,
            "production_deployed": False,
        }
        for key, expected_value in required.items():
            if binding.get(key) != expected_value:
                raise GeneSetSubtypeQueryAssetError(
                    f"Gene Set/ranked-subtype binding has invalid {key}: {binding.get(key)!r}"
                )
        expected_unavailable = [
            {"cancer_id": cancer, "reason": reason}
            for cancer, reason in sorted(TYPED_UNAVAILABLE.items())
        ]
        if binding.get("typed_unavailable_cancers") != expected_unavailable:
            raise GeneSetSubtypeQueryAssetError(
                "Gene Set/ranked-subtype typed unavailable cancers drifted"
            )
        sources = binding.get("sources")
        if not isinstance(sources, Mapping):
            raise GeneSetSubtypeQueryAssetError("Binding sources are missing")
        source_roles = {
            "audited_gene_set_release_manifest",
            "audited_gene_set_independent_audit_binding",
            "subtype_materialization_success",
            "subtype_materialization_coverage_audit",
            "deterministic_subtype_implementation",
        }
        if set(sources) != source_roles:
            raise GeneSetSubtypeQueryAssetError("Binding source roles are incomplete")
        source_paths = {
            role: _resolve_record(sources[role], f"source {role}")
            for role in sorted(source_roles)
        }
        upstream = _read_json(
            source_paths["audited_gene_set_release_manifest"],
            "upstream Gene Set manifest",
        )
        if any(
            upstream.get(key) != value
            for key, value in {
                "format": "CC_HHGT_V3_2_GENE_SET_PARITY_RELEASE_V1",
                "analysis_version": ANALYSIS_VERSION,
                "target_level": "exact_pathway",
                "family_to_exact_broadcast": False,
                "old_checkpoints_used": False,
                "old_predictions_used": False,
                "old_rankings_used": False,
            }.items()
        ):
            raise GeneSetSubtypeQueryAssetError(
                "Upstream Gene Set authority violates exact V3.2 provenance"
            )
        artifacts = binding.get("artifacts")
        expected_artifact_roles = {
            "exact_pathway_associations",
            "gene_set_catalog",
            "gene_set_members",
            "gene_set_gmt",
            "gene_set_enrichment",
            "gene_set_coverage",
            "gene_set_report",
            "ranked_subtypes",
            "cancer_program_subtypes",
            "subtype_stability",
        }
        if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifact_roles:
            raise GeneSetSubtypeQueryAssetError("Binding artifact roles are incomplete")
        paths = {
            role: _resolve_record(artifacts[role], role)
            for role in sorted(expected_artifact_roles - {"gene_set_members"})
        }
        member_record = artifacts.get("gene_set_members")
        part_records = (
            member_record.get("part_artifacts")
            if isinstance(member_record, Mapping)
            else None
        )
        if not isinstance(part_records, list) or not part_records:
            raise GeneSetSubtypeQueryAssetError("Gene Set member parts are missing")
        member_parts = [
            _resolve_record(record, f"member part {index}")
            for index, record in enumerate(part_records)
        ]
        tree = hashlib.sha256(
            "\n".join(
                f"{path.name}\t{artifact_sha256(path)}" for path in member_parts
            ).encode()
        ).hexdigest()
        if tree != member_record.get("sha256_tree"):
            raise GeneSetSubtypeQueryAssetError("Gene Set member tree SHA drift")

        audit_path = _safe_file(
            audit_binding_path, "Gene Set/ranked-subtype independent-audit binding"
        )
        expected_audit = str(expected_audit_binding_sha256 or "").lower()
        if not _SHA256.fullmatch(expected_audit):
            raise GeneSetSubtypeQueryAssetError(
                "Gene Set/ranked-subtype query requires an expected audit SHA256"
            )
        observed_audit = artifact_sha256(audit_path)
        if observed_audit != expected_audit:
            raise GeneSetSubtypeQueryAssetError(
                "Gene Set/ranked-subtype independent-audit SHA mismatch"
            )
        audit = _read_json(
            audit_path, "Gene Set/ranked-subtype independent-audit binding"
        )
        audit_required = {
            "format": AUDIT_BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS",
            "fail_count": 0,
            "accepted_for_api_integration": True,
            "independent_of_release_materializer": True,
            "release_materializer_imported": False,
            "subtype_implementation_imported": False,
            "release_ready": False,
            "production_deployed": False,
        }
        for key, expected_value in audit_required.items():
            if audit.get(key) != expected_value:
                raise GeneSetSubtypeQueryAssetError(
                    f"Independent audit has invalid {key}: {audit.get(key)!r}"
                )
        audit_release = audit.get("release_binding")
        if (
            not isinstance(audit_release, Mapping)
            or Path(str(audit_release.get("path", ""))).resolve() != source
            or audit_release.get("sha256") != observed
        ):
            raise GeneSetSubtypeQueryAssetError(
                "Independent audit is tied to a different release binding"
            )
        report_path = _resolve_record(audit.get("report"), "independent audit report")
        report = _read_json(report_path, "independent audit report")
        checks = report.get("checks")
        if (
            report.get("format") != AUDIT_FORMAT
            or report.get("status") != "PASS"
            or report.get("accepted_for_api_integration") is not True
            or report.get("fail_count") != 0
            or report.get("pass_count") != audit.get("pass_count")
            or not isinstance(report.get("pass_count"), int)
            or report["pass_count"] < 80
            or not isinstance(checks, list)
            or len(checks) != report["pass_count"]
            or any(not isinstance(item, Mapping) or item.get("status") != "PASS" for item in checks)
            or report.get("stability_values_independently_recomputed") is not True
            or report.get("release_ready") is not False
            or report.get("production_deployed") is not False
        ):
            raise GeneSetSubtypeQueryAssetError(
                "Independent audit did not pass the full API gate"
            )
        success = _read_json(source.parent / "SUCCESS.json", "release SUCCESS")
        if (
            success.get("binding") != source.name
            or success.get("binding_sha256") != observed
            or success.get("trained_model") is not False
            or success.get("release_ready") is not False
            or success.get("production_deployed") is not False
        ):
            raise GeneSetSubtypeQueryAssetError("Release SUCCESS marker is stale")

        self.binding_path = source
        self.binding_sha256 = observed
        self.binding = binding
        self.audit_binding_path = audit_path
        self.audit_binding_sha256 = observed_audit
        self.audit = audit
        self.audit_report_path = report_path
        self.paths = paths
        self.member_parts = member_parts
        self.require_formal_authority = require_formal_authority
        self._coverage = self._load_coverage()
        self._validate_tables()

    def _connect(self):
        return duckdb.connect(":memory:")

    def _relation(self, role: str) -> str:
        return _parquet(self.paths[role])

    def _members(self) -> str:
        paths = ",".join(_sql_literal(path) for path in self.member_parts)
        return f"read_parquet([{paths}], hive_partitioning=false)"

    def _load_coverage(self) -> dict[str, dict[str, Any]]:
        frame = pd.read_csv(self.paths["gene_set_coverage"], sep="\t")
        required = {
            "cancer_id",
            "publishable_genesets",
            "exact_pathways",
            "member_rows",
            "coverage_status",
        }
        if missing := required - set(frame.columns):
            raise GeneSetSubtypeQueryAssetError(
                f"Gene Set coverage lacks columns: {sorted(missing)}"
            )
        records: dict[str, dict[str, Any]] = {}
        for row in _records(frame):
            cancer = str(row["cancer_id"])
            status = str(row["coverage_status"])
            available = status == "AVAILABLE"
            row["available"] = available
            row["unavailable_reason"] = None if available else status
            records[cancer] = row
        observed_unavailable = {
            cancer: str(record["unavailable_reason"])
            for cancer, record in records.items()
            if not record["available"]
        }
        if observed_unavailable != TYPED_UNAVAILABLE:
            raise GeneSetSubtypeQueryAssetError(
                "Coverage does not preserve CHOL/UCS typed unavailable states"
            )
        return records

    def _validate_tables(self) -> None:
        exact = self._relation("exact_pathway_associations")
        catalog = self._relation("gene_set_catalog")
        enrichment = self._relation("gene_set_enrichment")
        context = self._relation("ranked_subtypes")
        program = self._relation("cancer_program_subtypes")
        stability = self._relation("subtype_stability")
        con = self._connect()
        try:
            exact_columns = {
                str(row[0])
                for row in con.execute(f"DESCRIBE SELECT * FROM {exact}").fetchall()
            }
            stats = {
                "exact_associations": int(con.execute(f"SELECT count(*) FROM {exact}").fetchone()[0]),
                "exact_non_exact": int(con.execute(f"SELECT count(*) FROM {exact} WHERE pathway_target_level<>'exact_pathway'").fetchone()[0]),
                "exact_family_as_target": int(con.execute(f"SELECT count(*) FROM {exact} WHERE pathway_id=pathway_family_id").fetchone()[0]),
                "exact_invalid_probability": int(con.execute(f"SELECT count(*) FROM {exact} WHERE association_membership_probability IS NOT NULL AND (NOT isfinite(association_membership_probability) OR association_membership_probability NOT BETWEEN 0 AND 1)").fetchone()[0]),
                "gene_sets": int(con.execute(f"SELECT count(*) FROM {catalog}").fetchone()[0]),
                "enrichment": int(con.execute(f"SELECT count(*) FROM {enrichment}").fetchone()[0]),
                "contexts": int(con.execute(f"SELECT count(*) FROM {context}").fetchone()[0]),
                "programs": int(con.execute(f"SELECT count(*) FROM {program}").fetchone()[0]),
                "stability": int(con.execute(f"SELECT count(*) FROM {stability}").fetchone()[0]),
                "non_exact": int(con.execute(f"SELECT count(*) FROM {catalog} WHERE pathway_target_level<>'exact_pathway'").fetchone()[0]),
                "family_broadcast": int(con.execute(f"SELECT count(*) FROM {enrichment} WHERE family_to_exact_broadcast").fetchone()[0]),
            }
        finally:
            con.close()
        forbidden_private_labels = {
            "label",
            "labels",
            "target",
            "y",
            "y_true",
            "training_label",
            "heldout_label",
            "private_label",
        }
        if (
            stats["exact_non_exact"]
            or stats["exact_family_as_target"]
            or stats["exact_invalid_probability"]
            or stats["non_exact"]
            or stats["family_broadcast"]
            or exact_columns & forbidden_private_labels
        ):
            raise GeneSetSubtypeQueryAssetError(
                "Gene Set/ranked-subtype tables violate exact-pathway semantics"
            )
        if self.require_formal_authority:
            expected = {
                "exact_associations": 3_300_000,
                "exact_non_exact": 0,
                "exact_family_as_target": 0,
                "exact_invalid_probability": 0,
                "gene_sets": 54_380,
                "enrichment": 54_380,
                "contexts": 54_380,
                "programs": 31,
                "stability": 668_274,
                "non_exact": 0,
                "family_broadcast": 0,
            }
            if stats != expected or len(self._coverage) != 33:
                raise GeneSetSubtypeQueryAssetError(
                    f"Formal Gene Set/ranked-subtype counts drifted: {stats}"
                )

    def _provenance(self) -> dict[str, Any]:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "binding_sha256": self.binding_sha256,
            "independent_audit_binding_sha256": self.audit_binding_sha256,
            "independent_audit_pass_count": self.audit["pass_count"],
            "deterministic_v32_derived_head": True,
            "trained_model": False,
            "initialization": "v32_core_frozen_new_head",
            "initialization_semantics": self.binding["initialization_semantics"],
            "source_target_level": "exact_pathway",
            "family_to_exact_broadcast": False,
            "old_checkpoints_used": False,
            "old_predictions_used": False,
            "old_rankings_used": False,
            "primary_score_preserved": True,
            "subtype_feedback_to_model": False,
            "missingness_encoding": "null_with_reason",
            "unavailable_fill_value": None,
            "private_training_labels_exposed": False,
            "release_ready": False,
            "production_deployed": False,
        }

    def _result(
        self,
        query_kind: str,
        rows: list[dict[str, Any]],
        filters: Mapping[str, Any],
        *,
        limit: int | None = None,
        offset: int | None = None,
        availability: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "module": "gene_set_ranked_subtype",
            "query_kind": query_kind,
            "returned_rows": len(rows),
            "rows": rows,
            "filters": dict(filters),
            "provenance": self._provenance(),
        }
        if limit is not None:
            result["limit"] = limit
        if offset is not None:
            result["offset"] = offset
        if availability is not None:
            result["availability"] = dict(availability)
        return result

    def exact_pathway_capability(self) -> dict[str, Any]:
        declaration = self.binding["artifacts"]["exact_pathway_associations"]
        counts = self.binding["counts"]
        return {
            "status": "ENABLED_CURRENT_V32_EXACT_PATHWAY_HASH_PINNED",
            "module_id": "exact_pathway",
            "analysis_version": ANALYSIS_VERSION,
            "artifact": {
                "path": str(self.paths["exact_pathway_associations"]),
                "sha256": declaration["sha256"],
                "rows": declaration["rows"],
            },
            "counts": {
                "rows": counts["exact_association_rows"],
                "unique_keys": counts["exact_association_unique_keys"],
                "cancers": counts["exact_association_cancers"],
                "lncrnas": counts["exact_association_lncrnas"],
                "exact_pathways": counts["exact_association_pathways"],
                "null_probability_rows": counts[
                    "exact_association_null_probability_rows"
                ],
            },
            "filters": ["cancer_id", "lncrna_id", "pathway_id", "availability"],
            "target_level": "exact_pathway",
            "availability_encoding": "boolean_plus_nullable_probability_and_reason",
            "unavailable_fill_value": None,
            "family_to_exact_broadcast": False,
            "private_training_labels_exposed": False,
            "old_checkpoints_used": False,
            "old_predictions_used": False,
            "old_rankings_used": False,
            "release_ready": False,
            "production_deployed": False,
            "provenance": self._provenance(),
        }

    def query_exact_pathway_associations(
        self,
        *,
        cancer_id: str | None = None,
        lncrna_id: str | None = None,
        pathway_id: str | None = None,
        availability: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        cancer = _cancer(cancer_id)
        lncrna = _clean_text(lncrna_id, "lncrna_id") if lncrna_id else None
        pathway = _clean_text(pathway_id, "pathway_id") if pathway_id else None
        relation = self._relation("exact_pathway_associations")
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("cancer_id", cancer),
            ("lncrna_id", lncrna),
            ("pathway_id", pathway),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        if availability is not None:
            clauses.append(
                "association_membership_probability IS NOT NULL"
                if availability
                else "association_membership_probability IS NULL"
            )
        where = " AND ".join(clauses) if clauses else "TRUE"
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT cancer_id,lncrna_id,pathway_id,pathway_family_id,
                       shared_or_local_scope,
                       CAST(association_membership_probability AS DOUBLE)
                         association_membership_probability,
                       CAST(association_direction_probability AS DOUBLE)
                         association_direction_probability,
                       association_direction,n_folds_available,analysis_version,
                       training_run_id,changes_primary_ranking,
                       association_membership_probability IS NOT NULL availability,
                       CASE WHEN association_membership_probability IS NULL
                            THEN 'UNAVAILABLE_CURRENT_V32_EXACT_PROBABILITY_NULL'
                            ELSE NULL END unavailable_reason
                FROM {relation} WHERE {where}
                ORDER BY cancer_id,lncrna_id,pathway_id LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            "current_v32_exact_pathway_associations",
            _records(frame),
            {
                "cancer_id": cancer,
                "lncrna_id": lncrna,
                "pathway_id": pathway,
                "availability": availability,
            },
            limit=limit,
            offset=offset,
        )

    def query_exact_pathway_stats(
        self, *, cancer_id: str | None = None
    ) -> dict[str, Any]:
        cancer = _cancer(cancer_id)
        relation = self._relation("exact_pathway_associations")
        where = "WHERE cancer_id=?" if cancer is not None else ""
        con = self._connect()
        try:
            row = con.execute(
                f"""
                SELECT count(*) AS row_count,
                       count(*) FILTER
                         (WHERE association_membership_probability IS NOT NULL)
                         available_rows,
                       count(*) FILTER
                         (WHERE association_membership_probability IS NULL)
                         unavailable_rows,
                       count(DISTINCT cancer_id) cancers,
                       count(DISTINCT lncrna_id) lncrnas,
                       count(DISTINCT pathway_id) exact_pathways,
                       min(association_membership_probability) min_probability,
                       avg(association_membership_probability) mean_probability,
                       max(association_membership_probability) max_probability
                FROM {relation} {where}
                """,
                [cancer] if cancer is not None else [],
            ).fetchdf()
        finally:
            con.close()
        return {
            "module": "exact_pathway",
            "query_kind": "current_v32_exact_pathway_stats",
            "filters": {"cancer_id": cancer},
            "stats": _records(row)[0],
            "availability_encoding": "boolean_plus_nullable_probability_and_reason",
            "unavailable_fill_value": None,
            "family_to_exact_broadcast": False,
            "private_training_labels_exposed": False,
            "provenance": self._provenance(),
        }

    def query_exact_pathway_search(
        self, *, query: str, limit: int = 20
    ) -> dict[str, Any]:
        term = _clean_text(query, "query", maximum=100)
        if len(term) < 2:
            raise GeneSetSubtypeQueryInputError("query must contain at least 2 characters")
        limit, _ = _bounds(limit, 0, maximum=100)
        relation = self._relation("exact_pathway_associations")
        pattern = f"%{term}%"
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT entity_type,entity_id FROM (
                  SELECT DISTINCT 'lncrna' entity_type,lncrna_id entity_id
                  FROM {relation} WHERE lncrna_id ILIKE ?
                  UNION
                  SELECT DISTINCT 'exact_pathway' entity_type,pathway_id entity_id
                  FROM {relation} WHERE pathway_id ILIKE ?
                  UNION
                  SELECT DISTINCT 'pathway_family' entity_type,pathway_family_id entity_id
                  FROM {relation} WHERE pathway_family_id ILIKE ?
                  UNION
                  SELECT DISTINCT 'cancer' entity_type,cancer_id entity_id
                  FROM {relation} WHERE cancer_id ILIKE ?
                ) q ORDER BY entity_type,entity_id LIMIT ?
                """,
                [pattern, pattern, pattern, pattern, limit],
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            "current_v32_exact_pathway_search",
            _records(frame),
            {"query": term},
            limit=limit,
            offset=0,
        )

    def query_exact_pathway_cancers(self) -> dict[str, Any]:
        rows = []
        for cancer_id in sorted(self._coverage):
            row = {"cancer_id": cancer_id, **dict(self._coverage[cancer_id])}
            rows.append(row)
        return self._result(
            "current_v32_exact_pathway_cancers",
            rows,
            {},
            limit=len(rows),
            offset=0,
        )

    def query_exact_pathway_datasets(self) -> dict[str, Any]:
        sources = self.binding.get("sources", {})
        rows = [
            {
                "dataset_id": str(source_id),
                "sha256": str(record.get("sha256", "")),
                "bytes": record.get("bytes"),
                "role": "HASH_BOUND_V32_EXACT_PATHWAY_AUTHORITY",
            }
            for source_id, record in sorted(sources.items())
            if isinstance(record, Mapping)
        ]
        return self._result(
            "current_v32_exact_pathway_datasets",
            rows,
            {},
            limit=len(rows),
            offset=0,
        )

    def _availability(self, cancer_id: str | None) -> dict[str, Any] | None:
        if cancer_id is None:
            return None
        if cancer_id not in self._coverage:
            raise GeneSetSubtypeQueryInputError(
                f"cancer_id is outside the formal 33-cancer universe: {cancer_id}"
            )
        return dict(self._coverage[cancer_id])

    def query_gene_set_catalog(
        self,
        *,
        cancer_id: str | None = None,
        pathway_id: str | None = None,
        direction: str | None = None,
        min_member_count: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        cancer = _cancer(cancer_id)
        pathway = _clean_text(pathway_id, "pathway_id") if pathway_id else None
        direction = _direction(direction)
        if min_member_count is not None and (
            isinstance(min_member_count, bool) or int(min_member_count) < 1
        ):
            raise GeneSetSubtypeQueryInputError("min_member_count must be positive")
        availability = self._availability(cancer)
        relation = self._relation("gene_set_catalog")
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("cancer_id", cancer),
            ("pathway_id", pathway),
            ("direction", direction),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        if min_member_count is not None:
            clauses.append("member_count>=?")
            params.append(int(min_member_count))
        where = " AND ".join(clauses) if clauses else "TRUE"
        con = self._connect()
        try:
            frame = con.execute(
                f"SELECT * FROM {relation} WHERE {where} "
                "ORDER BY cancer_id,pathway_id,direction,geneset_id LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            "gene_set_catalog",
            _records(frame),
            {
                "cancer_id": cancer,
                "pathway_id": pathway,
                "direction": direction,
                "min_member_count": min_member_count,
            },
            limit=limit,
            offset=offset,
            availability=availability,
        )

    def query_gene_set_detail(
        self,
        *,
        gene_set_id: str,
        member_limit: int = 200,
        member_offset: int = 0,
    ) -> dict[str, Any]:
        gene_set = _clean_text(gene_set_id, "gene_set_id", maximum=128)
        member_limit, member_offset = _bounds(
            member_limit, member_offset, maximum=MAX_MEMBER_LIMIT
        )
        catalog = self._relation("gene_set_catalog")
        enrichment = self._relation("gene_set_enrichment")
        context = self._relation("ranked_subtypes")
        members = self._members()
        con = self._connect()
        try:
            detail = con.execute(
                f"""
                SELECT c.*,e.mean_membership_probability,e.mean_discovery_probability,
                       e.mean_confidence_probability,e.genomic_available_member_count,
                       e.single_cell_available_member_count,e.evidence_available_member_count,
                       e.physical_supported_member_count,e.best_physical_ora_fdr,
                       e.independent_support_channel_fraction,e.enrichment_scope,
                       s.pathway_context_subtype_id,s.classification_status subtype_status,
                       s.selected_k,s.proposed_k,s.silhouette,s.bootstrap_ari,
                       s.mean_pairwise_similarity
                FROM {catalog} c
                LEFT JOIN {enrichment} e ON c.geneset_id=e.geneset_id
                LEFT JOIN {context} s
                  ON c.cancer_id=s.cancer_id AND c.pathway_id=s.pathway_id
                 AND c.pathway_family_id=s.pathway_family_id
                WHERE c.geneset_id=?
                """,
                [gene_set],
            ).fetchdf()
            member_frame = con.execute(
                f"SELECT * FROM {members} WHERE geneset_id=? "
                "ORDER BY geneset_rank,lncrna_id LIMIT ? OFFSET ?",
                [gene_set, member_limit, member_offset],
            ).fetchdf()
        finally:
            con.close()
        if detail.empty:
            raise GeneSetSubtypeQueryNotFoundError(
                f"Unknown V3.2 Gene Set: {gene_set}"
            )
        detail_row = _records(detail)[0]
        return {
            "module": "gene_set_ranked_subtype",
            "query_kind": "gene_set_detail",
            "gene_set": detail_row,
            "members": _records(member_frame),
            "member_returned_rows": len(member_frame),
            "member_limit": member_limit,
            "member_offset": member_offset,
            "availability": dict(self._coverage[str(detail_row["cancer_id"])]),
            "provenance": self._provenance(),
        }

    def query_gene_set_enrichment(
        self,
        *,
        gene_set_id: str | None = None,
        cancer_id: str | None = None,
        pathway_id: str | None = None,
        min_support_fraction: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        gene_set = (
            _clean_text(gene_set_id, "gene_set_id", maximum=128)
            if gene_set_id
            else None
        )
        cancer = _cancer(cancer_id)
        pathway = _clean_text(pathway_id, "pathway_id") if pathway_id else None
        if min_support_fraction is not None and not 0 <= float(min_support_fraction) <= 1:
            raise GeneSetSubtypeQueryInputError(
                "min_support_fraction must be within [0,1]"
            )
        availability = self._availability(cancer)
        relation = self._relation("gene_set_enrichment")
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("geneset_id", gene_set),
            ("cancer_id", cancer),
            ("pathway_id", pathway),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        if min_support_fraction is not None:
            clauses.append("independent_support_channel_fraction>=?")
            params.append(float(min_support_fraction))
        where = " AND ".join(clauses) if clauses else "TRUE"
        con = self._connect()
        try:
            frame = con.execute(
                f"SELECT * FROM {relation} WHERE {where} "
                "ORDER BY cancer_id,pathway_id,geneset_id LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            "gene_set_enrichment",
            _records(frame),
            {
                "gene_set_id": gene_set,
                "cancer_id": cancer,
                "pathway_id": pathway,
                "min_support_fraction": min_support_fraction,
            },
            limit=limit,
            offset=offset,
            availability=availability,
        )

    def query_subtype_overview(
        self, *, cancer_id: str | None = None
    ) -> dict[str, Any]:
        cancer = _cancer(cancer_id)
        if cancer is not None:
            self._availability(cancer)
        relation = self._relation("cancer_program_subtypes")
        con = self._connect()
        try:
            frame = con.execute(
                f"SELECT * FROM {relation} "
                + ("WHERE cancer_id=? " if cancer is not None else "")
                + "ORDER BY cancer_id",
                [cancer] if cancer is not None else [],
            ).fetchdf()
        finally:
            con.close()
        program_by_cancer = {
            str(row["cancer_id"]): row for row in _records(frame)
        }
        cancers = [cancer] if cancer is not None else sorted(self._coverage)
        rows: list[dict[str, Any]] = []
        for cancer_key in cancers:
            coverage = dict(self._coverage[cancer_key])
            program = program_by_cancer.get(cancer_key)
            rows.append(
                {
                    "cancer_id": cancer_key,
                    "available": bool(coverage["available"]),
                    "unavailable_reason": coverage["unavailable_reason"],
                    "coverage_status": coverage["coverage_status"],
                    "publishable_genesets": coverage["publishable_genesets"],
                    "exact_pathways": coverage["exact_pathways"],
                    "member_rows": coverage["member_rows"],
                    "cancer_program": program,
                }
            )
        return self._result(
            "ranked_subtype_overview",
            rows,
            {"cancer_id": cancer},
        )

    def query_subtype_detail(
        self,
        *,
        cancer_id: str | None = None,
        pathway_id: str | None = None,
        classification_status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        cancer = _cancer(cancer_id)
        pathway = _clean_text(pathway_id, "pathway_id") if pathway_id else None
        status = (
            _clean_text(classification_status, "classification_status", maximum=64)
            if classification_status
            else None
        )
        if cancer is None and pathway is None:
            raise GeneSetSubtypeQueryInputError(
                "ranked subtype detail requires cancer_id or pathway_id"
            )
        availability = self._availability(cancer)
        context = self._relation("ranked_subtypes")
        catalog = self._relation("gene_set_catalog")
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("s.cancer_id", cancer),
            ("s.pathway_id", pathway),
            ("s.classification_status", status),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        where = " AND ".join(clauses)
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT s.*,c.geneset_id,c.geneset_name,c.direction,c.member_count
                FROM {context} s JOIN {catalog} c
                  USING (cancer_id,pathway_id,pathway_family_id)
                WHERE {where}
                ORDER BY s.pathway_id,s.cancer_id LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            "ranked_subtype_detail",
            _records(frame),
            {
                "cancer_id": cancer,
                "pathway_id": pathway,
                "classification_status": status,
            },
            limit=limit,
            offset=offset,
            availability=availability,
        )

    def query_subtype_stability(
        self,
        *,
        pathway_id: str,
        cancer_id: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        pathway = _clean_text(pathway_id, "pathway_id")
        cancer = _cancer(cancer_id)
        availability = self._availability(cancer)
        stability = self._relation("subtype_stability")
        context = self._relation("ranked_subtypes")
        clauses = ["pathway_id=?"]
        params: list[Any] = [pathway]
        if cancer is not None:
            clauses.append("(cancer_a=? OR cancer_b=?)")
            params.extend([cancer, cancer])
        con = self._connect()
        try:
            frame = con.execute(
                f"SELECT * FROM {stability} WHERE {' AND '.join(clauses)} "
                "ORDER BY cancer_a,cancer_b LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchdf()
            context_frame = con.execute(
                f"SELECT * FROM {context} WHERE pathway_id=? "
                "ORDER BY cancer_id",
                [pathway],
            ).fetchdf()
        finally:
            con.close()
        return {
            **self._result(
                "ranked_subtype_stability",
                _records(frame),
                {"pathway_id": pathway, "cancer_id": cancer},
                limit=limit,
                offset=offset,
                availability=availability,
            ),
            "pathway_contexts": _records(context_frame),
            "pathway_context_count": len(context_frame),
        }


__all__ = [
    "AUDIT_BINDING_FORMAT",
    "AUDIT_FORMAT",
    "GeneSetSubtypeQueryAssetError",
    "GeneSetSubtypeQueryError",
    "GeneSetSubtypeQueryInputError",
    "GeneSetSubtypeQueryNotFoundError",
    "GeneSetSubtypeReleaseQuery",
    "MAX_MEMBER_LIMIT",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
]
