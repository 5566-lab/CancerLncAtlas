"""Hash-pinned read-only queries for a V3.2 physical-interaction release."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .interaction_release import (
    ANALYSIS_VERSION,
    FORMAL_CANDIDATE_SHA256,
    FORMAL_MEMBERSHIP_SHA256,
    GLOBAL_SCOPE,
    RELEASE_FORMAT,
)
from .release_registry import artifact_sha256


MAX_QUERY_LIMIT = 1_000
MAX_QUERY_OFFSET = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACTS = {
    "relationships": "physical_interaction_relationships.parquet",
    "evidence": "relationship_evidence.parquet",
    "enrichment": "interaction_exact_pathway_enrichment.parquet",
    "unmapped": "unmapped_physical_partners.parquet",
}


class InteractionQueryError(RuntimeError):
    """Base interaction query error."""


class InteractionQueryAssetError(InteractionQueryError):
    """Raised when a release artifact or hash is invalid."""


class InteractionQueryInputError(InteractionQueryError):
    """Raised when a query filter is invalid."""


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _canonical_lnc(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"^(?:LNC|LNCRNA):", "", text)
    text = re.sub(r"\.\d+$", "", text)
    if not text:
        raise InteractionQueryInputError("lncrna_id is required")
    return "LNC:" + text


def _clean(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise InteractionQueryInputError(f"{label} is required")
    if len(text) > 256:
        raise InteractionQueryInputError(f"{label} is too long")
    return text


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise InteractionQueryInputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise InteractionQueryInputError(f"offset must be 0..{MAX_QUERY_OFFSET}")
    return int(limit), int(offset)


class InteractionReleaseQuery:
    """Validated immutable view with parameterised DuckDB queries."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_manifest_sha256: str | None,
    ) -> None:
        source = Path(manifest_path).resolve()
        if not source.is_file():
            raise InteractionQueryAssetError(f"Interaction manifest is missing: {source}")
        expected = str(expected_manifest_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise InteractionQueryAssetError(
                "Interaction query requires an expected manifest SHA256"
            )
        observed = artifact_sha256(source)
        if observed != expected:
            raise InteractionQueryAssetError(
                f"Interaction manifest SHA256 mismatch: {observed} != {expected}"
            )
        try:
            manifest = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InteractionQueryAssetError("Interaction manifest is invalid JSON") from exc
        required_flags = {
            "format": RELEASE_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "SUCCESS_NEWLY_MATERIALIZED_V32",
            "formal_authority_enforced": True,
            "release_ready": False,
            "production_deployed": False,
            "old_interaction_pathway_support_loaded": False,
            "old_predictions_used": False,
            "old_rankings_used": False,
            "old_checkpoints_used": False,
            "family_to_exact_broadcast": False,
            "global_facts_broadcast_to_cancers": False,
        }
        for key, expected_value in required_flags.items():
            if manifest.get(key) != expected_value:
                raise InteractionQueryAssetError(
                    f"Interaction manifest has invalid {key}: {manifest.get(key)!r}"
                )
        binding = manifest.get("fresh_evidence_output_binding")
        if (
            not isinstance(binding, dict)
            or not str(binding.get("evidence_training_run_id", "")).startswith(
                "V32-EVIDENCE-TRAIN-"
            )
            or binding.get("five_fresh_private_heads_verified") is not True
            or not _SHA256.fullmatch(str(binding.get("sha256", "")))
            or not _SHA256.fullmatch(str(binding.get("success_sha256", "")))
            or not _SHA256.fullmatch(str(binding.get("physical_facts_sha256", "")))
        ):
            raise InteractionQueryAssetError(
                "Interaction manifest lacks a formal fresh Evidence output binding"
            )
        semantic_wrapper = manifest.get("fresh_evidence_semantic_wrapper")
        if (
            not isinstance(semantic_wrapper, dict)
            or semantic_wrapper.get("evidence_training_run_id")
            != binding.get("evidence_training_run_id")
            or semantic_wrapper.get("confidence_only") is not True
            or semantic_wrapper.get("affects_discovery") is not False
            or semantic_wrapper.get("affects_primary_ranking") is not False
            or semantic_wrapper.get("direct_exact_pathway_assertion") is not False
            or not _SHA256.fullmatch(str(semantic_wrapper.get("sha256", "")))
            or not _SHA256.fullmatch(
                str(semantic_wrapper.get("success_sha256", ""))
            )
            or not _SHA256.fullmatch(
                str(semantic_wrapper.get("independent_post_audit_sha256", ""))
            )
        ):
            raise InteractionQueryAssetError(
                "Interaction manifest lacks the audited Evidence semantic wrapper"
            )
        inputs = manifest.get("inputs")
        if not isinstance(inputs, dict):
            raise InteractionQueryAssetError("Interaction manifest has no input map")
        expected_inputs = {
            "exact_membership": FORMAL_MEMBERSHIP_SHA256,
            "exact_candidates": FORMAL_CANDIDATE_SHA256,
        }
        for role, digest in expected_inputs.items():
            declaration = inputs.get(role)
            if not isinstance(declaration, dict) or declaration.get("sha256") != digest:
                raise InteractionQueryAssetError(
                    f"Interaction {role} is not current formal V3.2 authority"
                )
        evidence_declaration = inputs.get("evidence_output_binding")
        if (
            not isinstance(evidence_declaration, dict)
            or evidence_declaration.get("sha256") != binding.get("sha256")
        ):
            raise InteractionQueryAssetError(
                "Interaction Evidence binding input declaration is inconsistent"
            )
        wrapper_declaration = inputs.get("evidence_semantic_wrapper")
        if (
            not isinstance(wrapper_declaration, dict)
            or wrapper_declaration.get("sha256")
            != semantic_wrapper.get("sha256")
        ):
            raise InteractionQueryAssetError(
                "Interaction Evidence semantic wrapper input declaration is inconsistent"
            )
        declarations = manifest.get("artifacts")
        if not isinstance(declarations, dict):
            raise InteractionQueryAssetError("Interaction manifest has no artifact map")
        root = source.parent.resolve()
        paths: dict[str, Path] = {}
        for role, name in _ARTIFACTS.items():
            declaration = declarations.get(name)
            if not isinstance(declaration, dict):
                raise InteractionQueryAssetError(f"Interaction manifest lacks {name}")
            relative = Path(str(declaration.get("path", "")))
            if relative.is_absolute():
                raise InteractionQueryAssetError(f"Interaction artifact path must be relative: {name}")
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise InteractionQueryAssetError(
                    f"Interaction artifact escaped release root: {name}"
                ) from exc
            if not path.is_file() or path.is_symlink():
                raise InteractionQueryAssetError(f"Interaction artifact is missing/unsafe: {path}")
            if path.name != name:
                raise InteractionQueryAssetError(f"Interaction artifact role/path mismatch: {name}")
            if artifact_sha256(path) != declaration.get("sha256"):
                raise InteractionQueryAssetError(f"Interaction artifact SHA drift: {name}")
            rows = declaration.get("rows")
            if (
                isinstance(rows, bool)
                or not isinstance(rows, int)
                or rows < 0
                or (role != "unmapped" and rows == 0)
            ):
                raise InteractionQueryAssetError(f"Interaction artifact is empty: {name}")
            paths[role] = path
        self.manifest = manifest
        self.manifest_path = source
        self.manifest_sha256 = observed
        self.paths = paths
        self._validate_tables()

    def _connect(self):
        try:
            import duckdb
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise InteractionQueryAssetError("duckdb is required for interaction queries") from exc
        return duckdb.connect(":memory:")

    def _validate_tables(self) -> None:
        required = {
            "relationships": {
                "relationship_id", "cancer_scope", "lncrna_id", "partner_gene_id",
                "physical_fact_count", "availability", "is_prediction", "analysis_version",
            },
            "evidence": {
                "relationship_id", "physical_fact_id", "cancer_scope", "lncrna_id",
                "partner_gene_id", "evidence_type", "is_prediction", "analysis_version",
            },
            "enrichment": {
                "enrichment_id", "cancer_scope", "lncrna_id", "pathway_id",
                "overlap_target_count", "ora_p_value", "ora_fdr", "fold_enrichment",
                "availability", "family_to_exact_broadcast", "analysis_version",
            },
            "unmapped": {
                "relationship_id", "cancer_scope", "lncrna_id", "partner_gene_id",
                "physical_fact_count", "availability", "is_prediction",
                "analysis_version", "mapping_status",
            },
        }
        con = self._connect()
        try:
            for role, columns in required.items():
                relation = f"read_parquet({_sql_path(self.paths[role])})"
                observed_columns = {
                    row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
                }
                missing = sorted(columns - observed_columns)
                if missing:
                    raise InteractionQueryAssetError(
                        f"Interaction {role} lacks columns: {missing}"
                    )
                declared = int(
                    self.manifest["artifacts"][_ARTIFACTS[role]]["rows"]
                )
                count = int(con.execute(f"SELECT count(*) FROM {relation}").fetchone()[0])
                if count != declared:
                    raise InteractionQueryAssetError(
                        f"Interaction {role} row count drift: {count} != {declared}"
                    )
            relationships = f"read_parquet({_sql_path(self.paths['relationships'])})"
            evidence = f"read_parquet({_sql_path(self.paths['evidence'])})"
            enrichment = f"read_parquet({_sql_path(self.paths['enrichment'])})"
            unmapped = f"read_parquet({_sql_path(self.paths['unmapped'])})"
            bad = con.execute(
                f"""
                SELECT
                  (SELECT count(*) FROM {relationships}
                   WHERE NOT availability OR is_prediction OR analysis_version <> ?) AS bad_rel,
                  (SELECT count(*) FROM {evidence}
                   WHERE is_prediction OR analysis_version <> ?) AS bad_evidence,
                  (SELECT count(*) FROM {enrichment}
                   WHERE NOT availability OR family_to_exact_broadcast
                      OR analysis_version <> ? OR ora_p_value < 0 OR ora_p_value > 1
                      OR ora_fdr < 0 OR ora_fdr > 1) AS bad_enrichment,
                  (SELECT count(*) FROM {evidence} e
                   LEFT JOIN {relationships} r USING (relationship_id)
                   WHERE r.relationship_id IS NULL) AS orphan_evidence,
                  (SELECT count(*) FROM {unmapped}
                   WHERE NOT availability OR is_prediction OR analysis_version <> ?
                      OR mapping_status <> 'PARTNER_OUTSIDE_EXACT_MEMBERSHIP_UNIVERSE'
                      OR relationship_id IS NULL OR trim(relationship_id) = ''
                      OR lncrna_id IS NULL OR trim(lncrna_id) = ''
                      OR partner_gene_id IS NULL OR trim(partner_gene_id) = '') AS bad_unmapped,
                  (SELECT count(*) - count(DISTINCT relationship_id)
                   FROM {unmapped}) AS duplicate_unmapped
                """,
                [ANALYSIS_VERSION, ANALYSIS_VERSION, ANALYSIS_VERSION, ANALYSIS_VERSION],
            ).fetchone()
            if any(int(value) for value in bad):
                raise InteractionQueryAssetError(
                    f"Interaction release semantic validation failed: {tuple(map(int, bad))}"
                )
        finally:
            con.close()

    def _result(self, kind: str, rows: list[dict[str, Any]], filters: dict[str, Any], limit: int, offset: int) -> dict[str, Any]:
        return {
            "module": "physical_interaction",
            "query_kind": kind,
            "filters": filters,
            "limit": limit,
            "offset": offset,
            "returned_rows": len(rows),
            "rows": rows,
            "provenance": {
                "analysis_version": ANALYSIS_VERSION,
                "manifest_path": str(self.manifest_path),
                "manifest_sha256": self.manifest_sha256,
                "old_results_used": False,
                "global_facts_broadcast_to_cancers": False,
                "evidence_semantic_wrapper_sha256": self.manifest[
                    "fresh_evidence_semantic_wrapper"
                ]["sha256"],
            },
        }

    def query_relationships(
        self,
        *,
        lncrna_id: Any,
        cancer_scope: Any | None = None,
        partner_gene_id: Any | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        lnc = _canonical_lnc(lncrna_id)
        clauses = ["lncrna_id = ?"]
        parameters: list[Any] = [lnc]
        scope = None
        if cancer_scope is not None:
            raw_scope = _clean(cancer_scope, "cancer_scope").upper()
            scope = GLOBAL_SCOPE if raw_scope in {"GLOBAL", "PAN_CANCER", GLOBAL_SCOPE} else raw_scope
            clauses.append("cancer_scope = ?")
            parameters.append(scope)
        partner = None
        if partner_gene_id is not None:
            partner = re.sub(r"^(?:GENE|PROTEIN):", "", _clean(partner_gene_id, "partner_gene_id").upper())
            partner = re.sub(r"\.\d+$", "", partner)
            clauses.append("partner_gene_id = ?")
            parameters.append(partner)
        relation = f"read_parquet({_sql_path(self.paths['relationships'])})"
        sql = f"""
            SELECT * FROM {relation}
            WHERE {' AND '.join(clauses)}
            ORDER BY physical_fact_count DESC, independent_pmid_count DESC,
                     cancer_scope, partner_gene_id
            LIMIT ? OFFSET ?
        """
        parameters.extend([limit, offset])
        con = self._connect()
        try:
            frame = con.execute(sql, parameters).fetchdf()
        finally:
            con.close()
        return self._result(
            "relationships", frame.to_dict("records"),
            {"lncrna_id": lnc, "cancer_scope": scope, "partner_gene_id": partner},
            limit, offset,
        )

    def query_enrichment(
        self,
        *,
        lncrna_id: Any,
        cancer_scope: Any | None = None,
        max_fdr: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        lnc = _canonical_lnc(lncrna_id)
        clauses = ["lncrna_id = ?"]
        parameters: list[Any] = [lnc]
        scope = None
        if cancer_scope is not None:
            raw_scope = _clean(cancer_scope, "cancer_scope").upper()
            scope = GLOBAL_SCOPE if raw_scope in {"GLOBAL", "PAN_CANCER", GLOBAL_SCOPE} else raw_scope
            clauses.append("cancer_scope = ?")
            parameters.append(scope)
        fdr = None
        if max_fdr is not None:
            fdr = float(max_fdr)
            if not 0 <= fdr <= 1:
                raise InteractionQueryInputError("max_fdr must be within 0..1")
            clauses.append("ora_fdr <= ?")
            parameters.append(fdr)
        relation = f"read_parquet({_sql_path(self.paths['enrichment'])})"
        sql = f"""
            SELECT * FROM {relation}
            WHERE {' AND '.join(clauses)}
            ORDER BY ora_fdr, ora_p_value, fold_enrichment DESC, pathway_id
            LIMIT ? OFFSET ?
        """
        parameters.extend([limit, offset])
        con = self._connect()
        try:
            frame = con.execute(sql, parameters).fetchdf()
        finally:
            con.close()
        return self._result(
            "interaction_exact_pathway_enrichment", frame.to_dict("records"),
            {"lncrna_id": lnc, "cancer_scope": scope, "max_fdr": fdr},
            limit, offset,
        )

    def query_evidence(
        self, *, relationship_id: Any, limit: int = 200, offset: int = 0
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        relationship = _clean(relationship_id, "relationship_id")
        relation = f"read_parquet({_sql_path(self.paths['evidence'])})"
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT * FROM {relation}
                WHERE relationship_id = ?
                ORDER BY physical_fact_id
                LIMIT ? OFFSET ?
                """,
                [relationship, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            "relationship_evidence", frame.to_dict("records"),
            {"relationship_id": relationship}, limit, offset,
        )

    def query_unmapped_partners(
        self,
        *,
        lncrna_id: Any | None = None,
        cancer_scope: Any | None = None,
        partner_gene_id: Any | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return bound physical partners outside the exact membership universe.

        These are retained physical facts, not exact-pathway negatives and not
        predictions.  A missing exact-pathway mapping must never silently drop
        the underlying relationship.
        """

        limit, offset = _bounds(limit, offset)
        clauses: list[str] = []
        parameters: list[Any] = []
        lnc = None
        if lncrna_id is not None:
            lnc = _canonical_lnc(lncrna_id)
            clauses.append("lncrna_id = ?")
            parameters.append(lnc)
        scope = None
        if cancer_scope is not None:
            raw_scope = _clean(cancer_scope, "cancer_scope").upper()
            scope = (
                GLOBAL_SCOPE
                if raw_scope in {"GLOBAL", "PAN_CANCER", GLOBAL_SCOPE}
                else raw_scope
            )
            clauses.append("cancer_scope = ?")
            parameters.append(scope)
        partner = None
        if partner_gene_id is not None:
            partner = re.sub(
                r"^(?:GENE|PROTEIN):",
                "",
                _clean(partner_gene_id, "partner_gene_id").upper(),
            )
            partner = re.sub(r"\.\d+$", "", partner)
            clauses.append("partner_gene_id = ?")
            parameters.append(partner)
        where = " AND ".join(clauses) if clauses else "TRUE"
        relation = f"read_parquet({_sql_path(self.paths['unmapped'])})"
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT * FROM {relation}
                WHERE {where}
                ORDER BY physical_fact_count DESC, independent_pmid_count DESC,
                         cancer_scope, lncrna_id, partner_gene_id
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        result = self._result(
            "unmapped_physical_partners",
            frame.to_dict("records"),
            {
                "lncrna_id": lnc,
                "cancer_scope": scope,
                "partner_gene_id": partner,
            },
            limit,
            offset,
        )
        result.update(
            {
                "mapping_status": "PARTNER_OUTSIDE_EXACT_MEMBERSHIP_UNIVERSE",
                "exact_pathway_negative_claimed": False,
                "affects_primary_ranking": False,
                "unmapped_rows_bound": self.manifest["artifacts"][
                    _ARTIFACTS["unmapped"]
                ]["rows"],
            }
        )
        return result


__all__ = [
    "InteractionQueryAssetError",
    "InteractionQueryError",
    "InteractionQueryInputError",
    "InteractionReleaseQuery",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
]
