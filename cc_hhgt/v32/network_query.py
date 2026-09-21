"""Fail-closed query loader for the hash-bound unified V3.2 network."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .input_lineage import artifact_sha256
from .network_release import (
    ANALYSIS_VERSION,
    EDGE_TYPES,
    FORMAL_CANDIDATE_SHA256,
    FORMAL_EXACT_LINEAGE_SHA256,
    FORMAL_EXPERIMENT_AUDIT_SHA256,
    FORMAL_EXPERIMENT_BRIDGE_SHA256,
    FORMAL_FUSION_AUDIT_SHA256,
    FORMAL_FUSION_BINDING_SHA256,
    FORMAL_FUSION_SCORES_SHA256,
    FORMAL_MEMBERSHIP_SHA256,
    FORMAL_PHYSICAL_AUDIT_SHA256,
    FORMAL_PHYSICAL_MANIFEST_SHA256,
    FORMAL_PHYSICAL_RELATIONSHIPS_SHA256,
    FORMAL_PRIMARY_SHA256,
    MANIFEST_FILE,
    MEMBERSHIP_EDGE_TYPE,
    MODEL_EDGE_TYPE,
    NETWORK_EDGE_FILE,
    NETWORK_NODE_FILE,
    NODE_TYPES,
    PHYSICAL_EDGE_TYPE,
    RELEASE_FORMAT,
    RELEASE_STATUS,
    SUCCESS_FILE,
)


MAX_QUERY_LIMIT = 500
MAX_QUERY_OFFSET = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class NetworkQueryError(RuntimeError):
    """Base network query error."""


class NetworkQueryAssetError(NetworkQueryError):
    """Raised when a release or binding is invalid."""


class NetworkQueryInputError(NetworkQueryError):
    """Raised for invalid query filters."""


def _sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _clean(value: Any, role: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise NetworkQueryInputError(f"{role} must be non-empty")
    if len(text) > 512:
        raise NetworkQueryInputError(f"{role} is too long")
    return text


def _lnc(value: Any) -> str:
    text = _clean(value, "lncrna_id").upper()
    for prefix in ("LNCRNA:", "LNC:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = re.sub(r"\.\d+$", "", text)
    return "LNC:" + text


def _gene(value: Any) -> str:
    text = _clean(value, "protein_gene_id").upper()
    text = re.sub(r"^(?:GENE:|PROTEIN:)", "", text)
    return re.sub(r"\.\d+$", "", text)


def _node(value: Any, node_type: str | None = None) -> str:
    text = _clean(value, "node_id")
    upper = text.upper()
    if upper.startswith("LNC:") or upper.startswith("LNCRNA:"):
        return _lnc(text)
    if upper.startswith("GENE:") or upper.startswith("PROTEIN:"):
        return "GENE:" + _gene(text)
    if upper.startswith("PATHWAY:"):
        return "PATHWAY:" + text[len("PATHWAY:") :]
    if node_type == "lncRNA":
        return _lnc(text)
    if node_type == "protein_gene":
        return "GENE:" + _gene(text)
    if node_type == "exact_pathway":
        return "PATHWAY:" + text
    raise NetworkQueryInputError(
        "node_id must include LNC:, GENE: or PATHWAY: unless node_type is supplied"
    )


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise NetworkQueryInputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise NetworkQueryInputError(f"offset must be 0..{MAX_QUERY_OFFSET}")
    return int(limit), int(offset)


def _records(frame: Any) -> list[dict[str, Any]]:
    """Return JSON-safe records while preserving typed missingness as null."""

    return frame.astype(object).where(frame.notna(), None).to_dict("records")


class UnifiedNetworkQuery:
    """Validated immutable DuckDB view over the unified network release."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_manifest_sha256: str | None,
    ) -> None:
        source = Path(manifest_path).resolve()
        if source.is_symlink() or not source.is_file():
            raise NetworkQueryAssetError(f"network manifest is missing/unsafe: {source}")
        expected = str(expected_manifest_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise NetworkQueryAssetError("network query requires an expected manifest SHA256")
        observed = artifact_sha256(source)
        if observed != expected:
            raise NetworkQueryAssetError(
                f"network manifest SHA256 mismatch: {observed} != {expected}"
            )
        try:
            manifest = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NetworkQueryAssetError("network manifest is invalid JSON") from exc
        required = {
            "format": RELEASE_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "module_id": "unified_network",
            "status": RELEASE_STATUS,
            "release_ready": False,
            "production_deployed": False,
            "formal_authority_enforced": True,
            "source_generation": "CURRENT_V3.2_ONLY",
            "primary_frozen": True,
            "primary_ranking_unchanged": True,
            "secondary_scores_remain_secondary": True,
            "native_expert_probabilities_public": True,
            "native_missingness_encoding": "availability_boolean_plus_nullable_probability",
            "missing_native_probability_imputed_to_zero": False,
            "family_to_exact_broadcast": False,
            "global_physical_facts_broadcast_to_cancers": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "experiment_route": "SUPPORT_ATTRIBUTE_ALREADY_ABSORBED_IN_EVIDENCE_TRANSFORMER",
            "experiment_separate_fusion_created": False,
            "experiment_changes_primary_ranking": False,
            "edge_types": list(EDGE_TYPES),
            "node_types": list(NODE_TYPES),
        }
        for key, wanted in required.items():
            if manifest.get(key) != wanted:
                raise NetworkQueryAssetError(
                    f"network manifest has invalid {key}: {manifest.get(key)!r}"
                )
        inputs = manifest.get("inputs")
        if not isinstance(inputs, dict):
            raise NetworkQueryAssetError("network manifest lacks input bindings")
        pinned = {
            "primary": FORMAL_PRIMARY_SHA256,
            "exact_lineage": FORMAL_EXACT_LINEAGE_SHA256,
            "fusion_scores": FORMAL_FUSION_SCORES_SHA256,
            "fusion_binding": FORMAL_FUSION_BINDING_SHA256,
            "fusion_audit": FORMAL_FUSION_AUDIT_SHA256,
            "membership": FORMAL_MEMBERSHIP_SHA256,
            "candidate": FORMAL_CANDIDATE_SHA256,
            "physical_relationships": FORMAL_PHYSICAL_RELATIONSHIPS_SHA256,
            "physical_manifest": FORMAL_PHYSICAL_MANIFEST_SHA256,
            "physical_audit": FORMAL_PHYSICAL_AUDIT_SHA256,
            "experiment_bridge": FORMAL_EXPERIMENT_BRIDGE_SHA256,
            "experiment_audit": FORMAL_EXPERIMENT_AUDIT_SHA256,
        }
        for role, digest in pinned.items():
            declaration = inputs.get(role)
            if not isinstance(declaration, dict) or declaration.get("sha256") != digest:
                raise NetworkQueryAssetError(
                    f"network {role} is not the formal current V3.2 authority"
                )
        query_link = manifest.get("experiment_query_link")
        if (
            not isinstance(query_link, dict)
            or query_link.get("bridge_sha256") != FORMAL_EXPERIMENT_BRIDGE_SHA256
            or query_link.get("separate_probability_head") is not False
            or query_link.get("double_counted_in_network_score") is not False
            or not _SHA256.fullmatch(str(query_link.get("sha256", "")))
        ):
            raise NetworkQueryAssetError("network experiment query link is invalid")
        root = source.parent.resolve()
        declarations = manifest.get("artifacts")
        if not isinstance(declarations, dict):
            raise NetworkQueryAssetError("network manifest lacks artifacts")
        paths: dict[str, Path] = {}
        for name in (NETWORK_NODE_FILE, NETWORK_EDGE_FILE):
            declaration = declarations.get(name)
            if not isinstance(declaration, dict):
                raise NetworkQueryAssetError(f"network manifest lacks {name}")
            relative = Path(str(declaration.get("path", "")))
            if relative.is_absolute():
                raise NetworkQueryAssetError(f"network artifact path must be relative: {name}")
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise NetworkQueryAssetError(f"network artifact escaped release root: {name}") from exc
            if path.is_symlink() or not path.is_file() or path.name != name:
                raise NetworkQueryAssetError(f"network artifact is missing/unsafe: {path}")
            if artifact_sha256(path) != declaration.get("sha256"):
                raise NetworkQueryAssetError(f"network artifact SHA256 drift: {name}")
            rows = declaration.get("rows")
            if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
                raise NetworkQueryAssetError(f"network artifact row declaration invalid: {name}")
            paths[name] = path
        success_path = root / SUCCESS_FILE
        try:
            success = json.loads(success_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NetworkQueryAssetError("network SUCCESS marker is invalid") from exc
        if (
            success.get("status") != RELEASE_STATUS
            or success.get("release_ready") is not False
            or success.get("production_deployed") is not False
            or success.get("manifest") != MANIFEST_FILE
            or success.get("manifest_sha256") != observed
        ):
            raise NetworkQueryAssetError("network SUCCESS marker is stale")
        self.manifest = manifest
        self.manifest_path = source
        self.manifest_sha256 = observed
        self.nodes_path = paths[NETWORK_NODE_FILE]
        self.edges_path = paths[NETWORK_EDGE_FILE]
        self.experiment_query_link = query_link
        self._validate_tables()

    def _connect(self):
        try:
            import duckdb
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise NetworkQueryAssetError("duckdb is required for network queries") from exc
        return duckdb.connect(":memory:")

    def _validate_tables(self) -> None:
        node_required = {
            "node_id", "entity_id", "node_type", "display_label",
            "in_model_candidate_authority", "in_exact_membership",
            "in_physical_interaction", "analysis_version", "generation",
        }
        edge_required = {
            "edge_id", "edge_type", "source_node_id", "target_node_id", "directed",
            "cancer_id", "cancer_scope", "lncrna_id", "pathway_id",
            "protein_gene_id", "primary_probability",
            "discovery_adjusted_probability", "fused_confidence_probability",
            "genomic_native_available", "genomic_native_probability",
            "single_cell_native_available", "single_cell_native_probability",
            "evidence_transformer_native_available",
            "evidence_transformer_native_probability", "experiment_support_available",
            "experiment_event_count", "experiment_source_record_count",
            "experiment_support_role", "physical_fact_count", "availability",
            "is_prediction", "primary_ranking_unchanged",
            "adjusted_ranking_is_secondary", "used_for_primary_release",
            "family_to_exact_broadcast", "changes_primary_ranking",
            "analysis_version", "generation",
        }
        con = self._connect()
        try:
            nrel = f"read_parquet({_sql_path(self.nodes_path)})"
            erel = f"read_parquet({_sql_path(self.edges_path)})"
            ncols = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {nrel}").fetchall()}
            ecols = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {erel}").fetchall()}
            if missing := sorted(node_required - ncols):
                raise NetworkQueryAssetError(f"network nodes lack columns: {missing}")
            if missing := sorted(edge_required - ecols):
                raise NetworkQueryAssetError(f"network edges lack columns: {missing}")
            node_declared = int(self.manifest["artifacts"][NETWORK_NODE_FILE]["rows"])
            edge_declared = int(self.manifest["artifacts"][NETWORK_EDGE_FILE]["rows"])
            observed = con.execute(
                f"""
                SELECT
                  (SELECT count(*) FROM {nrel}),
                  (SELECT count(*) FROM {erel}),
                  (SELECT count(*) - count(DISTINCT node_id) FROM {nrel}),
                  (SELECT count(*) - count(DISTINCT edge_id) FROM {erel}),
                  (SELECT count(*) FROM {nrel} WHERE node_type NOT IN ({','.join('?' for _ in NODE_TYPES)})
                     OR analysis_version <> ?),
                  (SELECT count(*) FROM {erel} WHERE edge_type NOT IN ({','.join('?' for _ in EDGE_TYPES)})
                     OR analysis_version <> ? OR family_to_exact_broadcast
                     OR changes_primary_ranking),
                  (SELECT count(*) FROM {erel} WHERE edge_type=? AND (
                     (genomic_native_available AND genomic_native_probability IS NULL)
                     OR (NOT genomic_native_available AND genomic_native_probability IS NOT NULL)
                     OR (single_cell_native_available AND single_cell_native_probability IS NULL)
                     OR (NOT single_cell_native_available AND single_cell_native_probability IS NOT NULL)
                     OR (evidence_transformer_native_available AND evidence_transformer_native_probability IS NULL)
                     OR (NOT evidence_transformer_native_available AND evidence_transformer_native_probability IS NOT NULL)
                     OR NOT primary_ranking_unchanged OR NOT adjusted_ranking_is_secondary
                     OR used_for_primary_release)),
                  (SELECT count(*) FROM {erel} WHERE edge_type<>? AND (
                     genomic_native_available IS NOT NULL OR genomic_native_probability IS NOT NULL
                     OR single_cell_native_available IS NOT NULL OR single_cell_native_probability IS NOT NULL
                     OR evidence_transformer_native_available IS NOT NULL
                     OR evidence_transformer_native_probability IS NOT NULL)),
                  (SELECT count(*) FROM {erel} WHERE experiment_support_available
                     AND (experiment_event_count IS NULL OR experiment_event_count <= 0
                       OR experiment_support_role <> 'ABSORBED_IN_EVIDENCE_TRANSFORMER_NO_SEPARATE_FUSION')),
                  (SELECT count(*) FROM {erel} WHERE NOT experiment_support_available
                     AND (experiment_event_count IS NOT NULL OR experiment_source_record_count IS NOT NULL
                       OR experiment_support_role IS NOT NULL))
                """,
                [
                    *NODE_TYPES, ANALYSIS_VERSION,
                    *EDGE_TYPES, ANALYSIS_VERSION,
                    MODEL_EDGE_TYPE, MODEL_EDGE_TYPE,
                ],
            ).fetchone()
            if int(observed[0]) != node_declared or int(observed[1]) != edge_declared:
                raise NetworkQueryAssetError("network table row count drift")
            if any(int(value) for value in observed[2:]):
                raise NetworkQueryAssetError(
                    f"network table semantic validation failed: {tuple(map(int, observed[2:]))}"
                )
        finally:
            con.close()

    def _result(
        self,
        kind: str,
        rows: list[dict[str, Any]],
        filters: dict[str, Any],
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        return {
            "module": "unified_network",
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
                "release_ready": False,
                "production_deployed": False,
                "primary_frozen": True,
                "family_to_exact_broadcast": False,
                "native_missingness_encoding": "availability_boolean_plus_nullable_probability",
                "experiment_bridge_sha256": self.experiment_query_link["bridge_sha256"],
                "experiment_double_counted": False,
            },
        }

    def query_nodes(
        self,
        *,
        node_id: Any | None = None,
        node_type: str | None = None,
        entity_id: Any | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        if node_type is not None and node_type not in NODE_TYPES:
            raise NetworkQueryInputError(f"unknown node_type: {node_type}")
        clauses: list[str] = []
        parameters: list[Any] = []
        canonical_node = None
        if node_id is not None:
            canonical_node = _node(node_id, node_type=node_type)
            clauses.append("node_id = ?")
            parameters.append(canonical_node)
        clean_entity = None
        if entity_id is not None:
            clean_entity = _clean(entity_id, "entity_id")
            clauses.append("entity_id = ?")
            parameters.append(clean_entity)
        if node_type is not None:
            clauses.append("node_type = ?")
            parameters.append(node_type)
        where = " AND ".join(clauses) if clauses else "TRUE"
        relation = f"read_parquet({_sql_path(self.nodes_path)})"
        con = self._connect()
        try:
            frame = con.execute(
                f"SELECT * FROM {relation} WHERE {where} ORDER BY node_type, node_id LIMIT ? OFFSET ?",
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            "nodes",
            _records(frame),
            {"node_id": canonical_node, "node_type": node_type, "entity_id": clean_entity},
            limit,
            offset,
        )

    def query_neighborhood(
        self,
        *,
        node_id: Any,
        node_type: str | None = None,
        edge_type: str | None = None,
        cancer_id: Any | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        canonical = _node(node_id, node_type=node_type)
        if edge_type is not None and edge_type not in EDGE_TYPES:
            raise NetworkQueryInputError(f"unknown edge_type: {edge_type}")
        clauses = ["(source_node_id = ? OR target_node_id = ?)"]
        parameters: list[Any] = [canonical, canonical]
        if edge_type is not None:
            clauses.append("edge_type = ?")
            parameters.append(edge_type)
        cancer = None
        if cancer_id is not None:
            cancer = _clean(cancer_id, "cancer_id").upper()
            clauses.append("cancer_id = ?")
            parameters.append(cancer)
        relation = f"read_parquet({_sql_path(self.edges_path)})"
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT * FROM {relation}
                WHERE {' AND '.join(clauses)}
                ORDER BY edge_type, cancer_id NULLS LAST,
                         primary_probability DESC NULLS LAST, edge_id
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            "neighborhood",
            _records(frame),
            {"node_id": canonical, "edge_type": edge_type, "cancer_id": cancer},
            limit,
            offset,
        )

    def query_model_associations(
        self,
        *,
        lncrna_id: Any,
        cancer_id: Any | None = None,
        pathway_id: Any | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        lnc = _lnc(lncrna_id)
        clauses = ["edge_type = ?", "lncrna_id = ?"]
        parameters: list[Any] = [MODEL_EDGE_TYPE, lnc]
        cancer = None
        if cancer_id is not None:
            cancer = _clean(cancer_id, "cancer_id").upper()
            clauses.append("cancer_id = ?")
            parameters.append(cancer)
        pathway = None
        if pathway_id is not None:
            pathway = _clean(pathway_id, "pathway_id")
            clauses.append("pathway_id = ?")
            parameters.append(pathway)
        relation = f"read_parquet({_sql_path(self.edges_path)})"
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT * FROM {relation} WHERE {' AND '.join(clauses)}
                ORDER BY primary_probability DESC, cancer_id, pathway_id
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            "model_exact_pathway_associations",
            _records(frame),
            {"lncrna_id": lnc, "cancer_id": cancer, "pathway_id": pathway},
            limit,
            offset,
        )

    def query_experiment_supported_associations(
        self,
        *,
        lncrna_id: Any,
        cancer_id: Any | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        lnc = _lnc(lncrna_id)
        clauses = ["edge_type = ?", "lncrna_id = ?", "experiment_support_available"]
        parameters: list[Any] = [MODEL_EDGE_TYPE, lnc]
        cancer = None
        if cancer_id is not None:
            cancer = _clean(cancer_id, "cancer_id").upper()
            clauses.append("cancer_id = ?")
            parameters.append(cancer)
        relation = f"read_parquet({_sql_path(self.edges_path)})"
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT cancer_id, lncrna_id, pathway_id, primary_probability,
                       fused_confidence_probability,
                       evidence_transformer_native_available,
                       evidence_transformer_native_probability,
                       experiment_event_count, experiment_source_record_count,
                       experiment_support_role, primary_ranking_unchanged,
                       changes_primary_ranking
                FROM {relation} WHERE {' AND '.join(clauses)}
                ORDER BY experiment_event_count DESC, primary_probability DESC, pathway_id
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        result = self._result(
            "experiment_supported_model_associations",
            _records(frame),
            {"lncrna_id": lnc, "cancer_id": cancer},
            limit,
            offset,
        )
        result["experiment_query_link"] = dict(self.experiment_query_link)
        result["separate_experiment_probability_head"] = False
        result["double_counted_in_score"] = False
        return result


__all__ = [
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
    "NetworkQueryAssetError",
    "NetworkQueryError",
    "NetworkQueryInputError",
    "UnifiedNetworkQuery",
]
