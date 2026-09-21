"""Hash-pinned read-only queries for fresh V3.2 structural Drug mechanisms."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .drug_mechanism import (
    ANALYSIS_VERSION,
    FORMAL_CANDIDATE_SHA256,
    FORMAL_MEMBERSHIP_SHA256,
    MECHANISM_FORMAT,
    PROBABILITY_COLUMN,
)
from .release_registry import artifact_sha256


MAX_QUERY_LIMIT = 1_000
MAX_QUERY_OFFSET = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_NAME = "lncrna_exact_pathway_target_drug_mechanisms.parquet"


class DrugMechanismQueryError(RuntimeError):
    """Base structural Drug mechanism query error."""


class DrugMechanismQueryAssetError(DrugMechanismQueryError):
    """Raised when a release artifact, lineage declaration, or hash is invalid."""


class DrugMechanismQueryInputError(DrugMechanismQueryError):
    """Raised when a query filter is invalid."""


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _canonical_lnc(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"^(?:LNC|LNCRNA):", "", text)
    text = re.sub(r"\.\d+$", "", text)
    if not text:
        raise DrugMechanismQueryInputError("lncrna_id is required")
    return "LNC:" + text


def _clean(value: Any, label: str, *, uppercase: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        raise DrugMechanismQueryInputError(f"{label} is required")
    if len(text) > 256:
        raise DrugMechanismQueryInputError(f"{label} is too long")
    return text.upper() if uppercase else text


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise DrugMechanismQueryInputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise DrugMechanismQueryInputError(f"offset must be 0..{MAX_QUERY_OFFSET}")
    return int(limit), int(offset)


class DrugMechanismReleaseQuery:
    """Immutable structural-hypothesis view backed by a pinned manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_manifest_sha256: str | None,
    ) -> None:
        source = Path(manifest_path).resolve()
        if not source.is_file():
            raise DrugMechanismQueryAssetError(f"Drug mechanism manifest is missing: {source}")
        expected = str(expected_manifest_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise DrugMechanismQueryAssetError(
                "Drug mechanism query requires an expected manifest SHA256"
            )
        observed = artifact_sha256(source)
        if observed != expected:
            raise DrugMechanismQueryAssetError(
                f"Drug mechanism manifest SHA256 mismatch: {observed} != {expected}"
            )
        try:
            manifest = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DrugMechanismQueryAssetError("Drug mechanism manifest is invalid JSON") from exc
        required = {
            "format": MECHANISM_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "module_id": "drug",
            "status": "SUCCESS_NEWLY_MATERIALIZED_V32",
            "release_ready": False,
            "production_deployed": False,
            "native_target_keys": ["cancer_id", "lncrna_id", "drug_id"],
            "actionability_separate_from_exact_pathway": True,
            "does_not_change_primary_pathway_ranking": True,
            "drug_response_association_probability_not_efficacy_or_direction": True,
            "signed_rho_private_only": True,
            "mechanism_semantics": "STRUCTURAL_HYPOTHESIS_NOT_MODEL_ATTRIBUTION",
            "causal_mechanism_claimed": False,
            "target_contribution_claimed": False,
            "family_to_exact_broadcast": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "current_v32_drug_predictions_used_as_binding": True,
        }
        for key, expected_value in required.items():
            if manifest.get(key) != expected_value:
                raise DrugMechanismQueryAssetError(
                    f"Drug mechanism manifest has invalid {key}: {manifest.get(key)!r}"
                )
        binding = manifest.get("formal_drug_bundle_binding")
        if not isinstance(binding, dict) or not str(binding.get("training_run_id", "")).strip():
            raise DrugMechanismQueryAssetError("Drug mechanism manifest is not formal-bundle bound")
        for key in ("success_sha256", "lineage_sha256"):
            if not _SHA256.fullmatch(str(binding.get(key, ""))):
                raise DrugMechanismQueryAssetError(f"Drug mechanism binding lacks valid {key}")
        inputs = manifest.get("inputs")
        if not isinstance(inputs, dict):
            raise DrugMechanismQueryAssetError("Drug mechanism manifest lacks inputs")
        expected_inputs = {
            "current_exact_candidates": FORMAL_CANDIDATE_SHA256,
            "current_exact_membership": FORMAL_MEMBERSHIP_SHA256,
        }
        for role, digest in expected_inputs.items():
            declaration = inputs.get(role)
            if not isinstance(declaration, dict) or declaration.get("sha256") != digest:
                raise DrugMechanismQueryAssetError(f"Drug mechanism {role} is not formal authority")

        declaration = manifest.get("artifact")
        if not isinstance(declaration, dict):
            raise DrugMechanismQueryAssetError("Drug mechanism manifest lacks artifact")
        relative = Path(str(declaration.get("path", "")))
        if relative.is_absolute():
            raise DrugMechanismQueryAssetError("Drug mechanism artifact path must be relative")
        root = source.parent.resolve()
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise DrugMechanismQueryAssetError("Drug mechanism artifact escaped release root") from exc
        if artifact.name != _ARTIFACT_NAME:
            raise DrugMechanismQueryAssetError("Drug mechanism artifact role/path mismatch")
        if not artifact.is_file() or artifact.is_symlink():
            raise DrugMechanismQueryAssetError(f"Drug mechanism artifact is missing/unsafe: {artifact}")
        if artifact_sha256(artifact) != declaration.get("sha256"):
            raise DrugMechanismQueryAssetError("Drug mechanism artifact SHA drift")
        rows = declaration.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise DrugMechanismQueryAssetError("Drug mechanism artifact is empty")

        self.manifest = manifest
        self.manifest_path = source
        self.manifest_sha256 = observed
        self.artifact_path = artifact
        self.training_run_id = str(binding["training_run_id"])
        self._validate_table(rows)

    def _connect(self):
        try:
            import duckdb
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise DrugMechanismQueryAssetError("duckdb is required for Drug mechanism queries") from exc
        return duckdb.connect(":memory:")

    def _validate_table(self, declared_rows: int) -> None:
        required = {
            "mechanism_path_id", "cancer_id", "lncrna_id", "pathway_id",
            "target_gene_id", "drug_id", PROBABILITY_COLUMN, "training_run_id",
            "mechanism_semantics", "causal_mechanism_claimed",
            "target_contribution_claimed", "family_to_exact_broadcast", "analysis_version",
        }
        relation = f"read_parquet({_sql_path(self.artifact_path)})"
        con = self._connect()
        try:
            columns = {
                row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
            }
            if missing := sorted(required - columns):
                raise DrugMechanismQueryAssetError(
                    f"Drug mechanism artifact lacks columns: {missing}"
                )
            audit = con.execute(
                f"""
                SELECT
                  count(*),
                  sum(CASE WHEN analysis_version <> ? OR training_run_id <> ?
                                 OR mechanism_semantics <> 'STRUCTURAL_HYPOTHESIS_NOT_MODEL_ATTRIBUTION'
                                 OR causal_mechanism_claimed OR target_contribution_claimed
                                 OR family_to_exact_broadcast
                                 OR {PROBABILITY_COLUMN} IS NULL
                                 OR NOT isfinite({PROBABILITY_COLUMN})
                                 OR {PROBABILITY_COLUMN} NOT BETWEEN 0 AND 1
                           THEN 1 ELSE 0 END),
                  count(DISTINCT mechanism_path_id),
                  count(DISTINCT (cancer_id, lncrna_id, pathway_id, target_gene_id, drug_id))
                FROM {relation}
                """,
                [ANALYSIS_VERSION, self.training_run_id],
            ).fetchone()
            if (
                int(audit[0]) != declared_rows
                or int(audit[1] or 0)
                or int(audit[2]) != declared_rows
                or int(audit[3]) != declared_rows
            ):
                raise DrugMechanismQueryAssetError(
                    f"Drug mechanism semantic/row validation failed: {tuple(map(int, audit))}"
                )
        finally:
            con.close()

    def query_mechanisms(
        self,
        *,
        lncrna_id: Any,
        cancer_id: Any | None = None,
        drug_id: Any | None = None,
        pathway_id: Any | None = None,
        target_gene_id: Any | None = None,
        min_probability: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        lnc = _canonical_lnc(lncrna_id)
        clauses = ["lncrna_id = ?"]
        parameters: list[Any] = [lnc]
        filters: dict[str, Any] = {"lncrna_id": lnc}
        for value, label, uppercase in (
            (cancer_id, "cancer_id", True),
            (drug_id, "drug_id", False),
            (pathway_id, "pathway_id", False),
            (target_gene_id, "target_gene_id", True),
        ):
            clean = None if value is None else _clean(value, label, uppercase=uppercase)
            filters[label] = clean
            if clean is not None:
                clauses.append(f"{label} = ?")
                parameters.append(clean)
        probability = None
        if min_probability is not None:
            probability = float(min_probability)
            if not 0 <= probability <= 1:
                raise DrugMechanismQueryInputError("min_probability must be within 0..1")
            clauses.append(f"{PROBABILITY_COLUMN} >= ?")
            parameters.append(probability)
        filters["min_probability"] = probability
        relation = f"read_parquet({_sql_path(self.artifact_path)})"
        parameters.extend([limit, offset])
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT * FROM {relation}
                WHERE {' AND '.join(clauses)}
                ORDER BY {PROBABILITY_COLUMN} DESC, drug_id, pathway_id, target_gene_id
                LIMIT ? OFFSET ?
                """,
                parameters,
            ).fetchdf()
        finally:
            con.close()
        rows = frame.to_dict("records")
        return {
            "module": "drug",
            "query_kind": "structural_mechanisms",
            "native_target_keys": ["cancer_id", "lncrna_id", "drug_id"],
            "actionability_separate_from_exact_pathway": True,
            "does_not_change_primary_pathway_ranking": True,
            "drug_response_association_probability_not_efficacy_or_direction": True,
            "signed_rho_private_only": True,
            "mechanism_semantics": "STRUCTURAL_HYPOTHESIS_NOT_MODEL_ATTRIBUTION",
            "causal_mechanism_claimed": False,
            "target_contribution_claimed": False,
            "filters": filters,
            "limit": limit,
            "offset": offset,
            "returned_rows": len(rows),
            "rows": rows,
            "provenance": {
                "analysis_version": ANALYSIS_VERSION,
                "training_run_id": self.training_run_id,
                "manifest_path": str(self.manifest_path),
                "manifest_sha256": self.manifest_sha256,
                "historical_results_used": False,
            },
        }

    def download_manifest(self) -> dict[str, Any]:
        declaration = self.manifest["artifact"]
        return {
            "format": "CC_HHGT_V3_2_DRUG_MECHANISM_DOWNLOAD_MANIFEST_V1",
            "module": "drug",
            "result_role": "FRESH_V3_2_STRUCTURAL_MECHANISM_HYPOTHESES",
            "artifact_count": 1,
            "total_bytes": int(self.artifact_path.stat().st_size),
            "artifacts": [
                {
                    "relative_path": _ARTIFACT_NAME,
                    "bytes": int(self.artifact_path.stat().st_size),
                    "rows": int(declaration["rows"]),
                    "sha256": str(declaration["sha256"]),
                    "download_url": (
                        "/v3.2-staging/drug/structural-mechanisms/download"
                    ),
                }
            ],
            "manifest_sha256": self.manifest_sha256,
            "mechanism_semantics": "STRUCTURAL_HYPOTHESIS_NOT_MODEL_ATTRIBUTION",
            "causal_mechanism_claimed": False,
            "target_contribution_claimed": False,
            "does_not_change_primary_pathway_ranking": True,
            "production_deployed": False,
        }

    def resolve_download(self) -> dict[str, Any]:
        declaration = self.manifest["artifact"]
        if not self.artifact_path.is_file() or self.artifact_path.is_symlink():
            raise DrugMechanismQueryAssetError("Drug mechanism artifact became unsafe")
        artifact_bytes = int(self.artifact_path.stat().st_size)
        if artifact_bytes <= 0:
            raise DrugMechanismQueryAssetError("Drug mechanism artifact became empty")
        observed = artifact_sha256(self.artifact_path)
        if observed != declaration["sha256"]:
            raise DrugMechanismQueryAssetError("Drug mechanism artifact SHA drift")
        return {
            "path": str(self.artifact_path),
            "relative_path": _ARTIFACT_NAME,
            "download_name": "V32_DRUG_STRUCTURAL_MECHANISMS.parquet",
            "bytes": artifact_bytes,
            "rows": int(declaration["rows"]),
            "sha256": observed,
            "manifest_sha256": self.manifest_sha256,
        }


__all__ = [
    "DrugMechanismQueryAssetError",
    "DrugMechanismQueryError",
    "DrugMechanismQueryInputError",
    "DrugMechanismReleaseQuery",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
]
