"""Hash-pinned read-only queries for fresh V3.2 perturbation facts.

This is a confidence-evidence fact layer.  It exposes literal
partner-to-exact-pathway mappings but never manufactures a probability and
never changes the discovery or primary rankings.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .experiment_perturbation import (
    ANALYSIS_VERSION,
    ARTIFACT_FILENAMES,
    BINDING_FORMAT,
    FORMAL_ASSAY_DETAIL_SHA256,
    FORMAL_CANDIDATE_UNIVERSE_SHA256,
    FORMAL_EVIDENCE_EVENT_SHA256,
    FORMAL_PATHWAY_MEMBERS_SHA256,
    PENDING_CONFIDENCE_REASON,
    file_sha256,
)


MAX_QUERY_LIMIT = 1_000
MAX_QUERY_OFFSET = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANCER = re.compile(r"^[A-Z0-9_-]{2,16}$")
_EXACT_ROLE = "v32_perturbation_exact_pathway"


class ExperimentPerturbationQueryError(RuntimeError):
    """Base experiment-perturbation query error."""


class ExperimentPerturbationQueryAssetError(ExperimentPerturbationQueryError):
    """Raised when a binding or fact artifact is stale/unsafe."""


class ExperimentPerturbationQueryInputError(ExperimentPerturbationQueryError):
    """Raised when a query filter is invalid."""


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ExperimentPerturbationQueryAssetError(
            f"{label} is missing or unsafe: {source}"
        )
    return source


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentPerturbationQueryAssetError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ExperimentPerturbationQueryAssetError(f"{label} must be a JSON object")
    return value


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _canonical_lnc(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    text = re.sub(r"^(?:LNC|LNCRNA|GENE):", "", text)
    text = re.sub(r"\.\d+$", "", text)
    if not text or len(text) > 128 or any(ord(character) < 32 for character in text):
        raise ExperimentPerturbationQueryInputError("lncrna_id is invalid")
    return "LNC:" + text


def _canonical_cancer(value: Any | None) -> str | None:
    if value is None:
        return None
    cancer = str(value).strip().upper()
    if not _CANCER.fullmatch(cancer):
        raise ExperimentPerturbationQueryInputError("cancer_id is invalid")
    return cancer


def _clean(value: Any | None, label: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > 256 or any(ord(character) < 32 for character in text):
        raise ExperimentPerturbationQueryInputError(f"{label} is invalid")
    return text


def _canonical_partner(value: Any | None) -> str | None:
    partner = _clean(value, "partner_id")
    if partner is None:
        return None
    partner = re.sub(r"^(?:GENE|PROTEIN):", "", partner.upper())
    return re.sub(r"\.\d+$", "", partner)


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise ExperimentPerturbationQueryInputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise ExperimentPerturbationQueryInputError(
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


class ExperimentPerturbationReleaseQuery:
    """Validated immutable view of fresh exact-pathway perturbation facts."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
    ) -> None:
        source = _safe_file(binding_path, "Experiment perturbation binding")
        expected = str(expected_binding_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise ExperimentPerturbationQueryAssetError(
                "Experiment perturbation query requires an expected binding SHA256"
            )
        observed = file_sha256(source)
        if observed != expected:
            raise ExperimentPerturbationQueryAssetError(
                f"Experiment perturbation binding SHA mismatch: {observed} != {expected}"
            )
        binding = _load_json(source, "Experiment perturbation binding")
        required = {
            "binding_format": BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "module_id": "experiment_perturbation",
            "status": "MATERIALISATION_SUCCESS_RELEASE_PENDING",
            "release_ready": False,
        }
        for key, value in required.items():
            if binding.get(key) != value:
                raise ExperimentPerturbationQueryAssetError(
                    f"Experiment perturbation binding has invalid {key}: {binding.get(key)!r}"
                )
        if binding.get("release_blockers") != [PENDING_CONFIDENCE_REASON]:
            raise ExperimentPerturbationQueryAssetError(
                "Experiment perturbation confidence gate is not explicitly pending"
            )
        invariants = binding.get("invariants")
        required_invariants = {
            "functional_perturbation_only": True,
            "partner_literal_exact_member_mapping_only": True,
            "family_broadcast_used": False,
            "pan_cancer_explicit": True,
            "old_results_used": False,
            "independent_probability_generated": False,
            "changes_primary_ranking": False,
        }
        if not isinstance(invariants, dict) or any(
            invariants.get(key) != value for key, value in required_invariants.items()
        ):
            raise ExperimentPerturbationQueryAssetError(
                "Experiment perturbation invariants are invalid"
            )
        if binding.get("input_sha256") != {
            "assay_detail": FORMAL_ASSAY_DETAIL_SHA256,
            "candidate_universe": FORMAL_CANDIDATE_UNIVERSE_SHA256,
            "evidence_event": FORMAL_EVIDENCE_EVENT_SHA256,
            "pathway_members": FORMAL_PATHWAY_MEMBERS_SHA256,
        }:
            raise ExperimentPerturbationQueryAssetError(
                "Experiment perturbation binding is not tied to formal V3.2 inputs"
            )

        manifest_declaration = binding.get("manifest")
        if not isinstance(manifest_declaration, dict):
            raise ExperimentPerturbationQueryAssetError(
                "Experiment perturbation binding lacks its manifest"
            )
        manifest_path = _safe_file(
            manifest_declaration.get("path", ""), "Experiment perturbation manifest"
        )
        try:
            manifest_path.relative_to(source.parent.resolve())
        except ValueError as exc:
            raise ExperimentPerturbationQueryAssetError(
                "Experiment perturbation manifest escaped the binding root"
            ) from exc
        if file_sha256(manifest_path) != manifest_declaration.get("sha256"):
            raise ExperimentPerturbationQueryAssetError(
                "Experiment perturbation manifest SHA drift"
            )
        manifest = _load_json(manifest_path, "Experiment perturbation manifest")
        confidence_policy = manifest.get("confidence_impact_policy")
        if (
            manifest.get("analysis_version") != ANALYSIS_VERSION
            or manifest.get("status") != "MATERIALISATION_SUCCESS"
            or manifest.get("release_ready") is not False
            or not isinstance(confidence_policy, dict)
            or confidence_policy.get("endpoint") != "confidence_only"
            or confidence_policy.get("status") != PENDING_CONFIDENCE_REASON
            or confidence_policy.get("independent_probability_generated") is not False
            or confidence_policy.get("changes_discovery_ranking") is not False
            or confidence_policy.get("changes_primary_ranking") is not False
        ):
            raise ExperimentPerturbationQueryAssetError(
                "Experiment perturbation manifest semantic contract is invalid"
            )

        declarations = binding.get("artifacts")
        if not isinstance(declarations, dict) or set(declarations) != set(ARTIFACT_FILENAMES):
            raise ExperimentPerturbationQueryAssetError(
                "Experiment perturbation artifact map is incomplete"
            )
        paths: dict[str, Path] = {}
        for role, filename in ARTIFACT_FILENAMES.items():
            declaration = declarations.get(role)
            if not isinstance(declaration, dict):
                raise ExperimentPerturbationQueryAssetError(
                    f"Experiment perturbation artifact declaration is invalid: {role}"
                )
            digest = str(declaration.get("sha256", "")).lower()
            rows = declaration.get("rows")
            if (
                not _SHA256.fullmatch(digest)
                or isinstance(rows, bool)
                or not isinstance(rows, int)
                or rows < 0
            ):
                raise ExperimentPerturbationQueryAssetError(
                    f"Experiment perturbation artifact binding is invalid: {role}"
                )
            path = _safe_file(declaration.get("path", ""), f"Perturbation artifact {role}")
            try:
                path.relative_to(source.parent.resolve())
            except ValueError as exc:
                raise ExperimentPerturbationQueryAssetError(
                    f"Experiment perturbation artifact escaped the binding root: {role}"
                ) from exc
            if path.name != filename or file_sha256(path) != digest:
                raise ExperimentPerturbationQueryAssetError(
                    f"Experiment perturbation artifact SHA/path drift: {role}"
                )
            paths[role] = path

        self.binding = binding
        self.binding_path = source
        self.binding_sha256 = observed
        self.manifest = manifest
        self.paths = paths
        self.exact_path = paths[_EXACT_ROLE]
        self._validate_exact_table(int(declarations[_EXACT_ROLE]["rows"]))

    @staticmethod
    def _connect():
        return duckdb.connect(database=":memory:")

    def _validate_exact_table(self, expected_rows: int) -> None:
        relation = f"read_parquet({_sql_path(self.exact_path)})"
        required = {
            "perturbation_event_id",
            "source_event_id",
            "cancer_id",
            "lncrna_id",
            "pathway_id",
            "partner_id",
            "assay_family",
            "assay_detail",
            "assay_detail_available",
            "assay_detail_unavailable_reason",
            "perturbation_methods",
            "perturbation_method_available",
            "readout_assays",
            "readout_assay_available",
            "assay_detail_match_route",
            "assay_detail_evidence_strength",
            "assay_detail_manual_review_required",
            "direction_class",
            "evidence_fact_available",
            "family_broadcast_used",
            "independent_probability_generated",
            "confidence_impact_status",
            "changes_primary_ranking",
            "changes_discovery_ranking",
            "release_ready",
        }
        connection = self._connect()
        try:
            columns = {
                row[0]
                for row in connection.execute(
                    f"DESCRIBE SELECT * FROM {relation}"
                ).fetchall()
            }
            missing = sorted(required - columns)
            forbidden = sorted(
                {
                    "probability",
                    "prediction",
                    "score",
                    "rank",
                    "logit",
                    "checkpoint",
                }.intersection(columns)
            )
            if missing or forbidden:
                raise ExperimentPerturbationQueryAssetError(
                    f"Perturbation exact table schema invalid; missing={missing}, forbidden={forbidden}"
                )
            row = connection.execute(
                f"""
                SELECT count(*),
                       count_if(cancer_id IS NULL OR trim(cancer_id) = ''
                                OR lncrna_id IS NULL OR trim(lncrna_id) = ''
                                OR pathway_id IS NULL OR trim(pathway_id) = ''
                                OR partner_id IS NULL OR trim(partner_id) = ''),
                       count_if(evidence_fact_available IS DISTINCT FROM true
                                OR family_broadcast_used IS DISTINCT FROM false
                                OR independent_probability_generated IS DISTINCT FROM false
                                OR changes_primary_ranking IS DISTINCT FROM false
                                OR changes_discovery_ranking IS DISTINCT FROM false
                                OR release_ready IS DISTINCT FROM false),
                       count_if(confidence_impact_status <> ?)
                       ,count_if(
                           (assay_detail_available IS TRUE
                            AND (assay_detail IS NULL OR trim(assay_detail) = ''))
                           OR (assay_detail_available IS FALSE
                               AND (assay_detail_unavailable_reason IS NULL
                                    OR trim(assay_detail_unavailable_reason) = ''))
                           OR (assay_detail_evidence_strength = 'LOW'
                               AND assay_detail_manual_review_required IS DISTINCT FROM true)
                       )
                FROM {relation}
                """,
                [PENDING_CONFIDENCE_REASON],
            ).fetchone()
        finally:
            connection.close()
        if tuple(map(int, row)) != (expected_rows, 0, 0, 0, 0):
            raise ExperimentPerturbationQueryAssetError(
                f"Perturbation exact table semantic validation failed: {row}"
            )

    def capability_status(self) -> dict[str, Any]:
        return {
            "module_id": "experiment_perturbation",
            "status": self.binding["status"],
            "target_level": "cancer_x_lncrna_x_exact_pathway_x_perturbation_fact",
            "fresh_v32_facts": True,
            "exact_pathway_fact_rows": self.binding["artifacts"][_EXACT_ROLE]["rows"],
            "assay_detail_available_rows": self.manifest.get(
                "assay_detail_contract", {}
            ).get("assay_detail_available_rows", 0),
            "assay_detail_manual_review_required_rows": self.manifest.get(
                "assay_detail_contract", {}
            ).get("manual_review_required_rows", 0),
            "confidence_only": True,
            "confidence_integration_status": PENDING_CONFIDENCE_REASON,
            "independent_probability_generated": False,
            "affects_discovery": False,
            "changes_primary_ranking": False,
            "release_ready": False,
            "production_deployed": False,
            "binding_sha256": self.binding_sha256,
        }

    def query_exact_pathway_facts(
        self,
        *,
        cancer_id: Any | None = None,
        lncrna_id: Any | None = None,
        pathway_id: Any | None = None,
        partner_id: Any | None = None,
        direction_class: Any | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        cancer = _canonical_cancer(cancer_id)
        lnc = _canonical_lnc(lncrna_id)
        pathway = _clean(pathway_id, "pathway_id")
        partner = _canonical_partner(partner_id)
        direction = _clean(direction_class, "direction_class")
        if direction is not None:
            direction = direction.lower()
            if direction not in {"negative", "neutral", "positive"}:
                raise ExperimentPerturbationQueryInputError(
                    "direction_class must be negative, neutral, or positive"
                )
        limit, offset = _bounds(limit, offset)
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("cancer_id", cancer),
            ("lncrna_id", lnc),
            ("pathway_id", pathway),
            ("partner_id", partner),
            ("direction_class", direction),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        where = " AND ".join(clauses) if clauses else "TRUE"
        relation = f"read_parquet({_sql_path(self.exact_path)})"
        connection = self._connect()
        try:
            frame = connection.execute(
                f"""
                SELECT * FROM {relation}
                WHERE {where}
                ORDER BY cancer_id, lncrna_id, pathway_id, perturbation_event_id
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            connection.close()
        rows = [
            {str(key): _json_value(value) for key, value in record.items()}
            for record in frame.to_dict("records")
        ]
        return {
            "module": "experiment_perturbation",
            "query_kind": "fresh_exact_pathway_perturbation_facts",
            "confidence_only": True,
            "confidence_integration_status": PENDING_CONFIDENCE_REASON,
            "independent_probability_generated": False,
            "affects_discovery": False,
            "changes_primary_ranking": False,
            "filters": {
                "cancer_id": cancer,
                "lncrna_id": lnc,
                "pathway_id": pathway,
                "partner_id": partner,
                "direction_class": direction,
                "limit": limit,
                "offset": offset,
            },
            "returned_rows": len(rows),
            "rows": rows,
            "provenance": {
                "analysis_version": ANALYSIS_VERSION,
                "binding_path": str(self.binding_path),
                "binding_sha256": self.binding_sha256,
                "old_results_used": False,
                "family_broadcast_used": False,
                "release_ready": False,
                "production_deployed": False,
            },
        }


__all__ = [
    "ExperimentPerturbationQueryAssetError",
    "ExperimentPerturbationQueryError",
    "ExperimentPerturbationQueryInputError",
    "ExperimentPerturbationReleaseQuery",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
]
