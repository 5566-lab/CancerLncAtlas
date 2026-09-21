"""Read-only typed queries for the hash-bound experiment/Evidence bridge."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .experiment_perturbation_bridge import (
    ABSORPTION_STATUS,
    BINDING_FORMAT,
    CONFIDENCE_ROUTE,
    file_sha256,
)


MAX_QUERY_LIMIT = 1_000
MAX_QUERY_OFFSET = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANCER = re.compile(r"^[A-Z0-9_-]{2,16}$")


class ExperimentPerturbationBridgeQueryError(RuntimeError):
    pass


class ExperimentPerturbationBridgeAssetError(ExperimentPerturbationBridgeQueryError):
    pass


class ExperimentPerturbationBridgeInputError(ExperimentPerturbationBridgeQueryError):
    pass


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _canonical_lnc(value: Any | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"^(?:LNC|LNCRNA|GENE):", "", str(value).strip(), flags=re.I)
    text = re.sub(r"\.\d+$", "", text.upper())
    if not text or len(text) > 128:
        raise ExperimentPerturbationBridgeInputError("lncrna_id is invalid")
    return "LNC:" + text


def _clean(value: Any | None, label: str, maximum: int = 512) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > maximum or any(ord(character) < 32 for character in text):
        raise ExperimentPerturbationBridgeInputError(f"{label} is invalid")
    return text


def _canonical_cancer(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not _CANCER.fullmatch(text):
        raise ExperimentPerturbationBridgeInputError("cancer_id is invalid")
    return text


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise ExperimentPerturbationBridgeInputError("limit must be 1..1000")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise ExperimentPerturbationBridgeInputError("offset must be 0..1000000")
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


class ExperimentPerturbationBridgeQuery:
    """Validated exact-key and event-level query facade."""

    def __init__(self, binding_path: str | Path, *, expected_binding_sha256: str) -> None:
        path = Path(binding_path).resolve()
        expected = str(expected_binding_sha256).lower()
        if not _SHA256.fullmatch(expected):
            raise ExperimentPerturbationBridgeAssetError("Expected binding SHA-256 is required")
        if not path.is_file() or path.is_symlink():
            raise ExperimentPerturbationBridgeAssetError(f"Binding is missing or unsafe: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise ExperimentPerturbationBridgeAssetError(
                f"Bridge binding SHA mismatch: {observed} != {expected}"
            )
        try:
            binding = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentPerturbationBridgeAssetError("Bridge binding is invalid JSON") from exc
        required = {
            "binding_format": BINDING_FORMAT,
            "status": "SUCCESS_HASH_BOUND_ABSORPTION_BRIDGE",
            "release_ready": True,
            "production_deployed": False,
            "route_determination": ABSORPTION_STATUS,
            "fully_absorbed_by_evidence_transformer": True,
            "confidence_route": CONFIDENCE_ROUTE,
            "separate_fusion_forbidden": True,
        }
        for key, value in required.items():
            if binding.get(key) != value:
                raise ExperimentPerturbationBridgeAssetError(
                    f"Bridge binding semantic mismatch for {key}"
                )
        invariants = binding.get("invariants", {})
        if not (
            invariants.get("experiment_native_event_identity_retained") is True
            and invariants.get("duplicate_fusion_created") is False
            and invariants.get("changes_primary_ranking") is False
            and invariants.get("changes_discovery_ranking") is False
        ):
            raise ExperimentPerturbationBridgeAssetError("Bridge invariants are invalid")
        paths: dict[str, Path] = {}
        for role in ("lineage_bridge", "exact_query", "source_removal_ablation"):
            declaration = binding.get("artifacts", {}).get(role, {})
            artifact = Path(str(declaration.get("path", ""))).resolve()
            try:
                artifact.relative_to(path.parent)
            except ValueError as exc:
                raise ExperimentPerturbationBridgeAssetError(
                    f"Artifact escaped binding root: {role}"
                ) from exc
            if not artifact.is_file() or artifact.is_symlink():
                raise ExperimentPerturbationBridgeAssetError(f"Artifact is missing: {role}")
            if file_sha256(artifact) != declaration.get("sha256"):
                raise ExperimentPerturbationBridgeAssetError(f"Artifact hash drift: {role}")
            paths[role] = artifact
        self.binding_path = path
        self.binding_sha256 = observed
        self.binding = binding
        self.paths = paths

    @staticmethod
    def _connect():
        return duckdb.connect(database=":memory:")

    def capability_status(self) -> dict[str, Any]:
        return {
            "module_id": "experiment_perturbation_evidence_bridge",
            "experiment_native_event_query": True,
            "experiment_native_event_download_artifact": str(self.paths["lineage_bridge"]),
            "exact_key_query": True,
            "source_removal_attribution_query": True,
            "route_determination": ABSORPTION_STATUS,
            "confidence_route": CONFIDENCE_ROUTE,
            "separate_fusion_forbidden": True,
            "affects_primary": False,
            "affects_discovery": False,
            "production_deployed": False,
            "binding_sha256": self.binding_sha256,
        }

    def _query(
        self,
        role: str,
        *,
        cancer_id: Any | None,
        lncrna_id: Any | None,
        pathway_id: Any | None,
        partner_id: Any | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        cancer = _canonical_cancer(cancer_id)
        lnc = _canonical_lnc(lncrna_id)
        pathway = _clean(pathway_id, "pathway_id")
        partner = _clean(partner_id, "partner_id") if role == "lineage_bridge" else None
        limit, offset = _bounds(limit, offset)
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("cancer_id", cancer),
            ("lncrna_id", lnc),
            ("pathway_id", pathway),
            ("partner_id", partner),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                parameters.append(value)
        where = " AND ".join(clauses) if clauses else "TRUE"
        order_tail = ",experiment_native_event_id" if role == "lineage_bridge" else ""
        connection = self._connect()
        try:
            frame = connection.execute(
                f"""
                SELECT * FROM read_parquet({_sql_path(self.paths[role])})
                WHERE {where}
                ORDER BY cancer_id,lncrna_id,pathway_id{order_tail}
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
            "module": "experiment_perturbation_evidence_bridge",
            "query_kind": (
                "experiment_native_event_lineage" if role == "lineage_bridge" else "experiment_exact_key"
            ),
            "availability": bool(rows),
            "unavailable_reason": "" if rows else "NO_EXPERIMENT_PERTURBATION_EVENT_FOR_FILTERS",
            "route_determination": ABSORPTION_STATUS,
            "confidence_route": CONFIDENCE_ROUTE,
            "separate_fusion_forbidden": True,
            "affects_primary": False,
            "affects_discovery": False,
            "filters": {
                "cancer_id": cancer,
                "lncrna_id": lnc,
                "pathway_id": pathway,
                "partner_id": partner,
                "limit": limit,
                "offset": offset,
            },
            "returned_rows": len(rows),
            "rows": rows,
            "provenance": {
                "binding_path": str(self.binding_path),
                "binding_sha256": self.binding_sha256,
                "production_deployed": False,
            },
        }

    def query_exact_keys(
        self,
        *,
        cancer_id: Any | None = None,
        lncrna_id: Any | None = None,
        pathway_id: Any | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        return self._query(
            "exact_query",
            cancer_id=cancer_id,
            lncrna_id=lncrna_id,
            pathway_id=pathway_id,
            partner_id=None,
            limit=limit,
            offset=offset,
        )

    def query_native_events(
        self,
        *,
        cancer_id: Any | None = None,
        lncrna_id: Any | None = None,
        pathway_id: Any | None = None,
        partner_id: Any | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        return self._query(
            "lineage_bridge",
            cancer_id=cancer_id,
            lncrna_id=lncrna_id,
            pathway_id=pathway_id,
            partner_id=partner_id,
            limit=limit,
            offset=offset,
        )


__all__ = [
    "ExperimentPerturbationBridgeAssetError",
    "ExperimentPerturbationBridgeInputError",
    "ExperimentPerturbationBridgeQuery",
    "ExperimentPerturbationBridgeQueryError",
]
