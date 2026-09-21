"""Hash-pinned read-only queries for V3.2 secondary multimodal scores."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .multimodal_fusion import ANALYSIS_VERSION, TARGET_KEYS, artifact_sha256


MAX_QUERY_LIMIT = 1_000
MAX_QUERY_OFFSET = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANCER = re.compile(r"^[A-Z0-9_-]{2,16}$")


class MultimodalFusionQueryError(RuntimeError):
    """Base fusion query error."""


class MultimodalFusionQueryAssetError(MultimodalFusionQueryError):
    """Raised when a binding or score artifact is stale/unsafe."""


class MultimodalFusionQueryInputError(MultimodalFusionQueryError):
    """Raised when a query filter is invalid."""


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise MultimodalFusionQueryAssetError(f"{label} is missing or unsafe: {source}")
    return source


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MultimodalFusionQueryAssetError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise MultimodalFusionQueryAssetError(f"{label} must be an object")
    return payload


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
        raise MultimodalFusionQueryInputError("lncrna_id is invalid")
    return "LNC:" + text


def _canonical_cancer(value: Any | None) -> str | None:
    if value is None:
        return None
    cancer = str(value).strip().upper()
    if not _CANCER.fullmatch(cancer):
        raise MultimodalFusionQueryInputError("cancer_id is invalid")
    return cancer


def _canonical_pathway(value: Any | None) -> str | None:
    if value is None:
        return None
    pathway = str(value).strip()
    if not pathway or len(pathway) > 256 or any(ord(character) < 32 for character in pathway):
        raise MultimodalFusionQueryInputError("pathway_id is invalid")
    return pathway


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise MultimodalFusionQueryInputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise MultimodalFusionQueryInputError(f"offset must be 0..{MAX_QUERY_OFFSET}")
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


class MultimodalFusionReleaseQuery:
    """Validated immutable view of one secondary fusion release."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
    ) -> None:
        source = _safe_file(binding_path, "Multimodal fusion binding")
        expected = str(expected_binding_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise MultimodalFusionQueryAssetError(
                "Multimodal fusion query requires an expected binding SHA256"
            )
        observed = artifact_sha256(source)
        if observed != expected:
            raise MultimodalFusionQueryAssetError(
                f"Multimodal fusion binding SHA mismatch: {observed} != {expected}"
            )
        binding = _load_json(source, "Multimodal fusion binding")
        required = {
            "format": "CC_HHGT_V3_2_MULTIMODAL_FUSION_BINDING_V1",
            "analysis_version": ANALYSIS_VERSION,
            "module_id": "multimodal_fusion",
            "status": "SUCCESS_SECONDARY_FUSION",
            "primary_score_preserved": True,
            "primary_ranking_unchanged": True,
            "adjusted_ranking_is_secondary": True,
            "zero_total_contribution_exact_primary_fallback": True,
            "native_expert_probabilities_public": True,
            "release_ready": False,
            "production_deployed": False,
        }
        for key, value in required.items():
            if binding.get(key) != value:
                raise MultimodalFusionQueryAssetError(
                    f"Multimodal fusion binding has invalid {key}: {binding.get(key)!r}"
                )
        public_rows = binding.get("public_rows")
        if not isinstance(public_rows, int) or public_rows < 1:
            raise MultimodalFusionQueryAssetError("Fusion public row count is invalid")
        native_experts = binding.get("public_native_experts")
        if (
            not isinstance(native_experts, list)
            or not native_experts
            or len(native_experts) != len(set(native_experts))
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[a-z0-9_]{1,64}", value) is None
                for value in native_experts
            )
        ):
            raise MultimodalFusionQueryAssetError(
                "Fusion binding lacks distinct public native experts"
            )
        artifacts = binding.get("artifacts")
        if not isinstance(artifacts, dict) or "secondary_scores" not in artifacts:
            raise MultimodalFusionQueryAssetError("Fusion binding lacks secondary scores")
        paths: dict[str, Path] = {}
        for role, declaration in artifacts.items():
            if not isinstance(declaration, dict):
                raise MultimodalFusionQueryAssetError(f"Fusion artifact is invalid: {role}")
            digest = str(declaration.get("sha256", "")).lower()
            if not _SHA256.fullmatch(digest):
                raise MultimodalFusionQueryAssetError(f"Fusion artifact SHA is invalid: {role}")
            path = _safe_file(declaration.get("path", ""), f"Fusion artifact {role}")
            if artifact_sha256(path) != digest:
                raise MultimodalFusionQueryAssetError(f"Fusion artifact SHA drift: {role}")
            paths[role] = path
        success = _load_json(source.parent / "SUCCESS.json", "Fusion SUCCESS")
        if (
            success.get("status") != binding["status"]
            or success.get("binding") != source.name
            or success.get("binding_sha256") != observed
            or success.get("primary_ranking_unchanged") is not True
            or success.get("release_ready") is not False
            or success.get("production_deployed") is not False
        ):
            raise MultimodalFusionQueryAssetError("Fusion SUCCESS marker is stale")
        self.binding = binding
        self.binding_path = source
        self.binding_sha256 = observed
        self.paths = paths
        self.score_path = paths["secondary_scores"]
        self.training_run_id = str(binding.get("training_run_id", ""))
        self.native_experts = tuple(native_experts)
        self._validate_scores(public_rows)

    @staticmethod
    def _connect():
        return duckdb.connect(database=":memory:")

    def _validate_scores(self, expected_rows: int) -> None:
        relation = _relation(self.score_path)
        connection = self._connect()
        try:
            columns = {
                row[0]
                for row in connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
            }
            required = set(TARGET_KEYS) | {
                "primary_probability",
                "discovery_adjusted_probability",
                "fused_confidence_probability",
                "analysis_version",
                "training_run_id",
                "primary_ranking_unchanged",
                "adjusted_ranking_is_secondary",
                "used_for_primary_release",
                "historical_predictions_used",
                "historical_rankings_used",
            }
            for expert in self.native_experts:
                required.add(f"{expert}_native_probability")
                required.add(f"{expert}_native_available")
            missing = sorted(required - columns)
            forbidden = sorted(
                {"fusion_target", "held_out_proxy_label", "test_membership_label"}.intersection(columns)
            )
            if missing or forbidden:
                raise MultimodalFusionQueryAssetError(
                    f"Fusion public score schema invalid; missing={missing}, forbidden={forbidden}"
                )
            native_checks = ",\n".join(
                (
                    f"count_if(\"{expert}_native_available\" AND "
                    f"(\"{expert}_native_probability\" IS NULL OR "
                    f"NOT isfinite(\"{expert}_native_probability\") OR "
                    f"\"{expert}_native_probability\" < 0 OR "
                    f"\"{expert}_native_probability\" > 1)) + "
                    f"count_if(NOT \"{expert}_native_available\" AND "
                    f"\"{expert}_native_probability\" IS NOT NULL)"
                )
                for expert in self.native_experts
            )
            row = connection.execute(
                f"""
                SELECT
                  count(*) AS rows,
                  count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS keys,
                  count_if(analysis_version != ?) AS wrong_version,
                  count_if(training_run_id != ?) AS wrong_run,
                  count_if(primary_probability IS NULL OR primary_probability < 0 OR primary_probability > 1) AS bad_primary,
                  count_if(discovery_adjusted_probability IS NULL OR discovery_adjusted_probability < 0 OR discovery_adjusted_probability > 1) AS bad_discovery,
                  count_if(fused_confidence_probability IS NULL OR fused_confidence_probability < 0 OR fused_confidence_probability > 1) AS bad_confidence,
                  count_if(primary_ranking_unchanged IS DISTINCT FROM true
                           OR adjusted_ranking_is_secondary IS DISTINCT FROM true
                           OR used_for_primary_release IS DISTINCT FROM false
                           OR historical_predictions_used IS DISTINCT FROM false
                           OR historical_rankings_used IS DISTINCT FROM false) AS bad_policy,
                  {native_checks}
                FROM {relation}
                """,
                [ANALYSIS_VERSION, self.training_run_id],
            ).fetchone()
        finally:
            connection.close()
        expected = (expected_rows, expected_rows, 0, 0, 0, 0, 0, 0) + (
            (0,) * len(self.native_experts)
        )
        if row != expected:
            raise MultimodalFusionQueryAssetError(
                f"Fusion public score semantic validation failed: {row}"
            )

    def capability_status(self) -> dict[str, Any]:
        return {
            "module": "multimodal_fusion",
            "status": self.binding["status"],
            "primary_score_preserved": True,
            "primary_ranking_unchanged": True,
            "adjusted_ranking_is_secondary": True,
            "zero_total_contribution_exact_primary_fallback": True,
            "native_expert_probabilities_public": True,
            "public_native_experts": list(self.native_experts),
            "public_rows": self.binding["public_rows"],
            "release_ready": False,
            "production_deployed": False,
            "binding_sha256": self.binding_sha256,
        }

    def query_scores(
        self,
        *,
        lncrna_id: Any,
        cancer_id: Any | None = None,
        pathway_id: Any | None = None,
        order_by: str = "discovery",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        lnc = _canonical_lnc(lncrna_id)
        cancer = _canonical_cancer(cancer_id)
        pathway = _canonical_pathway(pathway_id)
        if order_by not in {"primary", "discovery", "confidence"}:
            raise MultimodalFusionQueryInputError(
                "order_by must be primary, discovery, or confidence"
            )
        limit, offset = _bounds(limit, offset)
        clauses = ["lncrna_id = ?"]
        parameters: list[Any] = [lnc]
        for column, value in (("cancer_id", cancer), ("pathway_id", pathway)):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        order_column = {
            "primary": "primary_probability",
            "discovery": "discovery_adjusted_probability",
            "confidence": "fused_confidence_probability",
        }[order_by]
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""
                SELECT * FROM {_relation(self.score_path)}
                WHERE {' AND '.join(clauses)}
                ORDER BY {order_column} DESC, cancer_id, pathway_id
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            connection.close()
        records = [
            {str(key): _json_value(value) for key, value in row.items()}
            for row in rows.to_dict("records")
        ]
        return {
            "module": "multimodal_fusion",
            "query_kind": "lncrna_secondary_scores",
            "primary_score_preserved": True,
            "primary_ranking_unchanged": True,
            "adjusted_ranking_is_secondary": True,
            "filters": {
                "lncrna_id": lnc,
                "cancer_id": cancer,
                "pathway_id": pathway,
                "order_by": order_by,
                "limit": limit,
                "offset": offset,
            },
            "returned_rows": len(records),
            "rows": records,
            "provenance": {
                "analysis_version": ANALYSIS_VERSION,
                "training_run_id": self.training_run_id,
                "binding_path": str(self.binding_path),
                "binding_sha256": self.binding_sha256,
                "release_ready": False,
                "production_deployed": False,
            },
        }


__all__ = [
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
    "MultimodalFusionQueryAssetError",
    "MultimodalFusionQueryError",
    "MultimodalFusionQueryInputError",
    "MultimodalFusionReleaseQuery",
]
