"""Hash-pinned read-only queries for the fresh V3.2 continuous activity head."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .continuous_activity import ANALYSIS_VERSION, CHECKPOINT_FORMAT, MODULE_ID


BINDING_FORMAT = "CC_HHGT_V3_2_CONTINUOUS_ACTIVITY_BINDING_V1"
MAX_QUERY_LIMIT = 1_000
MAX_QUERY_OFFSET = 21_303_824
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANCER = re.compile(r"^[A-Z0-9_-]{2,16}$")


class ContinuousActivityQueryError(RuntimeError):
    """Base continuous-activity query error."""


class ContinuousActivityQueryAssetError(ContinuousActivityQueryError):
    """Raised when a binding or artifact is missing, stale, or malformed."""


class ContinuousActivityQueryInputError(ContinuousActivityQueryError):
    """Raised when a query filter is invalid."""


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuousActivityQueryAssetError(f"{label} is missing or invalid") from exc
    if not isinstance(value, dict):
        raise ContinuousActivityQueryAssetError(f"{label} must be a JSON object")
    return value


def _safe_bound_file(root: Path, relative: Any, label: str) -> Path:
    value = Path(str(relative or ""))
    path = value.resolve() if value.is_absolute() else (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ContinuousActivityQueryAssetError(f"{label} escapes the release root") from exc
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise ContinuousActivityQueryAssetError(f"{label} is missing, empty, or unsafe")
    return path


def _identifier(value: Any, label: str, *, upper: bool = False) -> str:
    result = str(value or "").strip()
    if upper:
        result = result.upper()
    if not result or len(result) > 256 or any(ord(character) < 32 for character in result):
        raise ContinuousActivityQueryInputError(f"{label} is invalid")
    return result


def _cancer(value: Any) -> str:
    result = _identifier(value, "cancer_id", upper=True)
    if not _CANCER.fullmatch(result):
        raise ContinuousActivityQueryInputError("cancer_id is invalid")
    return result


def _lncrna(value: Any | None) -> str | None:
    if value is None:
        return None
    result = _identifier(value, "lncrna_id", upper=True)
    result = re.sub(r"^(?:LNC|LNCRNA):", "", result)
    result = re.sub(r"\.\d+$", "", result)
    return "LNC:" + result


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise ContinuousActivityQueryInputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise ContinuousActivityQueryInputError(f"offset must be 0..{MAX_QUERY_OFFSET}")
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


class ContinuousActivityReleaseQuery:
    """Validated immutable view of one V3.2 continuous-activity release."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
        require_formal: bool = True,
    ) -> None:
        source = Path(binding_path).resolve()
        if not source.is_file() or source.is_symlink():
            raise ContinuousActivityQueryAssetError("Continuous binding is missing or unsafe")
        expected = str(expected_binding_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise ContinuousActivityQueryAssetError("Expected binding SHA256 is required")
        observed = artifact_sha256(source)
        if observed != expected:
            raise ContinuousActivityQueryAssetError(
                f"Continuous binding SHA mismatch: {observed} != {expected}"
            )
        binding = _json(source, "Continuous binding")
        required = {
            "binding_format": BINDING_FORMAT,
            "status": "PASS",
            "analysis_version": ANALYSIS_VERSION,
            "model_version": "V3.2",
            "module_id": MODULE_ID,
            "target_level": "pathway_activity",
            "folds": 5,
            "primary_exact_pathway_ranking_changed": False,
            "independent_from_exact_primary": True,
            "unavailable_encoding": "null_with_reason",
            "unavailable_fill_value": None,
        }
        for key, expected_value in required.items():
            if binding.get(key) != expected_value:
                raise ContinuousActivityQueryAssetError(
                    f"Continuous binding has invalid {key}: {binding.get(key)!r}"
                )
        if require_formal and binding.get("cancers") != 33:
            raise ContinuousActivityQueryAssetError("Continuous binding is not the 33-cancer authority")
        training = binding.get("new_training")
        if not isinstance(training, dict) or any(
            (
                training.get("completed") is not True,
                training.get("model_version") != "V3.2",
                training.get("initialization") != "random",
                training.get("new_parameters_from_scratch") is not True,
                training.get("checkpoint_format") != CHECKPOINT_FORMAT,
                training.get("old_checkpoint_loaded") is not False,
                training.get("old_predictions_used_as_features") is not False,
                training.get("legacy_derived_inputs") != [],
            )
        ):
            raise ContinuousActivityQueryAssetError("Continuous new-training lineage is invalid")
        if not _SHA256.fullmatch(str(training.get("checkpoint_sha256", ""))):
            raise ContinuousActivityQueryAssetError("Continuous checkpoint hash is invalid")

        success = _json(source.parent / "SUCCESS.json", "Continuous SUCCESS")
        if any(
            (
                success.get("status") != "PASS",
                success.get("analysis_version") != ANALYSIS_VERSION,
                success.get("model_version") != "V3.2",
                success.get("module_id") != MODULE_ID,
                success.get("output_binding_sha256") != observed,
                success.get("folds") != 5,
                success.get("fresh_checkpoints") != int(binding.get("cancers", 0)) * 5,
                success.get("primary_exact_pathway_ranking_changed") is not False,
            )
        ):
            raise ContinuousActivityQueryAssetError("Continuous SUCCESS marker is stale")
        if require_formal and success.get("cancers") != 33:
            raise ContinuousActivityQueryAssetError("Continuous SUCCESS is not formal authority")

        artifact_declarations = binding.get("artifacts")
        required_artifacts = {
            "v32_continuous_pathway_activity_oof": "oof",
            "v32_continuous_pathway_activity_metrics": "metrics",
            "v32_continuous_pathway_activity_attributions": "attributions",
        }
        if not isinstance(artifact_declarations, dict):
            raise ContinuousActivityQueryAssetError("Continuous artifact declarations are missing")
        paths: dict[str, Path] = {}
        counts: dict[str, int] = {}
        for artifact_id, role in required_artifacts.items():
            declaration = artifact_declarations.get(artifact_id)
            if not isinstance(declaration, dict):
                raise ContinuousActivityQueryAssetError(f"Missing artifact: {artifact_id}")
            path = _safe_bound_file(source.parent, declaration.get("path"), artifact_id)
            digest = str(declaration.get("sha256", ""))
            if not _SHA256.fullmatch(digest) or artifact_sha256(path) != digest:
                raise ContinuousActivityQueryAssetError(f"Artifact SHA drift: {artifact_id}")
            if int(declaration.get("bytes", -1)) != path.stat().st_size:
                raise ContinuousActivityQueryAssetError(f"Artifact byte count drift: {artifact_id}")
            counts[role] = int(declaration.get("rows", -1))
            if counts[role] <= 0:
                raise ContinuousActivityQueryAssetError(f"Artifact is empty: {artifact_id}")
            paths[role] = path

        checkpoint = binding.get("checkpoint_manifest")
        if not isinstance(checkpoint, dict):
            raise ContinuousActivityQueryAssetError("Checkpoint manifest declaration is missing")
        checkpoint_path = _safe_bound_file(
            source.parent, checkpoint.get("path"), "checkpoint manifest"
        )
        if (
            artifact_sha256(checkpoint_path) != checkpoint.get("sha256")
            or checkpoint.get("sha256") != training.get("checkpoint_sha256")
            or int(checkpoint.get("rows", -1)) != int(binding.get("cancers", 0)) * 5
        ):
            raise ContinuousActivityQueryAssetError("Checkpoint manifest is stale")
        paths["checkpoints"] = checkpoint_path

        self.binding_path = source
        self.binding_sha256 = observed
        self.binding = binding
        self.paths = paths
        self.counts = counts
        self._validate_tables(require_formal=require_formal)

    @staticmethod
    def _connect():
        return duckdb.connect(":memory:")

    def _relation(self, role: str) -> str:
        value = self.paths[role].as_posix().replace("'", "''")
        return f"read_parquet('{value}')"

    def _validate_tables(self, *, require_formal: bool) -> None:
        con = self._connect()
        try:
            oof = self._relation("oof")
            metrics = self._relation("metrics")
            attributions = self._relation("attributions")
            checkpoints = self._relation("checkpoints")
            oof_audit = con.execute(
                f"""
                SELECT count(*), count(DISTINCT (cancer_id,sample_id,pathway_id)),
                       count(DISTINCT cancer_id), count(DISTINCT sample_id),
                       count(DISTINCT pathway_id),
                       count_if(NOT isfinite(observed_activity)
                                OR NOT isfinite(predicted_activity)
                                OR NOT isfinite(null_predicted_activity)
                                OR split <> 'outer_test' OR NOT available
                                OR unavailable_reason IS NOT NULL
                                OR analysis_version <> ? OR module_id <> ?)
                FROM {oof}
                """,
                [ANALYSIS_VERSION, MODULE_ID],
            ).fetchone()
            metric_audit = con.execute(
                f"""
                SELECT count(*), count(DISTINCT (cancer_id,patient_fold_id,pathway_id)),
                       count(DISTINCT patient_fold_id),
                       count_if(outer_test_used_for_tuning
                                OR selected_alpha <= 0 OR n_train < 3
                                OR n_validation < 3 OR n_test < 3
                                OR (NOT metric_available AND unavailable_reason IS NULL)
                                OR analysis_version <> ? OR module_id <> ?)
                FROM {metrics}
                """,
                [ANALYSIS_VERSION, MODULE_ID],
            ).fetchone()
            attribution_audit = con.execute(
                f"""
                SELECT count(*), count(DISTINCT (cancer_id,pathway_id,lncrna_id)),
                       min(attribution_rank), max(attribution_rank),
                       count_if(attribution_rank < 1 OR attribution_rank > 25
                                OR selection_stability <= 0 OR selection_stability > 1
                                OR sign_consistency <= 0 OR sign_consistency > 1
                                OR direction NOT IN ('positive','negative')
                                OR analysis_version <> ? OR module_id <> ?)
                FROM {attributions}
                """,
                [ANALYSIS_VERSION, MODULE_ID],
            ).fetchone()
            checkpoint_audit = con.execute(
                f"""
                SELECT count(*), count(DISTINCT (cancer_id,patient_fold_id)),
                       count(DISTINCT checkpoint_sha256), count(DISTINCT cancer_id),
                       count(DISTINCT patient_fold_id),
                       count_if(initial_parameter_sha256=final_parameter_sha256
                                OR n_features < 2 OR n_components < 1 OR n_pathways < 1)
                FROM {checkpoints}
                """
            ).fetchone()
        finally:
            con.close()
        if (
            int(oof_audit[0]) != self.counts["oof"]
            or int(oof_audit[1]) != self.counts["oof"]
            or int(oof_audit[2]) != int(self.binding["cancers"])
            or int(oof_audit[5] or 0)
            or int(metric_audit[0]) != self.counts["metrics"]
            or int(metric_audit[1]) != self.counts["metrics"]
            or int(metric_audit[2]) != 5
            or int(metric_audit[3] or 0)
            or int(attribution_audit[0]) != self.counts["attributions"]
            or int(attribution_audit[1]) != self.counts["attributions"]
            or int(attribution_audit[2]) != 1
            or int(attribution_audit[3]) != 25
            or int(attribution_audit[4] or 0)
            or int(checkpoint_audit[0]) != int(self.binding["cancers"]) * 5
            or int(checkpoint_audit[1]) != int(self.binding["cancers"]) * 5
            or int(checkpoint_audit[2]) != int(self.binding["cancers"]) * 5
            or int(checkpoint_audit[3]) != int(self.binding["cancers"])
            or int(checkpoint_audit[4]) != 5
            or int(checkpoint_audit[5] or 0)
        ):
            raise ContinuousActivityQueryAssetError("Continuous output semantic audit failed")
        if require_formal and (
            int(oof_audit[3]) != 10_432
            or int(oof_audit[4]) != 2_135
            or self.counts != {
                "oof": 21_303_825,
                "metrics": 336_680,
                "attributions": 1_683_400,
            }
        ):
            raise ContinuousActivityQueryAssetError("Continuous formal counts drifted")

    def capability_status(self) -> dict[str, Any]:
        return {
            "module_id": MODULE_ID,
            "status": self.binding["scientific_status"],
            "performance_outcome": self.binding["performance_outcome"],
            "model_version": "V3.2",
            "new_training": True,
            "cancers": int(self.binding["cancers"]),
            "folds": 5,
            "oof_rows": self.counts["oof"],
            "metrics_rows": self.counts["metrics"],
            "attribution_rows": self.counts["attributions"],
            "model_mse": float(self.binding["model_mse"]),
            "null_mse": float(self.binding["null_mse"]),
            "primary_exact_pathway_ranking_changed": False,
            "binding_sha256": self.binding_sha256,
        }

    def query_oof(
        self,
        *,
        cancer_id: Any,
        pathway_id: Any | None = None,
        sample_id: Any | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        cancer = _cancer(cancer_id)
        pathway = None if pathway_id is None else _identifier(pathway_id, "pathway_id")
        sample = None if sample_id is None else _identifier(sample_id, "sample_id")
        limit, offset = _bounds(limit, offset)
        clauses = ["cancer_id = ?"]
        parameters: list[Any] = [cancer]
        if pathway is not None:
            clauses.append("pathway_id = ?")
            parameters.append(pathway)
        if sample is not None:
            clauses.append("sample_id = ?")
            parameters.append(sample)
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT * FROM {self._relation('oof')}
                WHERE {' AND '.join(clauses)}
                ORDER BY pathway_id, sample_id
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return {
            "module_id": MODULE_ID,
            "query_kind": "outer_test_prediction",
            "filters": {"cancer_id": cancer, "pathway_id": pathway, "sample_id": sample},
            "limit": limit,
            "offset": offset,
            "returned_rows": len(frame),
            "results": _records(frame),
            "primary_exact_pathway_ranking_changed": False,
            "binding_sha256": self.binding_sha256,
        }

    def query_metrics(
        self,
        *,
        cancer_id: Any,
        pathway_id: Any | None = None,
        patient_fold_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        cancer = _cancer(cancer_id)
        pathway = None if pathway_id is None else _identifier(pathway_id, "pathway_id")
        if patient_fold_id is not None and (
            isinstance(patient_fold_id, bool) or not 0 <= int(patient_fold_id) < 5
        ):
            raise ContinuousActivityQueryInputError("patient_fold_id must be 0..4")
        limit, offset = _bounds(limit, offset)
        clauses = ["cancer_id = ?"]
        parameters: list[Any] = [cancer]
        if pathway is not None:
            clauses.append("pathway_id = ?")
            parameters.append(pathway)
        if patient_fold_id is not None:
            clauses.append("patient_fold_id = ?")
            parameters.append(int(patient_fold_id))
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT * FROM {self._relation('metrics')}
                WHERE {' AND '.join(clauses)}
                ORDER BY pathway_id, patient_fold_id
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return {
            "module_id": MODULE_ID,
            "query_kind": "outer_test_metrics",
            "filters": {
                "cancer_id": cancer,
                "pathway_id": pathway,
                "patient_fold_id": patient_fold_id,
            },
            "limit": limit,
            "offset": offset,
            "returned_rows": len(frame),
            "results": _records(frame),
            "primary_exact_pathway_ranking_changed": False,
            "binding_sha256": self.binding_sha256,
        }

    def query_attributions(
        self,
        *,
        cancer_id: Any,
        pathway_id: Any | None = None,
        lncrna_id: Any | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        cancer = _cancer(cancer_id)
        pathway = None if pathway_id is None else _identifier(pathway_id, "pathway_id")
        lnc = _lncrna(lncrna_id)
        limit, offset = _bounds(limit, offset)
        clauses = ["cancer_id = ?"]
        parameters: list[Any] = [cancer]
        if pathway is not None:
            clauses.append("pathway_id = ?")
            parameters.append(pathway)
        if lnc is not None:
            clauses.append("lncrna_id = ?")
            parameters.append(lnc)
        con = self._connect()
        try:
            frame = con.execute(
                f"""
                SELECT * FROM {self._relation('attributions')}
                WHERE {' AND '.join(clauses)}
                ORDER BY pathway_id, attribution_rank
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return {
            "module_id": MODULE_ID,
            "query_kind": "lncrna_attribution",
            "filters": {"cancer_id": cancer, "pathway_id": pathway, "lncrna_id": lnc},
            "limit": limit,
            "offset": offset,
            "returned_rows": len(frame),
            "results": _records(frame),
            "primary_exact_pathway_ranking_changed": False,
            "binding_sha256": self.binding_sha256,
        }


__all__ = [
    "BINDING_FORMAT",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
    "ContinuousActivityQueryError",
    "ContinuousActivityQueryAssetError",
    "ContinuousActivityQueryInputError",
    "ContinuousActivityReleaseQuery",
]
