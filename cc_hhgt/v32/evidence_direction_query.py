"""Fail-closed query loader for audited V3.2 Evidence direction probabilities."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import duckdb


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
RELEASE_FORMAT = "CC_HHGT_V3_2_EVIDENCE_DIRECTION_PROBABILITY_BINDING_V1"
RELEASE_STATUS = "SUCCESS_V32_CHECKPOINT_REINFERENCE"
OUTPUT_FORMAT = "CC_HHGT_V3_2_EVIDENCE_DIRECTION_PROBABILITIES_V1"
AUDIT_BINDING_FORMAT = (
    "CC_HHGT_V3_2_EVIDENCE_DIRECTION_INDEPENDENT_AUDIT_BINDING_V1"
)
AUDIT_STATUS = "PASS_INDEPENDENT_V32_EVIDENCE_DIRECTION_AUDIT"
MAX_QUERY_LIMIT = 500
MAX_QUERY_OFFSET = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvidenceDirectionQueryError(RuntimeError):
    """Base error for Evidence direction queries."""


class EvidenceDirectionQueryAssetError(EvidenceDirectionQueryError):
    """Raised when a mounted release or audit binding drifts."""


class EvidenceDirectionQueryInputError(EvidenceDirectionQueryError):
    """Raised for invalid public filters."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceDirectionQueryAssetError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceDirectionQueryAssetError(f"{label} is not an object")
    return value


def _pinned(path: str | Path, expected: str | None, label: str) -> tuple[Path, str]:
    resolved = Path(path).resolve()
    expected = str(expected or "").lower()
    if (
        not _SHA256.fullmatch(expected)
        or resolved.is_symlink()
        or not resolved.is_file()
        or _sha256(resolved) != expected
    ):
        raise EvidenceDirectionQueryAssetError(f"{label} SHA256 mismatch")
    return resolved, expected


def _declared(
    declaration: Mapping[str, Any], parent: Path, label: str
) -> tuple[Path, str]:
    if not isinstance(declaration, Mapping):
        raise EvidenceDirectionQueryAssetError(f"Missing {label} declaration")
    declared = Path(str(declaration.get("path", "")))
    path = declared.resolve() if declared.is_absolute() else (parent / declared).resolve()
    return _pinned(path, str(declaration.get("sha256", "")), label)


def _require(value: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, wanted in expected.items():
        observed = value.get(key)
        if observed != wanted or type(observed) is not type(wanted):
            raise EvidenceDirectionQueryAssetError(
                f"{label} violates {key}: {observed!r}"
            )


def _clean(value: Any, label: str, *, upper: bool = False) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 512:
        raise EvidenceDirectionQueryInputError(f"{label} must be 1..512 characters")
    return text.upper() if upper else text


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_QUERY_LIMIT:
        raise EvidenceDirectionQueryInputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or not 0 <= offset <= MAX_QUERY_OFFSET
    ):
        raise EvidenceDirectionQueryInputError(
            f"offset must be 0..{MAX_QUERY_OFFSET}"
        )
    return limit, offset


def _records(frame: Any) -> list[dict[str, Any]]:
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


class EvidenceDirectionProbabilityQuery:
    """Query only after both the release and independent audit are hash-pinned."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
        audit_binding_path: str | Path,
        expected_audit_binding_sha256: str | None,
    ) -> None:
        release_path, release_sha = _pinned(
            binding_path, expected_binding_sha256, "Evidence direction binding"
        )
        audit_path, audit_sha = _pinned(
            audit_binding_path,
            expected_audit_binding_sha256,
            "Evidence direction independent audit binding",
        )
        release = _json(release_path, "Evidence direction binding")
        audit = _json(audit_path, "Evidence direction independent audit binding")
        _require(
            release,
            {
                "format": RELEASE_FORMAT,
                "status": RELEASE_STATUS,
                "analysis_version": ANALYSIS_VERSION,
                "source_checkpoints_newly_trained_v32": True,
                "old_checkpoint_loaded": False,
                "old_predictions_used_as_features": False,
                "historical_rankings_used": False,
                "primary_ranking_unchanged": True,
                "discovery_ranking_unchanged": True,
                "used_for_fusion": False,
                "production_deployed": False,
                "release_ready": False,
            },
            "Evidence direction binding",
        )
        _require(
            audit,
            {
                "format": AUDIT_BINDING_FORMAT,
                "status": AUDIT_STATUS,
                "analysis_version": ANALYSIS_VERSION,
                "three_class_probabilities_valid": True,
                "typed_nulls_valid": True,
                "source_checkpoints_newly_trained_v32": True,
                "old_checkpoint_loaded": False,
                "historical_prediction_used": False,
                "primary_ranking_unchanged": True,
                "discovery_ranking_unchanged": True,
                "used_for_fusion": False,
                "production_deployed": False,
                "release_ready": False,
            },
            "Evidence direction independent audit binding",
        )
        audited_release_path, audited_release_sha = _declared(
            audit.get("release_binding", {}), audit_path.parent, "audited release binding"
        )
        if audited_release_path != release_path or audited_release_sha != release_sha:
            raise EvidenceDirectionQueryAssetError(
                "Independent audit is not bound to the mounted release"
            )
        artifact_path, artifact_sha = _declared(
            release.get("artifact", {}), release_path.parent, "direction probability artifact"
        )
        audited_artifact_path, audited_artifact_sha = _declared(
            audit.get("artifact", {}), audit_path.parent, "audited probability artifact"
        )
        if (
            audited_artifact_path != artifact_path
            or audited_artifact_sha != artifact_sha
        ):
            raise EvidenceDirectionQueryAssetError(
                "Independent audit artifact differs from release artifact"
            )
        report_path, report_sha = _declared(
            audit.get("report", {}), audit_path.parent, "independent audit report"
        )
        report = _json(report_path, "independent audit report")
        _require(
            report,
            {
                "status": AUDIT_STATUS,
                "analysis_version": ANALYSIS_VERSION,
                "auditor_independent_of_materializer": True,
                "materializer_module_imported": False,
                "production_deployed": False,
                "release_ready": False,
            },
            "independent audit report",
        )
        if report.get("artifact", {}).get("sha256") != artifact_sha:
            raise EvidenceDirectionQueryAssetError("Audit report artifact SHA256 mismatch")
        self.binding_path = release_path
        self.binding_sha256 = release_sha
        self.audit_binding_path = audit_path
        self.audit_binding_sha256 = audit_sha
        self.audit_report_path = report_path
        self.audit_report_sha256 = report_sha
        self.artifact_path = artifact_path
        self.artifact_sha256 = artifact_sha
        self.release = release
        self.audit = audit
        self._validate_schema()

    def _assert_pins(self) -> None:
        pins = (
            (self.binding_path, self.binding_sha256, "release binding"),
            (self.audit_binding_path, self.audit_binding_sha256, "audit binding"),
            (self.audit_report_path, self.audit_report_sha256, "audit report"),
            (self.artifact_path, self.artifact_sha256, "probability artifact"),
        )
        for path, expected, label in pins:
            if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
                raise EvidenceDirectionQueryAssetError(f"Mounted {label} drifted")

    def _validate_schema(self) -> None:
        required = {
            "cancer_id",
            "lncrna_id",
            "pathway_id",
            "published_direction",
            "recovered_direction",
            "direction_negative_probability",
            "direction_neutral_probability",
            "direction_positive_probability",
            "direction_entropy",
            "direction_probability_available",
            "direction_probability_unavailable_reason",
            "published_direction_matches_recovered_argmax",
            "evidence_confidence_available",
            "evidence_fold",
            "event_count",
            "analysis_version",
            "output_format",
            "changes_primary_ranking",
            "changes_discovery_ranking",
            "used_for_fusion",
            "historical_checkpoint_used",
            "historical_prediction_used",
        }
        con = duckdb.connect(":memory:")
        try:
            columns = {
                row[0]
                for row in con.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)",
                    [str(self.artifact_path)],
                ).fetchall()
            }
        finally:
            con.close()
        if missing := sorted(required - columns):
            raise EvidenceDirectionQueryAssetError(
                f"Direction probability artifact lacks columns: {missing}"
            )

    def capability(self) -> dict[str, Any]:
        self._assert_pins()
        return {
            "module": "evidence_direction_probabilities",
            "status": "QUERYABLE_AUDITED_STAGING",
            "probability_classes": ["negative", "neutral", "positive"],
            "counts": dict(self.release.get("counts", {})),
            "typed_nulls": True,
            "source_generation": "CURRENT_V3.2_ONLY",
            "primary_ranking_unchanged": True,
            "discovery_ranking_unchanged": True,
            "used_for_fusion": False,
            "production_deployed": False,
            "release_ready": False,
            "provenance": self._provenance(),
        }

    def _provenance(self) -> dict[str, Any]:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "binding_path": str(self.binding_path),
            "binding_sha256": self.binding_sha256,
            "audit_binding_path": str(self.audit_binding_path),
            "audit_binding_sha256": self.audit_binding_sha256,
            "artifact_sha256": self.artifact_sha256,
            "old_checkpoint_loaded": False,
            "historical_prediction_used": False,
            "primary_ranking_unchanged": True,
            "discovery_ranking_unchanged": True,
            "used_for_fusion": False,
        }

    def query(
        self,
        *,
        cancer_id: Any | None = None,
        lncrna_id: Any | None = None,
        pathway_id: Any | None = None,
        available: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._assert_pins()
        limit, offset = _bounds(limit, offset)
        if available is not None and not isinstance(available, bool):
            raise EvidenceDirectionQueryInputError("available must be true, false or null")
        filters: dict[str, Any] = {
            "cancer_id": None,
            "lncrna_id": None,
            "pathway_id": None,
            "available": available,
        }
        clauses: list[str] = []
        parameters: list[Any] = []
        if cancer_id is not None:
            filters["cancer_id"] = _clean(cancer_id, "cancer_id", upper=True)
            clauses.append("cancer_id = ?")
            parameters.append(filters["cancer_id"])
        if lncrna_id is not None:
            lnc = _clean(lncrna_id, "lncrna_id", upper=True)
            lnc = re.sub(r"^(?:LNC:|LNCRNA:)", "", lnc)
            lnc = re.sub(r"\.\d+$", "", lnc)
            filters["lncrna_id"] = lnc
            clauses.append("lncrna_id = ?")
            parameters.append(lnc)
        if pathway_id is not None:
            filters["pathway_id"] = _clean(pathway_id, "pathway_id")
            clauses.append("pathway_id = ?")
            parameters.append(filters["pathway_id"])
        if available is not None:
            clauses.append("direction_probability_available = ?")
            parameters.append(available)
        where = " AND ".join(clauses) if clauses else "TRUE"
        relation = "read_parquet(?)"
        con = duckdb.connect(":memory:")
        try:
            frame = con.execute(
                f"""
                SELECT * FROM {relation}
                WHERE {where}
                ORDER BY direction_probability_available DESC,
                         direction_entropy ASC NULLS LAST,
                         cancer_id, lncrna_id, pathway_id
                LIMIT ? OFFSET ?
                """,
                [str(self.artifact_path), *parameters, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        rows = _records(frame)
        return {
            "module": "evidence_direction_probabilities",
            "filters": filters,
            "limit": limit,
            "offset": offset,
            "returned_rows": len(rows),
            "rows": rows,
            "typed_nulls": True,
            "probability_classes": ["negative", "neutral", "positive"],
            "provenance": self._provenance(),
        }


__all__ = [
    "EvidenceDirectionProbabilityQuery",
    "EvidenceDirectionQueryAssetError",
    "EvidenceDirectionQueryError",
    "EvidenceDirectionQueryInputError",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
]
