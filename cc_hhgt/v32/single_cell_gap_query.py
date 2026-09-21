"""Fail-closed query surface for the formal V3.2 single-cell gap audit.

The audit is deliberately descriptive.  It reports raw-H5 lncRNA detection
coverage and typed gaps, but it must never be presented as a prediction head
or as a score-changing fusion expert.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_GAP_AUDIT_BINDING_V1"
REPORT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_GAP_AUDIT_V1"
LNCRNA_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_LNCRNA_DETECTION_33C_V1"
AUDIT_BINDING_FORMAT = (
    "CC_HHGT_V3_2_SINGLE_CELL_GAP_INDEPENDENT_AUDIT_BINDING_V1"
)
AUDIT_REPORT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_GAP_INDEPENDENT_AUDIT_V1"
MAX_QUERY_LIMIT = 100
MAX_QUERY_OFFSET = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANCER_ID = re.compile(r"^[A-Z0-9]{2,12}$")


class SingleCellGapQueryAssetError(RuntimeError):
    """Raised when the formal release or independent audit is not hash closed."""


class SingleCellGapQueryInputError(ValueError):
    """Raised for invalid public query parameters."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SingleCellGapQueryAssetError(f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleCellGapQueryAssetError(f"Cannot read {label}") from exc
    return dict(_mapping(value, label))


def _declared_file(declaration: Any, label: str) -> tuple[Path, str]:
    item = _mapping(declaration, label)
    path = Path(str(item.get("path", ""))).resolve()
    digest = str(item.get("sha256", "")).lower()
    if not path.is_file() or path.is_symlink():
        raise SingleCellGapQueryAssetError(f"{label} is missing or unsafe")
    if not _SHA256.fullmatch(digest) or _sha256_file(path) != digest:
        raise SingleCellGapQueryAssetError(f"{label} hash mismatch")
    expected_bytes = item.get("bytes")
    if not isinstance(expected_bytes, int) or path.stat().st_size != expected_bytes:
        raise SingleCellGapQueryAssetError(f"{label} byte count mismatch")
    return path, digest


class SingleCellGapAuditQuery:
    """Validated read-only view of current V3.2 single-cell coverage and gaps."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str,
        audit_binding_path: str | Path,
        expected_audit_binding_sha256: str,
    ) -> None:
        self.binding_path = Path(binding_path).resolve()
        self.audit_binding_path = Path(audit_binding_path).resolve()
        binding_sha = str(expected_binding_sha256).lower()
        audit_sha = str(expected_audit_binding_sha256).lower()
        for path, digest, label in (
            (self.binding_path, binding_sha, "single-cell gap release binding"),
            (self.audit_binding_path, audit_sha, "single-cell gap audit binding"),
        ):
            if not path.is_file() or path.is_symlink():
                raise SingleCellGapQueryAssetError(f"{label} is missing or unsafe")
            if not _SHA256.fullmatch(digest) or _sha256_file(path) != digest:
                raise SingleCellGapQueryAssetError(f"{label} hash mismatch")

        binding = _read_json(self.binding_path, "single-cell gap release binding")
        if binding.get("format") != BINDING_FORMAT:
            raise SingleCellGapQueryAssetError("Single-cell gap binding format mismatch")
        if binding.get("analysis_version") != ANALYSIS_VERSION:
            raise SingleCellGapQueryAssetError("Single-cell gap analysis version mismatch")
        if binding.get("status") != "SUCCESS_HASH_BOUND_TYPED_GAP_AUDIT":
            raise SingleCellGapQueryAssetError("Single-cell gap binding is not successful")
        if binding.get("production_deployed") is not False:
            raise SingleCellGapQueryAssetError("Single-cell gap audit may not claim deployment")
        if binding.get("release_ready") is not False:
            raise SingleCellGapQueryAssetError("Single-cell gap audit may not claim release readiness")
        if binding.get("historical_results_promoted") is not False:
            raise SingleCellGapQueryAssetError("Historical single-cell results were promoted")
        if binding.get("candidate_results_promoted") is not False:
            raise SingleCellGapQueryAssetError("Candidate single-cell results were promoted")

        outputs = _mapping(binding.get("outputs"), "single-cell gap outputs")
        report_path, report_sha = _declared_file(
            outputs.get("SINGLE_CELL_GAP_AUDIT.json"),
            "single-cell gap report",
        )
        detection_path, detection_sha = _declared_file(
            outputs.get("LNCRNA_DETECTION_33C.json"),
            "single-cell lncRNA detection report",
        )
        report = _read_json(report_path, "single-cell gap report")
        detection = _read_json(detection_path, "single-cell lncRNA detection report")
        self._validate_report(report)
        entries = self._validate_detection(detection)

        audit_binding = _read_json(
            self.audit_binding_path, "single-cell independent audit binding"
        )
        if audit_binding.get("format") != AUDIT_BINDING_FORMAT:
            raise SingleCellGapQueryAssetError("Single-cell audit binding format mismatch")
        if audit_binding.get("analysis_version") != ANALYSIS_VERSION:
            raise SingleCellGapQueryAssetError("Single-cell audit analysis version mismatch")
        if audit_binding.get("status") != "PASS_HASH_BOUND":
            raise SingleCellGapQueryAssetError("Single-cell independent audit did not pass")
        if audit_binding.get("production_deployed") is not False:
            raise SingleCellGapQueryAssetError("Single-cell audit may not claim deployment")
        if audit_binding.get("release_binding_sha256") != binding_sha:
            raise SingleCellGapQueryAssetError("Single-cell audit/release SHA mismatch")
        checks = audit_binding.get("checks")
        if not isinstance(checks, int) or checks <= 0:
            raise SingleCellGapQueryAssetError("Single-cell audit check count is invalid")
        if audit_binding.get("failed_checks") != 0:
            raise SingleCellGapQueryAssetError("Single-cell independent audit has failures")
        audit_report_path = Path(str(audit_binding.get("report_path", ""))).resolve()
        audit_report_sha = str(audit_binding.get("report_sha256", "")).lower()
        if (
            not audit_report_path.is_file()
            or audit_report_path.is_symlink()
            or not _SHA256.fullmatch(audit_report_sha)
            or _sha256_file(audit_report_path) != audit_report_sha
        ):
            raise SingleCellGapQueryAssetError("Single-cell independent audit report mismatch")
        audit_report = _read_json(audit_report_path, "single-cell independent audit report")
        if audit_report.get("format") != AUDIT_REPORT_FORMAT:
            raise SingleCellGapQueryAssetError("Single-cell audit report format mismatch")
        if audit_report.get("status") != "PASS":
            raise SingleCellGapQueryAssetError("Single-cell audit report did not pass")
        if audit_report.get("checks") != checks or audit_report.get("failed_checks") != 0:
            raise SingleCellGapQueryAssetError("Single-cell audit check totals mismatch")
        if audit_report.get("release_binding_sha256") != binding_sha:
            raise SingleCellGapQueryAssetError("Single-cell audit report/release SHA mismatch")

        self.binding_sha256 = binding_sha
        self.audit_binding_sha256 = audit_sha
        self.report_sha256 = report_sha
        self.detection_sha256 = detection_sha
        self.audit_report_sha256 = audit_report_sha
        self.binding = binding
        self.report = report
        self.detection = detection
        self.audit = audit_binding
        self._entries = entries

    @staticmethod
    def _validate_report(report: Mapping[str, Any]) -> None:
        if report.get("format") != REPORT_FORMAT:
            raise SingleCellGapQueryAssetError("Single-cell gap report format mismatch")
        if report.get("analysis_version") != ANALYSIS_VERSION:
            raise SingleCellGapQueryAssetError("Single-cell gap report version mismatch")
        required_false = (
            "changes_fusion_score",
            "changes_primary_score",
            "historical_checkpoints_used",
            "historical_predictions_used",
            "historical_rankings_used",
            "numeric_values_invented",
            "production_deployed",
            "release_ready",
        )
        if any(report.get(key) is not False for key in required_false):
            raise SingleCellGapQueryAssetError("Single-cell gap report violates score/lineage policy")
        gaps = report.get("typed_gaps")
        if not isinstance(gaps, list) or len(gaps) != 3:
            raise SingleCellGapQueryAssetError("Single-cell typed gaps are incomplete")
        if any(not isinstance(row, Mapping) for row in gaps):
            raise SingleCellGapQueryAssetError("Single-cell typed gap row is invalid")

    @staticmethod
    def _validate_detection(detection: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        if detection.get("format") != LNCRNA_FORMAT:
            raise SingleCellGapQueryAssetError("Single-cell detection format mismatch")
        if detection.get("analysis_version") != ANALYSIS_VERSION:
            raise SingleCellGapQueryAssetError("Single-cell detection version mismatch")
        entries = detection.get("entries")
        if not isinstance(entries, list) or len(entries) != 33:
            raise SingleCellGapQueryAssetError("Single-cell detection must cover 33 cancers")
        clean: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in entries:
            row = dict(_mapping(raw, "single-cell detection entry"))
            cancer = str(row.get("cancer_id", "")).upper()
            if not _CANCER_ID.fullmatch(cancer) or cancer in seen:
                raise SingleCellGapQueryAssetError("Invalid or duplicate cancer in detection report")
            if not isinstance(row.get("detected_lncrna"), int) or row["detected_lncrna"] < 0:
                raise SingleCellGapQueryAssetError("Invalid detected lncRNA count")
            if not isinstance(row.get("cells"), int) or row["cells"] < 0:
                raise SingleCellGapQueryAssetError("Invalid single-cell count")
            seen.add(cancer)
            clean.append(row)
        counts = _mapping(detection.get("counts"), "single-cell detection counts")
        if counts.get("cancers") != 33 or counts.get("cells") != sum(r["cells"] for r in clean):
            raise SingleCellGapQueryAssetError("Single-cell detection aggregate mismatch")
        return tuple(sorted(clean, key=lambda row: row["cancer_id"]))

    @staticmethod
    def _public_entry(row: Mapping[str, Any]) -> dict[str, Any]:
        # Server paths and file sizes are lineage internals, not public API fields.
        allowed = (
            "cancer_id",
            "dataset_id",
            "cells",
            "detected_lncrna",
            "lncrna_universe",
            "detected_fraction",
            "formal_eligible",
            "coverage_status",
            "limitation",
            "raw_h5_sha256",
            "detected_lncrna_set_sha256",
        )
        return {key: row.get(key) for key in allowed}

    def capability(self) -> dict[str, Any]:
        return {
            "module": "single_cell_gap_audit",
            "analysis_version": ANALYSIS_VERSION,
            "status": self.report["status"],
            "audit_only": True,
            "changes_primary_score": False,
            "changes_fusion_score": False,
            "historical_results_promoted": False,
            "candidate_results_promoted": False,
            "production_deployed": False,
            "release_ready": False,
            "capability_status": dict(self.binding["capability_status"]),
            "lncrna_detection_counts": dict(self.detection["counts"]),
            "ucell": {
                key: self.report["ucell"].get(key)
                for key in (
                    "status",
                    "covered_cancers",
                    "formal_cancers_covered",
                    "formal_cancers_total",
                    "cells",
                    "pathways_available",
                    "pathways_total",
                )
            },
            "pseudotime": {
                key: self.report["pseudotime"].get(key)
                for key in ("status", "numeric_rows", "unavailable_reason")
            },
            "figures": {
                key: self.report["figures"].get(key)
                for key in ("status", "current_v32_figure_files", "unavailable_reason")
            },
            "binding_sha256": self.binding_sha256,
            "report_sha256": self.report_sha256,
            "detection_sha256": self.detection_sha256,
            "independent_audit": {
                "status": self.audit["status"],
                "checks": self.audit["checks"],
                "failed_checks": 0,
                "binding_sha256": self.audit_binding_sha256,
                "report_sha256": self.audit_report_sha256,
            },
        }

    def gaps(self) -> dict[str, Any]:
        return {
            "module": "single_cell_gap_audit",
            "analysis_version": ANALYSIS_VERSION,
            "changes_primary_score": False,
            "changes_fusion_score": False,
            "rows": [dict(row) for row in self.report["typed_gaps"]],
            "count": len(self.report["typed_gaps"]),
        }

    def query_coverage(
        self,
        *,
        cancer_id: str | None = None,
        formal_eligible: bool | None = None,
        limit: int = 33,
        offset: int = 0,
    ) -> dict[str, Any]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_QUERY_LIMIT:
            raise SingleCellGapQueryInputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
        if not isinstance(offset, int) or isinstance(offset, bool) or not 0 <= offset <= MAX_QUERY_OFFSET:
            raise SingleCellGapQueryInputError(f"offset must be 0..{MAX_QUERY_OFFSET}")
        normalized_cancer: str | None = None
        if cancer_id is not None:
            normalized_cancer = cancer_id.strip().upper()
            if not _CANCER_ID.fullmatch(normalized_cancer):
                raise SingleCellGapQueryInputError("Invalid cancer_id")
        rows = [
            row
            for row in self._entries
            if (normalized_cancer is None or row["cancer_id"] == normalized_cancer)
            and (formal_eligible is None or row.get("formal_eligible") is formal_eligible)
        ]
        page = rows[offset : offset + limit]
        return {
            "module": "single_cell_gap_audit",
            "analysis_version": ANALYSIS_VERSION,
            "definition": self.detection["definition"],
            "filters": {
                "cancer_id": normalized_cancer,
                "formal_eligible": formal_eligible,
            },
            "total": len(rows),
            "limit": limit,
            "offset": offset,
            "rows": [self._public_entry(row) for row in page],
            "binding_sha256": self.binding_sha256,
            "detection_sha256": self.detection_sha256,
        }


__all__ = [
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
    "SingleCellGapAuditQuery",
    "SingleCellGapQueryAssetError",
    "SingleCellGapQueryInputError",
]
