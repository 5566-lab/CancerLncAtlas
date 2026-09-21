"""Hash-bound query surface for the V3.2 exact-pathway report manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CANCERLNCATLAS_V32_EXACT_PATHWAY_REPORT_BINDING_V1"
MANIFEST_FORMAT = "CANCERLNCATLAS_V32_EXACT_PATHWAY_REPORT_MANIFEST_V1"
AUDIT_BINDING_FORMAT = (
    "CANCERLNCATLAS_V32_EXACT_PATHWAY_REPORT_INDEPENDENT_AUDIT_BINDING_V1"
)


class ExactPathwayReportAssetError(RuntimeError):
    """Raised when the exact-pathway report authority is incomplete or drifts."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExactPathwayReportAssetError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ExactPathwayReportAssetError(f"{label} must be an object")
    return value


def _declared_file(
    declaration: Mapping[str, Any], *, base: Path, label: str
) -> Path:
    raw = Path(str(declaration.get("path", "")))
    source = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
    if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
        raise ExactPathwayReportAssetError(f"Missing or unsafe {label}: {source}")
    expected = str(declaration.get("sha256", "")).lower()
    if len(expected) != 64 or sha256_file(source) != expected:
        raise ExactPathwayReportAssetError(f"{label} SHA256 drift")
    if declaration.get("bytes") not in (None, source.stat().st_size):
        raise ExactPathwayReportAssetError(f"{label} byte-count drift")
    return source


class ExactPathwayReportQuery:
    """Validate the report binding, independent audit, and every source record."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str,
        audit_binding_path: str | Path,
        expected_audit_binding_sha256: str,
    ) -> None:
        binding_source = Path(binding_path).resolve()
        if sha256_file(binding_source) != str(expected_binding_sha256).lower():
            raise ExactPathwayReportAssetError("Exact report binding SHA256 drift")
        binding = _json(binding_source, "exact report binding")
        if (
            binding.get("format") != BINDING_FORMAT
            or binding.get("analysis_version") != ANALYSIS_VERSION
            or binding.get("status") != "PASS_HASH_BOUND"
            or binding.get("old_checkpoint_loaded") is not False
            or binding.get("old_predictions_used_as_features") is not False
            or binding.get("primary_ranking_authoritative") is not True
        ):
            raise ExactPathwayReportAssetError("Exact report binding semantics are invalid")
        declaration = binding.get("v32_exact_pathway_report_manifest")
        if not isinstance(declaration, Mapping):
            raise ExactPathwayReportAssetError("Exact report binding lacks contract record")
        manifest_source = _declared_file(
            declaration, base=binding_source.parent, label="exact report manifest"
        )
        manifest = _json(manifest_source, "exact report manifest")
        if (
            manifest.get("format") != MANIFEST_FORMAT
            or manifest.get("analysis_version") != ANALYSIS_VERSION
            or manifest.get("status") != "PASS_HASH_BOUND"
            or manifest.get("prediction_rows") != 3_300_000
            or manifest.get("fold_count") != 5
            or manifest.get("trained_from_scratch") is not True
            or manifest.get("old_checkpoint_loaded") is not False
            or manifest.get("old_predictions_used_as_features") is not False
            or manifest.get("old_rankings_used_as_outputs") is not False
        ):
            raise ExactPathwayReportAssetError("Exact report manifest semantics are invalid")
        sources = manifest.get("sources")
        if not isinstance(sources, Mapping) or len(sources) < 6:
            raise ExactPathwayReportAssetError("Exact report manifest lacks source records")
        for source_id, record in sources.items():
            if not isinstance(record, Mapping):
                raise ExactPathwayReportAssetError(f"Invalid source record: {source_id}")
            _declared_file(record, base=manifest_source.parent, label=str(source_id))

        audit_source = Path(audit_binding_path).resolve()
        if sha256_file(audit_source) != str(expected_audit_binding_sha256).lower():
            raise ExactPathwayReportAssetError("Exact report audit binding SHA256 drift")
        audit = _json(audit_source, "exact report independent-audit binding")
        if (
            audit.get("format") != AUDIT_BINDING_FORMAT
            or audit.get("status") != "PASS"
            or audit.get("fail_count") != 0
            or audit.get("release_binding_sha256")
            != str(expected_binding_sha256).lower()
            or audit.get("report_manifest_sha256") != sha256_file(manifest_source)
        ):
            raise ExactPathwayReportAssetError("Exact report independent audit is invalid")
        self.binding = binding
        self.manifest = manifest
        self.audit = audit

    def capability(self) -> dict[str, Any]:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "status": "ENABLED_HASH_BOUND_INDEPENDENTLY_AUDITED",
            "artifact_id": "v32_exact_pathway_report_manifest",
            "training_run_id": self.manifest["training_run_id"],
            "prediction_rows": self.manifest["prediction_rows"],
            "fold_count": self.manifest["fold_count"],
            "trained_from_scratch": True,
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "old_rankings_used_as_outputs": False,
            "primary_ranking_authoritative": True,
            "independent_audit_pass_count": self.audit["pass_count"],
            "production_deployed": False,
        }


__all__ = [
    "ANALYSIS_VERSION",
    "AUDIT_BINDING_FORMAT",
    "BINDING_FORMAT",
    "ExactPathwayReportAssetError",
    "ExactPathwayReportQuery",
    "MANIFEST_FORMAT",
    "sha256_file",
]
