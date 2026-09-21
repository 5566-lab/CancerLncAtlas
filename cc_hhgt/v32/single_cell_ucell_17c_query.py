"""Audited, read-only query binding for fresh V3.2 UCell in 17 cancers.

The HNSC result was produced and audited before the generalized 16-cancer
batch.  This module binds both immutable authorities without copying or
rewriting either result.  UCell remains a pathway-signature result; it must
never be presented as a direct lncRNA score or as pseudotime evidence.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .hnsc_ucell_query import (
    HNSCUCellPilotQuery,
    _duckdb_connect,
)
from .input_lineage import artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BATCH_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_16C_FULL_BATCH_V1"
AUDIT_BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_16C_AUDIT_BINDING_V1"
AUDIT_REPORT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_16C_INDEPENDENT_AUDIT_V1"
BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_17C_QUERY_BINDING_V1"
BINDING_STATUS = "PASS_HASH_BOUND_17_CANCER_FULL_UCELL_QUERY"
PAGE_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_17C_QUERY_PAGE_V1"
DOWNLOAD_MANIFEST_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_17C_DOWNLOAD_MANIFEST_V1"
RESULT_ROLE = "FRESH_V3_2_CELL_LEVEL_EXACT_PATHWAY_UCELL_SECONDARY_FEATURE"

BATCH_CANCERS = (
    "ACC", "CHOL", "DLBC", "ESCA", "GBM", "KIRC", "LAML", "LGG",
    "LUSC", "MESO", "PCPG", "READ", "SARC", "SKCM", "THYM", "UCEC",
)
FORMAL_CANCERS = (
    "ACC", "CHOL", "DLBC", "ESCA", "GBM", "HNSC", "KIRC", "LAML",
    "LGG", "LUSC", "MESO", "PCPG", "READ", "SARC", "SKCM", "THYM",
    "UCEC",
)
EXPECTED_PATHWAYS = 2_135
MAX_QUERY_LIMIT = 500
MAX_QUERY_OFFSET = 2_000_000_000
MAX_TEXT_FILTER_LENGTH = 512
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CELL_PART = re.compile(r"^cell_level_ucell/part-[0-9]{5}\.parquet$")
_DOWNLOAD_EXACT = {
    "ucell_donor_celltype.parquet",
    "ucell_pathway_availability.parquet",
    "PRESENT_MISSING_COUNT_AUDIT.json",
}

_CELL_COLUMNS = {
    "dataset_id", "cancer_id", "cell_id", "patient_id", "cell_type_major",
    "pathway_id", "ucell_available", "unavailable_reason", "ucell_score",
}
_AGGREGATE_COLUMNS = {
    "dataset_id", "cancer_id", "pathway_id", "patient_id", "cell_type_major",
    "cell_count", "ucell_available", "unavailable_reason", "ucell_score_mean",
}
_PATHWAY_COLUMNS = {
    "pathway_id", "signature_gene_count", "present_gene_count",
    "missing_gene_count", "max_rank", "ucell_available", "unavailable_reason",
    "complete_signature_length_used_in_denominator",
    "matrix_missing_member_policy",
    "matrix_missing_member_is_expression_data_imputation", "smoothing_used",
}


class SingleCellUCell17CError(RuntimeError):
    """Base error for the formal 17-cancer UCell query."""


class SingleCellUCell17CAssetError(SingleCellUCell17CError):
    """Raised when a source, hash binding, or schema is invalid."""


class SingleCellUCell17CInputError(SingleCellUCell17CError):
    """Raised for invalid or scientifically inapplicable query filters."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SingleCellUCell17CAssetError(message)


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise SingleCellUCell17CAssetError(f"{label} is missing or unsafe: {source}")
    return source


def _safe_root(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_dir() or source.is_symlink():
        raise SingleCellUCell17CAssetError(f"{label} is missing or unsafe: {source}")
    return source


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleCellUCell17CAssetError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SingleCellUCell17CAssetError(f"{label} must be a JSON object")
    return value


def _declared_sha(value: Any, label: str) -> str:
    digest = str(value or "").lower()
    if _SHA256.fullmatch(digest) is None:
        raise SingleCellUCell17CAssetError(f"Invalid SHA256 for {label}")
    return digest


def _normalise_cancer(value: str) -> str:
    if not isinstance(value, str):
        raise SingleCellUCell17CInputError("cancer_id must be a string")
    cancer = value.strip().upper()
    if cancer not in FORMAL_CANCERS:
        raise SingleCellUCell17CInputError(
            "cancer_id must be one of: " + ", ".join(FORMAL_CANCERS)
        )
    return cancer


def _normalise_filter(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SingleCellUCell17CInputError(f"{label} must be a string")
    text = value.strip()
    if not text or len(text) > MAX_TEXT_FILTER_LENGTH:
        raise SingleCellUCell17CInputError(
            f"{label} must be non-empty and at most {MAX_TEXT_FILTER_LENGTH} characters"
        )
    return text


def _normalise_availability(value: str) -> str:
    if not isinstance(value, str):
        raise SingleCellUCell17CInputError(
            "availability must be ALL, AVAILABLE, or TYPED_UNAVAILABLE"
        )
    normal = value.strip().upper()
    if normal not in {"ALL", "AVAILABLE", "TYPED_UNAVAILABLE"}:
        raise SingleCellUCell17CInputError(
            "availability must be ALL, AVAILABLE, or TYPED_UNAVAILABLE"
        )
    return normal


def _normalise_download_path(value: str) -> str:
    if not isinstance(value, str):
        raise SingleCellUCell17CInputError("relative_path must be a string")
    candidate = PurePosixPath(value.strip().replace("\\", "/"))
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or ":" in candidate.parts[0]
    ):
        raise SingleCellUCell17CInputError("relative_path is unsafe")
    relative = candidate.as_posix()
    if relative not in _DOWNLOAD_EXACT and _CELL_PART.fullmatch(relative) is None:
        raise SingleCellUCell17CInputError(
            "relative_path is not an allowed UCell release artifact"
        )
    return relative


def _validate_page(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_QUERY_LIMIT:
        raise SingleCellUCell17CInputError(
            f"limit must be an integer in 1..{MAX_QUERY_LIMIT}"
        )
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or not 0 <= offset <= MAX_QUERY_OFFSET
    ):
        raise SingleCellUCell17CInputError(
            f"offset must be an integer in 0..{MAX_QUERY_OFFSET}"
        )
    return limit, offset


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
        for row in frame.to_dict(orient="records")
    ]


def _relation(path: Path, *, glob: bool = False) -> str:
    source = path.resolve().as_posix()
    if glob:
        source = source.rstrip("/") + "/part-*.parquet"
    return "read_parquet('" + source.replace("'", "''") + "', union_by_name=true)"


def _read_schema(path: Path, required: set[str], label: str) -> None:
    try:
        observed = set(pq.read_schema(path).names)
    except Exception as exc:
        raise SingleCellUCell17CAssetError(
            f"{label} is unreadable Parquet: {path}"
        ) from exc
    missing = sorted(required - observed)
    _require(not missing, f"{label} lacks columns: {missing}")
    forbidden = sorted(
        name for name in observed if name.casefold() in {"lncrna_id", "lncrna_symbol"}
    )
    _require(not forbidden, f"{label} falsely exposes lncRNA UCell columns: {forbidden}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_and_validate_16c(
    *,
    batch_binding_path: str | Path,
    audit_binding_path: str | Path,
    rehash_trees: bool,
) -> dict[str, Any]:
    batch_path = _safe_file(batch_binding_path, "16-cancer UCell batch binding")
    audit_path = _safe_file(audit_binding_path, "16-cancer UCell audit binding")
    batch_sha = artifact_sha256(batch_path)
    audit_sha = artifact_sha256(audit_path)
    batch = _load_json(batch_path, "16-cancer UCell batch binding")
    audit = _load_json(audit_path, "16-cancer UCell audit binding")
    _require(batch.get("format") == BATCH_FORMAT, "16-cancer batch format drift")
    _require(batch.get("analysis_version") == ANALYSIS_VERSION, "16-cancer analysis drift")
    _require(
        batch.get("status") == "SUCCESS_16_MISSING_FORMAL_CANCERS_FULL_UCELL",
        "16-cancer batch is not successful",
    )
    _require(batch.get("cancers") == list(BATCH_CANCERS), "16-cancer order drift")
    _require(batch.get("cancer_count") == 16, "16-cancer count drift")
    for field, expected in (
        ("existing_hnsc_result_reused_or_modified", False),
        ("historical_sc_trajectory_used", False),
        ("historical_predictions_used", False),
        ("historical_rankings_used", False),
        ("historical_checkpoints_used", False),
        ("retraining_started", False),
        ("release_ready", False),
        ("production_deployed", False),
    ):
        _require(batch.get(field) is expected, f"16-cancer batch invalid {field}")

    _require(audit.get("format") == AUDIT_BINDING_FORMAT, "16-cancer audit format drift")
    _require(audit.get("analysis_version") == ANALYSIS_VERSION, "16-cancer audit analysis drift")
    _require(
        audit.get("status") == "PASS_HASH_BOUND_16_CANCER_FULL_UCELL",
        "16-cancer independent audit did not pass",
    )
    declared_batch = audit.get("batch_binding")
    _require(isinstance(declared_batch, dict), "Audit lacks batch binding declaration")
    _require(
        Path(str(declared_batch.get("path", ""))).resolve() == batch_path
        and _declared_sha(declared_batch.get("sha256"), "audit batch binding") == batch_sha,
        "Audit is not bound to the supplied batch",
    )
    report_decl = audit.get("report")
    _require(isinstance(report_decl, dict), "Audit lacks report declaration")
    report_path = _safe_file(str(report_decl.get("path", "")), "16-cancer audit report")
    report_sha = _declared_sha(report_decl.get("sha256"), "audit report")
    _require(artifact_sha256(report_path) == report_sha, "16-cancer audit report SHA drift")
    report = _load_json(report_path, "16-cancer independent audit report")
    _require(report.get("format") == AUDIT_REPORT_FORMAT, "16-cancer report format drift")
    _require(report.get("status") == "PASS", "16-cancer report is not PASS")
    _require(report.get("cancers") == list(BATCH_CANCERS), "16-cancer report order drift")
    _require(report.get("historical_outputs_used") is False, "Historical UCell output reuse detected")
    _require(
        report.get("unavailable_values_filled_with_zero_or_half") is False,
        "Typed-unavailable UCell values were imputed",
    )

    batch_records = batch.get("records")
    audit_records = report.get("records")
    _require(isinstance(batch_records, list) and len(batch_records) == 16, "Batch records drift")
    _require(isinstance(audit_records, list) and len(audit_records) == 16, "Audit records drift")
    by_audit = {str(row.get("cancer_id")): row for row in audit_records}
    records: list[dict[str, Any]] = []
    for raw in batch_records:
        cancer = str(raw.get("cancer_id", ""))
        _require(cancer in BATCH_CANCERS, f"Unexpected batch cancer: {cancer}")
        root = _safe_root(raw.get("output_root", ""), f"{cancer} UCell root")
        _require(
            root == batch_path.parent / f"cancer_id={cancer}",
            f"Non-canonical UCell root for {cancer}",
        )
        output_sha = _declared_sha(raw.get("output_sha256"), f"{cancer} output")
        audited = by_audit.get(cancer)
        _require(isinstance(audited, dict), f"Audit lacks {cancer}")
        _require(audited.get("output_sha256") == output_sha, f"Audit output hash drift: {cancer}")
        if rehash_trees:
            _require(artifact_sha256(root) == output_sha, f"Runtime output tree drift: {cancer}")
        success = _safe_file(root / "SUCCESS.json", f"{cancer} SUCCESS")
        success_sha = _declared_sha(raw.get("success_sha256"), f"{cancer} SUCCESS")
        _require(artifact_sha256(success) == success_sha, f"SUCCESS SHA drift: {cancer}")
        manifest = _safe_file(root / "FILE_MANIFEST.parquet", f"{cancer} manifest")
        manifest_sha = _declared_sha(audited.get("manifest_sha256"), f"{cancer} manifest")
        _require(artifact_sha256(manifest) == manifest_sha, f"Manifest SHA drift: {cancer}")
        parts = sorted((root / "cell_level_ucell").glob("part-*.parquet"))
        _require(bool(parts), f"{cancer} has no cell-level UCell partitions")
        _read_schema(parts[0], _CELL_COLUMNS, f"{cancer} cell-level UCell")
        _read_schema(root / "ucell_donor_celltype.parquet", _AGGREGATE_COLUMNS, f"{cancer} aggregate UCell")
        _read_schema(root / "ucell_pathway_availability.parquet", _PATHWAY_COLUMNS, f"{cancer} pathway availability")
        record = dict(audited)
        record.update(
            {
                "source_kind": "FRESH_GENERALIZED_16_CANCER_BATCH",
                "output_root": str(root),
                "output_sha256": output_sha,
                "success_sha256": success_sha,
                "manifest_sha256": manifest_sha,
                "pseudotime_status": "OUT_OF_SCOPE_SEPARATE_DIAGNOSTIC_RUNNER",
            }
        )
        records.append(record)
    records.sort(key=lambda row: row["cancer_id"])
    _require([row["cancer_id"] for row in records] == list(BATCH_CANCERS), "Validated record order drift")
    return {
        "batch_path": batch_path,
        "batch_sha256": batch_sha,
        "audit_path": audit_path,
        "audit_sha256": audit_sha,
        "report_path": report_path,
        "report_sha256": report_sha,
        "records": records,
        "totals": dict(report.get("totals", {})),
    }


def build_single_cell_ucell_17c_query_binding(
    *,
    batch_binding_path: str | Path,
    audit_binding_path: str | Path,
    hnsc_binding_path: str | Path,
    hnsc_binding_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate both fresh authorities and create one immutable 17-cancer binding."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise SingleCellUCell17CAssetError(
            f"Refusing to overwrite 17-cancer UCell binding: {destination}"
        )
    validated = _load_and_validate_16c(
        batch_binding_path=batch_binding_path,
        audit_binding_path=audit_binding_path,
        rehash_trees=True,
    )
    hnsc_path = _safe_file(hnsc_binding_path, "HNSC UCell binding")
    hnsc_sha = _declared_sha(hnsc_binding_sha256, "HNSC UCell binding")
    _require(artifact_sha256(hnsc_path) == hnsc_sha, "HNSC UCell binding SHA drift")
    hnsc = HNSCUCellPilotQuery(hnsc_path, expected_binding_sha256=hnsc_sha)
    hnsc_counts = dict(hnsc.binding["validated_counts"])
    hnsc_record = {
        "cancer_id": "HNSC",
        "source_kind": "FRESH_HNSC_AUDITED_PILOT_AUTHORITY",
        "output_root": str(hnsc.result_root),
        "cells": int(hnsc_counts["cells"]),
        "pathways_total": int(hnsc_counts["pathways_total"]),
        "pathways_available": int(hnsc_counts["pathways_available"]),
        "pathways_typed_unavailable": int(hnsc_counts["pathways_typed_unavailable"]),
        "coverage_rows": int(hnsc_counts["cell_level_coverage_rows"]),
        "numeric_rows": int(hnsc_counts["cell_level_numeric_rows"]),
        "typed_unavailable_rows": int(hnsc_counts["cell_level_typed_unavailable_rows"]),
        "donor_celltype_groups": int(hnsc_counts["donor_celltype_groups"]),
        "pseudotime_status": "TYPED_UNAVAILABLE_NO_EXPLICIT_TRAJECTORY_ROOT",
    }
    all_records = [*validated["records"], hnsc_record]
    all_records.sort(key=lambda row: row["cancer_id"])
    combined_totals = {
        "cells": sum(int(row["cells"]) for row in all_records),
        "coverage_rows": sum(int(row["coverage_rows"]) for row in all_records),
        "numeric_rows": sum(int(row["numeric_rows"]) for row in all_records),
        "typed_unavailable_rows": sum(int(row["typed_unavailable_rows"]) for row in all_records),
        "donor_celltype_groups": sum(int(row["donor_celltype_groups"]) for row in all_records),
    }
    destination.mkdir(parents=True, exist_ok=False)
    binding_path = destination / "SINGLE_CELL_UCELL_17C_QUERY_BINDING.json"
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": BINDING_STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_role": RESULT_ROLE,
        "scope": "FORMAL_17_CANCER_CELL_LEVEL_EXACT_PATHWAY_UCELL",
        "cancers": list(FORMAL_CANCERS),
        "cancer_count": 17,
        "pathways_per_cancer": EXPECTED_PATHWAYS,
        "query_axes": ["cancer_id", "pathway_id", "cell_id", "cell_type_major", "patient_id"],
        "lncrna_query_applicable": False,
        "lncrna_query_unavailable_reason": "UCELL_IS_PATHWAY_SIGNATURE_LEVEL_NOT_LNCRNA_LEVEL",
        "typed_unavailable_preserved": True,
        "unavailable_values_filled_with_zero_or_half": False,
        "historical_outputs_used": False,
        "pseudotime_is_separate_diagnostic_scope": True,
        "ucell_capability_release_ready": True,
        "single_cell_module_complete": False,
        "release_ready": False,
        "production_deployed": False,
        "sources": {
            "batch_16c": {
                "path": str(validated["batch_path"]),
                "sha256": validated["batch_sha256"],
            },
            "independent_audit_16c": {
                "path": str(validated["audit_path"]),
                "sha256": validated["audit_sha256"],
            },
            "independent_audit_report_16c": {
                "path": str(validated["report_path"]),
                "sha256": validated["report_sha256"],
            },
            "hnsc_query_binding": {"path": str(hnsc_path), "sha256": hnsc_sha},
        },
        "records": all_records,
        "totals": combined_totals,
    }
    _atomic_json(binding_path, binding)
    binding_sha = artifact_sha256(binding_path)
    success = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_UCELL_17C_QUERY_BINDING_SUCCESS_V1",
        "status": BINDING_STATUS,
        "binding_path": str(binding_path),
        "binding_sha256": binding_sha,
        "cancer_count": 17,
        "totals": combined_totals,
        "ucell_capability_release_ready": True,
        "single_cell_module_complete": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _atomic_json(destination / "SUCCESS.json", success)
    return {
        "binding_path": str(binding_path),
        "binding_sha256": binding_sha,
        "success_path": str(destination / "SUCCESS.json"),
        "binding": binding,
    }


class SingleCellUCell17CQuery:
    """Hash-bound query facade across the HNSC and generalized 16-cancer runs."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
        runtime_rehash_trees: bool = True,
    ) -> None:
        source = _safe_file(binding_path, "17-cancer UCell query binding")
        expected = _declared_sha(expected_binding_sha256, "17-cancer UCell binding")
        observed = artifact_sha256(source)
        _require(observed == expected, "17-cancer UCell binding SHA mismatch")
        binding = _load_json(source, "17-cancer UCell query binding")
        _require(binding.get("format") == BINDING_FORMAT, "17-cancer binding format drift")
        _require(binding.get("analysis_version") == ANALYSIS_VERSION, "17-cancer analysis drift")
        _require(binding.get("status") == BINDING_STATUS, "17-cancer binding is not PASS")
        _require(binding.get("cancers") == list(FORMAL_CANCERS), "17-cancer binding order drift")
        _require(binding.get("cancer_count") == 17, "17-cancer binding count drift")
        for field, expected_value in (
            ("typed_unavailable_preserved", True),
            ("unavailable_values_filled_with_zero_or_half", False),
            ("historical_outputs_used", False),
            ("pseudotime_is_separate_diagnostic_scope", True),
            ("ucell_capability_release_ready", True),
            ("single_cell_module_complete", False),
            ("release_ready", False),
            ("production_deployed", False),
        ):
            _require(binding.get(field) is expected_value, f"17-cancer binding invalid {field}")
        sources = binding.get("sources")
        _require(isinstance(sources, dict), "17-cancer binding lacks sources")
        batch_decl = sources.get("batch_16c")
        audit_decl = sources.get("independent_audit_16c")
        hnsc_decl = sources.get("hnsc_query_binding")
        _require(isinstance(batch_decl, dict), "17-cancer binding lacks 16-cancer batch")
        _require(isinstance(audit_decl, dict), "17-cancer binding lacks independent audit")
        _require(isinstance(hnsc_decl, dict), "17-cancer binding lacks HNSC authority")
        validated = _load_and_validate_16c(
            batch_binding_path=str(batch_decl.get("path", "")),
            audit_binding_path=str(audit_decl.get("path", "")),
            rehash_trees=runtime_rehash_trees,
        )
        _require(
            validated["batch_sha256"] == _declared_sha(batch_decl.get("sha256"), "batch source"),
            "17-cancer batch source SHA drift",
        )
        _require(
            validated["audit_sha256"] == _declared_sha(audit_decl.get("sha256"), "audit source"),
            "17-cancer audit source SHA drift",
        )
        hnsc_path = _safe_file(str(hnsc_decl.get("path", "")), "HNSC source binding")
        hnsc_sha = _declared_sha(hnsc_decl.get("sha256"), "HNSC source binding")
        self.hnsc = HNSCUCellPilotQuery(hnsc_path, expected_binding_sha256=hnsc_sha)
        records = binding.get("records")
        _require(isinstance(records, list) and len(records) == 17, "17-cancer records drift")
        self.records = {str(row.get("cancer_id")): dict(row) for row in records}
        _require(set(self.records) == set(FORMAL_CANCERS), "17-cancer record membership drift")
        validated_by_cancer = {row["cancer_id"]: row for row in validated["records"]}
        for cancer in BATCH_CANCERS:
            _require(
                self.records[cancer].get("output_sha256")
                == validated_by_cancer[cancer].get("output_sha256"),
                f"17-cancer record hash drift: {cancer}",
            )
        self.binding = binding
        self.binding_sha256 = observed

    def capability_status(self, cancer_id: str | None = None) -> dict[str, Any]:
        if cancer_id is None:
            return {
                "result_role": RESULT_ROLE,
                "scope": "FORMAL_17_CANCER_CELL_LEVEL_EXACT_PATHWAY_UCELL",
                "cancers": list(FORMAL_CANCERS),
                "cancer_count": 17,
                "pathways_per_cancer": EXPECTED_PATHWAYS,
                "totals": dict(self.binding["totals"]),
                "query_axes": list(self.binding["query_axes"]),
                "lncrna_query_applicable": False,
                "typed_unavailable_preserved": True,
                "historical_outputs_used": False,
                "ucell_capability_release_ready": True,
                "single_cell_module_complete": False,
                "pseudotime_is_separate_diagnostic_scope": True,
                "production_deployed": False,
            }
        cancer = _normalise_cancer(cancer_id)
        record = dict(self.records[cancer])
        record.update(
            {
                "result_role": RESULT_ROLE,
                "scope": "FORMAL_17_CANCER_CELL_LEVEL_EXACT_PATHWAY_UCELL",
                "lncrna_query_applicable": False,
                "typed_unavailable_preserved": True,
                "ucell_capability_release_ready": True,
                "single_cell_module_complete": False,
                "production_deployed": False,
            }
        )
        return record

    def query_lncrna(self, lncrna_id: str, **_: Any) -> dict[str, Any]:
        _normalise_filter(lncrna_id, "lncrna_id")
        raise SingleCellUCell17CInputError(
            "lncRNA query is not applicable: UCell scores pathway signatures, not lncRNAs"
        )

    def _root(self, cancer: str) -> Path:
        if cancer == "HNSC":
            return self.hnsc.result_root
        return Path(str(self.records[cancer]["output_root"])).resolve()

    def _download_records(self, cancer: str) -> list[dict[str, Any]]:
        root = _safe_root(self._root(cancer), f"{cancer} UCell root")
        manifest_path = _safe_file(
            root / "FILE_MANIFEST.parquet", f"{cancer} FILE_MANIFEST.parquet"
        )
        if cancer == "HNSC":
            expected_manifest_sha = _declared_sha(
                self.hnsc.binding["sources"]["controls"]["FILE_MANIFEST.parquet"][
                    "sha256"
                ],
                "HNSC manifest",
            )
        else:
            expected_manifest_sha = _declared_sha(
                self.records[cancer].get("manifest_sha256"), f"{cancer} manifest"
            )
        _require(
            artifact_sha256(manifest_path) == expected_manifest_sha,
            f"{cancer} manifest SHA drift",
        )
        try:
            manifest = pd.read_parquet(manifest_path)
        except Exception as exc:
            raise SingleCellUCell17CAssetError(
                f"Unreadable {cancer} UCell manifest"
            ) from exc
        required = {"relative_path", "size_bytes", "sha256"}
        _require(required.issubset(manifest.columns), f"{cancer} manifest schema drift")
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in manifest.to_dict(orient="records"):
            try:
                relative = _normalise_download_path(str(raw["relative_path"]))
            except SingleCellUCell17CInputError:
                # Pseudotime artifacts remain a separate diagnostic scope.
                continue
            _require(relative not in seen, f"Duplicate {cancer} download path: {relative}")
            seen.add(relative)
            path = (root / PurePosixPath(relative)).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise SingleCellUCell17CAssetError(
                    f"{cancer} download path escapes result root: {relative}"
                ) from exc
            _require(path.is_file() and not path.is_symlink(), f"Missing {cancer} artifact: {relative}")
            size = int(raw["size_bytes"])
            digest = _declared_sha(raw["sha256"], f"{cancer}/{relative}")
            _require(path.stat().st_size == size, f"{cancer} artifact size drift: {relative}")
            records.append(
                {
                    "relative_path": relative,
                    "bytes": size,
                    "sha256": digest,
                    "download_url": (
                        f"/v3.2-staging/single-cell/ucell/{cancer}/download/{relative}"
                    ),
                }
            )
        records.sort(key=lambda row: str(row["relative_path"]))
        _require(bool(records), f"{cancer} manifest exposes no UCell artifacts")
        _require(
            any(str(row["relative_path"]).startswith("cell_level_ucell/") for row in records),
            f"{cancer} manifest lacks cell-level partitions",
        )
        _require(
            _DOWNLOAD_EXACT.issubset({str(row["relative_path"]) for row in records}),
            f"{cancer} manifest lacks required UCell summary artifacts",
        )
        return records

    def download_manifest(self, *, cancer_id: str) -> dict[str, Any]:
        cancer = _normalise_cancer(cancer_id)
        records = self._download_records(cancer)
        canonical = "\n".join(
            f"{row['relative_path']}\t{row['sha256']}\t{row['bytes']}" for row in records
        ) + "\n"
        return {
            "format": DOWNLOAD_MANIFEST_FORMAT,
            "result_role": RESULT_ROLE,
            "scope": "FORMAL_17_CANCER_CELL_LEVEL_EXACT_PATHWAY_UCELL",
            "cancer_id": cancer,
            "binding_sha256": self.binding_sha256,
            "artifact_count": len(records),
            "total_bytes": sum(int(row["bytes"]) for row in records),
            "tree_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "artifacts": records,
            "typed_unavailable_preserved": True,
            "pseudotime_artifacts_included": False,
            "production_deployed": False,
        }

    def resolve_download(self, *, cancer_id: str, relative_path: str) -> dict[str, Any]:
        cancer = _normalise_cancer(cancer_id)
        relative = _normalise_download_path(relative_path)
        records = {
            str(row["relative_path"]): row for row in self._download_records(cancer)
        }
        if relative not in records:
            raise SingleCellUCell17CInputError(
                f"UCell artifact is not declared for {cancer}: {relative}"
            )
        record = records[relative]
        root = _safe_root(self._root(cancer), f"{cancer} UCell root")
        path = _safe_file(root / PurePosixPath(relative), f"{cancer}/{relative}")
        observed = artifact_sha256(path)
        _require(observed == record["sha256"], f"{cancer} artifact SHA drift: {relative}")
        return {
            **record,
            "path": str(path),
            "cancer_id": cancer,
            "binding_sha256": self.binding_sha256,
            "download_name": f"V32_UCELL_{cancer}_{relative.replace('/', '__')}",
        }

    @staticmethod
    def _availability_clause(availability: str, clauses: list[str]) -> str:
        normal = _normalise_availability(availability)
        if normal == "AVAILABLE":
            clauses.append("ucell_available = TRUE")
        elif normal == "TYPED_UNAVAILABLE":
            clauses.append("ucell_available = FALSE")
        return normal

    def _page(
        self,
        *,
        cancer: str,
        relation: str,
        columns: list[str],
        clauses: list[str],
        parameters: list[Any],
        order_by: str,
        query_level: str,
        filters: Mapping[str, Any],
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        limit, offset = _validate_page(limit, offset)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        connection = _duckdb_connect()
        try:
            total = int(
                connection.execute(
                    f"SELECT count(*)::BIGINT FROM {relation}{where}", parameters
                ).fetchone()[0]
            )
            frame = connection.execute(
                f"SELECT {', '.join(columns)} FROM {relation}{where} "
                f"ORDER BY {order_by} LIMIT ? OFFSET ?",
                [*parameters, limit, offset],
            ).fetchdf()
        except Exception as exc:
            raise SingleCellUCell17CAssetError(
                f"UCell query failed for {cancer}: {exc}"
            ) from exc
        finally:
            connection.close()
        next_offset = offset + len(frame)
        return {
            "format": PAGE_FORMAT,
            "result_role": RESULT_ROLE,
            "scope": "FORMAL_17_CANCER_CELL_LEVEL_EXACT_PATHWAY_UCELL",
            "cancer_id": cancer,
            "query_level": query_level,
            "filters": dict(filters),
            "total_matching_rows": total,
            "returned_rows": len(frame),
            "limit": limit,
            "offset": offset,
            "next_offset": next_offset if next_offset < total else None,
            "rows": _records(frame),
            "typed_unavailable_preserved": True,
            "lncrna_query_applicable": False,
            "ucell_capability_release_ready": True,
            "single_cell_module_complete": False,
            "production_deployed": False,
        }

    def query_cell_scores(
        self,
        *,
        cancer_id: str,
        pathway_id: str | None = None,
        cell_id: str | None = None,
        cell_type_major: str | None = None,
        patient_id: str | None = None,
        availability: str = "ALL",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        cancer = _normalise_cancer(cancer_id)
        values = {
            "pathway_id": _normalise_filter(pathway_id, "pathway_id"),
            "cell_id": _normalise_filter(cell_id, "cell_id"),
            "cell_type_major": _normalise_filter(cell_type_major, "cell_type_major"),
            "patient_id": _normalise_filter(patient_id, "patient_id"),
        }
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in values.items():
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        normal = self._availability_clause(availability, clauses)
        return self._page(
            cancer=cancer,
            relation=_relation(self._root(cancer) / "cell_level_ucell", glob=True),
            columns=[
                "dataset_id", "cancer_id", "cell_id", "patient_id",
                "cell_type_major", "pathway_id", "ucell_available",
                "unavailable_reason", "ucell_score",
            ],
            clauses=clauses,
            parameters=parameters,
            order_by="pathway_id, cell_id, patient_id, cell_type_major",
            query_level="CELL_PATHWAY",
            filters={**values, "availability": normal},
            limit=limit,
            offset=offset,
        )

    def query_donor_celltype_scores(
        self,
        *,
        cancer_id: str,
        pathway_id: str | None = None,
        cell_type_major: str | None = None,
        patient_id: str | None = None,
        availability: str = "ALL",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        cancer = _normalise_cancer(cancer_id)
        values = {
            "pathway_id": _normalise_filter(pathway_id, "pathway_id"),
            "cell_type_major": _normalise_filter(cell_type_major, "cell_type_major"),
            "patient_id": _normalise_filter(patient_id, "patient_id"),
        }
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in values.items():
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        normal = self._availability_clause(availability, clauses)
        return self._page(
            cancer=cancer,
            relation=_relation(self._root(cancer) / "ucell_donor_celltype.parquet"),
            columns=[
                "dataset_id", "cancer_id", "pathway_id", "patient_id",
                "cell_type_major", "cell_count", "ucell_available",
                "unavailable_reason", "ucell_score_mean",
            ],
            clauses=clauses,
            parameters=parameters,
            order_by="pathway_id, patient_id, cell_type_major",
            query_level="DONOR_CELLTYPE_PATHWAY",
            filters={**values, "availability": normal},
            limit=limit,
            offset=offset,
        )

    def query_pathway_availability(
        self,
        *,
        cancer_id: str,
        pathway_id: str | None = None,
        availability: str = "ALL",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        cancer = _normalise_cancer(cancer_id)
        pathway = _normalise_filter(pathway_id, "pathway_id")
        clauses: list[str] = []
        parameters: list[Any] = []
        if pathway is not None:
            clauses.append("pathway_id = ?")
            parameters.append(pathway)
        normal = self._availability_clause(availability, clauses)
        return self._page(
            cancer=cancer,
            relation=_relation(self._root(cancer) / "ucell_pathway_availability.parquet"),
            columns=[
                "pathway_id", "signature_gene_count", "present_gene_count",
                "missing_gene_count", "max_rank", "ucell_available",
                "unavailable_reason", "complete_signature_length_used_in_denominator",
                "matrix_missing_member_policy",
                "matrix_missing_member_is_expression_data_imputation", "smoothing_used",
            ],
            clauses=clauses,
            parameters=parameters,
            order_by="pathway_id",
            query_level="PATHWAY_AVAILABILITY",
            filters={"pathway_id": pathway, "availability": normal},
            limit=limit,
            offset=offset,
        )

    def pseudotime_status(self, cancer_id: str) -> dict[str, Any]:
        cancer = _normalise_cancer(cancer_id)
        return {
            "format": "CC_HHGT_V3_2_SINGLE_CELL_PSEUDOTIME_SCOPE_STATUS_V1",
            "cancer_id": cancer,
            "availability": False,
            "pseudotime_numeric_values": 0,
            "unavailable_reason": self.records[cancer]["pseudotime_status"],
            "separate_diagnostic_scope": True,
            "ucell_results_are_not_pseudotime": True,
            "single_cell_module_complete": False,
            "production_deployed": False,
        }


__all__ = [
    "ANALYSIS_VERSION",
    "BINDING_FORMAT",
    "BINDING_STATUS",
    "BATCH_CANCERS",
    "FORMAL_CANCERS",
    "DOWNLOAD_MANIFEST_FORMAT",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
    "SingleCellUCell17CAssetError",
    "SingleCellUCell17CError",
    "SingleCellUCell17CInputError",
    "SingleCellUCell17CQuery",
    "build_single_cell_ucell_17c_query_binding",
]
