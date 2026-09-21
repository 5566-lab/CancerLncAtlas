"""Independent release audit for the 16-cancer fresh cell-level UCell batch.

The auditor deliberately does not import the UCell materializer or batch
runner.  It re-hashes every declared output, validates Parquet schemas and
statistics, and reconstructs the row-count contract from each cancer's
availability table.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BATCH_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_16C_FULL_BATCH_V1"
AUDIT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_16C_INDEPENDENT_AUDIT_V1"
BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_UCELL_16C_AUDIT_BINDING_V1"
EXPECTED_CANCERS = (
    "ACC", "CHOL", "DLBC", "ESCA", "GBM", "KIRC", "LAML", "LGG",
    "LUSC", "MESO", "PCPG", "READ", "SARC", "SKCM", "THYM", "UCEC",
)
EXPECTED_PATHWAYS = 2_135
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CELL_COLUMNS = {
    "dataset_id", "cancer_id", "cell_id", "patient_id", "cell_type_major",
    "pathway_id", "ucell_available", "unavailable_reason", "ucell_score",
}
_AGGREGATE_COLUMNS = {
    "dataset_id", "cancer_id", "pathway_id", "patient_id",
    "cell_type_major", "cell_count", "ucell_available",
    "unavailable_reason", "ucell_score_mean",
}


class UCell16CIndependentAuditError(RuntimeError):
    """Raised when a full UCell batch does not pass independent validation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UCell16CIndependentAuditError(message)


def _json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"Missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UCell16CIndependentAuditError(f"Invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"Missing or unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    """Reproduce the batch runner's input-lineage directory hash."""

    _require(path.is_dir() and not path.is_symlink(), f"Missing or unsafe tree: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    _require(bool(files), f"Empty artifact tree: {path}")
    digest = hashlib.sha256()
    for item in files:
        _require(not item.is_symlink(), f"Symlink in artifact tree: {item}")
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _declared_sha(value: Any, label: str) -> str:
    digest = str(value).lower()
    _require(_SHA256.fullmatch(digest) is not None, f"Invalid SHA256 for {label}")
    return digest


def _within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise UCell16CIndependentAuditError(f"{label} escapes batch root") from exc
    _require(not path.is_symlink(), f"Unsafe symlink for {label}")
    return resolved


def _column_statistics(path: Path, column: str) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    _require(column in names, f"{path} lacks {column}")
    index = names.index(column)
    null_count = 0
    minimum: Any = None
    maximum: Any = None
    for group_index in range(parquet.metadata.num_row_groups):
        statistics = parquet.metadata.row_group(group_index).column(index).statistics
        _require(statistics is not None, f"{path} lacks statistics for {column}")
        _require(statistics.null_count is not None, f"{path} lacks null count for {column}")
        null_count += int(statistics.null_count)
        if statistics.has_min_max:
            low, high = statistics.min, statistics.max
            if isinstance(low, bytes):
                low = low.decode("utf-8")
            if isinstance(high, bytes):
                high = high.decode("utf-8")
            minimum = low if minimum is None or low < minimum else minimum
            maximum = high if maximum is None or high > maximum else maximum
    return {"null_count": null_count, "min": minimum, "max": maximum}


def _parquet_rows(path: Path, required: set[str], label: str) -> int:
    _require(path.is_file() and not path.is_symlink(), f"Missing or unsafe {label}")
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    _require(required.issubset(names), f"{label} lacks columns: {sorted(required - names)}")
    return int(parquet.metadata.num_rows)


def _audit_manifest(root: Path, expected_sha: str) -> tuple[int, list[Path]]:
    manifest_path = root / "FILE_MANIFEST.parquet"
    _require(_file_sha256(manifest_path) == expected_sha, "FILE_MANIFEST SHA drift")
    manifest = pd.read_parquet(manifest_path)
    required = {"relative_path", "size_bytes", "sha256"}
    _require(required.issubset(manifest.columns), "FILE_MANIFEST schema drift")
    _require(not manifest.relative_path.astype(str).duplicated().any(), "Duplicate manifest path")
    declared_paths: list[Path] = []
    for row in manifest.itertuples(index=False):
        relative = Path(str(row.relative_path))
        _require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe manifest path")
        path = _within(root / relative, root, "manifest artifact")
        _require(path.is_file(), f"Manifest artifact missing: {relative}")
        _require(path.stat().st_size == int(row.size_bytes), f"Manifest size drift: {relative}")
        _require(_file_sha256(path) == _declared_sha(row.sha256, str(relative)), f"Manifest hash drift: {relative}")
        declared_paths.append(path)
    return len(manifest), declared_paths


def _audit_cancer(record: Mapping[str, Any], batch_root: Path) -> dict[str, Any]:
    cancer = str(record.get("cancer_id", "")).upper()
    _require(cancer in EXPECTED_CANCERS, f"Unexpected cancer: {cancer}")
    root = _within(Path(str(record.get("output_root", ""))), batch_root, f"{cancer} output")
    _require(root == batch_root / f"cancer_id={cancer}", f"Non-canonical output root: {cancer}")
    _require(_tree_sha256(root) == _declared_sha(record.get("output_sha256"), cancer), f"Output tree SHA drift: {cancer}")
    success_path = root / "SUCCESS.json"
    _require(_file_sha256(success_path) == _declared_sha(record.get("success_sha256"), f"{cancer} SUCCESS"), f"SUCCESS SHA drift: {cancer}")
    success = _json(success_path, f"{cancer} SUCCESS")
    _require(success.get("status") == "SUCCESS", f"{cancer} is not successful")
    _require(success.get("cancer_id") == cancer, f"{cancer} SUCCESS cancer drift")
    for field, expected in (
        ("full_cell_level_completed", True),
        ("unavailable_values_filled_with_zero_or_half", False),
        ("single_cell_module_complete", False),
        ("release_ready", False),
        ("pseudotime_numeric_output", False),
    ):
        _require(success.get(field) is expected, f"{cancer} invalid {field}")
    _require(
        success.get("pseudotime_status") == "OUT_OF_SCOPE_SEPARATE_DIAGNOSTIC_RUNNER",
        f"{cancer} pseudotime scope drift",
    )

    manifest_sha = _declared_sha(success.get("file_manifest_sha256"), f"{cancer} manifest")
    manifest_entries, declared_paths = _audit_manifest(root, manifest_sha)
    declared_relative = {path.relative_to(root).as_posix() for path in declared_paths}
    parts = sorted((root / "cell_level_ucell").glob("part-*.parquet"))
    _require(bool(parts), f"{cancer} has no cell-level partitions")
    expected_declared = {
        *(path.relative_to(root).as_posix() for path in parts),
        "ucell_pathway_availability.parquet",
        "ucell_donor_celltype.parquet",
        "PRESENT_MISSING_COUNT_AUDIT.json",
    }
    _require(declared_relative == expected_declared, f"{cancer} manifest content drift")

    availability_path = root / "ucell_pathway_availability.parquet"
    availability = pd.read_parquet(
        availability_path, columns=["pathway_id", "ucell_available", "unavailable_reason"]
    )
    _require(len(availability) == EXPECTED_PATHWAYS, f"{cancer} pathway count drift")
    _require(not availability.pathway_id.astype(str).duplicated().any(), f"{cancer} duplicate pathway")
    available = int(availability.ucell_available.astype(bool).sum())
    unavailable = EXPECTED_PATHWAYS - available
    _require(available > 0 and unavailable >= 0, f"{cancer} invalid availability counts")
    _require(
        availability.loc[~availability.ucell_available.astype(bool), "unavailable_reason"].notna().all(),
        f"{cancer} unavailable pathways lack reasons",
    )

    coverage_rows = 0
    score_nulls = 0
    score_min: float | None = None
    score_max: float | None = None
    for part in parts:
        rows = _parquet_rows(part, _CELL_COLUMNS, f"{cancer} cell-level part")
        _require(rows % EXPECTED_PATHWAYS == 0, f"{cancer} part row count is not whole-cell coverage")
        cancer_stats = _column_statistics(part, "cancer_id")
        _require(cancer_stats["min"] == cancer_stats["max"] == cancer, f"{cancer} partition cancer drift")
        scores = _column_statistics(part, "ucell_score")
        score_nulls += int(scores["null_count"])
        if scores["min"] is not None:
            low, high = float(scores["min"]), float(scores["max"])
            _require(math.isfinite(low) and math.isfinite(high), f"{cancer} non-finite score")
            _require(-1e-7 <= low <= high <= 1.0000001, f"{cancer} score outside [0,1]")
            score_min = low if score_min is None else min(score_min, low)
            score_max = high if score_max is None else max(score_max, high)
        coverage_rows += rows
    cells = coverage_rows // EXPECTED_PATHWAYS
    numeric_rows = cells * available
    unavailable_rows = cells * unavailable
    _require(score_nulls == unavailable_rows, f"{cancer} typed-null count drift")

    aggregate_path = root / "ucell_donor_celltype.parquet"
    aggregate_rows = _parquet_rows(aggregate_path, _AGGREGATE_COLUMNS, f"{cancer} aggregate")
    _require(aggregate_rows % EXPECTED_PATHWAYS == 0, f"{cancer} aggregate coverage drift")
    groups = aggregate_rows // EXPECTED_PATHWAYS
    aggregate_nulls = _column_statistics(aggregate_path, "ucell_score_mean")["null_count"]
    _require(aggregate_nulls == groups * unavailable, f"{cancer} aggregate typed-null drift")
    count_audit = _json(root / "PRESENT_MISSING_COUNT_AUDIT.json", f"{cancer} count audit")
    _require(count_audit.get("status") == "PASS", f"{cancer} count audit failed")

    expected_values = {
        "cells": cells,
        "pathways_total_coverage": EXPECTED_PATHWAYS,
        "pathways_available": available,
        "pathways_typed_unavailable": unavailable,
        "cell_level_coverage_rows": coverage_rows,
        "cell_level_numeric_score_rows": numeric_rows,
        "cell_level_typed_unavailable_rows": unavailable_rows,
    }
    for field, expected in expected_values.items():
        _require(int(success.get(field, -1)) == expected, f"{cancer} SUCCESS {field} drift")
    for field, expected in (
        ("cells", cells),
        ("pathways_total", EXPECTED_PATHWAYS),
        ("pathways_available", available),
        ("pathways_typed_unavailable", unavailable),
        ("coverage_rows", coverage_rows),
        ("numeric_rows", numeric_rows),
        ("typed_unavailable_rows", unavailable_rows),
    ):
        _require(int(record.get(field, -1)) == expected, f"{cancer} batch record {field} drift")
    return {
        "cancer_id": cancer,
        "output_root": str(root),
        "output_sha256": record["output_sha256"],
        "success_sha256": record["success_sha256"],
        "manifest_sha256": manifest_sha,
        "manifest_entries": manifest_entries,
        "partitions": len(parts),
        "cells": cells,
        "pathways_total": EXPECTED_PATHWAYS,
        "pathways_available": available,
        "pathways_typed_unavailable": unavailable,
        "coverage_rows": coverage_rows,
        "numeric_rows": numeric_rows,
        "typed_unavailable_rows": unavailable_rows,
        "donor_celltype_groups": groups,
        "score_min": score_min,
        "score_max": score_max,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit_full_ucell_16c(
    batch_binding_path: str | Path,
    *,
    expected_batch_binding_sha256: str,
    output_dir: str | Path,
    auditor_code_path: str | Path,
) -> dict[str, Any]:
    source = Path(batch_binding_path).resolve()
    expected_sha = _declared_sha(expected_batch_binding_sha256, "batch binding")
    _require(_file_sha256(source) == expected_sha, "Full batch binding SHA drift")
    batch = _json(source, "full UCell batch binding")
    _require(batch.get("format") == BATCH_FORMAT, "Full batch format drift")
    _require(batch.get("analysis_version") == ANALYSIS_VERSION, "Analysis version drift")
    _require(
        batch.get("status") == "SUCCESS_16_MISSING_FORMAL_CANCERS_FULL_UCELL",
        "Full batch is not successful",
    )
    _require(batch.get("cancers") == list(EXPECTED_CANCERS), "Full batch cancer order drift")
    _require(batch.get("cancer_count") == 16, "Full batch cancer count drift")
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
        _require(batch.get(field) is expected, f"Full batch invalid {field}")
    records = batch.get("records")
    _require(isinstance(records, list) and len(records) == 16, "Full batch records drift")
    _require(
        [str(row.get("cancer_id")) for row in records] == list(EXPECTED_CANCERS),
        "Full batch record order drift",
    )
    batch_root = source.parent.resolve()
    audited = [_audit_cancer(row, batch_root) for row in records]
    totals = {
        "cells": sum(row["cells"] for row in audited),
        "coverage_rows": sum(row["coverage_rows"] for row in audited),
        "numeric_rows": sum(row["numeric_rows"] for row in audited),
        "typed_unavailable_rows": sum(row["typed_unavailable_rows"] for row in audited),
    }
    for field, expected in totals.items():
        _require(int(batch.get(field, -1)) == expected, f"Full batch total {field} drift")

    destination = Path(output_dir).resolve()
    _require(not destination.exists(), f"Refusing independent-audit output reuse: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    auditor_path = Path(auditor_code_path).resolve()
    auditor_sha = _file_sha256(auditor_path)
    report_path = destination / "INDEPENDENT_AUDIT.json"
    report = {
        "format": AUDIT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS",
        "audit_kind": "INDEPENDENT_16_CANCER_FULL_CELL_LEVEL_UCELL_RELEASE",
        "batch_binding": {"path": str(source), "sha256": expected_sha},
        "auditor": {"path": str(auditor_path), "sha256": auditor_sha},
        "auditor_imports_materializer_or_batch_runner": False,
        "cancers": list(EXPECTED_CANCERS),
        "cancer_count": 16,
        "pathways_total_per_cancer": EXPECTED_PATHWAYS,
        "totals": totals,
        "records": audited,
        "typed_unavailable_preserved": True,
        "unavailable_values_filled_with_zero_or_half": False,
        "historical_outputs_used": False,
        "existing_hnsc_result_reused_or_modified": False,
        "pseudotime_is_separate_diagnostic_scope": True,
        "retraining_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _atomic_json(report_path, report)
    binding_path = destination / "INDEPENDENT_AUDIT_BINDING.json"
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS_HASH_BOUND_16_CANCER_FULL_UCELL",
        "report": {"path": str(report_path), "sha256": _file_sha256(report_path)},
        "batch_binding": {"path": str(source), "sha256": expected_sha},
        "auditor": {"path": str(auditor_path), "sha256": auditor_sha},
        "cancer_count": 16,
        "totals": totals,
        "release_ready": False,
        "production_deployed": False,
    }
    _atomic_json(binding_path, binding)
    success = {
        "status": binding["status"],
        "binding_path": str(binding_path),
        "binding_sha256": _file_sha256(binding_path),
        "report_sha256": binding["report"]["sha256"],
        "cancer_count": 16,
        "totals": totals,
        "release_ready": False,
        "production_deployed": False,
    }
    _atomic_json(destination / "SUCCESS.json", success)
    return success


__all__ = [
    "AUDIT_FORMAT",
    "BATCH_FORMAT",
    "BINDING_FORMAT",
    "EXPECTED_CANCERS",
    "UCell16CIndependentAuditError",
    "audit_full_ucell_16c",
]
