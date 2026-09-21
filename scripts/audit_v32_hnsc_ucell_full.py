#!/usr/bin/env python3
"""Independent read-only audit of the completed HNSC UCell full pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPECTED_CELLS = 5902
EXPECTED_PATHWAYS = 2135
EXPECTED_AVAILABLE = 2120
EXPECTED_UNAVAILABLE = 15
EXPECTED_PARTITIONS = 93
PSEUDOTIME_REASON = "NO_EXPLICIT_TRAJECTORY_ROOT_OR_ORDERED_SOURCE_STATE"


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_audit(root: Path, runner_path: Path) -> dict[str, Any]:
    success_path = root / "SUCCESS.json"
    status_path = root / "STATUS.json"
    handoff_path = root / "CELL_LEVEL_HANDOFF.json"
    lineage_path = root / "LINEAGE.json"
    manifest_path = root / "FILE_MANIFEST.parquet"
    for path in (success_path, status_path, handoff_path, lineage_path, manifest_path):
        require(path.is_file(), f"Missing control artifact: {path}")
    success = load_json(success_path)
    status = load_json(status_path)
    handoff = load_json(handoff_path)
    lineage = load_json(lineage_path)
    require(success.get("status") == "SUCCESS", "SUCCESS marker is not successful")
    require(status.get("status") == "PASS", "STATUS is not PASS")
    require(handoff.get("status") == "PASS", "handoff is not PASS")
    require(lineage.get("status") == "PASS", "lineage is not PASS")

    control_shas = {
        "success_sha256": sha256(success_path),
        "status_sha256": sha256(status_path),
        "handoff_sha256": sha256(handoff_path),
        "lineage_sha256": sha256(lineage_path),
        "file_manifest_sha256": sha256(manifest_path),
    }
    require(success["status_sha256"] == control_shas["status_sha256"], "status SHA drift")
    require(success["handoff_sha256"] == control_shas["handoff_sha256"], "handoff SHA drift")
    require(success["lineage_sha256"] == control_shas["lineage_sha256"], "lineage SHA drift")
    require(
        success["file_manifest_sha256"] == control_shas["file_manifest_sha256"],
        "manifest SHA drift",
    )
    require(
        status["runtime_code"]["runner_sha256"] == sha256(runner_path),
        "runtime runner SHA drift",
    )

    manifest = pd.read_parquet(manifest_path)
    require(not manifest.relative_path.astype(str).duplicated().any(), "manifest duplicates")
    manifest_mismatches: list[str] = []
    for row in manifest.itertuples(index=False):
        relative = str(row.relative_path)
        require(".." not in Path(relative).parts, "manifest path traversal")
        path = root / relative
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(row.size_bytes)
            or sha256(path) != str(row.sha256)
        ):
            manifest_mismatches.append(relative)
    require(not manifest_mismatches, f"manifest mismatch: {manifest_mismatches[:5]}")

    parts = sorted((root / "cell_level_ucell").glob("part-*.parquet"))
    require(len(parts) == EXPECTED_PARTITIONS, "cell-level partition count drift")
    coverage_rows = 0
    numeric_rows = 0
    available_rows = 0
    unavailable_rows = 0
    cell_ids: set[str] = set()
    reason_counts: dict[str, int] = {}
    score_min = np.inf
    score_max = -np.inf
    for part in parts:
        metadata_rows = pq.ParquetFile(part).metadata.num_rows
        frame = pd.read_parquet(
            part,
            columns=[
                "cell_id",
                "pathway_id",
                "ucell_available",
                "unavailable_reason",
                "ucell_score",
            ],
        )
        require(len(frame) == metadata_rows, f"row metadata mismatch: {part.name}")
        available = frame.ucell_available.astype(bool)
        score_present = frame.ucell_score.notna()
        require(score_present.equals(available), f"numeric/null availability mismatch: {part.name}")
        require(
            frame.loc[available, "unavailable_reason"].isna().all(),
            f"available reason is non-null: {part.name}",
        )
        require(
            frame.loc[~available, "unavailable_reason"].notna().all(),
            f"unavailable reason is null: {part.name}",
        )
        per_cell = frame.groupby("cell_id", observed=True).agg(
            rows=("pathway_id", "size"),
            pathways=("pathway_id", "nunique"),
            available=("ucell_available", "sum"),
        )
        require(per_cell.rows.eq(EXPECTED_PATHWAYS).all(), f"coverage per cell drift: {part.name}")
        require(per_cell.pathways.eq(EXPECTED_PATHWAYS).all(), f"pathway uniqueness drift: {part.name}")
        require(per_cell.available.eq(EXPECTED_AVAILABLE).all(), f"available per cell drift: {part.name}")
        overlap = cell_ids & set(per_cell.index.astype(str))
        require(not overlap, f"cell repeated across partitions: {next(iter(overlap), None)}")
        cell_ids.update(per_cell.index.astype(str))
        counts = frame.loc[~available, "unavailable_reason"].astype(str).value_counts()
        for reason, count in counts.items():
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + int(count)
        numeric = frame.loc[available, "ucell_score"].astype(float)
        require(np.isfinite(numeric).all(), f"non-finite numeric score: {part.name}")
        require(numeric.between(0.0, 1.0).all(), f"score outside [0,1]: {part.name}")
        score_min = min(score_min, float(numeric.min()))
        score_max = max(score_max, float(numeric.max()))
        coverage_rows += len(frame)
        numeric_rows += int(score_present.sum())
        available_rows += int(available.sum())
        unavailable_rows += int((~available).sum())

    expected_coverage = EXPECTED_CELLS * EXPECTED_PATHWAYS
    expected_numeric = EXPECTED_CELLS * EXPECTED_AVAILABLE
    expected_unavailable = EXPECTED_CELLS * EXPECTED_UNAVAILABLE
    require(len(cell_ids) == EXPECTED_CELLS, "unique cell count drift")
    require(coverage_rows == expected_coverage, "total coverage row drift")
    require(numeric_rows == expected_numeric == available_rows, "numeric row drift")
    require(unavailable_rows == expected_unavailable, "typed-unavailable row drift")
    expected_reason_rows = {
        "FULL_SIGNATURE_SIZE_AT_OR_EXCEEDS_UCELL_MAX_RANK_DOMAIN": EXPECTED_CELLS * 14,
        "NO_MEMBER_PROTEIN_IN_MATRIX": EXPECTED_CELLS,
    }
    require(reason_counts == expected_reason_rows, "cell-level unavailable reasons drift")

    availability = pd.read_parquet(root / "ucell_pathway_availability.parquet")
    require(len(availability) == EXPECTED_PATHWAYS, "availability sidecar row drift")
    require(int(availability.ucell_available.sum()) == EXPECTED_AVAILABLE, "availability count drift")
    sidecar_reasons = (
        availability.loc[~availability.ucell_available, "unavailable_reason"]
        .value_counts()
        .astype(int)
        .to_dict()
    )
    require(
        sidecar_reasons
        == {
            "FULL_SIGNATURE_SIZE_AT_OR_EXCEEDS_UCELL_MAX_RANK_DOMAIN": 14,
            "NO_MEMBER_PROTEIN_IN_MATRIX": 1,
        },
        "sidecar reason drift",
    )
    require(
        availability.signature_gene_count.eq(
            availability.present_gene_count + availability.missing_gene_count
        ).all(),
        "signature n != present + missing",
    )

    aggregate = pd.read_parquet(root / "ucell_donor_celltype.parquet")
    groups = int(status["donor_celltype_groups"])
    require(len(aggregate) == EXPECTED_PATHWAYS * groups, "aggregate row drift")
    aggregate_available = aggregate.ucell_available.astype(bool)
    require(
        aggregate.ucell_score_mean.notna().equals(aggregate_available),
        "aggregate null availability mismatch",
    )

    pseudo_cells = pd.read_parquet(root / "pseudotime_cell_availability.parquet")
    require(len(pseudo_cells) == EXPECTED_CELLS, "pseudotime cell row drift")
    require(not pseudo_cells.pseudotime_available.any(), "numeric pseudotime marked available")
    require(pseudo_cells.pseudotime_value.isna().all(), "pseudotime cell has numeric values")
    require(
        pseudo_cells.unavailable_reason.eq(PSEUDOTIME_REASON).all(),
        "pseudotime cell reason drift",
    )
    pseudo_pathways = pd.read_parquet(root / "pseudotime_pathway_availability.parquet")
    require(len(pseudo_pathways) == EXPECTED_PATHWAYS, "pseudotime pathway row drift")
    require(not pseudo_pathways.pseudotime_available.any(), "pathway pseudotime available")
    require(pseudo_pathways.pseudotime_effect.isna().all(), "pathway pseudotime has numeric values")
    require(
        pseudo_pathways.unavailable_reason.eq(PSEUDOTIME_REASON).all(),
        "pseudotime pathway reason drift",
    )

    for record in (status, handoff, lineage):
        require(record.get("historical_sc_trajectory_used") is False, "historical trajectory used")
        require(record.get("historical_predictions_used") is False, "historical prediction used")
        require(record.get("historical_rankings_used") is False, "historical ranking used")
        require(record.get("historical_checkpoints_used") is False, "historical checkpoint used")
        require(record.get("pseudotime_numeric_output") is False, "numeric pseudotime claimed")
        require(record.get("single_cell_module_complete") is False, "module falsely complete")
        require(record.get("release_ready") is False, "release falsely ready")

    return {
        "format": "CC_HHGT_V3_2_HNSC_UCELL_FULL_INDEPENDENT_AUDIT_V1",
        "status": "PASS",
        "result_root": str(root),
        "control_shas": control_shas,
        "runtime_runner_path": str(runner_path),
        "runtime_runner_sha256": sha256(runner_path),
        "process_id_recorded": int(success["process_id"]),
        "partitions": len(parts),
        "cells": len(cell_ids),
        "pathways_total": EXPECTED_PATHWAYS,
        "pathways_available": EXPECTED_AVAILABLE,
        "pathways_typed_unavailable": EXPECTED_UNAVAILABLE,
        "cell_level_coverage_rows": coverage_rows,
        "cell_level_numeric_rows": numeric_rows,
        "cell_level_typed_unavailable_rows": unavailable_rows,
        "unavailable_reason_row_counts": reason_counts,
        "score_min": score_min,
        "score_max": score_max,
        "donor_celltype_groups": groups,
        "donor_celltype_rows": len(aggregate),
        "pseudotime_cell_rows": len(pseudo_cells),
        "pseudotime_pathway_rows": len(pseudo_pathways),
        "pseudotime_numeric_values": 0,
        "manifest_entries": len(manifest),
        "manifest_mismatches": 0,
        "unavailable_values_filled_with_zero_or_half": False,
        "historical_outputs_used": False,
        "single_cell_module_complete": False,
        "release_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--runner-path", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_json.resolve()
    if output.exists():
        raise AuditError(f"Audit output reuse is forbidden: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_audit(args.result_root.resolve(), args.runner_path.resolve())
    atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
