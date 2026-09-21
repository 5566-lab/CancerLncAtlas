#!/usr/bin/env python3
"""Read-only r6-bound preflight for fresh cell-level UCell/pseudotime work."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from cc_hhgt.v32.input_lineage import artifact_sha256  # noqa: E402
from cc_hhgt.v32.single_cell_partition_builder import (  # noqa: E402
    _decode,
    _stable_gene_id,
)
from cc_hhgt.v32.single_cell_cell_level import (  # noqa: E402
    NO_MEMBER_PROTEIN_IN_MATRIX,
    SIGNATURE_AT_OR_ABOVE_MAX_RANK,
    UCELL_FORMULA,
    UCELL_MISSING_MEMBER_POLICY,
)


PREFLIGHT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_CELL_LEVEL_PREFLIGHT_V2"
FORBIDDEN_SOURCE_TOKENS = (
    "sc_trajectory",
    "prediction",
    "ranking",
    "checkpoint",
)
ROOT_COLUMNS = (
    "trajectory_root",
    "is_trajectory_root",
    "root_cell",
    "root_state",
    "pseudotime_root",
)
ORDER_COLUMNS = (
    "ordered_state",
    "state_order",
    "trajectory_order",
    "timepoint",
    "time_point",
    "collection_day",
    "day",
)
LINEAGE_COLUMNS = (
    "lineage_id",
    "lineage",
    "lineage_support",
    "author_malignant",
)


class CellLevelPreflightError(RuntimeError):
    """Raised when r6 authority or source prerequisites fail closed."""


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_sha(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise CellLevelPreflightError(f"Missing {role}: {path}")
    observed = artifact_sha256(path)
    if observed != str(expected):
        raise CellLevelPreflightError(
            f"{role} SHA drift: {observed} != {expected}"
        )
    return observed


def _assert_fresh_source(path: Path, role: str) -> None:
    lowered = str(path).lower().replace("\\", "/")
    matched = [token for token in FORBIDDEN_SOURCE_TOKENS if token in lowered]
    if matched:
        raise CellLevelPreflightError(
            f"{role} path contains forbidden historical-result token(s): {matched}"
        )


def _explicit_root_count(frame: pd.DataFrame, column: str) -> int:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return int(values.fillna(False).sum())
    lowered = values.fillna("").astype(str).str.strip().str.lower()
    if set(lowered.unique()).issubset({"", "true", "false", "1", "0", "yes", "no"}):
        return int(lowered.isin({"true", "1", "yes"}).sum())
    # A categorical root-state column is explicit if it contains at least one
    # non-empty declaration and at least one non-root/other value.
    nonempty = lowered.ne("")
    return int(nonempty.sum()) if nonempty.any() and lowered.nunique() > 1 else 0


def run_preflight(
    *,
    r6_root: str | Path,
    cancer_id: str,
    expected_run_status_sha256: str,
    expected_handoff_sha256: str,
    expected_manifest_sha256: str,
    max_rank: int = 1500,
    chunk_size: int = 64,
) -> dict[str, Any]:
    cancer = str(cancer_id).upper().strip()
    root = Path(r6_root).resolve()
    run_path = root / "RUN_STATUS.json"
    handoff_path = root / "TRAINING_HANDOFF.json"
    manifest_path = root / "dataset_manifest_33c.parquet"
    run_sha = _validate_sha(
        run_path, expected_run_status_sha256, "r6 RUN_STATUS"
    )
    handoff_sha = _validate_sha(
        handoff_path, expected_handoff_sha256, "r6 TRAINING_HANDOFF"
    )
    manifest_sha = _validate_sha(
        manifest_path, expected_manifest_sha256, "r6 dataset manifest"
    )
    run = _load_json(run_path)
    handoff = _load_json(handoff_path)
    manifest = pd.read_parquet(manifest_path)
    if not run.get("blocked_fresh_direct_id_counts_typed_unavailable"):
        raise CellLevelPreflightError("r6 blocked-count semantic correction is absent")
    if not handoff.get("blocked_fresh_direct_id_counts_typed_unavailable"):
        raise CellLevelPreflightError("r6 handoff does not bind the semantic correction")
    if cancer not in run.get("per_cancer", {}):
        raise CellLevelPreflightError(f"Cancer is absent from r6 authority: {cancer}")
    record = run["per_cancer"][cancer]
    manifest_row = manifest.loc[manifest.cancer_id.astype(str).eq(cancer)]
    if len(manifest_row) != 1:
        raise CellLevelPreflightError("r6 manifest lacks a unique cancer row")
    if record.get("formal_eligible") is not True or not bool(
        manifest_row.formal_eligible.iloc[0]
    ):
        raise CellLevelPreflightError(f"Cancer is not r6 formal-eligible: {cancer}")

    h5_path = Path(record["h5_path"]).resolve()
    metadata_path = Path(record["metadata_path"]).resolve()
    annotation_path = Path(record["annotation_path"]).resolve()
    membership_path = Path(handoff["exact_membership_path"]).resolve()
    for path, role in (
        (h5_path, "cell-level H5"),
        (metadata_path, "cell metadata"),
        (annotation_path, "GENCODE annotation"),
        (membership_path, "exact membership"),
    ):
        _assert_fresh_source(path, role)
    h5_sha = _validate_sha(h5_path, record["h5_sha256"], "H5")
    metadata_sha = _validate_sha(
        metadata_path, record["metadata_sha256"], "cell metadata"
    )
    annotation_sha = _validate_sha(
        annotation_path, record["annotation_sha256"], "GENCODE annotation"
    )
    membership_sha = _validate_sha(
        membership_path, handoff["exact_membership_sha256"], "exact membership"
    )

    metadata = pd.read_parquet(metadata_path)
    required_meta = {"cell_id", "patient_id", "cell_type_major", "dataset_id", "cancer_id"}
    if missing := sorted(required_meta - set(metadata.columns)):
        raise CellLevelPreflightError(f"Cell metadata lacks columns: {missing}")
    if metadata.cell_id.astype(str).duplicated().any():
        raise CellLevelPreflightError("Cell metadata IDs are duplicated")
    if set(metadata.cancer_id.astype(str).str.upper()) != {cancer}:
        raise CellLevelPreflightError("Metadata cancer ID disagrees with authority")

    annotation = pd.read_parquet(annotation_path)
    required_annotation = {"gene_id", "gene_symbol", "gene_class", "unique_symbol"}
    if missing := sorted(required_annotation - set(annotation.columns)):
        raise CellLevelPreflightError(f"GENCODE cache lacks columns: {missing}")
    annotation["gene_id"] = annotation.gene_id.map(_stable_gene_id)
    by_id = annotation.set_index("gene_id")
    unique_symbol_rows = annotation.loc[annotation.unique_symbol.astype(bool)]
    by_symbol = unique_symbol_rows.set_index("gene_symbol")
    membership = pd.read_parquet(membership_path)
    if set(membership.columns) < {"pathway_id", "gene_id"}:
        raise CellLevelPreflightError("Exact membership lacks pathway/gene keys")
    membership["gene_id"] = membership.gene_id.map(_stable_gene_id)
    if membership.pathway_id.astype(str).nunique() != 2135:
        raise CellLevelPreflightError("Exact membership is not the 2,135-pathway authority")

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise CellLevelPreflightError("h5py is required for H5 preflight") from exc

    with h5py.File(h5_path, "r") as handle:
        group = handle["matrix"] if "matrix" in handle else handle
        required_h5 = {"data", "indices", "indptr", "shape", "barcodes", "features"}
        if not required_h5.issubset(group.keys()):
            raise CellLevelPreflightError("H5 is not a 10x sparse matrix")
        shape = tuple(int(value) for value in np.asarray(group["shape"][:]))
        barcodes = _decode(group["barcodes"][:])
        ids = _decode(group["features"]["id"][:])
        names = _decode(group["features"]["name"][:])
        if shape != (len(ids), len(barcodes)):
            raise CellLevelPreflightError("H5 dimensions disagree with features/barcodes")
        indexer = pd.Index(metadata.cell_id.astype(str)).get_indexer(barcodes)
        if (indexer < 0).any() or len(barcodes) != len(metadata):
            raise CellLevelPreflightError("H5 barcodes and metadata are not one-to-one")
        metadata = metadata.iloc[indexer].reset_index(drop=True)

        protein_rows: list[int] = []
        protein_ids: list[str] = []
        for feature_index, (raw_id, raw_name) in enumerate(zip(ids, names, strict=True)):
            direct = _stable_gene_id(raw_id)
            if direct in by_id.index:
                annotation_row = by_id.loc[direct]
                stable = direct
            elif raw_name in by_symbol.index:
                annotation_row = by_symbol.loc[raw_name]
                stable = str(
                    unique_symbol_rows.loc[
                        unique_symbol_rows.gene_symbol.eq(raw_name), "gene_id"
                    ].iloc[0]
                )
            else:
                continue
            if str(annotation_row.gene_class) == "protein_coding":
                protein_rows.append(feature_index)
                protein_ids.append(stable)
        unique_proteins = set(protein_ids)
        duplicated_protein_ids = len(protein_ids) - len(unique_proteins)

        indptr = np.asarray(group["indptr"][:], dtype=np.int64)
        indices = np.asarray(group["indices"][:], dtype=np.int64)
        protein_feature_mask = np.zeros(shape[0], dtype=bool)
        protein_feature_mask[np.asarray(protein_rows, dtype=np.int64)] = True
        protein_detected = np.empty(shape[1], dtype=np.int32)
        for cell_index in range(shape[1]):
            start, stop = indptr[cell_index : cell_index + 2]
            protein_detected[cell_index] = int(
                protein_feature_mask[indices[start:stop]].sum()
            )
        data = group["data"]
        data_min = np.inf
        data_max = -np.inf
        nonfinite = 0
        negative = 0
        block = 1_000_000
        for start in range(0, len(data), block):
            values = np.asarray(data[start : start + block])
            if values.size:
                finite = np.isfinite(values)
                nonfinite += int((~finite).sum())
                negative += int((values < 0).sum())
                if finite.any():
                    data_min = min(data_min, float(values[finite].min()))
                    data_max = max(data_max, float(values[finite].max()))
        matrix_dtype = str(data.dtype)
        stored_nnz = int(len(data))

    available_counts = (
        membership.assign(in_matrix=membership.gene_id.isin(unique_proteins))
        .groupby("pathway_id", observed=True)
        .agg(
            signature_genes=("gene_id", "nunique"),
            genes_in_matrix=("in_matrix", "sum"),
        )
    )
    available_counts["genes_in_matrix"] = available_counts.genes_in_matrix.astype(int)
    available_counts["genes_missing_from_matrix"] = (
        available_counts.signature_genes - available_counts.genes_in_matrix
    ).astype(int)
    available_counts["ucell_available"] = (
        available_counts.genes_in_matrix.gt(0)
        & available_counts.signature_genes.lt(int(max_rank))
    )
    available_counts["unavailable_reason"] = np.select(
        [
            available_counts.genes_in_matrix.eq(0),
            available_counts.signature_genes.ge(int(max_rank)),
        ],
        [
            NO_MEMBER_PROTEIN_IN_MATRIX,
            SIGNATURE_AT_OR_ABOVE_MAX_RANK,
        ],
        default=None,
    )

    root_columns = [column for column in ROOT_COLUMNS if column in metadata]
    root_counts = {
        column: _explicit_root_count(metadata, column) for column in root_columns
    }
    order_columns = [column for column in ORDER_COLUMNS if column in metadata]
    usable_order_columns = []
    for column in order_columns:
        values = metadata[column].dropna()
        if len(values) and values.astype(str).str.strip().ne("").any() and values.nunique() >= 3:
            usable_order_columns.append(column)
    lineage_columns = [column for column in LINEAGE_COLUMNS if column in metadata]
    explicit_root_available = any(value > 0 for value in root_counts.values())
    ordered_source_state_available = bool(usable_order_columns)
    pseudotime_available = bool(
        (explicit_root_available or ordered_source_state_available)
        and lineage_columns
        and metadata.patient_id.astype(str).nunique() >= 5
    )
    pseudotime_reason = (
        None
        if pseudotime_available
        else "NO_EXPLICIT_TRAJECTORY_ROOT_OR_ORDERED_SOURCE_STATE"
    )

    cell_count = int(len(metadata))
    ucell_pathways = int(available_counts.ucell_available.sum())
    unavailable_counts = (
        available_counts.loc[~available_counts.ucell_available, "unavailable_reason"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    dense_chunk_bytes = int(len(unique_proteins) * int(chunk_size) * 8)
    result = {
        "format": PREFLIGHT_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "cancer_id": cancer,
        "dataset_id": str(metadata.dataset_id.iloc[0]),
        "r6_authority": {
            "root": str(root),
            "run_status_path": str(run_path),
            "run_status_sha256": run_sha,
            "training_handoff_path": str(handoff_path),
            "training_handoff_sha256": handoff_sha,
            "dataset_manifest_path": str(manifest_path),
            "dataset_manifest_sha256": manifest_sha,
            "formal_eligible": True,
        },
        "inputs": {
            "h5_path": str(h5_path),
            "h5_sha256": h5_sha,
            "metadata_path": str(metadata_path),
            "metadata_sha256": metadata_sha,
            "annotation_path": str(annotation_path),
            "annotation_sha256": annotation_sha,
            "exact_membership_path": str(membership_path),
            "exact_membership_sha256": membership_sha,
            "exact_membership_rows": int(len(membership)),
            "exact_pathways": int(membership.pathway_id.nunique()),
        },
        "h5_audit": {
            "shape_features_by_cells": list(shape),
            "stored_nnz": stored_nnz,
            "matrix_dtype": matrix_dtype,
            "data_min": None if not np.isfinite(data_min) else data_min,
            "data_max": None if not np.isfinite(data_max) else data_max,
            "nonfinite_values": nonfinite,
            "negative_values": negative,
            "protein_feature_rows": int(len(protein_rows)),
            "unique_protein_gene_ids": int(len(unique_proteins)),
            "duplicated_protein_feature_ids": int(duplicated_protein_ids),
            "median_detected_protein_genes_per_cell": float(
                np.median(protein_detected)
            ),
            "min_detected_protein_genes_per_cell": int(protein_detected.min()),
            "max_detected_protein_genes_per_cell": int(protein_detected.max()),
            "barcode_metadata_one_to_one": True,
        },
        "metadata_audit": {
            "cells": cell_count,
            "donors": int(metadata.patient_id.astype(str).nunique()),
            "cell_types": int(metadata.cell_type_major.astype(str).nunique()),
            "cell_type_counts": metadata.cell_type_major.astype(str).value_counts().to_dict(),
            "metadata_columns": list(map(str, metadata.columns)),
            "lineage_columns": lineage_columns,
            "root_columns": root_columns,
            "root_counts": root_counts,
            "ordered_state_columns": order_columns,
            "usable_ordered_state_columns": usable_order_columns,
        },
        "ucell": {
            "algorithm": "UCELL_MANN_WHITNEY_U_UPDATED_NORMALIZATION",
            "formula": UCELL_FORMULA,
            "rank_universe": "GENCODE_V50_PRIMARY_PROTEIN_CODING_GENES_IN_H5",
            "ties_method": "average",
            "max_rank": int(max_rank),
            "complete_exact_signature_length_used_in_denominator": True,
            "missing_gene_policy": UCELL_MISSING_MEMBER_POLICY,
            "missing_member_contribution_to_rank_sum": "maxRank_per_member",
            "missing_member_treatment_is_expression_data_imputation": False,
            "domain_gate": "0 < present_genes AND complete_signature_genes < maxRank",
            "smoothing": False,
            "pathways_available": ucell_pathways,
            "pathways_unavailable": int(len(available_counts) - ucell_pathways),
            "unavailable_reason_counts": unavailable_counts,
            "cell_level_output_rows_expected": int(cell_count * ucell_pathways),
            "numeric_output_permitted": bool(
                nonfinite == 0
                and negative == 0
                and duplicated_protein_ids == 0
                and ucell_pathways > 0
            ),
            "unavailable_values_filled_with_zero_or_half": False,
            "reference": "https://code.bioconductor.org/browse/UCell/",
        },
        "pseudotime": {
            "algorithm_if_prerequisites_exist": (
                "WITHIN_DECLARED_LINEAGE_HVG_PCA_KNN_DIFFUSION_DISTANCE_FROM_EXPLICIT_ROOT"
            ),
            "requires_explicit_root_or_ordered_source_state": True,
            "requires_declared_lineage": True,
            "explicit_root_available": explicit_root_available,
            "ordered_source_state_available": ordered_source_state_available,
            "numeric_output_permitted": pseudotime_available,
            "unavailable_reason": pseudotime_reason,
            "typed_unavailable_required": not pseudotime_available,
            "unavailable_values_filled_with_zero_or_half": False,
        },
        "resource_budget": {
            "workers": 1,
            "cell_chunk_size": int(chunk_size),
            "dense_expression_chunk_bytes": dense_chunk_bytes,
            "estimated_peak_rss_gib": 4.0,
            "estimated_runtime_minutes": [5, 20],
            "estimated_output_gib": [0.15, 0.6],
        },
        "pilot_scope_if_started": {
            "fresh_cell_level_ucell": True,
            "fresh_pseudotime": pseudotime_available,
            "pseudotime_typed_unavailable": not pseudotime_available,
        },
        "pilot_safe_to_start": bool(
            nonfinite == 0
            and negative == 0
            and duplicated_protein_ids == 0
            and ucell_pathways > 0
        ),
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "historical_sc_trajectory_used": False,
        "single_cell_module_complete": False,
        "training_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r6-root", required=True, type=Path)
    parser.add_argument("--cancer", required=True)
    parser.add_argument("--expected-run-status-sha256", required=True)
    parser.add_argument("--expected-handoff-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--max-rank", type=int, default=1500)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_preflight(
        r6_root=args.r6_root,
        cancer_id=args.cancer,
        expected_run_status_sha256=args.expected_run_status_sha256,
        expected_handoff_sha256=args.expected_handoff_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        max_rank=args.max_rank,
        chunk_size=args.chunk_size,
    )
    if args.output_json is not None:
        output = args.output_json.resolve()
        if output.exists():
            raise CellLevelPreflightError(f"Preflight output reuse is forbidden: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
