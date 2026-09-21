#!/usr/bin/env python3
"""Compare read-only 10x H5 caches with Seurat-derived aggregate signatures."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


EXPECTED_ROOT = Path("./data/CancerLncAtlas")
RUN_ID = "single_cell_rescue_20260828"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def read_tsv_gz(path: Path, key: str, numeric: tuple[str, ...]) -> dict[str, tuple[float, ...]]:
    out: dict[str, tuple[float, ...]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            out[row[key]] = tuple(float(row[name]) for name in numeric)
    return out


def inspect_h5(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        shape = tuple(int(x) for x in matrix["shape"][:])
        barcodes = decode(matrix["barcodes"][:])
        feature_group = matrix["features"]
        if "name" in feature_group:
            features = decode(feature_group["name"][:])
        elif "id" in feature_group:
            features = decode(feature_group["id"][:])
        else:
            raise KeyError("matrix/features lacks both name and id")
        indptr = matrix["indptr"][:].astype(np.int64, copy=False)
        indices = matrix["indices"][:].astype(np.int64, copy=False)
        data = matrix["data"][:].astype(np.float64, copy=False)

    if len(barcodes) != shape[1] or len(features) != shape[0]:
        raise ValueError(f"shape mismatch in {path}")
    cell_nnz = np.diff(indptr)
    cell_sum = np.zeros(shape[1], dtype=np.float64)
    for idx in range(shape[1]):
        cell_sum[idx] = data[indptr[idx] : indptr[idx + 1]].sum()
    feature_sum = np.bincount(indices, weights=data, minlength=shape[0]).astype(np.float64)
    feature_ncell = np.bincount(indices, minlength=shape[0]).astype(np.int64)
    return {
        "shape": shape,
        "barcodes": barcodes,
        "features": features,
        "cell_sum": cell_sum,
        "cell_nnz": cell_nnz,
        "feature_sum": feature_sum,
        "feature_ncell": feature_ncell,
        "nnz": int(data.size),
        "total_counts": float(data.sum()),
    }


def compare_one(root: Path, cancer: str) -> dict[str, object]:
    metadata = root / "metadata" / RUN_ID / f"{cancer}_cell_metadata_rescue_candidate.tsv.gz"
    feature_sig = root / "metadata" / RUN_ID / f"{cancer}_feature_count_signature.tsv.gz"
    h5 = Path("./data/CancerLncAtlas/processed/sc_tool_input") / cancer / "raw_feature_bc_matrix.h5"
    base: dict[str, object] = {
        "cancer_id": cancer,
        "metadata_path": str(metadata),
        "feature_signature_path": str(feature_sig),
        "h5_path": str(h5),
    }
    if not metadata.exists() or not feature_sig.exists() or not h5.exists():
        return {**base, "status": "MISSING_REQUIRED_INPUT"}

    rds_cells = read_tsv_gz(metadata, "cell_id", ("n_count_rna", "n_feature_rna_from_counts"))
    rds_features = read_tsv_gz(feature_sig, "feature_id", ("n_count_rna", "n_cells_detected"))
    h5_info = inspect_h5(h5)
    raw_barcodes = h5_info["barcodes"]
    legacy_prefix = f"LEGACY_SC_{cancer}:"
    barcodes = [
        value[len(legacy_prefix) :] if value.startswith(legacy_prefix) else value
        for value in raw_barcodes
    ]
    features = h5_info["features"]
    cell_set_match = set(barcodes) == set(rds_cells)
    feature_set_match = set(features) == set(rds_features)
    cell_order_match = barcodes == list(rds_cells)
    feature_order_match = features == list(rds_features)

    matched_cell_counts = 0
    matched_cell_features = 0
    if cell_set_match:
        for i, barcode in enumerate(barcodes):
            expected_count, expected_feature = rds_cells[barcode]
            matched_cell_counts += bool(np.isclose(h5_info["cell_sum"][i], expected_count, rtol=0, atol=1e-8))
            matched_cell_features += bool(h5_info["cell_nnz"][i] == expected_feature)

    matched_feature_counts = 0
    matched_feature_cells = 0
    if feature_set_match:
        for i, feature in enumerate(features):
            expected_count, expected_cells = rds_features[feature]
            matched_feature_counts += bool(np.isclose(h5_info["feature_sum"][i], expected_count, rtol=0, atol=1e-8))
            matched_feature_cells += bool(h5_info["feature_ncell"][i] == expected_cells)

    aggregate_exact = (
        cell_set_match
        and feature_set_match
        and matched_cell_counts == len(barcodes)
        and matched_cell_features == len(barcodes)
        and matched_feature_counts == len(features)
        and matched_feature_cells == len(features)
    )
    if aggregate_exact:
        status = "PASS_AGGREGATE_COUNTS_AND_IDENTITIES_EXACT"
    elif cancer == "CESC":
        status = "EXPECTED_REPLACEMENT_SOURCE_DIFFERS_FROM_CURRENT_H5_REBUILD_REQUIRED"
    else:
        status = "FAIL_RDS_H5_MISMATCH"
    return {
        **base,
        "h5_sha256": sha256(h5),
        "h5_features": h5_info["shape"][0],
        "h5_cells": h5_info["shape"][1],
        "h5_nnz": h5_info["nnz"],
        "h5_total_counts": h5_info["total_counts"],
        "rds_cells": len(rds_cells),
        "rds_features": len(rds_features),
        "barcode_set_match": cell_set_match,
        "barcode_order_match": cell_order_match,
        "barcode_normalization": (
            f"STRIP_LITERAL_PREFIX:{legacy_prefix}"
            if cancer != "CESC"
            else "NO_LEGACY_PREFIX_NORMALIZATION_REPLACEMENT_SOURCE"
        ),
        "feature_set_match": feature_set_match,
        "feature_order_match": feature_order_match,
        "cell_count_sum_matches": matched_cell_counts,
        "cell_detected_feature_matches": matched_cell_features,
        "feature_count_sum_matches": matched_feature_counts,
        "feature_detected_cell_matches": matched_feature_cells,
        "comparison_scope": "BARCODE_AND_FEATURE_IDENTITY_PLUS_EXACT_ROW_AND_COLUMN_AGGREGATES_NOT_ELEMENTWISE",
        "status": status,
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} {EXPECTED_ROOT}")
    root = Path(sys.argv[1]).resolve()
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing output root outside {EXPECTED_ROOT}: {root}")
    cancers = ["BLCA", "BRCA", "COAD", "LIHC", "LUAD", "OV", "PAAD", "PRAD", "STAD", "THCA", "CESC"]
    rows = []
    for cancer in cancers:
        try:
            rows.append(compare_one(root, cancer))
        except Exception as exc:  # typed failure is retained in the manifest
            rows.append({"cancer_id": cancer, "status": f"AUDIT_ERROR:{type(exc).__name__}:{exc}"})

    manifest_dir = root / "manifests" / RUN_ID
    manifest_dir.mkdir(parents=True, exist_ok=True)
    json_path = manifest_dir / "rds_h5_signature_comparison.json"
    json_path.write_text(
        json.dumps(
            {
                "format": "CANCERLNCATLAS_SINGLE_CELL_RDS_H5_SIGNATURE_AUDIT_V1",
                "comparison_is_elementwise": False,
                "source_root_read_only": "./data/CancerLncAtlas",
                "output_root": str(root),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    tsv_path = manifest_dir / "rds_h5_signature_comparison.tsv"
    columns = sorted({key for row in rows for key in row})
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"RESULT_PATH\t{json_path}")
    print(f"RESULT_SHA256\t{sha256(json_path)}")
    print(f"PASS\t{sum(row.get('status', '').startswith('PASS_') for row in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
