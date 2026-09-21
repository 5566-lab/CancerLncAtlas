#!/usr/bin/env python3
"""Merge independent TGCT cohorts and independently audit formal context eligibility."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from materialize_v32_tgct_gse261811_rescue import canonical_gene_id, sha256, write_10x_h5


def decode(values) -> np.ndarray:
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values], dtype=str)


def read_h5(path: Path):
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(value) for value in group["shape"][:])
        matrix = sparse.csc_matrix((group["data"][:], group["indices"][:], group["indptr"][:]), shape=shape)
        ids = decode(group["features"]["id"][:])
        names = decode(group["features"]["name"][:])
        barcodes = decode(group["barcodes"][:])
    if matrix.shape != (len(ids), len(barcodes)):
        raise RuntimeError(f"H5 sparse closure drift: {path}")
    return matrix, ids, names, barcodes


def collapse_and_remap(matrix, ids: np.ndarray, names: np.ndarray, union_index: dict[str, int]):
    canonical = np.asarray([canonical_gene_id(value) for value in ids], dtype=str)
    destination = np.asarray([union_index[value] for value in canonical], dtype=int)
    projector = sparse.csr_matrix(
        (np.ones(len(destination), dtype=np.int8), (destination, np.arange(len(destination)))),
        shape=(len(union_index), len(destination)),
    )
    remapped = (projector @ matrix).tocsc()
    name_map = {}
    for key, name in zip(canonical, names, strict=True):
        name_map.setdefault(key, str(name))
    return remapped, name_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-h5", required=True, type=Path)
    parser.add_argument("--old-metadata", required=True, type=Path)
    parser.add_argument("--new-root", required=True, type=Path)
    parser.add_argument("--lncrna-ids-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable TGCT merged output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    building = output / f".building.{os.getpid()}"; building.mkdir()

    roots = [
        (args.old_h5.resolve(), args.old_metadata.resolve(), "GSE228501"),
        ((args.new_root / "raw_feature_bc_matrix.h5").resolve(), (args.new_root / "cell_metadata.tsv.gz").resolve(), "GSE261811"),
    ]
    loaded = []; all_ids = set(); source_receipts = []; patient_sets: list[set[str]] = []
    for h5_path, metadata_path, accession in roots:
        matrix, ids, names, barcodes = read_h5(h5_path)
        metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
        if len(metadata) != len(barcodes) or metadata.cell_id.astype(str).duplicated().any():
            raise RuntimeError(f"metadata/barcode closure drift: {accession}")
        position = {barcode: index for index, barcode in enumerate(barcodes)}
        columns = metadata.cell_id.astype(str).map(position)
        if columns.isna().any() and "raw_cell_id" in metadata:
            normalized_position = {
                (barcode[len("LEGACY_SC_TGCT:"):] if barcode.startswith("LEGACY_SC_TGCT:") else barcode): index
                for index, barcode in enumerate(barcodes)
            }
            columns = metadata.raw_cell_id.astype(str).map(normalized_position)
        if columns.isna().any():
            raise RuntimeError(f"metadata IDs absent from H5: {accession}")
        matrix = matrix[:, columns.astype(int).to_numpy()]
        all_ids.update(canonical_gene_id(value) for value in ids)
        loaded.append((matrix, ids, names, metadata, accession))
        patient_sets.append(set(metadata.patient_id.astype(str)))
        source_receipts.append({
            "accession": accession, "h5_path": str(h5_path), "h5_sha256": sha256(h5_path),
            "metadata_path": str(metadata_path), "metadata_sha256": sha256(metadata_path),
            "cells": len(metadata), "patients": metadata.patient_id.nunique(), "features": len(ids),
        })
    union_ids = np.asarray(sorted(all_ids), dtype=str); union_index = {value: index for index, value in enumerate(union_ids)}
    matrices = []; metadata_parts = []; name_map = {}
    for matrix, ids, names, metadata, accession in loaded:
        remapped, local_names = collapse_and_remap(matrix, ids, names, union_index)
        matrices.append(remapped); name_map.update({key: value for key, value in local_names.items() if key not in name_map})
        metadata = metadata.copy(); metadata["source_accession"] = accession
        metadata_parts.append(metadata)
    combined = sparse.hstack(matrices, format="csc")
    metadata = pd.concat(metadata_parts, ignore_index=True, sort=False)
    harmonize = {"Germ_tumor": "Malignant_candidate", "Fibroblast": "Fibroblast_stromal"}
    metadata["cell_type_major_original"] = metadata.cell_type_major
    metadata["cell_type_major"] = metadata.cell_type_major.replace(harmonize)
    metadata["author_malignant"] = metadata.cell_type_major.eq("Malignant_candidate")
    if combined.shape[1] != len(metadata) or metadata.cell_id.astype(str).duplicated().any():
        raise RuntimeError("merged TGCT identity closure failed")
    if patient_sets[0] & patient_sets[1]:
        raise RuntimeError("TGCT source patient identifiers overlap")
    if metadata.patient_id.nunique() != 8:
        raise RuntimeError(f"expected eight independent TGCT patients, observed {metadata.patient_id.nunique()}")
    union_names = np.asarray([name_map.get(value, value) for value in union_ids], dtype=str)
    h5_path = building / "raw_feature_bc_matrix.h5"; metadata_path = building / "cell_metadata.tsv.gz"
    write_10x_h5(h5_path, combined, union_ids, union_names, metadata.cell_id.astype(str).to_numpy())
    metadata.to_csv(metadata_path, sep="\t", index=False, compression="gzip")

    formal_lnc = {canonical_gene_id(value) for value in json.loads(args.lncrna_ids_json.read_text(encoding="utf-8"))}
    lnc_count = len(set(union_ids) & formal_lnc)
    contexts = metadata.groupby("cell_type_major", observed=True).agg(cells=("cell_id", "size"), donors=("patient_id", "nunique")).reset_index()
    formal_contexts = contexts.loc[(contexts.donors >= 5) & (contexts.cells >= 100)].copy()
    required = {"T_cell", "B_cell", "Myeloid", "Endothelial", "Fibroblast_stromal", "Malignant_candidate"}
    if not required.issubset(set(formal_contexts.cell_type_major.astype(str))):
        raise RuntimeError(f"TGCT merged formal context gate failed: {sorted(required - set(formal_contexts.cell_type_major.astype(str)))}")

    # Independent reopen: reconstruct shape and verify exact barcode order.
    audited_matrix, audited_ids, _, audited_barcodes = read_h5(h5_path)
    if audited_matrix.shape != combined.shape or not np.array_equal(audited_ids, union_ids):
        raise RuntimeError("independent merged H5 shape/feature audit failed")
    if not np.array_equal(audited_barcodes, metadata.cell_id.astype(str).to_numpy()):
        raise RuntimeError("independent merged H5 barcode audit failed")
    contexts.to_csv(building / "context_coverage.tsv", sep="\t", index=False)
    audit = {
        "format": "CANCERLNCATLAS_V32_TGCT_MERGED_RESCUE_AUDIT_V1", "status": "PASS",
        "sources": source_receipts, "cells": len(metadata), "patients": metadata.patient_id.nunique(),
        "features": len(union_ids), "lncrna_features": lnc_count, "nonzero": int(combined.nnz),
        "formal_contexts": formal_contexts.to_dict(orient="records"),
        "source_patient_sets_disjoint": True, "h5_reopened_and_sparse_shape_verified": True,
        "barcode_order_exact": True, "formal_context_minimum_donors": 5,
        "annotation_harmonization": harmonize,
    }
    audit_path = building / "INDEPENDENT_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    success = {
        "format": audit["format"], "status": "SUCCESS_FORMAL_TGCT_RESCUE_INPUT",
        "audit_sha256": sha256(audit_path), "h5_sha256": sha256(h5_path),
        "metadata_sha256": sha256(metadata_path), "cells": len(metadata), "patients": 8,
        "lncrna_features": lnc_count, "formal_context_count": len(formal_contexts),
    }
    (building / "SUCCESS.json").write_text(json.dumps(success, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for child in building.iterdir():
        os.replace(child, output / child.name)
    building.rmdir()
    print(json.dumps(success, sort_keys=True))


if __name__ == "__main__":
    main()
