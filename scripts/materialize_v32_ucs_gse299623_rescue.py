#!/usr/bin/env python3
"""Rebuild the full-transcriptome UCS GSE299623 matrix after excluding CS6."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from materialize_v32_tgct_gse261811_rescue import (
    canonical_gene_id, sha256, write_10x_h5,
)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_raw_triplet(root: Path, accession: str):
    from scipy.io import mmread
    feature_path = next(iter(root.glob(f"{accession}_*_features.tsv.gz")), None)
    barcode_path = next(iter(root.glob(f"{accession}_*_barcodes.tsv.gz")), None)
    matrix_path = next(iter(root.glob(f"{accession}_*_matrix.mtx.gz")), None)
    if not feature_path or not barcode_path or not matrix_path:
        raise RuntimeError(f"incomplete raw 10x triplet: {accession}")
    features = pd.read_csv(feature_path, sep="\t", header=None)
    ids = features.iloc[:, 0].map(canonical_gene_id).to_numpy(str)
    names = (features.iloc[:, 1].astype(str) if features.shape[1] > 1 else pd.Series(ids)).to_numpy(str)
    feature_type = (features.iloc[:, 2].astype(str) if features.shape[1] > 2 else pd.Series("Gene Expression", index=features.index))
    keep = feature_type.eq("Gene Expression").to_numpy()
    barcodes = pd.read_csv(barcode_path, sep="\t", header=None)[0].astype(str).to_numpy()
    matrix = mmread(matrix_path).tocsc()[keep]
    ids, names = ids[keep], names[keep]
    if matrix.shape != (len(ids), len(barcodes)):
        raise RuntimeError(f"matrix identity drift: {accession}")
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"duplicate versionless feature IDs: {accession}")
    return ids, names, barcodes, matrix, {
        "accession": accession,
        "features_sha256": sha256(feature_path), "barcodes_sha256": sha256(barcode_path),
        "matrix_sha256": sha256(matrix_path), "raw_cells": len(barcodes),
        "raw_features": len(ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--lncrna-ids-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable UCS rescue output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    building = output / f".building.{os.getpid()}"; building.mkdir()

    metadata = pd.read_csv(args.metadata, sep="\t")
    required = {"cell_id", "raw_cell_id", "patient_id", "patient_id_raw", "sample_class", "cell_type_major"}
    if required - set(metadata.columns):
        raise RuntimeError(f"metadata schema missing: {sorted(required - set(metadata.columns))}")
    metadata["accession"] = metadata.raw_cell_id.astype(str).str.extract(r"^(GSM\d+)", expand=False)
    metadata["raw_barcode_for_matrix"] = metadata.raw_cell_id.astype(str).str.rsplit(":", n=1).str[-1]
    if metadata.accession.isna().any() or metadata[["accession", "raw_barcode_for_matrix"]].duplicated().any():
        raise RuntimeError("metadata raw barcode identity is not one-to-one")
    if metadata.raw_cell_id.astype(str).str.contains("GSM9042132|CS6").any():
        raise RuntimeError("excluded GSM9042132/CS6 remains in metadata")

    from scipy import sparse
    matrices = []; metadata_parts = []; receipts = []
    reference_ids: np.ndarray | None = None; reference_names: np.ndarray | None = None
    for accession in sorted(metadata.accession.unique()):
        ids, names, barcodes, matrix, receipt = read_raw_triplet(args.raw_root.resolve(), accession)
        if reference_ids is None:
            reference_ids, reference_names = ids, names
        elif not np.array_equal(ids, reference_ids):
            raise RuntimeError(f"feature order differs across UCS samples: {accession}")
        local = metadata.loc[metadata.accession.eq(accession)].copy()
        index = {barcode: position for position, barcode in enumerate(barcodes)}
        columns = local.raw_barcode_for_matrix.map(index)
        if columns.isna().any():
            examples = local.loc[columns.isna(), "raw_barcode_for_matrix"].head().tolist()
            raise RuntimeError(f"metadata cells absent from raw matrix {accession}: {examples}")
        columns = columns.astype(int).to_numpy()
        if len(set(columns)) != len(columns):
            raise RuntimeError(f"raw matrix cells selected twice: {accession}")
        matrices.append(matrix[:, columns].tocsc())
        metadata_parts.append(local)
        receipt.update(retained_cells=len(local), retained_nonzero=int(matrices[-1].nnz))
        receipts.append(receipt)
    assert reference_ids is not None and reference_names is not None
    combined = sparse.hstack(matrices, format="csc")
    aligned_metadata = pd.concat(metadata_parts, ignore_index=True)
    if combined.shape[1] != len(aligned_metadata):
        raise RuntimeError("rebuilt UCS matrix/metadata cell closure failed")
    aligned_metadata = aligned_metadata.drop(columns=["accession", "raw_barcode_for_matrix"])
    h5_path = building / "raw_feature_bc_matrix.h5"
    metadata_path = building / "cell_metadata.tsv.gz"
    write_10x_h5(h5_path, combined, reference_ids, reference_names,
                 aligned_metadata.cell_id.astype(str).to_numpy())
    aligned_metadata.to_csv(metadata_path, sep="\t", index=False, compression="gzip")
    formal_lnc = {canonical_gene_id(value) for value in json.loads(args.lncrna_ids_json.read_text(encoding="utf-8"))}
    lnc_count = len(set(reference_ids) & formal_lnc)
    tumors = aligned_metadata.loc[aligned_metadata.sample_class.astype(str).str.lower().eq("tumor")]
    context_donors = (tumors.groupby("cell_type_major", observed=True).patient_id.nunique().sort_values(ascending=False).to_dict())
    audit = {
        "format": "CANCERLNCATLAS_V32_UCS_GSE299623_RESCUE_V1", "status": "PASS",
        "excluded_accession": "GSM9042132", "excluded_patient": "CS6",
        "cells": len(aligned_metadata), "features": len(reference_ids), "nonzero": int(combined.nnz),
        "lncrna_features": lnc_count, "donors": aligned_metadata.patient_id.nunique(),
        "tumor_donors": tumors.patient_id.nunique(), "normal_donors": aligned_metadata.loc[~aligned_metadata.index.isin(tumors.index), "patient_id"].nunique(),
        "tumor_context_donor_counts": context_donors, "samples": receipts,
        "formal_minimum_five_tumor_donors": tumors.patient_id.nunique() >= 5,
        "metadata_source_sha256": sha256(args.metadata.resolve()),
    }
    if lnc_count < 1000:
        raise RuntimeError(f"full-transcriptome rescue unexpectedly has only {lnc_count} formal lncRNAs")
    atomic_json(building / "AUDIT.json", audit)
    atomic_json(building / "SUCCESS.json", {
        "format": audit["format"], "status": "SUCCESS_RESCUE_SOURCE_MATERIALIZED",
        "audit_sha256": sha256(building / "AUDIT.json"),
        "h5_sha256": sha256(h5_path), "metadata_sha256": sha256(metadata_path),
        "cells": len(aligned_metadata), "features": len(reference_ids), "lncrna_features": lnc_count,
        "tumor_donors": int(tumors.patient_id.nunique()),
    })
    for child in building.iterdir():
        os.replace(child, output / child.name)
    building.rmdir()
    print(json.dumps(json.loads((output / "SUCCESS.json").read_text()), sort_keys=True))


if __name__ == "__main__":
    main()
