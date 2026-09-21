#!/usr/bin/env python3
"""Promote BRCA/COAD/OV legacy matrices using official E-MTAB-8107 SDRF.

The old Seurat objects retained source-sample identifiers but treated them as
unverified patient IDs.  The official SDRF maps each processed source file to
an individual and sampling site.  This materializer collapses multi-sample
individuals and removes the exact duplicated BRCA scrJUQ058 typo copy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from materialize_v32_tgct_gse261811_rescue import (
    canonical_gene_id,
    sha256,
    write_10x_h5,
)


CANCERS = ("BRCA", "COAD", "OV")
DUPLICATE_SOURCE_SAMPLE = "scrJUQ058"
DUPLICATE_CANONICAL_SAMPLE = "sc5rJUQ058"


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=str)


def read_h5(path: Path):
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(value) for value in group["shape"][:])
        matrix = sparse.csc_matrix(
            (
                group["data"][:],
                group["indices"][:].astype(np.int32, copy=False),
                group["indptr"][:].astype(np.int64, copy=False),
            ),
            shape=shape,
        )
        barcodes = decode(group["barcodes"][:])
        feature_group = group["features"]
        ids = decode(feature_group["id"][:] if "id" in feature_group else feature_group["name"][:])
        names = decode(feature_group["name"][:] if "name" in feature_group else feature_group["id"][:])
    if matrix.shape != (len(ids), len(barcodes)):
        raise RuntimeError(f"H5 shape identity drift: {path}")
    return matrix, barcodes, ids, names


def source_sample(value: str) -> str:
    prefix = "2_pancancer_"
    text = str(value)
    return text[len(prefix):] if text.startswith(prefix) else text


def load_sdrf(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {
        "Derived Array Data File", "Characteristics[individual]",
        "Characteristics[disease]", "Characteristics[sampling site]",
        "Characteristics[organism part]",
    }
    if required - set(frame.columns):
        raise RuntimeError(f"SDRF schema missing: {sorted(required - set(frame.columns))}")
    frame = frame[list(required)].drop_duplicates("Derived Array Data File")
    frame["source_sample"] = frame["Derived Array Data File"].astype(str).str.removesuffix(".counts.csv")
    if frame.source_sample.duplicated().any():
        raise RuntimeError("official SDRF source sample is not one-to-one")
    return frame


def promote_one(
    cancer: str, h5_root: Path, metadata_root: Path, sdrf: pd.DataFrame,
    sdrf_path: Path, formal_lnc: set[str], formal_lnc_symbols: set[str],
    annotation_path: Path, output_root: Path,
) -> dict[str, Any]:
    output = output_root / cancer
    output.mkdir(parents=True, exist_ok=False)
    building = output / f".building.{os.getpid()}"
    building.mkdir()
    metadata_path = metadata_root / f"{cancer}_cell_metadata_rescue_candidate.tsv.gz"
    source_h5 = h5_root / cancer / "raw_feature_bc_matrix.h5"
    metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    if metadata.cell_id.duplicated().any():
        raise RuntimeError(f"legacy metadata cell IDs are duplicated: {cancer}")
    metadata["source_sample"] = metadata.patient_id_candidate.astype(str).map(source_sample)
    duplicate_mask = metadata.source_sample.eq(DUPLICATE_SOURCE_SAMPLE) if cancer == "BRCA" else pd.Series(False, index=metadata.index)
    duplicate_cells = int(duplicate_mask.sum())
    if cancer == "BRCA":
        canonical = metadata.loc[metadata.source_sample.eq(DUPLICATE_CANONICAL_SAMPLE), "cell_id"].astype(str).str.replace(
            f"2_pancancer_{DUPLICATE_CANONICAL_SAMPLE}_", "", regex=False
        )
        duplicate = metadata.loc[duplicate_mask, "cell_id"].astype(str).str.replace(
            f"2_pancancer_{DUPLICATE_SOURCE_SAMPLE}_", "", regex=False
        )
        if duplicate_cells != 347 or set(canonical) != set(duplicate):
            raise RuntimeError("BRCA scrJUQ058 is not the expected exact 347-cell duplicate")
        metadata = metadata.loc[~duplicate_mask].copy()

    disease = {"BRCA": "breast cancer", "COAD": "colorectal cancer", "OV": "ovarian cancer"}[cancer]
    official = sdrf.loc[sdrf["Characteristics[disease]"].astype(str).str.lower().eq(disease)].copy()
    official = official.set_index("source_sample", drop=False)
    missing = sorted(set(metadata.source_sample) - set(official.index))
    if missing:
        raise RuntimeError(f"legacy samples absent from official SDRF for {cancer}: {missing}")
    metadata["source_individual"] = metadata.source_sample.map(official["Characteristics[individual]"].astype(str))
    metadata["sampling_site"] = metadata.source_sample.map(official["Characteristics[sampling site]"].astype(str))
    metadata["organism_part"] = metadata.source_sample.map(official["Characteristics[organism part]"].astype(str))
    metadata["patient_id"] = "SC_EMTAB8107_" + cancer + ":" + metadata.source_individual.astype(str)
    metadata["patient_id_verified"] = True
    metadata["sample_id"] = "SC_EMTAB8107_" + cancer + ":" + metadata.source_sample.astype(str)
    metadata["sample_class"] = np.where(
        metadata.sampling_site.astype(str).str.lower().str.contains("normal"), "Normal", "Tumor"
    )
    metadata["cell_type_major"] = metadata.cell_type_major_candidate.astype(str)
    metadata["dataset_id"] = "SC_E_MTAB_8107_" + cancer
    metadata["analysis_role"] = "FORMAL_SOURCE_VERIFIED_PATIENT_MAPPING"
    metadata["mapping_authority"] = "E-MTAB-8107.sdrf.txt"

    matrix, raw_barcodes, feature_ids, feature_names = read_h5(source_h5)
    literal_prefix = f"LEGACY_SC_{cancer}:"
    normalized_barcodes = np.asarray([
        value[len(literal_prefix):] if value.startswith(literal_prefix) else value
        for value in raw_barcodes
    ], dtype=str)
    position = {value: index for index, value in enumerate(normalized_barcodes)}
    if len(position) != len(normalized_barcodes):
        raise RuntimeError(f"source H5 has duplicate barcode identities: {cancer}")
    columns = metadata.cell_id.astype(str).map(position)
    if columns.isna().any():
        raise RuntimeError(f"corrected metadata cells absent from source H5: {cancer}")
    columns_array = columns.astype(int).to_numpy()
    corrected = matrix[:, columns_array].tocsc()
    if corrected.shape[1] != len(metadata):
        raise RuntimeError(f"corrected H5/metadata cell closure failed: {cancer}")
    output_barcodes = ("SC_EMTAB8107_" + cancer + ":" + metadata.cell_id.astype(str)).to_numpy()
    corrected_h5 = building / "raw_feature_bc_matrix.h5"
    corrected_metadata = building / "cell_metadata.tsv.gz"
    write_10x_h5(corrected_h5, corrected, feature_ids, feature_names, output_barcodes)
    metadata["cell_id"] = output_barcodes
    metadata.to_csv(corrected_metadata, sep="\t", index=False, compression="gzip")

    stable_feature_ids = [canonical_gene_id(value) for value in feature_ids]
    direct_rows = {i for i, value in enumerate(stable_feature_ids) if value in formal_lnc}
    symbol_rows = {i for i, value in enumerate(feature_names) if value in formal_lnc_symbols}
    lnc_count_direct = len(direct_rows)
    lnc_count_symbol = len(symbol_rows - direct_rows)
    lnc_count = len(direct_rows | symbol_rows)
    tumors = metadata.loc[metadata.sample_class.eq("Tumor")]
    context = tumors.groupby("cell_type_major", observed=True).agg(
        cells=("cell_id", "size"), donors=("patient_id", "nunique")
    ).reset_index()
    formal_contexts = context.loc[(context.donors >= 5) & (context.cells >= 50)]
    if metadata.patient_id.nunique() < 5 or formal_contexts.empty or lnc_count < 1000:
        raise RuntimeError(f"formal promotion gate failed for {cancer}")
    with h5py.File(corrected_h5, "r") as handle:
        reopened_shape = tuple(int(value) for value in handle["matrix/shape"][:])
        reopened_cells = int(handle["matrix/barcodes"].shape[0])
    if reopened_shape != corrected.shape or reopened_cells != len(metadata):
        raise RuntimeError(f"independent corrected H5 reopen failed: {cancer}")

    audit = {
        "format": "CANCERLNCATLAS_V32_LEGACY_EMTAB8107_PROMOTION_V1",
        "status": "PASS",
        "cancer_id": cancer,
        "source_samples": int(metadata.source_sample.nunique()),
        "verified_patients": int(metadata.patient_id.nunique()),
        "tumor_patients": int(tumors.patient_id.nunique()),
        "cells": int(corrected.shape[1]),
        "features": int(corrected.shape[0]),
        "nonzero": int(corrected.nnz),
        "lncrna_features": int(lnc_count),
        "lncrna_direct_stable_id_features": int(lnc_count_direct),
        "lncrna_unique_gencode_symbol_features": int(lnc_count_symbol),
        "exact_duplicate_cells_removed": duplicate_cells,
        "duplicate_source_removed": DUPLICATE_SOURCE_SAMPLE if duplicate_cells else None,
        "formal_contexts": formal_contexts.to_dict(orient="records"),
        "source_h5_sha256": sha256(source_h5),
        "source_metadata_sha256": sha256(metadata_path),
        "mapping_sdrf_sha256": sha256(sdrf_path),
        "gencode_annotation_sha256": sha256(annotation_path),
        "patient_mapping_source_verified": True,
        "multi_sample_individuals_collapsed": True,
        "independent_h5_reopen": "PASS",
    }
    atomic_json(building / "AUDIT.json", audit)
    atomic_json(building / "SUCCESS.json", {
        "format": audit["format"], "status": "SUCCESS_FORMAL_PROMOTION",
        "cancer_id": cancer, "audit_sha256": sha256(building / "AUDIT.json"),
        "h5_sha256": sha256(corrected_h5), "metadata_sha256": sha256(corrected_metadata),
        "patients": audit["verified_patients"], "tumor_patients": audit["tumor_patients"],
        "cells": audit["cells"], "lncrna_features": audit["lncrna_features"],
        "formal_context_count": len(formal_contexts),
    })
    for child in building.iterdir():
        os.replace(child, output / child.name)
    building.rmdir()
    return json.loads((output / "SUCCESS.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-root", required=True, type=Path)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--sdrf", required=True, type=Path)
    parser.add_argument("--lncrna-ids-json", required=True, type=Path)
    parser.add_argument("--lncrna-symbols-json", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable E-MTAB-8107 promotion output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sdrf = load_sdrf(args.sdrf.resolve())
    formal_lnc = {
        canonical_gene_id(value)
        for value in json.loads(args.lncrna_ids_json.read_text(encoding="utf-8"))
    }
    formal_lnc_symbols = set(json.loads(
        args.lncrna_symbols_json.read_text(encoding="utf-8")
    ))
    if len(formal_lnc_symbols) < 1000:
        raise RuntimeError("GENCODE-derived unique lncRNA symbol universe is unexpectedly small")
    results = [
        promote_one(
            cancer, args.h5_root.resolve(), args.metadata_root.resolve(), sdrf,
            args.sdrf.resolve(), formal_lnc, formal_lnc_symbols,
            args.annotation.resolve(), output,
        )
        for cancer in CANCERS
    ]
    atomic_json(output / "SUCCESS.json", {
        "format": "CANCERLNCATLAS_V32_LEGACY_EMTAB8107_PROMOTION_COLLECTION_V1",
        "status": "SUCCESS_FORMAL_PROMOTION_COLLECTION",
        "cancers": list(CANCERS), "cancer_count": len(CANCERS), "results": results,
        "mapping_sdrf_sha256": sha256(args.sdrf.resolve()),
    })
    print((output / "SUCCESS.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
