#!/usr/bin/env python3
"""Materialize the 11-patient full-transcriptome GSE139829 UVM rescue."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import tarfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from materialize_v32_tgct_gse261811_rescue import (
    canonical_gene_id,
    sha256,
    write_10x_h5,
)


MARKERS = {
    "T": ("CD3D", "CD3E", "TRAC", "IL7R", "CD8A"),
    "B": ("CD79A", "MS4A1", "CD74", "CD37", "CD19"),
    "Myeloid": ("LST1", "TYROBP", "FCER1G", "CTSS", "AIF1"),
    "Endothelial": ("PECAM1", "VWF", "EMCN", "KDR", "ENG"),
    "Fibroblast_stromal": ("COL1A1", "COL1A2", "DCN", "COL3A1", "LUM"),
    "Malignant_candidate": ("MLANA", "PMEL", "MITF", "TYR", "DCT", "SOX10", "MIA"),
}


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_soft(path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                accession = line.split("=", 1)[1].strip()
                current = {"accession": accession}
                records[accession] = current
            elif current is not None and line.startswith("!Sample_title = "):
                current["title"] = line.split("=", 1)[1].strip()
            elif current is not None and line.startswith("!Sample_source_name_ch1 = "):
                current["source"] = line.split("=", 1)[1].strip()
            elif current is not None and line.startswith("!Sample_characteristics_ch1 = sample type:"):
                current["sample_type"] = line.split(":", 1)[1].strip().lower()
            elif current is not None and line.startswith("!Sample_characteristics_ch1 = gep class:"):
                current["gep_class"] = line.split(":", 1)[1].strip()
    if len(records) != 11:
        raise RuntimeError(f"expected 11 GEO samples in SOFT, observed {len(records)}")
    if sorted(value.get("sample_type") for value in records.values()).count("primary") != 8:
        raise RuntimeError("GEO SOFT does not contain the declared eight primary samples")
    if sorted(value.get("sample_type") for value in records.values()).count("metastatic") != 3:
        raise RuntimeError("GEO SOFT does not contain the declared three metastatic samples")
    return records


def _gzip_member(archive: tarfile.TarFile, name: str) -> BinaryIO:
    member = archive.getmember(name)
    if not member.isfile() or member.issym() or member.islnk():
        raise RuntimeError(f"unsafe/non-file archive member: {name}")
    raw = archive.extractfile(member)
    if raw is None:
        raise RuntimeError(f"cannot read archive member: {name}")
    return gzip.GzipFile(fileobj=raw)


def read_sample(archive: tarfile.TarFile, accession: str, stem: str):
    from scipy.io import mmread
    from scipy import sparse

    feature_name = f"{accession}_{stem}_genes.tsv.gz"
    barcode_name = f"{accession}_{stem}_barcodes.tsv.gz"
    matrix_name = f"{accession}_{stem}_matrix.mtx.gz"
    with _gzip_member(archive, feature_name) as handle:
        features = pd.read_csv(handle, sep="\t", header=None)
    with _gzip_member(archive, barcode_name) as handle:
        barcodes = pd.read_csv(handle, sep="\t", header=None)[0].astype(str).to_numpy()
    with _gzip_member(archive, matrix_name) as handle:
        matrix = mmread(handle).tocsc()
    ids = features.iloc[:, 0].map(canonical_gene_id).to_numpy(str)
    names = (
        features.iloc[:, 1].astype(str).to_numpy()
        if features.shape[1] > 1
        else ids.copy()
    )
    if matrix.shape != (len(ids), len(barcodes)):
        raise RuntimeError(f"matrix identity drift: {accession}/{matrix.shape}")
    if len(set(ids)) != len(ids):
        codes, unique_ids = pd.factorize(ids, sort=True)
        projector = sparse.csr_matrix(
            (np.ones(len(codes), dtype=np.int8), (codes, np.arange(len(codes)))),
            shape=(len(unique_ids), len(codes)),
        )
        matrix = (projector @ matrix).tocsc()
        first_names = pd.DataFrame({"id": ids, "name": names}).groupby("id", sort=True).name.first()
        ids = np.asarray(unique_ids, dtype=str)
        names = first_names.reindex(ids).astype(str).to_numpy()
    # GEO provides the unfiltered 737,280-barcode 10x droplet matrix for each
    # sample, not a filtered cell matrix.  Empty/background droplets must not
    # become biological replicates.  Apply an explicit sample-local cell QC
    # gate before annotation or cross-sample concatenation.
    raw_barcodes = int(len(barcodes))
    library_size = np.asarray(matrix.sum(axis=0)).reshape(-1)
    detected_features = np.diff(matrix.indptr)
    mitochondrial_rows = np.asarray([
        str(name).upper().startswith("MT-") for name in names
    ], dtype=bool)
    mitochondrial_counts = (
        np.asarray(matrix[mitochondrial_rows].sum(axis=0)).reshape(-1)
        if mitochondrial_rows.any() else np.zeros(matrix.shape[1], dtype=float)
    )
    mitochondrial_fraction = np.divide(
        mitochondrial_counts, library_size,
        out=np.ones_like(library_size, dtype=float), where=library_size > 0,
    )
    keep = (
        (detected_features >= 200)
        & (detected_features <= 6000)
        & (mitochondrial_fraction <= 0.25)
    )
    if int(keep.sum()) < 100:
        raise RuntimeError(f"sample-local cell QC retained too few cells: {accession}/{int(keep.sum())}")
    matrix = matrix[:, keep].tocsc()
    barcodes = barcodes[keep]
    qc = {
        "raw_droplet_barcodes": raw_barcodes,
        "nonempty_barcodes": int((library_size > 0).sum()),
        "retained_cells": int(keep.sum()),
        "minimum_detected_features": 200,
        "maximum_detected_features": 6000,
        "maximum_mitochondrial_fraction": 0.25,
        "cell_filter_applied_before_annotation": True,
    }
    return ids, names, barcodes, matrix, qc


def broad_annotation(matrix, feature_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    name_to_rows: dict[str, list[int]] = {}
    for position, name in enumerate(feature_names.astype(str)):
        name_to_rows.setdefault(name.upper(), []).append(position)
    library = np.asarray(matrix.sum(axis=0)).reshape(-1)
    scale = np.divide(10_000.0, library, out=np.zeros_like(library, dtype=float), where=library > 0)
    scores = np.zeros((len(MARKERS), matrix.shape[1]), dtype=np.float32)
    for index, marker_names in enumerate(MARKERS.values()):
        rows = sorted({row for marker in marker_names for row in name_to_rows.get(marker, [])})
        if not rows:
            continue
        selected = matrix[rows].astype(np.float32).multiply(scale)
        selected.data = np.log1p(selected.data)
        scores[index] = np.asarray(selected.mean(axis=0)).reshape(-1)
    best = scores.argmax(axis=0)
    maximum = scores[best, np.arange(matrix.shape[1])]
    second = np.partition(scores, -2, axis=0)[-2]
    labels = np.asarray(list(MARKERS), dtype=object)[best]
    confident = (maximum >= 0.08) & ((maximum - second) >= 0.015)
    labels[~confident] = "Other"
    confidence = np.divide(
        maximum - second, maximum + 1e-6,
        out=np.zeros_like(maximum), where=maximum > 0,
    )
    return labels.astype(str), confidence.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--soft", required=True, type=Path)
    parser.add_argument("--lncrna-ids-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable UVM rescue output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    building = output / f".building.{os.getpid()}"
    building.mkdir()

    source = parse_soft(args.soft.resolve())
    from scipy import sparse

    matrices = []
    metadata_parts = []
    sample_receipts = []
    reference_ids: np.ndarray | None = None
    reference_names: np.ndarray | None = None
    with tarfile.open(args.archive.resolve(), "r:") as archive:
        member_names = [member.name for member in archive.getmembers() if member.isfile()]
        for accession in sorted(source):
            prefixes = [name for name in member_names if name.startswith(f"{accession}_") and name.endswith("_genes.tsv.gz")]
            if len(prefixes) != 1:
                raise RuntimeError(f"expected one genes member for {accession}: {prefixes}")
            stem = prefixes[0][len(accession) + 1 : -len("_genes.tsv.gz")]
            ids, names, barcodes, matrix, qc = read_sample(archive, accession, stem)
            if reference_ids is None:
                reference_ids, reference_names = ids, names
            elif not np.array_equal(ids, reference_ids):
                raise RuntimeError(f"feature order differs across UVM samples: {accession}")
            labels, confidence = broad_annotation(matrix, names)
            patient = stem
            full_barcodes = np.asarray([f"GSE139829:{patient}:{value}" for value in barcodes], dtype=str)
            sample_type = source[accession]["sample_type"]
            patient_id = f"SC_GSE139829_UVM:{patient}"
            matrices.append(matrix)
            metadata_parts.append(pd.DataFrame({
                "patient_id_raw": patient,
                "sample_id_raw": accession,
                "sample_class": "Tumor",
                "tumor_site": sample_type,
                "raw_cell_id": barcodes,
                "cell_type_raw": labels,
                "cell_type_major": labels,
                "author_malignant": False,
                "lineage_support": labels == "Malignant_candidate",
                "tumor_origin_support": True,
                "scevan_tumor": False,
                "scevan_call": "not_assessed",
                "doublet_flag": False,
                "doublet_score": np.nan,
                "cell_id": full_barcodes,
                "dataset_id": "SC_GSE139829_UVM",
                "cancer_id": "UVM",
                "patient_id": patient_id,
                "sample_id": f"SC_GSE139829_UVM:{accession}",
                "histology": "uveal_melanoma",
                "gep_class": source[accession].get("gep_class", "unknown"),
                "source_object": "GSE139829 processed raw 10x matrix",
                "source_annotation_source": "computed_broad_marker_scores_from_raw_counts",
                "source_candidate_method": "melanocytic_tumor_and_microenvironment_marker_lineage",
                "cell_type_score_confidence": confidence,
                "analysis_role": "rescue_independent_patient",
                "expression_scale": "raw_counts",
            }))
            sample_receipts.append({
                "accession": accession,
                "patient_id": patient_id,
                "sample_type": sample_type,
                "gep_class": source[accession].get("gep_class", "unknown"),
                "features": int(matrix.shape[0]),
                "cells": int(matrix.shape[1]),
                "nonzero": int(matrix.nnz),
                "cell_qc": qc,
            })

    assert reference_ids is not None and reference_names is not None
    combined = sparse.hstack(matrices, format="csc")
    metadata = pd.concat(metadata_parts, ignore_index=True)
    if combined.shape[1] != len(metadata) or metadata.cell_id.duplicated().any():
        raise RuntimeError("combined UVM matrix/metadata cell closure failed")
    formal_lnc = {
        canonical_gene_id(value)
        for value in json.loads(args.lncrna_ids_json.read_text(encoding="utf-8"))
    }
    lnc_count = len(set(reference_ids) & formal_lnc)
    if lnc_count < 1000:
        raise RuntimeError(f"full-transcriptome UVM rescue has only {lnc_count} formal lncRNAs")
    context = metadata.groupby("cell_type_major", observed=True).agg(
        cells=("cell_id", "size"), donors=("patient_id", "nunique")
    ).reset_index()
    formal_contexts = context.loc[(context.donors >= 5) & (context.cells >= 50)]
    if formal_contexts.empty:
        raise RuntimeError("UVM rescue has no replicated five-donor cell context")

    h5_path = building / "raw_feature_bc_matrix.h5"
    metadata_path = building / "cell_metadata.tsv.gz"
    write_10x_h5(
        h5_path, combined, reference_ids, reference_names,
        metadata.cell_id.astype(str).to_numpy(),
    )
    metadata.to_csv(metadata_path, sep="\t", index=False, compression="gzip")
    # Independent reopen checks prove that the published shape and identities
    # are not inferred only from in-memory objects.
    import h5py
    with h5py.File(h5_path, "r") as handle:
        stored_shape = tuple(int(value) for value in handle["matrix/shape"][:])
        stored_barcodes = handle["matrix/barcodes"].shape[0]
    if stored_shape != combined.shape or stored_barcodes != len(metadata):
        raise RuntimeError("independent H5 reopen closure failed")

    audit = {
        "format": "CANCERLNCATLAS_V32_UVM_GSE139829_RESCUE_V1",
        "status": "PASS",
        "cells": int(combined.shape[1]),
        "features": int(combined.shape[0]),
        "nonzero": int(combined.nnz),
        "lncrna_features": int(lnc_count),
        "patients": int(metadata.patient_id.nunique()),
        "primary_patients": int(metadata.loc[metadata.tumor_site.eq("primary"), "patient_id"].nunique()),
        "metastatic_patients": int(metadata.loc[metadata.tumor_site.eq("metastatic"), "patient_id"].nunique()),
        "context_counts": context.to_dict(orient="records"),
        "formal_contexts": formal_contexts.to_dict(orient="records"),
        "samples": sample_receipts,
        "archive_sha256": sha256(args.archive.resolve()),
        "soft_sha256": sha256(args.soft.resolve()),
        "source_build": "GRCh38_Ensembl_93",
        "raw_10x_droplet_matrices_filtered_before_cell_level_analysis": True,
        "total_raw_droplet_barcodes": int(sum(item["cell_qc"]["raw_droplet_barcodes"] for item in sample_receipts)),
        "independent_h5_reopen": "PASS",
    }
    atomic_json(building / "AUDIT.json", audit)
    atomic_json(building / "SUCCESS.json", {
        "format": audit["format"],
        "status": "SUCCESS_RESCUE_SOURCE_MATERIALIZED",
        "audit_sha256": sha256(building / "AUDIT.json"),
        "h5_sha256": sha256(h5_path),
        "metadata_sha256": sha256(metadata_path),
        "cells": audit["cells"],
        "features": audit["features"],
        "lncrna_features": audit["lncrna_features"],
        "patients": audit["patients"],
        "formal_context_count": len(formal_contexts),
    })
    for child in building.iterdir():
        os.replace(child, output / child.name)
    building.rmdir()
    print((output / "SUCCESS.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
