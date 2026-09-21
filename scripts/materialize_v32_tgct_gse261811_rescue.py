#!/usr/bin/env python3
"""Materialize and broadly annotate the four-patient GSE261811 TGCT rescue."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any

import numpy as np
import pandas as pd


SAMPLES = {
    "GSM8152420_Tumor2EC.tar.gz": ("Tumor2EC", "embryonal_carcinoma"),
    "GSM8152421_Tumor4SE.tar.gz": ("Tumor4SE", "seminoma"),
    "GSM8152422_Tumor12MIX.tar.gz": ("Tumor12MIX", "mixed_germ_cell_tumor"),
    "GSM8152423_Tumor13SE.tar.gz": ("Tumor13SE", "seminoma"),
}
MARKERS = {
    "T_cell": ("CD3D", "CD3E", "TRAC", "IL7R", "CD8A"),
    "B_cell": ("CD79A", "MS4A1", "CD74", "CD37", "CD19"),
    "Myeloid": ("LST1", "TYROBP", "FCER1G", "CTSS", "AIF1"),
    "Endothelial": ("PECAM1", "VWF", "EMCN", "KDR", "ENG"),
    "Fibroblast": ("COL1A1", "COL1A2", "DCN", "COL3A1", "LUM"),
    "Germ_tumor": ("POU5F1", "NANOG", "SOX17", "TFAP2C", "KIT", "PRAME", "EPCAM"),
}


def canonical_gene_id(value: Any) -> str:
    text = str(value).strip()
    if ":" in text:
        prefix, remainder = text.split(":", 1)
        if prefix.upper() in {"LNC", "GENE", "PROTEIN"}:
            text = remainder
    return text.split(".", 1)[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def locate_one(root: Path, names: tuple[str, ...]) -> Path:
    matches = [path for name in names for path in root.rglob(name)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one of {names}, observed {matches}")
    return matches[0]


def read_sample(archive: Path):
    from scipy.io import mmread
    from scipy import sparse

    with tempfile.TemporaryDirectory(prefix="tgct_gse261811_") as directory:
        temporary = Path(directory)
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle.getmembers():
                destination = (temporary / member.name).resolve()
                try:
                    destination.relative_to(temporary.resolve())
                except ValueError as exc:
                    raise RuntimeError(f"unsafe tar member: {member.name}") from exc
                if member.issym() or member.islnk():
                    raise RuntimeError(f"linked tar member is forbidden: {member.name}")
                handle.extract(member, temporary)
        features_path = locate_one(temporary, ("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"))
        barcodes_path = locate_one(temporary, ("barcodes.tsv.gz", "barcodes.tsv"))
        matrix_path = locate_one(temporary, ("matrix.mtx.gz", "matrix.mtx"))
        features = pd.read_csv(features_path, sep="\t", header=None, compression="infer")
        barcodes = pd.read_csv(barcodes_path, sep="\t", header=None, compression="infer")[0].astype(str)
        matrix = mmread(matrix_path).tocsc()
        if matrix.shape != (len(features), len(barcodes)):
            raise RuntimeError(f"matrix/feature/barcode mismatch in {archive.name}")
        ids = features.iloc[:, 0].map(canonical_gene_id)
        names = features.iloc[:, 1].astype(str) if features.shape[1] >= 2 else ids
        feature_type = features.iloc[:, 2].astype(str) if features.shape[1] >= 3 else pd.Series("Gene Expression", index=features.index)
        keep = feature_type.eq("Gene Expression")
        matrix = matrix[keep.to_numpy()].tocsc()
        table = pd.DataFrame({"gene_id": ids[keep].to_numpy(), "gene_name": names[keep].to_numpy()})
        if table.gene_id.duplicated().any():
            # Sum duplicated versionless Ensembl identifiers without changing cells.
            codes, uniques = pd.factorize(table.gene_id, sort=True)
            projector = sparse.csr_matrix(
                (np.ones(len(codes), dtype=np.int8), (codes, np.arange(len(codes)))),
                shape=(len(uniques), len(codes)),
            )
            matrix = (projector @ matrix).tocsc()
            names_by_id = table.groupby("gene_id", sort=True).gene_name.first()
            table = pd.DataFrame({"gene_id": uniques, "gene_name": names_by_id.reindex(uniques).to_numpy()})
        return table.reset_index(drop=True), barcodes.reset_index(drop=True), matrix


def broad_annotation(matrix, feature_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy import sparse

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
    # Avoid pretending that noise is a cell identity.  The threshold is a
    # staging rule; final UCell analyses retain this confidence explicitly.
    confident = (maximum >= 0.08) & ((maximum - second) >= 0.015)
    labels[~confident] = "Other"
    confidence = np.divide(
        maximum - second, maximum + 1e-6,
        out=np.zeros_like(maximum), where=maximum > 0,
    )
    return labels.astype(str), confidence.astype(np.float32)


def write_10x_h5(path: Path, matrix, ids: np.ndarray, names: np.ndarray, barcodes: np.ndarray) -> None:
    import h5py

    matrix = matrix.tocsc()
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("barcodes", data=np.asarray(barcodes, dtype="S"), compression="gzip")
        group.create_dataset("data", data=matrix.data, compression="gzip")
        group.create_dataset("indices", data=matrix.indices.astype(np.int32), compression="gzip")
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64), compression="gzip")
        group.create_dataset("shape", data=np.asarray(matrix.shape, dtype=np.int64))
        features = group.create_group("features")
        features.create_dataset("id", data=np.asarray(ids, dtype="S"), compression="gzip")
        features.create_dataset("name", data=np.asarray(names, dtype="S"), compression="gzip")
        features.create_dataset("feature_type", data=np.asarray(["Gene Expression"] * len(ids), dtype="S"), compression="gzip")
        features.create_dataset("genome", data=np.asarray(["GRCh38"] * len(ids), dtype="S"), compression="gzip")
        features.create_dataset("_all_tag_keys", data=np.asarray(["genome"], dtype="S"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--lncrna-intervals", type=Path)
    parser.add_argument("--lncrna-ids-json", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable TGCT rescue output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    building = output / f".building.{os.getpid()}"
    building.mkdir()
    from scipy import sparse

    samples = []
    all_ids: set[str] = set()
    for filename, (patient, histology) in SAMPLES.items():
        archive = args.archive_root.resolve() / filename
        features, barcodes, matrix = read_sample(archive)
        samples.append((filename, patient, histology, features, barcodes, matrix))
        all_ids.update(features.gene_id)
    union_ids = np.asarray(sorted(all_ids), dtype=str)
    union_index = {value: index for index, value in enumerate(union_ids)}
    symbol_map: dict[str, str] = {}
    for _, _, _, features, _, _ in samples:
        symbol_map.update(dict(zip(features.gene_id, features.gene_name, strict=True)))
    matrices, barcode_parts, metadata_parts, sample_receipts = [], [], [], []
    offset = 0
    for filename, patient, histology, features, barcodes, matrix in samples:
        row_map = features.gene_id.map(union_index).to_numpy(int)
        coo = matrix.tocoo()
        remapped = sparse.coo_matrix(
            (coo.data, (row_map[coo.row], coo.col)),
            shape=(len(union_ids), matrix.shape[1]),
        ).tocsc()
        labels, confidence = broad_annotation(remapped, np.asarray([symbol_map.get(value, value) for value in union_ids]))
        raw_barcodes = barcodes.astype(str).to_numpy()
        full_barcodes = np.asarray([f"GSE261811:{patient}:{value}" for value in raw_barcodes])
        matrices.append(remapped); barcode_parts.append(full_barcodes)
        patient_id = f"SC_GSE261811_TGCT:{patient}"
        sample_id = f"SC_GSE261811_TGCT:{filename.split('_')[0]}_{patient}"
        metadata_parts.append(pd.DataFrame({
            "patient_id_raw": patient, "sample_id_raw": filename.split("_")[0],
            "sample_class": "Tumor", "raw_cell_id": raw_barcodes,
            "cell_type_raw": labels, "cell_type_major": labels,
            "author_malignant": False,
            "lineage_support": labels == "Germ_tumor",
            "tumor_origin_support": True,
            "scevan_tumor": False, "scevan_call": "not_assessed",
            "doublet_flag": False, "doublet_score": np.nan,
            "cell_id": full_barcodes, "dataset_id": "SC_GSE261811_TGCT",
            "cancer_id": "TGCT", "patient_id": patient_id,
            "sample_id": sample_id, "histology": histology,
            "source_object": "GSE261811 processed raw 10x matrix",
            "source_annotation_source": "computed_broad_marker_scores_from_raw_counts",
            "source_candidate_method": "germ_cell_tumor_and_microenvironment_marker_lineage",
            "cell_type_score_confidence": confidence,
            "analysis_role": "rescue_candidate_independent_patients",
            "expression_scale": "raw_counts",
        }))
        sample_receipts.append({
            "archive": str((args.archive_root / filename).resolve()),
            "archive_sha256": sha256((args.archive_root / filename).resolve()),
            "patient_id": patient_id, "histology": histology,
            "features": int(remapped.shape[0]), "cells": int(remapped.shape[1]),
            "nonzero": int(remapped.nnz),
        })
        offset += remapped.shape[1]
    combined = sparse.hstack(matrices, format="csc")
    combined_barcodes = np.concatenate(barcode_parts)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    if combined.shape[1] != len(metadata) or len(set(combined_barcodes)) != len(combined_barcodes):
        raise RuntimeError("combined TGCT cell closure failed")
    union_names = np.asarray([symbol_map.get(value, value) for value in union_ids])
    h5_path = building / "raw_feature_bc_matrix.h5"
    metadata_path = building / "cell_metadata.tsv.gz"
    write_10x_h5(h5_path, combined, union_ids, union_names, combined_barcodes)
    metadata.to_csv(metadata_path, sep="\t", index=False, compression="gzip")
    if args.lncrna_ids_json:
        lnc_ids = {canonical_gene_id(value) for value in json.loads(args.lncrna_ids_json.read_text(encoding="utf-8"))}
    elif args.lncrna_intervals:
        intervals = pd.read_parquet(args.lncrna_intervals)
        id_column = "entity_id" if "entity_id" in intervals else "gene_id"
        type_column = "entity_type" if "entity_type" in intervals else "gene_type"
        lnc_ids = set(intervals.loc[
            intervals[type_column].astype(str).str.lower().str.contains("lnc|antisense|processed_transcript|sense_intronic|sense_overlapping"),
            id_column,
        ].astype(str).str.split(".", regex=False).str[0])
    else:
        parser.error("one of --lncrna-intervals or --lncrna-ids-json is required")
    lnc_count = len(set(union_ids) & lnc_ids)
    audit = {
        "format": "CANCERLNCATLAS_V32_TGCT_GSE261811_RESCUE_AUDIT_V1",
        "status": "PASS_STAGED_RESCUE_NOT_YET_MERGED_WITH_GSE228501",
        "patients": 4, "patient_ids": [f"SC_GSE261811_TGCT:{item[0]}" for item in SAMPLES.values()],
        "cells": int(combined.shape[1]), "features": int(combined.shape[0]),
        "nonzero": int(combined.nnz), "lncrna_features": lnc_count,
        "cell_type_counts": metadata.cell_type_major.value_counts().to_dict(),
        "samples": sample_receipts,
        "source_independent_of_existing_GSE228501": True,
        "formal_donor_count_after_verified_merge_expected": 8,
        "cell_type_annotation": "computed_broad_marker_scores_from_raw_counts",
        "requires_independent_annotation_audit": True,
    }
    atomic_json(building / "AUDIT.json", audit)
    success = {
        "format": audit["format"], "status": "SUCCESS_RESCUE_SOURCE_MATERIALIZED",
        "audit_sha256": sha256(building / "AUDIT.json"),
        "h5_sha256": sha256(h5_path), "metadata_sha256": sha256(metadata_path),
        "patients": 4, "cells": int(combined.shape[1]), "lncrna_features": lnc_count,
    }
    atomic_json(building / "SUCCESS.json", success)
    for child in building.iterdir():
        os.replace(child, output / child.name)
    building.rmdir()
    print(json.dumps(success, sort_keys=True))


if __name__ == "__main__":
    main()
