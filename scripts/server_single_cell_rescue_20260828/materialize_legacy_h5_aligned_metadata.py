#!/usr/bin/env python3
"""Build H5-barcode-aligned legacy metadata candidates without promoting donors."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import h5py


EXPECTED_ROOT = Path("./data/CancerLncAtlas")
PUBLIC_H5 = Path("./data/CancerLncAtlas/processed/sc_tool_input")
RUN_ID = "single_cell_rescue_20260828"
CANCERS = ("BLCA", "BRCA", "COAD", "LIHC", "LUAD", "OV", "PAAD", "PRAD", "STAD", "THCA")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def materialize(root: Path, cancer: str) -> dict[str, object]:
    source = root / "metadata" / RUN_ID / f"{cancer}_cell_metadata_rescue_candidate.tsv.gz"
    h5 = PUBLIC_H5 / cancer / "raw_feature_bc_matrix.h5"
    out_dir = root / "processed" / RUN_ID / cancer
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "cell_metadata.h5_aligned.patient_unverified.tsv.gz"
    prefix = f"LEGACY_SC_{cancer}:"
    dataset_id = f"LEGACY_SC_{cancer}"
    with h5py.File(h5, "r") as handle:
        h5_barcodes = decode(handle["matrix/barcodes"][:])

    output_fields = [
        "cell_id", "raw_cell_id", "dataset_id", "cancer_id",
        "patient_id_raw", "patient_id_candidate", "patient_source_column",
        "patient_id_verified", "sample_id_raw", "sample_id_candidate",
        "cell_type_raw", "cell_type_major_candidate", "celltype_source_column",
        "compartment_candidate", "author_malignant_raw", "n_count_rna",
        "n_feature_rna_from_counts", "source_object", "source_h5",
        "source_class", "analysis_role",
    ]
    rows = 0
    donors: set[str] = set()
    with gzip.open(source, "rt", encoding="utf-8", newline="") as inp, gzip.open(
        output, "wt", encoding="utf-8", newline="", compresslevel=6
    ) as out:
        reader = csv.DictReader(inp, delimiter="\t")
        writer = csv.DictWriter(out, fieldnames=output_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index, record in enumerate(reader):
            if index >= len(h5_barcodes):
                raise RuntimeError(f"{cancer}: RDS metadata has more rows than H5 barcodes")
            raw_cell = record["cell_id"]
            expected = prefix + raw_cell
            observed = h5_barcodes[index]
            if observed != expected:
                raise RuntimeError(
                    f"{cancer}: barcode mismatch at {index}: expected={expected!r}, observed={observed!r}"
                )
            donor_raw = record.get("patient_id_raw", "").strip()
            donors.add(donor_raw)
            writer.writerow({
                "cell_id": observed,
                "raw_cell_id": raw_cell,
                "dataset_id": dataset_id,
                "cancer_id": cancer,
                "patient_id_raw": donor_raw,
                "patient_id_candidate": f"{dataset_id}:{donor_raw}" if donor_raw else "",
                "patient_source_column": record.get("patient_source_column", ""),
                "patient_id_verified": "false",
                "sample_id_raw": donor_raw,
                "sample_id_candidate": f"{dataset_id}:{donor_raw}" if donor_raw else "",
                "cell_type_raw": record.get("cell_type_raw", ""),
                "cell_type_major_candidate": record.get("cell_type_major_candidate", ""),
                "celltype_source_column": record.get("celltype_source_column", ""),
                "compartment_candidate": record.get("compartment_candidate", ""),
                "author_malignant_raw": record.get("author_malignant_raw", ""),
                "n_count_rna": record.get("n_count_rna", ""),
                "n_feature_rna_from_counts": record.get("n_feature_rna_from_counts", ""),
                "source_object": record.get("source_object", ""),
                "source_h5": str(h5),
                "source_class": "legacy_seurat_counts_h5_identity_verified",
                "analysis_role": "RESCUE_CANDIDATE_H5_ALIGNED_PATIENT_MAPPING_UNVERIFIED",
            })
            rows += 1
    if rows != len(h5_barcodes):
        raise RuntimeError(f"{cancer}: row count {rows} != H5 barcode count {len(h5_barcodes)}")
    return {
        "cancer_id": cancer,
        "source_metadata": str(source),
        "source_metadata_sha256": sha256(source),
        "source_h5": str(h5),
        "source_h5_sha256": sha256(h5),
        "output_path": str(output),
        "output_sha256": sha256(output),
        "rows": rows,
        "sample_or_donor_candidates": len(donors),
        "barcode_order_exact_after_literal_prefix": True,
        "literal_prefix": prefix,
        "patient_id_verified": False,
        "status": "PASS_H5_ALIGNED_METADATA_CANDIDATE_PATIENT_MAPPING_REQUIRED",
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} {EXPECTED_ROOT}")
    root = Path(sys.argv[1]).resolve()
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing output root outside {EXPECTED_ROOT}: {root}")
    rows = [materialize(root, cancer) for cancer in CANCERS]
    manifest_dir = root / "manifests" / RUN_ID
    json_path = manifest_dir / "legacy_h5_aligned_metadata_gate.json"
    json_path.write_text(json.dumps({
        "format": "CANCERLNCATLAS_LEGACY_H5_ALIGNED_METADATA_GATE_V1",
        "formal_promotion_allowed": False,
        "formal_blocker": "PATIENT_OR_DONOR_MAPPING_NOT_SOURCE_VERIFIED",
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tsv_path = manifest_dir / "legacy_h5_aligned_metadata_gate.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"RESULT_PATH\t{json_path}")
    print(f"RESULT_SHA256\t{sha256(json_path)}")
    print(f"PASS\t{sum(row['status'].startswith('PASS_') for row in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
