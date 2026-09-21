#!/usr/bin/env python3
"""Build the 23-cancer r11 single-cell contract from verified rescue outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import h5py
import pandas as pd


NEW_SOURCES = {
    "BRCA": ("emtab", "BRCA"),
    "COAD": ("emtab", "COAD"),
    "OV": ("emtab", "OV"),
    "CESC": ("direct", "single_cell_rescue_cesc_gse297041_20260903_r1"),
    "TGCT": ("direct", "single_cell_rescue_tgct_merged_20260903_r1"),
    "UVM": ("direct", "single_cell_rescue_uvm_gse139829_materialized_20260903_r2"),
}
TYPED_UNAVAILABLE = {
    "BLCA": "SOURCE_ACCESSION_AND_SAMPLE_TO_INDIVIDUAL_AUTHORITY_NOT_IDENTIFIED",
    "KICH": "INSUFFICIENT_INDEPENDENT_DONORS_3_LT_5_NO_RESCUE_SOURCE_IDENTIFIED",
    "KIRP": "INSUFFICIENT_INDEPENDENT_DONORS_3_LT_5_NO_RESCUE_SOURCE_IDENTIFIED",
    "LIHC": "SOURCE_ACCESSION_AND_SAMPLE_TO_INDIVIDUAL_AUTHORITY_NOT_IDENTIFIED",
    "LUAD": "SOURCE_ACCESSION_AND_SAMPLE_TO_INDIVIDUAL_AUTHORITY_NOT_IDENTIFIED",
    "PAAD": "SOURCE_ACCESSION_AND_SAMPLE_TO_INDIVIDUAL_AUTHORITY_NOT_IDENTIFIED",
    "PRAD": "SOURCE_ACCESSION_AND_SAMPLE_TO_INDIVIDUAL_AUTHORITY_NOT_IDENTIFIED",
    "STAD": "SOURCE_ACCESSION_AND_SAMPLE_TO_INDIVIDUAL_AUTHORITY_NOT_IDENTIFIED",
    "THCA": "SOURCE_ACCESSION_AND_SAMPLE_TO_INDIVIDUAL_AUTHORITY_NOT_IDENTIFIED",
    "UCS": "GSE299623_RELEASE_CONTAINS_18082_FEATURES_BUT_ONLY_38_FORMAL_LNCRNAS",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_success(root: Path) -> dict[str, Any]:
    path = root / "SUCCESS.json"
    if not path.is_file():
        raise RuntimeError(f"rescue SUCCESS missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not str(result.get("status", "")).startswith("SUCCESS"):
        raise RuntimeError(f"rescue SUCCESS status drift: {path}")
    return result


def source_root(results_root: Path, emtab_root: Path, cancer: str) -> Path:
    source_type, name = NEW_SOURCES[cancer]
    return emtab_root / name if source_type == "emtab" else results_root / name


def hardlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-r7-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--emtab-root", required=True, type=Path)
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable r11 contract output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    old_run = json.loads((args.old_r7_root / "RUN_STATUS.json").read_text(encoding="utf-8"))
    old_formal = set(map(str, old_run["formal_eligible_cancers"]))
    if len(old_formal) != 17 or old_formal & set(NEW_SOURCES):
        raise RuntimeError("old formal17 scope drift or rescue overlap")
    annotation = Path(next(iter(old_run["per_cancer"].values()))["annotation_path"])
    membership = Path(next(iter(old_run["per_cancer"].values()))["exact_membership_path"])
    per_cancer = dict(old_run["per_cancer"])
    new_receipts = {}

    for cancer in sorted(NEW_SOURCES):
        root = source_root(args.results_root.resolve(), args.emtab_root.resolve(), cancer)
        success = load_success(root)
        source_h5 = root / "raw_feature_bc_matrix.h5"
        source_metadata = root / "cell_metadata.tsv.gz"
        if success.get("h5_sha256") != sha256(source_h5):
            raise RuntimeError(f"rescue H5 SHA mismatch: {cancer}")
        if success.get("metadata_sha256") != sha256(source_metadata):
            raise RuntimeError(f"rescue metadata SHA mismatch: {cancer}")
        destination = args.processed_root.resolve() / cancer / "v32_rescue_20260903"
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f"immutable processed rescue target exists: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        h5_path = destination / "raw_feature_bc_matrix.h5"
        metadata_path = destination / "cell_metadata.parquet"
        hardlink_or_copy(source_h5, h5_path)
        metadata = pd.read_csv(source_metadata, sep="\t", low_memory=False)
        required = {"cell_id", "patient_id", "cell_type_major", "dataset_id", "cancer_id"}
        if required - set(metadata.columns):
            raise RuntimeError(f"rescue metadata schema missing for {cancer}")
        if set(metadata.cancer_id.astype(str).str.upper()) != {cancer}:
            raise RuntimeError(f"rescue metadata cancer drift: {cancer}")
        if metadata.cell_id.astype(str).duplicated().any() or metadata.patient_id.astype(str).str.strip().eq("").any():
            raise RuntimeError(f"rescue metadata identity drift: {cancer}")
        dataset_ids = sorted(set(metadata.dataset_id.astype(str)))
        if not dataset_ids or any(not value.strip() for value in dataset_ids):
            raise RuntimeError(f"rescue dataset ID is blank: {cancer}")
        if len(dataset_ids) == 1:
            formal_dataset_id = dataset_ids[0]
        else:
            # A cancer-level formal input may deliberately merge independent
            # public cohorts (TGCT does).  Preserve the source identity on every
            # cell while exposing one unambiguous dataset ID to the legacy r7
            # runner, which requires a scalar dataset contract per cancer.
            if "source_dataset_id" in metadata:
                existing = metadata.source_dataset_id.fillna("").astype(str)
                if (existing.str.strip() == "").any():
                    raise RuntimeError(f"blank pre-existing source dataset ID: {cancer}")
            else:
                metadata["source_dataset_id"] = metadata.dataset_id.astype(str)
            formal_dataset_id = f"SC_MERGED_{cancer}_V32_RESCUE_20260903"
            metadata["dataset_id"] = formal_dataset_id
        metadata.to_parquet(metadata_path, index=False, compression="zstd")
        with h5py.File(h5_path, "r") as handle:
            shape = tuple(int(value) for value in handle["matrix/shape"][:])
            barcode_count = int(handle["matrix/barcodes"].shape[0])
            feature_count = int(handle["matrix/features/id"].shape[0])
            nnz = int(handle["matrix/data"].shape[0])
        if shape != (feature_count, barcode_count) or barcode_count != len(metadata):
            raise RuntimeError(f"rescue H5/metadata closure failed: {cancer}")
        record = {
            "cancer_id": cancer,
            "formal_eligible": True,
            "blocking_reason": None,
            "dataset_id": formal_dataset_id,
            "source_dataset_ids": dataset_ids,
            "multiple_source_datasets_preserved": len(dataset_ids) > 1,
            "h5_path": str(h5_path),
            "h5_sha256": sha256(h5_path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "annotation_path": str(annotation),
            "annotation_sha256": sha256(annotation),
            "exact_membership_path": str(membership),
            "exact_membership_sha256": sha256(membership),
            "measurement_scale": "raw_counts",
            "cells": barcode_count,
            "features": feature_count,
            "nnz": nnz,
            "fresh_direct_id_lncrna_feature_count": int(success["lncrna_features"]),
            "fresh_rescue": True,
            "source_success_path": str(root / "SUCCESS.json"),
            "source_success_sha256": sha256(root / "SUCCESS.json"),
        }
        per_cancer[cancer] = record
        new_receipts[cancer] = record

    formal = sorted(old_formal | set(NEW_SOURCES))
    universe = set(per_cancer)
    if len(formal) != 23 or len(universe) != 33 or set(TYPED_UNAVAILABLE) != universe - set(formal):
        raise RuntimeError("r11 23+10 scope partition closure failed")
    for cancer, reason in TYPED_UNAVAILABLE.items():
        record = dict(per_cancer[cancer])
        record.update(formal_eligible=False, blocking_reason=reason)
        per_cancer[cancer] = record

    old_manifest = pd.read_parquet(args.old_r7_root / "dataset_manifest_33c.parquet")
    if set(old_manifest.cancer_id.astype(str)) != universe:
        raise RuntimeError("old dataset manifest universe drift")
    manifest = old_manifest.copy()
    for column in ("h5_path", "metadata_path", "dataset_id", "measurement_scale", "blocking_reason"):
        if column not in manifest:
            manifest[column] = None
    for cancer in universe:
        record = per_cancer[cancer]
        mask = manifest.cancer_id.astype(str).eq(cancer)
        manifest.loc[mask, "formal_eligible"] = bool(record["formal_eligible"])
        for column in ("h5_path", "metadata_path", "dataset_id", "measurement_scale", "blocking_reason"):
            if column in record:
                manifest.loc[mask, column] = record[column]
    manifest_path = output / "dataset_manifest_33c.parquet"
    manifest.to_parquet(manifest_path, index=False, compression="zstd")

    run_status = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_RUN_STATUS_V1",
        "status": "PASS_INPUT_CONTRACT_READY",
        "formal_eligible_cancers": formal,
        "formal_eligible_cancer_count": len(formal),
        "typed_unavailable": TYPED_UNAVAILABLE,
        "typed_unavailable_cancer_count": len(TYPED_UNAVAILABLE),
        "derived_assets_bound": False,
        "historical_assets_relabelled_fresh": False,
        "per_cancer": per_cancer,
    }
    atomic_json(output / "RUN_STATUS.json", run_status)
    handoff = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R11_TRAINING_HANDOFF_V1",
        "status": "PASS",
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": sha256(manifest_path),
        "historical_assets_relabelled_fresh": False,
        "assets": {
            "historical_predictions": {"path": None},
            "historical_rankings": {"path": None},
            "historical_checkpoints": {"path": None},
            "historical_trajectory": {"path": None},
        },
    }
    atomic_json(output / "TRAINING_HANDOFF.json", handoff)
    formal_cancers = []
    for cancer in formal:
        record = per_cancer[cancer]
        with h5py.File(record["h5_path"], "r") as handle:
            shape = [int(value) for value in handle["matrix/shape"][:]]
        formal_cancers.append({"cancer_id": cancer, "matrix_header": {"features": shape[0], "cells": shape[1]}})
    preflight = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_SERVER_PREFLIGHT_V1",
        "status": "PASS",
        "r7_root": str(output),
        "all_formal_raw_schemas_pass": True,
        "safe_to_start_fresh_cell_level_compute": True,
        "heavy_recompute_started": False,
        "formal_cancers": formal_cancers,
        "new_rescue_cancers": sorted(NEW_SOURCES),
        "new_rescue_receipts": new_receipts,
    }
    atomic_json(output / "SERVER_PREFLIGHT.json", preflight)
    atomic_json(output / "SUCCESS.json", {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_CONTRACT_SUCCESS_V1",
        "status": "SUCCESS",
        "formal_eligible_cancer_count": len(formal),
        "typed_unavailable_cancer_count": len(TYPED_UNAVAILABLE),
        "new_rescue_cancers": sorted(NEW_SOURCES),
        "run_status_sha256": sha256(output / "RUN_STATUS.json"),
        "preflight_sha256": sha256(output / "SERVER_PREFLIGHT.json"),
        "dataset_manifest_sha256": sha256(manifest_path),
    })
    print((output / "SUCCESS.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
