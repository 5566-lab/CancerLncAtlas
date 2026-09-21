#!/usr/bin/env python3
"""Fail-closed finalizer for the 2026-08-28 single-cell rescue audit."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_ROOT = Path("./data/CancerLncAtlas")
RUN_ID = "single_cell_rescue_20260828"
LEGACY = {"BLCA", "BRCA", "COAD", "LIHC", "LUAD", "OV", "PAAD", "PRAD", "STAD", "THCA"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_source_record(record: dict) -> dict:
    """Rehash a rescue source; size/mtime are never accepted as identity."""

    path = Path(record["path"])
    if not record.get("exists") or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Source disappeared or became unsafe after audit: {path}")
    expected_sha = str(record.get("sha256", "")).lower()
    if not SHA256_RE.fullmatch(expected_sha):
        raise RuntimeError(f"Source manifest lacks a valid SHA-256: {path}")
    stat = path.stat()
    current_mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    observed_sha = sha256(path)
    if (
        stat.st_size != int(record["bytes"])
        or current_mtime != str(record["mtime_utc"])
        or observed_sha != expected_sha
    ):
        raise RuntimeError(f"Source size/mtime/SHA-256 changed after read-only audit: {path}")
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_utc": current_mtime,
        "sha256": observed_sha,
        "identity_gate": "SIZE_MTIME_AND_FULL_SHA256",
        "unchanged": True,
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} {EXPECTED_ROOT}")
    root = Path(sys.argv[1]).resolve()
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing output root outside {EXPECTED_ROOT}: {root}")
    manifest = root / "manifests" / RUN_ID
    required = [
        manifest / "source_files.json",
        manifest / "seurat_source_audit.tsv",
        manifest / "rds_h5_signature_comparison.json",
        manifest / "legacy_h5_aligned_metadata_gate.json",
        manifest / "rescue_rds_lncrna_feature_universe.json",
        manifest / "typed_gap_context_reaudit.json",
        manifest / "UCS_GSE299623_EXCLUSION_GATE.txt",
        manifest / "UCS_GSE299623_METADATA_EXCLUSION_GATE.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    with (manifest / "seurat_source_audit.tsv").open("r", encoding="utf-8", newline="") as handle:
        seurat_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(seurat_rows) != 11 or any(not row["status"].startswith("RDS_AUDIT_COMPLETE") for row in seurat_rows):
        raise SystemExit("Seurat audit is not complete 11/11")

    comparison = read_json(manifest / "rds_h5_signature_comparison.json")["rows"]
    comparison_by_cancer = {row["cancer_id"]: row for row in comparison}
    legacy_comparison_pass = {
        cancer for cancer in LEGACY
        if comparison_by_cancer.get(cancer, {}).get("status") == "PASS_AGGREGATE_COUNTS_AND_IDENTITIES_EXACT"
    }
    if legacy_comparison_pass != LEGACY:
        raise SystemExit(f"Legacy RDS/H5 comparison did not pass: {sorted(LEGACY - legacy_comparison_pass)}")
    if comparison_by_cancer.get("CESC", {}).get("status") != "EXPECTED_REPLACEMENT_SOURCE_DIFFERS_FROM_CURRENT_H5_REBUILD_REQUIRED":
        raise SystemExit("CESC replacement/current-H5 relationship is not typed as expected")

    aligned = read_json(manifest / "legacy_h5_aligned_metadata_gate.json")
    aligned_rows = aligned["rows"]
    if len(aligned_rows) != 10 or any(
        row["status"] != "PASS_H5_ALIGNED_METADATA_CANDIDATE_PATIENT_MAPPING_REQUIRED"
        or row["patient_id_verified"] is not False
        for row in aligned_rows
    ):
        raise SystemExit("Legacy aligned metadata gate is not pass-10/10 patient-unverified")

    ucs_matrix_gate = (manifest / "UCS_GSE299623_EXCLUSION_GATE.txt").read_text(encoding="utf-8")
    if "status=PASS" not in ucs_matrix_gate or "excluded_members=3" not in ucs_matrix_gate or "raw_source_deleted=false" not in ucs_matrix_gate:
        raise SystemExit("UCS matrix exclusion gate failed")
    ucs_metadata = read_json(manifest / "UCS_GSE299623_METADATA_EXCLUSION_GATE.json")
    if ucs_metadata.get("status") != "PASS" or ucs_metadata.get("output_donors") != 7 or ucs_metadata.get("raw_source_deleted") is not False:
        raise SystemExit("UCS metadata exclusion gate failed")

    feature_rows = {
        row["cancer_id"]: row
        for row in read_json(manifest / "rescue_rds_lncrna_feature_universe.json")["rows"]
    }
    if feature_rows.get("CESC", {}).get("gencode_v50_lncrna_features", 0) < 1000:
        raise SystemExit("CESC replacement did not rescue the minimum lncRNA universe")
    cesc_audit = next(row for row in seurat_rows if row["cancer_id"] == "CESC")
    if cesc_audit.get("celltype_source_column", "") not in ("", "NA"):
        raise SystemExit("CESC cell-type status changed; finalizer contract needs review")

    source_manifest = read_json(manifest / "source_files.json")
    source_postcheck = []
    for record in source_manifest["files"]:
        try:
            source_postcheck.append(verify_source_record(record))
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    typed = read_json(manifest / "typed_gap_context_reaudit.json")["rows"]
    typed_by_cancer = {row["cancer_id"]: row for row in typed}
    summary = {
        "format": "CANCERLNCATLAS_SINGLE_CELL_RESCUE_AUDIT_SUCCESS_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS_AUDIT_AND_STAGING_COMPLETE_NOT_FORMAL_TRAINING_READY",
        "workspace": str(root),
        "public8_source_policy": "READ_ONLY",
        "source_size_mtime_postcheck_pass": True,
        "source_full_sha256_postcheck_pass": True,
        "source_identity_gate": "SIZE_MTIME_AND_FULL_SHA256",
        "source_files_postchecked": len(source_postcheck),
        "legacy_seurat_audit_complete": "11/11_INCLUDING_CESC_REPLACEMENT",
        "legacy_rds_h5_count_identity_pass": "10/10_AFTER_LITERAL_BARCODE_PREFIX_NORMALIZATION",
        "legacy_h5_aligned_metadata_candidates": 10,
        "legacy_formal_promotion_allowed": False,
        "legacy_formal_blocker": "SOURCE_VERIFIED_SAMPLE_TO_PATIENT_MAPPING_REQUIRED",
        "cesc": {
            "replacement_cells": int(cesc_audit["cells"]),
            "replacement_case_candidates": int(cesc_audit["donor_candidates"]),
            "replacement_lncrna_features": feature_rows["CESC"]["gencode_v50_lncrna_features"],
            "replacement_lncrna_detected": feature_rows["CESC"]["gencode_v50_lncrna_detected"],
            "current_h5_rebuild_required": True,
            "celltype_annotation_required": True,
            "formal_ready": False,
        },
        "ucs": {
            "GSM9042132_CS6_excluded_from_staging": True,
            "excluded_cells_from_metadata": ucs_metadata["excluded_rows"],
            "remaining_cells_in_metadata": ucs_metadata["output_rows"],
            "remaining_donors": ucs_metadata["output_donors"],
            "raw_source_deleted": False,
            "full_feature_matrix_rebuild_required": True,
            "formal_ready": False,
        },
        "typed_gaps": {
            cancer: {
                "donors": typed_by_cancer[cancer]["donors"],
                "lncrna_features": typed_by_cancer[cancer]["lncrna_features_v32_explicit_id_reaudit"],
                "status": typed_by_cancer[cancer]["status"],
            }
            for cancer in ("KICH", "KIRP", "TGCT", "UCS", "UVM")
        },
        "formal_training_ready": False,
        "downloads_started": False,
        "fastq_downloads_started": False,
        "required_next_actions": [
            "source-verify legacy sample-to-patient mappings and collapse multi-sample donors",
            "annotate CESC cell types and rebuild its H5 from GSE297041",
            "rebuild UCS GSE299623 matrix without GSM9042132/CS6 and remap full GENCODE features",
            "add independent donors for KICH, KIRP, and TGCT until at least five donors share a valid context",
            "locate a full-transcriptome UVM source; use FASTQ only if processed counts are unavailable",
        ],
    }
    success = manifest / "SUCCESS.json"
    success.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    postcheck = manifest / "source_size_mtime_postcheck.json"
    postcheck.write_text(json.dumps({"status": "PASS", "files": source_postcheck}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_targets = sorted(
        path for path in manifest.iterdir()
        if path.is_file() and path.name not in {"DELIVERY_SHA256.tsv"}
    )
    checksum_path = manifest / "DELIVERY_SHA256.tsv"
    with checksum_path.open("w", encoding="utf-8") as handle:
        handle.write("sha256\tpath\n")
        for path in checksum_targets:
            handle.write(f"{sha256(path)}\t{path}\n")
    print(f"RESULT_PATH\t{success}")
    print(f"RESULT_SHA256\t{sha256(success)}")
    print(f"DELIVERY_SHA256\t{sha256(checksum_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
