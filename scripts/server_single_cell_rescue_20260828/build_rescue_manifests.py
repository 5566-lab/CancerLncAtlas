#!/usr/bin/env python3
"""Materialise immutable source, exclusion, disk, and typed-gap manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_ROOT = Path("./data/CancerLncAtlas")
PUBLIC_ROOT = Path("./data/CancerLncAtlas")
RUN_ID = "single_cell_rescue_20260828"
LEGACY = ["BLCA", "BRCA", "COAD", "LIHC", "LUAD", "OV", "PAAD", "PRAD", "STAD", "THCA"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str, cancer: str | None = None) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "role": role, "cancer_id": cancer, "exists": False}
    info = path.stat()
    return {
        "path": str(path),
        "role": role,
        "cancer_id": cancer,
        "exists": True,
        "bytes": info.st_size,
        "mtime_utc": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} {EXPECTED_ROOT}")
    root = Path(sys.argv[1]).resolve()
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing output root outside {EXPECTED_ROOT}: {root}")
    manifest_dir = root / "manifests" / RUN_ID
    staging_dir = root / "staging" / RUN_ID
    manifest_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    disk = shutil.disk_usage(root)
    disk_manifest = {
        "format": "CANCERLNCATLAS_SINGLE_CELL_RESCUE_DISK_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(root),
        "workspace_resolved": str(root.resolve()),
        "public8_source_policy": "READ_ONLY_NO_SOURCE_MUTATION",
        "bytes_total": disk.total,
        "bytes_used": disk.used,
        "bytes_free": disk.free,
    }
    write_json(manifest_dir / "disk_and_workspace.json", disk_manifest)

    sources: list[dict[str, object]] = []
    for cancer in LEGACY:
        sources.append(file_record(PUBLIC_ROOT / "raw" / "single_cell" / f"01_{cancer}_processed_seurat.rds", "legacy_seurat_rds", cancer))
        sources.append(file_record(PUBLIC_ROOT / "processed" / "sc_tool_input" / cancer / "raw_feature_bc_matrix.h5", "legacy_h5_cache", cancer))
    sources.extend([
        file_record(PUBLIC_ROOT / "raw" / "single_cell_23" / "GSE297041" / "GSE297041_CESC_18_scRNA_rmdoublet_0.2_cluster_ident_without_anchor.rds", "CESC_GSE297041_replacement_candidate", "CESC"),
        file_record(PUBLIC_ROOT / "processed" / "sc_tool_input" / "CESC" / "raw_feature_bc_matrix.h5", "CESC_current_formal_h5", "CESC"),
        file_record(PUBLIC_ROOT / "raw" / "single_cell_23" / "GSE299623" / "GSE299623_RAW.tar", "UCS_GSE299623_raw_tar", "UCS"),
    ])
    source_manifest = {
        "format": "CANCERLNCATLAS_SINGLE_CELL_RESCUE_SOURCE_MANIFEST_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(PUBLIC_ROOT),
        "source_root_access_policy": "READ_ONLY",
        "output_root": str(root),
        "files": sources,
    }
    write_json(manifest_dir / "source_files.json", source_manifest)
    with (manifest_dir / "source_files.tsv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["cancer_id", "role", "path", "exists", "bytes", "mtime_utc", "sha256"]
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sources)

    exclusions = [{
        "cancer_id": "UCS",
        "dataset_id": "SC_GSE299623_UCS_RESCUE",
        "accession": "GSE299623",
        "sample_accession": "GSM9042132",
        "source_sample_label": "CS6",
        "observed_tar_prefix": "GSM9042132_A2_1",
        "action": "EXCLUDE_FROM_RESCUE_STAGING_ONLY",
        "raw_source_deleted": False,
        "reason": "USER_REPORTED_DATABASE_PROBLEM_REQUIRES_SAMPLE_EXCLUSION",
        "identity_evidence": "NCBI_GEO_GSE299623_SAMPLE_TABLE_GSM9042132_EQUALS_CS6",
    }]
    exclusion_path = staging_dir / "UCS_GSE299623_EXCLUSIONS.tsv"
    with exclusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exclusions[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(exclusions)

    typed_gaps = [
        {
            "cancer_id": "KICH",
            "current_typed_status": "INSUFFICIENT_DONOR_REPLICATION",
            "required_rescue": "ADDITIONAL_INDEPENDENT_KICH_DONORS_PER_COMPARTMENT",
            "do_not_substitute": "KIRC_OR_OTHER_RCC_SUBTYPES",
            "download_policy": "SEARCH_PROCESSED_COUNTS_AND_METADATA_BEFORE_FASTQ",
        },
        {
            "cancer_id": "KIRP",
            "current_typed_status": "INSUFFICIENT_DONOR_REPLICATION",
            "required_rescue": "ADDITIONAL_INDEPENDENT_KIRP_DONORS_PER_COMPARTMENT",
            "do_not_substitute": "KIRC_OR_CC_RCC",
            "download_policy": "SEARCH_PROCESSED_COUNTS_AND_METADATA_BEFORE_FASTQ",
        },
        {
            "cancer_id": "TGCT",
            "current_typed_status": "NO_CONTEXT_WITH_MINIMUM_DONOR_REPLICATION",
            "required_rescue": "RETEST_BROAD_COMPARTMENTS_THEN_ADD_INDEPENDENT_DONORS_IF_STILL_FAILING",
            "do_not_substitute": "MORE_CELLS_FROM_EXISTING_DONORS",
            "download_policy": "NO_DOWNLOAD_UNTIL_CONTEXT_COLLAPSE_REAUDIT",
        },
        {
            "cancer_id": "UCS",
            "current_typed_status": "LOW_LNCRNA_FEATURE_UNIVERSE_AND_SAMPLE_EXCLUSION_REQUIRED",
            "required_rescue": "REBUILD_FROM_GSE299623_RAW_MTX_EXCLUDING_GSM9042132_CS6_AND_REMAP_FULL_FEATURE_IDS",
            "do_not_substitute": "UCEC_OR_UNFILTERED_CS6",
            "download_policy": "NO_DOWNLOAD_CURRENT_RAW_TAR_PRESENT",
        },
        {
            "cancer_id": "UVM",
            "current_typed_status": "LOW_LNCRNA_FEATURE_UNIVERSE",
            "required_rescue": "LOCATE_FULL_TRANSCRIPTOME_COUNTS_OR_REPROCESS_RAW_READS_WITH_CURRENT_GENCODE",
            "do_not_substitute": "SKCM_OR_CUTANEOUS_MELANOMA",
            "download_policy": "PROCESSED_COUNTS_FIRST_FASTQ_ONLY_IF_UNAVAILABLE",
        },
    ]
    typed_path = manifest_dir / "typed_supplement_requirements.tsv"
    with typed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(typed_gaps[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(typed_gaps)
    write_json(manifest_dir / "typed_supplement_requirements.json", {
        "format": "CANCERLNCATLAS_SINGLE_CELL_TYPED_SUPPLEMENT_REQUIREMENTS_V1",
        "rows": typed_gaps,
    })

    outputs = [
        manifest_dir / "disk_and_workspace.json",
        manifest_dir / "source_files.json",
        manifest_dir / "source_files.tsv",
        exclusion_path,
        typed_path,
        manifest_dir / "typed_supplement_requirements.json",
    ]
    checksums = manifest_dir / "initial_manifest_sha256.tsv"
    with checksums.open("w", encoding="utf-8") as handle:
        handle.write("sha256\tpath\n")
        for path in outputs:
            handle.write(f"{sha256(path)}\t{path}\n")
    print(f"RESULT_PATH\t{manifest_dir}")
    print(f"RESULT_SHA256\t{sha256(checksums)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
