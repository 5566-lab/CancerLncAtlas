#!/usr/bin/env python3
"""Audit preprocessing and annotation evidence for all formal SC cohorts.

This audit deliberately separates a Harmony-corrected embedding used by the
preprocessing/annotation workflow from the expression values used to form
metacells and donor-resolved association targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd


FORMAT = "CANCERLNCATLAS_V32_SC_PREPROCESSING_AUDIT_V1"
EXPECTED_CANCERS = 28
EXPECTED_PREPROCESSING_METHOD = "Harmony_by_sample_within_cancer"
EXPECTED_BATCH_SCOPE = "WITHIN_CANCER_SAMPLE_BATCHES_IN_EMBEDDING_SPACE"
DEFAULT_ACTIVE_STATUS = Path(
    "/dell_2/DSC/CancerLncAtlas/runtime/"
    "single_cell_cell_level_r11_rescued_20260903_r1/RUN_STATUS.json"
)
DEFAULT_REPAIR_ROOT = Path("/dsk2/user/dengsc/sc_preprocessing_repair_20261004_r1")
BOOLEAN_EVIDENCE = (
    "preprocessing_pass_qc",
    "batch_correction_applied",
    "annotation_completed",
)
IDENTITY_COLUMNS = (
    "cell_id",
    "dataset_id",
    "cancer_id",
    "patient_id",
    "sample_id",
    "cell_type_major",
    "compartment",
    "lineage",
)
ANNOTATION_COLUMNS = (
    "annotation_mapping_status",
    "annotation_rule_id",
    "annotation_authority_version",
    "annotation_authority_sha256",
    "reference_annotation_source",
)


def sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"cannot hash missing or symlinked input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing or symlinked {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_table(frame: pd.DataFrame, parquet: Path, tsv: Path) -> None:
    parquet_temporary = parquet.with_suffix(parquet.suffix + ".partial")
    tsv_temporary = tsv.with_suffix(tsv.suffix + ".partial")
    frame.to_parquet(parquet_temporary, index=False)
    frame.to_csv(tsv_temporary, sep="\t", index=False, lineterminator="\n")
    os.replace(parquet_temporary, parquet)
    os.replace(tsv_temporary, tsv)


def verify_declared_file(path_value: object, digest_value: object, label: str) -> dict:
    path = Path(str(path_value))
    expected = str(digest_value)
    if len(expected) != 64:
        raise RuntimeError(f"{label} lacks a SHA256 declaration")
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} hash drift")
    return {"path": str(path), "sha256": observed, "bytes": path.stat().st_size}


def nonblank(frame: pd.DataFrame, column: str, eligible: pd.Series) -> pd.Series:
    return frame.loc[eligible, column].fillna("").astype(str).str.strip().ne("")


def unique_nonblank(frame: pd.DataFrame, column: str, eligible: pd.Series) -> list[str]:
    values = frame.loc[eligible, column].fillna("").astype(str).str.strip()
    return sorted(values.loc[values.ne("")].unique().tolist())


def audit_cancer(
    cancer_id: str,
    record: dict,
    repair_root: Path,
    annotation_authority: dict,
) -> dict[str, object]:
    if record.get("formal_eligible") is not True:
        raise RuntimeError(f"{cancer_id}: cohort is not formally eligible")
    gates = {
        "preprocessing_required": True,
        "preprocessing_complete": True,
        "preprocessing_status": "VERIFIED_COMPLETED",
        "training_input_binding_ready": True,
    }
    for key, expected in gates.items():
        if record.get(key) != expected:
            raise RuntimeError(f"{cancer_id}: active status {key} is not {expected!r}")

    metadata_path = Path(str(record.get("metadata_path", "")))
    metadata_file = verify_declared_file(
        metadata_path, record.get("metadata_sha256"), f"{cancer_id} metadata"
    )
    receipt_path = Path(str(record.get("preprocessing_receipt_path", "")))
    receipt_file = verify_declared_file(
        receipt_path,
        record.get("preprocessing_receipt_sha256"),
        f"{cancer_id} preprocessing receipt",
    )
    receipt = read_json(receipt_path, f"{cancer_id} preprocessing receipt")
    if (
        receipt.get("status") != "PASS"
        or str(receipt.get("cancer_id", "")).upper() != cancer_id
        or receipt.get("target_host") != "149"
    ):
        raise RuntimeError(f"{cancer_id}: invalid preprocessing receipt identity/status")
    if receipt.get("batch_correction_scope") != EXPECTED_BATCH_SCOPE:
        raise RuntimeError(f"{cancer_id}: batch correction was not cancer scoped")
    if receipt.get("expression_matrix_batch_corrected") is not False:
        raise RuntimeError(
            f"{cancer_id}: receipt improperly claims the expression matrix is Harmony corrected"
        )
    source_annotation = verify_declared_file(
        receipt.get("source_annotation_path"),
        receipt.get("source_annotation_sha256"),
        f"{cancer_id} source annotation",
    )
    receipt_metadata = Path(str(receipt.get("metadata_target", "")))
    if sha256(receipt_metadata) != metadata_file["sha256"]:
        raise RuntimeError(f"{cancer_id}: receipt metadata differs from active metadata")
    supporting_files = {}
    for prefix in ("completion_log", "success", "object_audit"):
        path_key, sha_key = f"{prefix}_path", f"{prefix}_sha256"
        if path_key in receipt or sha_key in receipt:
            supporting_files[prefix] = verify_declared_file(
                receipt.get(path_key), receipt.get(sha_key), f"{cancer_id} {prefix}"
            )

    job_root = repair_root / "jobs" / cancer_id
    manifest_path = job_root / "dataset_manifest_33c.parquet"
    handoff_path = job_root / "TRAINING_HANDOFF.json"
    handoff = read_json(handoff_path, f"{cancer_id} training handoff")
    manifest_file = verify_declared_file(
        manifest_path,
        handoff.get("dataset_manifest_sha256"),
        f"{cancer_id} dataset manifest",
    )
    manifest = pd.read_parquet(manifest_path)
    selected = manifest.loc[
        manifest.cancer_id.astype(str).str.upper().eq(cancer_id)
    ]
    if len(selected) != 1 or not bool(selected.iloc[0].formal_eligible):
        raise RuntimeError(f"{cancer_id}: formal manifest row is absent or ambiguous")
    manifest_row = selected.iloc[0]
    measurement_scale = str(manifest_row.measurement_scale)
    if measurement_scale not in {"raw_counts", "log_normalized"}:
        raise RuntimeError(f"{cancer_id}: unsupported expression scale {measurement_scale}")
    dataset_id = str(record.get("dataset_id", ""))
    if str(manifest_row.dataset_id) != dataset_id:
        raise RuntimeError(f"{cancer_id}: dataset ID differs between status and manifest")

    metadata = pd.read_parquet(metadata_path)
    required = {
        "analysis_eligible_tumor_context",
        "preprocessing_evidence_sha256",
        "preprocessing_method",
        *BOOLEAN_EVIDENCE,
        *IDENTITY_COLUMNS,
        *ANNOTATION_COLUMNS,
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise RuntimeError(f"{cancer_id}: metadata lacks {missing}")
    for column in (*BOOLEAN_EVIDENCE, "analysis_eligible_tumor_context"):
        if not pd.api.types.is_bool_dtype(metadata[column]) or metadata[column].isna().any():
            raise RuntimeError(f"{cancer_id}: {column} must be non-null boolean")
    if metadata.cell_id.astype(str).duplicated().any():
        raise RuntimeError(f"{cancer_id}: cell IDs are duplicated")
    eligible = metadata.analysis_eligible_tumor_context.eq(True)
    if not eligible.any():
        raise RuntimeError(f"{cancer_id}: no training-eligible cells")
    for column in BOOLEAN_EVIDENCE:
        if not metadata.loc[eligible, column].eq(True).all():
            raise RuntimeError(f"{cancer_id}: eligible cell fails {column}")
    for column in (*IDENTITY_COLUMNS, *ANNOTATION_COLUMNS, "preprocessing_method"):
        if not nonblank(metadata, column, eligible).all():
            raise RuntimeError(f"{cancer_id}: eligible cell has blank {column}")
    if set(metadata.loc[eligible, "cancer_id"].astype(str).str.upper()) != {cancer_id}:
        raise RuntimeError(f"{cancer_id}: eligible cancer identity drift")
    if set(metadata.loc[eligible, "dataset_id"].astype(str)) != {dataset_id}:
        raise RuntimeError(f"{cancer_id}: eligible dataset identity drift")
    if set(metadata.loc[eligible, "preprocessing_evidence_sha256"].astype(str)) != {
        receipt_file["sha256"]
    }:
        raise RuntimeError(f"{cancer_id}: per-cell preprocessing receipt binding drift")
    methods = unique_nonblank(metadata, "preprocessing_method", eligible)
    if methods != [EXPECTED_PREPROCESSING_METHOD]:
        raise RuntimeError(f"{cancer_id}: unexpected preprocessing method {methods}")
    annotation_sources = unique_nonblank(
        metadata, "reference_annotation_source", eligible
    )
    if annotation_sources != [str(receipt.get("annotation_source", ""))]:
        raise RuntimeError(f"{cancer_id}: annotation source differs from receipt")
    authority_sha = str(record.get("annotation_authority_sha256", ""))
    authority_version = str(annotation_authority.get("authority_version", ""))
    if set(metadata.loc[eligible, "annotation_authority_sha256"].astype(str)) != {
        authority_sha
    }:
        raise RuntimeError(f"{cancer_id}: per-cell annotation authority hash drift")
    if set(metadata.loc[eligible, "annotation_authority_version"].astype(str)) != {
        authority_version
    }:
        raise RuntimeError(f"{cancer_id}: per-cell annotation authority version drift")
    if (
        authority_sha != str(annotation_authority.get("annotation_authority_sha256", ""))
        or authority_version != str(annotation_authority.get("authority_version", ""))
    ):
        raise RuntimeError(f"{cancer_id}: annotation authority declaration drift")
    if "expression_scale" in metadata:
        expression_scales = unique_nonblank(metadata, "expression_scale", eligible)
        if expression_scales != [measurement_scale]:
            raise RuntimeError(
                f"{cancer_id}: metadata expression scale {expression_scales} "
                f"!= {measurement_scale}"
            )
        expression_scale_binding = "METADATA_AND_FORMAL_MANIFEST_AGREE"
    else:
        expression_scale_binding = "FORMAL_MANIFEST_AUTHORITY"

    optional_verification = {}
    for column in ("patient_id_verified", "source_review_include"):
        if column in metadata:
            if not pd.api.types.is_bool_dtype(metadata[column]):
                raise RuntimeError(f"{cancer_id}: {column} must be boolean")
            bad = int((~metadata.loc[eligible, column].eq(True)).sum())
            if bad:
                raise RuntimeError(f"{cancer_id}: {bad} eligible cells fail {column}")
            optional_verification[column] = True
        else:
            optional_verification[column] = "NOT_PRESENT_NOT_REQUIRED_BY_ACTIVE_SCHEMA"

    mapping_counts = (
        metadata.loc[eligible, "annotation_mapping_status"]
        .astype(str)
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    reference_status_counts = (
        metadata.loc[eligible, "reference_annotation_status"]
        .astype(str)
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
        if "reference_annotation_status" in metadata
        else {}
    )
    return {
        "cancer_id": cancer_id,
        "dataset_id": dataset_id,
        "status": "PASS",
        "raw_cells": int(len(metadata)),
        "eligible_cells": int(eligible.sum()),
        "eligible_donors": int(metadata.loc[eligible, "patient_id"].nunique()),
        "eligible_samples": int(metadata.loc[eligible, "sample_id"].nunique()),
        "eligible_celltypes": int(metadata.loc[eligible, "cell_type_major"].nunique()),
        "measurement_scale": measurement_scale,
        "measurement_scale_binding": expression_scale_binding,
        "preprocessing_method": methods[0],
        "preprocessing_origin": str(receipt.get("preprocessing_origin", "")),
        "batch_correction_scope": str(receipt["batch_correction_scope"]),
        "expression_matrix_batch_corrected": False,
        "annotation_source": annotation_sources[0],
        "annotation_authority_version": authority_version,
        "annotation_authority_sha256": authority_sha,
        "annotation_mapping_status_counts": mapping_counts,
        "reference_annotation_status_counts": reference_status_counts,
        "eligible_preprocessing_pass_qc": True,
        "eligible_batch_correction_applied": True,
        "eligible_annotation_completed": True,
        "eligible_patient_id_complete": True,
        "eligible_sample_id_complete": True,
        "eligible_celltype_complete": True,
        "donor_identity_column": "patient_id",
        "optional_identity_verification": optional_verification,
        "metadata": metadata_file,
        "preprocessing_receipt": receipt_file,
        "source_annotation": source_annotation,
        "supporting_files": supporting_files,
        "dataset_manifest": manifest_file,
    }


def flatten(report: dict[str, object]) -> dict[str, object]:
    return {
        "cancer_id": report["cancer_id"],
        "dataset_id": report["dataset_id"],
        "status": report["status"],
        "raw_cells": report["raw_cells"],
        "eligible_cells": report["eligible_cells"],
        "eligible_donors": report["eligible_donors"],
        "eligible_samples": report["eligible_samples"],
        "eligible_celltypes": report["eligible_celltypes"],
        "measurement_scale": report["measurement_scale"],
        "measurement_scale_binding": report["measurement_scale_binding"],
        "preprocessing_method": report["preprocessing_method"],
        "preprocessing_origin": report["preprocessing_origin"],
        "batch_correction_scope": report["batch_correction_scope"],
        "expression_matrix_batch_corrected": report[
            "expression_matrix_batch_corrected"
        ],
        "annotation_source": report["annotation_source"],
        "annotation_authority_version": report["annotation_authority_version"],
        "annotation_authority_sha256": report["annotation_authority_sha256"],
        "annotation_mapping_status_counts": json.dumps(
            report["annotation_mapping_status_counts"], sort_keys=True
        ),
        "reference_annotation_status_counts": json.dumps(
            report["reference_annotation_status_counts"], sort_keys=True
        ),
        "eligible_preprocessing_pass_qc": report["eligible_preprocessing_pass_qc"],
        "eligible_batch_correction_applied": report[
            "eligible_batch_correction_applied"
        ],
        "eligible_annotation_completed": report["eligible_annotation_completed"],
        "eligible_patient_id_complete": report["eligible_patient_id_complete"],
        "eligible_sample_id_complete": report["eligible_sample_id_complete"],
        "eligible_celltype_complete": report["eligible_celltype_complete"],
        "donor_identity_column": report["donor_identity_column"],
        "metadata_path": report["metadata"]["path"],
        "metadata_sha256": report["metadata"]["sha256"],
        "preprocessing_receipt_path": report["preprocessing_receipt"]["path"],
        "preprocessing_receipt_sha256": report["preprocessing_receipt"]["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--active-status", type=Path, default=DEFAULT_ACTIVE_STATUS)
    parser.add_argument("--repair-root", type=Path, default=DEFAULT_REPAIR_ROOT)
    args = parser.parse_args()
    if socket.gethostname() != "149":
        raise SystemExit("WRONG_HOST")
    if not args.output_root.is_dir() or args.output_root.is_symlink():
        raise SystemExit("OUTPUT_ROOT_MISSING_OR_UNSAFE")

    active = read_json(args.active_status, "active RUN_STATUS")
    records = active.get("per_cancer")
    if not isinstance(records, dict):
        raise RuntimeError("active RUN_STATUS lacks per_cancer records")
    cancers = sorted(
        str(cancer).upper()
        for cancer, record in records.items()
        if isinstance(record, dict) and record.get("formal_eligible") is True
    )
    if len(cancers) != EXPECTED_CANCERS:
        raise RuntimeError(
            f"expected {EXPECTED_CANCERS} active formal cancers, observed {len(cancers)}"
        )

    authority_path = Path(str(active.get("annotation_authority_path", "")))
    authority_file_digests = {
        str(record.get("annotation_authority_file_sha256", ""))
        for cancer, record in records.items()
        if str(cancer).upper() in cancers and isinstance(record, dict)
    }
    if len(authority_file_digests) != 1:
        raise RuntimeError("per-cancer annotation authority file hashes disagree")
    authority_file = verify_declared_file(
        authority_path,
        next(iter(authority_file_digests)),
        "annotation authority",
    )
    authority = read_json(authority_path, "annotation authority")
    if (
        authority.get("format") != "CC_HHGT_V3_2_SINGLE_CELL_ANNOTATION_V2"
        or authority.get("annotation_authority_sha256")
        != active.get("annotation_authority_sha256")
        or authority.get("authority_version")
        != active.get("annotation_authority_version")
    ):
        raise RuntimeError("active annotation authority contract drift")

    reports = [
        audit_cancer(cancer, records[cancer], args.repair_root, authority)
        for cancer in cancers
    ]
    table = pd.DataFrame([flatten(report) for report in reports]).sort_values(
        "cancer_id", kind="mergesort"
    )
    parquet_path = args.output_root / "SINGLE_CELL_PREPROCESSING_AUDIT.parquet"
    tsv_path = args.output_root / "SINGLE_CELL_PREPROCESSING_AUDIT.tsv"
    atomic_table(table, parquet_path, tsv_path)
    raw_cells = int(sum(int(report["raw_cells"]) for report in reports))
    eligible_cells = int(sum(int(report["eligible_cells"]) for report in reports))
    scales = table.measurement_scale.value_counts().sort_index().astype(int).to_dict()
    result = {
        "format": FORMAT,
        "target_host": "149",
        "status": "PASS",
        "auditor_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "active_cancers": cancers,
        "active_cancer_count": len(cancers),
        "raw_cells": raw_cells,
        "eligible_training_cells": eligible_cells,
        "eligible_cells_failing_preprocessing_pass_qc": 0,
        "eligible_cells_failing_batch_correction_applied": 0,
        "eligible_cells_failing_annotation_completed": 0,
        "eligible_cells_with_blank_patient_id": 0,
        "eligible_cells_with_blank_sample_id": 0,
        "eligible_cells_with_blank_cell_type": 0,
        "measurement_scale_counts": scales,
        "preprocessing_contract": {
            "method": EXPECTED_PREPROCESSING_METHOD,
            "scope": EXPECTED_BATCH_SCOPE,
            "all_eligible_cells_pass": True,
            "all_eligible_cells_batch_correction_applied": True,
            "all_eligible_cells_annotation_completed": True,
        },
        "expression_semantics": {
            "harmony_embedding_used_as_expression_matrix": False,
            "expression_matrix_batch_corrected": False,
            "training_expression_source": (
                "ORIGINAL_DECLARED_RAW_COUNTS_OR_LOG_NORMALIZED_VALUES; "
                "HARMONY_COORDINATES_ARE_NOT_SUBSTITUTED_FOR_EXPRESSION"
            ),
            "metacell_target": "WITHIN_DONOR_METACELL_SPEARMAN",
            "cross_donor_expression_correction_required": False,
            "cross_donor_rationale": (
                "Expression and pathway activity are ranked only among metacells "
                "inside one donor and cell type. Donor effects are then carried as "
                "separate observations and combined after the within-donor target is "
                "computed; raw expression values are never compared across donors."
            ),
            "leakage_rationale": (
                "Joint cross-donor correction could use held-out donor information. "
                "The formal target instead preserves donor-specific heterogeneity and "
                "uses donor-blocked outer folds."
            ),
        },
        "annotation_semantics": {
            "authority": authority_file,
            "authority_version": authority["authority_version"],
            "authority_semantic_sha256": authority["annotation_authority_sha256"],
            "celltype_complete_for_every_eligible_cell": True,
            "ontology_note": (
                "UNMAPPED_EXPLICIT is an explicit canonical annotation without a "
                "claimed exact Cell Ontology mapping; it is not a missing cell type."
            ),
        },
        "active_run_status": {
            "path": str(args.active_status),
            "sha256": sha256(args.active_status),
        },
        "table_files": {
            "parquet": {
                "path": str(parquet_path),
                "sha256": sha256(parquet_path),
                "bytes": parquet_path.stat().st_size,
            },
            "tsv": {
                "path": str(tsv_path),
                "sha256": sha256(tsv_path),
                "bytes": tsv_path.stat().st_size,
            },
        },
        "cancers": reports,
        "finished_unix": time.time(),
    }
    output = args.output_root / "SINGLE_CELL_PREPROCESSING_AUDIT.json"
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "cancers": result["active_cancer_count"],
        "raw_cells": raw_cells,
        "eligible_training_cells": eligible_cells,
        "measurement_scale_counts": scales,
        "output": str(output),
        "sha256": sha256(output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
