#!/usr/bin/env python3
"""Fail-closed post-download gate for full33 GDC masked-segment CNV."""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--manifest-tsv", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--download-root", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--entity-intervals", required=True)
    parser.add_argument("--pathway-membership", required=True)
    parser.add_argument("--sample-gene-mutation", required=True)
    parser.add_argument("--sample-lncrna-mutation", required=True)
    parser.add_argument("--mc3", required=True)
    parser.add_argument("--core-embedding-manifest", required=True)
    parser.add_argument("--training-output-root", required=True)
    parser.add_argument("--streaming-output-root")
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser


def _shell_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in parts)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.gdc_segment_cnv import (
        TCGA_CANCERS,
        audit_selected_segment_download,
        full33_gate_implementation_hashes,
        materialize_verified_segment_staging,
        sha256_file,
    )
    from cc_hhgt.v32.patient_fold_authority import (
        validate_frozen_v32_patient_fold_binding,
    )
    import pandas as pd

    output = _resolve(root, args.output_root)
    if output.exists():
        raise RuntimeError(f"Full33 preflight refuses output reuse: {output}")
    manifest_tsv = _resolve(root, args.manifest_tsv)
    manifest_json = _resolve(root, args.manifest_json)
    download_root = _resolve(root, args.download_root)
    staging_root = _resolve(root, args.staging_root)
    binding_args = {
        "candidates": args.candidates,
        "patient_folds": args.patient_folds,
        "patient_fold_authority_receipt": args.patient_fold_authority_receipt,
        "entity_intervals": args.entity_intervals,
        "pathway_membership": args.pathway_membership,
        "sample_gene_mutation": args.sample_gene_mutation,
        "sample_lncrna_mutation": args.sample_lncrna_mutation,
        "mc3": args.mc3,
        "core_embedding_manifest": args.core_embedding_manifest,
    }
    binding_paths = {label: _resolve(root, value) for label, value in binding_args.items()}
    missing = [label for label, path in binding_paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Bound training inputs are missing: {missing}")
    audit = audit_selected_segment_download(manifest_tsv, manifest_json, download_root)
    observed_segment_cancers = {str(row["cancer_id"]).upper() for row in audit["records"]}
    if observed_segment_cancers != set(TCGA_CANCERS):
        raise RuntimeError(
            f"Selected segment manifest is not full33: missing={sorted(set(TCGA_CANCERS) - observed_segment_cancers)}"
        )
    candidate_cancers = set(
        pd.read_parquet(binding_paths["candidates"], columns=["cancer_id"])
        .cancer_id.astype(str).str.upper().unique()
    )
    fold_authority_audit = validate_frozen_v32_patient_fold_binding(
        binding_paths["patient_folds"],
        binding_paths["patient_fold_authority_receipt"],
    )
    folds = pd.read_csv(binding_paths["patient_folds"], sep="\t")
    fold_column = "patient_fold_id"
    fold_required = {
        "cancer_id", "sample_id", "patient_id", "patient_fold_id", "fold_seed"
    }
    if missing_fold_columns := sorted(fold_required - set(folds.columns)):
        raise RuntimeError(
            f"Patient fold manifest lacks explicit authority columns: {missing_fold_columns}"
        )
    folds = folds.copy()
    folds["cancer_id"] = folds.cancer_id.astype(str).str.upper()
    folds["patient_id"] = folds.patient_id.astype(str).str.strip()
    fold_patients = folds[
        ["cancer_id", "patient_id", "patient_fold_id", "fold_seed"]
    ].drop_duplicates()
    if fold_patients.duplicated(["cancer_id", "patient_id"]).any():
        raise RuntimeError("Patient fold authority maps one explicit patient to multiple folds")
    fold_cancers = set(folds.cancer_id.astype(str).str.upper().unique())
    fold_ids = set(pd.to_numeric(folds[fold_column], errors="raise").astype(int).unique())
    if candidate_cancers != set(TCGA_CANCERS) or fold_cancers != set(TCGA_CANCERS):
        raise RuntimeError("Candidate/fold authorities do not cover exactly the same full33 cancers")
    if fold_ids != set(range(5)):
        raise RuntimeError("Patient fold authority is not exactly five-fold 0..4")
    per_cancer_fold_counts = fold_patients.groupby("cancer_id", observed=True)[fold_column].nunique()
    if not per_cancer_fold_counts.eq(5).all():
        raise RuntimeError("At least one cancer lacks one or more patient folds")
    bindings = {
        label: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for label, path in binding_paths.items()
    }
    output.mkdir(parents=True)
    segment_by_patient = {}
    for record in audit["records"]:
        patient_id = str(record["case_submitter_id"]).replace(".", "-")[:12].upper()
        sample_patient_id = str(record["sample_submitter_id"]).replace(".", "-")[:12].upper()
        if patient_id != sample_patient_id:
            raise RuntimeError(f"Segment case/sample patient mismatch: {record['file_id']}")
        key = (str(record["cancer_id"]).upper(), patient_id)
        if key in segment_by_patient:
            raise RuntimeError(f"Multiple selected segment UUIDs normalize to patient key: {key}")
        segment_by_patient[key] = record
    callability_rows = []
    for row in fold_patients.sort_values(["cancer_id", "patient_id"]).itertuples(index=False):
        key = (str(row.cancer_id), str(row.patient_id))
        record = segment_by_patient.get(key)
        callability_rows.append(
            {
                "cancer_id": key[0],
                "patient_id": key[1],
                "patient_fold_id": int(getattr(row, fold_column)),
                "segment_manifest_available": record is not None,
                "selected_file_id": record["file_id"] if record else "",
                "cnv_callability_state": "SELECTED_SEGMENT_BOUND" if record else "TYPED_UNAVAILABLE",
                "cnv_unavailable_reason": "" if record else "NO_SELECTED_GDC_MASKED_SEGMENT_FOR_PATIENT",
            }
        )
    callability_path = output / "PATIENT_CNV_SEGMENT_CALLABILITY.tsv"
    with callability_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(callability_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(callability_rows)
    fold_patient_keys = {(row["cancer_id"], row["patient_id"]) for row in callability_rows}
    extra_selected_keys = sorted(set(segment_by_patient) - fold_patient_keys)
    staging = None
    staging_error = None
    gate_status = audit["status"]
    if audit["status"] == "READY":
        try:
            staging = materialize_verified_segment_staging(audit, staging_root)
            staging_manifest = output / "VERIFIED_STAGING_FILES.tsv"
            with staging_manifest.open("w", encoding="utf-8", newline="") as handle:
                columns = ["file_id", "cancer_id", "source_path", "staged_path", "size", "md5"]
                writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(staging["records"])
        except Exception as exc:
            gate_status = "BLOCKED"
            staging_error = {"type": type(exc).__name__, "message": str(exc)}
    retry_path = output / "RETRY_ONLY_MISSING_OR_INVALID.gdc_manifest.tsv"
    retry_columns = ["id", "filename", "md5", "size", "state"]
    with retry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=retry_columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {
                "id": row["file_id"],
                "filename": row["file_name"],
                "md5": row["md5sum"],
                "size": row["file_size"],
                "state": "submitted",
            }
            for row in audit["retry_rows"]
        )

    payload = {
        key: value for key, value in audit.items() if key not in {"columns", "retry_rows", "records"}
    }
    payload.update(
        {
            "status": gate_status,
            "format": "CC_HHGT_V3_2_FULL33_SEGMENT_DOWNLOAD_GATE_V1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "bindings": bindings,
            "implementation_sha256": full33_gate_implementation_hashes(root),
            "source_download_root": str(download_root),
            "source_inventory_sha256": audit["inventory_sha256"],
            "staging_root": str(staging_root) if staging else None,
            "staging_inventory_sha256": staging["inventory_sha256"] if staging else None,
            "staged_files": staging["files"] if staging else 0,
            "staging_publish_method": staging.get("publish_method") if staging else None,
            "staging_manifest": (
                {"path": str(staging_manifest), "sha256": sha256_file(staging_manifest)}
                if staging else None
            ),
            "required_cancers": list(TCGA_CANCERS),
            "patient_folds": 5,
            "patient_fold_authority": {
                **fold_authority_audit,
                "fold_patient_id_source": "EXPLICIT_PATIENT_ID_COLUMN",
                "fold_sample_id_fallback_used": False,
                "gdc_case_and_sample_barcode_normalization_is_raw_source_mapping_only": True,
            },
            "retry_manifest": {
                "path": str(retry_path),
                "sha256": sha256_file(retry_path),
                "rows": int(audit["retry_files"]),
            },
            "candidate_only": True,
            "formal_v32_primary_unchanged": True,
            "training_started": False,
            "missing_is_never_zero": True,
            "patient_segment_callability": {
                "path": str(callability_path),
                "sha256": sha256_file(callability_path),
                "fold_patients": len(callability_rows),
                "selected_segment_bound": sum(row["segment_manifest_available"] for row in callability_rows),
                "typed_unavailable": sum(not row["segment_manifest_available"] for row in callability_rows),
                "typed_unavailable_reason": "NO_SELECTED_GDC_MASKED_SEGMENT_FOR_PATIENT",
                "extra_selected_patients_not_in_folds": len(extra_selected_keys),
            },
            "staging_error": staging_error,
            "training_blocker": (
                "STREAMING_CNV_SUCCESS_REQUIRED; direct global raw-segment long-form remains forbidden"
            ),
        }
    )
    marker_name = "DOWNLOAD_COMPLETE.json" if gate_status == "READY" else "BLOCKED.json"
    marker = output / marker_name
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    retry_command = _shell_command(
        [
            "gdc-client", "download", "-m", str(retry_path), "-d", str(download_root),
        ]
    )
    streaming_output = (
        _resolve(root, args.streaming_output_root)
        if args.streaming_output_root else _resolve(root, args.training_output_root).with_name("full33_segment_streaming_cnv")
    )
    streaming_command = None
    if gate_status == "READY":
        streaming_command = _shell_command(
            [
                args.python_bin, str(root / "scripts/run_v32_streaming_segment_cnv.py"),
                "--repo-root", str(root), "--download-complete", str(marker),
                "--staging-root", str(staging_root), "--candidates", str(binding_paths["candidates"]),
                "--patient-folds", str(binding_paths["patient_folds"]),
                "--patient-fold-authority-receipt",
                str(binding_paths["patient_fold_authority_receipt"]),
                "--entity-intervals", str(binding_paths["entity_intervals"]),
                "--pathway-membership", str(binding_paths["pathway_membership"]),
                "--sample-gene-mutation", str(binding_paths["sample_gene_mutation"]),
                "--sample-lncrna-mutation", str(binding_paths["sample_lncrna_mutation"]),
                "--mc3", str(binding_paths["mc3"]),
                "--core-embedding-manifest", str(binding_paths["core_embedding_manifest"]),
                "--output-root", str(streaming_output),
            ]
        )
    commands = {
        "status": gate_status,
        "retry_only_command": retry_command if audit["retry_files"] else None,
        "formal_training_command": None,
        "streaming_materialization_command": streaming_command,
        "formal_training_command_withheld_until_streaming_success": True,
        "training_blocker": payload["training_blocker"],
    }
    commands_path = output / "AUTHORIZED_NEXT_COMMANDS.json"
    commands_path.write_text(
        json.dumps(commands, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**commands, "gate": str(marker)}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if gate_status == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
