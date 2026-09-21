#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from cc_hhgt.v30_integrity import atomic_write_json, file_sha256, verify_file_manifest
from cc_hhgt.v30_samples import build_canonical_sample_outputs


TARGET_STATES = ["stemness_rna::RNAss", "stemness_dna::DNAss"]


def load_manifest(path: Path) -> list[dict]:
    frame = pd.read_csv(path, sep="\t")
    required = {"relative_path", "size_bytes", "sha256"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Input manifest is invalid; missing {missing}")
    return frame.to_dict("records")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the tumor-only canonical V3.0 sample and state universe")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--minimum-observed", type=int, default=30)
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    input_manifest = Path(args.input_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"Refusing to reuse canonical output root: {output_root}")
    records = load_manifest(input_manifest)
    mismatches = verify_file_manifest(input_root, records)
    if mismatches:
        raise RuntimeError(f"Frozen input assets do not match: {mismatches[:20]}")
    provenance_path = input_manifest.parent / "PROVENANCE.json"
    if not provenance_path.exists():
        raise RuntimeError("Frozen provenance is missing beside the input manifest")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if str(provenance.get("run_id")) != args.run_id:
        raise RuntimeError("Canonical sample run_id does not match frozen provenance")
    protocol_path = Path(str(provenance["protocol_path"])).resolve()
    if not protocol_path.exists():
        raise RuntimeError(f"Frozen protocol is missing: {protocol_path}")

    fold_path = input_root / "adapter" / "cancer_specific_patient_fold_manifest.tsv"
    score_path = input_root / "TCGA_sample_scores" / "03_final_tables" / "TCGA_sample_score_matrix.tsv.gz"
    fold = pd.read_csv(fold_path, sep="\t")
    score = pd.read_csv(
        score_path,
        sep="\t",
        usecols=["sample_id", "patient_id", "cancer_type", "sample_type_code", "is_tumor", *TARGET_STATES],
    )
    canonical, expanded, state_values, eligibility, audit = build_canonical_sample_outputs(
        fold,
        score,
        target_states=TARGET_STATES,
        reference_only=[],
        minimum_observed=args.minimum_observed,
    )
    temporary_root = output_root.parent / f".{output_root.name}.tmp"
    if temporary_root.exists():
        raise RuntimeError(f"Stale temporary canonical root exists: {temporary_root}")
    temporary_root.mkdir(parents=True)
    lineage = {
        "run_id": args.run_id,
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": file_sha256(input_manifest),
        "fold_source": str(fold_path),
        "fold_source_sha256": file_sha256(fold_path),
        "state_source": str(score_path),
        "state_source_sha256": file_sha256(score_path),
        "sample_universe_sha256": audit["sample_universe_sha256"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
    }
    for frame in (canonical, expanded, state_values, eligibility):
        frame["run_id"] = args.run_id
        frame["input_manifest_sha256"] = lineage["input_manifest_sha256"]
    canonical.to_parquet(temporary_root / "canonical_samples.parquet", index=False, compression="zstd")
    expanded.to_parquet(temporary_root / "canonical_patient_fold_manifest.parquet", index=False, compression="zstd")
    state_values.to_parquet(temporary_root / "canonical_state_complete_cases.parquet", index=False, compression="zstd")
    eligibility.to_csv(temporary_root / "state_complete_case_eligibility.tsv", sep="\t", index=False)
    atomic_write_json(temporary_root / "LINEAGE.json", lineage)
    success = {
        **audit,
        **lineage,
        "status": "PASS",
        "normal_samples_required": 0,
        "state_imputed_rows_required": 0,
        "output_contract": protocol_path.name,
    }
    if success["normal_samples"] or success["non_tumor_score_rows"] or success["state_nan_rows"]:
        raise RuntimeError(f"Canonical sample gate failed: {success}")
    # SUCCESS is deliberately the last file written before the atomic directory rename.
    atomic_write_json(temporary_root / "SUCCESS.json", success)
    os.replace(temporary_root, output_root)
    print(json.dumps({**success, "output_root": str(output_root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

