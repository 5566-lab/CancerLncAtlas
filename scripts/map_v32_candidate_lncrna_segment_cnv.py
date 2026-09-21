#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cc_hhgt.v32.gdc_segment_cnv import SegmentCNVError, sha256_file
from cc_hhgt.v32.genomic_training import segment_calls_from_raw
from cc_hhgt.v32.patient_fold_authority import (
    validate_frozen_v32_patient_fold_binding,
)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, sep="\t")


def canonical_lnc_core(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.removeprefix("LNC:")
        .str.split(".", regex=False)
        .str[0]
    )


def candidate_lncrna_ids(candidates: pd.DataFrame) -> set[str]:
    if "lncrna_id" in candidates:
        values = candidates.lncrna_id.dropna()
    elif "gene_id" in candidates:
        values = candidates.gene_id.dropna()
    else:
        raise SegmentCNVError("Candidate table lacks lncrna_id/gene_id")
    cores = canonical_lnc_core(values)
    return {f"LNC:{value}" for value in cores if value and value.lower() != "nan"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Map candidate lncRNA intervals to masked CNV segments")
    parser.add_argument("--segment-root", type=Path, required=True)
    parser.add_argument("--entity-intervals", type=Path, required=True)
    parser.add_argument("--patient-folds", type=Path, required=True)
    parser.add_argument("--patient-fold-authority-receipt", type=Path, required=True)
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-load1", type=float, default=12.0)
    args = parser.parse_args()
    patient_authority_audit = validate_frozen_v32_patient_fold_binding(
        args.patient_folds, args.patient_fold_authority_receipt
    )
    output_text = str(args.output_root).lower()
    if "candidate" not in output_text or "final_release" in output_text:
        raise SegmentCNVError("Output must be candidate-only and outside final_release")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SegmentCNVError("Refusing to overwrite a non-empty candidate output directory")
    if hasattr(os, "getloadavg") and os.getloadavg()[0] > args.max_load1:
        raise SegmentCNVError(
            f"Current load1={os.getloadavg()[0]:.2f} exceeds max-load1={args.max_load1}; mapping not started"
        )
    intervals = read_table(args.entity_intervals)
    candidates = read_table(args.candidate_table)
    folds = read_table(args.patient_folds)
    entity_col = "entity_id" if "entity_id" in intervals else "lncrna_id"
    type_col = "entity_type" if "entity_type" in intervals else "feature_type"
    allowed = candidate_lncrna_ids(candidates)
    allowed_cores = {value.removeprefix("LNC:") for value in allowed}
    interval_cores = canonical_lnc_core(intervals[entity_col])
    intervals = intervals.loc[
        intervals[type_col].astype(str).str.lower().isin(["lncrna", "lncrna_gene"])
        & interval_cores.isin(allowed_cores)
    ].copy()
    intervals[entity_col] = "LNC:" + canonical_lnc_core(intervals[entity_col])
    if not allowed or intervals.empty:
        raise SegmentCNVError("Candidate lncRNA namespace mapping produced an empty interval set")
    _, lnc, audit = segment_calls_from_raw(args.segment_root, intervals, folds)
    args.output_root.mkdir(parents=True, exist_ok=False)
    output = args.output_root / "candidate_lncrna_segment_cnv_calls.parquet"
    lnc.to_parquet(output, index=False, compression="zstd")
    manifest = {
        "format": "CC_HHGT_V3_2_CANDIDATE_LNCRNA_SEGMENT_CNV_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "formal_v32_unchanged": True,
        "candidate_lncrnas": len(allowed),
        "candidate_lncrna_namespace": "LNC:UNVERSIONED_ENSEMBL_OR_SOURCE_ID",
        "mapped_interval_rows": len(intervals),
        "segment_audit": audit,
        "patient_fold_authority": patient_authority_audit,
        "output": {"path": output.name, "sha256": sha256_file(output), "rows": len(lnc)},
    }
    (args.output_root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
