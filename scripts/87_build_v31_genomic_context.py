#!/usr/bin/env python3
"""Build canonical, outcome-free mutation/CNV context for residual B3."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cc_hhgt.common import file_sha256, write_table
from cc_hhgt.v30_integrity import atomic_write_json, merkle_sha256


REQUIRED_FORMAL = {"HNSC", "LGG"}
LOCAL_CNV_FEATURES = (
    "cnv_local_sample_count",
    "cnv_local_mean",
    "cnv_local_sd",
    "cnv_local_mean_absolute",
    "cnv_local_amplified_fraction",
    "cnv_local_deleted_fraction",
    "cnv_local_sample_coverage",
)


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _registered_cancers(path: Path) -> tuple[str, ...]:
    folds = pd.read_csv(path, sep="\t")
    cancers = tuple(sorted(folds.test_cancer.astype(str).unique()))
    if len(cancers) != 33 or not REQUIRED_FORMAL.issubset(cancers):
        raise RuntimeError("Genomic context requires the exact 33-cancer formal scope including HNSC/LGG")
    return cancers


def _candidate_keys(
    asset_results: Path, cancers: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lnc_parts: list[pd.DataFrame] = []
    pathway_parts: list[pd.DataFrame] = []
    for cancer in cancers:
        pathway_paths = sorted(
            (asset_results / "tables" / "candidate_universe" / f"cancer_id={cancer}").glob(
                "*.parquet"
            )
        )
        state_paths = sorted(
            (asset_results / "tables" / "strict_state_candidate" / f"cancer_id={cancer}").glob(
                "*.parquet"
            )
        )
        if not pathway_paths or not state_paths:
            raise RuntimeError(f"Frozen candidates missing for {cancer}")
        for path in pathway_paths:
            frame = pd.read_parquet(path, columns=["cancer_id", "lncrna_id", "pathway_family_id"])
            lnc_parts.append(frame[["cancer_id", "lncrna_id"]])
            pathway_parts.append(frame[["cancer_id", "pathway_family_id"]])
        for path in state_paths:
            frame = pd.read_parquet(path, columns=["cancer_id", "lncrna_id"])
            lnc_parts.append(frame)
    lnc = (
        pd.concat(lnc_parts, ignore_index=True)
        .drop_duplicates()
        .sort_values(["cancer_id", "lncrna_id"], kind="stable")
        .reset_index(drop=True)
    )
    pathway = (
        pd.concat(pathway_parts, ignore_index=True)
        .drop_duplicates()
        .sort_values(["cancer_id", "pathway_family_id"], kind="stable")
        .reset_index(drop=True)
    )
    return lnc, pathway


def aggregate_lnc_mutation(
    events: pd.DataFrame,
    canonical_patients: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "cancer_id",
        "patient_id",
        "lncrna_id",
        "exonic_variant_count",
        "splice_variant_count",
        "promoter_variant_count",
        "recurrent_variant_count",
        "max_vaf",
        "length_normalized_mutation_burden",
        "tmb_adjusted_mutation_burden",
        "locus_coverage_available",
        "absence_is_wildtype",
    }
    missing = required - set(events)
    if missing:
        raise RuntimeError(f"lncRNA mutation table lacks {sorted(missing)}")
    allowed = canonical_patients[["cancer_id", "patient_id"]].drop_duplicates()
    frame = events.merge(allowed, on=["cancer_id", "patient_id"], how="inner", validate="many_to_one")
    numeric = [
        "exonic_variant_count",
        "splice_variant_count",
        "promoter_variant_count",
        "recurrent_variant_count",
        "max_vaf",
        "length_normalized_mutation_burden",
        "tmb_adjusted_mutation_burden",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.groupby(["cancer_id", "lncrna_id"], observed=True, sort=False)
    output = grouped.agg(
        genomic_lnc_observed_event_patients=("patient_id", "nunique"),
        genomic_lnc_exonic_events=("exonic_variant_count", "sum"),
        genomic_lnc_splice_events=("splice_variant_count", "sum"),
        genomic_lnc_promoter_events=("promoter_variant_count", "sum"),
        genomic_lnc_recurrent_events=("recurrent_variant_count", "sum"),
        genomic_lnc_max_vaf=("max_vaf", "max"),
        genomic_lnc_mean_length_normalized_burden=("length_normalized_mutation_burden", "mean"),
        genomic_lnc_mean_tmb_adjusted_burden=("tmb_adjusted_mutation_burden", "mean"),
        genomic_lnc_covered_event_rows=("locus_coverage_available", "sum"),
        genomic_lnc_absence_wildtype_rows=("absence_is_wildtype", "sum"),
    ).reset_index()
    totals = allowed.groupby("cancer_id", observed=True).patient_id.nunique()
    output["genomic_lnc_canonical_patient_count"] = output.cancer_id.map(totals).astype(float)
    output["genomic_lnc_observed_event_fraction"] = (
        output.genomic_lnc_observed_event_patients
        / output.genomic_lnc_canonical_patient_count.clip(lower=1)
    )
    return output


def aggregate_pathway_mutation(
    events: pd.DataFrame,
    canonical_patients: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "cancer_id",
        "patient_id",
        "pathway_family_id",
        "n_mutated_genes",
        "weighted_mutation_score",
        "n_driver_genes",
        "n_hotspot_genes",
        "loss_of_function_burden",
        "activating_burden",
        "driver_annotation_available",
        "hotspot_annotation_available",
        "functional_annotation_available",
        "absence_is_wildtype",
    }
    missing = required - set(events)
    if missing:
        raise RuntimeError(f"pathway mutation table lacks {sorted(missing)}")
    allowed = canonical_patients[["cancer_id", "patient_id"]].drop_duplicates()
    frame = events.merge(allowed, on=["cancer_id", "patient_id"], how="inner", validate="many_to_one")
    numeric = [
        "n_mutated_genes",
        "weighted_mutation_score",
        "n_driver_genes",
        "n_hotspot_genes",
        "loss_of_function_burden",
        "activating_burden",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.groupby(["cancer_id", "pathway_family_id"], observed=True, sort=False)
    output = grouped.agg(
        genomic_pathway_observed_event_patients=("patient_id", "nunique"),
        genomic_pathway_mean_mutated_genes=("n_mutated_genes", "mean"),
        genomic_pathway_mean_weighted_score=("weighted_mutation_score", "mean"),
        genomic_pathway_mean_driver_genes=("n_driver_genes", "mean"),
        genomic_pathway_mean_hotspot_genes=("n_hotspot_genes", "mean"),
        genomic_pathway_mean_loss_of_function=("loss_of_function_burden", "mean"),
        genomic_pathway_mean_activating=("activating_burden", "mean"),
        genomic_pathway_driver_rows=("driver_annotation_available", "sum"),
        genomic_pathway_hotspot_rows=("hotspot_annotation_available", "sum"),
        genomic_pathway_functional_rows=("functional_annotation_available", "sum"),
        genomic_pathway_absence_wildtype_rows=("absence_is_wildtype", "sum"),
    ).reset_index()
    totals = allowed.groupby("cancer_id", observed=True).patient_id.nunique()
    output["genomic_pathway_canonical_patient_count"] = output.cancer_id.map(totals).astype(float)
    output["genomic_pathway_observed_event_fraction"] = (
        output.genomic_pathway_observed_event_patients
        / output.genomic_pathway_canonical_patient_count.clip(lower=1)
    )
    return output


def _mask_merged_features(
    keys: pd.DataFrame,
    aggregate: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    result = keys.merge(aggregate, on=key_columns, how="left", validate="one_to_one")
    features = [column for column in aggregate.columns if column not in key_columns]
    for feature in features:
        result[f"{feature}__available"] = result[feature].notna()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-results", required=True)
    parser.add_argument("--mutation-root", required=True)
    parser.add_argument("--canonical-manifest", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--local-cnv-context", required=True)
    parser.add_argument("--local-cnv-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    asset_results = Path(args.asset_results).resolve()
    mutation_root = Path(args.mutation_root).resolve()
    canonical_path = Path(args.canonical_manifest).resolve()
    fold_path = Path(args.fold_manifest).resolve()
    cnv_path = Path(args.local_cnv_context).resolve()
    cnv_manifest_path = Path(args.local_cnv_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"Genomic context builder refuses reuse: {output_root}")
    output_root.mkdir(parents=True)
    source_paths = [
        mutation_root / "sample_lncRNA_mutation.parquet",
        mutation_root / "sample_pathway_mutation.parquet",
        canonical_path,
        fold_path,
        cnv_path,
        cnv_manifest_path,
        Path(__file__).resolve(),
    ]
    for path in source_paths:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    cnv_manifest = json.loads(cnv_manifest_path.read_text(encoding="utf-8"))
    if str(cnv_manifest.get("table_sha256", "")).lower() != file_sha256(cnv_path).lower():
        raise RuntimeError("Local CNV compact context hash mismatch")

    cancers = _registered_cancers(fold_path)
    canonical = pd.read_csv(canonical_path, sep="\t", usecols=["cancer_id", "patient_id"])
    canonical = canonical.loc[canonical.cancer_id.astype(str).isin(cancers)].drop_duplicates()
    lnc_keys, pathway_keys = _candidate_keys(asset_results, cancers)
    lnc_events = pd.read_parquet(mutation_root / "sample_lncRNA_mutation.parquet")
    pathway_events = pd.read_parquet(mutation_root / "sample_pathway_mutation.parquet")
    lnc_aggregate = aggregate_lnc_mutation(lnc_events, canonical)
    pathway_aggregate = aggregate_pathway_mutation(pathway_events, canonical)
    lnc = _mask_merged_features(
        lnc_keys, lnc_aggregate, ["cancer_id", "lncrna_id"]
    )
    pathway = _mask_merged_features(
        pathway_keys,
        pathway_aggregate,
        ["cancer_id", "pathway_family_id"],
    )

    compact = pd.read_parquet(cnv_path)
    cnv_columns = ["cancer_id", "lncrna_id"]
    for feature in LOCAL_CNV_FEATURES:
        cnv_columns.extend([feature, f"{feature}__available"])
    missing_cnv = sorted(set(cnv_columns) - set(compact))
    if missing_cnv:
        raise RuntimeError(f"Local CNV compact context lacks {missing_cnv}")
    compact = compact[cnv_columns]
    lnc = lnc.merge(compact, on=["cancer_id", "lncrna_id"], how="left", validate="one_to_one")
    for feature in LOCAL_CNV_FEATURES:
        mask = f"{feature}__available"
        lnc[mask] = lnc[mask].fillna(False).astype(bool)
        lnc.loc[~lnc[mask], feature] = np.nan

    lnc_path = output_root / "GENOMIC_LNCRNA_CONTEXT.parquet"
    pathway_path = output_root / "GENOMIC_PATHWAY_CONTEXT.parquet"
    _atomic_table(lnc, lnc_path)
    _atomic_table(pathway, pathway_path)
    lineage_rows = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in source_paths
    ]
    lineage_path = output_root / "GENOMIC_CONTEXT_INPUT_LINEAGE.tsv"
    _atomic_table(pd.DataFrame(lineage_rows), lineage_path)
    audit: dict[str, Any] = {
        "status": "PASS",
        "stage": "PHASE_B3_GENOMIC_CONTEXT_BUILD",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": "CPU_HEAVY_MEMORY",
        "registered_cancers": list(cancers),
        "reference_only_cancers_excluded": [],
        "required_formal_cancers_included": sorted(REQUIRED_FORMAL),
        "canonical_patient_count": int(len(canonical)),
        "lncrna_context_rows": int(len(lnc)),
        "pathway_context_rows": int(len(pathway)),
        "mutation_absence_policy": "UNAVAILABLE_UNLESS_OBSERVED; NEVER_ASSUME_WILDTYPE",
        "local_cnv_cancers": sorted(
            compact.loc[
                compact[[f"{feature}__available" for feature in LOCAL_CNV_FEATURES]].any(axis=1),
                "cancer_id",
            ].astype(str).unique()
        ),
        "outcome_derived_columns_used": 0,
        "full_cancer_model_training_started": False,
        "lncrna_context_sha256": file_sha256(lnc_path),
        "pathway_context_sha256": file_sha256(pathway_path),
        "failures": [],
    }
    audit_path = output_root / "GENOMIC_CONTEXT_AUDIT.json"
    atomic_write_json(audit_path, audit)
    manifest_rows = [
        {
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in (lnc_path, pathway_path, lineage_path, audit_path)
    ]
    manifest_path = output_root / "GENOMIC_CONTEXT_SHA256.tsv"
    _atomic_table(pd.DataFrame(manifest_rows), manifest_path)
    success = {
        "status": "PASS",
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_merkle_sha256": merkle_sha256(manifest_rows),
        "full_cancer_model_training_started": False,
        "success_written_last": True,
    }
    atomic_write_json(output_root / "SUCCESS.json", success)
    print(json.dumps(success, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
