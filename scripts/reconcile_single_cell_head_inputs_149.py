"""Repair stale declarations and exercise the real consumers on server 149."""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from cc_hhgt.v32.single_cell_manifest_reconciliation import (
    adapt_verified_head_tables, file_sha256, reconcile_bound_manifest, verify_file,
)
from cc_hhgt.v32.single_cell_training import (
    FORMAL_SINGLE_CELL_CANCERS, _normalise_lnc_celltype, build_blocked_folds, build_domain_features,
    exact_candidate_join, normalise_dataset_manifest,
    normalise_single_cell_associations, split_block_ids,
)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_input_manifest(output_root, source_inputs, candidate_path, receipt):
    output_root, source_inputs, candidate_path = map(Path, (output_root, source_inputs, candidate_path))
    new = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_HEAD_INPUTS_V2",
        "status": "PASS_RECONCILED_AND_CONSUMER_VALIDATED", "target_host": "149",
        "source_assembly_manifest": str(source_inputs / "INPUT_MANIFEST.json"),
        "source_assembly_manifest_sha256": file_sha256(source_inputs / "INPUT_MANIFEST.json"),
        "reconciliation_receipt": str(output_root / "MANIFEST_RECONCILIATION.json"),
        "reconciliation_sha256": file_sha256(output_root / "MANIFEST_RECONCILIATION.json"),
        "binding_sha256": receipt["binding_sha256"],
        "qualified_cancers": receipt["qualified_cancers"],
        "training_cancers": receipt["training_cancers"],
        "per_cancer": receipt["coverage"],
        "consumer_adaptation": receipt["input_adaptation"],
        "candidates": {"path": str(candidate_path), "sha256": file_sha256(candidate_path)},
    }
    for key, name in (("association", "sc_association.parquet"), ("celltype", "lnc_celltype.parquet"),
                      ("dataset_manifest", "dataset_manifest.parquet"),
                      ("dataset_manifest_training", "dataset_manifest.training.parquet")):
        new[key] = {"path": str(output_root / name), "sha256": file_sha256(output_root / name),
                    "rows": pq.ParquetFile(output_root / name).metadata.num_rows}
    save(output_root / "INPUT_MANIFEST.json", new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--assembled-inputs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("Execution requires target_host=149")
    args.output_root.mkdir(exist_ok=False, parents=True)
    save(args.output_root / "LAUNCH_PREFLIGHT.json", {
        "target_host": "149", "actual_host": socket.gethostname(),
        "operation": "RECONCILE_AND_VALIDATE_SINGLE_CELL_INPUTS_CPU_ONLY",
    })
    raw = pd.read_parquet(args.source_root / "dataset_manifest_33c.parquet")
    if set(raw.cancer_id) != set(FORMAL_SINGLE_CELL_CANCERS):
        raise RuntimeError("Manifest must account for all 33 cancers")
    fixed, receipt = reconcile_bound_manifest(
        raw, args.source_root / "RUN_STATUS.json", args.binding,
    )
    qualified = normalise_dataset_manifest(fixed)
    expected = set(receipt["bound_cancers"])
    if set(qualified.loc[qualified.qualified, "cancer_id"]) != expected:
        raise RuntimeError("Qualified coverage does not equal audited binding")
    for name in ("dataset_manifest.parquet", "dataset_manifest.training.parquet"):
        fixed.to_parquet(args.output_root / name, index=False, compression="zstd")
    old = json.loads((args.assembled_inputs / "INPUT_MANIFEST.json").read_text())
    if old["binding"]["sha256"] != receipt["binding_sha256"]:
        raise RuntimeError("Assembled inputs use a different binding")
    for key, name in (("association", "sc_association.parquet"), ("celltype", "lnc_celltype.parquet")):
        source = args.assembled_inputs / name
        verify_file(source, old[key]["sha256"])
    parts = args.output_root / "parts"
    parts.mkdir()
    candidate_path = Path(old["candidates"]["path"])
    verify_file(candidate_path, old["candidates"]["sha256"])
    run = json.loads((args.source_root / "RUN_STATUS.json").read_text())
    coverage, blocks = [], []
    old_by_cancer = {r["cancer_id"]: r for r in old["per_cancer"]}
    for cancer in sorted(FORMAL_SINGLE_CELL_CANCERS):
        source = run["per_cancer"][cancer]
        row = {"cancer_id": cancer, "qualified": cancer in expected,
               "training_rows": 0, "reason": source.get("blocking_reason")}
        if cancer not in expected:
            metadata = Path(source["metadata_path"])
            row["h5_exists"] = Path(source["h5_path"]).is_file()
            row["metadata_exists"] = metadata.is_file()
            if metadata.is_file():
                md = pd.read_parquet(metadata, columns=["patient_id"])
                row["observed_donors"] = int(md.patient_id.nunique())
            coverage.append(row)
            continue
        evidence = receipt["evidence"][cancer]
        root = Path(evidence["output_root"])
        files = pd.read_parquet(root / "FILE_MANIFEST.parquet").set_index("relative_path")
        for name, key in (("association_evidence.parquet", "association_sha256"),
                          ("lncrna_donor_celltype_summary.parquet", "celltype_sha256")):
            expected_hash = files.at[name, "sha256"]
            verify_file(root / name, expected_hash)
            if old_by_cancer[cancer][key] != expected_hash:
                raise RuntimeError(f"Assembly partition provenance mismatch: {cancer}")
        raw_assoc = pd.read_parquet(args.assembled_inputs / "sc_association.parquet", filters=[("cancer_id", "=", cancer)])
        raw_cell = pd.read_parquet(args.assembled_inputs / "lnc_celltype.parquet", filters=[("cancer_id", "=", cancer)])
        if len(raw_assoc) != evidence["association_rows"]:
            raise RuntimeError(f"Association row count mismatch: {cancer}")
        if len(raw_cell) != old_by_cancer[cancer]["celltype_rows"]:
            raise RuntimeError(f"Celltype row count mismatch: {cancer}")
        candidates = pd.read_parquet(candidate_path, filters=[("cancer_id", "=", cancer)],
                                     columns=["cancer_id", "lncrna_id", "pathway_id"])
        adapted_assoc, adapted_cell = adapt_verified_head_tables(raw_assoc, raw_cell, candidates, qualified)
        cell = _normalise_lnc_celltype(adapted_cell)
        if set(cell.dataset_id) != {evidence["dataset_id"]}:
            raise RuntimeError(f"Celltype dataset mismatch: {cancer}")
        assoc = normalise_single_cell_associations(adapted_assoc, qualified)
        if len(assoc) != len(raw_assoc) or not assoc.formal_row.all():
            raise RuntimeError(f"Unexpected association filtering: {cancer}")
        joined = exact_candidate_join(assoc, candidates) if len(assoc) else assoc
        row.update(donors=evidence["donors"], lncrnas=evidence["lncrnas"],
                   association_rows=len(assoc), celltype_source_rows=len(raw_cell),
                   celltype_rows=len(adapted_cell), normalized_celltype_rows=len(cell),
                   training_rows=len(joined), candidate_rows=len(candidates))
        if len(joined):
            domain = build_domain_features(joined.head(5000), cell, pd.DataFrame())
            row["domain_sample_rows"] = len(domain)
            row["domain_sample_lnc_feature_available"] = int(domain[:, 7].sum())
            if row["domain_sample_lnc_feature_available"] != len(domain):
                raise RuntimeError(f"Compartment feature join incomplete: {cancer}")
            blocks.append(joined[["cancer_id", "dataset_id", "donor_id"]].drop_duplicates())
            row["reason"] = None
        elif not len(assoc):
            row["reason"] = "NO_ASSOCIATION_ROWS_AFTER_DONOR_REPLICATION_FILTER"
        else:
            row["reason"] = "NO_EXACT_CANDIDATE_OVERLAP"
        coverage.append(row)
        if len(adapted_assoc):
            adapted_assoc.to_parquet(parts / (cancer + ".association.parquet"), index=False, compression="zstd")
        adapted_cell.to_parquet(parts / (cancer + ".celltype.parquet"), index=False, compression="zstd")
        print(json.dumps(row), flush=True)
        del raw_assoc, raw_cell, cell, assoc, joined, candidates, adapted_assoc, adapted_cell
    for key, name, field in (("association", "sc_association.parquet", "association_rows"),
                              ("celltype", "lnc_celltype.parquet", "celltype_rows")):
        source_field = field if key == "association" else "celltype_source_rows"
        if sum(r.get(source_field, 0) for r in coverage) != old[key]["rows"]:
            raise RuntimeError(f"Unaccounted source input rows: {name}")
        sources = sorted(parts.glob("*." + key + ".parquet"))
        with pq.ParquetWriter(args.output_root / name, pq.read_schema(sources[0]), compression="zstd") as writer:
            for source in sources:
                for batch in pq.ParquetFile(source).iter_batches(batch_size=65536):
                    writer.write_batch(batch)
        total = pq.ParquetFile(args.output_root / name).metadata.num_rows
        if total != sum(r.get(field, 0) for r in coverage):
            raise RuntimeError(f"Unaccounted input rows: {name}")
    folds = build_blocked_folds(pd.concat(blocks, ignore_index=True))
    for fold in range(5):
        splits = split_block_ids(folds, fold)
        if not all(splits.values()):
            raise RuntimeError(f"Empty train/validation/test block set: {fold}")
    folds.to_csv(args.output_root / "TRAINING_BLOCKS.tsv", sep="\t", index=False)
    pd.DataFrame(coverage).to_csv(args.output_root / "CANCER_COVERAGE.tsv", sep="\t", index=False)
    receipt.update(status="PASS", coverage=coverage, qualified_cancers=sorted(expected),
                   training_cancers=sorted(r["cancer_id"] for r in coverage if r["training_rows"]),
                   fold_block_counts={str(k): int(v) for k, v in folds.single_cell_fold_id.value_counts().items()},
                   partition_isolation="DATASET_LEVEL_NO_DONOR_COLUMN_IN_ASSOCIATIONS",
                   source_manifest_sha256=file_sha256(args.source_root / "dataset_manifest_33c.parquet"),
                   manifest_sha256=file_sha256(args.output_root / "dataset_manifest.training.parquet"),
                   input_adaptation={"source_tier": "VERIFIED_DATASET_MANIFEST",
                     "lncrna_id": "EXACT_CANDIDATE_NAMESPACE",
                     "celltype": "CELL_WEIGHTED_WITHIN_DONOR_COMPARTMENT_THEN_CONSUMER_DONOR_MEAN",
                     "n_observations": "n_donors"},
                   training_started=False)
    save(args.output_root / "MANIFEST_RECONCILIATION.json", receipt)
    write_input_manifest(args.output_root, args.assembled_inputs, candidate_path, receipt)
    print(json.dumps({"status": "PASS", "qualified": len(expected),
                      "with_training_rows": len(receipt["training_cancers"]),
                      "fold_block_counts": receipt["fold_block_counts"]}), flush=True)


if __name__ == "__main__":
    main()
