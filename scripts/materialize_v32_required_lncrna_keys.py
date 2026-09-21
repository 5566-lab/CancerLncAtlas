#!/usr/bin/env python3
"""Materialize full V3.2 expression-eligible lncRNA keys from graph authority."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_tsv_gz(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    with gzip.open(partial, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, sep="\t", index=False, lineterminator="\n")
    if partial.stat().st_size <= 0:
        raise RuntimeError(f"Empty staged key table: {partial}")
    os.replace(partial, path)


def materialize(detection_path: Path, output: Path) -> dict[str, object]:
    detection = pd.read_parquet(detection_path)
    required = {"cancer_id", "lncrna_id"}
    if missing := sorted(required - set(detection)):
        raise RuntimeError(f"Detection authority lacks columns: {missing}")
    if detection["cancer_id"].nunique() != 33 or detection["lncrna_id"].nunique() != 16889:
        raise RuntimeError("Detection authority is not the exact 33 x 16,889 grid")
    rate_column = (
        "detection_rate_logcpm_gt0"
        if "detection_rate_logcpm_gt0" in detection
        else "detection_rate"
    )
    if rate_column not in detection:
        raise RuntimeError("Detection authority lacks a detection-rate column")
    detection[rate_column] = pd.to_numeric(detection[rate_column], errors="raise")
    eligible = detection.loc[
        detection[rate_column].ge(0.10), ["cancer_id", "lncrna_id"]
    ].copy()
    eligible["cancer_id"] = eligible["cancer_id"].astype(str)
    eligible["gene_id"] = (
        eligible["lncrna_id"].astype(str).str.removeprefix("LNC:").str.split(".").str[0]
    )
    if eligible.empty or eligible.duplicated(["cancer_id", "gene_id"]).any():
        raise RuntimeError("Expression-eligible lncRNA keys are empty or duplicated")
    candidate = (
        eligible.assign(lncrna_id="LNC:" + eligible["gene_id"])[["cancer_id", "lncrna_id"]]
        .sort_values(["cancer_id", "lncrna_id"], kind="stable")
        .reset_index(drop=True)
    )
    genes = (
        eligible[["gene_id"]].drop_duplicates().sort_values("gene_id", kind="stable").reset_index(drop=True)
    )
    output.mkdir(parents=True, exist_ok=False)
    candidate_path = output / "candidate_cancer_lncrna_keys.tsv.gz"
    genes_path = output / "required_lncrna_genes.tsv.gz"
    atomic_tsv_gz(candidate, candidate_path)
    atomic_tsv_gz(genes, genes_path)
    audit: dict[str, object] = {
        "format": "CC_HHGT_V3_2_REQUIRED_LNCRNA_KEYS_V1",
        "status": "PASS_REQUIRED_LNCRNA_KEYS",
        "source_detection": {"path": str(detection_path), "sha256": sha256(detection_path)},
        "detection_threshold": {"column": rate_column, "operator": ">=", "value": 0.10},
        "cancers": int(candidate["cancer_id"].nunique()),
        "candidate_cancer_lncrna_rows": int(len(candidate)),
        "required_gene_ids": int(len(genes)),
        "candidate_keys": {"path": str(candidate_path), "sha256": sha256(candidate_path)},
        "required_genes": {"path": str(genes_path), "sha256": sha256(genes_path)},
    }
    (output / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = materialize(args.detection.resolve(strict=True), args.output.resolve())
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
