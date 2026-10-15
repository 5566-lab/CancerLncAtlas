#!/usr/bin/env python3
"""Build coordinate-only GRCh38 candidate links for the distal prediction head."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
from pathlib import Path

import pandas as pd

from cc_hhgt.v32.distal_topology import POLICY, coordinate_distal_candidates, gene_core, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promoters", type=Path, required=True, help="GENCODE GRCh38 BED0 table with header")
    parser.add_argument("--peak-set", type=Path, required=True, help="Official GRCh38 ATAC 1-based peak table")
    parser.add_argument("--required-genes", type=Path, required=True, help="TSV gene_id universe, not Data S7 subset")
    parser.add_argument("--reference-build", choices=["GRCh38"], required=True)
    parser.add_argument("--max-distance-bp", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if socket.gethostname() != "149":
        raise RuntimeError("Distal topology preparation must execute on target host 149")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "LAUNCH_PREFLIGHT.json").write_text(json.dumps({
        "target_host": "149", "observed_host": socket.gethostname(), "training_started": False,
        "stage": "COORDINATE_ONLY_DISTAL_CANDIDATES"}, indent=2))
    inputs = {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in {
        "promoters": args.promoters, "peak_set": args.peak_set, "required_genes": args.required_genes}.items()}
    required = pd.read_csv(args.required_genes, sep="\t").gene_id
    links = coordinate_distal_candidates(pd.read_csv(args.promoters, sep="\t"),
        pd.read_csv(args.peak_set, sep="\t"), required, max_distance_bp=args.max_distance_bp)
    target = args.output / "INDEPENDENT_DISTAL_LINKS.tsv.gz"
    links.to_csv(target, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    # Retain the complete gene universe, including zero-peak genes in coverage.
    coverage = pd.DataFrame({"gene_id": sorted(set(gene_core(required)))})
    counts = links.assign(gene_id=gene_core(links.lncrna_id)).groupby("gene_id").size()
    coverage["linked_peaks"] = coverage.gene_id.map(counts).fillna(0).astype(int)
    coverage.to_csv(args.output / "GENE_COVERAGE.tsv", sep="\t", index=False)
    audit = {"status": "PASS", "policy": POLICY, "target_host": "149", "training_started": False,
        "coordinate_build": args.reference_build, "max_distance_bp": args.max_distance_bp,
        "cohort_expression_used": False, "cohort_pathway_activity_used": False,
        "regulatory_validation_claimed": False, "candidate_semantics": "COORDINATE_PROXIMITY_HYPOTHESIS",
        "promoter_overlap_excluded": True, "cohort_fold_specific_selection_needed": False,
        "rows": len(links), "gene_universe": len(coverage), "genes_without_linked_peak": int(coverage.linked_peaks.eq(0).sum()),
        "inputs": inputs, "output": str(target.resolve()), "output_sha256": sha256(target),
        "output_md5": hashlib.md5(target.read_bytes()).hexdigest()}
    Path(str(target) + ".audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
