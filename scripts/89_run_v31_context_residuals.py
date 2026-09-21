#!/usr/bin/env python3
"""Run the CPU-only V3.1 multimodal residual three-cancer pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cc_hhgt.v31_context_stage import load_context_asset, run_multimodal_context_stage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-base", required=True)
    parser.add_argument("--best-simple-gate", required=True)
    parser.add_argument("--graph-predictions", required=True)
    parser.add_argument("--graph-gate", required=True)
    parser.add_argument("--b2-hard-report", required=True)
    parser.add_argument("--rna-table", required=True)
    parser.add_argument("--rna-success", required=True)
    parser.add_argument("--genomic-lnc-table", required=True)
    parser.add_argument("--genomic-pathway-table", required=True)
    parser.add_argument("--genomic-success", required=True)
    parser.add_argument("--atac-table")
    parser.add_argument("--atac-success")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--shrinkage-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0]
    )
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    if (args.atac_table is None) != (args.atac_success is None):
        raise RuntimeError("ATAC table and SUCCESS must be supplied together")

    rna, rna_lineage = load_context_asset(
        Path(args.rna_table),
        Path(args.rna_success),
        expected_table_name="FULL_RNA_COEXPRESSION_CONTEXT.parquet",
    )
    genomic_lnc, genomic_lnc_lineage = load_context_asset(
        Path(args.genomic_lnc_table),
        Path(args.genomic_success),
        expected_table_name="GENOMIC_LNCRNA_CONTEXT.parquet",
    )
    genomic_pathway, genomic_pathway_lineage = load_context_asset(
        Path(args.genomic_pathway_table),
        Path(args.genomic_success),
        expected_table_name="GENOMIC_PATHWAY_CONTEXT.parquet",
    )
    atac = None
    atac_lineage = {"status": "UNAVAILABLE"}
    if args.atac_table is not None:
        atac, atac_lineage = load_context_asset(
            Path(args.atac_table), Path(args.atac_success)
        )
    payload = run_multimodal_context_stage(
        residual_base_path=Path(args.residual_base),
        best_simple_gate_path=Path(args.best_simple_gate),
        graph_predictions_path=Path(args.graph_predictions),
        graph_gate_path=Path(args.graph_gate),
        b2_hard_report_path=Path(args.b2_hard_report),
        modules={
            "RNA": (rna, None),
            "GENOMIC": (genomic_lnc, genomic_pathway),
            "ATAC": (atac, None) if atac is not None else None,
            "SINGLECELL": None,
        },
        module_lineage={
            "RNA": rna_lineage,
            "GENOMIC_LNCRNA": genomic_lnc_lineage,
            "GENOMIC_PATHWAY": genomic_pathway_lineage,
            "ATAC": atac_lineage,
            "SINGLECELL": {
                "status": "UNAVAILABLE",
                "reason": "NO_REAL_PATIENT_LEVEL_PSEUDOBULK_ASSET",
            },
        },
        output_dir=Path(args.output_dir),
        shrinkage_grid=args.shrinkage_grid,
        aggregation_git_commit=args.git_commit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
