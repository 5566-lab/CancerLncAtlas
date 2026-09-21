#!/usr/bin/env python3
"""Materialize the fresh V3.2 bulk lncRNA-gene coexpression release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.bulk_coexpression_release import (
    materialize_bulk_coexpression_release,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lncrna-expression-root",
        type=Path,
        default=Path("artifacts/input/formal_lncRNA_expression"),
    )
    parser.add_argument(
        "--gene-expression-root",
        type=Path,
        default=Path(
            "D:/model/V3_1_EXACT_33C_Formal_20260823/migration/input/"
            "input_snapshot/parquet/bulk_gene_expression"
        ),
    )
    parser.add_argument(
        "--covariates",
        type=Path,
        default=Path(
            "D:/model/V3_1_EXACT_33C_Formal_20260823/migration/input/"
            "input_snapshot/processed/tcga_association_covariates.parquet"
        ),
    )
    parser.add_argument(
        "--exact-candidates",
        type=Path,
        default=Path(
            "artifacts/formal_prepared/FORMAL_CANDIDATE_UNIVERSE.parquet"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--non-formal-fixture", action="store_true")
    parser.add_argument("--min-samples", type=int, default=40)
    parser.add_argument("--min-variance", type=float, default=0.01)
    parser.add_argument("--min-abs-rho", type=float, default=0.20)
    parser.add_argument("--max-fdr", type=float, default=0.05)
    parser.add_argument("--max-edges-per-direction", type=int, default=75)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-clusters", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=20260826)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_bulk_coexpression_release(
        lncrna_expression_root=args.lncrna_expression_root,
        gene_expression_root=args.gene_expression_root,
        covariates_path=args.covariates,
        exact_candidate_path=args.exact_candidates,
        output_root=args.output_root,
        runner_path=Path(__file__),
        strict_formal_authority=not args.non_formal_fixture,
        min_samples=args.min_samples,
        min_variance=args.min_variance,
        min_abs_rho=args.min_abs_rho,
        max_fdr=args.max_fdr,
        max_edges_per_direction=args.max_edges_per_direction,
        block_size=args.block_size,
        max_clusters=args.max_clusters,
        random_state=args.random_state,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
