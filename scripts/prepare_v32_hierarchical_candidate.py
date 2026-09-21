#!/usr/bin/env python3
"""Create pair-blocked modality-enriched folds for hierarchical HHGT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--formal-prepared-root", required=True)
    parser.add_argument("--genomic-predictions", required=True)
    parser.add_argument("--genomic-lineage", required=True)
    parser.add_argument("--atac-predictions")
    parser.add_argument("--atac-lineage")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--expected-graph-variant", choices=("G0", "G1", "G2"), default="G2")
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.hierarchical_candidate_preparation import prepare_hierarchical_candidate_folds

    resolve = lambda value: Path(value).resolve() if Path(value).is_absolute() else (root / value).resolve()
    result = prepare_hierarchical_candidate_folds(
        formal_prepared_root=resolve(args.formal_prepared_root),
        genomic_predictions_path=resolve(args.genomic_predictions),
        genomic_lineage_path=resolve(args.genomic_lineage),
        atac_predictions_path=resolve(args.atac_predictions) if args.atac_predictions else None,
        atac_lineage_path=resolve(args.atac_lineage) if args.atac_lineage else None,
        output_root=resolve(args.output_root),
        seed=args.seed,
        expected_graph_variant=args.expected_graph_variant,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
