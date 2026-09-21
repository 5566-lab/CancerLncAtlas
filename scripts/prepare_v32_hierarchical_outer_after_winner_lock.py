#!/usr/bin/env python3
"""Build hierarchical sealed outer payloads after validation winner lock."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--hierarchical-prepared-root", required=True)
    parser.add_argument("--hierarchical-preparation-manifest", required=True)
    parser.add_argument("--sealed-test-manifest", required=True)
    parser.add_argument("--genomic-predictions", required=True)
    parser.add_argument("--genomic-lineage", required=True)
    parser.add_argument("--atac-predictions")
    parser.add_argument("--atac-lineage")
    parser.add_argument("--routing-winner-declaration", required=True)
    parser.add_argument("--routing-winner-declaration-sha256", required=True)
    parser.add_argument("--hierarchical-validation-success", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.hierarchical_candidate_preparation import (
        prepare_hierarchical_outer_after_winner_lock,
    )

    result = prepare_hierarchical_outer_after_winner_lock(
        hierarchical_prepared_root=args.hierarchical_prepared_root,
        hierarchical_preparation_manifest_path=args.hierarchical_preparation_manifest,
        sealed_test_manifest_path=args.sealed_test_manifest,
        genomic_predictions_path=args.genomic_predictions,
        genomic_lineage_path=args.genomic_lineage,
        atac_predictions_path=args.atac_predictions,
        atac_lineage_path=args.atac_lineage,
        routing_winner_declaration_path=args.routing_winner_declaration,
        routing_winner_declaration_sha256=args.routing_winner_declaration_sha256,
        hierarchical_validation_success_path=args.hierarchical_validation_success,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
