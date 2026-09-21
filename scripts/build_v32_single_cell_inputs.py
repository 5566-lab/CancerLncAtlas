#!/usr/bin/env python3
"""Build fresh V3.2 single-cell standard inputs from expression facts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_input_builder import (  # noqa: E402
    SingleCellInputBuildConfig,
    build_v32_single_cell_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build V3.2 single-cell association/activity/lnc-celltype inputs "
            "only from donor pseudobulk expression facts and the pinned exact-2135 membership."
        )
    )
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--expression-facts", required=True, type=Path)
    parser.add_argument("--exact-membership", required=True, type=Path)
    parser.add_argument("--membership-provenance", required=True, type=Path)
    parser.add_argument("--exact-candidates", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--build-run-id", required=True)
    parser.add_argument("--min-association-observations", type=int, default=5)
    parser.add_argument("--low-feature-universe-threshold", type=int, default=1_000)
    parser.add_argument("--association-chunk-size", type=int, default=100_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_v32_single_cell_inputs(
        dataset_manifest_path=args.dataset_manifest,
        expression_facts_path=args.expression_facts,
        exact_membership_path=args.exact_membership,
        membership_provenance_path=args.membership_provenance,
        exact_candidates_path=args.exact_candidates,
        output_root=args.output_root,
        build_run_id=args.build_run_id,
        config=SingleCellInputBuildConfig(
            min_association_observations=args.min_association_observations,
            low_feature_universe_threshold=args.low_feature_universe_threshold,
            association_chunk_size=args.association_chunk_size,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
