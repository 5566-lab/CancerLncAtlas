#!/usr/bin/env python3
"""Build one fresh V3.2 single-cell cancer partition from source H5."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_partition_builder import (  # noqa: E402
    build_single_cell_partition,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", required=True)
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--annotation-provenance", required=True, type=Path)
    parser.add_argument("--exact-membership", required=True, type=Path)
    parser.add_argument("--exact-candidates", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-tier", required=True)
    parser.add_argument("--measurement-scale", required=True)
    parser.add_argument("--audited-lnc-feature-count", required=True, type=int)
    parser.add_argument("--min-association-observations", type=int, default=5)
    parser.add_argument("--association-chunk-size", type=int, default=100_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_single_cell_partition(
        cancer_id=args.cancer,
        h5_path=args.h5,
        metadata_path=args.metadata,
        annotation_parquet_path=args.annotation,
        annotation_provenance_path=args.annotation_provenance,
        membership_path=args.exact_membership,
        candidate_path=args.exact_candidates,
        output_root=args.output_root,
        source_tier=args.source_tier,
        measurement_scale=args.measurement_scale,
        audited_lnc_feature_count=args.audited_lnc_feature_count,
        min_association_observations=args.min_association_observations,
        association_chunk_size=args.association_chunk_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
