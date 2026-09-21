#!/usr/bin/env python3
"""Materialize State Gene Sets from the fresh, typed V3.2 State release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.state_gene_set_release import materialize_state_gene_set_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-predictions', required=True, type=Path)
    parser.add_argument('--state-lineage', required=True, type=Path)
    parser.add_argument('--state-checkpoints', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fixture-authority', action='store_true')
    parser.add_argument('--cancer-max-members', type=int, default=100)
    parser.add_argument('--pan-cancer-max-members', type=int, default=200)
    parser.add_argument('--minimum-set-members', type=int, default=3)
    parser.add_argument('--pan-cancer-min-cancers', type=int, default=5)
    parser.add_argument('--pan-cancer-min-direction-consistency', type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_state_gene_set_release(
        state_prediction_path=args.state_predictions,
        state_lineage_path=args.state_lineage,
        state_checkpoint_manifest_path=args.state_checkpoints,
        output_root=args.output,
        runner_path=Path(__file__),
        strict_formal_authority=not args.fixture_authority,
        cancer_max_members=args.cancer_max_members,
        pan_cancer_max_members=args.pan_cancer_max_members,
        minimum_set_members=args.minimum_set_members,
        pan_cancer_min_cancers=args.pan_cancer_min_cancers,
        pan_cancer_min_direction_consistency=args.pan_cancer_min_direction_consistency,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
