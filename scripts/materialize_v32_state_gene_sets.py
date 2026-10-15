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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current-manifest', type=Path,
        help='CURRENT_STATE_CONTEXT_INPUT_V1: native State lineage, current core and selected analysis; set policy lives in the manifest')
    parser.add_argument('--state-predictions', type=Path)
    parser.add_argument('--state-lineage', type=Path)
    parser.add_argument('--state-checkpoints', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fixture-authority', action='store_true')
    parser.add_argument('--cancer-max-members', type=int)
    parser.add_argument('--pan-cancer-max-members', type=int)
    parser.add_argument('--minimum-set-members', type=int)
    parser.add_argument('--pan-cancer-min-cancers', type=int)
    parser.add_argument('--pan-cancer-min-direction-consistency', type=float)
    args = parser.parse_args(argv)
    legacy = [args.state_predictions, args.state_lineage, args.state_checkpoints]
    if args.current_manifest and (any(legacy) or args.fixture_authority):
        parser.error('Current native State inputs cannot be mixed with legacy or fixture authority')
    if not args.current_manifest and not all(legacy):
        parser.error('Provide --current-manifest or all three legacy State inputs')
    defaults = dict(cancer_max_members=100, pan_cancer_max_members=200, minimum_set_members=3,
        pan_cancer_min_cancers=5, pan_cancer_min_direction_consistency=.60)
    if args.current_manifest and any(getattr(args,key) is not None for key in defaults):
        parser.error('Current State set policy must be recorded in the input manifest')
    for key,value in defaults.items():
        if getattr(args,key) is None: setattr(args,key,value)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.current_manifest:
        from cc_hhgt.v32.current_state_context import materialize_current_state_context
        result = materialize_current_state_context(args.current_manifest, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
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
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
