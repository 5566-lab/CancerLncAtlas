#!/usr/bin/env python3
'''Materialize fresh V3.2 bulk expression landscape and exact-model coverage.'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
V32_MODULE_ROOT = ROOT / 'cc_hhgt' / 'v32'
if str(V32_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(V32_MODULE_ROOT))

from bulk_expression_release import materialize_bulk_expression_release  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detection-grid', required=True, type=Path)
    parser.add_argument('--threshold-success', required=True, type=Path)
    parser.add_argument('--expression-root', required=True, type=Path)
    parser.add_argument('--exact-candidates', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fixture-authority', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_bulk_expression_release(
        detection_grid_path=args.detection_grid,
        threshold_success_path=args.threshold_success,
        expression_root=args.expression_root,
        exact_candidate_path=args.exact_candidates,
        output_root=args.output,
        runner_path=Path(__file__),
        strict_formal_authority=not args.fixture_authority,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
