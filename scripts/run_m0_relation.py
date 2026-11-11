#!/usr/bin/env python3
"""Run the policy-scoped relation M0, using a verified CCHHGT prepared capsule."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cla_engine.m0_relation import run_m0_relation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--policy', required=True, help='TARGET_POLICY.tsv')
    parser.add_argument('--out', required=True)
    parser.add_argument('--cancers', nargs='+', default=['BRCA', 'LUAD', 'COAD'])
    parser.add_argument('--identity-map', help='Explicit one-to-one source_target_id,canonical_id TSV')
    parser.add_argument('--penalty', choices=['l1', 'l2'], default='l1')
    args = parser.parse_args()
    receipt = run_m0_relation(args.prepared, args.policy, args.out,
        cancers=args.cancers, identity_map=args.identity_map, penalty=args.penalty)
    print(json.dumps({'format': receipt['format'], 'cancers': receipt['cancers'],
                      'targets': len(receipt['admitted_targets']), 'out': args.out,
                      'outer_test_read': receipt['outer_test_read']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
