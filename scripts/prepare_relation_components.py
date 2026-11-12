#!/usr/bin/env python3
"""Materialize source features for a frozen dual-cohort relation view on 149."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cla_engine.relation_components import materialize_relation_components, CONTEXTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', required=True, type=Path,
                        help='Pinned input bindings or a previous LAUNCH_PREFLIGHT.json')
    parser.add_argument('--cancer', required=True)
    parser.add_argument('--context', required=True, choices=CONTEXTS)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--budget', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=20261111)
    args = parser.parse_args()
    document = json.loads(args.bindings.read_text())
    result = materialize_relation_components(document.get('bindings', document), args.cancer,
        args.context, args.out, budget=args.budget, seed=args.seed,
        progress=lambda item: print(json.dumps(item), flush=True))
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'code_files')}, indent=2))


if __name__ == '__main__':
    main()
