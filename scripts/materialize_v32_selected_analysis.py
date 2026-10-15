#!/usr/bin/env python3
"""Materialize selected-score Gene Sets, ranked subtypes, coverage and GMT on 149."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cc_hhgt.v32.selected_genesets import materialize_selected_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(materialize_selected_analysis(args.manifest, args.output)))
    return 0


if __name__ == "__main__": raise SystemExit(main())
