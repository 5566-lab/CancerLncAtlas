#!/usr/bin/env python3
"""Bind native signed donor-OOF contexts to the current selected analysis."""
import argparse
import json
from cc_hhgt.v32.current_single_cell_context import materialize_current_single_cell_context


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(materialize_current_single_cell_context(args.manifest,args.output)))
    return 0


if __name__=="__main__":raise SystemExit(main())
