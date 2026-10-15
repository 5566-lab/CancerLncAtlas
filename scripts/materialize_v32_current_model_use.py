#!/usr/bin/env python3
"""Inspect native model inputs and their unresolved provenance on host 149."""
import argparse
import json
from cc_hhgt.v32.current_model_use import materialize_current_model_use


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(materialize_current_model_use(args.manifest,args.output)))
    return 0


if __name__=="__main__":raise SystemExit(main())
