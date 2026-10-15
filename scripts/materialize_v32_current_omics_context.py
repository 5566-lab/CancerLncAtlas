#!/usr/bin/env python3
"""Connect native genomic or distal analyses to the same selected release."""
import argparse
import json
from cc_hhgt.v32.current_omics_context import materialize_current_omics_context


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(materialize_current_omics_context(args.manifest,args.output)))
    return 0


if __name__=="__main__":raise SystemExit(main())
