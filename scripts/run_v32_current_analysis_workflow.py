#!/usr/bin/env python3
"""Build current selected analyses, contexts and web registration on host 149.

This command consumes native results; it never starts an optimizer or deploys.
"""
import argparse
import json
from cc_hhgt.v32.current_analysis_workflow import run_current_analysis_workflow


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run_current_analysis_workflow(args.manifest,args.output)))
    return 0


if __name__=="__main__":raise SystemExit(main())
