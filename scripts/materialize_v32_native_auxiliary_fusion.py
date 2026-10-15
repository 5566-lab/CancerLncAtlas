#!/usr/bin/env python3
"""Consume native Mutation/CNV/ATAC OOF outputs on host 149 without fitting."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cc_hhgt.v32.native_auxiliary_fusion import materialize_native_auxiliary
from cc_hhgt.v32.current_core_export import sha256


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--expected-core-run-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    binding=materialize_native_auxiliary(args.manifest,args.output,expected_core_run_id=args.expected_core_run_id)
    print(json.dumps({'binding':str(binding),'sha256':sha256(binding),'training_started':False}));return 0


if __name__=='__main__':raise SystemExit(main())
