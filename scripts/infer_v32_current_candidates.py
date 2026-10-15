#!/usr/bin/env python3
"""Infer all candidates with the frozen current V4 core on 149, without labels or fitting."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cc_hhgt.v32.current_core_prediction import materialize_current_predictions
from cc_hhgt.v32.current_core_export import sha256


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--expected-core-run-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args(argv)
    binding=materialize_current_predictions(args.manifest,args.output,expected_run_id=args.expected_core_run_id,device=args.device)
    print(json.dumps({'binding':str(binding),'sha256':sha256(binding),'training_started':False}))
    return 0


if __name__=='__main__':raise SystemExit(main())
