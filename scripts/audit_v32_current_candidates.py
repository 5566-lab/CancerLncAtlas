#!/usr/bin/env python3
"""Audit current predictions, optionally replaying the five frozen models on 149."""
import argparse
import json
from pathlib import Path
import socket
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cc_hhgt.v32.current_core_prediction import audit_current_predictions


def main(argv=None):
    if socket.gethostname()!='149':raise RuntimeError('Current prediction audit requires host 149')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay-models',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args(argv)
    preflight=args.output.with_name(args.output.name+'.preflight.json')
    if args.output.exists() or preflight.exists():raise ValueError('Audit output must be new')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    preflight.write_text(json.dumps({'target_host':'149','training_started':False,
        'replay_models':args.replay_models,'binding_sha256':args.expected_sha256}))
    result=audit_current_predictions(args.binding,expected_sha=args.expected_sha256,replay_models=args.replay_models,device=args.device)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
