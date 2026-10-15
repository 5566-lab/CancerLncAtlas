#!/usr/bin/env python3
"""Replay a selected score, ranked Gene Sets, subtypes, coverage and GMT on 149."""
import argparse
import json
from pathlib import Path
import socket
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cc_hhgt.v32.selected_genesets import audit_selected_analysis


def main(argv=None):
    if socket.gethostname() != "149": raise RuntimeError("Selected analysis audit must execute on host 149")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding",type=Path,required=True)
    parser.add_argument("--expected-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    preflight=args.output.with_name(args.output.name+".preflight.json")
    if args.output.exists() or preflight.exists(): raise ValueError("Audit output must be new")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    preflight.write_text(json.dumps({"target_host":"149","training_started":False,"binding_sha256":args.expected_sha256}))
    result=audit_selected_analysis(args.binding,expected_sha=args.expected_sha256)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__": raise SystemExit(main())
