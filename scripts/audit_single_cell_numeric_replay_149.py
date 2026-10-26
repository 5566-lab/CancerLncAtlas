"""Reconstruct completed SC-HHGT predictions without fitting on host 149."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    from cc_hhgt.v32.single_cell_numeric_replay import audit,record
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root",type=Path,required=True)
    parser.add_argument("--training-success-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args(argv)
    result=audit(args.training_root,expected_success_sha256=args.training_success_sha256,output=args.output)
    print(json.dumps({"binding":record(result),"training_started":False}));return 0


if __name__=="__main__":raise SystemExit(main())
