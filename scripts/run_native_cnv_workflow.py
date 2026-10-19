"""Execute declared native CNV statistics/corrected E1 on 149 without fitting."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from cc_hhgt.v32.native_cnv_workflow import preflight,run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--plan",action="store_true")
    args=parser.parse_args(argv)
    if args.plan: print(json.dumps(preflight(args.manifest,args.output),indent=2))
    else: print(run(args.manifest,args.output))


if __name__=="__main__":main()
