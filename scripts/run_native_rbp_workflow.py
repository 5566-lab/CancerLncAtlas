"""Execute cached ENCODE KD or RBNS analysis on 149 without fitting/downloads."""
import argparse
import json
from cc_hhgt.v32 import native_rbp_workflow as workflow


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--plan",action="store_true");args=parser.parse_args(argv)
    if args.plan:print(json.dumps(workflow.plan(args.manifest,args.output)));return 0
    path=workflow.run(args.manifest,args.output)
    print(json.dumps({"binding":workflow.record(path),"training_started":False}));return 0


if __name__=="__main__":raise SystemExit(main())
