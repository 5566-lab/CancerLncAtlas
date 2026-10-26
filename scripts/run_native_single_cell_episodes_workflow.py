"""Prepare native episode assembly or collect completed assets; never fit LASSO."""
import argparse
import json
from pathlib import Path
from cc_hhgt.v32 import native_single_cell_episodes_workflow as workflow


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='phase',required=True)
    prepare=sub.add_parser('prepare'); prepare.add_argument('--manifest',type=Path,required=True)
    prepare.add_argument('--output',type=Path,required=True); prepare.add_argument('--plan',action='store_true')
    collect=sub.add_parser('collect')
    for name in ('preparation','completed'):
        collect.add_argument('--'+name,type=Path,required=True); collect.add_argument('--'+name+'-sha256',required=True)
    collect.add_argument('--output',type=Path,required=True); args=parser.parse_args(argv)
    if args.phase=='prepare':
        if args.plan: print(json.dumps(workflow.plan(args.manifest,args.output))); return 0
        path=workflow.prepare(args.manifest,args.output)
    else:
        path=workflow.collect({'path':str(args.preparation.resolve()),'sha256':args.preparation_sha256},
            {'path':str(args.completed.resolve()),'sha256':args.completed_sha256},args.output)
    print(json.dumps({'binding':workflow.record(path),'training_started':False})); return 0


if __name__=='__main__': raise SystemExit(main())
