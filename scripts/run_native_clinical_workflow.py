"""Prepare native Clinical inputs or collect completed results; never dispatch fit."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    from cc_hhgt.v32 import native_clinical_workflow as workflow
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="phase",required=True)
    prepare=sub.add_parser("prepare");prepare.add_argument("--manifest",type=Path,required=True)
    prepare.add_argument("--output",type=Path,required=True);prepare.add_argument("--plan",action="store_true")
    collect=sub.add_parser("collect");collect.add_argument("--preparation",type=Path,required=True)
    collect.add_argument("--preparation-sha256",required=True)
    for kind in ("risk","entity"):
        collect.add_argument("--"+kind+"-completed",type=Path,required=True)
        collect.add_argument("--"+kind+"-completed-sha256",required=True)
    collect.add_argument("--output",type=Path,required=True);args=parser.parse_args(argv)
    if args.phase=="prepare":
        if args.plan:print(json.dumps(workflow.plan(args.manifest,args.output)));return 0
        path=workflow.prepare(args.manifest,args.output)
    else:
        path=workflow.collect({"path":str(args.preparation.resolve()),"sha256":args.preparation_sha256},
            {"path":str(args.risk_completed.resolve()),"sha256":args.risk_completed_sha256},
            {"path":str(args.entity_completed.resolve()),"sha256":args.entity_completed_sha256},args.output)
    print(json.dumps({"binding":workflow.record(path),"training_started":False}));return 0


if __name__=="__main__":raise SystemExit(main())
