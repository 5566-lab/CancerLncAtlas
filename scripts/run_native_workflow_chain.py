"""Initialize or resume declared native workflows; no fitting is dispatched."""
import argparse
import json
from cc_hhgt.v32 import native_workflow_chain as workflow


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest="phase",required=True)
    plan=commands.add_parser("plan");plan.add_argument("--manifest",required=True)
    init=commands.add_parser("init");init.add_argument("--manifest",required=True);init.add_argument("--output",required=True)
    advance=commands.add_parser("advance");advance.add_argument("--output",required=True);advance.add_argument("--external-inputs")
    recover=commands.add_parser("recover");recover.add_argument("--output",required=True);recover.add_argument("--stage",required=True)
    recover.add_argument("--completed",required=True);recover.add_argument("--completed-sha256",required=True)
    args=parser.parse_args(argv)
    if args.phase=="plan":print(json.dumps(workflow.plan(args.manifest)));return 0
    if args.phase=="recover":
        from pathlib import Path
        path=workflow.recover(args.output,args.stage,{"path":str(Path(args.completed).resolve()),"sha256":args.completed_sha256})
    else:path=(workflow.initialize(args.manifest,args.output) if args.phase=="init"
               else workflow.advance(args.output,args.external_inputs))
    value=workflow.read(path)
    print(json.dumps({"binding":workflow.record(path),"status":value["status"],"training_started":False}))
    return 1 if value["status"]=="FAILED" else 0


if __name__=="__main__":raise SystemExit(main())
