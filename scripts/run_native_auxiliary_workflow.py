#!/usr/bin/env python3
"""Prepare native CNV/ATAC inputs or collect completed results on 149; never fit."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cc_hhgt.v32 import native_auxiliary_workflow as workflow
from cc_hhgt.v32.formal_expression_sources import record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--plan", action="store_true")
    collect = sub.add_parser("collect")
    collect.add_argument("--preparation", type=Path, required=True)
    collect.add_argument("--preparation-sha256", required=True)
    collect.add_argument("--completed", type=Path, required=True)
    collect.add_argument("--completed-sha256", required=True)
    collect.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "prepare":
        if args.plan:
            print(json.dumps(workflow.plan(args.manifest, args.output))); return 0
        path = workflow.prepare(args.manifest, args.output)
    else:
        path = workflow.collect({"path":str(args.preparation.resolve()), "sha256":args.preparation_sha256},
            {"path":str(args.completed.resolve()), "sha256":args.completed_sha256}, args.output)
    print(json.dumps({"binding":record(path), "training_started":False})); return 0


if __name__ == "__main__": raise SystemExit(main())
