"""Materialize a pinned GENCODE GTF for native bulk consumers on host 149."""
import argparse
import json
from cc_hhgt.v32 import native_annotation_workflow as workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--plan", action="store_true"); args = parser.parse_args(argv)
    if args.plan: print(json.dumps(workflow.plan(args.manifest, args.output))); return 0
    print(json.dumps({"binding": workflow.record(workflow.run(args.manifest, args.output)), "training_started": False}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
