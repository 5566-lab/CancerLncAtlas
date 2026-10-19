"""Use completed native runs for frozen inference, selection and publication."""
import argparse
import json
from cc_hhgt.v32.native_results_workflow import preflight, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if args.plan: print(json.dumps(preflight(args.manifest, args.output), indent=2))
    else: print(run(args.manifest, args.output))
    return 0


if __name__ == "__main__": raise SystemExit(main())
