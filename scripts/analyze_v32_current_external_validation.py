#!/usr/bin/env python3
"""Evaluate declared raw external observations against bound exact/fusion scores; no fitting."""
import argparse
import json
from pathlib import Path
from cc_hhgt.v32.external_validation_current import materialize_current_external


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(materialize_current_external(args.manifest, args.output)))
    return 0


if __name__ == "__main__": raise SystemExit(main())
