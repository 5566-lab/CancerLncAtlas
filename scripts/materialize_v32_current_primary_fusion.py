#!/usr/bin/env python3
"""Prepare private native validation supervision for fusion development on 149."""
import argparse
import json
from pathlib import Path

from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.current_primary_fusion import materialize_primary_fusion_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-core-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    path = materialize_primary_fusion_inputs(args.binding, args.output,
        expected_sha=args.expected_sha256, expected_core_run_id=args.expected_core_run_id)
    print(json.dumps({"status": "PREPARED_NOT_TRAINED", "binding_path": str(path), "binding_sha256": sha256(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
