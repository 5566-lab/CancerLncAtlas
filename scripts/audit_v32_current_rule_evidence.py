#!/usr/bin/env python3
"""Recompute an existing rule release from its sources and independent SQL."""
import argparse
import json
from pathlib import Path
import socket
from cc_hhgt.v32.evidence_rule_release import audit_rule_release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if socket.gethostname() != "149": raise RuntimeError("Rule evidence audit must execute on host 149")
    with args.output.with_suffix(args.output.suffix+".preflight.json").open("x", encoding="utf-8") as stream:
        json.dump({"target_host": "149", "training_started": False, "binding_sha256": args.expected_sha256}, stream)
    audit = audit_rule_release(args.binding, expected_sha=args.expected_sha256)
    with args.output.open("x", encoding="utf-8") as stream: json.dump(audit, stream, indent=2)
    return 0


if __name__ == "__main__": raise SystemExit(main())
