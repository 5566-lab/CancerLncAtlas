#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], repo: Path) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=repo)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="V3 default: formal gate -> isolated 43c -> release audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--gate-output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    shared = [
        "--config", args.config,
        "--run-id", args.run_id,
        "--input-root", args.input_root,
        "--input-manifest", args.input_manifest,
    ]
    run(
        [args.python, "scripts/71_v30_formal_training_gate.py", *shared,
         "--run-root", args.run_root, "--output-root", args.training_root,
         "--output", args.gate_output],
        repo,
    )
    run(
        [args.python, "scripts/43c_run_v29_strict_matrix_parallel.py", *shared,
         "--formal-gate", args.gate_output, "--output-root", args.training_root,
         "--python", args.python, "--jobs", str(args.jobs),
         *(["--pilot"] if args.pilot else [])],
        repo,
    )
    run(
        [args.python, "scripts/72_audit_v30_state_matrix.py", *shared,
         "--formal-gate", args.gate_output, "--training-root", args.training_root,
         "--output-root", args.audit_output,
         *(["--pilot"] if args.pilot else [])],
        repo,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
