#!/usr/bin/env python3
"""Finalize only the exact B1 three-cancer Fixed Graph matrix."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from cc_hhgt.v31_fixed_graph import finalize_fixed_graph


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot-gate", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    if status.strip():
        raise RuntimeError("B1 finalization requires a clean aggregation worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    payload = finalize_fixed_graph(
        Path(args.fixed_root).resolve(),
        Path(args.output_dir).resolve(),
        config_path=Path(args.config).resolve(),
        pilot_gate_path=Path(args.pilot_gate).resolve(),
        aggregation_git_commit=commit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
