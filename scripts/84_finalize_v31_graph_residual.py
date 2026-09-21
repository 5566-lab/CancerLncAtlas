#!/usr/bin/env python3
"""Finalize Graph residual only after the exact three-cancer matrix finishes."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from cc_hhgt.common import load_config
from cc_hhgt.v31_graph_residual_stage import finalize_graph_residual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--best-simple-gate", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status.strip():
        raise RuntimeError("Graph residual finalization requires a clean worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    cfg = load_config(args.config)
    payload = finalize_graph_residual(
        Path(args.run_root).resolve(),
        Path(args.output_dir).resolve(),
        lambdas=list(map(float, cfg["residual_learning"]["shrinkage_grid"])),
        best_simple_gate_path=Path(args.best_simple_gate).resolve(),
        aggregation_git_commit=commit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
