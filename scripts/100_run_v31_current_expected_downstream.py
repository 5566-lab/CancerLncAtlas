#!/usr/bin/env python3
"""Run HARD-GO-only R_SELECTED atlas downstream benchmarks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from cc_hhgt.v31_current_expected_downstream import (
    run_current_expected_downstream,
)


def _clean_commit(repo: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status.strip():
        raise RuntimeError("Current Expected downstream requires a clean worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-predictions", required=True)
    parser.add_argument("--b3-gate", required=True)
    parser.add_argument("--graph-predictions", required=True)
    parser.add_argument("--graph-gate", required=True)
    parser.add_argument("--hard-report", required=True)
    parser.add_argument("--frozen-dual-axis-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    commit = _clean_commit(repo)
    payload = run_current_expected_downstream(
        selected_predictions_path=Path(args.selected_predictions).resolve(),
        b3_gate_path=Path(args.b3_gate).resolve(),
        graph_predictions_path=Path(args.graph_predictions).resolve(),
        graph_gate_path=Path(args.graph_gate).resolve(),
        hard_report_path=Path(args.hard_report).resolve(),
        frozen_dual_axis_root=Path(args.frozen_dual_axis_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        aggregation_git_commit=commit,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
