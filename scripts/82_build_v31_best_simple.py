#!/usr/bin/env python3
"""Build BestSimpleLOCO only after the exact Fixed Graph B1 gate passes."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from cc_hhgt.v31_simple_builder import build_best_simple


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-results", required=True)
    parser.add_argument("--fixed-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--b1-gate", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    if status.strip():
        raise RuntimeError("BestSimpleLOCO requires a clean aggregation worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    payload = build_best_simple(
        Path(args.asset_results).resolve(),
        Path(args.fixed_root).resolve(),
        Path(args.output_dir).resolve(),
        b1_gate_path=Path(args.b1_gate).resolve(),
        fold_manifest_path=Path(args.fold_manifest).resolve(),
        aggregation_git_commit=commit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
