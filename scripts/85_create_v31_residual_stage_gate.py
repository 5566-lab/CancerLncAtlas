#!/usr/bin/env python3
"""Freeze the exact code and inputs authorized for the three-cancer B2 run."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from cc_hhgt.common import file_sha256, load_config
from cc_hhgt.v30_integrity import atomic_write_json, merkle_sha256
from cc_hhgt.v31_graph_residual_stage import expected_graph_residual_matrix
from cc_hhgt.v31_pilot_gate import PILOT_CANCERS, PILOT_SEEDS


def _pass_gate(path: Path, name: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("failures"):
        raise RuntimeError(f"{name} gate is not PASS: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot-gate", required=True)
    parser.add_argument("--b1-gate", required=True)
    parser.add_argument("--best-simple-gate", required=True)
    parser.add_argument("--residual-base", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"Residual stage gate already exists: {output}")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status.strip():
        raise RuntimeError("Residual stage gate requires a clean git worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    pilot_gate_path = Path(args.pilot_gate).resolve()
    b1_gate_path = Path(args.b1_gate).resolve()
    simple_gate_path = Path(args.best_simple_gate).resolve()
    residual_base_path = Path(args.residual_base).resolve()
    pilot_gate = _pass_gate(pilot_gate_path, "Pilot")
    b1_gate = _pass_gate(b1_gate_path, "B1")
    simple_gate = _pass_gate(simple_gate_path, "BestSimple")
    if int(b1_gate.get("tasks_completed", 0)) != 12:
        raise RuntimeError("B1 gate is not the exact 12-task matrix")
    if simple_gate.get("selection_used_test_labels") is not False:
        raise RuntimeError("BestSimple gate lacks test-independent selection proof")
    if not residual_base_path.is_file():
        raise RuntimeError(f"Residual base is missing: {residual_base_path}")

    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, capture_output=True, check=True
    ).stdout.decode("utf-8").split("\0")
    code_rows = []
    for relative in sorted(item for item in tracked if item):
        path = repo / relative
        code_rows.append(
            {
                "relative_path": relative.replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    code_manifest = output.with_name("V31_RESIDUAL_CODE_MANIFEST.tsv")
    pd.DataFrame(code_rows).to_csv(code_manifest, sep="\t", index=False)
    code_contract = {
        "root": str(repo),
        "manifest": str(code_manifest),
        "manifest_sha256": file_sha256(code_manifest),
        "merkle_sha256": merkle_sha256(code_rows),
        "files": len(code_rows),
    }
    shrinkage_grid = list(map(float, cfg["residual_learning"]["shrinkage_grid"]))
    payload = {
        "status": "PASS",
        "analysis_version": cfg["analysis_version"],
        "git_commit": commit,
        "pilot_cancers": list(PILOT_CANCERS),
        "model_seeds": list(PILOT_SEEDS),
        "shrinkage_grid": shrinkage_grid,
        "expected_b2_tasks": len(expected_graph_residual_matrix(shrinkage_grid)),
        "full_cancer_training_authorized": False,
        "selection_scope": "pooled_outer_validation",
        "test_metric_controls_selection": False,
        "failures": [],
        "paths": {
            "output_root": str(Path(args.output_root).resolve()),
            "repo_root": str(repo),
            "config": str(Path(args.config).resolve()),
            "pilot_gate": str(pilot_gate_path),
            "b1_gate": str(b1_gate_path),
            "best_simple_gate": str(simple_gate_path),
            "residual_base": str(residual_base_path),
            "asset_results": pilot_gate["paths"]["asset_results"],
        },
        "manifest_contracts": [code_contract],
        "guarded_sha256": {
            str(Path(args.config).resolve()): file_sha256(Path(args.config)),
            str(pilot_gate_path): file_sha256(pilot_gate_path),
            str(b1_gate_path): file_sha256(b1_gate_path),
            str(simple_gate_path): file_sha256(simple_gate_path),
            str(residual_base_path): file_sha256(residual_base_path),
        },
        "asset_verification_at_task_boundaries": True,
        "resume_allowed": False,
    }
    atomic_write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
