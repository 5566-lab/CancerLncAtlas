#!/usr/bin/env python3
"""Register the exact A2-approved V3.1 three-cancer B1 scope."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from cc_hhgt.common import file_sha256, load_config
from cc_hhgt.v30_integrity import atomic_write_json, merkle_sha256, verify_file_manifest
from cc_hhgt.v31_pilot_gate import (
    DIAGNOSTIC_CONTRACT,
    PILOT_CANCERS,
    PILOT_SEEDS,
    PRIMARY_CONTRACT,
    _records,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase-a2-gate", required=True)
    parser.add_argument("--source-formal-gate", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    output = Path(args.output).resolve()
    if output_root.exists():
        raise RuntimeError(f"Pilot training root must be new at gate creation: {output_root}")
    if output.exists():
        raise RuntimeError(f"Pilot gate output already exists: {output}")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    if status.strip():
        raise RuntimeError("Pilot gate requires a clean git worktree")
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    phase_path = Path(args.phase_a2_gate).resolve()
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    if phase.get("status") != "PASS" or phase.get("passed") is not True:
        raise RuntimeError("Phase A2 did not pass")
    phase_commit = str(phase.get("git_commit", ""))
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", phase_commit, git_commit], cwd=repo
    )
    if ancestor.returncode:
        raise RuntimeError("Current B1 code is not descended from the A2-audited graph core")

    source_gate_path = Path(args.source_formal_gate).resolve()
    source_gate = json.loads(source_gate_path.read_text(encoding="utf-8"))
    if source_gate.get("status") != "PASS":
        raise RuntimeError("Frozen V3.0 source gate is not PASS")
    source_contracts = source_gate.get("manifest_contracts", [])
    if len(source_contracts) < 3:
        raise RuntimeError("Frozen V3.0 source gate lacks input/asset contracts")
    # The old code contract is intentionally replaced by the current tracked
    # code manifest.  Frozen input and built assets remain byte-identical.
    frozen_contracts = [source_contracts[0], source_contracts[2]]
    mismatches = []
    for contract in frozen_contracts:
        mismatches.extend(
            verify_file_manifest(Path(contract["root"]), _records(Path(contract["manifest"])))
        )
    if mismatches:
        raise RuntimeError(f"Frozen source assets changed: {mismatches[:20]}")

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
    code_manifest = output.with_name("V31_PILOT_CODE_MANIFEST.tsv")
    pd.DataFrame(code_rows).to_csv(code_manifest, sep="\t", index=False)
    code_contract = {
        "root": str(repo),
        "manifest": str(code_manifest),
        "manifest_sha256": file_sha256(code_manifest),
        "merkle_sha256": merkle_sha256(code_rows),
        "files": len(code_rows),
    }
    manifest_contracts = [*frozen_contracts, code_contract]
    payload = {
        "status": "PASS",
        "analysis_version": cfg["analysis_version"],
        "git_commit": git_commit,
        "phase_a2_git_commit": phase_commit,
        "pilot_cancers": list(PILOT_CANCERS),
        "primary_contract": PRIMARY_CONTRACT,
        "diagnostic_contract": DIAGNOSTIC_CONTRACT,
        "primary_seeds": list(PILOT_SEEDS),
        "diagnostic_seeds": [PILOT_SEEDS[0]],
        "expected_primary_tasks": 9,
        "expected_diagnostic_tasks": 3,
        "expected_b1_tasks": 12,
        "full_cancer_training_authorized": False,
        "failures": [],
        "paths": {
            "output_root": str(output_root),
            "asset_results": source_gate["paths"]["asset_results"],
            "phase_a2_gate": str(phase_path),
            "source_formal_gate": str(source_gate_path),
            "repo_root": str(repo),
            "config": str(Path(args.config).resolve()),
        },
        "phase_a2_gate_sha256": file_sha256(phase_path),
        "source_formal_gate_sha256": file_sha256(source_gate_path),
        "manifest_contracts": manifest_contracts,
        "guarded_sha256": {
            str(Path(args.config).resolve()): file_sha256(Path(args.config)),
            str(phase_path): file_sha256(phase_path),
            str(source_gate_path): file_sha256(source_gate_path),
        },
        "asset_verification_at_task_boundaries": True,
        "resume_allowed": False,
    }
    atomic_write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
