#!/usr/bin/env python3
"""Read-only compatibility audit for the interrupted G0/G1/G2 run.

This audit is intentionally metadata-only: it never mutates a checkpoint,
loads CUDA, or attempts a resume.  It records why the old 4090 checkpoints
cannot be used as formal input to a new guard/code or 5090 run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
from pathlib import Path
from typing import Any, Mapping

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_summary(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    runtime = state.get("runtime_fingerprint")
    artifacts = state.get("artifact_hashes")
    guard = state.get("last_optimizer_guard")
    if not isinstance(runtime, Mapping):
        runtime = {}
    if not isinstance(artifacts, Mapping):
        artifacts = {}
    if not isinstance(guard, Mapping):
        guard = {}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "checkpoint_format": state.get("checkpoint_format"),
        "task_id": state.get("task_id"),
        "patient_fold": state.get("patient_fold"),
        "graph_variant": state.get("graph_variant"),
        "cycle": state.get("cycle"),
        "optimizer_steps": state.get("optimizer_steps"),
        "code_sha256": artifacts.get("code_sha256"),
        "artifact_hashes": dict(artifacts),
        "runtime_fingerprint": dict(runtime),
        "hardware_class": runtime.get("hardware_class"),
        "device": runtime.get("device"),
        "gpu_name": runtime.get("gpu_name"),
        "cuda_runtime": runtime.get("cuda_runtime"),
        "torch_version": runtime.get("torch_version"),
        "parameter_delta_positive": guard.get("parameter_delta_positive"),
        "update_probe_parameter": guard.get("update_probe_parameter"),
        "last_optimizer_guard": dict(guard),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--static-auth", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.evidence_root.resolve()
    static_auth = load_json(args.static_auth.resolve())
    checkpoint_paths = sorted(root.rglob("last_training_state.pt"))
    success_paths = sorted(root.rglob("SUCCESS.json"))
    checkpoints = [checkpoint_summary(path) for path in checkpoint_paths]
    old_code_hashes = sorted(
        {str(row.get("code_sha256")) for row in checkpoints if row.get("code_sha256")}
    )
    current_code_hash = str(static_auth.get("code_tree_sha256", ""))
    runtime_hardware = sorted(
        {str(row.get("gpu_name")) for row in checkpoints if row.get("gpu_name")}
    )
    report = {
        "format": "CANCERLNCATLAS_V32_G012_CHECKPOINT_COMPATIBILITY_AUDIT_V1",
        "status": "PASS_METADATA_ONLY",
        "host_observed": socket.gethostname(),
        "host_expected": "149",
        "evidence_root": str(root),
        "static_auth": str(args.static_auth.resolve()),
        "checkpoint_count": len(checkpoints),
        "success_count": len(success_paths),
        "checkpoints": checkpoints,
        "old_checkpoint_code_sha256": old_code_hashes,
        "current_static_auth_code_tree_sha256": current_code_hash,
        "code_hashes_equal_to_current_static_auth": bool(old_code_hashes)
        and old_code_hashes == [current_code_hash],
        "future_guard_fix_requires_new_code_hash": True,
        "old_runtime_gpu_names": runtime_hardware,
        "formal_resume_on_149": False,
        "formal_resume_on_rtx5090": False,
        "formal_resume_allowed": False,
        "reasons": [
            "the archived checkpoints match the current pre-fix static-auth code hash",
            "the planned zero-delta guard fix must be a new code namespace/hash",
            "training.py rejects artifact_hashes drift before loading optimizer state",
            "training.py rejects runtime_fingerprint drift across 4090, CPU/149, and 5090",
            "the current corrected launcher requires new-run SUCCESS artifacts and forbids old checkpoint loading",
        ],
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
