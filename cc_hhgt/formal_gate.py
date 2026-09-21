from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import file_sha256
from .v30_integrity import verify_file_manifest


def _records(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, sep="\t")
    required = {"relative_path", "size_bytes", "sha256"}
    if not required.issubset(frame):
        raise RuntimeError(f"Invalid formal manifest {path}: missing {sorted(required - set(frame))}")
    return frame.to_dict("records")


def validate_formal_training_gate(
    path: str | Path,
    cfg: dict[str, Any],
    *,
    verify_assets: bool,
    trusted_sha256: str | None = None,
    run_id: str | None = None,
    input_root: str | Path | None = None,
    input_manifest: str | Path | None = None,
    output_root: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate the fail-closed V3 formal gate and all frozen assets.

    Asset verification cannot be disabled by a trusted hash.  The legacy
    argument remains only so old callers fail with an explicit explanation.
    """
    if trusted_sha256 is not None:
        raise RuntimeError("V3 formal training forbids trusted-gate asset-validation bypass")
    if not verify_assets:
        raise RuntimeError("V3 formal training requires full asset verification at every task boundary")
    gate_path = Path(path).resolve()
    if not gate_path.exists():
        raise RuntimeError(f"Formal training gate is missing: {gate_path}")
    gate_sha256 = file_sha256(gate_path)
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("required_failures"):
        raise RuntimeError(
            f"Formal training gate did not pass: status={payload.get('status')}, "
            f"failures={payload.get('required_failures', [])}"
        )
    if payload.get("analysis_version") != cfg.get("analysis_version"):
        raise RuntimeError(
            f"Formal gate/config version mismatch: {payload.get('analysis_version')} != {cfg.get('analysis_version')}"
        )
    contract = cfg.get("formal_contract", {})
    expected_tasks = int(contract.get("expected_tasks", 0))
    expected_folds = int(contract.get("expected_folds", 0))
    expected_models = int(contract.get("expected_models", 0))
    expected_seeds = int(contract.get("expected_seeds", 0))
    if expected_tasks != expected_folds * expected_models * expected_seeds:
        raise RuntimeError(
            "Formal config task matrix is internally inconsistent: "
            f"{expected_folds} x {expected_models} x {expected_seeds} != {expected_tasks}"
        )
    if int(payload.get("expected_tasks", 0)) != expected_tasks:
        raise RuntimeError(
            f"Formal training gate must authorize exactly {expected_tasks} tasks, "
            f"observed {payload.get('expected_tasks')}"
        )
    expected_run = run_id or payload.get("run_id")
    if expected_run != payload.get("run_id"):
        raise RuntimeError(f"Formal gate run_id mismatch: expected={expected_run}, gate={payload.get('run_id')}")
    if not all([input_root, input_manifest, output_root]):
        raise RuntimeError("V3 formal gate validation requires input_root, input_manifest, and output_root")
    resolved_input = Path(input_root).resolve()
    resolved_manifest = Path(input_manifest).resolve()
    resolved_output = Path(output_root).resolve()
    contracts = payload.get("paths", {})
    expected_paths = {
        "input_root": resolved_input,
        "input_manifest": resolved_manifest,
        "output_root": resolved_output,
    }
    for key, observed in expected_paths.items():
        expected = Path(contracts.get(key, "__MISSING__")).resolve()
        if observed != expected:
            raise RuntimeError(f"Formal gate path mismatch for {key}: expected={expected}, observed={observed}")

    mismatches: list[dict[str, str]] = []
    for contract in payload.get("manifest_contracts", []):
        root = Path(contract["root"]).resolve()
        manifest = Path(contract["manifest"]).resolve()
        if not manifest.exists():
            mismatches.append({"relative_path": str(manifest), "expected": "EXISTS", "observed": "MISSING"})
            continue
        if file_sha256(manifest).lower() != str(contract["manifest_sha256"]).lower():
            mismatches.append({"relative_path": str(manifest), "expected": str(contract["manifest_sha256"]), "observed": file_sha256(manifest)})
            continue
        mismatches.extend(verify_file_manifest(root, _records(manifest)))
    for raw_path, expected in payload.get("guarded_sha256", {}).items():
        asset = Path(raw_path)
        observed = file_sha256(asset) if asset.is_file() else "MISSING"
        if observed.lower() != str(expected).lower():
            mismatches.append({"relative_path": str(asset), "expected": str(expected), "observed": observed})
    if mismatches:
        raise RuntimeError(f"Formal assets changed after the gate: {mismatches[:20]}")
    return payload, gate_sha256
