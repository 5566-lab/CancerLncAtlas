"""Immutable code/input gate for the V3.1 Graph residual GPU stage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import file_sha256
from .v30_integrity import verify_file_manifest
from .v31_pilot_gate import PILOT_CANCERS, PILOT_SEEDS, _records


def validate_residual_stage_gate(
    path: str | Path,
    cfg: dict[str, Any],
    *,
    output_root: str | Path,
    verify_files: bool = True,
) -> tuple[dict[str, Any], str]:
    gate_path = Path(path).resolve()
    if not gate_path.is_file():
        raise RuntimeError(f"Residual stage gate is missing: {gate_path}")
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("failures"):
        raise RuntimeError("Residual stage gate is not PASS")
    if payload.get("analysis_version") != cfg.get("analysis_version"):
        raise RuntimeError("Residual stage gate/config analysis version mismatch")
    if Path(payload["paths"]["output_root"]).resolve() != Path(output_root).resolve():
        raise RuntimeError("Residual stage output root drift")
    if tuple(payload.get("pilot_cancers", [])) != PILOT_CANCERS:
        raise RuntimeError("Residual stage cancer scope drift")
    if tuple(map(int, payload.get("model_seeds", []))) != PILOT_SEEDS:
        raise RuntimeError("Residual stage seed scope drift")
    configured_grid = tuple(
        map(float, cfg.get("residual_learning", {}).get("shrinkage_grid", []))
    )
    registered_grid = tuple(map(float, payload.get("shrinkage_grid", [])))
    if configured_grid != registered_grid or not configured_grid:
        raise RuntimeError("Residual stage shrinkage grid drift")
    if int(payload.get("expected_b2_tasks", 0)) != len(PILOT_CANCERS) * len(PILOT_SEEDS) * len(configured_grid):
        raise RuntimeError("Residual stage task count drift")
    if payload.get("full_cancer_training_authorized") is not False:
        raise RuntimeError("Residual stage gate must forbid full-cancer training")
    if verify_files:
        mismatches: list[dict[str, str]] = []
        for contract in payload.get("manifest_contracts", []):
            manifest = Path(contract["manifest"]).resolve()
            observed = file_sha256(manifest) if manifest.is_file() else "MISSING"
            if observed.lower() != str(contract["manifest_sha256"]).lower():
                mismatches.append(
                    {
                        "relative_path": str(manifest),
                        "expected": str(contract["manifest_sha256"]),
                        "observed": observed,
                    }
                )
                continue
            mismatches.extend(
                verify_file_manifest(Path(contract["root"]), _records(manifest))
            )
        for raw_path, expected in payload.get("guarded_sha256", {}).items():
            guarded = Path(raw_path)
            observed = file_sha256(guarded) if guarded.is_file() else "MISSING"
            if observed.lower() != str(expected).lower():
                mismatches.append(
                    {
                        "relative_path": str(guarded),
                        "expected": str(expected),
                        "observed": observed,
                    }
                )
        if mismatches:
            raise RuntimeError(f"Residual stage assets changed: {mismatches[:20]}")
    return payload, file_sha256(gate_path)
