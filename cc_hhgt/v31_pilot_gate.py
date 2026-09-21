"""Fail-closed provenance gate for the fixed three-cancer V3.1 pilot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import file_sha256
from .v30_integrity import verify_file_manifest


PILOT_CANCERS = ("BRCA", "COAD", "KIRP")
PILOT_SEEDS = (20260726, 20261726, 20262726)
PRIMARY_CONTRACT = "CONTRACT-T"
DIAGNOSTIC_CONTRACT = "CONTRACT-S"


def _records(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, sep="\t")
    required = {"relative_path", "size_bytes", "sha256"}
    if not required.issubset(frame):
        raise RuntimeError(
            f"Invalid manifest {path}: missing {sorted(required - set(frame))}"
        )
    return frame.to_dict("records")


def verify_manifest_contract(contract: dict[str, Any]) -> list[dict[str, str]]:
    root = Path(contract["root"]).resolve()
    manifest = Path(contract["manifest"]).resolve()
    if not manifest.is_file():
        return [{"relative_path": str(manifest), "expected": "EXISTS", "observed": "MISSING"}]
    observed_manifest = file_sha256(manifest)
    if observed_manifest.lower() != str(contract["manifest_sha256"]).lower():
        return [
            {
                "relative_path": str(manifest),
                "expected": str(contract["manifest_sha256"]),
                "observed": observed_manifest,
            }
        ]
    return verify_file_manifest(root, _records(manifest))


def validate_v31_pilot_gate(
    gate_path: str | Path,
    cfg: dict[str, Any],
    *,
    output_root: str | Path,
    verify_assets: bool = True,
) -> tuple[dict[str, Any], str]:
    path = Path(gate_path).resolve()
    if not path.is_file():
        raise RuntimeError(f"V3.1 pilot gate is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("failures"):
        raise RuntimeError(
            f"V3.1 pilot gate failed: status={payload.get('status')}, "
            f"failures={payload.get('failures')}"
        )
    if payload.get("analysis_version") != cfg.get("analysis_version"):
        raise RuntimeError("V3.1 pilot gate/config analysis_version mismatch")
    if Path(payload["paths"]["output_root"]).resolve() != Path(output_root).resolve():
        raise RuntimeError("V3.1 pilot output root differs from the registered gate")
    if tuple(payload.get("pilot_cancers", [])) != PILOT_CANCERS:
        raise RuntimeError("V3.1 pilot cancer scope drift")
    if tuple(payload.get("primary_seeds", [])) != PILOT_SEEDS:
        raise RuntimeError("V3.1 primary seed scope drift")
    if payload.get("primary_contract") != PRIMARY_CONTRACT:
        raise RuntimeError("V3.1 primary contract drift")
    if payload.get("diagnostic_contract") != DIAGNOSTIC_CONTRACT:
        raise RuntimeError("V3.1 diagnostic contract drift")
    if payload.get("full_cancer_training_authorized") is not False:
        raise RuntimeError("Pilot gate must explicitly forbid full-cancer training")
    phase_path = Path(payload["paths"]["phase_a2_gate"]).resolve()
    if not phase_path.is_file() or file_sha256(phase_path) != payload["phase_a2_gate_sha256"]:
        raise RuntimeError("Phase A2 gate changed after V3.1 pilot registration")
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    if phase.get("status") != "PASS" or phase.get("passed") is not True:
        raise RuntimeError("Phase A2 is not PASS")

    if verify_assets:
        mismatches: list[dict[str, str]] = []
        for contract in payload.get("manifest_contracts", []):
            mismatches.extend(verify_manifest_contract(contract))
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
            raise RuntimeError(f"V3.1 pilot assets changed: {mismatches[:20]}")
    return payload, file_sha256(path)

