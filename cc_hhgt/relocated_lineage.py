"""Bind relocated formal outputs to both training and execution gates.

The formal training gate contains absolute paths.  A byte-identical gate can
therefore only be revalidated on the machine where it was issued.  Formal
outputs copied to another machine still need two independent guarantees:

1. each task remains bound to the original training-gate SHA256; and
2. the relocated input, code, and asset files pass a newly issued local gate.

This module compares the complete gate documents after removing only their
machine-specific absolute paths.  It does not provide an asset-validation
bypass; callers must first fully validate the execution-site gate.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import file_sha256
from .v30_integrity import canonical_json_sha256


def normalized_relocatable_gate(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the full gate contract with only absolute locations normalized."""
    normalized = copy.deepcopy(payload)

    paths = normalized.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise RuntimeError("Formal gate lacks the registered path contract")
    normalized["paths"] = {str(key): "<RELOCATED_ABSOLUTE_PATH>" for key in sorted(paths)}

    manifests = normalized.get("manifest_contracts")
    if not isinstance(manifests, list) or not manifests:
        raise RuntimeError("Formal gate lacks frozen manifest contracts")
    for index, contract in enumerate(manifests):
        if not isinstance(contract, dict) or "root" not in contract or "manifest" not in contract:
            raise RuntimeError(f"Invalid formal manifest contract at index {index}")
        contract["root"] = "<RELOCATED_MANIFEST_ROOT>"
        contract["manifest"] = "<RELOCATED_MANIFEST_PATH>"

    guarded = normalized.get("guarded_sha256")
    if not isinstance(guarded, dict) or not guarded:
        raise RuntimeError("Formal gate lacks guarded SHA256 assets")
    # Absolute path keys differ by machine.  The multiset retains duplicate
    # hashes and is sufficient because the execution gate independently maps
    # and verifies every local path before this comparison is called.
    normalized["guarded_sha256"] = sorted(str(value).lower() for value in guarded.values())
    return normalized


def validate_relocated_training_lineage(
    training_gate_path: str | Path,
    execution_gate: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    """Validate a training gate against an already verified execution gate.

    Returns ``(training_gate, training_gate_sha256, normalized_contract_sha256)``.
    """
    path = Path(training_gate_path).resolve()
    if not path.is_file():
        raise RuntimeError(f"Training-lineage gate is missing: {path}")
    training_gate = json.loads(path.read_text(encoding="utf-8"))
    if training_gate.get("status") != "PASS" or training_gate.get("required_failures"):
        raise RuntimeError("Training-lineage gate is not a passing formal gate")
    if execution_gate.get("status") != "PASS" or execution_gate.get("required_failures"):
        raise RuntimeError("Execution-site gate is not a passing formal gate")

    training_contract = normalized_relocatable_gate(training_gate)
    execution_contract = normalized_relocatable_gate(execution_gate)
    training_digest = canonical_json_sha256(training_contract)
    execution_digest = canonical_json_sha256(execution_contract)
    if training_digest != execution_digest:
        raise RuntimeError(
            "Training-lineage and execution-site gates differ beyond absolute relocation paths: "
            f"{training_digest} != {execution_digest}"
        )

    # This explicit count also makes accidental conversion from a multiset to
    # a set fail closed during future refactors.
    if Counter(training_contract["guarded_sha256"]) != Counter(
        execution_contract["guarded_sha256"]
    ):
        raise RuntimeError("Relocated guarded SHA256 multiset differs from the training gate")
    return training_gate, file_sha256(path), training_digest
