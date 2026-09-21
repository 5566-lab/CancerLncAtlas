from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "finalize_v32_single_cell_r11_formal23",
    ROOT / "scripts/finalize_v32_single_cell_r11_formal23.py",
)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_status = tmp_path / "RUN_STATUS.json"
    binding = tmp_path / "COHORT_BINDING.json"
    audit = tmp_path / "AUDIT.json"
    write_json(
        run_status,
        {
            "format": "CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_RUN_STATUS_V1",
            "status": "PASS_INPUT_CONTRACT_READY",
            "formal_eligible_cancers": sorted(MODULE.FORMAL23),
            "typed_unavailable": {key: "TYPED_REASON" for key in MODULE.TYPED10},
        },
    )
    write_json(
        binding,
        {
            "format": "CC_HHGT_V3_2_SINGLE_CELL_R11_FORMAL23_BINDING_V1",
            "status": "BOUND_23_OF_23_PENDING_INDEPENDENT_AUDIT",
            "formal_bound_cancer_count": 23,
            "generated_in_this_run_count": 10,
            "external_immutable_binding_count": 13,
            "formal_cancer_universe": sorted(MODULE.FORMAL23),
        },
    )
    write_json(
        audit,
        {
            "format": "CC_HHGT_V3_2_SINGLE_CELL_R11_FORMAL23_INDEPENDENT_AUDIT_V1",
            "status": "PASS_23_OF_23_INDEPENDENTLY_VERIFIED",
            "formal_cancer_count": 23,
            "generated_in_current_r11_run_count": 10,
            "immutable_upstream_binding_count": 13,
            "formal_cancer_universe": sorted(MODULE.FORMAL23),
            "cohort_success_sha256": sha256(binding),
            "total_cells": 123,
            "total_association_evidence_rows": 456,
        },
    )
    return run_status, binding, audit


def test_finalizer_requires_the_independent_audit_to_bind_exact_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_status, binding, audit = fixture_files(tmp_path)
    output = tmp_path / "SUCCESS.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalizer",
            "--run-status", str(run_status),
            "--binding", str(binding),
            "--audit", str(audit),
            "--output", str(output),
        ],
    )
    assert MODULE.main() == 0
    success = json.loads(output.read_text(encoding="utf-8"))
    assert success["status"] == "SUCCESS"
    assert success["formal_eligible_cancer_count"] == 23
    assert success["typed_unavailable_cancer_count"] == 10
    assert success["website_bound"] is False

    broken = json.loads(audit.read_text(encoding="utf-8"))
    broken["cohort_success_sha256"] = "0" * 64
    write_json(audit, broken)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalizer",
            "--run-status", str(run_status),
            "--binding", str(binding),
            "--audit", str(audit),
            "--output", str(tmp_path / "BROKEN_SUCCESS.json"),
        ],
    )
    with pytest.raises(RuntimeError, match="does not attest"):
        MODULE.main()
