from __future__ import annotations

import json

import pandas as pd
import pytest

from cc_hhgt.common import file_sha256
from cc_hhgt.v31_residual_gate import validate_residual_stage_gate


def test_residual_stage_gate_rejects_guarded_file_drift(tmp_path) -> None:
    guarded = tmp_path / "guarded.txt"
    guarded.write_text("frozen", encoding="utf-8")
    manifest = tmp_path / "code.tsv"
    pd.DataFrame(
        [
            {
                "relative_path": "guarded.txt",
                "size_bytes": guarded.stat().st_size,
                "sha256": file_sha256(guarded),
            }
        ]
    ).to_csv(manifest, sep="\t", index=False)
    output_root = tmp_path / "output"
    gate = tmp_path / "gate.json"
    payload = {
        "status": "PASS",
        "failures": [],
        "analysis_version": "residual-test",
        "pilot_cancers": ["BRCA", "COAD", "KIRP"],
        "model_seeds": [20260726, 20261726, 20262726],
        "shrinkage_grid": [0.01],
        "expected_b2_tasks": 9,
        "full_cancer_training_authorized": False,
        "paths": {"output_root": str(output_root)},
        "manifest_contracts": [
            {
                "root": str(tmp_path),
                "manifest": str(manifest),
                "manifest_sha256": file_sha256(manifest),
            }
        ],
        "guarded_sha256": {str(guarded): file_sha256(guarded)},
    }
    gate.write_text(json.dumps(payload), encoding="utf-8")
    validate_residual_stage_gate(
        gate,
        {"analysis_version": "residual-test", "residual_learning": {"shrinkage_grid": [0.01]}},
        output_root=output_root,
    )
    guarded.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="assets changed"):
        validate_residual_stage_gate(
            gate,
            {"analysis_version": "residual-test", "residual_learning": {"shrinkage_grid": [0.01]}},
            output_root=output_root,
        )
