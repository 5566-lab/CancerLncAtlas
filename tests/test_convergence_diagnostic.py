from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cc_hhgt.convergence_diagnostic import audit_convergence_task


def _write_task(root: Path, *, hard_cap: bool) -> None:
    root.mkdir(parents=True)
    epochs = [1, 35, 70, 105]
    pd.DataFrame(
        {
            "epoch": epochs,
            "validation_full_coverage_cycle": [True] * 4,
            "runtime_num_chunks": [35] * 4,
            "pathway_checkpoint_score": [0.20, 0.25, 0.25, 0.25],
            "state_checkpoint_score": [0.20, 0.26, 0.26, 0.26],
        }
    ).to_csv(root / "training_history.tsv", sep="\t", index=False)
    (root / "best_pathway.pt").write_bytes(b"path")
    (root / "best_state.pt").write_bytes(b"state")
    (root / "SUCCESS.json").write_text(
        json.dumps(
            {
                "model_name": "cc_hhgt",
                "fold_id": "LOCO_ACC",
                "seed": 20260726,
                "configured_epoch_cap": 105 if hard_cap else 1000,
                "epochs_completed": 105,
                "hit_hard_epoch_cap": hard_cap,
                "stopped_by_joint_patience": not hard_cap,
                "configured_runtime_checkpoint_patience_cycles": 3,
                "pathway_stale_validation_cycles": 3,
                "state_stale_validation_cycles": 3,
                "checkpoint_min_delta": 1e-4,
                "best_pathway_epoch": 35,
                "best_state_epoch": 35,
            }
        ),
        encoding="utf-8",
    )


def test_natural_joint_patience_is_the_only_pass(tmp_path: Path) -> None:
    root = tmp_path / "task"
    _write_task(root, hard_cap=False)
    result = audit_convergence_task(
        root,
        expected_model="cc_hhgt",
        expected_fold="LOCO_ACC",
        expected_seed=20260726,
        expected_cap=1000,
    )
    assert result["status"] == "PASS_CONVERGED"
    assert result["release_eligible"] is False


def test_hard_cap_never_passes(tmp_path: Path) -> None:
    root = tmp_path / "task"
    _write_task(root, hard_cap=True)
    result = audit_convergence_task(
        root,
        expected_model="cc_hhgt",
        expected_fold="LOCO_ACC",
        expected_seed=20260726,
        expected_cap=105,
    )
    assert result["status"].startswith("FAIL_HARD_CAP")
