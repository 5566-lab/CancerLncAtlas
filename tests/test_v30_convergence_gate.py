from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_audit_script():
    spec = importlib.util.spec_from_file_location(
        "audit_v30_convergence", ROOT / "scripts" / "72_audit_v30_state_matrix.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hard_epoch_cap_is_release_failure() -> None:
    module = load_audit_script()
    assert module.hard_epoch_cap_violation(
        {"epochs_completed": 500, "configured_epoch_cap": 500, "hit_hard_epoch_cap": True},
        500,
    )


def test_patience_stop_before_cap_passes_convergence_gate() -> None:
    module = load_audit_script()
    assert not module.hard_epoch_cap_violation(
        {
            "epochs_completed": 337,
            "configured_epoch_cap": 500,
            "hit_hard_epoch_cap": False,
            "stopped_by_joint_patience": True,
        },
        500,
    )


def test_non_cap_stop_without_joint_patience_is_still_invalid() -> None:
    module = load_audit_script()
    assert module.hard_epoch_cap_violation(
        {
            "epochs_completed": 337,
            "configured_epoch_cap": 500,
            "hit_hard_epoch_cap": False,
            "stopped_by_joint_patience": False,
        },
        500,
    )


def test_task_cannot_claim_a_different_epoch_cap() -> None:
    module = load_audit_script()
    assert module.hard_epoch_cap_violation(
        {"epochs_completed": 250, "configured_epoch_cap": 250, "hit_hard_epoch_cap": True},
        500,
    )


def test_patience_stop_before_cap2000_passes() -> None:
    module = load_audit_script()
    assert not module.hard_epoch_cap_violation(
        {
            "epochs_completed": 1015,
            "configured_epoch_cap": 2000,
            "hit_hard_epoch_cap": False,
            "stopped_by_joint_patience": True,
        },
        2000,
    )


def test_cap2000_is_failure_even_if_patience_flag_is_set() -> None:
    module = load_audit_script()
    assert module.hard_epoch_cap_violation(
        {
            "epochs_completed": 2000,
            "configured_epoch_cap": 2000,
            "hit_hard_epoch_cap": True,
            "stopped_by_joint_patience": True,
        },
        2000,
    )
