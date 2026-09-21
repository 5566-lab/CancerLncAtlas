from __future__ import annotations

from pathlib import Path

from cc_hhgt.v31_pilot_gate import (
    DIAGNOSTIC_CONTRACT,
    PILOT_CANCERS,
    PILOT_SEEDS,
    PRIMARY_CONTRACT,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_graph_scope_is_exact_and_has_no_full_mode() -> None:
    source = (ROOT / "scripts/79_run_v31_fixed_graph_pilot.py").read_text(
        encoding="utf-8"
    )
    assert PILOT_CANCERS == ("BRCA", "COAD", "KIRP")
    assert PILOT_SEEDS == (20260726, 20261726, 20262726)
    assert PRIMARY_CONTRACT == "CONTRACT-T"
    assert DIAGNOSTIC_CONTRACT == "CONTRACT-S"
    assert 'choices=["smoke", "b1"]' in source
    assert "full" not in source.split("choices=", 1)[1].split(")", 1)[0]
    assert 'if len(matrix) != 12:' in source
    assert '"full_cancer_training_started": False' in source


def test_runner_reuses_schedule_but_fresh_initializes_every_seed() -> None:
    source = (ROOT / "scripts/79_run_v31_fixed_graph_pilot.py").read_text(
        encoding="utf-8"
    )
    assert "bundle = load_graph_bundle(" in source
    assert "for seed in seeds:" in source
    assert "train_multitask_fold(" in source
    assert "_bundle=bundle" in source
    assert '"fresh_initialization": True' in source
    assert source.count("validate_v31_pilot_gate(") >= 3


def test_runtime_early_stopping_is_counted_in_full_validation_cycles() -> None:
    source = (ROOT / "cc_hhgt/v29_multitask.py").read_text(encoding="utf-8")
    config = (ROOT / "config/model_v3_1_fixed_graph_pilot.yaml").read_text(
        encoding="utf-8"
    )
    assert "checkpoint_patience_cycles: 3" in config
    assert "pathway_stale_validation_cycles >= checkpoint_patience_cycles" in source
    assert "state_stale_validation_cycles >= checkpoint_patience_cycles" in source
    assert '"runtime_validation_cycles_completed": runtime_validation_cycles' in source


def test_gate_replaces_old_code_contract_but_keeps_input_and_assets() -> None:
    source = (
        ROOT / "scripts/78_create_v31_fixed_graph_pilot_gate.py"
    ).read_text(encoding="utf-8")
    assert "frozen_contracts = [source_contracts[0], source_contracts[2]]" in source
    assert '"full_cancer_training_authorized": False' in source
    assert '"expected_b1_tasks": 12' in source
    assert "merge-base" in source and "--is-ancestor" in source
