from __future__ import annotations

import json
from pathlib import Path

import torch

from cc_hhgt.v32.single_cell_independent_audit import (
    AUDIT_FORMAT,
    BINDING_FORMAT,
    deterministic_block_folds,
    file_sha256,
    independently_rebuild_initial_state_sha256,
    split_blocks,
    state_dict_sha256,
    verify_report_binding,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "artifacts" / "v32_single_cell_fresh_20260826_r1_independent_audit"


def test_state_hash_is_content_sensitive_and_seed_rebuild_is_deterministic() -> None:
    state = {"weight": torch.tensor([[1.0, 2.0]]), "bias": torch.tensor([0.5])}
    baseline = state_dict_sha256(state)
    changed = {name: value.clone() for name, value in state.items()}
    changed["weight"][0, 0] += 1.0
    assert state_dict_sha256(changed) != baseline
    first = independently_rebuild_initial_state_sha256(
        seed=31, core_features=8, domain_features=3, hidden_features=5, dropout=0.1
    )
    second = independently_rebuild_initial_state_sha256(
        seed=31, core_features=8, domain_features=3, hidden_features=5, dropout=0.1
    )
    assert first == second


def test_dataset_block_folds_are_deterministic_disjoint_and_complete() -> None:
    blocks = [f"C{i}|dataset:D{i}" for i in range(17)]
    first = deterministic_block_folds(blocks, 20260825)
    assert first == deterministic_block_folds(reversed(blocks), 20260825)
    assert set(first.values()) == set(range(5))
    for fold in range(5):
        split = split_blocks(first, fold)
        assert not (split["train"] & split["validation"])
        assert not (split["train"] & split["test"])
        assert not (split["validation"] & split["test"])
        assert set().union(*split.values()) == set(blocks)


def test_checked_in_formal_independent_audit_is_pass_and_hash_bound() -> None:
    report_path = AUDIT_ROOT / "INDEPENDENT_AUDIT.json"
    binding_path = AUDIT_ROOT / "INDEPENDENT_AUDIT_BINDING.json"
    success_path = AUDIT_ROOT / "SUCCESS.json"
    assert report_path.is_file() and binding_path.is_file() and success_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    success = json.loads(success_path.read_text(encoding="utf-8"))
    assert report["format"] == AUDIT_FORMAT
    assert binding["format"] == BINDING_FORMAT
    assert report["status"] == binding["status"] == success["status"] == "PASS"
    assert report["failed_checks"] == []
    assert all(row["status"] == "PASS" for row in report["checks"])
    assert verify_report_binding(report_path, binding)
    assert success["audit_report_sha256"] == file_sha256(report_path)
    assert success["binding_sha256"] == file_sha256(binding_path)
    assert report["summary"]["candidate_rows"] == 3_300_000
    assert report["summary"]["typed_rows"] == 7_814_014
    assert report["summary"]["available_rows"] == 6_214_014
    assert report["summary"]["null_rows"] == 1_600_000
    assert len(report["summary"]["formal_cancers"]) == 17
    assert len(report["summary"]["typed_unavailable_cancers"]) == 16
    assert len(report["checkpoint_audit"]) == 5
    assert all(
        row["initial_parameter_sha256"]
        == row["independently_rebuilt_initial_parameter_sha256"]
        and row["initial_parameter_sha256"] != row["final_parameter_sha256"]
        for row in report["checkpoint_audit"]
    )
    assert report["core_audit"]["before_sha256"] == report["core_audit"]["after_sha256"]
    assert report["aggregation_adapter_audit"]["typed_to_exact_aggregate_difference_rows"] == 0
    assert report["aggregation_adapter_audit"]["adapter_stats"]["rows"] == 3_300_000
    assert report["release_decision"]["full_universe_fusion_adapter_materialization"] == "AUTHORIZED"


def test_binding_verifier_fails_closed_on_tampered_hash() -> None:
    report_path = AUDIT_ROOT / "INDEPENDENT_AUDIT.json"
    binding = json.loads(
        (AUDIT_ROOT / "INDEPENDENT_AUDIT_BINDING.json").read_text(encoding="utf-8")
    )
    binding["audit_report_sha256"] = "0" * 64
    assert not verify_report_binding(report_path, binding)
