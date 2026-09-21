from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from scripts.transfer_v32_payloads_to_local_gpu import (
    build_plan,
    load_copy_manifest,
)


def _load_supervisor():
    path = Path(__file__).resolve().parents[1] / "scripts" / (
        "run_v32_single_cell_r7_17c_cohort.py"
    )
    spec = importlib.util.spec_from_file_location("r7_17c_cohort_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cohort_scope_and_memory_gate_are_frozen() -> None:
    supervisor = _load_supervisor()
    assert len(supervisor.FORMAL_CANCERS) == 17
    assert supervisor.MEMORY_LIMIT_BYTES == 512 * 1024**2
    assert supervisor.TOOL_MANIFEST_SHA256 == (
        "06090c75a4c3a722c5be99c3ba6f3ae205c0905dbf023a68201c80d162fbf8d1"
    )
    assert len(supervisor.TOOL_SHAS) == 6


def test_runner_arguments_pin_donor_aware_fresh_contract() -> None:
    supervisor = _load_supervisor()
    plan = Path("./data/CancerLncAtlas/runtime/audits/test.json")
    argv = supervisor.runner_args(
        mode="plan",
        cancer="HNSC",
        plan_path=plan,
        output_root=None,
    )
    joined = " ".join(argv)
    assert "--cancer-id HNSC" in joined
    assert "--min-donors 5" in joined
    assert "--min-cells-per-donor-context 20" in joined
    assert "--chunk-cells 64" in joined
    assert "--plan-output-json" in argv
    assert "--resume" not in argv


def test_cohort_contract_hash_is_order_invariant_for_mapping_keys() -> None:
    supervisor = _load_supervisor()
    assert supervisor.canonical_sha256({"b": 2, "a": 1}) == (
        supervisor.canonical_sha256({"a": 1, "b": 2})
    )


def test_supervisor_transfer_bridge_is_single_file_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = root / "artifacts" / (
        "single_cell_r7_17c_supervisor_bridge_20260829_r1"
    )
    manifest, manifest_sha = load_copy_manifest(artifact / "COPY_MANIFEST.json")
    plan = build_plan(manifest, manifest_sha256=manifest_sha)
    validation = json.loads(
        (artifact / "VALIDATION.json").read_text(encoding="utf-8")
    )
    assert plan["entry_count"] == 1
    assert plan["total_bytes"] == 29_487
    assert validation["memory_limit_bytes"] == 512 * 1024**2
    assert validation["raw_h5_full_sha_first"] is True
    assert validation["typed_failure_stops_cohort"] is True
    assert validation["port_8260_touched"] is False
