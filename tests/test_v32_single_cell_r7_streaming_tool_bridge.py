from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.materialize_v32_single_cell_r7_streaming_tool_bridge import (
    EXPECTED,
    TARGET_ROOT,
    StreamingToolBridgeError,
    materialize,
)
from scripts.transfer_v32_payloads_to_local_gpu import (
    build_plan,
    load_copy_manifest,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "single_cell_r7_streaming_tool_bridge_20260829_r5"


def test_streaming_tool_manifest_is_production_loadable_and_exact() -> None:
    manifest, manifest_sha = load_copy_manifest(ARTIFACT / "COPY_MANIFEST.json")
    plan = build_plan(manifest, manifest_sha256=manifest_sha)
    assert plan["target_root"] == TARGET_ROOT
    assert plan["entry_count"] == 6
    assert plan["total_bytes"] == 97_875
    assert manifest["overwrite_permitted"] is False
    assert manifest["symlinks_permitted"] is False
    for row in manifest["entries"]:
        relative = row["target_path"].removeprefix(TARGET_ROOT + "/")
        expected_bytes, expected_sha = EXPECTED[relative]
        source = Path(row["source_path"])
        assert source.stat().st_size == row["bytes"] == expected_bytes
        assert sha256_file(source) == row["sha256"] == expected_sha


def test_streaming_tool_validation_binds_safety_contract() -> None:
    validation = json.loads((ARTIFACT / "VALIDATION.json").read_text(encoding="utf-8"))
    assert validation["no_cell_level_pathway_matrix"] is True
    assert validation["donor_is_biological_replicate"] is True
    assert validation["per_cancer_atomic_publish"] is True
    assert validation["resume_checkpoint_is_sufficient_statistics_only"] is True
    assert validation["historical_r5_r6_derived_inputs_permitted"] is False
    assert validation["supersedes_copy_manifest_sha256"] == (
        "ad173a711effadaf8313ebcf101b2914c7328d00e44b2e3574ed28502f2aa991"
    )
    assert "r4 HNSC pilot succeeded" in validation["supersession_reason"]
    assert validation["raw_h5_full_sha_gate_before_matrix_access"] is True
    assert validation["resource_estimate_includes_runtime_baseline"] is True
    with pytest.raises(StreamingToolBridgeError, match="reuse is forbidden"):
        materialize(ARTIFACT)
