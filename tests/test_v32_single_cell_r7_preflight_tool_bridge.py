from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from scripts.materialize_v32_single_cell_r7_preflight_tool_bridge import (
    EXPECTED,
    TARGET_ROOT,
    ToolBridgeError,
    materialize_tool_bridge,
)
from scripts.transfer_v32_payloads_to_local_gpu import (
    build_plan,
    load_copy_manifest,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "artifacts" / "single_cell_r7_preflight_tool_bridge_20260829_r3"
)


def test_production_loader_and_plan_accept_two_tool_leaves() -> None:
    manifest_path = ARTIFACT / "COPY_MANIFEST.json"
    manifest, manifest_sha = load_copy_manifest(manifest_path)
    plan = build_plan(manifest, manifest_sha256=manifest_sha)
    assert plan["target_root"] == TARGET_ROOT
    assert plan["entry_count"] == 2
    assert plan["total_bytes"] == 33_440
    assert manifest["overwrite_permitted"] is False
    assert manifest["symlinks_permitted"] is False
    assert manifest["payloads_are_byte_identical"] is True
    assert {Path(row["target_path"]).name for row in manifest["entries"]} == {
        "preflight_v32_single_cell_r7_portable_server.py",
        "COPY_MANIFEST.json",
    }
    for row in manifest["entries"]:
        source = Path(row["source_path"])
        expected = EXPECTED[Path(row["target_path"]).name]
        assert source.is_file() and not source.is_symlink()
        assert source.stat().st_size == row["bytes"] == expected["bytes"]
        assert sha256_file(source) == row["sha256"] == expected["sha256"]


def test_tool_bridge_is_non_overwriting_and_validation_is_bound() -> None:
    validation = json.loads(
        (ARTIFACT / "TOOL_BRIDGE_VALIDATION.json").read_text(encoding="utf-8")
    )
    assert validation["production_loader"] == "load_copy_manifest"
    assert validation["production_planner"] == "build_plan"
    assert validation["entry_count"] == 2
    assert validation["standalone_preflight"] is True
    assert validation["remote_r7_bridge_manifest_included"] is True
    assert validation["supersedes_tool_bridge_copy_manifest_sha256"] == (
        "212923c8959f9b06bb2fe143011b90bbdf8af261892eb5ff07e6b2eaf9d7a701"
    )
    assert validation["upload_started"] is False
    with pytest.raises(ToolBridgeError, match="reuse/overwrite is forbidden"):
        materialize_tool_bridge(ARTIFACT)


def test_preflight_has_no_project_runtime_imports_and_expected_dependencies() -> None:
    path = ROOT / "scripts" / "preflight_v32_single_cell_r7_portable_server.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert not ({"cc_hhgt", "scripts"} & roots)
    assert {"numpy", "pandas"}.issubset(roots)
    # h5py and pyarrow are intentionally lazy server dependencies.
    source = path.read_text(encoding="utf-8")
    assert "import h5py" in source
    assert "import pyarrow.parquet as pq" in source
