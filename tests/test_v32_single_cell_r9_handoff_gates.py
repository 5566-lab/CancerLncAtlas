from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_v32_single_cell_r7_streaming.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("r9_handoff_gate_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_handoff(tmp_path: Path, runner) -> dict:
    cancer = "ACC"
    contract = "a" * 64
    producer_pid = 12345
    publish = tmp_path / f".cancer_id={cancer}.{contract[:16]}.publish.{producer_pid}"
    final = tmp_path / f"cancer_id={cancer}"
    publish.mkdir()
    for relative in runner.POST_BH_RELATIVE_PATHS:
        if relative != "association_evidence.parquet":
            (publish / relative).write_bytes(b"fixture")
    (publish / ".association_evidence_raw.parquet").write_bytes(b"fixture")
    return {
        "format": runner.POST_BH_HANDOFF_FORMAT,
        "association_engine": runner.ASSOCIATION_ENGINE,
        "cancer_id": cancer,
        "producer_pid": producer_pid,
        "publish": str(publish),
        "final": str(final),
        "raw_path": str(publish / ".association_evidence_raw.parquet"),
        "evidence_path": str(publish / "association_evidence.parquet"),
        "scratch": str(publish / ".association_duckdb_scratch"),
        "raw_rows": 1,
        "relative_paths": list(runner.POST_BH_RELATIVE_PATHS),
        "lineage_base": {"cancer_id": cancer, "contract_sha256": contract},
        "success_base": {"cancer_id": cancer},
        "_verified_handoff_sha256": "b" * 64,
    }


@pytest.fixture()
def runner(monkeypatch: pytest.MonkeyPatch):
    value = load_runner()
    monkeypatch.setattr(value, "_authorized_output", lambda _path: None)
    return value


def test_rejects_publish_final_different_parents(tmp_path: Path, runner) -> None:
    handoff = valid_handoff(tmp_path, runner)
    other = tmp_path / "other"
    other.mkdir()
    handoff["final"] = str(other / "cancer_id=ACC")
    with pytest.raises(runner.R7StreamingRunError, match="topology or name drift"):
        runner._post_bh_exec_publish(handoff)


def test_rejects_raw_path_outside_fixed_publish_basename(tmp_path: Path, runner) -> None:
    handoff = valid_handoff(tmp_path, runner)
    handoff["raw_path"] = str(tmp_path / "other_raw.parquet")
    with pytest.raises(runner.R7StreamingRunError, match="fixed path drift"):
        runner._post_bh_exec_publish(handoff)


def test_rejects_cancer_mismatch_across_payloads(tmp_path: Path, runner) -> None:
    handoff = valid_handoff(tmp_path, runner)
    handoff["success_base"]["cancer_id"] = "GBM"
    with pytest.raises(runner.R7StreamingRunError, match="cancer ID differs"):
        runner._post_bh_exec_publish(handoff)


@pytest.mark.parametrize(
    "invalid_paths",
    [
        ["association_evidence.parquet"],
        [
            "lncrna_donor_celltype_summary.parquet",
            "pathway_availability.parquet",
            "pathway_donor_celltype_summary.parquet",
            "association_evidence.parquet",
            "association_lncrna_testability.parquet",
            "association_context_availability.parquet",
            "../RESOURCE_ESTIMATE.json",
        ],
    ],
)
def test_rejects_nonexact_release_file_set(
    tmp_path: Path, runner, invalid_paths: list[str]
) -> None:
    handoff = valid_handoff(tmp_path, runner)
    handoff["relative_paths"] = invalid_paths
    with pytest.raises(runner.R7StreamingRunError, match="file set/order drift"):
        runner._post_bh_exec_publish(handoff)


def test_rejects_symlinked_staged_release_leaf(tmp_path: Path, runner) -> None:
    handoff = valid_handoff(tmp_path, runner)
    publish = Path(handoff["publish"])
    leaf = publish / "RESOURCE_ESTIMATE.json"
    leaf.unlink()
    target = publish / "resource_target.json"
    target.write_bytes(b"fixture")
    try:
        os.symlink(target, leaf)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    with pytest.raises(runner.R7StreamingRunError, match="absent/unsafe"):
        runner._post_bh_exec_publish(handoff)
