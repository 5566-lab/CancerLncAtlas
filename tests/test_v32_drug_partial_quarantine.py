from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "quarantine_v32_drug_partial_uploads_server.py"
SPEC = importlib.util.spec_from_file_location("drug_partial_quarantine", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(MODULE, "EXPECTED_AUTHORIZED_FILES", 2)
    monkeypatch.setattr(MODULE, "EXPECTED_PARTIAL_FILES", 2)
    monkeypatch.setattr(MODULE, "EXPECTED_AVAILABLE_PARTIAL_FILES", 1)
    monkeypatch.setattr(MODULE, "EXPECTED_FACTOR_PARTIAL_FILES", 1)
    root = tmp_path / "artifact"
    available = root / "drug_response_association" / "cancer-X" / "part-00000.parquet"
    factor = root / "drug_sparse_query_factors" / "factor.parquet"
    available.parent.mkdir(parents=True)
    factor.parent.mkdir(parents=True)
    payloads = [(available, b"available"), (factor, b"factor")]
    records = []
    for path, data in payloads:
        path.write_bytes(data)
        digest = _sha(data)
        path.with_name(path.name + ".partial." + digest).write_bytes(data)
        records.append({"path": str(path), "sha256": digest, "bytes": len(data)})
    binding_path = tmp_path / "binding.json"
    binding_sha = _write_json(
        binding_path,
        {
            "format": MODULE.BINDING_FORMAT,
            "analysis_version": MODULE.ANALYSIS_VERSION,
            "components": {"drug": {"payloads": records}},
        },
    )
    core_path = tmp_path / "core.json"
    core_sha = _write_json(
        core_path,
        {
            "status": "PASS",
            "artifact_root_authority": {
                "path": str(root),
                "payload_files_hash_and_bytes_verified": 2,
            },
        },
    )
    return root, payloads, binding_path, binding_sha, core_path, core_sha


def test_plan_then_apply_preserves_canonical_and_moves_only_exact_partials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, payloads, binding, binding_sha, core, core_sha = _fixture(tmp_path, monkeypatch)
    quarantine = tmp_path / "quarantine"
    plan = MODULE.quarantine(
        artifact_root=root,
        binding_path=binding,
        binding_sha256=binding_sha,
        core_receipt_path=core,
        core_receipt_sha256=core_sha,
        quarantine_root=quarantine,
        apply=False,
        approved_plan=None,
    )
    assert plan["status"] == "PLAN_PASS"
    assert plan["pre_operation"]["physical_files"] == 4
    assert plan["pre_operation"]["unexpected_files"] == 2
    assert plan["pre_operation"]["unexpected_files_other_than_typed_upload_partials"] == 0
    assert len(plan["pre_operation"]["physical_inventory_sha256"]) == 64

    receipt = MODULE.quarantine(
        artifact_root=root,
        binding_path=binding,
        binding_sha256=binding_sha,
        core_receipt_path=core,
        core_receipt_sha256=core_sha,
        quarantine_root=quarantine,
        apply=True,
        approved_plan=plan,
    )
    assert receipt["status"] == "PASS"
    assert receipt["post_operation"]["artifact_files"] == 2
    assert receipt["post_operation"]["unexpected_files"] == 0
    assert receipt["post_operation"]["quarantine_files"] == 2
    for path, data in payloads:
        assert path.read_bytes() == data
        assert not list(path.parent.glob(path.name + ".partial.*"))
    assert len(list(quarantine.rglob("*.partial.*"))) == 2


def test_content_mismatch_fails_before_any_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, payloads, binding, binding_sha, core, core_sha = _fixture(tmp_path, monkeypatch)
    canonical, _ = payloads[0]
    partial = next(canonical.parent.glob(canonical.name + ".partial.*"))
    partial.write_bytes(b"mismatch")
    quarantine = tmp_path / "quarantine"
    with pytest.raises(MODULE.PartialQuarantineError, match="Partial bytes differ"):
        MODULE.quarantine(
            artifact_root=root,
            binding_path=binding,
            binding_sha256=binding_sha,
            core_receipt_path=core,
            core_receipt_sha256=core_sha,
            quarantine_root=quarantine,
            apply=True,
            approved_plan={},
        )
    assert not quarantine.exists()
    assert partial.exists()
