from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_metacell_materialization_149.py"
)
SPEC = importlib.util.spec_from_file_location("metacell_materialization_validator", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validator)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_authority(tmp_path: Path, count: int = 28):
    cancers = [f"C{index:02d}" for index in range(count)]
    active = tmp_path / "RUN_STATUS.json"
    write_json(
        active,
        {"per_cancer": {cancer: {"formal_eligible": True} for cancer in cancers}},
    )
    repair = tmp_path / "repair"
    for cancer in cancers:
        job = repair / "jobs" / cancer
        job.mkdir(parents=True)
        for name in validator.REQUIRED_JOB_FILES:
            (job / name).write_bytes(b"job")
    return cancers, active, repair


def test_full_authority_is_exact_28_intersection(tmp_path):
    cancers, active, repair = make_authority(tmp_path)
    expected, evidence = validator.authorized_cancers(
        active, repair, require_full=True
    )
    assert expected == cancers
    assert evidence["authorized_cancer_count"] == 28

    (repair / "jobs" / cancers[-1] / "TRAINING_HANDOFF.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        validator.authorized_cancers(active, repair, require_full=True)


def test_full_validation_roots_reject_missing_and_extra(tmp_path):
    expected = ["A", "B"]
    cancers_root = tmp_path / "cancers"
    for cancer in expected:
        write_json(cancers_root / cancer / "SUCCESS.json", {})
    roots = validator.validation_roots(tmp_path, expected, None)
    assert [path.name for path in roots] == expected

    (cancers_root / "EXTRA").mkdir()
    with pytest.raises(RuntimeError, match="extra"):
        validator.validation_roots(tmp_path, expected, None)

    assert validator.validation_roots(tmp_path, expected, "A") == [cancers_root / "A"]


def test_preflight_rehashes_all_bound_inputs(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    binding = tmp_path / "PREPROCESSING_BINDING.json"
    candidate = tmp_path / "FORMAL_CANDIDATE_UNIVERSE.parquet"
    core_root = tmp_path / "core"
    core_manifest = core_root / "CORE_EMBEDDING_MANIFEST.json"
    binding.write_bytes(b"binding")
    candidate.write_bytes(b"candidate")
    core_manifest.parent.mkdir()
    core_manifest.write_bytes(b"core")
    write_json(
        output / "LAUNCH_PREFLIGHT.json",
        {
            "format": validator.PREFLIGHT_FORMAT,
            "run_id": "test-run",
            "target_host": "149",
            "cpu_only": True,
            "paid_compute": False,
            "output_root": str(output),
            "preprocessing_binding_path": str(binding),
            "preprocessing_binding_sha256": validator.sha256(binding),
            "candidate_path": str(candidate),
            "candidate_sha256": validator.sha256(candidate),
            "core_manifest_path": str(core_manifest),
            "core_manifest_sha256": validator.sha256(core_manifest),
            "old_result_overwritten": False,
            "old_result_backed_up": False,
            "formal_pointer_update_authorized": False,
        },
    )
    evidence = validator.validate_launch_preflight(output, core_root, candidate)
    assert evidence["bindings"]["candidate_universe"]["sha256"] == validator.sha256(
        candidate
    )

    candidate.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="candidate_universe hash drift"):
        validator.validate_launch_preflight(output, core_root, candidate)


def test_receipt_record_binds_path_size_and_hash(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"content")
    record = {
        "path": str(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": validator.sha256(artifact),
    }
    assert validator.verify_file_record(
        record, artifact, label="artifact"
    )["sha256"] == record["sha256"]

    wrong = dict(record, bytes=record["bytes"] + 1)
    with pytest.raises(RuntimeError, match="byte count drift"):
        validator.verify_file_record(wrong, artifact, label="artifact")
