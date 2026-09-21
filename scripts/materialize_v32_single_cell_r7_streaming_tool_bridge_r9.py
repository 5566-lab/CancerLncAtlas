#!/usr/bin/env python3
"""Freeze the complete r9 single-cell code and immutable validation receipts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts.transfer_v32_payloads_to_local_gpu import (  # noqa: E402
    COPY_FORMAT,
    _canonical_json,
    build_plan,
    load_copy_manifest,
    sha256_file,
)


TARGET_ROOT = (
    "./data/CancerLncAtlas/runtime/tools/"
    "single_cell_r7_streaming_20260829_r9"
)
SUPERSEDED_MANIFEST_SHA256 = (
    "89360f86e1c96f36c69fcf347ed752ba657bb388ebfe62fdd0500a5a852ab55f"
)
EXTERNAL_TESTED_RUNNER = Path(
    "D:/model/r9_candidate_r1_equivalence_runner.audit-copy.py"
)

# source key -> (target relative path, bytes, SHA-256)
EXPECTED: dict[str, tuple[str, int, str]] = {
    "cc_hhgt/__init__.py": (
        "cc_hhgt/__init__.py",
        98,
        "e246acd89da9a9733fbac09b2770161b5ed199695a93a832ca965d8d201b78dc",
    ),
    "cc_hhgt/v32/__init__.py": (
        "cc_hhgt/v32/__init__.py",
        363,
        "359f958ad6e7e67078201282718193b40bc0d09bf15d27cbefb3a604b535ca91",
    ),
    "cc_hhgt/v32/contracts.py": (
        "cc_hhgt/v32/contracts.py",
        8_363,
        "387e64d4931bac936e7affa4a3f974812dbfbb16d50deacd670685c796d41f46",
    ),
    "cc_hhgt/v32/single_cell_cell_level.py": (
        "cc_hhgt/v32/single_cell_cell_level.py",
        12_413,
        "08e5b3519539fe488c78626aab7a7ef94bad4f4ef7f94b2f656f029bd5056815",
    ),
    "cc_hhgt/v32/single_cell_r7_streaming.py": (
        "cc_hhgt/v32/single_cell_r7_streaming.py",
        31_679,
        "7133182f386ca74ea8b36013f16ef7679cb95306949cff8eb91ffee24ee36fd3",
    ),
    "scripts/run_v32_single_cell_r7_streaming.py": (
        "scripts/run_v32_single_cell_r7_streaming.py",
        90_242,
        "903658231bfe31387d12cab56b9de45ec88258e386f9be0e3ba442b61495932c",
    ),
    "scripts/audit_v32_single_cell_r9_bh_equivalence.py": (
        "scripts/audit_v32_single_cell_r9_bh_equivalence.py",
        22_054,
        "99c4ece2b7aee3f3fba2a13f72152af6f7638462dc334a1d1ed307e6e687a396",
    ),
    "scripts/audit_v32_single_cell_r9_formal17.py": (
        "scripts/audit_v32_single_cell_r9_formal17.py",
        21_499,
        "6bacccd8ee63d715385b89df9488487f9643a0b80847b6f00458317235cb1cc9",
    ),
    "scripts/bind_v32_single_cell_r9_final_runner_equivalence.py": (
        "scripts/bind_v32_single_cell_r9_final_runner_equivalence.py",
        7_376,
        "95dfa1eafc24608b8f3a49875086f5a5583cc529f4de363511aae9c91f3311f3",
    ),
    "scripts/verify_v32_single_cell_r9_large_equivalence.py": (
        "scripts/verify_v32_single_cell_r9_large_equivalence.py",
        7_869,
        "fa4ffd245dea5bcd0c0ecad95d1ce2af8e261f0b1572186bee62a0a47b88eb89",
    ),
    "scripts/audit_v32_single_cell_r9_runtime_binding.py": (
        "scripts/audit_v32_single_cell_r9_runtime_binding.py",
        9_809,
        "93bb2dc5e60ed5e619a9c53ef841b4e8e2b05c800318d843814a32e0edc4baf2",
    ),
    "tests/test_v32_single_cell_r9_handoff_gates.py": (
        "tests/test_v32_single_cell_r9_handoff_gates.py",
        4_269,
        "e5e7aac950a1e222f33fb0e5540b0b70e774e0d3485af1390d51c368ba02d486",
    ),
    (
        "artifacts/single_cell_r9_acc_bh_process_isolation_audit_20260829_r1/"
        "ACC_R8_MEMORY_FAILURE_AND_R9_FIX.json"
    ): (
        "audit/acc_bh_process_isolation/ACC_R8_MEMORY_FAILURE_AND_R9_FIX.json",
        2_896,
        "2749c9539f0e9ce444740a8a99f5f0aab99f00472ae6383614720c4279003076",
    ),
    (
        "artifacts/single_cell_r9_acc_bh_process_isolation_audit_20260829_r1/"
        "ACC_R8_MEMORY_FAILURE_AND_R9_FIX.md"
    ): (
        "audit/acc_bh_process_isolation/ACC_R8_MEMORY_FAILURE_AND_R9_FIX.md",
        2_210,
        "0ea9290a9d317aa514a7c812b6b7bca5d8067bb6c9796021b1a0f5def05ed1d3",
    ),
    (
        "artifacts/single_cell_r9_equivalence_20260829_r1/"
        "AUDITOR_RESOURCE_FAILURE.launch.log"
    ): (
        "audit/equivalence/AUDITOR_RESOURCE_FAILURE.launch.log",
        1_510,
        "70945929640d8993ab7a0d5628b199b6e5d83f857edd009769e50f2f3532fec0",
    ),
    (
        "artifacts/single_cell_r9_equivalence_20260829_r1/"
        "INDEPENDENT_AUDIT.json"
    ): (
        "audit/equivalence/INDEPENDENT_AUDIT.json",
        2_306,
        "1cf5b2bec2674e8fc0912eb83460e3894205fb5a06d9f6e912223022d61f83fe",
    ),
    (
        "artifacts/single_cell_r9_equivalence_20260829_r1/"
        "tested_audit_v32_single_cell_r9_bh_equivalence.py"
    ): (
        "audit/equivalence/tested_audit_v32_single_cell_r9_bh_equivalence.py",
        17_384,
        "13b73de6886da209b45a4a14d21fb4668ed888fb0778a6c9664dd34a7ffa2664",
    ),
    (
        "artifacts/single_cell_r9_equivalence_auditor_failure_20260829_r1/"
        "AUDITOR_RESOURCE_FAILURE.json"
    ): (
        "audit/equivalence_auditor_failure/AUDITOR_RESOURCE_FAILURE.json",
        2_859,
        "eb5563fafb50e542f1160da4958bbb3c22849763e732c5e9bdc1404003caf619",
    ),
    (
        "artifacts/single_cell_r9_equivalence_auditor_failure_20260829_r1/"
        "README.md"
    ): (
        "audit/equivalence_auditor_failure/README.md",
        1_128,
        "6ef341c022b5992d19c2cbbe77bd8ffcf7e138171d3379c7e9cde17b02ae071a",
    ),
    (
        "artifacts/single_cell_r9_final_runner_equivalence_binding_20260829_r1/"
        "BINDING.json"
    ): (
        "audit/final_runner_equivalence/BINDING.json",
        3_158,
        "b3fde75af751a6d4af42ee34f57e3fd29eb0af04354ba2ea50639decb1f11aa8",
    ),
    (
        "artifacts/single_cell_r9_handoff_negative_gates_20260829_r1/"
        "AUDIT.json"
    ): (
        "audit/handoff_negative_gates/AUDIT.json",
        1_201,
        "81d4926cf11be3a1215811d9bc545b6386ae1cbd7f88048cf1a413a3de098373",
    ),
    (
        "artifacts/single_cell_r9_handoff_negative_gates_20260829_r1/"
        "handoff_negative_gate.pytest.log"
    ): (
        "audit/handoff_negative_gates/handoff_negative_gate.pytest.log",
        99,
        "306718181394925827f5111f8f177a5dc4ae97b107c8da3b1b97388e90c9a518",
    ),
    (
        "artifacts/single_cell_r9_interruption_safety_20260829_r1/AUDIT.json"
    ): (
        "audit/interruption_safety/AUDIT.json",
        1_250,
        "55ba7231f22bbf6b870f71a34d7a1544c8e273f4bbfef04a5438af087095f052",
    ),
    (
        "artifacts/single_cell_r9_interruption_safety_20260829_r1/"
        "interruption.log"
    ): (
        "audit/interruption_safety/interruption.log",
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "@external-tested-runner": (
        "audit/equivalence/tested_candidate_runner.py",
        86_763,
        "0e6a93ffb40f44bf12150f5bd9620eaf471b87606007a189a42291f4576eb13c",
    ),
}
EXPECTED_ENTRY_COUNT = 25
EXPECTED_TOTAL_BYTES = 338_798


class StreamingToolBridgeError(RuntimeError):
    """Raised when the immutable r9 tool bridge drifts."""


def _exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _source_path(source_key: str) -> Path:
    if source_key == "@external-tested-runner":
        return EXTERNAL_TESTED_RUNNER.resolve()
    return (ROOT / source_key).resolve()


def materialize(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if output.exists() or output.is_symlink():
        raise StreamingToolBridgeError(f"tool bridge reuse is forbidden: {output}")
    entries = []
    for source_key, (target_relative, expected_bytes, expected_sha) in EXPECTED.items():
        source = _source_path(source_key)
        if source.is_symlink() or not source.is_file():
            raise StreamingToolBridgeError(f"tool source is absent/unsafe: {source}")
        if source.stat().st_size != expected_bytes or sha256_file(source) != expected_sha:
            raise StreamingToolBridgeError(f"tool source drift: {source}")
        entries.append(
            {
                "kind": "file",
                "source_path": str(source),
                "target_path": f"{TARGET_ROOT}/{target_relative}",
                "bytes": expected_bytes,
                "sha256": expected_sha,
            }
        )
    manifest = {
        "format": COPY_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "run_id": "v32-single-cell-r7-streaming-tool-20260829-r9-process-isolated-bh",
        "target_root": TARGET_ROOT,
        "payloads_are_byte_identical": True,
        "symlinks_permitted": False,
        "overwrite_permitted": False,
        "entry_count": len(entries),
        "total_bytes": sum(row["bytes"] for row in entries),
        "entries": entries,
        "upload_started": False,
        "production_deployed": False,
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise StreamingToolBridgeError(f"unsafe bridge temporary exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        manifest_path = temporary / "COPY_MANIFEST.json"
        _exclusive(manifest_path, _canonical_json(manifest))
        loaded, manifest_sha = load_copy_manifest(manifest_path)
        plan = build_plan(loaded, manifest_sha256=manifest_sha)
        if (
            plan["entry_count"] != EXPECTED_ENTRY_COUNT
            or plan["total_bytes"] != EXPECTED_TOTAL_BYTES
        ):
            raise StreamingToolBridgeError("r9 transfer plan lost a leaf")
        validation = {
            "format": "CANCERLNCATLAS_V32_SINGLE_CELL_R9_TOOL_VALIDATION_V1",
            "copy_manifest_sha256": manifest_sha,
            "target_root": TARGET_ROOT,
            "entry_count": EXPECTED_ENTRY_COUNT,
            "total_bytes": EXPECTED_TOTAL_BYTES,
            "supersedes_copy_manifest_sha256": SUPERSEDED_MANIFEST_SHA256,
            "supersession_reason": (
                "r8's BH finalization shared the raw-scoring process lifetime and "
                "ACC exceeded the 512 MiB formal process peak by 4,190,208 bytes; "
                "r9 preserves exact BH semantics but hands the immutable raw spool "
                "to a fresh same-PID os.execve process image"
            ),
            "full_code_leaf_hashes_frozen": True,
            "runtime_binding_required_before_formal_execution": True,
            "runtime_binding_script_included": True,
            "formal17_auditor_included": True,
            "tested_candidate_runner_included": True,
            "tested_and_final_bh_ast_binding_status": (
                "PASS_FINAL_RUNNER_BH_IMPLEMENTATION_IDENTICAL_TO_TESTED_RUNNER"
            ),
            "final_runner_binding_sha256": (
                "b3fde75af751a6d4af42ee34f57e3fd29eb0af04354ba2ea50639decb1f11aa8"
            ),
            "large_equivalence_receipt_sha256": (
                "1cf5b2bec2674e8fc0912eb83460e3894205fb5a06d9f6e912223022d61f83fe"
            ),
            "large_equivalence_status": "PASS_EXACT_LARGE_FIXTURE_EQUIVALENCE",
            "old_aggregate_auditor_classification": (
                "AUDITOR_RESOURCE_FAILURE_NOT_DATA_OR_RESULT_FAILURE"
            ),
            "old_aggregate_auditor_failure_receipt_sha256": (
                "eb5563fafb50e542f1160da4958bbb3c22849763e732c5e9bdc1404003caf619"
            ),
            "negative_handoff_gates": "6_PASSED",
            "negative_handoff_gate_receipt_sha256": (
                "81d4926cf11be3a1215811d9bc545b6386ae1cbd7f88048cf1a413a3de098373"
            ),
            "interruption_safety": "PASS",
            "interruption_safety_receipt_sha256": (
                "55ba7231f22bbf6b870f71a34d7a1544c8e273f4bbfef04a5438af087095f052"
            ),
            "association_bh_family": "COMPARTMENT_ORDER",
            "association_family_scope_changed": False,
            "association_total_tests_denominator_changed": False,
            "association_execution_strategy": (
                "SERIAL_PER_COMPARTMENT_EXTERNAL_SORT_FRESH_BH_PROCESS_V2"
            ),
            "post_bh_handoff": "RAW_SPOOL_SHA256_GATED_OS_EXECVE_SAME_PID_V1",
            "duckdb_memory_limit": "64MB",
            "duckdb_memory_limit_increased": False,
            "memory_limit_bytes": 512 * 1024**2,
            "memory_limit_increased": False,
            "candidate_equivalence_outputs_reusable_for_formal": False,
            "formal_acc_must_recompute_from_raw_h5": True,
            "remaining_cancers": ["ACC", "UCEC", "READ", "GBM", "PCPG"],
            "remaining_cancers_started": False,
            "historical_derived_inputs_permitted": False,
            "raw_h5_full_sha_gate_before_matrix_access": True,
            "per_cancer_atomic_publish": True,
            "port_8260_touched": False,
            "upload_started": False,
            "production_deployed": False,
        }
        _exclusive(temporary / "VALIDATION.json", _canonical_json(validation))
        os.rename(temporary, output)
        return validation
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    result = materialize(parser.parse_args().output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
