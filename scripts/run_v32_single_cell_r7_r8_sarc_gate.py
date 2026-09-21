#!/usr/bin/env python3
"""Run only SARC r8/r4 under the unchanged 512 MiB child-process gate."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_v32_single_cell_r7_r6_sarc_gate as _gate  # noqa: E402


_gate.FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R7_R8_SARC_GATE_V1"
_gate.FAILURE_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R7_R8_SARC_TYPED_FAILURE_V1"
_gate.TOOL_ROOT = Path(
    "./data/CancerLncAtlas/runtime/tools/"
    "single_cell_r7_streaming_20260829_r8"
)
_gate.RUNNER = _gate.TOOL_ROOT / "scripts/run_v32_single_cell_r7_streaming.py"
_gate.TOOL_MANIFEST_SHA256 = (
    "89360f86e1c96f36c69fcf347ed752ba657bb388ebfe62fdd0500a5a852ab55f"
)
_gate.TOOL_SHAS = {
    "cc_hhgt/__init__.py": "e246acd89da9a9733fbac09b2770161b5ed199695a93a832ca965d8d201b78dc",
    "cc_hhgt/v32/__init__.py": "359f958ad6e7e67078201282718193b40bc0d09bf15d27cbefb3a604b535ca91",
    "cc_hhgt/v32/contracts.py": "387e64d4931bac936e7affa4a3f974812dbfbb16d50deacd670685c796d41f46",
    "cc_hhgt/v32/single_cell_cell_level.py": "08e5b3519539fe488c78626aab7a7ef94bad4f4ef7f94b2f656f029bd5056815",
    "cc_hhgt/v32/single_cell_r7_streaming.py": "7133182f386ca74ea8b36013f16ef7679cb95306949cff8eb91ffee24ee36fd3",
    "scripts/run_v32_single_cell_r7_streaming.py": "1d91fac0de8246c57c5d5c2a7299d178abc964a9360af99f397d48eb7ddcddfa",
    "audit/MEMORY_ROOT_CAUSE_AND_REMEDIATION.json": "f8ad8a6c182f109a78737da61aecb0797d26cf8cf5fe26342cbb8fdc43b0665d",
    "audit/MEMORY_ROOT_CAUSE_AND_REMEDIATION.md": "6196263f29852efbab6b92fbf4f01dd7497fccaf597c36a6b44f89fd5ee841a8",
    "audit/SARC_R6_SCHEMA_FAILURE_AND_R7_FIX.json": "196b1fc127c1fb1baa770f235fd87323be490d843aa4f84465b81e081b6806e1",
    "audit/SARC_R6_SCHEMA_FAILURE_AND_R7_FIX.md": "3d6ba2fd99c8009dc78548f3ed3945033da6a4fd54f5f291953a120652549127",
    "audit/SARC_R3_DUCKDB_FAILURE_AND_R4_FIX.json": "b123de01e499a17a91c9201ad409d12fee3e3603dfbf3480959ad29c4ed427ad",
    "audit/SARC_R3_DUCKDB_FAILURE_AND_R4_FIX.md": "2ffb03ce41fa0b1d000f64bb3a6b6a0a5cfaa190bf30501fe865e5263979b639",
}

_original_audit_output = _gate.audit_output


def audit_output(output_root, plan):
    record = _original_audit_output(output_root, plan)
    success = _gate.load_json(output_root / f"cancer_id={_gate.CANCER}" / "SUCCESS.json")
    expected = {
        "association_execution_strategy": "SERIAL_PER_COMPARTMENT_EXTERNAL_SORT_V1",
        "association_bh_family": "COMPARTMENT_ORDER",
        "association_stage_key_policy": (
            "UNIQUE_COMPARTMENT_LNCRNA_PATHWAY_CARRY_PAYLOAD_NO_PARQUET_ROWID_JOIN"
        ),
    }
    for key, value in expected.items():
        if success.get(key) != value:
            raise _gate.SarcGateError(f"SARC r4 {key} drift")
    record.update(expected)
    record["association_family_scope_changed"] = False
    record["association_total_tests_denominator_changed"] = False
    return record


_gate.audit_output = audit_output

CANCER = _gate.CANCER
MEMORY_LIMIT_BYTES = _gate.MEMORY_LIMIT_BYTES
POLL_SECONDS = _gate.POLL_SECONDS
TOOL_ROOT = _gate.TOOL_ROOT
TOOL_MANIFEST_SHA256 = _gate.TOOL_MANIFEST_SHA256
TOOL_SHAS = _gate.TOOL_SHAS
SarcGateError = _gate.SarcGateError
runner_args = _gate.runner_args
validate_plan = _gate.validate_plan
verify_frozen_inputs = _gate.verify_frozen_inputs


def main() -> int:
    return _gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
