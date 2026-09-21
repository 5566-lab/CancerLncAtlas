#!/usr/bin/env python3
"""Run ACC plus four unstarted cancers with frozen r9 and bind formal 17.

HNSC, SARC, and the ten cancers completed before the immutable r8 typed
failure are SHA-bound upstream results.  Only ACC, UCEC, READ, GBM, and PCPG
are planned and run, serially, in a new output root.  ACC is recomputed from
raw input; its failed r8 partial spool is never reused as a formal result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
from typing import Any


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R7_R9_REMAINING5_SUPERVISOR_V1"
SUCCESS_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R7_R9_FORMAL17_BINDING_V1"
FAILURE_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R7_R9_REMAINING5_TYPED_FAILURE_V1"
BASE_SUPERVISOR = Path(
    "./data/CancerLncAtlas/runtime/tools/"
    "single_cell_r7_17c_supervisor_20260829_r1/scripts/"
    "run_v32_single_cell_r7_17c_cohort.py"
)
BASE_SUPERVISOR_SHA256 = (
    "c50bebed211306f9a17115720e067133b85d09ad3bcc0dfa29a0dfe4edae7996"
)
TOOL_ROOT = Path(
    "./data/CancerLncAtlas/runtime/tools/"
    "single_cell_r7_streaming_20260829_r9"
)
TOOL_MANIFEST_SHA256 = (
    "7b226f9fa09627ceead790718186d489e483cf20ad75d77fe2d5b18e7c37e572"
)
TOOL_SHAS = {
    "cc_hhgt/__init__.py": (
        98,
        "e246acd89da9a9733fbac09b2770161b5ed199695a93a832ca965d8d201b78dc",
    ),
    "cc_hhgt/v32/__init__.py": (
        363,
        "359f958ad6e7e67078201282718193b40bc0d09bf15d27cbefb3a604b535ca91",
    ),
    "cc_hhgt/v32/contracts.py": (
        8_363,
        "387e64d4931bac936e7affa4a3f974812dbfbb16d50deacd670685c796d41f46",
    ),
    "cc_hhgt/v32/single_cell_cell_level.py": (
        12_413,
        "08e5b3519539fe488c78626aab7a7ef94bad4f4ef7f94b2f656f029bd5056815",
    ),
    "cc_hhgt/v32/single_cell_r7_streaming.py": (
        31_679,
        "7133182f386ca74ea8b36013f16ef7679cb95306949cff8eb91ffee24ee36fd3",
    ),
    "scripts/run_v32_single_cell_r7_streaming.py": (
        90_242,
        "903658231bfe31387d12cab56b9de45ec88258e386f9be0e3ba442b61495932c",
    ),
}
PYTHON = Path("${PRIVATE_WORK_ROOT}/miniconda3/bin/python")
PYTHON_RESOLVED = Path("${PRIVATE_WORK_ROOT}/miniconda3/bin/python3.13")
PYTHON_TRUST_ROOT = Path("${PRIVATE_WORK_ROOT}/miniconda3")
PYTHON_SHA256 = "dcb43d1acbc001b6ca88ed41f47273d53b4658766875852ac1ede3003bba9329"
PYTHON_STAT = {
    "size": 35_604_512,
    "mode": 0o100775,
    "uid": 1001,
    "gid": 1001,
    "dev": 2081,
    "inode": 34_359_900_582,
}
RUNTIME_BINDING = Path(
    "./data/CancerLncAtlas/runtime/audits/"
    "single_cell_r9_runtime_binding_20260829_r1/RUNTIME_BINDING.json"
)
RUNTIME_BINDING_SHA256 = (
    "0287998ee55e3a4a1d1381f4e1994aca75ea58d0c1bd2e810645fb523b3a6a92"
)
RUNTIME_BINDING_CONTRACT_SHA256 = (
    "291759551ec7d2788cf2fab671f6b4a9fe92917b0c13a47d45541728115ff99a"
)
RUNTIME_DISTRIBUTION_SET_SHA256 = (
    "b612c398704f93e2091d5ef7a96d450ac13d9d8521a8eca204c6b9b24ddf9b2a"
)
MEMORY_LIMIT_BYTES = 512 * 1024**2
POLL_SECONDS = 0.25
FORMAL_CANCERS = frozenset(
    {
        "ACC", "CHOL", "DLBC", "ESCA", "GBM", "HNSC", "KIRC", "LAML",
        "LGG", "LUSC", "MESO", "PCPG", "READ", "SARC", "SKCM", "THYM",
        "UCEC",
    }
)
REMAINING_EXECUTION_ORDER = ("ACC", "UCEC", "READ", "GBM", "PCPG")
REMAINING_CANCERS = frozenset(REMAINING_EXECUTION_ORDER)
EXTERNAL_CANCERS = FORMAL_CANCERS - REMAINING_CANCERS
if FORMAL_CANCERS != EXTERNAL_CANCERS | REMAINING_CANCERS:
    raise RuntimeError("formal cancer partition is not exact")

HNSC_OUTPUT = Path(
    "./data/CancerLncAtlas/results/model/"
    "v32_single_cell_r7_fresh_streaming_17c_20260829_r3/cancer_id=HNSC"
)
HNSC_SUCCESS_SHA256 = (
    "045c8c6c2b908ea1abca0561d0f14514923a5a78e38fafe08f010d0419fb895d"
)
HNSC_AUDIT = Path(
    "./data/CancerLncAtlas/runtime/audits/"
    "single_cell_r7_17c_20260829_r3/records/01_HNSC.json"
)
HNSC_AUDIT_SHA256 = (
    "41a367d4c82150ea6808c6f8f5140494232bfe53c4cad70c4489884642404ea4"
)
HNSC_PRODUCER_TOOL_MANIFEST_SHA256 = (
    "06090c75a4c3a722c5be99c3ba6f3ae205c0905dbf023a68201c80d162fbf8d1"
)
SARC_OUTPUT = Path(
    "./data/CancerLncAtlas/results/model/"
    "v32_single_cell_r7_fresh_streaming_r8_sarc_20260829_r4/cancer_id=SARC"
)
SARC_SUCCESS_SHA256 = (
    "4ef8b869bcc5f5728a940820bd2412ba91dd344139c494130198509b6458870d"
)
SARC_AUDIT = Path(
    "./data/CancerLncAtlas/runtime/audits/"
    "single_cell_r7_r8_sarc_20260829_r4/SARC_GATE_SUCCESS.json"
)
SARC_AUDIT_SHA256 = (
    "67469b2d83b99cd1d06dd78fe81c5292aa39ff18f02d54b346e01532dce282c2"
)
R8_TOOL_MANIFEST_SHA256 = (
    "89360f86e1c96f36c69fcf347ed752ba657bb388ebfe62fdd0500a5a852ab55f"
)
R8_COMPLETED_OUTPUT_ROOT = Path(
    "./data/CancerLncAtlas/results/model/"
    "v32_single_cell_r7_fresh_streaming_r8_remaining15_20260829_r1"
)
R8_COMPLETED_AUDIT_ROOT = Path(
    "./data/CancerLncAtlas/runtime/audits/"
    "single_cell_r7_r8_remaining15_20260829_r1"
)
R8_TYPED_FAILURE = R8_COMPLETED_AUDIT_ROOT / "TYPED_FAILURE.json"
R8_TYPED_FAILURE_SHA256 = (
    "5c71aa44f3738265cc106f4e3b1c02881daef5e91a50629137ab92cf85e590b3"
)
R8_TYPED_FAILURE_CONTRACT_SHA256 = (
    "c6f34ac875f9e6a9481ba3c22edc94fba14af0c2e3c969f223cd2c5cb56bf7bc"
)
R8_COMPLETED_BINDINGS = {
    "SKCM": {
        "index": 1,
        "success_sha256": "f832fcda15a3e10d11faf128106049f8fadcc41a801c08a3bb3d731d8842aa99",
        "audit_sha256": "013b4cb2b21ec77359b4694f31d0a6426eda8c06c85c0e163f024d5a09f1abfe",
    },
    "CHOL": {
        "index": 2,
        "success_sha256": "38f8dff9826706416322c5db93cb9f447b465003baf9818b0fd5669e04ec4ac7",
        "audit_sha256": "bc1d4afb3d12c0f71ef5b8ec3c1b71cd490d6a91692923b695228dd679a706c2",
    },
    "LGG": {
        "index": 3,
        "success_sha256": "9bd830be14b2b4a3a0d0fbbe5af00eba679affa07bcf78ed374fb225096f7cd0",
        "audit_sha256": "c076eff89c5c47ccf2de70f1857cec68e3c976acd792c62e849c10f02b9bc728",
    },
    "DLBC": {
        "index": 4,
        "success_sha256": "77f65e86e7a5cf95ef78aeb351e4026142eb9b5192f8c45b01a0ba64ebfd6a1b",
        "audit_sha256": "778ce2cff43d3027125f3b3d1cf5355a1a8d38e36d3bd0b22bf273919f88ea7a",
    },
    "KIRC": {
        "index": 5,
        "success_sha256": "5e7127910b242b10b0773bb69131f55125ac18d1a8ac4522685714e7ab8025c8",
        "audit_sha256": "de70e856306d9cb6e4711b062d164c917e98042f9142eaa4f2a678c931c94e48",
    },
    "MESO": {
        "index": 6,
        "success_sha256": "2e03b433036ce6a08c2c54c312500ee2dc6b2c15616e6e33cc34cb2e32b55733",
        "audit_sha256": "c3efdd0a49b2d605c8ccf4e6f64ca39f1aced7ac575c78b9144fb209dcf035f6",
    },
    "LAML": {
        "index": 7,
        "success_sha256": "f333866f5e98eaf2af68d8c4eae052444fca8cc668fa5397eb3c88332dbe508a",
        "audit_sha256": "d876c3daf45d63cf34f5ec0ef0e4083d769e9c96c00feb1b7dc9184c29adde3c",
    },
    "LUSC": {
        "index": 8,
        "success_sha256": "acb32a43f49386d5388d704ccba93ddfb90c3cc04982b49b2b7a0a57c1cd1f43",
        "audit_sha256": "9144e37958f6fa8aec29de559cd8fc3e812550438b2efaee3470642d45a5c917",
    },
    "ESCA": {
        "index": 9,
        "success_sha256": "1f40e6307bf6ca44c9c247c76e88df08e737bd87774f6c47e9f5a4d4213b4850",
        "audit_sha256": "74d286acb0d670f9990a657ab0c42fed26bd56c5e2d5555e527d389667c78c37",
    },
    "THYM": {
        "index": 10,
        "success_sha256": "f2fb6062ed3ee2e31cced2124ada450a1954c8425b2fd7d16784aaa3f9a0ad95",
        "audit_sha256": "603c0461a2ca54d46fe07ca2275dddd0366d27a5738c3b6eb4b0261a250e4cb8",
    },
}


class RemainingCohortError(RuntimeError):
    """Raised when a frozen dependency or formal binding cannot be proved."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RemainingCohortError(f"cannot hash absent/unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RemainingCohortError(f"JSON is absent/unsafe: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_python_identity() -> dict[str, Any]:
    if not PYTHON.is_symlink() or not PYTHON.is_file():
        raise RemainingCohortError("pinned Python link is absent or unsafe")
    resolved = PYTHON.resolve(strict=True)
    trust_root = PYTHON_TRUST_ROOT.resolve(strict=True)
    if resolved != PYTHON_RESOLVED or trust_root not in resolved.parents:
        raise RemainingCohortError("pinned Python resolves outside trusted root")
    observed = resolved.stat()
    identity = {
        "link_path": str(PYTHON),
        "link_target": os.readlink(PYTHON),
        "resolved_target": str(resolved),
        "trusted_conda_root": str(trust_root),
        "sha256": sha256_file(resolved),
        "size": int(observed.st_size),
        "mode": int(observed.st_mode),
        "uid": int(observed.st_uid),
        "gid": int(observed.st_gid),
        "dev": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "regular_file": stat.S_ISREG(observed.st_mode),
        "owner_executable": bool(observed.st_mode & stat.S_IXUSR),
    }
    for key, expected in PYTHON_STAT.items():
        if identity[key] != expected:
            raise RemainingCohortError(f"pinned Python stat drift: {key}")
    if identity["sha256"] != PYTHON_SHA256:
        raise RemainingCohortError("pinned Python SHA drift")
    if not identity["regular_file"] or not identity["owner_executable"]:
        raise RemainingCohortError("pinned Python target is not executable")
    identity["identity_contract_sha256"] = canonical_sha256(identity)
    return identity


def verify_runtime_binding() -> dict[str, Any]:
    if sha256_file(RUNTIME_BINDING) != RUNTIME_BINDING_SHA256:
        raise RemainingCohortError("frozen runtime-binding receipt SHA drift")
    receipt = load_json(RUNTIME_BINDING)
    if (
        receipt.get("status") != "PASS_CODE_DEPENDENCY_RUNTIME_BOUND"
        or receipt.get("runtime_binding_contract_sha256")
        != RUNTIME_BINDING_CONTRACT_SHA256
        or receipt.get("copy_manifest_sha256") != TOOL_MANIFEST_SHA256
        or receipt.get("installed_distribution_set_sha256")
        != RUNTIME_DISTRIBUTION_SET_SHA256
        or int(receipt.get("verified_tool_leaf_count", -1)) != 25
        or int(receipt.get("verified_tool_total_bytes", -1)) != 338_798
    ):
        raise RemainingCohortError("frozen runtime-binding receipt semantics drift")
    dependencies = receipt.get("direct_dependencies")
    native_extensions = receipt.get("native_extensions")
    if not isinstance(dependencies, dict) or not isinstance(native_extensions, dict):
        raise RemainingCohortError("runtime binding lost dependency identities")
    for name, row in {**dependencies, **native_extensions}.items():
        if not isinstance(row, dict):
            raise RemainingCohortError(f"invalid runtime dependency row: {name}")
        path = Path(str(row.get("module_file", row.get("path", ""))))
        expected_bytes = int(row.get("module_file_bytes", row.get("bytes", -1)))
        expected_sha = str(row.get("module_file_sha256", row.get("sha256", "")))
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
            raise RemainingCohortError(f"runtime dependency binary drift: {name}")
    for name, row in dependencies.items():
        distribution = str(row.get("distribution", ""))
        if importlib.metadata.version(distribution) != str(row.get("version", "")):
            raise RemainingCohortError(f"runtime distribution version drift: {name}")
    distributions = sorted(
        {
            (
                str(distribution.metadata.get("Name", "")).lower(),
                str(distribution.version),
            )
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
    )
    if canonical_sha256(distributions) != RUNTIME_DISTRIBUTION_SET_SHA256:
        raise RemainingCohortError("installed distribution set drift after binding")
    return {
        "path": str(RUNTIME_BINDING),
        "sha256": RUNTIME_BINDING_SHA256,
        "contract_sha256": RUNTIME_BINDING_CONTRACT_SHA256,
        "installed_distribution_set_sha256": RUNTIME_DISTRIBUTION_SET_SHA256,
        "verified_direct_dependencies": sorted(dependencies),
        "verified_native_extensions": sorted(native_extensions),
    }


def external_bindings() -> dict[str, dict[str, Any]]:
    if sha256_file(R8_TYPED_FAILURE) != R8_TYPED_FAILURE_SHA256:
        raise RemainingCohortError("immutable r8 typed-failure receipt drift")
    r8_failure = load_json(R8_TYPED_FAILURE)
    expected_completed = [
        "SKCM", "CHOL", "LGG", "DLBC", "KIRC", "MESO", "LAML", "LUSC",
        "ESCA", "THYM",
    ]
    if (
        r8_failure.get("status") != "TYPED_FAILURE_STOPPED"
        or r8_failure.get("reason") != "OBSERVED_PROCESS_MEMORY_EXCEEDED_512_MIB"
        or r8_failure.get("cancer_id") != "ACC"
        or r8_failure.get("completed_before_failure") != expected_completed
        or r8_failure.get("failure_contract_sha256")
        != R8_TYPED_FAILURE_CONTRACT_SHA256
    ):
        raise RemainingCohortError("immutable r8 failure semantics drift")
    expected = {
        "HNSC": (HNSC_OUTPUT, HNSC_SUCCESS_SHA256, HNSC_AUDIT,
                 HNSC_AUDIT_SHA256, HNSC_PRODUCER_TOOL_MANIFEST_SHA256),
        "SARC": (SARC_OUTPUT, SARC_SUCCESS_SHA256, SARC_AUDIT,
                 SARC_AUDIT_SHA256, R8_TOOL_MANIFEST_SHA256),
    }
    for cancer, record in R8_COMPLETED_BINDINGS.items():
        index = int(record["index"])
        expected[cancer] = (
            R8_COMPLETED_OUTPUT_ROOT / f"cancer_id={cancer}",
            str(record["success_sha256"]),
            R8_COMPLETED_AUDIT_ROOT / "records" / f"{index:02d}_{cancer}.json",
            str(record["audit_sha256"]),
            R8_TOOL_MANIFEST_SHA256,
        )
    bindings: dict[str, dict[str, Any]] = {}
    for cancer, (root, success_sha, audit, audit_sha, tool_sha) in expected.items():
        success_path = root / "SUCCESS.json"
        if sha256_file(success_path) != success_sha or sha256_file(audit) != audit_sha:
            raise RemainingCohortError(f"{cancer} immutable external binding drift")
        success = load_json(success_path)
        if success.get("status") != "SUCCESS" or success.get("cancer_id") != cancer:
            raise RemainingCohortError(f"{cancer} external SUCCESS contract drift")
        if success.get("historical_derived_results_used") is not False:
            raise RemainingCohortError(f"{cancer} external result used historical data")
        if success.get("source_generation") != "V3.2_R7_FRESH_FROM_RAW_H5":
            raise RemainingCohortError(f"{cancer} external source generation drift")
        audit_value = load_json(audit)
        audit_status = audit_value.get("status")
        if audit_status != "SUCCESS_AUDITED":
            raise RemainingCohortError(f"{cancer} audit status drift")
        if audit_value.get("cancer_id") not in (None, cancer):
            raise RemainingCohortError(f"{cancer} audit cancer ID drift")
        bindings[cancer] = {
            "cancer_id": cancer,
            "generated_in_r9_run": False,
            "output_root": str(root),
            "success_sha256": success_sha,
            "contract_sha256": str(success["contract_sha256"]),
            "lineage_sha256": str(success["lineage_sha256"]),
            "file_manifest_sha256": str(success["file_manifest_sha256"]),
            "association_evidence_rows": int(success["association_evidence_rows"]),
            "cells": int(success["cells"]),
            "donors": int(success["donors"]),
            "lncrnas": int(success["lncrnas"]),
            "audit_path": str(audit),
            "audit_sha256": audit_sha,
            "producer_tool_manifest_sha256": tool_sha,
            "immutable_r8_typed_failure_sha256": (
                R8_TYPED_FAILURE_SHA256 if cancer in R8_COMPLETED_BINDINGS else None
            ),
            "immutable_r8_typed_failure_contract_sha256": (
                R8_TYPED_FAILURE_CONTRACT_SHA256
                if cancer in R8_COMPLETED_BINDINGS else None
            ),
        }
    if set(bindings) != EXTERNAL_CANCERS:
        raise RemainingCohortError("fixed upstream binding set is not exact")
    return bindings


def _load_base():
    if sha256_file(BASE_SUPERVISOR) != BASE_SUPERVISOR_SHA256:
        raise RemainingCohortError("frozen base supervisor SHA drift")
    spec = importlib.util.spec_from_file_location("r7_r9_remaining5_base", BASE_SUPERVISOR)
    if spec is None or spec.loader is None:
        raise RemainingCohortError("cannot load frozen base supervisor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_base(base, audit_root: Path) -> None:
    base.FORMAT = FORMAT
    base.SUCCESS_FORMAT = SUCCESS_FORMAT
    base.FAILURE_FORMAT = FAILURE_FORMAT
    base.TOOL_ROOT = TOOL_ROOT
    base.RUNNER = TOOL_ROOT / "scripts/run_v32_single_cell_r7_streaming.py"
    base.TOOL_MANIFEST_SHA256 = TOOL_MANIFEST_SHA256
    base.TOOL_SHAS = TOOL_SHAS
    base.PYTHON = PYTHON
    base.MEMORY_LIMIT_BYTES = MEMORY_LIMIT_BYTES

    original_ordered = base.ordered_formal_cancers
    original_runner_args = base.runner_args
    original_audit_partition = base.audit_partition
    original_validate_tool = base.validate_frozen_tool
    original_exclusive_json = base.exclusive_json
    original_atomic_json = base.atomic_json
    original_failure_payload = base.failure_payload

    def ordered_formal_cancers():
        full = original_ordered()
        if {row["cancer_id"] for row in full} != FORMAL_CANCERS:
            raise RemainingCohortError("formal cancer universe drift")
        remaining = [row for row in full if row["cancer_id"] in REMAINING_CANCERS]
        if tuple(row["cancer_id"] for row in remaining) != REMAINING_EXECUTION_ORDER:
            raise RemainingCohortError("remaining execution order drift")
        return remaining

    def runner_args(**kwargs):
        cancer = str(kwargs.get("cancer", ""))
        if cancer not in REMAINING_CANCERS:
            raise RemainingCohortError(f"external cancer run forbidden: {cancer}")
        argv = original_runner_args(**kwargs)
        index = argv.index("--association-pathway-block")
        argv[index + 1] = "32"
        return argv

    def validate_frozen_tool():
        identity = verify_python_identity()
        runtime_binding = verify_runtime_binding()
        bindings = external_bindings()
        observed = original_validate_tool()
        identity_path = audit_root / "PINNED_PYTHON_IDENTITY.json"
        binding_path = audit_root / "EXTERNAL_BINDINGS_AUDIT.json"
        original_exclusive_json(identity_path, identity)
        binding_audit = {
            "format": SUCCESS_FORMAT,
            "status": "TWELVE_IMMUTABLE_UPSTREAM_BINDINGS_VERIFIED",
            "bindings": bindings,
            "r8_typed_failure_path": str(R8_TYPED_FAILURE),
            "r8_typed_failure_sha256": R8_TYPED_FAILURE_SHA256,
            "r8_typed_failure_contract_sha256": R8_TYPED_FAILURE_CONTRACT_SHA256,
            "r8_acc_partial_reused": False,
            "r8_first_ten_recomputed": False,
            "sarc_r3_partial_reused": False,
            "hnsc_overwritten": False,
            "timestamp_unix": time.time(),
        }
        binding_audit["binding_contract_sha256"] = canonical_sha256(binding_audit)
        original_exclusive_json(binding_path, binding_audit)
        return {
            "runner_files": observed,
            "base_supervisor": {
                "path": str(BASE_SUPERVISOR),
                "sha256": BASE_SUPERVISOR_SHA256,
            },
            "pinned_python_identity": identity,
            "runtime_binding": runtime_binding,
        }

    def process_memory_bytes(process_id: int) -> tuple[int, int]:
        status_path = Path(f"/proc/{int(process_id)}/status")
        if not status_path.is_file():
            return 0, 0
        rss = high_water = 0
        for line in status_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                high_water = int(line.split()[1]) * 1024
        return rss, high_water

    def process_group_memory(process_group_id: int) -> tuple[int, int, int, int]:
        group_rss = maximum_member_rss = maximum_member_hwm = member_count = 0
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            process_id = int(entry.name)
            try:
                if os.getpgid(process_id) != process_group_id:
                    continue
            except (ProcessLookupError, PermissionError):
                continue
            rss, high_water = process_memory_bytes(process_id)
            group_rss += rss
            maximum_member_rss = max(maximum_member_rss, rss)
            maximum_member_hwm = max(maximum_member_hwm, high_water)
            member_count += 1
        return group_rss, maximum_member_rss, maximum_member_hwm, member_count

    def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)

    def run_monitored(argv, *, log_path, memory_limit_bytes):
        if log_path.exists() or log_path.is_symlink():
            raise RemainingCohortError(f"log reuse is forbidden: {log_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(TOOL_ROOT)
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "OMP_DYNAMIC": "FALSE",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "BLIS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            }
        )
        started = time.time()
        maximum_group_rss = maximum_member_rss = maximum_member_hwm = 0
        maximum_group_member_count = 0
        maximum_supervisor_rss = maximum_supervisor_hwm = 0
        maximum_total_rss = maximum_total_hwm_guard = 0
        memory_exceeded = False
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, env=environment,
                start_new_session=True,
            )
            while process.poll() is None:
                group_rss, member_rss, member_hwm, member_count = (
                    process_group_memory(process.pid)
                )
                maximum_group_rss = max(maximum_group_rss, group_rss)
                maximum_member_rss = max(maximum_member_rss, member_rss)
                maximum_member_hwm = max(maximum_member_hwm, member_hwm)
                maximum_group_member_count = max(
                    maximum_group_member_count, member_count
                )
                supervisor_rss, supervisor_hwm = process_memory_bytes(os.getpid())
                maximum_supervisor_rss = max(maximum_supervisor_rss, supervisor_rss)
                maximum_supervisor_hwm = max(maximum_supervisor_hwm, supervisor_hwm)
                total_rss = supervisor_rss + group_rss
                total_hwm_guard = supervisor_hwm + member_hwm
                maximum_total_rss = max(maximum_total_rss, total_rss)
                maximum_total_hwm_guard = max(
                    maximum_total_hwm_guard, total_hwm_guard
                )
                if max(total_rss, total_hwm_guard) > int(memory_limit_bytes):
                    memory_exceeded = True
                    terminate_process_group(process)
                    break
                time.sleep(POLL_SECONDS)
            return_code = process.wait()
            log.flush()
            os.fsync(log.fileno())
        formal_value = max(maximum_total_rss, maximum_total_hwm_guard)
        return {
            "argv": argv,
            "return_code": int(return_code),
            "elapsed_seconds": round(time.time() - started, 3),
            "maximum_rss_bytes_observed": int(maximum_member_rss),
            "maximum_high_water_bytes_observed": int(maximum_member_hwm),
            "maximum_process_group_rss_sum_bytes_observed": int(maximum_group_rss),
            "maximum_process_group_member_count_observed": int(
                maximum_group_member_count
            ),
            "maximum_supervisor_rss_bytes_observed": int(maximum_supervisor_rss),
            "maximum_supervisor_high_water_bytes_observed": int(
                maximum_supervisor_hwm
            ),
            "maximum_supervisor_plus_child_group_rss_bytes_observed": int(
                maximum_total_rss
            ),
            "maximum_supervisor_plus_member_hwm_guard_bytes_observed": int(
                maximum_total_hwm_guard
            ),
            "formal_gate_value_bytes": int(formal_value),
            "memory_limit_bytes": int(memory_limit_bytes),
            "memory_exceeded": memory_exceeded,
            "memory_metric": (
                "MAX_OF_SUPERVISOR_PLUS_CHILD_GROUP_RSS_SUM_AND_"
                "SUPERVISOR_PLUS_MEMBER_VMHWM_GUARD"
            ),
            "process_group_rss_sum_measured": True,
            "supervisor_memory_included": True,
            "thread_pins": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "BLIS_NUM_THREADS": "1",
            },
            "poll_seconds": POLL_SECONDS,
            "log_path": str(log_path),
            "log_sha256": sha256_file(log_path),
        }

    def audit_partition(**kwargs):
        record = original_audit_partition(**kwargs)
        final = Path(kwargs["final"])
        success = load_json(final / "SUCCESS.json")
        expected = {
            "association_engine": "DUCKDB_EXTERNAL_BH_SORT_PROCESS_ISOLATED_V2",
            "association_duckdb_memory_limit": "64MB",
            "association_execution_strategy": (
                "SERIAL_PER_COMPARTMENT_EXTERNAL_SORT_FRESH_BH_PROCESS_V2"
            ),
            "association_bh_family": "COMPARTMENT_ORDER",
            "association_stage_key_policy": (
                "UNIQUE_COMPARTMENT_LNCRNA_PATHWAY_CARRY_PAYLOAD_NO_PARQUET_ROWID_JOIN"
            ),
        }
        for key, value in expected.items():
            if success.get(key) != value:
                raise RemainingCohortError(f"{record['cancer_id']} {key} drift")
        if int(success.get("parquet_batch_rows", 0)) != 16_384:
            raise RemainingCohortError(f"{record['cancer_id']} batch size drift")
        record.update(expected)
        record["association_family_scope_changed"] = False
        record["association_total_tests_denominator_changed"] = False
        record["formal_transition"] = "OS_EXECVE_REPLACES_EXPRESSION_PROCESS_IMAGE"
        record["parent_helper_overlap"] = False
        if not success.get("post_bh_exec_handoff_sha256"):
            raise RemainingCohortError(
                f"{record['cancer_id']} post-BH handoff SHA absent"
            )
        record["post_bh_exec_handoff_sha256"] = success[
            "post_bh_exec_handoff_sha256"
        ]
        return record

    def build_formal_binding(payload: dict[str, Any]) -> dict[str, Any]:
        value = dict(payload)
        per_cancer = list(value.get("per_cancer", []))
        if tuple(row.get("cancer_id") for row in per_cancer) != REMAINING_EXECUTION_ORDER:
            raise RemainingCohortError("completed remaining-cancer order drift")
        bindings = external_bindings()
        for row in per_cancer:
            cancer = str(row["cancer_id"])
            bindings[cancer] = {
                "cancer_id": cancer,
                "generated_in_r9_run": True,
                "output_root": str(row["output_root"]),
                "success_sha256": str(row["success_sha256"]),
                "contract_sha256": str(row["contract_sha256"]),
                "lineage_sha256": str(row["lineage_sha256"]),
                "file_manifest_sha256": str(row["file_manifest_sha256"]),
                "association_evidence_rows": int(row["association_evidence_rows"]),
                "cells": int(row["cells"]),
                "donors": int(row["donors"]),
                "lncrnas": int(row["lncrnas"]),
                "producer_tool_manifest_sha256": TOOL_MANIFEST_SHA256,
            }
        if set(bindings) != FORMAL_CANCERS:
            raise RemainingCohortError("formal binding set is not exactly 17 cancers")
        ordered_bindings = [bindings[cancer] for cancer in sorted(FORMAL_CANCERS)]
        value.pop("cohort_contract_sha256", None)
        value.update(
            {
                "format": SUCCESS_FORMAT,
                "status": "SUCCESS_17_OF_17_BOUND_AND_AUDITED",
                "formal_cancer_universe": sorted(FORMAL_CANCERS),
                "formal_bound_cancer_count": 17,
                "generated_in_this_run_count": 5,
                "external_immutable_binding_count": 12,
                "remaining_execution_order": list(REMAINING_EXECUTION_ORDER),
                "formal_bindings": ordered_bindings,
                "total_cells_all_bound": sum(row["cells"] for row in ordered_bindings),
                "total_association_evidence_rows_all_bound": sum(
                    row["association_evidence_rows"] for row in ordered_bindings
                ),
                "sarc_r3_partial_reused": False,
                "hnsc_overwritten": False,
                "r8_acc_partial_reused": False,
                "r8_typed_failure_path": str(R8_TYPED_FAILURE),
                "r8_typed_failure_sha256": R8_TYPED_FAILURE_SHA256,
                "r8_typed_failure_contract_sha256": (
                    R8_TYPED_FAILURE_CONTRACT_SHA256
                ),
                "r8_first_ten_recomputed": False,
                "single_physical_root_required": False,
                "aggregation_strategy": "IMMUTABLE_MULTI_ROOT_SHA256_BINDING_MANIFEST_V1",
            }
        )
        value["cohort_contract_sha256"] = canonical_sha256(value)
        return value

    def exclusive_json(path, payload):
        value = payload
        if path.name == "COHORT_SUCCESS.json":
            value = build_formal_binding(payload)
        elif path.name == "COHORT_RUNNING.json":
            value = dict(payload)
            value.update(
                {
                    "formal_cancer_universe": sorted(FORMAL_CANCERS),
                    "remaining_execution_order": list(REMAINING_EXECUTION_ORDER),
                    "external_immutable_cancers": sorted(EXTERNAL_CANCERS),
                    "memory_limit_bytes": MEMORY_LIMIT_BYTES,
                    "memory_metric": (
                        "MAX_OF_SUPERVISOR_PLUS_CHILD_GROUP_RSS_SUM_AND_"
                        "SUPERVISOR_PLUS_MEMBER_VMHWM_GUARD"
                    ),
                    "process_group_rss_sum_measured": True,
                    "supervisor_memory_included": True,
                    "thread_pins": {
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1",
                        "BLIS_NUM_THREADS": "1",
                    },
                    "poll_seconds": POLL_SECONDS,
                    "sarc_r3_partial_reused": False,
                    "hnsc_overwritten": False,
                    "r8_acc_partial_reused": False,
                }
            )
        return original_exclusive_json(path, value)

    def atomic_json(path, payload):
        value = payload
        if path.name == "COHORT_STATUS.json" and str(payload.get("status", "")).startswith("SUCCESS_17"):
            value = build_formal_binding(payload)
            value["format"] = FORMAT
            value.pop("cohort_contract_sha256", None)
            value["cohort_contract_sha256"] = canonical_sha256(value)
        return original_atomic_json(path, value)

    def failure_payload(**kwargs):
        value = original_failure_payload(**kwargs)
        status_path = Path(kwargs["audit_root"]) / "COHORT_STATUS.json"
        completed = []
        if status_path.is_file() and not status_path.is_symlink():
            completed = load_json(status_path).get("completed_cancers", [])
        value.update(
            {
                "formal_cancer_universe": sorted(FORMAL_CANCERS),
                "remaining_execution_order": list(REMAINING_EXECUTION_ORDER),
                "completed_before_failure": completed,
                "subsequent_cancers_not_started": True,
                "sarc_r3_partial_reused": False,
                "hnsc_overwritten": False,
                "r8_acc_partial_reused": False,
                "r8_typed_failure_sha256": R8_TYPED_FAILURE_SHA256,
                "r8_typed_failure_contract_sha256": (
                    R8_TYPED_FAILURE_CONTRACT_SHA256
                ),
                "r8_first_ten_recomputed": False,
            }
        )
        value.pop("failure_contract_sha256", None)
        value["failure_contract_sha256"] = canonical_sha256(value)
        return value

    base.ordered_formal_cancers = ordered_formal_cancers
    base.runner_args = runner_args
    base.validate_frozen_tool = validate_frozen_tool
    base.run_monitored = run_monitored
    base.audit_partition = audit_partition
    base.exclusive_json = exclusive_json
    base.atomic_json = atomic_json
    base.failure_payload = failure_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    audit_root = args.audit_root.resolve()
    authorized = Path("./data/CancerLncAtlas")
    if authorized not in output_root.parents or authorized not in audit_root.parents:
        raise RemainingCohortError("paths leave authorized ${PRIVATE_WORK_ROOT} tree")
    if output_root.exists() or output_root.is_symlink():
        raise RemainingCohortError(f"new output root already exists: {output_root}")
    if audit_root.exists() or audit_root.is_symlink():
        raise RemainingCohortError(f"new audit root already exists: {audit_root}")
    base = _load_base()
    configure_base(base, audit_root)
    try:
        return_code = base.execute(
            argparse.Namespace(
                output_root=output_root, audit_root=audit_root, resume=False
            )
        )
    except Exception as exc:
        failure_path = audit_root / "TYPED_FAILURE.json"
        if audit_root.is_dir() and not failure_path.exists():
            failure = {
                "format": FAILURE_FORMAT,
                "status": "TYPED_FAILURE_STOPPED",
                "stage": "SUPERVISOR",
                "reason": "SUPERVISOR_CONTRACT_ERROR",
                "detail": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                "formal_cancer_universe": sorted(FORMAL_CANCERS),
                "remaining_execution_order": list(REMAINING_EXECUTION_ORDER),
                "subsequent_cancers_not_started": True,
                "memory_limit_bytes": MEMORY_LIMIT_BYTES,
                "tool_manifest_sha256": TOOL_MANIFEST_SHA256,
                "output_root": str(output_root),
                "audit_root": str(audit_root),
                "sarc_r3_partial_reused": False,
                "hnsc_overwritten": False,
                "r8_acc_partial_reused": False,
                "r8_typed_failure_sha256": R8_TYPED_FAILURE_SHA256,
                "r8_typed_failure_contract_sha256": (
                    R8_TYPED_FAILURE_CONTRACT_SHA256
                ),
                "r8_first_ten_recomputed": False,
                "production_deployed": False,
                "port_8260_touched": False,
                "timestamp_unix": time.time(),
            }
            failure["failure_contract_sha256"] = canonical_sha256(failure)
            base.exclusive_json(failure_path, failure)
        raise
    if return_code == 0:
        success_path = audit_root / "COHORT_SUCCESS.json"
        success = load_json(success_path)
        if success.get("status") != "SUCCESS_17_OF_17_BOUND_AND_AUDITED":
            raise RemainingCohortError("final formal binding status drift")
        print(json.dumps(success, ensure_ascii=False, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
