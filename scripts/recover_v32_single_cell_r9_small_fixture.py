#!/usr/bin/env python3
"""Recreate and restore the consumed immutable R9 small BH fixture."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping


AUTHORIZED_ROOT = Path("./data/CancerLncAtlas")
GENERATOR = AUTHORIZED_ROOT / (
    "runtime/tools/single_cell_r7_streaming_20260829_r9/scripts/"
    "audit_v32_single_cell_r9_bh_equivalence.py"
)
GENERATOR_SHA256 = "99c4ece2b7aee3f3fba2a13f72152af6f7638462dc334a1d1ed307e6e687a396"
PYTHON_SHA256 = "8a9082ea4d03f7bed8b8802934fb4f7c13ebcc182cf8e939946f52058968175f"
EXPECTED_SHA256 = "588d4b019344ee3cace06e5e303242643938249710fe225221266ae632566172"
TARGET = AUTHORIZED_ROOT / (
    "runtime/audits/single_cell_r9_equivalence_20260829_r1/small_source.parquet"
)
FORMAT = "CANCERLNCATLAS_V32_SINGLE_CELL_R9_SMALL_FIXTURE_RECOVERY_V1"


class RecoveryError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"absent/unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RecoveryError(f"immutable JSON reuse forbidden: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_generator():
    spec = importlib.util.spec_from_file_location("r9_fixture_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise RecoveryError("cannot load bound R9 fixture generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recover(root: Path) -> dict[str, Any]:
    root = root.resolve()
    authorized = AUTHORIZED_ROOT.resolve(strict=True)
    if authorized not in root.parents or "/dsk2" in str(root) or "/tmp" in str(root):
        raise RecoveryError("recovery root leaves authorized ${PRIVATE_WORK_ROOT} project")
    if root.exists() or root.is_symlink():
        raise RecoveryError(f"recovery root reuse forbidden: {root}")
    if TARGET.exists() or TARGET.is_symlink():
        raise RecoveryError("target is not absent; refusing to overwrite")
    if sha256_file(GENERATOR) != GENERATOR_SHA256:
        raise RecoveryError("fixture generator SHA drift")
    python = Path(sys.executable).resolve(strict=True)
    if sha256_file(python) != PYTHON_SHA256 or "/dsk2" in str(python):
        raise RecoveryError("standalone Python binding drift")
    root.mkdir(parents=True, exist_ok=False)
    candidate = root / "small_source.parquet"
    load_generator().make_small_raw(candidate)
    candidate_sha = sha256_file(candidate)
    if candidate_sha != EXPECTED_SHA256:
        raise RecoveryError(
            f"regenerated fixture SHA mismatch: {candidate_sha} != {EXPECTED_SHA256}"
        )
    with candidate.open("rb") as source, TARGET.open("xb") as destination:
        shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    target_sha = sha256_file(TARGET)
    if target_sha != EXPECTED_SHA256:
        raise RecoveryError(f"restored target SHA mismatch: {target_sha}")
    receipt: dict[str, Any] = {
        "format": FORMAT,
        "status": "PASS_EXACT_FIXTURE_RESTORED",
        "reason_for_recovery": (
            "STANDALONE_EQUIVALENCE_R2_PASSED_FROZEN_RAW_DIRECTLY_TO_"
            "CONSUMING_R9_FINALIZER"
        ),
        "generator_path": str(GENERATOR),
        "generator_sha256": GENERATOR_SHA256,
        "standalone_python_path": str(python),
        "standalone_python_sha256": PYTHON_SHA256,
        "candidate_path": str(candidate),
        "candidate_sha256": candidate_sha,
        "restored_target_path": str(TARGET),
        "restored_target_sha256": target_sha,
        "exact_prior_sha_restored": True,
        "overwrite_performed": False,
        "reads_dsk2": False,
        "writes_dsk2": False,
        "writes_tmp": False,
        "production_deployed": False,
        "port_8260_touched": False,
        "timestamp_unix": time.time(),
    }
    receipt["receipt_contract_sha256"] = canonical_sha256(receipt)
    exclusive_json(root / "RECOVERY.json", receipt)
    return receipt


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: recover_v32_single_cell_r9_small_fixture.py ROOT")
    root = Path(sys.argv[1]).resolve()
    try:
        receipt = recover(root)
    except Exception as exc:
        try:
            authorized = AUTHORIZED_ROOT.resolve(strict=True)
            if authorized in root.parents:
                root.mkdir(parents=True, exist_ok=True)
                failure: dict[str, Any] = {
                    "format": FORMAT,
                    "status": "TYPED_FAILURE_PRESERVED",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "target_path": str(TARGET),
                    "target_exists_after_failure": TARGET.exists(),
                    "reads_dsk2": False,
                    "writes_dsk2": False,
                    "writes_tmp": False,
                    "production_deployed": False,
                    "port_8260_touched": False,
                    "timestamp_unix": time.time(),
                }
                failure["failure_contract_sha256"] = canonical_sha256(failure)
                exclusive_json(root / "TYPED_FAILURE.json", failure)
        except Exception:
            pass
        raise
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
