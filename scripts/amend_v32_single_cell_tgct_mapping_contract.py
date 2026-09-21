#!/usr/bin/env python3
"""Create an immutable contract correcting one cancer's lncRNA mapping count."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

import h5py
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("v32_sc_r7_runner_for_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--cancer-id", default="TGCT")
    parser.add_argument("--expected-parent-count", type=int, default=15518)
    parser.add_argument("--expected-runner-count", type=int, default=15565)
    args = parser.parse_args()
    cancer = str(args.cancer_id).upper()
    source = args.input_root.resolve()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable correction output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    parent_success_path = source / "SUCCESS.json"
    parent_success = json.loads(parent_success_path.read_text(encoding="utf-8"))
    expected = {
        "RUN_STATUS.json": parent_success["run_status_sha256"],
        "SERVER_PREFLIGHT.json": parent_success["preflight_sha256"],
        "dataset_manifest_33c.parquet": parent_success["dataset_manifest_sha256"],
    }
    for name, expected_sha in expected.items():
        observed = sha256(source / name)
        if observed != expected_sha:
            raise RuntimeError(f"parent contract hash drift: {name}: {observed} != {expected_sha}")

    status = json.loads((source / "RUN_STATUS.json").read_text(encoding="utf-8"))
    preflight = json.loads((source / "SERVER_PREFLIGHT.json").read_text(encoding="utf-8"))
    record = status["per_cancer"][cancer]
    old_count = int(record["fresh_direct_id_lncrna_feature_count"])
    if old_count != int(args.expected_parent_count):
        raise RuntimeError(f"unexpected parent {cancer} count: {old_count}")

    with h5py.File(record["h5_path"], "r") as handle:
        group = handle["matrix"] if "matrix" in handle else handle
        feature_ids = decode(group["features"]["id"][:])
        feature_names = decode(group["features"]["name"][:])
    annotation = pd.read_parquet(record["annotation_path"])
    runner = load_runner(args.runner.resolve())
    mapping = runner._build_feature_mapping(
        feature_ids=feature_ids,
        feature_names=feature_names,
        annotation=annotation,
        measurement_scale=str(record["measurement_scale"]),
    )
    corrected_count = len(mapping["lncrna_ids"])
    if corrected_count != int(args.expected_runner_count):
        raise RuntimeError(f"independent runner mapping count drift: {corrected_count}")

    correction = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_FEATURE_MAPPING_CONTRACT_CORRECTION_V1",
        "status": "PASS",
        "cancer_id": cancer,
        "parent_contract_root": str(source),
        "parent_success_sha256": sha256(parent_success_path),
        "source_receipt_count": old_count,
        "runner_mapping_count": corrected_count,
        "cause": (
            "SOURCE_RESCUE_RECEIPT_USED_A_SEPARATE_FROZEN_LNCRNA_ID_LIST; "
            "FORMAL_R7_RUNNER_REMAPS_THE_FEATURE_UNIVERSE_AGAINST_GENCODE_V50"
        ),
        "source_data_changed": False,
        "formal_runner_changed": False,
        "correction_scope": f"{cancer}_EXPECTED_MAPPED_LNCRNA_COUNT_ONLY",
    }
    correction_path = output / f"{cancer}_MAPPING_CORRECTION_AUDIT.json"
    atomic_json(correction_path, correction)

    corrected_record = dict(record)
    corrected_record["source_receipt_lncrna_feature_count"] = old_count
    corrected_record["fresh_direct_id_lncrna_feature_count"] = corrected_count
    corrected_record["lncrna_count_correction_audit_path"] = str(correction_path)
    corrected_record["lncrna_count_correction_audit_sha256"] = sha256(correction_path)
    status["format"] = "CC_HHGT_V3_2_SINGLE_CELL_R12_FEATURE_MAPPING_CORRECTED_RUN_STATUS_V1"
    status["status"] = f"PASS_INPUT_CONTRACT_READY_WITH_{cancer}_MAPPING_CORRECTION"
    status["parent_contract_root"] = str(source)
    status["per_cancer"][cancer] = corrected_record
    status_path = output / "RUN_STATUS.json"
    atomic_json(status_path, status)

    # The formal runner intentionally accepts only the frozen R7/R11 preflight
    # schema identifiers.  Keep the accepted schema and record the immutable
    # correction lineage in a separate field instead of inventing a format the
    # runner must be weakened to accept.
    preflight["format"] = "CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_SERVER_PREFLIGHT_V1"
    preflight["contract_correction_format"] = (
        "CC_HHGT_V3_2_SINGLE_CELL_R12_FEATURE_MAPPING_CORRECTED_SERVER_PREFLIGHT_V1"
    )
    preflight["r7_root"] = str(output)
    preflight["new_rescue_receipts"][cancer] = corrected_record
    preflight["mapping_correction_cancer_id"] = cancer
    preflight["mapping_correction_audit_path"] = str(correction_path)
    preflight["mapping_correction_audit_sha256"] = sha256(correction_path)
    preflight_path = output / "SERVER_PREFLIGHT.json"
    atomic_json(preflight_path, preflight)

    for name in ("dataset_manifest_33c.parquet", "TRAINING_HANDOFF.json"):
        link_or_copy(source / name, output / name)
    success = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R12_FEATURE_MAPPING_CORRECTED_CONTRACT_SUCCESS_V1",
        "status": "SUCCESS",
        "parent_success_sha256": sha256(parent_success_path),
        "correction_audit_sha256": sha256(correction_path),
        "run_status_sha256": sha256(status_path),
        "preflight_sha256": sha256(preflight_path),
        "dataset_manifest_sha256": sha256(output / "dataset_manifest_33c.parquet"),
        "training_handoff_sha256": sha256(output / "TRAINING_HANDOFF.json"),
        "corrected_cancer_id": cancer,
        "corrected_lncrna_count": corrected_count,
        "source_data_changed": False,
        "formal_runner_changed": False,
    }
    atomic_json(output / "SUCCESS.json", success)
    print(json.dumps(success, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
