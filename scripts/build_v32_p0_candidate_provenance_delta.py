#!/usr/bin/env python3
"""Build a non-overwriting r3 resume manifest and P0 provenance delta."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
R3_ROOT = ARTIFACTS / "v32_production_portable_overlay_20260829_r3"
R3_COPY_MANIFEST = R3_ROOT / "COPY_MANIFEST.json"
R3_JOURNAL = (
    R3_ROOT
    / "_transfer_state_r1"
    / (
        "TRANSFER_JOURNAL."
        "09f2ae4c810a392b0718f0e2d3e02bc6e5487783a9a0d2644a240141d1705669"
        ".jsonl"
    )
)
OUTPUT = ARTIFACTS / "v32_p0_candidate_provenance_delta_20260829_r1"
DELTA_TARGET_ROOT = PurePosixPath(
    "./data/CancerLncAtlas/runtime/web_candidates/"
    "v32_20260829_r3_p0_delta_r1/v32"
)
FORMAL_CANDIDATE = (
    ARTIFACTS / "formal_prepared" / "FORMAL_CANDIDATE_UNIVERSE.parquet"
)
DIAGNOSTIC_CANDIDATE = (
    ARTIFACTS
    / "diagnostic_leaky_split_association_20260825"
    / "formal_prepared"
    / "FORMAL_CANDIDATE_UNIVERSE.parquet"
)
EXPECTED_R3_MANIFEST_SHA256 = (
    "09f2ae4c810a392b0718f0e2d3e02bc6e5487783a9a0d2644a240141d1705669"
)
EXPECTED_CANDIDATE_SHA256 = (
    "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
)
EXPECTED_DRIFT_SUFFIXES = {
    "cc_hhgt/v32/single_cell_training.py",
    "scripts/materialize_v32_bulk_coexpression.py",
}
COPY_FORMAT = "CANCERLNCATLAS_V32_PORTABLE_COPY_MANIFEST_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def write_json(path: Path, payload: Any) -> str:
    content = canonical_bytes(payload)
    write_new(path, content)
    return hashlib.sha256(content).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verified_targets(path: Path) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "file_verified":
                verified[str(row["target_path"])] = row
    return verified


def source_suffix(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rewrite_candidate_path(value: Any, formal_target: str) -> Any:
    if isinstance(value, dict):
        return {
            key: rewrite_candidate_path(item, formal_target)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rewrite_candidate_path(item, formal_target) for item in value]
    if isinstance(value, str) and (
        "diagnostic_leaky_split_association_20260825" in value
        and value.replace("\\", "/").endswith(
            "/formal_prepared/FORMAL_CANDIDATE_UNIVERSE.parquet"
        )
    ):
        return formal_target
    return value


def correction_record() -> dict[str, Any]:
    return {
        "format": "CANCERLNCATLAS_V32_CANDIDATE_PROVENANCE_CORRECTION_V1",
        "status": "PASS",
        "correction_scope": "PATH_PROVENANCE_ONLY",
        "candidate_content_sha256": EXPECTED_CANDIDATE_SHA256,
        "candidate_bytes_unchanged": True,
        "statistical_outputs_recomputed": False,
        "base_overlay": "v32_production_portable_overlay_20260829_r3",
        "base_overlay_immutable": True,
    }


def entry(source: Path, target_relative: str, *, source_override: Path | None = None) -> dict[str, Any]:
    actual = source.resolve()
    declared_source = (source_override or source).resolve()
    target = DELTA_TARGET_ROOT / PurePosixPath(target_relative)
    return {
        "bytes": actual.stat().st_size,
        "exists": True,
        "file_count": 1,
        "kind": "file",
        "sha256": sha256_file(actual),
        "source_path": str(declared_source),
        "target_path": str(target),
        "target_relative_path": PurePosixPath(target_relative).as_posix(),
    }


def corrected_jsons(stage: Path, final_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    formal_target = str(
        DELTA_TARGET_ROOT
        / "artifacts/formal_prepared/FORMAL_CANDIDATE_UNIVERSE.parquet"
    )
    specs = {
        "expression": (
            "v32_bulk_expression_release_20260826_r2_nullable_counts",
            "BULK_EXPRESSION_BINDING.json",
        ),
        "coexpression": (
            "v32_bulk_coexpression_release_20260826_r1",
            "BULK_COEXPRESSION_BINDING.json",
        ),
    }
    records: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    corrected_sources: dict[str, tuple[Path, str, str]] = {}

    for module, (folder, _binding_name) in specs.items():
        base = R3_ROOT / "artifacts" / folder / "SOURCE_INPUTS.json"
        payload = rewrite_candidate_path(load_json(base), formal_target)
        payload["portable_provenance_correction"] = correction_record()
        relative = f"artifacts/{folder}/SOURCE_INPUTS.json"
        staged = stage / "overlay" / Path(relative)
        digest = write_json(staged, payload)
        final = final_root / "overlay" / Path(relative)
        corrected_sources[module] = (staged, str(DELTA_TARGET_ROOT / relative), digest)
        manifest_entries.append(entry(staged, relative, source_override=final))
        records.append(
            {
                "artifact": relative,
                "sha256": digest,
                "diagnostic_path_references": canonical_bytes(payload).count(
                    b"diagnostic_leaky_split_association_20260825"
                ),
                "formal_candidate_references": canonical_bytes(payload).count(
                    b"artifacts/formal_prepared/FORMAL_CANDIDATE_UNIVERSE.parquet"
                ),
            }
        )

    coexp_folder = specs["coexpression"][0]
    base_lineage = R3_ROOT / "artifacts" / coexp_folder / "MODULE_LINEAGE.json"
    lineage = deepcopy(load_json(base_lineage))
    lineage["source_input_sha256"] = corrected_sources["coexpression"][2]
    lineage["portable_provenance_correction"] = correction_record()
    lineage_relative = f"artifacts/{coexp_folder}/MODULE_LINEAGE.json"
    lineage_staged = stage / "overlay" / Path(lineage_relative)
    lineage_sha = write_json(lineage_staged, lineage)
    lineage_final = final_root / "overlay" / Path(lineage_relative)
    manifest_entries.append(
        entry(lineage_staged, lineage_relative, source_override=lineage_final)
    )
    records.append(
        {
            "artifact": lineage_relative,
            "sha256": lineage_sha,
            "diagnostic_path_references": 0,
            "formal_candidate_references": 0,
        }
    )

    for module, (folder, binding_name) in specs.items():
        base = R3_ROOT / "artifacts" / folder / binding_name
        payload = rewrite_candidate_path(load_json(base), formal_target)
        payload["portable_provenance_correction"] = correction_record()
        payload["artifacts"]["source_inputs"]["path"] = corrected_sources[module][1]
        payload["artifacts"]["source_inputs"]["sha256"] = corrected_sources[module][2]
        if module == "coexpression":
            payload["artifacts"]["module_lineage"]["path"] = str(
                DELTA_TARGET_ROOT / lineage_relative
            )
            payload["artifacts"]["module_lineage"]["sha256"] = lineage_sha
        relative = f"artifacts/{folder}/{binding_name}"
        staged = stage / "overlay" / Path(relative)
        digest = write_json(staged, payload)
        final = final_root / "overlay" / Path(relative)
        manifest_entries.append(entry(staged, relative, source_override=final))
        content = canonical_bytes(payload)
        records.append(
            {
                "artifact": relative,
                "sha256": digest,
                "diagnostic_path_references": content.count(
                    b"diagnostic_leaky_split_association_20260825"
                ),
                "formal_candidate_references": content.count(
                    b"artifacts/formal_prepared/FORMAL_CANDIDATE_UNIVERSE.parquet"
                ),
            }
        )

    if any(row["diagnostic_path_references"] for row in records):
        raise RuntimeError("Corrected JSON still refers to the diagnostic candidate path")
    if sum(row["formal_candidate_references"] for row in records) < 4:
        raise RuntimeError("Corrected JSON lacks formal candidate path bindings")
    return records, manifest_entries


def run_tests(stage: Path) -> dict[str, Any]:
    base_temp = stage / "_pytest_tmp"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--basetemp",
        str(base_temp),
        "tests/test_v32_single_cell_training.py",
        "tests/test_v32_full_model_contract.py",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if base_temp.exists():
        shutil.rmtree(base_temp)
    write_new(stage / "TEST_RESULTS.txt", result.stdout.encode("utf-8"))
    if result.returncode:
        raise RuntimeError(f"Focused P0 tests failed:\n{result.stdout}")
    return {
        "status": "PASS",
        "returncode": result.returncode,
        "command": command,
        "summary": result.stdout.strip().splitlines()[-1],
    }


def build() -> Path:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing artifact reuse: {OUTPUT}")
    for required in (R3_COPY_MANIFEST, R3_JOURNAL, FORMAL_CANDIDATE, DIAGNOSTIC_CANDIDATE):
        if not required.is_file():
            raise FileNotFoundError(required)
    r3_sha_before = sha256_file(R3_COPY_MANIFEST)
    if r3_sha_before != EXPECTED_R3_MANIFEST_SHA256:
        raise RuntimeError(f"r3 manifest drift: {r3_sha_before}")

    formal_sha = sha256_file(FORMAL_CANDIDATE)
    diagnostic_sha = sha256_file(DIAGNOSTIC_CANDIDATE)
    formal_bytes = FORMAL_CANDIDATE.stat().st_size
    diagnostic_bytes = DIAGNOSTIC_CANDIDATE.stat().st_size
    if not (
        formal_sha
        == diagnostic_sha
        == EXPECTED_CANDIDATE_SHA256
        and formal_bytes == diagnostic_bytes
    ):
        raise RuntimeError("Formal and diagnostic candidate contents are not byte-identical")

    original = load_json(R3_COPY_MANIFEST)
    if original.get("format") != COPY_FORMAT:
        raise RuntimeError("r3 copy manifest format drift")
    verified = verified_targets(R3_JOURNAL)
    drift: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for index, record in enumerate(original["entries"], start=1):
        source = Path(record["source_path"]).resolve()
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"Unsafe or missing r3 source: {source}")
        current_bytes = source.stat().st_size
        current_sha = sha256_file(source)
        declared_bytes = int(record["bytes"])
        declared_sha = str(record["sha256"])
        changed = current_bytes != declared_bytes or current_sha != declared_sha
        if changed:
            target = str(record["target_path"])
            proof = verified.get(target)
            drift.append(
                {
                    "source_path": str(source),
                    "source_suffix": source_suffix(source),
                    "target_path": target,
                    "declared_sha256": declared_sha,
                    "declared_bytes": declared_bytes,
                    "current_sha256": current_sha,
                    "current_bytes": current_bytes,
                    "old_r3_remote_verified": bool(
                        proof
                        and proof.get("sha256") == declared_sha
                        and int(proof.get("bytes", -1)) == declared_bytes
                    ),
                }
            )
        elif str(record["target_path"]) not in verified:
            remaining.append(record)
        if index % 100 == 0:
            print(f"rehash {index}/{len(original['entries'])}", flush=True)

    observed_drift = {row["source_suffix"] for row in drift}
    if observed_drift != EXPECTED_DRIFT_SUFFIXES:
        raise RuntimeError(
            f"Unexpected r3 source drift: observed={sorted(observed_drift)} "
            f"expected={sorted(EXPECTED_DRIFT_SUFFIXES)}"
        )
    if not all(row["old_r3_remote_verified"] for row in drift):
        raise RuntimeError("A drifted old-r3 source lacks a file_verified journal proof")

    stage = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT.name}.staging.", dir=str(OUTPUT.parent))
    )
    final_root = OUTPUT.resolve()
    try:
        remaining_manifest = {
            "format": COPY_FORMAT,
            "target_root": original["target_root"],
            "payloads_are_byte_identical": True,
            "symlinks_permitted": False,
            "overwrite_permitted": False,
            "entries": remaining,
        }
        remaining_path = stage / "R3_REMAINING_UNCHANGED_COPY_MANIFEST.json"
        remaining_sha = write_json(remaining_path, remaining_manifest)

        corrected_records, corrected_entries = corrected_jsons(stage, final_root)
        delta_entries = [
            entry(
                FORMAL_CANDIDATE,
                "artifacts/formal_prepared/FORMAL_CANDIDATE_UNIVERSE.parquet",
            ),
            entry(
                ROOT / "cc_hhgt/v32/single_cell_training.py",
                "cc_hhgt/v32/single_cell_training.py",
            ),
            entry(
                ROOT / "cc_hhgt/v32/full_model_contract.py",
                "cc_hhgt/v32/full_model_contract.py",
            ),
            entry(
                ROOT / "scripts/materialize_v32_bulk_coexpression.py",
                "scripts/materialize_v32_bulk_coexpression.py",
            ),
            entry(
                Path(__file__).resolve(),
                "scripts/build_v32_p0_candidate_provenance_delta.py",
            ),
            *corrected_entries,
        ]
        delta_entries.sort(key=lambda row: row["target_path"])
        delta_manifest = {
            "format": COPY_FORMAT,
            "target_root": str(DELTA_TARGET_ROOT),
            "payloads_are_byte_identical": True,
            "symlinks_permitted": False,
            "overwrite_permitted": False,
            "entries": delta_entries,
        }
        delta_path = stage / "P0_CURRENT_DELTA_COPY_MANIFEST.json"
        delta_sha = write_json(delta_path, delta_manifest)
        test_audit = run_tests(stage)

        r3_sha_after = sha256_file(R3_COPY_MANIFEST)
        if r3_sha_after != r3_sha_before:
            raise RuntimeError("r3 manifest changed while building the delta")
        audit = {
            "format": "CANCERLNCATLAS_V32_P0_PROVENANCE_FAILCLOSE_AUDIT_V1",
            "status": "PASS",
            "base_r3_immutable": True,
            "base_r3_copy_manifest_path": str(R3_COPY_MANIFEST.resolve()),
            "base_r3_copy_manifest_sha256_before": r3_sha_before,
            "base_r3_copy_manifest_sha256_after": r3_sha_after,
            "candidate_binding_correction": {
                "classification": "PROCESSING_PATH_PROVENANCE_ERROR_NOT_DATA_CONTENT_ERROR",
                "formal_path": str(FORMAL_CANDIDATE.resolve()),
                "diagnostic_alias_path": str(DIAGNOSTIC_CANDIDATE.resolve()),
                "formal_sha256": formal_sha,
                "diagnostic_alias_sha256": diagnostic_sha,
                "formal_bytes": formal_bytes,
                "diagnostic_alias_bytes": diagnostic_bytes,
                "byte_identical": True,
                "corrected_json": corrected_records,
                "corrected_json_diagnostic_reference_count": sum(
                    row["diagnostic_path_references"] for row in corrected_records
                ),
                "corrected_json_formal_reference_count": sum(
                    row["formal_candidate_references"] for row in corrected_records
                ),
            },
            "r3_resume": {
                "original_entries": len(original["entries"]),
                "journal_verified_targets": len(verified),
                "verified_drift_excluded": drift,
                "remaining_entries": len(remaining),
                "remaining_bytes": sum(int(row["bytes"]) for row in remaining),
                "remaining_sources_fully_rehashed": True,
                "remaining_copy_manifest_path": str(
                    (final_root / remaining_path.name).resolve()
                ),
                "remaining_copy_manifest_sha256": remaining_sha,
            },
            "p0_delta": {
                "target_root": str(DELTA_TARGET_ROOT),
                "entry_count": len(delta_entries),
                "bytes": sum(int(row["bytes"]) for row in delta_entries),
                "copy_manifest_path": str((final_root / delta_path.name).resolve()),
                "copy_manifest_sha256": delta_sha,
                "overwrites_r3": False,
            },
            "single_cell_fail_close": {
                "outcome_derived_target_role": "training_label",
                "raw_data_training_target_rejected": True,
                "historical_label_generation_rejected": True,
                "empty_formal_datasets_cannot_unlock": True,
                "zero_trained_folds_cannot_unlock": True,
                "tests": test_audit,
                "code_sha256": {
                    "single_cell_training.py": sha256_file(
                        ROOT / "cc_hhgt/v32/single_cell_training.py"
                    ),
                    "full_model_contract.py": sha256_file(
                        ROOT / "cc_hhgt/v32/full_model_contract.py"
                    ),
                    "test_v32_single_cell_training.py": sha256_file(
                        ROOT / "tests/test_v32_single_cell_training.py"
                    ),
                    "test_v32_full_model_contract.py": sha256_file(
                        ROOT / "tests/test_v32_full_model_contract.py"
                    ),
                },
            },
            "production_deployed": False,
        }
        audit_path = stage / "AUDIT.json"
        audit_sha = write_json(audit_path, audit)
        markdown = (
            "# V3.2 P0 provenance and single-cell fail-close audit\n\n"
            "Status: **PASS**\n\n"
            f"- Formal and diagnostic-alias candidate files are byte-identical: "
            f"`{formal_sha}` ({formal_bytes:,} bytes).\n"
            "- Corrected portable bindings contain zero diagnostic-path references.\n"
            f"- r3 remains immutable at `{r3_sha_after}`.\n"
            f"- Old-r3 journal verified {len(verified)} targets; the new remaining "
            f"manifest contains {len(remaining)} fully rehashed files.\n"
            f"- P0 delta contains {len(delta_entries)} files under a distinct target root.\n"
            f"- Tests: {test_audit['summary']}.\n"
            f"- Audit SHA-256: `{audit_sha}`.\n"
        )
        write_new(stage / "AUDIT.md", markdown.encode("utf-8"))
        commands = {
            "remaining_dry_run": [
                sys.executable,
                "scripts/transfer_v32_payloads_to_local_gpu.py",
                "--copy-manifest",
                str(final_root / remaining_path.name),
                "--state-dir",
                str(final_root / "_transfer_state_remaining"),
            ],
            "remaining_execute_required_confirmation_sha256": remaining_sha,
            "delta_dry_run": [
                sys.executable,
                "scripts/transfer_v32_payloads_to_local_gpu.py",
                "--copy-manifest",
                str(final_root / delta_path.name),
                "--state-dir",
                str(final_root / "_transfer_state_delta"),
            ],
            "delta_execute_required_confirmation_sha256": delta_sha,
            "policy": "DRY_RUN_REMOTE_AUDIT_BEFORE_EXECUTE_AND_NEVER_OVERWRITE_R3",
        }
        write_json(stage / "TRANSFER_COMMANDS.json", commands)
        os.replace(stage, OUTPUT)
    except Exception:
        print(f"FAILED_STAGING_PRESERVED={stage}", file=sys.stderr, flush=True)
        raise
    print(str(OUTPUT), flush=True)
    return OUTPUT


if __name__ == "__main__":
    build()
