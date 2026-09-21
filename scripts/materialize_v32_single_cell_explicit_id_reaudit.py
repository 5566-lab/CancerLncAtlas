#!/usr/bin/env python3
"""Materialize the explicit-ID supplement to the V3.2 raw-H5 re-audit."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
CANCER_ORDER = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)
EXPECTED_CANCERS = set(CANCER_ORDER)
R1_BINDING_SHA256 = "6f0e637ba45f189ade5825758d06ddb01a3700f914b03632422bdc5359f89164"
R1_TABLE_SHA256 = "750f0a95cc8438a31b1298bd0d19ead4f089ffc599b3d14db1b667a8c1c2126b"
R1_RAW_SHA256 = "d6b2b3c25b11fb7cd4b4a3545b999aa1ef6cc8c74f2f282413ae560313722ce4"
MANIFEST_SHA256 = "bea2d20a261063dacdd05b3b18865f8de0fec3504086e08e6295b63fe0a4ffad"
MAPPING_POLICY = (
    "VERSION_STRIP;DIRECT_STABLE_ENSEMBL;"
    "EXACT_UNIQUE_FULL_ANNOTATION_SYMBOL;STABLE_ID_DEDUP"
)


class ExplicitIdAuditError(RuntimeError):
    """Raised when the explicit-ID supplement cannot prove its authorities."""


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ExplicitIdAuditError(f"Missing or unsafe file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def set_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(values))).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExplicitIdAuditError(f"JSON is not an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def require_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ExplicitIdAuditError(f"{label} SHA256 drift: {observed} != {expected}")
    return observed


def run_explicit_id_scan(
    scanner: Path,
    host: str,
    ssh_executable: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    source = scanner.read_text(encoding="utf-8")
    scanner_sha256 = sha256_file(scanner)
    collected: dict[str, dict[str, Any]] = {}
    annotation: dict[str, Any] | None = None
    if checkpoint_path.exists():
        checkpoint = load_json(checkpoint_path)
        rows = checkpoint.get("per_cancer", [])
        if (
            checkpoint.get("scanner_sha256") != scanner_sha256
            or checkpoint.get("remote_writes_performed") is not False
            or not isinstance(rows, list)
        ):
            raise ExplicitIdAuditError(
                f"Unsafe or stale explicit-ID checkpoint: {checkpoint_path}"
            )
        for row in rows:
            if not isinstance(row, dict):
                raise ExplicitIdAuditError(f"Malformed checkpoint row: {checkpoint_path}")
            cancer = str(row.get("cancer_id", ""))
            if cancer not in EXPECTED_CANCERS or cancer in collected:
                raise ExplicitIdAuditError(f"Unexpected checkpoint cancer: {cancer}")
            collected[cancer] = row
        annotation_value = checkpoint.get("annotation")
        if annotation_value is not None and not isinstance(annotation_value, dict):
            raise ExplicitIdAuditError(f"Malformed checkpoint annotation: {checkpoint_path}")
        annotation = annotation_value
        print(
            f"EXPLICIT_ID_CHECKPOINT_LOADED\t{len(collected)}/33",
            file=sys.stderr,
            flush=True,
        )

    consecutive_empty_failures = 0
    while len(collected) < len(CANCER_ORDER):
        remaining = [cancer for cancer in CANCER_ORDER if cancer not in collected]
        command = [
            ssh_executable,
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            "-o", "ConnectionAttempts=1",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=20",
            host,
            "python3 - --cancers " + ",".join(remaining),
        ]
        before = len(collected)
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
        )
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(source)
        process.stdin.close()
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if line.startswith("ANNOTATION_JSON\t"):
                current = json.loads(line.split("\t", 1)[1])
                if annotation is not None and canonical_sha256(current) != canonical_sha256(annotation):
                    process.kill()
                    raise ExplicitIdAuditError("Annotation changed across retry connections")
                annotation = current
            elif line.startswith("CANCER_JSON\t"):
                row = json.loads(line.split("\t", 1)[1])
                cancer = str(row.get("cancer_id", ""))
                if cancer not in remaining or cancer in collected:
                    process.kill()
                    raise ExplicitIdAuditError(f"Unexpected streamed cancer row: {cancer}")
                collected[cancer] = row
                write_json(
                    checkpoint_path,
                    {
                        "scanner_sha256": scanner_sha256,
                        "annotation": annotation,
                        "per_cancer": [
                            collected[key] for key in CANCER_ORDER if key in collected
                        ],
                        "remote_writes_performed": False,
                    },
                )
                print(
                    f"EXPLICIT_ID_LOCAL_CHECKPOINT\t{len(collected)}/33\t{cancer}",
                    file=sys.stderr,
                    flush=True,
                )
        return_code = process.wait()
        consecutive_empty_failures = (
            consecutive_empty_failures + 1 if len(collected) == before else 0
        )
        if return_code and len(collected) < len(CANCER_ORDER):
            print(
                f"EXPLICIT_ID_REMOTE_RETRY\texit={return_code}\t"
                f"completed={len(collected)}/33",
                file=sys.stderr,
                flush=True,
            )
        if consecutive_empty_failures >= 3:
            raise ExplicitIdAuditError(
                "Explicit-ID scanner made no progress in three retries; "
                f"completed={len(collected)}"
            )
    if annotation is None:
        raise ExplicitIdAuditError("Explicit-ID scanner omitted annotation provenance")
    return {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_EXPLICIT_ID_RAW_H5_REAUDIT_V1",
        "remote_host_role": "COMPUTE_HOST_READ_ONLY",
        "remote_writes_performed": False,
        "cancers": list(CANCER_ORDER),
        "annotation": annotation,
        "per_cancer": [collected[cancer] for cancer in CANCER_ORDER],
    }


def validate_id_row(row: dict[str, Any], reference: dict[str, Any]) -> None:
    cancer = str(row.get("cancer_id", ""))
    universe = row.get("lncRNA_universe_ids")
    detected = row.get("detected_lncRNA_ids")
    if not isinstance(universe, list) or not isinstance(detected, list):
        raise ExplicitIdAuditError(f"Explicit IDs are not lists: {cancer}")
    if universe != sorted(set(universe)) or detected != sorted(set(detected)):
        raise ExplicitIdAuditError(f"Explicit IDs are not sorted unique sets: {cancer}")
    if not set(detected).issubset(universe):
        raise ExplicitIdAuditError(f"Detected IDs are not a universe subset: {cancer}")
    comparisons = {
        "lncRNA_universe_count": (
            len(universe), int(reference["formal_policy_lncRNA_universe_count"])
        ),
        "detected_lncRNA_count": (
            len(detected), int(reference["formal_policy_detected_lncRNA_count"])
        ),
        "lncRNA_universe_set_sha256": (
            set_sha256(universe), str(reference["formal_policy_universe_sha256"])
        ),
        "detected_lncRNA_set_sha256": (
            set_sha256(detected), str(reference["formal_policy_detected_sha256"])
        ),
        "cell_count": (int(row["cell_count"]), int(reference["cell_count"])),
        "feature_count": (int(row["feature_count"]), int(reference["feature_count"])),
        "stored_nnz": (int(row["stored_nnz"]), int(reference["stored_nnz"])),
        "h5_path": (str(row["h5_path"]), str(reference["h5_path"])),
        "h5_size_bytes": (
            int(row["h5_size_bytes"]), int(reference["h5_size_bytes"])
        ),
        "h5_mtime_utc": (
            str(row["h5_mtime_utc"]), str(reference["h5_mtime_utc"])
        ),
    }
    for label, (observed, expected) in comparisons.items():
        if observed != expected:
            raise ExplicitIdAuditError(
                f"r2 explicit-ID/r1 mismatch for {cancer} {label}: "
                f"{observed!r} != {expected!r}"
            )
    if row.get("matrix_orientation_validated") is not True:
        raise ExplicitIdAuditError(f"Sparse orientation was not validated: {cancer}")
    if row.get("detection_value_threshold") != "FINITE_VALUE_GT_0":
        raise ExplicitIdAuditError(f"Detection threshold drift: {cancer}")
    if int(row.get("minimum_detected_cell_count", 0)) != 1:
        raise ExplicitIdAuditError(f"Minimum-cell threshold drift: {cancer}")


def materialize(
    *,
    r1_root: Path,
    manifest_path: Path,
    scanner_path: Path,
    output_root: Path,
    remote_host: str,
    ssh_executable: str,
) -> dict[str, Any]:
    destination = output_root.resolve()
    if destination.exists():
        raise ExplicitIdAuditError(f"Refusing to overwrite explicit-ID audit: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    r1_binding_path = r1_root / "SINGLE_CELL_LNCRNA_REAUDIT_BINDING.json"
    r1_table_path = r1_root / "single_cell_lncrna_expression_33c_reaudit.tsv"
    r1_raw_path = r1_root / "RAW_H5_RECOMPUTATION.json"
    require_hash(r1_binding_path, R1_BINDING_SHA256, "r1 binding")
    require_hash(r1_table_path, R1_TABLE_SHA256, "r1 per-cancer table")
    require_hash(r1_raw_path, R1_RAW_SHA256, "r1 raw recomputation")
    require_hash(manifest_path, MANIFEST_SHA256, "r6 dataset manifest")
    r1_binding = load_json(r1_binding_path)
    r1_raw = load_json(r1_raw_path)
    r1_table = pd.read_csv(r1_table_path, sep="\t", dtype={"cancer_id": "string"})
    manifest = pd.read_parquet(manifest_path)
    if (
        r1_binding.get("status") != "SUCCESS_CODE_BOUND_33_CANCER_REAUDIT"
        or r1_raw.get("remote_writes_performed") is not False
        or len(r1_table) != 33
        or set(r1_table.cancer_id.astype(str)) != EXPECTED_CANCERS
        or len(manifest) != 33
        or set(manifest.cancer_id.astype(str)) != EXPECTED_CANCERS
    ):
        raise ExplicitIdAuditError("r1 or manifest 33-cancer authority is inconsistent")

    checkpoint_path = destination.parent / f".{destination.name}.explicit-id-checkpoint.json"
    raw = run_explicit_id_scan(
        scanner_path, remote_host, ssh_executable, checkpoint_path
    )
    if (
        raw.get("remote_writes_performed") is not False
        or set(raw.get("cancers", [])) != EXPECTED_CANCERS
        or raw.get("annotation", {}).get("release") != "GENCODE v50"
        or raw.get("annotation", {}).get("strict_primary_lncRNA_count") != 34_866
        or raw.get("annotation", {}).get("mapping_policy") != MAPPING_POLICY
        or raw.get("annotation", {}).get("full_gtf", {}).get("sha256")
        != r1_raw.get("annotation", {}).get("full_gtf", {}).get("sha256")
        or raw.get("annotation", {}).get("strict_primary_lncRNA_set_sha256")
        != r1_raw.get("annotation", {}).get("strict_primary_lncRNA_set_sha256")
    ):
        raise ExplicitIdAuditError("Explicit-ID annotation/scope drift from r1")

    r1_rows = {row["cancer_id"]: row for row in r1_raw["per_cancer"]}
    explicit_rows = {row["cancer_id"]: row for row in raw["per_cancer"]}
    if set(r1_rows) != EXPECTED_CANCERS or set(explicit_rows) != EXPECTED_CANCERS:
        raise ExplicitIdAuditError("Raw per-cancer row scope is not the exact 33 cancers")
    for cancer in CANCER_ORDER:
        validate_id_row(explicit_rows[cancer], r1_rows[cancer])

    detected_sets = {
        cancer: set(explicit_rows[cancer]["detected_lncRNA_ids"])
        for cancer in CANCER_ORDER
    }
    detected_union = set().union(*detected_sets.values())
    detected_intersection = set.intersection(*detected_sets.values())
    global_sets = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_GLOBAL_DETECTED_ID_SETS_V1",
        "scope": "ALL_33_CANCERS_REGARDLESS_OF_FORMAL_ELIGIBILITY",
        "cancer_count": 33,
        "covered_cancers": list(CANCER_ORDER),
        "uncovered_cancers": [],
        "annotation_release": "GENCODE v50",
        "mapping_policy": MAPPING_POLICY,
        "detection_value_threshold": "FINITE_VALUE_GT_0",
        "minimum_detected_cell_count": 1,
        "detected_union_count": len(detected_union),
        "detected_union_sha256": set_sha256(detected_union),
        "detected_union_ids": sorted(detected_union),
        "detected_intersection_count": len(detected_intersection),
        "detected_intersection_sha256": set_sha256(detected_intersection),
        "detected_intersection_ids": sorted(detected_intersection),
        "remote_writes_performed": False,
    }

    r1_table = r1_table.set_index("cancer_id")
    manifest = manifest.set_index("cancer_id")
    records: list[dict[str, Any]] = []
    for cancer in CANCER_ORDER:
        prior = r1_table.loc[cancer]
        source = manifest.loc[cancer]
        row = explicit_rows[cancer]
        records.append(
            {
                "cancer_id": cancer,
                "source_dataset_id": str(source.dataset_id),
                "expression_source_tier": str(source.expression_source_tier),
                "source_generation": str(source.source_generation),
                "measurement_scale": str(prior.measurement_scale),
                "raw_h5_file_count": 1,
                "raw_h5_path": str(prior.h5_path),
                "raw_h5_sha256": str(prior.h5_sha256),
                "raw_h5_size_bytes": int(prior.h5_size_bytes),
                "cell_count": int(prior.cell_count),
                "lncRNA_annotation_release": "GENCODE v50",
                "lncRNA_annotation_strict_stable_id_count": 34_866,
                "lncRNA_id_mapping_policy": MAPPING_POLICY,
                "detection_value_threshold": "FINITE_VALUE_GT_0",
                "minimum_detected_cell_count": 1,
                "lncRNA_universe_count": int(prior.lncRNA_universe_count),
                "detected_lncRNA_count": int(prior.detected_lncRNA_count),
                "lncRNA_universe_set_sha256": str(
                    row["lncRNA_universe_set_sha256"]
                ),
                "detected_lncRNA_set_sha256": str(
                    row["detected_lncRNA_set_sha256"]
                ),
                "formal_eligible": bool(prior.formal_eligible),
                "failure_or_limitation_reason": str(
                    prior.failure_or_limitation_reason
                ),
                "h5_coverage_status": "COVERED_RAW_H5_READ_ONLY_SCANNED",
            }
        )
    table = pd.DataFrame(records)
    uncovered = sorted(EXPECTED_CANCERS - set(table.cancer_id))
    if uncovered or len(table) != 33 or int(table.raw_h5_file_count.sum()) != 33:
        raise ExplicitIdAuditError(f"Unexpected H5 coverage gap: {uncovered}")

    summary = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_EXPLICIT_ID_REAUDIT_SUMMARY_V1",
        "status": "SUCCESS_CODE_BOUND_33_CANCER_EXPLICIT_ID_REAUDIT",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cancer_count": 33,
        "covered_raw_h5_cancers": 33,
        "raw_h5_file_count": 33,
        "uncovered_cancers": uncovered,
        "total_cells": int(table.cell_count.sum()),
        "annotation_strict_lncRNA_count": 34_866,
        "formal_eligible_cancers": int(table.formal_eligible.sum()),
        "detected_union_count": len(detected_union),
        "detected_union_sha256": set_sha256(detected_union),
        "detected_intersection_count": len(detected_intersection),
        "detected_intersection_sha256": set_sha256(detected_intersection),
        "r1_count_and_set_hashes_reproduced_cancers": 33,
        "remote_writes_performed": False,
        "existing_artifacts_overwritten": False,
        "release_ready": False,
        "production_deployed": False,
    }

    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    stage.mkdir(parents=False, exist_ok=False)
    try:
        table_path = stage / "single_cell_lncrna_expression_33c_explicit_ids.tsv"
        raw_path = stage / "EXPLICIT_ID_RECOMPUTATION.json"
        global_path = stage / "GLOBAL_DETECTED_ID_SETS.json"
        summary_path = stage / "AUDIT_SUMMARY.json"
        input_path = stage / "INPUT_AUTHORITIES.json"
        methods_path = stage / "AUDIT_METHODS.md"
        table.to_csv(table_path, sep="\t", index=False)
        write_json(raw_path, raw)
        write_json(global_path, global_sets)
        write_json(summary_path, summary)
        inputs = {
            "r1_binding": {
                "path": str(r1_binding_path.resolve()),
                "sha256": sha256_file(r1_binding_path),
            },
            "r1_per_cancer_table": {
                "path": str(r1_table_path.resolve()),
                "sha256": sha256_file(r1_table_path),
            },
            "r1_raw_h5_recomputation": {
                "path": str(r1_raw_path.resolve()),
                "sha256": sha256_file(r1_raw_path),
            },
            "r6_dataset_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "gencode_v50": raw["annotation"],
            "raw_h5_files": [
                {
                    "cancer_id": cancer,
                    "path": str(r1_table.loc[cancer].h5_path),
                    "sha256": str(r1_table.loc[cancer].h5_sha256),
                    "size_bytes": int(r1_table.loc[cancer].h5_size_bytes),
                }
                for cancer in CANCER_ORDER
            ],
            "remote_access": "READ_ONLY_STDIN_PYTHON_AND_STDOUT_JSON",
            "remote_writes_performed": False,
        }
        write_json(input_path, inputs)
        methods_path.write_text(
            "# V3.2 33-cancer explicit lncRNA stable-ID re-audit\n\n"
            "This supplement leaves the completed r1 audit unchanged and performs "
            "a second read-only matrix scan solely to materialize the actual stable-ID "
            "sets needed for global union/intersection. No remote file is written.\n\n"
            "- Annotation: GENCODE v50 strict primary-chromosome lncRNA stable-ID "
            "universe (34,866 IDs).\n"
            "- Mapping: Ensembl version stripping, direct stable-ID match, exact "
            "case-sensitive symbol fallback only when unique in the full lncRNA plus "
            "protein-coding annotation, then stable-ID deduplication.\n"
            "- Detection: at least one finite stored value strictly greater than zero "
            "in at least one cell. Sparse missing entries, explicit zero, negative, "
            "and non-finite values do not count.\n"
            "- Orientation: feature-by-cell CSC is required and revalidated.\n"
            "- Each per-cancer universe/detected count and set SHA-256 must exactly "
            "reproduce the full r1 raw-H5 scan before union/intersection is emitted.\n"
            "- The global union/intersection scope is all 33 H5-covered cancers, "
            "independent of formal downstream eligibility.\n",
            encoding="utf-8",
        )
        artifacts = {
            "per_cancer_table": table_path,
            "explicit_id_recomputation": raw_path,
            "global_detected_id_sets": global_path,
            "summary": summary_path,
            "input_authorities": input_path,
            "methods": methods_path,
        }
        binding = {
            "format": "CC_HHGT_V3_2_SINGLE_CELL_EXPLICIT_ID_REAUDIT_BINDING_V1",
            "status": summary["status"],
            "analysis_version": summary["analysis_version"],
            "remote_writes_performed": False,
            "existing_artifacts_overwritten": False,
            "release_ready": False,
            "production_deployed": False,
            "counts": summary,
            "code": {
                "materializer": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__)),
                },
                "explicit_id_scanner": {
                    "path": str(scanner_path.resolve()),
                    "sha256": sha256_file(scanner_path),
                },
            },
            "artifacts": {
                role: {
                    "path": str(destination / path.name),
                    "sha256": sha256_file(path),
                }
                for role, path in artifacts.items()
            },
            "input_authorities_sha256": canonical_sha256(inputs),
        }
        binding_path = stage / "SINGLE_CELL_EXPLICIT_ID_REAUDIT_BINDING.json"
        write_json(binding_path, binding)
        binding_sha = sha256_file(binding_path)
        write_json(
            stage / "SUCCESS.json",
            {
                "status": summary["status"],
                "binding": binding_path.name,
                "binding_sha256": binding_sha,
                "remote_writes_performed": False,
                "existing_artifacts_overwritten": False,
                "release_ready": False,
                "production_deployed": False,
            },
        )
        stage.replace(destination)
        checkpoint_path.unlink(missing_ok=True)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "output_root": str(destination),
        "binding_sha256": binding_sha,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r1-root",
        type=Path,
        default=REPO / "artifacts" / "single_cell_lncrna_expression_reaudit_20260826_r1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            REPO / "artifacts" /
            "single_cell_h5_33c_partitions_20260826_r6_semantic_correction" /
            "dataset_manifest_33c.parquet"
        ),
    )
    parser.add_argument(
        "--scanner",
        type=Path,
        default=REPO / "scripts" / "audit_v32_single_cell_h5_explicit_ids_readonly.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-host", default="COMPUTE_HOST")
    parser.add_argument("--ssh-executable", default="ssh.exe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize(
        r1_root=args.r1_root,
        manifest_path=args.manifest,
        scanner_path=args.scanner,
        output_root=args.output,
        remote_host=args.remote_host,
        ssh_executable=args.ssh_executable,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
