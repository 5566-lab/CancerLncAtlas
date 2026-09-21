#!/usr/bin/env python3
"""Materialize a code-bound 33-cancer single-cell lncRNA re-audit."""
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
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
EXPECTED_CANCERS = {
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
}
CANCER_ORDER = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)
AUDIT_TSV_SHA256 = "61f48ba6de3436877c90fe8ea0663a6c0477fdae24ac7c0f8e2f228b115bec0b"
GATE_SHA256 = "cbf88cfabaf9b7719cd3514e1896c34c695f5861c40143ed747089d4ec18ec00"
RUN_STATUS_SHA256 = "3a9ec28a6168f1e38d65521d9e2d7fff818becef6f6d74a3dd9846c1715c0423"
MANIFEST_SHA256 = "bea2d20a261063dacdd05b3b18865f8de0fec3504086e08e6295b63fe0a4ffad"


class ReauditError(RuntimeError):
    """Raised when the immutable re-audit cannot prove its inputs."""


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ReauditError(f"Missing or unsafe file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReauditError(f"JSON is not an object: {path}")
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
        raise ReauditError(f"{label} SHA256 drift: {observed} != {expected}")
    return observed


def run_raw_h5_scan(
    scanner: Path,
    host: str,
    ssh_executable: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    source = scanner.read_text(encoding="utf-8")
    collected: dict[str, dict[str, Any]] = {}
    annotation: dict[str, Any] | None = None
    scanner_sha256 = sha256_file(scanner)
    if checkpoint_path.exists():
        checkpoint = load_json(checkpoint_path)
        checkpoint_rows = checkpoint.get("per_cancer", [])
        if (
            checkpoint.get("scanner_sha256") != scanner_sha256
            or checkpoint.get("remote_writes_performed") is not False
            or not isinstance(checkpoint_rows, list)
        ):
            raise ReauditError(
                f"Unsafe or stale raw-H5 checkpoint: {checkpoint_path}"
            )
        for row in checkpoint_rows:
            if not isinstance(row, dict):
                raise ReauditError(f"Malformed raw-H5 checkpoint: {checkpoint_path}")
            cancer = str(row.get("cancer_id", ""))
            if cancer not in CANCER_ORDER or cancer in collected:
                raise ReauditError(
                    f"Unexpected checkpoint cancer row: {cancer or '<missing>'}"
                )
            collected[cancer] = row
        checkpoint_annotation = checkpoint.get("annotation")
        if checkpoint_annotation is not None and not isinstance(
            checkpoint_annotation, dict
        ):
            raise ReauditError(f"Malformed checkpoint annotation: {checkpoint_path}")
        annotation = checkpoint_annotation
        print(
            f"LOCAL_CHECKPOINT_LOADED\t{len(collected)}/33",
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
                    raise ReauditError("GENCODE annotation changed across retry connections")
                annotation = current
            elif line.startswith("CANCER_JSON\t"):
                row = json.loads(line.split("\t", 1)[1])
                cancer = str(row.get("cancer_id", ""))
                if cancer not in remaining or cancer in collected:
                    process.kill()
                    raise ReauditError(f"Unexpected streamed cancer row: {cancer}")
                collected[cancer] = row
                write_json(
                    checkpoint_path,
                    {
                        "scanner_sha256": sha256_file(scanner),
                        "annotation": annotation,
                        "per_cancer": [collected[key] for key in CANCER_ORDER if key in collected],
                        "remote_writes_performed": False,
                    },
                )
                print(
                    f"LOCAL_CHECKPOINT\t{len(collected)}/33\t{cancer}",
                    file=sys.stderr,
                    flush=True,
                )
        return_code = process.wait()
        if len(collected) == before:
            consecutive_empty_failures += 1
        else:
            consecutive_empty_failures = 0
        if return_code and len(collected) < len(CANCER_ORDER):
            print(
                f"REMOTE_RETRY\texit={return_code}\tcompleted={len(collected)}/33",
                file=sys.stderr,
                flush=True,
            )
        if consecutive_empty_failures >= 3:
            raise ReauditError(
                f"Raw H5 scanner made no progress in three retries; completed={len(collected)}"
            )
    if annotation is None:
        raise ReauditError("Raw H5 scanner did not return annotation provenance")
    payload = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_RAW_H5_READ_ONLY_REAUDIT_V1",
        "remote_host_role": "COMPUTE_HOST_READ_ONLY",
        "remote_writes_performed": False,
        "cancers": list(CANCER_ORDER),
        "annotation": annotation,
        "per_cancer": [collected[cancer] for cancer in CANCER_ORDER],
    }
    return payload


def limitation_reason(status: dict[str, Any], source_tier: str) -> str:
    reasons: list[str] = []
    blocking = status.get("blocking_reason")
    if blocking:
        reasons.append(str(blocking))
    if source_tier == "source_normalized":
        reasons.append("SOURCE_NORMALIZED_NONCOUNT_MEASUREMENT")
    if source_tier == "legacy_counts":
        reasons.append("LEGACY_SEURAT_EXPORT_UNVERIFIED_COUNTS")
    return ";".join(dict.fromkeys(reasons)) if reasons else "NONE"


def materialize(
    *,
    audit_tsv: Path,
    audit_summary_path: Path,
    gate_path: Path,
    run_status_path: Path,
    manifest_path: Path,
    scanner_path: Path,
    output_root: Path,
    remote_host: str,
    ssh_executable: str,
) -> dict[str, Any]:
    destination = output_root.resolve()
    if destination.exists():
        raise ReauditError(f"Refusing to overwrite re-audit: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    require_hash(audit_tsv, AUDIT_TSV_SHA256, "existing expression audit")
    require_hash(gate_path, GATE_SHA256, "current formal gate")
    require_hash(run_status_path, RUN_STATUS_SHA256, "current run status")
    require_hash(manifest_path, MANIFEST_SHA256, "current dataset manifest")
    gate = load_json(gate_path)
    run = load_json(run_status_path)
    old_summary = load_json(audit_summary_path)
    if (
        gate.get("format") != "CC_HHGT_V3_2_SINGLE_CELL_FORMAL_33C_INPUT_GATE_V2"
        or gate.get("run_status_sha256") != RUN_STATUS_SHA256
        or gate.get("dataset_manifest_sha256") != MANIFEST_SHA256
        or gate.get("scope_is_full_33") is not True
        or run.get("scope_is_full_33") is not True
    ):
        raise ReauditError("Current gate/status authority is inconsistent")
    old = pd.read_csv(audit_tsv, sep="\t", dtype={"cancer_id": "string"})
    manifest = pd.read_parquet(manifest_path)
    if (
        len(old) != 33
        or set(old.cancer_id.astype(str)) != EXPECTED_CANCERS
        or old.cancer_id.duplicated().any()
        or len(manifest) != 33
        or set(manifest.cancer_id.astype(str)) != EXPECTED_CANCERS
        or manifest.cancer_id.duplicated().any()
        or set(run.get("per_cancer", {})) != EXPECTED_CANCERS
    ):
        raise ReauditError("One or more 33-cancer authorities have scope drift")

    checkpoint_path = destination.parent / f".{destination.name}.raw-h5-checkpoint.json"
    raw = run_raw_h5_scan(
        scanner_path, remote_host, ssh_executable, checkpoint_path
    )
    remote_rows = pd.DataFrame(raw.get("per_cancer", []))
    if (
        len(remote_rows) != 33
        or set(remote_rows.cancer_id.astype(str)) != EXPECTED_CANCERS
        or remote_rows.cancer_id.duplicated().any()
        or raw.get("annotation", {}).get("strict_primary_lncRNA_count") != 34_866
        or raw.get("annotation", {}).get(
            "full_and_long_strict_stable_id_sets_equal"
        ) is not True
    ):
        raise ReauditError("Raw-H5 recomputation did not return the exact 33-cancer scope")

    old = old.set_index("cancer_id")
    remote_rows = remote_rows.set_index("cancer_id")
    manifest = manifest.set_index("cancer_id")
    records: list[dict[str, Any]] = []
    for cancer in sorted(EXPECTED_CANCERS):
        prior = old.loc[cancer]
        scanned = remote_rows.loc[cancer]
        status = run["per_cancer"][cancer]
        manifest_row = manifest.loc[cancer]
        built = str(status.get("status", "")).startswith("SUCCESS_PARTITION_BUILT")
        old_count_match = (
            int(scanned.audit_policy_lncRNA_universe_count)
            == int(prior.canonical_lnc_feature_unique)
        )
        old_detected_match = (
            int(scanned.audit_policy_detected_lncRNA_count)
            == int(prior.canonical_lnc_detected_ge1_cell)
        )
        cell_count_match = int(scanned.cell_count) == int(prior.cell_count)
        h5_hash_match = str(scanned.h5_sha256) == str(status.get("h5_sha256"))
        fresh_expected = status.get("fresh_direct_id_lncrna_feature_count")
        formal_feature_count_match = (
            int(scanned.formal_policy_lncRNA_universe_count) == int(fresh_expected)
            if built and fresh_expected is not None
            else pd.NA
        )
        formal_eligible = bool(status.get("formal_eligible", False))
        if formal_eligible != (cancer in set(gate["formal_eligible_cancers"])):
            raise ReauditError(f"Gate/status formal eligibility mismatch: {cancer}")
        records.append(
            {
                "cancer_id": cancer,
                "cell_count": int(scanned.cell_count),
                "annotation_strict_lncRNA_count": 34_866,
                "lncRNA_universe_count": int(
                    scanned.formal_policy_lncRNA_universe_count
                ),
                "detected_lncRNA_count": int(
                    scanned.formal_policy_detected_lncRNA_count
                ),
                "undetected_lncRNA_count": int(
                    scanned.formal_policy_undetected_lncRNA_count
                ),
                "formal_eligible": formal_eligible,
                "partition_status": str(status.get("status")),
                "failure_or_limitation_reason": limitation_reason(
                    status, str(prior.source_tier)
                ),
                "downstream_candidate_lncrna_count_in_h5": (
                    status.get("candidate_lncrnas_in_h5") if built else pd.NA
                ),
                "source_tier": str(prior.source_tier),
                "measurement_scale": str(prior.measurement_scale),
                "old_audit_lncRNA_universe_count": int(
                    prior.canonical_lnc_feature_unique
                ),
                "old_audit_detected_lncRNA_count": int(
                    prior.canonical_lnc_detected_ge1_cell
                ),
                "formal_minus_old_audit_universe_count": int(
                    scanned.formal_minus_audit_universe_count
                ),
                "formal_minus_old_audit_detected_count": int(
                    scanned.formal_minus_audit_detected_count
                ),
                "old_audit_universe_reproduced": bool(old_count_match),
                "old_audit_detected_reproduced": bool(old_detected_match),
                "formal_partition_feature_count_reproduced": formal_feature_count_match,
                "cell_count_reproduced": bool(cell_count_match),
                "h5_sha256_reproduced": bool(h5_hash_match),
                "matrix_orientation": str(scanned.matrix_orientation),
                "matrix_orientation_validated": bool(
                    scanned.matrix_orientation_validated
                ),
                "feature_count": int(scanned.feature_count),
                "stored_nnz": int(scanned.stored_nnz),
                "stored_zero_entries": int(scanned.stored_zero_entries),
                "stored_negative_entries": int(scanned.stored_negative_entries),
                "stored_nonfinite_entries": int(scanned.stored_nonfinite_entries),
                "matrix_data_dtype": str(scanned.matrix_data_dtype),
                "h5_path": str(scanned.h5_path),
                "h5_sha256": str(scanned.h5_sha256),
                "h5_size_bytes": int(scanned.h5_size_bytes),
                "formal_universe_set_sha256": str(
                    scanned.formal_policy_universe_sha256
                ),
                "formal_detected_set_sha256": str(
                    scanned.formal_policy_detected_sha256
                ),
                "old_quality_flags": (
                    "" if pd.isna(prior.quality_flags) else str(prior.quality_flags)
                ),
            }
        )
    table = pd.DataFrame(records)
    for column in (
        "downstream_candidate_lncrna_count_in_h5",
        "formal_partition_feature_count_reproduced",
    ):
        table[column] = table[column].astype(
            "Int64" if column.endswith("count_in_h5") else "boolean"
        )
    if (
        not table.cell_count_reproduced.all()
        or not table.h5_sha256_reproduced.all()
        or not table.matrix_orientation_validated.all()
    ):
        raise ReauditError("Raw H5 authority drift or sparse orientation failure")

    built_mask = table.partition_status.str.startswith("SUCCESS_PARTITION_BUILT")
    summary = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_LNCRNA_EXPRESSION_REAUDIT_SUMMARY_V1",
        "audit_strength": (
            "RAW_H5_READ_ONLY_RECOMPUTATION_PLUS_CODE_MANIFEST_GATE_RECONCILIATION"
        ),
        "status": "SUCCESS_CODE_BOUND_33_CANCER_REAUDIT",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cancer_count": 33,
        "total_cells": int(table.cell_count.sum()),
        "annotation_strict_lncRNA_count": 34_866,
        "detected_lncRNA_count_min": int(table.detected_lncRNA_count.min()),
        "detected_lncRNA_count_median": float(table.detected_lncRNA_count.median()),
        "detected_lncRNA_count_max": int(table.detected_lncRNA_count.max()),
        "formal_eligible_cancers": int(table.formal_eligible.sum()),
        "metadata_blocked_cancers": int(
            table.partition_status.eq("BLOCKED_MISSING_CELL_METADATA").sum()
        ),
        "limited_feature_universe_cancers": int(
            table.failure_or_limitation_reason.str.contains(
                "LIMITED_LNCRNA_FEATURE_UNIVERSE", regex=False
            ).sum()
        ),
        "insufficient_replication_cancers": int(
            table.failure_or_limitation_reason.str.contains(
                "NO_CONTEXT_WITH_MINIMUM_DONOR_REPLICATION", regex=False
            ).sum()
        ),
        "source_normalized_cancers": int(
            table.source_tier.eq("source_normalized").sum()
        ),
        "old_audit_universe_exactly_reproduced_cancers": int(
            table.old_audit_universe_reproduced.sum()
        ),
        "old_audit_detected_exactly_reproduced_cancers": int(
            table.old_audit_detected_reproduced.sum()
        ),
        "formal_partition_feature_count_reproduced_built_cancers": int(
            table.loc[built_mask, "formal_partition_feature_count_reproduced"].sum()
        ),
        "built_partition_cancers": int(built_mask.sum()),
        "h5_sha256_reproduced_cancers": int(table.h5_sha256_reproduced.sum()),
        "matrix_orientation_validated_cancers": int(
            table.matrix_orientation_validated.sum()
        ),
        "remote_writes_performed": False,
        "existing_artifacts_overwritten": False,
        "release_ready": False,
        "production_deployed": False,
    }

    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    stage.mkdir(parents=False, exist_ok=False)
    try:
        table_path = stage / "single_cell_lncrna_expression_33c_reaudit.tsv"
        raw_path = stage / "RAW_H5_RECOMPUTATION.json"
        summary_path = stage / "AUDIT_SUMMARY.json"
        input_path = stage / "INPUT_AUTHORITIES.json"
        methods_path = stage / "AUDIT_METHODS.md"
        table.to_csv(table_path, sep="\t", index=False, na_rep="NA")
        write_json(raw_path, raw)
        write_json(summary_path, summary)
        inputs = {
            "existing_expression_audit": {
                "path": str(audit_tsv.resolve()), "sha256": sha256_file(audit_tsv)
            },
            "existing_audit_summary": {
                "path": str(audit_summary_path.resolve()),
                "sha256": sha256_file(audit_summary_path),
                "declared_table_sha256": old_summary.get("artifacts", {})
                .get("per_cancer_tsv", {})
                .get("sha256"),
            },
            "formal_gate": {
                "path": str(gate_path.resolve()), "sha256": sha256_file(gate_path)
            },
            "run_status": {
                "path": str(run_status_path.resolve()),
                "sha256": sha256_file(run_status_path),
            },
            "dataset_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "remote_annotation": raw["annotation"],
            "remote_h5_files": [
                {
                    "cancer_id": row["cancer_id"],
                    "path": row["h5_path"],
                    "sha256": row["h5_sha256"],
                    "size_bytes": row["h5_size_bytes"],
                }
                for row in raw["per_cancer"]
            ],
            "remote_access": "READ_ONLY_STDIN_PYTHON_AND_STDOUT_JSON",
            "remote_writes_performed": False,
        }
        write_json(input_path, inputs)
        methods_path.write_text(
            "# V3.2 33-cancer single-cell lncRNA expression re-audit\n\n"
            "This is a read-only raw-H5 recomputation, not training. No remote file "
            "was created, changed, or deleted.\n\n"
            "- `annotation_strict_lncRNA_count` is the fixed GENCODE v50 strict "
            "primary-chromosome stable-ID universe (34,866).\n"
            "- `lncRNA_universe_count` is the number of those stable IDs present in "
            "the cancer H5 after direct Ensembl matching, version stripping, exact "
            "unique full-annotation symbol fallback, and stable-ID deduplication.\n"
            "- `detected_lncRNA_count` requires at least one finite stored matrix "
            "value strictly greater than zero. Missing sparse entries, explicit zero, "
            "negative, and non-finite values do not count.\n"
            "- Every H5 must be feature-by-cell CSC: `len(indptr) == shape[1] + 1`, "
            "with monotone pointers, matching feature/barcode dimensions, and in-range "
            "feature indices. Any violation fails the audit.\n"
            "- Multiple feature rows mapping to one lncRNA are deduplicated to the "
            "stable ENSG ID. For the >=1-cell endpoint, any positive mapped row is "
            "sufficient and duplicate rows cannot inflate the unique-ID count.\n"
            "- `formal_eligible` and typed reasons come from the hash-bound current r6 "
            "gate/status. They are not inferred from detection alone. KICH, KIRP, and "
            "TGCT fail the five-donor context requirement; CESC, UCS, and UVM have "
            "limited feature universes; ten legacy cancers lack donor/cell-type metadata.\n"
            "- HNSC and LGG are formal-eligible but remain source-normalized, so their "
            "positive detection is valid while magnitudes are not raw-count comparable.\n",
            encoding="utf-8",
        )
        artifacts = {
            "per_cancer_table": table_path,
            "raw_h5_recomputation": raw_path,
            "summary": summary_path,
            "input_authorities": input_path,
            "methods": methods_path,
        }
        binding = {
            "format": "CC_HHGT_V3_2_SINGLE_CELL_LNCRNA_EXPRESSION_REAUDIT_BINDING_V1",
            "status": summary["status"],
            "analysis_version": summary["analysis_version"],
            "audit_strength": summary["audit_strength"],
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
                "raw_h5_scanner": {
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
        binding_path = stage / "SINGLE_CELL_LNCRNA_REAUDIT_BINDING.json"
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
    audit_root = REPO / "artifacts" / "single_cell_lncrna_expression_audit_20260826"
    gate_root = (
        REPO / "artifacts" /
        "single_cell_h5_33c_partitions_20260826_r6_semantic_correction"
    )
    parser.add_argument(
        "--audit-tsv", type=Path,
        default=audit_root / "single_cell_lncrna_expression_33c.tsv"
    )
    parser.add_argument(
        "--audit-summary", type=Path, default=audit_root / "AUDIT_SUMMARY.json"
    )
    parser.add_argument(
        "--gate", type=Path, default=gate_root / "FORMAL_33C_INPUT_GATE.json"
    )
    parser.add_argument(
        "--run-status", type=Path, default=gate_root / "RUN_STATUS.json"
    )
    parser.add_argument(
        "--manifest", type=Path, default=gate_root / "dataset_manifest_33c.parquet"
    )
    parser.add_argument(
        "--scanner", type=Path,
        default=REPO / "scripts" / "audit_v32_single_cell_h5_readonly.py"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-host", default="COMPUTE_HOST")
    parser.add_argument("--ssh-executable", default="ssh.exe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize(
        audit_tsv=args.audit_tsv,
        audit_summary_path=args.audit_summary,
        gate_path=args.gate,
        run_status_path=args.run_status,
        manifest_path=args.manifest,
        scanner_path=args.scanner,
        output_root=args.output,
        remote_host=args.remote_host,
        ssh_executable=args.ssh_executable,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
