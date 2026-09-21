#!/usr/bin/env python3
"""Independent byte/schema/row/order verification for the ACC BH fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R9_LARGE_EQUIVALENCE_INDEPENDENT_RECEIPT_V1"
EXPECTED_COMPARTMENTS = {"malignant": 0, "immune": 1, "stromal": 2}


class EquivalenceVerificationError(RuntimeError):
    """Raised when the r8/r9 equivalence fixture is not exactly equivalent."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise EquivalenceVerificationError(f"absent or unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EquivalenceVerificationError(f"absent or unsafe JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EquivalenceVerificationError(f"JSON root is not an object: {path}")
    return value


def audit_parquet(path: Path, expected_rows: int) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    try:
        rows = int(parquet.metadata.num_rows)
        row_groups = int(parquet.metadata.num_row_groups)
        schema = parquet.schema_arrow.remove_metadata()
        schema_sha = hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()
        columns = [
            "compartment", "bh_q_global_tests", "nominal_p", "lncrna_id",
            "pathway_id", "fdr_0_10_pass", "cell_as_independent_replicate",
        ]
        if not set(columns).issubset(schema.names):
            raise EquivalenceVerificationError(f"required schema missing: {path}")
        previous: tuple[Any, ...] | None = None
        compartment_rows = {key: 0 for key in EXPECTED_COMPARTMENTS}
        fdr_rows = {key: 0 for key in EXPECTED_COMPARTMENTS}
        for batch in parquet.iter_batches(batch_size=16_384, columns=columns):
            values = [column.to_pylist() for column in batch.columns]
            for row in zip(*values, strict=True):
                compartment = str(row[0])
                if compartment not in EXPECTED_COMPARTMENTS:
                    raise EquivalenceVerificationError(
                        f"unexpected compartment {compartment}: {path}"
                    )
                q_value = float(row[1])
                nominal_p = float(row[2])
                if (
                    not math.isfinite(q_value)
                    or not math.isfinite(nominal_p)
                    or not 0.0 <= q_value <= 1.0
                    or not 0.0 <= nominal_p <= 1.0
                ):
                    raise EquivalenceVerificationError(f"invalid p/q value: {path}")
                key = (
                    EXPECTED_COMPARTMENTS[compartment], q_value, nominal_p,
                    str(row[3]), str(row[4]),
                )
                if previous is not None and key < previous:
                    raise EquivalenceVerificationError(f"sort-order drift: {path}")
                previous = key
                if bool(row[5]) != (q_value <= 0.10):
                    raise EquivalenceVerificationError(f"FDR flag drift: {path}")
                if bool(row[6]):
                    raise EquivalenceVerificationError(
                        f"cell used as independent replicate: {path}"
                    )
                compartment_rows[compartment] += 1
                fdr_rows[compartment] += int(bool(row[5]))
    finally:
        parquet.close()
    if rows != int(expected_rows):
        raise EquivalenceVerificationError(f"row drift: {rows} != {expected_rows}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "rows": rows,
        "row_groups": row_groups,
        "schema_sha256": schema_sha,
        "sorted_by_compartment_q_p_lncrna_pathway": True,
        "compartment_rows": compartment_rows,
        "fdr_0_10_pass_rows": fdr_rows,
    }


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r8-evidence", required=True, type=Path)
    parser.add_argument("--r9-evidence", required=True, type=Path)
    parser.add_argument("--r8-result", required=True, type=Path)
    parser.add_argument("--r9-result", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--audit-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.audit_root.resolve()
    if root.exists() or root.is_symlink():
        raise EquivalenceVerificationError(f"audit root reuse forbidden: {root}")
    if args.expected_rows <= 0:
        raise EquivalenceVerificationError("expected rows must be positive")
    r8_path = args.r8_evidence.resolve()
    r9_path = args.r9_evidence.resolve()
    r8_result_path = args.r8_result.resolve()
    r9_result_path = args.r9_result.resolve()
    r8_result = load_json(r8_result_path)
    r9_result = load_json(r9_result_path)
    for label, result in (("r8", r8_result), ("r9", r9_result)):
        if int(result.get("rows", -1)) != int(args.expected_rows):
            raise EquivalenceVerificationError(f"{label} result row drift")
    r8 = audit_parquet(r8_path, args.expected_rows)
    r9 = audit_parquet(r9_path, args.expected_rows)
    if r8["sha256"] != r9["sha256"] or r8["bytes"] != r9["bytes"]:
        raise EquivalenceVerificationError("r8/r9 evidence is not byte-identical")
    if r8["schema_sha256"] != r9["schema_sha256"]:
        raise EquivalenceVerificationError("r8/r9 evidence schema differs")
    if r8["compartment_rows"] != r9["compartment_rows"]:
        raise EquivalenceVerificationError("r8/r9 compartment row counts differ")
    if r8["fdr_0_10_pass_rows"] != r9["fdr_0_10_pass_rows"]:
        raise EquivalenceVerificationError("r8/r9 FDR-pass counts differ")
    root.mkdir(parents=True)
    value = {
        "format": FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "PASS_EXACT_LARGE_FIXTURE_EQUIVALENCE",
        "expected_rows": int(args.expected_rows),
        "byte_identical": True,
        "schema_identical": True,
        "row_count_identical": True,
        "sort_order_independently_verified_for_both": True,
        "compartment_counts_identical": True,
        "fdr_pass_counts_identical": True,
        "r8": r8,
        "r9": r9,
        "r8_result_path": str(r8_result_path),
        "r8_result_sha256": sha256_file(r8_result_path),
        "r9_result_path": str(r9_result_path),
        "r9_result_sha256": sha256_file(r9_result_path),
        "production_deployed": False,
    }
    value["receipt_contract_sha256"] = canonical_sha256(value)
    report_path = root / "AUDIT.json"
    exclusive_json(report_path, value)
    print(
        json.dumps(
            {**value, "audit_path": str(report_path), "audit_sha256": sha256_file(report_path)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
