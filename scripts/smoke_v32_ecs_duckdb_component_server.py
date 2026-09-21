#!/usr/bin/env python3
"""Pandas-free, binding-pinned DuckDB smokes for Evidence/Clinical/State."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import duckdb


FORMAT = "CANCERLNCATLAS_V32_ECS_DUCKDB_RUNTIME_QUERY_SMOKE_V1"
RUNTIME_FORMAT = "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_RUNTIME_BINDINGS_V1"
ALLOWED_ROOT = Path("./data/CancerLncAtlas")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allowed_file(path_value: str, label: str) -> Path:
    path = Path(path_value).resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), f"Missing or symlinked {label}: {path}")
    require(ALLOWED_ROOT in path.parents, f"{label} escaped ${PRIVATE_WORK_ROOT} authority: {path}")
    return path


def load_pinned(path: Path, expected: str, label: str) -> dict[str, Any]:
    expected = expected.lower()
    require(SHA256.fullmatch(expected) is not None, f"Invalid {label} SHA256")
    path = allowed_file(str(path), label)
    observed = sha256_file(path)
    require(observed == expected, f"{label} SHA256 mismatch: {observed}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path = path.absolute()
    require(not path.exists(), f"Refusing to overwrite output: {path}")
    require(ALLOWED_ROOT in path.parents, "Output escaped ${PRIVATE_WORK_ROOT} authority")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)


def quoted(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def one_row(connection: duckdb.DuckDBPyConnection, path: Path, where: str) -> dict[str, Any]:
    cursor = connection.execute(f"SELECT * FROM read_parquet({quoted(path)}) WHERE {where} LIMIT 1")
    row = cursor.fetchone()
    require(row is not None, f"No row matched: {where}")
    columns = [item[0] for item in cursor.description]
    return dict(zip(columns, row, strict=True))


def artifact_path(binding: Mapping[str, Any], declaration: Mapping[str, Any], receipt_rows: Mapping[str, Mapping[str, Any]], label: str) -> Path:
    path = allowed_file(str(declaration.get("path", "")), label)
    expected_sha = str(declaration.get("sha256", "")).lower()
    require(SHA256.fullmatch(expected_sha) is not None, f"Invalid {label} artifact SHA")
    receipt = receipt_rows.get(str(path))
    require(isinstance(receipt, Mapping), f"{label} is absent from the rehashed quarantine receipt")
    require(receipt.get("sha256") == expected_sha, f"{label} SHA differs from quarantine receipt")
    require(receipt.get("canonical_rehashed_equal") is True and receipt.get("partial_rehashed_equal") is True, f"{label} was not byte-for-byte rehashed")
    return path


def evidence_smoke(runtime: Mapping[str, Any], receipt_rows: Mapping[str, Mapping[str, Any]], connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    declaration = runtime["bindings"]["evidence"]
    binding = load_pinned(Path(declaration["binding_path"]), declaration["binding_sha256"], "Evidence binding")
    require(binding.get("release_ready") is False and binding.get("production_deployed") is False, "Evidence release flags drifted")
    predictions = artifact_path(binding, binding["artifacts"]["evidence_predictions"], receipt_rows, "Evidence predictions")
    available = one_row(connection, predictions, "availability=true")
    unavailable = one_row(connection, predictions, "availability=false")
    require(available.get("availability") is True and available.get("evidence_confidence_probability") is not None, "Evidence available semantics failed")
    require(unavailable.get("availability") is False and unavailable.get("evidence_confidence_probability") is None and bool(unavailable.get("failure_reason")), "Evidence unavailable semantics failed")

    direction_decl = runtime["bindings"]["evidence_direction"]
    direction_binding = load_pinned(Path(direction_decl["binding_path"]), direction_decl["binding_sha256"], "Direction binding")
    load_pinned(Path(direction_decl["audit_binding_path"]), direction_decl["audit_binding_sha256"], "Direction audit binding")
    require(direction_binding.get("release_ready") is False and direction_binding.get("production_deployed") is False, "Direction release flags drifted")
    direction_path = artifact_path(direction_binding, direction_binding["artifact"], receipt_rows, "Direction probabilities")
    direction_available = one_row(connection, direction_path, "direction_probability_available=true")
    direction_unavailable = one_row(connection, direction_path, "direction_probability_available=false")
    probability_columns = ("direction_negative_probability", "direction_neutral_probability", "direction_positive_probability")
    require(all(direction_available.get(name) is not None for name in probability_columns), "Direction available probabilities contain NULL")
    require(all(direction_unavailable.get(name) is None for name in probability_columns), "Direction unavailable probabilities are not NULL")
    require(bool(direction_unavailable.get("direction_probability_unavailable_reason")), "Direction unavailable reason is empty")
    return {"component": "evidence", "binding": declaration, "direction_binding": direction_decl, "probes": {"confidence_available": available, "confidence_unavailable": unavailable, "direction_available": direction_available, "direction_unavailable": direction_unavailable}, "typed_available_verified": True, "typed_unavailable_verified": True, "scientific_status": "partial_not_publishable"}


def clinical_smoke(runtime: Mapping[str, Any], receipt_rows: Mapping[str, Mapping[str, Any]], connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    declaration = runtime["bindings"]["clinical"]
    binding = load_pinned(Path(declaration["binding_path"]), declaration["binding_sha256"], "Clinical binding")
    require(binding.get("release_ready") is False and binding.get("production_deployed") is False, "Clinical release flags drifted")
    statistics = artifact_path(binding, binding["artifacts"]["statistics"], receipt_rows, "Clinical statistics")
    available = one_row(connection, statistics, "availability=true")
    unavailable = one_row(connection, statistics, "availability=false")
    require(available.get("availability") is True and not available.get("failure_reason"), "Clinical available semantics failed")
    require(unavailable.get("availability") is False and bool(unavailable.get("failure_reason")), "Clinical unavailable semantics failed")
    return {"component": "clinical", "binding": declaration, "clean_directory_runtime_view_bound": bool(declaration.get("clean_directory_runtime_view_bound")), "probes": {"available": available, "unavailable": unavailable}, "typed_available_verified": True, "typed_unavailable_verified": True, "scientific_status": "secondary_fresh_lncrna_survival_statistics"}


def state_smoke(runtime: Mapping[str, Any], receipt_rows: Mapping[str, Mapping[str, Any]], connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    declaration = runtime["bindings"]["state_gene_set"]
    binding = load_pinned(Path(declaration["binding_path"]), declaration["binding_sha256"], "State binding")
    require(binding.get("release_ready") is False and binding.get("production_deployed") is False, "State release flags drifted")
    catalog = artifact_path(binding, binding["artifacts"]["catalog"], receipt_rows, "State catalog")
    members = artifact_path(binding, binding["artifacts"]["members"], receipt_rows, "State members")
    rnass = one_row(connection, catalog, "state_id='stemness_rna::RNAss'")
    dnass = one_row(connection, catalog, "state_id='stemness_dna::DNAss'")
    gene_set_id = str(rnass["gene_set_id"]).replace("'", "''")
    member = one_row(connection, members, f"gene_set_id='{gene_set_id}'")
    return {"component": "state_gene_set", "binding": declaration, "probes": {"RNAss": rnass, "DNAss": dnass, "member": member}, "typed_available_unavailable": "NOT_APPLICABLE_NO_AVAILABILITY_FIELD_IN_STATE_GENE_SET_QUERY_CONTRACT", "scientific_status": "secondary_state_gene_set_and_report"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("evidence", "clinical", "state_gene_set"), required=True)
    parser.add_argument("--runtime-bindings", type=Path, required=True)
    parser.add_argument("--runtime-bindings-sha256", required=True)
    parser.add_argument("--quarantine-receipt", type=Path, required=True)
    parser.add_argument("--quarantine-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.absolute().exists(), f"Refusing to overwrite output: {args.output}")
    base = {"format": FORMAT, "component": args.component, "status": "FAIL", "query_engine": "DUCKDB_FETCHONE_NO_PANDAS_BINDING_COMPATIBILITY_SMOKE", "main_score_changed": False, "production_port_8260_touched": False, "production_deployed": False, "release_ready": False}
    try:
        require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE must be exactly 1")
        require(not any(item.startswith("${PRIVATE_WORK_ROOT}/") for item in sys.path), "Unauthorized /dsk2 sys.path entry")
        runtime = load_pinned(args.runtime_bindings, args.runtime_bindings_sha256, "runtime bindings")
        require(runtime.get("format") == RUNTIME_FORMAT and runtime.get("release_ready") is False and runtime.get("production_deployed") is False, "Runtime binding contract drifted")
        quarantine = load_pinned(args.quarantine_receipt, args.quarantine_receipt_sha256, "quarantine receipt")
        require(quarantine.get("status") == "PASS" and quarantine.get("validated_files") == 59 and quarantine.get("moved_files") == 59, "Quarantine receipt is not complete PASS")
        receipt_rows = {str(Path(row["canonical_path"]).resolve()): row for row in quarantine["records"]}
        connection = duckdb.connect(":memory:")
        connection.execute("SET threads=1")
        runner = {"evidence": evidence_smoke, "clinical": clinical_smoke, "state_gene_set": state_smoke}[args.component]
        base["result"] = runner(runtime, receipt_rows, connection)
        base["runtime_identity"] = {"python": {"executable": sys.executable, "version": sys.version}, "duckdb": {"version": duckdb.__version__, "path": str(Path(duckdb.__file__).resolve())}, "python_no_user_site": True, "unauthorized_dsk2_sys_path_entries": []}
        require(str(Path(duckdb.__file__).resolve()).startswith(str(ALLOWED_ROOT) + "/"), "DuckDB loaded outside ${PRIVATE_WORK_ROOT}")
        base["status"] = "PASS"
    except Exception as error:
        base["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        write_exclusive(args.output, base)
        raise
    write_exclusive(args.output, base)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
