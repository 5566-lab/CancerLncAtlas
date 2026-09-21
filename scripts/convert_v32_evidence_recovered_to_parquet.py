#!/usr/bin/env python3
"""Convert an independently validated recovered Evidence TSV to Parquet.

The conversion is representation-only.  DuckDB compares the source and
converted tables with ``EXCEPT ALL`` in both directions before a receipt can
pass.  The script never writes into either the historical input tree or the
rematerialization directory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping


FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_RECOVERED_PARQUET_CONVERSION_V1"
REMAT_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_INTERACTION_REMATERIALIZATION_V1"
VALIDATION_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_INTERACTION_INDEPENDENT_VALIDATION_V1"
PASS_STATUS = "PASS_LOSSLESS_PARQUET_CONVERSION"

EMPTY_EVENT_COLUMNS = (
    "evidence_event_id", "lncrna_id", "partner_id", "pathway_id",
    "pathway_family_id", "cancer_id", "source_database", "source_dataset",
    "source_record_id", "pmid", "experiment_family", "relation_type",
    "direction", "is_experimental", "is_predicted", "source_row_sha256",
)


class ConversionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def convert(
    *,
    rematerialization_root: Path,
    validation_root: Path,
    output_root: Path,
    memory_limit: str,
    threads: int,
) -> dict[str, Any]:
    try:
        import duckdb
    except ImportError as exc:
        raise ConversionError("duckdb is required for recovered Evidence conversion") from exc

    remat = rematerialization_root.resolve()
    validation = validation_root.resolve()
    output = output_root.resolve()
    temporary = output.with_name(f".{output.name}.tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"Conversion refuses output reuse: {output}")
    if not remat.is_dir() or remat.is_symlink() or not validation.is_dir() or validation.is_symlink():
        raise ConversionError("Rematerialization/validation root is missing or unsafe")
    manifest_path = remat / "MANIFEST.json"
    validation_path = validation / "VALIDATION.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
    if manifest.get("format") != REMAT_FORMAT or manifest.get("status") != "PASS_REMATERIALIZED_REQUIRES_INDEPENDENT_VALIDATION":
        raise ConversionError("Rematerialization manifest is not admissible")
    if validation_payload.get("format") != VALIDATION_FORMAT or validation_payload.get("status") != "PASS_INDEPENDENT_REMATERIALIZATION_VALIDATION":
        raise ConversionError("Independent validation receipt is not passing")
    if Path(validation_payload.get("rematerialization_root", "")).resolve() != remat:
        raise ConversionError("Independent validation binds a different rematerialization root")
    relation_tsv = remat / "interaction_relation_recovered.tsv.gz"
    event_tsv = remat / "evidence_event_empty_by_contract.tsv"
    relation_sha = sha256_file(relation_tsv)
    event_sha = sha256_file(event_tsv)
    if relation_sha != manifest["outputs"]["interaction_relation"]["sha256"]:
        raise ConversionError("Recovered interaction SHA drift")
    if event_sha != manifest["outputs"]["evidence_event"]["sha256"]:
        raise ConversionError("Empty Evidence event SHA drift")
    if relation_sha != validation_payload["inputs"]["interaction_relation"]["sha256"]:
        raise ConversionError("Independent validation does not bind the recovered interaction")
    if sha256_file(manifest_path) != validation_payload["inputs"]["manifest"]["sha256"]:
        raise ConversionError("Independent validation does not bind the rematerialization manifest")
    with event_tsv.open("r", encoding="utf-8", newline="") as handle:
        event_rows = list(csv.reader(handle, delimiter="\t"))
    if len(event_rows) != 1 or tuple(event_rows[0]) != EMPTY_EVENT_COLUMNS:
        raise ConversionError(
            "Empty Evidence event authority lacks the canonical header-only schema"
        )
    expected_rows = int(manifest["counts"]["rows"])

    temporary.mkdir(parents=True)
    spill = temporary / "duckdb_spill"
    spill.mkdir()
    relation_parquet = temporary / "interaction_relation_recovered.parquet"
    event_parquet = temporary / "evidence_event_empty_by_contract.parquet"
    database = temporary / "conversion.duckdb"
    connection = duckdb.connect(str(database))
    try:
        connection.execute(f"SET memory_limit={quote(memory_limit)}")
        connection.execute(f"SET threads={max(1, int(threads))}")
        connection.execute(f"SET temp_directory={quote(spill)}")
        connection.execute(
            "CREATE TEMP VIEW source_relation AS SELECT * FROM read_csv("
            f"{quote(relation_tsv)}, delim='\\t', header=true, all_varchar=true, "
            "compression='gzip', nullstr='', quote='\"', escape='\"', sample_size=-1)"
        )
        source_rows = int(connection.execute("SELECT count(*) FROM source_relation").fetchone()[0])
        if source_rows != expected_rows:
            raise ConversionError(f"Source row count drift: {source_rows} != {expected_rows}")
        connection.execute(
            f"COPY (SELECT * FROM source_relation) TO {quote(relation_parquet)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        connection.execute(
            f"CREATE TEMP VIEW converted_relation AS SELECT * FROM read_parquet({quote(relation_parquet)})"
        )
        metrics = connection.execute(
            "SELECT count(*) AS rows, count(DISTINCT interaction_id) AS interaction_ids, "
            "count(DISTINCT source_row_id) AS source_row_ids, "
            "sum(CASE WHEN source_row_sha256 IS NULL OR length(source_row_sha256) <> 64 THEN 1 ELSE 0 END) AS bad_row_hashes "
            "FROM converted_relation"
        ).fetchone()
        if tuple(map(int, metrics)) != (expected_rows, expected_rows, expected_rows, 0):
            raise ConversionError(f"Converted identity invariant failed: {metrics}")
        source_minus_converted = int(
            connection.execute(
                "SELECT count(*) FROM (SELECT * FROM source_relation EXCEPT ALL SELECT * FROM converted_relation)"
            ).fetchone()[0]
        )
        converted_minus_source = int(
            connection.execute(
                "SELECT count(*) FROM (SELECT * FROM converted_relation EXCEPT ALL SELECT * FROM source_relation)"
            ).fetchone()[0]
        )
        if source_minus_converted or converted_minus_source:
            raise ConversionError(
                "Parquet conversion changed table semantics: "
                f"source_minus_converted={source_minus_converted}, "
                f"converted_minus_source={converted_minus_source}"
            )
        source_counts = {
            str(database_name): int(count)
            for database_name, count in connection.execute(
                "SELECT source_database, count(*) FROM converted_relation GROUP BY source_database ORDER BY source_database"
            ).fetchall()
        }
        empty_projection = ", ".join(
            f"CAST(NULL AS VARCHAR) AS \"{column}\"" for column in EMPTY_EVENT_COLUMNS
        )
        connection.execute(
            f"COPY (SELECT {empty_projection} WHERE false) TO {quote(event_parquet)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        event_metrics = connection.execute(
            f"SELECT count(*), count(*) FILTER (WHERE evidence_event_id IS NOT NULL) FROM read_parquet({quote(event_parquet)})"
        ).fetchone()
        if tuple(map(int, event_metrics)) != (0, 0):
            raise ConversionError(f"Empty Evidence event parquet is not empty: {event_metrics}")
    finally:
        connection.close()
    if database.exists():
        database.unlink()
    if spill.exists():
        shutil.rmtree(spill)

    generated = datetime.now(timezone.utc).isoformat()
    relation_output = output / relation_parquet.name
    event_output = output / event_parquet.name
    report = {
        "format": FORMAT,
        "status": PASS_STATUS,
        "generated_at_utc": generated,
        "output_root": str(output),
        "formal_training_started": False,
        "production_deployed": False,
        "production_port_8260_touched": False,
        "contract": {
            "independent_validation_required_and_bound": True,
            "all_columns_varchar_preserved": True,
            "source_minus_converted_rows": source_minus_converted,
            "converted_minus_source_rows": converted_minus_source,
            "interaction_id_unique": True,
            "source_row_id_unique": True,
            "empty_event_authority_preserved": True,
            "historical_output_overwritten": False,
        },
        "inputs": {
            "rematerialization_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "independent_validation": {"path": str(validation_path), "sha256": sha256_file(validation_path)},
            "interaction_relation_tsv_gz": {"path": str(relation_tsv), "sha256": relation_sha},
            "evidence_event_tsv": {"path": str(event_tsv), "sha256": event_sha},
        },
        "counts": {
            "rows": expected_rows,
            "unique_interaction_ids": expected_rows,
            "unique_source_row_ids": expected_rows,
            "bad_source_row_sha256": 0,
            "empty_evidence_event_rows": 0,
            "by_source_database": source_counts,
        },
        "outputs": {
            "interaction_relation": {
                "path": str(relation_output), "bytes": int(relation_parquet.stat().st_size),
                "sha256": sha256_file(relation_parquet),
            },
            "evidence_event": {
                "path": str(event_output), "bytes": int(event_parquet.stat().st_size),
                "sha256": sha256_file(event_parquet),
            },
        },
    }
    manifest_output = temporary / "CONVERSION_MANIFEST.json"
    atomic_json(manifest_output, report)
    success = {
        "format": FORMAT, "status": PASS_STATUS, "generated_at_utc": generated,
        "conversion_manifest": {
            "path": str(output / manifest_output.name), "sha256": sha256_file(manifest_output),
        },
        "rows": expected_rows, "formal_training_started": False,
        "production_deployed": False, "production_port_8260_touched": False,
    }
    atomic_json(temporary / "SUCCESS.json", success)
    os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rematerialization-root", required=True, type=Path)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--memory-limit", default="64GB")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    report = convert(
        rematerialization_root=args.rematerialization_root,
        validation_root=args.validation_root,
        output_root=args.output_root,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    print(json.dumps({"status": report["status"], "output_root": report["output_root"], "counts": report["counts"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
