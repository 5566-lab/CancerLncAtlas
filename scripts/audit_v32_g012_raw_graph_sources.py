#!/usr/bin/env python3
"""Read-only schema/content audit for proposed V3.2 G0/G1/G2 graph sources.

This utility never writes beside an input.  It records Parquet metadata,
column names/types, a small JSON-safe sample, null counts, and bounded value
counts so that a fresh authority materializer can be written against observed
raw schemas instead of historical prepared graph rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _audit(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(f"Unsafe graph source: {resolved}")
    parquet = pq.ParquetFile(resolved)
    frame = pd.read_parquet(resolved)
    if len(frame) != parquet.metadata.num_rows:
        raise RuntimeError(f"Parquet row-count drift: {resolved}")
    value_counts: dict[str, Any] = {}
    for column in frame.columns:
        distinct = int(frame[column].nunique(dropna=False))
        if distinct <= 30:
            counts = frame[column].value_counts(dropna=False).head(30)
            value_counts[str(column)] = [
                {"value": _json_value(key), "rows": int(value)}
                for key, value in counts.items()
            ]
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
        "rows": int(len(frame)),
        "row_groups": int(parquet.metadata.num_row_groups),
        "schema": [
            {"column": field.name, "arrow_type": str(field.type)}
            for field in parquet.schema_arrow
        ],
        "null_counts": {str(column): int(frame[column].isna().sum()) for column in frame},
        "distinct_counts": {
            str(column): int(frame[column].nunique(dropna=False)) for column in frame
        },
        "bounded_value_counts": value_counts,
        "head": [
            {str(column): _json_value(value) for column, value in row.items()}
            for row in frame.head(5).to_dict(orient="records")
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    payload = {
        "format": "CANCERLNCATLAS_V32_G012_RAW_SOURCE_AUDIT_V1",
        "status": "PASS_READ_ONLY_SCHEMA_AUDIT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [_audit(path) for path in args.source],
        "gates": {
            "input_writes": False,
            "historical_prepared_graph_rows_used": False,
            "raw_sources_only": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
