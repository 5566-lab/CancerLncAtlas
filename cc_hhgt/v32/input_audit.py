from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds

from .contracts import canonical_json_sha256, file_sha256


def _inspect_path(path: Path, required_columns: Sequence[str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "required_columns": list(required_columns),
    }
    if not path.exists():
        record.update({"status": "MISSING", "missing_columns": list(required_columns)})
        return record
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": file_sha256(path)})
    try:
        suffix = "".join(path.suffixes).lower()
        if path.is_dir() or suffix.endswith(".parquet"):
            dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
            columns = list(dataset.schema.names)
            record.update(
                {
                    "columns": columns,
                    "rows": int(dataset.count_rows()),
                    "missing_columns": sorted(set(required_columns) - set(columns)),
                }
            )
        else:
            record["missing_columns"] = []
    except Exception as exc:  # audit must report rather than mutate or repair
        record.update({"schema_error": f"{type(exc).__name__}: {exc}"})
        record.setdefault("missing_columns", list(required_columns))
    record["status"] = "PASS" if not record.get("missing_columns") and not record.get("schema_error") else "FAIL"
    return record


def audit_inputs(
    inputs: Mapping[str, str | Path],
    required_columns: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Read-only path/schema inventory.

    This function never creates directories, rewrites input files, or repairs
    schemas.  It is safe to run during the CODE_ONLY phase.
    """

    requirements = required_columns or {}
    records = {
        name: _inspect_path(Path(value), requirements.get(name, ()))
        for name, value in sorted(inputs.items())
    }
    payload = {
        "audit_mode": "READ_ONLY",
        "input_count": len(records),
        "status": "PASS" if records and all(r["status"] == "PASS" for r in records.values()) else "FAIL",
        "inputs": records,
    }
    payload["audit_sha256"] = canonical_json_sha256(payload)
    return payload
