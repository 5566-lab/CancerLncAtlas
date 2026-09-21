#!/usr/bin/env python3
"""Bind a freshly computed single-cell extension cohort for independent audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CANCERS = ("BRCA", "CESC", "COAD", "OV", "TGCT", "UVM")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise RuntimeError(f"immutable binding already exists: {path}")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--cancers", nargs="+", default=list(DEFAULT_CANCERS))
    args = parser.parse_args()
    cancers = tuple(str(value).upper() for value in args.cancers)
    if not cancers or len(set(cancers)) != len(cancers):
        raise RuntimeError("extension cancer universe is empty or duplicated")
    root = args.results_root.resolve()
    bindings: list[dict[str, Any]] = []
    for cancer in sorted(cancers):
        output = root / "partitions" / f"cancer_id={cancer}"
        success_path = output / "SUCCESS.json"
        success = load_json(success_path)
        if success.get("status") != "SUCCESS" or success.get("cancer_id") != cancer:
            raise RuntimeError(f"fresh extension SUCCESS drift: {cancer}")
        if success.get("historical_derived_results_used") is not False:
            raise RuntimeError(f"historical derived result used: {cancer}")
        if success.get("source_generation") != "V3.2_R7_FRESH_FROM_RAW_H5":
            raise RuntimeError(f"source generation drift: {cancer}")
        bindings.append(
            {
                "cancer_id": cancer,
                "generated_in_r9_run": True,
                "output_root": str(output),
                "success_sha256": sha256(success_path),
                "contract_sha256": str(success["contract_sha256"]),
                "lineage_sha256": str(success["lineage_sha256"]),
                "file_manifest_sha256": str(success["file_manifest_sha256"]),
                "association_evidence_rows": int(success["association_evidence_rows"]),
                "cells": int(success["cells"]),
                "donors": int(success["donors"]),
                "lncrnas": int(success["lncrnas"]),
            }
        )
    count = len(bindings)
    value = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R11_EXTENSION_BINDING_V1",
        "status": f"SUCCESS_{count}_OF_{count}_BOUND_AND_AUDITED",
        "formal_cancer_universe": sorted(cancers),
        "formal_bound_cancer_count": count,
        "generated_in_this_run_count": count,
        "external_immutable_binding_count": 0,
        "formal_bindings": bindings,
        "total_cells_all_bound": sum(row["cells"] for row in bindings),
        "total_association_evidence_rows_all_bound": sum(
            row["association_evidence_rows"] for row in bindings
        ),
        "historical_assets_relabelled_fresh": False,
        "production_deployed": False,
    }
    exclusive_json(args.output_json.resolve(), value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
