#!/usr/bin/env python3
"""Smoke the real hash-bound 17-cancer UCell query across both source runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_ucell_17c_query import (  # noqa: E402
    SingleCellUCell17CQuery,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--binding", required=True, type=Path)
    value.add_argument("--binding-sha256", required=True)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--skip-runtime-rehash", action="store_true")
    return value


def _write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing smoke output reuse: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parser().parse_args()
    query = SingleCellUCell17CQuery(
        args.binding,
        expected_binding_sha256=args.binding_sha256,
        runtime_rehash_trees=not args.skip_runtime_rehash,
    )
    capability = query.capability_status()
    if capability["cancer_count"] != 17 or capability["ucell_capability_release_ready"] is not True:
        raise RuntimeError("17-cancer capability contract failed")
    acc_available = query.query_pathway_availability(
        cancer_id="ACC", availability="AVAILABLE", limit=3
    )
    acc_donor = query.query_donor_celltype_scores(
        cancer_id="ACC", availability="AVAILABLE", limit=3
    )
    acc_typed_null = query.query_cell_scores(
        cancer_id="ACC", availability="TYPED_UNAVAILABLE", limit=1
    )
    hnsc_cell = query.query_cell_scores(
        cancer_id="HNSC", availability="AVAILABLE", limit=3
    )
    if not acc_available["rows"] or not acc_donor["rows"] or not hnsc_cell["rows"]:
        raise RuntimeError("A real UCell query unexpectedly returned no available rows")
    if acc_typed_null["returned_rows"] != 1:
        raise RuntimeError("ACC typed-unavailable smoke did not return one row")
    null_row = acc_typed_null["rows"][0]
    if null_row["ucell_available"] is not False or null_row["ucell_score"] is not None:
        raise RuntimeError("Typed-unavailable UCell value was not preserved as null")
    payload = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_UCELL_17C_REAL_QUERY_SMOKE_V1",
        "status": "PASS",
        "binding": str(args.binding.resolve()),
        "binding_sha256": query.binding_sha256,
        "runtime_tree_rehash_performed": not args.skip_runtime_rehash,
        "capability": capability,
        "queries": {
            "acc_pathway_available": acc_available,
            "acc_donor_celltype_available": acc_donor,
            "acc_cell_typed_unavailable": acc_typed_null,
            "hnsc_cell_available": hnsc_cell,
        },
        "both_source_runs_queried": True,
        "typed_unavailable_null_preserved": True,
        "historical_outputs_used": False,
        "production_deployed": False,
    }
    _write_new(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
