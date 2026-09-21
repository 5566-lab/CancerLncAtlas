#!/usr/bin/env python3
"""Bind the 13 immutable completed cancers and 10 r11 results into formal23."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


OLD_ROOTS = {
    "ACC": "v32_single_cell_r7_fresh_streaming_r9_remaining5_20260829_r1",
    "HNSC": "v32_single_cell_r7_fresh_streaming_17c_20260829_r3",
    "SARC": "v32_single_cell_r7_fresh_streaming_r8_sarc_20260829_r4",
    **{cancer: "v32_single_cell_r7_fresh_streaming_r8_remaining15_20260829_r1" for cancer in (
        "CHOL", "DLBC", "ESCA", "KIRC", "LAML", "LGG", "LUSC", "MESO", "SKCM", "THYM"
    )},
}
NEW_CANCERS = ("BRCA", "CESC", "COAD", "GBM", "OV", "PCPG", "READ", "TGCT", "UCEC", "UVM")


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


def binding(cancer: str, output: Path, generated: bool) -> dict[str, Any]:
    success_path = output / "SUCCESS.json"
    success = load_json(success_path)
    if success.get("status") != "SUCCESS" or success.get("cancer_id") != cancer:
        raise RuntimeError(f"single-cell SUCCESS drift: {cancer}")
    if success.get("historical_derived_results_used") is not False:
        raise RuntimeError(f"historical derived result used: {cancer}")
    if success.get("source_generation") != "V3.2_R7_FRESH_FROM_RAW_H5":
        raise RuntimeError(f"source generation drift: {cancer}")
    return {
        "cancer_id": cancer,
        "generated_in_current_r11_run": generated,
        "source_output_generation": (
            "CURRENT_R11_MISSING10" if generated else "IMMUTABLE_PRIOR_V32_FRESH"
        ),
        "output_root": str(output.resolve()),
        "success_sha256": sha256(success_path),
        "contract_sha256": str(success["contract_sha256"]),
        "lineage_sha256": str(success["lineage_sha256"]),
        "file_manifest_sha256": str(success["file_manifest_sha256"]),
        "association_evidence_rows": int(success["association_evidence_rows"]),
        "cells": int(success["cells"]),
        "donors": int(success["donors"]),
        "lncrnas": int(success["lncrnas"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--new-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    rows = [
        binding(cancer, args.model_root.resolve() / root / f"cancer_id={cancer}", False)
        for cancer, root in OLD_ROOTS.items()
    ]
    rows.extend(
        binding(cancer, args.new_root.resolve() / "partitions" / f"cancer_id={cancer}", True)
        for cancer in NEW_CANCERS
    )
    rows.sort(key=lambda row: row["cancer_id"])
    cancers = [row["cancer_id"] for row in rows]
    if len(rows) != 23 or len(set(cancers)) != 23:
        raise RuntimeError("formal23 binding closure failed")
    value = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R11_FORMAL23_BINDING_V1",
        "status": "BOUND_23_OF_23_PENDING_INDEPENDENT_AUDIT",
        "formal_cancer_universe": cancers,
        "formal_bound_cancer_count": 23,
        "generated_in_this_run_count": 10,
        "external_immutable_binding_count": 13,
        "formal_bindings": rows,
        "total_cells_all_bound": sum(row["cells"] for row in rows),
        "total_association_evidence_rows_all_bound": sum(row["association_evidence_rows"] for row in rows),
        "historical_assets_relabelled_fresh": False,
        "production_deployed": False,
    }
    output = args.output_json.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    if output.exists():
        temporary.unlink()
        raise RuntimeError(f"immutable binding exists: {output}")
    os.replace(temporary, output)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
