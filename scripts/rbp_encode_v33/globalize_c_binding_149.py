#!/usr/bin/env python3
"""Create the global physical-binding source for the final C graph on host 149.

This reuses the already merged typed binding table (ruling B and ENCODE v2).
It does not repeat source acquisition, hash existing files, or touch old outputs.
Assay contexts remain inspectable as provenance in the two original columns.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import pandas as pd


def globalize_binding(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "lncrna_id", "protein_id", "cancer_id", "is_context_specific",
        "graph_assay_class", "weight", "relation_type",
    }
    missing = sorted(required - set(source))
    if missing:
        raise ValueError(f"Typed binding is missing columns: {missing}")
    context = source["is_context_specific"]
    if context.isna().any() or not pd.api.types.is_bool_dtype(context.dtype):
        raise ValueError("Binding context flags must be non-null booleans")
    if source[["lncrna_id", "protein_id", "graph_assay_class"]].isna().any().any():
        raise ValueError("Binding endpoints and assay classes must be present")
    result = source.copy()
    result["assay_record_cancer_id"] = source.cancer_id.astype("string")
    result["assay_record_context_specific"] = context.astype(bool)
    result["cancer_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["is_context_specific"] = False
    audit = {
        "source_rows": int(len(source)),
        "globalized_rows": int(len(result)),
        "source_cancer_tagged_rows": int(source.cancer_id.notna().sum()),
        "source_context_flagged_rows": int(context.sum()),
        "remaining_cancer_tagged_rows": int(result.cancer_id.notna().sum()),
        "remaining_context_flagged_rows": int(result.is_context_specific.sum()),
        "predicted_rows_retained_as_separate_class": int(result.graph_assay_class.eq("predicted").sum()),
        "policy": "physical_binding_global; assay_context_provenance_only",
        "file_hashes_computed": False,
        "target_host": "149",
    }
    if audit["source_rows"] != audit["globalized_rows"]:
        raise AssertionError("Binding rows changed during context projection")
    if audit["remaining_cancer_tagged_rows"] or audit["remaining_context_flagged_rows"]:
        raise AssertionError("Cancer-specific binding survived global projection")
    return result, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("CancerLncAtlas CPU work must execute on host 149")
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing generation: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output parent must be preflighted: {output.parent}")
    result, audit = globalize_binding(pd.read_parquet(source))
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        result.to_parquet(temporary, index=False)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    audit.update({"source_path": str(source), "output_path": str(output),
                  "output_bytes": output.stat().st_size})
    receipt = output.with_suffix(".json")
    receipt.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
