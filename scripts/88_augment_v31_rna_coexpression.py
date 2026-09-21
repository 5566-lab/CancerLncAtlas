#!/usr/bin/env python3
"""Add frozen outcome-free coexpression summaries to full RNA context."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cc_hhgt.common import file_sha256, write_table
from cc_hhgt.v30_integrity import atomic_write_json, merkle_sha256
from cc_hhgt.v31_rna_context import augment_rna_context


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-rna-root", required=True)
    parser.add_argument("--coexpression-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    base_root = Path(args.base_rna_root).resolve()
    coexpression_root = Path(args.coexpression_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"RNA augmentation refuses reuse: {output_root}")
    base_table = base_root / "FULL_RNA_CONTEXT.parquet"
    base_audit_path = base_root / "FULL_RNA_CONTEXT_AUDIT.json"
    base_success_path = base_root / "SUCCESS.json"
    for path in (base_table, base_audit_path, base_success_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    base_success = json.loads(base_success_path.read_text(encoding="utf-8"))
    base_audit = json.loads(base_audit_path.read_text(encoding="utf-8"))
    if base_success.get("status") != "PASS" or base_audit.get("status") != "PASS":
        raise RuntimeError("Frozen full RNA context is not a PASS")
    if base_success.get("table_sha256") != file_sha256(base_table):
        raise RuntimeError("Frozen full RNA context hash mismatch")

    output_root.mkdir(parents=True)
    augmented, context_audit = augment_rna_context(
        pd.read_parquet(base_table), coexpression_root
    )
    table_path = output_root / "FULL_RNA_COEXPRESSION_CONTEXT.parquet"
    _atomic_table(augmented, table_path)
    source_rows = context_audit.pop("sources")
    lineage_path = output_root / "RNA_COEXPRESSION_INPUT_LINEAGE.tsv"
    _atomic_table(pd.DataFrame(source_rows), lineage_path)
    audit = {
        **context_audit,
        "stage": "PHASE_B3_TARGET_RNA_COEXPRESSION_AUGMENTATION",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(augmented)),
        "base_rna_table_sha256": file_sha256(base_table),
        "base_rna_audit_sha256": file_sha256(base_audit_path),
        "base_rna_success_sha256": file_sha256(base_success_path),
        "table_sha256": file_sha256(table_path),
        "lineage_sha256": file_sha256(lineage_path),
        "full_cancer_model_training_started": False,
        "failures": [],
    }
    audit_path = output_root / "RNA_COEXPRESSION_CONTEXT_AUDIT.json"
    atomic_write_json(audit_path, audit)
    manifest_rows = [
        {
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in (table_path, lineage_path, audit_path)
    ]
    manifest_path = output_root / "RNA_COEXPRESSION_CONTEXT_SHA256.tsv"
    _atomic_table(pd.DataFrame(manifest_rows), manifest_path)
    success = {
        "status": "PASS",
        "table_sha256": file_sha256(table_path),
        "audit_sha256": file_sha256(audit_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_merkle_sha256": merkle_sha256(manifest_rows),
        "full_cancer_model_training_started": False,
        "success_written_last": True,
    }
    atomic_write_json(output_root / "SUCCESS.json", success)
    print(json.dumps(success, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
