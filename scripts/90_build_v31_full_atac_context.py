#!/usr/bin/env python3
"""Build full registered-scope official TCGA ATAC context on the CPU server."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cc_hhgt.common import file_sha256, write_table
from cc_hhgt.v30_integrity import atomic_write_json, merkle_sha256
from cc_hhgt.v31_atac_context import derive_full_scope_r_script, finalize_atac_context


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_tsv_gzip(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.gz")
    temporary.unlink(missing_ok=True)
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, sep="\t", index=False)
    os.replace(temporary, path)


def _run(command: list[str], log_path: Path, *, environment: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--base-rna-table", required=True)
    parser.add_argument("--base-rna-success", required=True)
    parser.add_argument("--frozen-r-script", required=True)
    parser.add_argument("--frozen-datas7-script", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    asset_root = Path(args.asset_root).resolve()
    base_table = Path(args.base_rna_table).resolve()
    base_success_path = Path(args.base_rna_success).resolve()
    frozen_r = Path(args.frozen_r_script).resolve()
    frozen_datas7 = Path(args.frozen_datas7_script).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"Full ATAC context builder refuses reuse: {output_root}")
    matrix_path = asset_root / "atac" / "TCGA-ATAC_PanCan_Log2Norm_Counts.rds"
    mapping_path = asset_root / "atac" / "TCGA_identifier_mapping.txt"
    peak_path = asset_root / "atac" / "TCGA-ATAC_PanCancer_PeakSet.txt"
    workbook_path = asset_root / "atac" / "TCGA-ATAC_DataS7_PeakToGeneLinks_v2.xlsx"
    promoter_path = asset_root / "gencode" / "gencode.v36.transcript_promoters.minus1000_plus100.bed.gz"
    required = [
        base_table, base_success_path, frozen_r, frozen_datas7, matrix_path,
        mapping_path, peak_path, workbook_path, promoter_path,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    base_success = json.loads(base_success_path.read_text(encoding="utf-8"))
    if base_success.get("status") != "PASS" or base_success.get("table_sha256") != file_sha256(base_table):
        raise RuntimeError("Full RNA candidate source is not an immutable PASS")
    output_root.mkdir(parents=True)
    intermediate = output_root / "intermediate"
    intermediate.mkdir()

    base = pd.read_parquet(base_table, columns=["cancer_id", "lncrna_id"])
    keys = base.drop_duplicates().sort_values(["cancer_id", "lncrna_id"], kind="stable")
    if keys.cancer_id.nunique() != 31 or keys.duplicated(["cancer_id", "lncrna_id"]).any():
        raise RuntimeError("Full RNA table does not provide exact 31-cancer unique keys")
    candidate_path = intermediate / "candidate_cancer_lncrna_keys.tsv.gz"
    _atomic_tsv_gzip(keys, candidate_path)

    derived_source, patch_audit = derive_full_scope_r_script(
        frozen_r.read_text(encoding="utf-8")
    )
    derived_r = intermediate / "90_extract_v31_full_atac_context.R"
    derived_r.write_text(derived_source, encoding="utf-8", newline="\n")
    datas7_path = intermediate / "datas7_static_lnc_peak_links.tsv.gz"
    datas7_audit_path = intermediate / "DATAS7_TOPOLOGY_AUDIT.json"
    _run(
        [
            "/usr/bin/python3", str(frozen_datas7),
            "--workbook", str(workbook_path),
            "--promoters", str(promoter_path),
            "--candidate-keys", str(candidate_path),
            "--output", str(datas7_path),
            "--audit", str(datas7_audit_path),
        ],
        output_root / "DATAS7_BUILD.log",
    )
    r_environment = os.environ.copy()
    r_environment.update(
        {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"}
    )
    _run(
        [
            "/usr/bin/Rscript", str(derived_r), str(asset_root),
            str(candidate_path), str(datas7_path), str(intermediate),
        ],
        output_root / "ATAC_BUILD.log",
        environment=r_environment,
    )
    r_audit_path = intermediate / "ATAC_BUILD_AUDIT.json"
    r_features_path = intermediate / "atac_compact_features.tsv.gz"
    r_audit = json.loads(r_audit_path.read_text(encoding="utf-8"))
    if r_audit.get("status") != "PASS":
        raise RuntimeError("Full-scope R ATAC aggregation is not PASS")
    final, final_audit = finalize_atac_context(
        keys,
        pd.read_csv(r_features_path, sep="\t", compression="gzip"),
        covered_cancers=r_audit["atac_covered_cancers"],
    )
    table_path = output_root / "FULL_ATAC_CONTEXT.parquet"
    _atomic_table(final, table_path)
    patch_audit.update(
        {
            "frozen_r_script": str(frozen_r),
            "frozen_r_script_sha256": file_sha256(frozen_r),
            "derived_r_script": str(derived_r),
            "derived_r_script_sha256": file_sha256(derived_r),
        }
    )
    atomic_write_json(output_root / "ATAC_R_SCOPE_PATCH_AUDIT.json", patch_audit)
    audit = {
        **final_audit,
        "stage": "PHASE_B3_FULL_ATAC_CONTEXT_BUILD",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_normalized_matrix_read": True,
        "official_sample_lookup_read": True,
        "official_peak_set_read": True,
        "datas7_policy": "STATIC_TOPOLOGY_ONLY_NO_CORRELATION_OR_FDR",
        "pf_use_authorized": False,
        "r_build_audit_sha256": file_sha256(r_audit_path),
        "datas7_audit_sha256": file_sha256(datas7_audit_path),
        "table_sha256": file_sha256(table_path),
        "full_cancer_model_training_started": False,
        "failures": [],
    }
    audit_path = output_root / "FULL_ATAC_CONTEXT_AUDIT.json"
    atomic_write_json(audit_path, audit)
    lineage_rows = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in required
    ]
    lineage_path = output_root / "FULL_ATAC_CONTEXT_INPUT_LINEAGE.tsv"
    _atomic_table(pd.DataFrame(lineage_rows), lineage_path)
    output_files = [
        table_path, audit_path, lineage_path,
        output_root / "ATAC_R_SCOPE_PATCH_AUDIT.json",
        candidate_path, datas7_path, datas7_audit_path, r_features_path,
        intermediate / "atac_promoter_sample_context.tsv.gz",
        r_audit_path, derived_r,
    ]
    manifest_rows = [
        {
            "relative_path": str(path.relative_to(output_root)),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in output_files
    ]
    manifest_path = output_root / "FULL_ATAC_CONTEXT_SHA256.tsv"
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
