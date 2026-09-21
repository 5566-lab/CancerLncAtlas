"""Independently validate the revision-3 per-cancer readiness matrix.

This is a read-only audit.  It does not prepare graphs, train a model, or write
an authority receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


TRUE_ATAC_RAW_GAPS = {
    "DLBC",
    "KICH",
    "LAML",
    "OV",
    "PAAD",
    "READ",
    "SARC",
    "THYM",
    "UCS",
    "UVM",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(repo_root: Path) -> dict[str, object]:
    matrix_path = repo_root / "docs/v32_g012_33_cancer_status_matrix_20260829.tsv"
    mutation_path = (
        repo_root
        / "artifacts/v32_full_multitask/genomic_fresh_rerun1/"
        "mutation_cnv_typed_predictions.parquet"
    )
    cnv_audit_path = (
        repo_root
        / "artifacts/v32_streaming_cnv_independent_audit_20260829_r2_AUDIT.json"
    )
    atac_readiness_path = (
        repo_root
        / "artifacts/v32_atac_readiness_audit_20260829_r1/ATAC_CANCER_READINESS.tsv"
    )

    matrix = pd.read_csv(matrix_path, sep="\t")
    if len(matrix) != 33 or matrix["cancer_id"].nunique() != 33:
        raise AssertionError("matrix must contain 33 unique cancer rows")

    mutation = pd.read_parquet(mutation_path, columns=["cancer_id"])
    mutation_counts = mutation.groupby("cancer_id", sort=True).size()
    if set(mutation_counts.index) != set(matrix["cancer_id"]):
        raise AssertionError("Mutation and matrix cancer sets differ")
    if not mutation_counts.eq(100_000).all():
        raise AssertionError("Mutation fixed expert is not 100,000 rows/cancer")

    cnv_audit = json.loads(cnv_audit_path.read_text(encoding="utf-8"))
    if cnv_audit.get("status") != "PASS" or cnv_audit.get("cancer_count") != 33:
        raise AssertionError("CNV materialization audit is not a 33-cancer PASS")
    cnv_patients = {
        row["cancer_id"]: int(row["patients"]) for row in cnv_audit["cancers"]
    }
    matrix_cnv_patients = dict(
        zip(matrix["cancer_id"], matrix["cnv_patients"].astype(int), strict=True)
    )
    if matrix_cnv_patients != cnv_patients:
        raise AssertionError("CNV patient counts differ from independent audit")

    atac = pd.read_csv(atac_readiness_path, sep="\t")
    atac_patients = dict(
        zip(atac["cancer_id"], atac["raw_atac_patients"].astype(int), strict=True)
    )
    matrix_atac_patients = dict(
        zip(matrix["cancer_id"], matrix["atac_raw_patients"].astype(int), strict=True)
    )
    if matrix_atac_patients != atac_patients:
        raise AssertionError("ATAC raw-patient counts differ from readiness audit")
    observed_gaps = set(atac.loc[~atac["raw_atac_available"], "cancer_id"])
    matrix_gaps = set(
        matrix.loc[
            matrix["atac_fresh_oof"].eq(
                "TYPED_TRUE_RAW_GAP_CURRENT_OFFICIAL_MATRIX"
            ),
            "cancer_id",
        ]
    )
    if observed_gaps != TRUE_ATAC_RAW_GAPS or matrix_gaps != TRUE_ATAC_RAW_GAPS:
        raise AssertionError("ATAC typed raw-gap set differs from source evidence")

    fixed_columns = {
        "common_g0_g1_g2": "NOT_READY_DESIGN_ONLY",
        "mutation_fixed_3m3": "READY_FIXED_3M3_NOT_COMMON_BOUND",
        "cnv_materialization": "PASS_MEASURED_NOT_TRAINED",
        "cnv_five_fold_oof": "NOT_READY",
        "routing_comparison": "NOT_READY",
    }
    for column, expected in fixed_columns.items():
        if not matrix[column].eq(expected).all():
            raise AssertionError(f"{column} contains an overstated state")

    return {
        "format": "CC_HHGT_V3_2_G012_33_CANCER_STATUS_MATRIX_AUDIT_V1",
        "status": "PASS",
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256(matrix_path),
        "cancer_count": 33,
        "mutation_rows": int(len(mutation)),
        "mutation_cancers": int(mutation_counts.size),
        "cnv_materialized_cancers": int(cnv_audit["cancer_count"]),
        "atac_raw_covered_cancers": int(atac["raw_atac_available"].sum()),
        "atac_true_raw_gap_cancers": sorted(TRUE_ATAC_RAW_GAPS),
        "g012_training_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(audit(args.repo_root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
