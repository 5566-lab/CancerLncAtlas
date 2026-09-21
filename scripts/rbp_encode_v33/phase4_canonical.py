"""Phase 4: cross-database duplicate audit on the real interaction table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
sys.path.insert(0, str(WORK / "code"))

from cc_hhgt.v32.rbp_assay import classify_assay  # noqa: E402
from cc_hhgt.v32.rbp_canonical import (  # noqa: E402
    collapse_canonical_experiments,
    duplicate_report,
)

STD = Path(
    "${PRIVATE_WORK_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/"
    "V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized"
)
OUT = WORK / "manifests"

print("=== loading interaction_relation ===", flush=True)
ir = pd.read_parquet(
    STD / "interaction_relation.parquet",
    columns=["lncrna_id", "partner_id", "partner_type", "pmid", "source_database",
             "source_record_id", "cell_line", "tissue", "experiment_raw",
             "is_experimental", "is_predicted"],
)
print(f"    rows: {len(ir):,}", flush=True)

print("\n=== pmid field shape ===", flush=True)
sample = ir.pmid.dropna().astype(str).head(5).tolist()
print(f"    samples: {sample}")
multi = ir.pmid.astype(str).str.contains(r"[;|,]", regex=True, na=False).sum()
print(f"    rows with multiple pmid separators: {multi:,}")
print(f"    empty pmid rows: {int(ir.pmid.isna().sum() + (ir.pmid.astype(str).str.strip()=='').sum()):,}")

print("\n=== classifying assay_subtype ===", flush=True)
raw = ir.experiment_raw
assignments = [
    classify_assay(value, is_predicted=pred, is_experimental=exp)
    for value, pred, exp in zip(
        raw.tolist(),
        ir.is_predicted.tolist(),
        ir.is_experimental.tolist(),
    )
]
ir["assay_subtype"] = [a.assay_subtype for a in assignments]
ir["graph_assay_class"] = [a.graph_assay_class for a in assignments]
del assignments

print("\n=== source_database distribution ===", flush=True)
print(ir.source_database.astype(str).value_counts().head(20).to_string())

print("\n=== potential cross-database duplicates (relaxed key, report only) ===", flush=True)
report = duplicate_report(ir, top=20)
print(report[["rows", "database_count", "databases", "assay_subtype", "pmids"]].to_string(index=False))

print("\n=== canonical collapse ===", flush=True)
result = collapse_canonical_experiments(ir)
print(json.dumps(result.audit, indent=6))

# Restrict the headline numbers to the RBP-relevant subset as well.
rbp_mask = ir.partner_type.astype(str).str.lower().str.strip().eq("rbp")
print("\n=== same audit restricted to partner_type == 'rbp' ===", flush=True)
rbp_result = collapse_canonical_experiments(ir.loc[rbp_mask].copy())
print(json.dumps(rbp_result.audit, indent=6))

payload = {
    "all_rows": result.audit,
    "rbp_rows": rbp_result.audit,
    "top_cross_database_duplicates": report.to_dict("records"),
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "PHASE4_CANONICAL_AUDIT.json").write_text(
    json.dumps(payload, indent=2, default=str), encoding="utf-8"
)
print(f"\nwritten: {OUT / 'PHASE4_CANONICAL_AUDIT.json'}")
