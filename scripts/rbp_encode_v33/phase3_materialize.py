"""Phase 3: materialise typed lncRNA-protein binding and prove legacy equivalence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
sys.path.insert(0, str(WORK / "code"))

from cc_hhgt.v32.rbp_assay import classify_assay  # noqa: E402
from cc_hhgt.v32.rbp_typed_binding import (  # noqa: E402
    BindingTyping,
    build_typed_lnc_protein_binding,
    verify_legacy_equivalence,
)

STD = Path(
    "${PRIVATE_WORK_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/"
    "V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized"
)
OUT = WORK / "outputs" / "phase3_typed_binding"
OUT.mkdir(parents=True, exist_ok=True)

print("=== loading frozen standardized inputs ===", flush=True)
interaction = pd.read_parquet(STD / "interaction_relation.parquet")
protein_gene = pd.read_parquet(STD / "protein_gene_map.parquet")
dim_lnc = pd.read_parquet(STD / "dim_lncRNA.parquet")
dim_cancer = pd.read_parquet(STD / "dim_cancer.parquet")
frozen = pd.read_parquet(STD / "lncRNA_protein_relation_context.parquet")
print(f"    interaction_relation rows : {len(interaction):,}")
print(f"    frozen binding rows       : {len(frozen):,}", flush=True)

# --------------------------------------------------------------------------
# 1. LEGACY EQUIVALENCE  (Phase 12 gate #12)
# --------------------------------------------------------------------------
print("\n=== 1. legacy mode (both switches OFF) ===", flush=True)
legacy, legacy_audit = build_typed_lnc_protein_binding(
    interaction, protein_gene, dim_lnc, dim_cancer,
    typing=BindingTyping(include_rbp_partner_type=False, group_by_assay_class=False),
    classify=classify_assay,
)
print(f"    produced rows : {len(legacy):,}")
comparison = verify_legacy_equivalence(legacy, frozen)
print(f"    equivalence   : {comparison['status']}")
print(f"    {json.dumps({k: v for k, v in comparison.items() if k != 'compared_columns'}, indent=6)}")

# --------------------------------------------------------------------------
# 2. TYPED MODE  (rbp included, grouped by assay class)
# --------------------------------------------------------------------------
print("\n=== 2. typed mode (rbp included + assay-class grouping) ===", flush=True)
typed, typed_audit = build_typed_lnc_protein_binding(
    interaction, protein_gene, dim_lnc, dim_cancer,
    typing=BindingTyping(include_rbp_partner_type=True, group_by_assay_class=True),
    classify=classify_assay,
)
print(f"    typed rows : {len(typed):,}")
print("    relation_type counts:")
for rel, n in sorted(typed_audit["relation_type_counts"].items(), key=lambda x: -x[1]):
    print(f"       {rel:42s} {n:>10,}")
print("    graph_assay_class counts:")
for cls, n in sorted(typed_audit["graph_assay_class_counts"].items(), key=lambda x: -x[1]):
    print(f"       {cls:42s} {n:>10,}")

# --------------------------------------------------------------------------
# 3. DELTA versus the frozen table
# --------------------------------------------------------------------------
print("\n=== 3. delta analysis ===", flush=True)
frozen_pairs = set(zip(frozen.lncrna_id.astype(str), frozen.protein_id.astype(str)))
typed_pairs = set(zip(typed.lncrna_id.astype(str), typed.protein_id.astype(str)))
print(f"    frozen distinct (lnc,protein) pairs : {len(frozen_pairs):,}")
print(f"    typed  distinct (lnc,protein) pairs : {len(typed_pairs):,}")
print(f"    pairs added                         : {len(typed_pairs - frozen_pairs):,}")

typed.to_parquet(OUT / "lnc_protein_binding_typed.parquet", index=False)
(OUT / "PHASE3_TYPED_BINDING_AUDIT.json").write_text(
    json.dumps(
        {
            "legacy_audit": legacy_audit,
            "typed_audit": typed_audit,
            "legacy_equivalence": comparison,
            "frozen_rows": int(len(frozen)),
            "typed_rows": int(len(typed)),
            "pairs_added": len(typed_pairs - frozen_pairs),
        },
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)
print(f"\nwritten: {OUT / 'lnc_protein_binding_typed.parquet'}")
print(f"written: {OUT / 'PHASE3_TYPED_BINDING_AUDIT.json'}")
