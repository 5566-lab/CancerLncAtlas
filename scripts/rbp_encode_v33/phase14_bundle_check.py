"""Phase 14 smoke criterion, corrected: schema presence is not active edges.

The bundle deliberately carries the COMPLETE G2 relation schema in every arm as
empty tensors, so ``HeteroData.edge_types`` lists all relations regardless of
arm.  A check that only inspects the schema would "pass" trivially and verify
nothing.

The two claims are distinct and both matter:

1. **schema presence** -- the relation type is registered, which is what lets the
   graph carry typed message passing at all;
2. **active edges**   -- the arm's edge index actually contains rows for that
   relation, which is what the model sees.

This script reports both, per arm and per relation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
sys.path.insert(0, str(WORK / "code"))

from cc_hhgt.v32.formal_graph import (  # noqa: E402
    build_variant_runtime_bundle,
    materialize_typed_lnc_protein_binding,
)
from cc_hhgt.v32.formal_graph_authority import (  # noqa: E402
    RELATION_SCHEMA_TYPED_ASSAY_CLASS_V1,
    build_bound_formal_graph,
    load_formal_graph_input_authority,
)
from cc_hhgt.v32.patient_fold_authority import (  # noqa: E402
    validate_frozen_v32_patient_fold_binding,
)
from cc_hhgt.v32.patient_folds import assign_outer_split  # noqa: E402

FROZEN = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_local_cnv_corrected_graph_20260903_r1")
PREPARED = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_local_cnv_formal_prepared_20260903_r1")
AUTH = WORK / "outputs" / "phase7_typed_authority"
STATIC = AUTH / "static"
FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = AUTH / f"fold_{FOLD}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


print(f"=== rebuilding fold {FOLD} authority (typed generation) ===", flush=True)
fold_map = PREPARED / "SAMPLE_PATIENT_FOLD_MAP.tsv"
audit = validate_frozen_v32_patient_fold_binding(
    fold_map, PREPARED / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"
)
fold_manifest = pd.read_csv(
    fold_map, sep="\t", dtype={"cancer_id": str, "sample_id": str, "patient_id": str}
)
receipt_path = AUTH / "GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json"
graph_inputs = load_formal_graph_input_authority(
    receipt_path=receipt_path,
    receipt_sha256=sha256(receipt_path),
    patient_authority_audit=audit,
    static_inputs={
        name: (str(STATIC / f"{name}.parquet"), sha256(STATIC / f"{name}.parquet"))
        for name in ("detection", "signed_membership", "pathway_hierarchy",
                     "lnc_protein_binding", "protein_gene_encoding", "ppi")
    },
    fold_expression_pattern=str(
        Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_patient_first_20260830_r1")
        / "folds/fold_{fold}/train_expression"
    ),
    fold_coexpression_pattern=str(FROZEN / "folds/fold_{fold}/train_coexpression"),
    relation_schema_generation=RELATION_SCHEMA_TYPED_ASSAY_CLASS_V1,
)
candidates = pd.read_parquet(PREPARED / "FORMAL_CANDIDATE_UNIVERSE.parquet")
split_manifest = assign_outer_split(fold_manifest, FOLD, n_folds=5, validation_offset=1)
bound = build_bound_formal_graph(
    graph_inputs, outer_fold=FOLD, split_manifest=split_manifest,
    candidate_pairs=candidates,
    binding_materializer=materialize_typed_lnc_protein_binding,
    binding_generation=RELATION_SCHEMA_TYPED_ASSAY_CLASS_V1,
)
print("    authority built", flush=True)

report: dict[str, object] = {}
for variant in ("G0", "G1", "G2"):
    print(f"=== bundle {variant} ===", flush=True)
    bundle = build_variant_runtime_bundle(bound.authority, variant, edge_chunk_size=250_000)
    data = bundle.hetero_data
    schema_relations = sorted({
        str(et[1]) for et in data.edge_types if str(et[1]).startswith("binds_protein")
    })
    active = bundle.active_edges
    if active is not None and len(active):
        rows = active.loc[active.relation_type.astype(str).str.startswith("binds_protein")]
    else:
        rows = active
    active_counts = (
        {str(k): int(v) for k, v in rows.relation_type.value_counts().items()}
        if rows is not None and len(rows) else {}
    )
    report[variant] = {
        "schema_binding_relations": schema_relations,
        "schema_binding_relation_count": len(schema_relations),
        "active_edges_total": int(len(bundle.edges)),
        "active_binding_edges": int(sum(active_counts.values())),
        "active_binding_relation_counts": active_counts,
        "hetero_data_edge_type_count": len(list(data.edge_types)),
    }
    print(f"    schema binding relations : {len(schema_relations)} (metadata)")
    print(f"    active edges            : {len(bundle.edges):,}")
    print(f"    active BINDING edges    : {sum(active_counts.values()):,}")
    for rel, n in sorted(active_counts.items(), key=lambda kv: -kv[1]):
        print(f"        {rel:44s} {n:>10,}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "RUNTIME_BUNDLE_METADATA.json").write_text(
    json.dumps(report, indent=2, default=str), encoding="utf-8"
)

g0 = report["G0"]; g1 = report["G1"]; g2 = report["G2"]
print()
print("=== Phase 14 criterion, stated correctly ===")
print(f"  schema: typed binding relations registered in every arm : "
      f"{g0['schema_binding_relation_count']} == {g1['schema_binding_relation_count']} "
      f"== {g2['schema_binding_relation_count']}")
print(f"  active binding edges  G0 : {g0['active_binding_edges']:,}   (must be 0)")
print(f"  active binding edges  G1 : {g1['active_binding_edges']:,}")
print(f"  active binding edges  G2 : {g2['active_binding_edges']:,}")
print(f"  G1 == G2 on active binding : {g1['active_binding_edges'] == g2['active_binding_edges']}")
print(f"  predicted active edges     : "
      f"{g1['active_binding_relation_counts'].get('binds_protein_predicted', 0):,} "
      f"(must be 0 under the conservative default)")
print(f"written: {OUT / 'RUNTIME_BUNDLE_METADATA.json'}")
