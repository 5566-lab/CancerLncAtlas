"""Phase 14 smoke criterion: do typed relations actually reach HeteroData metadata?

The plan is explicit -- "禁止 GPU 启动后再发现 ENCODE 没进模型" -- and the same
discipline applies to the typed relations: the check is that they appear in the
graph's own metadata, not that they exist somewhere in a table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
sys.path.insert(0, str(WORK / "code"))

from cc_hhgt.v32.formal_graph import (  # noqa: E402
    build_variant_runtime_bundle,
    materialize_typed_lnc_protein_binding,
    typed_global_role,
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
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


print(f"=== rebuilding fold {FOLD} authority (typed generation) ===", flush=True)
fold_map = PREPARED / "SAMPLE_PATIENT_FOLD_MAP.tsv"
fold_receipt = PREPARED / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"
audit = validate_frozen_v32_patient_fold_binding(fold_map, fold_receipt)
fold_manifest = pd.read_csv(
    fold_map, sep="\t", dtype={"cancer_id": str, "sample_id": str, "patient_id": str}
)
receipt_path = AUTH / "GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json"
static_inputs = {
    name: (str(STATIC / f"{name}.parquet"), sha256(STATIC / f"{name}.parquet"))
    for name in ("detection", "signed_membership", "pathway_hierarchy",
                 "lnc_protein_binding", "protein_gene_encoding", "ppi")
}
graph_inputs = load_formal_graph_input_authority(
    receipt_path=receipt_path,
    receipt_sha256=sha256(receipt_path),
    patient_authority_audit=audit,
    static_inputs=static_inputs,
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
    graph_inputs,
    outer_fold=FOLD,
    split_manifest=split_manifest,
    candidate_pairs=candidates,
    binding_materializer=materialize_typed_lnc_protein_binding,
    binding_generation=RELATION_SCHEMA_TYPED_ASSAY_CLASS_V1,
)
print("    authority built", flush=True)

report: dict[str, object] = {}
for variant in ("G0", "G1", "G2"):
    print(f"=== building runtime bundle {variant} ===", flush=True)
    bundle = build_variant_runtime_bundle(bound.authority, variant, edge_chunk_size=250_000)
    data = bundle.hetero_data
    edge_types = sorted(tuple(str(x) for x in et) for et in data.edge_types)
    typed_present = sorted(
        {et[1] for et in edge_types if et[1].startswith("binds_protein_")}
    )
    binding_roles = sorted(
        {str(r) for r in bundle.edges.edge_role.unique()} if len(bundle.edges) else []
    )
    report[variant] = {
        "node_types": sorted(str(nt) for nt in data.node_types),
        "edge_type_count": len(edge_types),
        "binds_protein_relations_in_metadata": typed_present,
        "binding_roles_active": [r for r in binding_roles if "binding" in r or "eclip" in r],
        "active_edges": int(len(bundle.edges)),
        "hetero_data_edge_types": [list(et) for et in edge_types],
    }
    print(f"    node_types : {report[variant]['node_types']}")
    print(f"    edge types : {len(edge_types)}")
    print(f"    typed relations in metadata: {typed_present}")
    print(f"    active edges: {len(bundle.edges):,}", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "RUNTIME_BUNDLE_METADATA.json").write_text(
    json.dumps(report, indent=2, default=str), encoding="utf-8"
)

g0 = set(report["G0"]["binds_protein_relations_in_metadata"])
g1 = set(report["G1"]["binds_protein_relations_in_metadata"])
g2 = set(report["G2"]["binds_protein_relations_in_metadata"])
print()
print("=== Phase 14 smoke criterion ===")
print(f"  G0 typed binding relations : {sorted(g0) or 'none (expected)'}")
print(f"  G1 typed binding relations : {len(g1)}")
print(f"  G2 typed binding relations : {len(g2)}")
print(f"  G0 has no binding          : {len(g0) == 0}")
print(f"  G1 == G2 on binding schema : {g1 == g2}")
print(f"  predicted absent everywhere: "
      f"{not any('predicted' in r for r in g1 | g2)}")
print(f"written: {OUT / 'RUNTIME_BUNDLE_METADATA.json'}")
