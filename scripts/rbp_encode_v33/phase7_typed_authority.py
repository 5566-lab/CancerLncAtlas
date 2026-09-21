"""Materialise a typed-relation G012 graph authority generation.

Writes only under the new work root.  The frozen receipt and every formal
artifact are read-only inputs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
sys.path.insert(0, str(WORK / "code"))

from cc_hhgt.v32.formal_graph import materialize_typed_lnc_protein_binding  # noqa: E402
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
STATIC_SRC = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_patient_first_20260830_r1/static")
TYPED_BINDING = WORK / "outputs" / "phase3_typed_binding" / "lnc_protein_binding_typed.parquet"

OUT = WORK / "outputs" / "phase7_typed_authority"
STATIC_OUT = OUT / "static"
OUT.mkdir(parents=True, exist_ok=True)
STATIC_OUT.mkdir(parents=True, exist_ok=True)

SMOKE_FOLDS = [int(x) for x in (sys.argv[1:] or ["0"])]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- inputs
print("=== patient authority ===", flush=True)
fold_map = PREPARED / "SAMPLE_PATIENT_FOLD_MAP.tsv"
fold_receipt = PREPARED / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"
audit = validate_frozen_v32_patient_fold_binding(fold_map, fold_receipt)
print("   manifest_sha256:", audit["manifest_sha256"][:24], "…")
fold_manifest = pd.read_csv(
    fold_map, sep="\t", dtype={"cancer_id": str, "sample_id": str, "patient_id": str}
)

frozen_receipt = json.loads((FROZEN / "GRAPH_INPUT_AUTHORITY_RECEIPT.json").read_text())

# ---------------------------------------------------- typed static artifacts
print("\n=== building the typed static artifact set ===", flush=True)
for name in ("detection", "signed_membership", "pathway_hierarchy",
             "protein_gene_encoding", "ppi"):
    src = STATIC_SRC / f"{name}.parquet"
    dst = STATIC_OUT / f"{name}.parquet"
    if not dst.exists():
        shutil.copyfile(src, dst)
    declared = frozen_receipt["artifacts"][("lnc_protein_binding" if name == "binding" else name)]
    print(f"   {name:24s} {sha256(dst)[:16]}  (frozen {declared['sha256'][:16]})")

typed = pd.read_parquet(TYPED_BINDING)
typed_binding_out = STATIC_OUT / "lnc_protein_binding.parquet"
keep = ["lncrna_id", "protein_id", "weight", "cancer_id", "is_context_specific",
        "source_database", "graph_assay_class", "relation_type",
        "experiment_family", "assay_subtype"]
typed[keep].to_parquet(typed_binding_out, index=False)
typed_binding_sha = sha256(typed_binding_out)
print(f"   typed binding            {typed_binding_sha[:16]}  rows={len(typed):,}")

# ------------------------------------------------------------ typed receipt
print("\n=== writing the typed generation receipt ===", flush=True)
receipt = json.loads(json.dumps(frozen_receipt))
receipt["format"] = frozen_receipt["format"]
receipt["status"] = frozen_receipt["status"]
receipt["relation_schema_generation"] = RELATION_SCHEMA_TYPED_ASSAY_CLASS_V1
receipt["frozen_input_receipt_sha256"] = sha256(FROZEN / "GRAPH_INPUT_AUTHORITY_RECEIPT.json")
receipt["gates"] = dict(frozen_receipt["gates"])
# A typed generation genuinely does NOT have an identical relation schema.
receipt["gates"]["same_node_and_relation_schema_all_variants"] = False
receipt["gates"]["typed_assay_class_relations"] = True
receipt["artifacts"] = json.loads(json.dumps(frozen_receipt["artifacts"]))
receipt["artifacts"]["lnc_protein_binding"] = {
    **frozen_receipt["artifacts"]["lnc_protein_binding"],
    "path": str(typed_binding_out),
    "sha256": typed_binding_sha,
    "rows": int(len(typed)),
    "typed_generation": RELATION_SCHEMA_TYPED_ASSAY_CLASS_V1,
}
for name in ("detection", "signed_membership", "pathway_hierarchy",
             "protein_gene_encoding", "ppi"):
    receipt["artifacts"][name] = {
        **frozen_receipt["artifacts"][name],
        "path": str(STATIC_OUT / f"{name}.parquet"),
        "sha256": sha256(STATIC_OUT / f"{name}.parquet"),
    }
receipt_path = OUT / "GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json"
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
receipt_sha = sha256(receipt_path)
print(f"   {receipt_path.name}  sha256={receipt_sha[:24]}…")

# ------------------------------------------------------------- load + build
print("\n=== loading graph inputs under the typed generation ===", flush=True)
static_inputs = {
    name: (str(STATIC_OUT / f"{name}.parquet"), sha256(STATIC_OUT / f"{name}.parquet"))
    for name in ("detection", "signed_membership", "pathway_hierarchy",
                 "lnc_protein_binding", "protein_gene_encoding", "ppi")
}
graph_inputs = load_formal_graph_input_authority(
    receipt_path=receipt_path,
    receipt_sha256=receipt_sha,
    patient_authority_audit=audit,
    static_inputs=static_inputs,
    fold_expression_pattern=str(
        Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_patient_first_20260830_r1")
        / "folds/fold_{fold}/train_expression"
    ),
    fold_coexpression_pattern=str(FROZEN / "folds/fold_{fold}/train_coexpression"),
    relation_schema_generation=RELATION_SCHEMA_TYPED_ASSAY_CLASS_V1,
)
print("   typed generation accepted")

candidates = pd.read_parquet(PREPARED / "FORMAL_CANDIDATE_UNIVERSE.parquet")
print(f"   candidates: {len(candidates):,} rows, "
      f"{candidates.cancer_id.nunique()} cancers")

results = {}
for fold in SMOKE_FOLDS:
    print(f"\n=== fold {fold} ===", flush=True)
    split_manifest = assign_outer_split(fold_manifest, fold, n_folds=5, validation_offset=1)
    bound = build_bound_formal_graph(
        graph_inputs,
        outer_fold=fold,
        split_manifest=split_manifest,
        candidate_pairs=candidates,
        binding_materializer=materialize_typed_lnc_protein_binding,
        binding_generation=RELATION_SCHEMA_TYPED_ASSAY_CLASS_V1,
    )
    auth = bound.authority
    binding_relations = {
        key: value for key, value in auth.manifest["relation_counts"].items()
        if key.startswith("binds_protein")
    }
    results[fold] = {
        "edges": auth.manifest["g2_edge_count"],
        "variant_edge_counts": auth.manifest["variant_edge_counts"],
        "binding_relations": binding_relations,
        "node_counts": auth.manifest["node_counts"],
        "binding_generation": bound.binding.get("binding_generation"),
        "binding_materializer": bound.binding.get("binding_materializer"),
    }
    print(f"   edges={auth.manifest['g2_edge_count']:,}  "
          f"variants={auth.manifest['variant_edge_counts']}")
    for key, value in sorted(binding_relations.items(), key=lambda kv: -kv[1]):
        print(f"      {key:44s} {value:>10,}")
    fold_dir = OUT / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "AUTHORITY_SUMMARY.json").write_text(
        json.dumps(results[fold], indent=2, default=str), encoding="utf-8"
    )

(OUT / "PHASE7_TYPED_AUTHORITY_SUMMARY.json").write_text(
    json.dumps(
        {
            "relation_schema_generation": RELATION_SCHEMA_TYPED_ASSAY_CLASS_V1,
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_sha,
            "folds": results,
        },
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)
print(f"\nwritten: {OUT / 'PHASE7_TYPED_AUTHORITY_SUMMARY.json'}")
