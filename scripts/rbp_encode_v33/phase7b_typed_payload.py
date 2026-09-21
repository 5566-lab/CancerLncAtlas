"""Produce the typed prepared payload by surgical substitution.

The frozen payload stores node INDICES and the LASSO ``base_logit``.  Measured
against the frozen authority records, the node universe is identical between the
legacy and typed generations (all six node types, and ``node_sha256`` is the same
for every fold), so the batches and their LASSO logits can be carried over
verbatim -- no LASSO re-fit, which is the reuse requirement.

What this script replaces, and only this:

* ``bundle``                  -> the typed GraphBundle for the requested variant
* ``formal_graph_authority``  -> the typed variant binding
* ``formal_graph_variant``    -> unchanged value, restated for clarity

Everything else -- ``train_batches``, ``validation_batches``, ``label_contract``,
``artifact_hashes``, ``patient_fold_authority``, ``input_scope`` -- is copied
through untouched.

It fails closed if the node universe moved, because reusing index-based batches
across a changed universe would misalign silently rather than raise.
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
    bind_formal_graph_variant,
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
OUT = WORK / "outputs" / "phase7_typed_payload"

FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
VARIANTS = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("G1",)
DRY_RUN = "--dry-run" in sys.argv

OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------- node gate
print(f"=== node-universe gate (fold {FOLD}) ===", flush=True)
frozen_record = json.loads(
    (PREPARED / f"GRAPH_AUTHORITIES/PATIENT_FOLD_{FOLD}.json").read_text()
)["graph"]
frozen_nodes = frozen_record["node_counts"]

# ------------------------------------------------------------- typed build
print("=== building the typed authority ===", flush=True)
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

typed_nodes = bound.authority.manifest["node_counts"]
print(f"    frozen nodes: {frozen_nodes}")
print(f"    typed  nodes: {typed_nodes}")

if frozen_nodes != typed_nodes:
    raise SystemExit(
        "FAIL-CLOSED: the node universe moved between the legacy and typed "
        "generations. Reusing the frozen index-based batches would misalign "
        "silently, so no payload was written."
    )
print("    node universe identical -> frozen batches and LASSO logits are reusable",
      flush=True)

if DRY_RUN:
    print("\n--dry-run: gate passed, no payload written")
    raise SystemExit(0)

# ---------------------------------------------------- surgical substitution
for variant in VARIANTS:
    src = PREPARED / variant / f"PATIENT_FOLD_{FOLD}.pt"
    if not src.is_file():
        print(f"  {variant}: frozen payload absent at {src}")
        continue
    print(f"\n=== {variant}: loading frozen payload ===", flush=True)
    import torch

    payload = torch.load(src, map_location="cpu", weights_only=False)
    print(f"    keys: {sorted(payload)}")

    print(f"    building typed bundle for {variant}", flush=True)
    bundle = build_variant_runtime_bundle(bound.authority, variant)
    graph_binding = bind_formal_graph_variant(bound.binding, bound.authority, variant)

    payload["bundle"] = bundle
    payload["formal_graph_authority"] = graph_binding
    payload["formal_graph_variant"] = variant

    dst_dir = OUT / variant
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"PATIENT_FOLD_{FOLD}.pt"
    torch.save(payload, dst)
    print(f"    written: {dst}  ({dst.stat().st_size / 1024**3:.2f} GB)")
    del payload, bundle

print("\ndone")
