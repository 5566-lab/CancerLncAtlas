#!/usr/bin/env python3
"""Replace only binding in an existing G2 C overlay, on host 149.

The patient fold, labels, other graph relations, node inventory and relation
schema are reused.  Existing large inputs are read without file rehashing.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import socket
from pathlib import Path

import pandas as pd

from cc_hhgt.v32.c_graph_overlay import OVERLAY_FORMAT
from cc_hhgt.v32.formal_graph import (
    TYPED_BINDING_ROLES, build_formal_graph_authority,
    build_variant_runtime_bundle, materialize_typed_lnc_protein_binding,
)
from cc_hhgt.v32.formal_graph_authority import (
    bind_formal_graph_variant, validate_formal_graph_payload_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-overlay", required=True, type=Path)
    parser.add_argument("--global-binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != "149" or args.fold not in range(5):
        raise RuntimeError("Final C graph CPU work requires host 149 and fold 0..4")
    old_path = args.old_overlay.resolve(strict=True)
    binding_path = args.global_binding.resolve(strict=True)
    output = args.output.resolve()
    if not args.check_only and (output.exists() or output.with_suffix(".json").exists()):
        raise FileExistsError("Refusing to overwrite an existing C graph generation")
    import torch

    with old_path.open("rb") as handle:
        old = torch.load(handle, map_location="cpu", weights_only=False)
    if old.get("format") != OVERLAY_FORMAT or int(old.get("patient_fold", -1)) != args.fold:
        raise RuntimeError("Old C overlay format or fold mismatch")
    prior = old["bundle"]
    typed = pd.read_parquet(binding_path)
    if typed.cancer_id.notna().any() or typed.is_context_specific.isna().any() or typed.is_context_specific.any():
        raise RuntimeError("Global binding table contains cancer-restricted rows")
    new_binding = materialize_typed_lnc_protein_binding(
        typed,
        include_predicted=False,
        include_context_eclip=False,
        require_global_binding=True,
        candidate_lncrnas=prior.node_maps["lncRNA"],
    )
    old_binding_mask = prior.edges.edge_role.astype(str).isin(TYPED_BINDING_ROLES)
    unchanged = prior.edges.loc[~old_binding_mask].copy()
    required_nodes = prior.nodes[["node_type", "canonical_id"]].copy()
    authority = build_formal_graph_authority(
        [unchanged, new_binding], outer_fold=args.fold, required_nodes=required_nodes,
    )
    if not authority.nodes[["node_type", "canonical_id", "node_index_within_type"]].equals(
        prior.nodes[["node_type", "canonical_id", "node_index_within_type"]]
    ):
        raise RuntimeError("Global binding changed the frozen node inventory")
    if len(authority.edges) <= len(prior.edges):
        raise RuntimeError("Global binding did not restore any previously omitted edges")
    final_binding_rows = authority.edges.edge_role.astype(str).isin(TYPED_BINDING_ROLES)
    if authority.edges.loc[final_binding_rows, "cancer_id"].notna().any() or authority.edges.loc[
        final_binding_rows, "is_context_specific"
    ].any():
        raise RuntimeError("Cancer-specific binding remains in the final G2 graph")
    if authority.edges.relation_type.eq("binds_protein_predicted").any():
        raise RuntimeError("Predicted binding entered the final G2 graph")
    bundle = build_variant_runtime_bundle(authority, "G2")
    prior_binding = old["formal_graph_authority"]
    new_binding_authority = copy.deepcopy(prior_binding)
    new_binding_authority["graph"].update({
        "edge_sha256": authority.safe_graph.manifest["edge_sha256"],
        "variant_edge_counts": authority.manifest["variant_edge_counts"],
    })
    new_binding_authority["global_binding_correction"] = {
        "policy": "physical_binding_global; assay_context_provenance_only",
        "source_path": str(binding_path),
        "source_bytes": binding_path.stat().st_size,
        "parent_graph_authority": prior_binding["graph"],
        "file_hashes_computed": False,
    }
    new_binding_authority = bind_formal_graph_variant(
        new_binding_authority, authority, "G2",
    )
    validate_formal_graph_payload_binding(
        new_binding_authority, outer_fold=args.fold, variant="G2", bundle=bundle,
    )
    audit = {
        "target_host": "149", "patient_fold": args.fold,
        "graph_variant": "G2", "binding_context_policy": "GLOBAL_PHYSICAL_BINDING",
        "old_edges": int(len(prior.edges)), "new_edges": int(len(authority.edges)),
        "added_edges": int(len(authority.edges) - len(prior.edges)),
        "old_binding_edges": int(old_binding_mask.sum()),
        "new_binding_edges": int(final_binding_rows.sum()),
        "predicted_binding_edges": 0,
        "contextual_binding_edges": 0,
        "new_graph_model_ready": True,
        "source_overlay": str(old_path),
        "global_binding": str(binding_path),
        "file_hashes_computed": False,
    }
    if args.check_only:
        print(json.dumps({**audit, "status": "PASS_G2_GLOBAL_BINDING_IN_MEMORY"}), flush=True)
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": OVERLAY_FORMAT,
        "patient_fold": args.fold,
        "formal_graph_variant": "G2",
        "source_prepared_sha256": old["source_prepared_sha256"],
        "graph_authority_receipt_sha256": old["graph_authority_receipt_sha256"],
        "formal_graph_authority": new_binding_authority,
        "bundle": bundle,
        "binding_context_policy": "GLOBAL_PHYSICAL_BINDING",
        "global_binding_source": {"path": str(binding_path), "bytes": binding_path.stat().st_size},
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt = {**audit, "status": "PASS_G2_GLOBAL_BINDING_SIDECAR",
               "path": str(output), "bytes": output.stat().st_size,
               "format": OVERLAY_FORMAT,
               "source_prepared_sha256": old["source_prepared_sha256"],
               "graph_authority_receipt_sha256": old["graph_authority_receipt_sha256"]}
    output.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
