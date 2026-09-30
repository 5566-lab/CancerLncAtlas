#!/usr/bin/env python3
"""Build only the C graph sidecar; never rewrite the 170 GB A fold payloads.

Run on host 149.  ``--check-only`` rebuilds the graph in memory and writes
nothing.  The non-check path writes one G2 fold sidecar and a small receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.c_graph_overlay import OVERLAY_FORMAT, TYPED_GENERATION  # noqa: E402
from cc_hhgt.v32.formal_graph import (  # noqa: E402
    build_variant_runtime_bundle, materialize_typed_lnc_protein_binding,
)
from cc_hhgt.v32.formal_graph_authority import (  # noqa: E402
    bind_formal_graph_variant, build_bound_formal_graph,
    load_formal_graph_input_authority, validate_formal_graph_payload_binding,
)
from cc_hhgt.v32.patient_fold_authority import (  # noqa: E402
    validate_frozen_v32_patient_fold_binding,
)
from cc_hhgt.v32.patient_folds import assign_outer_split  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_pattern(receipt: dict, artifact: str) -> str:
    first = str(receipt["fold_artifacts"]["0"][artifact]["path"])
    if "/fold_0/" not in first:
        raise RuntimeError(f"Cannot derive five-fold {artifact} pattern")
    pattern = first.replace("/fold_0/", "/fold_{fold}/")
    for fold in range(5):
        if pattern.format(fold=fold) != receipt["fold_artifacts"][str(fold)][artifact]["path"]:
            raise RuntimeError(f"Five-fold {artifact} paths do not share one pattern")
    return pattern


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--source-static-auth", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("C graph construction must run on host 149")
    if args.fold not in range(5):
        raise RuntimeError("C graph fold must be 0..4")
    authority_root = args.authority_root.resolve(strict=True)
    prepared_root = args.prepared_root.resolve(strict=True)
    receipt_path = authority_root / "GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_sha = sha256(receipt_path)
    summary = json.loads((authority_root / "PHASE7_TYPED_AUTHORITY_SUMMARY.json").read_text())
    if summary.get("receipt_sha256") != receipt_sha or summary.get("relation_schema_generation") != TYPED_GENERATION:
        raise RuntimeError("C authority summary is not bound to the typed receipt")
    parent_auth = json.loads(args.source_static_auth.read_text(encoding="utf-8"))
    if parent_auth.get("host") != "149" or parent_auth.get("prepared_root") != str(prepared_root):
        raise RuntimeError("A prepared authority host or root mismatch")
    parents = parent_auth["variants"]["G2"]["fold_inputs"]
    matches = [row for row in parents if int(row["fold"]) == args.fold]
    if len(matches) != 1:
        raise RuntimeError("A prepared authority lacks one G2 parent fold")
    parent = matches[0]
    parent_path = prepared_root / "G2" / f"PATIENT_FOLD_{args.fold}.pt"
    if parent.get("path") != str(parent_path) or parent_path.stat().st_size != parent.get("bytes"):
        raise RuntimeError("A prepared parent path or size drift")
    old_binding = json.loads(
        (prepared_root / "GRAPH_AUTHORITIES" / f"PATIENT_FOLD_{args.fold}.json").read_text()
    )
    audit = validate_frozen_v32_patient_fold_binding(
        prepared_root / "SAMPLE_PATIENT_FOLD_MAP.tsv",
        prepared_root / "PATIENT_FOLD_AUTHORITY_RECEIPT.json",
    )
    fold_map = pd.read_csv(
        prepared_root / "SAMPLE_PATIENT_FOLD_MAP.tsv", sep="\t",
        dtype={"cancer_id": str, "sample_id": str, "patient_id": str},
    )
    artifacts = receipt["artifacts"]
    graph_inputs = load_formal_graph_input_authority(
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha,
        patient_authority_audit=audit,
        static_inputs={name: (record["path"], record["sha256"])
                       for name, record in artifacts.items()},
        fold_expression_pattern=fold_pattern(receipt, "train_expression"),
        fold_coexpression_pattern=fold_pattern(receipt, "train_coexpression"),
        relation_schema_generation=TYPED_GENERATION,
    )
    candidates = pd.read_parquet(prepared_root / "FORMAL_CANDIDATE_UNIVERSE.parquet")
    split = assign_outer_split(fold_map, args.fold, n_folds=5, validation_offset=1)
    bound = build_bound_formal_graph(
        graph_inputs, outer_fold=args.fold, split_manifest=split,
        candidate_pairs=candidates,
        binding_materializer=materialize_typed_lnc_protein_binding,
        binding_generation=TYPED_GENERATION,
    )
    expected = summary["folds"][str(args.fold)]
    if bound.authority.manifest["variant_edge_counts"] != expected["variant_edge_counts"]:
        raise RuntimeError("C graph edge counts differ from the frozen v2 summary")
    new_node_sha = bound.binding["graph"]["node_sha256"]
    if new_node_sha != old_binding["graph"]["node_sha256"]:
        raise RuntimeError("C node SHA256 differs from A; candidate indices cannot be reused")
    bundle = build_variant_runtime_bundle(bound.authority, "G2")
    binding = bind_formal_graph_variant(bound.binding, bound.authority, "G2")
    validate_formal_graph_payload_binding(binding, outer_fold=args.fold, variant="G2", bundle=bundle)
    relations = bundle.edges.relation_type.astype(str).value_counts()
    for relation, count in expected["binding_relations"].items():
        if int(relations.get(relation, 0)) != int(count):
            raise RuntimeError(f"C graph {relation} count differs from the v2 summary")
    if int(relations.get("binds_protein_predicted", 0)) or int(relations.get("binds_protein", 0)):
        raise RuntimeError("C graph contains predicted or legacy flat binding edges")
    print(json.dumps({
        "status": "PASS_C_GRAPH_IN_MEMORY", "host": "149", "fold": args.fold,
        "node_sha256": new_node_sha,
        "graph_receipt_sha256": receipt_sha,
        "variant_edge_count": len(bundle.edges),
        "eclip_edges": int(relations.get("binds_protein_eclip", 0)),
        "source_prepared_path": str(parent_path),
        "source_prepared_sha256": parent["sha256"],
        "wrote_sidecar": not args.check_only,
    }), flush=True)
    if args.check_only:
        return 0
    import torch

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / f"C_G2_PATIENT_FOLD_{args.fold}.pt"
    receipt_out = output_root / f"C_G2_PATIENT_FOLD_{args.fold}.json"
    if target.exists() or receipt_out.exists():
        raise FileExistsError("C graph overlay output already exists; refusing overwrite")
    payload = {
        "format": OVERLAY_FORMAT,
        "patient_fold": args.fold,
        "formal_graph_variant": "G2",
        "source_prepared_sha256": parent["sha256"],
        "graph_authority_receipt_sha256": receipt_sha,
        "formal_graph_authority": binding,
        "bundle": bundle,
    }
    temporary = output_root / f".{target.name}.{os.getpid()}.tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    sidecar_receipt = {
        "format": OVERLAY_FORMAT,
        "host": "149", "fold": args.fold, "variant": "G2",
        "path": str(target), "sha256": sha256(target),
        "bytes": target.stat().st_size,
        "source_prepared_path": str(parent_path),
        "source_prepared_sha256": parent["sha256"],
        "graph_authority_receipt_sha256": receipt_sha,
        "node_sha256": new_node_sha,
        "active_eclip_edges": int(relations.get("binds_protein_eclip", 0)),
        "predicted_edges": 0,
    }
    receipt_out.write_text(json.dumps(sidecar_receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "WROTE_C_GRAPH_SIDECAR", **sidecar_receipt}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
