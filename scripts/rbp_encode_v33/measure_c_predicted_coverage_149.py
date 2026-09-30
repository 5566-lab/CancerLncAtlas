#!/usr/bin/env python3
"""Count C binding coverage with and without predictions, without writing a graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code-root", type=Path, required=True)
    p.add_argument("--typed-summary", type=Path, required=True)
    p.add_argument("--candidate-universe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("C coverage measurement must run on host 149")
    if args.output.exists():
        raise FileExistsError("C coverage audit already exists; refusing overwrite")
    preflight = args.output.with_name("C_PREDICTED_COVERAGE_PREFLIGHT.json")
    if preflight.exists():
        raise FileExistsError("C coverage preflight already exists; refusing duplicate run")
    preflight.write_text(json.dumps({
        "format": "C_PREDICTED_COVERAGE_PREFLIGHT_V1", "target_host": "149",
        "workload": "CPU_ONLY_COUNT", "graph_changed": False,
        "paid_gpu_allowed": False,
    }, sort_keys=True) + "\n", encoding="utf-8")
    sys.path.insert(0, str(args.code_root.resolve(strict=True)))
    from cc_hhgt.v32.formal_graph import materialize_typed_lnc_protein_binding

    summary = json.loads(args.typed_summary.read_text(encoding="utf-8"))
    table = Path(summary["binding_table"]).resolve(strict=True)
    if sha256(table) != summary["binding_table_sha256"]:
        raise RuntimeError("C typed binding table differs from authority summary")
    typed = pd.read_parquet(table)
    candidate_path = args.candidate_universe.resolve(strict=True)
    candidate_sha = sha256(candidate_path)
    candidates = pd.read_parquet(candidate_path, columns=["lncrna_id"])
    candidate_lncs = set(candidates.lncrna_id.astype(str))
    nonpred = materialize_typed_lnc_protein_binding(
        typed, include_predicted=False, candidate_lncrnas=candidate_lncs,
    )
    withpred = materialize_typed_lnc_protein_binding(
        typed, include_predicted=True, candidate_lncrnas=candidate_lncs,
    )
    counts = {str(k): int(v) for k, v in nonpred.relation_type.value_counts().items()}
    expected = summary["folds"]["0"]["binding_relations"]
    if counts != expected:
        raise RuntimeError("Measured C binding relations differ from fold-0 authority")
    predicted = withpred.loc[withpred.relation_type.eq("binds_protein_predicted")]
    nonpred_lncs = set(nonpred.source_id.astype(str))
    nonpred_proteins = set(nonpred.target_id.astype(str))
    pred_lncs = set(predicted.source_id.astype(str))
    pred_proteins = set(predicted.target_id.astype(str))
    nonpred_pairs = set(zip(nonpred.source_id.astype(str), nonpred.target_id.astype(str)))
    pred_pairs = set(zip(predicted.source_id.astype(str), predicted.target_id.astype(str)))
    result = {
        "format": "C_GRAPH_PREDICTED_COVERAGE_AUDIT_V1",
        "target_host": "149", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "graph_changed": False, "gpu_started": False,
        "typed_binding_table": str(table),
        "typed_binding_sha256": summary["binding_table_sha256"],
        "candidate_universe": str(candidate_path),
        "candidate_universe_sha256": candidate_sha,
        "candidate_lncrnas": len(candidate_lncs),
        "nonpredicted_edges": len(nonpred),
        "predicted_edges_if_admitted": len(predicted),
        "nonpredicted_relation_counts": counts,
        "nonpredicted_lncrnas": len(nonpred_lncs),
        "predicted_only_lncrnas": len(pred_lncs - nonpred_lncs),
        "predicted_only_lncrnas_fraction_of_candidates": (
            len(pred_lncs - nonpred_lncs) / max(len(candidate_lncs), 1)
        ),
        "nonpredicted_proteins": len(nonpred_proteins),
        "predicted_only_proteins": len(pred_proteins - nonpred_proteins),
        "nonpredicted_pairs": len(nonpred_pairs),
        "predicted_only_pairs": len(pred_pairs - nonpred_pairs),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
