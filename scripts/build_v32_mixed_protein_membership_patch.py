#!/usr/bin/env python3
"""Materialize a protein-only exact-pathway membership for mixed ORA.

The legacy staging membership contains lncRNA identifiers in addition to
protein-coding genes.  Mixed-list ORA is defined over protein genes, so this
creates a new immutable, query-only annotation filtered against the registered
identifier map and the current exact-pathway association universe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.input_lineage import artifact_sha256  # noqa: E402


def build(*, membership: Path, identifier_map: Path, association: Path, output: Path) -> dict[str, object]:
    membership = membership.resolve()
    identifier_map = identifier_map.resolve()
    association = association.resolve()
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite membership: {output}")
    raw = pd.read_parquet(membership, columns=["pathway_id", "gene_id"])
    ids = pd.read_parquet(identifier_map, columns=["canonical_id", "entity_type"])
    assoc = pd.read_parquet(association, columns=["pathway_id"])
    protein_ids = set(ids.loc[ids.entity_type.astype(str).eq("protein_coding_gene"), "canonical_id"].astype(str))
    pathways = set(assoc.pathway_id.astype(str))
    frame = raw.copy()
    frame["pathway_id"] = frame.pathway_id.astype(str).str.strip()
    frame["gene_id"] = frame.gene_id.astype(str).str.strip()
    frame = frame.loc[frame.pathway_id.isin(pathways) & frame.gene_id.isin(protein_ids)]
    frame = frame.drop_duplicates(["pathway_id", "gene_id"]).sort_values(["pathway_id", "gene_id"], kind="mergesort")
    if frame.empty or frame.pathway_id.nunique() != len(pathways):
        raise RuntimeError("Protein-only membership lost one or more exact pathways")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, compression="zstd")
    return {
        "status": "PASS",
        "rows": int(len(frame)),
        "pathways": int(frame.pathway_id.nunique()),
        "protein_genes": int(frame.gene_id.nunique()),
        "source_membership_sha256": artifact_sha256(membership),
        "source_identifier_map_sha256": artifact_sha256(identifier_map),
        "source_association_sha256": artifact_sha256(association),
        "output": str(output),
        "output_sha256": artifact_sha256(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--identifier-map", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(membership=args.membership, identifier_map=args.identifier_map, association=args.association, output=args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
