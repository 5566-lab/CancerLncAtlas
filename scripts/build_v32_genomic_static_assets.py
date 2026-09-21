#!/usr/bin/env python3
"""Build exact-pathway Ensembl membership and GRCh38 CNV intervals for V3.2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cc_hhgt.v32.input_lineage import artifact_sha256


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build(
    pathway_membership_path: Path,
    dim_gene_path: Path,
    lncrna_intervals_path: Path,
    output_root: Path,
) -> dict[str, object]:
    membership_raw = pd.read_parquet(pathway_membership_path)
    dim_gene = pd.read_parquet(dim_gene_path)
    lnc_raw = pd.read_parquet(lncrna_intervals_path)
    for required, frame, name in (
        ({"pathway_id", "gene_symbol"}, membership_raw, "pathway membership"),
        (
            {"ensembl_gene_id", "gene_symbol", "gene_type", "chromosome", "start", "end"},
            dim_gene,
            "gene dimension",
        ),
        ({"lncrna_id", "feature", "chromosome", "start0", "end0"}, lnc_raw, "lncRNA intervals"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} lacks required columns: {missing}")

    gene_map = dim_gene.loc[
        dim_gene.gene_symbol.notna() & dim_gene.ensembl_gene_id.notna(),
        ["gene_symbol", "ensembl_gene_id", "gene_type", "chromosome", "start", "end"],
    ].copy()
    # Exact pathway sources are symbol based.  Prefer protein-coding mappings;
    # a duplicated symbol may legitimately map to more than one current stable ID.
    coding = gene_map.gene_type.astype(str).str.lower().eq("protein_coding")
    coding_symbols = set(gene_map.loc[coding, "gene_symbol"].astype(str))
    gene_map = gene_map.loc[
        coding | ~gene_map.gene_symbol.astype(str).isin(coding_symbols)
    ].drop_duplicates(["gene_symbol", "ensembl_gene_id"])
    membership = membership_raw[["pathway_id", "gene_symbol"]].dropna().merge(
        gene_map[["gene_symbol", "ensembl_gene_id"]],
        on="gene_symbol",
        how="inner",
        validate="many_to_many",
    )
    membership = membership.rename(columns={"ensembl_gene_id": "gene_id"})[
        ["pathway_id", "gene_id"]
    ].drop_duplicates()
    if membership.empty:
        raise ValueError("No pathway genes mapped to current Ensembl IDs")

    pathway_gene_ids = set(membership.gene_id.astype(str))
    gene_intervals = gene_map.loc[
        gene_map.ensembl_gene_id.astype(str).isin(pathway_gene_ids),
        ["ensembl_gene_id", "chromosome", "start", "end"],
    ].rename(columns={"ensembl_gene_id": "entity_id"})
    gene_intervals.insert(0, "entity_type", "gene")
    lnc_intervals = lnc_raw.loc[
        lnc_raw.feature.astype(str).str.lower().eq("gene"),
        ["lncrna_id", "chromosome", "start0", "end0"],
    ].rename(
        columns={"lncrna_id": "entity_id", "start0": "start", "end0": "end"}
    )
    # GTF start0 is zero based; convert to the one-based coordinate convention
    # used by GDC segments.  Midpoint mapping is otherwise unchanged.
    lnc_intervals["start"] = pd.to_numeric(lnc_intervals.start, errors="raise") + 1
    lnc_intervals.insert(0, "entity_type", "lncrna")
    intervals = pd.concat([gene_intervals, lnc_intervals], ignore_index=True)
    intervals["chromosome"] = intervals.chromosome.astype(str).str.replace(
        r"^chr", "", regex=True
    )
    intervals["start"] = pd.to_numeric(intervals.start, errors="raise").astype("int64")
    intervals["end"] = pd.to_numeric(intervals.end, errors="raise").astype("int64")
    intervals = intervals.loc[intervals.end.ge(intervals.start)].drop_duplicates()

    output_root.mkdir(parents=True, exist_ok=True)
    membership_output = output_root / "exact_pathway_gene_membership_ensembl.parquet"
    interval_output = output_root / "grch38_gene_lncrna_intervals.parquet"
    _atomic_parquet(membership, membership_output)
    _atomic_parquet(intervals, interval_output)
    mapped_pathways = membership.pathway_id.nunique()
    source_pathways = membership_raw.pathway_id.nunique()
    lineage: dict[str, object] = {
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "artifact_role": "STATIC_ANNOTATION_ONLY",
        "model_result": False,
        "old_checkpoint_loaded": False,
        "old_predictions_used": False,
        "old_rankings_used": False,
        "reference_build": "GRCh38",
        "source_artifacts": [
            {"path": str(path.resolve()), "sha256": artifact_sha256(path)}
            for path in (pathway_membership_path, dim_gene_path, lncrna_intervals_path)
        ],
        "exact_membership": {
            "path": str(membership_output.resolve()),
            "sha256": artifact_sha256(membership_output),
            "rows": int(len(membership)),
            "mapped_pathways": int(mapped_pathways),
            "source_pathways": int(source_pathways),
        },
        "cnv_intervals": {
            "path": str(interval_output.resolve()),
            "sha256": artifact_sha256(interval_output),
            "rows": int(len(intervals)),
            "gene_rows": int(intervals.entity_type.eq("gene").sum()),
            "lncrna_rows": int(intervals.entity_type.eq("lncrna").sum()),
        },
    }
    _atomic_json(lineage, output_root / "GENOMIC_STATIC_ASSET_LINEAGE.json")
    return {
        "status": "SUCCESS",
        "membership": str(membership_output.resolve()),
        "intervals": str(interval_output.resolve()),
        "mapped_pathways": int(mapped_pathways),
        "source_pathways": int(source_pathways),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pathway-membership", type=Path, required=True)
    parser.add_argument("--dim-gene", type=Path, required=True)
    parser.add_argument("--lncrna-intervals", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(
        args.pathway_membership,
        args.dim_gene,
        args.lncrna_intervals,
        args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
