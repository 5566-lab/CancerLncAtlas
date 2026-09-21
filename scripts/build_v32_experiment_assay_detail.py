#!/usr/bin/env python3
"""Build the traceable V3.2 perturbation assay-detail sidecar."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cc_hhgt.v32.experiment_assay_detail import (
    fetch_pubmed_records,
    materialise_assay_detail,
    write_pubmed_cache,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-event",
        type=Path,
        default=ROOT.parent / "processed/evidence_event.parquet",
    )
    parser.add_argument(
        "--ncpath",
        type=Path,
        default=(
            ROOT
            / "inputs/v32_full_multitask/evidence_sources/ncpath/extracted/lncRNA_interaction.txt.gz"
        ),
    )
    parser.add_argument(
        "--dim-lncrna",
        type=Path,
        default=(
            ROOT.parent
            / "CC_HHGT_v2_8_gdc_star/input_snapshot/processed/dimensions/dim_lncRNA.parquet"
        ),
    )
    parser.add_argument(
        "--dim-gene",
        type=Path,
        default=(
            ROOT.parent
            / "CC_HHGT_v2_8_gdc_star/input_snapshot/processed/dimensions/dim_gene.parquet"
        ),
    )
    parser.add_argument(
        "--pubmed-cache",
        type=Path,
        default=(
            ROOT
            / "inputs/v32_full_multitask/evidence_sources/pubmed/functional_perturbation_title_abstract.jsonl"
        ),
    )
    parser.add_argument("--fetch-pubmed", action="store_true")
    parser.add_argument("--email")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fetch_pubmed:
        evidence = pd.read_parquet(
            args.evidence_event, columns=["pmid", "experiment_family"]
        )
        selected = evidence.loc[
            evidence.experiment_family.astype(str).str.lower().str.contains("perturb")
        ]
        pmids = sorted(
            {
                str(value).strip()
                for value in selected.pmid
                if str(value).strip().isdigit()
            }
        )
        records = fetch_pubmed_records(pmids, email=args.email)
        write_pubmed_cache(records, args.pubmed_cache)
    cache = args.pubmed_cache if args.pubmed_cache.is_file() else None
    result = materialise_assay_detail(
        evidence_event_path=args.evidence_event,
        ncpath_path=args.ncpath,
        dim_lncrna_path=args.dim_lncrna,
        dim_gene_path=args.dim_gene,
        output_dir=args.output_dir,
        pubmed_cache_path=cache,
    )
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
