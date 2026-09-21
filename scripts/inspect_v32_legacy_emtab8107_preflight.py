#!/usr/bin/env python3
"""Read-only preflight for E-MTAB-8107 legacy formal promotions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd

from materialize_v32_tgct_gse261811_rescue import canonical_gene_id


def decode(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--sdrf", type=Path, required=True)
    parser.add_argument("--lncrna-ids-json", type=Path, required=True)
    args = parser.parse_args()
    formal_lnc = {canonical_gene_id(value) for value in json.loads(args.lncrna_ids_json.read_text())}
    sdrf = pd.read_csv(args.sdrf, sep="\t")
    sdrf["source_sample"] = sdrf["Derived Array Data File"].astype(str).str.removesuffix(".counts.csv")
    mapping = sdrf.drop_duplicates("source_sample").set_index("source_sample")
    results = []
    for cancer, disease in (("BRCA", "breast cancer"), ("COAD", "colorectal cancer"), ("OV", "ovarian cancer")):
        metadata = pd.read_csv(args.metadata_root / f"{cancer}_cell_metadata_rescue_candidate.tsv.gz", sep="\t", low_memory=False)
        metadata["source_sample"] = metadata.patient_id_candidate.astype(str).str.removeprefix("2_pancancer_")
        if cancer == "BRCA":
            metadata = metadata.loc[~metadata.source_sample.eq("scrJUQ058")].copy()
        local = mapping.loc[mapping["Characteristics[disease]"].astype(str).str.lower().eq(disease)]
        metadata["individual"] = metadata.source_sample.map(local["Characteristics[individual]"].astype(str))
        metadata["sampling_site"] = metadata.source_sample.map(local["Characteristics[sampling site]"].astype(str))
        tumor = metadata.loc[~metadata.sampling_site.astype(str).str.lower().str.contains("normal")]
        contexts = tumor.groupby("cell_type_major_candidate").agg(cells=("cell_id", "size"), donors=("individual", "nunique")).reset_index()
        with h5py.File(args.h5_root / cancer / "raw_feature_bc_matrix.h5", "r") as handle:
            features = handle["matrix/features"]
            ids = decode(features["id"][:] if "id" in features else features["name"][:])
            names = decode(features["name"][:] if "name" in features else features["id"][:])
        results.append({
            "cancer": cancer,
            "cells": len(metadata),
            "source_samples": metadata.source_sample.nunique(),
            "patients": metadata.individual.nunique(),
            "tumor_patients": tumor.individual.nunique(),
            "unmapped_cells": int(metadata.individual.isna().sum()),
            "feature_id_examples": ids[:5],
            "feature_name_examples": names[:5],
            "lncrna_overlap_ids": len({canonical_gene_id(value) for value in ids} & formal_lnc),
            "lncrna_overlap_names": len({canonical_gene_id(value) for value in names} & formal_lnc),
            "contexts": contexts.to_dict(orient="records"),
        })
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
