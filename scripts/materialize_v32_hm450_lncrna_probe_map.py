#!/usr/bin/env python3
"""Map the official GDC HM450 GRCh38 manifest to V3.2 lncRNA regions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import duckdb
import pandas as pd


EXPECTED_GDC_HM450_MD5 = "e163fc110043abb5a7ef623816383bb9"


def checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hm450-manifest", type=Path, required=True)
    parser.add_argument("--promoters", type=Path, required=True)
    parser.add_argument("--required-genes", type=Path, required=True)
    parser.add_argument("--peak-set", type=Path, required=True)
    parser.add_argument("--datas7-links", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: path.resolve(strict=True)
        for name, path in {
            "hm450_manifest": args.hm450_manifest,
            "promoters": args.promoters,
            "required_genes": args.required_genes,
            "peak_set": args.peak_set,
            "datas7_links": args.datas7_links,
        }.items()
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"HM450 lncRNA probe-map output reuse refused: {output}")
    observed_md5 = checksum(paths["hm450_manifest"], "md5")
    if observed_md5 != EXPECTED_GDC_HM450_MD5:
        raise RuntimeError("Official GDC HM450 manifest MD5 mismatch")

    probes = pd.read_csv(paths["hm450_manifest"], sep="\t", compression="gzip")
    required_probe_columns = {"CpG_chrm", "CpG_beg", "CpG_end", "probeID"}
    if missing := sorted(required_probe_columns - set(probes)):
        raise RuntimeError(f"GDC HM450 manifest lacks columns: {missing}")
    probes = probes[["CpG_chrm", "CpG_beg", "CpG_end", "probeID"]].copy()
    probes["CpG_beg"] = pd.to_numeric(probes["CpG_beg"], errors="coerce")
    probes["CpG_end"] = pd.to_numeric(probes["CpG_end"], errors="coerce")
    probes = probes.loc[
        probes["CpG_chrm"].astype(str).str.startswith("chr")
        & probes["CpG_beg"].ge(0)
        & probes["CpG_end"].gt(probes["CpG_beg"])
    ].copy()
    probes["chromosome"] = probes["CpG_chrm"].astype(str)
    probes["probe_start0"] = probes["CpG_beg"].astype("int64")
    probes["probe_end0"] = probes["CpG_end"].astype("int64")
    probes["probe_id"] = probes["probeID"].astype(str)
    probes = probes[["probe_id", "chromosome", "probe_start0", "probe_end0"]].drop_duplicates()
    if len(probes) < 450_000 or probes["probe_id"].duplicated().any():
        raise RuntimeError("GDC HM450 GRCh38 probe coordinates are incomplete or duplicated")

    required = pd.read_csv(paths["required_genes"], sep="\t", compression="gzip")
    if "gene_id" not in required:
        raise RuntimeError("Required lncRNA table lacks gene_id")
    required_ids = set(required["gene_id"].astype(str).str.split(".").str[0])
    if len(required_ids) < 1:
        raise RuntimeError("Required lncRNA set is empty")

    promoters = pd.read_csv(paths["promoters"], sep="\t", compression="gzip")
    promoters = promoters.rename(columns={"#chrom": "chromosome"})
    required_promoter_columns = {"chromosome", "start", "end", "gene_id"}
    if missing := sorted(required_promoter_columns - set(promoters)):
        raise RuntimeError(f"GENCODE promoter BED lacks columns: {missing}")
    promoters["gene_id"] = promoters["gene_id"].astype(str).str.split(".").str[0]
    promoters = promoters.loc[promoters["gene_id"].isin(required_ids)].copy()
    promoters["promoter_start0"] = pd.to_numeric(promoters["start"], errors="raise").astype("int64")
    promoters["promoter_end0"] = pd.to_numeric(promoters["end"], errors="raise").astype("int64")
    promoters = promoters[["chromosome", "promoter_start0", "promoter_end0", "gene_id"]].drop_duplicates()

    links = pd.read_csv(paths["datas7_links"], sep="\t", compression="gzip")
    if not {"lncrna_id", "peak_name"}.issubset(links):
        raise RuntimeError("DataS7 topology lacks lncRNA/peak keys")
    links["gene_id"] = links["lncrna_id"].astype(str).str.removeprefix("LNC:").str.split(".").str[0]
    links = links.loc[links["gene_id"].isin(required_ids), ["gene_id", "peak_name"]].drop_duplicates()
    peaks = pd.read_csv(paths["peak_set"], sep="\t")
    if not {"seqnames", "start", "end", "name"}.issubset(peaks):
        raise RuntimeError("ATAC peak set lacks coordinate keys")
    peaks = peaks.loc[peaks["name"].astype(str).isin(set(links["peak_name"].astype(str)))].copy()
    if set(links["peak_name"]) - set(peaks["name"].astype(str)):
        raise RuntimeError("DataS7 topology includes peaks absent from the official peak set")
    peaks["chromosome"] = peaks["seqnames"].astype(str)
    peaks["peak_start0"] = pd.to_numeric(peaks["start"], errors="raise").astype("int64") - 1
    peaks["peak_end0"] = pd.to_numeric(peaks["end"], errors="raise").astype("int64")
    distal = links.merge(
        peaks[["name", "chromosome", "peak_start0", "peak_end0"]],
        left_on="peak_name",
        right_on="name",
        validate="many_to_one",
    )[["chromosome", "peak_start0", "peak_end0", "gene_id", "peak_name"]]

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='32GB'")
    con.register("probes", probes)
    con.register("promoters", promoters)
    con.register("distal", distal)
    mapping = con.execute(
        """
        SELECT DISTINCT p.probe_id, 'LNC:' || r.gene_id AS lncrna_id,
               'promoter' AS region_type
        FROM probes p
        JOIN promoters r
          ON p.chromosome = r.chromosome
         AND p.probe_start0 < r.promoter_end0
         AND p.probe_end0 > r.promoter_start0
        UNION
        SELECT DISTINCT p.probe_id, 'LNC:' || r.gene_id AS lncrna_id,
               'distal' AS region_type
        FROM probes p
        JOIN distal r
          ON p.chromosome = r.chromosome
         AND p.probe_start0 < r.peak_end0
         AND p.probe_end0 > r.peak_start0
        """
    ).fetchdf()
    con.close()
    if mapping.empty or mapping.duplicated().any():
        raise RuntimeError("HM450 lncRNA region probe map is empty or duplicated")
    mapping = mapping.sort_values(["probe_id", "lncrna_id", "region_type"], kind="stable").reset_index(drop=True)
    output.mkdir(parents=True)
    target = output / "HM450_LNCRNA_REGION_PROBES.parquet"
    temporary = output / ".HM450_LNCRNA_REGION_PROBES.tmp.parquet"
    mapping.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target)
    by_region = mapping.groupby("region_type", observed=True).agg(
        mapping_rows=("probe_id", "size"),
        probes=("probe_id", "nunique"),
        lncrnas=("lncrna_id", "nunique"),
    ).reset_index()
    audit = {
        "format": "CC_HHGT_V3_2_HM450_LNCRNA_REGION_PROBE_MAP_V1",
        "status": "PASS_HM450_LNCRNA_REGION_PROBE_MAP",
        "coordinate_build": "GRCh38",
        "coordinate_contract": "BED0_HALF_OPEN_INTERVAL_OVERLAP",
        "promoter_definition": "GENCODE_V36_TRANSCRIPT_TSS_MINUS1000_PLUS100",
        "distal_definition": "DATAS7_STATIC_LINKED_TCGA_ATAC_PEAK",
        "gdc_hm450_manifest_md5": observed_md5,
        "valid_hm450_probes": int(probes.probe_id.nunique()),
        "required_lncrnas": len(required_ids),
        "mapping_rows": int(len(mapping)),
        "mapped_probes": int(mapping.probe_id.nunique()),
        "mapped_lncrnas": int(mapping.lncrna_id.nunique()),
        "by_region": by_region.to_dict(orient="records"),
        "inputs": {name: {"path": str(path), "sha256": checksum(path)} for name, path in paths.items()},
        "output": {"path": str(target), "sha256": checksum(target), "bytes": target.stat().st_size},
    }
    (output / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
