#!/usr/bin/env python3
"""Derive deterministic transcript-level promoter BED from GENCODE GTF.

The promoter definition follows Corces et al. (Science 2018): positions
-1000 through +100 bp relative to each transcription start site, inclusive.
Output uses BED 0-based, half-open coordinates and deterministic gzip mtime=0.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


REFERENCE_CHROMS = tuple([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"])
CHROM_ORDER = {chrom: idx for idx, chrom in enumerate(REFERENCE_CHROMS)}


def parse_attrs(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for field in raw.rstrip(";\n").split("; "):
        if not field:
            continue
        key, _, value = field.partition(" ")
        attrs[key] = value.strip().strip('"')
    return attrs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()

    rows: list[tuple[int, int, int, str, str, str, str, str, str]] = []
    with gzip.open(args.gtf, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "transcript":
                continue
            chrom, _, _, start_raw, end_raw, _, strand, _, attrs_raw = fields
            if chrom not in CHROM_ORDER or strand not in {"+", "-"}:
                continue
            start_1 = int(start_raw)
            end_1 = int(end_raw)
            tss0 = start_1 - 1 if strand == "+" else end_1 - 1
            if strand == "+":
                promoter_start = max(0, tss0 - 1000)
                promoter_end = tss0 + 101
            else:
                promoter_start = max(0, tss0 - 100)
                promoter_end = tss0 + 1001
            attrs = parse_attrs(attrs_raw)
            transcript_id = attrs.get("transcript_id", "")
            gene_id = attrs.get("gene_id", "")
            gene_name = attrs.get("gene_name", "")
            gene_type = attrs.get("gene_type", attrs.get("gene_biotype", ""))
            rows.append(
                (
                    CHROM_ORDER[chrom],
                    promoter_start,
                    promoter_end,
                    chrom,
                    transcript_id,
                    gene_id,
                    gene_name,
                    gene_type,
                    strand,
                )
            )

    rows.sort(key=lambda row: (row[0], row[1], row[2], row[4]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            header = (
                "#chrom\tstart\tend\ttranscript_id\tgene_id\tgene_name\tgene_type\tstrand\t"
                "definition=-1000_to_+100_inclusive_from_TSS;coordinates=BED0_half_open\n"
            )
            zipped.write(header.encode("utf-8"))
            for _, start, end, chrom, transcript_id, gene_id, gene_name, gene_type, strand in rows:
                zipped.write(
                    f"{chrom}\t{start}\t{end}\t{transcript_id}\t{gene_id}\t{gene_name}\t{gene_type}\t{strand}\n".encode(
                        "utf-8"
                    )
                )

    metadata = {
        "source_gtf": str(args.gtf.resolve()),
        "source_gtf_sha256": sha256(args.gtf),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "n_transcript_promoters": len(rows),
        "reference_chromosomes": list(REFERENCE_CHROMS),
        "feature": "transcript",
        "promoter_definition": "-1000 through +100 bp relative to strand-aware TSS, inclusive",
        "coordinate_system": "BED 0-based half-open",
        "gzip_mtime": 0,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
