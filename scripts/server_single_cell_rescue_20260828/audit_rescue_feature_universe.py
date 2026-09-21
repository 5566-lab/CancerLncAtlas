#!/usr/bin/env python3
"""Map rescue RDS features to GENCODE v50 lncRNA genes without broadcasting symbols."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_ROOT = Path("./data/CancerLncAtlas")
ANNOTATION = Path("./data/CancerLncAtlas/raw/gencode_v50/gencode.v50.annotation.gtf.gz")
RUN_ID = "single_cell_rescue_20260828"
PRIMARY_CHROMS = {*(f"chr{index}" for index in range(1, 23)), "chrX", "chrY"}
ATTR = re.compile(r'(\w+) "([^"]*)"')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(value: str) -> str:
    text = value.strip()
    upper = text.upper()
    for prefix in ("GENE:", "PROTEIN:", "LNC:", "LNCRNA:"):
        if upper.startswith(prefix):
            text = text[len(prefix) :]
            break
    return re.sub(r"\.\d+$", "", text).upper()


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} {EXPECTED_ROOT}")
    root = Path(sys.argv[1]).resolve()
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing output root outside {EXPECTED_ROOT}: {root}")

    by_id: dict[str, str] = {}
    symbol_values: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with gzip.open(ANNOTATION, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene" or fields[0] not in PRIMARY_CHROMS:
                continue
            attrs = dict(ATTR.findall(fields[8]))
            stable = stable_id(attrs.get("gene_id", ""))
            token = re.sub(
                r"[^a-z0-9]+", "_",
                attrs.get("gene_type", attrs.get("gene_biotype", "")).strip().lower(),
            ).strip("_")
            gene_class = "lncRNA" if token == "lncrna" else "protein_coding" if token == "protein_coding" else ""
            symbol = attrs.get("gene_name", "").strip()
            if not stable or not symbol or not gene_class:
                continue
            previous = by_id.setdefault(stable, gene_class)
            if previous != gene_class:
                raise RuntimeError(f"GENCODE class conflict for {stable}")
            symbol_values[symbol].add((stable, gene_class))
    symbol_map = {
        symbol: next(iter(values))
        for symbol, values in symbol_values.items()
        if len({stable for stable, _ in values}) == 1
    }
    lnc_ids = {stable for stable, gene_class in by_id.items() if gene_class == "lncRNA"}
    if len(lnc_ids) != 34866:
        raise RuntimeError(f"Unexpected GENCODE v50 strict lncRNA count: {len(lnc_ids)}")

    rows: list[dict[str, object]] = []
    metadata_dir = root / "metadata" / RUN_ID
    for path in sorted(metadata_dir.glob("*_feature_count_signature.tsv.gz")):
        cancer = path.name.split("_", 1)[0]
        total = direct_id = unique_symbol = 0
        mapped_ids: set[str] = set()
        detected_ids: set[str] = set()
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle, delimiter="\t"):
                total += 1
                feature = record["feature_id"].strip()
                candidate = stable_id(feature)
                if by_id.get(candidate) == "lncRNA":
                    mapped_id, method = candidate, "direct"
                else:
                    symbol = symbol_map.get(feature)
                    mapped_id = symbol[0] if symbol and symbol[1] == "lncRNA" else None
                    method = "symbol" if mapped_id else "unmapped"
                if mapped_id:
                    mapped_ids.add(mapped_id)
                    direct_id += int(method == "direct")
                    unique_symbol += int(method == "symbol")
                    if float(record["n_cells_detected"]) > 0:
                        detected_ids.add(mapped_id)
        rows.append({
            "cancer_id": cancer,
            "source_feature_signature": str(path),
            "source_feature_signature_sha256": sha256(path),
            "total_features": total,
            "gencode_v50_lncrna_features": len(mapped_ids),
            "gencode_v50_lncrna_detected": len(detected_ids),
            "mapped_by_direct_stable_ensembl": direct_id,
            "mapped_by_exact_unique_symbol": unique_symbol,
            "mapping_contract": "VERSION_STRIP_DIRECT_STABLE_ENSEMBL_OR_EXACT_UNIQUE_FULL_ANNOTATION_SYMBOL_NO_BROADCAST",
        })

    out_dir = root / "manifests" / RUN_ID
    json_path = out_dir / "rescue_rds_lncrna_feature_universe.json"
    json_path.write_text(json.dumps({
        "format": "CANCERLNCATLAS_RESCUE_RDS_LNCRNA_FEATURE_UNIVERSE_V1",
        "annotation_path": str(ANNOTATION),
        "annotation_sha256": sha256(ANNOTATION),
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tsv_path = out_dir / "rescue_rds_lncrna_feature_universe.tsv"
    columns = list(rows[0]) if rows else []
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"RESULT_PATH\t{json_path}")
    print(f"RESULT_SHA256\t{sha256(json_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
