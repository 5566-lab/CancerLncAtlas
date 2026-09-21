#!/usr/bin/env python3
"""Read-only explicit stable-ID scan for the V3.2 33-cancer H5 collection.

This standalone program is streamed to ``python3 -`` on COMPUTE_HOST.  It
prints one annotation record and one cancer record at a time to stdout.  It
never opens a remote path for writing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


PROJECT_ROOT = Path("./data/CancerLncAtlas")
SC_ROOT = PROJECT_ROOT / "processed/sc_tool_input"
FULL_GTF = PROJECT_ROOT / "raw/gencode_v50/gencode.v50.annotation.gtf.gz"
CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)
PRIMARY_CHROMS = {*(f"chr{index}" for index in range(1, 23)), "chrX", "chrY"}
ATTR = re.compile(r'(\w+) "([^"]*)"')
CHUNK_NNZ = 16_000_000


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def set_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(values))).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def formal_stable(value: Any) -> str:
    text = str(value).strip()
    upper = text.upper()
    for prefix in ("GENE:", "PROTEIN:", "LNC:", "LNCRNA:"):
        if upper.startswith(prefix):
            text = text[len(prefix) :]
            break
    return re.sub(r"\.\d+$", "", text).upper()


def load_annotation() -> dict[str, Any]:
    by_id: dict[str, str] = {}
    symbols: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with gzip.open(FULL_GTF, "rt", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene" or fields[0] not in PRIMARY_CHROMS:
                continue
            attrs = dict(ATTR.findall(fields[8]))
            stable = formal_stable(attrs.get("gene_id", ""))
            token = re.sub(
                r"[^a-z0-9]+",
                "_",
                attrs.get("gene_type", attrs.get("gene_biotype", "")).strip().lower(),
            ).strip("_")
            gene_class = (
                "lncRNA" if token == "lncrna" else
                "protein_coding" if token == "protein_coding" else ""
            )
            symbol = attrs.get("gene_name", "").strip()
            if not stable or not symbol or not gene_class:
                continue
            previous = by_id.setdefault(stable, gene_class)
            if previous != gene_class:
                raise RuntimeError(f"GENCODE class conflict for {stable}")
            symbols[symbol].add((stable, gene_class))
    symbol_map = {
        symbol: next(iter(values))
        for symbol, values in symbols.items()
        if len({stable for stable, _ in values}) == 1
    }
    strict_ids = {stable for stable, value in by_id.items() if value == "lncRNA"}
    if len(strict_ids) != 34_866:
        raise RuntimeError(f"Unexpected GENCODE v50 strict lncRNA count: {len(strict_ids)}")
    return {
        "by_id": by_id,
        "symbol_map": symbol_map,
        "strict_ids": strict_ids,
        "public": {
            "release": "GENCODE v50",
            "strict_primary_lncRNA_count": len(strict_ids),
            "strict_primary_lncRNA_set_sha256": set_sha256(strict_ids),
            "full_gtf": {
                "path": str(FULL_GTF),
                "sha256": sha256_file(FULL_GTF),
                "size_bytes": FULL_GTF.stat().st_size,
            },
            "mapping_policy": (
                "VERSION_STRIP;DIRECT_STABLE_ENSEMBL;"
                "EXACT_UNIQUE_FULL_ANNOTATION_SYMBOL;STABLE_ID_DEDUP"
            ),
        },
    }


def decode(dataset: h5py.Dataset) -> list[str]:
    return [
        value.decode("utf-8", errors="replace")
        if isinstance(value, (bytes, np.bytes_))
        else str(value)
        for value in dataset[()]
    ]


def locate_matrix(handle: h5py.File) -> h5py.Group:
    groups: list[h5py.Group] = []

    def visitor(_: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Group) and {
            "data", "indices", "indptr", "shape"
        }.issubset(obj.keys()):
            groups.append(obj)

    handle.visititems(visitor)
    if len(groups) != 1:
        raise RuntimeError(f"Expected one sparse matrix group, found {len(groups)}")
    return groups[0]


def feature_vectors(group: h5py.Group) -> tuple[list[str], list[str]]:
    feature_group = next(
        (
            group[key]
            for key in ("features", "gene", "genes")
            if key in group and isinstance(group[key], h5py.Group)
        ),
        None,
    )
    if feature_group is None:
        raise RuntimeError("Sparse H5 lacks a features group")
    id_key = next(
        (key for key in ("id", "gene_ids", "ensembl_id") if key in feature_group),
        None,
    )
    name_key = next(
        (key for key in ("name", "gene_names", "symbol") if key in feature_group),
        None,
    )
    if id_key is None and name_key is None:
        raise RuntimeError("Feature ID and name are both unavailable")
    ids = decode(feature_group[id_key]) if id_key else []
    names = decode(feature_group[name_key]) if name_key else []
    ids = ids or list(names)
    names = names or list(ids)
    if len(ids) != len(names):
        raise RuntimeError("Feature ID/name lengths disagree")
    return ids, names


def map_features(
    ids: list[str], names: list[str], annotation: dict[str, Any]
) -> tuple[list[str | None], Counter[str]]:
    by_id = annotation["by_id"]
    symbols = annotation["symbol_map"]
    mapped: list[str | None] = []
    methods: Counter[str] = Counter()
    for raw_id, raw_name in zip(ids, names, strict=True):
        stable_id = formal_stable(raw_id)
        stable_name = formal_stable(raw_name)
        if by_id.get(stable_id) == "lncRNA":
            value, method = stable_id, "feature_id_stable_ensembl"
        elif by_id.get(stable_name) == "lncRNA":
            value, method = stable_name, "feature_name_stable_ensembl"
        else:
            symbol = symbols.get(raw_name)
            value = symbol[0] if symbol and symbol[1] == "lncRNA" else None
            method = "unique_full_annotation_symbol" if value else "unmapped"
        mapped.append(value)
        methods[method] += 1
    return mapped, methods


def scan_cancer(cancer: str, annotation: dict[str, Any]) -> dict[str, Any]:
    path = SC_ROOT / cancer / "raw_feature_bc_matrix.h5"
    before = path.stat()
    with h5py.File(path, "r") as handle:
        group = locate_matrix(handle)
        shape = tuple(int(value) for value in group["shape"][()])
        if len(shape) != 2:
            raise RuntimeError(f"Non-matrix sparse shape: {shape}")
        n_features, n_cells = shape
        indptr = np.asarray(group["indptr"][()], dtype=np.int64)
        data = group["data"]
        indices = group["indices"]
        nnz = int(data.shape[0])
        if (
            len(indptr) != n_cells + 1
            or indptr[0] != 0
            or indptr[-1] != nnz
            or np.any(indptr[1:] < indptr[:-1])
            or int(indices.shape[0]) != nnz
        ):
            raise RuntimeError(
                f"Not a valid feature-by-cell CSC matrix: shape={shape}, "
                f"indptr={len(indptr)}, nnz={nnz}"
            )
        ids, names = feature_vectors(group)
        if len(ids) != n_features:
            raise RuntimeError("Feature vector length disagrees with matrix rows")
        if "barcodes" in group and int(group["barcodes"].shape[0]) != n_cells:
            raise RuntimeError("Barcode count disagrees with matrix columns")
        mapped, methods = map_features(ids, names, annotation)
        mapped_row = np.fromiter(
            (value is not None for value in mapped), dtype=bool, count=n_features
        )
        positive_rows = np.zeros(n_features, dtype=bool)
        min_index = n_features
        max_index = -1
        for start in range(0, nnz, CHUNK_NNZ):
            stop = min(start + CHUNK_NNZ, nnz)
            rows = np.asarray(indices[start:stop])
            values = np.asarray(data[start:stop])
            if rows.size:
                min_index = min(min_index, int(rows.min()))
                max_index = max(max_index, int(rows.max()))
                relevant = mapped_row[rows]
                positive = relevant & np.isfinite(values) & (values > 0)
                if np.any(positive):
                    positive_rows[np.unique(rows[positive])] = True
            print(
                f"EXPLICIT_ID_CHUNK\t{cancer}\t{stop}/{nnz}",
                file=sys.stderr,
                flush=True,
            )
        if min_index < 0 or max_index >= n_features:
            raise RuntimeError(f"Sparse feature indices out of range: {min_index}, {max_index}")
        universe = sorted({value for value in mapped if value})
        detected = sorted(
            {
                value
                for value, present in zip(mapped, positive_rows, strict=True)
                if value and present
            }
        )
        matrix_group_path = str(group.name)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"H5 changed during explicit-ID scan: {cancer}")
    return {
        "cancer_id": cancer,
        "h5_path": str(path),
        "h5_size_bytes": int(after.st_size),
        "h5_mtime_utc": dt.datetime.fromtimestamp(
            after.st_mtime, dt.timezone.utc
        ).isoformat(),
        "matrix_group_path": matrix_group_path,
        "matrix_orientation": "features_by_cells_csc",
        "matrix_orientation_validated": True,
        "feature_count": n_features,
        "cell_count": n_cells,
        "stored_nnz": nnz,
        "lncRNA_universe_count": len(universe),
        "detected_lncRNA_count": len(detected),
        "lncRNA_universe_set_sha256": set_sha256(universe),
        "detected_lncRNA_set_sha256": set_sha256(detected),
        "lncRNA_universe_ids": universe,
        "detected_lncRNA_ids": detected,
        "mapping_method_counts": dict(sorted(methods.items())),
        "detection_value_threshold": "FINITE_VALUE_GT_0",
        "minimum_detected_cell_count": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancers", default=",".join(CANCERS))
    args = parser.parse_args()
    selected = tuple(
        value.strip().upper() for value in args.cancers.split(",") if value.strip()
    )
    if not selected or set(selected) - set(CANCERS) or len(set(selected)) != len(selected):
        raise SystemExit("Invalid --cancers scope")
    annotation = load_annotation()
    print(
        "ANNOTATION_JSON\t"
        + json.dumps(annotation["public"], ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    for index, cancer in enumerate(selected, start=1):
        row = scan_cancer(cancer, annotation)
        print(
            "CANCER_JSON\t" + json.dumps(row, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        print(
            f"EXPLICIT_ID_AUDIT\t{index}/{len(selected)}\t{cancer}\t"
            f"universe={row['lncRNA_universe_count']}\t"
            f"detected={row['detected_lncRNA_count']}",
            file=sys.stderr,
            flush=True,
        )
        gc.collect()
    print(
        "COMPLETE_JSON\t"
        + json.dumps(
            {
                "cancers": list(selected),
                "remote_writes_performed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
