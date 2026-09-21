#!/usr/bin/env python3
"""Read-only raw-H5 recomputation for the V3.2 33-cancer lncRNA audit.

This standalone program is streamed to ``python3 -`` on COMPUTE_HOST.  It
prints one JSON document to stdout and progress to stderr.  It never opens a
remote path for writing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import gzip
import hashlib
import json
import math
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
LONG_GTF = PROJECT_ROOT / "raw/gencode_v50/gencode.v50.long_noncoding_RNAs.gtf.gz"
CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)
PRIMARY_CHROMS = {*(f"chr{index}" for index in range(1, 23)), "chrX", "chrY"}
ENSG = re.compile(r"^(ENSG\d+)(?:\.\d+)?$", re.IGNORECASE)
ATTR = re.compile(r'(\w+) "([^"]*)"')
CHUNK_NNZ = 8_000_000


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def set_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(values))).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def audit_stable(value: Any) -> str | None:
    match = ENSG.fullmatch(str(value).strip())
    return match.group(1).upper() if match else None


def formal_stable(value: Any) -> str:
    text = str(value).strip()
    upper = text.upper()
    for prefix in ("GENE:", "PROTEIN:", "LNC:", "LNCRNA:"):
        if upper.startswith(prefix):
            text = text[len(prefix) :]
            break
    return re.sub(r"\.\d+$", "", text).upper()


def load_annotations() -> dict[str, Any]:
    long_ids: set[str] = set()
    long_symbols: dict[str, set[str]] = defaultdict(set)
    with gzip.open(LONG_GTF, "rt", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene" or fields[0] not in PRIMARY_CHROMS:
                continue
            attrs = dict(ATTR.findall(fields[8]))
            stable = audit_stable(attrs.get("gene_id", ""))
            gene_type = attrs.get("gene_type", attrs.get("gene_biotype", ""))
            if stable is None or gene_type != "lncRNA":
                continue
            long_ids.add(stable)
            symbol = attrs.get("gene_name", "").strip()
            if symbol:
                long_symbols[symbol].add(stable)
                long_symbols[symbol.upper()].add(stable)
    audit_symbol_map = {
        symbol: next(iter(values))
        for symbol, values in long_symbols.items()
        if len(values) == 1
    }

    full_by_id: dict[str, str] = {}
    full_symbols: dict[str, set[tuple[str, str]]] = defaultdict(set)
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
            gene_class = "lncRNA" if token == "lncrna" else (
                "protein_coding" if token == "protein_coding" else ""
            )
            symbol = attrs.get("gene_name", "").strip()
            if not stable or not symbol or not gene_class:
                continue
            previous = full_by_id.setdefault(stable, gene_class)
            if previous != gene_class:
                raise RuntimeError(f"GENCODE class conflict for {stable}")
            full_symbols[symbol].add((stable, gene_class))
    formal_symbol_map = {
        symbol: next(iter(values))
        for symbol, values in full_symbols.items()
        if len({stable for stable, _ in values}) == 1
    }
    full_lnc_ids = {stable for stable, value in full_by_id.items() if value == "lncRNA"}
    if full_lnc_ids != long_ids:
        raise RuntimeError(
            "Full and long-noncoding GENCODE v50 strict stable-ID sets differ: "
            f"full={len(full_lnc_ids)}, long={len(long_ids)}"
        )
    return {
        "strict_ids": full_lnc_ids,
        "audit_symbol_map": audit_symbol_map,
        "formal_by_id": full_by_id,
        "formal_symbol_map": formal_symbol_map,
        "full_gtf": {
            "path": str(FULL_GTF),
            "sha256": sha256_file(FULL_GTF),
            "size_bytes": FULL_GTF.stat().st_size,
        },
        "long_gtf": {
            "path": str(LONG_GTF),
            "sha256": sha256_file(LONG_GTF),
            "size_bytes": LONG_GTF.stat().st_size,
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
    id_key = next((key for key in ("id", "gene_ids", "ensembl_id") if key in feature_group), None)
    name_key = next((key for key in ("name", "gene_names", "symbol") if key in feature_group), None)
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
    ids: list[str],
    names: list[str],
    annotation: dict[str, Any],
) -> tuple[list[str | None], list[str | None], Counter[str], Counter[str]]:
    strict_ids = annotation["strict_ids"]
    audit_symbols = annotation["audit_symbol_map"]
    formal_by_id = annotation["formal_by_id"]
    formal_symbols = annotation["formal_symbol_map"]
    audit_values: list[str | None] = []
    formal_values: list[str | None] = []
    audit_methods: Counter[str] = Counter()
    formal_methods: Counter[str] = Counter()
    for raw_id, raw_name in zip(ids, names, strict=True):
        direct_id = audit_stable(raw_id)
        direct_name = audit_stable(raw_name)
        if direct_id in strict_ids:
            audit_value, audit_method = direct_id, "feature_id_stable_ensembl"
        elif direct_name in strict_ids:
            audit_value, audit_method = direct_name, "feature_name_stable_ensembl"
        else:
            audit_value = audit_symbols.get(raw_name) or audit_symbols.get(raw_name.upper())
            audit_method = "unambiguous_lnc_only_symbol" if audit_value else "unmapped"

        formal_id = formal_stable(raw_id)
        formal_name = formal_stable(raw_name)
        if formal_by_id.get(formal_id) == "lncRNA":
            formal_value, formal_method = formal_id, "feature_id_stable_ensembl"
        elif formal_by_id.get(formal_name) == "lncRNA":
            formal_value, formal_method = formal_name, "feature_name_stable_ensembl"
        else:
            symbol = formal_symbols.get(raw_name)
            formal_value = symbol[0] if symbol and symbol[1] == "lncRNA" else None
            formal_method = "unique_full_annotation_symbol" if formal_value else "unmapped"
        audit_values.append(audit_value)
        formal_values.append(formal_value)
        audit_methods[audit_method] += 1
        formal_methods[formal_method] += 1
    return audit_values, formal_values, audit_methods, formal_methods


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
        audit_map, formal_map, audit_methods, formal_methods = map_features(
            ids, names, annotation
        )
        matrix_group_path = str(group.name)
        positive_rows = np.zeros(n_features, dtype=bool)
        stored_zero = stored_negative = stored_nonfinite = 0
        min_index = n_features
        max_index = -1
        for start in range(0, nnz, CHUNK_NNZ):
            stop = min(start + CHUNK_NNZ, nnz)
            values = np.asarray(data[start:stop])
            rows = np.asarray(indices[start:stop], dtype=np.int64)
            if rows.size:
                min_index = min(min_index, int(rows.min()))
                max_index = max(max_index, int(rows.max()))
            finite = np.isfinite(values)
            positive = finite & (values > 0)
            if np.any(positive):
                positive_rows[np.unique(rows[positive])] = True
            stored_zero += int(np.count_nonzero(finite & (values == 0)))
            stored_negative += int(np.count_nonzero(finite & (values < 0)))
            stored_nonfinite += int(np.count_nonzero(~finite))
            print(
                f"H5_CHUNK\t{cancer}\t{stop}/{nnz}",
                file=sys.stderr,
                flush=True,
            )
        if min_index < 0 or max_index >= n_features:
            raise RuntimeError(f"Sparse feature indices out of range: {min_index}, {max_index}")
        audit_universe = {value for value in audit_map if value}
        formal_universe = {value for value in formal_map if value}
        audit_detected = {
            value for value, present in zip(audit_map, positive_rows, strict=True)
            if value and present
        }
        formal_detected = {
            value for value, present in zip(formal_map, positive_rows, strict=True)
            if value and present
        }
        duplicate_formal_ids = len(formal_map) - len({value for value in formal_map if value}) - formal_methods["unmapped"]
        matrix_dtype = str(data.dtype)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"H5 changed during scan: {cancer}")
    return {
        "cancer_id": cancer,
        "h5_path": str(path),
        "h5_sha256": sha256_file(path),
        "h5_size_bytes": int(after.st_size),
        "h5_mtime_utc": dt.datetime.fromtimestamp(
            after.st_mtime, dt.timezone.utc
        ).isoformat(),
        "matrix_group_path": matrix_group_path,
        "matrix_orientation": "features_by_cells_csc",
        "matrix_orientation_validated": True,
        "matrix_data_dtype": matrix_dtype,
        "feature_count": n_features,
        "cell_count": n_cells,
        "stored_nnz": nnz,
        "stored_zero_entries": stored_zero,
        "stored_negative_entries": stored_negative,
        "stored_nonfinite_entries": stored_nonfinite,
        "audit_policy_lncRNA_universe_count": len(audit_universe),
        "audit_policy_detected_lncRNA_count": len(audit_detected),
        "formal_policy_lncRNA_universe_count": len(formal_universe),
        "formal_policy_detected_lncRNA_count": len(formal_detected),
        "formal_policy_undetected_lncRNA_count": len(formal_universe - formal_detected),
        "audit_policy_universe_sha256": set_sha256(audit_universe),
        "audit_policy_detected_sha256": set_sha256(audit_detected),
        "formal_policy_universe_sha256": set_sha256(formal_universe),
        "formal_policy_detected_sha256": set_sha256(formal_detected),
        "formal_minus_audit_universe_count": len(formal_universe) - len(audit_universe),
        "formal_minus_audit_detected_count": len(formal_detected) - len(audit_detected),
        "formal_only_universe_count": len(formal_universe - audit_universe),
        "audit_only_universe_count": len(audit_universe - formal_universe),
        "duplicate_formal_mapped_feature_rows_beyond_unique_ids": duplicate_formal_ids,
        "audit_mapping_method_counts": dict(sorted(audit_methods.items())),
        "formal_mapping_method_counts": dict(sorted(formal_methods.items())),
        "zero_missing_policy": "ONLY_FINITE_STORED_VALUE_GT_0_COUNTS_AS_DETECTED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancers", default=",".join(CANCERS))
    args = parser.parse_args()
    selected = tuple(
        value.strip().upper()
        for value in args.cancers.split(",")
        if value.strip()
    )
    if not selected or set(selected) - set(CANCERS):
        raise SystemExit("Invalid --cancers scope")
    annotation = load_annotations()
    annotation_public = {
        "release": "GENCODE v50",
        "strict_primary_lncRNA_count": len(annotation["strict_ids"]),
        "strict_primary_lncRNA_set_sha256": set_sha256(annotation["strict_ids"]),
        "full_gtf": annotation["full_gtf"],
        "long_noncoding_gtf": annotation["long_gtf"],
        "full_and_long_strict_stable_id_sets_equal": True,
    }
    print(
        "ANNOTATION_JSON\t"
        + json.dumps(annotation_public, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    per_cancer = []
    for index, cancer in enumerate(selected, start=1):
        row = scan_cancer(cancer, annotation)
        per_cancer.append(row)
        print(
            "CANCER_JSON\t" + json.dumps(row, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        print(
            f"RAW_H5_AUDIT\t{index}/{len(selected)}\t{cancer}\t"
            f"cells={row['cell_count']}\tformal_features="
            f"{row['formal_policy_lncRNA_universe_count']}\tdetected="
            f"{row['formal_policy_detected_lncRNA_count']}",
            file=sys.stderr,
            flush=True,
        )
        gc.collect()
    payload = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_RAW_H5_READ_ONLY_REAUDIT_V1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "remote_host_role": "COMPUTE_HOST_READ_ONLY",
        "remote_writes_performed": False,
        "cancers": list(selected),
        "annotation": annotation_public,
        "statistics_definition": {
            "lncRNA_universe_count": (
                "unique strict GENCODE v50 stable lncRNA IDs present/mapped in H5"
            ),
            "detected_lncRNA_count": (
                "unique mapped stable IDs with at least one finite matrix value > 0"
            ),
            "deduplication": "strip version and deduplicate to stable ENSG ID",
            "sparse_orientation": (
                "feature-by-cell CSC required: len(indptr)=shape[1]+1"
            ),
        },
        "per_cancer": per_cancer,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
