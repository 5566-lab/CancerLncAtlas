#!/usr/bin/env python3
"""Find a real formal key whose Drug absence is caused only by core coverage.

This is a read-only, streaming factor audit.  It avoids materialising the
62-million-row conceptual relation and does not use DuckDB or spill storage.
The emitted witness must still be checked through ``DrugSparseQueryBundle``
before it is accepted into the typed-probe contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_rows(
    path: Path, key_columns: tuple[str, ...]
) -> dict[tuple[str, ...], int]:
    columns = [*key_columns, "fold_id"]
    result: dict[tuple[str, ...], int] = defaultdict(int)
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=131_072, columns=columns
    ):
        values = batch.to_pydict()
        for index, fold in enumerate(values["fold_id"]):
            key = tuple(str(values[name][index]) for name in key_columns)
            result[key] |= 1 << int(fold)
    return dict(result)


def _edge_map(path: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = defaultdict(list)
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=131_072, columns=["pathway_id", "drug_id"]
    ):
        values = batch.to_pydict()
        for pathway, drug in zip(values["pathway_id"], values["drug_id"]):
            result[str(pathway)].append(str(drug))
    return {
        pathway: tuple(sorted(set(drugs)))
        for pathway, drugs in result.items()
    }


def _artifact_path(
    root: Path,
    manifest: dict[str, Any],
    role: str,
    allowed_roots: Iterable[Path] | None = None,
) -> Path:
    raw = str(manifest["artifacts"][role]["path"])
    declared = Path(raw)
    if not raw or declared.is_absolute() or ".." in declared.parts:
        raise RuntimeError(f"Artifact path is not canonical relative: {role}")
    path = (root / declared).resolve(strict=True)
    scope = tuple(allowed_roots or (root,))
    if not any(candidate == path or candidate in path.parents for candidate in scope):
        raise RuntimeError(f"Artifact escaped bundle root: {role}: {path}")
    current = root / declared
    while current != root:
        if current.is_symlink():
            raise RuntimeError(f"Factor path contains a symlink: {current}")
        current = current.parent
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Factor must be one non-symlink Parquet file: {path}")
    return path


def discover(
    manifest_path: Path,
    expected_sha256: str,
    *,
    allowed_roots: Iterable[Path] | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    expected = str(expected_sha256).lower()
    if SHA256.fullmatch(expected) is None:
        raise RuntimeError("Expected manifest SHA256 is invalid")
    observed = sha256_file(manifest_path)
    if observed != expected:
        raise RuntimeError(f"Manifest SHA256 mismatch: {observed} != {expected}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_run_status") != "SUCCESS":
        raise RuntimeError("Core-unavailable witness requires a SUCCESS bundle")
    root = (
        manifest_path.parent.resolve()
        if artifact_root is None
        else Path(artifact_root).resolve(strict=True)
    )
    allowed = tuple(Path(item).resolve() for item in (allowed_roots or (root,)))
    exact_path = _artifact_path(root, manifest, "exact_candidates", allowed)
    edge_path = _artifact_path(root, manifest, "pathway_drug_edges", allowed)
    assay_path = _artifact_path(root, manifest, "assay_fold_eligibility", allowed)
    expression_path = _artifact_path(
        root, manifest, "expression_fold_coverage", allowed
    )
    core_path = _artifact_path(root, manifest, "core_entity_availability", allowed)

    edges = _edge_map(edge_path)
    assay_masks = _mask_rows(assay_path, ("cancer_id", "drug_id"))
    expression_masks = _mask_rows(
        expression_path, ("cancer_id", "lncrna_id")
    )
    core_masks = _mask_rows(core_path, ("entity_type", "entity_id"))

    rows_examined = 0
    conceptual_pairs_examined = 0
    for batch in pq.ParquetFile(exact_path).iter_batches(
        batch_size=65_536,
        columns=["cancer_id", "lncrna_id", "pathway_id"],
    ):
        values = batch.to_pydict()
        for cancer_value, lnc_value, pathway_value in zip(
            values["cancer_id"], values["lncrna_id"], values["pathway_id"]
        ):
            rows_examined += 1
            cancer = str(cancer_value)
            lncrna = str(lnc_value)
            pathway = str(pathway_value)
            expression_mask = expression_masks.get((cancer, lncrna), 0)
            if not expression_mask:
                continue
            cancer_core = core_masks.get(("cancer", cancer), 0)
            lnc_core = core_masks.get(("lncRNA", lncrna), 0)
            for drug in edges.get(pathway, ()):
                assay_mask = assay_masks.get((cancer, drug), 0)
                eligible_mask = assay_mask & expression_mask
                if not eligible_mask:
                    continue
                conceptual_pairs_examined += 1
                drug_core = core_masks.get(("drug_target", drug), 0)
                fully_supported_core_mask = (
                    eligible_mask & cancer_core & lnc_core & drug_core
                )
                if not fully_supported_core_mask:
                    return {
                        "status": "FOUND",
                        "expected_failure_reason": "CURRENT_V32_CORE_UNAVAILABLE",
                        "manifest_sha256": observed,
                        "key": {
                            "cancer_id": cancer,
                            "lncrna_id": lncrna,
                            "drug_id": drug,
                        },
                        "witness_pathway_id": pathway,
                        "fold_masks": {
                            "assay": assay_mask,
                            "expression": expression_mask,
                            "eligible": eligible_mask,
                            "cancer_core": cancer_core,
                            "lncrna_core": lnc_core,
                            "drug_core": drug_core,
                            "fully_supported_core": fully_supported_core_mask,
                        },
                        "exact_rows_examined": rows_examined,
                        "eligible_conceptual_pairs_examined": conceptual_pairs_examined,
                        "semantic_sampling": False,
                    }
    return {
        "status": "UNREACHABLE_IN_FORMAL_FACTORS",
        "expected_failure_reason": "CURRENT_V32_CORE_UNAVAILABLE",
        "manifest_sha256": observed,
        "exact_rows_examined": rows_examined,
        "eligible_conceptual_pairs_examined": conceptual_pairs_examined,
        "semantic_sampling": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    payload = discover(
        Path(args.manifest),
        args.sha256,
        artifact_root=Path(args.artifact_root),
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "FOUND" else 2


if __name__ == "__main__":
    raise SystemExit(main())
