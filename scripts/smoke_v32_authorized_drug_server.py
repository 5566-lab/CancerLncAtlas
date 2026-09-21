#!/usr/bin/env python3
"""Run a real available/typed-absent smoke against an authorized Drug bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cc_hhgt.v32.drug_sparse_query import (
    STRICT_VALIDATION_STRATEGY,
    _available_partition_groups,
    load_drug_sparse_query_bundle,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite smoke receipt: {output}")
    bundle = load_drug_sparse_query_bundle(
        args.manifest, expected_manifest_sha256=args.sha256
    )
    groups = _available_partition_groups(bundle.paths["available_predictions"])
    physical_parts = {item for files in groups.values() for item in files}
    if len(physical_parts) != 694:
        raise RuntimeError(
            f"Formal Drug runtime requires all 694 physical partitions, got {len(physical_parts)}"
        )
    available = bundle.resolve(
        "BLCA", "LNC:ENSG00000099869", "DRUG:02ebaf6144107cac"
    )
    absent = bundle.resolve(
        "BLCA", "LNC:ENSG00000099869", "STRICT_AUDIT_UNKNOWN_DRUG"
    )
    if available["availability"] is not True:
        raise RuntimeError("Known available Drug query did not resolve")
    if absent["availability"] is not False or absent["failure_reason"] != "OUTSIDE_CONCEPTUAL_UNIVERSE":
        raise RuntimeError("Known absent Drug query did not fail closed")
    payload = {
        "format": "CANCERLNCATLAS_V32_AUTHORIZED_DRUG_RUNTIME_SMOKE_V1",
        "status": "PASS",
        "manifest": {
            "path": str(Path(args.manifest).resolve()),
            "sha256": sha256_file(Path(args.manifest)),
        },
        "available_probe": available,
        "typed_absent_probe": absent,
        "strict_validation": {
            "strategy": STRICT_VALIDATION_STRATEGY,
            "semantic_sampling": False,
            "physical_partitions_validated": len(physical_parts),
            "logical_cancer_partitions_validated": sum(map(len, groups.values())),
            "cancers_validated": len(groups),
            "available_rows_validated": bundle.manifest["available_rows"],
            "conceptual_candidate_rows_validated": bundle.manifest[
                "conceptual_candidate_rows"
            ],
            "five_fold_support_validated": True,
            "cross_file_key_uniqueness_validated_per_cancer": True,
            "all_artifact_hashes_and_rows_validated": True,
        },
        "scientific_status": "diagnostic_only",
        "evidence_scope": "CELL_LINE_ASSOCIATION_NOT_TCGA_PATIENT_RESPONSE",
        "changes_primary_ranking": False,
        "production_deployed": False,
        "release_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with output.open("xb") as stream:
        stream.write(encoded)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
