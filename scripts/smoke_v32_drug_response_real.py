#!/usr/bin/env python
"""Exercise one real available and one real typed-absent Drug R6 key."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.drug_sparse_query import (  # noqa: E402
    PROBABILITY_COLUMN,
    load_drug_sparse_query_bundle,
)
from cc_hhgt.v32.input_lineage import artifact_sha256  # noqa: E402


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing output reuse: {output}")

    bundle = load_drug_sparse_query_bundle(
        args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    available_path = bundle.paths["available_predictions"]
    seed_rows = ds.dataset(available_path, format="parquet").head(1).to_pylist()
    if len(seed_rows) != 1:
        raise RuntimeError("Drug R6 available relation has no smoke key")
    seed = seed_rows[0]
    observed = bundle.resolve(
        seed["cancer_id"], seed["lncrna_id"], seed["drug_id"]
    )
    if observed.get("availability") is not True:
        raise RuntimeError(f"Real available key was not queryable: {observed}")
    probability = observed.get(PROBABILITY_COLUMN)
    if not isinstance(probability, float) or not 0.0 <= probability <= 1.0:
        raise RuntimeError("Real available Drug probability is invalid")
    absent = bundle.resolve(
        seed["cancer_id"], seed["lncrna_id"], "STRICT_AUDIT_UNKNOWN_DRUG"
    )
    if (
        absent.get("availability") is not False
        or absent.get(PROBABILITY_COLUMN) is not None
        or absent.get("failure_reason") != "OUTSIDE_CONCEPTUAL_UNIVERSE"
    ):
        raise RuntimeError(f"Typed absent Drug key failed closed: {absent}")

    payload = {
        "format": "CC_HHGT_V3_2_DRUG_RESPONSE_R6_REAL_SMOKE_V1",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": artifact_sha256(args.manifest),
        "available_relation_path": str(available_path),
        "available_relation_rows": int(bundle.manifest["available_rows"]),
        "available_relation_sha256": str(
            bundle.manifest["artifacts"]["available_predictions"]["sha256"]
        ),
        "real_available_query": observed,
        "real_typed_absent_query": absent,
        "semantics": {
            "drug_response_association_probability_not_efficacy_or_direction": True,
            "tcga_patient_response_claimed": False,
            "does_not_change_primary_pathway_ranking": True,
            "old_predictions_used": False,
        },
        "runtime": {
            "python": sys.executable,
            "duckdb_memory_limit": os.environ.get(
                "CC_HHGT_DRUG_SPARSE_DUCKDB_MEMORY_LIMIT"
            ),
            "duckdb_threads": os.environ.get("CC_HHGT_DRUG_SPARSE_DUCKDB_THREADS"),
        },
    }
    payload["semantic_sha256"] = _canonical_sha256(payload)
    output.mkdir(parents=True)
    destination = output / "REAL_SMOKE.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "path": str(destination),
        "sha256": artifact_sha256(destination),
        "status": "PASS",
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
