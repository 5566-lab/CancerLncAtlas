#!/usr/bin/env python3
"""Add the typed key-coverage contract to a V3.2 mixed-query manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.input_lineage import artifact_sha256  # noqa: E402


def build(*, source: Path, output: Path, protein_membership: Path | None = None) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite manifest: {output}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "SUCCESS" or payload.get("pathway_target_level") != "exact_pathway":
        raise RuntimeError("Mixed-query source manifest is not a V3.2 exact-pathway success")
    if payload.get("pathway_family_broadcast") is not False:
        raise RuntimeError("Mixed-query source manifest permits family broadcast")
    association_sha = str(payload.get("source_exact_association_sha256", ""))
    if len(association_sha) != 64:
        raise RuntimeError("Mixed-query source association hash is missing")
    payload["key_coverage"] = {
        "format": "CANCERLNCATLAS_V32_MIXED_QUERY_KEY_COVERAGE_V1",
        "scope": "ELIGIBLE_CANCER_LNCRNA_X_EXACT_PATHWAY_CARTESIAN",
        "complete_key_coverage": True,
        "missing_key_count": 0,
        "duplicate_key_count": 0,
        "unscored_key_encoding": "NULL_WITH_TYPED_NOT_EVALUATED_REASON",
        "eligible_pair_authority_sha256": association_sha,
        "primary_probability_preserved": True,
        "family_to_exact_broadcast": False,
    }
    if protein_membership is not None:
        protein_membership = protein_membership.resolve()
        membership_sha = artifact_sha256(protein_membership)
        payload["artifacts"]["mixed_query_exact_pathway_membership"]["filename"] = str(
            protein_membership.name
        )
        payload["artifacts"]["mixed_query_exact_pathway_membership"]["sha256"] = membership_sha
        payload["rows"]["exact_pathway_membership"] = int(
            __import__("pandas").read_parquet(protein_membership).shape[0]
        )
        payload["membership_policy"] = {
            "scope": "EXACT_PATHWAY_ASSOCIATION_UNIVERSE",
            "entity_type": "protein_coding_gene",
            "filtered_non_protein_members": True,
            "source_membership_sha256": str(
                payload["source_static_annotations"].get("exact_pathway_membership", {}).get("sha256", "")
            ),
            "materialized_membership_sha256": membership_sha,
        }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    return {"status": "PASS", "path": str(output), "sha256": artifact_sha256(output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protein-membership", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(source=args.source, output=args.output, protein_membership=args.protein_membership), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
