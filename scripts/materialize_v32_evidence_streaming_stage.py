#!/usr/bin/env python3
"""Materialize fresh V3.2 Evidence training inputs without starting training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.evidence_streaming_training import (  # noqa: E402
    materialize_streaming_evidence_stage,
)


ROLES = (
    "interaction_relation",
    "empty_evidence_event",
    "exact_pathway_members",
    "formal_candidates",
    "conversion_manifest",
    "patient_fold_authority",
    "patient_fold_receipt",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hash-pinned, bounded-memory Evidence event-bag stage. "
            "This command never trains or publishes a model."
        )
    )
    parser.add_argument("--interaction-relation", required=True, type=Path)
    parser.add_argument("--empty-evidence-event", required=True, type=Path)
    parser.add_argument("--exact-pathway-members", required=True, type=Path)
    parser.add_argument("--formal-candidates", required=True, type=Path)
    parser.add_argument("--conversion-manifest", required=True, type=Path)
    parser.add_argument("--patient-fold-authority", required=True, type=Path)
    parser.add_argument("--patient-fold-receipt", required=True, type=Path)
    parser.add_argument("--expected-hashes-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--row-group-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help="Disable the 3.3M/33-cancer/6,160,707-row cardinality gates for tests only.",
    )
    args = parser.parse_args()
    expected_hashes = json.loads(
        args.expected_hashes_json.read_text(encoding="utf-8")
    )
    if set(expected_hashes) != set(ROLES):
        raise SystemExit(
            "expected-hashes JSON must contain exactly: " + ", ".join(ROLES)
        )
    manifest = materialize_streaming_evidence_stage(
        interaction_relation_path=args.interaction_relation,
        empty_evidence_event_path=args.empty_evidence_event,
        exact_pathway_members_path=args.exact_pathway_members,
        formal_candidates_path=args.formal_candidates,
        conversion_manifest_path=args.conversion_manifest,
        patient_fold_authority_path=args.patient_fold_authority,
        patient_fold_receipt_path=args.patient_fold_receipt,
        output_root=args.output_root,
        expected_hashes=expected_hashes,
        strict_formal=not args.fixture_mode,
        batch_size=args.batch_size,
        row_group_size=args.row_group_size,
        seed=args.seed,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_root": manifest["output_root"],
                "counts": manifest["counts"],
                "formal_training_started": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
