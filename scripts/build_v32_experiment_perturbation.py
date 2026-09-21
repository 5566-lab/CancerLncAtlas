#!/usr/bin/env python3
"""Build the local, raw-input-only V3.2 perturbation fact layer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cc_hhgt.v32.experiment_perturbation import (  # noqa: E402
    FORMAL_ASSAY_DETAIL_SHA256,
    FORMAL_CANDIDATE_UNIVERSE_SHA256,
    FORMAL_EVIDENCE_EVENT_SHA256,
    FORMAL_PATHWAY_MEMBERS_SHA256,
    materialise_experiment_perturbation,
    validate_experiment_perturbation_binding,
)


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPOSITORY_ROOT / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialise functional-perturbation facts using only the current raw "
            "evidence table and current V3.2 exact authorities."
        )
    )
    parser.add_argument(
        "--evidence-event",
        default=r"D:\model\processed\evidence_event.parquet",
    )
    parser.add_argument(
        "--candidate-universe",
        default="artifacts/formal_prepared/FORMAL_CANDIDATE_UNIVERSE.parquet",
    )
    parser.add_argument(
        "--pathway-members",
        default="inputs/v32_full_multitask/genomic/exact_pathway_gene_membership_ensembl.parquet",
    )
    parser.add_argument(
        "--assay-detail",
        default=(
            "artifacts/v32_experiment_assay_detail_20260828_r1/"
            "v32_experiment_assay_detail.parquet"
        ),
        help="Hash-bound additive assay detail; never overwrites the authoritative event table.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/v32_experiment_perturbation_fresh_20260826_r1",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-hash an already materialised binding without rebuilding it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    if args.verify_only:
        binding = validate_experiment_perturbation_binding(output_dir)
        print(json.dumps(binding, indent=2, sort_keys=True))
        return 0

    result = materialise_experiment_perturbation(
        evidence_event_path=_resolve(args.evidence_event),
        candidate_universe_path=_resolve(args.candidate_universe),
        pathway_members_path=_resolve(args.pathway_members),
        assay_detail_path=_resolve(args.assay_detail),
        output_dir=output_dir,
        expected_hashes={
            "evidence_event": FORMAL_EVIDENCE_EVENT_SHA256,
            "candidate_universe": FORMAL_CANDIDATE_UNIVERSE_SHA256,
            "pathway_members": FORMAL_PATHWAY_MEMBERS_SHA256,
            "assay_detail": FORMAL_ASSAY_DETAIL_SHA256,
        },
        formal=True,
        runner_path=Path(__file__).resolve(),
    )
    # Re-open and independently re-hash everything before reporting success.
    validate_experiment_perturbation_binding(result.output_dir)
    summary = {
        "status": result.manifest["status"],
        "release_ready": result.manifest["release_ready"],
        "release_blockers": result.manifest["release_blockers"],
        "output_dir": str(result.output_dir),
        "manifest": str(result.manifest_path),
        "binding": str(result.binding_path),
        "row_audit": result.manifest["row_audit"],
        "artifact_sha256": {
            key: value["sha256"]
            for key, value in result.manifest["artifacts"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
