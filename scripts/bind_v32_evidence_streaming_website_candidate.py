#!/usr/bin/env python3
"""Bind the fresh streaming Evidence output to an isolated website candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.evidence_streaming_website import (  # noqa: E402
    materialize_streaming_evidence_website_binding,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--stage-manifest", required=True, type=Path)
    parser.add_argument("--candidate-events", required=True, type=Path)
    parser.add_argument("--event-lineage", required=True, type=Path)
    parser.add_argument("--independent-audit", required=True, type=Path)
    parser.add_argument("--independent-audit-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--fixture-authority",
        action="store_true",
        help="Allow small test fixtures; never use for the formal 3.3M candidate.",
    )
    args = parser.parse_args(argv)
    result = materialize_streaming_evidence_website_binding(
        prediction_manifest_path=args.prediction_manifest,
        predictions_path=args.predictions,
        training_manifest_path=args.training_manifest,
        stage_manifest_path=args.stage_manifest,
        candidate_events_path=args.candidate_events,
        event_lineage_path=args.event_lineage,
        independent_audit_path=args.independent_audit,
        expected_independent_audit_sha256=args.independent_audit_sha256,
        output_root=args.output_root,
        strict_formal=not args.fixture_authority,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
