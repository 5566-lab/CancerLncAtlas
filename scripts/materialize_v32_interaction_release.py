#!/usr/bin/env python
"""Materialise fresh V3.2 physical relationships and exact-pathway ORA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.interaction_release import materialize_interaction_release  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-facts", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument(
        "--evidence-binding",
        type=Path,
        help="Required for formal runs: fresh EVIDENCE_OUTPUT_BINDING.json",
    )
    parser.add_argument(
        "--evidence-semantic-wrapper",
        type=Path,
        help="Required for formal runs: audited EVIDENCE_SEMANTIC_WRAPPER.json",
    )
    parser.add_argument(
        "--evidence-semantic-wrapper-sha256",
        help="Required formal SHA256 pin for the Evidence semantic wrapper",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fixture-authority",
        action="store_true",
        help="Disable formal SHA/count pins only for bounded test fixtures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_interaction_release(
        physical_facts_path=args.physical_facts,
        membership_path=args.membership,
        candidates_path=args.candidates,
        evidence_binding_path=args.evidence_binding,
        evidence_semantic_wrapper_path=args.evidence_semantic_wrapper,
        expected_evidence_semantic_wrapper_sha256=(
            args.evidence_semantic_wrapper_sha256
        ),
        output_root=args.output,
        strict_formal_authority=not args.fixture_authority,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
