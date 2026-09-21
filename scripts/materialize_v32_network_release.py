#!/usr/bin/env python
"""Materialise the hash-bound current-V3.2 unified network release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.network_release import materialize_network_release  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--exact-lineage", required=True, type=Path)
    parser.add_argument("--fusion-scores", required=True, type=Path)
    parser.add_argument("--fusion-binding", required=True, type=Path)
    parser.add_argument("--fusion-audit", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--physical-relationships", required=True, type=Path)
    parser.add_argument("--physical-manifest", required=True, type=Path)
    parser.add_argument("--physical-audit", required=True, type=Path)
    parser.add_argument("--experiment-bridge", required=True, type=Path)
    parser.add_argument("--experiment-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument(
        "--fixture-authority",
        action="store_true",
        help="Disable formal SHA/count pins only for bounded tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_network_release(
        primary_path=args.primary,
        exact_lineage_path=args.exact_lineage,
        fusion_scores_path=args.fusion_scores,
        fusion_binding_path=args.fusion_binding,
        fusion_audit_path=args.fusion_audit,
        membership_path=args.membership,
        candidate_path=args.candidates,
        physical_relationships_path=args.physical_relationships,
        physical_manifest_path=args.physical_manifest,
        physical_audit_path=args.physical_audit,
        experiment_bridge_path=args.experiment_bridge,
        experiment_audit_path=args.experiment_audit,
        output_root=args.output,
        execution_code_paths=(Path(__file__), ROOT / "cc_hhgt/v32/network_release.py"),
        strict_formal_authority=not args.fixture_authority,
        memory_limit=args.memory_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
