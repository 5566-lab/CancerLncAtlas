#!/usr/bin/env python3
"""Infer hierarchical outer folds after a hash-bound validation lock."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--checkpoint-pattern", required=True)
    parser.add_argument("--winner-locked-test-manifest", required=True)
    parser.add_argument("--routing-winner-declaration", required=True)
    parser.add_argument("--routing-winner-declaration-sha256", required=True)
    parser.add_argument("--primary-outer-predictions", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--training-budget-id", required=True)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.hierarchical_candidate_inference import (
        materialize_hierarchical_outer_after_winner_lock,
    )

    result = materialize_hierarchical_outer_after_winner_lock(
        repo_root=root,
        config_path=args.config,
        prepared_root=args.prepared_root,
        checkpoint_pattern=args.checkpoint_pattern,
        winner_locked_test_manifest_path=args.winner_locked_test_manifest,
        routing_winner_declaration_path=args.routing_winner_declaration,
        routing_winner_declaration_sha256=args.routing_winner_declaration_sha256,
        primary_outer_predictions_path=args.primary_outer_predictions,
        output_root=args.output_root,
        training_budget_id=args.training_budget_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
