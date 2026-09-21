#!/usr/bin/env python3
"""Run post-lock Mutation/CNV/ATAC ablations for hierarchical HHGT."""
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
    parser.add_argument("--outer-predictions", required=True)
    parser.add_argument("--outer-success", required=True)
    parser.add_argument("--routing-winner-declaration", required=True)
    parser.add_argument("--routing-winner-declaration-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.hierarchical_modality_ablation import (
        materialize_hierarchical_modality_ablation_after_winner_lock,
    )

    result = materialize_hierarchical_modality_ablation_after_winner_lock(
        repo_root=root,
        config_path=args.config,
        prepared_root=args.prepared_root,
        checkpoint_pattern=args.checkpoint_pattern,
        outer_predictions_path=args.outer_predictions,
        outer_success_path=args.outer_success,
        routing_winner_declaration_path=args.routing_winner_declaration,
        routing_winner_declaration_sha256=args.routing_winner_declaration_sha256,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
