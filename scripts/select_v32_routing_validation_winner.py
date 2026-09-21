#!/usr/bin/env python3
"""Lock the V3.2 routing architecture using validation-only predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--external-predictions", required=True)
    parser.add_argument("--hierarchical-predictions", required=True)
    parser.add_argument("--external-contract", required=True)
    parser.add_argument("--hierarchical-contract", required=True)
    parser.add_argument("--external-success", required=True)
    parser.add_argument("--hierarchical-success", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--minimum-logloss-improvement", type=float, default=1e-4)
    parser.add_argument("--brier-tolerance", type=float, default=0.0)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.routing_validation_winner import (
        lock_routing_validation_winner,
    )

    result = lock_routing_validation_winner(
        external_predictions_path=args.external_predictions,
        hierarchical_predictions_path=args.hierarchical_predictions,
        external_contract_path=args.external_contract,
        hierarchical_contract_path=args.hierarchical_contract,
        external_success_path=args.external_success,
        hierarchical_success_path=args.hierarchical_success,
        output_root=args.output_root,
        run_id=args.run_id,
        minimum_logloss_improvement=args.minimum_logloss_improvement,
        brier_tolerance=args.brier_tolerance,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
