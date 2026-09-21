#!/usr/bin/env python3
"""Lock the fresh G0/G1/G2 winner using validation loss only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--prepared-parent", required=True)
    parser.add_argument("--run-parent", required=True)
    parser.add_argument("--candidate-authority", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pair-fold-seed", type=int, default=20260826)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.repo_root).resolve()))
    from cc_hhgt.v32.g012_winner_lock import lock_g012_validation_winner

    result = lock_g012_validation_winner(
        repo_root=args.repo_root,
        prepared_parent=args.prepared_parent,
        run_parent=args.run_parent,
        candidate_authority_path=args.candidate_authority,
        output_root=args.output_root,
        pair_fold_seed=args.pair_fold_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
