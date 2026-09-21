#!/usr/bin/env python3
"""Extract immutable five-fold primary train/validation/test views."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--candidate-authority", required=True)
    parser.add_argument("--budget-contract", required=True)
    parser.add_argument("--sealed-test-acceptance", required=True)
    parser.add_argument("--expected-graph-variant", choices=("G0", "G1", "G2"), default="G2")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.primary_fold_views import extract_primary_fold_views

    result = extract_primary_fold_views(
        prepared_root=args.prepared_root,
        candidate_authority_path=args.candidate_authority,
        budget_contract_path=args.budget_contract,
        sealed_test_acceptance_path=args.sealed_test_acceptance,
        expected_graph_variant=args.expected_graph_variant,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
