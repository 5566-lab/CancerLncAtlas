#!/usr/bin/env python3
"""Fairly compare external and end-to-end cancer-level routing candidates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--primary-oof", required=True)
    parser.add_argument("--external-oof", required=True)
    parser.add_argument("--hierarchical-oof", required=True)
    parser.add_argument("--external-contract", required=True)
    parser.add_argument("--hierarchical-contract", required=True)
    parser.add_argument("--hierarchical-probability-column", default="hierarchical_probability")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.routed_fair_comparison import (
        compare_routed_candidates,
        materialize_routed_comparison,
        validate_equal_contracts,
    )

    resolve = lambda value: Path(value).resolve() if Path(value).is_absolute() else (root / value).resolve()
    paths = {
        "primary_oof": resolve(args.primary_oof),
        "external_oof": resolve(args.external_oof),
        "hierarchical_oof": resolve(args.hierarchical_oof),
        "external_contract": resolve(args.external_contract),
        "hierarchical_contract": resolve(args.hierarchical_contract),
    }
    contract = validate_equal_contracts(
        json.loads(paths["external_contract"].read_text(encoding="utf-8")),
        json.loads(paths["hierarchical_contract"].read_text(encoding="utf-8")),
    )
    metrics = compare_routed_candidates(
        pd.read_parquet(paths["primary_oof"]),
        pd.read_parquet(paths["external_oof"]),
        pd.read_parquet(paths["hierarchical_oof"]),
        hierarchical_probability_column=args.hierarchical_probability_column,
    )
    result = materialize_routed_comparison(
        metrics,
        output_root=resolve(args.output_root),
        run_id=args.run_id,
        contract=contract,
        sources=paths,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
