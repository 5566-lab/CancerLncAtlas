#!/usr/bin/env python
"""Materialise V3.2 structural lncRNA--pathway--target--drug paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.drug_mechanism import materialize_drug_mechanisms  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--drug-targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixture-authority", action="store_true")
    args = parser.parse_args()
    result = materialize_drug_mechanisms(
        predictions_path=args.predictions,
        candidates_path=args.candidates,
        membership_path=args.membership,
        drug_targets_path=args.drug_targets,
        output_root=args.output,
        strict_formal_authority=not args.fixture_authority,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
