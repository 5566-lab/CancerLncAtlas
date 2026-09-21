#!/usr/bin/env python3
"""Materialize fresh V3.2 lncRNA log-rank and Kaplan--Meier facts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cc_hhgt.v32.clinical_km_release import (  # noqa: E402
    materialize_clinical_km_release,
)


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path(
            "inputs/v32_full_multitask/clinical/TCGA-CDR-SupplementalTableS1.xlsx"
        ),
    )
    parser.add_argument(
        "--expression-root",
        type=Path,
        default=Path("artifacts/input/formal_lncRNA_expression"),
    )
    parser.add_argument(
        "--exact-candidates",
        type=Path,
        default=Path("artifacts/formal_prepared/FORMAL_CANDIDATE_UNIVERSE.parquet"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fixture-authority",
        action="store_true",
        help="Allow synthetic fixture hashes/counts; never use for formal materialization.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    result = materialize_clinical_km_release(
        workbook_path=_resolve(root, args.workbook),
        expression_root=_resolve(root, args.expression_root),
        exact_candidate_path=_resolve(root, args.exact_candidates),
        output_root=_resolve(root, args.output),
        runner_path=Path(__file__),
        strict_formal_authority=not args.fixture_authority,
        progress=not args.quiet,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
