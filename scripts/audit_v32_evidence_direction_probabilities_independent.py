#!/usr/bin/env python3
"""Independently audit a hash-pinned V3.2 Evidence direction release."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--expected-binding-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-total-rows", type=int, default=3_300_000)
    parser.add_argument("--expected-available-rows", type=int, default=825_753)
    parser.add_argument("--expected-cancer-count", type=int, default=33)
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.evidence_direction_independent_audit import (
        audit_evidence_direction_release,
    )

    result = audit_evidence_direction_release(
        binding_path=_resolve(root, args.binding),
        expected_binding_sha256=args.expected_binding_sha256,
        output_root=_resolve(root, args.output_root),
        expected_total_rows=args.expected_total_rows,
        expected_available_rows=args.expected_available_rows,
        expected_cancer_count=args.expected_cancer_count,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
