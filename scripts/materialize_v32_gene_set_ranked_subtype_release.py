#!/usr/bin/env python
"""Bind audited V3.2 exact Gene Sets to deterministic ranked subtypes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.gene_set_subtype_release import (  # noqa: E402
    materialize_gene_set_ranked_subtype_release,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-set-manifest", required=True, type=Path)
    parser.add_argument("--gene-set-manifest-sha256", required=True)
    parser.add_argument("--gene-set-audit-binding", required=True, type=Path)
    parser.add_argument("--gene-set-audit-binding-sha256", required=True)
    parser.add_argument("--subtype-root", required=True, type=Path)
    parser.add_argument(
        "--subtype-implementation",
        type=Path,
        default=ROOT / "cc_hhgt" / "v32" / "subtypes.py",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fixture-authority",
        action="store_true",
        help="Disable only the fixed formal row-count pins for bounded tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_gene_set_ranked_subtype_release(
        gene_set_manifest_path=args.gene_set_manifest,
        expected_gene_set_manifest_sha256=args.gene_set_manifest_sha256,
        gene_set_audit_binding_path=args.gene_set_audit_binding,
        expected_gene_set_audit_binding_sha256=args.gene_set_audit_binding_sha256,
        subtype_root=args.subtype_root,
        subtype_implementation_path=args.subtype_implementation,
        output_root=args.output,
        strict_formal_authority=not args.fixture_authority,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
