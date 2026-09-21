#!/usr/bin/env python3
"""Materialise the independently authorised full-universe single-cell expert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_fusion_binding import (  # noqa: E402
    materialize_single_cell_fusion_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--single-cell-exact", type=Path, required=True)
    parser.add_argument("--independent-audit-binding", type=Path, required=True)
    parser.add_argument("--independent-audit-binding-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nonformal-test-mode", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    result = materialize_single_cell_fusion_binding(
        candidates_path=args.candidates,
        exact_source_path=args.single_cell_exact,
        independent_audit_binding_path=args.independent_audit_binding,
        expected_audit_binding_sha256=args.independent_audit_binding_sha256,
        output_root=args.output,
        strict_formal=not args.nonformal_test_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
