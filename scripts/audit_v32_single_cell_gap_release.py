"""Independently audit a materialized V3.2 single-cell gap release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_gap_audit import (  # noqa: E402
    audit_single_cell_gap_release,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--expected-binding-sha256")
    args = parser.parse_args()
    result = audit_single_cell_gap_release(
        release_root=args.release_root,
        audit_root=args.audit_root,
        expected_binding_sha256=args.expected_binding_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
