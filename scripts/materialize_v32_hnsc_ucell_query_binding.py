#!/usr/bin/env python3
"""Materialize the strict read-only binding for the audited HNSC UCell pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.hnsc_ucell_query import (  # noqa: E402
    build_hnsc_ucell_query_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--independent-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_hnsc_ucell_query_binding(
        result_root=args.result_root,
        independent_audit_path=args.independent_audit,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["binding"]["status"],
                "binding_path": result["binding_path"],
                "binding_sha256": result["binding_sha256"],
                "scope": result["binding"]["scope"],
                "single_cell_module_complete": False,
                "release_ready": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
