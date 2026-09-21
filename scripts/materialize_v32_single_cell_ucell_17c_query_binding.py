#!/usr/bin/env python3
"""Materialize the audited, immutable 17-cancer V3.2 UCell query binding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_ucell_17c_query import (  # noqa: E402
    build_single_cell_ucell_17c_query_binding,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--batch-binding", required=True, type=Path)
    value.add_argument("--audit-binding", required=True, type=Path)
    value.add_argument("--hnsc-binding", required=True, type=Path)
    value.add_argument("--hnsc-binding-sha256", required=True)
    value.add_argument("--output-dir", required=True, type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    result = build_single_cell_ucell_17c_query_binding(
        batch_binding_path=args.batch_binding,
        audit_binding_path=args.audit_binding,
        hnsc_binding_path=args.hnsc_binding,
        hnsc_binding_sha256=args.hnsc_binding_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
