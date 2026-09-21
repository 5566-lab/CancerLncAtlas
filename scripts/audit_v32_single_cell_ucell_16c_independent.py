#!/usr/bin/env python3
"""Run the independent audit for a successful 16-cancer full UCell batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_ucell_16c_independent_audit import (  # noqa: E402
    audit_full_ucell_16c,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-binding", required=True, type=Path)
    parser.add_argument("--expected-batch-binding-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = audit_full_ucell_16c(
        args.batch_binding,
        expected_batch_binding_sha256=args.expected_batch_binding_sha256,
        output_dir=args.output_dir,
        auditor_code_path=Path(__file__),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
