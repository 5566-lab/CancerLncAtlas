#!/usr/bin/env python3
"""Server-side, no-compute preflight for the V3.2 SC diagnostic rerun."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_diagnostic_rerun import (  # noqa: E402
    write_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", choices=("formal17", "coverage33"), required=True)
    parser.add_argument(
        "--defer-remote-input-checks",
        action="store_true",
        help="Only for local plan materialization; cannot authorize execution.",
    )
    args = parser.parse_args()
    payload = write_preflight(
        args.config,
        args.output,
        scope=args.scope,
        verify_inputs=not args.defer_remote_input_checks,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

