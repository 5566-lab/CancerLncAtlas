#!/usr/bin/env python3
"""Generalized fresh cell-level UCell runner for the 17 formal V3.2 cancers."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


PILOT_RUNNER = Path(__file__).with_name(
    "run_v32_single_cell_cell_level_pilot.py"
).resolve()


def _load_pilot_module():
    spec = importlib.util.spec_from_file_location(
        "v32_single_cell_cell_level_pilot_frozen", PILOT_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen UCell implementation: {PILOT_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    pilot = _load_pilot_module()
    args = pilot.build_parser().parse_args()
    result = pilot.run_general(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
