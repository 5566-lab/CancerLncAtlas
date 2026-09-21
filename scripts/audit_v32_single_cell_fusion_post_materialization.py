#!/usr/bin/env python3
"""Independently audit the materialized 3.3M single-cell fusion expert."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_fusion_post_audit import (  # noqa: E402
    run_post_materialization_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--materialization-root", type=Path,
        default=ROOT / "artifacts" / "v32_single_cell_fusion_adapter_20260826_r1",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "artifacts"
        / "v32_single_cell_fusion_adapter_20260826_r1_independent_post_audit",
    )
    args = parser.parse_args(argv)
    result = run_post_materialization_audit(
        repo_root=args.repo_root,
        materialization_root=args.materialization_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
