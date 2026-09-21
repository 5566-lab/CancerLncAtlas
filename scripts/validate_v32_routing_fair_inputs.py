#!/usr/bin/env python3
"""Independently validate V3.2 fair inputs without running either arm or a winner."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--staging-success", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.routing_fair_input import validate_staged_routing_fair_inputs

    result = validate_staged_routing_fair_inputs(
        staging_success_path=_resolve(root, args.staging_success),
        output_root=_resolve(root, args.output_root),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
