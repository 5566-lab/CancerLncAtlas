#!/usr/bin/env python3
"""Materialize the canonical full-universe V3.2 Evidence fusion expert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.evidence_fusion_adapter import (  # noqa: E402
    materialize_evidence_fusion_adapter,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--semantic-wrapper", type=Path, required=True)
    parser.add_argument("--semantic-wrapper-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--non-formal",
        action="store_true",
        help="Disable the pinned formal row/SHA checks (tests only).",
    )
    args = parser.parse_args()
    result = materialize_evidence_fusion_adapter(
        candidates_path=args.candidates,
        semantic_wrapper_path=args.semantic_wrapper,
        expected_semantic_wrapper_sha256=args.semantic_wrapper_sha256,
        output_root=args.output,
        strict_formal=not args.non_formal,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
