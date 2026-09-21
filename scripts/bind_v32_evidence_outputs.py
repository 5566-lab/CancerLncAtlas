#!/usr/bin/env python
"""Create the immutable post-training binding for fresh V3.2 Evidence outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.evidence_output_binding import (  # noqa: E402
    materialize_evidence_output_binding,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-output", required=True, type=Path)
    parser.add_argument("--evidence-code", required=True, type=Path)
    parser.add_argument("--evidence-runner", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--path-relocations",
        type=Path,
        help="Optional JSON mapping of immutable declared source paths to local mirrors",
    )
    parser.add_argument("--fixture-authority", action="store_true")
    args = parser.parse_args()
    relocations = None
    if args.path_relocations is not None:
        relocations = json.loads(args.path_relocations.read_text(encoding="utf-8"))
        if not isinstance(relocations, dict):
            raise SystemExit("--path-relocations must contain a JSON object")
    result = materialize_evidence_output_binding(
        evidence_output_root=args.evidence_output,
        evidence_code_path=args.evidence_code,
        evidence_runner_path=args.evidence_runner,
        exact_membership_path=args.membership,
        exact_candidates_path=args.candidates,
        output_root=args.output,
        strict_formal_authority=not args.fixture_authority,
        path_relocations=relocations,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
