#!/usr/bin/env python
"""Build a local semantic wrapper around an audited V3.2 R2 Evidence binding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.evidence_semantic_wrapper import (  # noqa: E402
    materialize_evidence_semantic_wrapper,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-binding", required=True, type=Path)
    parser.add_argument("--independent-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = materialize_evidence_semantic_wrapper(
        r2_binding_path=args.r2_binding,
        independent_audit_path=args.independent_audit,
        output_root=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
