#!/usr/bin/env python
"""Independently verify a strict V3.2 completeness r3 binding."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.integrated_completeness_strict_r3_independent import (  # noqa: E402
    independent_audit_strict_integrated_completeness_r3,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--expected-binding-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = independent_audit_strict_integrated_completeness_r3(
        repo_root=args.repo_root,
        binding_path=args.binding,
        expected_binding_sha256=args.expected_binding_sha256,
        output_root=args.output_root,
        auditor_code_path=(
            ROOT / "cc_hhgt/v32/integrated_completeness_strict_r3_independent.py"
        ),
        runner_code_path=Path(__file__).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
