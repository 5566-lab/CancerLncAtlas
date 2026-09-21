#!/usr/bin/env python3
"""Run the independent verifier for strict V3.2 audit r5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cc_hhgt.v32.integrated_completeness_strict_r5_independent import (
    independent_audit_strict_integrated_completeness_r5,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = independent_audit_strict_integrated_completeness_r5(
        repo_root=args.repo_root,
        binding_path=args.binding,
        expected_binding_sha256=args.binding_sha256,
        output_root=args.output_root,
        auditor_code_path=Path(__file__).resolve().parents[1]
        / "cc_hhgt/v32/integrated_completeness_strict_r5_independent.py",
        runner_code_path=Path(__file__).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
