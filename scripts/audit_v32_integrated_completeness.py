#!/usr/bin/env python
"""Materialize the fail-closed V3.2 integrated completeness audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.integrated_completeness import (  # noqa: E402
    materialize_integrated_completeness_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--parity-contract",
        type=Path,
        default=ROOT / "config/v32_historical_capability_parity.yaml",
    )
    parser.add_argument(
        "--unified-bindings",
        type=Path,
        default=ROOT / "config/v32_unified_staging_bindings.json",
    )
    parser.add_argument(
        "--web-catalog",
        type=Path,
        default=ROOT / "website/frontend/v32-capability-catalog.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_integrated_completeness_audit(
        repo_root=args.repo_root,
        parity_contract_path=args.parity_contract,
        unified_bindings_path=args.unified_bindings,
        web_catalog_path=args.web_catalog,
        output_root=args.output_root,
        evaluator_code_path=ROOT / "cc_hhgt/v32/integrated_completeness.py",
        runner_code_path=Path(__file__).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
