#!/usr/bin/env python
"""Materialize strict V3.2 integrated-completeness audit r4."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.integrated_completeness_strict_r4 import (  # noqa: E402
    materialize_strict_integrated_completeness_audit_r4,
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
    parser.add_argument(
        "--frontend-html",
        type=Path,
        default=ROOT / "website/frontend/v32-staging.html",
    )
    parser.add_argument(
        "--frontend-javascript",
        type=Path,
        default=ROOT / "website/frontend/assets/v32-staging.js",
    )
    parser.add_argument(
        "--main-app", type=Path, default=ROOT / "website/backend/app.py"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_strict_integrated_completeness_audit_r4(
        repo_root=args.repo_root,
        parity_contract_path=args.parity_contract,
        unified_bindings_path=args.unified_bindings,
        web_catalog_path=args.web_catalog,
        frontend_html_path=args.frontend_html,
        frontend_javascript_path=args.frontend_javascript,
        main_app_path=args.main_app,
        output_root=args.output_root,
        evaluator_code_path=(
            ROOT / "cc_hhgt/v32/integrated_completeness_strict_r4.py"
        ),
        runner_code_path=Path(__file__).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
