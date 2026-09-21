#!/usr/bin/env python3
"""Run the strict V3.2 four-gate completeness audit r5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cc_hhgt.v32.integrated_completeness_strict_r5 import (
    materialize_strict_integrated_completeness_audit_r5,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--parity-contract", type=Path, required=True)
    parser.add_argument("--unified-bindings", type=Path, required=True)
    parser.add_argument("--web-catalog", type=Path, required=True)
    parser.add_argument("--frontend-html", type=Path, required=True)
    parser.add_argument("--frontend-javascript", type=Path, required=True)
    parser.add_argument("--main-app", type=Path, required=True)
    parser.add_argument("--download-binding-sha256", required=True)
    parser.add_argument("--download-audit-sha256", required=True)
    parser.add_argument("--single-cell-deployment-sha256", required=True)
    parser.add_argument("--single-cell-audit-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_strict_integrated_completeness_audit_r5(
        repo_root=args.repo_root,
        parity_contract_path=args.parity_contract,
        unified_bindings_path=args.unified_bindings,
        web_catalog_path=args.web_catalog,
        frontend_html_path=args.frontend_html,
        frontend_javascript_path=args.frontend_javascript,
        main_app_path=args.main_app,
        download_binding_sha256=args.download_binding_sha256,
        download_audit_sha256=args.download_audit_sha256,
        single_cell_deployment_sha256=args.single_cell_deployment_sha256,
        single_cell_audit_sha256=args.single_cell_audit_sha256,
        output_root=args.output_root,
        evaluator_code_path=Path(__file__).resolve().parents[1]
        / "cc_hhgt/v32/integrated_completeness_strict_r5.py",
        runner_code_path=Path(__file__).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
