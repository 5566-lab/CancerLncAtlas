#!/usr/bin/env python3
"""Materialize the V3.2 r7 download catalog."""
from __future__ import annotations

import argparse
import json

from cc_hhgt.v32.download_catalog_r7 import materialize_download_catalog_r7


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--single-cell-deployment-path", required=True)
    parser.add_argument("--single-cell-deployment-sha256", required=True)
    parser.add_argument("--single-cell-audit-path", required=True)
    parser.add_argument("--single-cell-audit-sha256", required=True)
    args = parser.parse_args()
    result = materialize_download_catalog_r7(
        repo_root=args.repo_root,
        output_root=args.output_root,
        deployment_relative_path=args.single_cell_deployment_path,
        deployment_sha256=args.single_cell_deployment_sha256,
        audit_relative_path=args.single_cell_audit_path,
        audit_sha256=args.single_cell_audit_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
