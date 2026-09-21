#!/usr/bin/env python
"""Materialise the hash-bound V3.2 Gene Set parity release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.gene_set_parity_release import (  # noqa: E402
    materialize_gene_set_parity_release,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--member-root", required=True, type=Path)
    parser.add_argument("--gmt", required=True, type=Path)
    parser.add_argument("--materialization-success", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--fusion-binding", required=True, type=Path)
    parser.add_argument("--fusion-binding-sha256", required=True)
    parser.add_argument("--fusion-post-audit", required=True, type=Path)
    parser.add_argument("--fusion-post-audit-sha256", required=True)
    parser.add_argument("--physical-manifest", required=True, type=Path)
    parser.add_argument("--physical-manifest-sha256", required=True)
    parser.add_argument("--physical-post-audit", required=True, type=Path)
    parser.add_argument("--physical-post-audit-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fixture-authority",
        action="store_true",
        help="Disable fixed formal row/SHA pins only for bounded tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = materialize_gene_set_parity_release(
        master_path=args.master,
        member_root=args.member_root,
        gmt_path=args.gmt,
        materialization_success_path=args.materialization_success,
        coverage_path=args.coverage,
        primary_path=args.primary,
        fusion_binding_path=args.fusion_binding,
        expected_fusion_binding_sha256=args.fusion_binding_sha256,
        fusion_post_audit_path=args.fusion_post_audit,
        expected_fusion_post_audit_sha256=args.fusion_post_audit_sha256,
        physical_manifest_path=args.physical_manifest,
        expected_physical_manifest_sha256=args.physical_manifest_sha256,
        physical_post_audit_path=args.physical_post_audit,
        expected_physical_post_audit_sha256=args.physical_post_audit_sha256,
        output_root=args.output,
        strict_formal_authority=not args.fixture_authority,
    )
    print(json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
