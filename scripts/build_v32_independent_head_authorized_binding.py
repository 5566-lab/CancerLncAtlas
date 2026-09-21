#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cc_hhgt.v32.independent_head_authorized_binding import build_authorized_binding


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--portable-overlay-root", required=True)
    parser.add_argument("--copy-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--publication-root", required=True)
    parser.add_argument("--drug-materialized-root", required=True)
    args = parser.parse_args()
    result = build_authorized_binding(
        repo_root=args.repo_root,
        portable_overlay_root=args.portable_overlay_root,
        copy_manifest_path=args.copy_manifest,
        output_root=args.output_root,
        publication_root=args.publication_root,
        drug_materialized_root=args.drug_materialized_root,
    )
    print(result["status"])
    print(result["totals"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
