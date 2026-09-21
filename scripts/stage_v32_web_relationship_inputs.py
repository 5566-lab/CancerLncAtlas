#!/usr/bin/env python3
"""Audit selection readiness or stage public-only V3.2 web fold inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cc_hhgt.v32.web_relationship_input_staging import (
    audit_web_relationship_selection_readiness,
    stage_fresh_v32_web_relationship_inputs,
)


def _declaration(value: str) -> dict[str, str]:
    try:
        path, sha256 = value.rsplit("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected PATH=SHA256") from exc
    if not path or not sha256:
        raise argparse.ArgumentTypeError("expected PATH=SHA256")
    return {"path": path, "sha256": sha256.lower()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit-selection",
        help="Inspect small selection documents; never scan 3.3M-row inputs",
    )
    audit.add_argument("--evidence", action="append", type=_declaration, default=[])
    audit.add_argument("--selection-authority", type=_declaration)
    audit.add_argument("--relative-to", type=Path)
    audit.add_argument("--output-root", required=True, type=Path)

    stage = subparsers.add_parser(
        "stage", help="Validate all sources and create sanitized fold Parquet files"
    )
    stage.add_argument("--source-manifest", required=True, type=Path)
    stage.add_argument("--expected-source-manifest-sha256", required=True)
    stage.add_argument("--output-root", required=True, type=Path)
    stage.add_argument("--memory-limit", default="1GB")
    stage.add_argument("--temp-directory", type=Path)
    args = parser.parse_args()

    if args.command == "audit-selection":
        result = audit_web_relationship_selection_readiness(
            evidence_declarations=args.evidence,
            selection_authority_declaration=args.selection_authority,
            output_root=args.output_root,
            relative_to=args.relative_to,
        )
    else:
        result = stage_fresh_v32_web_relationship_inputs(
            source_manifest_path=args.source_manifest,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
            output_root=args.output_root,
            memory_limit=args.memory_limit,
            temp_directory=args.temp_directory,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ready_for_formal_input_staging", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
