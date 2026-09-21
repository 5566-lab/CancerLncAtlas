#!/usr/bin/env python3
"""Run the bounded context-aware V2 single-cell preflight only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_context_v2 import run_context_v2_preflight  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute context coverage, leakage, lineage, donor-split, and "
            "fresh-core gates without starting V2 training."
        )
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--sc-association", required=True, type=Path)
    parser.add_argument("--lnc-celltype", required=True, type=Path)
    parser.add_argument("--activity", required=True, type=Path)
    parser.add_argument("--pseudotime", type=Path)
    parser.add_argument("--ucell", type=Path)
    parser.add_argument("--core-embedding-manifest", required=True, type=Path)
    parser.add_argument("--formal-v1-binding", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_context_v2_preflight(
        candidates_path=args.candidates,
        dataset_manifest_path=args.dataset_manifest,
        association_path=args.sc_association,
        lnc_celltype_path=args.lnc_celltype,
        activity_path=args.activity,
        pseudotime_path=args.pseudotime,
        ucell_path=args.ucell,
        core_embedding_manifest_path=args.core_embedding_manifest,
        formal_v1_binding_path=args.formal_v1_binding,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    # A successfully completed preflight is useful even when the donor-resolved
    # training gate remains closed.  Context-coverage failure is fail-closed.
    return 0 if result.get("context_gate_pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())

