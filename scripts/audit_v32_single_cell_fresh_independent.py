#!/usr/bin/env python3
"""Run the fail-closed independent audit of formal V3.2 single-cell outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_independent_audit import run_independent_audit  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--training-root", type=Path,
        default=ROOT / "artifacts" / "v32_single_cell_fresh_20260826_r1",
    )
    parser.add_argument(
        "--input-root", type=Path,
        default=ROOT / "artifacts" / "single_cell_training_inputs_20260826_r1",
    )
    parser.add_argument(
        "--candidate-universe", type=Path,
        default=ROOT / "artifacts" / "formal_prepared" / "FORMAL_CANDIDATE_UNIVERSE.parquet",
    )
    parser.add_argument(
        "--core-manifest", type=Path,
        default=ROOT / "artifacts" / "v32_full_multitask" / "core_embeddings"
        / "CORE_EMBEDDING_MANIFEST.json",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "artifacts" / "v32_single_cell_fresh_20260826_r1_independent_audit",
    )
    parser.add_argument(
        "--expected-training-run-id",
        default="V32-SINGLE-CELL-FRESH-20260826-R1",
        help="Exact training_run_id that every audited artifact must contain.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_independent_audit(
        repo_root=args.repo_root,
        training_root=args.training_root,
        input_root=args.input_root,
        candidate_universe=args.candidate_universe,
        core_manifest=args.core_manifest,
        output_root=args.output_root,
        expected_training_run_id=args.expected_training_run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
