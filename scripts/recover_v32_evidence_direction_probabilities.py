#!/usr/bin/env python3
"""Recover the three Evidence direction probabilities from formal V3.2 heads."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--evidence-binding", type=Path, required=True)
    parser.add_argument("--expected-evidence-binding-sha256", required=True)
    parser.add_argument("--core-embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--mc-samples", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--inference-seed", type=int, default=20270826)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.evidence_direction_recovery import (
        recover_evidence_direction_probabilities,
    )

    result = recover_evidence_direction_probabilities(
        evidence_binding_path=_resolve(root, args.evidence_binding),
        expected_evidence_binding_sha256=args.expected_evidence_binding_sha256,
        core_embedding_root=_resolve(root, args.core_embedding_root),
        output_root=_resolve(root, args.output_root),
        batch_size=args.batch_size,
        mc_samples=args.mc_samples,
        dropout=args.dropout,
        inference_seed=args.inference_seed,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
