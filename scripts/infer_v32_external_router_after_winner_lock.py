#!/usr/bin/env python3
"""Infer external-router outer folds after a hash-bound validation lock."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--primary-fusion-frame", required=True)
    parser.add_argument("--genomic-predictions", required=True)
    parser.add_argument("--atac-predictions")
    parser.add_argument("--validation-gates", required=True)
    parser.add_argument("--external-validation-predictions", required=True)
    parser.add_argument("--external-validation-success", required=True)
    parser.add_argument("--winner-declaration", required=True)
    parser.add_argument("--winner-declaration-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.cancer_modality_router import (
        build_cancer_router_frame,
        materialize_external_router_outer_after_winner_lock,
    )

    primary = _resolve(root, args.primary_fusion_frame)
    genomic = _resolve(root, args.genomic_predictions)
    atac = _resolve(root, args.atac_predictions)
    assert primary and genomic
    frame = build_cancer_router_frame(
        pd.read_parquet(primary),
        pd.read_parquet(genomic),
        pd.read_parquet(atac) if atac else None,
    )
    result = materialize_external_router_outer_after_winner_lock(
        frame,
        validation_gates_path=args.validation_gates,
        external_validation_predictions_path=args.external_validation_predictions,
        external_validation_success_path=args.external_validation_success,
        winner_declaration_path=args.winner_declaration,
        winner_declaration_sha256=args.winner_declaration_sha256,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
