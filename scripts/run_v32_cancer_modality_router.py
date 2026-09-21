#!/usr/bin/env python3
"""Train and freeze the V3.2 external cancer-by-modality routed candidate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--primary-fusion-frame", required=True)
    parser.add_argument("--genomic-predictions", required=True)
    parser.add_argument("--genomic-lineage", required=True)
    parser.add_argument("--atac-predictions")
    parser.add_argument("--atac-lineage")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--min-validation-rows", type=int, default=30)
    parser.add_argument("--min-available-train-rows", type=int, default=30)
    parser.add_argument("--min-available-validation-rows", type=int, default=12)
    parser.add_argument("--validation-logloss-delta", type=float, default=1e-4)
    parser.add_argument("--modality-ablation-logloss-delta", type=float, default=0.0)
    parser.add_argument("--brier-tolerance", type=float, default=0.0)
    parser.add_argument("--frozen-gate-support-folds", type=int, default=3)
    parser.add_argument("--training-budget-id", default="v32-routed-equal-budget-v1")
    return parser


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.cancer_modality_router import (
        CancerRouterConfig,
        build_cancer_router_frame,
        materialize_cancer_modality_router,
        train_cancer_modality_router,
        validate_atac_oof_lineage,
        validate_patient_oof_lineage,
    )

    primary_path = _resolve(root, args.primary_fusion_frame)
    genomic_path = _resolve(root, args.genomic_predictions)
    lineage_path = _resolve(root, args.genomic_lineage)
    atac_path = _resolve(root, args.atac_predictions)
    atac_lineage_path = _resolve(root, args.atac_lineage)
    output_path = _resolve(root, args.output_root)
    assert primary_path and genomic_path and lineage_path and output_path
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    from cc_hhgt.v32.multimodal_fusion import artifact_sha256

    validate_patient_oof_lineage(
        lineage,
        predictions_sha256=artifact_sha256(genomic_path),
    )
    if (atac_path is None) != (atac_lineage_path is None):
        raise RuntimeError("--atac-predictions and --atac-lineage must be supplied together")
    if atac_path is not None and atac_lineage_path is not None:
        atac_lineage = json.loads(atac_lineage_path.read_text(encoding="utf-8"))
        validate_atac_oof_lineage(
            atac_lineage,
            predictions_sha256=artifact_sha256(atac_path),
        )
    frame = build_cancer_router_frame(
        pd.read_parquet(primary_path),
        pd.read_parquet(genomic_path),
        pd.read_parquet(atac_path) if atac_path else None,
    )
    config = CancerRouterConfig(
        seed=args.seed,
        min_train_rows=args.min_train_rows,
        min_validation_rows=args.min_validation_rows,
        min_available_train_rows=args.min_available_train_rows,
        min_available_validation_rows=args.min_available_validation_rows,
        validation_logloss_delta=args.validation_logloss_delta,
        modality_ablation_logloss_delta=args.modality_ablation_logloss_delta,
        brier_tolerance=args.brier_tolerance,
        frozen_gate_support_folds=args.frozen_gate_support_folds,
    )
    result = train_cancer_modality_router(frame, config=config)
    sources = {
        "primary_fusion_frame": primary_path,
        "genomic_predictions": genomic_path,
        "genomic_lineage": lineage_path,
    }
    if atac_path:
        sources["atac_predictions"] = atac_path
        assert atac_lineage_path is not None
        sources["atac_lineage"] = atac_lineage_path
    success = materialize_cancer_modality_router(
        result,
        output_root=output_path,
        run_id=args.run_id,
        config=config,
        source_artifacts=sources,
        training_budget_id=args.training_budget_id,
    )
    print(json.dumps(success, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
