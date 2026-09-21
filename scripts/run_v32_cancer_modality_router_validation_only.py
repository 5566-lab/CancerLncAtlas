#!/usr/bin/env python3
"""Fit V3.2 external-router gates using inner validation only."""
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
    parser.add_argument("--genomic-lineage", required=True)
    parser.add_argument("--atac-predictions")
    parser.add_argument("--atac-lineage")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--training-budget-id", required=True)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--min-validation-rows", type=int, default=30)
    parser.add_argument("--min-available-train-rows", type=int, default=30)
    parser.add_argument("--min-available-validation-rows", type=int, default=12)
    parser.add_argument("--validation-logloss-delta", type=float, default=1e-4)
    parser.add_argument("--modality-ablation-logloss-delta", type=float, default=0.0)
    parser.add_argument("--brier-tolerance", type=float, default=0.0)
    parser.add_argument("--frozen-gate-support-folds", type=int, default=3)
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.cancer_modality_router import (
        CancerRouterConfig,
        build_cancer_router_frame,
        materialize_cancer_modality_router_validation_only,
        train_cancer_modality_router_validation_only,
        validate_atac_oof_lineage,
        validate_patient_oof_lineage,
    )
    from cc_hhgt.v32.multimodal_fusion import artifact_sha256

    primary = _resolve(root, args.primary_fusion_frame)
    genomic = _resolve(root, args.genomic_predictions)
    genomic_lineage = _resolve(root, args.genomic_lineage)
    atac = _resolve(root, args.atac_predictions)
    atac_lineage = _resolve(root, args.atac_lineage)
    output = _resolve(root, args.output_root)
    assert primary and genomic and genomic_lineage and output
    if (atac is None) != (atac_lineage is None):
        raise RuntimeError("ATAC predictions and lineage must be supplied together")
    genomic_lineage_value = json.loads(genomic_lineage.read_text(encoding="utf-8"))
    validate_patient_oof_lineage(
        genomic_lineage_value, predictions_sha256=artifact_sha256(genomic)
    )
    source_artifacts: dict[str, Path] = {
        "primary_fusion_frame": primary,
        "genomic_predictions": genomic,
        "genomic_lineage": genomic_lineage,
    }
    atac_frame = None
    if atac is not None and atac_lineage is not None:
        validate_atac_oof_lineage(
            json.loads(atac_lineage.read_text(encoding="utf-8")),
            predictions_sha256=artifact_sha256(atac),
        )
        atac_frame = pd.read_parquet(atac)
        source_artifacts.update(
            {"atac_predictions": atac, "atac_lineage": atac_lineage}
        )
    frame = build_cancer_router_frame(
        pd.read_parquet(primary), pd.read_parquet(genomic), atac_frame
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
    result = train_cancer_modality_router_validation_only(frame, config=config)
    success = materialize_cancer_modality_router_validation_only(
        result,
        output_root=output,
        run_id=args.run_id,
        config=config,
        source_artifacts=source_artifacts,
        training_budget_id=args.training_budget_id,
    )
    print(json.dumps(success, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
