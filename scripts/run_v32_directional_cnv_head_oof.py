#!/usr/bin/env python3
"""CLI for fresh formal-target directional CNV-only five-fold OOF."""
from __future__ import annotations

import argparse
import json

from cc_hhgt.v32.cnv_only_training_v2 import run_directional_cnv_only_oof
from cc_hhgt.v32.genomic_training import GenomicTrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--directional-root", required=True)
    parser.add_argument("--association-root", required=True)
    parser.add_argument("--core-embedding-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-pair-callable", type=int, default=12)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--max-validation-rows", type=int, default=100_000)
    parser.add_argument("--prediction-batch-size", type=int, default=16_384)
    args = parser.parse_args()
    config = GenomicTrainingConfig(
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        hidden_features=args.hidden_features, dropout=args.dropout,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        patience=args.patience, min_pair_callable=args.min_pair_callable,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        prediction_batch_size=args.prediction_batch_size,
    )
    result = run_directional_cnv_only_oof(
        candidates_path=args.candidates, folds_path=args.patient_folds,
        directional_root=args.directional_root,
        association_root=args.association_root,
        core_manifest_path=args.core_embedding_manifest,
        output_root=args.output_root, training_run_id=args.training_run_id,
        config=config,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
