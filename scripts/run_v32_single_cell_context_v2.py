#!/usr/bin/env python3
"""Explicitly gated V3.2 single-cell context-V2 training runner.

Do not invoke this runner before root review.  The current aggregate target
table is expected to fail its strict donor-resolved split gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_context_v2 import (  # noqa: E402
    ROOT_APPROVAL_TOKEN,
    run_context_v2_training,
)
from cc_hhgt.v32.single_cell_training import SingleCellTrainingConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run explicitly approved, strict donor-blocked context-V2 training."
    )
    parser.add_argument("--execute-full-training", action="store_true", required=True)
    parser.add_argument("--root-approval-token", required=True)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--expected-execution-code-sha256", required=True)
    parser.add_argument("--donor-fold-manifest", required=True, type=Path)
    parser.add_argument("--expected-donor-fold-manifest-sha256", required=True)
    parser.add_argument("--fold-local-feature-manifest", required=True, type=Path)
    parser.add_argument(
        "--expected-fold-local-feature-manifest-sha256", required=True
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
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--max-validation-rows", type=int, default=100_000)
    parser.add_argument("--prediction-batch-size", type=int, default=16_384)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute_full_training or args.root_approval_token != ROOT_APPROVAL_TOKEN:
        raise SystemExit("Full context-V2 training lacks explicit root approval")
    config = SingleCellTrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_features=args.hidden_features,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        prediction_batch_size=args.prediction_batch_size,
    )
    result = run_context_v2_training(
        candidates_path=args.candidates,
        dataset_manifest_path=args.dataset_manifest,
        association_path=args.sc_association,
        lnc_celltype_path=args.lnc_celltype,
        activity_path=args.activity,
        pseudotime_path=args.pseudotime,
        ucell_path=args.ucell,
        core_embedding_manifest_path=args.core_embedding_manifest,
        formal_v1_binding_path=args.formal_v1_binding,
        preflight_path=args.preflight,
        expected_preflight_sha256=args.expected_preflight_sha256,
        expected_execution_code_sha256=args.expected_execution_code_sha256,
        donor_fold_manifest_path=args.donor_fold_manifest,
        expected_donor_fold_manifest_sha256=(
            args.expected_donor_fold_manifest_sha256
        ),
        fold_local_feature_manifest_path=args.fold_local_feature_manifest,
        expected_fold_local_feature_manifest_sha256=(
            args.expected_fold_local_feature_manifest_sha256
        ),
        output_root=args.output_root,
        training_run_id=args.training_run_id,
        root_approval_token=args.root_approval_token,
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
