#!/usr/bin/env python3
"""Run leakage-safe V3.2 single-cell private-head training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_training import (  # noqa: E402
    SingleCellTrainingConfig,
    run_single_cell_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the five donor/dataset-blocked V3.2 single-cell private heads "
            "from exact-pathway non-predictive source assets."
        )
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--sc-association", required=True, type=Path)
    parser.add_argument("--lnc-celltype", required=True, type=Path)
    parser.add_argument("--core-embedding-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--activity", type=Path)
    parser.add_argument("--pseudotime", type=Path)
    parser.add_argument("--ucell", type=Path)
    parser.add_argument(
        "--association-generation", default="DECLARED_SINGLE_CELL_ASSOCIATION_SOURCE"
    )
    parser.add_argument(
        "--lnc-celltype-generation", default="DECLARED_SINGLE_CELL_NONPREDICTIVE_SOURCE"
    )
    parser.add_argument(
        "--activity-generation", default="DECLARED_SINGLE_CELL_NONPREDICTIVE_SOURCE"
    )
    parser.add_argument(
        "--pseudotime-generation", default="DECLARED_SINGLE_CELL_NONPREDICTIVE_SOURCE"
    )
    parser.add_argument(
        "--ucell-generation", default="DECLARED_SINGLE_CELL_NONPREDICTIVE_SOURCE"
    )
    parser.add_argument(
        "--figure-root", action="append", default=[], type=Path,
        help="Optional source figure directory; may be supplied more than once.",
    )
    parser.add_argument("--lncrna-node-type", default="lncRNA")
    parser.add_argument("--pathway-node-type", default="pathway")
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
    result = run_single_cell_training(
        candidates_path=args.candidates,
        dataset_manifest_path=args.dataset_manifest,
        association_path=args.sc_association,
        lnc_celltype_path=args.lnc_celltype,
        core_embedding_manifest_path=args.core_embedding_manifest,
        output_root=args.output_root,
        training_run_id=args.training_run_id,
        activity_path=args.activity,
        pseudotime_path=args.pseudotime,
        ucell_path=args.ucell,
        figure_roots=args.figure_root,
        association_generation=args.association_generation,
        lnc_celltype_generation=args.lnc_celltype_generation,
        activity_generation=args.activity_generation,
        pseudotime_generation=args.pseudotime_generation,
        ucell_generation=args.ucell_generation,
        config=config,
        lncrna_node_type=args.lncrna_node_type,
        pathway_node_type=args.pathway_node_type,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
