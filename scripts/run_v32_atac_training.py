#!/usr/bin/env python3
"""Train the five-fold fresh V3.2 ATAC co-accessibility expert."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.atac_training import (  # noqa: E402
    AtacTrainingConfig,
    run_atac_training,
)
from cc_hhgt.v32.patient_fold_authority import (  # noqa: E402
    validate_frozen_v32_patient_fold_binding,
)


def _parser() -> argparse.ArgumentParser:
    defaults = AtacTrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--pathway-membership", required=True)
    parser.add_argument("--materialization-success", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--min-pathway-promoter-genes",
        type=int,
        default=defaults.min_pathway_promoter_genes,
    )
    parser.add_argument(
        "--min-inner-train-patients",
        type=int,
        default=defaults.min_inner_train_patients,
    )
    parser.add_argument(
        "--min-outer-train-patients",
        type=int,
        default=defaults.min_outer_train_patients,
    )
    parser.add_argument("--max-training-rows", type=int, default=defaults.max_training_rows)
    parser.add_argument(
        "--prediction-pair-batch-size",
        type=int,
        default=defaults.prediction_pair_batch_size,
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--l2", type=float, default=defaults.l2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_frozen_v32_patient_fold_binding(
        args.patient_folds, args.patient_fold_authority_receipt
    )
    config = AtacTrainingConfig(
        seed=args.seed,
        min_pathway_promoter_genes=args.min_pathway_promoter_genes,
        min_inner_train_patients=args.min_inner_train_patients,
        min_outer_train_patients=args.min_outer_train_patients,
        max_training_rows=args.max_training_rows,
        prediction_pair_batch_size=args.prediction_pair_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    result = run_atac_training(
        candidates_path=args.candidates,
        patient_folds_path=args.patient_folds,
        pathway_membership_path=args.pathway_membership,
        materialization_success_path=args.materialization_success,
        output_root=args.output_root,
        training_run_id=args.training_run_id,
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
