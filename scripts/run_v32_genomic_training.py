#!/usr/bin/env python3
"""Run fresh five-fold V3.2 Mutation/CNV private-head training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train availability-aware V3.2 Mutation/CNV private heads from raw/non-predictive "
            "patient calls and frozen fold-specific V3.2 core embeddings."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--pathway-gene-membership", required=True)
    parser.add_argument("--sample-gene-mutation", required=True)
    parser.add_argument("--sample-lncrna-mutation", required=True)
    parser.add_argument("--mc3", required=True, help="Original MC3 MAF; hashed for source lineage")
    parser.add_argument("--core-embedding-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--training-run-id", required=True)

    cnv = parser.add_argument_group("CNV input (choose paired long calls or raw segments)")
    cnv.add_argument("--cnv-gene-calls")
    cnv.add_argument("--cnv-lncrna-calls")
    cnv.add_argument(
        "--cnv-entity-coverage",
        help="Explicit cancer/entity callability sidecar for sparse GISTIC calls",
    )
    cnv.add_argument("--cnv-segment-root")
    cnv.add_argument(
        "--cnv-download-complete",
        help="READY post-download marker; mandatory with --cnv-segment-root",
    )
    cnv.add_argument("--entity-intervals")
    cnv.add_argument("--gistic-source", help="Optional raw GISTIC source retained in lineage")
    cnv.add_argument(
        "--cnv-streaming-success",
        help="Full33 hash-verified compact segment CNV SUCCESS.json",
    )

    parser.add_argument("--lncrna-node-type", default="lncRNA")
    parser.add_argument("--pathway-node-type", default="pathway")
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
    parser.add_argument("--cnv-event-threshold", type=float, default=0.30)
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
    from cc_hhgt.v32.genomic_training import GenomicTrainingConfig, run_genomic_training
    from cc_hhgt.v32.patient_fold_authority import (
        validate_frozen_v32_patient_fold_binding,
    )

    patient_folds = _resolve(root, args.patient_folds)
    patient_fold_receipt = _resolve(root, args.patient_fold_authority_receipt)
    validate_frozen_v32_patient_fold_binding(patient_folds, patient_fold_receipt)

    config = GenomicTrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_features=args.hidden_features,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_pair_callable=args.min_pair_callable,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        prediction_batch_size=args.prediction_batch_size,
        cnv_event_threshold=args.cnv_event_threshold,
    )
    result = run_genomic_training(
        candidates_path=_resolve(root, args.candidates),
        patient_folds_path=patient_folds,
        pathway_gene_membership_path=_resolve(root, args.pathway_gene_membership),
        sample_gene_mutation_path=_resolve(root, args.sample_gene_mutation),
        sample_lncrna_mutation_path=_resolve(root, args.sample_lncrna_mutation),
        mc3_path=_resolve(root, args.mc3),
        core_embedding_manifest_path=_resolve(root, args.core_embedding_manifest),
        output_root=_resolve(root, args.output_root),
        training_run_id=args.training_run_id,
        cnv_gene_calls_path=_resolve(root, args.cnv_gene_calls),
        cnv_lncrna_calls_path=_resolve(root, args.cnv_lncrna_calls),
        cnv_entity_coverage_path=_resolve(root, args.cnv_entity_coverage),
        cnv_segment_root=_resolve(root, args.cnv_segment_root),
        cnv_download_complete_path=_resolve(root, args.cnv_download_complete),
        entity_intervals_path=_resolve(root, args.entity_intervals),
        gistic_source_path=_resolve(root, args.gistic_source),
        cnv_streaming_success_path=_resolve(root, args.cnv_streaming_success),
        lncrna_node_type=args.lncrna_node_type,
        pathway_node_type=args.pathway_node_type,
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
