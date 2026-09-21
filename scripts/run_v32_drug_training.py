#!/usr/bin/env python3
"""Run fresh five-fold V3.2 cell-line drug private-head training."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _sha256(value: str) -> str:
    if re.fullmatch(r"[0-9A-Fa-f]{64}", value) is None:
        raise argparse.ArgumentTypeError(
            "expected exactly 64 hexadecimal characters (a SHA-256 digest)"
        )
    return value.lower()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the V3.2 drug module from cell-line level GDSC/PRISM response, "
            "lncRNA expression, static mappings/targets and a frozen V3.2 core."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--exact-candidates", required=True)
    parser.add_argument(
        "--drug-candidates",
        help=(
            "Factored V3.2 Drug candidate directory (formal default). A legacy explicit "
            "table requires --allow-legacy-dense-candidates."
        ),
    )
    parser.add_argument(
        "--allow-legacy-dense-candidates",
        action="store_true",
        help="Development fixtures only; permits the memory-resident legacy runner",
    )
    parser.add_argument("--raw-drug-response", required=True)
    parser.add_argument("--raw-lncrna-expression", required=True)
    parser.add_argument("--cell-line-map", required=True)
    parser.add_argument("--drug-gene-target", required=True)
    parser.add_argument("--curated-drug-response")
    parser.add_argument("--core-embedding-manifest", required=True)
    parser.add_argument(
        "--expected-staging-manifest-sha256",
        type=_sha256,
        help="Required for factored/formal streaming runs",
    )
    parser.add_argument(
        "--expected-exact-release-prediction-sha256",
        type=_sha256,
        help="Required for factored/formal streaming runs",
    )
    parser.add_argument(
        "--expected-exact-release-lineage-sha256",
        type=_sha256,
        help="Required for factored/formal streaming runs",
    )
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
    parser.add_argument("--min-cell-lines-per-dataset", type=int, default=5)
    parser.add_argument("--min-total-cell-lines", type=int, default=8)
    parser.add_argument("--positive-abs-rho", type=float, default=0.30)
    parser.add_argument("--negative-abs-rho", type=float, default=0.10)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--max-validation-rows", type=int, default=100_000)
    parser.add_argument("--prediction-batch-size", type=int, default=16_384)
    parser.add_argument("--candidate-chunk-rows", type=int, default=25_000)
    parser.add_argument("--private-evaluation-rows-per-fold", type=int, default=50_000)
    parser.add_argument("--estimated-public-bytes-per-row", type=int, default=160)
    parser.add_argument("--atomic-write-multiplier", type=float, default=2.25)
    parser.add_argument("--disk-reserve-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--disk-reserve-fraction", type=float, default=0.15)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate lineage/fold eligibility/disk budget and stop before training",
    )
    parser.add_argument(
        "--allow-untrainable-folds",
        action="store_true",
        help="Diagnostic development only; formal runs require all five fresh heads",
    )
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
    from cc_hhgt.v32.drug_training import DrugTrainingConfig, run_drug_training

    common = dict(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_features=args.hidden_features,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_cell_lines_per_dataset=args.min_cell_lines_per_dataset,
        min_total_cell_lines=args.min_total_cell_lines,
        positive_abs_rho=args.positive_abs_rho,
        negative_abs_rho=args.negative_abs_rho,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        prediction_batch_size=args.prediction_batch_size,
        require_all_folds=not args.allow_untrainable_folds,
    )
    candidate_path = _resolve(root, args.drug_candidates)
    factored = bool(
        candidate_path is not None
        and candidate_path.is_dir()
        and (candidate_path / "FACTORED_UNIVERSE.json").is_file()
    )
    if not factored and not args.allow_legacy_dense_candidates:
        raise SystemExit(
            "Formal V3.2 Drug training requires --drug-candidates pointing to a "
            "FACTORED_UNIVERSE.json directory; the legacy dense path is fail-closed."
        )
    if factored:
        missing_hashes = [
            flag
            for flag, value in (
                (
                    "--expected-staging-manifest-sha256",
                    args.expected_staging_manifest_sha256,
                ),
                (
                    "--expected-exact-release-prediction-sha256",
                    args.expected_exact_release_prediction_sha256,
                ),
                (
                    "--expected-exact-release-lineage-sha256",
                    args.expected_exact_release_lineage_sha256,
                ),
            )
            if value is None
        ]
        if missing_hashes:
            raise SystemExit(
                "Factored/formal V3.2 Drug training requires: "
                + ", ".join(missing_hashes)
            )
        from cc_hhgt.v32.drug_training_streaming import (
            StreamingDrugTrainingConfig,
            run_streaming_drug_training,
        )

        config = StreamingDrugTrainingConfig(
            **common,
            candidate_chunk_rows=args.candidate_chunk_rows,
            private_evaluation_rows_per_fold=args.private_evaluation_rows_per_fold,
            estimated_public_bytes_per_row=args.estimated_public_bytes_per_row,
            atomic_write_multiplier=args.atomic_write_multiplier,
            disk_reserve_bytes=args.disk_reserve_bytes,
            disk_reserve_fraction=args.disk_reserve_fraction,
            preflight_only=args.preflight_only,
        )
        result = run_streaming_drug_training(
            exact_candidates_path=_resolve(root, args.exact_candidates),
            factored_drug_candidates_path=candidate_path,
            raw_drug_response_path=_resolve(root, args.raw_drug_response),
            raw_lncrna_expression_path=_resolve(root, args.raw_lncrna_expression),
            cell_line_map_path=_resolve(root, args.cell_line_map),
            drug_gene_target_path=_resolve(root, args.drug_gene_target),
            curated_drug_response_path=_resolve(root, args.curated_drug_response),
            core_embedding_manifest_path=_resolve(root, args.core_embedding_manifest),
            output_root=_resolve(root, args.output_root),
            training_run_id=args.training_run_id,
            expected_staging_manifest_sha256=args.expected_staging_manifest_sha256,
            expected_exact_release_prediction_sha256=(
                args.expected_exact_release_prediction_sha256
            ),
            expected_exact_release_lineage_sha256=(
                args.expected_exact_release_lineage_sha256
            ),
            config=config,
        )
    else:
        config = DrugTrainingConfig(**common)
        result = run_drug_training(
            exact_candidates_path=_resolve(root, args.exact_candidates),
            drug_candidates_path=candidate_path,
            raw_drug_response_path=_resolve(root, args.raw_drug_response),
            raw_lncrna_expression_path=_resolve(root, args.raw_lncrna_expression),
            cell_line_map_path=_resolve(root, args.cell_line_map),
            drug_gene_target_path=_resolve(root, args.drug_gene_target),
            curated_drug_response_path=_resolve(root, args.curated_drug_response),
            core_embedding_manifest_path=_resolve(root, args.core_embedding_manifest),
            output_root=_resolve(root, args.output_root),
            training_run_id=args.training_run_id,
            config=config,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
