#!/usr/bin/env python3
"""Stage, fill, seal, or reassemble bounded V3.2 genomic memmaps."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_table(path: Path):
    import pandas as pd

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    commands = parser.add_subparsers(dest="command", required=True)

    stage = commands.add_parser("stage", help="partition keys and initialize memmaps")
    stage.add_argument("--candidate-authority", required=True)
    stage.add_argument("--genomic-training-source", required=True)
    stage.add_argument("--cnv-streaming-success", required=True)
    stage.add_argument("--cnv-streaming-success-sha256", required=True)
    stage.add_argument("--budget-contract", required=True)
    stage.add_argument("--output-root", required=True)
    stage.add_argument(
        "--skip-host-budget-check",
        action="store_true",
        help="test/smoke only; formal server launchers must never set this",
    )

    write = commands.add_parser("write", help="write one exact keyed fold slice")
    write.add_argument("--staging-success", required=True)
    write.add_argument("--modality", choices=("mutation", "cnv"), required=True)
    write.add_argument("--patient-fold", type=int, choices=range(5), required=True)
    write.add_argument("--cancer-id", required=True)
    write.add_argument("--prediction-table", required=True)
    write.add_argument("--probability-column", required=True)
    write.add_argument("--available-column", required=True)

    seal = commands.add_parser("seal", help="seal only a complete receipt-bound memmap")
    seal.add_argument("--staging-success", required=True)
    seal.add_argument("--output-path")

    reassemble = commands.add_parser(
        "reassemble", help="stream a sealed memmap into candidate-ordered Parquet"
    )
    reassemble.add_argument("--predictions-ready", required=True)
    reassemble.add_argument("--output-root", required=True)
    reassemble.add_argument("--training-run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.genomic_resource_staging import (
        reassemble_genomic_memmap_outputs,
        seal_genomic_memmap_predictions,
        stage_genomic_cancer_memmaps,
        write_genomic_memmap_partition,
    )

    if args.command == "stage":
        result = stage_genomic_cancer_memmaps(
            candidate_authority_path=_resolve(root, args.candidate_authority),
            genomic_training_source_path=_resolve(root, args.genomic_training_source),
            cnv_streaming_success_path=_resolve(root, args.cnv_streaming_success),
            cnv_streaming_success_sha256=args.cnv_streaming_success_sha256,
            budget_contract_path=_resolve(root, args.budget_contract),
            output_root=_resolve(root, args.output_root),
            enforce_host_budget=not args.skip_host_budget_check,
        )
    elif args.command == "write":
        table = _read_table(_resolve(root, args.prediction_table))
        required = {"cancer_id", "lncrna_id", "pathway_id", args.probability_column, args.available_column}
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"Prediction table lacks required columns: {missing}")
        result = write_genomic_memmap_partition(
            staging_success_path=_resolve(root, args.staging_success),
            modality=args.modality,
            patient_fold=args.patient_fold,
            cancer_id=args.cancer_id,
            candidate_keys=table[["cancer_id", "lncrna_id", "pathway_id"]],
            probability=table[args.probability_column].to_numpy(),
            available=table[args.available_column].to_numpy(),
        )
    elif args.command == "seal":
        result = seal_genomic_memmap_predictions(
            staging_success_path=_resolve(root, args.staging_success),
            output_path=_resolve(root, args.output_path) if args.output_path else None,
        )
    else:
        result = reassemble_genomic_memmap_outputs(
            predictions_ready_path=_resolve(root, args.predictions_ready),
            output_root=_resolve(root, args.output_root),
            training_run_id=args.training_run_id,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
