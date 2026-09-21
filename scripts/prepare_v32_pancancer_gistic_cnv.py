#!/usr/bin/env python3
"""Prepare an all-cancer sparse GISTIC CNV routed-candidate bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--gistic", required=True)
    parser.add_argument("--gencode-gtf", required=True)
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--entity-intervals", required=True)
    parser.add_argument("--pathway-membership", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--min-patients-per-cancer", type=int, default=12)
    parser.add_argument("--parquet-row-group-size", type=int, default=100_000)
    parser.add_argument("--audit-only", action="store_true")
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.pancancer_cnv import (
        GisticPreparationConfig,
        prepare_pancancer_gistic_bundle,
    )
    from cc_hhgt.v32.patient_fold_authority import (
        validate_frozen_v32_patient_fold_binding,
    )

    patient_folds = _resolve(root, args.patient_folds)
    patient_fold_receipt = _resolve(root, args.patient_fold_authority_receipt)
    validate_frozen_v32_patient_fold_binding(patient_folds, patient_fold_receipt)

    result = prepare_pancancer_gistic_bundle(
        gistic_path=_resolve(root, args.gistic),
        gencode_gtf_path=_resolve(root, args.gencode_gtf),
        patient_folds_path=patient_folds,
        entity_intervals_path=_resolve(root, args.entity_intervals),
        pathway_membership_path=_resolve(root, args.pathway_membership),
        candidates_path=_resolve(root, args.candidates),
        output_root=_resolve(root, args.output_root),
        run_id=args.run_id,
        config=GisticPreparationConfig(
            min_patients_per_cancer=args.min_patients_per_cancer,
            parquet_row_group_size=args.parquet_row_group_size,
        ),
        audit_only=args.audit_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
