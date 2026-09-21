#!/usr/bin/env python3
"""Materialize fresh V3.2 patient-gene ATAC accessibility from official assets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.atac_materialization import run_atac_materialization  # noqa: E402
from cc_hhgt.v32.patient_fold_authority import (  # noqa: E402
    validate_frozen_v32_patient_fold_binding,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--pathway-membership", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--peak-set", required=True)
    parser.add_argument("--promoters", required=True)
    parser.add_argument("--extractor-script", required=True)
    parser.add_argument("--input-contract", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rscript-bin", default="/usr/bin/Rscript")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_frozen_v32_patient_fold_binding(
        args.patient_folds, args.patient_fold_authority_receipt
    )
    result = run_atac_materialization(
        candidates_path=args.candidates,
        patient_folds_path=args.patient_folds,
        pathway_membership_path=args.pathway_membership,
        matrix_path=args.matrix,
        mapping_path=args.mapping,
        peak_set_path=args.peak_set,
        promoters_path=args.promoters,
        extractor_script_path=args.extractor_script,
        input_contract_path=args.input_contract,
        output_root=args.output_root,
        rscript_bin=args.rscript_bin,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
