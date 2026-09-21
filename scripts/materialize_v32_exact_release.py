#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.exact_release import materialize_exact_release
from cc_hhgt.v32.patient_fold_authority import (
    validate_frozen_v32_patient_fold_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize sanitized V3.2 exact ensemble")
    parser.add_argument("--fold-predictions", nargs=5, required=True)
    parser.add_argument("--fold-metrics", nargs=5, required=True)
    parser.add_argument("--training-success", nargs=5, required=True)
    parser.add_argument("--checkpoints", nargs=5, required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--patient-fold-manifest", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--prep-summary", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    validate_frozen_v32_patient_fold_binding(
        args.patient_fold_manifest, args.patient_fold_authority_receipt
    )
    result = materialize_exact_release(
        fold_prediction_paths=args.fold_predictions,
        fold_metrics_paths=args.fold_metrics,
        training_success_paths=args.training_success,
        checkpoint_paths=args.checkpoints,
        candidate_path=args.candidates,
        fold_manifest_path=args.patient_fold_manifest,
        prep_summary_path=args.prep_summary,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
