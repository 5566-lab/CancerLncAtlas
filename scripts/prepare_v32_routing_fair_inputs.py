#!/usr/bin/env python3
"""Retired single-table V3.2 routing input entry point.

One pair-aggregated table does not give the external and hierarchical arms the
same fold-specific training targets/features.  Formal callers must use
``extract_v32_primary_fold_views.py`` followed by the independent audit and the
fold-specific modality stager.  This entry point deliberately fails before it
can create an output directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--candidate-authority", required=True)
    parser.add_argument("--patient-fold-manifest", required=True)
    parser.add_argument("--formal-prepared-root", required=True)
    parser.add_argument("--primary-oof", required=True)
    parser.add_argument("--genomic-predictions", required=True)
    parser.add_argument("--genomic-lineage", required=True)
    parser.add_argument("--genomic-success", required=True)
    parser.add_argument("--atac-predictions", required=True)
    parser.add_argument("--atac-lineage", required=True)
    parser.add_argument("--atac-training-success", required=True)
    parser.add_argument("--atac-audit", required=True)
    parser.add_argument("--atac-audit-success", required=True)
    parser.add_argument("--budget-contract", required=True)
    parser.add_argument("--atac-freeze-contract", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    raise SystemExit(
        "SINGLE_TABLE_FAIR_CONTRACT_RETIRED: use five outer-fold "
        "train/validation/test shared views; no arm training is authorized"
    )
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.routing_fair_input import stage_routing_fair_inputs

    result = stage_routing_fair_inputs(
        candidate_authority_path=_resolve(root, args.candidate_authority),
        patient_fold_manifest_path=_resolve(root, args.patient_fold_manifest),
        formal_prepared_root=_resolve(root, args.formal_prepared_root),
        primary_oof_path=_resolve(root, args.primary_oof),
        genomic_predictions_path=_resolve(root, args.genomic_predictions),
        genomic_lineage_path=_resolve(root, args.genomic_lineage),
        genomic_success_path=_resolve(root, args.genomic_success),
        atac_predictions_path=_resolve(root, args.atac_predictions),
        atac_lineage_path=_resolve(root, args.atac_lineage),
        atac_training_success_path=_resolve(root, args.atac_training_success),
        atac_audit_path=_resolve(root, args.atac_audit),
        atac_audit_success_path=_resolve(root, args.atac_audit_success),
        budget_contract_path=_resolve(root, args.budget_contract),
        atac_freeze_contract_path=_resolve(root, args.atac_freeze_contract),
        output_root=_resolve(root, args.output_root),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
