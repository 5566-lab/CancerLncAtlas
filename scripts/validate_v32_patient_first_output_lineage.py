#!/usr/bin/env python3
"""Validate and optionally seal a current V3.2 patient-first output lineage."""
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
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--fold-lineage", action="append", required=True)
    parser.add_argument("--run-lineage", action="append", default=[])
    parser.add_argument("--component", required=True)
    parser.add_argument("--output-audit")
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.patient_first_lineage import (
        validate_patient_first_output_lineage,
        write_patient_first_output_lineage_audit,
    )

    audit = validate_patient_first_output_lineage(
        patient_folds_path=_resolve(root, args.patient_folds),
        patient_fold_authority_receipt_path=_resolve(
            root, args.patient_fold_authority_receipt
        ),
        fold_lineage_paths=[_resolve(root, value) for value in args.fold_lineage],
        run_lineage_paths=[_resolve(root, value) for value in args.run_lineage],
        component=args.component,
    )
    if args.output_audit:
        destination = write_patient_first_output_lineage_audit(
            audit, _resolve(root, args.output_audit)
        )
        audit = {**audit, "audit_path": str(destination)}
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
