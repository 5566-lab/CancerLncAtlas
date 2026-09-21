#!/usr/bin/env python3
"""Fail-closed shared binding gate for every formal V3.2 fold consumer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.patient_fold_authority import (
        validate_frozen_v32_patient_fold_binding,
    )

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    result = validate_frozen_v32_patient_fold_binding(
        resolve(args.patient_folds),
        resolve(args.patient_fold_authority_receipt),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
