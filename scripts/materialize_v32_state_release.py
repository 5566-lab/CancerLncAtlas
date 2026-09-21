#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.state_release import materialize_state_release
from cc_hhgt.v32.patient_fold_authority import (
    validate_frozen_v32_patient_fold_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize complete V3.2 State release")
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--source-lineage", type=Path, required=True)
    parser.add_argument("--patient-fold-manifest", type=Path, required=True)
    parser.add_argument(
        "--patient-fold-authority-receipt", type=Path, required=True
    )
    parser.add_argument("--core-embedding-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    validate_frozen_v32_patient_fold_binding(
        args.patient_fold_manifest, args.patient_fold_authority_receipt
    )
    result = materialize_state_release(
        source_release_path=args.source_release,
        source_lineage_path=args.source_lineage,
        fold_manifest_path=args.patient_fold_manifest,
        core_manifest_path=args.core_embedding_manifest,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
