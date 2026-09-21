#!/usr/bin/env python3
"""Build an immutable V3.2 patient-first five-fold authority.

This is the only supported replacement for the historical formal preparation
fold export.  It reads explicit ``patient_id`` values from formal pathway
activity and never derives a patient from ``sample_id``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.patient_fold_authority import (  # noqa: E402
    DEFAULT_SEED,
    TCGA_CANCERS,
    build_patient_fold_authority,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-patients", type=int, default=10_432)
    parser.add_argument("--memory-limit", default="2GB")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_patient_fold_authority(
        activity_root=Path(args.activity_root),
        output_root=Path(args.output_root),
        seed=args.seed,
        expected_cancers=TCGA_CANCERS,
        expected_patients=args.expected_patients,
        memory_limit=args.memory_limit,
        launcher_path=Path(__file__),
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output_root": receipt["output_root"],
                "patients": receipt["observed"]["patients"],
                "samples": receipt["observed"]["samples"],
                "cancers": receipt["observed"]["cancers"],
                "seed": receipt["seed"],
                "old_prepare_v32_formal_executed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
