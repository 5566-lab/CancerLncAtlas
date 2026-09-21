#!/usr/bin/env python3
"""Audit fail-closed recovery of raw Evidence lncRNA identifiers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--top-unresolved", type=int, default=250)
    parser.add_argument("--skip-input-sha256", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from cc_hhgt.v32.evidence_identifier_recovery import audit_raw_identifier_recovery

    report = audit_raw_identifier_recovery(
        project_root=args.project_root,
        output_root=args.output_root,
        top_unresolved=args.top_unresolved,
        hash_inputs=not args.skip_input_sha256,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_root": report["output_root"],
                "counts": report["counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
