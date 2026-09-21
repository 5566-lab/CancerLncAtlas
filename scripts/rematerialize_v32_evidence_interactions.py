#!/usr/bin/env python3
"""Build a fresh collision-safe V3.2 Evidence interaction authority."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--recovery-audit", required=True, type=Path)
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=250_000)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from cc_hhgt.v32.evidence_interaction_rematerialization import (
        rematerialize_evidence_interactions,
    )

    report = rematerialize_evidence_interactions(
        project_root=args.project_root,
        output_root=args.output_root,
        recovery_audit_path=args.recovery_audit,
        compression_level=args.compression_level,
        progress_every=args.progress_every,
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
