#!/usr/bin/env python
"""Materialise the formal V3.2 experiment/Evidence anti-double-counting bridge."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cc_hhgt.v32.experiment_perturbation_bridge import (  # noqa: E402
    materialize_experiment_evidence_bridge,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-binding", required=True, type=Path)
    parser.add_argument("--evidence-binding", required=True, type=Path)
    parser.add_argument("--fusion-binding", required=True, type=Path)
    parser.add_argument("--fusion-post-audit-binding", required=True, type=Path)
    parser.add_argument("--core-embedding-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    result = materialize_experiment_evidence_bridge(
        experiment_binding_path=args.experiment_binding,
        evidence_binding_path=args.evidence_binding,
        fusion_binding_path=args.fusion_binding,
        fusion_post_audit_binding_path=args.fusion_post_audit_binding,
        core_embedding_root=args.core_embedding_root,
        output_dir=args.output_dir,
        runner_path=Path(__file__).resolve(),
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
