#!/usr/bin/env python3
"""Reconstruct native prepared features/labels and frozen L1 scores, without fit."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.cnv_input_sources import require_host
from cc_hhgt.v32.current_core_export import verified_path
from cc_hhgt.v32.formal_expression_sources import write
from cc_hhgt.v32.prepared_construction import replay


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True, type=Path)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--baseline-only", action="store_true")
    args = p.parse_args(argv); require_host()
    if args.output.exists(): raise FileExistsError("Use a new prepared-construction audit output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = {"target_host": "149", "status": "CHECKING_INPUTS", "training_started": False}
    write(args.output, base)
    try:
        path = verified_path({"path": str(args.prepared), "sha256": args.expected_sha256}, Path.cwd())
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        result = replay(payload, path.parent, baseline_only=args.baseline_only)
        if result is None: raise ValueError("Prepared input has no native feature/label construction authority")
        verified_path({"path": str(path), "sha256": args.expected_sha256}, path.parent)
        write(args.output, {**base, **result, "status": "PASS", "prepared_sha256": args.expected_sha256,
            "baseline_only": args.baseline_only, "neural_model_forwards_executed": False,
            "raw_study_ancestry_complete": False})
    except Exception as error:
        write(args.output, {**base, "status": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        raise
    return 0


if __name__ == "__main__": raise SystemExit(main())
