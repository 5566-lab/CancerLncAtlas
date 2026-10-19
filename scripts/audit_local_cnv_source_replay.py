#!/usr/bin/env python3
"""Audit one actual native local-CNV statistical partition on server 149."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.cnv_input_sources import require_host
from cc_hhgt.v32.current_core_export import verified_path
from cc_hhgt.v32.formal_expression_sources import write
from cc_hhgt.v32.local_cnv_sources import partition_binding
from cc_hhgt.v32.local_cnv_source_replay import replay


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt", required=True, type=Path)
    p.add_argument("--receipt-sha256", required=True)
    p.add_argument("--partition", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv); require_host()
    if args.output.exists(): raise FileExistsError("Use a new local-CNV numerical audit output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = {"target_host": "149", "training_started": False, "status": "CHECKING_INPUTS"}
    write(args.output, base)
    item = {"path": str(args.receipt), "sha256": args.receipt_sha256}
    try:
        path = verified_path(item, Path.cwd())
        binding = partition_binding(args.partition, path)
        result = replay(binding, path.parent) if binding is not None else None
        if result is None: raise ValueError("Local-CNV partition has no native numerical replay contract")
        verified_path(item, Path.cwd())
        write(args.output, {**base, **result, "status": "PASS", "producer_receipt_sha256": args.receipt_sha256})
    except Exception as error:
        write(args.output, {**base, "status": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        raise
    return 0


if __name__ == "__main__": raise SystemExit(main())
