#!/usr/bin/env python3
"""Reconstruct native signed CNV matrices from bound segment files on 149."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.cnv_input_sources import require_host
from cc_hhgt.v32.current_core_export import verified_path
from cc_hhgt.v32.formal_expression_sources import read, write
from cc_hhgt.v32.native_directional_cnv import FORMAT, PARTITION, REPLAY, ARRAYS, validate_release, validate_partition


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--success", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv); require_host()
    if args.output.exists(): raise FileExistsError("Use a new directional CNV audit output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = {"target_host":"149", "training_started":False, "model_forwards_executed":False,
            "raw_study_ancestry_complete":False, "format":REPLAY, "status":"CHECKING_INPUTS"}
    write(args.output, base)
    item = {"path":str(args.success), "sha256":args.expected_sha256}
    try:
        path = verified_path(item, Path.cwd()); value = read(path)
        if value.get("format") == FORMAT:
            _, value = validate_release(item, Path.cwd(), replay=True); count = len(value["cancers"])
        elif value.get("format") == PARTITION:
            validate_partition(item, Path.cwd(), replay=True); count = 1
        else: raise ValueError("Directional success lacks a native raw reconstruction contract")
        verified_path(item, Path.cwd())
        write(args.output, {**base, "status":"PASS", "source_release_sha256":item["sha256"],
                            "cancers_recomputed":count, "arrays_compared":count*len(ARRAYS)})
    except Exception as error:
        write(args.output, {**base, "status":"FAILED", "error_type":type(error).__name__, "error":str(error)})
        raise
    return 0


if __name__ == "__main__": raise SystemExit(main())
