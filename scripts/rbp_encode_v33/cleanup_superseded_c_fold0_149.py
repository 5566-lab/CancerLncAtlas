#!/usr/bin/env python3
"""Remove one obsolete pre-correction C sidecar after its CPU validation."""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--old-overlay", type=Path, required=True)
    parser.add_argument("--new-overlay", type=Path, required=True)
    parser.add_argument("--new-receipt", type=Path, required=True)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--cleanup-receipt", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("C fold cleanup must run on host 149")
    if args.cleanup_receipt.exists():
        raise FileExistsError(args.cleanup_receipt)
    old = args.old_overlay.resolve(strict=True)
    new = args.new_overlay.resolve(strict=True)
    if (old.name != f"C_G2_PATIENT_FOLD_{args.fold}.pt"
            or new.name != f"G2_GLOBAL_FOLD_{args.fold}.pt"):
        raise RuntimeError("Cleanup paths do not name the exact fold generations")
    old_receipt = old.with_suffix(".json")
    prior = json.loads(old_receipt.read_text(encoding="utf-8"))
    fresh = json.loads(args.new_receipt.read_text(encoding="utf-8"))
    cpu = json.loads(args.cpu_receipt.read_text(encoding="utf-8"))
    if (prior.get("fold") != args.fold or prior.get("path") != str(old)
            or old.stat().st_size != prior.get("bytes")
            or fresh.get("status") != "PASS_G2_GLOBAL_BINDING_SIDECAR"
            or fresh.get("patient_fold") != args.fold or fresh.get("path") != str(new)
            or fresh.get("bytes") != new.stat().st_size
            or fresh.get("source_overlay") != str(old)
            or cpu.get("status") != "PASS_G2_GLOBAL_C_CPU_MODEL_PREFLIGHT"
            or cpu.get("target_host") != "149" or cpu.get("fold") != args.fold
            or cpu.get("graph_overlay_path") != str(new)
            or cpu.get("graph_overlay_bytes") != new.stat().st_size):
        raise RuntimeError("Old/new fold lineage or CPU validation is incomplete")
    if any(old.samefile(path) for path in (new, args.new_receipt, args.cpu_receipt)):
        raise RuntimeError("Old fold path aliases a file that must be kept")
    size = old.stat().st_size
    old.unlink()
    old_receipt.unlink()
    result = {
        "status": ("SUPERSEDED_C_FOLD0_REMOVED" if args.fold == 0
                   else "SUPERSEDED_C_FOLD_REMOVED"),
        "target_host": "149", "fold": args.fold,
        "old_overlay_path": str(old), "old_overlay_bytes_removed": size,
        "old_receipt_path": str(old_receipt),
        "new_overlay_path": str(new), "new_overlay_bytes": new.stat().st_size,
        "cpu_validation_receipt": str(args.cpu_receipt),
        "parent_A_fold_preserved": True, "file_hashes_computed": False,
    }
    args.cleanup_receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
