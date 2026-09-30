#!/usr/bin/env python3
"""Seal actual C graph/model loading checks for all five folds on host 149."""
from __future__ import annotations

import argparse
import datetime
import json
import socket
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    args = parser.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("C five-fold CPU readiness must be sealed on host 149")
    root = args.runtime_root.resolve(strict=True)
    sha_ready = json.loads((root / "C_ALL_FOLDS_SHA_READY.json").read_text(encoding="utf-8"))
    if (sha_ready.get("status") != "PASS_C_ALL_FOLDS_INPUT_SHA256"
            or sha_ready.get("target_host") != "149"
            or [row.get("fold") for row in sha_ready.get("folds", [])] != list(range(5))):
        raise RuntimeError("Five-fold SHA256 readiness is missing or invalid")
    folds = []
    for row in sha_ready["folds"]:
        fold = row["fold"]
        suffix = "" if fold == 0 else f"_FOLD_{fold}"
        receipt = root / f"CPU_OVERLAY_PREFLIGHT{suffix}.json"
        exit_code = root / f"CPU_OVERLAY_PREFLIGHT{suffix}_EXIT_CODE"
        if exit_code.read_text(encoding="utf-8").strip() != "0":
            raise RuntimeError(f"C fold {fold} CPU model check failed")
        observed = json.loads(receipt.read_text(encoding="utf-8"))
        if (observed.get("status") != "PASS_C_FOLD0_CPU_OVERLAY_MODEL_PREFLIGHT"
                or observed.get("host") != "149"
                or observed.get("fold") != fold
                or observed.get("source_prepared_sha256") != row["source_sha256"]
                or observed.get("c_graph_overlay_sha256") != row["overlay_sha256"]
                or observed.get("c_graph_authority_receipt_sha256")
                != row["graph_authority_receipt_sha256"]
                or observed.get("gpu_started") is not False
                or min(observed.get("train_batches", 0), observed.get("validation_batches", 0),
                       observed.get("model_parameter_count", 0)) <= 0):
            raise RuntimeError(f"C fold {fold} CPU model check receipt disagrees with SHA256")
        folds.append({
            "fold": fold,
            "source_sha256": row["source_sha256"],
            "overlay_sha256": row["overlay_sha256"],
            "train_batches": observed["train_batches"],
            "validation_batches": observed["validation_batches"],
            "model_parameter_count": observed["model_parameter_count"],
        })
    output = root / "C_ALL_FOLDS_CPU_READY.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump({
            "status": "PASS_C_ALL_FOLDS_GRAPH_MODEL_LOAD",
            "target_host": "149",
            "paid_gpu_started": False,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "folds": folds,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "PASS_C_ALL_FOLDS_GRAPH_MODEL_LOAD",
                      "target_host": "149", "fold_count": len(folds)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
