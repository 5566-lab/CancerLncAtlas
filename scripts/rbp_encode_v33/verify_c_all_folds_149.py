#!/usr/bin/env python3
"""Independently hash all five C graph sidecars and their reused A fold inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path


OVERLAY_FORMAT = "CANCERLNCATLAS_V32_C_GRAPH_OVERLAY_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--static-auth", type=Path, required=True)
    args = p.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("C five-fold verification must run on host 149")
    root = args.runtime_root.resolve(strict=True)
    result = root / "C_ALL_FOLDS_SHA_READY.json"
    if result.exists():
        raise FileExistsError(f"Refusing to overwrite five-fold receipt: {result}")
    if (root / "C_GRAPH_QUEUE_EXIT_CODE").read_text().strip() != "0":
        raise RuntimeError("C graph queue did not finish successfully")
    ready0 = json.loads((root / "C_FOLD0_CPU_READINESS.json").read_text(encoding="utf-8"))
    if (ready0.get("status") != "PASS_C_FOLD0_CPU_INPUTS_READY"
            or ready0.get("preparation_host") != "149"):
        raise RuntimeError("Fold-0 CPU model preflight has not passed")
    authority = json.loads(args.static_auth.read_text(encoding="utf-8"))
    if authority.get("host") != "149":
        raise RuntimeError("A prepared input authority has the wrong host")
    parents = authority["variants"]["G2"]["fold_inputs"]
    if sorted(int(row["fold"]) for row in parents) != list(range(5)):
        raise RuntimeError("A input authority does not contain exactly five patient folds")

    folds = []
    for fold in range(5):
        if fold and (root / f"BUILD_FOLD_{fold}_EXIT_CODE").read_text().strip() != "0":
            raise RuntimeError(f"C graph fold {fold} builder failed")
        parent = next(row for row in parents if int(row["fold"]) == fold)
        source = Path(parent["path"]).resolve(strict=True)
        overlay = (root / "overlays" / f"C_G2_PATIENT_FOLD_{fold}.pt").resolve(strict=True)
        receipt_path = root / "overlays" / f"C_G2_PATIENT_FOLD_{fold}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("format") != OVERLAY_FORMAT or receipt.get("host") != "149"
                or receipt.get("fold") != fold or receipt.get("variant") != "G2"
                or receipt.get("path") != str(overlay)
                or receipt.get("source_prepared_path") != str(source)
                or receipt.get("source_prepared_sha256") != parent["sha256"]
                or receipt.get("bytes") != overlay.stat().st_size
                or parent.get("bytes") != source.stat().st_size
                or receipt.get("active_eclip_edges", 0) <= 0
                or receipt.get("predicted_edges") != 0):
            raise RuntimeError(f"C graph fold {fold} receipt or source metadata differs")
        source_sha = sha256(source)
        if source_sha != parent["sha256"]:
            raise RuntimeError(f"A patient fold {fold} SHA256 differs from frozen authority")
        print(json.dumps({"status": "A_SOURCE_SHA256_VERIFIED", "fold": fold}), flush=True)
        overlay_sha = sha256(overlay)
        if overlay_sha != receipt["sha256"]:
            raise RuntimeError(f"C graph fold {fold} SHA256 differs from its receipt")
        print(json.dumps({"status": "C_GRAPH_SHA256_VERIFIED", "fold": fold}), flush=True)
        folds.append({
            "fold": fold, "source_path": str(source), "source_bytes": source.stat().st_size,
            "source_sha256": source_sha, "overlay_path": str(overlay),
            "overlay_bytes": overlay.stat().st_size, "overlay_sha256": overlay_sha,
            "active_eclip_edges": receipt["active_eclip_edges"],
            "node_sha256": receipt["node_sha256"],
            "graph_authority_receipt_sha256": receipt["graph_authority_receipt_sha256"],
        })
    if (len({row["node_sha256"] for row in folds}) != 1
            or len({row["graph_authority_receipt_sha256"] for row in folds}) != 1
            or len({row["active_eclip_edges"] for row in folds}) != 1):
        raise RuntimeError("C fold node, authority, or eCLIP invariants disagree")
    if (folds[0]["source_sha256"] != ready0["source_prepared_sha256"]
            or folds[0]["overlay_sha256"] != ready0["c_graph_overlay_sha256"]):
        raise RuntimeError("Fold-0 CPU readiness differs from independently hashed files")
    payload = {
        "status": "PASS_C_ALL_FOLDS_INPUT_SHA256", "target_host": "149",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paid_gpu_started": False, "training_started": False,
        "source_bytes_reused": sum(row["source_bytes"] for row in folds),
        "new_overlay_bytes": sum(row["overlay_bytes"] for row in folds),
        "folds": folds,
    }
    temporary = root / f".{result.name}.{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, result)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": payload["status"], "target_host": "149",
                      "fold_count": len(folds), "receipt": str(result)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
