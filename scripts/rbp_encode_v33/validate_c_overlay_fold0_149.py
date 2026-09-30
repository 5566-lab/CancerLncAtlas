#!/usr/bin/env python3
"""CPU-only fold training-input rehearsal using the immutable A batches."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.c_graph_overlay import OVERLAY_FORMAT, load_c_graph_overlay  # noqa: E402
from cc_hhgt.v32.training import _validate_prepared, validate_formal_graph_variant_binding  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--overlay-receipt", type=Path, required=True)
    source_group = p.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--parent-hash-receipt", type=Path)
    source_group.add_argument("--all-folds-sha-ready", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("C input preflight must run on host 149")
    if args.output.exists():
        raise FileExistsError("C input preflight result already exists")
    sidecar = json.loads(args.overlay_receipt.read_text(encoding="utf-8"))
    if (sidecar.get("format") != OVERLAY_FORMAT or sidecar.get("host") != "149"
            or sidecar.get("fold") != args.fold or sidecar.get("variant") != "G2"):
        raise RuntimeError("C sidecar receipt has wrong scope")
    old_path = Path(sidecar["source_prepared_path"]).resolve(strict=True)
    overlay_path = Path(sidecar["path"]).resolve(strict=True)
    if args.parent_hash_receipt is not None:
        source_hash = args.parent_hash_receipt.read_text(encoding="utf-8").strip().split()
    else:
        ready = json.loads(args.all_folds_sha_ready.read_text(encoding="utf-8"))
        if (ready.get("status") != "PASS_C_ALL_FOLDS_INPUT_SHA256"
                or ready.get("target_host") != "149"):
            raise RuntimeError("Five-fold source SHA256 receipt is not ready")
        rows = [row for row in ready.get("folds", []) if row.get("fold") == args.fold]
        if len(rows) != 1:
            raise RuntimeError("Five-fold source SHA256 receipt has wrong fold scope")
        source_hash = [rows[0]["source_sha256"], rows[0]["source_path"]]
    if source_hash != [sidecar["source_prepared_sha256"], str(old_path)]:
        raise RuntimeError("A source SHA256 receipt mismatch")
    if overlay_path.stat().st_size != sidecar["bytes"] or sha256(overlay_path) != sidecar["sha256"]:
        raise RuntimeError("C sidecar SHA256 receipt mismatch")

    import torch
    from cc_hhgt.gnn import build_model

    with old_path.open("rb") as handle:
        old_payload = torch.load(handle, map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory(prefix="c_graph_cpu_preflight_") as temp:
        manifest_path = Path(temp) / "INPUT_MANIFEST.json"
        manifest_path.write_text(json.dumps({"fold_inputs": [{
            "fold": args.fold, "graph_overlay": {
                "format": OVERLAY_FORMAT, "path": str(overlay_path),
                "sha256": sidecar["sha256"],
                "source_prepared_sha256": sidecar["source_prepared_sha256"],
                "graph_authority_receipt_sha256": sidecar["graph_authority_receipt_sha256"],
            },
        }]}), encoding="utf-8")
        payload, observed_overlay_hash = load_c_graph_overlay(
            torch, input_manifest_path=manifest_path, overlay_path=overlay_path,
            fold=args.fold, parent_sha256=sidecar["source_prepared_sha256"],
            old_payload=old_payload,
        )
    _validate_prepared(payload, fold=args.fold, artifact_hashes={}, authorized_graph_overlay=True)
    validate_formal_graph_variant_binding(
        run_id=f"v32-g012-g2-c-overlay-fold-{args.fold}-cpu-preflight",
        config={"task_contract": {"graph_variant": "G2"}},
        payload=payload, prepared_path=old_path,
    )
    model = build_model(
        "cc_hhgt", payload["bundle"], int(payload["feature_dim"]),
        dict(payload["legacy_model_config"]),
    )
    receipt = {
        "status": "PASS_C_FOLD0_CPU_OVERLAY_MODEL_PREFLIGHT",
        "host": "149", "fold": args.fold, "gpu_started": False,
        "source_prepared_sha256": sidecar["source_prepared_sha256"],
        "c_graph_overlay_sha256": observed_overlay_hash,
        "c_graph_authority_receipt_sha256": sidecar["graph_authority_receipt_sha256"],
        "train_batches": len(payload["train_batches"]),
        "validation_batches": len(payload["validation_batches"]),
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
