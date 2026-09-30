#!/usr/bin/env python3
"""CPU-only final G2 fold rehearsal on host 149, without file hashing."""
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--overlay-receipt", type=Path, required=True)
    parser.add_argument("--all-folds-sha-ready", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("Final G2 CPU validation must run on host 149")
    if args.output.exists():
        raise FileExistsError(args.output)

    sidecar = json.loads(args.overlay_receipt.read_text(encoding="utf-8"))
    if (sidecar.get("status") != "PASS_G2_GLOBAL_BINDING_SIDECAR"
            or sidecar.get("target_host") != "149"
            or sidecar.get("patient_fold") != args.fold
            or sidecar.get("graph_variant") != "G2"
            or sidecar.get("binding_context_policy") != "GLOBAL_PHYSICAL_BINDING"
            or sidecar.get("format") != OVERLAY_FORMAT
            or sidecar.get("file_hashes_computed") is not False):
        raise RuntimeError("New G2 sidecar receipt has wrong scope")
    authority = json.loads(args.all_folds_sha_ready.read_text(encoding="utf-8"))
    if (authority.get("status") != "PASS_C_ALL_FOLDS_INPUT_SHA256"
            or authority.get("target_host") != "149"):
        raise RuntimeError("Historical parent receipt has wrong scope")
    matches = [row for row in authority.get("folds", []) if row.get("fold") == args.fold]
    if len(matches) != 1:
        raise RuntimeError("Historical parent receipt has no unique fold")
    parent = matches[0]
    parent_path = Path(parent["source_path"]).resolve(strict=True)
    overlay_path = Path(sidecar["path"]).resolve(strict=True)
    if (parent_path.stat().st_size != parent["source_bytes"]
            or overlay_path.stat().st_size != sidecar["bytes"]
            or sidecar["source_prepared_sha256"] != parent["source_sha256"]
            or sidecar["graph_authority_receipt_sha256"] != parent["graph_authority_receipt_sha256"]):
        raise RuntimeError("Parent or new overlay authorization drift")

    import torch
    from cc_hhgt.gnn import build_model

    with parent_path.open("rb") as handle:
        parent_payload = torch.load(handle, map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory(prefix="g2_global_cpu_preflight_") as temp:
        manifest = Path(temp) / "INPUT_MANIFEST.json"
        manifest.write_text(json.dumps({"fold_inputs": [{
            "fold": args.fold,
            "graph_overlay": {
                "format": OVERLAY_FORMAT,
                "path": str(overlay_path),
                "bytes": sidecar["bytes"],
                "verification_mode": "SIZE_AND_GRAPH_SEMANTICS_V1",
                "expected_graph_edges": sidecar["new_edges"],
                "expected_binding_edges": sidecar["new_binding_edges"],
                "source_prepared_sha256": parent["source_sha256"],
                "graph_authority_receipt_sha256": parent["graph_authority_receipt_sha256"],
            },
        }]}), encoding="utf-8")
        payload, observed_digest = load_c_graph_overlay(
            torch, input_manifest_path=manifest, overlay_path=overlay_path,
            fold=args.fold, parent_sha256=parent["source_sha256"],
            old_payload=parent_payload,
        )
    if observed_digest is not None:
        raise RuntimeError("Final G2 preflight unexpectedly hashed a file")
    _validate_prepared(payload, fold=args.fold, artifact_hashes={}, authorized_graph_overlay=True)
    validate_formal_graph_variant_binding(
        run_id=f"v32-g012-g2-global-c-fold-{args.fold}-cpu-preflight",
        config={"task_contract": {"graph_variant": "G2"}},
        payload=payload, prepared_path=parent_path,
    )
    model = build_model(
        "cc_hhgt", payload["bundle"], int(payload["feature_dim"]),
        dict(payload["legacy_model_config"]),
    )
    receipt = {
        "status": "PASS_G2_GLOBAL_C_CPU_MODEL_PREFLIGHT",
        "target_host": "149", "fold": args.fold, "graph_variant": "G2",
        "paid_gpu_started": False, "file_hashes_computed": False,
        "source_prepared_path": str(parent_path),
        "source_prepared_bytes": parent["source_bytes"],
        "source_prepared_sha256_reused": parent["source_sha256"],
        "graph_overlay_path": str(overlay_path),
        "graph_overlay_bytes": sidecar["bytes"],
        "graph_edges": sidecar["new_edges"],
        "binding_edges": sidecar["new_binding_edges"],
        "train_batches": len(payload["train_batches"]),
        "validation_batches": len(payload["validation_batches"]),
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
