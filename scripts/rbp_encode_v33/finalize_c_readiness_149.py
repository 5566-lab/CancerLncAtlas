#!/usr/bin/env python3
"""Seal CPU-only C fold-0 readiness without authorizing a paid instance."""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


def hash_line(path: Path) -> tuple[str, Path]:
    digest, filename = path.read_text(encoding="utf-8").strip().split()
    if len(digest) != 64:
        raise RuntimeError(f"Malformed SHA256 receipt: {path}")
    return digest, Path(filename)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", type=Path, required=True)
    args = p.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("CPU readiness must be sealed on host 149")
    root = args.runtime_root.resolve(strict=True)
    output = root / "C_FOLD0_CPU_READINESS.json"
    if output.exists():
        raise FileExistsError("C fold-0 readiness receipt already exists")
    model = json.loads((root / "CPU_OVERLAY_PREFLIGHT.json").read_text(encoding="utf-8"))
    code = json.loads((root / "C_CODE_ARCHIVE_READY.json").read_text(encoding="utf-8"))
    overlay = json.loads((root / "overlays" / "C_G2_PATIENT_FOLD_0.json").read_text(encoding="utf-8"))
    source_sha, source_path = hash_line(root / "SOURCE_A_FOLD0.sha256")
    wheel_sha, wheel_path = hash_line(root / "CLOUD_WHEELHOUSE.sha256")
    if ((root / "CPU_OVERLAY_PREFLIGHT_EXIT_CODE").read_text().strip() != "0"
            or model.get("status") != "PASS_C_FOLD0_CPU_OVERLAY_MODEL_PREFLIGHT"
            or code.get("status") != "PASS_C_CODE_ARCHIVE_149"
            or model.get("source_prepared_sha256") != source_sha
            or overlay.get("source_prepared_sha256") != source_sha
            or overlay.get("path") != str((root / "overlays" / "C_G2_PATIENT_FOLD_0.pt").resolve())
            or model.get("c_graph_overlay_sha256") != overlay.get("sha256")
            or code.get("code_archive_sha256") != hash_line(root / "C_TRAIN_CODE.sha256")[0]
            or str(source_path) != overlay.get("source_prepared_path")
            or wheel_path != root / "CLOUD_WHEELHOUSE.tar"
            or not wheel_path.is_file()):
        raise RuntimeError("C fold-0 CPU readiness receipts disagree")
    receipt = {
        "status": "PASS_C_FOLD0_CPU_INPUTS_READY",
        "preparation_host": "149",
        "gpu_started": False,
        "paid_gpu_allowed": False,
        "fold": 0,
        "source_prepared_sha256": source_sha,
        "source_prepared_bytes": source_path.stat().st_size,
        "c_graph_overlay_sha256": overlay["sha256"],
        "c_graph_overlay_bytes": overlay["bytes"],
        "active_eclip_edges": overlay["active_eclip_edges"],
        "predicted_edges": overlay["predicted_edges"],
        "graph_authority_receipt_sha256": overlay["graph_authority_receipt_sha256"],
        "code_archive_sha256": code["code_archive_sha256"],
        "code_tree_sha256": code["code_tree_sha256"],
        "wheel_archive_sha256": wheel_sha,
        "wheel_archive_bytes": wheel_path.stat().st_size,
        "train_batches": model["train_batches"],
        "validation_batches": model["validation_batches"],
        "model_parameter_count": model["model_parameter_count"],
        "next_gate": "CompShare billing mode and instance launch authorization",
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
