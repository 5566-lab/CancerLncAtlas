"""Bounded audit of unbound V3.2 staging capabilities.

This audit reads only the supplied r9 unified binding manifest.  It does not
scan the server, inspect sealed-test data, or infer a publication binding from
historical paths.  Its purpose is to make the fail-closed pending inventory
reproducible and explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _host() -> str:
    try:
        return subprocess.check_output(
            ["hostname", "-s"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return socket.gethostname().split(".", 1)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _classification(binding_id: str, reason: str) -> str:
    text = f"{binding_id} {reason}".lower()
    if binding_id.startswith("single_cell_"):
        return "LEGACY_SINGLE_CELL_CONTRACT_NOT_REBINDABLE"
    if any(token in text for token in ("metadata-only", "metadata", "private checkpoint", "efficacy")):
        return "METADATA_OR_SCIENTIFIC_CLOSURE_MISSING"
    if "old path" in text or "local-path" in text or "not server-rebound" in text:
        return "HISTORICAL_LINEAGE_WITHOUT_SERVER_BINDING"
    return "NO_CURRENT_SERVER_PATH_OR_INDEPENDENT_AUDIT"


def main() -> int:
    args = _args()
    if _host() != "149":
        raise SystemExit(f"BLOCKED_WRONG_HOST expected=149 observed={_host()}")
    root = args.root.resolve()
    manifest = args.manifest.resolve(strict=True)
    output = args.output.resolve()
    try:
        manifest.relative_to(root)
        output.relative_to(root)
    except ValueError as exc:
        raise SystemExit("BLOCKED_PATH_OUTSIDE_CANDIDATE_ROOT") from exc
    if output.exists():
        raise SystemExit(f"REFUSING_TO_OVERWRITE={output}")

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        raise SystemExit("INVALID_BINDINGS_OBJECT")
    pending: list[dict[str, Any]] = []
    violations: list[str] = []
    for binding_id, value in sorted(bindings.items()):
        if not isinstance(value, dict) or value.get("status") != "PENDING_FORMAL_SUCCESS":
            continue
        has_path = bool(str(value.get("path", "")).strip())
        has_sha = bool(str(value.get("sha256", "")).strip())
        if has_path or has_sha:
            violations.append(f"pending_binding_has_path_or_sha:{binding_id}")
        reason = str(value.get("reason", ""))
        pending.append(
            {
                "binding_id": binding_id,
                "status": value.get("status"),
                "reason": reason,
                "classification": _classification(binding_id, reason),
                "path_present": has_path,
                "sha256_present": has_sha,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "CANCERLNCATLAS_V32_R9_PENDING_BINDING_INVENTORY_V1",
        "status": "PASS" if not violations else "FAIL",
        "host": "149",
        "environment": "staging",
        "candidate_root": str(root),
        "manifest": {"path": str(manifest), "sha256": _sha256(manifest)},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "declared_binding_count": len(bindings),
        "pending_binding_count": len(pending),
        "pending_bindings_have_no_path_or_sha": not violations,
        "safe_closure_count": 0,
        "pending_bindings": pending,
        "classification_counts": {
            key: sum(1 for item in pending if item["classification"] == key)
            for key in sorted({item["classification"] for item in pending})
        },
        "violations": violations,
        "production_deployed": False,
        "release_ready": False,
        "training_started": False,
        "gpu_started": False,
        "sealed_test_read": False,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
