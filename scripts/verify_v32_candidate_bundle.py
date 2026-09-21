#!/usr/bin/env python3
"""Independently verify a materialized portable V3.2 candidate bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cc_hhgt.v32.portable_rebinding_audit import verify_portable_bundle


def _write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-missing-payloads", action="store_true")
    args = parser.parse_args()
    report = verify_portable_bundle(
        args.lineage,
        materialized_root=args.materialized_root,
        allowed_server_roots=(
            "./data/CancerLncAtlas",
            "./data/CancerLncAtlas",
        ),
        require_payloads=not args.allow_missing_payloads,
    )
    report_path = args.output_dir / "INDEPENDENT_REBIND_AUDIT.json"
    report_sha = _write_json(report_path, report)
    binding = {
        "format": "CANCERLNCATLAS_V32_PORTABLE_REBINDING_INDEPENDENT_AUDIT_BINDING_V1",
        "report": {"path": str(report_path.resolve()), "sha256": report_sha},
        "candidate_unified": report["candidate_unified"],
        "accepted_for_candidate_start": report["accepted_for_candidate_start"],
        "production_deployed": False,
        "release_ready": False,
    }
    binding_path = args.output_dir / "INDEPENDENT_REBIND_AUDIT_BINDING.json"
    binding_sha = _write_json(binding_path, binding)
    print(json.dumps({
        "accepted_for_candidate_start": report["accepted_for_candidate_start"],
        "report": str(report_path),
        "report_sha256": report_sha,
        "binding": str(binding_path),
        "binding_sha256": binding_sha,
    }, indent=2))
    return 0 if report["accepted_for_candidate_start"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
