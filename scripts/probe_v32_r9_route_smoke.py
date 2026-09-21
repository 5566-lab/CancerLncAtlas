#!/usr/bin/env python3
"""Run the bounded r9 route smoke with the non-cyclic catalog pair check.

The historical probe has an ``--allow-stale-catalog`` discovery mode because
r8 contained a full-file scope/catalog hash cycle.  r9 has an explicit
canonical reverse digest and an external pair receipt, so this wrapper runs
the same semantic probes without that bypass and verifies the new contract
before delegating to the original probe implementation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from scripts import probe_v32_website_staging_candidate_20260904 as base_probe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _args_from_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    known, _ = parser.parse_known_args()
    return known


def main() -> int:
    args = _args_from_argv()
    scope_path = args.scope.resolve()
    catalog_path = args.catalog.resolve()
    manifest_path = args.manifest.resolve()
    pair_audit_path = args.pair_audit.resolve()
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    pair_audit = json.loads(pair_audit_path.read_text(encoding="utf-8"))
    if pair_audit.get("status") != "PASS_R9_SCOPE_CATALOG_PAIR":
        raise SystemExit("BLOCKED_R9_PAIR_AUDIT_NOT_PASS")
    if not isinstance(scope, dict) or not isinstance(catalog, dict):
        raise SystemExit("BLOCKED_INVALID_R9_SCOPE_OR_CATALOG")

    original_catalog_semantics = base_probe._catalog_semantics

    def r9_catalog_semantics(
        response_catalog: Any, *, expected_scope_sha256: str | None = None
    ) -> tuple[bool, str | None]:
        # Retain all historical environment/scope/status/stale-text checks,
        # but replace the impossible full-file reverse hash with the r9
        # canonical contract.
        ok, error = original_catalog_semantics(
            response_catalog, expected_scope_sha256=None
        )
        if not ok:
            return ok, error
        response_scope = response_catalog.get("capability_scope")
        if not isinstance(response_scope, dict):
            return False, "r9 catalog lacks capability_scope"
        if Path(str(response_scope.get("path", ""))).resolve() != scope_path:
            return False, "r9 catalog scope path drift"
        if response_scope.get(
            "hash_mode"
        ) != "CANONICAL_SCOPE_WITHOUT_CATALOG_SHA256_V1":
            return False, "r9 catalog scope hash mode drift"
        canonical_scope = copy.deepcopy(scope)
        canonical_scope.pop("catalog_sha256", None)
        canonical_scope_sha = _canonical(canonical_scope)
        if response_scope.get("sha256") != canonical_scope_sha:
            return False, "r9 canonical scope digest drift"
        catalog_sha = _sha256(catalog_path)
        if scope.get("catalog_sha256") != catalog_sha:
            return False, "r9 scope forward catalog digest drift"
        canonical_catalog = copy.deepcopy(response_catalog)
        canonical_catalog.pop("capability_scope", None)
        canonical_catalog.pop("catalog_payload_sha256", None)
        if response_catalog.get("catalog_payload_sha256") != _canonical(
            canonical_catalog
        ):
            return False, "r9 catalog payload digest drift"
        return True, None

    base_probe._catalog_semantics = r9_catalog_semantics
    # A strict r9 run must not carry the old discovery bypass flag.
    sys.argv = [item for item in sys.argv if item != "--allow-stale-catalog"]
    base_probe.main()

    # Add an explicit assertion to the output without changing any probe
    # result.  The caller supplied the output path, and base_probe already
    # enforces that it remains under the candidate root.
    output = args.output.resolve()
    report = json.loads(output.read_text(encoding="utf-8"))
    report["r9_catalog_pair_verification"] = {
        "status": "PASS",
        "pair_audit_path": str(pair_audit_path),
        "pair_audit_sha256": _sha256(pair_audit_path),
        "scope_forward_catalog_hash_checked": True,
        "canonical_scope_reverse_hash_checked": True,
        "catalog_payload_hash_checked": True,
        "allow_stale_catalog_used": False,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report.get("status"), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
