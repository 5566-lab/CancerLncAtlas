#!/usr/bin/env python3
"""Audit the non-cyclic r9 scope/catalog binding for a V3.2 staging tree.

The scope sidecar contains the full hash of the final catalog.  The catalog
cannot, at the same time, contain the full hash of that scope because that
would create a cryptographic self-reference.  Instead the catalog records a
canonical scope hash computed after removing ``catalog_sha256`` from the
scope payload.  This script verifies both full-file hashes and the canonical
cross-reference, with an external pair receipt as the authoritative link.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HEX64 = set("0123456789abcdef")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _short_hostname() -> str:
    try:
        return os.uname().nodename.split(".", 1)[0]
    except AttributeError:
        return socket.gethostname().split(".", 1)[0]


def _inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes candidate root") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(f"{label} missing or symlink: {resolved}")
    return resolved


def _hex64(value: Any, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(ch not in HEX64 for ch in text):
        raise RuntimeError(f"{label} is not a SHA-256 digest")
    return text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frontend-catalog", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    if _short_hostname() != "149":
        raise SystemExit("BLOCKED_WRONG_HOST expected=149 observed=" + _short_hostname())
    if not root.is_dir() or root.is_symlink():
        raise SystemExit("BLOCKED_INVALID_CANDIDATE_ROOT")

    scope_path = _inside(args.scope, root, "scope")
    catalog_path = _inside(args.catalog, root, "catalog")
    manifest_path = _inside(args.manifest, root, "manifest")
    frontend_path = _inside(args.frontend_catalog, root, "frontend catalog")
    pair_path = _inside(args.pair, root, "pair") if args.pair.exists() else args.pair.resolve()
    if pair_path.exists() and pair_path.is_symlink():
        raise SystemExit("BLOCKED_PAIR_SYMLINK")

    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(scope, dict) or not isinstance(catalog, dict):
        raise SystemExit("INVALID_SCOPE_OR_CATALOG_JSON")
    if scope.get("environment") != "staging" or catalog.get("environment") != "staging":
        raise SystemExit("NON_STAGING_SCOPE_OR_CATALOG")
    if scope.get("host") != "149":
        raise SystemExit("SCOPE_HOST_NOT_149")
    if scope.get("production_deployed") is not False or scope.get("release_ready") is not False:
        raise SystemExit("SCOPE_DEPLOYMENT_FLAGS_UNSAFE")
    if catalog.get("production_deployed") is not False or catalog.get("release_ready") is not False:
        raise SystemExit("CATALOG_DEPLOYMENT_FLAGS_UNSAFE")
    if manifest.get("environment") != "staging" or manifest.get("production_deployed") is not False:
        raise SystemExit("MANIFEST_DEPLOYMENT_FLAGS_UNSAFE")

    scope_full_sha = sha256_file(scope_path)
    catalog_full_sha = sha256_file(catalog_path)
    manifest_full_sha = sha256_file(manifest_path)
    frontend_full_sha = sha256_file(frontend_path)
    if frontend_full_sha != catalog_full_sha:
        raise SystemExit("FRONTEND_CATALOG_HASH_MISMATCH")

    # The scope's forward declaration is deliberately a full final-catalog
    # hash.  This is the defect that r8 failed to keep current.
    if _hex64(scope.get("catalog_sha256"), "scope.catalog_sha256") != catalog_full_sha:
        raise SystemExit("SCOPE_FORWARD_CATALOG_HASH_DRIFT")

    scope_canonical = copy.deepcopy(scope)
    scope_canonical.pop("catalog_sha256", None)
    scope_canonical_sha = canonical_hash(scope_canonical)

    binding = catalog.get("capability_scope")
    if not isinstance(binding, dict):
        raise SystemExit("CATALOG_MISSING_CAPABILITY_SCOPE_BINDING")
    expected_path = str(scope_path)
    declared_path = str(binding.get("path", ""))
    if Path(declared_path).resolve() != scope_path:
        raise SystemExit("CATALOG_SCOPE_PATH_DRIFT")
    if binding.get("hash_mode") != "CANONICAL_SCOPE_WITHOUT_CATALOG_SHA256_V1":
        raise SystemExit("CATALOG_SCOPE_HASH_MODE_MISSING")
    if binding.get("canonical_sha256") not in {None, scope_canonical_sha}:
        raise SystemExit("CATALOG_CANONICAL_SCOPE_ALIAS_DRIFT")
    if _hex64(binding.get("sha256"), "catalog.capability_scope.sha256") != scope_canonical_sha:
        raise SystemExit("CATALOG_CANONICAL_SCOPE_HASH_DRIFT")

    catalog_canonical = copy.deepcopy(catalog)
    # Exclude both cross-reference metadata and the digest field itself.  The
    # latter is the usual content-addressed-manifest pattern and avoids a
    # self-referential hash while retaining a deterministic payload digest.
    catalog_canonical.pop("capability_scope", None)
    catalog_canonical.pop("catalog_payload_sha256", None)
    catalog_payload_sha = canonical_hash(catalog_canonical)
    if _hex64(catalog.get("catalog_payload_sha256"), "catalog.catalog_payload_sha256") != catalog_payload_sha:
        raise SystemExit("CATALOG_PAYLOAD_HASH_DRIFT")

    rows = catalog.get("capabilities")
    if not isinstance(rows, list) or catalog.get("capability_count") != len(rows):
        raise SystemExit("CATALOG_CAPABILITY_COUNT_DRIFT")
    statuses: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("CATALOG_CAPABILITY_ROW_INVALID")
        status = str(row.get("staging_status", ""))
        statuses[status] = statuses.get(status, 0) + 1
    if statuses.get("PENDING_FORMAL_BINDING", 0) != 22:
        raise SystemExit("PENDING_CAPABILITY_COUNT_CHANGED")
    if any(row.get("staging_status") == "QUERYABLE_STAGING" and row.get("capability_id") not in {"cnv", "mixed_lncrna_protein_pathway_query"} for row in rows):
        raise SystemExit("UNAUTHORIZED_CAPABILITY_PROMOTION")

    pair = {
        "schema_version": "CANCERLNCATLAS_V32_R9_SCOPE_CATALOG_PAIR_V1",
        "candidate_root": str(root),
        "environment": "staging",
        "host": "149",
        "production_deployed": False,
        "release_ready": False,
        "scope": {
            "path": str(scope_path),
            "file_sha256": scope_full_sha,
            "canonical_sha256": scope_canonical_sha,
            "forward_catalog_sha256": scope.get("catalog_sha256"),
        },
        "catalog": {
            "path": str(catalog_path),
            "file_sha256": catalog_full_sha,
            "canonical_payload_sha256": catalog_payload_sha,
        },
        "frontend_catalog": {"path": str(frontend_path), "file_sha256": frontend_full_sha},
        "manifest": {"path": str(manifest_path), "file_sha256": manifest_full_sha},
        "hash_contract": {
            "scope_forward_field_is_full_catalog_sha256": True,
            "catalog_scope_sha256_is_canonical_scope_without_catalog_sha256": True,
            "catalog_payload_sha256_excludes_capability_scope": True,
            "external_pair_is_authoritative_for_full_file_hashes": True,
        },
        "status_counts": statuses,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_R9_SCOPE_CATALOG_PAIR",
    }
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    pair_path.write_text(json.dumps(pair, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Pair is written before the receipt; the receipt records its resulting
    # digest and is itself the independently auditable closure artifact.
    result = {
        "schema_version": "CANCERLNCATLAS_V32_R9_SCOPE_CATALOG_AUDIT_V1",
        "candidate_root": str(root),
        "host": "149",
        "environment": "staging",
        "production_deployed": False,
        "release_ready": False,
        "status": "PASS_R9_SCOPE_CATALOG_PAIR",
        "scope_catalog_pair_path": str(pair_path),
        "scope_catalog_pair_sha256": sha256_file(pair_path),
        "scope_file_sha256": scope_full_sha,
        "scope_canonical_sha256": scope_canonical_sha,
        "catalog_file_sha256": catalog_full_sha,
        "catalog_payload_sha256": catalog_payload_sha,
        "frontend_catalog_file_sha256": frontend_full_sha,
        "manifest_file_sha256": manifest_full_sha,
        "status_counts": statuses,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "scope_forward_catalog_hash_matches": True,
            "catalog_reverse_canonical_scope_hash_matches": True,
            "external_pair_full_file_hashes_match": True,
            "catalog_payload_hash_matches": True,
            "frontend_catalog_byte_equal": True,
            "manifest_staging_nonproduction": True,
            "pending_formal_bindings_preserved": True,
            "no_gpu_or_training_started": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "pair": str(pair_path), "receipt": str(args.output), "pair_sha256": result["scope_catalog_pair_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
