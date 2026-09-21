#!/usr/bin/env python3
"""Audit the hash relationship between a staging scope and its catalog.

This is a read-only, dependency-free check intended for server-149 staging
trees.  A route smoke can prove that a scope file is syntactically valid while
still carrying a catalog hash from an older candidate.  The check therefore
verifies both the embedded scope->catalog declaration and, when supplied, an
external pair manifest.  It never rewrites either input.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def lower_hex(value: Any) -> str | None:
    if not isinstance(value, str) or len(value.strip()) != 64:
        return None
    text = value.strip().lower()
    return text if all(char in "0123456789abcdef" for char in text) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--pair", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scope_path = args.scope.resolve()
    catalog_path = args.catalog.resolve()
    report: dict[str, Any] = {
        "schema": "CANCERLNCATLAS_V32_SCOPE_CATALOG_PAIR_AUDIT_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope_path": str(scope_path),
        "catalog_path": str(catalog_path),
        "pair_path": str(args.pair.resolve()) if args.pair else None,
        "checks": {},
        "status": "FAIL",
    }
    try:
        scope = load_object(scope_path, "scope")
        catalog = load_object(catalog_path, "catalog")
        scope_sha = sha256_file(scope_path)
        catalog_sha = sha256_file(catalog_path)
        declared_catalog_sha = lower_hex(scope.get("catalog_sha256"))
        report["scope_sha256"] = scope_sha
        report["catalog_sha256"] = catalog_sha
        report["declared_scope_catalog_sha256"] = scope.get("catalog_sha256")
        report["checks"]["scope_catalog_hash"] = (
            declared_catalog_sha == catalog_sha
        )
        report["checks"]["scope_status"] = (
            scope.get("status") == "PASS"
            and scope.get("release_ready") is False
            and scope.get("production_deployed") is False
        )
        report["checks"]["catalog_status"] = (
            catalog.get("environment") == "staging"
            and catalog.get("release_ready") is False
            and catalog.get("production_deployed") is False
        )
        scope_binding = catalog.get("capability_scope")
        report["declared_catalog_scope_sha256"] = (
            scope_binding.get("sha256")
            if isinstance(scope_binding, dict)
            else None
        )
        report["checks"]["catalog_scope_hash"] = (
            isinstance(scope_binding, dict)
            and lower_hex(scope_binding.get("sha256")) == scope_sha
        )
        if args.pair:
            pair_path = args.pair.resolve()
            pair = load_object(pair_path, "pair manifest")
            report["pair_sha256"] = sha256_file(pair_path)
            report["pair"] = pair
            report["checks"]["pair_scope_hash"] = (
                lower_hex(pair.get("scope_sha256")) == scope_sha
            )
            report["checks"]["pair_catalog_hash"] = (
                lower_hex(pair.get("catalog_sha256")) == catalog_sha
            )
            report["checks"]["pair_status"] = pair.get("status") == "PASS"
        checks = report["checks"]
        report["status"] = "PASS" if all(checks.values()) else "FAIL"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
