#!/usr/bin/env python3
"""Read-only audit for the isolated V3.2 full-bindings probe manifest.

This script intentionally does not start a service, copy data, or mutate a
release tree.  It is designed to run on COMPUTE_HOST (149), where the r6
staging artifacts live.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import socket
import sys
from typing import Any, Iterable


SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PATH_KEY_RE = re.compile(
    r"(?:^|_)(?:path|file|root|dir|target|table|manifest|payload|artifact|tree|download|server)(?:$|_)",
    re.I,
)
SKIP_PREFIXES = ("http://", "https://", "s3://", "gs://", "ssh://")


def sha256_file(path: pathlib.Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def safe_json(path: pathlib.Path) -> tuple[Any | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh), None
    except Exception as exc:  # audit must report, not hide malformed files
        return None, f"{type(exc).__name__}: {exc}"


def within(root: pathlib.Path, path: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_manifest_path(root: pathlib.Path, value: str) -> tuple[pathlib.Path, str]:
    raw = pathlib.Path(value)
    if raw.is_absolute():
        return raw, "absolute"
    return root / raw, "repo_root_relative"


def is_path_string(value: Any, key: str) -> bool:
    if not isinstance(value, str) or not value or value.startswith(SKIP_PREFIXES):
        return False
    if value.startswith(("sha256:", "sha256/")):
        return False
    # A path-valued key is authoritative.  For free-form strings only inspect
    # obvious server paths; this avoids treating cancer names and URLs as files.
    if PATH_KEY_RE.search(key):
        return True
    return value.startswith(("${PRIVATE_WORK_ROOT}/", "${PRIVATE_WORK_ROOT}/", "${DATA_ROOT}/", "/tmp/"))


def walk_path_values(node: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{prefix}.{key}" if prefix else str(key)
            if is_path_string(value, str(key)):
                yield here, value
            yield from walk_path_values(value, here)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            yield from walk_path_values(value, f"{prefix}[{idx}]")


def flatten_sha_fields(node: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, str) and "sha256" in str(key).lower() and SHA_RE.match(value):
                yield here, value.lower()
            yield from flatten_sha_fields(value, here)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            yield from flatten_sha_fields(value, f"{prefix}[{idx}]")


def nearby_declared_sha(data: Any, path_key: str, path_value: str) -> str | None:
    """Find a sibling SHA for a deployment target where the schema has one."""
    # Most bindings use a sibling `<name>_sha256`; search the containing object
    # rather than guessing from arbitrary text.
    parts = re.split(r"[.\[]", path_key, maxsplit=1)
    # Re-walk by key path is needlessly brittle; a recursive sibling search is
    # sufficient and deterministic.
    wanted = pathlib.Path(path_value).name

    def visit(node: Any) -> str | None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and "sha256" in key.lower() and SHA_RE.match(value):
                    stem = key.lower().replace("_sha256", "").replace("sha256", "")
                    if not stem or stem in path_key.lower() or stem in wanted.lower():
                        return value.lower()
            for value in node.values():
                found = visit(value)
                if found:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = visit(value)
                if found:
                    return found
        return None

    return visit(data)


def check_path(
    *,
    root: pathlib.Path,
    label: str,
    value: str,
    declared_sha: str | None = None,
    hash_file: bool = True,
) -> dict[str, Any]:
    path, mode = resolve_manifest_path(root, value)
    item: dict[str, Any] = {
        "label": label,
        "declared_path": value,
        "resolution": mode,
        "resolved_path": str(path),
        "within_repo_root": within(root, path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "declared_sha256": declared_sha,
    }
    if path.exists() and path.is_file() and hash_file:
        try:
            actual = sha256_file(path)
            item["actual_sha256"] = actual
            item["sha_match"] = declared_sha is None or actual.lower() == declared_sha.lower()
            item["bytes"] = path.stat().st_size
        except Exception as exc:
            item["hash_error"] = f"{type(exc).__name__}: {exc}"
    elif path.exists():
        item["sha_match"] = None
    else:
        item["sha_match"] = False if declared_sha else None
    return item


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=pathlib.Path)
    ap.add_argument("--repo-root", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path)
    args = ap.parse_args()

    manifest = args.manifest.resolve()
    root = args.repo_root.resolve()
    data, parse_error = safe_json(manifest)
    report: dict[str, Any] = {
        "schema": "V32_FULL_BINDINGS_PROBE_AUDIT_V1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
        "repo_root": str(root),
        "manifest_parse_error": parse_error,
        "checks": [],
        "deployment_targets": [],
        "summary": {},
    }
    if parse_error or not isinstance(data, dict):
        report["summary"] = {"audit_status": "FAIL_MANIFEST_PARSE"}
    else:
        # Registry is a top-level special path, then every named binding.
        registry = data.get("registry")
        if isinstance(registry, dict) and isinstance(registry.get("path"), str):
            report["checks"].append(
                check_path(
                    root=root,
                    label="registry",
                    value=registry["path"],
                    declared_sha=registry.get("sha256"),
                )
            )
        bindings = data.get("bindings", {})
        if not isinstance(bindings, dict):
            bindings = {}
        for name, binding in bindings.items():
            if not isinstance(binding, dict):
                report["checks"].append(
                    {"label": f"binding:{name}", "error": "binding is not an object"}
                )
                continue
            if isinstance(binding.get("path"), str):
                report["checks"].append(
                    check_path(
                        root=root,
                        label=f"binding:{name}",
                        value=binding["path"],
                        declared_sha=binding.get("sha256"),
                    )
                )
            dep_value = binding.get("deployment_binding_path")
            if isinstance(dep_value, str):
                dep_check = check_path(
                    root=root,
                    label=f"deployment_binding:{name}",
                    value=dep_value,
                    declared_sha=binding.get("deployment_binding_sha256"),
                )
                report["checks"].append(dep_check)
                dep_path = pathlib.Path(dep_check["resolved_path"])
                dep_data, dep_error = safe_json(dep_path) if dep_path.is_file() else (None, "missing")
                dep_record: dict[str, Any] = {
                    "binding": name,
                    "deployment_binding": str(dep_path),
                    "parse_error": dep_error,
                    "target_checks": [],
                }
                if dep_error is None:
                    for path_key, path_value in walk_path_values(dep_data):
                        # The deployment binding itself often repeats its own
                        # path; skip that self-reference.
                        if path_value == dep_value or pathlib.Path(path_value).name == dep_path.name:
                            continue
                        target_sha = nearby_declared_sha(dep_data, path_key, path_value)
                        dep_record["target_checks"].append(
                            check_path(
                                root=root,
                                label=f"deployment_target:{name}:{path_key}",
                                value=path_value,
                                declared_sha=target_sha,
                                hash_file=bool(target_sha),
                            )
                        )
                report["deployment_targets"].append(dep_record)

        all_checks = report["checks"]
        missing = [x for x in all_checks if x.get("exists") is False]
        mismatched = [x for x in all_checks if x.get("sha_match") is False and x.get("exists")]
        malformed = [x for x in report["deployment_targets"] if x.get("parse_error")]
        target_checks = [x for d in report["deployment_targets"] for x in d["target_checks"]]
        target_missing = [x for x in target_checks if x.get("exists") is False]
        target_mismatch = [x for x in target_checks if x.get("sha_match") is False and x.get("exists")]
        report["summary"] = {
            "audit_status": "PASS" if not (missing or mismatched or malformed or target_missing or target_mismatch) else "FAIL",
            "manifest_bindings": len(bindings),
            "path_checks": len(all_checks),
            "path_missing": len(missing),
            "path_sha_mismatch": len(mismatched),
            "deployment_bindings": len(report["deployment_targets"]),
            "deployment_parse_failures": len(malformed),
            "deployment_target_checks": len(target_checks),
            "deployment_target_missing": len(target_missing),
            "deployment_target_sha_mismatch": len(target_mismatch),
            "full_staging_candidate": not (missing or mismatched or malformed or target_missing or target_mismatch),
        }
    output = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report["summary"].get("audit_status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
