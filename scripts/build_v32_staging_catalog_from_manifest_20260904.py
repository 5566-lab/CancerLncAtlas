#!/usr/bin/env python3
"""Rebuild the V3.2 staging catalog from one validated server manifest.

Only the catalog is written.  Scientific authority files are opened through
their hash-bound query loaders and are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_arg(value: str, *, root: Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or unsafe: {path}")
    return path


def _sha_arg(path: Path, value: str, label: str) -> str:
    expected = str(value).strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise RuntimeError(f"{label} SHA-256 is malformed")
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} SHA-256 drift: expected {expected}, observed {observed}")
    return expected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r11-handoff", type=Path, required=True)
    parser.add_argument("--r11-handoff-sha256", required=True)
    parser.add_argument("--r11-run-status", type=Path, required=True)
    parser.add_argument("--r11-run-status-sha256", required=True)
    parser.add_argument("--r11-success", type=Path, required=True)
    parser.add_argument("--r11-success-sha256", required=True)
    # The R7/HNSC gap bindings are retained as compatibility inputs, but the
    # current formal23 sidecar is sufficient for a new staging candidate.  An
    # omitted legacy pair is therefore valid; if one member is supplied, the
    # invocation below requires the complete path/SHA pair.
    parser.add_argument("--gap-binding", type=Path)
    parser.add_argument("--gap-binding-sha256")
    parser.add_argument("--gap-audit-binding", type=Path)
    parser.add_argument("--gap-audit-binding-sha256")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    manifest = _path_arg(str(args.manifest), root=repo_root, label="manifest")
    scope = _path_arg(str(args.scope), root=repo_root, label="capability scope")
    output = args.output.resolve()
    try:
        manifest.relative_to(repo_root)
        scope.relative_to(repo_root.parent)
        output.relative_to(repo_root.parent)
    except ValueError as exc:
        raise SystemExit("BLOCKED_CATALOG_PATH_ESCAPES_STAGING_ROOT") from exc
    scope_sha = _sha_arg(scope, args.scope_sha256, "capability scope")

    # Import after the path checks so a malformed invocation cannot trigger
    # any loader or data access outside the declared candidate.
    sys.path.insert(0, str(repo_root))
    from cc_hhgt.v32.unified_staging_bindings import load_unified_staging_bindings
    from scripts.build_v32_staging_web_catalog import build

    _registry, kwargs, _manifest_payload = load_unified_staging_bindings(
        manifest, repo_root=repo_root
    )

    def required_binding(name: str) -> tuple[Path, str]:
        path_key = f"{name}_path"
        sha_key = f"{name}_sha256"
        if path_key not in kwargs or sha_key not in kwargs:
            raise RuntimeError(f"unified manifest lacks {name} binding")
        path = _path_arg(kwargs[path_key], root=repo_root, label=name)
        return path, _sha_arg(path, kwargs[sha_key], name)

    formal23, formal23_sha = required_binding("single_cell_formal23_success")
    directional, directional_sha = required_binding("directional_cnv_binding")
    directional_audit, directional_audit_sha = required_binding(
        "directional_cnv_audit_binding"
    )

    handoff = _path_arg(str(args.r11_handoff), root=repo_root, label="R11 handoff")
    handoff_sha = _sha_arg(handoff, args.r11_handoff_sha256, "R11 handoff")
    run_status = _path_arg(
        str(args.r11_run_status), root=repo_root, label="R11 run status"
    )
    run_status_sha = _sha_arg(run_status, args.r11_run_status_sha256, "R11 run status")
    success = _path_arg(str(args.r11_success), root=repo_root, label="R11 success")
    success_sha = _sha_arg(success, args.r11_success_sha256, "R11 success")
    legacy_gap_values = (
        args.gap_binding,
        args.gap_binding_sha256,
        args.gap_audit_binding,
        args.gap_audit_binding_sha256,
    )
    if any(value is not None for value in legacy_gap_values) and not all(
        value is not None for value in legacy_gap_values
    ):
        raise RuntimeError(
            "legacy single-cell gap bindings must be supplied as a complete path/SHA pair"
        )
    if all(value is None for value in legacy_gap_values):
        gap = gap_sha = gap_audit = gap_audit_sha = None
    else:
        gap = _path_arg(str(args.gap_binding), root=repo_root, label="single-cell gap binding")
        gap_sha = _sha_arg(gap, args.gap_binding_sha256, "single-cell gap binding")
        gap_audit = _path_arg(
            str(args.gap_audit_binding), root=repo_root, label="single-cell gap audit binding"
        )
        gap_audit_sha = _sha_arg(
            gap_audit, args.gap_audit_binding_sha256, "single-cell gap audit binding"
        )

    payload: dict[str, Any] = build(
        single_cell_current_handoff_path=handoff,
        single_cell_current_handoff_sha256=handoff_sha,
        single_cell_current_run_status_path=run_status,
        single_cell_current_run_status_sha256=run_status_sha,
        single_cell_current_supersession_path=success,
        single_cell_current_supersession_sha256=success_sha,
        single_cell_release_binding_path=gap,
        single_cell_release_binding_sha256=gap_sha,
        single_cell_audit_binding_path=gap_audit,
        single_cell_audit_binding_sha256=gap_audit_sha,
        single_cell_formal23_success_path=formal23,
        single_cell_formal23_success_sha256=formal23_sha,
        directional_cnv_binding_path=directional,
        directional_cnv_binding_sha256=directional_sha,
        directional_cnv_audit_binding_path=directional_audit,
        directional_cnv_audit_binding_sha256=directional_audit_sha,
        capability_scope_path=scope,
        capability_scope_sha256=scope_sha,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), "sha256": _sha256(output)}))


if __name__ == "__main__":
    main()
