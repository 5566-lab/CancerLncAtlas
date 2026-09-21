"""Independent materialized-bundle verifier for portable V3.2 rebinding."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .production_binding_audit import sha256_file
from .portable_rebinding import PortableRebindingError, payload_summary


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_FILESYSTEM_POSIX = re.compile(
    r"^/(?:public\d+|dell_\d+|dsk\d+|home|mnt|opt|srv|var|tmp)(?:/|$)"
)


class PortableRebindingAuditError(RuntimeError):
    """Raised when the materialized candidate diverges from its lineage."""


def _walk_strings(value: Any):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def _materialized_path(
    target_path: str,
    *,
    declared_root: PurePosixPath,
    materialized_root: Path,
) -> Path:
    declared = PurePosixPath(target_path)
    try:
        relative = declared.relative_to(declared_root)
    except ValueError as exc:
        raise PortableRebindingAuditError(
            f"Target path escapes declared candidate root: {target_path}"
        ) from exc
    return materialized_root.joinpath(*relative.parts)


def verify_portable_bundle(
    lineage_path: str | Path,
    *,
    materialized_root: str | Path,
    allowed_server_roots: Iterable[str],
    require_payloads: bool = True,
) -> dict[str, Any]:
    lineage_source = Path(lineage_path).resolve()
    lineage = json.loads(lineage_source.read_text(encoding="utf-8"))
    if lineage.get("format") != "CANCERLNCATLAS_V32_PORTABLE_REBINDING_LINEAGE_V1":
        raise PortableRebindingAuditError("Portable rebinding lineage format mismatch")
    if lineage.get("production_deployed") is not False or lineage.get("release_ready") is not False:
        raise PortableRebindingAuditError("Candidate lineage may not claim release/deployment")
    declared_root = PurePosixPath(str(lineage["target_root"]))
    local_root = Path(materialized_root).resolve()
    allowed = tuple(str(PurePosixPath(item)) for item in allowed_server_roots)
    checks: list[dict[str, Any]] = []

    for record in lineage.get("json_rewrites", []):
        rebound = _materialized_path(
            record["target_path"],
            declared_root=declared_root,
            materialized_root=local_root,
        )
        original = _materialized_path(
            record["original_copy_target_path"],
            declared_root=declared_root,
            materialized_root=local_root,
        )
        rebound_ok = (
            not rebound.is_symlink()
            and rebound.is_file()
            and sha256_file(rebound) == record["rebound_sha256"]
        )
        original_ok = (
            not original.is_symlink()
            and original.is_file()
            and sha256_file(original) == record["source_sha256"]
        )
        checks.append({"kind": "rebound_json", "path": str(rebound), "pass": rebound_ok})
        checks.append({"kind": "original_json", "path": str(original), "pass": original_ok})

    payload_missing = 0
    payload_mismatch = 0
    for record in lineage.get("payload_invariants", []):
        target = _materialized_path(
            record["target_path"],
            declared_root=declared_root,
            materialized_root=local_root,
        )
        try:
            observed = payload_summary(target)
        except PortableRebindingError:
            observed = {}
        if record.get("kind") == "directory":
            passed = (
                observed.get("kind") == "directory"
                and observed.get("tree_sha256") == record.get("tree_sha256")
                and observed.get("file_count") == record.get("file_count")
                and observed.get("bytes") == record.get("bytes")
            )
        else:
            passed = (
                observed.get("kind") == "file"
                and observed.get("sha256") == record.get("sha256")
                and observed.get("bytes") == record.get("bytes")
            )
        if not target.exists():
            payload_missing += 1
        elif not passed:
            payload_mismatch += 1
        checks.append({"kind": "payload", "path": str(target), "pass": passed})

    candidate_manifest = _materialized_path(
        lineage["candidate_unified"]["target_path"],
        declared_root=declared_root,
        materialized_root=local_root,
    )
    manifest_sha_ok = (
        candidate_manifest.is_file()
        and sha256_file(candidate_manifest) == lineage["candidate_unified"]["sha256"]
    )
    checks.append(
        {"kind": "candidate_unified", "path": str(candidate_manifest), "pass": manifest_sha_ok}
    )
    copy_manifest = _materialized_path(
        lineage["copy_manifest"]["target_path"],
        declared_root=declared_root,
        materialized_root=local_root,
    )
    copy_manifest_ok = (
        not copy_manifest.is_symlink()
        and copy_manifest.is_file()
        and sha256_file(copy_manifest) == lineage["copy_manifest"]["sha256"]
    )
    checks.append(
        {"kind": "copy_manifest", "path": str(copy_manifest), "pass": copy_manifest_ok}
    )
    manifest = json.loads(candidate_manifest.read_text(encoding="utf-8")) if manifest_sha_ok else {}
    mounted = {
        capability: entry
        for capability, entry in manifest.get("bindings", {}).items()
        if entry.get("status") == "MOUNTED_HASH_PINNED"
    }
    path_policy_errors: list[dict[str, str]] = []
    mounted_targets = [manifest.get("registry", {}).get("path")]
    mounted_targets.extend(entry.get("path") for entry in mounted.values())
    for relative in mounted_targets:
        if not relative:
            continue
        source = (local_root / str(relative)).resolve()
        if not source.is_file():
            path_policy_errors.append({"path": str(source), "reason": "mounted_json_missing"})
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        for value in _walk_strings(payload):
            if _WINDOWS_ABSOLUTE.match(value):
                path_policy_errors.append({"path": value, "reason": "windows_absolute"})
            elif _FILESYSTEM_POSIX.match(value) and not any(
                value == root or value.startswith(f"{root}/") for root in allowed
            ):
                path_policy_errors.append({"path": value, "reason": "outside_authority"})

    failed_checks = sum(not item["pass"] for item in checks)
    accepted = (
        failed_checks == 0
        and not path_policy_errors
        and (not require_payloads or (payload_missing == 0 and payload_mismatch == 0))
        and manifest.get("production_deployed") is False
        and manifest.get("release_ready") is False
        and lineage.get("registry_transitive_unresolved_count") == 0
    )
    return {
        "format": "CANCERLNCATLAS_V32_PORTABLE_REBINDING_INDEPENDENT_AUDIT_V1",
        "lineage": {"path": str(lineage_source), "sha256": sha256_file(lineage_source)},
        "materialized_root": str(local_root),
        "candidate_unified": {
            "path": str(candidate_manifest),
            "sha256": sha256_file(candidate_manifest) if candidate_manifest.is_file() else None,
        },
        "mounted_capability_count": len(mounted),
        "pending_capability_count": sum(
            entry.get("status") == "PENDING_FORMAL_SUCCESS"
            for entry in manifest.get("bindings", {}).values()
        ),
        "checks": len(checks),
        "failed_checks": failed_checks,
        "payload_missing": payload_missing,
        "payload_mismatch": payload_mismatch,
        "path_policy_errors": path_policy_errors,
        "registry_transitive_unresolved_count": lineage.get(
            "registry_transitive_unresolved_count"
        ),
        "runtime_create_app_smoke_required": True,
        "accepted_for_candidate_start": accepted,
        "production_deployed": False,
        "release_ready": False,
    }


__all__ = [
    "PortableRebindingAuditError",
    "verify_portable_bundle",
]
