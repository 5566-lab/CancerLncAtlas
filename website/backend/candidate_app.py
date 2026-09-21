"""Isolated blue/green candidate entry point.

Launch this module on a non-production loopback port.  Import fails closed
unless the cloned authoritative app and the portable V3.2 manifest match their
operator-pinned SHA-256 values.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for the isolated candidate")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_gate(path_env: str, sha_env: str, *, expected_format: str, accepted_key: str) -> dict:
    source = Path(_required_env(path_env)).resolve()
    expected = _required_env(sha_env).lower()
    if not _SHA256.fullmatch(expected) or _sha256(source) != expected:
        raise RuntimeError(f"{path_env} SHA-256 mismatch")
    payload = __import__("json").loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != expected_format or payload.get(accepted_key) is not True:
        raise RuntimeError(f"{path_env} was not accepted")
    if payload.get("production_deployed") is not False or payload.get("release_ready") is not False:
        raise RuntimeError(f"{path_env} has invalid candidate status")
    return payload


if _required_env("CANCERLNCATLAS_V32_CANDIDATE_MODE") != "1":
    raise RuntimeError("candidate_app refuses to load outside candidate mode")

_authoritative_app = Path(__file__).with_name("app.py").resolve()
_authoritative_sha = _required_env(
    "CANCERLNCATLAS_AUTHORITATIVE_APP_SHA256"
).lower()
if not _SHA256.fullmatch(_authoritative_sha):
    raise RuntimeError("Invalid authoritative app SHA-256 declaration")
if _sha256(_authoritative_app) != _authoritative_sha:
    raise RuntimeError("The cloned authoritative app SHA-256 does not match")

_repo_root = Path(_required_env("CANCERLNCATLAS_V32_REPO_ROOT")).resolve()
_unified = Path(
    _required_env("CANCERLNCATLAS_V32_PORTABLE_BINDINGS")
).resolve()
_unified_sha = _required_env(
    "CANCERLNCATLAS_V32_PORTABLE_BINDINGS_SHA256"
).lower()
try:
    _unified.relative_to(_repo_root)
except ValueError as exc:
    raise RuntimeError("Portable bindings must stay under the candidate root") from exc
if not _SHA256.fullmatch(_unified_sha) or _sha256(_unified) != _unified_sha:
    raise RuntimeError("Portable unified binding SHA-256 mismatch")

_rebind_audit = _verified_gate(
    "CANCERLNCATLAS_V32_REBIND_AUDIT_BINDING",
    "CANCERLNCATLAS_V32_REBIND_AUDIT_BINDING_SHA256",
    expected_format="CANCERLNCATLAS_V32_PORTABLE_REBINDING_INDEPENDENT_AUDIT_BINDING_V1",
    accepted_key="accepted_for_candidate_start",
)
_portable_smoke = _verified_gate(
    "CANCERLNCATLAS_V32_PORTABLE_SMOKE_BINDING",
    "CANCERLNCATLAS_V32_PORTABLE_SMOKE_BINDING_SHA256",
    expected_format="CANCERLNCATLAS_V32_PORTABLE_CREATE_APP_SMOKE_BINDING_V1",
    accepted_key="accepted_for_candidate_web_import",
)
for _gate in (_rebind_audit, _portable_smoke):
    _declared = _gate.get("candidate_unified")
    if not isinstance(_declared, dict) or _declared.get("sha256") != _unified_sha:
        raise RuntimeError("Candidate gate is tied to a different unified binding")

# Importing the full app is intentional: the candidate preserves every route,
# frontend asset and lazy loader from the actual production release.
from website.backend.app import app  # noqa: E402
from cc_hhgt.v32.production_candidate import install_candidate_routes  # noqa: E402
from cc_hhgt.v32.unified_staging_bindings import (  # noqa: E402
    create_app_from_unified_bindings,
)


_v32_app = create_app_from_unified_bindings(_unified, repo_root=_repo_root)
_integration = install_candidate_routes(
    app,
    _v32_app,
    candidate_metadata={
        "authoritative_app_path": str(_authoritative_app),
        "authoritative_app_sha256": _authoritative_sha,
        "portable_bindings_path": str(_unified),
        "portable_bindings_sha256": _unified_sha,
        "candidate_port_required": 8262,
    },
)


__all__ = ["app"]
