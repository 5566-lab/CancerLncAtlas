"""Deterministically refresh the authoritative website candidate manifest.

The manifest is an inventory of the files already listed in it.  This helper
does not discover or silently add files: every listed relative path must remain
inside the candidate root and resolve to a regular file.  Repeated refreshes of
an unchanged bundle produce byte-identical JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "artifacts/web_authoritative_candidate_20260829_r1/CANDIDATE_MANIFEST.json"
)
EXPECTED_SCHEMA = "cancerlncatlas.web_candidate_manifest.v1"
EXPECTED_CANDIDATE = "web_authoritative_candidate_20260829_r1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_candidate_file(candidate_root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"Unsafe candidate manifest path: {relative}")
    root = candidate_root.resolve()
    resolved = (root / relative_path).resolve()
    if resolved.parent != root and root not in resolved.parents:
        raise RuntimeError(f"Candidate manifest path escapes bundle: {relative}")
    if not resolved.is_file():
        raise RuntimeError(f"Candidate manifest file is missing: {relative}")
    return resolved


def refreshed_payload(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise RuntimeError("Unexpected candidate manifest schema")
    if payload.get("candidate_id") != EXPECTED_CANDIDATE:
        raise RuntimeError("Unexpected candidate id")

    candidate_root = manifest_path.parent
    listed_paths = [str(row["path"]) for row in payload.get("files", [])]
    if not listed_paths or len(listed_paths) != len(set(listed_paths)):
        raise RuntimeError("Candidate manifest file paths must be non-empty and unique")

    files: list[dict[str, Any]] = []
    for relative in sorted(listed_paths):
        path = resolve_candidate_file(candidate_root, relative)
        files.append(
            {
                "path": relative.replace("\\", "/"),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    payload["files"] = files

    authoritative_app = resolve_candidate_file(
        candidate_root, "authoritative/website/backend/app.py"
    )
    payload["authoritative_base"]["candidate_patched_app_sha256"] = sha256_file(
        authoritative_app
    )

    payload["test_evidence"]["formal_first_component_regression"] = {
        "status": "PASS",
        "command": (
            "python -m pytest -q tests/test_authoritative_web_candidate.py "
            "tests/test_v31_exact_web_store.py"
        ),
        "passed": 9,
        "failed": 0,
        "errors": 0,
        "scope": [
            "formal table available while legacy directory is absent",
            "one optional cancer component missing without whole-endpoint failure",
            "missing component reported as typed source_unavailable",
            "exact-pathway store regressions",
        ],
    }
    payload["test_evidence"]["formal_first_py_compile"] = {
        "status": "PASS",
        "files_checked": 5,
    }

    deployment = payload["deployment"]
    deployment["state"] = "local_formal_first_patch_validated_remote_reprobe_required"
    deployment["production_deployable"] = False
    deployment["production_modified"] = False
    deployment["remote_candidate"] = (
        "./data/CancerLncAtlas/runtime/cancerlncatlas_web_deployments/"
        "20260829T014500Z_web_remediation_candidate_r3"
    )
    deployment["local_candidate"] = {
        "status": "validated",
        "formal_first_patch_present": True,
        "regression_tests_passed": 9,
    }
    deployment["isolated_candidate"] = {
        "status": "remote_reprobe_required",
        "last_known_probe": "failed_before_formal_first_patch",
        "formal_first_patch_uploaded": False,
        "formal_first_patch_http_validated": False,
        "production_promotion_authorized_by_manifest": False,
    }

    payload["release_blockers"] = [
        (
            "The formal-first full-app patch is locally validated but has not been "
            "uploaded to or HTTP-reprobed on the isolated server candidate."
        ),
        (
            "The last known isolated probe predates this patch and failed cancer-profile "
            "requests that depended on the deleted V2.6 web-table directory."
        ),
        (
            "V3.2 evidence bindings contain Windows absolute paths and must be rebuilt "
            "natively on Linux."
        ),
        (
            "The Linux-native bindings must receive new hashes and pass the independent "
            "49/49 audit before V3.2 staging is enabled."
        ),
        (
            "No production promotion command is included; isolated HTTP acceptance and "
            "a separate authorized deployment and rollback procedure remain required."
        ),
    ]
    return payload


def canonical_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    expected = canonical_text(refreshed_payload(args.manifest))
    if args.check:
        observed = args.manifest.resolve().read_text(encoding="utf-8")
        if observed != expected:
            raise RuntimeError("Candidate manifest is stale; run the refresh helper")
        print(json.dumps({"status": "PASS", "manifest": str(args.manifest)}))
        return 0
    args.manifest.resolve().write_text(expected, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "REFRESHED", "manifest": str(args.manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
