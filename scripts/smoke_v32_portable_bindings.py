#!/usr/bin/env python3
"""Import and probe a materialized portable binding graph before web launch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from cc_hhgt.v32.production_binding_audit import sha256_file
from cc_hhgt.v32.unified_staging_bindings import create_app_from_unified_bindings


PROBES = (
    "/v3.2-staging/health",
    "/v3.2-staging/release",
    "/v3.2-staging/cancers",
    "/v3.2-staging/datasets",
    "/v3.2-staging/scoring/architecture",
)


def _write(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--unified", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    unified = args.unified.resolve()
    try:
        unified.relative_to(repo)
        app = create_app_from_unified_bindings(unified, repo_root=repo)
        client = TestClient(app)
        probes = [
            {"path": path, "status": client.get(path).status_code}
            for path in PROBES
        ]
        errors = [item for item in probes if item["status"] != 200]
        route_count = len(app.routes)
        exception = None
    except Exception as exc:  # fail-closed audit boundary
        probes = []
        errors = [{"error": f"{type(exc).__name__}: {exc}"}]
        route_count = 0
        exception = errors[0]["error"]
    accepted = not errors and route_count > 0
    report = {
        "format": "CANCERLNCATLAS_V32_PORTABLE_CREATE_APP_SMOKE_V1",
        "repo_root": str(repo),
        "candidate_unified": {"path": str(unified), "sha256": sha256_file(unified)},
        "route_count": route_count,
        "probes": probes,
        "errors": errors,
        "exception": exception,
        "accepted_for_candidate_web_import": accepted,
        "production_deployed": False,
        "release_ready": False,
    }
    report_path = args.output_dir / "PORTABLE_CREATE_APP_SMOKE.json"
    report_sha = _write(report_path, report)
    binding = {
        "format": "CANCERLNCATLAS_V32_PORTABLE_CREATE_APP_SMOKE_BINDING_V1",
        "report": {"path": str(report_path), "sha256": report_sha},
        "candidate_unified": report["candidate_unified"],
        "accepted_for_candidate_web_import": accepted,
        "production_deployed": False,
        "release_ready": False,
    }
    binding_path = args.output_dir / "PORTABLE_CREATE_APP_SMOKE_BINDING.json"
    binding_sha = _write(binding_path, binding)
    print(json.dumps({
        "accepted_for_candidate_web_import": accepted,
        "binding": str(binding_path),
        "binding_sha256": binding_sha,
    }, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
