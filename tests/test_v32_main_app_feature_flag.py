from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "website" / "frontend"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(enabled: bool) -> dict[str, object]:
    program = r'''
import hashlib
import json
from fastapi.testclient import TestClient
from website.backend import app as module

paths = (
    "/v32-staging.html",
    "/assets/v32-staging.js",
    "/v32-capability-catalog.json",
    "/v3.2-staging/health",
)
with TestClient(module.app) as client:
    rows = {}
    for path in paths:
        response = client.get(path)
        rows[path] = {
            "status": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "sha256": hashlib.sha256(response.content).hexdigest(),
        }
print(json.dumps({
    "enabled": module.app.state.v32_staging_enabled,
    "rows": rows,
}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment["CANCERLNCATLAS_REQUIRE_V31_EXACT_PATHWAY"] = "0"
    environment["CANCERLNCATLAS_ENABLE_V32_STAGING"] = "1" if enabled else "0"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src"), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_explicit_flag_serves_hash_exact_v32_site_and_api() -> None:
    result = _probe(True)
    assert result["enabled"] is True
    rows = result["rows"]
    expected = {
        "/v32-staging.html": FRONTEND / "v32-staging.html",
        "/assets/v32-staging.js": FRONTEND / "assets" / "v32-staging.js",
        "/v32-capability-catalog.json": FRONTEND / "v32-capability-catalog.json",
    }
    for route, source in expected.items():
        assert rows[route]["status"] == 200
        assert rows[route]["sha256"] == _sha256(source)
    assert rows["/v3.2-staging/health"]["status"] == 200
    assert rows["/v3.2-staging/health"]["content_type"].startswith(
        "application/json"
    )


def test_default_disabled_mode_fails_closed_for_v32_surface() -> None:
    result = _probe(False)
    assert result["enabled"] is False
    assert result["rows"]["/v32-staging.html"]["status"] == 404
    assert result["rows"]["/v32-capability-catalog.json"]["status"] == 404
    assert result["rows"]["/v3.2-staging/health"]["status"] == 404
