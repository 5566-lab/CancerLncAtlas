#!/usr/bin/env python3
"""Promotion-blocking probes for the loopback V3.2 web candidate."""
from __future__ import annotations

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


REQUIRED_200 = (
    "/__candidate/health",
    "/api/meta",
    "/api/health",
    "/api/site/stats",
    "/api/site/search?" + urlencode({"q": "肺癌"}),
    "/api/site/sc-summary/BLCA",
    "/api/site/sc-umap/BLCA",
    "/api/site/cancer/BLCA",
    "/api/site/lncrna/MALAT1/visuals",
    "/api/site/genesets/PF%3A0001",
    "/api/site/clinical/visuals?" + urlencode({"cancer": "BLCA"}),
    "/api/site/mutation/status",
    "/api/site/mutation/cancer/BLCA",
    "/v2.7/health",
    "/v3.2-staging/health",
    "/v3.2-staging/release",
)


def _request(base: str, path: str) -> dict:
    started = time.monotonic()
    status = 0
    body = b""
    error = None
    try:
        with urlopen(Request(base + path, headers={"Accept": "application/json"}), timeout=30) as response:
            status = int(response.status)
            body = response.read(2 * 1024 * 1024)
    except HTTPError as exc:
        status = int(exc.code)
        body = exc.read(2 * 1024 * 1024)
        error = str(exc)
    except (URLError, TimeoutError, OSError) as exc:
        error = str(exc)
    try:
        parsed = json.loads(body) if body else None
    except json.JSONDecodeError:
        parsed = None
    return {
        "path": path,
        "status": status,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        "json": parsed,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8262")
    parser.add_argument("--output")
    args = parser.parse_args()
    parsed = urlparse(args.base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.port != 8262:
        raise SystemExit("Candidate probes are restricted to loopback port 8262")
    base = args.base_url.rstrip("/")
    results = [_request(base, path) for path in REQUIRED_200]
    by_path = {item["path"]: item for item in results}
    candidate = by_path["/__candidate/health"]
    candidate_body = candidate.get("json") if isinstance(candidate.get("json"), dict) else {}
    failures = [
        {"path": item["path"], "status": item["status"], "error": item["error"]}
        for item in results
        if item["status"] != 200
    ]
    if candidate_body.get("production_deployed") is not False:
        failures.append({"path": "/__candidate/health", "error": "candidate deployment flag invalid"})
    payload = {
        "format": "CANCERLNCATLAS_V32_WEB_CANDIDATE_PROBE_V1",
        "base_url": base,
        "required_probe_count": len(REQUIRED_200),
        "results": results,
        "failures": failures,
        "clinical_visuals_gate_pass": by_path[
            "/api/site/clinical/visuals?" + urlencode({"cancer": "BLCA"})
        ]["status"] == 200,
        "geneset_detail_gate_pass": by_path["/api/site/genesets/PF%3A0001"]["status"] == 200,
        "promotion_probe_pass": not failures,
        "production_deployed": False,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        from pathlib import Path
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
