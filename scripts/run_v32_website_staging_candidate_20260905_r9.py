#!/usr/bin/env python3
"""Start the hash-bound r9 V3.2 staging API on server 149 only.

This wrapper is deliberately server-only and never starts a GPU or touches
the production listener.  The r9 scope/catalog pair audit must pass before
the API process is opened.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path


ROOT = Path("./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r9")
CODE = ROOT / "code"
MANIFEST = CODE / "config/v32_server_unified_staging_bindings_20260905_r9.json"
CATALOG = ROOT / "catalog_candidate_20260905_r9/v32-capability-catalog_r9.json"
SCOPE = ROOT / "capability_scope_20260905_r9.json"
PAIR = ROOT / "receipts/R9_SCOPE_CATALOG_PAIR_20260905.json"
PAIR_AUDIT = ROOT / "receipts/R9_SCOPE_CATALOG_AUDIT_20260905.json"
PAIR_CHECKER = CODE / "scripts/audit_v32_r9_scope_catalog_pair.py"
SERVER = CODE / "scripts/serve_v32_unified_staging_candidate.py"


def _host() -> str:
    try:
        return subprocess.check_output(["hostname", "-s"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return socket.gethostname().split(".", 1)[0]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("V32_R9_PORT", "8303")))
    return parser.parse_args()


def main() -> int:
    args = _args()
    if _host() != "149":
        raise SystemExit(f"BLOCKED_WRONG_HOST expected=149 observed={_host()}")
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("BLOCKED_NON_LOOPBACK_STAGING_BIND")
    if args.port in {8260, 8261, 8262}:
        raise SystemExit(f"BLOCKED_PRODUCTION_OR_LEGACY_PORT={args.port}")
    required = (CODE, MANIFEST, CATALOG, SCOPE, PAIR, PAIR_AUDIT, PAIR_CHECKER, SERVER)
    if any(not path.exists() for path in required):
        raise SystemExit("BLOCKED_MISSING_R9_STAGING_ARTIFACT")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CODE)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Re-run the independent metadata gate immediately before startup.  It
    # writes only the r9 pair/audit receipts inside this isolated candidate.
    audit = subprocess.run(
        [
            sys.executable,
            str(PAIR_CHECKER),
            "--root", str(ROOT),
            "--scope", str(SCOPE),
            "--catalog", str(CATALOG),
            "--manifest", str(MANIFEST),
            "--frontend-catalog", str(CODE / "website/frontend/v32-capability-catalog.json"),
            "--pair", str(PAIR),
            "--output", str(PAIR_AUDIT),
        ],
        cwd=str(CODE),
        env=env,
        check=False,
    )
    if audit.returncode:
        raise SystemExit("BLOCKED_R9_SCOPE_CATALOG_PAIR_AUDIT")
    # The server entry point performs the manifest hash gate again before
    # opening a socket.  exec-style replacement preserves a single process
    # and makes external stop/monitoring unambiguous.
    os.execve(
        sys.executable,
        [
            sys.executable,
            str(SERVER),
            "--repo-root", str(CODE),
            "--manifest", str(MANIFEST),
            "--catalog", str(CATALOG),
            "--host", args.host,
            "--port", str(args.port),
        ],
        env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
