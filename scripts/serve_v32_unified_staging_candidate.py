#!/usr/bin/env python3
"""Serve one hash-bound V3.2 staging candidate on a loopback port.

This entry point is intentionally server-only.  It refuses to run anywhere
whose short hostname is not ``149`` and refuses non-loopback/production ports.
The application is built from the unified binding manifest before uvicorn is
started, so a stale or out-of-root sidecar fails before a socket is opened.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys


def _short_hostname() -> str:
    try:
        value = subprocess.check_output(
            ["hostname", "-s"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        value = socket.gethostname().split(".", 1)[0]
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8295)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if _short_hostname() != "149":
        raise SystemExit(
            f"BLOCKED_WRONG_HOST expected=149 observed={_short_hostname()}"
        )
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("BLOCKED_NON_LOOPBACK_STAGING_BIND")
    if args.port in {8260, 8261, 8262}:
        raise SystemExit(f"BLOCKED_PRODUCTION_OR_LEGACY_PORT={args.port}")

    repo_root = args.repo_root.resolve()
    manifest = args.manifest.resolve()
    catalog = args.catalog.resolve()
    if not repo_root.is_dir() or not manifest.is_file():
        raise SystemExit("BLOCKED_MISSING_STAGING_ROOT_OR_MANIFEST")
    if not catalog.is_file() or catalog.is_symlink():
        raise SystemExit("BLOCKED_MISSING_STAGING_CATALOG")
    try:
        manifest.relative_to(repo_root)
        catalog.relative_to(repo_root.parent)
    except ValueError as exc:
        raise SystemExit("BLOCKED_STAGING_PATH_ESCAPES_ALLOWED_ROOT") from exc

    # The overlay contains only corrected runtime modules.  Put it first, but
    # keep the complete copied candidate tree available as the fallback.
    roots = []
    if args.overlay_root is not None:
        overlay = args.overlay_root.resolve()
        if not overlay.is_dir():
            raise SystemExit(f"BLOCKED_MISSING_CODE_OVERLAY={overlay}")
        roots.append(overlay)
    roots.append(repo_root)
    for root in reversed(roots):
        sys.path.insert(0, str(root))
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    from cc_hhgt.v32.unified_staging_bindings import (  # noqa: PLC0415
        create_app_from_unified_bindings,
    )
    from fastapi.responses import FileResponse  # noqa: PLC0415
    from fastapi.staticfiles import StaticFiles  # noqa: PLC0415

    app = create_app_from_unified_bindings(manifest, repo_root=repo_root)

    frontend_root = repo_root / "website" / "frontend"
    assets_root = frontend_root / "assets"
    if not frontend_root.is_dir() or frontend_root.is_symlink():
        raise SystemExit("BLOCKED_MISSING_STAGING_FRONTEND")
    if not assets_root.is_dir() or assets_root.is_symlink():
        raise SystemExit("BLOCKED_MISSING_STAGING_ASSETS")
    # A page without its static bundle is not a publishable staging
    # candidate.  Fail before opening the socket rather than serving a
    # misleading half-rendered workbench.
    for required_asset in (
        "v32-staging.js",
        "v32-staging.css",
        "styles.css",
        "mark.svg",
    ):
        asset = assets_root / required_asset
        if not asset.is_file() or asset.is_symlink():
            raise SystemExit(f"BLOCKED_MISSING_STAGING_ASSET={required_asset}")

    app.mount(
        "/assets",
        StaticFiles(directory=str(assets_root), check_dir=True),
        name="staging-assets",
    )

    @app.get("/v32-capability-catalog.json", include_in_schema=False)
    def capability_catalog() -> FileResponse:
        return FileResponse(catalog, media_type="application/json")

    @app.get("/v32-staging.html", include_in_schema=False)
    def staging_page() -> FileResponse:
        page = frontend_root / "v32-staging.html"
        if not page.is_file() or page.is_symlink():
            raise FileNotFoundError(page)
        return FileResponse(page, media_type="text/html")

    @app.get("/", include_in_schema=False)
    def staging_root() -> FileResponse:
        page = frontend_root / "v32-staging.html"
        if not page.is_file() or page.is_symlink():
            raise FileNotFoundError(page)
        return FileResponse(page, media_type="text/html")

    import uvicorn  # noqa: PLC0415

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=True,
    )


if __name__ == "__main__":
    main()
