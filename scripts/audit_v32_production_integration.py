#!/usr/bin/env python3
"""Audit production route parity and V3.2 binding portability without writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cc_hhgt.v32.production_binding_audit import (
    audit_binding_portability,
    decorated_routes,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITATIVE_SHA256 = (
    "39702f422f3926e858a3c61956a4193fd7d84216fecec57124d34eec325ad157"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--authoritative-app",
        type=Path,
        default=ROOT.parent / "remote_original_r_figures/website/backend/app.py",
    )
    parser.add_argument(
        "--repo-app", type=Path, default=ROOT / "website/backend/app.py"
    )
    parser.add_argument(
        "--v32-api", type=Path, default=ROOT / "website/backend/v32_staging_api.py"
    )
    parser.add_argument(
        "--unified",
        type=Path,
        default=ROOT / "config/v32_unified_staging_bindings.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    authoritative_hash = sha256_file(args.authoritative_app)
    production = decorated_routes(args.authoritative_app)
    repo = decorated_routes(args.repo_app)
    v32 = decorated_routes(args.v32_api)
    portability = audit_binding_portability(
        args.unified,
        repo_root=args.repo_root,
        allowed_server_roots=(
            "./data/CancerLncAtlas",
            "./data/CancerLncAtlas",
        ),
    )
    payload = {
        "format": "CANCERLNCATLAS_V32_PRODUCTION_INTEGRATION_AUDIT_V1",
        "authoritative_app": {
            "path": str(args.authoritative_app.resolve()),
            "sha256": authoritative_hash,
            "expected_sha256": AUTHORITATIVE_SHA256,
            "hash_matches": authoritative_hash == AUTHORITATIVE_SHA256,
            "decorated_route_count": len(production),
        },
        "repo_app": {
            "path": str(args.repo_app.resolve()),
            "sha256": sha256_file(args.repo_app),
            "decorated_route_count": len(repo),
            "missing_production_routes": [
                {"method": method, "path": path}
                for method, path in sorted(production - repo)
            ],
            "covers_authoritative_production": production <= repo,
        },
        "v32_api": {
            "path": str(args.v32_api.resolve()),
            "sha256": sha256_file(args.v32_api),
            "decorated_route_count": len(v32),
            "conflicts_with_production": [
                {"method": method, "path": path}
                for method, path in sorted(production & v32)
            ],
        },
        "binding_portability": portability,
        "safe_to_replace_production_with_repo_app": False,
        "safe_to_upload_unified_binding_without_rebinding": portability[
            "production_portable"
        ],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if authoritative_hash == AUTHORITATIVE_SHA256 else 2


if __name__ == "__main__":
    raise SystemExit(main())
