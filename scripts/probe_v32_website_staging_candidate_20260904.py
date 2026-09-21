#!/usr/bin/env python3
"""Run a bounded HTTP smoke and write a hash-bound capability scope.

The probe is deliberately small: it exercises the catalog/release contract,
the corrected CNV overlay, the formal-23 single-cell overlay (including a
typed-unavailable cancer), and the mixed lncRNA/protein pathway query. It does
not enumerate large tables or touch any scientific authority files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hostname() -> str:
    try:
        return subprocess.check_output(
            ["hostname", "-s"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return socket.gethostname().split(".", 1)[0]


def _json_request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any, str | None]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(
        base_url.rstrip("/") + path,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=20) as response:  # nosec B310
            raw = response.read()
            status = int(response.status)
    except HTTPError as exc:
        raw = exc.read()
        status = int(exc.code)
    except (OSError, URLError, TimeoutError) as exc:
        return 0, None, f"{type(exc).__name__}: {exc}"
    try:
        return status, json.loads(raw.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, raw.decode("utf-8", errors="replace"), None


def _raw_request(
    base_url: str, path: str
) -> tuple[int, str | None, int, str | None]:
    """Fetch one finite static resource without assuming a JSON body."""
    request = Request(
        base_url.rstrip("/") + path,
        headers={"Accept": "*/*"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=20) as response:  # nosec B310
            raw = response.read()
            return (
                int(response.status),
                response.headers.get_content_type(),
                len(raw),
                None,
            )
    except HTTPError as exc:
        raw = exc.read()
        return int(exc.code), exc.headers.get_content_type(), len(raw), None
    except (OSError, URLError, TimeoutError) as exc:
        return 0, None, 0, f"{type(exc).__name__}: {exc}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8295")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--scope-sha256")
    parser.add_argument(
        "--allow-stale-catalog",
        action="store_true",
        help="Bootstrap mode: record route statuses without failing on old catalog text",
    )
    return parser.parse_args()


def _scope_template() -> dict[str, Any]:
    # Import the canonical list rather than maintaining a second capability
    # universe. The import also acts as a dependency/preflight check.
    from scripts.build_v32_staging_web_catalog import STAGING_BINDINGS

    return {
        capability_id: {
            "staging_status": "PENDING_FORMAL_BINDING",
            "verified_endpoints": [],
            "known_gap": (
                "No route was included in the bounded staging smoke; this "
                "capability remains explicitly unverified in the isolated "
                "candidate."
            ),
            "probe_id": None,
        }
        for capability_id in STAGING_BINDINGS
    }


def _catalog_semantics(
    catalog: Any, *, expected_scope_sha256: str | None = None
) -> tuple[bool, str | None]:
    if not isinstance(catalog, dict):
        return False, "catalog response is not an object"
    if catalog.get("environment") != "staging":
        return False, "catalog environment is not staging"
    if catalog.get("production_deployed") is not False:
        return False, "catalog claims production deployment"
    if catalog.get("release_ready") is not False:
        return False, "catalog claims release readiness"
    if expected_scope_sha256 is not None:
        scope_binding = catalog.get("capability_scope")
        if not isinstance(scope_binding, dict):
            return False, "catalog lacks the hash-bound capability scope"
        if (
            scope_binding.get("sha256") != expected_scope_sha256.lower()
            or scope_binding.get("probe_status") != "PASS"
        ):
            return False, "catalog capability-scope hash/status drift"
    rows = catalog.get("capabilities")
    if not isinstance(rows, list):
        return False, "catalog capabilities is not a list"
    single_cell = next(
        (
            item
            for item in rows
            if isinstance(item, dict) and item.get("capability_id") == "single_cell"
        ),
        None,
    )
    if single_cell is None:
        return False, "catalog lacks single_cell capability"
    scope = single_cell.get("accepted_release_scope")
    if not isinstance(scope, dict):
        return False, "catalog lacks accepted single-cell scope"
    if (
        scope.get("formal_eligible_cancer_count") != 23
        or scope.get("typed_unavailable_cancer_count") != 10
        or scope.get("full_33_single_cell_coverage_claimed") is not False
    ):
        return False, "catalog single-cell scope is not 23+10 typed-unavailable"
    serialized = json.dumps(single_cell, ensure_ascii=False, sort_keys=True)
    stale_fragments = (
        "fresh outputs remain 0/23",
        '"current_fresh_derived_cancers": 0',
        '"derived_assets_bound": false',
        "PENDING_FORMAL23_RECOMPUTE_AND_AUDIT",
        "17-cancer UCell",
        "17/33",
    )
    for fragment in stale_fragments:
        if fragment in serialized:
            return False, f"stale single-cell catalog text: {fragment}"
    return True, None


def _cnv_semantics(body: Any) -> tuple[bool, str | None]:
    if not isinstance(body, dict) or int(body.get("total_rows", 0)) <= 0:
        return False, "CNV query returned no rows"
    provenance = body.get("provenance")
    if not isinstance(provenance, dict):
        return False, "CNV response lacks provenance"
    expected_false = (
        "old_checkpoint_loaded",
        "old_predictions_used_as_features",
        "old_rankings_used_as_outputs",
    )
    if any(provenance.get(key) is not False for key in expected_false):
        return False, "CNV response contains a stale-input provenance flag"
    if provenance.get("new_training_attestation") is not True:
        return False, "CNV response lacks new-training attestation"
    return True, None


def _single_cell_semantics(
    body: Any, *, unavailable: bool = False
) -> tuple[bool, str | None]:
    if not isinstance(body, dict):
        return False, "single-cell response is not an object"
    if unavailable:
        if body.get("availability") is not False and body.get("status") not in {
            "TYPED_UNAVAILABLE",
            "AUDITED_UNAVAILABLE",
        }:
            text = json.dumps(body, ensure_ascii=False)
            if "typed" not in text.lower() and "unavailable" not in text.lower():
                return False, "typed-unavailable cancer was not explicit"
        return True, None
    if body.get("status") not in {"PASS", "SUCCESS", "READY", "AVAILABLE"}:
        if body.get("formal_eligible_cancer_count") != 23:
            return False, "formal single-cell capability is not successful"
    return True, None


def main() -> None:
    args = _parse_args()
    if _hostname() != "149":
        raise SystemExit(f"BLOCKED_WRONG_HOST expected=149 observed={_hostname()}")
    if not (
        args.base_url.startswith("http://127.0.0.1:")
        or args.base_url.startswith("http://localhost:")
    ):
        raise SystemExit("BLOCKED_NON_LOOPBACK_PROBE_URL")
    repo_root = args.repo_root.resolve()
    manifest = args.manifest.resolve()
    catalog_path = args.catalog.resolve()
    output = args.output.resolve()
    if not manifest.is_file() or not catalog_path.is_file():
        raise SystemExit("BLOCKED_MISSING_MANIFEST_OR_CATALOG")
    expected_scope_sha256 = None
    if args.scope is not None or args.scope_sha256 is not None:
        if args.scope is None or args.scope_sha256 is None:
            raise SystemExit("BLOCKED_SCOPE_BINDING_REQUIRES_PATH_AND_SHA256")
        scope_path = args.scope.resolve()
        if not scope_path.is_file() or scope_path.is_symlink():
            raise SystemExit("BLOCKED_MISSING_OR_UNSAFE_CAPABILITY_SCOPE")
        expected_scope_sha256 = _sha256(scope_path)
        if expected_scope_sha256 != args.scope_sha256.strip().lower():
            raise SystemExit("BLOCKED_CAPABILITY_SCOPE_SHA256_DRIFT")
    try:
        manifest.relative_to(repo_root)
        output.relative_to(repo_root.parent)
    except ValueError as exc:
        raise SystemExit("BLOCKED_PROBE_OUTPUT_ESCAPES_STAGING_ROOT") from exc

    probes: list[dict[str, Any]] = []

    def probe(
        probe_id: str,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        canonical_endpoint: str | None = None,
    ) -> tuple[int, Any]:
        status, body, error = _json_request(args.base_url, method, path, payload)
        probes.append(
            {
                "probe_id": probe_id,
                "method": method,
                "path": path,
                "canonical_endpoint": canonical_endpoint,
                "http_status": status,
                "error": error,
            }
        )
        return status, body

    overall_ok = True
    status, health = probe("health", "GET", "/v3.2-staging/health")
    health_ok = (
        status == 200
        and isinstance(health, dict)
        and health.get("environment") == "staging"
        and health.get("production_deployed") is False
    )
    overall_ok &= health_ok
    probes[-1]["semantic_ok"] = health_ok

    # The API probes below prove query semantics.  These finite requests also
    # prove that the isolated candidate can actually serve the publishable
    # frontend shell and assets; they never enumerate the static tree.
    static_resources = (
        ("frontend_root", "/", "text/html"),
        ("frontend_staging_html", "/v32-staging.html", "text/html"),
        ("frontend_staging_js", "/assets/v32-staging.js", "text/javascript"),
        ("frontend_staging_css", "/assets/v32-staging.css", "text/css"),
        ("frontend_shared_css", "/assets/styles.css", "text/css"),
        ("frontend_mark_svg", "/assets/mark.svg", "image/svg+xml"),
    )
    for probe_id, path, expected_content_type in static_resources:
        static_status, content_type, byte_count, static_error = _raw_request(
            args.base_url, path
        )
        static_ok = (
            static_status == 200
            and byte_count > 0
            and content_type is not None
            and (
                content_type == expected_content_type
                or content_type.startswith(expected_content_type + ";")
                or (
                    expected_content_type == "text/javascript"
                    and content_type == "application/javascript"
                )
            )
        )
        probes.append(
            {
                "probe_id": probe_id,
                "method": "GET",
                "path": path,
                "canonical_endpoint": f"GET {path}",
                "http_status": static_status,
                "content_type": content_type,
                "bytes": byte_count,
                "error": static_error,
                "semantic_ok": static_ok,
            }
        )
        overall_ok &= static_ok

    status, catalog = probe("catalog", "GET", "/v32-capability-catalog.json")
    catalog_ok, catalog_error = _catalog_semantics(
        catalog,
        expected_scope_sha256=(
            None if args.allow_stale_catalog else expected_scope_sha256
        ),
    )
    if args.allow_stale_catalog and status == 200:
        # The first pass is used only to discover the mounted route set.  A
        # second strict pass is mandatory after the catalog is rebuilt.
        catalog_ok = True
        catalog_error = None
    overall_ok &= status == 200 and catalog_ok
    probes[-1]["semantic_ok"] = catalog_ok
    probes[-1]["semantic_error"] = catalog_error

    status, release = probe("release_state", "GET", "/v3.2-staging/release")
    release_ok = (
        isinstance(release, dict)
        and release.get("environment") == "staging"
        and release.get("production_deployed") is False
        and release.get("release_ready") is False
    )
    overall_ok &= status == 200 and release_ok
    probes[-1]["semantic_ok"] = release_ok

    status, cnv = probe(
        "cnv_genomic",
        "GET",
        "/v3.2-staging/genomic?"
        + urlencode({"modality": "cnv", "limit": 1}),
        canonical_endpoint="GET /v3.2-staging/genomic?modality=cnv",
    )
    cnv_ok, cnv_error = _cnv_semantics(cnv)
    overall_ok &= status == 200 and cnv_ok
    probes[-1]["semantic_ok"] = cnv_ok
    probes[-1]["semantic_error"] = cnv_error

    status, cnv_coverage = probe(
        "cnv_coverage",
        "GET",
        "/v3.2-staging/cnv/coverage",
        canonical_endpoint="GET /v3.2-staging/cnv/coverage",
    )
    cnv_coverage_ok = status == 200 and isinstance(cnv_coverage, dict)
    overall_ok &= cnv_coverage_ok
    probes[-1]["semantic_ok"] = cnv_coverage_ok

    status, sc_capability = probe(
        "single_cell_capability",
        "GET",
        "/v3.2-staging/single-cell/context/capability",
        canonical_endpoint="GET /v3.2-staging/single-cell/context/capability",
    )
    sc_capability_ok = status == 200 and isinstance(sc_capability, dict)
    if sc_capability_ok:
        sc_capability_ok &= (
            sc_capability.get("formal_eligible_cancer_count") == 23
            and sc_capability.get("typed_unavailable_cancer_count") == 10
        )
    overall_ok &= sc_capability_ok
    probes[-1]["semantic_ok"] = sc_capability_ok

    status, sc_assoc = probe(
        "single_cell_association",
        "GET",
        "/v3.2-staging/single-cell/exact-pathway-associations?"
        + urlencode({"cancer_id": "BRCA", "limit": 1}),
        canonical_endpoint="GET /v3.2-staging/single-cell/exact-pathway-associations",
    )
    sc_assoc_ok = status == 200 and isinstance(sc_assoc, dict)
    overall_ok &= sc_assoc_ok
    probes[-1]["semantic_ok"] = sc_assoc_ok

    status, sc_expression = probe(
        "single_cell_expression",
        "GET",
        "/v3.2-staging/single-cell/lncrna-expression-summary?"
        + urlencode({"cancer_id": "BRCA", "limit": 1}),
        canonical_endpoint="GET /v3.2-staging/single-cell/lncrna-expression-summary",
    )
    sc_expression_ok = status == 200 and isinstance(sc_expression, dict)
    overall_ok &= sc_expression_ok
    probes[-1]["semantic_ok"] = sc_expression_ok

    status, sc_audit = probe(
        "single_cell_audit",
        "GET",
        "/v3.2-staging/single-cell/audit/capability",
        canonical_endpoint="GET /v3.2-staging/single-cell/audit/capability",
    )
    sc_audit_ok = status == 200 and isinstance(sc_audit, dict)
    overall_ok &= sc_audit_ok
    probes[-1]["semantic_ok"] = sc_audit_ok

    status, sc_coverage = probe(
        "single_cell_coverage",
        "GET",
        "/v3.2-staging/single-cell/audit/coverage?"
        + urlencode({"cancer_id": "BRCA", "limit": 1}),
        canonical_endpoint="GET /v3.2-staging/single-cell/audit/coverage",
    )
    sc_coverage_ok = status == 200 and isinstance(sc_coverage, dict)
    overall_ok &= sc_coverage_ok
    probes[-1]["semantic_ok"] = sc_coverage_ok

    status, sc_unavailable = probe(
        "single_cell_typed_unavailable",
        "GET",
        "/v3.2-staging/single-cell/exact-pathway-associations?"
        + urlencode({"cancer_id": "BLCA", "limit": 1}),
        canonical_endpoint="GET /v3.2-staging/single-cell/exact-pathway-associations",
    )
    sc_unavailable_ok = (
        status == 200 and _single_cell_semantics(sc_unavailable, unavailable=True)[0]
    )
    overall_ok &= sc_unavailable_ok
    probes[-1]["semantic_ok"] = sc_unavailable_ok

    status, mixed = probe(
        "mixed_lncrna_protein_pathway",
        "POST",
        "/v3.2-staging/enrichment/mixed-exact-pathway",
        payload={"members": ["TP53", "BRCA1"], "cancer_id": "BRCA", "top_k": 1},
        canonical_endpoint="POST /v3.2-staging/enrichment/mixed-exact-pathway",
    )
    mixed_ok = status == 200 and isinstance(mixed, dict)
    overall_ok &= mixed_ok
    probes[-1]["semantic_ok"] = mixed_ok

    scope = _scope_template()
    if cnv_ok and cnv_coverage_ok:
        scope["cnv"].update(
            staging_status="QUERYABLE_STAGING",
            verified_endpoints=[
                "GET /v3.2-staging/genomic?modality=cnv",
                "GET /v3.2-staging/cnv/coverage",
            ],
            known_gap=None,
            probe_id="cnv_genomic+cnv_coverage",
        )
    sc_routes = []
    if sc_capability_ok:
        sc_routes.append("GET /v3.2-staging/single-cell/context/capability")
    if sc_assoc_ok or sc_unavailable_ok:
        sc_routes.append(
            "GET /v3.2-staging/single-cell/exact-pathway-associations"
        )
    if sc_expression_ok:
        sc_routes.append(
            "GET /v3.2-staging/single-cell/lncrna-expression-summary"
        )
    if sc_audit_ok:
        sc_routes.append("GET /v3.2-staging/single-cell/audit/capability")
    if sc_coverage_ok:
        sc_routes.append("GET /v3.2-staging/single-cell/audit/coverage")
    if sc_routes:
        scope["single_cell"].update(
            staging_status="PARTIAL_STAGING",
            verified_endpoints=sorted(set(sc_routes)),
            known_gap=(
                "Bounded smoke verified formal-23 context/association/expression/"
                "audit routes only; 23 eligible cancers plus 10 typed-unavailable "
                "remain the accepted scope. Unprobed routes stay unavailable."
            ),
            probe_id="single_cell_formal23_bounded",
        )
    if mixed_ok:
        scope["mixed_lncrna_protein_pathway_query"].update(
            staging_status="QUERYABLE_STAGING",
            verified_endpoints=[
                "POST /v3.2-staging/enrichment/mixed-exact-pathway"
            ],
            known_gap=None,
            probe_id="mixed_lncrna_protein_pathway",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "CANCERLNCATLAS_V32_STAGING_CAPABILITY_SCOPE_V1",
        "environment": "staging",
        "host": "149",
        "status": "PASS" if overall_ok else "FAIL",
        "production_deployed": False,
        "release_ready": False,
        "base_url": args.base_url,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "catalog": str(catalog_path),
        "catalog_sha256": _sha256(catalog_path),
        "probe_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "probe_count": len(probes),
        "probes": probes,
        "capabilities": scope,
        "semantic_summary": {
            "formal_single_cell_scope": "23_PLUS_10_TYPED_UNAVAILABLE",
            "full_33_single_cell_coverage_claimed": False,
            "cnv_signed_directional_overlay": cnv_ok and cnv_coverage_ok,
            "mixed_lncrna_protein_pathway_query": mixed_ok,
        },
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": report["status"], "output": str(output)},
            ensure_ascii=False,
        )
    )
    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
