"""Fail-closed route composition for an isolated CancerLncAtlas candidate.

This module deliberately does not edit or replace the legacy production app.
It composes a separately launched candidate process by putting the audited
V3.2 routes in front of the legacy routes.  Only the explicitly enumerated
legacy compatibility signatures may be replaced; every other legacy route is
left intact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from fastapi import APIRouter


V32_PREFIX = "/v3.2-staging"
COMPATIBILITY_OVERRIDE_SIGNATURES = frozenset(
    {
        ("GET", "/api/site/mutation/status"),
        ("GET", "/api/site/mutation/cancer/{cancer_id}"),
        ("GET", "/api/site/mutation/lncrna/{lncrna}"),
        ("GET", "/v2.7/health"),
        ("GET", "/v2.7/version"),
        ("GET", "/v2.7/mutation/cancer/{cancer_id}"),
        ("GET", "/v2.7/mutation/lncrna/{lncrna_id}"),
        ("POST", "/v2.7/predict/mutation-context"),
    }
)


class CandidateIntegrationError(RuntimeError):
    """Raised when candidate route composition is ambiguous or unsafe."""


@dataclass(frozen=True)
class CandidateIntegrationReport:
    v32_route_count: int
    v32_native_route_count: int
    compatibility_override_count: int
    removed_legacy_route_count: int
    preserved_legacy_route_count: int
    overridden_signatures: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "v32_route_count": self.v32_route_count,
            "v32_native_route_count": self.v32_native_route_count,
            "compatibility_override_count": self.compatibility_override_count,
            "removed_legacy_route_count": self.removed_legacy_route_count,
            "preserved_legacy_route_count": self.preserved_legacy_route_count,
            "overridden_signatures": [
                {"method": method, "path": path}
                for method, path in self.overridden_signatures
            ],
        }


def _route_signatures(route: Any) -> frozenset[tuple[str, str]]:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if not isinstance(path, str) or not methods:
        return frozenset()
    return frozenset((str(method).upper(), path) for method in methods)


def _select_v32_routes(routes: Iterable[Any]) -> tuple[list[Any], set[tuple[str, str]]]:
    selected: list[Any] = []
    signatures: set[tuple[str, str]] = set()
    for route in routes:
        route_signatures = _route_signatures(route)
        if not route_signatures:
            continue
        path = next(iter(route_signatures))[1]
        allowed = path == V32_PREFIX or path.startswith(f"{V32_PREFIX}/")
        allowed = allowed or bool(
            route_signatures & COMPATIBILITY_OVERRIDE_SIGNATURES
        )
        if not allowed:
            continue
        duplicate = signatures & route_signatures
        if duplicate:
            raise CandidateIntegrationError(
                f"Duplicate selected V3.2 route signatures: {sorted(duplicate)!r}"
            )
        selected.append(route)
        signatures.update(route_signatures)
    missing = COMPATIBILITY_OVERRIDE_SIGNATURES - signatures
    if missing:
        raise CandidateIntegrationError(
            f"V3.2 candidate lacks required compatibility routes: {sorted(missing)!r}"
        )
    if not any(path.startswith(V32_PREFIX) for _, path in signatures):
        raise CandidateIntegrationError("V3.2 candidate has no native staging routes")
    return selected, signatures


def install_candidate_routes(
    legacy_app: Any,
    v32_app: Any,
    *,
    candidate_metadata: dict[str, Any],
) -> CandidateIntegrationReport:
    """Prepend validated V3.2 routes and replace only declared legacy conflicts.

    FastAPI resolves equal paths in registration order.  Merely calling
    ``include_router`` after the full legacy app has registered its routes does
    not repair the old mutation endpoints.  This function therefore composes a
    new route ordering for a *candidate-only* process.  It never mutates files
    or a separately running production process.
    """

    if getattr(legacy_app.state, "v32_candidate_integrated", False):
        raise CandidateIntegrationError("Candidate routes were already integrated")
    selected, selected_signatures = _select_v32_routes(v32_app.router.routes)

    preserved: list[Any] = []
    removed: list[Any] = []
    for route in legacy_app.router.routes:
        signatures = _route_signatures(route)
        if signatures & COMPATIBILITY_OVERRIDE_SIGNATURES:
            removed.append(route)
        else:
            preserved.append(route)

    removed_signatures = set().union(
        *(_route_signatures(route) for route in removed)
    ) if removed else set()
    missing_legacy = COMPATIBILITY_OVERRIDE_SIGNATURES - removed_signatures
    if missing_legacy:
        raise CandidateIntegrationError(
            "Authoritative legacy app no longer exposes the expected override "
            f"surface: {sorted(missing_legacy)!r}"
        )

    report = CandidateIntegrationReport(
        v32_route_count=len(selected),
        v32_native_route_count=sum(
            1
            for route in selected
            if str(getattr(route, "path", "")).startswith(V32_PREFIX)
        ),
        compatibility_override_count=sum(
            1
            for route in selected
            if _route_signatures(route) & COMPATIBILITY_OVERRIDE_SIGNATURES
        ),
        removed_legacy_route_count=len(removed),
        preserved_legacy_route_count=len(preserved),
        overridden_signatures=tuple(sorted(COMPATIBILITY_OVERRIDE_SIGNATURES)),
    )

    health_router = APIRouter()

    @health_router.get("/__candidate/health", include_in_schema=False)
    def candidate_health() -> dict[str, Any]:
        return {
            "status": "CANDIDATE_READY",
            "production_deployed": False,
            "candidate_metadata": candidate_metadata,
            "integration": report.as_dict(),
        }

    # Include the health route in an empty temporary router and then move the
    # resulting APIRoute before every legacy route, including the SPA catch-all.
    health_routes = list(health_router.routes)
    legacy_app.router.routes[:] = health_routes + selected + preserved
    legacy_app.state.v32_candidate_integrated = True
    legacy_app.state.v32_candidate_metadata = candidate_metadata
    legacy_app.state.v32_candidate_integration = report.as_dict()
    return report


__all__ = [
    "COMPATIBILITY_OVERRIDE_SIGNATURES",
    "CandidateIntegrationError",
    "CandidateIntegrationReport",
    "V32_PREFIX",
    "install_candidate_routes",
]
