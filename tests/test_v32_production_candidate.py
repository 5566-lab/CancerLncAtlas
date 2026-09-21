from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from cc_hhgt.v32.production_candidate import (
    COMPATIBILITY_OVERRIDE_SIGNATURES,
    CandidateIntegrationError,
    install_candidate_routes,
)


def _legacy_app() -> FastAPI:
    app = FastAPI()

    @app.get("/keep")
    def keep():
        return {"source": "legacy"}

    for index, (method, path) in enumerate(sorted(COMPATIBILITY_OVERRIDE_SIGNATURES)):
        async def legacy(index=index):
            return {"source": "legacy", "index": index}

        app.add_api_route(path, legacy, methods=[method], name=f"legacy_{index}")

    @app.get("/{page_path:path}")
    def catch_all(page_path: str):
        return {"source": "spa", "path": page_path}

    return app


def _v32_app(*, omit: tuple[str, str] | None = None) -> FastAPI:
    app = FastAPI()

    @app.get("/v3.2-staging/health")
    def health():
        return {"source": "v32"}

    for index, (method, path) in enumerate(sorted(COMPATIBILITY_OVERRIDE_SIGNATURES)):
        if (method, path) == omit:
            continue

        async def v32(index=index):
            return {"source": "v32", "index": index}

        app.add_api_route(path, v32, methods=[method], name=f"v32_{index}")
    return app


def test_candidate_preserves_legacy_and_replaces_only_explicit_conflicts():
    legacy = _legacy_app()
    report = install_candidate_routes(
        legacy,
        _v32_app(),
        candidate_metadata={"test": True},
    )
    assert report.compatibility_override_count == len(COMPATIBILITY_OVERRIDE_SIGNATURES)
    assert report.removed_legacy_route_count == len(COMPATIBILITY_OVERRIDE_SIGNATURES)
    client = TestClient(legacy)
    assert client.get("/keep").json() == {"source": "legacy"}
    assert client.get("/v3.2-staging/health").json() == {"source": "v32"}
    assert client.get("/api/site/mutation/status").json()["source"] == "v32"
    candidate = client.get("/__candidate/health")
    assert candidate.status_code == 200
    assert candidate.json()["production_deployed"] is False
    assert client.get("/unrelated-page").json()["source"] == "spa"


def test_candidate_fails_closed_when_a_compatibility_override_is_missing():
    omitted = next(iter(COMPATIBILITY_OVERRIDE_SIGNATURES))
    with pytest.raises(CandidateIntegrationError, match="lacks required"):
        install_candidate_routes(
            _legacy_app(),
            _v32_app(omit=omitted),
            candidate_metadata={},
        )


def test_candidate_cannot_be_integrated_twice():
    legacy = _legacy_app()
    v32 = _v32_app()
    install_candidate_routes(legacy, v32, candidate_metadata={})
    with pytest.raises(CandidateIntegrationError, match="already integrated"):
        install_candidate_routes(legacy, v32, candidate_metadata={})
