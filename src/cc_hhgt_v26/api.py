"""FastAPI compatibility application for the historical online query API.

The production-site module imports ``app`` and the lazy ``engine`` factory
from here.  Restoring this small adapter preserves the historical routes
without loading a model or prediction table during application import.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .query_service import (
    MAX_MEMBERS,
    MAX_NETWORK_NODES,
    MAX_TOP_K,
    QueryEngine,
)


class SetRequest(BaseModel):
    members: list[str] = Field(min_length=1, max_length=MAX_MEMBERS)
    cancer_id: str | None = None
    top_k: int = Field(default=20, ge=1, le=MAX_TOP_K)

    @field_validator("members")
    @classmethod
    def nonempty_members(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("members cannot contain blank values")
        return values


class MixedSetRequest(SetRequest):
    network_nodes: int = Field(default=100, ge=1, le=MAX_NETWORK_NODES)


@lru_cache(maxsize=1)
def engine() -> QueryEngine:
    """Load the legacy query engine only when an old endpoint needs it."""

    return QueryEngine()


app = FastAPI(
    title="CC-HHGT legacy Query API",
    version="2.9-compat",
    description=(
        "Historical online-query compatibility surface retained alongside "
        "the independently hash-bound V3.2 application."
    ),
)


@app.get("/v2.6/health")
def health() -> dict[str, str]:
    try:
        engine()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "healthy", "version": "2.9-compat"}


@app.get("/v2.6/version")
def version() -> dict[str, object]:
    return {
        "version": "2.9-compat",
        "model_fallback": False,
        "online_full_graph_loaded": False,
        "historical_surface_retained": True,
    }


@app.post("/v2.6/predict/mixed-gene-set")
def mixed_gene_set(request: MixedSetRequest) -> dict[str, object]:
    return engine().mixed_gene_set(
        request.members,
        request.cancer_id,
        request.top_k,
        request.network_nodes,
    )


@app.post("/v2.6/predict/custom-gene-set-lncrna")
def custom_gene_set(request: SetRequest) -> dict[str, object]:
    return engine().custom_gene_set(
        request.members, request.cancer_id, request.top_k
    )


@app.post("/v2.6/predict/protein-set-lncrna")
def protein_set(request: SetRequest) -> dict[str, object]:
    return engine().protein_set(
        request.members, request.cancer_id, request.top_k
    )
