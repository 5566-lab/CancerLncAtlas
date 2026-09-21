"""Isolated, non-production API for one V3.2 streaming Evidence candidate.

This module is not imported by the production application.  It has no default
listener and never starts or modifies production port 8260.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from cc_hhgt.v32.evidence_streaming_website import (
    MAX_QUERY_LIMIT,
    MAX_QUERY_OFFSET,
    StreamingEvidenceAssetError,
    StreamingEvidenceInputError,
    StreamingEvidenceWebsiteQuery,
)


class EvidenceConfidenceRow(BaseModel):
    cancer_id: str
    lncrna_id: str
    pathway_id: str
    evidence_confidence_probability: float | None
    direction: str | None
    uncertainty: float | None
    availability: bool
    unavailable_reason: str
    model_consumed_event_count: int = Field(
        description="Events actually consumed by the Evidence head after deterministic capping."
    )
    event_count_cap: int = Field(
        description="Maximum events the Evidence head can consume for one exact key; fixed at 64."
    )
    event_count_capped: bool = Field(
        description="True when total_exact_event_count exceeds model_consumed_event_count."
    )
    total_exact_event_count: int = Field(
        description="Exact uncapped COUNT(*) from candidate_exact_events.parquet for this key."
    )
    evidence_fold: int | None
    failure_reason: str
    analysis_version: str
    training_run_id: str
    changes_primary_ranking: bool
    main_ranking_modified: bool


class EvidenceEventRow(BaseModel):
    event_id: str
    source_event_id: str
    cancer_id: str
    lncrna_id: str
    pathway_id: str
    partner_id: str | None
    member_type: str | None
    route_type: str | None
    source_database: str
    source_dataset: str
    source_record_id: str
    pmid: str | None
    experiment_type: str | None
    relation_type: str | None
    direction_raw: str | None
    direction_target: int | None
    confidence_target: float | None
    tissue: str | None
    cell_line: str | None
    species: str | None
    is_experimental: bool
    is_computational: bool
    is_physical: bool
    is_model_prediction: bool
    lineage_id: str | None
    source_kind: str | None
    mapping_route: str | None
    source_row_sha256: str | None
    family_broadcast_used: bool


class EvidenceQueryProvenance(BaseModel):
    analysis_version: str
    training_run_id: str
    binding_sha256: str
    independent_audit_sha256: str


class EvidenceConfidenceResponse(BaseModel):
    module: str
    query_kind: str
    candidate_only: bool
    release_ready: bool
    publication_blocked_by_core_lineage: bool
    prediction_role: str
    changes_primary_ranking: bool
    filters: dict[str, Any]
    limit: int
    offset: int
    returned_rows: int
    rows: list[EvidenceConfidenceRow]
    provenance: EvidenceQueryProvenance


class EvidenceEventsResponse(BaseModel):
    module: str
    query_kind: str
    candidate_only: bool
    release_ready: bool
    publication_blocked_by_core_lineage: bool
    prediction_role: str
    changes_primary_ranking: bool
    filters: dict[str, Any]
    limit: int
    offset: int
    returned_rows: int
    rows: list[EvidenceEventRow]
    provenance: EvidenceQueryProvenance


class EvidenceCapabilityResponse(BaseModel):
    module: str
    analysis_version: str
    status: str
    candidate_only: bool
    release_ready: bool
    publication_allowed_by_independent_audit: bool
    publication_blocked_by_core_lineage: bool
    core_lineage_verdict: str
    prediction_role: str
    changes_primary_ranking: bool
    coverage: dict[str, Any]
    query_contract: dict[str, Any]
    provenance: dict[str, str]


def create_evidence_candidate_app(
    binding_path: str | Path,
    *,
    binding_sha256: str,
) -> FastAPI:
    """Create an isolated hash-pinned app; no server process is started."""

    query = StreamingEvidenceWebsiteQuery(
        binding_path,
        expected_binding_sha256=binding_sha256,
    )
    app = FastAPI(
        title="CancerLncAtlas V3.2 Evidence Candidate",
        version="3.2-candidate",
        description=(
            "Isolated candidate API. candidate_only/release_ready fields are "
            "authoritative; this is not the production port 8260 service."
        ),
    )

    @app.get(
        "/v3.2-candidate/evidence/capability",
        response_model=EvidenceCapabilityResponse,
    )
    def capability() -> dict[str, Any]:
        return query.capability()

    @app.get(
        "/v3.2-candidate/evidence/confidence",
        response_model=EvidenceConfidenceResponse,
    )
    def confidence(
        lncrna_id: str,
        cancer_id: str | None = None,
        pathway_id: str | None = None,
        availability: bool | None = None,
        min_confidence: float | None = Query(default=None, ge=0, le=1),
        limit: int = Query(default=50, ge=1, le=MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        try:
            return query.query_confidence(
                lncrna_id=lncrna_id,
                cancer_id=cancer_id,
                pathway_id=pathway_id,
                availability=availability,
                min_confidence=min_confidence,
                limit=limit,
                offset=offset,
            )
        except StreamingEvidenceInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except StreamingEvidenceAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get(
        "/v3.2-candidate/evidence/events",
        response_model=EvidenceEventsResponse,
    )
    def events(
        cancer_id: str,
        lncrna_id: str,
        pathway_id: str,
        limit: int = Query(default=200, ge=1, le=MAX_QUERY_LIMIT),
        offset: int = Query(default=0, ge=0, le=MAX_QUERY_OFFSET),
    ) -> dict[str, Any]:
        try:
            return query.query_events(
                cancer_id=cancer_id,
                lncrna_id=lncrna_id,
                pathway_id=pathway_id,
                limit=limit,
                offset=offset,
            )
        except StreamingEvidenceInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except StreamingEvidenceAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get(
        "/v3.2-candidate/evidence/download/{artifact_id}",
        response_class=FileResponse,
        responses={
            200: {
                "description": "Hash-pinned candidate-only Parquet artifact.",
                "content": {"application/vnd.apache.parquet": {}},
                "headers": {
                    "X-Artifact-SHA256": {"schema": {"type": "string"}},
                    "X-Candidate-Only": {"schema": {"type": "string"}},
                    "X-Release-Ready": {"schema": {"type": "string"}},
                },
            }
        },
    )
    def download(artifact_id: str) -> FileResponse:
        try:
            declaration = query.download(artifact_id)
        except StreamingEvidenceInputError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StreamingEvidenceAssetError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        response = FileResponse(
            declaration["path"],
            media_type=declaration["media_type"],
            filename=declaration["filename"],
        )
        response.headers["X-Artifact-SHA256"] = declaration["sha256"]
        response.headers["X-Candidate-Only"] = "true"
        response.headers["X-Release-Ready"] = "false"
        return response

    return app


def app_from_environment() -> FastAPI:
    binding = os.getenv("CANCERLNCATLAS_V32_EVIDENCE_CANDIDATE_BINDING")
    digest = os.getenv("CANCERLNCATLAS_V32_EVIDENCE_CANDIDATE_BINDING_SHA256")
    if not binding or not digest:
        raise RuntimeError(
            "Evidence candidate API requires a binding path and expected SHA256"
        )
    return create_evidence_candidate_app(binding, binding_sha256=digest)


__all__ = [
    "EvidenceCapabilityResponse",
    "EvidenceConfidenceResponse",
    "EvidenceConfidenceRow",
    "EvidenceEventRow",
    "EvidenceEventsResponse",
    "app_from_environment",
    "create_evidence_candidate_app",
]
