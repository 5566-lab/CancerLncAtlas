#!/usr/bin/env python3
"""Acceptance audit for an isolated V3.2 streaming Evidence website candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from cc_hhgt.v32.evidence_streaming_website import (  # noqa: E402
    EVENT_COUNT_CAP,
    NO_EVENT_REASON,
    StreamingEvidenceWebsiteQuery,
)
from cc_hhgt.v32.release_registry import artifact_sha256  # noqa: E402
from website.backend.v32_evidence_candidate_api import (  # noqa: E402
    create_evidence_candidate_app,
)


MAX_INTERACTIVE_QUERY_SECONDS = 5.0


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _resolve_openapi_schema(document: dict, schema: dict) -> dict:
    reference = schema.get("$ref")
    if not reference:
        return schema
    prefix = "#/components/schemas/"
    _require(reference.startswith(prefix), f"unexpected OpenAPI reference: {reference}")
    return document["components"]["schemas"][reference.removeprefix(prefix)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"website acceptance refuses output reuse: {output}")

    query = StreamingEvidenceWebsiteQuery(
        args.binding,
        expected_binding_sha256=args.binding_sha256,
    )
    predictions = f"read_parquet({_sql_path(query.paths['evidence_predictions'])})"
    con = duckdb.connect(":memory:")
    try:
        available = con.execute(
            f"""
            SELECT cancer_id, lncrna_id, pathway_id
            FROM {predictions}
            WHERE availability AND event_count < 64
            ORDER BY cancer_id, lncrna_id, pathway_id
            LIMIT 1
            """
        ).fetchone()
        unavailable = con.execute(
            f"""
            SELECT cancer_id, lncrna_id, pathway_id
            FROM {predictions}
            WHERE NOT availability AND failure_reason=?
            ORDER BY cancer_id, lncrna_id, pathway_id
            LIMIT 1
            """,
            [NO_EVENT_REASON],
        ).fetchone()
        capped = con.execute(
            f"""
            SELECT cancer_id, lncrna_id, pathway_id, total_exact_event_count
            FROM read_parquet({_sql_path(query.paths['exact_event_counts'])})
            WHERE total_exact_event_count > 64
            ORDER BY total_exact_event_count DESC, cancer_id, lncrna_id, pathway_id
            LIMIT 1
            """
        ).fetchone()
    finally:
        con.close()
    _require(available is not None, "no available Evidence probe exists")
    _require(unavailable is not None, "no typed-unavailable Evidence probe exists")
    _require(capped is not None, "no >64-event Evidence probe exists")

    started = time.perf_counter()
    available_bare = query.query_confidence(
        cancer_id=available[0], lncrna_id=available[1], pathway_id=available[2]
    )
    available_query_seconds = time.perf_counter() - started
    available_prefixed = query.query_confidence(
        cancer_id=available[0],
        lncrna_id="LNC:" + str(available[1]).removeprefix("LNC:"),
        pathway_id=available[2],
    )
    _require(available_bare["returned_rows"] == 1, "available exact probe did not resolve")
    _require(
        available_bare["rows"] == available_prefixed["rows"],
        "bare ENSG and LNC:ENSG aliases are not equivalent",
    )
    available_row = available_bare["rows"][0]
    _require(available_row["availability"] is True, "available flag drift")
    _require(
        available_row["evidence_confidence_probability"] is not None,
        "available probability is null",
    )
    _require(
        0 < available_row["model_consumed_event_count"] <= EVENT_COUNT_CAP
        and available_row["event_count_cap"] == EVENT_COUNT_CAP
        and available_row["total_exact_event_count"]
        >= available_row["model_consumed_event_count"]
        and available_row["event_count_capped"]
        is (
            available_row["total_exact_event_count"]
            > available_row["model_consumed_event_count"]
        ),
        "available model-consumed/total event-count semantics drift",
    )
    _require("event_count" not in available_row, "ambiguous event_count field is exposed")

    started = time.perf_counter()
    unavailable_result = query.query_confidence(
        cancer_id=unavailable[0],
        lncrna_id=unavailable[1],
        pathway_id=unavailable[2],
    )
    unavailable_query_seconds = time.perf_counter() - started
    _require(unavailable_result["returned_rows"] == 1, "unavailable probe did not resolve")
    unavailable_row = unavailable_result["rows"][0]
    _require(unavailable_row["availability"] is False, "unavailable flag drift")
    _require(
        unavailable_row["evidence_confidence_probability"] is None
        and unavailable_row["uncertainty"] is None
        and unavailable_row["direction"] is None,
        "typed-unavailable row emitted numeric/directional values",
    )
    _require(
        unavailable_row["failure_reason"] == NO_EVENT_REASON,
        "typed-unavailable reason drift",
    )
    _require(
        unavailable_row["model_consumed_event_count"] == 0
        and unavailable_row["event_count_cap"] == EVENT_COUNT_CAP
        and unavailable_row["event_count_capped"] is False
        and unavailable_row["total_exact_event_count"] == 0,
        "typed-unavailable event-count semantics drift",
    )
    empty_events = query.query_events(
        cancer_id=unavailable[0],
        lncrna_id=unavailable[1],
        pathway_id=unavailable[2],
    )
    _require(empty_events["rows"] == [], "no-event exact probe returned event rows")

    started = time.perf_counter()
    capped_result = query.query_confidence(
        cancer_id=capped[0], lncrna_id=capped[1], pathway_id=capped[2]
    )
    capped_query_seconds = time.perf_counter() - started
    _require(capped_result["returned_rows"] == 1, ">64-event exact probe did not resolve")
    capped_row = capped_result["rows"][0]
    _require(
        capped_row["model_consumed_event_count"] == EVENT_COUNT_CAP
        and capped_row["event_count_cap"] == EVENT_COUNT_CAP
        and capped_row["event_count_capped"] is True
        and capped_row["total_exact_event_count"] == int(capped[3]),
        ">64-event count semantics drift",
    )
    query_timings = {
        "available_uncapped_seconds": available_query_seconds,
        "typed_unavailable_seconds": unavailable_query_seconds,
        "available_capped_gt64_seconds": capped_query_seconds,
    }
    started = time.perf_counter()
    source_rows = query.query_events(
        cancer_id=available[0], lncrna_id=available[1], pathway_id=available[2]
    )
    query_timings["available_event_trace_seconds"] = time.perf_counter() - started
    latency_pass = all(
        value <= MAX_INTERACTIVE_QUERY_SECONDS for value in query_timings.values()
    )
    _require(source_rows["returned_rows"] > 0, "available probe has no source events")
    _require(
        all(
            row.get("source_database")
            and row.get("direction_raw") is not None
            and row.get("family_broadcast_used") is False
            and row.get("is_model_prediction") is False
            for row in source_rows["rows"]
        ),
        "event source/direction/family-broadcast contract drift",
    )

    app = create_evidence_candidate_app(
        args.binding,
        binding_sha256=args.binding_sha256,
    )
    client = TestClient(app)
    capability_http = client.get("/v3.2-candidate/evidence/capability")
    confidence_http = client.get(
        "/v3.2-candidate/evidence/confidence",
        params={
            "cancer_id": unavailable[0],
            "lncrna_id": unavailable[1],
            "pathway_id": unavailable[2],
        },
    )
    events_http = client.get(
        "/v3.2-candidate/evidence/events",
        params={
            "cancer_id": available[0],
            "lncrna_id": available[1],
            "pathway_id": available[2],
            "limit": 2,
        },
    )
    _require(
        capability_http.status_code == confidence_http.status_code == events_http.status_code == 200,
        "candidate HTTP endpoints did not all return 200",
    )
    openapi = client.get("/openapi.json")
    _require(openapi.status_code == 200, "candidate OpenAPI schema is unavailable")
    openapi_document = openapi.json()
    api_paths = set(openapi_document.get("paths", {}))
    expected_api_paths = {
        "/v3.2-candidate/evidence/capability",
        "/v3.2-candidate/evidence/confidence",
        "/v3.2-candidate/evidence/events",
        "/v3.2-candidate/evidence/download/{artifact_id}",
    }
    _require(expected_api_paths <= api_paths, "candidate API/download schema is incomplete")
    confidence_schema = _resolve_openapi_schema(
        openapi_document,
        openapi_document["paths"]["/v3.2-candidate/evidence/confidence"]["get"]
        ["responses"]["200"]["content"]["application/json"]["schema"],
    )
    confidence_row_schema = _resolve_openapi_schema(
        openapi_document,
        confidence_schema["properties"]["rows"]["items"],
    )
    confidence_fields = set(confidence_row_schema.get("properties", {}))
    count_fields = {
        "model_consumed_event_count",
        "event_count_cap",
        "event_count_capped",
        "total_exact_event_count",
    }
    _require(
        count_fields <= confidence_fields and "event_count" not in confidence_fields,
        "candidate confidence OpenAPI count schema is ambiguous",
    )
    downloads = {
        role: query.download(role)
        for role in ("evidence_predictions", "candidate_exact_events")
    }
    _require(
        all(value["release_ready"] is False for value in downloads.values()),
        "candidate download claimed release readiness",
    )

    report = {
        "format": "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_WEBSITE_ACCEPTANCE_V1",
        "status": (
            "BLOCKED_CANDIDATE_QUERY_LATENCY"
            if not latency_pass
            else (
                "BLOCKED_ISOLATED_QUERY_CANDIDATE_NOT_PUBLISHABLE"
                if str(query.binding["status"]).startswith("BLOCKED_")
                else "PASS_ISOLATED_QUERY_CANDIDATE_AUDIT_ELIGIBLE"
            )
        ),
        "candidate_only": True,
        "release_ready": False,
        "publication_allowed_by_independent_audit": query.binding[
            "publication_allowed_by_independent_audit"
        ],
        "publication_blocked_by_core_lineage": query.binding[
            "publication_blocked_by_core_lineage"
        ],
        "core_lineage_verdict": query.binding["core_lineage_verdict"],
        "production_deployed": False,
        "port_8260_touched": False,
        "binding": {
            "path": str(query.binding_path),
            "sha256": query.binding_sha256,
        },
        "checks": {
            "bare_and_prefixed_lnc_alias_equivalence": True,
            "exact_lncrna_pathway_cancer_query": True,
            "available_probability_direction_uncertainty": True,
            "model_consumed_capped_and_total_exact_event_counts": True,
            "typed_unavailable_null_never_zero": True,
            "event_sources_and_direction": True,
            "family_broadcast_zero": True,
            "no_event_exact_empty_event_fallback": True,
            "http_capability_confidence_events_200": True,
            "download_and_explicit_openapi_schema": True,
            "interactive_query_latency": latency_pass,
        },
        "query_latency": {
            "threshold_seconds": MAX_INTERACTIVE_QUERY_SECONDS,
            "passed": latency_pass,
            **query_timings,
        },
        "probes": {
            "available": available_bare,
            "typed_unavailable": unavailable_result,
            "available_capped_gt64": capped_result,
            "available_event_rows": source_rows["rows"][:2],
        },
        "downloads": downloads,
        "api_paths": sorted(expected_api_paths),
    }
    output.mkdir(parents=True)
    report_path = output / "WEBSITE_ACCEPTANCE.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "report_sha256": artifact_sha256(report_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if latency_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
