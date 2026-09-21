from __future__ import annotations

import json
from pathlib import Path

import pytest

from cc_hhgt.v32.external_validation_query import ExternalValidationReleaseQuery


REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "artifacts" / "v32_external_validation_fresh_20260826_r1"
BINDING = RELEASE / "EXTERNAL_VALIDATION_BINDING.json"
BINDING_SHA256 = "62d617fbe2fcf3d6a8afa33e969587d885498eba1d10a96ed1f9821331937713"
METRICS_SHA256 = "8c4ee059481c20b1367639d5ff2fe00c704b5e6d1f4b06badf0f4a47de027e53"
DETAILS_SHA256 = "fa08527b98764244d11e04cf32f94ba5920b56f728ce25cdc7ca7d74f0b6da6d"
OVERLAP_SHA256 = "88a3ce37f7494da1e3769c3cca87cb5b2556b535803bab5d389620abcaeeea07"
RANKS_SHA256 = "de467987a3b53337173e9907ad609c827db154306f12c6efa4595b26fd758aef"


@pytest.fixture(scope="module")
def formal_query() -> ExternalValidationReleaseQuery:
    return ExternalValidationReleaseQuery(
        BINDING,
        expected_binding_sha256=BINDING_SHA256,
    )


def test_formal_external_validation_is_hash_pinned_and_fresh(
    formal_query: ExternalValidationReleaseQuery,
) -> None:
    success = json.loads((RELEASE / "SUCCESS.json").read_text(encoding="utf-8"))
    assert success["binding_sha256"] == BINDING_SHA256
    assert formal_query.counts == {
        "task_records": 645_602,
        "training_union_distinct_pmids": 8_437,
        "external_overlap_records": 29_569,
        "primary_eligible_records": 15_726,
        "primary_unique_positives": 7_193,
        "primary_rank_available": 4_327,
        "secondary_unique_positives": 56_480,
        "secondary_rank_available": 14_554,
        "detail_rows": 63_673,
        "metric_rows": 496,
        "available_metric_rows": 476,
        "overlap_audit_rows": 31_408,
        "fresh_rank_rows": 76_734,
        "source_summary_rows": 4,
        "rank_cutoffs": 4,
    }
    assert formal_query.binding["artifacts"]["metrics"]["sha256"] == METRICS_SHA256
    assert formal_query.binding["artifacts"]["details"]["sha256"] == DETAILS_SHA256
    assert (
        formal_query.binding["artifacts"]["overlap_audit"]["sha256"]
        == OVERLAP_SHA256
    )
    assert formal_query.binding["artifacts"]["fresh_ranks"]["sha256"] == RANKS_SHA256
    assert formal_query.binding["fresh_rank_recovery_calculation"] is True
    assert formal_query.binding["pmid_overlap_recomputed"] is True
    assert formal_query.binding["current_v32_predictions_used_as_only_rank_source"] is True
    for key in (
        "historical_metrics_used",
        "historical_details_used",
        "historical_ranks_used",
        "historical_predictions_used",
        "changes_primary_ranking",
        "release_ready",
        "production_deployed",
    ):
        assert formal_query.binding[key] is False


def test_formal_query_returns_recovery_and_typed_unavailable_cohort(
    formal_query: ExternalValidationReleaseQuery,
) -> None:
    detail = formal_query.query_lncrna(
        lncrna_id="ENSG00000215417.1",
        cancer_id="CESC",
        validation_role="primary_known_positive",
        source_database="GSE85011",
    )
    assert detail["returned_rows"] == 1
    assert detail["rows"][0]["availability"] is True
    assert detail["rows"][0]["rank"] == 1_407
    assert detail["rows"][0]["independent_pmid_count"] == 1
    unavailable = formal_query.query_metrics(
        validation_role="primary_known_positive",
        source_database="LncRNADisease",
        cancer_id="BRCA",
        k=100,
    )
    assert unavailable["returned_rows"] == 1
    assert unavailable["rows"][0]["availability"] is False
    assert (
        unavailable["rows"][0]["failure_reason"]
        == "NO_INDEPENDENT_POSITIVE_AFTER_PMID_EXCLUSION"
    )
    assert unavailable["rows"][0]["hits_at_k"] is None
    assert unavailable["provenance"]["historical_metrics_used"] is False
