from __future__ import annotations

import json
from pathlib import Path

from cc_hhgt.v32.clinical_km_query import ClinicalKMReleaseQuery


REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "artifacts" / "v32_clinical_km_fresh_20260826_r1"
BINDING = RELEASE / "CLINICAL_KM_BINDING.json"
BINDING_SHA256 = "c1905f2669dbb670bcad675c7d0ef82503cd15951cf41c8e0645b3a04c0ea6e9"
STATISTICS_SHA256 = "831cf584e6a11a7dfd2a945b8cd293dfcc412f019b65157520cea3c7d9798052"
CURVES_SHA256 = "242a838ac25c54222142683d85bda4a29f25308ba8bdac5e22813a3bdd850d06"


def test_formal_clinical_km_release_is_hash_pinned_and_complete() -> None:
    success = json.loads((RELEASE / "SUCCESS.json").read_text(encoding="utf-8"))
    assert success["binding_sha256"] == BINDING_SHA256
    query = ClinicalKMReleaseQuery(
        BINDING,
        expected_binding_sha256=BINDING_SHA256,
    )
    assert query.counts == {
        "candidate_pairs": 76_734,
        "statistics_rows": 460_404,
        "available_rows": 355_445,
        "unavailable_rows": 104_959,
        "curve_rows": 4_265_340,
        "cancers": 33,
        "lncrnas": 8_541,
        "endpoints": 6,
        "dfs_rows": 76_734,
        "dfs_available_rows": 0,
        "horizons": 6,
    }
    assert query.binding["artifacts"]["statistics"]["sha256"] == STATISTICS_SHA256
    assert query.binding["artifacts"]["curves"]["sha256"] == CURVES_SHA256
    for key in (
        "historical_derived_outputs_used",
        "historical_predictions_used",
        "historical_checkpoints_used",
        "historical_rankings_used",
        "historical_web_tables_used",
        "changes_primary_ranking",
        "release_ready",
        "production_deployed",
    ):
        assert query.binding[key] is False


def test_formal_query_returns_fresh_os_and_typed_unavailable_dfs() -> None:
    query = ClinicalKMReleaseQuery(
        BINDING,
        expected_binding_sha256=BINDING_SHA256,
    )
    os_result = query.query_pair(
        cancer_id="ACC",
        lncrna_id="ENSG00000099869",
        clinical_endpoint="OS",
    )
    assert os_result["returned_statistics_rows"] == 1
    assert os_result["returned_curve_rows"] == 12
    assert os_result["statistics"][0]["availability"] is True
    assert os_result["provenance"]["historical_derived_outputs_used"] is False
    dfs_result = query.query_pair(
        cancer_id="ACC",
        lncrna_id="ENSG00000099869",
        clinical_endpoint="DFS",
    )
    assert dfs_result["returned_statistics_rows"] == 1
    assert dfs_result["returned_curve_rows"] == 0
    assert dfs_result["statistics"][0]["availability"] is False
    assert dfs_result["statistics"][0]["failure_reason"] == "NO_DISTINCT_DFS_SOURCE"
