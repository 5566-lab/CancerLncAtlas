from __future__ import annotations

from pathlib import Path

import pytest

from cc_hhgt.v32.continuous_activity_query import (
    ContinuousActivityQueryAssetError,
    ContinuousActivityQueryInputError,
    ContinuousActivityReleaseQuery,
)


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts" / "v32_continuous_pathway_activity_fresh_20260826_r1"
BINDING_SHA256 = "42ed326b81ae23de2cf1ea7423013d0a6511dbce1b72e4ebd1359ef8d3c2d143"


@pytest.fixture(scope="module")
def release() -> ContinuousActivityReleaseQuery:
    if not (RELEASE / "OUTPUT_BINDING.json").is_file():
        pytest.skip("Formal continuous-activity release is not materialized")
    return ContinuousActivityReleaseQuery(
        RELEASE / "OUTPUT_BINDING.json",
        expected_binding_sha256=BINDING_SHA256,
    )


def test_formal_release_is_hash_pinned_and_queryable(
    release: ContinuousActivityReleaseQuery,
) -> None:
    capability = release.capability_status()
    assert capability["status"] == "READY"
    assert capability["performance_outcome"] == "INCREMENT"
    assert capability["cancers"] == 33
    assert capability["folds"] == 5
    assert capability["oof_rows"] == 21_303_825
    assert capability["primary_exact_pathway_ranking_changed"] is False

    oof = release.query_oof(
        cancer_id="ACC", pathway_id="DOROTHEA:AHR", limit=2
    )
    metrics = release.query_metrics(
        cancer_id="ACC", pathway_id="DOROTHEA:AHR", limit=2
    )
    attributions = release.query_attributions(
        cancer_id="ACC", pathway_id="DOROTHEA:AHR", limit=2
    )
    assert oof["returned_rows"] == 2
    assert metrics["returned_rows"] == 2
    assert attributions["returned_rows"] == 2
    assert all(row["split"] == "outer_test" for row in oof["results"])
    assert all(row["outer_test_used_for_tuning"] is False for row in metrics["results"])
    assert all(row["lncrna_id"].startswith("LNC:") for row in attributions["results"])


def test_query_rejects_unpinned_or_invalid_inputs(
    release: ContinuousActivityReleaseQuery,
) -> None:
    with pytest.raises(ContinuousActivityQueryAssetError):
        ContinuousActivityReleaseQuery(
            RELEASE / "OUTPUT_BINDING.json", expected_binding_sha256=None
        )
    with pytest.raises(ContinuousActivityQueryInputError):
        release.query_oof(cancer_id="bad cancer", limit=10)
    with pytest.raises(ContinuousActivityQueryInputError):
        release.query_metrics(cancer_id="ACC", patient_fold_id=5)
    with pytest.raises(ContinuousActivityQueryInputError):
        release.query_attributions(cancer_id="ACC", limit=0)
