from __future__ import annotations

import json
from pathlib import Path

import pytest

from cc_hhgt.v32.single_cell_formal_context_query import (
    EXPECTED_COUNTS,
    FORMAL_CANCERS,
    SingleCellFormalContextAssetError,
    SingleCellFormalContextInputError,
    SingleCellFormalContextQuery,
    artifact_sha256,
    cell_type_compartment,
)


ROOT = Path(__file__).resolve().parents[1]
BINDING = (
    ROOT
    / "artifacts"
    / "v32_single_cell_formal_context_20260826_r2_celltype_gap_explicit"
    / "SINGLE_CELL_FORMAL_CONTEXT_BINDING.json"
)
AUDIT_BINDING = (
    ROOT
    / "artifacts"
    / "v32_single_cell_formal_context_20260826_r2_celltype_gap_explicit_independent_audit"
    / "INDEPENDENT_AUDIT_BINDING.json"
)
EXPANSION_PLAN = (
    ROOT
    / "artifacts"
    / "v32_single_cell_ucell_17c_expansion_plan_20260826_r1"
    / "UCELL_17C_EXPANSION_PLAN.json"
)


@pytest.fixture(scope="module")
def query() -> SingleCellFormalContextQuery:
    return SingleCellFormalContextQuery(
        BINDING,
        expected_binding_sha256=artifact_sha256(BINDING),
    )


def test_capability_reports_exact_formal_coverage_and_honest_gaps(
    query: SingleCellFormalContextQuery,
) -> None:
    result = query.capability_status()
    assert result["formal_context_cancer_count"] == 17
    assert set(result["formal_context_cancers"]) == set(FORMAL_CANCERS)
    assert result["raw_h5_cancer_count"] == 33
    for role, expected in EXPECTED_COUNTS.items():
        for key, value in expected.items():
            assert result["metrics"][role][key] == value
    assert result["cell_level_ucell"]["status"] == "PARTIAL_1_OF_17_FORMAL_CANCERS"
    assert result["cell_level_ucell"]["covered_cancers"] == ["HNSC"]
    assert result["cell_level_ucell"]["numeric_rows_hnsc"] == 12_512_240
    assert result["single_cell_currently_changes_secondary_score"] is False
    assert result["changes_exact_primary_score"] is False
    assert result["release_ready"] is False
    assert result["production_deployed"] is False
    assert (
        query.binding["model_gaps"][
            "celltype_specific_pathway_activity_consumed_by_current_private_head"
        ]
        is False
    )
    assert (
        query.binding["model_gaps"][
            "celltype_specific_pathway_feature_requires_retraining"
        ]
        is True
    )

    independent = json.loads(AUDIT_BINDING.read_text(encoding="utf-8"))
    assert independent["status"] == "PASS"
    assert independent["source_binding_sha256"] == artifact_sha256(BINDING)
    assert independent["check_count"] == 25
    assert independent["failure_count"] == 0
    assert independent["release_ready"] is False
    assert independent["production_deployed"] is False


def test_context_queries_keep_facts_activity_and_predictions_distinct(
    query: SingleCellFormalContextQuery,
) -> None:
    expression = query.query_lncrna_celltype(
        cancer_id="ACC",
        compartment="IMMUNE",
        availability="AVAILABLE",
        limit=3,
    )
    assert expression["learned_association"] is False
    assert expression["value_semantics"] == "NONPREDICTIVE_EXPRESSION_DETECTION_FACT"
    assert expression["returned_rows"] == 3
    assert all(row["compartment"] == "IMMUNE" for row in expression["rows"])
    assert all(row["lnc_celltype_available"] is True for row in expression["rows"])

    activity = query.query_pathway_activity(
        cancer_id="ACC",
        compartment="MALIGNANT_CANDIDATE",
        availability="AVAILABLE",
        limit=3,
    )
    assert activity["learned_association"] is False
    assert activity["activity_is_not_learned_association"] is True
    assert activity["value_semantics"] == (
        "V3.2_PROTEIN_ONLY_WITHIN_PSEUDOBULK_RANK_MEAN_V1"
    )
    assert activity["returned_rows"] == 3
    assert all(
        row["compartment"] == "MALIGNANT_CANDIDATE" for row in activity["rows"]
    )
    assert all(row["activity_available"] is True for row in activity["rows"])

    prediction = query.query_celltype_predictions(
        cancer_id="ACC",
        compartment="STROMAL",
        availability="AVAILABLE",
        limit=3,
    )
    assert prediction["learned_association"] is True
    assert prediction["returned_rows"] == 3
    assert all(row["compartment"] == "STROMAL" for row in prediction["rows"])
    assert all(row["single_cell_available"] is True for row in prediction["rows"])
    assert all(row["changes_primary_ranking"] is False for row in prediction["rows"])
    assert prediction["single_cell_currently_changes_secondary_score"] is False
    assert prediction["changes_exact_primary_score"] is False


def test_blocked_cancer_is_typed_unavailable_and_labels_are_not_overclaimed(
    query: SingleCellFormalContextQuery,
) -> None:
    blocked = query.query_lncrna_celltype(
        cancer_id="BRCA", availability="UNAVAILABLE", limit=2
    )
    assert blocked["returned_rows"] == 2
    assert all(row["compartment"] == "UNAVAILABLE" for row in blocked["rows"])
    assert all(row["lnc_detection_rate"] is None for row in blocked["rows"])
    assert all(
        row["lnc_celltype_unavailable_reason"] == "DONOR_CELLTYPE_METADATA_UNAVAILABLE"
        for row in blocked["rows"]
    )
    assert cell_type_compartment("Malignant") == "MALIGNANT"
    assert cell_type_compartment("Malignant_candidate") == "MALIGNANT_CANDIDATE"
    assert cell_type_compartment("Epithelial") == "OTHER_UNRESOLVED"
    assert cell_type_compartment("Myeloid") == "IMMUNE"
    assert cell_type_compartment("Fibroblast_stromal") == "STROMAL"


def test_query_fails_closed_on_bad_sha_and_bad_filters(
    query: SingleCellFormalContextQuery,
) -> None:
    with pytest.raises(SingleCellFormalContextAssetError, match="SHA mismatch"):
        SingleCellFormalContextQuery(BINDING, expected_binding_sha256="0" * 64)
    with pytest.raises(SingleCellFormalContextInputError, match="33-cancer"):
        query.query_lncrna_celltype(cancer_id="NOT_A_CANCER")
    with pytest.raises(SingleCellFormalContextInputError, match="compartment"):
        query.query_pathway_activity(cancer_id="ACC", compartment="CANCER")
    with pytest.raises(SingleCellFormalContextInputError, match="availability"):
        query.query_celltype_predictions(cancer_id="ACC", availability="YES")


def test_ucell_expansion_plan_is_preflight_only_and_does_not_claim_completion() -> None:
    plan = json.loads(EXPANSION_PLAN.read_text(encoding="utf-8"))
    assert plan["status"] == "PREFLIGHT_LAUNCH_READY_FULL_COMPUTE_RUNNER_NOT_READY"
    assert set(plan["formal_cancers"]) == set(FORMAL_CANCERS)
    assert plan["formal_cancer_count"] == 17
    assert plan["formal_cells_scanned"] == 1_060_327
    assert plan["current_cell_level_ucell"]["covered_count"] == 1
    assert plan["current_cell_level_ucell"]["missing_formal_count"] == 16
    assert plan["execution_state"] == {
        "preflight_started": False,
        "cell_level_ucell_started": False,
        "retraining_started": False,
        "duplicate_run_started": False,
    }
    assert plan["historical_sc_trajectory_used"] is False
    assert plan["changes_exact_primary_score"] is False
    assert plan["release_ready"] is False
    assert plan["production_deployed"] is False
    assert plan["gates"][1]["status"] == "BLOCKED_BY_HNSC_ONLY_IMPLEMENTATION"
