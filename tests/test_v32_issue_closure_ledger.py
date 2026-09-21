from __future__ import annotations

import json
from pathlib import Path

import pytest

from cc_hhgt.v32.issue_closure_ledger import (
    EXPECTED_TRUTH_IDS,
    SYNTHETIC_IDS,
    build_ledger,
    load_hash_bound,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _build() -> dict:
    return build_ledger(
        source_records={},
        truth=_load(
            "artifacts/data_processing_truth_matrix_20260829_r2_expanded/TRUTH_MATRIX.json"
        ),
        runtime=_load(
            "artifacts/v32_runtime_acceptance_8261_fullapp_4c3a12b3_20260829_r4_lowpressure/RUNTIME_ACCEPTANCE.json"
        ),
        readiness=_load(
            "artifacts/v32_web_relationship_selection_readiness_20260829_r2/WEB_RELATIONSHIP_INPUT_READINESS.json"
        ),
        handoff=_load(
            "artifacts/v32_single_cell_r11_rescued_contract_20260903_r1_server_manifest/TRAINING_HANDOFF.json"
        ),
        run_status=_load(
            "artifacts/v32_single_cell_r11_rescued_contract_20260903_r1_server_manifest/RUN_STATUS.json"
        ),
        supersession=_load(
            "artifacts/v32_single_cell_r11_rescued_contract_20260903_r1_server_manifest/SUCCESS.json"
        ),
        catalog=_load("website/frontend/v32-capability-catalog.json"),
    )


def test_ledger_covers_exact_truth_matrix_plus_two_new_issues() -> None:
    ledger = _build()
    ids = [row["issue_id"] for row in ledger["items"]]
    assert ids == [*EXPECTED_TRUTH_IDS, *SYNTHETIC_IDS]
    assert ledger["summary"]["item_count"] == 39
    assert ledger["summary"]["closure_status_counts"] == {
        "CLOSED": 6,
        "PARTIAL": 20,
        "BLOCKED": 13,
    }
    assert ledger["summary"]["release_complete"] is False
    assert ledger["immutability"]["production_port_8260_touched"] is False


def test_nonclosed_items_name_missing_evidence_and_http_never_implies_science() -> None:
    ledger = _build()
    for row in ledger["items"]:
        if row["closure_status"] == "CLOSED":
            assert row["missing_evidence"] == []
        else:
            assert row["missing_evidence"]
    assert "never promoted" in ledger["status_contract"]["http_200_semantics"]
    downloads = next(
        row for row in ledger["items"] if row["issue_id"] == "HTML_DOWNLOAD_LIST_KEYS_404"
    )
    assert downloads["closure_status"] == "PARTIAL"
    assert "lacks per-file V3.2 scientific attestation" in downloads["scientific_conclusion"]


def test_r11_supersedes_hnsc_and_separates_all_from_formal_remaining() -> None:
    ledger = _build()
    facts = ledger["current_facts"]
    assert facts["single_cell_current_fresh_derived_outputs"] == 0
    assert facts["single_cell_formal_eligible"] == 23
    assert facts["single_cell_remaining_all_cancers"] == 33
    assert facts["single_cell_remaining_formal_cancers"] == 23
    assert facts["old_hnsc_one_cancer_result"] == (
        "SUPERSEDED_DIAGNOSTIC_NOT_CURRENT_RELEASE"
    )
    catalog_issue = next(
        row
        for row in ledger["items"]
        if row["issue_id"] == "STAGING_CATALOG_STATIC_UCELL_17_OF_17"
    )
    assert catalog_issue["closure_status"] == "PARTIAL"
    assert "0/23" in catalog_issue["scientific_conclusion"]


def test_hash_bound_loader_fails_closed_on_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 drift"):
        load_hash_bound(source, "0" * 64, "fixture")
