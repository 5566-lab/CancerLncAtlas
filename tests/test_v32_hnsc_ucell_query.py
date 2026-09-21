from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.hnsc_ucell_query import (
    EXPECTED_CELL_LEVEL_ROWS,
    EXPECTED_NUMERIC_ROWS,
    EXPECTED_TYPED_UNAVAILABLE_ROWS,
    HNSCUCellAssetError,
    HNSCUCellInputError,
    HNSCUCellPilotQuery,
    PINNED_SHA256,
    artifact_sha256,
    build_hnsc_ucell_query_binding,
)


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    REPO
    / "artifacts"
    / "single_cell_cell_level_hnsc_ucell_pilot_20260826_r1_query_snapshot"
)
AUDIT_PATH = (
    REPO
    / "artifacts"
    / "single_cell_cell_level_hnsc_ucell_pilot_20260826_r1_controls"
    / "AUDIT.json"
)


@pytest.fixture(scope="session")
def hnsc_bound_query(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("hnsc-ucell-binding") / "binding"
    built = build_hnsc_ucell_query_binding(
        result_root=RESULT_ROOT,
        independent_audit_path=AUDIT_PATH,
        output_dir=output,
    )
    query = HNSCUCellPilotQuery(
        built["binding_path"],
        expected_binding_sha256=built["binding_sha256"],
    )
    return query, built


def test_binding_is_exact_hnsc_pilot_not_module_or_release(hnsc_bound_query) -> None:
    query, built = hnsc_bound_query
    binding = built["binding"]
    assert binding["scope"] == "HNSC_ONLY_PILOT"
    assert binding["cancer_id"] == "HNSC"
    assert binding["dataset_id"] == "SC_GSE103322_HNSC"
    assert binding["query_axes"] == [
        "pathway_id",
        "cell_id",
        "cell_type_major",
        "patient_id",
    ]
    assert binding["lncrna_query_applicable"] is False
    assert binding["single_cell_module_complete"] is False
    assert binding["release_ready"] is False
    assert binding["production_deployed"] is False
    assert binding["validated_counts"] == {
        "manifest_entries": 98,
        "cells": 5_902,
        "pathways_total": 2_135,
        "pathways_available": 2_120,
        "pathways_typed_unavailable": 15,
        "cell_level_coverage_rows": EXPECTED_CELL_LEVEL_ROWS,
        "cell_level_numeric_rows": EXPECTED_NUMERIC_ROWS,
        "cell_level_typed_unavailable_rows": EXPECTED_TYPED_UNAVAILABLE_ROWS,
        "donor_celltype_groups": 101,
        "donor_celltype_rows": 215_635,
        "pseudotime_numeric_values": 0,
    }
    controls = binding["sources"]["controls"]
    for name, digest in PINNED_SHA256.items():
        if name != "AUDIT.json":
            assert controls[name] == {"relative_path": name, "sha256": digest}
    status = query.capability_status()
    assert status["scope"] == "HNSC_ONLY_PILOT"
    assert status["historical_inputs_used"] is False
    assert status["pseudotime_numeric_values"] == 0
    assert status["single_cell_module_complete"] is False
    assert status["release_ready"] is False


def test_cell_query_pages_deterministically_and_preserves_typed_unavailable(
    hnsc_bound_query,
) -> None:
    query, _ = hnsc_bound_query
    cell_id = pd.read_parquet(
        RESULT_ROOT / "pseudotime_cell_availability.parquet",
        columns=["cell_id"],
    ).iloc[0]["cell_id"]
    first = query.query_cell_scores(cell_id=str(cell_id), limit=7, offset=0)
    second = query.query_cell_scores(cell_id=str(cell_id), limit=7, offset=7)
    assert first["query_level"] == "CELL_PATHWAY"
    assert first["total_matching_rows"] == 2_135
    assert first["returned_rows"] == 7
    assert first["next_offset"] == 7
    first_keys = [row["pathway_id"] for row in first["rows"]]
    second_keys = [row["pathway_id"] for row in second["rows"]]
    assert first_keys == sorted(first_keys)
    assert set(first_keys).isdisjoint(second_keys)

    unavailable = query.query_cell_scores(
        cell_id=str(cell_id),
        availability="TYPED_UNAVAILABLE",
        limit=20,
    )
    assert unavailable["total_matching_rows"] == 15
    assert unavailable["returned_rows"] == 15
    assert all(row["ucell_available"] is False for row in unavailable["rows"])
    assert all(row["ucell_score"] is None for row in unavailable["rows"])
    assert all(row["unavailable_reason"] for row in unavailable["rows"])
    assert not any(row["ucell_score"] in {0, 0.5} for row in unavailable["rows"])

    available = query.query_cell_scores(
        cell_id=str(cell_id), availability="AVAILABLE", limit=3
    )
    assert available["total_matching_rows"] == 2_120
    assert all(row["ucell_score"] is not None for row in available["rows"])
    assert all(row["unavailable_reason"] is None for row in available["rows"])


def test_pathway_availability_reports_exact_14_plus_1_reasons(hnsc_bound_query) -> None:
    query, _ = hnsc_bound_query
    result = query.query_pathway_availability(
        availability="TYPED_UNAVAILABLE", limit=20
    )
    assert result["total_matching_rows"] == 15
    assert result["returned_rows"] == 15
    reasons = pd.Series([row["unavailable_reason"] for row in result["rows"]]).value_counts()
    assert reasons.to_dict() == {
        "FULL_SIGNATURE_SIZE_AT_OR_EXCEEDS_UCELL_MAX_RANK_DOMAIN": 14,
        "NO_MEMBER_PROTEIN_IN_MATRIX": 1,
    }
    assert all(row["complete_signature_length_used_in_denominator"] for row in result["rows"])
    assert all(
        row["matrix_missing_member_is_expression_data_imputation"] is False
        for row in result["rows"]
    )
    assert all(row["smoothing_used"] is False for row in result["rows"])


def test_pathway_and_donor_celltype_filters_are_paginated(hnsc_bound_query) -> None:
    query, _ = hnsc_bound_query
    unavailable_pathway = query.query_pathway_availability(
        availability="TYPED_UNAVAILABLE", limit=1
    )["rows"][0]["pathway_id"]
    result = query.query_donor_celltype_scores(
        pathway_id=unavailable_pathway,
        availability="TYPED_UNAVAILABLE",
        limit=5,
    )
    assert result["query_level"] == "DONOR_CELLTYPE_PATHWAY"
    assert result["total_matching_rows"] == 101
    assert result["returned_rows"] == 5
    assert result["next_offset"] == 5
    assert all(row["pathway_id"] == unavailable_pathway for row in result["rows"])
    assert all(row["ucell_score_mean"] is None for row in result["rows"])
    probe = result["rows"][0]
    exact = query.query_donor_celltype_scores(
        pathway_id=unavailable_pathway,
        patient_id=probe["patient_id"],
        cell_type_major=probe["cell_type_major"],
        availability="TYPED_UNAVAILABLE",
        limit=2,
    )
    assert exact["total_matching_rows"] == 1
    assert exact["rows"][0]["unavailable_reason"] is not None


def test_pseudotime_query_is_typed_unavailable_with_zero_numeric_values(
    hnsc_bound_query,
) -> None:
    query, _ = hnsc_bound_query
    cells = query.query_pseudotime_availability(level="CELL", limit=4)
    pathways = query.query_pseudotime_availability(level="PATHWAY", limit=4)
    assert cells["total_matching_rows"] == 5_902
    assert pathways["total_matching_rows"] == 2_135
    assert cells["pseudotime_numeric_values"] == 0
    assert pathways["pseudotime_numeric_values"] == 0
    assert all(row["pseudotime_available"] is False for row in cells["rows"])
    assert all(row["pseudotime_value"] is None for row in cells["rows"])
    assert all(row["pseudotime_effect"] is None for row in pathways["rows"])
    assert all(
        row["unavailable_reason"]
        == "NO_EXPLICIT_TRAJECTORY_ROOT_OR_ORDERED_SOURCE_STATE"
        for row in [*cells["rows"], *pathways["rows"]]
    )


def test_lncrna_query_is_explicitly_inapplicable(hnsc_bound_query) -> None:
    query, _ = hnsc_bound_query
    with pytest.raises(HNSCUCellInputError, match="not applicable"):
        query.query_lncrna("ENSG00000200001")
    assert query.capability_status()["lncrna_query_applicable"] is False


def test_query_requires_exact_binding_hash(hnsc_bound_query) -> None:
    _, built = hnsc_bound_query
    with pytest.raises(HNSCUCellAssetError, match="expected binding"):
        HNSCUCellPilotQuery(
            built["binding_path"], expected_binding_sha256=None
        )
    with pytest.raises(HNSCUCellAssetError, match="SHA mismatch"):
        HNSCUCellPilotQuery(
            built["binding_path"], expected_binding_sha256="0" * 64
        )


def test_tampered_binding_semantics_and_success_fail_closed(
    hnsc_bound_query, tmp_path: Path
) -> None:
    _, built = hnsc_bound_query
    source_dir = Path(built["binding_path"]).parent

    semantic_dir = tmp_path / "semantic"
    shutil.copytree(source_dir, semantic_dir)
    semantic_path = semantic_dir / "HNSC_UCELL_QUERY_BINDING.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["single_cell_module_complete"] = True
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    with pytest.raises(HNSCUCellAssetError, match="single_cell_module_complete"):
        HNSCUCellPilotQuery(
            semantic_path,
            expected_binding_sha256=artifact_sha256(semantic_path),
        )

    stale_dir = tmp_path / "stale"
    shutil.copytree(source_dir, stale_dir)
    stale_path = stale_dir / "HNSC_UCELL_QUERY_BINDING.json"
    stale_success_path = stale_dir / "SUCCESS.json"
    stale_success = json.loads(stale_success_path.read_text(encoding="utf-8"))
    stale_success["binding_sha256"] = "f" * 64
    stale_success_path.write_text(json.dumps(stale_success), encoding="utf-8")
    with pytest.raises(HNSCUCellAssetError, match="binding_sha256"):
        HNSCUCellPilotQuery(
            stale_path,
            expected_binding_sha256=artifact_sha256(stale_path),
        )


def test_source_control_hash_drift_fails_before_query(
    hnsc_bound_query, tmp_path: Path
) -> None:
    _, built = hnsc_bound_query
    binding_dir = tmp_path / "binding"
    shutil.copytree(Path(built["binding_path"]).parent, binding_dir)
    tampered_root = tmp_path / "tampered-root"
    tampered_root.mkdir()
    shutil.copy2(RESULT_ROOT / "SUCCESS.json", tampered_root / "SUCCESS.json")
    with (tampered_root / "SUCCESS.json").open("ab") as handle:
        handle.write(b"\n")

    binding_path = binding_dir / "HNSC_UCELL_QUERY_BINDING.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["sources"]["result_root"] = str(tampered_root)
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    binding_sha = artifact_sha256(binding_path)
    success_path = binding_dir / "SUCCESS.json"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    success["binding_sha256"] = binding_sha
    success_path.write_text(json.dumps(success), encoding="utf-8")
    with pytest.raises(HNSCUCellAssetError, match="Pinned SUCCESS.json SHA drift"):
        HNSCUCellPilotQuery(
            binding_path,
            expected_binding_sha256=binding_sha,
        )


def test_builder_refuses_overwrite_and_inputs_are_bounded(
    hnsc_bound_query, tmp_path: Path
) -> None:
    query, built = hnsc_bound_query
    with pytest.raises(HNSCUCellAssetError, match="overwrite"):
        build_hnsc_ucell_query_binding(
            result_root=RESULT_ROOT,
            independent_audit_path=AUDIT_PATH,
            output_dir=Path(built["binding_path"]).parent,
        )
    with pytest.raises(HNSCUCellInputError, match="limit"):
        query.query_cell_scores(limit=501)
    with pytest.raises(HNSCUCellInputError, match="offset"):
        query.query_cell_scores(offset=EXPECTED_CELL_LEVEL_ROWS + 1)
    with pytest.raises(HNSCUCellInputError, match="availability"):
        query.query_cell_scores(availability="MISSING_MEANS_ZERO")
    with pytest.raises(HNSCUCellInputError, match="CELL or PATHWAY"):
        query.query_pseudotime_availability(level="DONOR")
    with pytest.raises(HNSCUCellInputError, match="invalid for pathway"):
        query.query_pseudotime_availability(level="PATHWAY", cell_id="cell")
    with pytest.raises(HNSCUCellInputError, match="invalid for cell"):
        query.query_pseudotime_availability(level="CELL", pathway_id="pathway")
