from __future__ import annotations

import pandas as pd
import pytest

import cc_hhgt.v32.single_cell_partition_builder as partition
from cc_hhgt.v32.single_cell_input_builder import (
    EXACT_PATHWAY_COUNT,
    EXPECTED_CANCERS,
    compute_exact_pathway_activity,
)
from cc_hhgt.v32.single_cell_training import FORMAL_SINGLE_CELL_CANCERS


def _small_authority() -> tuple[pd.DataFrame, pd.DataFrame]:
    pathways = [f"PW:{index:04d}" for index in range(EXACT_PATHWAY_COUNT)]
    candidates = pd.DataFrame(
        [
            {"cancer_id": "ACC", "lncrna_id": "LNC:ENSG_L1", "pathway_id": pathway}
            for pathway in pathways
        ]
        + [
            {"cancer_id": cancer, "lncrna_id": "LNC:ENSG_L1", "pathway_id": pathways[0]}
            for cancer in sorted(EXPECTED_CANCERS - {"ACC"})
        ]
    )
    membership = pd.DataFrame(
        {"pathway_id": pathways, "gene_id": ["ENSG_P1"] * len(pathways)}
    )
    return membership, candidates


def test_trainer_and_partition_share_exact_33_cancer_authority() -> None:
    assert FORMAL_SINGLE_CELL_CANCERS == EXPECTED_CANCERS
    assert len(FORMAL_SINGLE_CELL_CANCERS) == 33


def test_exact_pathway_availability_has_reasons_and_never_imputes() -> None:
    activity = pd.DataFrame({"pathway_id": ["PW:A", "PW:B"]})
    association = pd.DataFrame({"pathway_id": ["PW:A"]})
    result = partition.build_exact_pathway_availability(
        ["PW:A", "PW:B", "PW:C"], activity, association
    ).set_index("pathway_id")
    assert result.loc["PW:A", "association_available"]
    assert result.loc["PW:B", "association_unavailable_reason"] == (
        "NO_FINITE_DONOR_ASSOCIATION"
    )
    assert result.loc["PW:C", "activity_unavailable_reason"] == (
        "NO_MEMBER_PROTEIN_IN_MATRIX"
    )
    assert result.loc["PW:C", "association_unavailable_reason"] == (
        "NO_MEMBER_PROTEIN_IN_MATRIX"
    )
    assert not result.numeric_imputation_used.any()


def test_sparse_activity_matches_generic_edge_expanded_definition() -> None:
    genes = ["ENSG_P1", "ENSG_P2", "ENSG_P3"]
    groups = pd.DataFrame(
        {
            "donor_id": ["D1", "D2"],
            "cell_type": ["Malignant", "Malignant"],
            "cell_state": ["", ""],
        }
    )
    values = pd.DataFrame(
        [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], index=genes
    )
    rows = []
    for gene in genes:
        for group_index, group in groups.iterrows():
            rows.append(
                {
                    "dataset_id": "SC_TEST",
                    "cancer_id": "ACC",
                    "donor_id": group.donor_id,
                    "cell_type": group.cell_type,
                    "cell_state": group.cell_state,
                    "gene_id": gene,
                    "gene_class": "protein_coding",
                    "expression": values.loc[gene, group_index],
                    "n_cells": 10,
                    "source_tier": "primary_raw_count",
                    "expression_source_tier": "raw_counts",
                    "quality_flags": "NONE",
                    "feature_universe_status": "PASS",
                }
            )
    membership = pd.DataFrame(
        {
            "pathway_id": ["PW:A", "PW:A", "PW:B"],
            "gene_id": ["ENSG_P1", "ENSG_P3", "ENSG_P2"],
            "membership_weight": [1.0, 1.0, 1.0],
        }
    )
    generic = compute_exact_pathway_activity(pd.DataFrame(rows), membership)
    sparse = partition.compute_exact_pathway_activity_from_aggregates(
        aggregate_expression=values.to_numpy(),
        gene_order=genes,
        gene_class={gene: "protein_coding" for gene in genes},
        group_records=groups,
        group_cell_counts=pd.Series([10, 10]).to_numpy(),
        membership=membership,
        dataset_id="SC_TEST",
        cancer_id="ACC",
        source_tier="primary_raw_count",
        expression_source_tier="raw_counts",
        quality_flags="NONE",
        feature_universe_status="PASS",
    )
    keys = ["donor_id", "cell_type", "cell_state", "pathway_id"]
    left = generic.sort_values(keys).reset_index(drop=True)
    right = sparse.sort_values(keys).reset_index(drop=True)
    assert left[keys].equals(right[keys])
    pd.testing.assert_series_equal(left.activity, right.activity, check_names=False)


def test_current_authority_is_hash_bound_then_filtered_to_exact_namespace(
    tmp_path, monkeypatch
) -> None:
    membership, candidates = _small_authority()
    membership_path = tmp_path / "membership.parquet"
    candidate_path = tmp_path / "candidates.parquet"
    membership.to_parquet(membership_path, index=False)
    candidates.to_parquet(candidate_path, index=False)
    monkeypatch.setattr(partition, "CURRENT_V32_CANDIDATE_ROWS", len(candidates))
    monkeypatch.setattr(partition, "CURRENT_V32_EXACT_MEMBERSHIP_EDGES", len(membership))

    def fake_sha(path):
        return (
            partition.CURRENT_V32_EXACT_MEMBERSHIP_SHA256
            if str(path).endswith("membership.parquet")
            else partition.CURRENT_V32_CANDIDATE_SHA256
        )

    monkeypatch.setattr(partition, "artifact_sha256", fake_sha)
    exact, candidate, audit = partition.load_current_v32_authority(
        membership_path, candidate_path
    )
    assert exact.pathway_id.nunique() == EXACT_PATHWAY_COUNT
    assert candidate.cancer_id.nunique() == 33
    assert audit["family_to_exact_broadcast"] is False
    assert audit["membership_source_sha256"] == (
        partition.CURRENT_V32_EXACT_MEMBERSHIP_SHA256
    )
    assert audit["candidate_sha256"] == partition.CURRENT_V32_CANDIDATE_SHA256


def test_current_authority_rejects_any_hash_drift(tmp_path, monkeypatch) -> None:
    membership, candidates = _small_authority()
    membership_path = tmp_path / "membership.parquet"
    candidate_path = tmp_path / "candidates.parquet"
    membership.to_parquet(membership_path, index=False)
    candidates.to_parquet(candidate_path, index=False)
    monkeypatch.setattr(partition, "artifact_sha256", lambda _: "0" * 64)
    with pytest.raises(partition.SingleCellPartitionBuildError, match="not the pinned"):
        partition.load_current_v32_authority(membership_path, candidate_path)


def test_missing_metadata_is_a_persisted_block_not_fake_partition(
    tmp_path, monkeypatch
) -> None:
    cancer_root = tmp_path / "processed" / "sc_tool_input" / "CESC"
    cancer_root.mkdir(parents=True)
    h5_path = cancer_root / "raw_feature_bc_matrix.h5"
    h5_path.write_bytes(b"not-read-when-metadata-is-missing")
    metadata_path = cancer_root / "cell_metadata.parquet"
    membership = pd.DataFrame(
        {"pathway_id": ["PW:A"], "gene_id": ["ENSG_P1"], "membership_weight": [1.0]}
    )
    candidates = pd.DataFrame(
        {"cancer_id": ["CESC"], "lncrna_id": ["LNC:ENSG_L1"], "pathway_id": ["PW:A"]}
    )
    monkeypatch.setattr(
        partition,
        "load_current_v32_authority",
        lambda *_: (membership, candidates, {"status": "PINNED_TEST_AUTHORITY"}),
    )
    output = tmp_path / "new_partition"
    status = partition.build_single_cell_partition(
        cancer_id="CESC",
        h5_path=h5_path,
        metadata_path=metadata_path,
        annotation_parquet_path=tmp_path / "unused_annotation.parquet",
        annotation_provenance_path=tmp_path / "unused_annotation.json",
        membership_path=tmp_path / "unused_membership.parquet",
        candidate_path=tmp_path / "unused_candidates.parquet",
        output_root=output,
        source_tier="raw_counts",
        measurement_scale="raw_counts",
        audited_lnc_feature_count=53,
    )
    assert status["status"] == "BLOCKED_MISSING_CELL_METADATA"
    assert status["formal_eligible"] is False
    assert status["blocking_reason"] == "DONOR_CELLTYPE_METADATA_UNAVAILABLE"
    assert status["feature_universe_status"] == "LIMITED"
    assert status["audited_lncrna_feature_universe_count"] == 53
    assert status["fresh_direct_id_lncrna_feature_count"] is None
    assert status["fresh_minus_audit_feature_count"] is None
    assert status["feature_count_tolerance"] is None
    assert status["lncrna_feature_universe_count_semantics"] == (
        "READ_ONLY_AUDIT_COUNT_METADATA_BLOCKED_H5_NOT_SCANNED"
    )
    assert (output / "PARTITION_STATUS.json").is_file()
    availability = pd.read_parquet(output / "exact_pathway_availability.parquet")
    assert availability.association_unavailable_reason.eq(
        "DONOR_CELLTYPE_METADATA_UNAVAILABLE"
    ).all()
    assert not availability.numeric_imputation_used.any()
    assert not (output / "gene_expression_facts.parquet").exists()


def test_historical_sc_trajectory_path_is_rejected_before_read(tmp_path) -> None:
    h5_root = tmp_path / "sc_trajectory" / "ACC"
    h5_root.mkdir(parents=True)
    h5_path = h5_root / "raw_feature_bc_matrix.h5"
    h5_path.write_bytes(b"forbidden")
    with pytest.raises(partition.SingleCellPartitionBuildError, match="forbidden"):
        partition.build_single_cell_partition(
            cancer_id="ACC",
            h5_path=h5_path,
            metadata_path=tmp_path / "metadata.parquet",
            annotation_parquet_path=tmp_path / "annotation.parquet",
            annotation_provenance_path=tmp_path / "annotation.json",
            membership_path=tmp_path / "membership.parquet",
            candidate_path=tmp_path / "candidate.parquet",
            output_root=tmp_path / "output",
            source_tier="raw_counts",
            measurement_scale="raw_counts",
            audited_lnc_feature_count=1000,
        )
