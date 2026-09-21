from __future__ import annotations

from pathlib import Path
import importlib.util
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cc_hhgt.v32.single_cell_cell_level import build_ucell_pathway_contract

from cc_hhgt.v32.single_cell_r7_streaming import (
    INSUFFICIENT_DONOR_REPLICATION,
    StreamingContractError,
    StreamingGroupAccumulator,
    assert_fresh_r7_source,
    atomic_publish_directory,
    classify_compartment,
    compute_donor_associations,
    estimate_streaming_resources,
    load_checkpoint,
    read_csc_column_block,
    select_smallest_formal_cancer,
    stream_donor_association_chunks,
    write_checkpoint,
)


class _TrackedArray:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)
        self.requests: list[object] = []

    def __getitem__(self, key):
        self.requests.append(key)
        if key == slice(None):
            raise AssertionError("full H5 payload read is forbidden")
        return self.values[key]


def test_csc_reader_only_reads_the_requested_cell_block() -> None:
    # CSC columns: [row0=1,row2=2], [row1=3], [row0=4,row1=5], [row2=6]
    data = _TrackedArray([1, 2, 3, 4, 5, 6])
    indices = _TrackedArray([0, 2, 1, 0, 1, 2])
    indptr = _TrackedArray([0, 2, 3, 5, 6])
    block = read_csc_column_block(
        data=data,
        indices=indices,
        indptr=indptr,
        shape=(3, 4),
        start=1,
        stop=3,
    )
    np.testing.assert_array_equal(block.toarray(), [[0, 4], [3, 5], [0, 0]])
    assert data.requests == [slice(2, 5, None)]
    assert indices.requests == [slice(2, 5, None)]
    assert indptr.requests == [slice(1, 4, None)]


def test_resource_estimate_never_materializes_cells_by_pathways() -> None:
    estimate = estimate_streaming_resources(
        cells=1_919_578,
        features=56_875,
        protein_genes=20_094,
        lncrnas=15_502,
        pathways=2_135,
        donor_celltype_groups=220,
        nnz=344_901_490,
        chunk_cells=64,
    )
    assert estimate["forbidden_cell_by_pathway_elements"] == 1_919_578 * 2_135
    assert estimate["largest_dense_block_elements"] <= 20_094 * 64
    assert estimate["cell_level_pathway_matrix_persisted"] is False
    assert estimate["estimated_peak_ram_bytes"] < 512 * 1024**2
    assert estimate["runtime_baseline_budget_bytes"] == 352 * 1024**2
    assert estimate["estimate_includes_interpreter_runtime_baseline"] is True
    assert estimate["association_evidence_accumulated_in_python"] is False
    assert estimate["association_bh_external_spill_enabled"] is True
    assert estimate["association_workspace_bytes"] > (
        15_502 * 32 * 8
    )
    assert estimate["checkpoint_bytes"] < 256 * 1024**2


def test_compartment_mapping_is_explicit_and_unknown_is_not_forced() -> None:
    assert classify_compartment("Malignant_candidate") == "malignant"
    assert classify_compartment("T_cell") == "immune"
    assert classify_compartment("T") == "immune"
    assert classify_compartment("B") == "immune"
    assert classify_compartment("Lymphocyte") == "immune"
    assert classify_compartment("Plasma") == "immune"
    assert classify_compartment("Hematopoietic_progenitor") == "immune"
    assert classify_compartment("Fibroblast_stromal") == "stromal"
    assert classify_compartment("Endothelial") == "stromal"
    assert classify_compartment("Endothelium") == "stromal"
    assert classify_compartment("Fibroblast") == "stromal"
    assert classify_compartment("Epithelial") == "other_unresolved"
    assert classify_compartment("Neural") == "other_unresolved"
    assert classify_compartment("new_label") == "other_unresolved"


def test_streaming_accumulator_uses_sufficient_statistics_only() -> None:
    accumulator = StreamingGroupAccumulator(
        lncrna_count=2,
        pathway_count=2,
        group_count=2,
    )
    lnc = sparse.csc_matrix(
        np.asarray([[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0]])
    )
    pathways = np.asarray([[0.1, 0.3, 0.5, 0.7], [0.2, 0.4, 0.6, 0.8]])
    accumulator.update(lnc, pathways, np.asarray([0, 0, 1, 1]))
    result = accumulator.finalize()
    np.testing.assert_array_equal(result["cell_counts"], [2, 2])
    np.testing.assert_array_equal(result["lncrna_detect_counts"], [[1, 1], [1, 1]])
    np.testing.assert_allclose(result["lncrna_expression_means"], [[0.5, 1.0], [1.5, 2.0]])
    np.testing.assert_allclose(result["pathway_activity_means"], [[0.2, 0.6], [0.3, 0.7]])


def test_association_counts_donors_not_cells_as_replicates() -> None:
    donors = [f"D{i}" for i in range(5)]
    lnc = pd.DataFrame({"L1": [0.0, 1.0, 2.0, 3.0, 4.0]}, index=donors)
    pathway = pd.DataFrame({"P1": [0.1, 0.2, 0.3, 0.4, 0.5]}, index=donors)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_donor_associations(
            lnc,
            pathway,
            compartment="malignant",
            min_donors=5,
            min_abs_rho=0.5,
            max_nominal_p=0.05,
        )
    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert result.availability["status"] == "AVAILABLE"
    assert result.availability["biological_unit"] == "DONOR"
    assert result.availability["donor_count"] == 5
    assert len(result.evidence) == 1
    row = result.evidence.iloc[0]
    assert row.n_donors == 5
    assert row.spearman_rho == pytest.approx(1.0)
    assert row.nominal_p == 0.0
    assert bool(row.bh_q_is_conservative_upper_bound) is True
    assert "CONSERVATIVE_UPPER_BOUND" in row.multiple_testing_adjustment
    assert bool(row.cell_as_independent_replicate) is False


def test_insufficient_donors_emits_typed_null_not_numeric_association() -> None:
    donors = ["D1", "D2", "D3", "D4"]
    result = compute_donor_associations(
        pd.DataFrame({"L1": [0.0, 1.0, 2.0, 3.0]}, index=donors),
        pd.DataFrame({"P1": [0.1, 0.2, 0.3, 0.4]}, index=donors),
        compartment="immune",
        min_donors=5,
    )
    assert result.evidence.empty
    assert result.availability["status"] == "TYPED_UNAVAILABLE"
    assert result.availability["unavailable_reason"] == INSUFFICIENT_DONOR_REPLICATION


def test_low_memory_association_chunks_match_reference_numerics() -> None:
    donors = [f"D{i}" for i in range(7)]
    lnc = pd.DataFrame(
        {
            "L2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            "L1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "L0": [1.0] * 7,
        },
        index=donors,
    )
    pathway = pd.DataFrame(
        {
            "P2": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "P1": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "P0": [0.2] * 7,
        },
        index=donors,
    )
    reference = compute_donor_associations(
        lnc,
        pathway,
        compartment="malignant",
        min_donors=5,
        min_abs_rho=0.5,
        max_nominal_p=0.05,
        pathway_block=2,
    )
    chunks = []
    availability = stream_donor_association_chunks(
        lnc,
        pathway,
        compartment="malignant",
        evidence_sink=chunks.append,
        min_donors=5,
        min_abs_rho=0.5,
        max_nominal_p=0.05,
        pathway_block=2,
    )
    observed = pd.concat(
        [
            pd.DataFrame(
                {
                    "lncrna_id": lnc.columns.to_numpy()[chunk.lncrna_rows],
                    "pathway_id": pathway.columns.to_numpy()[chunk.pathway_rows],
                    "spearman_rho": chunk.spearman_rho,
                    "nominal_p": chunk.nominal_p,
                }
            )
            for chunk in chunks
        ],
        ignore_index=True,
    ).sort_values(["lncrna_id", "pathway_id"]).reset_index(drop=True)
    expected = reference.evidence.loc[
        :, ["lncrna_id", "pathway_id", "spearman_rho", "nominal_p"]
    ].sort_values(["lncrna_id", "pathway_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, expected)
    assert availability["tested_pair_count"] == reference.availability["tested_pair_count"]
    assert availability["retained_evidence_count"] == len(expected)
    assert all(len(chunk.nominal_p) <= len(lnc) * 2 for chunk in chunks)


def test_checkpoint_resume_is_bound_to_contract_and_payload_sha(tmp_path: Path) -> None:
    arrays = {
        "cell_counts": np.asarray([2, 3], dtype=np.int64),
        "pathway_sums": np.asarray([[1.0, 2.0]], dtype=np.float64),
    }
    write_checkpoint(
        tmp_path,
        contract_sha256="a" * 64,
        next_cell=5,
        total_cells=10,
        arrays=arrays,
    )
    state, loaded = load_checkpoint(tmp_path, expected_contract_sha256="a" * 64)
    assert state["next_cell"] == 5
    np.testing.assert_array_equal(loaded["cell_counts"], arrays["cell_counts"])
    with pytest.raises(StreamingContractError, match="contract"):
        load_checkpoint(tmp_path, expected_contract_sha256="b" * 64)
    with (tmp_path / "CHECKPOINT.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(StreamingContractError, match="SHA"):
        load_checkpoint(tmp_path, expected_contract_sha256="a" * 64)


def test_atomic_publish_never_overwrites_a_cancer_result(tmp_path: Path) -> None:
    staging = tmp_path / "HNSC.publish.tmp"
    staging.mkdir()
    (staging / "SUCCESS.json").write_text("{}\n", encoding="utf-8")
    final = tmp_path / "cancer_id=HNSC"
    atomic_publish_directory(staging, final)
    assert (final / "SUCCESS.json").is_file()
    other = tmp_path / "other.tmp"
    other.mkdir()
    with pytest.raises(StreamingContractError, match="exists"):
        atomic_publish_directory(other, final)


def test_historical_derived_paths_cannot_be_relabelled_fresh() -> None:
    assert_fresh_r7_source(
        Path("./data/CancerLncAtlas/processed/sc_tool_input/HNSC/raw_feature_bc_matrix.h5"),
        role="raw_h5",
    )
    assert_fresh_r7_source(
        Path("./data/CancerLncAtlas/runtime/single_cell_cell_level_r7_portable_supersession_20260829/authority/gencode_v50_gene_annotation.parquet"),
        role="annotation",
    )
    with pytest.raises(StreamingContractError, match="historical derived"):
        assert_fresh_r7_source(
            Path("./data/CancerLncAtlas/results/sc_trajectory/HNSC/ucell_r6.parquet"),
            role="activity",
        )


def test_smallest_formal_cancer_is_selected_from_preflight_counts() -> None:
    report = {
        "formal_cancers": [
            {"cancer_id": "ACC", "matrix_header": {"cells": 94_903}},
            {"cancer_id": "HNSC", "matrix_header": {"cells": 5_902}},
            {"cancer_id": "SARC", "matrix_header": {"cells": 6_228}},
        ]
    }
    assert select_smallest_formal_cancer(report) == {
        "cancer_id": "HNSC",
        "cells": 5_902,
    }


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / (
        "run_v32_single_cell_r7_streaming.py"
    )
    spec = importlib.util.spec_from_file_location("r7_streaming_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unique_symbol_fallback_retains_stable_gene_identity() -> None:
    runner = _load_runner()
    annotation = pd.DataFrame(
        {
            "gene_id": ["ENSG000001.7", "ENSG000002.3"],
            "gene_symbol": ["PC1", "LNC2"],
            "gene_class": ["protein_coding", "lncRNA"],
            "unique_symbol": [True, True],
        }
    )
    mapping = runner._build_feature_mapping(
        feature_ids=["ENSG000001.9", "unversioned_source_alias"],
        feature_names=["PC1", "LNC2"],
        annotation=annotation,
        measurement_scale="raw_counts",
    )
    assert mapping["protein_ids"] == ("ENSG000001",)
    assert mapping["lncrna_ids"] == ("ENSG000002",)
    assert mapping["lncrna_symbols"] == ("LNC2",)
    assert mapping["mapping_route_counts"] == {
        "DIRECT_STABLE_ID": 1,
        "UNIQUE_GENCODE_SYMBOL": 1,
        "UNMAPPED": 0,
    }


def test_stable_gene_id_strips_known_prefix_and_terminal_version_only() -> None:
    runner = _load_runner()
    assert runner._stable_gene_id("GENE:ENSG000003.12") == "ENSG000003"
    assert runner._stable_gene_id("lnc:ENSG000004.2") == "ENSG000004"
    assert runner._stable_gene_id("HLA.DRA") == "HLA.DRA"


def test_raw_count_measurement_gate_rejects_fractional_source_values() -> None:
    runner = _load_runner()
    runner._validate_measurement_block(
        sparse.csc_matrix([[0.0, 2.0], [3.0, 4.0]]),
        measurement_scale="raw_counts",
    )
    with pytest.raises(runner.R7StreamingRunError, match="fractional"):
        runner._validate_measurement_block(
            sparse.csc_matrix([[0.0, 2.5]]), measurement_scale="raw_counts"
        )


def test_per_cancer_materializer_publishes_summaries_not_cell_pathway_rows(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    membership = pd.DataFrame(
        {"pathway_id": ["P1", "P1", "P2"], "gene_id": ["G1", "G2", "G2"]}
    )
    pathway_contract = build_ucell_pathway_contract(
        membership, protein_gene_ids=["G1", "G2"], max_rank=10
    )
    groups = pd.DataFrame(
        {
            "patient_id": [f"D{i}" for i in range(5)],
            "cell_type_major": ["Malignant"] * 5,
            "compartment": ["malignant"] * 5,
        }
    )
    accumulator = StreamingGroupAccumulator(
        lncrna_count=2, pathway_count=2, group_count=5
    )
    accumulator.update(
        sparse.csc_matrix(
            np.asarray([[0.0, 1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0, 0.0]])
        ),
        np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]),
        np.arange(5),
    )
    work = tmp_path / "work"
    write_checkpoint(
        work,
        contract_sha256="c" * 64,
        next_cell=5,
        total_cells=5,
        arrays=accumulator.state_arrays(),
    )
    bundle = {
        "cancer_id": "HNSC",
        "dataset_id": "SYNTHETIC",
        "contract_sha256": "c" * 64,
        "contract": {
            "format": "synthetic",
            "cell_level_pathway_matrix_persisted": False,
            "historical_checkpoints_used": False,
        },
        "pathway_contract": pathway_contract,
        "group_frame": groups,
        "mapping": {
            "lncrna_ids": ("L1", "L2"),
            "lncrna_symbols": ("L1", "L2"),
        },
        "measurement_scale": "raw_counts",
        "resource_estimate": {"cell_level_pathway_matrix_persisted": False},
        "input_paths": {},
        "input_shas": {},
        "code_shas": {},
        "shape": (2, 5),
        "metadata": pd.DataFrame({"patient_id": [f"D{i}" for i in range(5)]}),
        "keep_mask": np.ones(5, dtype=bool),
    }
    output_parent = tmp_path / "results"
    output_parent.mkdir()
    args = SimpleNamespace(
        output_parent=output_parent,
        min_cells_per_donor_context=1,
        min_lncrna_detect_rate=0.0,
        min_lncrna_detecting_donors=1,
        min_donors=5,
        min_abs_rho=0.5,
        max_nominal_p=0.05,
        association_pathway_block=2,
        resume=False,
    )
    result = runner._materialize_outputs(bundle, accumulator, work, args)
    final = output_parent / "cancer_id=HNSC"
    assert result["status"] == "SUCCESS"
    assert result["cell_level_pathway_rows_written"] == 0
    assert not (final / "cell_level_ucell").exists()
    assert (final / "lncrna_donor_celltype_summary.parquet").is_file()
    assert (final / "pathway_donor_celltype_summary.parquet").is_file()
    assert (final / "association_context_availability.parquet").is_file()
    testability = pd.read_parquet(final / "association_lncrna_testability.parquet")
    assert set(testability.source_generation) == {"V3.2_R7_FRESH_FROM_RAW_H5"}
    evidence = pd.read_parquet(final / "association_evidence.parquet")
    assert evidence.bh_q_is_conservative_upper_bound.astype(bool).all()
    reference = compute_donor_associations(
        pd.DataFrame(
            {"L1": [0.0, 1.0, 2.0, 3.0, 4.0], "L2": [4.0, 3.0, 2.0, 1.0, 0.0]},
            index=[f"D{i}" for i in range(5)],
        ),
        pd.DataFrame(
            {"P1": [0.1, 0.2, 0.3, 0.4, 0.5], "P2": [0.5, 0.4, 0.3, 0.2, 0.1]},
            index=[f"D{i}" for i in range(5)],
        ),
        compartment="malignant",
        min_donors=5,
        min_abs_rho=0.5,
        max_nominal_p=0.05,
        pathway_block=2,
    )
    compare = [
        "lncrna_id",
        "pathway_id",
        "spearman_rho",
        "nominal_p",
        "bh_q_global_tests",
    ]
    pd.testing.assert_frame_equal(
        evidence.loc[:, compare].reset_index(drop=True),
        reference.evidence.loc[:, compare].reset_index(drop=True),
        check_dtype=False,
    )
    assert not (final / ".association_evidence_raw.parquet").exists()
    assert not (final / ".association_duckdb_scratch").exists()
    success = pd.read_json(final / "SUCCESS.json", typ="series")
    assert bool(success["cell_as_independent_replicate"]) is False
    assert bool(success["historical_derived_results_used"]) is False
    assert bool(success["raw_h5_full_sha256_verified"]) is True
