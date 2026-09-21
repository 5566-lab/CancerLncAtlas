from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from cc_hhgt.v32 import segment_cnv_streaming as streaming
from cc_hhgt.v32.gdc_segment_cnv import selected_segment_target
from cc_hhgt.v32.segment_cnv_streaming import (
    CompactCNVPartition,
    StreamingCNVError,
    materialize_cancer_partition,
    validate_cancer_partition,
)
from cc_hhgt.v32.genomic_training import candidate_statistics


def _fixture(tmp_path: Path, patients: int = 10):
    candidates = pd.DataFrame(
        [{"cancer_id": "BRCA", "lncrna_id": "L1", "pathway_id": "P1"}]
    )
    folds = pd.DataFrame(
        [
            {"cancer_id": "BRCA", "patient_id": f"P{i:04d}", "patient_fold_id": i % 5}
            for i in range(patients)
        ]
    )
    intervals = pd.DataFrame(
        [
            {"entity_type": "lncrna", "entity_id": "L1", "chromosome": "1", "start": 10, "end": 20},
            {"entity_type": "gene", "entity_id": "G1", "chromosome": "1", "start": 30, "end": 40},
            {"entity_type": "gene", "entity_id": "G2", "chromosome": "1", "start": 50, "end": 60},
        ]
    )
    membership = pd.DataFrame(
        [{"pathway_id": "P1", "gene_id": "G1"}, {"pathway_id": "P1", "gene_id": "G2"}]
    )
    staging = tmp_path / "staging"
    selected = {}
    for index in range(patients - 1):
        patient = f"P{index:04d}"
        row = {
            "cancer_id": "BRCA", "file_id": f"uuid-{index}", "file_name": f"{patient}.seg.txt",
            "case_submitter_id": patient, "sample_submitter_id": patient,
        }
        path = selected_segment_target(row, staging)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = 0.6 if index % 2 else 0.0
        pd.DataFrame(
            [
                {"Chromosome": "1", "Start": 1, "End": 25, "Segment_Mean": value},
                {"Chromosome": "1", "Start": 26, "End": 45, "Segment_Mean": value},
                {"Chromosome": "1", "Start": 46, "End": 70, "Segment_Mean": value},
            ]
        ).to_csv(path, sep="\t", index=False)
        selected[("BRCA", patient)] = row
    return candidates, folds, intervals, membership, staging, selected


def test_compact_partition_is_fold_resumable_and_never_materializes_long_rows(tmp_path: Path) -> None:
    candidates, folds, intervals, membership, staging, selected = _fixture(tmp_path)
    output = tmp_path / "compact"
    payload = materialize_cancer_partition(
        cancer_id="BRCA", staging_root=staging, output_root=output,
        candidates=candidates, folds=folds, intervals=intervals, membership=membership,
        selected_rows=selected, context_sha256="a" * 64, max_memory_bytes=1_000_000_000,
    )
    assert payload["max_materialized_long_rows"] == 0
    assert payload["typed_unavailable_patients"] == 1
    assert {item["patient_fold"] for item in payload["folds"]} == set(range(5))
    for fold in range(5):
        assert (output / "cancer=BRCA" / f"fold={fold}" / "SUCCESS.json").is_file()
    assert validate_cancer_partition(output / "cancer=BRCA", context_sha256="a" * 64)["status"] == "SUCCESS"
    partition = CompactCNVPartition(output / "cancer=BRCA")
    domain, labels, available, reasons = partition.candidate_statistics_arrays(
        candidates, folds.patient_id.tolist(), min_pair_callable=2
    )
    assert domain.shape == (1, 6)
    assert available[0]
    assert labels[0] == 1
    assert reasons[0] == ""
    assert partition.lc[-1, 0] == 0
    assert pd.isna(partition.lb[-1, 0])

    long_lnc = []
    long_pathway = []
    for index in range(9):
        patient = f"P{index:04d}"
        event = bool(index % 2)
        burden = 0.6 if event else 0.0
        long_lnc.append({"cancer_id": "BRCA", "patient_id": patient, "entity_id": "L1", "event": event, "callable": True, "burden": burden})
        long_pathway.append({"cancer_id": "BRCA", "patient_id": patient, "entity_id": "P1", "event": event, "callable": True, "burden": burden * 2})
    legacy = candidate_statistics(
        candidates, pd.DataFrame(long_lnc), pd.DataFrame(long_pathway),
        folds.patient_id.tolist(), modality="cnv", min_pair_callable=2,
    )
    np.testing.assert_allclose(domain, legacy.domain, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(labels, legacy.labels)
    np.testing.assert_array_equal(available, legacy.available)


def test_large_synthetic_memory_gate_blocks_before_publish(tmp_path: Path) -> None:
    candidates, folds, intervals, membership, staging, selected = _fixture(tmp_path, patients=100)
    large = pd.concat([intervals] * 50_000, ignore_index=True)
    output = tmp_path / "compact"
    with pytest.raises(StreamingCNVError, match="MEMORY_GATE"):
        materialize_cancer_partition(
            cancer_id="BRCA", staging_root=staging, output_root=output,
            candidates=candidates, folds=folds, intervals=large, membership=membership,
            selected_rows=selected, context_sha256="b" * 64, max_memory_bytes=1024,
        )
    assert not (output / "cancer=BRCA").exists()


def test_observed_process_memory_gate_blocks_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates, folds, intervals, membership, staging, selected = _fixture(tmp_path)
    monkeypatch.setattr(
        streaming, "_observed_process_memory_bytes", lambda: (2_000_000, 2_500_000)
    )
    output = tmp_path / "compact"
    with pytest.raises(StreamingCNVError, match="OBSERVED_MEMORY_GATE"):
        materialize_cancer_partition(
            cancer_id="BRCA", staging_root=staging, output_root=output,
            candidates=candidates, folds=folds, intervals=intervals, membership=membership,
            selected_rows=selected, context_sha256="d" * 64, max_memory_bytes=1_000_000,
        )
    assert not (output / "cancer=BRCA").exists()


def test_candidate_reader_filters_before_pandas_materialization(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "candidates.parquet"
    pd.DataFrame(
        [
            {"cancer_id": "BRCA", "lncrna_id": "L1", "pathway_id": "P1"},
            {"cancer_id": "LUAD", "lncrna_id": "L2", "pathway_id": "P2"},
        ]
    ).to_parquet(path, index=False)
    observed = streaming._read_cancer_candidates(path, "brca")
    assert observed.to_dict("records") == [
        {"pathway_id": "P1", "lncrna_id": "L1", "cancer_id": "BRCA"}
    ]


def test_row_gate_failure_retains_incomplete_but_never_publishes(tmp_path: Path) -> None:
    candidates, folds, intervals, membership, staging, selected = _fixture(tmp_path)
    output = tmp_path / "compact"
    with pytest.raises(StreamingCNVError, match="ROW_GATE"):
        materialize_cancer_partition(
            cancer_id="BRCA", staging_root=staging, output_root=output,
            candidates=candidates, folds=folds, intervals=intervals, membership=membership,
            selected_rows=selected, context_sha256="c" * 64, max_segment_rows=1,
        )
    assert not (output / "cancer=BRCA").exists()
    evidence = list(output.glob(".cancer=BRCA.building.*/INCOMPLETE.json"))
    assert len(evidence) == 1
    assert json.loads(evidence[0].read_text(encoding="utf-8"))["status"] == "FAILED"
