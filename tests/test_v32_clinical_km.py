from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.clinical_km_query import (
    ClinicalKMQueryAssetError,
    ClinicalKMQueryInputError,
    ClinicalKMReleaseQuery,
)
from cc_hhgt.v32.clinical_km_release import (
    DFS_FAILURE_REASON,
    ClinicalKMReleaseError,
    artifact_sha256,
    materialize_clinical_km_release,
    vectorized_logrank_km,
)


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "materialize_v32_clinical_km_release.py"


def _build_fixture(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    workbook = root / "TCGA-CDR.xlsx"
    patients = [f"TCGA-AA-{index:04d}" for index in range(1, 9)]
    time = np.arange(1, 9, dtype=float) * 100.0
    event = np.array([1, 1, 1, 0, 1, 0, 1, 1], dtype=float)
    main = pd.DataFrame(
        {
            "bcr_patient_barcode": patients,
            "type": ["BRCA"] * len(patients),
            "OS": event,
            "OS.time": time,
            "DSS": event,
            "DSS.time": time,
            "PFI": event,
            "PFI.time": time,
            "DFI": event,
            "DFI.time": time,
        }
    )
    extra = pd.DataFrame(
        {
            "bcr_patient_barcode": patients,
            "type": ["BRCA"] * len(patients),
            "PFS": event,
            "PFS.time": time,
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        main.to_excel(writer, sheet_name="TCGA-CDR", index=False)
        extra.to_excel(writer, sheet_name="ExtraEndpoints", index=False)

    expression_root = root / "expression"
    partition_root = expression_root / "cancer_id=BRCA"
    partition_root.mkdir(parents=True)
    rows = []
    for index, patient in enumerate(patients):
        for lnc, value in (
            ("LNC:ENSG00000000001", float(index)),
            ("LNC:ENSG00000000002", 1.0),
        ):
            rows.append(
                {
                    "cancer_id": "BRCA",
                    "sample_id": patient + "-01A",
                    "patient_id": patient,
                    "lncrna_id": lnc,
                    "logcpm": value,
                }
            )
    pd.DataFrame(rows).to_parquet(partition_root / "part-0.parquet", index=False)
    candidates = root / "candidates.parquet"
    pd.DataFrame(
        {
            "cancer_id": ["BRCA"] * 4,
            "lncrna_id": [
                "LNC:ENSG00000000001",
                "LNC:ENSG00000000001",
                "LNC:ENSG00000000002",
                "LNC:ENSG00000000002",
            ],
            "pathway_id": ["P1", "P2", "P1", "P2"],
        }
    ).to_parquet(candidates, index=False)
    return {
        "workbook": workbook,
        "expression": expression_root,
        "candidates": candidates,
    }


def _materialize(root: Path) -> tuple[dict[str, Path], Path, dict[str, object]]:
    inputs = _build_fixture(root / "inputs")
    output = root / "release"
    result = materialize_clinical_km_release(
        workbook_path=inputs["workbook"],
        expression_root=inputs["expression"],
        exact_candidate_path=inputs["candidates"],
        output_root=output,
        runner_path=RUNNER,
        strict_formal_authority=False,
    )
    return inputs, output, result


def _scalar_logrank(expression: np.ndarray, time: np.ndarray, event: np.ndarray):
    high = expression >= np.median(expression)
    observed = float(event[high].sum())
    expected = 0.0
    variance = 0.0
    for event_time in sorted(set(time[event == 1])):
        risk = time >= event_time
        n = float(risk.sum())
        n_high = float((risk & high).sum())
        deaths = time == event_time
        d = float(event[deaths].sum())
        expected += d * n_high / n
        if n > 1:
            variance += n_high * (n - n_high) * d * (n - d) / (n * n * (n - 1))
    z = (observed - expected) / math.sqrt(variance)
    return observed - expected, variance, z, math.erfc(abs(z) / math.sqrt(2.0))


def test_vectorized_logrank_matches_independent_scalar_oracle() -> None:
    expression = np.array(
        [
            [0.0, 3.0],
            [1.0, 7.0],
            [2.0, 1.0],
            [3.0, 5.0],
            [4.0, 2.0],
            [5.0, 8.0],
            [6.0, 4.0],
            [7.0, 6.0],
        ]
    )
    time = np.array([10, 10, 20, 30, 30, 40, 50, 60], dtype=float)
    event = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    result = vectorized_logrank_km(expression, time, event)
    for column in range(expression.shape[1]):
        expected = _scalar_logrank(expression[:, column], time, event)
        assert np.allclose(
            [
                result["observed_minus_expected_high"][column],
                result["variance"][column],
                result["z"][column],
                result["p_value"][column],
            ],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )
    assert np.all(result["survival_high"][1:] <= result["survival_high"][:-1] + 1e-12)
    assert np.all(result["survival_low"][1:] <= result["survival_low"][:-1] + 1e-12)


def test_materializer_builds_typed_six_endpoint_universe_and_curves(tmp_path: Path) -> None:
    _, output, result = _materialize(tmp_path)
    assert result["counts"] == {
        "candidate_pairs": 2,
        "statistics_rows": 12,
        "available_rows": 5,
        "unavailable_rows": 7,
        "curve_rows": 60,
        "cancers": 1,
        "lncrnas": 2,
        "endpoints": 6,
        "dfs_rows": 2,
        "dfs_available_rows": 0,
        "horizons": 6,
    }
    statistics = pd.read_parquet(output / "clinical_km_statistics.parquet")
    assert len(statistics) == 2 * 6
    assert set(statistics.clinical_endpoint) == {"OS", "DSS", "PFI", "PFS", "DFI", "DFS"}
    assert not statistics.duplicated(
        ["cancer_id", "lncrna_id", "clinical_endpoint"]
    ).any()
    dfs = statistics.loc[statistics.clinical_endpoint.eq("DFS")]
    dfi = statistics.loc[statistics.clinical_endpoint.eq("DFI")]
    assert not dfs.availability.any()
    assert set(dfs.failure_reason) == {DFS_FAILURE_REASON}
    assert dfs.logrank_p_value.isna().all() and dfs.median_logcpm.isna().all()
    assert dfi.availability.any()
    constant = statistics.loc[
        statistics.lncrna_id.eq("LNC:ENSG00000000002")
        & ~statistics.clinical_endpoint.eq("DFS")
    ]
    assert not constant.availability.any()
    assert set(constant.failure_reason) == {"MEDIAN_SPLIT_SINGLE_GROUP"}
    curves = pd.read_parquet(output / "clinical_km_curves")
    assert len(curves) == 60
    assert set(curves.expression_group) == {"HIGH", "LOW"}
    assert set(curves.horizon_years) == {0, 1, 2, 3, 5, 10}
    assert not curves.clinical_endpoint.eq("DFS").any()


def test_hash_pinned_query_returns_statistics_curves_and_provenance(tmp_path: Path) -> None:
    _, output, result = _materialize(tmp_path)
    query = ClinicalKMReleaseQuery(
        output / "CLINICAL_KM_BINDING.json",
        expected_binding_sha256=result["binding_sha256"],
        require_formal_authority=False,
    )
    response = query.query_pair(
        cancer_id="brca",
        lncrna_id="ENSG00000000001.9",
        clinical_endpoint="os",
    )
    assert response["returned_statistics_rows"] == 1
    assert response["returned_curve_rows"] == 12
    assert response["statistics"][0]["availability"] is True
    assert response["fresh_statistical_calculation"] is True
    assert response["training_not_applicable"] is True
    assert response["changes_primary_ranking"] is False
    assert response["provenance"]["historical_predictions_used"] is False
    dfs = query.query_pair(
        cancer_id="BRCA",
        lncrna_id="LNC:ENSG00000000001",
        clinical_endpoint="DFS",
    )
    assert dfs["returned_curve_rows"] == 0
    assert dfs["statistics"][0]["failure_reason"] == DFS_FAILURE_REASON


def test_query_and_materializer_fail_closed_on_hash_or_overwrite(tmp_path: Path) -> None:
    inputs, output, result = _materialize(tmp_path)
    binding = output / "CLINICAL_KM_BINDING.json"
    with pytest.raises(ClinicalKMQueryAssetError, match="SHA mismatch"):
        ClinicalKMReleaseQuery(
            binding,
            expected_binding_sha256="0" * 64,
            require_formal_authority=False,
        )
    with pytest.raises(ClinicalKMReleaseError, match="overwrite"):
        materialize_clinical_km_release(
            workbook_path=inputs["workbook"],
            expression_root=inputs["expression"],
            exact_candidate_path=inputs["candidates"],
            output_root=output,
            runner_path=RUNNER,
            strict_formal_authority=False,
        )
    assert artifact_sha256(binding) == result["binding_sha256"]


def test_query_detects_source_drift_and_validates_inputs(tmp_path: Path) -> None:
    inputs, output, result = _materialize(tmp_path)
    partition = inputs["expression"] / "cancer_id=BRCA" / "part-0.parquet"
    frame = pd.read_parquet(partition)
    frame.loc[0, "logcpm"] += 1.0
    frame.to_parquet(partition, index=False)
    with pytest.raises(ClinicalKMQueryAssetError, match="source SHA drift"):
        ClinicalKMReleaseQuery(
            output / "CLINICAL_KM_BINDING.json",
            expected_binding_sha256=result["binding_sha256"],
            require_formal_authority=False,
        )


def test_query_semantic_gate_rejects_events_above_endpoint_patients(
    tmp_path: Path,
) -> None:
    _, output, _ = _materialize(tmp_path)
    binding = json.loads(
        (output / "CLINICAL_KM_BINDING.json").read_text(encoding="utf-8")
    )
    statistics_path = output / "clinical_km_statistics.parquet"
    statistics = pd.read_parquet(statistics_path)
    index = statistics.index[statistics["availability"]].tolist()[0]
    statistics.loc[index, "n_events"] = (
        int(statistics.loc[index, "n_endpoint_patients"]) + 1
    )
    statistics.to_parquet(statistics_path, index=False)

    query = object.__new__(ClinicalKMReleaseQuery)
    query.paths = {
        "statistics": statistics_path,
        "curves": output / "clinical_km_curves",
    }
    query.counts = binding["counts"]
    query.computation_run_id = binding["computation_run_id"]
    with pytest.raises(ClinicalKMQueryAssetError, match="semantic validation"):
        query._validate_tables()


def test_strict_authority_and_query_bounds_fail_closed(tmp_path: Path) -> None:
    inputs = _build_fixture(tmp_path / "inputs")
    with pytest.raises(ClinicalKMReleaseError, match="workbook SHA256 mismatch"):
        materialize_clinical_km_release(
            workbook_path=inputs["workbook"],
            expression_root=inputs["expression"],
            exact_candidate_path=inputs["candidates"],
            output_root=tmp_path / "strict-release",
            runner_path=RUNNER,
            strict_formal_authority=True,
        )
    _, output, result = _materialize(tmp_path / "second")
    query = ClinicalKMReleaseQuery(
        output / "CLINICAL_KM_BINDING.json",
        expected_binding_sha256=result["binding_sha256"],
        require_formal_authority=False,
    )
    with pytest.raises(ClinicalKMQueryInputError, match="lncrna_id"):
        query.query_lncrna_survival(lncrna_id="")
    with pytest.raises(ClinicalKMQueryInputError, match="clinical_endpoint"):
        query.query_lncrna_survival(
            lncrna_id="ENSG00000000001", clinical_endpoint="RFS"
        )
    with pytest.raises(ClinicalKMQueryInputError, match="limit"):
        query.query_lncrna_survival(
            lncrna_id="ENSG00000000001", limit=199
        )


def test_directory_hash_uses_platform_stable_casefold_relative_order(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "cancer_id=ACC").mkdir(parents=True)
    (root / "SUCCESS.json").write_bytes(b"success\n")
    (root / "cancer_id=ACC" / "QC.json").write_bytes(b"qc\n")
    entries = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    digest = hashlib.sha256()
    for relative in sorted(entries, key=str.casefold):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entries[relative].encode("ascii"))
        digest.update(b"\n")
    assert artifact_sha256(root) == digest.hexdigest()
