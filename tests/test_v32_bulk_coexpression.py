from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.bulk_coexpression_query import (
    BulkCoexpressionQueryAssetError,
    BulkCoexpressionReleaseQuery,
)
from cc_hhgt.v32.bulk_coexpression_release import (
    artifact_sha256,
    materialize_bulk_coexpression_release,
)


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    rng = np.random.default_rng(20260826)
    cancer = "TST"
    n = 60
    patients = [f"TCGA-AA-{index:04d}" for index in range(n)]
    samples = [f"{patient}-01A-11R-TEST-00" for patient in patients]
    signal = rng.normal(size=n)
    lnc_values = {
        "LNC:ENSG00000000001": signal + rng.normal(scale=0.02, size=n),
        "LNC:ENSG00000000002": -signal + rng.normal(scale=0.02, size=n),
        "LNC:ENSG00000000003": np.ones(n),
    }
    gene_values = {
        "GENE:ENSG00000010001": signal + rng.normal(scale=0.02, size=n),
        "GENE:ENSG00000010002": -signal + rng.normal(scale=0.02, size=n),
    }
    for index in range(3, 23):
        gene_values[f"GENE:ENSG0000001{index:04d}"] = rng.normal(size=n)

    lnc_rows = []
    for lncrna_id, values in lnc_values.items():
        for patient, sample, value in zip(patients, samples, values):
            lnc_rows.append(
                {
                    "cancer_id": cancer,
                    "sample_id": sample,
                    "patient_id": patient,
                    "lncrna_id": lncrna_id,
                    "logcpm": float(value),
                }
            )
    gene_rows = []
    for gene_id, values in gene_values.items():
        for patient, sample, value in zip(patients, samples, values):
            gene_rows.append(
                {
                    "cancer_id": cancer,
                    "sample_id": sample,
                    "patient_id": patient,
                    "gene_id": gene_id,
                    "logcpm": float(value),
                }
            )
    lnc_root = tmp_path / "lnc"
    gene_root = tmp_path / "gene"
    (lnc_root / f"cancer_id={cancer}").mkdir(parents=True)
    (gene_root / f"cancer_id={cancer}").mkdir(parents=True)
    pd.DataFrame(lnc_rows).to_parquet(
        lnc_root / f"cancer_id={cancer}" / "part-0.parquet", index=False
    )
    pd.DataFrame(gene_rows).to_parquet(
        gene_root / f"cancer_id={cancer}" / "part-0.parquet", index=False
    )
    covariates = pd.DataFrame(
        {
            "cancer_id": cancer,
            "sample_id": samples,
            "patient_id": patients,
            "purity": rng.uniform(0.6, 0.95, n),
            "age_years": rng.uniform(35, 80, n),
            "sex": ["female", "male"] * (n // 2),
            "stage": ["I", "II", "III"] * (n // 3),
        }
    )
    cov_path = tmp_path / "covariates.parquet"
    covariates.to_parquet(cov_path, index=False)
    candidates = pd.DataFrame(
        [
            {
                "cancer_id": cancer,
                "lncrna_id": lncrna_id,
                "pathway_id": pathway,
            }
            for lncrna_id in lnc_values
            for pathway in ("PATHWAY:A", "PATHWAY:B")
        ]
    )
    candidate_path = tmp_path / "candidates.parquet"
    candidates.to_parquet(candidate_path, index=False)
    output = tmp_path / "release"
    result = materialize_bulk_coexpression_release(
        lncrna_expression_root=lnc_root,
        gene_expression_root=gene_root,
        covariates_path=cov_path,
        exact_candidate_path=candidate_path,
        output_root=output,
        runner_path=Path(__file__),
        strict_formal_authority=False,
        min_samples=40,
        min_variance=0.01,
        min_abs_rho=0.40,
        max_fdr=0.05,
        max_edges_per_direction=5,
        block_size=2,
        max_clusters=4,
        random_state=11,
    )
    return {
        "output": output,
        "binding": Path(result["binding_path"]),
        "binding_sha256": str(result["binding_sha256"]),
    }


def test_fresh_coexpression_query_and_typed_unavailability(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    query = BulkCoexpressionReleaseQuery(
        fixture["binding"],
        expected_binding_sha256=fixture["binding_sha256"],
        require_formal_authority=False,
    )
    result = query.query_lncrna(
        lncrna_id="ENSG00000000001",
        cancer_id="tst",
        limit=50,
    )
    assert result["returned_edge_rows"] >= 2
    assert result["association_not_physical_binding"] is True
    assert result["provenance"]["historical_derived_outputs_used"] is False
    assert all(row["fdr"] <= 0.05 for row in result["edge_rows"])
    unavailable = query.query_lncrna(
        lncrna_id="LNC:ENSG00000000003",
        cancer_id="TST",
    )
    assert unavailable["returned_edge_rows"] == 0
    assert unavailable["availability_rows"][0]["availability"] is False
    assert (
        unavailable["availability_rows"][0]["failure_reason"]
        == "LNC_LOW_VARIANCE_OR_MISSING_EXPRESSION"
    )
    clusters = query.query_clusters(cancer_id="TST", limit=50)
    assert clusters["returned_rows"] >= 1


def test_binding_requires_expected_hash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(BulkCoexpressionQueryAssetError):
        BulkCoexpressionReleaseQuery(
            fixture["binding"],
            expected_binding_sha256=None,
            require_formal_authority=False,
        )
    with pytest.raises(BulkCoexpressionQueryAssetError):
        BulkCoexpressionReleaseQuery(
            fixture["binding"],
            expected_binding_sha256="0" * 64,
            require_formal_authority=False,
        )


def test_bound_artifact_drift_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    binding = json.loads(Path(fixture["binding"]).read_text(encoding="utf-8"))
    summary = Path(binding["artifacts"]["coexpression_cancer_summary"]["path"])
    frame = pd.read_parquet(summary)
    frame.loc[0, "edges"] = int(frame.loc[0, "edges"]) + 1
    frame.to_parquet(summary, index=False)
    assert artifact_sha256(summary) != binding["artifacts"][
        "coexpression_cancer_summary"
    ]["sha256"]
    with pytest.raises(BulkCoexpressionQueryAssetError, match="SHA drift"):
        BulkCoexpressionReleaseQuery(
            fixture["binding"],
            expected_binding_sha256=fixture["binding_sha256"],
            require_formal_authority=False,
        )
