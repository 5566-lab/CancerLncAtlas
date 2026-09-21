from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.v31_rna_context import augment_rna_context, summarize_coexpression_edges


def test_coexpression_summary_preserves_sign_and_missingness(tmp_path: Path) -> None:
    edges = pd.DataFrame(
        {
            "lncrna_id": ["L1", "L1", "L2"],
            "gene_id": ["G1", "G2", "G1"],
            "rho": [0.5, -0.25, 0.1],
            "method": ["covariate_residual_spearman_effective_df"] * 3,
            "p_value": [0.01, 0.02, 0.03],
            "fdr": [0.03, 0.03, 0.03],
        }
    )
    summary = summarize_coexpression_edges(
        edges[["lncrna_id", "gene_id", "rho", "method"]]
    ).set_index("lncrna_id")
    assert np.isclose(summary.loc["L1", "log1p_coexpression_degree"], np.log1p(2))
    assert summary.loc["L1", "positive_edge_fraction"] == 0.5
    assert np.isclose(summary.loc["L1", "mean_rho"], 0.125)
    assert np.isclose(summary.loc["L1", "mean_absolute_rho"], 0.375)
    assert summary.loc["L1", "maximum_absolute_rho"] == 0.5

    root = tmp_path / "coexpression"
    partition = root / "cancer_id=BRCA"
    partition.mkdir(parents=True)
    edges.to_parquet(partition / "part-0.parquet", index=False)
    base = pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA", "COAD"],
            "lncrna_id": ["L1", "L3", "L1"],
            "bulk_detection_rate": [0.5, 0.4, 0.3],
            "bulk_detection_rate__available": [True, True, True],
        }
    )
    augmented, audit = augment_rna_context(base, root)
    matched = augmented.loc[
        augmented.cancer_id.eq("BRCA") & augmented.lncrna_id.eq("L1")
    ].iloc[0]
    missing_edge = augmented.loc[
        augmented.cancer_id.eq("BRCA") & augmented.lncrna_id.eq("L3")
    ].iloc[0]
    missing_partition = augmented.loc[augmented.cancer_id.eq("COAD")].iloc[0]
    assert matched["mean_rho__available"]
    assert not missing_edge["mean_rho__available"] and np.isnan(missing_edge.mean_rho)
    assert not missing_partition["mean_rho__available"] and np.isnan(missing_partition.mean_rho)
    assert audit["available_cancers"] == ["BRCA"]
    assert audit["unavailable_cancers"] == ["COAD"]
    assert "p_value" not in augmented and "fdr" not in augmented


def test_coexpression_rejects_outcome_columns() -> None:
    edges = pd.DataFrame(
        {
            "lncrna_id": ["L1"],
            "gene_id": ["G1"],
            "rho": [0.2],
            "method": ["covariate_residual_spearman_effective_df"],
            "proxy_label": [1],
        }
    )
    try:
        summarize_coexpression_edges(edges)
    except RuntimeError as exc:
        assert "Outcome-derived" in str(exc)
    else:
        raise AssertionError("Outcome-derived coexpression column was accepted")
