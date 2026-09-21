from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "materialize_v32_methylation_lncrna.py"
SPEC = importlib.util.spec_from_file_location("materialize_v32_methylation_lncrna", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_single_beta_file_maps_promoter_and_distal_without_zero_imputation(tmp_path: Path) -> None:
    probe_map = pd.DataFrame(
        {
            "probe_id": ["cg1", "cg2", "cg3"],
            "lncrna_id": ["LNC:G1", "LNC:G1", "LNC:G2"],
            "region_type": ["promoter", "promoter", "distal"],
        }
    )
    probe_map_path = tmp_path / "probe_map.parquet"
    probe_map.to_parquet(probe_map_path, index=False)
    beta_path = tmp_path / "sample.txt"
    beta_path.write_text("cg1\t0.2\ncg2\t0.6\ncg3\tNA\ncg4\t0.9\n", encoding="utf-8")
    MODULE._initialise_worker(str(probe_map_path))
    result, record = MODULE._process_file(
        {
            "path": str(beta_path),
            "file_id": "F1",
            "sample_id": "S1",
            "patient_id": "P1",
            "cancer_id": "BRCA",
        }
    )
    g1 = result.loc[result.lncrna_id.eq("LNC:G1")].iloc[0]
    assert g1.promoter_methylation_beta == pytest.approx(0.4)
    assert g1.promoter_probe_count == 2
    assert pd.isna(g1.distal_methylation_beta)
    assert record["promoter_available"] == 1
    assert record["distal_available"] == 0


def test_single_beta_file_rejects_out_of_range_beta(tmp_path: Path) -> None:
    probe_map_path = tmp_path / "probe_map.parquet"
    pd.DataFrame(
        {"probe_id": ["cg1"], "lncrna_id": ["LNC:G1"], "region_type": ["promoter"]}
    ).to_parquet(probe_map_path, index=False)
    beta_path = tmp_path / "bad.txt"
    beta_path.write_text("cg1\t1.2\n", encoding="utf-8")
    MODULE._initialise_worker(str(probe_map_path))
    with pytest.raises(RuntimeError, match="outside"):
        MODULE._process_file(
            {
                "path": str(beta_path),
                "file_id": "F1",
                "sample_id": "S1",
                "patient_id": "P1",
                "cancer_id": "BRCA",
            }
        )
