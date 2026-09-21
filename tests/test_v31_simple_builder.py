from __future__ import annotations

import pandas as pd

from cc_hhgt.v31_simple_builder import load_full_source_universe


def test_full_source_loader_uses_frozen_pathway_and_state_outcomes(tmp_path) -> None:
    tables = tmp_path / "tables"
    for cancer in ("A", "B"):
        pathway_dir = tables / "candidate_universe" / f"cancer_id={cancer}"
        pathway_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "candidate_id": [f"p:{cancer}"],
                "cancer_id": [cancer],
                "lncrna_id": ["L1"],
                "pathway_family_id": ["P1"],
                "label": [1 if cancer == "A" else 0],
                "bulk_effect": [0.4 if cancer == "A" else -0.2],
                "direction": ["positive" if cancer == "A" else "negative"],
            }
        ).to_parquet(pathway_dir / "part.parquet", index=False)
        state_dir = tables / "strict_state_candidate" / f"cancer_id={cancer}"
        state_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "candidate_id": [f"s:{cancer}"],
                "cancer_id": [cancer],
                "lncrna_id": ["L1"],
                "state_id": ["stemness_rna::RNAss"],
                "proxy_label": [1 if cancer == "A" else 0],
                "effect": [0.3 if cancer == "A" else -0.1],
                "n_observed": [50],
                "direction": ["positive" if cancer == "A" else "negative"],
                "evaluation_eligibility": ["ELIGIBLE"],
            }
        ).to_parquet(state_dir / "part.parquet", index=False)
    pathway = load_full_source_universe(tmp_path, ["A", "B"], "pathway")
    state = load_full_source_universe(tmp_path, ["A", "B"], "state")
    assert len(pathway) == 2 and len(state) == 2
    assert pathway.target_type.eq("pathway").all()
    assert pathway.effect.tolist() == [0.4, -0.2]
    assert state.target_type.eq("state").all()
    assert state.label.tolist() == [1, 0]
