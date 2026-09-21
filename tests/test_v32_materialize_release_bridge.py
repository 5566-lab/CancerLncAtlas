from pathlib import Path

import pandas as pd

from scripts.materialize_v32_release import (
    SUBTYPE_MEMBER_COLUMNS,
    _load_subtype_members,
)


def test_subtype_bridge_retains_membership_probability(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "geneset_id": ["GS1"],
            "geneset_rank": [1],
            "cancer_id": ["BRCA"],
            "lncrna_id": ["ENSG_LNC_1"],
            "pathway_id": ["HALLMARK_EMT"],
            "pathway_family_id": ["FAMILY_EMT"],
            "association_membership_probability": [0.91],
            "association_direction": ["positive"],
            "shared_or_local_scope": ["shared"],
            "unrelated_column": ["must_not_be_loaded"],
        }
    )
    path = tmp_path / "members.parquet"
    frame.to_parquet(path, index=False)

    loaded = _load_subtype_members([path])

    assert tuple(loaded.columns) == SUBTYPE_MEMBER_COLUMNS
    assert loaded.loc[0, "association_membership_probability"] == 0.91
