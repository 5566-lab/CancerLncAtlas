from __future__ import annotations

import pandas as pd

from scripts.run_v32_single_cell_33c_partition_build import (
    build_feature_count_delta_table,
)


def test_blocked_partition_never_materialises_a_fake_fresh_direct_count() -> None:
    audit = pd.DataFrame(
        {
            "cancer_id": ["ACC", "BLCA"],
            "canonical_lnc_feature_unique": [11969, 2909],
        }
    ).set_index("cancer_id")
    records = {
        "ACC": {
            "status": "SUCCESS_PARTITION_BUILT_FORMAL_ELIGIBLE",
            "formal_eligible": True,
            # Exercise backward-compatible reading of a successful r5 status.
            "lncrna_feature_universe_count": 11986,
            "fresh_minus_audit_feature_count": 17,
            "feature_count_tolerance": 240,
        },
        "BLCA": {
            "status": "BLOCKED_MISSING_CELL_METADATA",
            "formal_eligible": False,
            "blocking_reason": "DONOR_CELLTYPE_METADATA_UNAVAILABLE",
            # This legacy/audit field was present in r5 and must not leak into
            # the explicitly fresh direct-ID column.
            "lncrna_feature_universe_count": 2909,
        },
    }

    result = build_feature_count_delta_table(
        ["ACC", "BLCA"], records, audit
    ).set_index("cancer_id")

    assert result.loc["ACC", "fresh_direct_id_lncrna_feature_count"] == 11986
    assert result.loc["ACC", "within_delta_gate"] is True
    for column in (
        "fresh_direct_id_lncrna_feature_count",
        "fresh_minus_audit",
        "allowed_delta_max_25_or_2pct",
        "within_delta_gate",
    ):
        assert pd.isna(result.loc["BLCA", column])

