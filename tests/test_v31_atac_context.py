from __future__ import annotations

import numpy as np
import pandas as pd

from cc_hhgt.v31_atac_context import (
    ATAC_FEATURES,
    derive_full_scope_r_script,
    finalize_atac_context,
)


def test_r_scope_patch_is_exact_and_algorithm_preserving() -> None:
    source = '\n'.join(
        [
            'cancers <- c("BRCA", "COAD", "KIRP")',
            'stopifnot(identical(sort(unique(candidate$cancer_id)), cancers))',
            'mapping[, cancer_id := sub("-.*$", "", bam_prefix)]',
            '  cancers = cancer_audit',
        ]
    )
    derived, audit = derive_full_scope_r_script(source)
    assert "registered_cancers" in derived
    assert 'sub("x$", "", cancer_id)' in derived
    assert audit["scope_changes_only"] and not audit["algorithm_changed"]


def test_atac_padding_never_treats_missing_cancer_as_zero() -> None:
    cancers = [f"C{i:02d}" for i in range(31)]
    keys = pd.DataFrame(
        [(cancer, lnc) for cancer in cancers for lnc in ("L1", "L2")],
        columns=["cancer_id", "lncrna_id"],
    )
    rows = []
    for cancer in cancers[:2]:
        for lnc in ("L1", "L2"):
            row = {"cancer_id": cancer, "lncrna_id": lnc}
            row.update({feature: 1.0 for feature in ATAC_FEATURES})
            if lnc == "L2":
                row["atac_promoter_mean"] = np.nan
            rows.append(row)
    final, audit = finalize_atac_context(
        keys, pd.DataFrame(rows), covered_cancers=cancers[:2]
    )
    unavailable = final.loc[final.cancer_id.eq(cancers[-1])]
    assert unavailable[list(ATAC_FEATURES)].isna().all().all()
    assert not unavailable[[f"{x}__available" for x in ATAC_FEATURES]].any().any()
    missing_peak = final.loc[
        final.cancer_id.eq(cancers[0]) & final.lncrna_id.eq("L2")
    ].iloc[0]
    assert not missing_peak["atac_promoter_mean__available"]
    assert audit["data_level"].endswith("NOT_PF")
