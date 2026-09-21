"""Audited full-scope ATAC context helpers for the V3.1 LOCO pilot."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import require_columns


ATAC_FEATURES = (
    "atac_promoter_peak_count",
    "atac_promoter_mean",
    "atac_promoter_sd_across_samples",
    "atac_promoter_q90",
    "atac_promoter_positive_fraction",
    "atac_distal_peak_count",
    "atac_distal_mean",
    "atac_distal_sd_across_samples",
    "atac_distal_q90",
    "atac_distal_positive_fraction",
    "atac_sample_count",
)


def derive_full_scope_r_script(source: str) -> tuple[str, dict[str, Any]]:
    """Apply three exact, auditable scope substitutions to the frozen R code."""

    replacements = (
        ('cancers <- c("BRCA", "COAD", "KIRP")', '# cancers resolved from registered keys and official mapping below'),
        (
            'stopifnot(identical(sort(unique(candidate$cancer_id)), cancers))',
            'registered_cancers <- sort(unique(candidate$cancer_id))\n'
            'if (length(registered_cancers) != 31L) stop("expected exact 31-cancer registered scope")',
        ),
        (
            'mapping[, cancer_id := sub("-.*$", "", bam_prefix)]',
            'mapping[, cancer_id := sub("-.*$", "", bam_prefix)]\n'
            'mapping[, cancer_id := sub("x$", "", cancer_id)]\n'
            'cancers <- intersect(registered_cancers, sort(unique(mapping$cancer_id)))\n'
            'if (length(cancers) < 1L) stop("no registered cancer has official ATAC mapping")',
        ),
        (
            '  cancers = cancer_audit',
            '  registered_cancers = registered_cancers,\n'
            '  atac_covered_cancers = cancers,\n'
            '  atac_unavailable_cancers = setdiff(registered_cancers, cancers),\n'
            '  cancers = cancer_audit',
        ),
    )
    result = source
    counts: list[dict[str, Any]] = []
    for old, new in replacements:
        count = result.count(old)
        if count != 1:
            raise RuntimeError(
                f"Frozen ATAC R source contract drift for replacement {old!r}: count={count}"
            )
        result = result.replace(old, new, 1)
        counts.append({"source_fragment": old, "replacement_count": count})
    return result, {
        "status": "PASS",
        "algorithm_changed": False,
        "scope_changes_only": True,
        "exact_replacements": counts,
        "mapping_normalization": "remove historical trailing x from ACCx/GBMx/LGGx",
    }


def finalize_atac_context(
    candidate_keys: pd.DataFrame,
    covered_features: pd.DataFrame,
    *,
    covered_cancers: list[str] | tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pad official ATAC summaries to all registered candidates with masks."""

    keys = candidate_keys.loc[:, ["cancer_id", "lncrna_id"]].copy()
    keys["cancer_id"] = keys.cancer_id.astype(str)
    keys["lncrna_id"] = keys.lncrna_id.astype(str)
    keys = keys.drop_duplicates().sort_values(
        ["cancer_id", "lncrna_id"], kind="stable"
    ).reset_index(drop=True)
    if keys.cancer_id.nunique() != 31:
        raise RuntimeError("ATAC context requires exact 31-cancer candidate scope")
    if keys.duplicated(["cancer_id", "lncrna_id"]).any():
        raise RuntimeError("ATAC candidate keys are duplicated")
    require_columns(
        covered_features,
        ["cancer_id", "lncrna_id", *ATAC_FEATURES],
        "covered ATAC features",
    )
    features = covered_features.loc[:, ["cancer_id", "lncrna_id", *ATAC_FEATURES]].copy()
    features["cancer_id"] = features.cancer_id.astype(str)
    features["lncrna_id"] = features.lncrna_id.astype(str)
    if features.duplicated(["cancer_id", "lncrna_id"]).any():
        raise RuntimeError("Covered ATAC features contain duplicate keys")
    covered = set(map(str, covered_cancers))
    if set(features.cancer_id.unique()) != covered:
        raise RuntimeError("Covered ATAC feature cancer list disagrees with R audit")
    if not covered.issubset(set(keys.cancer_id.unique())):
        raise RuntimeError("ATAC mapping contains cancers outside registered scope")
    unexpected = features.merge(keys, on=["cancer_id", "lncrna_id"], how="left", indicator=True)
    if not unexpected._merge.eq("both").all():
        raise RuntimeError("ATAC output contains unregistered candidate keys")
    expected_covered = keys.loc[keys.cancer_id.isin(covered)]
    signature = expected_covered.merge(
        features[["cancer_id", "lncrna_id"]],
        on=["cancer_id", "lncrna_id"],
        how="left",
        indicator=True,
    )
    if not signature._merge.eq("both").all():
        raise RuntimeError("ATAC R output did not retain every covered-cancer candidate")

    final = keys.merge(
        features, on=["cancer_id", "lncrna_id"], how="left", validate="one_to_one"
    )
    for feature in ATAC_FEATURES:
        numeric = pd.to_numeric(final[feature], errors="coerce")
        available = np.isfinite(numeric.to_numpy(float)) & final.cancer_id.isin(covered)
        final[feature] = np.where(available, numeric, np.nan)
        final[f"{feature}__available"] = available
    unavailable = ~final.cancer_id.isin(covered)
    if final.loc[unavailable, list(ATAC_FEATURES)].notna().any().any():
        raise RuntimeError("Unavailable ATAC cancer contains feature values")
    if final.loc[
        unavailable, [f"{feature}__available" for feature in ATAC_FEATURES]
    ].any().any():
        raise RuntimeError("Unavailable ATAC cancer contains true masks")
    audit = {
        "status": "PASS",
        "registered_cancers": sorted(keys.cancer_id.unique()),
        "atac_covered_cancers": sorted(covered),
        "atac_unavailable_cancers": sorted(set(keys.cancer_id.unique()) - covered),
        "rows": int(len(final)),
        "covered_rows": int((~unavailable).sum()),
        "unavailable_rows": int(unavailable.sum()),
        "feature_columns": list(ATAC_FEATURES),
        "missing_policy": "EXPLICIT_FALSE_MASK_AND_NAN_NEVER_ZERO",
        "data_level": "CANCER_LEVEL_AGGREGATE_FOR_LOCO_ONLY_NOT_PF",
        "outcome_derived_columns_used": 0,
    }
    return final, audit
