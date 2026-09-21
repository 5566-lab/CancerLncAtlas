"""Outcome-free coexpression summaries for the V3.1 RNA residual."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import file_sha256, require_columns


ALLOWED_COEXPRESSION_METHOD = "covariate_residual_spearman_effective_df"
COEXPRESSION_FEATURES = (
    "log1p_coexpression_degree",
    "positive_edge_fraction",
    "mean_rho",
    "mean_absolute_rho",
    "maximum_absolute_rho",
)
FORBIDDEN_OUTCOME_COLUMNS = {
    "state_id",
    "effect",
    "fdr",
    "p_value",
    "proxy_label",
    "label_class",
    "state_value",
    "bulk_effect",
}


def summarize_coexpression_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Reduce eligible lncRNA--gene edges without using outcome statistics."""

    require_columns(edges, ["lncrna_id", "gene_id", "rho", "method"], "coexpression")
    overlap = sorted(FORBIDDEN_OUTCOME_COLUMNS.intersection(edges.columns))
    if overlap:
        raise RuntimeError(f"Outcome-derived columns entered coexpression context: {overlap}")
    methods = set(edges.method.dropna().astype(str).unique())
    if methods and methods != {ALLOWED_COEXPRESSION_METHOD}:
        raise RuntimeError(f"Unexpected coexpression methods: {sorted(methods)}")
    frame = edges.loc[:, ["lncrna_id", "rho"]].copy()
    frame["lncrna_id"] = frame.lncrna_id.astype(str)
    frame["rho"] = pd.to_numeric(frame.rho, errors="coerce")
    frame = frame.loc[np.isfinite(frame.rho.to_numpy(float))]
    if frame.empty:
        return pd.DataFrame(columns=["lncrna_id", *COEXPRESSION_FEATURES])
    frame["absolute_rho"] = frame.rho.abs()
    frame["positive"] = frame.rho.gt(0).astype(float)
    summary = frame.groupby("lncrna_id", observed=True, sort=False).agg(
        coexpression_degree=("rho", "size"),
        positive_edge_fraction=("positive", "mean"),
        mean_rho=("rho", "mean"),
        mean_absolute_rho=("absolute_rho", "mean"),
        maximum_absolute_rho=("absolute_rho", "max"),
    ).reset_index()
    summary["log1p_coexpression_degree"] = np.log1p(
        summary.pop("coexpression_degree").astype(float)
    )
    return summary.loc[:, ["lncrna_id", *COEXPRESSION_FEATURES]]


def augment_rna_context(
    base_context: pd.DataFrame,
    coexpression_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append masked coexpression summaries to a frozen RNA/OCLR context.

    A missing partition or a candidate without an eligible edge remains NaN
    with an explicit false availability mask.  It is never interpreted as a
    biological zero.
    """

    require_columns(base_context, ["cancer_id", "lncrna_id"], "RNA context")
    result = base_context.copy()
    result["cancer_id"] = result.cancer_id.astype(str)
    result["lncrna_id"] = result.lncrna_id.astype(str)
    if result.duplicated(["cancer_id", "lncrna_id"]).any():
        raise RuntimeError("RNA context contains duplicate cancer/lncRNA keys")
    collisions = [
        column
        for feature in COEXPRESSION_FEATURES
        for column in (feature, f"{feature}__available")
        if column in result
    ]
    if collisions:
        raise RuntimeError(f"RNA context already contains coexpression fields: {collisions}")

    parts: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for cancer, keys in result.groupby("cancer_id", observed=True, sort=True):
        path = Path(coexpression_root) / f"cancer_id={cancer}" / "part-0.parquet"
        part = keys.copy()
        if path.is_file():
            # The frozen edge table also carries p/FDR values describing the
            # expression--expression correlation itself.  They are not model
            # features and are deliberately never loaded here; the context is
            # computed from signed rho only.
            edges = pd.read_parquet(
                path, columns=["lncrna_id", "gene_id", "rho", "method"]
            )
            summary = summarize_coexpression_edges(edges)
            summary = summary.loc[summary.lncrna_id.isin(set(part.lncrna_id))]
            sources.append(
                {
                    "cancer_id": str(cancer),
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "status": "AVAILABLE",
                    "eligible_edge_rows": int(len(edges)),
                    "matched_lncrnas": int(summary.lncrna_id.nunique()),
                }
            )
            part = part.merge(summary, on="lncrna_id", how="left", validate="one_to_one")
        else:
            sources.append(
                {
                    "cancer_id": str(cancer),
                    "path": str(path.resolve()),
                    "size_bytes": 0,
                    "sha256": None,
                    "status": "UNAVAILABLE",
                    "eligible_edge_rows": 0,
                    "matched_lncrnas": 0,
                }
            )
            for feature in COEXPRESSION_FEATURES:
                part[feature] = np.nan
        for feature in COEXPRESSION_FEATURES:
            numeric = pd.to_numeric(part[feature], errors="coerce")
            available = np.isfinite(numeric.to_numpy(float))
            part[feature] = np.where(available, numeric, np.nan)
            part[f"{feature}__available"] = available
        parts.append(part)

    augmented = (
        pd.concat(parts, ignore_index=True, sort=False)
        .sort_values(["cancer_id", "lncrna_id"], kind="stable")
        .reset_index(drop=True)
    )
    audit = {
        "status": "PASS",
        "coexpression_method": ALLOWED_COEXPRESSION_METHOD,
        "features": list(COEXPRESSION_FEATURES),
        "sources": sources,
        "available_cancers": sorted(
            source["cancer_id"] for source in sources if source["status"] == "AVAILABLE"
        ),
        "unavailable_cancers": sorted(
            source["cancer_id"] for source in sources if source["status"] == "UNAVAILABLE"
        ),
        "outcome_derived_columns_used": 0,
        "missing_policy": "EXPLICIT_FALSE_MASK_AND_NAN_NEVER_ZERO",
    }
    return augmented, audit
