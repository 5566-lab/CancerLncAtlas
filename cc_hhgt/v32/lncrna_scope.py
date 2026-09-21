from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import dataframe_sha256


@dataclass(frozen=True)
class LncRnaScope:
    detection_by_cancer: pd.DataFrame
    summary: pd.DataFrame
    eligibility: pd.DataFrame
    source_universe_n: int
    shared_universe_n: int
    detection_sha256: str
    eligibility_sha256: str


def build_lncrna_scope(
    expression: pd.DataFrame,
    canonical_samples: pd.DataFrame,
    annotation_lncrnas: Iterable[str],
    *,
    value_column: str = "logcpm",
    within_cancer_min_detection_rate: float = 0.10,
    minimum_detected_cancers: int = 3,
    expected_source_n: int | None = 16889,
    expected_shared_n: int | None = 4712,
    enforce_expected_counts: bool = True,
) -> LncRnaScope:
    required_expression = {"cancer_id", "sample_id", "lncrna_id", value_column}
    required_samples = {"cancer_id", "sample_id"}
    if missing := sorted(required_expression - set(expression.columns)):
        raise ValueError(f"Expression table lacks columns: {missing}")
    if missing := sorted(required_samples - set(canonical_samples.columns)):
        raise ValueError(f"Canonical sample table lacks columns: {missing}")
    if not 0 < within_cancer_min_detection_rate <= 1:
        raise ValueError("within_cancer_min_detection_rate must be in (0, 1]")
    if minimum_detected_cancers < 1:
        raise ValueError("minimum_detected_cancers must be positive")

    samples = canonical_samples[["cancer_id", "sample_id"]].astype(str).drop_duplicates()
    if samples.sample_id.duplicated().any():
        raise RuntimeError("A canonical sample cannot belong to multiple cancers")
    annotation = pd.Index(sorted({str(value) for value in annotation_lncrnas}), name="lncrna_id")
    if annotation.empty:
        raise ValueError("The source lncRNA annotation universe is empty")
    source_n = int(len(annotation))
    if enforce_expected_counts and expected_source_n is not None and source_n != int(expected_source_n):
        raise RuntimeError(f"Source lncRNA universe is {source_n}, expected {expected_source_n}")

    values = expression.loc[:, list(required_expression)].copy()
    values[["cancer_id", "sample_id", "lncrna_id"]] = values[
        ["cancer_id", "sample_id", "lncrna_id"]
    ].astype(str)
    if values.duplicated(["cancer_id", "sample_id", "lncrna_id"]).any():
        raise RuntimeError("Expression contains duplicate cancer/sample/lncRNA rows")
    unknown_samples = values[["cancer_id", "sample_id"]].drop_duplicates().merge(
        samples, on=["cancer_id", "sample_id"], how="left", indicator=True
    )
    if unknown_samples._merge.ne("both").any():
        raise RuntimeError("Expression contains samples outside the canonical tumour manifest")
    unknown_lncrnas = sorted(set(values.lncrna_id) - set(annotation))
    if unknown_lncrnas:
        raise RuntimeError(f"Expression contains lncRNAs outside annotation: {unknown_lncrnas[:5]}")
    numeric = pd.to_numeric(values[value_column], errors="coerce")
    if numeric.isna().any():
        raise RuntimeError(f"{value_column} contains missing/non-numeric values")
    values["detected"] = numeric.gt(0)

    sample_counts = samples.groupby("cancer_id", observed=True).sample_id.nunique()
    detected = (
        values.loc[values.detected]
        .groupby(["cancer_id", "lncrna_id"], observed=True)
        .sample_id.nunique()
    )
    index = pd.MultiIndex.from_product(
        [sorted(samples.cancer_id.unique()), annotation], names=["cancer_id", "lncrna_id"]
    )
    detection = detected.reindex(index, fill_value=0).rename("n_detected_samples").reset_index()
    detection["n_canonical_samples"] = detection.cancer_id.map(sample_counts).astype(int)
    detection["detection_rate"] = detection.n_detected_samples / detection.n_canonical_samples
    detection["within_cancer_eligible"] = detection.detection_rate.ge(
        float(within_cancer_min_detection_rate)
    )

    summary = (
        detection.groupby("lncrna_id", as_index=False, observed=True)
        .agg(
            detected_cancers=("within_cancer_eligible", "sum"),
            max_detection_rate=("detection_rate", "max"),
            mean_detection_rate=("detection_rate", "mean"),
        )
        .sort_values("lncrna_id", kind="stable")
    )
    summary["pancancer_shared_eligible"] = summary.detected_cancers.ge(
        int(minimum_detected_cancers)
    )
    shared = set(summary.loc[summary.pancancer_shared_eligible, "lncrna_id"].astype(str))
    shared_n = len(shared)
    if enforce_expected_counts and expected_shared_n is not None and shared_n != int(expected_shared_n):
        raise RuntimeError(f"Shared lncRNA universe is {shared_n}, expected {expected_shared_n}")

    eligibility = detection.merge(
        summary[["lncrna_id", "detected_cancers", "pancancer_shared_eligible"]],
        on="lncrna_id",
        how="left",
        validate="many_to_one",
    )
    eligibility["shared_or_local_scope"] = np.select(
        [
            eligibility.within_cancer_eligible & eligibility.pancancer_shared_eligible,
            eligibility.within_cancer_eligible & ~eligibility.pancancer_shared_eligible,
        ],
        ["shared", "cancer_local"],
        default="not_expression_eligible",
    )
    eligibility["evaluated"] = True
    eligibility = eligibility.sort_values(["cancer_id", "lncrna_id"], kind="stable").reset_index(drop=True)
    detection = detection.sort_values(["cancer_id", "lncrna_id"], kind="stable").reset_index(drop=True)
    return LncRnaScope(
        detection_by_cancer=detection,
        summary=summary.reset_index(drop=True),
        eligibility=eligibility,
        source_universe_n=source_n,
        shared_universe_n=shared_n,
        detection_sha256=dataframe_sha256(detection, ["cancer_id", "lncrna_id"]),
        eligibility_sha256=dataframe_sha256(eligibility, ["cancer_id", "lncrna_id"]),
    )
