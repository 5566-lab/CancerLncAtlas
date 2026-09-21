from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from ..common import stable_id
from ..stats import correlation_p_values, design_rank
from .contracts import dataframe_sha256


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    result = np.full(pvalues.shape, np.nan, dtype=float)
    valid = np.isfinite(pvalues)
    if not valid.any():
        return result
    values = pvalues[valid]
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    unsorted = np.empty_like(adjusted)
    unsorted[order] = np.clip(adjusted, 0.0, 1.0)
    result[valid] = unsorted
    return result


def _covariate_matrix(covariates: pd.DataFrame | None, samples: list[str]) -> np.ndarray:
    if covariates is None or covariates.empty:
        return np.ones((len(samples), 1), dtype=float)
    if "sample_id" not in covariates:
        raise ValueError("Covariates require sample_id")
    frame = covariates.copy()
    frame["sample_id"] = frame.sample_id.astype(str)
    if frame.sample_id.duplicated().any():
        raise RuntimeError("Covariates contain duplicate sample IDs")
    frame = frame.set_index("sample_id").reindex(samples)
    if frame.isna().all(axis=None):
        raise RuntimeError("No covariates align to the discovery samples")
    numeric = frame.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors="coerce")
    categoricals = frame.drop(columns=numeric.columns).astype("string")
    design_parts = [np.ones((len(frame), 1), dtype=float)]
    if not numeric.empty:
        numeric = numeric.fillna(numeric.median()).fillna(0.0)
        scale = numeric.std(ddof=0).replace(0, 1.0)
        design_parts.append(((numeric - numeric.mean()) / scale).to_numpy(float))
    if not categoricals.empty:
        encoded = pd.get_dummies(categoricals.fillna("MISSING"), drop_first=True, dtype=float)
        if not encoded.empty:
            design_parts.append(encoded.to_numpy(float))
    return np.column_stack(design_parts)


def _residualize(matrix: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients = np.linalg.pinv(design) @ matrix
    return matrix - design @ coefficients


def _rank_and_standardize(frame: pd.DataFrame, design: np.ndarray) -> np.ndarray:
    ranked = frame.rank(axis=0, method="average", na_option="keep").to_numpy(float)
    if np.isnan(ranked).any():
        means = np.nanmean(ranked, axis=0)
        rows, columns = np.where(np.isnan(ranked))
        ranked[rows, columns] = means[columns]
    residual = _residualize(ranked, design)
    residual -= residual.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.square(residual).sum(axis=0, keepdims=True))
    return np.divide(residual, norm, out=np.zeros_like(residual), where=norm > 0)


def _association_blocks(
    lnc: pd.DataFrame,
    pathway: pd.DataFrame,
    design: np.ndarray,
    *,
    pathway_block_size: int,
) -> Iterator[tuple[list[str], list[str], np.ndarray, np.ndarray]]:
    lnc_values = _rank_and_standardize(lnc, design)
    n = len(lnc)
    residual_design_rank = design_rank(design)
    degrees = n - residual_design_rank - 1
    if degrees <= 0:
        raise RuntimeError(
            "No residual correlation degrees of freedom: "
            f"n={n}, design_rank={residual_design_rank}"
        )
    for start in range(0, pathway.shape[1], int(pathway_block_size)):
        block = pathway.iloc[:, start : start + int(pathway_block_size)]
        pathway_values = _rank_and_standardize(block, design)
        rho = np.clip(lnc_values.T @ pathway_values, -0.999999, 0.999999)
        pvalue = correlation_p_values(
            rho,
            n,
            residual_design_rank=residual_design_rank,
        )
        yield list(map(str, lnc.columns)), list(map(str, block.columns)), rho, pvalue


def label_associations(
    associations: pd.DataFrame,
    *,
    weak_max_fdr: float = 0.10,
    weak_min_abs_effect: float = 0.15,
    strong_max_fdr: float = 0.05,
    strong_min_abs_effect: float = 0.20,
) -> pd.DataFrame:
    frame = associations.copy()
    strong = frame.discovery_fdr.le(strong_max_fdr) & frame.discovery_effect.abs().ge(
        strong_min_abs_effect
    )
    weak = frame.discovery_fdr.le(weak_max_fdr) & frame.discovery_effect.abs().ge(
        weak_min_abs_effect
    )
    frame["label_class"] = np.select(
        [strong, weak], ["strong_positive", "weak_positive"], default="unlabeled"
    )
    frame["proxy_label"] = frame.label_class.isin(
        ["strong_positive", "weak_positive"]
    ).astype("int8")
    frame["association_direction"] = np.where(
        frame.discovery_effect.ge(0), "positive", "negative"
    )
    return frame


def compute_fold_associations(
    lnc_expression: pd.DataFrame,
    pathway_activity: pd.DataFrame,
    split_manifest: pd.DataFrame,
    pathway_hierarchy: pd.DataFrame,
    lnc_eligibility: pd.DataFrame,
    *,
    split: str = "train",
    covariates: pd.DataFrame | None = None,
    expression_value_column: str = "logcpm",
    activity_value_column: str = "pathway_activity_scaled",
    pathway_block_size: int = 128,
    weak_max_fdr: float = 0.10,
    weak_min_abs_effect: float = 0.15,
    strong_max_fdr: float = 0.05,
    strong_min_abs_effect: float = 0.20,
) -> pd.DataFrame:
    """Compute fold-local, covariate-adjusted association candidates.

    Only sample IDs assigned to ``split`` are read into the calculation.  The
    resulting hash is therefore invariant to changes in other patient splits.
    """

    required_lnc = {"cancer_id", "sample_id", "lncrna_id", expression_value_column}
    required_activity = {"sample_id", "pathway_id", activity_value_column}
    required_split = {"cancer_id", "sample_id", "split", "outer_fold"}
    required_hierarchy = {"pathway_id", "pathway_family_id"}
    required_eligibility = {"cancer_id", "lncrna_id", "within_cancer_eligible", "shared_or_local_scope"}
    for name, frame, required in (
        ("lnc expression", lnc_expression, required_lnc),
        ("pathway activity", pathway_activity, required_activity),
        ("split manifest", split_manifest, required_split),
        ("pathway hierarchy", pathway_hierarchy, required_hierarchy),
        ("lnc eligibility", lnc_eligibility, required_eligibility),
    ):
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"{name} lacks columns: {missing}")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    outer_values = split_manifest.outer_fold.astype(int).unique()
    if len(outer_values) != 1:
        raise RuntimeError("A fold association call must contain exactly one outer_fold")
    outer_fold = int(outer_values[0])

    hierarchy = pathway_hierarchy[["pathway_id", "pathway_family_id"]].astype(str).drop_duplicates()
    if hierarchy.pathway_id.duplicated().any():
        raise RuntimeError("An exact pathway maps to multiple pathway families")
    eligible = lnc_eligibility.loc[
        lnc_eligibility.within_cancer_eligible.astype(bool),
        ["cancer_id", "lncrna_id", "shared_or_local_scope", "detection_rate"],
    ].copy()
    eligible[["cancer_id", "lncrna_id"]] = eligible[["cancer_id", "lncrna_id"]].astype(str)
    sample_scope = split_manifest.loc[
        split_manifest.split.astype(str).eq(split), ["cancer_id", "sample_id"]
    ].astype(str)
    expression = lnc_expression.merge(
        sample_scope, on=["cancer_id", "sample_id"], how="inner", validate="many_to_one"
    )
    activity = pathway_activity.copy()
    activity["sample_id"] = activity.sample_id.astype(str)
    rows: list[pd.DataFrame] = []
    for cancer, cancer_samples in sample_scope.groupby("cancer_id", observed=True, sort=True):
        sample_ids = sorted(cancer_samples.sample_id.astype(str).unique())
        cancer_lnc = expression.loc[expression.cancer_id.astype(str).eq(str(cancer))]
        lnc_wide = cancer_lnc.pivot(index="sample_id", columns="lncrna_id", values=expression_value_column)
        pathway_wide = activity.loc[activity.sample_id.isin(sample_ids)].pivot(
            index="sample_id", columns="pathway_id", values=activity_value_column
        )
        common = sorted(set(sample_ids) & set(lnc_wide.index.astype(str)) & set(pathway_wide.index.astype(str)))
        if len(common) < 8:
            raise RuntimeError(f"{cancer}/{split} has only {len(common)} aligned samples")
        lnc_ids = set(
            eligible.loc[eligible.cancer_id.eq(str(cancer)), "lncrna_id"].astype(str)
        )
        lnc_columns = sorted(set(map(str, lnc_wide.columns)) & lnc_ids)
        pathway_columns = sorted(set(map(str, pathway_wide.columns)) & set(hierarchy.pathway_id))
        lnc_wide = lnc_wide.reindex(index=common, columns=lnc_columns)
        pathway_wide = pathway_wide.reindex(index=common, columns=pathway_columns)
        local_covariates = None
        if covariates is not None:
            local_covariates = covariates.loc[covariates.sample_id.astype(str).isin(common)]
        design = _covariate_matrix(local_covariates, common)
        residual_design_rank = design_rank(design)
        correlation_df = len(common) - residual_design_rank - 1
        if correlation_df <= 0:
            raise RuntimeError(
                f"No residual correlation degrees of freedom for {cancer}/{split}: "
                f"n={len(common)}, design_rank={residual_design_rank}"
            )
        cancer_parts: list[pd.DataFrame] = []
        for lnc_names, pathway_names, rho, pvalue in _association_blocks(
            lnc_wide, pathway_wide, design, pathway_block_size=pathway_block_size
        ):
            lnc_grid = np.repeat(np.asarray(lnc_names, dtype=object), len(pathway_names))
            pathway_grid = np.tile(np.asarray(pathway_names, dtype=object), len(lnc_names))
            cancer_parts.append(
                pd.DataFrame(
                    {
                        "cancer_id": str(cancer),
                        "lncrna_id": lnc_grid,
                        "pathway_id": pathway_grid,
                        "discovery_effect": rho.reshape(-1),
                        "discovery_pvalue": pvalue.reshape(-1),
                        "association_n_samples": len(common),
                        "residual_design_rank": residual_design_rank,
                        "correlation_df": correlation_df,
                    }
                )
            )
        if not cancer_parts:
            continue
        cancer_frame = pd.concat(cancer_parts, ignore_index=True)
        cancer_frame["discovery_fdr"] = benjamini_hochberg(
            cancer_frame.discovery_pvalue.to_numpy(float)
        )
        rows.append(cancer_frame)
    if not rows:
        raise RuntimeError(f"No associations were produced for split={split}")
    result = pd.concat(rows, ignore_index=True)
    result = result.merge(hierarchy, on="pathway_id", how="left", validate="many_to_one")
    result = result.merge(
        eligible,
        on=["cancer_id", "lncrna_id"],
        how="left",
        validate="many_to_one",
    )
    if result[["pathway_family_id", "shared_or_local_scope"]].isna().any().any():
        raise RuntimeError("Association rows lack hierarchy or lncRNA scope")
    result = label_associations(
        result,
        weak_max_fdr=weak_max_fdr,
        weak_min_abs_effect=weak_min_abs_effect,
        strong_max_fdr=strong_max_fdr,
        strong_min_abs_effect=strong_min_abs_effect,
    )
    result["outer_fold"] = outer_fold
    result["association_split"] = split
    result["candidate_id"] = [
        stable_id("V32CAND", cancer, lnc, pathway)
        for cancer, lnc, pathway in result[["cancer_id", "lncrna_id", "pathway_id"]].itertuples(index=False, name=None)
    ]
    result = result.sort_values(
        ["cancer_id", "lncrna_id", "pathway_id"], kind="stable"
    ).reset_index(drop=True)
    result.attrs["association_sha256"] = dataframe_sha256(
        result, ["cancer_id", "lncrna_id", "pathway_id"]
    )
    result.attrs["data_scope"] = f"outer_fold={outer_fold};split={split}"
    return result
