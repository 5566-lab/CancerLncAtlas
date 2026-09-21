from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import first_existing_column, input_path, read_table, stable_id, write_table
from .io import read_cancer_partition
from .stats import bh_fdr, correlation_p_values, design_rank, prepare_design, rank_transform, residualize, standardize
from .v30_integrity import atomic_write_json, canonical_json_sha256


def _atomic_table(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary, **kwargs)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _target_states(cfg: dict[str, Any]) -> list[str]:
    states = list(map(str, cfg.get("sample_contract", {}).get("target_states", [])))
    if not states:
        raise RuntimeError("V3.0 formal build requires sample_contract.target_states")
    return states


def _canonical(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = read_table(Path(cfg["_canonical_samples_path"]))
    states = read_table(Path(cfg["_canonical_state_complete_cases_path"]))
    eligibility = read_table(Path(cfg["_canonical_eligibility_path"]))
    if samples.duplicated(["cancer_id", "sample_id"]).any():
        raise RuntimeError("Canonical samples are not unique by cancer_id/sample_id")
    if not samples.is_tumor.astype(bool).all() or samples.sample_type_code.astype(int).isin(range(10, 20)).any():
        raise RuntimeError("A non-tumor sample entered the V3.0 formal sample universe")
    if states.state_value.isna().any():
        raise RuntimeError("State complete-case table contains a missing outcome")
    if states.duplicated(["cancer_id", "sample_id", "state_id"]).any():
        raise RuntimeError("State complete cases are not unique by cancer/sample/state")
    allowed = set(_target_states(cfg))
    observed = set(states.state_id.astype(str))
    if not allowed.issubset(observed):
        raise RuntimeError(f"Canonical state table is missing {sorted(allowed - observed)}")
    return samples, states.loc[states.state_id.astype(str).isin(allowed)].copy(), eligibility.loc[
        eligibility.state_id.astype(str).isin(allowed)
    ].copy()


def _matrix(
    cfg: dict[str, Any],
    key: str,
    cancer: str,
    entity: str,
    aliases: list[str],
    canonical_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    root = input_path(cfg, key)
    if root is None:
        raise RuntimeError(f"Input {key} is not configured")
    frame = read_cancer_partition(root, cancer)
    if frame.empty:
        return pd.DataFrame(), None
    frame["sample_id"] = frame.sample_id.astype(str)
    frame = frame.loc[frame.sample_id.isin(canonical_ids)].copy()
    if frame.empty:
        return pd.DataFrame(), None
    duplicate = frame.duplicated(["sample_id", entity], keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, ["sample_id", entity]].head(10).to_dict("records")
        raise RuntimeError(f"Duplicate exact sample/entity rows for {key}/{cancer}: {examples}")
    value = first_existing_column(frame, aliases)
    frame[value] = pd.to_numeric(frame[value], errors="coerce")
    matrix = frame.pivot(index="sample_id", columns=entity, values=value)
    detection = None
    if entity == "lncrna_id":
        if "tpm" in frame:
            detection_frame = frame[["sample_id", entity, "tpm"]].copy()
            detection_frame["detected"] = pd.to_numeric(detection_frame.tpm, errors="coerce").gt(0.1).astype(float)
            detection = detection_frame.pivot(index="sample_id", columns=entity, values="detected").fillna(0.0)
        else:
            detection = matrix.notna().astype(float)
    return matrix, detection


def _design(cfg: dict[str, Any], cancer: str, sample_ids: list[str]) -> np.ndarray:
    cov_path = input_path(cfg, "bulk_covariates")
    if cov_path is None:
        return np.ones((len(sample_ids), 1), dtype=float)
    cov = read_table(cov_path)
    cov = cov.loc[
        cov.cancer_id.astype(str).eq(str(cancer)) & cov.sample_id.astype(str).isin(sample_ids)
    ].copy()
    if cov.duplicated("sample_id").any():
        raise RuntimeError(f"Duplicate canonical covariate rows for {cancer}")
    cov = cov.set_index(cov.sample_id.astype(str)).reindex(sample_ids)
    columns = [column for column in cfg.get("state_graph", {}).get("covariates", []) if column in cov]
    return prepare_design(cov[columns].reset_index(drop=True)) if columns else np.ones((len(sample_ids), 1))


def _residual_feature(frame: pd.DataFrame, design: np.ndarray) -> np.ndarray:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    return standardize(residualize(rank_transform(numeric.to_numpy(float)), design))


def _association(
    cfg: dict[str, Any],
    cancer: str,
    state_id: str,
    feature_matrix: pd.DataFrame,
    state_cases: pd.DataFrame,
    canonical_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outcome = state_cases.loc[
        state_cases.cancer_id.astype(str).eq(cancer) & state_cases.state_id.astype(str).eq(state_id),
        ["sample_id", "state_value"],
    ].copy()
    if outcome.duplicated("sample_id").any() or outcome.state_value.isna().any():
        raise RuntimeError(f"Invalid complete-case outcome for {cancer}/{state_id}")
    outcome = outcome.set_index(outcome.sample_id.astype(str)).state_value.astype(float)
    sample_ids = sorted(set(feature_matrix.index.astype(str)) & set(outcome.index.astype(str)))
    minimum = int(cfg["sample_contract"]["minimum_observed_per_cancer_state"])
    if len(sample_ids) < minimum:
        return pd.DataFrame(), {
            "cancer_id": cancer,
            "state_id": state_id,
            "eligibility": "UNAVAILABLE",
            "unavailable_reason": f"N_OBSERVED_LT_{minimum}",
            "n_observed": len(sample_ids),
            "n_missing": canonical_count - len(sample_ids),
        }
    matrix = feature_matrix.reindex(sample_ids)
    variable = matrix.var(axis=0, ddof=1).replace([np.inf, -np.inf], np.nan).fillna(0).gt(0)
    matrix = matrix.loc[:, variable]
    if matrix.empty:
        return pd.DataFrame(), {
            "cancer_id": cancer,
            "state_id": state_id,
            "eligibility": "UNAVAILABLE",
            "unavailable_reason": "NO_VARIABLE_FEATURES",
            "n_observed": len(sample_ids),
            "n_missing": canonical_count - len(sample_ids),
        }
    design = _design(cfg, cancer, sample_ids)
    rank = design_rank(design)
    df = len(sample_ids) - rank - 1
    if df <= 0:
        raise RuntimeError(f"Non-positive residual df for {cancer}/{state_id}: n={len(sample_ids)}, rank={rank}")
    x = _residual_feature(matrix, design)
    y_frame = pd.DataFrame({state_id: outcome.reindex(sample_ids).to_numpy(float)}, index=sample_ids)
    if y_frame[state_id].isna().any():
        raise RuntimeError("State NaN reached association statistics")
    y = _residual_feature(y_frame, design)[:, 0]
    effect = x.T @ y / max(len(sample_ids) - 1, 1)
    p_value = correlation_p_values(effect, len(sample_ids), residual_design_rank=rank)
    table = pd.DataFrame(
        {
            "feature_id": matrix.columns.astype(str),
            "state_id": state_id,
            "effect": effect,
            "p_value": p_value,
            "fdr": bh_fdr(p_value, total_tests=len(p_value)),
        }
    )
    audit = {
        "cancer_id": cancer,
        "state_id": state_id,
        "eligibility": "ELIGIBLE",
        "unavailable_reason": None,
        "n_observed": len(sample_ids),
        "n_missing": canonical_count - len(sample_ids),
        "design_rank": rank,
        "residual_df": df,
        "fdr_family_size": len(table),
    }
    return table, audit


def _lineage(cfg: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    base = dict(cfg["_formal_lineage"])
    payload = {
        **base,
        "sample_universe_sha256": cfg["_sample_universe_sha256"],
        "eligibility": audit["eligibility"],
        "unavailable_reason": audit.get("unavailable_reason"),
        "evaluation_policy": "STATE_SPECIFIC_COMPLETE_CASE",
        "n_observed": int(audit["n_observed"]),
        "n_missing": int(audit["n_missing"]),
        "design_rank": audit.get("design_rank"),
        "residual_df": audit.get("residual_df"),
        "fdr_family_size": audit.get("fdr_family_size"),
    }
    payload["lineage_sha256"] = canonical_json_sha256(payload)
    return payload


def _decorate(frame: pd.DataFrame, cfg: dict[str, Any], audit: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for key, value in _lineage(cfg, audit).items():
        out[key] = value
    out["n_samples"] = int(audit["n_observed"])
    out["residual_design_rank"] = audit.get("design_rank")
    out["correlation_df"] = audit.get("residual_df")
    out["df"] = audit.get("residual_df")
    return out


def build_state_model_eligibility(
    cfg: dict[str, Any],
    canonical_eligibility: pd.DataFrame,
    candidate_root: Path,
    targets: list[str],
    all_cancers: list[str],
) -> pd.DataFrame:
    """Publish label-evaluation availability without changing label thresholds."""
    rows: list[dict[str, Any]] = []
    canonical_lookup = {
        (str(row.cancer_id), str(row.state_id)): row
        for row in canonical_eligibility.itertuples(index=False)
    }
    for cancer in all_cancers:
        candidate_path = candidate_root / f"cancer_id={cancer}" / "part-0.parquet"
        candidate = read_table(candidate_path) if candidate_path.exists() else pd.DataFrame()
        changed = False
        for state_id in targets:
            base = canonical_lookup[(cancer, state_id)]
            group = candidate.loc[candidate.state_id.astype(str).eq(state_id)] if not candidate.empty else pd.DataFrame()
            positive = int(group.proxy_label.astype(float).gt(0.5).sum()) if not group.empty else 0
            unlabeled = int(group.proxy_label.astype(float).le(0.5).sum()) if not group.empty else 0
            status = str(base.eligibility)
            reason = None if status == "ELIGIBLE" else str(base.unavailable_reason)
            minimum = int(cfg["sample_contract"]["minimum_observed_per_cancer_state"])
            if status == "UNAVAILABLE" and str(base.cancer_role) == "FORMAL" and int(base.n_observed) < minimum:
                reason = f"N_OBSERVED_LT_{minimum}"
            if status == "ELIGIBLE" and positive == 0:
                status, reason = "UNAVAILABLE", "NO_PROXY_POSITIVE"
            elif status == "ELIGIBLE" and unlabeled == 0:
                status, reason = "UNAVAILABLE", "NO_PROXY_UNLABELED"
            rows.append(
                {
                    "cancer_id": cancer,
                    "state_id": state_id,
                    "cancer_role": str(base.cancer_role),
                    "n_observed": int(base.n_observed),
                    "n_missing": int(base.n_missing),
                    "candidate_rows": int(len(group)),
                    "proxy_positive": positive,
                    "proxy_unlabeled": unlabeled,
                    "evaluation_eligibility": status,
                    "evaluation_unavailable_reason": reason,
                    "sample_universe_sha256": cfg["_sample_universe_sha256"],
                    **cfg["_formal_lineage"],
                }
            )
            if not group.empty:
                mask = candidate.state_id.astype(str).eq(state_id)
                candidate.loc[mask, "evaluation_eligibility"] = status
                candidate.loc[mask, "evaluation_unavailable_reason"] = reason
                changed = True
        if changed:
            _atomic_table(candidate, candidate_path)
    output = pd.DataFrame(rows)
    _atomic_table(output, cfg["_results"] / "tables" / "state_model_eligibility.tsv")
    return output


def build_v30_state_assets(cfg: dict[str, Any], cancers: Iterable[str] | None = None) -> dict[str, Any]:
    samples, state_cases, canonical_eligibility = _canonical(cfg)
    targets = _target_states(cfg)
    reference = set(map(str, cfg.get("cancer_scope", {}).get("reference_only", [])))
    all_cancers = sorted(samples.cancer_id.astype(str).unique())
    cancer_list = list(map(str, cancers)) if cancers else all_cancers
    tables = cfg["_results"] / "tables"
    target_roots = {
        "pathway": tables / "pathway_state_edge",
        "gene": tables / "gene_state_edge",
        "lnc": tables / "strict_state_candidate",
    }
    for root in target_roots.values():
        root.mkdir(parents=True, exist_ok=True)

    sample_universe_values = samples.sample_universe_sha256.astype(str).unique()
    if len(sample_universe_values) != 1:
        raise RuntimeError("Canonical samples carry multiple sample-universe hashes")
    cfg["_sample_universe_sha256"] = str(sample_universe_values[0])
    settings = cfg["state_graph"]
    audit_rows: list[dict[str, Any]] = []

    formal_state_cases = state_cases.loc[~state_cases.cancer_id.astype(str).isin(reference)].copy()
    catalog = (
        formal_state_cases.groupby("state_id", observed=True)
        .agg(n_values=("state_value", "size"), n_samples=("sample_id", "nunique"), n_cancers=("cancer_id", "nunique"), mean_value=("state_value", "mean"), sd_value=("state_value", "std"), min_value=("state_value", "min"), max_value=("state_value", "max"))
        .reset_index()
    )
    catalog["required_for_v3_0"] = True
    catalog["node_label"] = catalog.state_id
    catalog["analysis_version"] = cfg["analysis_version"]
    _atomic_table(catalog, tables / "state_catalog.parquet")

    for cancer in cancer_list:
        canonical_ids = set(samples.loc[samples.cancer_id.astype(str).eq(cancer), "sample_id"].astype(str))
        canonical_count = len(canonical_ids)
        if cancer in reference:
            for state_id in targets:
                hit = canonical_eligibility.loc[
                    canonical_eligibility.cancer_id.astype(str).eq(cancer)
                    & canonical_eligibility.state_id.astype(str).eq(state_id)
                ]
                if len(hit) != 1 or str(hit.iloc[0].eligibility) != "UNAVAILABLE":
                    raise RuntimeError(f"Reference-only eligibility is invalid for {cancer}/{state_id}")
                for feature_type in ("pathway", "gene", "lnc"):
                    audit_rows.append(
                        {
                            "cancer_id": cancer,
                            "state_id": state_id,
                            "feature_type": feature_type,
                            "cancer_role": "REFERENCE_ONLY",
                            "eligibility": "UNAVAILABLE",
                            "unavailable_reason": "REFERENCE_ONLY",
                            "n_observed": int(hit.iloc[0].n_observed),
                            "n_missing": int(hit.iloc[0].n_missing),
                        }
                    )
            continue
        pathway, _ = _matrix(cfg, "bulk_pathway_activity", cancer, "pathway_id", cfg["column_aliases"]["pathway_value"], canonical_ids)
        gene, _ = _matrix(cfg, "bulk_gene_expression", cancer, "gene_id", cfg["column_aliases"]["expression_value"], canonical_ids)
        lnc, detection = _matrix(cfg, "bulk_lnc_expression", cancer, "lncrna_id", cfg["column_aliases"]["expression_value"], canonical_ids)
        if set(pathway.index.astype(str)) - canonical_ids or set(gene.index.astype(str)) - canonical_ids or set(lnc.index.astype(str)) - canonical_ids:
            raise RuntimeError(f"Non-canonical samples entered feature matrices for {cancer}")
        if not gene.empty:
            gene_variance = gene.var(axis=0, ddof=1)
            gene_keep = gene_variance.loc[gene_variance.ge(float(settings["min_gene_variance"]))].sort_values(ascending=False).head(int(settings["max_genes_per_cancer"])).index
            gene = gene.loc[:, gene_keep]
        detection_rate = detection.mean(axis=0) if detection is not None else pd.Series(dtype=float)
        if not lnc.empty:
            lnc_keep = detection_rate.loc[detection_rate.ge(float(settings["candidate_min_detection"]))].index
            lnc_variance = lnc.loc[:, lnc_keep].var(axis=0, ddof=1).sort_values(ascending=False)
            lnc_keep = lnc_variance.head(int(settings["candidate_max_lncRNAs"])).index
            lnc = lnc.loc[:, lnc_keep]

        output_parts: dict[str, list[pd.DataFrame]] = {"pathway": [], "gene": [], "lnc": []}
        for state_id in targets:
            eligibility_hit = canonical_eligibility.loc[
                canonical_eligibility.cancer_id.astype(str).eq(cancer)
                & canonical_eligibility.state_id.astype(str).eq(state_id)
            ]
            if len(eligibility_hit) != 1:
                raise RuntimeError(f"Missing unique eligibility row for {cancer}/{state_id}")
            for feature_type, matrix in (("pathway", pathway), ("gene", gene), ("lnc", lnc)):
                table, audit = _association(cfg, cancer, state_id, matrix, state_cases, canonical_count)
                audit.update({"feature_type": feature_type, "cancer_role": "REFERENCE_ONLY" if cancer in reference else "FORMAL"})
                expected_status = str(eligibility_hit.iloc[0].eligibility)
                if expected_status == "UNAVAILABLE" and audit["eligibility"] != "UNAVAILABLE":
                    raise RuntimeError(f"Canonical eligibility disagreement for {cancer}/{state_id}")
                audit_rows.append(audit)
                if table.empty:
                    continue
                table = _decorate(table, cfg, audit)
                table["cancer_id"] = cancer
                table["direction"] = np.where(table.effect.ge(0), "positive", "negative")
                if feature_type == "pathway":
                    table = table.rename(columns={"feature_id": "pathway_id"})
                    table = table.loc[table.effect.abs().ge(float(settings["min_abs_pathway_effect"])) & table.fdr.le(float(settings["max_fdr"]))].copy()
                    table["edge_id"] = [stable_id("PSTATE30", cancer, item, state_id) for item in table.pathway_id]
                    table["method"] = "canonical_complete_case_covariate_residual_spearman"
                elif feature_type == "gene":
                    table = table.rename(columns={"feature_id": "gene_id"})
                    table = table.loc[table.effect.abs().ge(float(settings["min_abs_gene_effect"])) & table.fdr.le(float(settings["max_fdr"]))].copy()
                    table = table.assign(abs_effect=table.effect.abs()).sort_values(["direction", "abs_effect"], ascending=[True, False]).groupby("direction", observed=True).head(int(settings["max_gene_edges_per_state_direction"])).drop(columns="abs_effect")
                    table["edge_id"] = [stable_id("GSTATE30", cancer, item, state_id) for item in table.gene_id]
                    table["method"] = "canonical_complete_case_covariate_residual_spearman"
                else:
                    table = table.rename(columns={"feature_id": "lncrna_id"})
                    table["detection_rate"] = table.lncrna_id.map(detection_rate.to_dict()).astype(float)
                    strong = table.effect.abs().ge(float(settings["candidate_strong_abs_effect"])) & table.fdr.le(float(settings["candidate_strong_fdr"]))
                    weak = table.effect.abs().ge(float(settings["candidate_weak_abs_effect"])) & table.fdr.le(float(settings["candidate_weak_fdr"])) & ~strong
                    table["label_class"] = np.select([strong, weak], ["strong_positive", "weak_positive"], default="unlabeled")
                    table["proxy_label"] = table.label_class.ne("unlabeled").astype("int8")
                    known = table.proxy_label.eq(1)
                    table["direction"] = np.where(known, np.where(table.effect.ge(0), "positive", "negative"), "unknown")
                    table["direction_label"] = np.where(known, table.effect.ge(0).astype(float), np.nan)
                    table["evaluation_sampling_policy"] = "full_detectable_lncrna_state_universe"
                    table["candidate_id"] = [stable_id("STATEPAIR30", cancer, item, state_id) for item in table.lncrna_id]
                    table["label_source"] = "canonical_complete_case_covariate_residual_spearman"
                    table["method"] = "canonical_complete_case_covariate_residual_spearman"
                if not table.empty:
                    output_parts[feature_type].append(table)
        for feature_type, parts in output_parts.items():
            if parts:
                _atomic_table(pd.concat(parts, ignore_index=True), target_roots[feature_type] / f"cancer_id={cancer}" / "part-0.parquet")

    # Evaluation eligibility is stricter than outcome complete-case
    # eligibility: AUROC/calibration are undefined when the prespecified proxy
    # label has only one class.  Do not change thresholds to manufacture a
    # positive class; record UNAVAILABLE explicitly instead.
    model_eligibility = build_state_model_eligibility(
        cfg, canonical_eligibility, target_roots["lnc"], targets, all_cancers
    )

    pathway_all = read_table(target_roots["pathway"]) if any(target_roots["pathway"].rglob("*.parquet")) else pd.DataFrame()
    family_member = read_table(tables / "pathway_family_member.parquet")
    if not pathway_all.empty:
        merged = pathway_all.merge(family_member[["pathway_id", "pathway_family_id", "membership_weight"]], on="pathway_id", how="inner")
        merged["weighted_effect"] = merged.effect * merged.membership_weight
        family = merged.groupby(["cancer_id", "pathway_family_id", "state_id"], observed=True).agg(effect_sum=("weighted_effect", "sum"), weight_sum=("membership_weight", "sum"), min_fdr=("fdr", "min"), n_pathways=("pathway_id", "nunique"), n_observed=("n_observed", "max"), n_missing=("n_missing", "max"), design_rank=("design_rank", "max"), residual_df=("residual_df", "max"), sample_universe_sha256=("sample_universe_sha256", "first"), lineage_sha256=("lineage_sha256", "first")).reset_index()
        family["effect"] = family.effect_sum / family.weight_sum.clip(lower=1e-12)
        family["direction"] = np.where(family.effect.ge(0), "positive", "negative")
        family["edge_id"] = [stable_id("PFSTATE30", c, p, s) for c, p, s in zip(family.cancer_id, family.pathway_family_id, family.state_id)]
        family["method"] = "canonical_membership_weighted_pathway_state"
        family = family.drop(columns=["effect_sum", "weight_sum"])
        family_root = tables / "pathway_family_state_edge"
        for cancer, part in family.groupby("cancer_id", observed=True, sort=True):
            _atomic_table(
                part.reset_index(drop=True),
                family_root / f"cancer_id={cancer}" / "part-0.parquet",
            )

    cancer_state = formal_state_cases.groupby(["cancer_id", "state_id"], observed=True).agg(effect=("state_value", "mean"), state_sd=("state_value", "std"), n_observed=("sample_id", "nunique")).reset_index()
    cancer_state["direction"] = "requires_fold_localization"
    cancer_state["edge_id"] = [stable_id("CSTATE30", c, s) for c, s in zip(cancer_state.cancer_id, cancer_state.state_id)]
    cancer_state["method"] = "canonical_raw_cancer_mean_requires_fold_local_standardization"
    _atomic_table(cancer_state, tables / "cancer_state_edge.parquet")

    audit_frame = pd.DataFrame(audit_rows)
    _atomic_table(audit_frame, tables / "state_asset_complete_case_audit.tsv")
    summary = {
        "status": "PASS",
        "run_id": cfg["_formal_lineage"]["run_id"],
        "sample_universe_sha256": cfg["_sample_universe_sha256"],
        "target_states": targets,
        "cancers": len(cancer_list),
        "formal_cancers": len([c for c in cancer_list if c not in reference]),
        "reference_only": sorted(reference),
        "state_nan_used_in_statistics": 0,
        "state_outcome_imputation": "NONE",
        "audit_rows": len(audit_frame),
        "formal_evaluable_by_state": (
            model_eligibility.loc[model_eligibility.cancer_role.eq("FORMAL")]
            .groupby("state_id", observed=True)
            .evaluation_eligibility.apply(lambda values: int(values.eq("ELIGIBLE").sum()))
            .to_dict()
        ),
        "unavailable": audit_frame.loc[audit_frame.eligibility.eq("UNAVAILABLE"), ["cancer_id", "state_id", "feature_type", "unavailable_reason"]].to_dict("records"),
    }
    atomic_write_json(tables / "V3_STATE_ASSET_AUDIT.json", summary)
    return summary
