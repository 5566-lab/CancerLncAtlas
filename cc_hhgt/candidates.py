from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .common import LOGGER, input_path, read_table, stable_id, write_table
from .evidence import parquet_glob


def build_bulk_detection(cfg: dict[str, Any]) -> pd.DataFrame:
    path = input_path(cfg, "bulk_lnc_expression")
    out_path = cfg["_results"] / "tables" / "bulk_lnc_detection.parquet"
    if out_path.exists():
        return read_table(out_path)
    con = duckdb.connect()
    canonical_path = cfg.get("_canonical_samples_path")
    if canonical_path:
        canonical_sql = str(Path(canonical_path).resolve()).replace("'", "''")
        source = f"""
        SELECT expression.*
        FROM read_parquet('{parquet_glob(path)}', hive_partitioning=1) expression
        INNER JOIN read_parquet('{canonical_sql}') canonical
          ON expression.cancer_id = canonical.cancer_id
         AND expression.sample_id = canonical.sample_id
        WHERE canonical.is_tumor = TRUE
        """
    else:
        source = f"SELECT * FROM read_parquet('{parquet_glob(path)}', hive_partitioning=1)"
    sql = f"""
    SELECT cancer_id, lncrna_id,
           count(*) AS n_measurements,
           count(DISTINCT sample_id) AS n_samples,
           avg(CASE WHEN logcpm > 0 THEN 1.0 ELSE 0.0 END) AS bulk_detection_rate,
           median(logcpm) AS bulk_median_logcpm,
           avg(logcpm) AS bulk_mean_logcpm,
           var_samp(logcpm) AS bulk_variance
    FROM ({source})
    GROUP BY 1,2
    """
    out = con.execute(sql).df()
    con.close()
    write_table(out, out_path)
    return out


def build_sc_detection(cfg: dict[str, Any]) -> pd.DataFrame:
    path = cfg["_standardized"] / "sc_lnc_celltype_summary.parquet"
    out_path = cfg["_results"] / "tables" / "sc_lnc_detection.parquet"
    if out_path.exists():
        return read_table(out_path)
    if not path.exists():
        out = pd.DataFrame(columns=["cancer_id", "lncrna_id", "sc_detection_rate", "sc_mean_expression", "sc_n_cells"])
        write_table(out, out_path)
        return out
    x = read_table(path)
    malignant = x.malignant_status.astype(str).str.lower().eq("malignant") if "malignant_status" in x else pd.Series(True, index=x.index)
    x = x.loc[malignant]
    out = x.groupby(["cancer_id", "lncrna_id"], as_index=False, observed=True).agg(
        sc_detection_rate=("detection_rate", "max"),
        sc_mean_expression=("mean_log_expression", "max"),
        sc_n_cells=("n_cells", "sum"),
        sc_n_datasets=("dataset_id", "nunique"),
    )
    write_table(out, out_path)
    return out


def build_pancancer_lnc_eligibility(
    bulk: pd.DataFrame,
    formal_cancers: list[str],
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Freeze an outcome-free pan-cancer lncRNA eligibility universe.

    A lncRNA is eligible only when it reaches the configured within-cancer
    sample detection rate in at least ``minimum_detected_cancers`` formal
    cancers.  This is intentionally computed from bulk expression only;
    pathway labels, evidence labels, state outcomes and test metrics never
    participate in the filter.
    """
    policy = settings.get("pancancer_lnc_filter", {})
    enabled = bool(policy.get("enabled", False))
    minimum_cancers = int(policy.get("minimum_detected_cancers", 1))
    minimum_rate = float(
        policy.get("within_cancer_min_sample_detection_rate", settings["bulk_min_detection_rate"])
    )
    if minimum_cancers < 1:
        raise ValueError("minimum_detected_cancers must be at least one")
    formal = set(map(str, formal_cancers))
    scoped = bulk.loc[bulk.cancer_id.astype(str).isin(formal)].copy()
    scoped["bulk_detection_rate"] = pd.to_numeric(scoped.bulk_detection_rate, errors="coerce").fillna(0.0)
    detected = scoped.loc[scoped.bulk_detection_rate.ge(minimum_rate)]
    counts = detected.groupby("lncrna_id", observed=True).cancer_id.nunique()
    summary = (
        scoped.groupby("lncrna_id", as_index=False, observed=True)
        .agg(
            measured_cancers=("cancer_id", "nunique"),
            maximum_bulk_detection_rate=("bulk_detection_rate", "max"),
            mean_bulk_detection_rate=("bulk_detection_rate", "mean"),
        )
    )
    summary["detected_cancers"] = summary.lncrna_id.map(counts).fillna(0).astype(int)
    summary["minimum_detected_cancers"] = minimum_cancers
    summary["within_cancer_min_sample_detection_rate"] = minimum_rate
    summary["formal_cancer_count"] = len(formal)
    summary["eligible"] = True if not enabled else summary.detected_cancers.ge(minimum_cancers)
    summary["filter_enabled"] = enabled
    summary["filter_semantics"] = (
        "OUTCOME_FREE_BULK_EXPRESSION:within_cancer_detection_rate"
        f">={minimum_rate:g}_in_at_least_{minimum_cancers}_formal_cancers"
    )
    return summary.sort_values("lncrna_id", kind="stable").reset_index(drop=True)


def build_candidate_universe(cfg: dict[str, Any]) -> pd.DataFrame:
    from .pathway_target import EXACT_PATHWAY_TARGET, pathway_target_level

    if pathway_target_level(cfg) == EXACT_PATHWAY_TARGET:
        from .exact_pathway_candidates import build_exact_candidate_universe

        return build_exact_candidate_universe(cfg)

    settings = cfg["candidate_universe"]
    bulk = build_bulk_detection(cfg)
    sc = build_sc_detection(cfg)
    observed = read_table(cfg["_results"] / "tables" / "pair_evidence.parquet")
    families = read_table(cfg["_results"] / "tables" / "pathway_family.parquet")
    cancers = read_table(cfg["_standardized"] / "dim_cancer.parquet")
    reference = set(cfg["analysis_cancers"].get("reference_only", []))
    excluded = set(cfg["analysis_cancers"].get("exclude_from_training", []))
    formal_cancers = [
        cancer for cancer in cancers.cancer_id.astype(str)
        if cancer not in reference | excluded
    ]
    pan_eligibility = build_pancancer_lnc_eligibility(bulk, formal_cancers, settings)
    write_table(
        pan_eligibility,
        cfg["_results"] / "tables" / "pancancer_lncrna_eligibility.parquet",
    )
    eligible_lnc = set(
        pan_eligibility.loc[pan_eligibility.eligible.astype(bool), "lncrna_id"].astype(str)
    )
    if settings.get("pancancer_lnc_filter", {}).get("enabled", False) and not eligible_lnc:
        raise RuntimeError("Pan-cancer lncRNA filter removed the complete candidate universe")

    detection = bulk.merge(sc, on=["cancer_id", "lncrna_id"], how="outer")
    detection["bulk_detection_rate"] = detection.bulk_detection_rate.fillna(0.0)
    detection["sc_detection_rate"] = detection.sc_detection_rate.fillna(0.0)
    detection["bulk_detected"] = detection.bulk_detection_rate >= settings["bulk_min_detection_rate"]
    detection["sc_detected"] = detection.sc_detection_rate >= settings["sc_min_detection_rate"]
    detection["candidate_detection_score"] = np.maximum(detection.bulk_detection_rate, detection.sc_detection_rate)
    detection["observed_pair_count"] = detection.merge(observed.groupby(["cancer_id", "lncrna_id"]).size().rename("n").reset_index(), on=["cancer_id", "lncrna_id"], how="left").n.fillna(0).to_numpy()
    detection = detection.loc[
        (detection.bulk_detected | detection.sc_detected | (detection.observed_pair_count > 0))
        & detection.lncrna_id.astype(str).isin(eligible_lnc)
    ]
    detection = detection.merge(
        pan_eligibility[["lncrna_id", "detected_cancers", "formal_cancer_count", "filter_semantics"]],
        on="lncrna_id",
        how="left",
        validate="many_to_one",
    )

    out_root = cfg["_results"] / "tables" / "candidate_universe"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    all_summary = []
    family_ids = families.pathway_family_id.astype(str).to_numpy()
    for cancer in cancers.cancer_id.astype(str):
        outfile = out_root / f"cancer_id={cancer}" / "part-0.parquet"
        if outfile.exists():
            part = pd.read_parquet(outfile)
            manifest_rows.append({"cancer_id": cancer, "n_candidates": len(part), "status": "CACHED", "path": str(outfile)})
            continue
        lnc = detection.loc[detection.cancer_id.astype(str) == cancer].copy()
        observed_lnc = set(observed.loc[observed.cancer_id.astype(str) == cancer, "lncrna_id"].astype(str))
        lnc["is_observed_lnc"] = lnc.lncrna_id.astype(str).isin(observed_lnc)
        lnc = lnc.sort_values(["is_observed_lnc", "candidate_detection_score", "observed_pair_count"], ascending=False)
        base = lnc.head(settings["max_lncRNAs_per_cancer"])
        if settings.get("include_all_observed_pairs", True):
            base = pd.concat([base, lnc.loc[lnc.is_observed_lnc]], ignore_index=True).drop_duplicates("lncrna_id")
        if base.empty:
            manifest_rows.append({"cancer_id": cancer, "n_candidates": 0, "status": "NO_LNCRNA", "path": str(outfile)})
            continue
        n = len(base) * len(family_ids)
        candidate = pd.DataFrame({
            "cancer_id": np.repeat(cancer, n),
            "lncrna_id": np.repeat(base.lncrna_id.astype(str).to_numpy(), len(family_ids)),
            "pathway_family_id": np.tile(family_ids, len(base)),
            "bulk_detection_rate": np.repeat(base.bulk_detection_rate.to_numpy(float), len(family_ids)),
            "sc_detection_rate": np.repeat(base.sc_detection_rate.to_numpy(float), len(family_ids)),
            "pancancer_detected_cancers": np.repeat(base.detected_cancers.to_numpy(int), len(family_ids)),
            "pancancer_formal_cancer_count": np.repeat(base.formal_cancer_count.to_numpy(int), len(family_ids)),
            "pancancer_filter_semantics": np.repeat(base.filter_semantics.astype(str).to_numpy(), len(family_ids)),
            "analysis_tier": "reference_only" if cancer in reference else "primary",
        })
        ev = observed.loc[observed.cancer_id.astype(str) == cancer]
        candidate = candidate.merge(ev, on=["cancer_id", "lncrna_id", "pathway_family_id"], how="left", suffixes=("", "_ev"))
        for col in cfg["training"]["feature_columns"]:
            if col not in candidate.columns:
                candidate[col] = 0.0
            candidate[col] = pd.to_numeric(candidate[col], errors="coerce").fillna(0.0)
        candidate["label_class"] = candidate.label_class.fillna("unlabeled") if "label_class" in candidate else "unlabeled"
        candidate["association_proxy_label"] = candidate.label_class.isin(
            ["strong_positive", "weak_positive"]
        ).astype(np.int8)
        candidate["strong_evidence_label"] = candidate.label_class.eq("strong_positive").astype(np.int8)
        candidate["label"] = candidate.strong_evidence_label
        candidate["label_semantics"] = "PATHWAY_STRONG_EVIDENCE_V1:strong_evidence_label"
        candidate["sample_weight"] = candidate.sample_weight.fillna(cfg["training"]["unlabeled_weight"]) if "sample_weight" in candidate else cfg["training"]["unlabeled_weight"]
        candidate["candidate_id"] = [stable_id("CAND", cancer, l, f) for l, f in zip(candidate.lncrna_id, candidate.pathway_family_id)]
        if cfg.get("_sample_universe_sha256"):
            candidate["sample_universe_sha256"] = cfg["_sample_universe_sha256"]
        candidate["evaluation_policy"] = "FROZEN_CANONICAL_SAMPLE_UNIVERSE"
        for key, value in cfg.get("_formal_lineage", {}).items():
            candidate[key] = value
        write_table(candidate, outfile)
        manifest_rows.append({
            "cancer_id": cancer,
            "n_lncRNAs": len(base),
            "n_pancancer_eligible_lncRNAs": len(eligible_lnc),
            "n_families": len(family_ids),
            "n_candidates": len(candidate),
            "n_observed_pairs": len(ev),
            "status": "PASS",
            "path": str(outfile),
        })
    manifest = pd.DataFrame(manifest_rows)
    write_table(manifest, cfg["_results"] / "tables" / "candidate_universe_manifest.tsv")
    return manifest
