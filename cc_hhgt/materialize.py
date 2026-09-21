from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import read_table, stable_id, write_table


def classify_members(pred: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    m = cfg["materialization"]
    evidence = pred.observed_evidence_score.fillna(0.0)
    probability = pred.calibrated_probability.fillna(0.0)
    uncertainty = pred.uncertainty.fillna(1.0)
    conditions = [
        evidence >= m["observed_core_evidence_min"],
        (evidence > 0) & (probability >= m["model_supported_probability_min"]) & (uncertainty <= m["max_uncertainty"]),
        (evidence <= m["predicted_candidate_max_observed_evidence"]) & (probability >= m["predicted_candidate_probability_min"]) & (uncertainty <= m["max_uncertainty"]),
    ]
    # Predicted candidate is evaluated before model-supported for low-evidence high-probability rows.
    pred["member_class"] = np.select([conditions[0], conditions[2], conditions[1]], ["observed_core", "predicted_candidate", "model_supported"], default="not_selected")
    observed_direction = pred["direction"].astype(str) if "direction" in pred.columns else pd.Series("unknown", index=pred.index)
    pred["final_direction"] = np.where(observed_direction.isin(["positive", "negative"]), observed_direction, pred.predicted_direction)
    pred["member_weight"] = (0.55 * probability + 0.45 * evidence) * (1.0 - uncertainty)
    return pred


def materialize_genesets(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = cfg["_results"] / "tables" / "all_candidate_prediction"
    pred = read_table(root)
    pred = classify_members(pred, cfg)
    selected = pred.loc[pred.member_class != "not_selected"].copy()
    groups = []
    members = []
    for (cancer, family, direction), group in selected.groupby(["cancer_id", "pathway_family_id", "final_direction"], observed=True):
        group = group.sort_values(["member_weight", "calibrated_probability", "observed_evidence_score"], ascending=False)
        group = group.head(cfg["materialization"]["max_members"])
        if len(group) < cfg["materialization"]["min_members"]:
            continue
        geneset_id = stable_id("LGS", cancer, family, direction, cfg["analysis_version"])
        groups.append({
            "geneset_id": geneset_id,
            "geneset_name": f"{cancer}__{family}__{str(direction).upper()}",
            "geneset_type": "cancer_pathway_family",
            "cancer_id": cancer,
            "pathway_family_id": family,
            "direction": direction,
            "member_count": len(group),
            "n_observed_core": int((group.member_class == "observed_core").sum()),
            "n_model_supported": int((group.member_class == "model_supported").sum()),
            "n_predicted_candidate": int((group.member_class == "predicted_candidate").sum()),
            "analysis_version": cfg["analysis_version"],
        })
        group = group.copy()
        group["geneset_id"] = geneset_id
        group["rank"] = np.arange(1, len(group) + 1)
        members.append(group)
    master = pd.DataFrame(groups)
    member = pd.concat(members, ignore_index=True) if members else pd.DataFrame()
    write_table(master, cfg["_results"] / "tables" / "geneset_master_model.parquet")
    write_table(member, cfg["_results"] / "tables" / "geneset_member_model.parquet")

    def write_gmt(path: Path, include: list[str]):
        with path.open("w", encoding="utf-8") as handle:
            if member.empty:
                return
            filtered = member.loc[member.member_class.isin(include)]
            names = master.set_index("geneset_id").geneset_name.to_dict()
            for geneset_id, group in filtered.groupby("geneset_id", observed=True):
                genes = group.sort_values("rank").lncrna_id.astype(str).tolist()
                if len(genes) >= cfg["materialization"]["min_members"]:
                    handle.write("\t".join([names[geneset_id], "CancerLncAtlas CC-HHGT"] + genes) + "\n")
    write_gmt(cfg["_results"] / "model_genesets.gmt", cfg["materialization"]["default_gmt_include"])
    write_gmt(cfg["_results"] / "model_genesets_with_predicted_candidates.gmt", cfg["materialization"]["extended_gmt_include"])
    return master, member
