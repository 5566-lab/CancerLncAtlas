#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from cc_hhgt.common import load_config, read_table, write_json, write_table
from cc_hhgt.stats import bh_fdr


def random_effects(effect: np.ndarray, n: np.ndarray):
    valid = np.isfinite(effect) & np.isfinite(n) & (n > 3) & (np.abs(effect) < 1)
    r = np.clip(effect[valid].astype(float), -0.999999, 0.999999)
    n = n[valid].astype(float)
    if not len(r):
        return (math.nan,) * 7 + (0,)
    z = np.arctanh(r); var = 1 / np.maximum(n - 3, 1); wf = 1 / var
    fixed = np.sum(wf * z) / np.sum(wf)
    q = float(np.sum(wf * (z - fixed) ** 2)); k = len(z)
    c = float(np.sum(wf) - np.sum(wf**2) / np.sum(wf))
    tau2 = max((q - (k - 1)) / c, 0) if k > 1 and c > 0 else 0.0
    w = 1 / (var + tau2); mean = float(np.sum(w * z) / np.sum(w)); se = float(np.sqrt(1 / np.sum(w)))
    p = float(2 * stats.norm.sf(abs(mean / se))) if se > 0 else math.nan
    lo, hi = math.tanh(mean - 1.96 * se), math.tanh(mean + 1.96 * se)
    i2 = max((q - (k - 1)) / q, 0) * 100 if q > 0 and k > 1 else 0.0
    return math.tanh(mean), lo, hi, p, tau2, i2, q, k


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pan-cancer lncRNA-RNAss random-effects meta-analysis")
    parser.add_argument("--config", default="config/model_v2_9_state_graph.yaml")
    parser.add_argument("--min-cancers", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config(args.config)
    release = cfg["_results"] / "v2_9_downstream" / "lncrna_state_release"
    frame = read_table(release / "rnass_state_significance_report.parquet")
    rows = []
    for lnc, group in frame.groupby("lncrna_id", observed=True):
        usable = group.loc[group.partial_rho.notna() & group.n_patients.gt(3)]
        if len(usable) < args.min_cancers:
            continue
        effect, lo, hi, p, tau2, i2, q, k = random_effects(usable.partial_rho.to_numpy(float), usable.n_patients.to_numpy(float))
        positive = int(usable.partial_rho.gt(0).sum()); negative = int(usable.partial_rho.lt(0).sum())
        probability = pd.to_numeric(usable.state_patient_probability, errors="coerce")
        weights = pd.to_numeric(usable.n_patients, errors="coerce").fillna(0).clip(lower=1)
        mean_prob = float(np.average(probability.fillna(probability.mean()).to_numpy(float), weights=weights)) if probability.notna().any() else math.nan
        rows.append({
            "lncrna_id": lnc,
            "state_id": "stemness_rna::RNAss",
            "pancancer_partial_rho": effect,
            "ci_lower": lo,
            "ci_upper": hi,
            "p_value": p,
            "tau2": tau2,
            "i2": i2,
            "q_statistic": q,
            "n_cancers": k,
            "n_positive_cancers": positive,
            "n_negative_cancers": negative,
            "direction_consistency": max(positive, negative) / max(k, 1),
            "mean_cancer_state_probability": mean_prob,
            "total_patients": int(usable.n_patients.sum()),
            "significant_cancer_count": int(usable.fdr.le(0.05).sum()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No pan-cancer RNAss associations met min-cancers")
    out["fdr"] = bh_fdr(out.p_value.to_numpy(float))
    out["direction"] = np.where(out.pancancer_partial_rho.ge(0), "positive", "negative")
    out["pancancer_rank_score"] = (
        0.5 * out.mean_cancer_state_probability.fillna(0.5)
        + 0.5 * (1.0 - out.fdr.fillna(1.0))
    )
    out["conclusion"] = np.select(
        [
            out.fdr.le(0.05) & out.direction_consistency.ge(0.75) & out.n_cancers.ge(10),
            out.fdr.le(0.10) & out.direction_consistency.ge(0.65),
        ],
        ["pan_cancer_robust", "pan_cancer_supported"],
        default="heterogeneous_or_exploratory",
    )
    gene = frame[[c for c in ["lncrna_id", "gene_symbol"] if c in frame]].drop_duplicates("lncrna_id")
    if "gene_symbol" in gene:
        out = out.merge(gene, on="lncrna_id", how="left")
    write_table(out.sort_values("pancancer_rank_score", ascending=False), release / "pancancer_lncrna_rnass_association.parquet")
    payload = {
        "status": "COMPLETED",
        "rows": len(out),
        "min_cancers": args.min_cancers,
        "robust": int(out.conclusion.eq("pan_cancer_robust").sum()),
    }
    write_json(payload, release / "PANCANCER_RNASS_SUCCESS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
