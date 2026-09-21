#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cc_hhgt.common import load_config, read_table, write_json, write_table
from cc_hhgt.stats import bh_fdr

KEYS = ["cancer_id", "patient_fold_id", "lncrna_id", "state_id"]


def random_effects_correlation(effect: np.ndarray, n: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(effect) & np.isfinite(n) & (n > 3) & (np.abs(effect) < 1)
    r = np.clip(effect[valid].astype(float), -0.999999, 0.999999)
    n = n[valid].astype(float)
    if not len(r):
        return {k: math.nan for k in ["effect", "ci_lower", "ci_upper", "p_value", "tau2", "i2"]} | {"k": 0}
    z = np.arctanh(r)
    variance = 1.0 / np.maximum(n - 3.0, 1.0)
    fixed_w = 1.0 / variance
    fixed_mean = np.sum(fixed_w * z) / np.sum(fixed_w)
    q = float(np.sum(fixed_w * (z - fixed_mean) ** 2))
    df = max(len(z) - 1, 1)
    c = float(np.sum(fixed_w) - np.sum(fixed_w**2) / np.sum(fixed_w))
    tau2 = max((q - (len(z) - 1)) / c, 0.0) if c > 0 and len(z) > 1 else 0.0
    w = 1.0 / (variance + tau2)
    mean = float(np.sum(w * z) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    zstat = mean / se if se > 0 else math.nan
    p = float(2 * stats.norm.sf(abs(zstat))) if np.isfinite(zstat) else math.nan
    lower = math.tanh(mean - 1.96 * se)
    upper = math.tanh(mean + 1.96 * se)
    i2 = max((q - (len(z) - 1)) / q, 0.0) * 100 if q > 0 and len(z) > 1 else 0.0
    return {
        "effect": math.tanh(mean),
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": p,
        "tau2": tau2,
        "i2": i2,
        "k": int(len(z)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cancer-specific V2.9 lncRNA-state significance reports")
    parser.add_argument("--config", default="config/model_v2_9_state_graph.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = cfg["_results"] / "v2_9_downstream"
    data_root = root / "lncrna_state_data"
    release = root / "lncrna_state_release"
    fold_pred = read_table(release / "lncrna_state_fold_prediction.parquet")
    data = read_table(data_root)
    use = data.merge(
        fold_pred[[c for c in KEYS + ["lncrna_state_probability", "state_direction_probability"] if c in fold_pred]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    rows = []
    for (cancer, lnc, state_id), group in use.groupby(["cancer_id", "lncrna_id", "state_id"], observed=True):
        meta = random_effects_correlation(group.test_effect.to_numpy(float), group.test_n_patients.to_numpy(float))
        finite_effect = group.test_effect.to_numpy(float)
        finite_effect = finite_effect[np.isfinite(finite_effect)]
        direction = np.sign(finite_effect)
        nonzero = direction[direction != 0]
        same_direction = int(max((nonzero > 0).sum(), (nonzero < 0).sum())) if len(nonzero) else 0
        n_folds = int(np.isfinite(group.test_effect).sum())
        prob = pd.to_numeric(group.lncrna_state_probability, errors="coerce")
        dir_prob = pd.to_numeric(group.state_direction_probability, errors="coerce")
        rows.append({
            "cancer_id": cancer,
            "lncrna_id": lnc,
            "state_id": state_id,
            "partial_rho": meta["effect"],
            "effect_beta": meta["effect"],
            "ci_lower": meta["ci_lower"],
            "ci_upper": meta["ci_upper"],
            "p_value": meta["p_value"],
            "tau2": meta["tau2"],
            "i2": meta["i2"],
            "n_folds_available": n_folds,
            "n_folds_same_direction": same_direction,
            "fold_selection_frequency": n_folds / 5.0,
            "n_patients": int(pd.to_numeric(group.test_n_patients, errors="coerce").fillna(0).sum()),
            "detection_rate": float(pd.to_numeric(group.train_detection_rate, errors="coerce").mean()),
            "state_patient_probability": float(prob.mean()) if prob.notna().any() else math.nan,
            "state_patient_probability_sd": float(prob.std()) if prob.notna().sum() > 1 else 0.0,
            "direction_probability": float(dir_prob.mean()) if dir_prob.notna().any() else math.nan,
            "predicted_effect": float(pd.to_numeric(group.test_effect, errors="coerce").mean()),
            "uncertainty": float(prob.std()) if prob.notna().sum() > 1 else math.nan,
        })
    report = pd.DataFrame(rows)
    report["fdr"] = report.groupby(["cancer_id", "state_id"], observed=True).p_value.transform(lambda x: bh_fdr(x.to_numpy(float)))
    report["replication_tier"] = np.select(
        [report.n_folds_same_direction.ge(4), report.n_folds_same_direction.ge(3)],
        ["robust_core", "replicated"],
        default="exploratory",
    )
    report["direction"] = np.where(report.partial_rho.ge(0), "positive", "negative")
    report["availability"] = np.where(report.n_folds_available.ge(1), "available", "unavailable")
    report["failure_reason"] = np.where(report.availability.eq("available"), pd.NA, "NO_HELDOUT_EFFECT")
    report["model_version"] = "CC-HHGT_v2.9-state-graph"
    report["conclusion"] = np.select(
        [
            report.fdr.le(0.05) & report.partial_rho.abs().ge(0.15) & report.n_folds_same_direction.ge(4),
            report.fdr.le(0.10) & report.partial_rho.abs().ge(0.12) & report.n_folds_same_direction.ge(3),
        ],
        ["significant_robust", "replicated_support"],
        default="exploratory_or_not_significant",
    )
    dim_path = Path("input_snapshot/processed/dimensions/dim_lncRNA.parquet")
    if dim_path.exists():
        dim = pd.read_parquet(dim_path, columns=["lncrna_id", "gene_symbol"]).drop_duplicates("lncrna_id")
        report = report.merge(dim, on="lncrna_id", how="left")
    out = release / "lncrna_state_significance_report.parquet"
    write_table(report, out)
    rnass = report.loc[report.state_id.astype(str).eq("stemness_rna::RNAss")].copy()
    write_table(rnass, release / "rnass_state_significance_report.parquet")
    payload = {
        "status": "COMPLETED",
        "rows": len(report),
        "rnass_rows": len(rnass),
        "states": sorted(report.state_id.astype(str).unique()),
        "cancers": int(report.cancer_id.nunique()),
    }
    write_json(payload, release / "STATE_SIGNIFICANCE_SUCCESS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
