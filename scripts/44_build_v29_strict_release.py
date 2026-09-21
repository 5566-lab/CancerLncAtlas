#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.common import load_config, read_table, write_json, write_table

MODEL_DIRS = {"rgcn": "rgcn", "hgt": "hgt", "cc_hhgt_strict": "cc_hhgt_strict"}
KEYS_PATHWAY = ["candidate_id", "cancer_id", "lncrna_id", "pathway_family_id"]
KEYS_STATE = ["candidate_id", "cancer_id", "lncrna_id", "state_id"]


def collect(strict_root: Path, model_name: str, task: str, expected_runs: int | None = None) -> pd.DataFrame:
    paths = sorted((strict_root / MODEL_DIRS[model_name]).glob("LOCO_*/seed_*/prediction_%s_calibrated.parquet" % task))
    if not paths:
        raise RuntimeError(f"No calibrated {task} predictions for {model_name}")
    if expected_runs is not None and len(paths) != expected_runs:
        raise RuntimeError(
            f"Incomplete calibrated {task} matrix for {model_name}: "
            f"found {len(paths)}, expected {expected_runs}"
        )
    frames = []
    for path in paths:
        frame = read_table(path)
        frame = frame.loc[frame.split.astype(str).eq("test")].copy()
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No test rows for {model_name}/{task}")
    return pd.concat(frames, ignore_index=True)


def aggregate(frame: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    agg = frame.groupby(keys, observed=True).agg(
        probability=("calibrated_probability", "mean"),
        probability_sd=("calibrated_probability", "std"),
        direction_probability=("direction_probability", "mean"),
        uncertainty=("uncertainty", "mean"),
        n_seeds=("seed", "nunique"),
        label_class=("label_class", "first"),
        proxy_label=("proxy_label", "first"),
    ).reset_index()
    agg = agg.rename(columns={
        "probability": f"{prefix}_probability",
        "probability_sd": f"{prefix}_probability_sd",
        "direction_probability": f"{prefix}_direction_probability",
        "uncertainty": f"{prefix}_uncertainty",
        "n_seeds": f"{prefix}_n_seeds",
    })
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V2.9 strict OOF release for pathway and state tasks")
    parser.add_argument("--config", default="config/model_v2_9_state_graph.yaml")
    parser.add_argument("--strict-root", help="Override source v2_9_strict model root")
    parser.add_argument("--release-root", help="Override strict_release output directory")
    args = parser.parse_args()
    cfg = load_config(args.config)
    strict_root = Path(args.strict_root).resolve() if args.strict_root else cfg["_results"] / "v2_9_strict"
    release = Path(args.release_root).resolve() if args.release_root else cfg["_results"] / "strict_release"
    release.mkdir(parents=True, exist_ok=True)
    folds = read_table(cfg["_results"] / "tables" / "fold_manifest.tsv")
    expected_runs = int(folds.fold_id.astype(str).nunique()) * 3

    pathway_parts = []
    state_parts = []
    for model in MODEL_DIRS:
        pathway_parts.append(aggregate(collect(strict_root, model, "pathway", expected_runs), KEYS_PATHWAY, model))
        state_parts.append(aggregate(collect(strict_root, model, "state", expected_runs), KEYS_STATE, model))
    pathway = pathway_parts[0]
    for part in pathway_parts[1:]:
        pathway = pathway.merge(part.drop(columns=[c for c in ["label_class", "proxy_label"] if c in part]), on=KEYS_PATHWAY, how="outer")
    state = state_parts[0]
    for part in state_parts[1:]:
        state = state.merge(part.drop(columns=[c for c in ["label_class", "proxy_label"] if c in part]), on=KEYS_STATE, how="outer")

    # Backward-compatible strict adapter columns use CC-HHGT-Strict.
    pathway["cross_cancer_probability"] = pathway["cc_hhgt_strict_probability"]
    pathway["cross_cancer_probability_sd"] = pathway["cc_hhgt_strict_probability_sd"].fillna(0)
    pathway["direction_probability"] = pathway["cc_hhgt_strict_direction_probability"]
    pathway["cold_start"] = np.where(pathway["cross_cancer_probability"].notna(), "graph_available", "graph_unavailable")
    write_table(pathway, release / "strict_cross_cancer_oof_prediction.parquet")
    write_table(state, release / "strict_state_oof_prediction.parquet")

    payload = {
        "status": "COMPLETED",
        "analysis_version": cfg["analysis_version"],
        "pathway_rows": len(pathway),
        "state_rows": len(state),
        "pathway_cancers": int(pathway.cancer_id.nunique()),
        "state_cancers": int(state.cancer_id.nunique()),
        "state_ids": sorted(state.state_id.dropna().astype(str).unique()),
        "models": list(MODEL_DIRS),
        "expected_runs_per_model": expected_runs,
        "strict_root": str(strict_root),
        "release_root": str(release),
    }
    write_json(payload, release / "SUCCESS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
