#!/usr/bin/env python3
"""Resume selected LOCO models without sharing a stage-status file.

This runner is intended for disjoint worker fold lists on CPU hosts.  It
validates a complete artifact set before skipping a fold and persists worker
status after every model so interrupted runs can be resumed safely.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

from cc_hhgt.baselines import train_baseline_fold
from cc_hhgt.common import configure_logging, load_config, read_table, utc_now, write_table
from cc_hhgt.gnn import train_gnn_fold


GNN_MODELS = {"rgcn", "hgt", "cc_hhgt"}


def expected_artifacts(model_name: str) -> list[str]:
    common = ["prediction_raw.parquet", "metrics.tsv", "metadata.json"]
    if model_name in GNN_MODELS:
        return ["best.pt", "training_history.tsv", *common]
    return ["model.joblib", *common]


def is_complete(model_dir: Path, model_name: str) -> bool:
    return all((model_dir / name).exists() and (model_dir / name).stat().st_size > 0 for name in expected_artifacts(model_name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--status-output", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    cfg = load_config(args.config)
    folds = read_table(cfg["_results"] / "tables" / "fold_manifest.tsv")
    requested = set(args.folds)
    folds = folds.loc[folds.fold_id.astype(str).isin(requested)]
    missing = requested - set(folds.fold_id.astype(str))
    if missing:
        raise ValueError(f"Unknown folds: {sorted(missing)}")
    unknown_models = set(args.models) - (GNN_MODELS | {"logistic", "hist_gradient_boosting"})
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")

    status_path = Path(args.status_output)
    rows: list[dict[str, object]] = []
    if status_path.exists():
        rows = read_table(status_path).to_dict(orient="records")

    for fold in folds.itertuples(index=False):
        fold_row = pd.Series(fold._asdict())
        for model_name in args.models:
            model_dir = cfg["_results"] / "models" / model_name / str(fold.fold_id)
            started_at = utc_now()
            row = {
                "fold_id": str(fold.fold_id),
                "test_cancer": str(fold.test_cancer),
                "model_name": model_name,
                "started_at": started_at,
                "finished_at": None,
                "status": "RUNNING",
                "error": None,
            }
            rows = [
                old
                for old in rows
                if not (str(old.get("fold_id")) == str(fold.fold_id) and str(old.get("model_name")) == model_name)
            ]
            rows.append(row)
            write_table(pd.DataFrame(rows), status_path)
            try:
                if is_complete(model_dir, model_name) and not args.force:
                    row["status"] = "SKIPPED_COMPLETE"
                elif model_name in GNN_MODELS:
                    train_gnn_fold(cfg, fold_row, model_name)
                    row["status"] = "SUCCESS"
                else:
                    train_baseline_fold(cfg, fold_row, model_name)
                    row["status"] = "SUCCESS"
            except Exception as exc:
                row["status"] = "FAILED"
                row["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                if args.fail_fast:
                    row["finished_at"] = utc_now()
                    write_table(pd.DataFrame(rows), status_path)
                    raise
            row["finished_at"] = utc_now()
            write_table(pd.DataFrame(rows), status_path)


if __name__ == "__main__":
    main()
