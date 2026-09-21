#!/usr/bin/env python3
"""Train resumable lncRNA-state patient and frozen-graph decoder heads."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cc_hhgt_v26.state_model import (  # noqa: E402
    SEEDS,
    STATE_DATA,
    STATE_MODEL_ROOT,
    STATE_RESULT_ROOT,
    train_state_task,
)

RUN = Path(os.getenv("CC_HHGT_STATE_RUN_ROOT", str(ROOT / "results" / "lncrna_state_run")))
MANIFEST = RUN / "task_manifest.tsv"


def initial_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(STATE_DATA.glob("cancer_id=*/part-0.parquet")):
        cancer = path.parent.name.split("=", 1)[1]
        folds = sorted(pd.read_parquet(path, columns=["patient_fold_id"])["patient_fold_id"].astype(str).unique())
        for fold in folds:
            for seed in SEEDS:
                result = STATE_RESULT_ROOT / cancer / fold / f"seed_{seed}"
                model = STATE_MODEL_ROOT / cancer / fold / f"seed_{seed}" / "best.pt"
                prediction = result / "prediction.parquet"
                required = {"state_patient_probability", "state_direction_probability", "test_membership_label"}
                complete = prediction.exists() and required.issubset(set(pq.read_schema(prediction).names)) and model.exists() and (result / "metrics.tsv").exists()
                rows.append({
                    "cancer_id": cancer,
                    "patient_fold_id": fold,
                    "seed": seed,
                    "status": "COMPLETED" if complete else "PENDING",
                    "started_at": pd.NA,
                    "finished_at": pd.NA,
                    "error": pd.NA,
                })
    return pd.DataFrame(rows)


def write_manifest(frame: pd.DataFrame) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MANIFEST, sep="\t", index=False)
    (RUN / "run_status.json").write_text(json.dumps({
        "updated_at": datetime.now().isoformat(),
        "counts": frame["status"].value_counts().to_dict(),
        "n_tasks": len(frame),
        "graph_encoder_retrained": False,
    }, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()
    if not (STATE_DATA / "SUCCESS.json").exists():
        raise RuntimeError("lncRNA-state data SUCCESS.json missing")
    frame = initial_manifest()
    write_manifest(frame)
    for index, row in frame.iterrows():
        if row["status"] == "COMPLETED":
            print(f"[skip] {row.cancer_id} {row.patient_fold_id} {row.seed}", flush=True)
            continue
        frame.loc[index, ["status", "started_at"]] = ["RUNNING", datetime.now().isoformat()]
        write_manifest(frame)
        try:
            train_state_task(str(row.cancer_id), str(row.patient_fold_id), int(row.seed), args.epochs, args.patience)
        except Exception:
            error = traceback.format_exc()
            frame.loc[index, ["status", "finished_at", "error"]] = ["FAILED", datetime.now().isoformat(), error]
            write_manifest(frame)
            (RUN / "FAILED.txt").write_text(error, encoding="utf-8")
            raise
        frame.loc[index, ["status", "finished_at", "error"]] = ["COMPLETED", datetime.now().isoformat(), pd.NA]
        write_manifest(frame)
    success = {"status": "COMPLETED", "completed_at": datetime.now().isoformat(), "n_tasks": len(frame), "graph_encoder_retrained": False}
    (RUN / "SUCCESS.json").write_text(json.dumps(success, indent=2), encoding="utf-8")
    print(json.dumps(success, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
