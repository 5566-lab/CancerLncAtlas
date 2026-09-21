"""Fail-closed resumable patient-fold adapter multiseed runner."""

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

from cc_hhgt_v26.adapter_model import (  # noqa: E402
    ADAPTER_DATA,
    MODEL_ROOT,
    RESULT_ROOT,
    SEEDS,
    train_task,
)


RUN_ROOT = Path(os.getenv("CC_HHGT_ADAPTER_RUN_ROOT", str(ROOT / "results" / "cancer_adapter_run")))
MANIFEST = RUN_ROOT / "task_manifest.tsv"


def initial_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(ADAPTER_DATA.glob("cancer_id=*/part-0.parquet")):
        cancer = path.parent.name.split("=", 1)[1]
        folds = pd.read_parquet(
            path, columns=["patient_fold_id"]
        )["patient_fold_id"].unique()
        for fold in sorted(folds):
            for seed in SEEDS:
                result = RESULT_ROOT / cancer / fold / f"seed_{seed}"
                model = MODEL_ROOT / cancer / fold / f"seed_{seed}" / "best.pt"
                prediction = result / "prediction.parquet"
                required_prediction_columns = {
                    "cancer_native_probability",
                    "cancer_joint_probability",
                    "patient_gate_native_weight",
                    "patient_gate_joint_weight",
                    "strict_available",
                }
                prediction_compatible = (
                    prediction.exists()
                    and required_prediction_columns.issubset(
                        set(pq.read_schema(prediction).names)
                    )
                )
                complete = (
                    prediction_compatible
                    and (result / "metrics.tsv").exists()
                    and model.exists()
                )
                rows.append(
                    {
                        "cancer_id": cancer,
                        "patient_fold_id": fold,
                        "seed": seed,
                        "status": "COMPLETED" if complete else "PENDING",
                        "started_at": pd.NA,
                        "finished_at": pd.NA,
                        "error": pd.NA,
                    }
                )
    return pd.DataFrame(rows)


def write_manifest(frame: pd.DataFrame) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MANIFEST, sep="\t", index=False)
    status = {
        "updated_at": datetime.now().isoformat(),
        "counts": frame["status"].value_counts().to_dict(),
        "n_tasks": int(len(frame)),
        "fail_closed": True,
        "model_fallback": False,
    }
    (RUN_ROOT / "run_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    if not (ADAPTER_DATA / "SUCCESS.json").exists():
        raise RuntimeError("adapter data SUCCESS.json is absent")
    frame = initial_manifest()
    write_manifest(frame)
    attribution_rows = []
    for index, row in frame.iterrows():
        if row["status"] == "COMPLETED":
            print(
                f"[skip] {row.cancer_id} {row.patient_fold_id} {row.seed}",
                flush=True,
            )
            continue
        frame.loc[index, ["status", "started_at"]] = [
            "RUNNING",
            datetime.now().isoformat(),
        ]
        write_manifest(frame)
        print(
            f"[start] {row.cancer_id} {row.patient_fold_id} seed={row.seed}",
            flush=True,
        )
        try:
            _, _, attribution = train_task(
                str(row.cancer_id),
                str(row.patient_fold_id),
                int(row.seed),
                args.epochs,
                args.patience,
            )
            for group, value in attribution.items():
                attribution_rows.append(
                    {
                        "cancer_id": row.cancer_id,
                        "patient_fold_id": row.patient_fold_id,
                        "seed": row.seed,
                        "feature_group": group,
                        "normalized_weight_attribution": value,
                    }
                )
        except Exception:
            error = traceback.format_exc()
            frame.loc[index, ["status", "finished_at", "error"]] = [
                "FAILED",
                datetime.now().isoformat(),
                error,
            ]
            write_manifest(frame)
            (RUN_ROOT / "FAILED.txt").write_text(error, encoding="utf-8")
            raise
        frame.loc[index, ["status", "finished_at", "error"]] = [
            "COMPLETED",
            datetime.now().isoformat(),
            pd.NA,
        ]
        write_manifest(frame)
    if attribution_rows:
        pd.DataFrame(attribution_rows).to_parquet(
            RESULT_ROOT / "cancer_adapter_feature_attribution.parquet",
            index=False,
            compression="zstd",
        )
    success = {
        "completed_at": datetime.now().isoformat(),
        "n_tasks": int(len(frame)),
        "status": "COMPLETED",
    }
    (RUN_ROOT / "SUCCESS.json").write_text(
        json.dumps(success, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
