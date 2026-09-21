from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .calibration import calibrate_fold_model
from .common import read_table, utc_now, write_json, write_table


GNN_MODELS = ("rgcn", "hgt", "cc_hhgt")
BASE_ARTIFACTS = (
    "best.pt",
    "training_history.tsv",
    "prediction_raw.parquet",
    "metrics.tsv",
    "metadata.json",
)
EMBEDDING_ARTIFACTS = ("node_embeddings.pt", "embedding_metadata.json")
CALIBRATION_ARTIFACTS = (
    "calibration.json",
    "prediction_calibrated.parquet",
    "metrics_calibrated.tsv",
)


def seed_plan(cfg: dict[str, Any]) -> tuple[int, list[int], list[int]]:
    settings = cfg.get("multiseed", {})
    primary = int(settings.get("primary_seed", cfg["random_seed"]))
    replicates = [int(seed) for seed in settings.get("replicate_seeds", [])]
    if primary != int(cfg["random_seed"]):
        raise ValueError("multiseed.primary_seed must equal random_seed")
    if primary in replicates or len(replicates) != len(set(replicates)):
        raise ValueError("multiseed replicate seeds must be unique and exclude primary_seed")
    return primary, replicates, [primary, *replicates]


def model_root_for_seed(cfg: dict[str, Any], seed: int) -> Path:
    primary, _, _ = seed_plan(cfg)
    if int(seed) == primary:
        return cfg["_results"] / "models"
    return cfg["_results"] / "models_multiseed" / f"seed_{int(seed)}"


def required_artifacts(
    cfg: dict[str, Any],
    include_calibration: bool = False,
) -> tuple[str, ...]:
    artifacts = BASE_ARTIFACTS
    if bool(cfg["training"].get("export_node_embeddings", False)):
        artifacts += EMBEDDING_ARTIFACTS
    if include_calibration:
        artifacts += CALIBRATION_ARTIFACTS
    return artifacts


def model_dir_complete(
    path: Path,
    artifacts: Iterable[str],
) -> bool:
    return all(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in artifacts
    )


def completion_audit(
    cfg: dict[str, Any],
    models: Iterable[str] = GNN_MODELS,
    folds: Iterable[str] | None = None,
    include_calibration: bool = False,
) -> pd.DataFrame:
    fold_manifest = read_table(cfg["_results"] / "tables" / "fold_manifest.tsv")
    fold_ids = list(folds) if folds is not None else fold_manifest.fold_id.astype(str).tolist()
    artifacts = required_artifacts(cfg, include_calibration=include_calibration)
    _, _, seeds = seed_plan(cfg)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        root = model_root_for_seed(cfg, seed)
        for model in models:
            for fold in fold_ids:
                directory = root / str(model) / str(fold)
                missing = [
                    name
                    for name in artifacts
                    if not (directory / name).is_file()
                    or (directory / name).stat().st_size == 0
                ]
                rows.append(
                    {
                        "experiment_seed": seed,
                        "model_name": str(model),
                        "fold_id": str(fold),
                        "model_dir": str(directory),
                        "complete": not missing,
                        "missing_artifacts": ";".join(missing),
                    }
                )
    return pd.DataFrame(rows)


def calibrate_all_seed_models(
    cfg: dict[str, Any],
    models: Iterable[str] = GNN_MODELS,
    folds: Iterable[str] | None = None,
) -> int:
    fold_manifest = read_table(cfg["_results"] / "tables" / "fold_manifest.tsv")
    fold_ids = list(folds) if folds is not None else fold_manifest.fold_id.astype(str).tolist()
    _, _, seeds = seed_plan(cfg)
    calibrated = 0
    for seed in seeds:
        model_root = model_root_for_seed(cfg, seed)
        for model in models:
            for fold in fold_ids:
                raw = model_root / str(model) / str(fold) / "prediction_raw.parquet"
                if not raw.exists():
                    raise FileNotFoundError(raw)
                calibrate_fold_model(
                    cfg,
                    str(model),
                    str(fold),
                    model_root=model_root,
                )
                calibrated += 1
    return calibrated


def _metric_columns(cfg: dict[str, Any], metrics: pd.DataFrame) -> list[str]:
    requested = cfg.get("multiseed", {}).get(
        "metric_columns",
        ["auroc", "auprc", "brier", "ece", "log_loss"],
    )
    return [str(column) for column in requested if column in metrics.columns]


def summarize_multiseed(
    cfg: dict[str, Any],
    models: Iterable[str] = GNN_MODELS,
    folds: Iterable[str] | None = None,
    calibrated: bool = True,
) -> dict[str, Any]:
    fold_manifest = read_table(cfg["_results"] / "tables" / "fold_manifest.tsv")
    if folds is not None:
        fold_manifest = fold_manifest.loc[
            fold_manifest.fold_id.astype(str).isin(set(map(str, folds)))
        ].copy()
    fold_lookup = fold_manifest.set_index("fold_id").test_cancer.astype(str).to_dict()
    _, _, seeds = seed_plan(cfg)
    rows: list[pd.DataFrame] = []
    metric_file = "metrics_calibrated.tsv" if calibrated else "metrics.tsv"
    for seed in seeds:
        model_root = model_root_for_seed(cfg, seed)
        for model in models:
            for fold in fold_manifest.fold_id.astype(str):
                path = model_root / str(model) / fold / metric_file
                if not path.exists():
                    raise FileNotFoundError(path)
                frame = read_table(path)
                frame = frame.loc[frame.split.astype(str).eq("test")].copy()
                frame["experiment_seed"] = seed
                frame["fold_id"] = fold
                frame["test_cancer"] = fold_lookup[fold]
                frame["model_name"] = str(model)
                rows.append(frame)
    all_metrics = pd.concat(rows, ignore_index=True)
    metric_columns = _metric_columns(cfg, all_metrics)
    if not metric_columns:
        raise ValueError("No configured multiseed metric columns are available")

    long = all_metrics.melt(
        id_vars=["experiment_seed", "model_name", "fold_id", "test_cancer"],
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )
    per_seed = (
        long.groupby(["experiment_seed", "model_name", "metric"], as_index=False)
        .agg(
            n_folds=("value", "count"),
            fold_mean=("value", "mean"),
            fold_sd=("value", "std"),
        )
    )
    summary_rows: list[dict[str, object]] = []
    confidence = float(cfg.get("multiseed", {}).get("confidence_level", 0.95))
    for (model, metric), group in per_seed.groupby(
        ["model_name", "metric"], observed=True
    ):
        values = group.fold_mean.dropna().to_numpy(float)
        n = len(values)
        mean = float(np.mean(values)) if n else np.nan
        sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        critical = (
            float(student_t.ppf((1 + confidence) / 2, df=n - 1))
            if n > 1
            else np.nan
        )
        margin = critical * sem if n > 1 else np.nan
        summary_rows.append(
            {
                "model_name": model,
                "metric": metric,
                "n_seeds": n,
                "mean_across_seed_fold_means": mean,
                "sd_across_seed_fold_means": sd,
                "sem": sem,
                "confidence_level": confidence,
                "ci_low": mean - margin if n > 1 else np.nan,
                "ci_high": mean + margin if n > 1 else np.nan,
            }
        )
    model_summary = pd.DataFrame(summary_rows)
    per_fold = (
        long.groupby(
            ["model_name", "fold_id", "test_cancer", "metric"],
            as_index=False,
        )
        .agg(
            n_seeds=("value", "count"),
            seed_mean=("value", "mean"),
            seed_sd=("value", "std"),
            seed_min=("value", "min"),
            seed_max=("value", "max"),
        )
    )

    pivot = long.pivot_table(
        index=["experiment_seed", "fold_id", "test_cancer", "metric"],
        columns="model_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    if {"cc_hhgt", "hgt"} <= set(pivot):
        pivot["cc_hhgt_minus_hgt"] = pivot.cc_hhgt - pivot.hgt
    if {"hgt", "rgcn"} <= set(pivot):
        pivot["hgt_minus_rgcn"] = pivot.hgt - pivot.rgcn

    out_dir = cfg["_results"] / "reports" / "multiseed"
    write_table(all_metrics, out_dir / "all_seed_test_metrics.tsv")
    write_table(per_seed, out_dir / "per_seed_fold_summary.tsv")
    write_table(model_summary, out_dir / "model_multiseed_summary.tsv")
    write_table(per_fold, out_dir / "per_fold_seed_stability.tsv")
    write_table(pivot, out_dir / "architecture_ablation_deltas.tsv")
    completion = completion_audit(
        cfg,
        models=models,
        folds=fold_manifest.fold_id.astype(str).tolist(),
        include_calibration=calibrated,
    )
    write_table(completion, out_dir / "multiseed_completion_audit.tsv")
    result = {
        "generated_at": utc_now(),
        "analysis_version": cfg["analysis_version"],
        "seeds": seeds,
        "models": list(models),
        "folds": int(len(fold_manifest)),
        "expected_model_fold_runs": int(len(seeds) * len(list(models)) * len(fold_manifest)),
        "complete_model_fold_runs": int(completion.complete.sum()),
        "calibrated": calibrated,
        "metric_columns": metric_columns,
        "confidence_interval": (
            "Student-t interval across seed-level mean LOCO metrics"
        ),
    }
    write_json(result, out_dir / "multiseed_summary.json")
    if not completion.complete.all():
        raise RuntimeError(
            f"Multiseed completion audit failed: "
            f"{int((~completion.complete).sum())} incomplete runs"
        )
    return result
