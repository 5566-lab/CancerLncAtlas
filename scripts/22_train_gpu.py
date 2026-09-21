#!/usr/bin/env python3
"""Resume-safe, single-GPU LOCO training with CUDA OOM fallback."""

from __future__ import annotations

import argparse
import copy
import gc
import os
import traceback
from pathlib import Path

import pandas as pd

from cc_hhgt.common import configure_logging, load_config, read_table, utc_now, write_table
from cc_hhgt.gnn import train_gnn_fold


GNN_MODELS = ("rgcn", "hgt", "cc_hhgt")
EXPECTED_ARTIFACTS = (
    "best.pt",
    "training_history.tsv",
    "prediction_raw.parquet",
    "metrics.tsv",
    "metadata.json",
)
EMBEDDING_ARTIFACTS = ("node_embeddings.pt", "embedding_metadata.json")


def expected_artifacts(cfg: dict[str, object]) -> tuple[str, ...]:
    training = cfg.get("training", {})
    export = bool(training.get("export_node_embeddings", False)) if isinstance(training, dict) else False
    return EXPECTED_ARTIFACTS + (EMBEDDING_ARTIFACTS if export else ())


def is_complete(model_dir: Path, artifacts: tuple[str, ...]) -> bool:
    return all((model_dir / name).exists() and (model_dir / name).stat().st_size > 0 for name in artifacts)


def remove_partial_artifacts(model_dir: Path, artifacts: tuple[str, ...]) -> None:
    for name in artifacts:
        path = model_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def is_cuda_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "outofmemoryerror" in text or "cuda out of memory" in text or "cuda error: out of memory" in text


def persist(rows: list[dict[str, object]], path: Path) -> None:
    write_table(pd.DataFrame(rows), path)


def adjust_fold_seeds(
    folds: pd.DataFrame,
    primary_seed: int,
    experiment_seed: int,
) -> pd.DataFrame:
    adjusted = folds.copy()
    adjusted["split_seed"] = (
        int(experiment_seed)
        + pd.to_numeric(adjusted.split_seed).astype(int)
        - int(primary_seed)
    )
    return adjusted


def experiment_output_paths(
    results: Path,
    primary_seed: int,
    experiment_seed: int,
) -> tuple[Path, Path]:
    if int(experiment_seed) == int(primary_seed):
        return (
            results / "models",
            results / "status" / "gpu_training_status.tsv",
        )
    return (
        results / "models_multiseed" / f"seed_{int(experiment_seed)}",
        results / "status" / f"gpu_training_status_seed_{int(experiment_seed)}.tsv",
    )


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/model_v2_1_gpu_4070tis.yaml")
    parser.add_argument("--models", nargs="+", choices=GNN_MODELS, default=list(GNN_MODELS))
    parser.add_argument("--folds", nargs="*", help="Default: all folds in fold_manifest.tsv")
    parser.add_argument("--status-output", help="Default: <results>/status/gpu_training_status.tsv")
    parser.add_argument(
        "--seed-base",
        type=int,
        help=(
            "Experiment base seed. The primary seed writes results/models; "
            "other seeds write results/models_multiseed/seed_<seed>."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Retrain complete model/fold artifacts")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install the CUDA PyTorch wheel and verify the NVIDIA driver; "
            "do not continue with the CPU wheel."
        )
    cfg = load_config(args.config)
    if str(cfg["training"].get("device", "")).lower() not in {"cuda", "cuda:0", "auto"}:
        raise ValueError("GPU runner requires training.device=cuda, cuda:0, or auto")

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    minimum_gb = float(cfg.get("gpu", {}).get("minimum_total_vram_gb", 14))
    if total_gb < minimum_gb:
        raise RuntimeError(f"GPU has {total_gb:.1f} GiB VRAM; configuration requires at least {minimum_gb:.1f} GiB")
    if bool(cfg.get("gpu", {}).get("enable_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    folds = read_table(cfg["_results"] / "tables" / "fold_manifest.tsv")
    primary_seed = int(
        cfg.get("multiseed", {}).get("primary_seed", cfg["random_seed"])
    )
    if primary_seed != int(cfg["random_seed"]):
        raise ValueError("multiseed.primary_seed must equal random_seed for fold alignment")
    experiment_seed = int(args.seed_base if args.seed_base is not None else primary_seed)
    folds = adjust_fold_seeds(folds, primary_seed, experiment_seed)
    model_root, default_status = experiment_output_paths(
        cfg["_results"], primary_seed, experiment_seed
    )
    artifacts = expected_artifacts(cfg)
    if args.folds:
        requested = set(args.folds)
        missing = requested - set(folds.fold_id.astype(str))
        if missing:
            raise ValueError(f"Unknown folds: {sorted(missing)}")
        folds = folds.loc[folds.fold_id.astype(str).isin(requested)]

    status_path = Path(args.status_output) if args.status_output else default_status
    status_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_table(status_path).to_dict(orient="records") if status_path.exists() else []
    fallback_caps = [
        int(x)
        for x in cfg.get("gpu", {}).get(
            "edge_cap_fallbacks",
            [cfg["training"]["max_edges_per_relation"]],
        )
    ]
    configured_cap = int(cfg["training"]["max_edges_per_relation"])
    caps = list(dict.fromkeys([configured_cap, *fallback_caps]))

    print(
        f"CUDA device: {props.name}; VRAM={total_gb:.1f} GiB; "
        f"PyTorch={torch.__version__}; CUDA runtime={torch.version.cuda}"
    )
    print(
        f"Folds={len(folds)}; models={','.join(args.models)}; edge caps={caps}; "
        f"experiment_seed={experiment_seed}; model_root={model_root}"
    )

    for fold in folds.itertuples(index=False):
        fold_row = pd.Series(fold._asdict())
        for model_name in args.models:
            model_dir = model_root / model_name / str(fold.fold_id)
            rows = [
                old
                for old in rows
                if not (
                    str(old.get("fold_id")) == str(fold.fold_id)
                    and str(old.get("model_name")) == model_name
                )
            ]
            if is_complete(model_dir, artifacts) and not args.force:
                rows.append(
                    {
                        "fold_id": str(fold.fold_id),
                        "test_cancer": str(fold.test_cancer),
                        "model_name": model_name,
                        "experiment_seed": experiment_seed,
                        "fold_split_seed": int(fold.split_seed),
                        "edge_cap": None,
                        "started_at": utc_now(),
                        "finished_at": utc_now(),
                        "status": "SKIPPED_COMPLETE",
                        "error": None,
                    }
                )
                persist(rows, status_path)
                continue

            success = False
            last_error_text = None
            for edge_cap in caps:
                started_at = utc_now()
                row = {
                    "fold_id": str(fold.fold_id),
                    "test_cancer": str(fold.test_cancer),
                    "model_name": model_name,
                    "experiment_seed": experiment_seed,
                    "fold_split_seed": int(fold.split_seed),
                    "edge_cap": edge_cap,
                    "started_at": started_at,
                    "finished_at": None,
                    "status": "RUNNING",
                    "error": None,
                }
                rows.append(row)
                persist(rows, status_path)
                run_cfg = copy.deepcopy(cfg)
                run_cfg["training"]["max_edges_per_relation"] = edge_cap
                run_cfg["_model_output_root"] = model_root
                run_cfg["_experiment_seed"] = experiment_seed
                try:
                    if args.force or not is_complete(model_dir, artifacts):
                        remove_partial_artifacts(model_dir, artifacts)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(0)
                    train_gnn_fold(run_cfg, fold_row, model_name)
                    row["status"] = "SUCCESS"
                    row["finished_at"] = utc_now()
                    row["peak_cuda_memory_gb"] = round(torch.cuda.max_memory_allocated(0) / 1024**3, 3)
                    persist(rows, status_path)
                    success = True
                    break
                except Exception as exc:
                    last_error_text = f"{type(exc).__name__}: {exc}"
                    row["finished_at"] = utc_now()
                    row["error"] = last_error_text
                    if is_cuda_oom(exc):
                        row["status"] = "OOM_RETRY"
                        persist(rows, status_path)
                        remove_partial_artifacts(model_dir, artifacts)
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                    row["status"] = "FAILED"
                    row["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    persist(rows, status_path)
                    if args.fail_fast:
                        raise
                    break
            if not success:
                if args.fail_fast and last_error_text is not None:
                    raise RuntimeError(last_error_text)
                print(f"FAILED: {fold.fold_id}/{model_name}; see {status_path}")

    failed = [
        row
        for row in rows
        if row.get("status") == "FAILED"
    ]
    if failed:
        raise SystemExit(f"{len(failed)} model/fold runs failed; see {status_path}")
    print(f"Training pass complete. Status: {status_path}")


if __name__ == "__main__":
    main()
