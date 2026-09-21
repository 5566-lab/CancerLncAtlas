#!/usr/bin/env python3
"""Train extra GPU seeds, calibrate all seeds, and summarize stability."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cc_hhgt.common import configure_logging, load_config
from cc_hhgt.multiseed import (
    GNN_MODELS,
    calibrate_all_seed_models,
    completion_audit,
    seed_plan,
    summarize_multiseed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/model_v2_1_gpu_4070tis.yaml")
    parser.add_argument("--models", nargs="+", choices=GNN_MODELS, default=list(GNN_MODELS))
    parser.add_argument("--folds", nargs="*")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    cfg = load_config(args.config)
    primary, replicates, seeds = seed_plan(cfg)
    if not bool(cfg.get("multiseed", {}).get("enabled", False)):
        raise RuntimeError("multiseed.enabled is false")
    primary_audit = completion_audit(
        cfg,
        models=args.models,
        folds=args.folds,
        include_calibration=False,
    )
    primary_audit = primary_audit.loc[
        primary_audit.experiment_seed.eq(primary)
    ]
    if not primary_audit.complete.all():
        raise RuntimeError(
            "Primary seed is incomplete. Run 03_train_all before multiseed; "
            f"missing runs={int((~primary_audit.complete).sum())}"
        )

    runner = Path(__file__).with_name("22_train_gpu.py")
    for seed in replicates:
        command = [
            sys.executable,
            str(runner),
            "--config",
            str(Path(args.config)),
            "--seed-base",
            str(seed),
            "--models",
            *args.models,
        ]
        if args.folds:
            command.extend(["--folds", *args.folds])
        if args.fail_fast:
            command.append("--fail-fast")
        if args.verbose:
            command.append("--verbose")
        print(f"Running multiseed replicate {seed}: {' '.join(command)}")
        subprocess.run(command, check=True)

    if not args.skip_calibration:
        calibrated = calibrate_all_seed_models(
            cfg,
            models=args.models,
            folds=args.folds,
        )
        print(f"Calibrated model/fold/seed runs: {calibrated}")
    result = summarize_multiseed(
        cfg,
        models=args.models,
        folds=args.folds,
        calibrated=not args.skip_calibration,
    )
    print(result)
    print(f"Completed seeds: {seeds}")


if __name__ == "__main__":
    main()
