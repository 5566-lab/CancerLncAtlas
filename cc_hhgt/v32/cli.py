"""Command line entry points for V3.2 dry-run and guarded execution."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .orchestration import (
    DEFAULT_RUN_ID,
    DEFAULT_SEED,
    build_dry_run_report,
    build_task_manifest,
    extract_execution_policy,
    write_dry_run_report,
    write_task_manifest,
)
from .training_guard import (
    TrainingAuthorizationError,
    guard_training_entry,
    load_structured_mapping,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cchhgt-v32",
        description="V3.2 code-only planning and fail-closed training entry",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    dry = commands.add_parser("dry-run", help="write a zero-cost blocked task plan")
    dry.add_argument("--config")
    dry.add_argument("--input-manifest")
    dry.add_argument("--output-dir", required=True)
    dry.add_argument("--run-id", default=DEFAULT_RUN_ID)
    dry.add_argument("--seed", type=int, default=DEFAULT_SEED)
    dry.add_argument("--owner", default="local4070")
    dry.add_argument("--hardware-class", default="LOCAL_RTX_4070_TI_SUPER_16GB")

    run = commands.add_parser(
        "run-shard", help="validate explicit authority, then dispatch one task"
    )
    run.add_argument("--allow-training", action="store_true")
    run.add_argument("--repo-root", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--input-manifest", required=True)
    run.add_argument("--task-manifest", required=True)
    run.add_argument("--approval", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--task-id", required=True)
    run.add_argument("--endpoint-id", default="local4070")
    run.add_argument(
        "--hardware-class", default="LOCAL_RTX_4070_TI_SUPER_16GB"
    )
    run.add_argument(
        "--trainer",
        default="cc_hhgt.v32.training:run_authorized_task",
        help="module:function imported only after the training guard passes",
    )
    return parser


def _dry_run(args: argparse.Namespace) -> int:
    config: dict[str, Any] = {}
    if args.config:
        config = load_structured_mapping(args.config)
    policy = extract_execution_policy(config)
    tasks = build_task_manifest(
        run_id=args.run_id,
        seed=args.seed,
        policy=policy,
        owner=args.owner,
        hardware_class=args.hardware_class,
    )
    output = Path(args.output_dir)
    task_path = write_task_manifest(output / "TASK_MANIFEST.tsv", tasks)
    paths = {"task_manifest": task_path}
    if args.config:
        paths["config"] = Path(args.config)
    if args.input_manifest:
        paths["input_manifest"] = Path(args.input_manifest)
    report = build_dry_run_report(
        run_id=args.run_id,
        seed=args.seed,
        policy=policy,
        artifact_paths=paths,
    )
    report["task_manifest_path"] = str(task_path.resolve())
    report_path = write_dry_run_report(output / "DRY_RUN_REPORT.json", report)
    print(
        json.dumps(
            {
                "status": "CODE_ONLY_TASKS_BLOCKED",
                "task_manifest": str(task_path.resolve()),
                "dry_run_report": str(report_path.resolve()),
                "tasks": len(tasks),
                "paid_tasks": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _import_trainer(specification: str):
    """Import a training callable only after authorization has succeeded."""

    if ":" not in specification:
        raise TrainingAuthorizationError("--trainer must use module:function syntax")
    module_name, function_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    trainer = getattr(module, function_name, None)
    if not callable(trainer):
        raise TrainingAuthorizationError(f"Trainer is not callable: {specification}")
    return trainer


def _run_shard(args: argparse.Namespace) -> int:
    # SECURITY BOUNDARY: do not add torch/model imports above this call.
    context = guard_training_entry(
        allow_training=args.allow_training,
        repo_root=args.repo_root,
        config_path=args.config,
        input_manifest_path=args.input_manifest,
        task_manifest_path=args.task_manifest,
        approval_path=args.approval,
        run_id=args.run_id,
        task_id=args.task_id,
        endpoint_id=args.endpoint_id,
        hardware_class=args.hardware_class,
        trainer_specification=args.trainer,
    )
    trainer = _import_trainer(args.trainer)
    result = trainer(context)
    return int(result or 0)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "dry-run":
            return _dry_run(args)
        if args.command == "run-shard":
            return _run_shard(args)
        raise AssertionError(f"Unhandled command: {args.command}")
    except TrainingAuthorizationError as exc:
        print(
            json.dumps(
                {
                    "status": "TRAINING_BLOCKED",
                    "reason": str(exc),
                    "torch_import_attempted": False,
                    "cuda_initialized": False,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 13


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
