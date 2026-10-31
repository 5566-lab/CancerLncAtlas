#!/usr/bin/env python3
"""Fit Evidence EventSet heads with independent validation and no evaluation leakage.

The stage supplies deterministic train and validation streams.  Evaluation is
never constructed by this runner and is opened only by the separate prediction
entry point after a completed, hash-bound V2 head exists.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.evidence_fit_streaming_v2 import (  # noqa: E402
    fit_private_eventset_streaming,
)
from cc_hhgt.v32.evidence_semantics_v2 import EVIDENCE_SEMANTICS  # noqa: E402
from cc_hhgt.v32.evidence_streaming_training import (  # noqa: E402
    iter_fold_trainer_batches,
    validate_streaming_stage,
)
from cc_hhgt.v32.evidence_training import (  # noqa: E402
    N_FOLDS,
    file_sha256,
    load_core_feature_bundle,
)
from cc_hhgt.v32.resume_state_r4 import (  # noqa: E402
    TrainingPaused,
    atomic_json,
    canonical_sha,
)

TRAINING_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_TRAINING_V2"
HEAD_RECEIPT_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_PRIVATE_HEAD_FIT_V2"


def _fresh_output_root(path: Path, *, resume: bool) -> tuple[Path, bool]:
    target = path.resolve()
    existed = target.exists()
    if existed:
        if not target.is_dir():
            raise FileExistsError(f"training output is not a directory: {target}")
        if not resume:
            raise FileExistsError(
                f"training refuses output reuse without --resume: {target}"
            )
    else:
        target.mkdir(parents=True)
    return target, existed


def _completed_fold(
    output_root: Path,
    fold: int,
    *,
    execution_binding_sha256: str,
) -> dict[str, object] | None:
    folder = output_root / f"patient_fold={fold}"
    receipt_path = folder / "HEAD_FIT.json"
    state_path = folder / "private_head_state.pt"
    if not folder.exists():
        return None
    if not receipt_path.is_file():
        return None
    if not state_path.is_file():
        raise RuntimeError(f"Fold {fold} completed receipt lacks its model state")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_validation = (fold + 1) % N_FOLDS
    expected_training = [
        value for value in range(N_FOLDS)
        if value not in {fold, expected_validation}
    ]
    if (
        receipt.get("format") != HEAD_RECEIPT_FORMAT
        or receipt.get("status") != "TRAINED_NOT_DEPLOYED"
        or receipt.get("patient_fold") != fold
        or receipt.get("pair_evaluation_fold") != fold
        or receipt.get("pair_validation_fold") != expected_validation
        or receipt.get("pair_training_folds") != expected_training
        or receipt.get("evaluation_used_for_selection") is not False
        or receipt.get("execution_binding_sha256") != execution_binding_sha256
        or file_sha256(state_path) != receipt.get("state_sha256")
    ):
        raise RuntimeError(
            f"Fold {fold} completed receipt/state does not match this execution"
        )
    return {
        "receipt": f"patient_fold={fold}/HEAD_FIT.json",
        "optimizer_steps": int(receipt["optimizer_steps"]),
        "status": "TRAINED_NOT_DEPLOYED",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--core-embedding-root", type=Path, required=True)
    parser.add_argument("--expected-core-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-graph-authority-receipt-sha256", required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--deadline",
        type=float,
        required=True,
        help="Absolute job deadline; resume cannot renew the original budget",
    )
    parser.add_argument("--checkpoint-steps", type=int, default=20)
    parser.add_argument("--fold", type=int, choices=range(N_FOLDS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--max-bags-per-batch", type=int, default=256)
    parser.add_argument("--event-feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.allow_training:
        parser.error("explicit --allow-training is required")
    if min(
        args.epochs,
        args.patience,
        args.batch_size,
        args.max_events,
        args.max_bags_per_batch,
        args.event_feature_dim,
        args.hidden_dim,
        args.checkpoint_steps,
    ) < 1:
        parser.error("training and checkpoint bounds must be positive")
    for name, value in (
        ("core manifest", args.expected_core_manifest_sha256),
        ("graph authority receipt", args.expected_graph_authority_receipt_sha256),
    ):
        if re.fullmatch(r"[0-9a-fA-F]{64}", str(value)) is None:
            parser.error(f"expected {name} SHA256 must be 64 hexadecimal characters")

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no implicit device fallback")

    stage_root = args.stage_root.resolve()
    core_root = args.core_embedding_root.resolve()
    stage_audit = validate_streaming_stage(stage_root)
    binding = {
        "stage_manifest_sha256": file_sha256(
            stage_root / "STAGING_MANIFEST.json"
        ),
        "core_manifest_sha256": args.expected_core_manifest_sha256.lower(),
        "graph_receipt_sha256": (
            args.expected_graph_authority_receipt_sha256.lower()
        ),
        "script_sha256": file_sha256(Path(__file__)),
        "fitter_sha256": file_sha256(
            ROOT / "cc_hhgt/v32/evidence_fit_streaming_v2.py"
        ),
        "configuration": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "max_events": args.max_events,
            "max_bags_per_batch": args.max_bags_per_batch,
            "event_feature_dim": args.event_feature_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "seed": args.seed,
            "device": args.device,
            "checkpoint_steps": args.checkpoint_steps,
        },
    }
    binding_sha = canonical_sha(binding)
    output_root, existed = _fresh_output_root(
        args.output_root, resume=args.resume
    )
    binding_path = output_root / "EXECUTION_BINDING.json"
    if binding_path.exists():
        prior = json.loads(binding_path.read_text(encoding="utf-8"))
        if (
            not args.resume
            or prior.get("binding_sha256") != binding_sha
            or not isinstance(prior.get("deadline"), (int, float))
        ):
            raise RuntimeError(
                "Evidence output reuse requires the same binding and --resume"
            )
        deadline = min(float(args.deadline), float(prior["deadline"]))
    else:
        if existed:
            raise RuntimeError(
                "Existing Evidence output lacks an execution binding"
            )
        deadline = float(args.deadline)
        atomic_json(
            binding_path,
            {
                "binding_sha256": binding_sha,
                "inputs": binding,
                "deadline": deadline,
            },
        )
    if deadline <= time.time():
        raise RuntimeError(
            "Evidence original budget exhausted; resume cannot reset the deadline"
        )

    stop_requested = [False]

    def on_signal(_signum, _frame):
        stop_requested[0] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    requested_folds = (
        [args.fold] if args.fold is not None else list(range(N_FOLDS))
    )
    for fold in requested_folds:
        completed = _completed_fold(
            output_root,
            fold,
            execution_binding_sha256=binding_sha,
        )
        if completed is not None:
            continue
        folder = output_root / f"patient_fold={fold}"
        folder.mkdir(parents=True, exist_ok=True)
        core = load_core_feature_bundle(
            core_root,
            patient_fold=fold,
            require_current_g2_authority=True,
            expected_manifest_sha256=args.expected_core_manifest_sha256,
            expected_graph_authority_receipt_sha256=(
                args.expected_graph_authority_receipt_sha256
            ),
        )

        def factory(split: str):
            return lambda: iter_fold_trainer_batches(
                stage_root,
                fold=fold,
                split=split,
                core=core,
                event_feature_dim=args.event_feature_dim,
                max_bags_per_batch=args.max_bags_per_batch,
                max_events_per_bag=args.max_events,
            )

        # The fitter API intentionally has no evaluation factory.
        try:
            fit = fit_private_eventset_streaming(
                factory("train"),
                factory("validation"),
                event_feature_dim=args.event_feature_dim,
                core_feature_dim=core.combined_dim,
                seed=args.seed + fold,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                device=args.device,
                checkpoint_root=folder / "resume_state",
                resume=args.resume,
                input_binding={**binding, "fold": fold},
                deadline=deadline,
                checkpoint_steps=args.checkpoint_steps,
                stop_requested=lambda: stop_requested[0],
            )
        except TrainingPaused as exc:
            atomic_json(
                output_root / "STATUS.json",
                {
                    "state": "PAUSED_BUDGET_OR_SIGNAL",
                    "fold": fold,
                    "reason": str(exc),
                    "training_complete": False,
                },
            )
            return 75
        if (
            fit.optimizer_steps <= 0
            or fit.initial_parameter_sha256 == fit.final_parameter_sha256
        ):
            raise RuntimeError("Untrained Evidence head cannot emit a receipt")

        state_path = folder / "private_head_state.pt"
        state_partial = folder / "private_head_state.pt.partial"
        with state_partial.open("wb") as handle:
            torch.save(fit.model.state_dict(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(state_partial, state_path)
        receipt = {
            "format": HEAD_RECEIPT_FORMAT,
            "status": "TRAINED_NOT_DEPLOYED",
            "evidence_semantics": EVIDENCE_SEMANTICS,
            "execution_binding_sha256": binding_sha,
            "patient_fold": fold,
            "pair_evaluation_fold": fold,
            "pair_validation_fold": (fold + 1) % N_FOLDS,
            "pair_training_folds": [
                value for value in range(N_FOLDS)
                if value not in {fold, (fold + 1) % N_FOLDS}
            ],
            "evaluation_used_for_selection": False,
            "initial_parameter_sha256": fit.initial_parameter_sha256,
            "final_parameter_sha256": fit.final_parameter_sha256,
            "optimizer_steps": fit.optimizer_steps,
            "history": fit.history,
            "confidence_supervision_available": (
                fit.confidence_supervision_available
            ),
            "direction_supervision_available": (
                fit.direction_supervision_available
            ),
            "core_combined_dim": core.combined_dim,
            "hyperparameters": {
                "event_feature_dim": args.event_feature_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "max_events": args.max_events,
                "seed": args.seed + fold,
            },
            "core_embedding_lineage": {
                "sha256": core.lineage.manifest_sha256,
                "formal_graph_variant": core.lineage.formal_graph_variant,
                "graph_authority_receipt_sha256": (
                    core.lineage.graph_authority_receipt_sha256
                ),
                "patient_fold_authority_receipt_sha256": (
                    core.lineage.patient_fold_authority_receipt_sha256
                ),
            },
            "state_sha256": file_sha256(state_path),
        }
        atomic_json(folder / "HEAD_FIT.json", receipt)
        del fit.model
        print(
            json.dumps(
                {
                    "fold": fold,
                    "receipt": f"patient_fold={fold}/HEAD_FIT.json",
                    "optimizer_steps": fit.optimizer_steps,
                    "status": "TRAINED_NOT_DEPLOYED",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    folds: dict[str, dict[str, object]] = {}
    for fold in range(N_FOLDS):
        completed = _completed_fold(
            output_root,
            fold,
            execution_binding_sha256=binding_sha,
        )
        if completed is not None:
            folds[str(fold)] = completed
    manifest = {
        "format": TRAINING_FORMAT,
        "status": "TRAINED_NOT_DEPLOYED",
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "stage_root": str(stage_root),
        "stage_validation": stage_audit,
        "folds": folds,
        "core_manifest_sha256": args.expected_core_manifest_sha256.lower(),
        "graph_authority_receipt_sha256": (
            args.expected_graph_authority_receipt_sha256.lower()
        ),
        "evaluation_fold_opened": False,
        "evaluation_used_for_selection": False,
        "changes_primary_ranking": False,
        "production_deployed": False,
        "head_model": "2_LAYER_EVENT_SET_TRANSFORMER",
        "requires_independent_calibration": True,
        "five_folds_complete": len(folds) == N_FOLDS,
        "execution_binding_sha256": binding_sha,
    }
    atomic_json(output_root / "TRAINING_MANIFEST.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
