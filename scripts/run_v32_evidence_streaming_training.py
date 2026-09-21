#!/usr/bin/env python3
"""Run fresh five-fold V3.2 private Evidence EventSet training from a validated streaming stage.

Consumes only the bounded-reader stage API (``iter_fold_trainer_batches``):
no raw interaction table is ever loaded whole.  Each fold fits one fresh
private attention head on frozen same-fold core embeddings, exactly like the
legacy runner but through the streaming stage contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.evidence_streaming_training import (  # noqa: E402
    iter_fold_trainer_batches,
    validate_streaming_stage,
)
from cc_hhgt.v32.evidence_training import (  # noqa: E402
    N_FOLDS,
    file_sha256,
    fit_private_eventset_head,
    load_core_feature_bundle,
)

TRAINING_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_TRAINING_V1"
HEAD_RECEIPT_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_PRIVATE_HEAD_FIT_V1"


def _fresh_output_root(path: Path) -> Path:
    target = path.resolve()
    if target.exists():
        raise FileExistsError(f"training refuses output reuse: {target}")
    target.mkdir(parents=True)
    return target


def _accumulate_examples(stage_root: Path, fold: int, split: str, core, *, max_bags: int, event_feature_dim: int):
    examples = []
    missing_core = []
    for batch in iter_fold_trainer_batches(
        stage_root,
        fold=fold,
        split=split,
        core=core,
        event_feature_dim=event_feature_dim,
        max_bags_per_batch=max_bags,
    ):
        examples.extend(batch.examples)
        missing_core.extend(batch.missing_core)
    return examples, missing_core


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--core-embedding-root", type=Path, required=True)
    parser.add_argument("--expected-core-manifest-sha256", required=True)
    parser.add_argument("--expected-graph-authority-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--max-bags-per-batch", type=int, default=256)
    parser.add_argument("--event-feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args(argv)

    stage_root = Path(args.stage_root).resolve()
    core_root = Path(args.core_embedding_root).resolve()
    output_root = _fresh_output_root(Path(args.output_root))

    stage_audit = validate_streaming_stage(stage_root)
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    folds: dict[str, dict[str, object]] = {}
    for fold in range(N_FOLDS):
        core = load_core_feature_bundle(
            core_root,
            patient_fold=fold,
            require_current_g2_authority=True,
            expected_manifest_sha256=args.expected_core_manifest_sha256,
            expected_graph_authority_receipt_sha256=(
                args.expected_graph_authority_receipt_sha256
            ),
        )
        train_examples, train_missing = _accumulate_examples(
            stage_root, fold, "train", core,
            max_bags=args.max_bags_per_batch,
            event_feature_dim=args.event_feature_dim,
        )
        evaluation_examples, evaluation_missing = _accumulate_examples(
            stage_root, fold, "evaluation", core,
            max_bags=args.max_bags_per_batch,
            event_feature_dim=args.event_feature_dim,
        )
        fit = fit_private_eventset_head(
            train_examples,
            evaluation_examples,
            event_feature_dim=args.event_feature_dim,
            core_feature_dim=core.combined_dim,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            device=device,
        )
        fold_dir = output_root / f"patient_fold={fold}"
        fold_dir.mkdir()
        # frozen model state
        torch = __import__("torch")
        torch.save(fit.model.state_dict(), fold_dir / "private_head_state.pt")
        # lineage + hashes
        del fit.model
        receipt = {
            "format": HEAD_RECEIPT_FORMAT,
            "patient_fold": fold,
            "device": device,
            "initial_parameter_sha256": fit.initial_parameter_sha256,
            "final_parameter_sha256": fit.final_parameter_sha256,
            "optimizer_steps": fit.optimizer_steps,
            "history": fit.history,
            "confidence_supervision_available": fit.confidence_supervision_available,
            "direction_supervision_available": fit.direction_supervision_available,
            "train_examples": len(train_examples),
            "evaluation_examples": len(evaluation_examples),
            "train_missing_core": len(train_missing),
            "evaluation_missing_core": len(evaluation_missing),
            "core_combined_dim": core.combined_dim,
            "core_embedding_lineage": {
                "path": str(core.lineage.export_paths),
                "sha256": core.lineage.manifest_sha256,
                "formal_graph_variant": core.lineage.formal_graph_variant,
                "graph_authority_receipt_sha256": (
                    core.lineage.graph_authority_receipt_sha256
                ),
                "patient_fold_authority_receipt_sha256": (
                    core.lineage.patient_fold_authority_receipt_sha256
                ),
            },
            "state_path": str(fold_dir / "private_head_state.pt"),
            "state_sha256": file_sha256(fold_dir / "private_head_state.pt"),
        }
        (fold_dir / "HEAD_FIT.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        folds[str(fold)] = {
            "status": "TRAINED",
            "train_examples": len(train_examples),
            "evaluation_examples": len(evaluation_examples),
            "train_missing_core": len(train_missing),
            "evaluation_missing_core": len(evaluation_missing),
            "optimizer_steps": fit.optimizer_steps,
            "final_parameter_sha256": fit.final_parameter_sha256,
            "head_receipt": f"patient_fold={fold}/HEAD_FIT.json",
        }
        print(f"fold {fold}: trained steps={fit.optimizer_steps} "
              f"train={len(train_examples)} eval={len(evaluation_examples)} "
              f"missing_core={len(train_missing) + len(evaluation_missing)}", flush=True)

    manifest = {
        "format": TRAINING_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "PASS_FRESH_V32_EVIDENCE_STREAMING_TRAINING",
        "stage_root": str(stage_root),
        "stage_validation": stage_audit.get("status", "PASS"),
        "core_embedding_root": str(core_root),
        "core_manifest_sha256": args.expected_core_manifest_sha256.lower(),
        "formal_graph_variant": "G2",
        "graph_authority_receipt_sha256": (
            args.expected_graph_authority_receipt_sha256.lower()
        ),
        "output_root": str(output_root),
        "device": device,
        "seed": args.seed,
        "hyperparameters": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "max_events": args.max_events,
            "max_bags_per_batch": args.max_bags_per_batch,
            "event_feature_dim": args.event_feature_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
        "folds": folds,
        "production_deployed": False,
        "port_8260_touched": False,
        "sealed_test_opened": False,
        "winner_lock_used": False,
    }
    manifest_path = output_root / "TRAINING_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "output_root": str(output_root),
        "manifest_sha256": file_sha256(manifest_path),
        "folds_trained": len(folds),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
