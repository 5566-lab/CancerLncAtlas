#!/usr/bin/env python3
"""Predict with the freshly trained five-fold V3.2 Evidence streaming heads.

Loads the per-fold private heads trained by run_v32_evidence_streaming_training.py,
predicts each candidate on its evaluation fold, then completes the full
candidate universe with explicit null+reason for missing evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from cc_hhgt.v32.evidence_streaming_training import (  # noqa: E402
    iter_fold_trainer_batches,
    validate_streaming_stage,
)
from cc_hhgt.v32.evidence_training import (  # noqa: E402
    N_FOLDS,
    EXACT_KEYS,
    PrivateHeadFit,
    build_fresh_private_eventset_head,
    complete_prediction_frame,
    file_sha256,
    load_core_feature_bundle,
    predict_private_eventset_head,
)

PREDICTION_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_PREDICTION_V1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--core-embedding-root", type=Path, required=True)
    parser.add_argument("--expected-core-manifest-sha256", required=True)
    parser.add_argument("--expected-graph-authority-receipt-sha256", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mc-samples", type=int, default=16)
    parser.add_argument("--max-bags-per-batch", type=int, default=256)
    parser.add_argument("--event-feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args(argv)

    stage_root = Path(args.stage_root).resolve()
    training_root = Path(args.training_root).resolve()
    core_root = Path(args.core_embedding_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"prediction refuses output reuse: {output_root}")
    output_root.mkdir(parents=True)

    validate_streaming_stage(stage_root)
    universe = pd.read_parquet(args.candidates)
    # The staged candidate key space uses bare ENSG lncRNA ids (the universe
    # ships with a display "LNC:" prefix); align the universe to the stage keys
    # before completion so the exact-pair merge actually matches.
    universe["lncrna_id"] = universe["lncrna_id"].astype(str).str.replace(
        r"^LNC:", "", regex=True
    )
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    fold_frames: list[pd.DataFrame] = []
    fold_records: dict[str, dict[str, object]] = {}
    for fold in range(N_FOLDS):
        receipt = json.loads(
            (training_root / f"patient_fold={fold}" / "HEAD_FIT.json").read_text(encoding="utf-8")
        )
        state_path = training_root / f"patient_fold={fold}" / "private_head_state.pt"
        torch = __import__("torch")
        lineage = receipt.get("core_embedding_lineage", {})
        if (
            lineage.get("sha256") != args.expected_core_manifest_sha256.lower()
            or lineage.get("formal_graph_variant") != "G2"
            or lineage.get("graph_authority_receipt_sha256")
            != args.expected_graph_authority_receipt_sha256.lower()
        ):
            raise RuntimeError(
                f"Fold {fold} Evidence head is not bound to the expected current G2 core"
            )
        core = load_core_feature_bundle(
            core_root,
            patient_fold=fold,
            require_current_g2_authority=True,
            expected_manifest_sha256=args.expected_core_manifest_sha256,
            expected_graph_authority_receipt_sha256=(
                args.expected_graph_authority_receipt_sha256
            ),
        )
        model = build_fresh_private_eventset_head(
            event_feature_dim=args.event_feature_dim,
            core_feature_dim=core.combined_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            seed=args.seed,
        )
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
        fit = PrivateHeadFit(
            model=model,
            initial_parameter_sha256=receipt["initial_parameter_sha256"],
            final_parameter_sha256=receipt["final_parameter_sha256"],
            history=receipt["history"],
            optimizer_steps=int(receipt["optimizer_steps"]),
            confidence_supervision_available=bool(receipt["confidence_supervision_available"]),
            direction_supervision_available=bool(receipt["direction_supervision_available"]),
        )
        examples = []
        missing_core = []
        for batch in iter_fold_trainer_batches(
            stage_root,
            fold=fold,
            split="evaluation",
            core=core,
            event_feature_dim=args.event_feature_dim,
            max_bags_per_batch=args.max_bags_per_batch,
        ):
            examples.extend(batch.examples)
            missing_core.extend(batch.missing_core)
        frame = predict_private_eventset_head(
            fit, examples,
            batch_size=args.batch_size,
            mc_samples=args.mc_samples,
            device=device,
        )
        del fit.model
        fold_frames.append(frame)
        fold_path = output_root / f"pair_fold={fold}_predictions.parquet"
        frame.to_parquet(fold_path, index=False)
        fold_records[str(fold)] = {
            "examples": len(examples),
            "missing_core": len(missing_core),
            "prediction_rows": len(frame),
            "available_rows": int(frame.availability.sum()) if len(frame) else 0,
            "receipt_sha256": receipt.get("state_sha256"),
            "prediction_path": fold_path.name,
        }
        print(f"fold {fold}: predicted {len(frame)} rows "
              f"available={int(frame.availability.sum()) if len(frame) else 0} "
              f"missing_core={len(missing_core)}", flush=True)

    predictions = pd.concat(fold_frames + [pd.DataFrame()], ignore_index=True) if fold_frames else pd.DataFrame()
    events = pd.read_parquet(stage_root / "candidate_exact_events_pair_folded.parquet")
    completed = complete_prediction_frame(
        universe,
        predictions,
        events,
        training_run_id="V32-EVIDENCE-STREAMING-R1",
    )
    out_path = output_root / "EVIDENCE_PREDICTIONS.parquet"
    completed.to_parquet(out_path, index=False)
    manifest = {
        "format": PREDICTION_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "PASS_FRESH_V32_EVIDENCE_STREAMING_PREDICTION",
        "training_root": str(training_root),
        "training_manifest_sha256": file_sha256(training_root / "TRAINING_MANIFEST.json"),
        "stage_root": str(stage_root),
        "output_root": str(output_root),
        "device": device,
        "candidate_universe_rows": len(universe),
        "prediction_rows": len(completed),
        "available_rows": int(completed.availability.sum()),
        "unavailable_rows": int((~completed.availability).sum()),
        "folds": fold_records,
        "predictions_parquet": "EVIDENCE_PREDICTIONS.parquet",
        "predictions_sha256": file_sha256(out_path),
        "production_deployed": False,
        "port_8260_touched": False,
        "sealed_test_opened": False,
        "winner_lock_used": False,
    }
    manifest_path = output_root / "PREDICTION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "prediction_rows": len(completed),
        "available_rows": manifest["available_rows"],
        "output_root": str(output_root),
        "manifest_sha256": file_sha256(manifest_path),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
