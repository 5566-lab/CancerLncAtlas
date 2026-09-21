#!/usr/bin/env python3
"""Score one held-out cancer's complete filtered pathway candidate universe.

This is a post-training, inference-only website-release stage.  It must never
replace the frozen identity-only LOCO evaluation tables or their metrics.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from cc_hhgt.common import (
    available_device,
    file_sha256,
    load_config,
    read_table,
    seed_everything,
    write_table,
)
from cc_hhgt.formal_gate import validate_formal_training_gate
from cc_hhgt.gnn import (
    build_model,
    candidate_tensors,
    load_graph_bundle,
    move_graph,
    require_torch_geometric,
    runtime_bundle_for_step,
)
from cc_hhgt.prediction_contract import candidate_universe_sha256
from cc_hhgt.pathway_target import require_exact_pathway_contract
from cc_hhgt.training_data import evaluation_candidates_from_files
from cc_hhgt.v29_multitask import (
    _pair_masked_batch,
    _pathway_forward,
    _slice_pathway_batch,
)
from cc_hhgt.v30_integrity import atomic_write_json


PURPOSE = "WEBSITE_FULL_UNIVERSE_INFERENCE"
OVERLAP_ATOL = 1e-5


def _atomic_write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _task_success(task_root: Path) -> dict:
    success_path = task_root / "SUCCESS.json"
    if not success_path.exists():
        raise RuntimeError(f"Formal task has no SUCCESS.json: {task_root}")
    payload = json.loads(success_path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETED":
        raise RuntimeError(f"Formal task is not completed: {success_path}")
    if bool(payload.get("hit_hard_epoch_cap")):
        raise RuntimeError(f"Hard-cap task cannot enter website inference: {task_root}")
    return payload


def _full_test_candidates(cfg: dict, fold_row: pd.Series) -> pd.DataFrame:
    inference_cfg = copy.deepcopy(cfg)
    inference_cfg["training"]["max_evaluation_pairs_per_cancer"] = 0
    frame = evaluation_candidates_from_files(
        inference_cfg, [str(fold_row.test_cancer)]
    )
    if frame.empty:
        raise RuntimeError(f"No full-universe candidates for {fold_row.test_cancer}")
    if set(frame.cancer_id.astype(str)) != {str(fold_row.test_cancer)}:
        raise RuntimeError("Full-universe candidate cancer mismatch")
    if frame.candidate_id.astype(str).duplicated().any():
        raise RuntimeError("Full-universe candidate IDs are not unique")
    if set(frame.evaluation_sampling_policy.astype(str)) != {"full_universe"}:
        raise RuntimeError("Full-universe inference unexpectedly used sampling")
    frame["split"] = "website_full_universe"
    return frame


def _infer_logits(
    cfg: dict,
    task_root: Path,
    fold_row: pd.Series,
    kind: str,
    seed: int,
    candidates: pd.DataFrame,
    microbatch_size: int,
) -> tuple[pd.DataFrame, int]:
    torch = require_torch_geometric()
    seed_everything(seed)
    device = available_device(cfg["training"].get("device", "auto"))
    if device == "unavailable":
        raise RuntimeError("PyTorch device is unavailable")
    if cfg.get("residual_learning", {}).get("enabled", False):
        raise RuntimeError("Residual models require a registered full-universe base")

    excluded = {str(fold_row.test_cancer), str(fold_row.validation_cancer)}
    bundle = load_graph_bundle(cfg, seed, excluded_cancers=excluded)
    checkpoint_path = task_root / "best_pathway.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if str(checkpoint.get("kind")) != kind:
        raise RuntimeError("Checkpoint model kind mismatch")
    features = list(checkpoint.get("features") or [])
    expected_features = [
        column for column in cfg["training"]["feature_columns"] if column in candidates
    ]
    if features != expected_features:
        raise RuntimeError(
            f"Checkpoint feature contract mismatch: {features} != {expected_features}"
        )

    mapped, batch = candidate_tensors(
        candidates, bundle, features, device, kind
    )
    expected_ids = set(candidates.candidate_id.astype(str))
    mapped_ids = set(mapped.candidate_id.astype(str))
    if len(mapped) != len(candidates) or mapped_ids != expected_ids:
        raise RuntimeError(
            "Graph mapping lost full-universe candidates: "
            f"expected={len(candidates)}, mapped={len(mapped)}"
        )
    mask_pair_evidence_for_all = bool(
        cfg.get("state_training", {}).get(
            "mask_pair_evidence_for_all_pathway_models", False
        )
    )
    target_batch = _pair_masked_batch(
        batch, zero_all=(mask_pair_evidence_for_all or kind == "cc_hhgt")
    )
    model = build_model(kind, bundle, len(features), cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    runtime_chunks = (
        list(range(bundle.runtime_num_chunks))
        if bundle.runtime_schedule is not None
        else [None]
    )
    logit_sum = np.zeros(len(mapped), dtype=np.float64)
    direction_sum = np.zeros(len(mapped), dtype=np.float64)
    with torch.no_grad():
        for runtime_chunk in runtime_chunks:
            evaluation_bundle = (
                runtime_bundle_for_step(bundle, int(runtime_chunk))
                if runtime_chunk is not None
                else bundle
            )
            graph = move_graph(evaluation_bundle, device, kind)
            encoded = model.encode(graph)
            for start in range(0, len(mapped), microbatch_size):
                stop = min(start + microbatch_size, len(mapped))
                indices = np.arange(start, stop, dtype=np.int64)
                current = _slice_pathway_batch(target_batch, indices)
                logits, directions, _ = _pathway_forward(
                    model, encoded, current, kind
                )
                logit_sum[start:stop] += logits.float().cpu().numpy()
                direction_sum[start:stop] += directions.float().cpu().numpy()

    raw_logit = logit_sum / len(runtime_chunks)
    direction_logit = direction_sum / len(runtime_chunks)
    output = mapped[
        [
            column
            for column in (
                "candidate_id",
                "cancer_id",
                "lncrna_id",
                "pathway_id",
                "pathway_family_id",
                "label_class",
                "proxy_label",
                "direction",
                "evaluation_sampling_policy",
                "evaluation_universe_rows",
            )
            if column in mapped
        ]
    ].copy()
    output["raw_logit"] = raw_logit
    output["raw_probability"] = expit(raw_logit)
    output["direction_positive_probability"] = expit(direction_logit)
    return output, len(runtime_chunks)


def _apply_registered_calibration(task_root: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    calibration_path = task_root / "calibration_pathway.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("method") != "temperature":
        raise RuntimeError("Only the registered temperature calibration is allowed")
    group = calibration.get("groups", {}).get("pathway", {})
    temperature = float(group.get("temperature", np.nan))
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Invalid registered pathway temperature")
    calibrated = expit(frame.raw_logit.to_numpy(float) / temperature)
    frame = frame.copy()
    frame["proxy_positive_probability"] = calibrated
    frame["calibrated_probability"] = calibrated
    frame["prediction_scale"] = "calibrated_probability"
    frame["uncertainty"] = 1.0 - np.abs(calibrated - 0.5) * 2.0
    return frame, temperature


def _audit_overlap(task_root: Path, full: pd.DataFrame) -> dict:
    formal = read_table(task_root / "prediction_pathway_calibrated.parquet")
    formal = formal.loc[formal.split.astype(str).eq("test")].copy()
    if formal.empty:
        raise RuntimeError("Formal task has no test predictions")
    overlap = formal.merge(
        full,
        on="candidate_id",
        how="inner",
        suffixes=("_formal", "_full"),
        validate="one_to_one",
    )
    if len(overlap) != len(formal):
        raise RuntimeError(
            f"Full inference does not cover every formal test row: {len(overlap)}/{len(formal)}"
        )
    raw_delta = np.abs(
        overlap.raw_logit_formal.to_numpy(float)
        - overlap.raw_logit_full.to_numpy(float)
    )
    calibrated_delta = np.abs(
        overlap.proxy_positive_probability_formal.to_numpy(float)
        - overlap.proxy_positive_probability_full.to_numpy(float)
    )
    raw_max = float(raw_delta.max(initial=0.0))
    calibrated_max = float(calibrated_delta.max(initial=0.0))
    if raw_max > OVERLAP_ATOL or calibrated_max > OVERLAP_ATOL:
        raise RuntimeError(
            "Formal/full inference overlap mismatch: "
            f"raw_max={raw_max}, calibrated_max={calibrated_max}"
        )
    return {
        "status": "PASS",
        "formal_test_rows": int(len(formal)),
        "overlap_rows": int(len(overlap)),
        "raw_logit_max_abs_delta": raw_max,
        "calibrated_probability_max_abs_delta": calibrated_max,
        "absolute_tolerance": OVERLAP_ATOL,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V3.1 inference-only complete pathway candidate scorer"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--formal-gate", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--model", choices=["rgcn", "hgt", "cc_hhgt"], required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--microbatch-size", type=int, default=8192)
    args = parser.parse_args()
    if args.microbatch_size < 1:
        raise ValueError("microbatch size must be positive")

    training_root = Path(args.training_root).resolve()
    gate_path = Path(args.formal_gate).resolve()
    gate_document = json.loads(gate_path.read_text(encoding="utf-8"))
    cfg = load_config(
        args.config,
        project_root_override=Path(gate_document["paths"]["run_root"]).resolve(),
        create_dirs=False,
    )
    require_exact_pathway_contract(cfg)
    gate, gate_sha256 = validate_formal_training_gate(
        gate_path,
        cfg,
        verify_assets=True,
        run_id=args.run_id,
        input_root=gate_document["paths"]["input_root"],
        input_manifest=gate_document["paths"]["input_manifest"],
        output_root=training_root,
    )
    cfg["_results"] = Path(gate["paths"]["asset_results"]).resolve()
    cfg["_standardized"] = cfg["_results"].parent / "standardized"
    cfg["_cache"] = cfg["_results"].parent / "cache"
    cfg["_strict_output_root"] = training_root
    cfg["_formal_training_gate_path"] = str(gate_path)
    cfg["_formal_training_gate_sha256"] = gate_sha256
    cfg["_run_id"] = args.run_id
    cfg["_sample_universe_sha256"] = gate["sample_universe_sha256"]
    provenance_path = (
        Path(gate["paths"]["run_root"]) / "provenance" / "PROVENANCE.json"
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    task_root = training_root / args.model / args.fold / f"seed_{args.seed}"
    task_success = _task_success(task_root)
    if task_success.get("run_id") != args.run_id:
        raise RuntimeError("Task run_id mismatch")
    if task_success.get("formal_training_gate_sha256") != gate_sha256:
        raise RuntimeError("Task formal-gate lineage mismatch")

    folds = read_table(cfg["_results"] / "tables" / "fold_manifest.tsv")
    hit = folds.loc[folds.fold_id.astype(str).eq(args.fold)]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one registered fold {args.fold}, observed {len(hit)}")
    fold_row = pd.Series(hit.iloc[0].to_dict())
    candidates = _full_test_candidates(cfg, fold_row)
    prediction, runtime_chunks = _infer_logits(
        cfg,
        task_root,
        fold_row,
        args.model,
        args.seed,
        candidates,
        args.microbatch_size,
    )
    prediction, temperature = _apply_registered_calibration(task_root, prediction)
    prediction["split"] = "website_full_universe"
    prediction["source_fold_id"] = args.fold
    prediction["source_model_name"] = args.model
    prediction["source_seed"] = args.seed
    prediction["prediction_purpose"] = PURPOSE
    prediction["runtime_ensemble_chunks"] = runtime_chunks
    prediction["checkpoint_sha256"] = file_sha256(task_root / "best_pathway.pt")
    prediction["calibration_model_sha256"] = file_sha256(
        task_root / "calibration_pathway.json"
    )
    prediction["formal_gate_sha256"] = gate_sha256
    prediction["sample_universe_sha256"] = gate["sample_universe_sha256"]
    test_cancer = str(fold_row.test_cancer)
    source_candidate_path = (
        cfg["_results"]
        / "tables"
        / "candidate_universe"
        / f"cancer_id={test_cancer}"
        / "part-0.parquet"
    )
    eligibility_path = (
        cfg["_results"] / "tables" / "pancancer_lncrna_eligibility.parquet"
    )
    prediction["source_candidate_file_sha256"] = file_sha256(source_candidate_path)
    prediction["pancancer_lncrna_eligibility_sha256"] = file_sha256(
        eligibility_path
    )
    prediction["input_merkle_sha256"] = gate["manifest_contracts"][0][
        "merkle_sha256"
    ]
    prediction["code_merkle_sha256"] = gate["manifest_contracts"][1][
        "merkle_sha256"
    ]
    prediction["asset_merkle_sha256"] = gate["asset_merkle_sha256"]
    prediction["config_merkle_sha256"] = provenance["config_merkle_sha256"]
    prediction["inference_script_sha256"] = file_sha256(Path(__file__).resolve())
    overlap = _audit_overlap(task_root, prediction)

    output_root = Path(args.output_root).resolve()
    output_dir = output_root / args.model / args.fold / f"seed_{args.seed}"
    if output_dir.exists():
        raise RuntimeError(f"Full-universe inference output must be new: {output_dir}")
    output_dir.mkdir(parents=True)
    output_path = output_dir / "prediction_pathway_full_universe.parquet"
    _atomic_write_table(prediction, output_path)
    success = {
        "status": "PASS",
        "analysis_version": cfg["analysis_version"],
        "run_id": args.run_id,
        "prediction_purpose": PURPOSE,
        "model_name": args.model,
        "fold_id": args.fold,
        "test_cancer": test_cancer,
        "seed": args.seed,
        "rows": int(len(prediction)),
        "unique_candidates": int(prediction.candidate_id.astype(str).nunique()),
        "candidate_universe_sha256": candidate_universe_sha256(
            prediction,
            ["candidate_id", "cancer_id", "lncrna_id", "pathway_id"],
        ),
        "pathway_target_level": "exact_pathway",
        "source_candidate_file": str(source_candidate_path),
        "source_candidate_file_sha256": file_sha256(source_candidate_path),
        "pancancer_lncrna_eligibility_file": str(eligibility_path),
        "pancancer_lncrna_eligibility_sha256": file_sha256(eligibility_path),
        "prediction_file": str(output_path),
        "prediction_file_sha256": file_sha256(output_path),
        "checkpoint_sha256": file_sha256(task_root / "best_pathway.pt"),
        "calibration_model_sha256": file_sha256(task_root / "calibration_pathway.json"),
        "temperature": temperature,
        "runtime_ensemble_chunks": runtime_chunks,
        "formal_gate_sha256": gate_sha256,
        "sample_universe_sha256": gate["sample_universe_sha256"],
        "input_merkle_sha256": gate["manifest_contracts"][0]["merkle_sha256"],
        "code_merkle_sha256": gate["manifest_contracts"][1]["merkle_sha256"],
        "asset_merkle_sha256": gate["asset_merkle_sha256"],
        "config_merkle_sha256": provenance["config_merkle_sha256"],
        "inference_script_sha256": file_sha256(Path(__file__).resolve()),
        "overlap_audit": overlap,
    }
    atomic_write_json(output_dir / "SUCCESS.json", success)
    print(json.dumps(success, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
