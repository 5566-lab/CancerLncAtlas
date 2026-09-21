#!/usr/bin/env python3
"""Score one cancer with the validation-selected model and all three seeds."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.common import file_sha256, load_config, read_table
from cc_hhgt.formal_gate import validate_formal_training_gate
from cc_hhgt.pathway_target import require_exact_pathway_contract
from cc_hhgt.prediction_contract import candidate_universe_sha256
from cc_hhgt.v30_integrity import atomic_write_json


SEEDS = (20260726, 20261726, 20262726)
MODELS = ("rgcn", "hgt", "cc_hhgt")
PURPOSE = "WEBSITE_SELECTED_MODEL_THREE_SEED_FULL_UNIVERSE_ENSEMBLE"
PUBLIC_IDENTITY = [
    "candidate_id",
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
]


def _load_single_seed_module(repo_root: Path):
    path = repo_root / "scripts" / "73_v31_full_universe_inference.py"
    spec = importlib.util.spec_from_file_location("v31_single_seed_full_inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load registered full-universe inference: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection(path: Path, run_id: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "PASS"
        or payload.get("run_id") != run_id
        or payload.get("selected_model") not in MODELS
        or payload.get("selection_split") != "val_only"
        or bool(payload.get("test_metrics_used"))
        or int(payload.get("seed_ensemble_size", -1)) != len(SEEDS)
        or int(payload.get("folds", -1)) != 33
        or not bool(payload.get("matrix_audit_release_eligible"))
    ):
        raise RuntimeError("Invalid validation-only website model selection")
    return payload


def _align_and_accumulate(
    reference: pd.DataFrame | None,
    prediction: pd.DataFrame,
    probability_sum: np.ndarray | None,
    probability_square_sum: np.ndarray | None,
    raw_logit_sum: np.ndarray | None,
    direction_sum: np.ndarray | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = sorted(
        set(PUBLIC_IDENTITY + [
            "proxy_positive_probability", "raw_logit",
            "direction_positive_probability",
        ])
        - set(prediction.columns)
    )
    if missing:
        raise RuntimeError(f"Seed inference lacks ensemble columns: {missing}")
    current = prediction.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    identity = current[PUBLIC_IDENTITY].astype(str)
    if identity.candidate_id.duplicated().any():
        raise RuntimeError("Seed inference contains duplicate candidates")
    if reference is None:
        reference = identity
        n = len(reference)
        probability_sum = np.zeros(n, dtype=np.float64)
        probability_square_sum = np.zeros(n, dtype=np.float64)
        raw_logit_sum = np.zeros(n, dtype=np.float64)
        direction_sum = np.zeros(n, dtype=np.float64)
    elif not identity.equals(reference):
        raise RuntimeError("Full-universe candidate identities drift across seeds")
    probability = pd.to_numeric(
        current.proxy_positive_probability, errors="coerce"
    ).to_numpy(float)
    raw_logit = pd.to_numeric(current.raw_logit, errors="coerce").to_numpy(float)
    direction = pd.to_numeric(
        current.direction_positive_probability, errors="coerce"
    ).to_numpy(float)
    if (
        not np.isfinite(probability).all()
        or not np.isfinite(raw_logit).all()
        or not np.isfinite(direction).all()
        or np.any((probability < 0) | (probability > 1))
        or np.any((direction < 0) | (direction > 1))
    ):
        raise RuntimeError("Seed inference contains invalid score values")
    assert probability_sum is not None
    assert probability_square_sum is not None
    assert raw_logit_sum is not None
    assert direction_sum is not None
    probability_sum += probability
    probability_square_sum += probability**2
    raw_logit_sum += raw_logit
    direction_sum += direction
    return reference, probability_sum, probability_square_sum, raw_logit_sum, direction_sum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--formal-gate", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--microbatch-size", type=int, default=8192)
    args = parser.parse_args()
    if args.microbatch_size < 1:
        raise ValueError("microbatch size must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    single = _load_single_seed_module(repo_root)
    selection_path = Path(args.selection).resolve()
    selection = _selection(selection_path, args.run_id)
    model_name = str(selection["selected_model"])
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

    folds = read_table(cfg["_results"] / "tables" / "fold_manifest.tsv")
    hit = folds.loc[folds.fold_id.astype(str).eq(args.fold)]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one registered fold {args.fold}, observed {len(hit)}")
    fold_row = pd.Series(hit.iloc[0].to_dict())
    candidates = single._full_test_candidates(cfg, fold_row)

    reference = None
    probability_sum = probability_square_sum = raw_logit_sum = direction_sum = None
    seed_audits = []
    for seed in SEEDS:
        task_root = training_root / model_name / args.fold / f"seed_{seed}"
        task_success = single._task_success(task_root)
        if (
            task_success.get("run_id") != args.run_id
            or task_success.get("formal_training_gate_sha256") != gate_sha256
        ):
            raise RuntimeError("Selected-model seed task lineage mismatch")
        prediction, runtime_chunks = single._infer_logits(
            cfg,
            task_root,
            fold_row,
            model_name,
            seed,
            candidates,
            args.microbatch_size,
        )
        prediction, temperature = single._apply_registered_calibration(
            task_root, prediction
        )
        overlap = single._audit_overlap(task_root, prediction)
        (
            reference,
            probability_sum,
            probability_square_sum,
            raw_logit_sum,
            direction_sum,
        ) = _align_and_accumulate(
            reference,
            prediction,
            probability_sum,
            probability_square_sum,
            raw_logit_sum,
            direction_sum,
        )
        seed_audits.append(
            {
                "seed": seed,
                "checkpoint_sha256": file_sha256(task_root / "best_pathway.pt"),
                "calibration_sha256": file_sha256(task_root / "calibration_pathway.json"),
                "temperature": temperature,
                "runtime_chunks": runtime_chunks,
                "overlap_audit": overlap,
            }
        )

    assert reference is not None
    assert probability_sum is not None and probability_square_sum is not None
    assert raw_logit_sum is not None and direction_sum is not None
    n_seeds = len(SEEDS)
    mean_probability = probability_sum / n_seeds
    variance = np.maximum(probability_square_sum / n_seeds - mean_probability**2, 0.0)
    output = reference.copy()
    output["proxy_positive_probability"] = mean_probability
    output["calibrated_probability"] = mean_probability
    output["seed_probability_std"] = np.sqrt(variance)
    output["raw_logit_mean"] = raw_logit_sum / n_seeds
    output["direction_positive_probability"] = direction_sum / n_seeds
    output["prediction_scale"] = "three_seed_mean_calibrated_probability"
    output["prediction_purpose"] = PURPOSE
    output["source_fold_id"] = args.fold
    output["source_model_name"] = model_name
    output["source_seed_count"] = n_seeds
    output["formal_gate_sha256"] = gate_sha256
    output["sample_universe_sha256"] = gate["sample_universe_sha256"]
    # Public score artifact deliberately excludes proxy_label and label_class.
    if {"proxy_label", "label_class"} & set(output.columns):
        raise RuntimeError("Held-out label entered website score artifact")

    output_root = Path(args.output_root).resolve()
    output_dir = output_root / model_name / args.fold
    if output_dir.exists():
        raise RuntimeError(f"Selected-model ensemble inference refuses reuse: {output_dir}")
    output_dir.mkdir(parents=True)
    output_path = output_dir / "prediction_exact_pathway_three_seed_ensemble.parquet"
    single._atomic_write_table(output, output_path)
    payload = {
        "status": "PASS",
        "run_id": args.run_id,
        "fold_id": args.fold,
        "test_cancer": str(fold_row.test_cancer),
        "selected_model": model_name,
        "selection_sha256": file_sha256(selection_path),
        "selection_test_metrics_used": False,
        "seed_count": n_seeds,
        "seeds": list(SEEDS),
        "rows": int(len(output)),
        "candidate_universe_sha256": candidate_universe_sha256(
            output, PUBLIC_IDENTITY
        ),
        "pathway_target_level": "exact_pathway",
        "heldout_label_columns_in_output": False,
        "prediction_file": str(output_path),
        "prediction_file_sha256": file_sha256(output_path),
        "formal_gate_sha256": gate_sha256,
        "seed_audits": seed_audits,
        "inference_script_sha256": file_sha256(Path(__file__).resolve()),
    }
    atomic_write_json(output_dir / "SUCCESS.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
