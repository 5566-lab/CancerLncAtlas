#!/usr/bin/env python3
"""Retrain patient/state decoder heads against an isolated strict embedding matrix."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from cc_hhgt.common import load_config, write_json


def validate_state_data(state_data: Path, cfg: dict) -> dict:
    success_path = state_data / "SUCCESS.json"
    if not success_path.exists():
        raise RuntimeError(f"Leakage-controlled state-data SUCCESS.json is missing: {state_data}")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    files = sorted(state_data.glob("cancer_id=*/part-0.parquet"))
    provenance_path = state_data / "fold_provenance.parquet"
    if not provenance_path.exists():
        raise RuntimeError(f"State-data fold provenance is missing: {provenance_path}")
    provenance = pd.read_parquet(provenance_path)
    manifest = pd.read_csv(
        cfg["_results"] / "tables" / "cancer_specific_patient_fold_manifest.tsv",
        sep="\t",
    )
    expected_folds = int(manifest[["cancer_id", "patient_fold_id"]].drop_duplicates().shape[0])
    observed_folds = int(provenance[["cancer_id", "patient_fold_id"]].drop_duplicates().shape[0])
    scope_ok = (
        success.get("status") == "COMPLETED"
        and success.get("direction_supervision_scope") == "positive_membership_pairs_only"
        and provenance["candidate_selection_scope"].eq("train_patients_only_per_fold").all()
        and provenance["residualizer_fit_scope"].eq("train_patients_only").all()
        and provenance["direction_supervision_scope"].eq("positive_membership_pairs_only").all()
    )
    direction_violations = 0
    for path in files:
        frame = pd.read_parquet(
            path,
            columns=[
                "train_membership_label", "train_direction_label",
                "validation_membership_label", "validation_direction_label",
                "test_membership_label", "test_direction_label",
            ],
        )
        for split in ("train", "validation", "test"):
            direction_violations += int(
                (
                    frame[f"{split}_membership_label"].ne(1.0)
                    & frame[f"{split}_direction_label"].notna()
                ).sum()
            )
    payload = {
        "status": "PASS" if len(files) == 33 and observed_folds == expected_folds and scope_ok and direction_violations == 0 else "FAIL",
        "state_data_root": str(state_data),
        "cancer_files": len(files),
        "expected_patient_folds": expected_folds,
        "observed_patient_folds": observed_folds,
        "fold_safe_scope": bool(scope_ok),
        "nonpositive_direction_label_violations": direction_violations,
    }
    if payload["status"] != "PASS":
        raise RuntimeError(f"Patient state-data prelaunch gate failed: {payload}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain patient-fold state heads using fixed-sampler graph embeddings")
    parser.add_argument("--config", default="config/model_v2_9_state_graph_local_run.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--strict-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--state-data-root", help="Reuse leakage-controlled patient-fold state table")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    strict_root = Path(args.strict_root).resolve()
    output_root = Path(args.output_root).resolve()
    state_data = (
        Path(args.state_data_root).resolve()
        if args.state_data_root
        else cfg["_results"] / "v2_9_downstream" / "lncrna_state_data"
    )
    if not strict_root.exists():
        raise RuntimeError(f"Strict embedding root does not exist: {strict_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    state_data_gate = validate_state_data(state_data, cfg)
    write_json(state_data_gate, output_root / "PATIENT_STATE_DATA_GATE.json")

    env = os.environ.copy()
    env.update(
        {
            "CC_HHGT_RESULT_ROOT": str(output_root),
            "CC_HHGT_V29_STRICT_ROOT": str(strict_root),
            "CC_HHGT_STATE_DATA_ROOT": str(state_data),
            "CC_HHGT_STATE_MODEL_ROOT": str(output_root / "models"),
            "CC_HHGT_STATE_RESULT_ROOT": str(output_root / "experts"),
            "CC_HHGT_STATE_RUN_ROOT": str(output_root / "run_control"),
            "CC_HHGT_STATE_RELEASE_ROOT": str(output_root / "release"),
            "CC_HHGT_STATE_TARGETS": "EXTEND::published_score,stemness_rna::RNAss,stemness_dna::DNAss,stemness_rna::EREG.EXPss",
            "CC_HHGT_REQUIRE_TELOMERASE_STATE": "1",
            "CC_HHGT_REQUIRE_STEMNESS_STATE": "1",
        }
    )
    commands = [
        [args.python, "scripts/28_train_lncrna_state_experts.py", "--epochs", str(args.epochs), "--patience", str(args.patience)],
        [args.python, "scripts/29_integrate_lncrna_state_moe.py"],
        [args.python, "scripts/30_audit_lncrna_state_model.py"],
    ]
    for command in commands:
        print("[run]", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=env)
    payload = {
        "status": "COMPLETED",
        "strict_root": str(strict_root),
        "state_data_root": str(state_data),
        "output_root": str(output_root),
        "expert_prediction_root": str(output_root / "experts"),
        "epochs": args.epochs,
        "patience": args.patience,
        "patient_state_data_gate": state_data_gate,
        "graph_encoder_retrained": False,
        "decoder_heads_retrained": True,
    }
    write_json(payload, output_root / "RETRAINED_STATE_PATIENT_HEADS_SUCCESS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
