"""Fail-closed aggregation for the V3.1 Fixed Graph B1 matrix."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import file_sha256, require_columns, write_table
from .metrics import binary_metrics
from .prediction_contract import candidate_universe_sha256
from .v30_integrity import atomic_write_json, merkle_sha256
from .v31_pilot_gate import (
    DIAGNOSTIC_CONTRACT,
    PILOT_CANCERS,
    PILOT_SEEDS,
    PRIMARY_CONTRACT,
)


def expected_b1_matrix() -> list[tuple[str, str, int]]:
    matrix = [
        (PRIMARY_CONTRACT, cancer, seed)
        for cancer in PILOT_CANCERS
        for seed in PILOT_SEEDS
    ]
    matrix.extend(
        (DIAGNOSTIC_CONTRACT, cancer, PILOT_SEEDS[0])
        for cancer in PILOT_CANCERS
    )
    if len(matrix) != 12:
        raise AssertionError("B1 matrix contract drift")
    return matrix


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _task_root(fixed_root: Path, contract: str, cancer: str, seed: int) -> Path:
    return fixed_root / contract / "cc_hhgt" / f"LOCO_{cancer}" / f"seed_{seed}"


def _target_columns(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    out = frame.copy()
    if task == "pathway":
        require_columns(out, ["pathway_family_id"], "pathway prediction")
        out["target_id"] = out.pathway_family_id.astype(str)
        out["target_type"] = "pathway"
        out["target_subtype"] = "Pathway"
    else:
        require_columns(out, ["state_id"], "state prediction")
        out["target_id"] = out.state_id.astype(str)
        out["target_type"] = "state"
        out["target_subtype"] = np.where(
            out.target_id.str.contains("RNAss", case=False, regex=False), "RNAss", "DNAss"
        )
    return out


def _read_task_predictions(
    task_root: Path,
    contract: str,
    cancer: str,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    success_path = task_root / "SUCCESS.json"
    if not success_path.is_file():
        raise RuntimeError(f"B1 task lacks SUCCESS.json: {task_root}")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if success.get("status") != "COMPLETED":
        raise RuntimeError(f"B1 task is not COMPLETED: {task_root}")
    if (
        success.get("contract") != contract
        or str(success.get("test_cancer")) != cancer
        or int(success.get("seed")) != seed
    ):
        raise RuntimeError(f"B1 SUCCESS identity mismatch: {task_root}")
    training_path = task_root / "TRAINING_SUCCESS.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("residual_learning", False):
        raise RuntimeError("B1 standalone task was accidentally trained as a residual")
    if training.get("graph_contract") != contract:
        raise RuntimeError(f"B1 training graph contract mismatch: {task_root}")

    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for task in ("pathway", "state"):
        raw_path = task_root / f"prediction_{task}_raw.parquet"
        calibrated_path = task_root / f"prediction_{task}_calibrated.parquet"
        calibration_path = task_root / f"calibration_{task}.json"
        raw = pd.read_parquet(raw_path)
        calibrated = pd.read_parquet(calibrated_path)
        for name, frame in (("raw", raw), ("calibrated", calibrated)):
            require_columns(
                frame,
                [
                    "candidate_id", "cancer_id", "lncrna_id", "proxy_label", "split",
                    "proxy_positive_probability", "prediction_scale",
                ],
                f"{name} {task} prediction",
            )
        raw_keys = raw[["candidate_id", "proxy_label", "split"]].sort_values("candidate_id").reset_index(drop=True)
        calibrated_keys = calibrated[["candidate_id", "proxy_label", "split"]].sort_values("candidate_id").reset_index(drop=True)
        if not raw_keys.equals(calibrated_keys):
            raise RuntimeError(f"Calibration changed candidate keys/labels: {task_root}/{task}")
        if set(raw.prediction_scale.astype(str)) != {"raw_probability"}:
            raise RuntimeError(f"Raw prediction scale mislabeled: {task_root}/{task}")
        if set(calibrated.prediction_scale.astype(str)) != {"calibrated_probability"}:
            raise RuntimeError(f"Calibrated prediction scale mislabeled: {task_root}/{task}")
        calibration_sha = file_sha256(calibration_path)
        for scale, path, full in (
            ("raw_probability", raw_path, raw),
            ("calibrated_probability", calibrated_path, calibrated),
        ):
            test = full.loc[full.split.astype(str).eq("test")].copy()
            if test.empty or set(test.cancer_id.astype(str)) != {cancer}:
                raise RuntimeError(f"Invalid B1 LOCO test rows: {path}")
            test = _target_columns(test, task)
            test["contract"] = contract
            test["seed"] = int(seed)
            test["metric_scope"] = "LOCO_test"
            test["source_prediction_file_sha256"] = file_sha256(path)
            test["candidate_universe_sha256"] = candidate_universe_sha256(test)
            test["calibration_model_sha256"] = (
                "NOT_APPLICABLE_RAW" if scale == "raw_probability" else calibration_sha
            )
            parts.append(test)
        raw_test = parts[-2]
        calibrated_test = parts[-1]
        for subtype in sorted(raw_test.target_subtype.unique()):
            raw_group = raw_test.loc[raw_test.target_subtype.eq(subtype)].sort_values("candidate_id")
            cal_group = calibrated_test.loc[calibrated_test.target_subtype.eq(subtype)].sort_values("candidate_id")
            raw_metric = binary_metrics(raw_group.proxy_label, raw_group.proxy_positive_probability)
            cal_metric = binary_metrics(cal_group.proxy_label, cal_group.proxy_positive_probability)
            auroc_delta = abs(float(raw_metric["auroc"]) - float(cal_metric["auroc"]))
            auprc_delta = abs(float(raw_metric["auprc"]) - float(cal_metric["auprc"]))
            status = "PASS" if max(auroc_delta, auprc_delta) <= 1e-10 else "FAIL"
            audit_rows.append(
                {
                    "contract": contract,
                    "cancer_id": cancer,
                    "seed": seed,
                    "target_subtype": subtype,
                    "candidate_count": len(raw_group),
                    "raw_calibrated_key_parity": True,
                    "auroc_abs_delta": auroc_delta,
                    "auprc_abs_delta": auprc_delta,
                    "status": status,
                }
            )
            if status != "PASS":
                raise RuntimeError(
                    f"Within-group temperature calibration changed ranking metric: {audit_rows[-1]}"
                )
    return pd.concat(parts, ignore_index=True, sort=False), audit_rows


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ["contract", "cancer_id", "seed", "target_subtype", "prediction_scale"]
    for keys, group in predictions.groupby(groups, observed=True, sort=True):
        values = binary_metrics(group.proxy_label, group.proxy_positive_probability)
        prevalence = float(values["positive_rate"])
        auprc = float(values["auprc"])
        rows.append(
            {
                **dict(zip(groups, keys)),
                "metric_scope": "LOCO_test",
                **values,
                "auprc_over_prevalence": auprc / prevalence if prevalence > 0 else np.nan,
                "auprc_minus_prevalence": auprc - prevalence,
                "candidate_universe_sha256": candidate_universe_sha256(group),
                "prediction_file_sha256": "SET_AFTER_WRITE",
                "calibration_model_sha256": (
                    "NOT_APPLICABLE_RAW"
                    if keys[-1] == "raw_probability"
                    else "MULTIPLE_TASK_FILES_SEE_PREDICTION_ROWS"
                ),
            }
        )
    return pd.DataFrame(rows)


def finalize_fixed_graph(
    fixed_root: Path,
    output_dir: Path,
    *,
    config_path: Path,
    pilot_gate_path: Path,
    aggregation_git_commit: str,
) -> dict[str, Any]:
    """Verify exact B1 completion and publish the standalone aggregate."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "FIXED_GRAPH_CONFIG.yaml",
        output_dir / "FIXED_GRAPH_PREDICTIONS.parquet",
        output_dir / "FIXED_GRAPH_METRICS.tsv",
        output_dir / "FIXED_GRAPH_B1_GATE.json",
        output_dir / "RAW_CALIBRATED_METRIC_AUDIT.tsv",
        output_dir / "FIXED_GRAPH_SHA256.tsv",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise RuntimeError(f"B1 finalizer refuses overwrite: {existing}")
    gate = json.loads(pilot_gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("expected_b1_tasks") != 12:
        raise RuntimeError("Registered V3.1 pilot gate is not a 12-task PASS")
    expected = expected_b1_matrix()
    observed_success = {
        path.parent.resolve()
        for path in fixed_root.glob("CONTRACT-*/cc_hhgt/LOCO_*/seed_*/SUCCESS.json")
    }
    expected_roots = {
        _task_root(fixed_root, contract, cancer, seed).resolve()
        for contract, cancer, seed in expected
    }
    if observed_success != expected_roots:
        raise RuntimeError(
            f"B1 task set mismatch: missing={sorted(map(str, expected_roots-observed_success))}, "
            f"extra={sorted(map(str, observed_success-expected_roots))}"
        )

    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for contract, cancer, seed in expected:
        prediction, audit = _read_task_predictions(
            _task_root(fixed_root, contract, cancer, seed), contract, cancer, seed
        )
        parts.append(prediction)
        audit_rows.extend(audit)
    predictions = pd.concat(parts, ignore_index=True, sort=False)

    # Every Contract-T seed must score the exact same registered candidates
    # and labels; seed variation is restricted to model output.
    for (contract, cancer, subtype), group in predictions.loc[
        predictions.prediction_scale.eq("raw_probability")
    ].groupby(["contract", "cancer_id", "target_subtype"], observed=True):
        signatures = []
        for seed, seed_group in group.groupby("seed", observed=True):
            ordered = seed_group[["candidate_id", "proxy_label"]].sort_values("candidate_id")
            signatures.append((int(seed), ordered.reset_index(drop=True)))
        reference_seed, reference = signatures[0]
        for current_seed, current in signatures[1:]:
            if not reference.equals(current):
                raise RuntimeError(
                    f"B1 seed candidate/label drift: {contract}/{cancer}/{subtype}/"
                    f"{reference_seed} vs {current_seed}"
                )

    config_copy = outputs[0]
    config_copy.write_bytes(config_path.read_bytes())
    prediction_path = outputs[1]
    _atomic_table(predictions, prediction_path)
    metrics = _metrics(predictions)
    metrics["prediction_file_sha256"] = file_sha256(prediction_path)
    _atomic_table(metrics, outputs[2])
    raw_calibrated_audit = pd.DataFrame(audit_rows)
    _atomic_table(raw_calibrated_audit, outputs[4])

    manifest_rows = []
    for path in outputs[:3] + [outputs[4]]:
        manifest_rows.append(
            {
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    _atomic_table(manifest, outputs[5])
    payload = {
        "status": "PASS",
        "stage": "PHASE_B1_FIXED_GRAPH_STANDALONE",
        "tasks_expected": 12,
        "tasks_completed": 12,
        "primary_contract_tasks": 9,
        "diagnostic_contract_tasks": 3,
        "contracts_reported_separately": True,
        "pilot_cancers": list(PILOT_CANCERS),
        "primary_seeds": list(PILOT_SEEDS),
        "training_git_commit": gate["git_commit"],
        "aggregation_git_commit": aggregation_git_commit,
        "config_sha256": file_sha256(config_copy),
        "pilot_gate_sha256": file_sha256(pilot_gate_path),
        "predictions_sha256": file_sha256(prediction_path),
        "metrics_sha256": file_sha256(outputs[2]),
        "raw_calibrated_audit_sha256": file_sha256(outputs[4]),
        "manifest_sha256": file_sha256(outputs[5]),
        "manifest_merkle_sha256": merkle_sha256(manifest_rows),
        "prediction_rows": int(len(predictions)),
        "metric_rows": int(len(metrics)),
        "full_cancer_training_started": False,
        "failures": [],
    }
    atomic_write_json(outputs[3], payload)
    return payload
