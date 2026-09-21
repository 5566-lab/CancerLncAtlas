"""Build the frozen BestSimpleLOCO offset for the V3.1 residual pilot."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import file_sha256, require_columns, write_table
from .metrics import binary_metrics
from .label_contract import load_label_contract, validate_model_target
from .prediction_contract import candidate_universe_sha256
from .v30_integrity import atomic_write_json, merkle_sha256
from .v31_pilot_gate import PILOT_CANCERS, PILOT_SEEDS, PRIMARY_CONTRACT
from .v31_simple_loco import SCORE_COLUMNS, fit_best_simple_loco


FIXED_CANDIDATE_SAMPLING_SEED = PILOT_SEEDS[0]


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_partitioned_cancers(
    table_root: Path,
    cancers: list[str],
    columns: list[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for cancer in sorted(cancers):
        paths = sorted((table_root / f"cancer_id={cancer}").glob("*.parquet"))
        if not paths:
            raise RuntimeError(f"No frozen candidate partition for {cancer}: {table_root}")
        for path in paths:
            frame = pd.read_parquet(path, columns=columns)
            if "cancer_id" not in frame:
                frame["cancer_id"] = cancer
            if set(frame.cancer_id.astype(str)) != {cancer}:
                raise RuntimeError(f"Partition cancer mismatch: {path}")
            parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["cancer_id", "candidate_id"]).any():
        raise RuntimeError(f"Duplicate candidate rows in {table_root}")
    return result


def load_full_source_universe(
    asset_results: Path,
    cancers: list[str],
    task: str,
) -> pd.DataFrame:
    tables = asset_results / "tables"
    if task == "pathway":
        frame = _read_partitioned_cancers(
            tables / "candidate_universe",
            cancers,
            [
                "candidate_id", "cancer_id", "lncrna_id", "pathway_family_id",
                "label", "bulk_effect", "direction",
            ],
        )
        frame["target_id"] = frame.pathway_family_id.astype(str)
        frame["target_type"] = "pathway"
        frame["effect"] = pd.to_numeric(frame.bulk_effect, errors="coerce")
        frame["n_observed"] = np.nan
        # Historical candidate assets store strong evidence in ``label``.
        # Convert once at this input boundary and retain explicit provenance.
        frame["strong_evidence_label"] = pd.to_numeric(
            frame.label, errors="raise"
        ).astype(np.int8)
        frame["label"] = frame.strong_evidence_label
        frame["label_semantics"] = "PATHWAY_STRONG_EVIDENCE_V1:strong_evidence_label"
    elif task == "state":
        frame = _read_partitioned_cancers(
            tables / "strict_state_candidate",
            cancers,
            [
                "candidate_id", "cancer_id", "lncrna_id", "state_id",
                "proxy_label", "effect", "n_observed", "direction",
                "evaluation_eligibility",
            ],
        )
        frame = frame.loc[
            frame.evaluation_eligibility.astype(str).eq("ELIGIBLE")
        ].copy()
        frame["target_id"] = frame.state_id.astype(str)
        frame["target_type"] = "state"
        frame["label"] = frame.proxy_label.astype(np.int8)
    else:
        raise ValueError(f"Unsupported simple-base task: {task}")
    return frame


def _read_b1_split(
    fixed_root: Path,
    heldout_cancer: str,
    task: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = (
        fixed_root
        / PRIMARY_CONTRACT
        / "cc_hhgt"
        / f"LOCO_{heldout_cancer}"
        / f"seed_{FIXED_CANDIDATE_SAMPLING_SEED}"
        / f"prediction_{task}_raw.parquet"
    )
    frame = pd.read_parquet(path)
    require_columns(
        frame,
        ["candidate_id", "cancer_id", "lncrna_id", "proxy_label", "split"],
        f"B1 {task} prediction",
    )
    if task == "pathway":
        frame["target_id"] = frame.pathway_family_id.astype(str)
        frame["target_type"] = "pathway"
        frame["association_proxy_label"] = frame.proxy_label.astype(np.int8)
        frame["label"] = frame.association_proxy_label
        frame["label_semantics"] = "PATHWAY_PROXY_V1:association_proxy_label"
    else:
        frame["target_id"] = frame.state_id.astype(str)
        frame["target_type"] = "state"
        frame["label"] = frame.proxy_label.astype(np.int8)
    train = frame.loc[frame.split.astype(str).eq("train")].copy()
    validation = frame.loc[frame.split.astype(str).isin(["val", "validation"])].copy()
    test = frame.loc[frame.split.astype(str).eq("test")].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError(f"B1 simple-base split is empty: {path}")
    if set(test.cancer_id.astype(str)) != {heldout_cancer}:
        raise RuntimeError(f"B1 test cancer mismatch: {path}")
    return train, validation, test


def _prediction_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    methods = {**SCORE_COLUMNS, "BestSimpleLOCO": "best_simple_score"}
    groups = ["loco_cancer", "task", "target_subtype", "split"]
    for keys, group in frame.groupby(groups, observed=True, sort=True):
        for method, column in methods.items():
            values = binary_metrics(group.label, group[column])
            prevalence = float(values["positive_rate"])
            auprc = float(values["auprc"])
            rows.append(
                {
                    **dict(zip(groups, keys)),
                    "method": method,
                    "prediction_scale": "raw_probability",
                    "metric_scope": (
                        "LOCO_test" if keys[-1] == "test" else "validation"
                    ),
                    **values,
                    "auprc_over_prevalence": auprc / prevalence if prevalence > 0 else np.nan,
                    "auprc_minus_prevalence": auprc - prevalence,
                    "candidate_universe_sha256": candidate_universe_sha256(group),
                }
            )
    return pd.DataFrame(rows)


def build_best_simple(
    asset_results: Path,
    fixed_root: Path,
    output_dir: Path,
    *,
    b1_gate_path: Path,
    fold_manifest_path: Path,
    aggregation_git_commit: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output_dir / "BEST_SIMPLE_LOCO_PREDICTIONS.parquet",
        "metrics": output_dir / "BEST_SIMPLE_LOCO_METRICS.tsv",
        "residual_base": output_dir / "BEST_SIMPLE_RESIDUAL_BASE.parquet",
        "selection": output_dir / "BEST_SIMPLE_SELECTION.tsv",
        "input_audit": output_dir / "BEST_SIMPLE_INPUT_AUDIT.tsv",
        "gate": output_dir / "BEST_SIMPLE_GATE.json",
        "manifest": output_dir / "BEST_SIMPLE_SHA256.tsv",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise RuntimeError(f"BestSimple builder refuses overwrite: {existing}")
    b1_gate = json.loads(b1_gate_path.read_text(encoding="utf-8"))
    if b1_gate.get("status") != "PASS" or b1_gate.get("tasks_completed") != 12:
        raise RuntimeError("Phase B1 is not a complete 12-task PASS")
    pathway_contract = load_label_contract(
        Path(__file__).resolve().parents[1]
        / "configs" / "label_contracts" / "PATHWAY_PROXY_V1.yaml"
    )
    pathway_contract_hashes = {
        "training_target_contract_sha256": pathway_contract["_contract_sha256"],
        "evaluation_target_contract_sha256": pathway_contract["_contract_sha256"],
        "baseline_target_contract_sha256": pathway_contract["_contract_sha256"],
    }
    folds = pd.read_csv(fold_manifest_path, sep="\t")
    prediction_parts: list[pd.DataFrame] = []
    base_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    for heldout_cancer in PILOT_CANCERS:
        fold_hits = folds.loc[folds.test_cancer.astype(str).eq(heldout_cancer)]
        if len(fold_hits) != 1:
            raise RuntimeError(f"Expected one registered fold for {heldout_cancer}")
        fold = fold_hits.iloc[0]
        train_cancers = sorted(filter(None, str(fold.train_cancers).split(";")))
        for task in ("pathway", "state"):
            sampled_train, validation, test = _read_b1_split(
                fixed_root, heldout_cancer, task
            )
            if set(sampled_train.cancer_id.astype(str)) != set(train_cancers):
                raise RuntimeError(
                    f"B1 sampled training cancers drift for {heldout_cancer}/{task}"
                )
            source_universe = load_full_source_universe(
                asset_results, train_cancers, task
            )
            if task == "pathway":
                for role, frame in {
                    "training": sampled_train,
                    "validation": validation,
                    "test": test,
                }.items():
                    validate_model_target(
                        frame,
                        "association_proxy_label",
                        pathway_contract,
                        pathway_contract_hashes,
                    )
                    if not np.array_equal(
                        frame.label.to_numpy(np.int8),
                        frame.association_proxy_label.to_numpy(np.int8),
                    ):
                        raise RuntimeError(f"BestSimple {role} compatibility label drift")
                if "strong_evidence_label" not in source_universe or not np.array_equal(
                    source_universe.label.to_numpy(np.int8),
                    source_universe.strong_evidence_label.to_numpy(np.int8),
                ):
                    raise RuntimeError("BestSimple source-support label semantics are ambiguous")
            fit = fit_best_simple_loco(
                sampled_train,
                {"val": validation, "test": test},
                source_universe=source_universe,
                seed=FIXED_CANDIDATE_SAMPLING_SEED,
                selection_split="val",
            )
            selected = fit.selected_methods.copy()
            selected["loco_cancer"] = heldout_cancer
            selected["task"] = task
            selection_parts.append(selected)
            for split, prediction in fit.query_predictions.items():
                current = prediction.copy()
                current["split"] = split
                current["loco_cancer"] = heldout_cancer
                current["task"] = task
                current["fold_id"] = str(fold.fold_id)
                current["candidate_sampling_seed"] = FIXED_CANDIDATE_SAMPLING_SEED
                base_parts.append(current)
                if split == "test":
                    prediction_parts.append(current)
            oof = fit.oof_predictions.copy()
            oof["split"] = "train"
            oof["loco_cancer"] = heldout_cancer
            oof["task"] = task
            oof["fold_id"] = str(fold.fold_id)
            oof["candidate_sampling_seed"] = FIXED_CANDIDATE_SAMPLING_SEED
            base_parts.append(oof)
            audit_rows.append(
                {
                    "loco_cancer": heldout_cancer,
                    "task": task,
                    "train_cancers": "|".join(train_cancers),
                    "source_universe_rows": int(len(source_universe)),
                    "sampled_training_rows": int(len(sampled_train)),
                    "validation_rows": int(len(validation)),
                    "test_rows": int(len(test)),
                    "source_universe_candidate_sha256": candidate_universe_sha256(source_universe),
                    "sampled_training_candidate_sha256": candidate_universe_sha256(sampled_train),
                    "validation_candidate_sha256": candidate_universe_sha256(validation),
                    "test_candidate_sha256": candidate_universe_sha256(test),
                    "selection_used_test_labels": False,
                    "status": "PASS",
                }
            )
            del source_universe

    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False)
    residual_base = pd.concat(base_parts, ignore_index=True, sort=False)
    selection = pd.concat(selection_parts, ignore_index=True, sort=False)
    metrics_input = pd.concat(
        [
            part
            for part in base_parts
            if set(part.split.astype(str).unique()).issubset({"val", "test"})
        ],
        ignore_index=True,
        sort=False,
    )
    metrics = _prediction_metrics(metrics_input)
    _atomic_table(predictions, paths["predictions"])
    _atomic_table(metrics, paths["metrics"])
    _atomic_table(residual_base, paths["residual_base"])
    _atomic_table(selection, paths["selection"])
    _atomic_table(pd.DataFrame(audit_rows), paths["input_audit"])
    output_files = [
        paths["predictions"], paths["metrics"], paths["residual_base"],
        paths["selection"], paths["input_audit"],
    ]
    manifest_rows = [
        {
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in output_files
    ]
    _atomic_table(pd.DataFrame(manifest_rows), paths["manifest"])
    payload = {
        "status": "PASS",
        "stage": "PHASE_B2_BEST_SIMPLE_LOCO",
        "pilot_cancers": list(PILOT_CANCERS),
        "selection_scope": "outer_validation",
        "selection_used_test_labels": False,
        "candidate_sampling_seed": FIXED_CANDIDATE_SAMPLING_SEED,
        "full_source_universe_used_for_aggregates": True,
        "whole_cancer_crossfit": True,
        "b1_gate_sha256": file_sha256(b1_gate_path),
        "fold_manifest_sha256": file_sha256(fold_manifest_path),
        "aggregation_git_commit": aggregation_git_commit,
        "prediction_rows": int(len(predictions)),
        "residual_base_rows": int(len(residual_base)),
        "metric_rows": int(len(metrics)),
        "manifest_sha256": file_sha256(paths["manifest"]),
        "manifest_merkle_sha256": merkle_sha256(manifest_rows),
        "failures": [],
    }
    atomic_write_json(paths["gate"], payload)
    return payload
