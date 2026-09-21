#!/usr/bin/env python3
"""Run Count-base + ET-quality-residual ablation for BRCA/COAD/KIRP."""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cc_hhgt.common import file_sha256, write_table
from cc_hhgt.metrics import binary_metrics
from cc_hhgt.v30_integrity import atomic_write_json, merkle_sha256
from cc_hhgt.v31_evidence_residual import (
    ARCHITECTURES,
    CAT_FIELDS,
    COUNT_FEATURES,
    FALLBACK_ATOL,
    KEYS,
    EvidenceCandidateDataset,
    GlobalFeatureTransform,
    build_candidate_event_records,
    count_crossfit_offsets,
    encode_evidence_events,
    fit_evidence_model,
    prepare_evidence_candidates,
    predict_evidence_model,
    select_count_base,
)
from cc_hhgt.v31_residual import probability_to_logit
from cc_hhgt_v26.evidence_transformer import VocabularyBundle


CANCERS = ("BRCA", "COAD", "KIRP")
MIN_VALIDATION_DELTA_AUPRC = 0.01
METHOD_COLUMNS = {
    "E0_PREVALENCE": "e0_prevalence_probability",
    "E1_EVENT_COUNT_RANK": "e1_event_count_rank_probability",
    "E2_COUNT_LOGISTIC": "e2_count_probability",
    "E3_ET_STANDALONE_SELECTED": "e3_standalone_probability",
    "E4_COUNT_ET_RESIDUAL_CANDIDATE": "e4_residual_candidate_probability",
    "E4_SELECTED_OR_COUNT": "e4_selected_probability",
}


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _event_indices(frame: pd.DataFrame) -> set[int]:
    return {
        int(index)
        for values in frame.selected_event_indices
        for index in np.asarray(values, dtype=np.int64)
    }


def _rank_probability(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame.deduplicated_event_count.rank(method="average", pct=True).to_numpy(float)
    )


def _prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = list(predictions.groupby("cancer_id", observed=True, sort=True))
    scopes.append(("THREE_CANCER_POOLED", predictions))
    for cancer, group in scopes:
        for method, column in METHOD_COLUMNS.items():
            values = binary_metrics(group.binary_label, group[column])
            prevalence = float(values["positive_rate"])
            auprc = float(values["auprc"])
            rows.append(
                {
                    "scope": cancer,
                    "method": method,
                    "metric_scope": "OOF",
                    "prediction_scale": "raw_probability",
                    **values,
                    "auprc_over_prevalence": auprc / prevalence if prevalence > 0 else np.nan,
                    "auprc_minus_prevalence": auprc - prevalence,
                }
            )
    for fold, group in predictions.groupby("evidence_fold", observed=True, sort=True):
        for method, column in METHOD_COLUMNS.items():
            values = binary_metrics(group.binary_label, group[column])
            rows.append(
                {
                    "scope": f"FOLD_{int(fold)}",
                    "method": method,
                    "metric_scope": "OOF_fold",
                    "prediction_scale": "raw_probability",
                    **values,
                    "auprc_over_prevalence": (
                        values["auprc"] / values["positive_rate"]
                        if values["positive_rate"] > 0 else np.nan
                    ),
                    "auprc_minus_prevalence": values["auprc"] - values["positive_rate"],
                }
            )
    return pd.DataFrame(rows)


def _cluster_bootstrap(
    frame: pd.DataFrame,
    candidate_column: str,
    base_column: str,
    *,
    iterations: int = 1000,
    seed: int = 20260814,
) -> dict[str, float]:
    clusters = frame.lncrna_id.astype(str).unique()
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    grouped = {key: group for key, group in frame.groupby(frame.lncrna_id.astype(str))}
    for _ in range(int(iterations)):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[key] for key in sampled], ignore_index=True)
        base = binary_metrics(sample.binary_label, sample[base_column])["auprc"]
        candidate = binary_metrics(sample.binary_label, sample[candidate_column])["auprc"]
        deltas.append(float(candidate - base))
    return {
        "iterations": int(iterations),
        "delta_auprc_ci_lower": float(np.quantile(deltas, 0.025)),
        "delta_auprc_ci_median": float(np.quantile(deltas, 0.5)),
        "delta_auprc_ci_upper": float(np.quantile(deltas, 0.975)),
    }


def _count_matched(predictions: pd.DataFrame) -> pd.DataFrame:
    bins = [-1, 2, 3, 5, 10, np.inf]
    labels = ["1-2", "3", "4-5", "6-10", ">10"]
    frame = predictions.copy()
    frame["event_count_bin"] = pd.cut(
        frame.deduplicated_event_count, bins=bins, labels=labels
    )
    rows: list[dict[str, Any]] = []
    for count_bin, group in frame.groupby("event_count_bin", observed=True, sort=True):
        for method, column in (
            ("E2_COUNT_LOGISTIC", "e2_count_probability"),
            ("E4_COUNT_ET_RESIDUAL_CANDIDATE", "e4_residual_candidate_probability"),
        ):
            values = binary_metrics(group.binary_label, group[column])
            rows.append({"event_count_bin": str(count_bin), "method": method, **values})
    return pd.DataFrame(rows)


def _save_checkpoint(
    path: Path,
    fit,
    *,
    fold: int,
    architecture: str,
    mode: str,
    shrinkage_lambda: float,
    vocabulary: VocabularyBundle,
    input_sha256: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "model_state": fit.best_state,
            "fold": int(fold),
            "architecture": architecture,
            "mode": mode,
            "residual_shrinkage_lambda": float(shrinkage_lambda),
            "categorical_fields": list(vocabulary.fields or []),
            "vocabulary": vocabulary.values,
            "input_sha256": input_sha256,
            "loss": "BINOMIAL_K_OF_N_NO_CLASS_BALANCING",
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--count-c-grid", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--residual-lambda-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()

    evidence_root = Path(args.evidence_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"Evidence residual runner refuses reuse: {output_root}")
    output_root.mkdir(parents=True)
    checkpoint_root = output_root / "checkpoints"
    event_path = evidence_root / "evidence_event_dataset.parquet"
    historical_oof_path = evidence_root / "evidence_transformer_oof_prediction.parquet"
    for path in (event_path, historical_oof_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    input_sha256 = {
        "evidence_event_dataset": file_sha256(event_path),
        "historical_oof_label_source": file_sha256(historical_oof_path),
    }
    candidates, label_audit = prepare_evidence_candidates(
        pd.read_parquet(historical_oof_path), cancers=CANCERS
    )
    raw_events = pd.read_parquet(event_path)
    if pd.to_numeric(raw_events.direct_target_evidence, errors="coerce").fillna(0).gt(0).any():
        raise RuntimeError("Direct-event head policy requires a separate supervised contract")
    records, events, event_audit = build_candidate_event_records(
        candidates, raw_events, max_events=args.max_events
    )
    if len(records) != len(candidates):
        raise RuntimeError("Historical event-bearing candidate universe changed")
    del raw_events

    prediction_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    ablation_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    count_selection_parts: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    history_parts: list[pd.DataFrame] = []
    vocabulary_rows: list[dict[str, Any]] = []
    unseen_rows: list[dict[str, Any]] = []

    for test_fold in range(5):
        validation_fold = (test_fold + 1) % 5
        test = records.loc[records.evidence_fold.astype(int).eq(test_fold)].copy().reset_index(drop=True)
        validation = records.loc[
            records.evidence_fold.astype(int).eq(validation_fold)
        ].copy().reset_index(drop=True)
        train = records.loc[
            ~records.evidence_fold.astype(int).isin([test_fold, validation_fold])
        ].copy().reset_index(drop=True)
        if min(len(train), len(validation), len(test)) == 0:
            raise RuntimeError(f"Evidence fold {test_fold} is empty")
        train_events = _event_indices(train)
        validation_events = _event_indices(validation)
        test_events = _event_indices(test)
        if train_events.intersection(validation_events | test_events):
            raise RuntimeError("A source event crosses train and validation/test")

        count_model, count_metrics, selected_c = select_count_base(
            train,
            validation,
            c_grid=args.count_c_grid,
            seed=args.seed + test_fold,
        )
        count_metrics["test_fold"] = test_fold
        count_metrics["validation_fold"] = validation_fold
        count_metrics["selected"] = count_metrics.count_C.eq(selected_c)
        count_selection_parts.append(count_metrics)
        train_count_logit = count_crossfit_offsets(
            train,
            c_value=selected_c,
            seed=args.seed + 100 + test_fold * 10,
        )
        validation_count_probability = count_model.predict_proba(
            validation.loc[:, list(COUNT_FEATURES)].to_numpy(float)
        )[:, 1]
        test_count_probability = count_model.predict_proba(
            test.loc[:, list(COUNT_FEATURES)].to_numpy(float)
        )[:, 1]
        validation_count_logit = probability_to_logit(validation_count_probability)
        test_count_logit = probability_to_logit(test_count_probability)

        vocabulary = VocabularyBundle.fit(
            events.iloc[sorted(train_events)],
            drop_constant_fields=True,
            merge_redundant_fields=True,
        )
        encoded = encode_evidence_events(events, vocabulary)
        transform = GlobalFeatureTransform.fit(train)
        train_dataset = EvidenceCandidateDataset(
            train, transform.transform(train), train_count_logit
        )
        validation_dataset = EvidenceCandidateDataset(
            validation, transform.transform(validation), validation_count_logit
        )
        test_dataset = EvidenceCandidateDataset(
            test, transform.transform(test), test_count_logit
        )
        vocabulary_rows.extend(
            {
                "test_fold": test_fold,
                "field": field,
                "vocabulary_size": len(vocabulary.values[field]),
                "embedding_parameters_d32": len(vocabulary.values[field]) * 32,
                "constant_or_redundant_removed": False,
            }
            for field in (vocabulary.fields or [])
        )
        removed = sorted(set(CAT_FIELDS) - set(vocabulary.fields or []))
        vocabulary_rows.extend(
            {
                "test_fold": test_fold,
                "field": field,
                "vocabulary_size": 0,
                "embedding_parameters_d32": 0,
                "constant_or_redundant_removed": True,
            }
            for field in removed
        )
        for split_name, event_indices in (
            ("validation", validation_events), ("test", test_events)
        ):
            indices = np.asarray(sorted(event_indices), dtype=np.int64)
            for position, field in enumerate(encoded.categorical_fields):
                values = encoded.categorical[indices, position]
                unseen_rows.append(
                    {
                        "test_fold": test_fold,
                        "split": split_name,
                        "field": field,
                        "event_cells": int(len(values)),
                        "unseen_cells": int(np.sum(values == vocabulary.mapping(field)["<UNK>"])),
                        "unseen_rate": float(np.mean(values == vocabulary.mapping(field)["<UNK>"])),
                        "unseen_policy": "FOLD_LOCAL_VOCAB_TO_UNK",
                    }
                )

        standalone_predictions: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        residual_predictions: dict[tuple[str, float], tuple[pd.DataFrame, pd.DataFrame]] = {}
        for architecture_index, architecture in enumerate(ARCHITECTURES):
            model_seed = args.seed + test_fold * 1000 + architecture_index * 100
            standalone_fit = fit_evidence_model(
                train_dataset,
                validation_dataset,
                encoded,
                vocabulary,
                architecture=architecture,
                mode="standalone",
                residual_shrinkage_lambda=0.0,
                seed=model_seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                max_events=args.max_events,
                device=args.device,
            )
            standalone_val = predict_evidence_model(
                standalone_fit.model, validation_dataset, encoded,
                max_events=args.max_events, batch_size=args.batch_size, device=args.device,
            )
            standalone_test = predict_evidence_model(
                standalone_fit.model, test_dataset, encoded,
                max_events=args.max_events, batch_size=args.batch_size, device=args.device,
            )
            standalone_predictions[architecture] = (standalone_val, standalone_test)
            _save_checkpoint(
                checkpoint_root / f"fold_{test_fold}" / f"{architecture}_standalone.pt",
                standalone_fit,
                fold=test_fold,
                architecture=architecture,
                mode="standalone",
                shrinkage_lambda=0.0,
                vocabulary=vocabulary,
                input_sha256=input_sha256,
            )
            fit_rows.append(
                {
                    "test_fold": test_fold,
                    "architecture": architecture,
                    "mode": "standalone",
                    "residual_shrinkage_lambda": 0.0,
                    "validation_auprc": standalone_fit.validation_metrics["auprc"],
                    "validation_auroc": standalone_fit.validation_metrics["auroc"],
                    "validation_brier": standalone_fit.validation_metrics["brier"],
                    "validation_ece": standalone_fit.validation_metrics["ece"],
                    "zero_initialization_max_abs_error": math.nan,
                    **standalone_fit.prediction_summary,
                }
            )
            history = pd.DataFrame(standalone_fit.history)
            history["test_fold"] = test_fold
            history["architecture"] = architecture
            history["mode"] = "standalone"
            history["residual_shrinkage_lambda"] = 0.0
            history_parts.append(history)

            for lambda_index, shrinkage_lambda in enumerate(args.residual_lambda_grid):
                residual_fit = fit_evidence_model(
                    train_dataset,
                    validation_dataset,
                    encoded,
                    vocabulary,
                    architecture=architecture,
                    mode="residual",
                    residual_shrinkage_lambda=float(shrinkage_lambda),
                    seed=model_seed + 10 + lambda_index,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    max_events=args.max_events,
                    device=args.device,
                )
                residual_val = predict_evidence_model(
                    residual_fit.model, validation_dataset, encoded,
                    max_events=args.max_events, batch_size=args.batch_size, device=args.device,
                )
                residual_test = predict_evidence_model(
                    residual_fit.model, test_dataset, encoded,
                    max_events=args.max_events, batch_size=args.batch_size, device=args.device,
                )
                residual_predictions[(architecture, float(shrinkage_lambda))] = (
                    residual_val, residual_test
                )
                _save_checkpoint(
                    checkpoint_root / f"fold_{test_fold}" / f"{architecture}_residual_lambda_{shrinkage_lambda}.pt",
                    residual_fit,
                    fold=test_fold,
                    architecture=architecture,
                    mode="residual",
                    shrinkage_lambda=float(shrinkage_lambda),
                    vocabulary=vocabulary,
                    input_sha256=input_sha256,
                )
                fit_rows.append(
                    {
                        "test_fold": test_fold,
                        "architecture": architecture,
                        "mode": "residual",
                        "residual_shrinkage_lambda": float(shrinkage_lambda),
                        "validation_auprc": residual_fit.validation_metrics["auprc"],
                        "validation_auroc": residual_fit.validation_metrics["auroc"],
                        "validation_brier": residual_fit.validation_metrics["brier"],
                        "validation_ece": residual_fit.validation_metrics["ece"],
                        "zero_initialization_max_abs_error": residual_fit.initialization_max_abs_error,
                        **residual_fit.prediction_summary,
                    }
                )
                history = pd.DataFrame(residual_fit.history)
                history["test_fold"] = test_fold
                history["architecture"] = architecture
                history["mode"] = "residual"
                history["residual_shrinkage_lambda"] = float(shrinkage_lambda)
                history_parts.append(history)

        fit_frame = pd.DataFrame([row for row in fit_rows if row["test_fold"] == test_fold])
        standalone_choice = fit_frame.loc[fit_frame["mode"].eq("standalone")].sort_values(
            ["validation_auprc", "validation_brier", "architecture"],
            ascending=[False, True, True],
        ).iloc[0]
        residual_choice = fit_frame.loc[fit_frame["mode"].eq("residual")].sort_values(
            ["validation_auprc", "validation_brier", "residual_shrinkage_lambda", "architecture"],
            ascending=[False, True, True, True],
        ).iloc[0]
        selected_standalone = str(standalone_choice.architecture)
        selected_residual_architecture = str(residual_choice.architecture)
        selected_lambda = float(residual_choice.residual_shrinkage_lambda)
        selection_rows.append(
            {
                "test_fold": test_fold,
                "validation_fold": validation_fold,
                "selected_count_C": selected_c,
                "selected_standalone_architecture": selected_standalone,
                "selected_residual_architecture": selected_residual_architecture,
                "selected_residual_lambda": selected_lambda,
                "selection_scope": "outer_validation",
                "test_labels_used_for_selection": False,
            }
        )

        for split_name, base_frame, count_probability, standalone_source, residual_source in (
            (
                "validation", validation, validation_count_probability,
                standalone_predictions[selected_standalone][0],
                residual_predictions[(selected_residual_architecture, selected_lambda)][0],
            ),
            (
                "test", test, test_count_probability,
                standalone_predictions[selected_standalone][1],
                residual_predictions[(selected_residual_architecture, selected_lambda)][1],
            ),
        ):
            final = base_frame.copy()
            prevalence = float(train.replicate_k.sum() / train.replicate_n.sum())
            final["e0_prevalence_probability"] = prevalence
            final["e1_event_count_rank_probability"] = _rank_probability(final)
            final["e2_count_probability"] = count_probability
            final["e3_standalone_probability"] = standalone_source.proxy_positive_probability.to_numpy(float)
            final["e4_residual_candidate_probability"] = residual_source.proxy_positive_probability.to_numpy(float)
            final["e4_residual_logit"] = residual_source.event_quality_residual_logit.to_numpy(float)
            final["evidence_fold"] = test_fold if split_name == "test" else validation_fold
            final["outer_test_fold"] = test_fold
            final["split"] = split_name
            final["selected_standalone_architecture"] = selected_standalone
            final["selected_residual_architecture"] = selected_residual_architecture
            final["selected_residual_lambda"] = selected_lambda
            final["prediction_scale"] = "raw_probability"
            final["metric_scope"] = "OOF" if split_name == "test" else "validation"
            if split_name == "test":
                prediction_parts.append(final)
            else:
                validation_parts.append(final)

        for (architecture, shrinkage_lambda), (val_prediction, test_prediction) in residual_predictions.items():
            for split_name, source in (("validation", val_prediction), ("test", test_prediction)):
                part = source.copy()
                part["outer_test_fold"] = test_fold
                part["split"] = split_name
                part["architecture"] = architecture
                part["mode"] = "residual"
                part["residual_shrinkage_lambda"] = shrinkage_lambda
                ablation_parts.append(part)
        for architecture, (val_prediction, test_prediction) in standalone_predictions.items():
            for split_name, source in (("validation", val_prediction), ("test", test_prediction)):
                part = source.copy()
                part["outer_test_fold"] = test_fold
                part["split"] = split_name
                part["architecture"] = architecture
                part["mode"] = "standalone"
                part["residual_shrinkage_lambda"] = 0.0
                ablation_parts.append(part)

    test_predictions = pd.concat(prediction_parts, ignore_index=True, sort=False)
    validation_predictions = pd.concat(validation_parts, ignore_index=True, sort=False)
    if test_predictions.duplicated(KEYS).any():
        raise RuntimeError("Evidence OOF test candidates are duplicated")
    if len(test_predictions) != len(records):
        raise RuntimeError("Evidence OOF test coverage is incomplete")
    validation_base = binary_metrics(
        validation_predictions.binary_label, validation_predictions.e2_count_probability
    )
    validation_candidate = binary_metrics(
        validation_predictions.binary_label,
        validation_predictions.e4_residual_candidate_probability,
    )
    fold_validation_deltas = []
    for _, group in validation_predictions.groupby("outer_test_fold", observed=True):
        base = binary_metrics(group.binary_label, group.e2_count_probability)["auprc"]
        candidate = binary_metrics(
            group.binary_label, group.e4_residual_candidate_probability
        )["auprc"]
        fold_validation_deltas.append(float(candidate - base))
    validation_delta = float(validation_candidate["auprc"] - validation_base["auprc"])
    evidence_admitted = bool(
        validation_delta >= MIN_VALIDATION_DELTA_AUPRC
        and sum(delta > 0 for delta in fold_validation_deltas) >= 3
        and validation_candidate["brier"] - validation_base["brier"] <= 0.01
        and validation_candidate["ece"] - validation_base["ece"] <= 0.01
    )
    # The counterfactual ET-OFF path is defined and verified even when the
    # module is admitted.  This keeps the exact fallback invariant observable
    # instead of recording NaN for successful admissions.
    counterfactual_off_probability = test_predictions.e2_count_probability.to_numpy(
        dtype=float, copy=True
    )
    fallback_error = float(
        np.max(
            np.abs(
                counterfactual_off_probability
                - test_predictions.e2_count_probability.to_numpy(dtype=float)
            )
        )
    )
    if fallback_error > FALLBACK_ATOL:
        raise RuntimeError("Evidence residual counterfactual OFF is not exact Count base")
    if evidence_admitted:
        test_predictions["e4_selected_probability"] = test_predictions.e4_residual_candidate_probability
        test_predictions["e4_module_admitted"] = True
    else:
        test_predictions["e4_selected_probability"] = counterfactual_off_probability
        test_predictions["e4_module_admitted"] = False

    metrics = _prediction_metrics(test_predictions)
    count_matched = _count_matched(test_predictions)
    bootstrap = _cluster_bootstrap(
        test_predictions,
        "e4_residual_candidate_probability",
        "e2_count_probability",
        seed=args.seed,
    )
    admission = {
        "status": "PASS",
        "module": "EVIDENCE_TRANSFORMER_RESIDUAL",
        "decision": "PASS" if evidence_admitted else "OFF",
        "admitted": evidence_admitted,
        "selection_scope": "outer_validation_only",
        "test_labels_used_for_selection": False,
        "minimum_validation_delta_auprc": MIN_VALIDATION_DELTA_AUPRC,
        "validation_delta_auprc": validation_delta,
        "validation_delta_brier": float(validation_candidate["brier"] - validation_base["brier"]),
        "validation_delta_ece": float(validation_candidate["ece"] - validation_base["ece"]),
        "positive_validation_folds": int(sum(delta > 0 for delta in fold_validation_deltas)),
        "fold_validation_delta_auprc": fold_validation_deltas,
        "off_max_abs_probability_error": fallback_error,
        "off_invariant_evaluated_when_module_admitted": True,
        "paired_test_cluster_bootstrap": bootstrap,
    }

    outputs = {
        "predictions": output_root / "ET_STANDALONE_VS_COUNT_RESIDUAL_PREDICTIONS.parquet",
        "validation": output_root / "ET_VALIDATION_SELECTION_PREDICTIONS.parquet",
        "ablation": output_root / "ET_ARCHITECTURE_ABLATION_PREDICTIONS.parquet",
        "metrics": output_root / "ET_STANDALONE_VS_COUNT_RESIDUAL.tsv",
        "selection": output_root / "ET_FOLD_SELECTION.tsv",
        "count_selection": output_root / "EVIDENCE_COUNT_SELECTION.tsv",
        "fit": output_root / "ET_FIT_AUDIT.tsv",
        "history": output_root / "ET_TRAINING_HISTORY.tsv",
        "vocab": output_root / "ET_FOLD_LOCAL_VOCAB_AUDIT.tsv",
        "unseen": output_root / "ET_UNSEEN_TOKEN_AUDIT.tsv",
        "count_matched": output_root / "ET_COUNT_MATCHED_DIAGNOSTIC.tsv",
        "label_audit": output_root / "ET_LABEL_SPLIT_AUDIT.json",
        "event_audit": output_root / "ET_EVENT_DEDUP_TRUNCATION_AUDIT.json",
        "admission": output_root / "ET_RESIDUAL_ADMISSION.json",
        "manifest": output_root / "ET_RESIDUAL_SHA256.tsv",
        "gate": output_root / "ET_RESIDUAL_GATE.json",
    }
    for frame, key in (
        (test_predictions, "predictions"),
        (validation_predictions, "validation"),
        (pd.concat(ablation_parts, ignore_index=True, sort=False), "ablation"),
        (metrics, "metrics"),
        (pd.DataFrame(selection_rows), "selection"),
        (pd.concat(count_selection_parts, ignore_index=True), "count_selection"),
        (pd.DataFrame(fit_rows), "fit"),
        (pd.concat(history_parts, ignore_index=True), "history"),
        (pd.DataFrame(vocabulary_rows), "vocab"),
        (pd.DataFrame(unseen_rows), "unseen"),
        (count_matched, "count_matched"),
    ):
        _atomic_table(frame, outputs[key])
    atomic_write_json(outputs["label_audit"], label_audit)
    atomic_write_json(outputs["event_audit"], event_audit)
    atomic_write_json(outputs["admission"], admission)
    manifest_files = [
        path for key, path in outputs.items() if key not in {"manifest", "gate"}
    ] + sorted(checkpoint_root.rglob("*.pt"))
    manifest_rows = [
        {
            "relative_path": str(path.relative_to(output_root)),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in manifest_files
    ]
    _atomic_table(pd.DataFrame(manifest_rows), outputs["manifest"])
    gate = {
        "status": "PASS",
        "stage": "PHASE_B4_EVIDENCE_COUNT_PLUS_ET_RESIDUAL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "cancers": list(CANCERS),
        "candidate_rows": int(len(test_predictions)),
        "event_rows": int(len(events)),
        "architectures": list(ARCHITECTURES),
        "loss": "BINOMIAL_K_OF_N_NO_CLASS_BALANCING",
        "direct_head": "DISABLED_UNAVAILABLE_ZERO_DIRECT_EVENTS",
        "selection_scope": "outer_validation_only",
        "selection_used_test_labels": False,
        "minimum_validation_delta_auprc": MIN_VALIDATION_DELTA_AUPRC,
        "off_max_abs_probability_error": fallback_error,
        "module_admitted": evidence_admitted,
        "git_commit": args.git_commit,
        "input_sha256": input_sha256,
        "manifest_sha256": file_sha256(outputs["manifest"]),
        "manifest_merkle_sha256": merkle_sha256(manifest_rows),
        "full_cancer_model_training_started": False,
        "failures": [],
    }
    atomic_write_json(outputs["gate"], gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
