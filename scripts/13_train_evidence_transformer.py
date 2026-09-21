#!/usr/bin/env python3
"""Train event-level evidence experts from already-built patient OOF labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cc_hhgt_v26.evidence_transformer import (  # noqa: E402
    CAT_FIELDS,
    NUM_FIELDS,
    KEYS,
    CandidateDataset,
    EvidenceStore,
    VocabularyBundle,
    clean_evidence_frame,
    fit_evidence_transformer,
    metrics,
    predict_evidence_transformer,
    stable_fold,
)

OUTPUT = Path(os.getenv("CC_HHGT_EVIDENCE_EVENT_ROOT", str(ROOT / "results" / "evidence_event_model")))


def event_indices_for_candidates(events: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
    """Return local + pan-cancer event rows applicable to candidates."""

    candidates = candidates[KEYS].drop_duplicates()
    local = events.loc[events.cancer_id.astype(str).ne("PAN_CANCER")].reset_index(names="_event_index")
    local_hit = local.merge(candidates, on=KEYS, how="inner")["_event_index"]
    global_events = events.loc[events.cancer_id.astype(str).eq("PAN_CANCER")].reset_index(names="_event_index")
    global_hit = global_events.merge(
        candidates[["lncrna_id", "pathway_family_id"]].drop_duplicates(),
        on=["lncrna_id", "pathway_family_id"],
        how="inner",
    )["_event_index"]
    return np.unique(np.concatenate([local_hit.to_numpy(np.int64), global_hit.to_numpy(np.int64)]))


def candidate_universe_sha256(frame: pd.DataFrame) -> str:
    keys = [column for column in [*KEYS, "evidence_fold"] if column in frame]
    canonical = frame[keys].astype(str).drop_duplicates().sort_values(keys, kind="stable")
    digest = hashlib.sha256()
    for row in canonical.itertuples(index=False, name=None):
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_patient_oof() -> pd.DataFrame:
    files = sorted((ROOT / "results" / "cancer_adapter").glob("*/*/seed_*/prediction.parquet"))
    if not files:
        raise FileNotFoundError("No patient adapter OOF prediction files found")
    parts = []
    columns = [*KEYS, "patient_fold_id", "seed", "test_membership_label"]
    for path in files:
        frame = pd.read_parquet(path)
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise RuntimeError(f"{path} lacks OOF label columns: {missing}")
        parts.append(frame[columns])
    raw = pd.concat(parts, ignore_index=True)
    raw["test_membership_label"] = pd.to_numeric(raw["test_membership_label"], errors="coerce")
    seed_agg = raw.groupby([*KEYS, "patient_fold_id"], observed=True, as_index=False).agg(
        fold_label=("test_membership_label", "mean"), n_seeds=("seed", "nunique")
    )
    expected_seeds = 3
    seed_agg = seed_agg.loc[seed_agg["n_seeds"].ge(expected_seeds)].copy()
    candidate = seed_agg.groupby(KEYS, observed=True, as_index=False).agg(
        label=("fold_label", "mean"),
        n_patient_folds_available=("patient_fold_id", "nunique"),
    )
    candidate = candidate.loc[candidate["label"].notna()].copy()
    return candidate


def load_score_candidates(events: pd.DataFrame, patient: pd.DataFrame) -> pd.DataFrame:
    frames = [patient[KEYS]]
    strict_path = ROOT / "results" / "strict_release" / "strict_cross_cancer_oof_prediction.parquet"
    if strict_path.exists():
        frames.append(pd.read_parquet(strict_path, columns=KEYS))
    graph_path = ROOT / "results" / "graph_expert_stacking" / "graph_expert_prediction.parquet"
    if graph_path.exists():
        frames.append(pd.read_parquet(graph_path, columns=KEYS))
    local = events.loc[events["cancer_id"].ne("PAN_CANCER"), KEYS]
    if not local.empty:
        frames.append(local)
    direct_global = events.loc[
        events["cancer_id"].eq("PAN_CANCER") & events["direct_target_evidence"].eq(1),
        ["lncrna_id", "pathway_family_id"],
    ].drop_duplicates()
    if not direct_global.empty:
        cancers = sorted(patient["cancer_id"].astype(str).unique())
        expanded = direct_global.assign(_key=1).merge(pd.DataFrame({"cancer_id": cancers, "_key": 1}), on="_key").drop(columns="_key")
        frames.append(expanded[KEYS])
    return pd.concat(frames, ignore_index=True).drop_duplicates(KEYS).reset_index(drop=True)


def add_event_counts(prediction: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach local + pan-cancer event counts to candidate predictions."""
    local = events.loc[events["cancer_id"].ne("PAN_CANCER")].groupby(
        KEYS, observed=True, as_index=False
    ).agg(
        n_local_evidence_events=("event_id", "nunique"),
        n_local_evidence_pmids=("pmid", lambda x: x[x.ne("unknown")].nunique()),
        n_local_direct_events=("direct_target_evidence", "sum"),
    )
    global_counts = events.loc[events["cancer_id"].eq("PAN_CANCER")].groupby(
        ["lncrna_id", "pathway_family_id"], observed=True, as_index=False
    ).agg(
        n_global_evidence_events=("event_id", "nunique"),
        n_global_evidence_pmids=("pmid", lambda x: x[x.ne("unknown")].nunique()),
        n_global_direct_events=("direct_target_evidence", "sum"),
    )
    result = prediction.merge(local, on=KEYS, how="left").merge(
        global_counts, on=["lncrna_id", "pathway_family_id"], how="left"
    )
    for column in [
        "n_local_evidence_events", "n_local_evidence_pmids", "n_local_direct_events",
        "n_global_evidence_events", "n_global_evidence_pmids", "n_global_direct_events",
    ]:
        result[column] = pd.to_numeric(result.get(column), errors="coerce").fillna(0)
    result["n_evidence_events"] = result["n_local_evidence_events"] + result["n_global_evidence_events"]
    result["n_evidence_pmids"] = result["n_local_evidence_pmids"] + result["n_global_evidence_pmids"]
    result["n_direct_events"] = result["n_local_direct_events"] + result["n_global_direct_events"]
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, default=OUTPUT / "evidence_event_dataset.parquet")
    ap.add_argument("--epochs", type=int, default=int(os.getenv("CC_HHGT_EVIDENCE_EPOCHS", "40")))
    ap.add_argument("--patience", type=int, default=int(os.getenv("CC_HHGT_EVIDENCE_PATIENCE", "6")))
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("CC_HHGT_EVIDENCE_BATCH_SIZE", "512")))
    ap.add_argument("--max-events", type=int, default=int(os.getenv("CC_HHGT_EVIDENCE_MAX_EVENTS", "64")))
    ap.add_argument("--seed", type=int, default=int(os.getenv("CC_HHGT_EVIDENCE_SEED", "20260731")))
    args = ap.parse_args()

    if not args.events.exists():
        raise FileNotFoundError(args.events)
    events = clean_evidence_frame(pd.read_parquet(args.events))
    patient = load_patient_oof()

    trainable = patient.copy()
    trainable["evidence_fold"] = [stable_fold(row, 5) for row in trainable[KEYS].itertuples(index=False, name=None)]
    oof_parts = []
    metric_rows = []
    for heldout in range(5):
        validation_fold = (heldout + 1) % 5
        test_frame = trainable.loc[trainable["evidence_fold"].eq(heldout)].copy()
        validation_frame = trainable.loc[trainable["evidence_fold"].eq(validation_fold)].copy()
        train_frame = trainable.loc[~trainable["evidence_fold"].isin([heldout, validation_fold])].copy()
        heldout_indices = event_indices_for_candidates(
            events, pd.concat([test_frame, validation_frame], ignore_index=True)
        )
        heldout_pmids = set(events.loc[heldout_indices, "pmid"].astype(str)) - {
            "", "unknown", "nan", "None"
        }
        train_event_indices = event_indices_for_candidates(events, train_frame)
        train_events = events.loc[train_event_indices]
        train_events = train_events.loc[~train_events.pmid.astype(str).isin(heldout_pmids)]
        vocab = VocabularyBundle.fit(
            train_events, drop_constant_fields=True, merge_redundant_fields=True
        )
        vocab_path = OUTPUT / f"evidence_vocabulary_fold_{heldout}.json"
        vocab.to_json(vocab_path)
        store = EvidenceStore(events, vocab)
        train_dataset = CandidateDataset(train_frame, store, max_events=args.max_events, forbidden_pmids=heldout_pmids)
        validation_dataset = CandidateDataset(validation_frame, store, max_events=args.max_events)
        test_dataset = CandidateDataset(test_frame, store, max_events=args.max_events)
        if len(train_dataset) < 100 or len(validation_dataset) < 20 or len(test_dataset) == 0:
            raise RuntimeError(
                f"Evidence fold {heldout} has insufficient event-bearing candidates: "
                f"train={len(train_dataset)} validation={len(validation_dataset)} test={len(test_dataset)}"
            )
        fit = fit_evidence_transformer(
            train_dataset, validation_dataset, vocab, store,
            seed=args.seed + heldout, epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size, max_events=args.max_events,
            direct_head_enabled=bool(train_events.direct_target_evidence.astype(int).eq(1).any()),
        )
        prediction = predict_evidence_transformer(
            fit.model, test_dataset, store,
            batch_size=args.batch_size, max_events=args.max_events,
        )
        prediction["evidence_fold"] = heldout
        prediction["prediction_scale"] = "raw_probability"
        prediction["metric_scope"] = "OOF"
        oof_parts.append(prediction)
        test_metrics = metrics(
            prediction["label"].to_numpy(float),
            prediction["neural_evidence_probability"].to_numpy(float),
        )
        metric_rows.append({
            "heldout_fold": heldout,
            "n_train": len(train_dataset),
            "n_validation": len(validation_dataset),
            "n_test": len(test_dataset),
            "prediction_scale": "raw_probability",
            "metric_scope": "OOF",
            "vocabulary_sha256": sha256(vocab_path),
            "unseen_token_policy": "fold-local vocabulary; unseen maps to <UNK>",
            **test_metrics,
        })

    oof = add_event_counts(pd.concat(oof_parts, ignore_index=True), events)
    oof_file = OUTPUT / "evidence_transformer_oof_prediction.parquet"
    oof.to_parquet(oof_file, index=False, compression="zstd")
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame["prediction_file_sha256"] = sha256(oof_file)
    metric_frame["candidate_universe_sha256"] = candidate_universe_sha256(oof)
    metric_frame["calibration_model_sha256"] = "NOT_APPLICABLE_RAW"
    metric_frame.to_csv(OUTPUT / "evidence_transformer_crossfit_metrics.tsv", sep="\t", index=False)

    validation_fold = 4
    final_train_frame = trainable.loc[~trainable["evidence_fold"].eq(validation_fold)].copy()
    final_validation_frame = trainable.loc[trainable["evidence_fold"].eq(validation_fold)].copy()
    final_validation_indices = event_indices_for_candidates(events, final_validation_frame)
    final_validation_pmids = set(events.loc[final_validation_indices, "pmid"].astype(str)) - {"", "unknown", "nan", "None"}
    final_train_event_indices = event_indices_for_candidates(events, final_train_frame)
    final_train_events = events.loc[final_train_event_indices]
    final_train_events = final_train_events.loc[~final_train_events.pmid.astype(str).isin(final_validation_pmids)]
    final_vocab = VocabularyBundle.fit(
        final_train_events, drop_constant_fields=True, merge_redundant_fields=True
    )
    final_vocab.to_json(OUTPUT / "evidence_vocabulary.json")
    final_store = EvidenceStore(events, final_vocab)
    final_train_dataset = CandidateDataset(final_train_frame, final_store, max_events=args.max_events, forbidden_pmids=final_validation_pmids)
    final_validation_dataset = CandidateDataset(final_validation_frame, final_store, max_events=args.max_events)
    final_fit = fit_evidence_transformer(
        final_train_dataset, final_validation_dataset, final_vocab, final_store,
        seed=args.seed + 100, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, max_events=args.max_events,
        direct_head_enabled=bool(final_train_events.direct_target_evidence.astype(int).eq(1).any()),
    )
    pd.DataFrame(final_fit.history).to_csv(OUTPUT / "evidence_transformer_history.tsv", sep="\t", index=False)
    torch.save({
        "model_state": final_fit.best_state,
        "vocab_sizes": [len(final_vocab.values[field]) for field in (final_vocab.fields or [])],
        "categorical_fields": final_vocab.fields,
        "numeric_fields": NUM_FIELDS,
        "max_events": args.max_events,
        "direct_evidence_policy": "confidence_only; direct head predicts monotonic event quality, not discovery membership",
        "indirect_evidence_policy": "eligible_for_discovery",
        "pmid_split_policy": "heldout PMID removed from training event sets",
        "seed": args.seed + 100,
    }, OUTPUT / "evidence_transformer.pt")

    score_candidates = load_score_candidates(events, patient)
    score_candidates["label"] = np.nan
    score_dataset = CandidateDataset(score_candidates, final_store, max_events=args.max_events, require_event=True)
    final_prediction = predict_evidence_transformer(
        final_fit.model, score_dataset, final_store,
        batch_size=args.batch_size, max_events=args.max_events,
    ).drop(columns="label")
    final_prediction = add_event_counts(final_prediction, events)
    final_prediction.to_parquet(OUTPUT / "evidence_transformer_prediction.parquet", index=False, compression="zstd")

    summary = {
        "status": "COMPLETED",
        "oof_rows": int(len(oof)),
        "scored_candidates": int(len(final_prediction)),
        "event_rows": int(len(events)),
        "direct_events": int(events["direct_target_evidence"].eq(1).sum()),
        "indirect_events": int(events["direct_target_evidence"].eq(0).sum()),
        "direct_evidence_used_in_discovery": False,
        "pmid_group_leakage_control": True,
        "patient_and_graph_models_retrained_by_this_stage": False,
        "underlying_graph_release": "V2.9 state-node strict retraining",
        "validation_metrics": final_fit.metrics,
    }
    (OUTPUT / "EVIDENCE_TRANSFORMER_SUCCESS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
