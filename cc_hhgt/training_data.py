from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import LOGGER, read_table
from .label_contract import attach_explicit_pathway_labels
from .pathway_target import pathway_target_column


def candidate_path(cfg: dict[str, Any], cancer_id: str) -> Path:
    return cfg["_results"] / "tables" / "candidate_universe" / f"cancer_id={cancer_id}" / "part-0.parquet"


def read_candidates(cfg: dict[str, Any], cancers: Iterable[str]) -> pd.DataFrame:
    target_column = pathway_target_column(cfg)
    required = {
        "candidate_id",
        "cancer_id",
        "lncrna_id",
        "pathway_family_id",
        target_column,
        "label_class",
        "direction",
        *cfg.get("training", {}).get("feature_columns", []),
    }
    frames = []
    for cancer in cancers:
        path = candidate_path(cfg, str(cancer))
        if path.exists():
            frames.append(pd.read_parquet(path, columns=sorted(required)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _proportional_quotas(counts: np.ndarray, total: int) -> np.ndarray:
    """Allocate an exact sample budget proportionally across source files."""
    counts = np.asarray(counts, dtype=np.int64)
    quotas = np.zeros(len(counts), dtype=np.int64)
    total = min(int(total), int(counts.sum()))
    if total <= 0 or not len(counts):
        return quotas
    raw = total * counts.astype(float) / float(counts.sum())
    quotas = np.minimum(np.floor(raw).astype(np.int64), counts)
    remaining = int(total - quotas.sum())
    if remaining:
        order = np.argsort(-(raw - quotas), kind="stable")
        for index in order:
            if remaining == 0:
                break
            if quotas[index] < counts[index]:
                quotas[index] += 1
                remaining -= 1
    if int(quotas.sum()) != total:
        raise RuntimeError(f"Unable to allocate candidate sample: requested={total}, allocated={int(quotas.sum())}")
    return quotas


def _required_candidate_columns(cfg: dict[str, Any]) -> list[str]:
    target_column = pathway_target_column(cfg)
    return sorted(
        {
            "candidate_id",
            "cancer_id",
            "lncrna_id",
            "pathway_family_id",
            target_column,
            "label_class",
            "direction",
            *cfg.get("training", {}).get("feature_columns", []),
        }
    )


def _attach_proxy_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = attach_explicit_pathway_labels(frame)
    # Compatibility alias for legacy model modules.  The authoritative target
    # is association_proxy_label and the equality is checked at attachment.
    out["proxy_label"] = out.association_proxy_label
    out["proxy_label_semantics"] = (
        "EXACT_PATHWAY_PATIENT_ASSOCIATION_V2:association_proxy_label"
        if "pathway_id" in out.columns
        else "PATHWAY_FAMILY_PROXY_V1:association_proxy_label"
    )
    known = out.association_proxy_label.eq(1) & out.direction.astype(str).isin(["positive", "negative"])
    out["direction_label"] = np.where(
        known,
        np.where(out.direction.astype(str).eq("positive"), 1.0, 0.0),
        np.nan,
    )
    return out


def _pu_target_counts(n_positive: int, n_unlabeled: int, cap: int, ratio: float) -> tuple[int, int]:
    if cap < 2 or ratio <= 0:
        raise ValueError(f"Invalid PU sampling configuration: cap={cap}, ratio={ratio}")
    if n_positive <= 0 or n_unlabeled <= 0:
        raise RuntimeError(
            "Pathway PU training requires both positive and unlabeled rows before sampling: "
            f"positive={n_positive}, unlabeled={n_unlabeled}"
        )
    target_positive = min(int(n_positive), max(1, int(cap // (1.0 + ratio))))
    target_unlabeled = min(int(n_unlabeled), int(round(target_positive * ratio)), cap - target_positive)
    # If unlabeled supply is the limiting class, use otherwise-idle capacity
    # for positives while retaining at least one unlabeled row.
    spare = cap - target_positive - target_unlabeled
    if spare > 0 and target_unlabeled == int(n_unlabeled):
        target_positive += min(spare, int(n_positive) - target_positive)
    return target_positive, target_unlabeled


def _sample_rows(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return frame.iloc[0:0].copy()
    return frame.sample(n=n, random_state=seed).copy() if len(frame) > n else frame.copy()


def sample_training_candidates_from_files(
    cfg: dict[str, Any], cancers: Iterable[str], seed: int
) -> pd.DataFrame:
    """Sample pathway PU rows one cancer at a time to bound host-memory use."""
    paths = [candidate_path(cfg, str(cancer)) for cancer in cancers]
    paths = [path for path in paths if path.exists()]
    if not paths:
        return pd.DataFrame()

    class_names = ("strong_positive", "weak_positive", "unlabeled")
    count_rows: list[list[int]] = []
    for path in paths:
        counts = pd.read_parquet(path, columns=["label_class"]).label_class.value_counts()
        count_rows.append([int(counts.get(name, 0)) for name in class_names])
    class_counts = np.asarray(count_rows, dtype=np.int64)
    strong_total, weak_total, unlabeled_total = map(int, class_counts.sum(axis=0))
    weak_n = min(int(weak_total), max(int(strong_total) * 2, 1000))
    cap = int(cfg["training"]["max_training_pairs_per_fold"])
    ratio = float(cfg["training"]["unlabeled_to_positive_ratio"])
    positive_n, unlabeled_n = _pu_target_counts(
        int(strong_total) + weak_n,
        int(unlabeled_total),
        cap,
        ratio,
    )
    strong_n = min(int(strong_total), positive_n)
    weak_n = min(weak_n, positive_n - strong_n)
    strong_quotas = _proportional_quotas(class_counts[:, 0], strong_n)
    weak_quotas = _proportional_quotas(class_counts[:, 1], weak_n)
    unlabeled_quotas = _proportional_quotas(class_counts[:, 2], unlabeled_n)

    required = _required_candidate_columns(cfg)
    selected: list[pd.DataFrame] = []
    for index, path in enumerate(paths):
        frame = pd.read_parquet(path, columns=required)
        for label, quota, offset in [
            ("strong_positive", int(strong_quotas[index]), 5_000),
            ("weak_positive", int(weak_quotas[index]), 10_000),
            ("unlabeled", int(unlabeled_quotas[index]), 20_000),
        ]:
            if quota <= 0:
                continue
            subset = frame.loc[frame.label_class.eq(label)]
            selected.append(_sample_rows(subset, quota, seed + offset + index))
    pre_sampled = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    return sample_training_pairs(cfg, pre_sampled, seed)


def sample_training_pairs(cfg: dict[str, Any], frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.drop_duplicates("candidate_id").copy()
    strong = frame.loc[frame.label_class.eq("strong_positive")]
    weak = frame.loc[frame.label_class.eq("weak_positive")]
    unlabeled = frame.loc[frame.label_class.eq("unlabeled")]
    weak_n = min(len(weak), max(len(strong) * 2, 1000))
    ratio = float(cfg["training"]["unlabeled_to_positive_ratio"])
    cap = int(cfg["training"]["max_training_pairs_per_fold"])
    positive_n, unlabeled_n = _pu_target_counts(len(strong) + weak_n, len(unlabeled), cap, ratio)
    strong_n = min(len(strong), positive_n)
    weak_n = min(weak_n, positive_n - strong_n)
    sampled = [
        _sample_rows(strong, strong_n, seed + 1_000),
        _sample_rows(weak, weak_n, seed + 2_000),
        _sample_rows(unlabeled, unlabeled_n, seed + 3_000),
    ]
    out = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    out = _attach_proxy_targets(out)
    observed_positive = int(out.proxy_label.sum())
    if len(out) > cap or observed_positive == 0 or observed_positive == len(out):
        raise RuntimeError(
            "Pathway PU sampler violated its fail-closed contract: "
            f"rows={len(out)}, cap={cap}, positive={observed_positive}, unlabeled={len(out) - observed_positive}"
        )
    return out


def _label_independent_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Select a fixed evaluation subset using only candidate identity."""
    if n <= 0 or len(frame) <= n:
        return frame.copy()
    keys = frame.candidate_id.astype(str) + f"|{int(seed)}"
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    indices = np.argpartition(hashes, n - 1)[:n]
    selected = frame.iloc[indices].copy()
    return selected.assign(_evaluation_hash=hashes[indices]).sort_values(
        ["_evaluation_hash", "candidate_id"], kind="stable"
    ).drop(columns="_evaluation_hash").reset_index(drop=True)


def evaluation_candidates_from_files(cfg: dict[str, Any], cancers: Iterable[str]) -> pd.DataFrame:
    """Build a prevalence-preserving evaluation set without consulting labels."""
    settings = cfg["training"]
    cap = int(settings.get("max_evaluation_pairs_per_cancer", 0))
    seed = int(settings.get("evaluation_seed", 20260726))
    policy = str(settings.get("evaluation_sampling_policy", "label_independent_deterministic"))
    if policy != "label_independent_deterministic":
        raise RuntimeError(f"Unsupported formal evaluation sampling policy: {policy}")
    frames: list[pd.DataFrame] = []
    for cancer in cancers:
        path = candidate_path(cfg, str(cancer))
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=_required_candidate_columns(cfg))
        universe_rows = len(frame)
        frame = _label_independent_sample(frame, cap, seed) if cap else frame
        frame["evaluation_sampling_policy"] = policy if cap and len(frame) < universe_rows else "full_universe"
        frame["evaluation_universe_rows"] = int(universe_rows)
        frames.append(_attach_proxy_targets(frame))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def split_for_fold(cfg: dict[str, Any], fold_row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_cancers = str(fold_row.train_cancers).split(";") if fold_row.train_cancers else []
    train = sample_training_candidates_from_files(cfg, train_cancers, int(fold_row.split_seed))
    val = evaluation_candidates_from_files(cfg, [fold_row.validation_cancer])
    test = evaluation_candidates_from_files(cfg, [fold_row.test_cancer])
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    return train, val, test
