from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .contracts import (
    FORBIDDEN_PRIMARY_FEATURES,
    TARGET_KEYS,
    assert_matched_candidate_universe,
    candidate_key_sha256,
)


SAFE_NUMERIC_FEATURES = (
    "discovery_effect",
    "discovery_neglog10_fdr",
    "detection_rate",
    "association_n_samples",
    "cross_cancer_support_frequency",
    "cross_cancer_direction_consistency",
    "cross_cancer_i2",
)


@dataclass(frozen=True)
class MatchedBaselineResult:
    penalty: str
    selected_c: float
    model: LogisticRegression
    validation_probability: np.ndarray
    test_probability: np.ndarray
    selection_table: pd.DataFrame
    candidate_sha256: str
    feature_contract: dict


def prepare_discovery_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(TARGET_KEYS) | {"pathway_family_id", "discovery_effect", "discovery_fdr"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Discovery features lack columns: {missing}")
    result = frame.copy()
    fdr = pd.to_numeric(result.discovery_fdr, errors="coerce").clip(lower=1e-300, upper=1.0)
    result["discovery_neglog10_fdr"] = -np.log10(fdr)
    return result


def attach_replication_labels(
    discovery_features: pd.DataFrame,
    replication_labels: pd.DataFrame,
    *,
    label_column: str = "proxy_label",
) -> pd.DataFrame:
    """Attach validation/test-patient labels without importing their features."""

    required_label = set(TARGET_KEYS) | {label_column}
    if missing := sorted(required_label - set(replication_labels.columns)):
        raise ValueError(f"Replication label table lacks columns: {missing}")
    discovery = prepare_discovery_features(discovery_features)
    labels = replication_labels[list(TARGET_KEYS) + [label_column]].copy()
    if labels.duplicated(list(TARGET_KEYS)).any():
        raise RuntimeError("Replication label table has duplicate candidate keys")
    before = candidate_key_sha256(discovery)
    result = discovery.drop(columns=[label_column], errors="ignore").merge(
        labels,
        on=list(TARGET_KEYS),
        how="inner",
        validate="one_to_one",
    )
    if candidate_key_sha256(result) != before:
        raise RuntimeError("Replication labels do not cover the discovery candidate universe")
    result["proxy_label"] = pd.to_numeric(result[label_column], errors="raise").astype(int)
    return result


def _safe_records(frame: pd.DataFrame) -> Iterator[dict[str, float]]:
    required = set(TARGET_KEYS) | {"pathway_family_id"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Baseline candidate table lacks columns: {missing}")
    numeric = {
        column: pd.to_numeric(frame.get(column, 0.0), errors="coerce")
        if column in frame
        else pd.Series(0.0, index=frame.index)
        for column in SAFE_NUMERIC_FEATURES
    }
    numeric = {name: values.fillna(0.0).to_numpy(float) for name, values in numeric.items()}
    identities = frame[["cancer_id", "lncrna_id", "pathway_id", "pathway_family_id"]].astype(str)
    for index, (cancer, lnc, pathway, family) in enumerate(
        identities.itertuples(index=False, name=None)
    ):
        record = {
            f"cancer::{cancer}": 1.0,
            f"lncrna::{lnc}": 1.0,
            f"pathway::{pathway}": 1.0,
            f"family::{family}": 1.0,
            f"cancer_lncrna::{cancer}::{lnc}": 1.0,
            f"cancer_pathway::{cancer}::{pathway}": 1.0,
            f"lncrna_pathway::{lnc}::{pathway}": 1.0,
        }
        for column in SAFE_NUMERIC_FEATURES:
            record[f"numeric::{column}"] = float(numeric[column][index])
        yield record


def build_safe_hashed_design(frame: pd.DataFrame, *, n_features: int = 2**20) -> csr_matrix:
    if n_features < 1024:
        raise ValueError("n_features must be at least 1024")
    # The design builder chooses its own explicit allow-list.  Merely adding a
    # forbidden evidence column to the source frame cannot change the matrix.
    hasher = FeatureHasher(
        n_features=int(n_features), input_type="dict", alternate_sign=False
    )
    return hasher.transform(_safe_records(frame)).tocsr()


def assert_no_forbidden_requested_features(feature_names: Iterable[str]) -> None:
    bad = sorted(set(map(str, feature_names)) & set(FORBIDDEN_PRIMARY_FEATURES))
    if bad:
        raise RuntimeError(f"Forbidden primary baseline features requested: {bad}")


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if np.unique(labels).size != 2:
        raise RuntimeError("Matched baseline evaluation requires both classes")
    prevalence = float(labels.mean())
    auprc = float(average_precision_score(labels, probability))
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": auprc,
        "auprc_lift": auprc - prevalence,
        "prevalence": prevalence,
    }


def fit_matched_logistic(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    penalty: str,
    seed: int = 20260726,
    c_grid: Iterable[float] = (0.01, 0.1, 1.0, 10.0),
    n_features: int = 2**20,
    max_iter: int = 2000,
) -> MatchedBaselineResult:
    """Fit a matched L1 LASSO or L2 Ridge candidate-replication baseline.

    This routine is implemented for the authorized training phase.  The
    CODE_ONLY CLI never calls it.
    """

    if penalty not in {"l1", "l2"}:
        raise ValueError("penalty must be l1 or l2")
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        if "proxy_label" not in frame:
            raise ValueError(f"{name} lacks proxy_label")
        if frame.duplicated(list(TARGET_KEYS)).any():
            raise RuntimeError(f"{name} has duplicate exact-pathway candidates")
    candidate_sha = assert_matched_candidate_universe(
        train=train, validation=validation, test=test
    )
    x_train = build_safe_hashed_design(train, n_features=n_features)
    x_validation = build_safe_hashed_design(validation, n_features=n_features)
    x_test = build_safe_hashed_design(test, n_features=n_features)
    y_train = pd.to_numeric(train.proxy_label, errors="raise").to_numpy(int)
    y_validation = pd.to_numeric(validation.proxy_label, errors="raise").to_numpy(int)
    if np.unique(y_train).size != 2:
        raise RuntimeError("Training candidates require both proxy classes")
    weight = np.where(
        train.get("label_class", pd.Series("unlabeled", index=train.index)).astype(str).eq("weak_positive"),
        0.35,
        np.where(y_train == 0, 0.12, 1.0),
    )
    candidates = []
    for c_value in sorted({float(value) for value in c_grid if float(value) > 0}):
        model = LogisticRegression(
            penalty=penalty,
            C=c_value,
            solver="saga",
            max_iter=int(max_iter),
            random_state=int(seed),
            n_jobs=1,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train, y_train, sample_weight=weight)
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        probability = model.predict_proba(x_validation)[:, 1]
        metrics = _metrics(y_validation, probability)
        metrics.update({"C": c_value, "converged": converged})
        candidates.append((metrics, model, probability))
    eligible = [item for item in candidates if item[0]["converged"]]
    if not eligible:
        raise RuntimeError(f"No converged {penalty} model")
    selected = max(
        eligible,
        key=lambda item: (item[0]["auprc_lift"], item[0]["auroc"], -item[0]["C"]),
    )
    metrics, model, validation_probability = selected
    feature_contract = {
        "allowed_numeric": list(SAFE_NUMERIC_FEATURES),
        "allowed_identity": [
            "cancer_id",
            "lncrna_id",
            "pathway_id",
            "pathway_family_id",
            "all_two_way_interactions",
        ],
        "forbidden_three_way_identity": "cancer_x_lncrna_x_pathway",
        "forbidden_primary_features": sorted(FORBIDDEN_PRIMARY_FEATURES),
        "selection_scope": "validation_only",
        "test_used_for_tuning": False,
        "candidate_sha256": candidate_sha,
    }
    return MatchedBaselineResult(
        penalty=penalty,
        selected_c=float(metrics["C"]),
        model=model,
        validation_probability=validation_probability,
        test_probability=model.predict_proba(x_test)[:, 1],
        selection_table=pd.DataFrame([item[0] for item in candidates]),
        candidate_sha256=candidate_sha,
        feature_contract=feature_contract,
    )
