"""Leakage-safe sparse logistic baselines for the exact-pathway target."""
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


IDENTITY_COLUMNS = (
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
)
OUTCOME_INDEPENDENT_NUMERIC = (
    "bulk_detection_rate",
    "sc_detection_rate",
)
LABEL_DERIVED_AVAILABILITY = (
    "bulk_available",
    "sc_available",
    "ucell_available",
    "replication_available",
    "interaction_available",
    "drug_available",
    "cellline_available",
)
FORBIDDEN_PAIR_EVIDENCE = (
    "bulk_support",
    "sc_ssgsea_support",
    "replication_support",
    "direction_consistency",
    "observed_evidence_score",
    "bulk_effect",
    "bulk_fdr",
    "sc_effect",
    "sc_fdr",
)


@dataclass(frozen=True)
class SparseBaselineFit:
    model: LogisticRegression
    penalty: str
    selected_c: float
    validation_metrics: pd.DataFrame
    validation_probability: np.ndarray
    test_probability: np.ndarray
    feature_contract: dict


def _records(frame: pd.DataFrame) -> Iterator[dict[str, float]]:
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Exact-pathway baseline lacks identity columns: {missing}")
    numeric = {
        column: pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(float)
        if column in frame
        else np.zeros(len(frame), dtype=float)
        for column in OUTCOME_INDEPENDENT_NUMERIC
    }
    for index, row in enumerate(
        frame[list(IDENTITY_COLUMNS)].astype(str).itertuples(index=False, name=None)
    ):
        lncrna, pathway, family = row
        record = {
            f"lncrna::{lncrna}": 1.0,
            f"pathway::{pathway}": 1.0,
            f"family::{family}": 1.0,
            f"pair::{lncrna}::{pathway}": 1.0,
        }
        for column in OUTCOME_INDEPENDENT_NUMERIC:
            record[f"numeric::{column}"] = float(numeric[column][index])
        yield record


def hashed_design(frame: pd.DataFrame, n_features: int = 2**20) -> csr_matrix:
    """Create the fixed, outcome-independent sparse design matrix.

    ``FeatureHasher`` is stateless, so validation/test transformations cannot
    alter a fitted vocabulary.  Cancer identifiers and pair evidence are
    deliberately absent from the record generator.
    """
    if n_features < 1024:
        raise ValueError("Sparse baseline hash dimension must be at least 1024")
    hasher = FeatureHasher(
        n_features=int(n_features), input_type="dict", alternate_sign=False
    )
    return hasher.transform(_records(frame)).tocsr()


def _binary_metrics(labels: Iterable, scores: Iterable) -> dict[str, float]:
    y = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy(float)
    p = pd.to_numeric(pd.Series(scores), errors="coerce").to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid].astype(int)
    p = p[valid]
    if len(y) == 0 or np.unique(y).size != 2:
        raise RuntimeError("Sparse baseline evaluation requires both proxy classes")
    prevalence = float(y.mean())
    auprc = float(average_precision_score(y, p))
    return {
        "n": int(len(y)),
        "positive_prevalence": prevalence,
        "auroc": float(roc_auc_score(y, p)),
        "auprc": auprc,
        "auprc_lift": auprc - prevalence,
    }


def _sample_weights(frame: pd.DataFrame, weak_positive_weight: float) -> np.ndarray:
    if "label_class" not in frame:
        return np.ones(len(frame), dtype=float)
    label_class = frame.label_class.astype(str)
    weights = np.ones(len(frame), dtype=float)
    weights[label_class.eq("weak_positive").to_numpy()] = float(weak_positive_weight)
    return weights


def fit_sparse_logistic_baseline(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    penalty: str,
    c_grid: Iterable[float] = (0.01, 0.1, 1.0, 10.0),
    seed: int,
    weak_positive_weight: float = 0.5,
    n_features: int = 2**20,
    max_iter: int = 2000,
) -> SparseBaselineFit:
    """Fit L1 (LASSO) or L2 (Ridge) logistic regression using validation only."""
    if penalty not in {"l1", "l2"}:
        raise ValueError("penalty must be l1 or l2")
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        if "proxy_label" not in frame:
            raise ValueError(f"{name} exact-pathway candidates lack proxy_label")
        if frame.candidate_id.astype(str).duplicated().any():
            raise RuntimeError(f"{name} contains duplicate candidate IDs")
    c_values = sorted({float(value) for value in c_grid if float(value) > 0})
    if not c_values:
        raise ValueError("c_grid must contain a positive penalty inverse")

    x_train = hashed_design(train, n_features)
    x_validation = hashed_design(validation, n_features)
    x_test = hashed_design(test, n_features)
    y_train = pd.to_numeric(train.proxy_label, errors="raise").to_numpy(int)
    y_validation = pd.to_numeric(validation.proxy_label, errors="raise").to_numpy(int)
    if np.unique(y_train).size != 2:
        raise RuntimeError("Sparse baseline training requires both proxy classes")
    weights = _sample_weights(train, weak_positive_weight)

    candidates: list[tuple[tuple[float, float, float], LogisticRegression, np.ndarray, dict]] = []
    for c_value in c_values:
        model = LogisticRegression(
            penalty=penalty,
            C=c_value,
            solver="saga",
            max_iter=int(max_iter),
            tol=1e-4,
            fit_intercept=True,
            random_state=int(seed),
            n_jobs=1,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train, y_train, sample_weight=weights)
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        probability = model.predict_proba(x_validation)[:, 1]
        metrics = _binary_metrics(y_validation, probability)
        metrics.update(
            {
                "penalty": penalty,
                "C": c_value,
                "converged": converged,
                "nonzero_coefficients": int(np.count_nonzero(model.coef_)),
            }
        )
        # Validation AUPRC lift is primary, AUROC secondary. Prefer stronger
        # regularization (smaller C) only after exact metric ties.
        key = (float(metrics["auprc_lift"]), float(metrics["auroc"]), -c_value)
        candidates.append((key, model, probability, metrics))
    eligible = [item for item in candidates if bool(item[3]["converged"])]
    if not eligible:
        raise RuntimeError(f"No converged {penalty} logistic baseline in the registered C grid")
    _, selected_model, validation_probability, selected_metrics = max(
        eligible, key=lambda item: item[0]
    )
    test_probability = selected_model.predict_proba(x_test)[:, 1]
    metric_frame = pd.DataFrame([item[3] for item in candidates]).sort_values("C")
    contract = {
        "target": "cancer_x_lncrna_x_exact_pathway association_proxy_label",
        "penalty": penalty,
        "selection_split": "validation_only",
        "test_used_for_tuning": False,
        "cancer_id_feature": False,
        "identity_features": [
            "lncrna_id", "pathway_id", "pathway_family_id", "lncrna_id_x_pathway_id"
        ],
        "numeric_features": list(OUTCOME_INDEPENDENT_NUMERIC),
        "forbidden_label_derived_availability": list(LABEL_DERIVED_AVAILABILITY),
        "forbidden_pair_evidence": list(FORBIDDEN_PAIR_EVIDENCE),
        "hash_dimension": int(n_features),
        "hash_alternate_sign": False,
        "selected_validation_metrics": selected_metrics,
    }
    return SparseBaselineFit(
        model=selected_model,
        penalty=penalty,
        selected_c=float(selected_model.C),
        validation_metrics=metric_frame,
        validation_probability=validation_probability,
        test_probability=test_probability,
        feature_contract=contract,
    )
