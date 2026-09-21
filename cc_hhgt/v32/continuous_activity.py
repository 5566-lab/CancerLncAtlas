"""Fresh V3.2 patient-level continuous exact-pathway activity head.

The exact-pathway classifier and this continuous endpoint have different
estimands. This module therefore never changes the primary association
probability. It fits a new fold-local PCA--ridge head from patient lncRNA
expression to continuous exact-pathway activity, tunes regularisation on the
validation patients, refits on train+validation, and evaluates the outer test
patients exactly once.

No historical checkpoint, prediction, ranking, or fitted feature is accepted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
MODULE_ID = "continuous_pathway_activity"
CHECKPOINT_FORMAT = "CC_HHGT_V3_2_CONTINUOUS_ACTIVITY_PCA_RIDGE_V1"


class ContinuousActivityError(RuntimeError):
    """Raised when a continuous-head contract is violated."""


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _string_array(values: Sequence[str], label: str) -> np.ndarray:
    result = np.asarray([str(value).strip() for value in values], dtype=str)
    if result.ndim != 1 or len(result) == 0 or np.any(result == ""):
        raise ContinuousActivityError(f"{label} must contain non-empty identifiers")
    if len(np.unique(result)) != len(result):
        raise ContinuousActivityError(f"{label} contains duplicates")
    return result


def audit_frozen_lnc_embeddings(embeddings: pd.DataFrame) -> dict[str, Any]:
    """Report whether exported lncRNA embeddings can distinguish lncRNAs."""

    feature_columns = sorted(
        column for column in embeddings.columns if str(column).startswith("core_feature_")
    )
    if "node_id" not in embeddings or not feature_columns:
        raise ContinuousActivityError("Frozen lncRNA embedding table lacks node_id/features")
    if embeddings.node_id.astype(str).duplicated().any():
        raise ContinuousActivityError("Frozen lncRNA embedding table duplicates node_id")
    values = embeddings[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ContinuousActivityError("Frozen lncRNA embeddings contain non-finite values")
    unique_vectors = int(np.unique(values, axis=0).shape[0])
    rank = int(np.linalg.matrix_rank(values))
    varying_features = int(np.count_nonzero(np.var(values, axis=0) > 1.0e-12))
    usable = unique_vectors > 1 and varying_features > 0
    return {
        "status": "PASS" if usable else "UNUSABLE_FOR_PATIENT_PROJECTION",
        "rows": int(len(embeddings)),
        "feature_count": int(len(feature_columns)),
        "unique_embedding_vectors": unique_vectors,
        "matrix_rank": rank,
        "varying_feature_count": varying_features,
        "usable_for_patient_expression_projection": bool(usable),
        "fallback": None if usable else "FRESH_FOLD_LOCAL_EXPRESSION_PCA",
    }


@dataclass(frozen=True)
class ContinuousActivityConfig:
    n_components: int = 64
    ridge_alphas: tuple[float, ...] = (
        1.0e-4,
        1.0e-3,
        1.0e-2,
        1.0e-1,
        1.0,
        10.0,
        100.0,
        1_000.0,
        10_000.0,
    )
    top_attributions: int = 25
    variance_floor: float = 1.0e-8

    def validate(self) -> None:
        if self.n_components < 1 or self.top_attributions < 1:
            raise ContinuousActivityError("PCA dimensions and attribution count must be positive")
        alphas = np.asarray(self.ridge_alphas, dtype=float)
        if len(alphas) < 2 or not np.isfinite(alphas).all() or np.any(alphas <= 0):
            raise ContinuousActivityError("ridge_alphas must contain at least two positive values")
        if len(np.unique(alphas)) != len(alphas):
            raise ContinuousActivityError("ridge_alphas contains duplicates")
        if not np.isfinite(self.variance_floor) or self.variance_floor <= 0:
            raise ContinuousActivityError("variance_floor must be finite and positive")


@dataclass(frozen=True)
class ContinuousActivityModel:
    feature_ids: np.ndarray
    pathway_ids: np.ndarray
    expression_median: np.ndarray
    expression_mean: np.ndarray
    expression_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    score_mean: np.ndarray
    score_scale: np.ndarray
    outcome_mean: np.ndarray
    ridge_coefficients: np.ndarray
    selected_alpha: float
    seed: int
    validation_normalized_mse: float
    initial_parameter_sha256: str
    final_parameter_sha256: str
    n_train: int
    n_validation: int
    n_test: int

    def metadata(self) -> dict[str, Any]:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "module_id": MODULE_ID,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "model": "fold_local_expression_pca_multioutput_ridge",
            "initialization": "random",
            "new_parameters_from_scratch": True,
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "primary_exact_pathway_ranking_changed": False,
            "outer_test_used_for_tuning": False,
            "selected_alpha": float(self.selected_alpha),
            "seed": int(self.seed),
            "validation_normalized_mse": float(self.validation_normalized_mse),
            "initial_parameter_sha256": self.initial_parameter_sha256,
            "final_parameter_sha256": self.final_parameter_sha256,
            "n_train": int(self.n_train),
            "n_validation": int(self.n_validation),
            "n_test": int(self.n_test),
            "n_features": int(len(self.feature_ids)),
            "n_components": int(self.pca_components.shape[0]),
            "n_pathways": int(len(self.pathway_ids)),
        }

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(self.metadata(), sort_keys=True, separators=(",", ":"))
        np.savez_compressed(
            destination,
            metadata_json=np.asarray(metadata),
            feature_ids=self.feature_ids,
            pathway_ids=self.pathway_ids,
            expression_median=self.expression_median,
            expression_mean=self.expression_mean,
            expression_scale=self.expression_scale,
            pca_mean=self.pca_mean,
            pca_components=self.pca_components,
            score_mean=self.score_mean,
            score_scale=self.score_scale,
            outcome_mean=self.outcome_mean,
            ridge_coefficients=self.ridge_coefficients,
        )


@dataclass(frozen=True)
class ContinuousActivityFit:
    model: ContinuousActivityModel
    test_prediction: np.ndarray
    metrics: pd.DataFrame
    attributions: pd.DataFrame


def _matrix(value: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ContinuousActivityError(f"{label} must be a non-empty matrix")
    return result


def _impute_and_scale_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    median = np.nanmedian(x, axis=0)
    median[~np.isfinite(median)] = 0.0
    imputed = np.where(np.isfinite(x), x, median)
    mean = imputed.mean(axis=0)
    scale = imputed.std(axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return (imputed - mean) / scale, median, mean, scale


def _impute_and_scale_apply(
    x: np.ndarray, median: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (np.where(np.isfinite(x), x, median) - mean) / scale


def _fit_pca(
    x: np.ndarray, requested_components: int, seed: int
) -> tuple[PCA, np.ndarray]:
    components = min(int(requested_components), x.shape[0] - 1, x.shape[1])
    if components < 1:
        raise ContinuousActivityError("Too few training patients/features for PCA")
    solver = "randomized" if components < min(x.shape) else "full"
    pca = PCA(n_components=components, svd_solver=solver, random_state=int(seed))
    scores = pca.fit_transform(x)
    if not np.isfinite(scores).all():
        raise ContinuousActivityError("PCA produced non-finite scores")
    return pca, scores


def _ridge_coefficients(z: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    gram = z.T @ z
    gram.flat[:: len(gram) + 1] += float(alpha)
    try:
        return np.linalg.solve(gram, z.T @ y)
    except np.linalg.LinAlgError as exc:
        raise ContinuousActivityError("Ridge normal equations are singular") from exc


def _normalized_mse(
    observed: np.ndarray, predicted: np.ndarray, train_variance: np.ndarray, floor: float
) -> tuple[float, np.ndarray]:
    by_outcome = np.mean(np.square(predicted - observed), axis=0)
    denominator = np.maximum(train_variance, float(floor))
    return float(np.mean(by_outcome / denominator)), by_outcome


def _vector_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, np.ndarray]:
    residual = observed - predicted
    rmse = np.sqrt(np.mean(np.square(residual), axis=0))
    centered = observed - observed.mean(axis=0)
    denominator = np.sum(np.square(centered), axis=0)
    r2 = np.full(observed.shape[1], np.nan, dtype=float)
    valid_r2 = denominator > 0
    r2[valid_r2] = 1.0 - np.sum(np.square(residual), axis=0)[valid_r2] / denominator[valid_r2]

    observed_rank = pd.DataFrame(observed).rank(axis=0, method="average").to_numpy(float)
    predicted_rank = pd.DataFrame(predicted).rank(axis=0, method="average").to_numpy(float)
    observed_rank -= observed_rank.mean(axis=0)
    predicted_rank -= predicted_rank.mean(axis=0)
    rank_denominator = np.sqrt(
        np.sum(np.square(observed_rank), axis=0)
        * np.sum(np.square(predicted_rank), axis=0)
    )
    spearman = np.full(observed.shape[1], np.nan, dtype=float)
    valid_rank = rank_denominator > 0
    spearman[valid_rank] = np.sum(
        observed_rank[:, valid_rank] * predicted_rank[:, valid_rank], axis=0
    ) / rank_denominator[valid_rank]
    return {"test_r2": r2, "test_rmse": rmse, "test_spearman": spearman}


def _top_attributions(
    feature_ids: np.ndarray,
    pathway_ids: np.ndarray,
    standardized_coefficients: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    if standardized_coefficients.shape != (len(feature_ids), len(pathway_ids)):
        raise ContinuousActivityError("Attribution coefficient dimensions disagree")
    take = min(int(top_n), len(feature_ids))
    rows: list[pd.DataFrame] = []
    for pathway_index, pathway_id in enumerate(pathway_ids):
        coefficient = standardized_coefficients[:, pathway_index]
        candidate = np.argpartition(np.abs(coefficient), -take)[-take:]
        candidate = candidate[
            np.lexsort((feature_ids[candidate], -np.abs(coefficient[candidate])))
        ]
        rows.append(
            pd.DataFrame(
                {
                    "pathway_id": str(pathway_id),
                    "lncrna_id": feature_ids[candidate],
                    "standardized_coefficient": coefficient[candidate],
                    "absolute_standardized_coefficient": np.abs(coefficient[candidate]),
                    "attribution_rank": np.arange(1, len(candidate) + 1, dtype=np.int16),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_fresh_pca_ridge_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    feature_ids: Sequence[str],
    pathway_ids: Sequence[str],
    seed: int,
    config: ContinuousActivityConfig = ContinuousActivityConfig(),
) -> ContinuousActivityFit:
    """Fit a leakage-safe fresh head and return outer-test predictions."""

    config.validate()
    x_train = _matrix(x_train, "x_train")
    y_train = _matrix(y_train, "y_train")
    x_validation = _matrix(x_validation, "x_validation")
    y_validation = _matrix(y_validation, "y_validation")
    x_test = _matrix(x_test, "x_test")
    y_test = _matrix(y_test, "y_test")
    features = _string_array(feature_ids, "feature_ids")
    pathways = _string_array(pathway_ids, "pathway_ids")
    if not (x_train.shape[1] == x_validation.shape[1] == x_test.shape[1] == len(features)):
        raise ContinuousActivityError("Expression feature dimensions disagree")
    if not (y_train.shape[1] == y_validation.shape[1] == y_test.shape[1] == len(pathways)):
        raise ContinuousActivityError("Pathway outcome dimensions disagree")
    if not (
        len(x_train) == len(y_train)
        and len(x_validation) == len(y_validation)
        and len(x_test) == len(y_test)
    ):
        raise ContinuousActivityError("Expression and pathway patient counts disagree")
    if min(len(x_train), len(x_validation), len(x_test)) < 3:
        raise ContinuousActivityError("Every split requires at least three patients")
    if not all(np.isfinite(value).all() for value in (y_train, y_validation, y_test)):
        raise ContinuousActivityError("Continuous pathway outcomes must never be imputed")

    train_z, train_median, train_mean, train_scale = _impute_and_scale_fit(x_train)
    validation_z = _impute_and_scale_apply(
        x_validation, train_median, train_mean, train_scale
    )
    pca, train_scores = _fit_pca(train_z, config.n_components, int(seed))
    validation_scores = pca.transform(validation_z)
    score_mean = train_scores.mean(axis=0)
    score_scale = train_scores.std(axis=0, ddof=0)
    score_scale[score_scale <= 0] = 1.0
    train_scores = (train_scores - score_mean) / score_scale
    validation_scores = (validation_scores - score_mean) / score_scale
    outcome_mean = y_train.mean(axis=0)
    train_outcome = y_train - outcome_mean
    train_variance = np.var(y_train, axis=0, ddof=0)

    candidates: list[tuple[float, float, np.ndarray]] = []
    for alpha in sorted(map(float, config.ridge_alphas), reverse=True):
        coefficient = _ridge_coefficients(train_scores, train_outcome, alpha)
        prediction = outcome_mean + validation_scores @ coefficient
        normalized, by_outcome = _normalized_mse(
            y_validation, prediction, train_variance, config.variance_floor
        )
        candidates.append((normalized, alpha, by_outcome))
    selected_normalized, selected_alpha, validation_mse = min(
        candidates, key=lambda item: (item[0], -item[1])
    )

    fit_x = np.concatenate([x_train, x_validation], axis=0)
    fit_y = np.concatenate([y_train, y_validation], axis=0)
    fit_z, median, mean, scale = _impute_and_scale_fit(fit_x)
    test_z = _impute_and_scale_apply(x_test, median, mean, scale)
    final_pca, fit_scores = _fit_pca(
        fit_z, config.n_components, int(seed) + 1_000_003
    )
    test_scores = final_pca.transform(test_z)
    final_score_mean = fit_scores.mean(axis=0)
    final_score_scale = fit_scores.std(axis=0, ddof=0)
    final_score_scale[final_score_scale <= 0] = 1.0
    fit_scores = (fit_scores - final_score_mean) / final_score_scale
    test_scores = (test_scores - final_score_mean) / final_score_scale
    final_outcome_mean = fit_y.mean(axis=0)
    coefficient = _ridge_coefficients(
        fit_scores, fit_y - final_outcome_mean, selected_alpha
    )
    prediction = final_outcome_mean + test_scores @ coefficient
    if not np.isfinite(prediction).all():
        raise ContinuousActivityError("Continuous head produced non-finite predictions")

    initial_sha = hashlib.sha256(
        f"{CHECKPOINT_FORMAT}|{int(seed)}|{len(features)}|{len(pathways)}|"
        f"{final_pca.n_components_}".encode("utf-8")
    ).hexdigest()
    final_sha = _array_sha256(
        median,
        mean,
        scale,
        final_pca.mean_,
        final_pca.components_,
        final_score_mean,
        final_score_scale,
        final_outcome_mean,
        coefficient,
    )
    if initial_sha == final_sha:
        raise ContinuousActivityError("Fresh head parameter hash did not change")
    model = ContinuousActivityModel(
        feature_ids=features,
        pathway_ids=pathways,
        expression_median=median,
        expression_mean=mean,
        expression_scale=scale,
        pca_mean=np.asarray(final_pca.mean_, dtype=float),
        pca_components=np.asarray(final_pca.components_, dtype=float),
        score_mean=np.asarray(final_score_mean, dtype=float),
        score_scale=np.asarray(final_score_scale, dtype=float),
        outcome_mean=np.asarray(final_outcome_mean, dtype=float),
        ridge_coefficients=np.asarray(coefficient, dtype=float),
        selected_alpha=float(selected_alpha),
        seed=int(seed),
        validation_normalized_mse=float(selected_normalized),
        initial_parameter_sha256=initial_sha,
        final_parameter_sha256=final_sha,
        n_train=int(len(x_train)),
        n_validation=int(len(x_validation)),
        n_test=int(len(x_test)),
    )
    metric_values = _vector_metrics(y_test, prediction)
    metrics = pd.DataFrame(
        {
            "pathway_id": pathways,
            "selected_alpha": float(selected_alpha),
            "validation_mse": validation_mse,
            **metric_values,
            "n_train": int(len(x_train)),
            "n_validation": int(len(x_validation)),
            "n_test": int(len(x_test)),
            "outer_test_used_for_tuning": False,
        }
    )
    original_scale_coefficients = final_pca.components_.T @ (
        coefficient / final_score_scale[:, None]
    )
    attributions = _top_attributions(
        features, pathways, original_scale_coefficients, config.top_attributions
    )
    return ContinuousActivityFit(
        model=model,
        test_prediction=prediction,
        metrics=metrics,
        attributions=attributions,
    )


__all__ = [
    "ANALYSIS_VERSION",
    "MODULE_ID",
    "CHECKPOINT_FORMAT",
    "ContinuousActivityError",
    "ContinuousActivityConfig",
    "ContinuousActivityModel",
    "ContinuousActivityFit",
    "audit_frozen_lnc_embeddings",
    "fit_fresh_pca_ridge_head",
]
