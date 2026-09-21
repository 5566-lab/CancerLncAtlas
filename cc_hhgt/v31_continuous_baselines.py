from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso, Ridge, lasso_path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def _impute_from_train(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    medians = np.nanmedian(train, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    result = [np.where(np.isfinite(train), train, medians)]
    result.extend(np.where(np.isfinite(frame), frame, medians) for frame in others)
    return tuple(result)


def _finite_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Continuous baseline metrics require finite values")
    correlation = (
        float(spearmanr(observed, predicted).statistic)
        if len(observed) >= 3
        and np.unique(observed).size > 1
        and np.unique(predicted).size > 1
        else float("nan")
    )
    return {
        "test_r2": float(r2_score(observed, predicted)),
        "test_rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "test_spearman": correlation,
    }


def fit_nested_continuous_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    lasso_alpha_ratios: Iterable[float] = tuple(np.logspace(0, -3, 30)),
    ridge_alphas: Iterable[float] = tuple(np.logspace(4, -4, 25)),
) -> dict[str, dict[str, Any]]:
    """Tune on validation only, refit on train+validation, evaluate test once."""

    arrays = [
        np.asarray(value, dtype=float)
        for value in (x_train, y_train, x_validation, y_validation, x_test, y_test)
    ]
    x_train, y_train, x_validation, y_validation, x_test, y_test = arrays
    if x_train.ndim != 2 or x_validation.ndim != 2 or x_test.ndim != 2:
        raise ValueError("Continuous baseline predictors must be matrices")
    if x_train.shape[1] == 0 or not (
        x_train.shape[1] == x_validation.shape[1] == x_test.shape[1]
    ):
        raise ValueError("Continuous baseline feature dimensions disagree")
    if min(len(y_train), len(y_validation), len(y_test)) < 3:
        raise ValueError("Continuous baseline requires at least three patients per split")
    if not all(np.isfinite(y).all() for y in (y_train, y_validation, y_test)):
        raise ValueError("Continuous pathway outcomes must never be imputed")

    train_i, validation_i, test_i = _impute_from_train(
        x_train, x_validation, x_test
    )
    scaler = StandardScaler().fit(train_i)
    train_z = scaler.transform(train_i)
    validation_z = scaler.transform(validation_i)
    centered_y = y_train - float(y_train.mean())

    ratios = np.asarray(list(lasso_alpha_ratios), dtype=float)
    if ratios.size == 0 or np.any(~np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError("Invalid LASSO alpha ratios")
    alpha_max = float(np.max(np.abs(train_z.T @ centered_y)) / len(y_train))
    alpha_max = max(alpha_max, np.finfo(float).eps)
    lasso_alphas = np.unique(alpha_max * ratios)[::-1]
    path_alphas, coefficients, _ = lasso_path(
        train_z,
        centered_y,
        alphas=lasso_alphas,
        max_iter=50_000,
        tol=1e-6,
    )
    lasso_validation = validation_z @ coefficients + float(y_train.mean())
    lasso_mse = np.mean(
        np.square(lasso_validation - y_validation[:, None]), axis=0
    )
    lasso_index = int(np.flatnonzero(lasso_mse == np.nanmin(lasso_mse))[0])
    selected_lasso_alpha = float(path_alphas[lasso_index])

    ridge_grid = np.asarray(list(ridge_alphas), dtype=float)
    if ridge_grid.size == 0 or np.any(~np.isfinite(ridge_grid)) or np.any(ridge_grid <= 0):
        raise ValueError("Invalid Ridge alpha grid")
    ridge_grid = np.unique(ridge_grid)[::-1]
    ridge_validation_mse: list[float] = []
    for alpha in ridge_grid:
        model = Ridge(alpha=float(alpha), fit_intercept=True).fit(train_z, y_train)
        ridge_validation_mse.append(
            float(mean_squared_error(y_validation, model.predict(validation_z)))
        )
    ridge_validation_mse_array = np.asarray(ridge_validation_mse)
    ridge_index = int(
        np.flatnonzero(ridge_validation_mse_array == ridge_validation_mse_array.min())[0]
    )
    selected_ridge_alpha = float(ridge_grid[ridge_index])

    fit_x = np.concatenate([x_train, x_validation], axis=0)
    fit_y = np.concatenate([y_train, y_validation], axis=0)
    fit_i, final_test_i = _impute_from_train(fit_x, x_test)
    final_scaler = StandardScaler().fit(fit_i)
    fit_z = final_scaler.transform(fit_i)
    test_z = final_scaler.transform(final_test_i)
    lasso = Lasso(
        alpha=selected_lasso_alpha,
        fit_intercept=True,
        max_iter=50_000,
        tol=1e-6,
        selection="cyclic",
    ).fit(fit_z, fit_y)
    ridge = Ridge(alpha=selected_ridge_alpha, fit_intercept=True).fit(fit_z, fit_y)
    lasso_prediction = lasso.predict(test_z)
    ridge_prediction = ridge.predict(test_z)
    return {
        "lasso": {
            "alpha": selected_lasso_alpha,
            "validation_mse": float(lasso_mse[lasso_index]),
            "coefficients": np.asarray(lasso.coef_, dtype=float),
            "intercept": float(lasso.intercept_),
            "prediction": np.asarray(lasso_prediction, dtype=float),
            "n_nonzero": int(np.count_nonzero(np.abs(lasso.coef_) > 1e-12)),
            **_finite_metrics(y_test, lasso_prediction),
        },
        "ridge": {
            "alpha": selected_ridge_alpha,
            "validation_mse": float(ridge_validation_mse_array[ridge_index]),
            "coefficients": np.asarray(ridge.coef_, dtype=float),
            "intercept": float(ridge.intercept_),
            "prediction": np.asarray(ridge_prediction, dtype=float),
            "n_nonzero": int(np.count_nonzero(np.abs(ridge.coef_) > 1e-12)),
            **_finite_metrics(y_test, ridge_prediction),
        },
        "audit": {
            "hyperparameter_selection_split": "validation_only",
            "outer_test_used_for_tuning": False,
            "refit_split": "train_plus_validation_after_tuning",
            "outcome_imputation": "NONE",
            "predictor_imputation": "training_median",
            "predictor_scaling": "training_standard_scaler",
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_validation)),
            "n_test": int(len(y_test)),
            "n_features": int(x_train.shape[1]),
        },
    }
