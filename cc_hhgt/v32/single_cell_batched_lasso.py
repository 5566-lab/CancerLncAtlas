"""Donor-safe batched independent L1 evidence for single-cell HHGT graphs.

Each response solves its own scalar Lasso objective.  Responses share only
matrix operations; there is no joint sparsity penalty or response leakage.
All filtering, centering, scaling, alpha selection, and bootstrap fitting are
performed after slicing to the declared source donors.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd
from scipy import linalg, sparse
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


LASSO_FORMAT = "CANCERLNCATLAS_V32_SC_BATCHED_INDEPENDENT_LASSO_V1"


@dataclass(frozen=True)
class BatchedLassoPolicy:
    alphas: tuple[float, ...] = (0.3, 0.1, 0.03, 0.01)
    max_genes: int = 1000
    min_donors: int = 4
    min_detection: float = 0.05
    inner_folds: int = 5
    bootstraps: int = 8
    min_bootstrap_successes: int | None = None
    max_iter: int = 2000
    tolerance: float = 1e-6
    kkt_check_interval: int = 10
    response_batch_size: int = 256
    coefficient_epsilon: float = 1e-8
    top_k_per_response: int | None = None
    seed: int = 1729

    def validated(self) -> "BatchedLassoPolicy":
        if not self.alphas or any(not np.isfinite(value) or value <= 0 for value in self.alphas):
            raise ValueError("SC_LASSO_ALPHAS")
        if len(set(map(float, self.alphas))) != len(self.alphas):
            raise ValueError("SC_LASSO_DUPLICATE_ALPHA")
        if self.max_genes <= 0 or self.min_donors < 3:
            raise ValueError("SC_LASSO_DIMENSION_POLICY")
        if not 0 <= self.min_detection <= 1:
            raise ValueError("SC_LASSO_DETECTION_POLICY")
        if self.inner_folds < 2 or self.bootstraps <= 0:
            raise ValueError("SC_LASSO_RESAMPLING_POLICY")
        if self.max_iter <= 0 or self.tolerance <= 0 or self.kkt_check_interval <= 0:
            raise ValueError("SC_LASSO_SOLVER_POLICY")
        if self.response_batch_size <= 0 or self.coefficient_epsilon < 0:
            raise ValueError("SC_LASSO_OUTPUT_POLICY")
        if self.min_bootstrap_successes is not None and not (
            1 <= self.min_bootstrap_successes <= self.bootstraps
        ):
            raise ValueError("SC_LASSO_BOOTSTRAP_SUCCESS_POLICY")
        if self.top_k_per_response is not None and self.top_k_per_response <= 0:
            raise ValueError("SC_LASSO_TOP_K_POLICY")
        return self


@dataclass(frozen=True)
class BatchedLassoResult:
    response_count: int
    gene_count: int
    response_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    response_index: np.ndarray
    gene_index: np.ndarray
    coefficient: np.ndarray
    stability: np.ndarray
    available: np.ndarray
    unavailable_reason: np.ndarray
    selected_alpha: np.ndarray
    inner_cv_mse: np.ndarray
    inner_null_mse: np.ndarray
    bootstrap_successes: np.ndarray
    audit: dict[str, Any]

    @property
    def unavailable(self) -> np.ndarray:
        return ~self.available

    def coefficient_coo(self) -> sparse.coo_matrix:
        return sparse.coo_matrix(
            (self.coefficient, (self.response_index, self.gene_index)),
            shape=(self.response_count, self.gene_count),
        )

    def stability_coo(self) -> sparse.coo_matrix:
        return sparse.coo_matrix(
            (self.stability, (self.response_index, self.gene_index)),
            shape=(self.response_count, self.gene_count),
        )

    def edge_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "response_index": self.response_index,
                "gene_index": self.gene_index,
                "response_id": np.asarray(self.response_ids, dtype=object)[self.response_index],
                "gene_id": np.asarray(self.gene_ids, dtype=object)[self.gene_index],
                "coefficient": self.coefficient,
                "stability": self.stability,
                "sign": np.sign(self.coefficient).astype(np.int8),
            }
        )

    def response_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "response_index": np.arange(self.response_count, dtype=np.int64),
                "response_id": self.response_ids,
                "available": self.available,
                "unavailable_reason": self.unavailable_reason,
                "selected_alpha": self.selected_alpha,
                "inner_cv_mse": self.inner_cv_mse,
                "inner_null_mse": self.inner_null_mse,
                "bootstrap_successes": self.bootstrap_successes,
            }
        )


@dataclass(frozen=True)
class _PreparedDesign:
    selected: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    standardized: np.ndarray
    weights: np.ndarray
    weighted_design: np.ndarray
    gram: np.ndarray | None
    lipschitz: float
    lipschitz_method: str


@dataclass(frozen=True)
class _SolverResult:
    coefficients: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    maximum_kkt_violation: np.ndarray


def donor_balanced_weights(donors: Iterable[object]) -> np.ndarray:
    donors = np.asarray(list(donors), dtype=str)
    if donors.ndim != 1 or not len(donors) or np.any(np.char.strip(donors) == ""):
        raise ValueError("SC_LASSO_DONOR_LABELS")
    _, inverse, counts = np.unique(donors, return_inverse=True, return_counts=True)
    weights = 1.0 / (len(counts) * counts[inverse])
    return weights.astype(np.float64, copy=False)


def _matrix_shape(value: Any, name: str) -> tuple[int, int]:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name}_SHAPE")
    return int(shape[0]), int(shape[1])


def _take_rows(value: Any, rows: np.ndarray, name: str) -> Any:
    if sparse.issparse(value):
        result = sparse.csr_matrix(value)[rows]
        if not np.isfinite(result.data).all():
            raise ValueError(f"NONFINITE_SOURCE_{name}")
        return result
    result = np.asarray(value)[rows]
    if not np.issubdtype(result.dtype, np.number):
        raise ValueError(f"{name}_DTYPE")
    result = result.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"NONFINITE_SOURCE_{name}")
    return result


def _dense(value: Any) -> np.ndarray:
    if sparse.issparse(value):
        return value.toarray().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _weighted_feature_statistics(
    coding: Any, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sparse.issparse(coding):
        coding = sparse.csr_matrix(coding, dtype=np.float64)
        mean = np.asarray(weights @ coding).ravel()
        squared = coding.copy()
        squared.data **= 2
        second = np.asarray(weights @ squared).ravel()
        positive = coding.copy()
        positive.data = (positive.data > 0).astype(np.float64)
        positive.eliminate_zeros()
        detection = np.asarray(weights @ positive).ravel()
    else:
        coding = np.asarray(coding, dtype=np.float64)
        mean = weights @ coding
        second = weights @ (coding * coding)
        detection = weights @ (coding > 0)
    variance = np.maximum(second - mean * mean, 0.0)
    return mean, variance, detection


def _prepare_design(
    coding: Any, donors: np.ndarray, policy: BatchedLassoPolicy
) -> _PreparedDesign | None:
    weights = donor_balanced_weights(donors)
    mean, variance, detection = _weighted_feature_statistics(coding, weights)
    selected = np.flatnonzero(
        (variance > 1e-8) & (detection >= policy.min_detection)
    )
    order = np.argsort(-variance[selected], kind="stable")
    selected = selected[order[: policy.max_genes]].astype(np.int64, copy=False)
    if not len(selected):
        return None
    scale = np.sqrt(variance[selected])
    values = _dense(coding[:, selected] if sparse.issparse(coding) else coding[:, selected])
    standardized = (values - mean[selected]) / scale
    weighted = standardized * np.sqrt(weights)[:, None]
    observation_count, feature_count = weighted.shape
    if feature_count <= observation_count:
        gram = weighted.T @ weighted
        spectral_matrix = gram
        method = "EXACT_PRIMAL_GRAM_EIGENVALUE"
    else:
        gram = None
        spectral_matrix = weighted @ weighted.T
        method = "EXACT_DUAL_GRAM_EIGENVALUE"
    dimension = spectral_matrix.shape[0]
    largest = float(
        linalg.eigh(
            spectral_matrix,
            eigvals_only=True,
            subset_by_index=(dimension - 1, dimension - 1),
            check_finite=False,
            driver="evr",
        )[0]
    )
    if not np.isfinite(largest) or largest <= 0:
        return None
    return _PreparedDesign(
        selected=selected,
        mean=mean[selected],
        scale=scale,
        standardized=standardized,
        weights=weights,
        weighted_design=weighted,
        gram=gram,
        lipschitz=largest,
        lipschitz_method=method,
    )


def _standardize_responses(
    responses: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = weights @ responses
    centered = responses - mean
    scale = np.sqrt(np.maximum(weights @ (centered * centered), 0.0))
    estimable = np.isfinite(scale) & (scale > 1e-8)
    standardized = np.zeros_like(responses, dtype=np.float64)
    standardized[:, estimable] = centered[:, estimable] / scale[estimable]
    return standardized, mean, scale, estimable


def _soft_threshold(value: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


def _kkt_violation(
    design: _PreparedDesign,
    cross: np.ndarray,
    coefficients: np.ndarray,
    alphas: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    gradient = _smooth_gradient(design, coefficients, cross)
    support = np.abs(coefficients) > epsilon
    residual = np.where(
        support,
        np.abs(gradient + np.sign(coefficients) * alphas[None, :]),
        np.maximum(np.abs(gradient) - alphas[None, :], 0.0),
    )
    return residual.max(axis=0, initial=0.0)


def _smooth_gradient(
    design: _PreparedDesign, coefficients: np.ndarray, cross: np.ndarray
) -> np.ndarray:
    if design.gram is not None:
        return design.gram @ coefficients - cross
    return design.weighted_design.T @ (
        design.weighted_design @ coefficients
    ) - cross


def _solve_independent_fista(
    design: _PreparedDesign,
    standardized_responses: np.ndarray,
    alphas: np.ndarray,
    policy: BatchedLassoPolicy,
    *,
    initial: np.ndarray | None = None,
) -> _SolverResult:
    """Solve independent scalar Lasso columns with batched matrix operations."""

    responses = np.asarray(standardized_responses, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)
    if responses.ndim != 2 or responses.shape[0] != len(design.weights):
        raise ValueError("SC_LASSO_STANDARDIZED_RESPONSE_SHAPE")
    if alphas.shape != (responses.shape[1],) or np.any(alphas <= 0):
        raise ValueError("SC_LASSO_RESPONSE_ALPHA_SHAPE")
    feature_count = len(design.selected)
    response_count = responses.shape[1]
    if initial is not None and np.asarray(initial).shape != (feature_count, response_count):
        raise ValueError("SC_LASSO_WARM_START_SHAPE")

    all_coefficients = np.zeros((feature_count, response_count), dtype=np.float64)
    all_converged = np.zeros(response_count, dtype=bool)
    all_iterations = np.full(response_count, policy.max_iter, dtype=np.int64)
    all_kkt = np.full(response_count, np.inf, dtype=np.float64)
    root_weight = np.sqrt(design.weights)[:, None]
    weighted_design = design.weighted_design

    for start in range(0, response_count, policy.response_batch_size):
        stop = min(start + policy.response_batch_size, response_count)
        target = responses[:, start:stop] * root_weight
        cross = weighted_design.T @ target
        alpha = alphas[start:stop]
        if initial is None:
            coefficient = np.zeros((feature_count, stop - start), dtype=np.float64)
        else:
            coefficient = np.asarray(initial[:, start:stop], dtype=np.float64).copy()
        extrapolated = coefficient.copy()
        acceleration = np.ones(stop - start, dtype=np.float64)
        active = np.ones(stop - start, dtype=bool)
        converged = np.zeros(stop - start, dtype=bool)
        iterations = np.full(stop - start, policy.max_iter, dtype=np.int64)

        for iteration in range(1, policy.max_iter + 1):
            gradient = _smooth_gradient(design, extrapolated, cross)
            updated = _soft_threshold(
                extrapolated - gradient / design.lipschitz,
                alpha[None, :] / design.lipschitz,
            )
            updated[:, ~active] = coefficient[:, ~active]
            next_acceleration = (
                1.0 + np.sqrt(1.0 + 4.0 * acceleration * acceleration)
            ) / 2.0
            momentum = (acceleration - 1.0) / next_acceleration
            next_extrapolated = updated + momentum[None, :] * (updated - coefficient)

            # Column-wise adaptive restart prevents oscillation on correlated genes.
            restart = np.sum(
                (extrapolated - updated) * (updated - coefficient), axis=0
            ) > 0
            next_extrapolated[:, restart] = updated[:, restart]
            next_acceleration[restart] = 1.0
            next_extrapolated[:, ~active] = updated[:, ~active]
            next_acceleration[~active] = 1.0
            coefficient = updated
            extrapolated = next_extrapolated
            acceleration = next_acceleration

            if iteration % policy.kkt_check_interval == 0 or iteration == policy.max_iter:
                violation = _kkt_violation(
                    design,
                    cross,
                    coefficient,
                    alpha,
                    policy.coefficient_epsilon,
                )
                newly_converged = active & (violation <= policy.tolerance)
                iterations[newly_converged] = iteration
                converged |= newly_converged
                active &= ~newly_converged
                if not active.any():
                    break

        violation = _kkt_violation(
            design,
            cross,
            coefficient,
            alpha,
            policy.coefficient_epsilon,
        )
        converged |= violation <= policy.tolerance
        iterations[(iterations == policy.max_iter) & converged] = iteration
        all_coefficients[:, start:stop] = coefficient
        all_converged[start:stop] = converged
        all_iterations[start:stop] = iterations
        all_kkt[start:stop] = violation

    return _SolverResult(
        coefficients=all_coefficients,
        converged=all_converged,
        iterations=all_iterations,
        maximum_kkt_violation=all_kkt,
    )


def _coordinate_descent_workers() -> int:
    raw = os.environ.get("SC_LASSO_SOLVER_THREADS", "1")
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError("SC_LASSO_SOLVER_THREADS") from error
    if workers <= 0:
        raise ValueError("SC_LASSO_SOLVER_THREADS")
    return workers


def _solve_independent_coordinate_descent(
    design: _PreparedDesign,
    standardized_responses: np.ndarray,
    alphas: np.ndarray,
    policy: BatchedLassoPolicy,
    *,
    initial: np.ndarray | None = None,
) -> _SolverResult:
    """Solve the same scalar objectives with parallel coordinate descent."""

    responses = np.asarray(standardized_responses, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)
    if responses.ndim != 2 or responses.shape[0] != len(design.weights):
        raise ValueError("SC_LASSO_STANDARDIZED_RESPONSE_SHAPE")
    if alphas.shape != (responses.shape[1],) or np.any(alphas <= 0):
        raise ValueError("SC_LASSO_RESPONSE_ALPHA_SHAPE")
    feature_count = len(design.selected)
    response_count = responses.shape[1]
    if initial is not None and np.asarray(initial).shape != (feature_count, response_count):
        raise ValueError("SC_LASSO_WARM_START_SHAPE")
    if response_count == 0:
        return _SolverResult(
            coefficients=np.zeros((feature_count, 0), dtype=np.float64),
            converged=np.zeros(0, dtype=bool),
            iterations=np.zeros(0, dtype=np.int64),
            maximum_kkt_violation=np.zeros(0, dtype=np.float64),
        )

    observation_count = len(design.weights)
    scale = math.sqrt(observation_count)
    root_weight = np.sqrt(design.weights)[:, None]
    x = np.asfortranarray(design.weighted_design * scale)
    targets = np.asfortranarray(responses * root_weight * scale)
    gram = np.ascontiguousarray(x.T @ x)

    def solve(response: int) -> tuple[np.ndarray, int]:
        model = Lasso(
            alpha=float(alphas[response]),
            fit_intercept=False,
            precompute=gram,
            copy_X=False,
            max_iter=policy.max_iter,
            tol=policy.tolerance,
            warm_start=initial is not None,
            selection="cyclic",
        )
        if initial is not None:
            model.coef_ = np.asarray(initial[:, response], dtype=np.float64).copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x, targets[:, response], check_input=False)
        return np.asarray(model.coef_, dtype=np.float64), int(model.n_iter_)

    workers = min(_coordinate_descent_workers(), response_count)
    with threadpool_limits(limits=1):
        if workers == 1:
            solved = [solve(response) for response in range(response_count)]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                solved = list(pool.map(solve, range(response_count)))
    coefficients = np.column_stack([value[0] for value in solved])
    iterations = np.asarray([value[1] for value in solved], dtype=np.int64)
    cross = design.weighted_design.T @ (responses * root_weight)
    violation = _kkt_violation(
        design, cross, coefficients, alphas, policy.coefficient_epsilon
    )
    converged = violation <= policy.tolerance
    return _SolverResult(
        coefficients=coefficients,
        converged=converged,
        iterations=iterations,
        maximum_kkt_violation=violation,
    )


def _donor_folds(
    donors: np.ndarray, fold_count: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    unique = np.unique(donors)
    rng = np.random.default_rng(seed)
    ordered = unique[rng.permutation(len(unique))]
    count = min(int(fold_count), len(ordered))
    return tuple(
        tuple(sorted(ordered[np.arange(len(ordered)) % count == fold].tolist()))
        for fold in range(count)
    )


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _matrix_hash(value: Any) -> str:
    digest = hashlib.sha256()
    if sparse.issparse(value):
        matrix = sparse.csr_matrix(value)
        digest.update(str(matrix.shape).encode("ascii"))
        for part in (matrix.indptr, matrix.indices, matrix.data):
            digest.update(np.ascontiguousarray(part).view(np.uint8))
    else:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _empty_result(
    *,
    response_count: int,
    gene_count: int,
    response_ids: tuple[str, ...],
    gene_ids: tuple[str, ...],
    reason: str,
    audit: dict[str, Any],
) -> BatchedLassoResult:
    return BatchedLassoResult(
        response_count=response_count,
        gene_count=gene_count,
        response_ids=response_ids,
        gene_ids=gene_ids,
        response_index=np.empty(0, dtype=np.int64),
        gene_index=np.empty(0, dtype=np.int64),
        coefficient=np.empty(0, dtype=np.float64),
        stability=np.empty(0, dtype=np.float64),
        available=np.zeros(response_count, dtype=bool),
        unavailable_reason=np.full(response_count, reason, dtype=object),
        selected_alpha=np.full(response_count, np.nan),
        inner_cv_mse=np.full(response_count, np.nan),
        inner_null_mse=np.full(response_count, np.nan),
        bootstrap_successes=np.zeros(response_count, dtype=np.int64),
        audit={**audit, "available_response_count": 0, "edge_count": 0},
    )


def _validate_ids(
    values: Iterable[object] | None, count: int, prefix: str
) -> tuple[str, ...]:
    if values is None:
        return tuple(f"{prefix}{index}" for index in range(count))
    result = tuple(map(str, values))
    if len(result) != count or len(set(result)) != count or any(not value for value in result):
        raise ValueError(f"SC_LASSO_{prefix.upper()}_IDENTITY")
    return result


def _apply_top_k(
    response_index: np.ndarray,
    gene_index: np.ndarray,
    coefficient: np.ndarray,
    top_k: int | None,
) -> np.ndarray:
    keep = np.ones(len(coefficient), dtype=bool)
    if top_k is None:
        return keep
    keep[:] = False
    for response in np.unique(response_index):
        rows = np.flatnonzero(response_index == response)
        order = np.lexsort((gene_index[rows], -np.abs(coefficient[rows])))
        keep[rows[order[:top_k]]] = True
    return keep


def fit_batched_lasso(
    coding: Any,
    responses: Any,
    donors: Iterable[object],
    *,
    source_donors: Iterable[object],
    excluded_donors: Iterable[object],
    response_ids: Iterable[object] | None = None,
    gene_ids: Iterable[object] | None = None,
    policy: BatchedLassoPolicy = BatchedLassoPolicy(),
) -> BatchedLassoResult:
    """Fit donor-cross-validated independent Lasso evidence in response batches.

    ``coding`` is metacell-by-gene and ``responses`` is metacell-by-lncRNA.
    Rows outside ``source_donors`` are never inspected beyond their donor label.
    ``excluded_donors`` must explicitly cover every observed non-source donor.
    """

    policy = policy.validated()
    coding_shape = _matrix_shape(coding, "SC_LASSO_CODING")
    response_shape = _matrix_shape(responses, "SC_LASSO_RESPONSES")
    if coding_shape[0] != response_shape[0]:
        raise ValueError("SC_LASSO_OBSERVATION_ALIGNMENT")
    donor_array = np.asarray(list(donors), dtype=str)
    if donor_array.shape != (coding_shape[0],) or np.any(np.char.strip(donor_array) == ""):
        raise ValueError("SC_LASSO_DONOR_ALIGNMENT")

    source = {str(value) for value in source_donors}
    excluded = {str(value) for value in excluded_donors}
    if not source or source.intersection(excluded):
        raise ValueError("SC_LASSO_DONOR_LEAK")
    observed = set(donor_array.tolist())
    if observed - source - excluded:
        raise ValueError("SC_LASSO_UNDECLARED_DONOR_SCOPE")

    response_count = response_shape[1]
    gene_count = coding_shape[1]
    response_identity = _validate_ids(response_ids, response_count, "response")
    gene_identity = _validate_ids(gene_ids, gene_count, "gene")
    source_rows = np.flatnonzero(np.isin(donor_array, sorted(source)))
    effective_donors = donor_array[source_rows]
    effective_unique = tuple(sorted(np.unique(effective_donors).tolist()))
    base_audit: dict[str, Any] = {
        "format": LASSO_FORMAT,
        "solver": "PARALLEL_INDEPENDENT_COORDINATE_DESCENT",
        "objective": "INDEPENDENT_WEIGHTED_SCALAR_LASSO_PER_RESPONSE",
        "multitask_penalty_used": False,
        "source_only_preprocessing": True,
        "held_out_donors_used": 0,
        "requested_source_donors": sorted(source),
        "effective_source_donors": list(effective_unique),
        "excluded_donors": sorted(excluded),
        "missing_context_source_donors": sorted(source - set(effective_unique)),
        "source_metacells": int(len(source_rows)),
        "coding_gene_count": gene_count,
        "lncrna_count": response_count,
        "alpha_candidates": list(map(float, policy.alphas)),
        "alpha_warm_start_order": [
            float(policy.alphas[index]) for index in np.argsort(
                -np.asarray(policy.alphas), kind="stable"
            )
        ],
        "inner_fold_policy": "DETERMINISTIC_DONOR_DISJOINT_K_FOLD",
        "bootstrap_policy": "DETERMINISTIC_DONOR_BLOCK_RESAMPLE_WITH_REPLACEMENT",
        "bootstrap_repeats": policy.bootstraps,
        "coefficient_semantics": "STANDARDIZED_CONDITIONAL_PREDICTIVE_NOT_CAUSAL_NOT_FDR",
        "top_k_per_response": policy.top_k_per_response,
    }
    if len(effective_unique) < policy.min_donors:
        return _empty_result(
            response_count=response_count,
            gene_count=gene_count,
            response_ids=response_identity,
            gene_ids=gene_identity,
            reason="INSUFFICIENT_SOURCE_DONORS",
            audit=base_audit,
        )

    source_coding = _take_rows(coding, source_rows, "CODING")
    source_responses = _dense(_take_rows(responses, source_rows, "RESPONSES"))
    base_audit["source_coding_sha256"] = _matrix_hash(source_coding)
    base_audit["source_responses_sha256"] = _matrix_hash(source_responses)
    base_audit["source_donor_order_sha256"] = _canonical_hash(effective_donors.tolist())

    folds = _donor_folds(effective_donors, policy.inner_folds, policy.seed)
    base_audit["inner_folds"] = [list(fold) for fold in folds]
    alpha_count = len(policy.alphas)
    cv_loss = np.zeros((alpha_count, response_count), dtype=np.float64)
    cv_null = np.zeros_like(cv_loss)
    cv_valid = np.ones((alpha_count, response_count), dtype=bool)
    cv_iterations: list[dict[str, Any]] = []

    alpha_order = np.argsort(-np.asarray(policy.alphas), kind="stable")
    for fold_number, held_donors in enumerate(folds):
        validation_mask = np.isin(effective_donors, held_donors)
        training_mask = ~validation_mask
        training_donors = effective_donors[training_mask]
        validation_donors = effective_donors[validation_mask]
        design = _prepare_design(source_coding[training_mask], training_donors, policy)
        if design is None:
            cv_valid[:, :] = False
            cv_iterations.append(
                {"fold": fold_number, "held_donors": list(held_donors), "selected_genes": 0}
            )
            continue
        training_response = source_responses[training_mask]
        standardized, target_mean, target_scale, estimable = _standardize_responses(
            training_response, design.weights
        )
        validation_response = source_responses[validation_mask]
        validation_weights = donor_balanced_weights(validation_donors)
        validation_coding = _dense(
            source_coding[validation_mask][:, design.selected]
            if sparse.issparse(source_coding)
            else source_coding[validation_mask][:, design.selected]
        )
        validation_standardized = (validation_coding - design.mean) / design.scale
        eligible = np.flatnonzero(estimable)
        warm: np.ndarray | None = None
        fold_record: dict[str, Any] = {
            "fold": fold_number,
            "held_donors": list(held_donors),
            "training_donors": sorted(np.unique(training_donors).tolist()),
            "selected_genes": int(len(design.selected)),
            "lipschitz_method": design.lipschitz_method,
            "alpha_iterations": {},
        }
        for alpha_index in alpha_order:
            cv_valid[alpha_index, ~estimable] = False
            if not len(eligible):
                continue
            alpha = float(policy.alphas[alpha_index])
            solved = _solve_independent_coordinate_descent(
                design,
                standardized[:, eligible],
                np.full(len(eligible), alpha),
                policy,
                initial=warm,
            )
            warm = solved.coefficients
            converged = np.zeros(response_count, dtype=bool)
            converged[eligible] = solved.converged
            cv_valid[alpha_index] &= converged
            local_valid = eligible[solved.converged]
            if len(local_valid):
                local_position = np.flatnonzero(solved.converged)
                predicted_standardized = (
                    validation_standardized @ solved.coefficients[:, local_position]
                )
                predicted = (
                    target_mean[local_valid]
                    + predicted_standardized * target_scale[local_valid]
                )
                error = validation_response[:, local_valid] - predicted
                null_error = (
                    validation_response[:, local_valid] - target_mean[local_valid]
                )
                cv_loss[alpha_index, local_valid] += validation_weights @ (error * error)
                cv_null[alpha_index, local_valid] += validation_weights @ (
                    null_error * null_error
                )
            fold_record["alpha_iterations"][str(alpha)] = {
                "converged_responses": int(solved.converged.sum()),
                "maximum_iterations": int(solved.iterations.max(initial=0)),
                "maximum_kkt_violation": float(
                    solved.maximum_kkt_violation.max(initial=0.0)
                ),
            }
        cv_iterations.append(fold_record)

    cv_loss /= len(folds)
    cv_null /= len(folds)
    masked_loss = np.where(cv_valid, cv_loss, np.inf)
    best_index = np.argmin(masked_loss, axis=0)
    has_candidate = np.isfinite(masked_loss[best_index, np.arange(response_count)])
    selected_alpha = np.full(response_count, np.nan)
    selected_alpha[has_candidate] = np.asarray(policy.alphas)[best_index[has_candidate]]
    inner_cv_mse = np.full(response_count, np.nan)
    inner_null_mse = np.full(response_count, np.nan)
    positions = np.flatnonzero(has_candidate)
    inner_cv_mse[positions] = cv_loss[best_index[positions], positions]
    inner_null_mse[positions] = cv_null[best_index[positions], positions]

    final_design = _prepare_design(source_coding, effective_donors, policy)
    unavailable_reason = np.full(response_count, "INNER_CV_UNAVAILABLE", dtype=object)
    available = np.zeros(response_count, dtype=bool)
    bootstrap_successes = np.zeros(response_count, dtype=np.int64)
    if final_design is None:
        base_audit.update(
            {
                "inner_cv_solver": cv_iterations,
                "unavailable_reason_counts": {"NO_ESTIMABLE_GENES": response_count},
            }
        )
        return BatchedLassoResult(
            response_count=response_count,
            gene_count=gene_count,
            response_ids=response_identity,
            gene_ids=gene_identity,
            response_index=np.empty(0, dtype=np.int64),
            gene_index=np.empty(0, dtype=np.int64),
            coefficient=np.empty(0),
            stability=np.empty(0),
            available=available,
            unavailable_reason=np.full(response_count, "NO_ESTIMABLE_GENES", dtype=object),
            selected_alpha=selected_alpha,
            inner_cv_mse=inner_cv_mse,
            inner_null_mse=inner_null_mse,
            bootstrap_successes=bootstrap_successes,
            audit={**base_audit, "available_response_count": 0, "edge_count": 0},
        )

    final_standardized, _, _, final_estimable = _standardize_responses(
        source_responses, final_design.weights
    )
    unavailable_reason[has_candidate & ~final_estimable] = "NON_ESTIMABLE_RESPONSE"
    final_eligible = np.flatnonzero(has_candidate & final_estimable)
    final_coefficients = np.zeros(
        (len(final_design.selected), response_count), dtype=np.float64
    )
    final_solver: _SolverResult | None = None
    if len(final_eligible):
        final_solver = _solve_independent_coordinate_descent(
            final_design,
            final_standardized[:, final_eligible],
            selected_alpha[final_eligible],
            policy,
        )
        final_coefficients[:, final_eligible] = final_solver.coefficients
        converged_response = final_eligible[final_solver.converged]
        unavailable_reason[final_eligible] = "FINAL_NOT_CONVERGED"
        unavailable_reason[converged_response] = "INSUFFICIENT_BOOTSTRAP_SUCCESS"
        provisional = np.zeros(response_count, dtype=bool)
        provisional[converged_response] = True
    else:
        provisional = np.zeros(response_count, dtype=bool)

    local_gene, response_index = np.nonzero(
        (np.abs(final_coefficients) > policy.coefficient_epsilon)
        & provisional[None, :]
    )
    gene_index = final_design.selected[local_gene]
    coefficient = final_coefficients[local_gene, response_index]
    top_keep = _apply_top_k(
        response_index.astype(np.int64),
        gene_index.astype(np.int64),
        coefficient,
        policy.top_k_per_response,
    )
    local_gene = local_gene[top_keep]
    response_index = response_index[top_keep].astype(np.int64, copy=False)
    gene_index = gene_index[top_keep].astype(np.int64, copy=False)
    coefficient = coefficient[top_keep]
    selection_count = np.zeros(len(coefficient), dtype=np.int64)

    bootstrap_rng = np.random.default_rng(np.random.SeedSequence(policy.seed).spawn(1)[0])
    unique_donors = np.unique(effective_donors)
    for _ in range(policy.bootstraps):
        sampled = bootstrap_rng.choice(unique_donors, len(unique_donors), replace=True)
        row_parts: list[np.ndarray] = []
        block_parts: list[np.ndarray] = []
        for draw, donor in enumerate(sampled):
            rows = np.flatnonzero(effective_donors == donor)
            row_parts.append(rows)
            block_parts.append(np.full(len(rows), f"draw_{draw}", dtype=object))
        bootstrap_rows = np.concatenate(row_parts)
        bootstrap_donors = np.concatenate(block_parts).astype(str)
        bootstrap_coding = source_coding[bootstrap_rows]
        bootstrap_response = source_responses[bootstrap_rows]
        bootstrap_design = _prepare_design(
            bootstrap_coding, bootstrap_donors, policy
        )
        if bootstrap_design is None:
            continue
        standardized, _, _, estimable = _standardize_responses(
            bootstrap_response, bootstrap_design.weights
        )
        eligible = np.flatnonzero(provisional & estimable)
        if not len(eligible):
            continue
        solved = _solve_independent_coordinate_descent(
            bootstrap_design,
            standardized[:, eligible],
            selected_alpha[eligible],
            policy,
        )
        successful = eligible[solved.converged]
        bootstrap_successes[successful] += 1
        if len(coefficient) and len(successful):
            response_to_local = np.full(response_count, -1, dtype=np.int64)
            response_to_local[eligible] = np.arange(len(eligible), dtype=np.int64)
            gene_to_local = np.full(gene_count, -1, dtype=np.int64)
            gene_to_local[bootstrap_design.selected] = np.arange(
                len(bootstrap_design.selected), dtype=np.int64
            )
            response_position = response_to_local[response_index]
            gene_position = gene_to_local[gene_index]
            valid_edge = (
                (response_position >= 0)
                & (gene_position >= 0)
                & solved.converged[np.maximum(response_position, 0)]
            )
            selected_edge = np.zeros(len(coefficient), dtype=bool)
            rows = np.flatnonzero(valid_edge)
            if len(rows):
                selected_edge[rows] = (
                    np.abs(
                        solved.coefficients[
                            gene_position[rows], response_position[rows]
                        ]
                    )
                    > policy.coefficient_epsilon
                )
            selection_count += selected_edge

    required_bootstraps = (
        policy.min_bootstrap_successes
        if policy.min_bootstrap_successes is not None
        else max(1, math.ceil(policy.bootstraps / 2))
    )
    available = provisional & (bootstrap_successes >= required_bootstraps)
    unavailable_reason[available] = ""
    stability = np.divide(
        selection_count,
        bootstrap_successes[response_index],
        out=np.zeros(len(selection_count), dtype=np.float64),
        where=bootstrap_successes[response_index] > 0,
    )
    keep_available = available[response_index]
    response_index = response_index[keep_available]
    gene_index = gene_index[keep_available]
    coefficient = coefficient[keep_available]
    stability = stability[keep_available]
    order = np.lexsort((gene_index, response_index))
    response_index = response_index[order]
    gene_index = gene_index[order]
    coefficient = coefficient[order]
    stability = stability[order]

    reason_counts = Counter(map(str, unavailable_reason[~available]))
    base_audit.update(
        {
            "inner_cv_solver": cv_iterations,
            "final_selected_gene_count": int(len(final_design.selected)),
            "final_lipschitz_method": final_design.lipschitz_method,
            "final_converged_response_count": int(provisional.sum()),
            "final_maximum_iterations": int(
                final_solver.iterations.max(initial=0) if final_solver is not None else 0
            ),
            "final_maximum_kkt_violation": float(
                final_solver.maximum_kkt_violation.max(initial=0.0)
                if final_solver is not None
                else 0.0
            ),
            "required_bootstrap_successes": int(required_bootstraps),
            "available_response_count": int(available.sum()),
            "unavailable_reason_counts": dict(sorted(reason_counts.items())),
            "edge_count": int(len(coefficient)),
            "positive_edge_count": int((coefficient > 0).sum()),
            "negative_edge_count": int((coefficient < 0).sum()),
            "alpha_selection_counts": {
                str(alpha): int(np.sum(selected_alpha[available] == alpha))
                for alpha in policy.alphas
            },
        }
    )
    return BatchedLassoResult(
        response_count=response_count,
        gene_count=gene_count,
        response_ids=response_identity,
        gene_ids=gene_identity,
        response_index=response_index,
        gene_index=gene_index,
        coefficient=coefficient,
        stability=stability,
        available=available,
        unavailable_reason=unavailable_reason,
        selected_alpha=selected_alpha,
        inner_cv_mse=inner_cv_mse,
        inner_null_mse=inner_null_mse,
        bootstrap_successes=bootstrap_successes,
        audit=base_audit,
    )


__all__ = [
    "BatchedLassoPolicy",
    "BatchedLassoResult",
    "LASSO_FORMAT",
    "donor_balanced_weights",
    "fit_batched_lasso",
]
