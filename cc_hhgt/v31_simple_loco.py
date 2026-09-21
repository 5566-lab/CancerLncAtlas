"""Leakage-safe cross-cancer baselines for the V3.1 residual pilot.

The functions in this module deliberately separate two operations:

* deriving sufficient statistics from *source* cancers; and
* attaching labels from a query cancer after its scores have been produced.

That separation is what makes the outcome-invariance tests meaningful.  A
query cancer is never present in its own feature aggregates, logistic fitting
rows, or model-selection folds.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import require_columns
from .metrics import binary_metrics


PAIR_KEYS = ("lncrna_id", "target_id", "target_type")
QUERY_KEYS = ("candidate_id", "cancer_id", *PAIR_KEYS)
FEATURE_COLUMNS = (
    "support_count",
    "support_frequency",
    "meta_effect_signed",
    "meta_effect_abs",
    "direction_consistency",
    "heterogeneity_Q",
    "heterogeneity_I2",
    "n_evaluable_cancers",
    "similarity_weighted_support",
    "similarity_weighted_effect_signed",
    "similarity_weighted_effect_abs",
    "lncrna_support_frequency",
    "target_support_frequency",
    "overall_prevalence",
    "rule_score",
)
LOGISTIC_FEATURE_COLUMNS = (
    "support_frequency",
    "meta_effect_abs",
    "direction_consistency",
    "heterogeneity_I2",
    "similarity_weighted_support",
    "similarity_weighted_effect_abs",
    "lncrna_support_frequency",
    "target_support_frequency",
    "log1p_n_evaluable_cancers",
)
SCORE_COLUMNS = {
    "prevalence": "score_prevalence",
    "frequency": "score_frequency",
    "meta_effect": "score_meta_effect",
    "rule_score": "score_rule",
    "similarity_weighted": "score_similarity_weighted",
    "logistic": "score_logistic",
}


class SimpleLocoContractError(RuntimeError):
    """Raised when a caller violates the cancer-level cross-fit contract."""


@dataclass(frozen=True)
class BestSimpleFit:
    """Frozen simple-base result for one outer LOCO fold."""

    oof_predictions: pd.DataFrame
    selection_metrics: pd.DataFrame
    selected_methods: pd.DataFrame
    query_predictions: Mapping[str, pd.DataFrame]


@dataclass
class CrossCancerSufficientStatistics:
    """Reusable pair/lncRNA/target sums for exact whole-cancer subtraction."""

    cancers: tuple[str, ...]
    pair_global: pd.DataFrame
    pair_by_cancer: Mapping[str, pd.DataFrame]
    lnc_global: pd.DataFrame
    lnc_by_cancer: Mapping[str, pd.DataFrame]
    target_global: pd.DataFrame
    target_by_cancer: Mapping[str, pd.DataFrame]
    total_positive: float
    total_count: int
    totals_by_cancer: Mapping[str, tuple[float, int]]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "CrossCancerSufficientStatistics":
        data = standardize_candidate_frame(frame)
        data = data.copy()
        data["positive"] = data.label.astype(float)
        data["effect_value"] = pd.to_numeric(data.effect, errors="coerce")
        data["effect_valid"] = (
            data.effect_value.notna()
            & data.effect_value.between(-1, 1, inclusive="both")
        )
        observed = pd.to_numeric(data.n_observed, errors="coerce")
        data["effect_weight"] = np.where(
            data.effect_valid,
            np.maximum(observed.fillna(4.0).to_numpy(float) - 3.0, 1.0),
            0.0,
        )
        data["effect_z"] = np.where(
            data.effect_valid,
            np.arctanh(data.effect_value.clip(-0.999999, 0.999999)),
            0.0,
        )
        data["effect_wz"] = data.effect_weight * data.effect_z
        data["effect_wz2"] = data.effect_weight * data.effect_z.pow(2)
        data["positive_known_direction"] = (
            data.positive * data.effect_valid.astype(float)
        )
        data["positive_direction"] = (
            data.positive_known_direction * data.effect_value.gt(0).astype(float)
        )
        data["row_count"] = 1.0
        data["effect_count"] = data.effect_valid.astype(float)
        value_columns = [
            "positive",
            "row_count",
            "effect_weight",
            "effect_wz",
            "effect_wz2",
            "effect_count",
            "positive_known_direction",
            "positive_direction",
        ]
        pair_by_cancer: dict[str, pd.DataFrame] = {}
        lnc_by_cancer: dict[str, pd.DataFrame] = {}
        target_by_cancer: dict[str, pd.DataFrame] = {}
        totals_by_cancer: dict[str, tuple[float, int]] = {}
        for cancer, group in data.groupby("cancer_id", observed=True, sort=True):
            cancer = str(cancer)
            pair_by_cancer[cancer] = group.set_index(list(PAIR_KEYS))[value_columns]
            lnc_by_cancer[cancer] = group.groupby("lncrna_id", observed=True)[
                ["positive", "row_count"]
            ].sum()
            target_by_cancer[cancer] = group.groupby(
                ["target_id", "target_type"], observed=True
            )[["positive", "row_count"]].sum()
            totals_by_cancer[cancer] = (float(group.positive.sum()), int(len(group)))
        pair_global = data.groupby(list(PAIR_KEYS), observed=True)[value_columns].sum()
        lnc_global = data.groupby("lncrna_id", observed=True)[
            ["positive", "row_count"]
        ].sum()
        target_global = data.groupby(["target_id", "target_type"], observed=True)[
            ["positive", "row_count"]
        ].sum()
        return cls(
            cancers=tuple(sorted(pair_by_cancer)),
            pair_global=pair_global,
            pair_by_cancer=pair_by_cancer,
            lnc_global=lnc_global,
            lnc_by_cancer=lnc_by_cancer,
            target_global=target_global,
            target_by_cancer=target_by_cancer,
            total_positive=float(data.positive.sum()),
            total_count=int(len(data)),
            totals_by_cancer=totals_by_cancer,
        )

    @staticmethod
    def _subtract(
        global_frame: pd.DataFrame,
        per_cancer: Mapping[str, pd.DataFrame],
        index: pd.Index,
        excluded: set[str],
    ) -> pd.DataFrame:
        if excluded:
            # Do not form global-minus-heldout.  Even though mathematically
            # correct, floating cancellation would let permuting a held-out
            # effect alter the last bit of a score.  Summing only admitted
            # cancers makes outcome invariance exact, not approximate.
            result = pd.DataFrame(0.0, index=index, columns=global_frame.columns)
            for cancer in sorted(set(per_cancer) - excluded):
                result += per_cancer[cancer].reindex(index).fillna(0.0).to_numpy(float)
        else:
            result = global_frame.reindex(index).fillna(0.0).astype(float)
        # Floating subtraction can leave tiny negative values in count/mass
        # columns.  Signed effect_z sums are intentionally not clipped.
        nonnegative = [
            column
            for column in result.columns
            if column != "effect_wz"
        ]
        result[nonnegative] = result[nonnegative].clip(lower=0.0)
        return result

    def features(
        self,
        query: pd.DataFrame,
        query_cancer: str,
        *,
        excluded_cancers: set[str] | None = None,
        cancer_similarity: Mapping[str, Mapping[str, float]] | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        qry = standardize_candidate_frame(query)
        query_cancer = str(query_cancer)
        observed = set(qry.cancer_id.unique())
        if observed != {query_cancer}:
            raise SimpleLocoContractError(
                f"query must contain exactly cancer {query_cancer}; observed={sorted(observed)}"
            )
        excluded = set(map(str, excluded_cancers or set()))
        if query_cancer in self.cancers and query_cancer not in excluded:
            raise SimpleLocoContractError(
                f"query cancer {query_cancer} must be wholly excluded from sufficient statistics"
            )
        remaining = [cancer for cancer in self.cancers if cancer not in excluded]
        if not remaining:
            raise SimpleLocoContractError("no source cancers remain after exclusion")
        identity = qry[list(QUERY_KEYS)].copy().reset_index(drop=True)
        pair_index = pd.MultiIndex.from_frame(identity[list(PAIR_KEYS)])
        pair = self._subtract(
            self.pair_global, self.pair_by_cancer, pair_index, excluded
        ).reset_index(drop=True)
        support_count = pair.positive.to_numpy(float)
        evaluable = pair.row_count.to_numpy(float)
        effect_weight = pair.effect_weight.to_numpy(float)
        effect_wz = pair.effect_wz.to_numpy(float)
        mean_z = np.divide(
            effect_wz,
            effect_weight,
            out=np.zeros(len(pair), dtype=float),
            where=effect_weight > 0,
        )
        q_value = np.maximum(
            0.0,
            pair.effect_wz2.to_numpy(float) - effect_weight * mean_z**2,
        )
        effect_k = pair.effect_count.to_numpy(float)
        i2 = np.divide(
            np.maximum(0.0, q_value - np.maximum(effect_k - 1.0, 0.0)),
            q_value,
            out=np.zeros(len(pair), dtype=float),
            where=q_value > 0,
        )
        known = pair.positive_known_direction.to_numpy(float)
        positive_direction = pair.positive_direction.to_numpy(float)
        direction_consistency = np.divide(
            np.maximum(positive_direction, known - positive_direction),
            known,
            out=np.full(len(pair), 0.5, dtype=float),
            where=known > 0,
        )

        similarity_positive = np.zeros(len(pair), dtype=float)
        similarity_weight_sum = np.zeros(len(pair), dtype=float)
        similarity_effect_wz = np.zeros(len(pair), dtype=float)
        similarity_effect_weight = np.zeros(len(pair), dtype=float)
        weights = _similarity_weights(
            pd.Series(remaining, dtype="string"), query_cancer, cancer_similarity
        )
        for cancer, weight in zip(remaining, weights, strict=True):
            contribution = self.pair_by_cancer[cancer].reindex(pair_index).fillna(0.0)
            similarity_positive += weight * contribution.positive.to_numpy(float)
            similarity_weight_sum += weight * contribution.row_count.to_numpy(float)
            similarity_effect_wz += weight * contribution.effect_wz.to_numpy(float)
            similarity_effect_weight += weight * contribution.effect_weight.to_numpy(float)
        similarity_support = np.divide(
            similarity_positive,
            similarity_weight_sum,
            out=np.zeros(len(pair), dtype=float),
            where=similarity_weight_sum > 0,
        )
        similarity_z = np.divide(
            similarity_effect_wz,
            similarity_effect_weight,
            out=np.zeros(len(pair), dtype=float),
            where=similarity_effect_weight > 0,
        )

        lnc_index = pd.Index(identity.lncrna_id, name="lncrna_id")
        lnc = self._subtract(
            self.lnc_global, self.lnc_by_cancer, lnc_index, excluded
        ).reset_index(drop=True)
        target_index = pd.MultiIndex.from_frame(identity[["target_id", "target_type"]])
        target = self._subtract(
            self.target_global, self.target_by_cancer, target_index, excluded
        ).reset_index(drop=True)
        total_positive = self.total_positive - sum(
            self.totals_by_cancer.get(cancer, (0.0, 0))[0] for cancer in excluded
        )
        total_count = self.total_count - sum(
            self.totals_by_cancer.get(cancer, (0.0, 0))[1] for cancer in excluded
        )
        overall = total_positive / total_count if total_count else 0.0

        result = identity.copy()
        result["support_count"] = support_count
        result["support_frequency"] = np.divide(
            support_count,
            evaluable,
            out=np.zeros(len(pair), dtype=float),
            where=evaluable > 0,
        )
        result["meta_effect_signed"] = np.tanh(mean_z)
        result["meta_effect_abs"] = np.abs(result.meta_effect_signed)
        result["direction_consistency"] = direction_consistency
        result["heterogeneity_Q"] = q_value
        result["heterogeneity_I2"] = i2
        result["n_evaluable_cancers"] = evaluable
        result["similarity_weighted_support"] = similarity_support
        result["similarity_weighted_effect_signed"] = np.tanh(similarity_z)
        result["similarity_weighted_effect_abs"] = np.abs(
            result.similarity_weighted_effect_signed
        )
        result["lncrna_support_frequency"] = np.divide(
            lnc.positive.to_numpy(float),
            lnc.row_count.to_numpy(float),
            out=np.full(len(lnc), overall, dtype=float),
            where=lnc.row_count.to_numpy(float) > 0,
        )
        result["target_support_frequency"] = np.divide(
            target.positive.to_numpy(float),
            target.row_count.to_numpy(float),
            out=np.full(len(target), overall, dtype=float),
            where=target.row_count.to_numpy(float) > 0,
        )
        result["overall_prevalence"] = overall
        result["rule_score"] = (
            result.support_frequency
            * (0.5 + 0.5 * result.direction_consistency)
            * (1.0 - 0.5 * result.heterogeneity_I2.clip(0, 1))
        ).clip(0, 1)
        result["log1p_n_evaluable_cancers"] = np.log1p(
            result.n_evaluable_cancers
        )
        result["source_cancer_count"] = len(remaining)
        result["source_cancers"] = "|".join(remaining)
        return result[
            [
                *QUERY_KEYS,
                *FEATURE_COLUMNS,
                "log1p_n_evaluable_cancers",
                "source_cancer_count",
                "source_cancers",
            ]
        ]


def standardize_candidate_frame(frame: pd.DataFrame, task_type: str | None = None) -> pd.DataFrame:
    """Return one canonical candidate row per cancer/pair.

    ``task_type`` is only needed when neither ``target_id`` nor
    ``target_type`` is already present.
    """

    out = frame.copy()
    require_columns(out, ["candidate_id", "cancer_id", "lncrna_id"], "candidate frame")
    if "target_id" not in out:
        if task_type == "state" and "state_id" in out:
            out["target_id"] = out["state_id"].astype(str)
        elif task_type == "pathway" and "pathway_family_id" in out:
            out["target_id"] = out["pathway_family_id"].astype(str)
        else:
            raise SimpleLocoContractError("target_id is missing and cannot be derived")
    if "target_type" not in out:
        if task_type not in {"state", "pathway"}:
            raise SimpleLocoContractError("target_type is missing and task_type was not supplied")
        out["target_type"] = task_type
    if "label" not in out:
        if "proxy_label" not in out:
            raise SimpleLocoContractError("candidate frame lacks label/proxy_label")
        out["label"] = pd.to_numeric(out["proxy_label"], errors="raise").astype(np.int8)
    else:
        out["label"] = pd.to_numeric(out["label"], errors="raise").astype(np.int8)
    if not out.label.isin([0, 1]).all():
        raise SimpleLocoContractError("simple LOCO labels must be binary")
    for column in ("candidate_id", "cancer_id", "lncrna_id", "target_id", "target_type"):
        out[column] = out[column].astype(str)
    duplicated = out.duplicated(["cancer_id", *PAIR_KEYS], keep=False)
    if duplicated.any():
        example = out.loc[duplicated, ["cancer_id", *PAIR_KEYS]].head(5).to_dict("records")
        raise SimpleLocoContractError(f"duplicate cancer/pair candidate rows: {example}")
    if "effect" not in out:
        out["effect"] = out["bulk_effect"] if "bulk_effect" in out else np.nan
    if "n_observed" not in out:
        out["n_observed"] = np.nan
    return out


def _similarity_weights(
    source_cancers: pd.Series,
    query_cancer: str,
    cancer_similarity: Mapping[str, Mapping[str, float]] | pd.DataFrame | None,
) -> np.ndarray:
    if cancer_similarity is None:
        return np.ones(len(source_cancers), dtype=float)
    if isinstance(cancer_similarity, pd.DataFrame):
        require_columns(
            cancer_similarity,
            ["query_cancer", "source_cancer", "weight"],
            "cancer similarity",
        )
        rows = cancer_similarity.loc[
            cancer_similarity.query_cancer.astype(str).eq(str(query_cancer))
        ]
        mapping = dict(zip(rows.source_cancer.astype(str), pd.to_numeric(rows.weight, errors="coerce")))
    else:
        mapping = dict(cancer_similarity.get(str(query_cancer), {}))
    weights = source_cancers.astype(str).map(mapping).fillna(0.0).to_numpy(float)
    if (~np.isfinite(weights)).any() or (weights < 0).any():
        raise SimpleLocoContractError("cancer-similarity weights must be finite and nonnegative")
    # A missing similarity table must not silently erase all evidence.  Equal
    # weights are a deterministic, target-independent fallback.
    if not np.any(weights > 0):
        weights = np.ones(len(source_cancers), dtype=float)
    return weights


def aggregate_cross_cancer_features(
    source: pd.DataFrame,
    query: pd.DataFrame,
    query_cancer: str,
    cancer_similarity: Mapping[str, Mapping[str, float]] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate source-cancer evidence for query candidates.

    The function fails closed if ``query_cancer`` occurs in ``source``.  It
    only reads candidate identity columns from ``query``; its labels, effects,
    FDR values, and graph predictions therefore cannot affect the result.
    """

    src = standardize_candidate_frame(source)
    qry = standardize_candidate_frame(query)
    query_cancer = str(query_cancer)
    observed_query_cancers = set(qry.cancer_id.unique())
    if observed_query_cancers != {query_cancer}:
        raise SimpleLocoContractError(
            f"query must contain exactly cancer {query_cancer}; observed={sorted(observed_query_cancers)}"
        )
    if src.cancer_id.eq(query_cancer).any():
        raise SimpleLocoContractError(f"query cancer {query_cancer} leaked into source rows")
    if src.empty:
        raise SimpleLocoContractError("source rows are empty")

    src = src.copy()
    src["effect_value"] = pd.to_numeric(src.effect, errors="coerce")
    src["effect_valid"] = src.effect_value.notna() & src.effect_value.between(-1, 1, inclusive="both")
    n_observed = pd.to_numeric(src.n_observed, errors="coerce")
    src["effect_weight"] = np.where(
        src.effect_valid,
        np.maximum(n_observed.fillna(4.0).to_numpy(float) - 3.0, 1.0),
        0.0,
    )
    src["effect_z"] = np.where(
        src.effect_valid,
        np.arctanh(src.effect_value.clip(-0.999999, 0.999999)),
        0.0,
    )
    src["effect_wz"] = src.effect_weight * src.effect_z
    src["effect_wz2"] = src.effect_weight * src.effect_z.pow(2)
    src["positive"] = src.label.astype(float)
    src["positive_known_direction"] = src.positive * src.effect_valid.astype(float)
    src["positive_direction"] = src.positive_known_direction * src.effect_value.gt(0).astype(float)
    src["similarity_weight"] = _similarity_weights(src.cancer_id, query_cancer, cancer_similarity)
    src["similarity_positive"] = src.similarity_weight * src.positive
    src["similarity_effect_weight"] = src.similarity_weight * src.effect_weight
    src["similarity_effect_wz"] = src.similarity_effect_weight * src.effect_z

    grouped = src.groupby(list(PAIR_KEYS), sort=False, observed=True)
    stats = grouped.agg(
        support_count=("positive", "sum"),
        n_evaluable_cancers=("cancer_id", "nunique"),
        effect_weight_sum=("effect_weight", "sum"),
        effect_wz_sum=("effect_wz", "sum"),
        effect_wz2_sum=("effect_wz2", "sum"),
        direction_known_count=("positive_known_direction", "sum"),
        positive_direction_count=("positive_direction", "sum"),
        similarity_weight_sum=("similarity_weight", "sum"),
        similarity_positive_sum=("similarity_positive", "sum"),
        similarity_effect_weight_sum=("similarity_effect_weight", "sum"),
        similarity_effect_wz_sum=("similarity_effect_wz", "sum"),
    )
    stats["support_frequency"] = stats.support_count / stats.n_evaluable_cancers.clip(lower=1)
    mean_z = stats.effect_wz_sum / stats.effect_weight_sum.replace(0, np.nan)
    stats["meta_effect_signed"] = np.tanh(mean_z).fillna(0.0)
    stats["meta_effect_abs"] = stats.meta_effect_signed.abs()
    stats["direction_consistency"] = np.where(
        stats.direction_known_count > 0,
        np.maximum(
            stats.positive_direction_count,
            stats.direction_known_count - stats.positive_direction_count,
        ) / stats.direction_known_count,
        0.5,
    )
    stats["heterogeneity_Q"] = (
        stats.effect_wz2_sum - stats.effect_weight_sum * mean_z.pow(2)
    ).clip(lower=0).fillna(0.0)
    effect_k = grouped.effect_valid.sum().astype(float)
    stats["heterogeneity_I2"] = np.where(
        stats.heterogeneity_Q > 0,
        np.maximum(0.0, (stats.heterogeneity_Q - np.maximum(effect_k - 1.0, 0.0)) / stats.heterogeneity_Q),
        0.0,
    )
    stats["similarity_weighted_support"] = (
        stats.similarity_positive_sum / stats.similarity_weight_sum.replace(0, np.nan)
    ).fillna(0.0)
    similarity_mean_z = (
        stats.similarity_effect_wz_sum / stats.similarity_effect_weight_sum.replace(0, np.nan)
    )
    stats["similarity_weighted_effect_signed"] = np.tanh(similarity_mean_z).fillna(0.0)
    stats["similarity_weighted_effect_abs"] = stats.similarity_weighted_effect_signed.abs()

    overall = float(src.positive.mean())
    lnc_support = src.groupby("lncrna_id", observed=True).positive.mean()
    target_support = src.groupby(["target_id", "target_type"], observed=True).positive.mean()
    identity = qry[list(QUERY_KEYS)].copy().reset_index(drop=True)
    result = identity.merge(stats.reset_index(), on=list(PAIR_KEYS), how="left", validate="many_to_one")
    result["lncrna_support_frequency"] = result.lncrna_id.map(lnc_support).fillna(overall)
    target_map = target_support.to_dict()
    result["target_support_frequency"] = [
        target_map.get((target, kind), overall)
        for target, kind in zip(result.target_id, result.target_type)
    ]
    result["overall_prevalence"] = overall
    zero_columns = (
        "support_count", "support_frequency", "meta_effect_signed", "meta_effect_abs",
        "heterogeneity_Q", "heterogeneity_I2", "n_evaluable_cancers",
        "similarity_weighted_support", "similarity_weighted_effect_signed",
        "similarity_weighted_effect_abs",
    )
    result[list(zero_columns)] = result[list(zero_columns)].fillna(0.0)
    result["direction_consistency"] = result.direction_consistency.fillna(0.5)
    result["rule_score"] = (
        result.support_frequency
        * (0.5 + 0.5 * result.direction_consistency)
        * (1.0 - 0.5 * result.heterogeneity_I2.clip(0, 1))
    ).clip(0, 1)
    result["log1p_n_evaluable_cancers"] = np.log1p(result.n_evaluable_cancers)
    result["source_cancer_count"] = int(src.cancer_id.nunique())
    result["source_cancers"] = "|".join(sorted(src.cancer_id.unique()))
    return result[[*QUERY_KEYS, *FEATURE_COLUMNS, "log1p_n_evaluable_cancers", "source_cancer_count", "source_cancers"]]


def _attach_nonparametric_scores(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["score_prevalence"] = out.overall_prevalence.clip(0, 1)
    out["score_frequency"] = out.support_frequency.clip(0, 1)
    out["score_meta_effect"] = out.meta_effect_abs.clip(0, 1)
    out["score_rule"] = out.rule_score.clip(0, 1)
    out["score_similarity_weighted"] = (
        0.65 * out.similarity_weighted_support + 0.35 * out.similarity_weighted_effect_abs
    ).clip(0, 1)
    return out


def _fit_logistic(features: pd.DataFrame, labels: np.ndarray, seed: int):
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return None, float(labels.mean()) if len(labels) else 0.5
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=int(seed),
        ),
    )
    model.fit(features[list(LOGISTIC_FEATURE_COLUMNS)].to_numpy(float), labels)
    return model, float(labels.mean())


def _predict_logistic(model, fallback: float, features: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.full(len(features), fallback, dtype=float)
    return model.predict_proba(features[list(LOGISTIC_FEATURE_COLUMNS)].to_numpy(float))[:, 1]


def nested_cancer_crossfit_predictions(
    train: pd.DataFrame,
    *,
    cancer_similarity: Mapping[str, Mapping[str, float]] | pd.DataFrame | None = None,
    seed: int = 20260814,
    source_universe: pd.DataFrame | None = None,
    statistics: CrossCancerSufficientStatistics | None = None,
) -> pd.DataFrame:
    """Produce simple scores whose complete fitting path excludes each query cancer.

    For a held-out training cancer ``c``, the logistic model is trained on
    rows from cancers ``d != c``.  Each such row is itself represented using
    sources that exclude both ``c`` and ``d``.  Thus ``c`` cannot leak through
    aggregate features used to fit the logistic coefficients.
    """

    data = standardize_candidate_frame(train)
    cancers = sorted(data.cancer_id.unique())
    if len(cancers) < 3:
        raise SimpleLocoContractError("nested cancer cross-fit requires at least three cancers")
    if statistics is not None and source_universe is not None:
        raise ValueError("provide statistics or source_universe, not both")
    stats = statistics or CrossCancerSufficientStatistics.from_frame(
        source_universe if source_universe is not None else data
    )
    if set(stats.cancers) != set(cancers):
        raise SimpleLocoContractError(
            "sufficient-statistics cancers must exactly equal cross-fit training cancers"
        )
    outputs: list[pd.DataFrame] = []
    for heldout in cancers:
        outer_source = data.loc[~data.cancer_id.eq(heldout)].copy()
        query = data.loc[data.cancer_id.eq(heldout)].copy()
        query_features = _attach_nonparametric_scores(
            stats.features(
                query,
                heldout,
                excluded_cancers={heldout},
                cancer_similarity=cancer_similarity,
            )
        )
        inner_parts: list[pd.DataFrame] = []
        for inner_heldout in sorted(outer_source.cancer_id.unique()):
            inner_query = outer_source.loc[outer_source.cancer_id.eq(inner_heldout)]
            inner_features = stats.features(
                inner_query,
                inner_heldout,
                excluded_cancers={heldout, inner_heldout},
                cancer_similarity=cancer_similarity,
            )
            inner_labels = inner_query.set_index("candidate_id").label
            inner_features["label"] = inner_features.candidate_id.map(inner_labels).astype(np.int8)
            inner_parts.append(inner_features)
        logistic_train = pd.concat(inner_parts, ignore_index=True)
        model, fallback = _fit_logistic(logistic_train, logistic_train.label.to_numpy(), seed)
        query_features["score_logistic"] = _predict_logistic(model, fallback, query_features)
        labels = query.set_index("candidate_id").label
        query_features["label"] = query_features.candidate_id.map(labels).astype(np.int8)
        query_features["crossfit_heldout_cancer"] = heldout
        outputs.append(query_features)
    result = pd.concat(outputs, ignore_index=True)
    if result.candidate_id.duplicated().any():
        raise SimpleLocoContractError("candidate_id is not globally unique in cross-fit output")
    return result


def _subtype(frame: pd.DataFrame) -> pd.Series:
    state = frame.target_type.eq("state")
    rnass = frame.target_id.str.contains("RNAss", case=False, regex=False)
    return pd.Series(np.where(~state, "Pathway", np.where(rnass, "RNAss", "DNAss")), index=frame.index)


def _select_methods(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = oof.copy()
    work["target_subtype"] = _subtype(work)
    metric_rows: list[dict[str, object]] = []
    for (cancer, subtype), group in work.groupby(
        ["cancer_id", "target_subtype"], observed=True, sort=True
    ):
        for method, column in SCORE_COLUMNS.items():
            row = {
                "cancer_id": cancer,
                "target_subtype": subtype,
                "method": method,
                **binary_metrics(group.label, group[column]),
            }
            metric_rows.append(row)
    per_cancer = pd.DataFrame(metric_rows)
    macro = per_cancer.groupby(["target_subtype", "method"], observed=True).agg(
        macro_auprc=("auprc", "mean"),
        macro_auroc=("auroc", "mean"),
        evaluable_cancers=("auprc", "count"),
    ).reset_index()
    selections: list[dict[str, object]] = []
    for subtype, group in macro.groupby("target_subtype", observed=True, sort=True):
        eligible = group.loc[group.macro_auprc.notna()].sort_values(
            ["macro_auprc", "macro_auroc", "method"],
            ascending=[False, False, True],
        )
        selected = "prevalence" if eligible.empty else str(eligible.iloc[0].method)
        selections.append({"target_subtype": subtype, "selected_method": selected})
    return per_cancer.merge(macro, on=["target_subtype", "method"], how="left"), pd.DataFrame(selections)


def _final_logistic_training_rows(
    train: pd.DataFrame,
    cancer_similarity: Mapping[str, Mapping[str, float]] | pd.DataFrame | None,
    statistics: CrossCancerSufficientStatistics,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for cancer in sorted(train.cancer_id.unique()):
        query = train.loc[train.cancer_id.eq(cancer)]
        features = statistics.features(
            query,
            cancer,
            excluded_cancers={str(cancer)},
            cancer_similarity=cancer_similarity,
        )
        labels = query.set_index("candidate_id").label
        features["label"] = features.candidate_id.map(labels).astype(np.int8)
        parts.append(features)
    return pd.concat(parts, ignore_index=True)


def fit_best_simple_loco(
    train: pd.DataFrame,
    queries: Mapping[str, pd.DataFrame],
    *,
    cancer_similarity: Mapping[str, Mapping[str, float]] | pd.DataFrame | None = None,
    seed: int = 20260814,
    source_universe: pd.DataFrame | None = None,
    selection_split: str | None = None,
) -> BestSimpleFit:
    """Cross-fit, select, and freeze BestSimpleLOCO for an outer fold.

    By default selection uses macro AUPRC from cancer-level OOF predictions
    within the outer training cancers.  Passing ``selection_split="val"``
    implements the preregistered outer-validation choice; test labels remain
    inaccessible in either case.  The returned ``z_base`` is a frozen offset
    suitable for additive residual training.
    """

    training = standardize_candidate_frame(train)
    statistics = CrossCancerSufficientStatistics.from_frame(
        source_universe if source_universe is not None else training
    )
    if set(statistics.cancers) != set(training.cancer_id.unique()):
        raise SimpleLocoContractError(
            "source universe must contain exactly the outer training cancers"
        )
    oof = nested_cancer_crossfit_predictions(
        training,
        cancer_similarity=cancer_similarity,
        seed=seed,
        statistics=statistics,
    )
    selection_metrics, selections = _select_methods(oof)
    selection_map = dict(zip(selections.target_subtype, selections.selected_method))
    oof["target_subtype"] = _subtype(oof)
    oof["best_simple_method"] = oof.target_subtype.map(selection_map)
    if oof.best_simple_method.isna().any():
        raise SimpleLocoContractError("cross-fit output lacks a selected simple method")
    oof["best_simple_score"] = [
        float(oof.iloc[i][SCORE_COLUMNS[method]])
        for i, method in enumerate(oof.best_simple_method)
    ]
    oof_probability = oof.best_simple_score.clip(1e-7, 1 - 1e-7)
    oof["z_base"] = np.log(oof_probability / (1.0 - oof_probability))
    oof["prediction_scale"] = "raw_probability"
    oof["metric_scope"] = "cancer_crossfit_OOF"
    oof["simple_base_fit_cancers"] = oof.source_cancers

    logistic_rows = _final_logistic_training_rows(
        training, cancer_similarity, statistics
    )
    model, fallback = _fit_logistic(logistic_rows, logistic_rows.label.to_numpy(), seed)
    query_predictions: dict[str, pd.DataFrame] = {}
    train_cancers = set(training.cancer_id.unique())
    for split_name, raw_query in queries.items():
        query = standardize_candidate_frame(raw_query)
        overlap = train_cancers.intersection(query.cancer_id.unique())
        if overlap:
            raise SimpleLocoContractError(
                f"query split {split_name} overlaps outer training cancers: {sorted(overlap)}"
            )
        parts: list[pd.DataFrame] = []
        for cancer, cancer_query in query.groupby("cancer_id", observed=True, sort=True):
            features = _attach_nonparametric_scores(
                statistics.features(
                    cancer_query,
                    str(cancer),
                    excluded_cancers=set(),
                    cancer_similarity=cancer_similarity,
                )
            )
            features["score_logistic"] = _predict_logistic(model, fallback, features)
            labels = cancer_query.set_index("candidate_id").label
            features["label"] = features.candidate_id.map(labels).astype(np.int8)
            parts.append(features)
        prediction = pd.concat(parts, ignore_index=True)
        prediction["target_subtype"] = _subtype(prediction)
        prediction["best_simple_method"] = prediction.target_subtype.map(selection_map)
        if prediction.best_simple_method.isna().any():
            missing = sorted(prediction.loc[prediction.best_simple_method.isna(), "target_subtype"].unique())
            raise SimpleLocoContractError(f"no cross-fit method selection for subtypes {missing}")
        prediction["best_simple_score"] = [
            float(prediction.iloc[i][SCORE_COLUMNS[method]])
            for i, method in enumerate(prediction.best_simple_method)
        ]
        clipped = prediction.best_simple_score.clip(1e-7, 1 - 1e-7)
        prediction["z_base"] = np.log(clipped / (1.0 - clipped))
        prediction["prediction_scale"] = "raw_probability"
        prediction["metric_scope"] = f"{split_name}"
        prediction["simple_base_fit_cancers"] = "|".join(sorted(train_cancers))
        query_predictions[str(split_name)] = prediction
    selection_scope = "training_cancer_crossfit_OOF"
    if selection_split is not None:
        if selection_split not in query_predictions:
            raise SimpleLocoContractError(
                f"selection split {selection_split} was not supplied as a query"
            )
        if str(selection_split).lower() in {"test", "loco_test"}:
            raise SimpleLocoContractError("BestSimpleLOCO selection on test is forbidden")
        selection_metrics, selections = _select_methods(
            query_predictions[selection_split]
        )
        selection_map = dict(zip(selections.target_subtype, selections.selected_method))
        selection_scope = str(selection_split)

        def overwrite_selection(frame: pd.DataFrame) -> None:
            frame["best_simple_method"] = frame.target_subtype.map(selection_map)
            if frame.best_simple_method.isna().any():
                missing = sorted(
                    frame.loc[frame.best_simple_method.isna(), "target_subtype"].unique()
                )
                raise SimpleLocoContractError(
                    f"validation selection lacks target subtypes {missing}"
                )
            frame["best_simple_score"] = [
                float(frame.iloc[i][SCORE_COLUMNS[method]])
                for i, method in enumerate(frame.best_simple_method)
            ]
            clipped_score = frame.best_simple_score.clip(1e-7, 1 - 1e-7)
            frame["z_base"] = np.log(clipped_score / (1.0 - clipped_score))

        overwrite_selection(oof)
        for prediction in query_predictions.values():
            overwrite_selection(prediction)
    oof["best_simple_selection_scope"] = selection_scope
    for prediction in query_predictions.values():
        prediction["best_simple_selection_scope"] = selection_scope
    selections["selection_scope"] = selection_scope
    return BestSimpleFit(
        oof_predictions=oof,
        selection_metrics=selection_metrics,
        selected_methods=selections,
        query_predictions=query_predictions,
    )
