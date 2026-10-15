"""Observed external metrics, tied-rank recovery and paired cluster uncertainty."""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

from .evidence_rule_summary import KEYS, candidate_aliases, stable_id

SCOPES = ("descriptive", "provenance_held_out", "pair_held_out", "lncrna_held_out")
CLASSIFICATION = ("auroc", "average_precision", "brier", "logloss")


def validate_scores(models):
    if not models: raise ValueError("At least one locked model score table is required")
    reference = None
    for model, frame in models.items():
        if not model or "score" not in frame: raise ValueError("Invalid external model score declaration")
        candidate_aliases(frame[KEYS])
        values = pd.to_numeric(frame.score, errors="coerce")
        if not np.isfinite(values).all() or not values.between(0, 1).all(): raise ValueError("External scores must be complete finite values in [0,1]")
        keys = frame[KEYS].sort_values(KEYS).reset_index(drop=True)
        if reference is None: reference = keys
        elif not keys.equals(reference): raise ValueError("Compared models must have the same complete exact candidate universe")


def aggregate_scores(frame, endpoint, policy):
    if endpoint == "EXACT_PATHWAY": return frame[[*KEYS, "score"]].copy()
    if endpoint != "CANCER_LNCRNA_KNOWN_POSITIVE": raise ValueError("Unknown external validation endpoint")
    ordered = frame.sort_values(["cancer_id", "lncrna_id", "score", "pathway_id"], ascending=[True, True, False, True])
    group = ordered.groupby(["cancer_id", "lncrna_id"], sort=False)
    maximum = group.score.max().rename("max")
    top5 = group.head(5).groupby(["cancer_id", "lncrna_id"]).score.mean().rename("top5")
    values = pd.concat([maximum, top5], axis=1)
    if policy == "MAX": values["score"] = values["max"]
    elif policy == "MEAN_TOP5": values["score"] = values.top5
    elif policy == "LEGACY_0.7_MAX_0.3_TOP5": values["score"] = .7*values["max"]+.3*values.top5
    else: raise ValueError("lncRNA ranking aggregation must be declared before evaluation")
    return values[["score"]].reset_index().assign(pathway_id="")


def novel_candidates(frame, used, scope, endpoint):
    if scope in {"descriptive", "provenance_held_out"}: return frame.copy()
    used_lnc = set(used.lncrna_id)-{""}
    if scope == "lncrna_held_out" or endpoint == "CANCER_LNCRNA_KNOWN_POSITIVE":
        return frame.loc[~frame.lncrna_id.isin(used_lnc)].copy()
    unspecified = set(used.loc[used.pathway_id.eq(""), "lncrna_id"])-{""}
    pairs = set(used[["lncrna_id", "pathway_id"]].itertuples(index=False, name=None))
    keep = [lnc not in unspecified and (lnc, pathway) not in pairs for lnc, pathway in frame[["lncrna_id", "pathway_id"]].itertuples(index=False, name=None)]
    return frame.loc[keep].copy()


def deduplicate_targets(events):
    rows = []
    for keys, group in events.groupby(KEYS, sort=True):
        labels = set(group.label)
        rows.append({**dict(zip(KEYS, keys)), "label": next(iter(labels)) if len(labels) == 1 else np.nan,
            "label_conflict": len(labels)>1, "source_records": len(group), "study_ids_json": json.dumps(sorted(set(group.study_id))),
            "source_ids_json": json.dumps(sorted(set(group.source_id))), "provenance_groups": group.study_id.nunique()})
    return pd.DataFrame(rows, columns=[*KEYS, "label", "label_conflict", "source_records", "study_ids_json", "source_ids_json", "provenance_groups"])


def cluster_labels(targets):
    """Shared lncRNA or overlapping study provenance creates one resampling block."""
    parent = {}
    def find(token):
        parent.setdefault(token, token)
        while parent[token] != token:
            parent[token] = parent[parent[token]]; token = parent[token]
        return token
    row_tokens = []
    for row in targets.itertuples():
        tokens = ["LNC:"+row.lncrna_id, *json.loads(row.study_ids_json)]
        for token in tokens: find(token)
        for token in tokens[1:]: parent[find(token)] = find(tokens[0])
        row_tokens.append(tokens[0])
    return np.array([find(token) for token in row_tokens], dtype=str)


def classification_metrics(labels, scores, weights=None):
    labels = np.asarray(labels, dtype=float); scores = np.asarray(scores, dtype=float)
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
    if len(weights) != len(labels) or not np.isfinite(weights).all() or (weights < 0).any(): raise ValueError("Invalid observation weights")
    keep = np.isfinite(labels) & np.isfinite(scores) & (weights > 0); y = labels[keep]; p = scores[keep]; w = weights[keep]
    if not np.isin(y, [0., 1.]).all(): raise ValueError("External binary metrics require measured 0/1 labels")
    if not np.all((p >= 0) & (p <= 1)): raise ValueError("External probability metrics require scores in [0,1]")
    if len(np.unique(y)) != 2: return {key: np.nan for key in CLASSIFICATION}
    positive = float((w*y).sum()); negative = float((w*(1-y)).sum())
    unique, group = np.unique(p, return_inverse=True)
    pos_group = np.bincount(group, weights=w*y, minlength=len(unique))
    neg_group = np.bincount(group, weights=w*(1-y), minlength=len(unique))
    auc = np.sum(pos_group*(np.cumsum(neg_group)-.5*neg_group))/(positive*negative)
    true_positive = np.cumsum(pos_group[::-1]); total = np.cumsum((pos_group+neg_group)[::-1])
    precision = true_positive/total
    ap = np.sum(np.diff(np.r_[0., true_positive/positive])*precision)
    clipped = np.clip(p, 1e-12, 1-1e-12)
    return {"auroc": float(auc), "average_precision": float(ap), "brier": float(np.average((y-p)**2, weights=w)),
        "logloss": float(-np.average(y*np.log(clipped)+(1-y)*np.log1p(-clipped), weights=w))}


def rank_details(targets, candidate_scores, cutoffs):
    result = targets.merge(candidate_scores[[*KEYS, "score"]], on=KEYS, how="left", validate="one_to_one")
    ordered = np.sort(candidate_scores.score.to_numpy(float)); scores = result.score.to_numpy(float)
    valid = np.isfinite(scores); left = np.searchsorted(ordered, scores, side="left"); right = np.searchsorted(ordered, scores, side="right")
    greater = len(ordered)-right; ties = right-left
    result["rank_min"] = np.where(valid, greater+1, np.nan)
    result["rank_max"] = np.where(valid, greater+ties, np.nan)
    result["rank_midpoint"] = (result.rank_min+result.rank_max)/2
    for cutoff in cutoffs:
        result[f"hit_{cutoff}"] = np.where(valid, np.clip((min(cutoff, len(ordered))-greater)/np.maximum(ties, 1), 0, 1), 0.)
    return result


def array_metrics(labels, scores, hits, cutoffs, weights=None):
    weights = np.ones(len(labels)) if weights is None else weights
    metrics = classification_metrics(labels, scores, weights)
    positive = (labels == 1) & (weights > 0); matched = np.isfinite(scores)
    for index, cutoff in enumerate(cutoffs):
        metrics[f"recall_all_positive_at_{cutoff}"] = float(np.average(hits[positive, index], weights=weights[positive])) if positive.any() else np.nan
        subset = positive & matched
        metrics[f"recall_matched_positive_at_{cutoff}"] = float(np.average(hits[subset, index], weights=weights[subset])) if subset.any() else np.nan
    return metrics


def metric_arrays(details, cutoffs):
    return (details.label.to_numpy(float), details.score.to_numpy(float),
            details[[f"hit_{cutoff}" for cutoff in cutoffs]].to_numpy(float))


def point_metrics(details, cutoffs):
    return array_metrics(*metric_arrays(details, cutoffs), cutoffs)


def evaluate_external(external, used, models, *, primary_model, cutoffs=(10, 25, 50, 100),
                      lncrna_aggregation="MAX", bootstrap_replicates=1000, seed=0):
    validate_scores(models)
    if primary_model not in models: raise ValueError("The primary comparison model must be locked before evaluation")
    if (not cutoffs or len(set(cutoffs)) != len(cutoffs) or any(type(k) is not int or k < 1 for k in cutoffs)
            or type(bootstrap_replicates) is not int or bootstrap_replicates < 0
            or (bootstrap_replicates and bootstrap_replicates < 20)):
        raise ValueError("Invalid external metric/uncertainty settings")
    metrics_rows = []; detail_frames = []; coverage_rows = []
    source_groups = ["ALL_SOURCES", *sorted(external.source_id.unique())]
    for endpoint, endpoint_events in external.groupby("endpoint", sort=True):
        aggregated = {name: aggregate_scores(frame, endpoint, lncrna_aggregation) for name, frame in models.items()}
        for cancer, cancer_events in endpoint_events.groupby("cancer_id", sort=True):
            for source_group in source_groups:
                raw = cancer_events if source_group == "ALL_SOURCES" else cancer_events.loc[cancer_events.source_id.eq(source_group)]
                if raw.empty: continue
                for scope in SCOPES:
                    column = "descriptive_eligible" if scope == "descriptive" else scope
                    eligible = raw.loc[raw[column]]
                    targets = deduplicate_targets(eligible)
                    clean = targets.loc[~targets.label_conflict.astype(bool)].reset_index(drop=True)
                    meta = {"endpoint": endpoint, "cancer_id": cancer, "source_group": source_group, "isolation_scope": scope}
                    coverage = {**meta, "source_records": len(raw), "eligible_source_records": len(eligible),
                        "deduplicated_targets": len(targets), "conflicting_targets_excluded": int(targets.label_conflict.sum()),
                        "eligible_positive_targets": int(clean.label.eq(1).sum()), "eligible_negative_targets": int(clean.label.eq(0).sum()),
                        "model_usage_provenance_complete": bool(raw.model_usage_provenance_complete.all()),
                        "model_usage_entity_complete": bool(raw.model_usage_entity_complete.all()),
                        "independence_scope": "REGISTERED_MODEL_USE_ONLY", "unregistered_model_use_excluded": False}
                    groups = cluster_labels(clean); block_ids = sorted(set(groups))
                    coverage["resampling_blocks"] = len(block_ids)
                    details = {}; point = {}; samples = {}; arrays = {}
                    for name, scores in aggregated.items():
                        universe = novel_candidates(scores.loc[scores.cancer_id.eq(cancer)], used, scope, endpoint)
                        details[name] = rank_details(clean, universe, cutoffs)
                        arrays[name] = metric_arrays(details[name], cutoffs)
                        point[name] = array_metrics(*arrays[name], cutoffs)
                        samples[name] = {key: [] for key in point[name]}
                        coverage_rows.append({**coverage, "model_id": name, "ranking_universe": len(universe),
                            "matched_targets": int(details[name].score.notna().sum()),
                            "matched_positive_targets": int((details[name].label.eq(1) & details[name].score.notna()).sum())})
                        if len(details[name]): detail_frames.append(details[name].assign(**meta, model_id=name))
                    # Identical resampled blocks across models: differences are paired.
                    if len(block_ids) >= 2 and bootstrap_replicates:
                        local_seed = int(stable_id("", [seed, meta])[:16], 16)
                        random = np.random.default_rng(local_seed)
                        group_indices = np.searchsorted(block_ids, groups)
                        for _ in range(bootstrap_replicates):
                            block_counts = np.bincount(random.integers(0, len(block_ids), len(block_ids)), minlength=len(block_ids))
                            weights = block_counts[group_indices]
                            for name in models:
                                sample = array_metrics(*arrays[name], cutoffs, weights=weights)
                                for key, value in sample.items(): samples[name][key].append(value)
                    for name in models:
                        for key, value in point[name].items():
                            distribution = np.asarray(samples[name][key], dtype=float)
                            comparisons = [(name, "", value, distribution)]
                            if name != primary_model:
                                comparisons.append((primary_model, name, point[primary_model][key]-value,
                                    np.asarray(samples[primary_model][key], dtype=float)-distribution))
                            for model_id, baseline, estimate, draws in comparisons:
                                finite = draws[np.isfinite(draws)]
                                ci = len(finite) >= max(20, int(.8*bootstrap_replicates)) and np.isfinite(estimate)
                                available = bool(np.isfinite(estimate))
                                metrics_rows.append({**meta, "model_id": model_id, "baseline_model_id": baseline,
                                    "metric": key, "value": float(estimate) if available else np.nan, "available": available,
                                    "unavailable_reason": None if available else "NO_MEASURED_CLASS_CONTRAST" if key in CLASSIFICATION else "NO_ELIGIBLE_MATCHED_POSITIVES" if "matched" in key else "NO_ELIGIBLE_POSITIVES",
                                    "ci_low": float(np.quantile(finite, .025)) if ci else np.nan,
                                    "ci_high": float(np.quantile(finite, .975)) if ci else np.nan,
                                    "ci_available": bool(ci), "valid_bootstrap_replicates": len(finite),
                                    "ci_unavailable_reason": None if ci else "NOT_REQUESTED" if not bootstrap_replicates else "METRIC_UNAVAILABLE" if not available else "INSUFFICIENT_RESAMPLING_BLOCKS" if len(block_ids)<2 else "INSUFFICIENT_VALID_BOOTSTRAP_REPLICATES",
                                    "higher_is_better": key not in {"brier", "logloss"}, "delta_definition": "PRIMARY_MINUS_BASELINE" if baseline else "NOT_A_DELTA",
                                    "tie_policy": "EXPECTED_UNIFORM_SELECTION_WITHIN_EQUAL_SCORE_TIES", "negative_labels_synthesized": False,
                                    "intervals_are_exploratory_unadjusted": True, "used_for_model_selection": False})
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(columns=[*KEYS, "label", "score", "endpoint", "source_group", "isolation_scope", "model_id"])
    return pd.DataFrame(metrics_rows), details, pd.DataFrame(coverage_rows)
