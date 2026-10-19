"""Preserve training-fold detection through inference and selected Gene Sets.

Agreement is descriptive: mixed fold classifications are never majority-voted
into shared/local. Scores remain the registered five-fold prediction mean.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from .current_core_export import verified_path
from .formal_expression_sources import read
from .evidence_rule_summary import KEYS

ENSEMBLE_BASIS = "FIVE_TRAINING_FOLD_AGREEMENT"
FOLD_BASIS = "OUTER_TRAIN_PATIENT_DETECTION"
STATES = ("shared", "cancer_local", "not_expression_eligible")
FOLD_COLUMNS = [f"scope_fold_{fold}" for fold in range(5)]
COUNTS = {"shared": "scope_shared_fold_count", "cancer_local": "scope_local_fold_count",
          "not_expression_eligible": "scope_ineligible_fold_count"}
ANNOTATIONS = ["scope_basis", *FOLD_COLUMNS, *COUNTS.values(), "scope_eligible_fold_count"]


def manifest_roles(manifest):
    native = "fold_local_scope" in manifest
    if native == ("lncrna_scope" in manifest):
        raise ValueError("Declare exactly one legacy or fold-local prediction scope")
    return (["candidate_authority", "fold_local_scope", "patient_map", "pathway_hierarchy"] if native
            else ["candidate_authority", "lncrna_scope"])


def summarize_scopes(frames):
    if set(frames) != set(range(5)):
        raise ValueError("Prediction scope requires all five training folds")
    result = None
    for fold in range(5):
        frame = frames[fold].sort_values(["cancer_id", "lncrna_id"]).reset_index(drop=True)
        if (frame.duplicated(["cancer_id", "lncrna_id"]).any()
                or not frame.shared_or_local_scope.isin(STATES).all()
                or not frame.fold_id.eq(fold).all() or not frame.fit_split.eq("train").all()):
            raise ValueError("Invalid per-fold detection classifications")
        keys = frame[["cancer_id", "lncrna_id"]]
        if result is None: result = keys.copy()
        elif not result[list(keys)].equals(keys): raise ValueError("Fold scope candidate grids differ")
        result[FOLD_COLUMNS[fold]] = frame.shared_or_local_scope
    values = result[FOLD_COLUMNS]
    for state, column in COUNTS.items(): result[column] = values.eq(state).sum(axis=1).astype(int)
    result["scope_eligible_fold_count"] = 5 - result.scope_ineligible_fold_count
    result["shared_or_local_scope"] = np.where(values.nunique(axis=1).eq(1), values.iloc[:, 0], "fold_variable")
    result["scope_basis"] = ENSEMBLE_BASIS
    validate_annotations(result)
    return result


def validate_annotations(frame):
    native = "scope_basis" in frame
    if not native:
        if set(ANNOTATIONS) & set(frame) or not frame.shared_or_local_scope.isin(STATES[:2]).all():
            raise ValueError("Non-legacy scope needs complete native fold annotations")
        return False
    if not set(ANNOTATIONS) <= set(frame) or not frame.scope_basis.eq(ENSEMBLE_BASIS).all():
        raise ValueError("Incomplete five-fold scope annotations")
    values = frame[FOLD_COLUMNS]
    if not values.isin(STATES).all().all(): raise ValueError("Invalid native fold scope state")
    for state, column in COUNTS.items():
        if not frame[column].eq(values.eq(state).sum(axis=1)).all():
            raise ValueError("Scope fold counts differ from actual fold states")
    if not frame.scope_eligible_fold_count.eq(5 - frame.scope_ineligible_fold_count).all():
        raise ValueError("Scope eligible fold count differs")
    expected = np.where(values.nunique(axis=1).eq(1), values.iloc[:, 0], "fold_variable")
    if not np.array_equal(frame.shared_or_local_scope.to_numpy(), expected):
        raise ValueError("Scope consensus differs from five-fold agreement")
    return True


def annotated_candidates(manifest, root, candidates):
    """Read the actual scope producer, not a user-relabelled aggregate table."""
    from .fold_local_scope import release, scope_input
    from .evidence_rule_sources import read_table
    roles = manifest_roles(manifest)
    sources = {key: verified_path(manifest[key], root) for key in roles}
    path, value = release(manifest["fold_local_scope"], root,
                          candidates=sources["candidate_authority"], patient_map=sources["patient_map"])
    frames = {fold: scope_input(value, path.parent, fold)[0] for fold in range(5)}
    scope = summarize_scopes(frames)
    hierarchy = read_table(sources["pathway_hierarchy"])
    from scripts.prepare_v32_formal import build_hierarchy
    hierarchy = build_hierarchy(sorted(candidates.pathway_id.unique()), hierarchy)
    result = candidates[KEYS].merge(hierarchy, on="pathway_id", how="left", validate="many_to_one")
    result = result.merge(scope, on=["cancer_id", "lncrna_id"], how="left", validate="many_to_one")
    if result[ANNOTATIONS].isna().any().any(): raise ValueError("Native scopes do not cover prediction candidates")
    validate_annotations(result)
    return result


def fold_annotations(candidates, fold):
    if not validate_annotations(candidates): return candidates.copy()
    result = candidates[[*KEYS, "pathway_family_id"]].copy()
    result["shared_or_local_scope"] = candidates[FOLD_COLUMNS[fold]]
    result["scope_basis"] = FOLD_BASIS
    result["scope_outer_fold"] = fold
    return result


def fold_source_record(manifest, root, fold):
    if "fold_local_scope" not in manifest: return None
    path = verified_path(manifest["fold_local_scope"], root)
    source = read(path)["folds"][str(fold)]
    return {"path": str(verified_path(source, path.parent)), "sha256": source["sha256"]}


def validate_prepared_scope(manifest, root, payload, candidates, batches, fold):
    """Bind the prediction's annotations to the scope actually used in training."""
    reads = payload.get("preparation_sources", {}).get("source_reads", [])
    detection = [r for r in reads if r["role"] == "detection"]
    members = [m for r in detection for m in r["members"] if m.get("upstream_scope")]
    if "fold_local_scope" not in manifest:
        if members: raise ValueError("Fold-local training cannot use legacy global prediction scope")
        return None
    expected = fold_source_record(manifest, root, fold)
    if (len(detection) != 1 or detection[0]["fold"] != fold or len(members) != 1
            or members[0]["upstream_scope"] != expected):
        raise ValueError("Prediction scope differs from actual prepared training scope")
    for role in ("candidate_authority", "patient_map", "pathway_hierarchy"):
        matched = [r for r in reads if r["role"] == role]
        if (len(matched) != 1 or matched[0]["sha256"] != manifest[role]["sha256"]
                or Path(matched[0]["path"]).resolve() != verified_path(manifest[role], root)):
            raise ValueError("Prediction input differs from actual preparation: " + role)
    from .fold_local_scope import validate_scope
    path, value = validate_scope(expected, root, members[0]["path"])
    scope = pd.read_parquet(verified_path(value["output"], path.parent))
    joined = candidates[KEYS].merge(scope, on=["cancer_id", "lncrna_id"], how="left", validate="many_to_one")
    if not np.array_equal(joined.shared_or_local_scope, candidates[FOLD_COLUMNS[fold]]):
        raise ValueError("Fold annotation differs from consumed detection scope")
    features = np.column_stack([joined.shared_or_local_scope.eq("shared"), joined.detection_rate,
                                joined.detected_cancers / joined.assessed_cancers]).astype(float)
    for batch in batches:
        observed = batch["conservation_context"].numpy()[:, [0, 1, 3]]
        if not np.allclose(observed, features[batch["candidate_positions"]], rtol=0, atol=1e-7):
            raise ValueError("Prediction features differ from bound training detection")
    return expected


def geneset_eligibility(scores, min_expression_folds):
    native = validate_annotations(scores)
    if native:
        if type(min_expression_folds) is not int or min_expression_folds not in range(1, 6):
            raise ValueError("Fold-local GeneSets require explicit min_expression_folds from 1 to 5")
        return scores.scope_eligible_fold_count.ge(min_expression_folds)
    if min_expression_folds is not None:
        raise ValueError("An expression-fold threshold requires native fold evidence")
    return pd.Series(True, index=scores.index)
