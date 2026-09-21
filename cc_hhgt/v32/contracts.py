from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTRACT_VERSION = "3.2.0"
ANALYSIS_VERSION = "CancerLncAtlas_V3.2_RS_CC-HHGT_1seed_CODE_ONLY"
TARGET_LEVEL = "exact_pathway"
TARGET_KEYS = ("cancer_id", "lncrna_id", "pathway_id")

PREDICTION_COLUMNS = (
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
    "association_membership_probability",
    "association_direction",
    "l1_probability",
    "ridge_probability",
    "graph_residual",
    "graph_gate",
    "fold_rank_percentile",
    "fold_selection_frequency",
    "shared_or_local_scope",
    "regulatory_evidence_confidence",
)

FORBIDDEN_PRIMARY_FEATURES = frozenset(
    {
        "bulk_available",
        "sc_available",
        "ucell_available",
        "replication_available",
        "interaction_available",
        "drug_available",
        "literature_support",
        "interaction_support",
        "perturbation_support",
        "drug_support",
        "observed_evidence_score",
        "regulatory_evidence_confidence",
        "proxy_label",
        "label",
        "label_class",
        "strong_association_label",
    }
)


def default_contract() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "execution_control": {
            "execution_mode": "CODE_ONLY",
            "training_authorized": False,
            "paid_enabled": False,
            "max_paid_hours": 0.0,
            "max_cost_cny": 0.0,
        },
        "task_contract": {
            "endpoint": "cancer_lncrna_exact_pathway_association_relevance",
            "target_level": TARGET_LEVEL,
            "target_column": "pathway_id",
            "target_keys": list(TARGET_KEYS),
            "pathway_family_role": "auxiliary_hierarchy_only",
            "family_may_replace_target": False,
            "patient_activity_is_separate_endpoint": True,
        },
        "candidate_universe": {
            "annotation_lncRNAs": 16889,
            "shared_lncRNAs": 4712,
            "within_cancer_min_detection_rate": 0.10,
            "minimum_detected_cancers": 3,
            "detection_rule": "logCPM_gt_0",
            "local_only_retained": True,
        },
        "crossfit": {
            "cancers": 33,
            "patient_folds": 5,
            "seed": 20260726,
            "validation_offset": 1,
        },
        "primary_model": {
            "kind": "cc_hhgt_bounded_l1_residual",
            "formula": "z_final=z_l1+lambda_cp*tanh(delta_cc_hhgt)",
            "pair_evidence_in_primary": False,
            "evidence_confidence_is_separate": True,
            "subtype_feedback": False,
        },
        "subtypes": {
            "top_k": 200,
            "rbo_p": 0.98,
            "rank_weight": 0.50,
            "composition_weight": 0.50,
            "pathway_k_candidates": [1, 2, 3, 4],
            "silhouette_min": 0.25,
            "bootstrap_ari_min": 0.75,
            "min_cancers": 6,
            "min_members": 10,
            "subtype_feedback": False,
        },
    }


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame, key: Sequence[str] | None = None) -> str:
    """Order-invariant hash for a typed table.

    Column order is canonicalized.  Rows are sorted by ``key`` when supplied,
    otherwise by every column.  Missing values are represented explicitly.
    """

    columns = sorted(map(str, frame.columns))
    canonical = frame.loc[:, columns].copy()
    sort_columns = list(key or columns)
    missing = sorted(set(sort_columns) - set(columns))
    if missing:
        raise ValueError(f"Hash key columns are missing: {missing}")
    for column in canonical.columns:
        if pd.api.types.is_float_dtype(canonical[column]):
            canonical[column] = pd.to_numeric(canonical[column], errors="coerce").round(12)
    canonical = canonical.sort_values(sort_columns, kind="stable", na_position="first")
    records = canonical.replace({np.nan: None}).to_dict("records")
    return canonical_json_sha256({"columns": columns, "records": records})


def validate_contract(config: Mapping[str, Any]) -> None:
    task = config.get("task_contract", {})
    if task.get("target_level") != TARGET_LEVEL:
        raise RuntimeError("V3.2 primary target must be exact_pathway")
    if task.get("target_column") != "pathway_id":
        raise RuntimeError("V3.2 primary target column must be pathway_id")
    if bool(task.get("family_may_replace_target", False)):
        raise RuntimeError("pathway_family may not replace the exact-pathway target")
    if task.get("pathway_family_role") != "auxiliary_hierarchy_only":
        raise RuntimeError("pathway_family role must remain auxiliary_hierarchy_only")

    model = config.get("primary_model", {})
    if bool(model.get("pair_evidence_in_primary", False)):
        raise RuntimeError("Pair-level evidence is forbidden in the primary score")
    if not bool(model.get("evidence_confidence_is_separate", False)):
        raise RuntimeError("Evidence confidence must remain a separate endpoint")
    if bool(model.get("subtype_feedback", False)):
        raise RuntimeError("The one-seed V3.2 pilot forbids subtype feedback")

    control = config.get("execution_control", config)
    mode = str(control.get("execution_mode", "")).upper()
    if mode == "CODE_ONLY":
        if bool(control.get("training_authorized", False)):
            raise RuntimeError("CODE_ONLY cannot authorize training")
        if bool(control.get("paid_enabled", False)):
            raise RuntimeError("CODE_ONLY cannot enable paid compute")
        if float(control.get("max_paid_hours", 0)) != 0:
            raise RuntimeError("CODE_ONLY max_paid_hours must be zero")
        if float(control.get("max_cost_cny", 0)) != 0:
            raise RuntimeError("CODE_ONLY max_cost_cny must be zero")


def validate_feature_role_manifest(feature_roles: Mapping[str, str]) -> None:
    bad = sorted(
        feature
        for feature, role in feature_roles.items()
        if feature in FORBIDDEN_PRIMARY_FEATURES and role == "primary_input"
    )
    if bad:
        raise RuntimeError(f"Forbidden primary features requested: {bad}")


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    missing = sorted(set(PREDICTION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"V3.2 prediction frame lacks columns: {missing}")
    if frame[list(TARGET_KEYS)].astype(str).duplicated().any():
        raise RuntimeError("V3.2 prediction frame has duplicate exact-pathway candidates")
    probability_columns = ["association_membership_probability", "l1_probability", "ridge_probability"]
    for column in probability_columns:
        value = pd.to_numeric(frame[column], errors="coerce")
        if value.isna().any() or ((value < 0) | (value > 1)).any():
            raise RuntimeError(f"{column} must be finite and within [0, 1]")
    scope = set(frame.shared_or_local_scope.astype(str))
    if not scope.issubset({"shared", "cancer_local"}):
        raise RuntimeError(f"Invalid shared_or_local_scope values: {sorted(scope)}")


def candidate_key_sha256(frame: pd.DataFrame) -> str:
    missing = sorted(set(TARGET_KEYS) - set(frame.columns))
    if missing:
        raise ValueError(f"Candidate frame lacks exact target keys: {missing}")
    keys = frame.loc[:, list(TARGET_KEYS)].astype(str).drop_duplicates()
    return dataframe_sha256(keys, TARGET_KEYS)


def assert_matched_candidate_universe(**frames: pd.DataFrame) -> str:
    hashes = {name: candidate_key_sha256(frame) for name, frame in frames.items()}
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"Candidate-universe mismatch: {hashes}")
    return next(iter(hashes.values()))
