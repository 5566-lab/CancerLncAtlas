"""Fresh V3.2 physical-interaction release materialisation.

The release is rebuilt from experimental physical facts and the current
exact-pathway membership.  Historical interaction-pathway support tables,
model probabilities and rankings are not accepted as inputs.  Global facts
remain global; they are never broadcast into cancer-specific observations.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .evidence_output_binding import (
    BINDING_FORMAT as EVIDENCE_BINDING_FORMAT,
    FORMAL_EVIDENCE_CODE_SHA256,
    FORMAL_EVIDENCE_RUNNER_SHA256,
)
from .evidence_semantic_wrapper import (
    SEMANTIC_WRAPPER_FORMAT,
    SEMANTIC_WRAPPER_STATUS,
)
from .input_lineage import artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
RELEASE_FORMAT = "CC_HHGT_V3_2_PHYSICAL_INTERACTION_RELEASE_V1"
GLOBAL_SCOPE = "GLOBAL_NOT_CANCER_SPECIFIC"
EXACT_PATHWAY_COUNT = 2_135
FORMAL_CANDIDATE_ROWS = 3_300_000
FORMAL_CANCER_COUNT = 33
FORMAL_CANDIDATE_SHA256 = (
    "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
)
FORMAL_MEMBERSHIP_SHA256 = (
    "0ae85904df979046fcbfb7f781977947392e99735b819f4d90803869831871ef"
)

_FORBIDDEN_INPUT_TOKENS = (
    "interaction_pathway_support",
    "prediction",
    "probability",
    "ranking",
    "checkpoint",
    "web_table",
    "relationship_evidence",
)


class InteractionReleaseError(RuntimeError):
    """Raised when interaction release lineage or semantics are invalid."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _stable_id(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    text = str(value).strip()
    upper = text.upper()
    for prefix in ("GENE:", "PROTEIN:", "LNC:", "LNCRNA:"):
        if upper.startswith(prefix):
            text = text[len(prefix) :]
            break
    return re.sub(r"\.\d+$", "", text).upper()


def _canonical_lnc(value: Any) -> str:
    stable = _stable_id(value)
    return "LNC:" + stable if stable else ""


def _clean_scope(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return GLOBAL_SCOPE
    text = str(value).strip().upper()
    return text if text and text not in {"NAN", "NONE", "NA", "GLOBAL", "PAN_CANCER"} else GLOBAL_SCOPE


def _assert_source_path(path: str | Path, role: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    token = re.sub(r"[^a-z0-9]+", "_", source.name.lower())
    found = [item for item in _FORBIDDEN_INPUT_TOKENS if item in token]
    if found:
        raise InteractionReleaseError(
            f"{role} is a forbidden historical/model-derived input: {source}; {found}"
        )
    if source.suffix.lower() in {".pt", ".pth", ".ckpt", ".pkl", ".pickle"}:
        raise InteractionReleaseError(f"{role} cannot be a checkpoint/model: {source}")
    return source


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_evidence_output_binding(
    binding_path: str | Path,
    *,
    facts_source: Path,
    membership_source: Path,
    candidate_source: Path,
) -> dict[str, Any]:
    """Rehash and bind the successful fresh pair-blocked Evidence run."""

    path = Path(binding_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise InteractionReleaseError(f"Evidence output binding is missing/unsafe: {path}")
    try:
        binding = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InteractionReleaseError("Evidence output binding is invalid JSON") from exc
    required = {
        "format": EVIDENCE_BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND",
        "release_ready": False,
        "production_deployed": False,
        "evidence_module_still_auxiliary_partial": True,
        "interaction_materialization_input_eligible": True,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "family_to_exact_broadcast": False,
        "five_fresh_private_heads_verified": True,
    }
    for key, expected in required.items():
        if binding.get(key) != expected:
            raise InteractionReleaseError(
                f"Evidence output binding has invalid {key}: {binding.get(key)!r}"
            )
    if int(binding.get("optimizer_steps_total", 0)) <= 0:
        raise InteractionReleaseError("Evidence output binding has no optimizer updates")
    run_id = str(binding.get("evidence_training_run_id", ""))
    if not run_id.startswith("V32-EVIDENCE-TRAIN-"):
        raise InteractionReleaseError("Evidence output binding lacks fresh training_run_id")

    authorities = binding.get("authorities")
    artifacts = binding.get("artifacts")
    checkpoints = binding.get("checkpoints")
    if not isinstance(authorities, Mapping) or not isinstance(artifacts, Mapping):
        raise InteractionReleaseError("Evidence output binding lacks authorities/artifacts")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {str(i) for i in range(5)}:
        raise InteractionReleaseError("Evidence output binding lacks five checkpoint declarations")

    def rehash(record: Any, role: str) -> Path:
        if not isinstance(record, Mapping):
            raise InteractionReleaseError(f"Evidence binding lacks {role}")
        source = Path(str(record.get("path", ""))).resolve()
        digest = str(record.get("sha256", ""))
        if artifact_sha256(source) != digest:
            raise InteractionReleaseError(f"Evidence binding {role} SHA256 drift")
        return source

    code = rehash(authorities.get("evidence_code"), "evidence code")
    runner = rehash(authorities.get("evidence_runner"), "evidence runner")
    if authorities["evidence_code"].get("sha256") != FORMAL_EVIDENCE_CODE_SHA256:
        raise InteractionReleaseError("Evidence binding code is not the formal R2 authority")
    if authorities["evidence_runner"].get("sha256") != FORMAL_EVIDENCE_RUNNER_SHA256:
        raise InteractionReleaseError("Evidence binding runner is not the formal R2 authority")
    if code.name != "evidence_training.py" or runner.name != "run_v32_evidence_training.py":
        raise InteractionReleaseError("Evidence binding code/runner role mismatch")
    bound_membership = rehash(authorities.get("exact_membership"), "exact membership")
    bound_candidates = rehash(authorities.get("exact_candidates"), "exact candidates")
    if bound_membership != membership_source.resolve():
        raise InteractionReleaseError("Evidence binding membership path differs from release input")
    if bound_candidates != candidate_source.resolve():
        raise InteractionReleaseError("Evidence binding candidate path differs from release input")
    if authorities["exact_membership"].get("sha256") != FORMAL_MEMBERSHIP_SHA256:
        raise InteractionReleaseError("Evidence binding membership is not formal V3.2 authority")
    if authorities["exact_candidates"].get("sha256") != FORMAL_CANDIDATE_SHA256:
        raise InteractionReleaseError("Evidence binding candidates are not formal V3.2 authority")
    for role in (
        "training_manifest", "training_success", "split_preflight", "split_preflight_success"
    ):
        rehash(authorities.get(role), role)
    for role, record in artifacts.items():
        rehash(record, f"artifact {role}")
    bound_facts = rehash(artifacts.get("physical_facts"), "physical facts")
    if bound_facts != facts_source.resolve():
        raise InteractionReleaseError("Evidence binding physical facts path differs from release input")
    for fold_id, record in checkpoints.items():
        checkpoint = rehash(record, f"fold {fold_id} checkpoint")
        if checkpoint.parent.name != f"patient_fold={fold_id}":
            raise InteractionReleaseError(f"Evidence fold {fold_id} checkpoint path mismatch")
        history = Path(str(record.get("training_history_path", ""))).resolve()
        if artifact_sha256(history) != record.get("training_history_sha256"):
            raise InteractionReleaseError(f"Evidence fold {fold_id} history SHA256 drift")
        if int(record.get("optimizer_steps", 0)) <= 0:
            raise InteractionReleaseError(f"Evidence fold {fold_id} has no optimizer steps")
        if record.get("initial_parameter_sha256") == record.get("final_parameter_sha256"):
            raise InteractionReleaseError(f"Evidence fold {fold_id} parameters did not change")
    success_path = path.parent / "SUCCESS.json"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if (
        success.get("status") != binding["status"]
        or success.get("release_ready") is not False
        or success.get("binding") != path.name
        or success.get("binding_sha256") != artifact_sha256(path)
    ):
        raise InteractionReleaseError("Evidence output binding success marker is stale")
    return {
        "path": str(path),
        "sha256": artifact_sha256(path),
        "success_path": str(success_path),
        "success_sha256": artifact_sha256(success_path),
        "evidence_training_run_id": run_id,
        "physical_facts_sha256": artifacts["physical_facts"]["sha256"],
        "five_fresh_private_heads_verified": True,
        "optimizer_steps_total": int(binding["optimizer_steps_total"]),
    }


def _validate_evidence_semantic_wrapper(
    wrapper_path: str | Path,
    *,
    expected_sha256: str,
    evidence_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash-bind the independently audited Evidence routing semantics."""

    requested = Path(wrapper_path)
    if requested.is_symlink():
        raise InteractionReleaseError(
            f"Evidence semantic wrapper is missing/unsafe: {requested}"
        )
    path = requested.resolve()
    if not path.is_file():
        raise InteractionReleaseError(
            f"Evidence semantic wrapper is missing/unsafe: {path}"
        )
    expected = str(expected_sha256 or "").lower()
    if not _SHA256.fullmatch(expected):
        raise InteractionReleaseError(
            "Formal interaction release requires an expected Evidence semantic wrapper SHA256"
        )
    observed = artifact_sha256(path)
    if observed != expected:
        raise InteractionReleaseError(
            f"Evidence semantic wrapper SHA256 mismatch: {observed} != {expected}"
        )
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InteractionReleaseError("Evidence semantic wrapper is invalid JSON") from exc
    required = {
        "format": SEMANTIC_WRAPPER_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": SEMANTIC_WRAPPER_STATUS,
        "release_ready": False,
        "production_deployed": False,
        "direct_target_evidence": True,
        "direct_target_scope": "LNCRNA_PARTNER_MAPPED_VIA_EXACT_MEMBER",
        "direct_exact_pathway_assertion": False,
        "unavailable_encoding": "null_with_reason",
        "confidence_only": True,
        "changes_primary_ranking": False,
        "affects_discovery": False,
        "affects_primary_ranking": False,
        "raw_prediction_direct_fusion_allowed": False,
        "canonical_candidate_adapter_required": True,
        "five_fresh_private_heads_verified": True,
        "prediction_role": "AUXILIARY_CONFIDENCE_ONLY",
    }
    for key, expected_value in required.items():
        if wrapper.get(key) != expected_value:
            raise InteractionReleaseError(
                f"Evidence semantic wrapper has invalid {key}: {wrapper.get(key)!r}"
            )
    if wrapper.get("evidence_training_run_id") != evidence_binding.get(
        "evidence_training_run_id"
    ):
        raise InteractionReleaseError(
            "Evidence semantic wrapper training_run_id differs from output binding"
        )
    if int(wrapper.get("optimizer_steps_total", 0)) != int(
        evidence_binding.get("optimizer_steps_total", -1)
    ):
        raise InteractionReleaseError(
            "Evidence semantic wrapper optimiser steps differ from output binding"
        )
    bound = wrapper.get("r2_evidence_binding")
    if not isinstance(bound, Mapping):
        raise InteractionReleaseError("Evidence semantic wrapper lacks its R2 binding")
    if (
        Path(str(bound.get("path", ""))).resolve()
        != Path(str(evidence_binding.get("path", ""))).resolve()
        or bound.get("sha256") != evidence_binding.get("sha256")
    ):
        raise InteractionReleaseError(
            "Evidence semantic wrapper is bound to a different R2 Evidence output"
        )
    post_audit = wrapper.get("independent_post_audit")
    if not isinstance(post_audit, Mapping):
        raise InteractionReleaseError(
            "Evidence semantic wrapper lacks its independent post-audit"
        )
    audit_path = Path(str(post_audit.get("path", ""))).resolve()
    if audit_path.is_symlink() or artifact_sha256(audit_path) != post_audit.get("sha256"):
        raise InteractionReleaseError(
            "Evidence semantic wrapper independent post-audit SHA256 drift"
        )
    folds = wrapper.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != {str(i) for i in range(5)}:
        raise InteractionReleaseError(
            "Evidence semantic wrapper does not prove five fresh folds"
        )
    if any(
        not isinstance(record, Mapping)
        or record.get("fresh_training_verified") is not True
        or int(record.get("optimizer_steps", 0)) <= 0
        for record in folds.values()
    ):
        raise InteractionReleaseError(
            "Evidence semantic wrapper has an invalid fresh-fold declaration"
        )
    success_path = path.parent / "SUCCESS.json"
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InteractionReleaseError(
            "Evidence semantic wrapper success marker is invalid"
        ) from exc
    if (
        success.get("status") != SEMANTIC_WRAPPER_STATUS
        or success.get("release_ready") is not False
        or success.get("wrapper") != path.name
        or success.get("wrapper_sha256") != observed
    ):
        raise InteractionReleaseError(
            "Evidence semantic wrapper success marker is stale"
        )
    return {
        "path": str(path),
        "sha256": observed,
        "success_path": str(success_path),
        "success_sha256": artifact_sha256(success_path),
        "independent_post_audit_path": str(audit_path),
        "independent_post_audit_sha256": post_audit["sha256"],
        "evidence_training_run_id": wrapper["evidence_training_run_id"],
        "confidence_only": True,
        "affects_discovery": False,
        "affects_primary_ranking": False,
        "direct_exact_pathway_assertion": False,
    }


def _relationship_id(scope: str, lncrna_id: str, partner_id: str) -> str:
    digest = hashlib.sha256(
        f"{scope}\0{lncrna_id}\0{partner_id}".encode("utf-8")
    ).hexdigest()
    return "REL32:" + digest[:24]


def _enrichment_id(scope: str, lncrna_id: str, pathway_id: str) -> str:
    digest = hashlib.sha256(
        f"{scope}\0{lncrna_id}\0{pathway_id}".encode("utf-8")
    ).hexdigest()
    return "INTPATH32:" + digest[:24]


def validate_current_exact_authority(
    membership: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    expected_pathways: int = EXACT_PATHWAY_COUNT,
    expected_candidates: int | None = None,
    expected_cancers: int | None = None,
) -> tuple[pd.DataFrame, set[str], dict[str, int]]:
    """Restrict membership to the exact pathways declared by current candidates."""

    required_members = {"pathway_id", "gene_id"}
    required_candidates = {"cancer_id", "lncrna_id", "pathway_id"}
    if missing := sorted(required_members - set(membership.columns)):
        raise InteractionReleaseError(f"Membership lacks columns: {missing}")
    if missing := sorted(required_candidates - set(candidates.columns)):
        raise InteractionReleaseError(f"Candidates lack columns: {missing}")
    candidate = candidates[list(required_candidates)].dropna().copy()
    candidate["cancer_id"] = candidate.cancer_id.astype(str).str.upper().str.strip()
    candidate["lncrna_id"] = candidate.lncrna_id.map(_canonical_lnc)
    candidate["pathway_id"] = candidate.pathway_id.astype(str).str.strip()
    if candidate.duplicated(["cancer_id", "lncrna_id", "pathway_id"]).any():
        raise InteractionReleaseError("Current exact candidates contain duplicate keys")
    if expected_candidates is not None and len(candidate) != int(expected_candidates):
        raise InteractionReleaseError(
            f"Candidate row authority mismatch: {len(candidate)} != {expected_candidates}"
        )
    pathway_ids = set(candidate.pathway_id)
    if len(pathway_ids) != int(expected_pathways):
        raise InteractionReleaseError(
            f"Exact pathway authority mismatch: {len(pathway_ids)} != {expected_pathways}"
        )
    cancer_count = int(candidate.cancer_id.nunique())
    if expected_cancers is not None and cancer_count != int(expected_cancers):
        raise InteractionReleaseError(
            f"Cancer authority mismatch: {cancer_count} != {expected_cancers}"
        )
    member = membership[["pathway_id", "gene_id"]].dropna().copy()
    member["pathway_id"] = member.pathway_id.astype(str).str.strip()
    member["gene_id"] = member.gene_id.map(_stable_id)
    member = member.loc[member.pathway_id.isin(pathway_ids) & member.gene_id.ne("")]
    member = member.drop_duplicates(["pathway_id", "gene_id"])
    if member.pathway_id.nunique() != len(pathway_ids):
        missing = sorted(pathway_ids - set(member.pathway_id))[:10]
        raise InteractionReleaseError(f"Exact pathways lack membership: {missing}")
    return (
        member.sort_values(["pathway_id", "gene_id"], kind="stable").reset_index(drop=True),
        pathway_ids,
        {
            "candidate_rows": int(len(candidate)),
            "candidate_cancers": cancer_count,
            "candidate_lncrnas": int(candidate.lncrna_id.nunique()),
            "exact_pathways": int(len(pathway_ids)),
            "exact_membership_edges": int(len(member)),
        },
    )


def build_physical_relationships(
    physical_facts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deduplicate raw facts while preserving one-row-per-fact drilldown."""

    required = {"physical_fact_id", "lncrna_id", "partner_id", "is_prediction"}
    if missing := sorted(required - set(physical_facts.columns)):
        raise InteractionReleaseError(f"Physical facts lack columns: {missing}")
    facts = physical_facts.copy()
    if facts.empty:
        raise InteractionReleaseError("Physical fact input is empty")
    predicted = facts.is_prediction.fillna(True).astype(bool)
    if predicted.any():
        raise InteractionReleaseError("Predicted interactions entered the physical-fact release")
    facts["physical_fact_id"] = facts.physical_fact_id.astype(str).str.strip()
    facts["lncrna_id"] = facts.lncrna_id.map(_canonical_lnc)
    facts["partner_id_raw"] = facts.partner_id.astype(str).str.strip()
    facts["partner_gene_id"] = facts.partner_id.map(_stable_id)
    facts["cancer_scope"] = (
        facts.cancer_id.map(_clean_scope)
        if "cancer_id" in facts
        else GLOBAL_SCOPE
    )
    invalid = (
        facts.physical_fact_id.eq("")
        | facts.lncrna_id.eq("LNC:")
        | facts.partner_gene_id.eq("")
    )
    rejected = facts.loc[invalid].copy()
    if not rejected.empty:
        rejected["rejection_reason"] = "MISSING_PHYSICAL_RELATIONSHIP_IDENTIFIER"
    facts = facts.loc[~invalid].copy()
    if facts.empty:
        raise InteractionReleaseError("No physical facts retain canonical identifiers")
    if facts.physical_fact_id.duplicated().any():
        duplicated = facts.loc[facts.physical_fact_id.duplicated(False), "physical_fact_id"].head(10)
        raise InteractionReleaseError(f"physical_fact_id is duplicated: {duplicated.tolist()}")
    keys = ["cancer_scope", "lncrna_id", "partner_gene_id"]
    facts["relationship_id"] = [
        _relationship_id(*row)
        for row in facts[keys].itertuples(index=False, name=None)
    ]
    for column in (
        "pmid", "source_database", "source_dataset", "source_record_id",
        "relation_type", "experiment_type", "source_row_sha256",
    ):
        if column not in facts:
            facts[column] = ""
        facts[column] = facts[column].fillna("").astype(str).str.strip()
    if "source_occurrence_count" not in facts:
        facts["source_occurrence_count"] = 1
    facts["source_occurrence_count"] = pd.to_numeric(
        facts.source_occurrence_count, errors="coerce"
    ).fillna(1).clip(lower=1).astype(int)
    relationships = facts.groupby(keys + ["relationship_id"], as_index=False, observed=True).agg(
        physical_fact_count=("physical_fact_id", "nunique"),
        source_occurrence_count=("source_occurrence_count", "sum"),
        independent_pmid_count=("pmid", lambda x: len({v for v in x if v})),
        independent_source_database_count=(
            "source_database", lambda x: len({v for v in x if v})
        ),
        independent_source_record_count=(
            "source_record_id", lambda x: len({v for v in x if v})
        ),
    )
    relationships["availability"] = True
    relationships["is_prediction"] = False
    relationships["analysis_version"] = ANALYSIS_VERSION
    relationships["generation"] = "V3.2_FRESH_FROM_RAW_PHYSICAL_FACTS"
    drilldown_columns = [
        "relationship_id", "physical_fact_id", "cancer_scope", "lncrna_id",
        "partner_gene_id", "partner_id_raw", "relation_type", "experiment_type",
        "source_database", "source_dataset", "source_record_id", "pmid",
        "source_row_sha256", "source_occurrence_count",
    ]
    drilldown = facts[drilldown_columns].copy()
    drilldown["evidence_type"] = "experimental_physical_interaction"
    drilldown["is_prediction"] = False
    drilldown["analysis_version"] = ANALYSIS_VERSION
    return (
        relationships.sort_values(keys, kind="stable").reset_index(drop=True),
        drilldown.sort_values(["relationship_id", "physical_fact_id"], kind="stable").reset_index(drop=True),
        rejected.reset_index(drop=True),
    )


def _bh_with_total_hypotheses(p_values: np.ndarray, total: int) -> np.ndarray:
    if len(p_values) == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    adjusted = ranked * float(total) / np.arange(1, len(ranked) + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def build_interaction_pathway_enrichment(
    relationships: pd.DataFrame,
    exact_membership: pd.DataFrame,
    *,
    total_pathways: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute exact-pathway ORA per lncRNA and evidence scope.

    Only pathways with at least one physical target are materialised, but BH
    uses ``total_pathways`` so the omitted zero-overlap hypotheses are counted.
    """

    required_rel = {"cancer_scope", "lncrna_id", "partner_gene_id", "relationship_id"}
    if missing := sorted(required_rel - set(relationships.columns)):
        raise InteractionReleaseError(f"Relationships lack columns: {missing}")
    required_member = {"pathway_id", "gene_id"}
    if missing := sorted(required_member - set(exact_membership.columns)):
        raise InteractionReleaseError(f"Membership lacks columns: {missing}")
    member = exact_membership[list(required_member)].drop_duplicates().copy()
    background = set(member.gene_id)
    if not background:
        raise InteractionReleaseError("Exact membership has an empty gene universe")
    rel = relationships.loc[relationships.partner_gene_id.isin(background)].copy()
    unmapped = relationships.loc[~relationships.partner_gene_id.isin(background)].copy()
    if not unmapped.empty:
        unmapped["mapping_status"] = "PARTNER_OUTSIDE_EXACT_MEMBERSHIP_UNIVERSE"
    if rel.empty:
        raise InteractionReleaseError(
            "No physical partner maps into the current exact-pathway gene universe"
        )
    joined = rel.merge(
        member,
        left_on="partner_gene_id",
        right_on="gene_id",
        how="inner",
        validate="many_to_many",
    )
    overlap = joined.groupby(
        ["cancer_scope", "lncrna_id", "pathway_id"], as_index=False, observed=True
    ).agg(
        overlap_target_count=("partner_gene_id", "nunique"),
        supporting_relationship_count=("relationship_id", "nunique"),
        supporting_target_ids=(
            "partner_gene_id", lambda x: ";".join(sorted(set(map(str, x))))
        ),
        supporting_relationship_ids=(
            "relationship_id", lambda x: ";".join(sorted(set(map(str, x))))
        ),
    )
    query_sizes = rel.groupby(
        ["cancer_scope", "lncrna_id"], observed=True
    ).partner_gene_id.nunique()
    pathway_sizes = member.groupby("pathway_id", observed=True).gene_id.nunique()
    universe_size = len(background)
    overlap["physical_target_count_in_universe"] = [
        int(query_sizes.loc[(scope, lnc)])
        for scope, lnc in overlap[["cancer_scope", "lncrna_id"]].itertuples(index=False, name=None)
    ]
    overlap["pathway_member_count"] = overlap.pathway_id.map(pathway_sizes).astype(int)
    overlap["membership_gene_universe_count"] = universe_size
    overlap["ora_p_value"] = [
        float(hypergeom.sf(int(k) - 1, universe_size, int(m), int(n)))
        for k, m, n in overlap[
            ["overlap_target_count", "pathway_member_count", "physical_target_count_in_universe"]
        ].itertuples(index=False, name=None)
    ]
    parts: list[pd.DataFrame] = []
    for _, part in overlap.groupby(["cancer_scope", "lncrna_id"], observed=True, sort=True):
        local = part.copy()
        local["ora_fdr"] = _bh_with_total_hypotheses(
            local.ora_p_value.to_numpy(float), int(total_pathways)
        )
        parts.append(local)
    result = pd.concat(parts, ignore_index=True)
    result["enrichment_id"] = [
        _enrichment_id(*row)
        for row in result[["cancer_scope", "lncrna_id", "pathway_id"]].itertuples(
            index=False, name=None
        )
    ]
    expected_overlap = (
        result.overlap_target_count / result.physical_target_count_in_universe
    )
    pathway_fraction = result.pathway_member_count / result.membership_gene_universe_count
    result["fold_enrichment"] = expected_overlap / pathway_fraction
    result["availability"] = True
    result["analysis_version"] = ANALYSIS_VERSION
    result["generation"] = "V3.2_FRESH_PHYSICAL_TARGET_EXACT_PATHWAY_ORA"
    result["family_to_exact_broadcast"] = False
    if not np.isfinite(result[["ora_p_value", "ora_fdr", "fold_enrichment"]]).all().all():
        raise InteractionReleaseError("Interaction-pathway enrichment is non-finite")
    return (
        result.sort_values(
            ["cancer_scope", "lncrna_id", "ora_fdr", "ora_p_value", "pathway_id"],
            kind="stable",
        ).reset_index(drop=True),
        unmapped.reset_index(drop=True),
    )


def materialize_interaction_release(
    *,
    physical_facts_path: str | Path,
    membership_path: str | Path,
    candidates_path: str | Path,
    evidence_binding_path: str | Path | None = None,
    evidence_semantic_wrapper_path: str | Path | None = None,
    expected_evidence_semantic_wrapper_sha256: str | None = None,
    output_root: str | Path,
    strict_formal_authority: bool = True,
) -> dict[str, Any]:
    """Write an immutable, fail-closed physical-interaction release."""

    facts_source = _assert_source_path(physical_facts_path, "physical facts")
    membership_source = _assert_source_path(membership_path, "exact membership")
    candidate_source = _assert_source_path(candidates_path, "exact candidates")
    evidence_binding: dict[str, Any] | None = None
    evidence_semantic_wrapper: dict[str, Any] | None = None
    if strict_formal_authority:
        if artifact_sha256(membership_source) != FORMAL_MEMBERSHIP_SHA256:
            raise InteractionReleaseError("Membership SHA is not current formal V3.2 authority")
        if artifact_sha256(candidate_source) != FORMAL_CANDIDATE_SHA256:
            raise InteractionReleaseError("Candidate SHA is not current formal V3.2 authority")
        if evidence_binding_path is None:
            raise InteractionReleaseError(
                "Formal interaction release requires a fresh Evidence output binding"
            )
        evidence_binding = _validate_evidence_output_binding(
            evidence_binding_path,
            facts_source=facts_source,
            membership_source=membership_source,
            candidate_source=candidate_source,
        )
        if evidence_semantic_wrapper_path is None:
            raise InteractionReleaseError(
                "Formal interaction release requires the Evidence semantic wrapper"
            )
        evidence_semantic_wrapper = _validate_evidence_semantic_wrapper(
            evidence_semantic_wrapper_path,
            expected_sha256=str(expected_evidence_semantic_wrapper_sha256 or ""),
            evidence_binding=evidence_binding,
        )
    output = Path(output_root).resolve()
    if output.exists():
        raise InteractionReleaseError(f"Interaction release output reuse is forbidden: {output}")
    output.mkdir(parents=True)
    try:
        facts = pd.read_parquet(facts_source)
        membership = pd.read_parquet(membership_source, columns=["pathway_id", "gene_id"])
        candidates = pd.read_parquet(
            candidate_source, columns=["cancer_id", "lncrna_id", "pathway_id"]
        )
        exact_membership, pathway_ids, authority = validate_current_exact_authority(
            membership,
            candidates,
            expected_pathways=EXACT_PATHWAY_COUNT if strict_formal_authority else int(candidates.pathway_id.nunique()),
            expected_candidates=FORMAL_CANDIDATE_ROWS if strict_formal_authority else None,
            expected_cancers=FORMAL_CANCER_COUNT if strict_formal_authority else None,
        )
        relationships, drilldown, rejected = build_physical_relationships(facts)
        enrichment, unmapped = build_interaction_pathway_enrichment(
            relationships, exact_membership, total_pathways=len(pathway_ids)
        )
        tables = {
            "physical_interaction_relationships.parquet": relationships,
            "relationship_evidence.parquet": drilldown,
            "interaction_exact_pathway_enrichment.parquet": enrichment,
            "rejected_physical_facts.parquet": rejected,
            "unmapped_physical_partners.parquet": unmapped,
        }
        for name, frame in tables.items():
            _atomic_parquet(frame, output / name)
        manifest: dict[str, Any] = {
            "format": RELEASE_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "module_id": "physical_interaction_release",
            "status": "SUCCESS_NEWLY_MATERIALIZED_V32",
            "formal_authority_enforced": bool(strict_formal_authority),
            "release_ready": False,
            "production_deployed": False,
            "source_generation": "CURRENT_V3.2_RAW_FACTS_PLUS_EXACT_STATIC_MEMBERSHIP",
            "old_interaction_pathway_support_loaded": False,
            "old_predictions_used": False,
            "old_rankings_used": False,
            "old_checkpoints_used": False,
            "family_to_exact_broadcast": False,
            "global_facts_broadcast_to_cancers": False,
            "fresh_evidence_output_binding": evidence_binding,
            "fresh_evidence_semantic_wrapper": evidence_semantic_wrapper,
            "statistical_method": "one-sided hypergeometric ORA; BH within scope/lncRNA over all exact pathways",
            "inputs": {
                "physical_facts": {"path": str(facts_source), "sha256": artifact_sha256(facts_source)},
                "exact_membership": {"path": str(membership_source), "sha256": artifact_sha256(membership_source)},
                "exact_candidates": {"path": str(candidate_source), "sha256": artifact_sha256(candidate_source)},
                "evidence_output_binding": (
                    {"path": evidence_binding["path"], "sha256": evidence_binding["sha256"]}
                    if evidence_binding is not None
                    else None
                ),
                "evidence_semantic_wrapper": (
                    {
                        "path": evidence_semantic_wrapper["path"],
                        "sha256": evidence_semantic_wrapper["sha256"],
                    }
                    if evidence_semantic_wrapper is not None
                    else None
                ),
            },
            "authority": authority,
            "counts": {
                "physical_fact_rows": int(len(facts)),
                "relationship_rows": int(len(relationships)),
                "relationship_evidence_rows": int(len(drilldown)),
                "interaction_exact_pathway_rows": int(len(enrichment)),
                "rejected_physical_fact_rows": int(len(rejected)),
                "unmapped_relationship_rows": int(len(unmapped)),
            },
            "artifacts": {},
        }
        for name, frame in tables.items():
            manifest["artifacts"][name] = {
                "path": name,
                "sha256": artifact_sha256(output / name),
                "rows": int(len(frame)),
            }
        _atomic_json(manifest, output / "INTERACTION_RELEASE_MANIFEST.json")
        success = {
            "status": manifest["status"],
            "release_ready": False,
            "manifest": "INTERACTION_RELEASE_MANIFEST.json",
            "manifest_sha256": artifact_sha256(output / "INTERACTION_RELEASE_MANIFEST.json"),
        }
        _atomic_json(success, output / "SUCCESS.json")
        return manifest
    except Exception:
        # This output is newly created and has not been registered.  Removing
        # a partial build is safe and prevents it from being mistaken for a release.
        import shutil

        shutil.rmtree(output, ignore_errors=True)
        raise


__all__ = [
    "ANALYSIS_VERSION",
    "EXACT_PATHWAY_COUNT",
    "FORMAL_CANDIDATE_SHA256",
    "FORMAL_MEMBERSHIP_SHA256",
    "GLOBAL_SCOPE",
    "InteractionReleaseError",
    "RELEASE_FORMAT",
    "build_interaction_pathway_enrichment",
    "build_physical_relationships",
    "materialize_interaction_release",
    "validate_current_exact_authority",
]
