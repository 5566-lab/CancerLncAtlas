from __future__ import annotations

from collections.abc import Mapping


EXACT_PATHWAY_TARGET = "exact_pathway"
PATHWAY_FAMILY_TARGET = "pathway_family"


def pathway_target_level(cfg: Mapping) -> str:
    """Return the registered primary Pathway target level.

    Historical configurations did not carry ``target_level`` and therefore
    retain their original pathway-family semantics.  V3.1 exact-pathway
    configurations must opt in explicitly; this prevents a filename or a
    display label from silently changing the statistical unit.
    """

    task = cfg.get("task_contract", {}).get("pathway", {})
    level = str(task.get("target_level", PATHWAY_FAMILY_TARGET)).strip().lower()
    aliases = {
        "family": PATHWAY_FAMILY_TARGET,
        "pathway-family": PATHWAY_FAMILY_TARGET,
        "pathway_family": PATHWAY_FAMILY_TARGET,
        "exact": EXACT_PATHWAY_TARGET,
        "pathway": EXACT_PATHWAY_TARGET,
        "exact-pathway": EXACT_PATHWAY_TARGET,
        "exact_pathway": EXACT_PATHWAY_TARGET,
    }
    try:
        return aliases[level]
    except KeyError as exc:
        raise ValueError(f"Unsupported Pathway target level: {level!r}") from exc


def pathway_target_column(cfg: Mapping) -> str:
    return "pathway_id" if pathway_target_level(cfg) == EXACT_PATHWAY_TARGET else "pathway_family_id"


def pathway_target_node_type(cfg: Mapping) -> str:
    return "pathway" if pathway_target_level(cfg) == EXACT_PATHWAY_TARGET else "pathway_family"


def pathway_target_unit(cfg: Mapping) -> list[str]:
    return ["cancer_id", "lncrna_id", pathway_target_column(cfg)]


def require_exact_pathway_contract(cfg: Mapping) -> None:
    """Fail closed when an exact-pathway release is only exact in name."""

    task = cfg.get("task_contract", {}).get("pathway", {})
    if pathway_target_level(cfg) != EXACT_PATHWAY_TARGET:
        raise RuntimeError("Website exact-pathway release requires target_level=exact_pathway")
    expected = "cancer_x_lncrna_x_exact_pathway"
    if str(task.get("target_unit", "")) != expected:
        raise RuntimeError(
            f"Exact-pathway target unit must be {expected!r}, got {task.get('target_unit')!r}"
        )
    if str(task.get("target_column", "")) != "pathway_id":
        raise RuntimeError("Exact-pathway target column must be pathway_id")
    if str(task.get("target_node_type", "")) != "pathway":
        raise RuntimeError("Exact-pathway target node type must be pathway")
    family = cfg.get("task_contract", {}).get("pathway_family", {})
    if bool(family.get("enabled", False)):
        raise RuntimeError("Pathway family cannot be a supervised head in the exact-pathway release")
    if not bool(family.get("enabled_as_hierarchy_context", False)):
        raise RuntimeError("Exact-pathway release requires pathway family as hierarchy context")
    if bool(family.get("may_replace_primary_target", True)):
        raise RuntimeError("Pathway family may never replace the exact-pathway target")
    state_training = cfg.get("state_training", {})
    if not bool(state_training.get("mask_pair_evidence_for_all_pathway_models", False)):
        raise RuntimeError(
            "Exact-pathway release must mask pair evidence for every pathway model"
        )
    if float(state_training.get("strict_pair_evidence_mask_probability", -1.0)) != 1.0:
        raise RuntimeError(
            "Exact-pathway release requires complete training-row pair-evidence masking"
        )
