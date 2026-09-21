"""Formal expert declarations for V3.2 exact-pathway residual fusion."""
from __future__ import annotations

from .multimodal_fusion import ExpertSpec, MultimodalFusionError


EXACT_PATHWAY_FUSION_EXPERTS: tuple[ExpertSpec, ...] = (
    ExpertSpec(
        expert_id="genomic",
        probability_column="mutation_cnv_context_probability",
        availability_column="genomic_available",
        endpoint_roles=("discovery", "confidence"),
        split_unit="patient",
        source_role="current_v32_oof_prediction",
        independent_of_primary_label=True,
        direct_target_evidence=False,
    ),
    ExpertSpec(
        expert_id="single_cell",
        probability_column="single_cell_replication_probability",
        availability_column="single_cell_available",
        endpoint_roles=("discovery", "confidence"),
        split_unit="donor_or_dataset",
        source_role="current_v32_oof_prediction",
        independent_of_primary_label=True,
        direct_target_evidence=False,
    ),
    ExpertSpec(
        expert_id="evidence_transformer",
        probability_column="evidence_confidence_probability",
        availability_column="availability",
        endpoint_roles=("confidence",),
        split_unit="publication_dataset_and_pair",
        source_role="current_v32_oof_prediction",
        independent_of_primary_label=True,
        direct_target_evidence=True,
    ),
)


def validate_exact_pathway_fusion_expert_policy() -> None:
    ids: set[str] = set()
    for spec in EXACT_PATHWAY_FUSION_EXPERTS:
        spec.validate()
        if spec.expert_id in ids:
            raise MultimodalFusionError(f"Duplicate formal expert: {spec.expert_id}")
        ids.add(spec.expert_id)
    if ids != {"genomic", "single_cell", "evidence_transformer"}:
        raise MultimodalFusionError("Formal exact-pathway expert set drifted")
    evidence = next(
        spec for spec in EXACT_PATHWAY_FUSION_EXPERTS
        if spec.expert_id == "evidence_transformer"
    )
    if evidence.endpoint_roles != ("confidence",) or not evidence.direct_target_evidence:
        raise MultimodalFusionError("Evidence Transformer must be direct confidence-only evidence")
    if any(spec.expert_id == "drug" for spec in EXACT_PATHWAY_FUSION_EXPERTS):
        raise MultimodalFusionError("Native cancer-lncRNA-drug scores cannot enter exact fusion")


def experts_for_endpoint(endpoint: str) -> tuple[ExpertSpec, ...]:
    if endpoint not in {"discovery", "confidence"}:
        raise MultimodalFusionError(f"Unsupported exact-pathway endpoint: {endpoint}")
    validate_exact_pathway_fusion_expert_policy()
    return tuple(
        spec for spec in EXACT_PATHWAY_FUSION_EXPERTS if endpoint in spec.endpoint_roles
    )


__all__ = [
    "EXACT_PATHWAY_FUSION_EXPERTS",
    "experts_for_endpoint",
    "validate_exact_pathway_fusion_expert_policy",
]
