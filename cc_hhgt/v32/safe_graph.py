from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import dataframe_sha256


EDGE_KEYS = ("source_type", "source_id", "relation_type", "target_type", "target_id")

# A role is not merely a descriptive label: it fixes endpoint types, relation
# name, split semantics, and the additional negative gates below.  Keeping this
# mapping explicit prevents a valid source from being relabelled just to pass a
# generic allowlist.
ROLE_CONTRACTS: dict[str, tuple[str, str, str, str]] = {
    "transductive_expression_eligibility": (
        "lncRNA", "expressed_in", "cancer", "all_samples_pre_outcome"
    ),
    "fold_train_coexpression_positive": (
        "lncRNA", "coexpressed_positive", "gene", "train"
    ),
    "fold_train_coexpression_negative": (
        "lncRNA", "coexpressed_negative", "gene", "train"
    ),
    "static_signed_pathway_membership_positive": (
        "gene", "member_of_positive", "pathway", "static"
    ),
    "static_signed_pathway_membership_negative": (
        "gene", "member_of_negative", "pathway", "static"
    ),
    "static_pathway_hierarchy": (
        "pathway", "member_of_family", "pathway_family", "static"
    ),
    "static_global_lnc_protein_binding": (
        "lncRNA", "binds_protein", "protein", "static"
    ),
    "static_protein_gene_encoding": (
        "protein", "encoded_by", "gene", "static"
    ),
    "static_symmetric_ppi": (
        "protein", "physical_interaction", "protein", "static"
    ),
    # Typed binding roles.  One role per relation: this mapping is structurally
    # one-role-one-relation, so a single role cannot carry several relation types.
    "static_global_lnc_protein_binding_eclip": (
        "lncRNA", "binds_protein_eclip", "protein", "static"
    ),
    "static_global_lnc_protein_binding_other_clip": (
        "lncRNA", "binds_protein_other_clip", "protein", "static"
    ),
    "static_global_lnc_protein_binding_rip": (
        "lncRNA", "binds_protein_rip", "protein", "static"
    ),
    "static_global_lnc_protein_binding_rna_capture": (
        "lncRNA", "binds_protein_rna_capture", "protein", "static"
    ),
    "static_global_lnc_protein_binding_other_physical": (
        "lncRNA", "binds_protein_other_physical", "protein", "static"
    ),
    "static_global_lnc_protein_binding_experimental_unspecified": (
        "lncRNA", "binds_protein_experimental_unspecified", "protein", "static"
    ),
    "static_global_lnc_protein_binding_predicted": (
        "lncRNA", "binds_protein_predicted", "protein", "static"
    ),
    "static_global_lnc_protein_binding_unknown": (
        "lncRNA", "binds_protein_unknown", "protein", "static"
    ),
    # Context-specific eCLIP.  Permitted to carry a cancer context, and required
    # to, so it can only ever enter its own context.
    "static_context_lnc_rbp_eclip": (
        "lncRNA", "binds_protein_eclip", "protein", "static"
    ),
}
ALLOWED_ROLES = frozenset(ROLE_CONTRACTS)
RELATION_POLARITY = {
    "coexpressed_positive": 1.0,
    "coexpressed_negative": -1.0,
    "member_of_positive": 1.0,
    "member_of_negative": -1.0,
    "expressed_in": 1.0,
    "member_of_family": 1.0,
    "binds_protein": 1.0,
    "binds_protein_eclip": 1.0,
    "binds_protein_other_clip": 1.0,
    "binds_protein_rip": 1.0,
    "binds_protein_rna_capture": 1.0,
    "binds_protein_other_physical": 1.0,
    "binds_protein_experimental_unspecified": 1.0,
    "binds_protein_predicted": 1.0,
    "binds_protein_unknown": 1.0,
    "encoded_by": 1.0,
    "physical_interaction": 1.0,
}
FORBIDDEN_ROLE_TOKENS = (
    "literature",
    "drug",
    "perturb",
    "interaction_evidence",
    "pair_evidence",
    "label",
    "outcome",
    "validation",
    "test",
)


@dataclass(frozen=True)
class SafeGraph:
    edges: pd.DataFrame
    manifest: dict


def _require_boolean(frame: pd.DataFrame, column: str, *, allow_null: bool = False) -> pd.Series:
    values = frame[column]
    nonnull = values.dropna()
    if not pd.api.types.is_bool_dtype(nonnull.dtype):
        raise RuntimeError(f"Safe graph {column} must be a typed boolean")
    if not allow_null and values.isna().any():
        raise RuntimeError(f"Safe graph {column} cannot be null")
    return values.astype("boolean")


#: Static edge roles allowed to carry a cancer context in the primary graph.
#: Every other static role must remain global.  Adding an entry widens the
#: primary graph's context surface and therefore requires review.
CONTEXT_BEARING_STATIC_ROLES = frozenset({"static_context_lnc_rbp_eclip"})

def _validate_ppi_reciprocity(frame: pd.DataFrame) -> None:
    ppi = frame.loc[frame.relation_type.eq("physical_interaction")]
    if ppi.empty:
        return
    if ppi.source_id.eq(ppi.target_id).any():
        raise RuntimeError("Safe graph PPI contains a self-loop")
    source = ppi.source_id.to_numpy(str)
    target = ppi.target_id.to_numpy(str)
    source_first = source <= target
    canonical = pd.DataFrame(
        {
            "left": np.where(source_first, source, target),
            "right": np.where(source_first, target, source),
            "source_id": source,
            "target_id": target,
            "weight": ppi.weight.to_numpy(float),
        }
    )
    for (left, right), group in canonical.groupby(["left", "right"], sort=False):
        directions = set(zip(group.source_id, group.target_id, strict=True))
        if len(group) != 2 or directions != {(left, right), (right, left)}:
            raise RuntimeError(
                "Safe graph PPI must contain exactly two reciprocal messages per canonical pair"
            )
        if not np.isclose(group.weight.iloc[0], group.weight.iloc[1], rtol=0.0, atol=0.0):
            raise RuntimeError("Safe graph reciprocal PPI messages have different weights")


def build_safe_graph(edges: pd.DataFrame, *, outer_fold: int) -> SafeGraph:
    required = set(EDGE_KEYS) | {
        "edge_role",
        "source_split",
        "outcome_derived",
        "observed",
        "weight",
        "relation_polarity",
        "raw_effect",
        "requires_fold_localization",
        "edge_outer_fold",
        "cancer_id",
        "is_context_specific",
    }
    if missing := sorted(required - set(edges.columns)):
        raise ValueError(f"Graph edge table lacks columns: {missing}")
    frame = edges.copy()
    for column in EDGE_KEYS + ("edge_role", "source_split"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"Safe graph {column} contains null/empty values")
        frame[column] = frame[column].astype(str).str.strip()
    if frame.duplicated(list(EDGE_KEYS)).any():
        raise RuntimeError("Safe graph input contains duplicate edge keys")

    outcome_derived = _require_boolean(frame, "outcome_derived")
    observed = _require_boolean(frame, "observed")
    # These flags are part of the security boundary, not optional metadata.
    # In particular, treating a null ``is_context_specific`` value as false
    # would let a cancer-local interaction masquerade as a static edge.
    localization = _require_boolean(frame, "requires_fold_localization")
    context_specific = _require_boolean(frame, "is_context_specific")
    weight = pd.to_numeric(frame.weight, errors="coerce")
    polarity = pd.to_numeric(frame.relation_polarity, errors="coerce")
    if not np.isfinite(weight.to_numpy(float)).all() or not weight.gt(0).all():
        raise RuntimeError("Safe graph weights must be finite positive magnitudes")
    if not np.isfinite(polarity.to_numpy(float)).all():
        raise RuntimeError("Safe graph relation polarity must be finite")
    frame["weight"] = weight.astype(float)
    frame["relation_polarity"] = polarity.astype(float)

    role = frame.edge_role.str.lower()
    forbidden_token = role.apply(lambda value: any(token in value for token in FORBIDDEN_ROLE_TOKENS))
    disallowed_role = ~role.isin(ALLOWED_ROLES)
    direct_target = (
        frame.source_type.eq("lncRNA") & frame.target_type.isin(["pathway", "exact_pathway"])
    ) | (
        frame.target_type.eq("lncRNA") & frame.source_type.isin(["pathway", "exact_pathway"])
    )
    rejected = forbidden_token | direct_target | outcome_derived.fillna(True) | ~observed.fillna(False)
    rejected |= localization.fillna(False) | disallowed_role

    for edge_role, (source_type, relation, target_type, split) in ROLE_CONTRACTS.items():
        selected = role.eq(edge_role)
        rejected |= selected & ~(
            frame.source_type.eq(source_type)
            & frame.relation_type.eq(relation)
            & frame.target_type.eq(target_type)
            & frame.source_split.eq(split)
        )
    expected_polarity = frame.relation_type.map(RELATION_POLARITY)
    rejected |= expected_polarity.isna() | ~np.isclose(
        frame.relation_polarity.to_numpy(float),
        expected_polarity.fillna(0.0).to_numpy(float),
        rtol=0.0,
        atol=0.0,
    )
    rejected |= frame.raw_effect.notna()

    coexpression = role.isin(
        {"fold_train_coexpression_positive", "fold_train_coexpression_negative"}
    )
    edge_fold = pd.to_numeric(frame.edge_outer_fold, errors="coerce")
    rejected |= coexpression & ~edge_fold.eq(int(outer_fold))
    rejected |= ~coexpression & frame.edge_outer_fold.notna()
    rejected |= coexpression & (
        frame.cancer_id.isna() | ~context_specific
    )

    expressed = role.eq("transductive_expression_eligibility")
    rejected |= expressed & (
        frame.cancer_id.isna()
        | ~frame.cancer_id.astype(str).eq(frame.target_id.astype(str))
        | ~context_specific
    )
    global_binding = role.eq("static_global_lnc_protein_binding")
    rejected |= global_binding & (
        context_specific | frame.cancer_id.notna()
    )
    # Static roles explicitly permitted to carry a cancer context.  This is the
    # ONLY widening of the primary graph's context surface, and it is deliberately
    # an allow-list: a new entry is a reviewed change, not an accident.  The
    # condition is the exact inverse of the global-binding rule -- such an edge
    # MUST carry a cancer and MUST be marked context-specific, so it can only ever
    # enter its own context and can never be globalised.
    context_static = role.isin(CONTEXT_BEARING_STATIC_ROLES)
    rejected |= context_static & (
        frame.cancer_id.isna() | ~context_specific
    )
    other_roles = ~(coexpression | expressed | global_binding | context_static)
    rejected |= other_roles & frame.cancer_id.notna()
    rejected |= other_roles & context_specific

    if rejected.any():
        examples = frame.loc[
            rejected,
            list(EDGE_KEYS)
            + [
                "edge_role",
                "source_split",
                "outcome_derived",
                "observed",
                "relation_polarity",
                "edge_outer_fold",
                "cancer_id",
                "is_context_specific",
            ],
        ].head(10)
        raise RuntimeError(
            "Unsafe edges were supplied to the primary graph; separate them before graph construction: "
            f"{examples.to_dict('records')}"
        )

    _validate_ppi_reciprocity(frame)
    frame["outer_fold"] = int(outer_fold)
    frame = frame.sort_values(list(EDGE_KEYS), kind="stable").reset_index(drop=True)
    manifest = {
        "outer_fold": int(outer_fold),
        "edge_count": int(len(frame)),
        "relation_counts": frame.groupby("relation_type", observed=True).size().sort_index().to_dict(),
        "role_counts": frame.groupby("edge_role", observed=True).size().sort_index().to_dict(),
        "allowed_roles": sorted(ALLOWED_ROLES),
        "pair_evidence_included": False,
        "validation_or_test_outcome_edges": 0,
        "transductive_expression_eligibility_declared": bool(expressed.any()),
        "ppi_reciprocal_message_count": int(frame.relation_type.eq("physical_interaction").sum()),
        "edge_sha256": dataframe_sha256(frame, EDGE_KEYS),
    }
    return SafeGraph(edges=frame, manifest=manifest)


def graph_is_invariant_to_external_evidence(
    base_edges: pd.DataFrame,
    evidence_edges: pd.DataFrame,
    *,
    outer_fold: int,
) -> bool:
    """Evidence is never concatenated; this helper makes that contract testable."""

    del evidence_edges
    first = build_safe_graph(base_edges, outer_fold=outer_fold)
    second = build_safe_graph(base_edges.copy(), outer_fold=outer_fold)
    return first.manifest["edge_sha256"] == second.manifest["edge_sha256"]
