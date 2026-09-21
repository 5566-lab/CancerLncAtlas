"""Leakage-safe fold graph contracts for V3.1.

The same :func:`build_fold_graph` implementation is used for real LOCO
evaluation and episodic pseudo-heldout training.  ``mode`` is provenance only;
it is deliberately absent from the filtering decision so the two paths cannot
silently drift.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import pandas as pd


class DeploymentContract(str, Enum):
    STRICT = "CONTRACT-S"
    TARGET_CONTEXT = "CONTRACT-T"


class FoldGraphMode(str, Enum):
    PSEUDOHELDOUT = "pseudoheldout"
    REAL_TEST = "real_test"


class RelationProvenance(str, Enum):
    GLOBAL_STATIC = "global_static"
    CANCER_UNLABELED_CONTEXT = "cancer_unlabeled_context"
    CANCER_OUTCOME_DERIVED = "cancer_outcome_derived"


OUTCOME_RELATIONS = frozenset(
    {
        "positively_associated_with_state",
        "negatively_associated_with_state",
        "state_profile_requires_fold_localization",
        "state_enriched",
        "state_depleted",
    }
)

STRICT_ALLOWED_HELDOUT_CONTEXT = frozenset({"expressed_in"})


@dataclass(frozen=True)
class FoldGraphResult:
    edges: pd.DataFrame
    audit: dict[str, object]


def _as_text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="string")
    return frame[column].astype("string").fillna(default)


def materialize_relation_provenance(edges: pd.DataFrame) -> pd.DataFrame:
    """Attach the locked three-class provenance to every edge.

    Existing explicit values are validated and retained.  Legacy V3.0 edges
    are classified conservatively: every cancer-specific edge is context,
    and any state/outcome relation is outcome-derived.  Cancer-specific
    curated interactions remain unlabeled context because their source event
    does not depend on the pathway/state response.
    """

    output = edges.copy()
    allowed = {item.value for item in RelationProvenance}
    if "relation_provenance" in output:
        explicit = output["relation_provenance"].astype("string")
        invalid = sorted(set(explicit.dropna().astype(str)) - allowed)
        if invalid:
            raise ValueError(f"Invalid relation_provenance values: {invalid}")
    else:
        explicit = pd.Series(pd.NA, index=output.index, dtype="string")

    relation = _as_text(output, "relation_type")
    source_database = _as_text(output, "source_database").str.lower()
    source_type = _as_text(output, "source_type")
    target_type = _as_text(output, "target_type")
    if "is_context_specific" in output:
        context = output["is_context_specific"].fillna(False).astype(bool)
    else:
        context = _as_text(output, "cancer_id").ne("")
    outcome = context & (
        relation.isin(OUTCOME_RELATIONS)
        | source_type.eq("state")
        | target_type.eq("state")
        | source_database.str.contains("sample_scores|state_outcome|effect|fdr", regex=True)
    )
    inferred = pd.Series(RelationProvenance.GLOBAL_STATIC.value, index=output.index, dtype="string")
    inferred.loc[context] = RelationProvenance.CANCER_UNLABELED_CONTEXT.value
    inferred.loc[outcome] = RelationProvenance.CANCER_OUTCOME_DERIVED.value
    output["relation_provenance"] = explicit.fillna(inferred)

    # Explicit provenance may never downgrade an obvious outcome edge.
    bad = outcome & output["relation_provenance"].ne(RelationProvenance.CANCER_OUTCOME_DERIVED.value)
    if bad.any():
        examples = output.loc[bad, [c for c in ("relation_type", "source_database", "cancer_id") if c in output]].head(10)
        raise RuntimeError(f"Outcome-derived edges were assigned a safer provenance:\n{examples}")
    return output


def _edge_fingerprint(edges: pd.DataFrame) -> str:
    columns = [
        column
        for column in (
            "edge_id",
            "source_type",
            "source_canonical_id",
            "relation_type",
            "target_type",
            "target_canonical_id",
            "cancer_id",
            "relation_provenance",
        )
        if column in edges
    ]
    # Fold construction preserves the frozen canonical edge order.  Keeping
    # that order avoids an 8.5M-row sort for every pseudoheldout parity check;
    # the fingerprint is intentionally order-sensitive so any upstream reorder
    # is also visible in provenance.
    canonical = edges[columns].astype("string").fillna("")
    digest = hashlib.sha256()
    # Vectorised CSV serialization is substantially faster than 8.5M Python
    # tuple iterations while preserving the exact column/order contract.
    chunk_size = 250_000
    for start in range(0, len(canonical), chunk_size):
        text = canonical.iloc[start : start + chunk_size].to_csv(
            index=False, header=False, lineterminator="\n", sep="\x1f"
        )
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def build_fold_graph(
    edges: pd.DataFrame,
    *,
    heldout_cancers: Iterable[str],
    mode: FoldGraphMode | str,
    contract: DeploymentContract | str,
    reference_only: Iterable[str] = (),
    compute_fingerprint: bool = True,
) -> FoldGraphResult:
    """Apply the single registered fold policy used by train episodes/test.

    ``CONTRACT-S`` retains only ``expressed_in`` among held-out unlabeled
    cancer context.  ``CONTRACT-T`` retains all held-out unlabeled context.
    Both contracts drop held-out outcome-derived edges and all context from
    reference-only cancers.
    """

    mode = FoldGraphMode(mode)
    contract = DeploymentContract(contract)
    heldout = frozenset(map(str, heldout_cancers))
    reference = frozenset(map(str, reference_only))
    if not heldout:
        raise ValueError("At least one heldout cancer is required")
    if heldout & reference:
        raise ValueError(f"Heldout and reference-only cancer sets overlap: {sorted(heldout & reference)}")

    original = materialize_relation_provenance(edges)
    cancer = _as_text(original, "cancer_id")
    relation = _as_text(original, "relation_type")
    provenance = original["relation_provenance"].astype(str)
    is_context = provenance.ne(RelationProvenance.GLOBAL_STATIC.value)

    remove_reference = is_context & cancer.isin(reference)
    heldout_rows = is_context & cancer.isin(heldout)
    remove_outcome = heldout_rows & provenance.eq(RelationProvenance.CANCER_OUTCOME_DERIVED.value)
    remove_unlabeled = pd.Series(False, index=original.index)
    if contract is DeploymentContract.STRICT:
        remove_unlabeled = (
            heldout_rows
            & provenance.eq(RelationProvenance.CANCER_UNLABELED_CONTEXT.value)
            & ~relation.isin(STRICT_ALLOWED_HELDOUT_CONTEXT)
        )

    keep = ~(remove_reference | remove_outcome | remove_unlabeled)
    output = original.loc[keep].copy().reset_index(drop=True)
    surviving = output.loc[_as_text(output, "cancer_id").isin(heldout)]
    leaked = surviving.loc[
        surviving["relation_provenance"].astype(str).eq(RelationProvenance.CANCER_OUTCOME_DERIVED.value)
    ]
    if len(leaked):
        raise RuntimeError(
            "Outcome-derived held-out edges survived fold filtering: "
            f"{leaked.relation_type.astype(str).value_counts().to_dict()}"
        )
    if contract is DeploymentContract.STRICT:
        invalid = surviving.loc[
            surviving["relation_provenance"].astype(str).eq(RelationProvenance.CANCER_UNLABELED_CONTEXT.value)
            & ~surviving.relation_type.astype(str).isin(STRICT_ALLOWED_HELDOUT_CONTEXT)
        ]
        if len(invalid):
            raise RuntimeError(f"Contract-S retained forbidden target context: {invalid.relation_type.unique()}")

    audit = {
        "policy_version": "V3.1_GRAPH_CONTRACT_1",
        "mode": mode.value,
        "contract": contract.value,
        "heldout_cancers": sorted(heldout),
        "reference_only": sorted(reference),
        "input_edges": int(len(original)),
        "output_edges": int(len(output)),
        "removed_reference_context": int(remove_reference.sum()),
        "removed_heldout_outcome": int(remove_outcome.sum()),
        "removed_heldout_unlabeled_context": int(remove_unlabeled.sum()),
        "heldout_outcome_edges_surviving": int(len(leaked)),
        "edge_fingerprint_sha256": _edge_fingerprint(output) if compute_fingerprint else "NOT_COMPUTED_RUNTIME",
        "filter_implementation": "cc_hhgt.graph_contract.build_fold_graph",
    }
    return FoldGraphResult(edges=output, audit=audit)


def assert_pseudoheldout_real_test_parity(
    edges: pd.DataFrame,
    *,
    heldout_cancers: Iterable[str],
    contract: DeploymentContract | str,
    reference_only: Iterable[str] = (),
) -> dict[str, object]:
    """Fail unless pseudoheldout and real-test calls produce identical edges."""

    pseudo = build_fold_graph(
        edges,
        heldout_cancers=heldout_cancers,
        mode=FoldGraphMode.PSEUDOHELDOUT,
        contract=contract,
        reference_only=reference_only,
    )
    real = build_fold_graph(
        edges,
        heldout_cancers=heldout_cancers,
        mode=FoldGraphMode.REAL_TEST,
        contract=contract,
        reference_only=reference_only,
    )
    first = str(pseudo.audit["edge_fingerprint_sha256"])
    second = str(real.audit["edge_fingerprint_sha256"])
    if first != second or len(pseudo.edges) != len(real.edges):
        raise RuntimeError("Pseudoheldout and real-test fold graph policies diverged")
    return {
        "status": "PASS",
        "contract": DeploymentContract(contract).value,
        "edge_count": int(len(real.edges)),
        "pseudoheldout_sha256": first,
        "real_test_sha256": second,
        "same_filter_function": True,
    }
