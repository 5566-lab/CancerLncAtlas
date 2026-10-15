"""Add experimental and predicted relations to the current typed C graph.

All three comparison arms share nodes, node features, relation schema and the
coexpression schedule. Existing C edges are retained exactly. New nodes append
to each type's index space, so the original candidate tensors remain valid.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from .contracts import dataframe_sha256
from .formal_graph import (TYPED_BINDING_ROLES, build_formal_graph_authority,
                           materialize_typed_lnc_protein_binding)
from .rbp_assay import classify_assay
from .safe_graph import EDGE_KEYS, build_safe_graph
from ..relation_sampling import RuntimeSamplingPolicy, runtime_chunk_positions, schedule_runtime_edges

ARMS = ("current_c", "experimental", "experimental_predicted")
LINEAGE_HASH_COLUMNS = ("lineage_id", "event_id", "raw_event_id", "partner_type", "source_row_sha256", "source_sha256")
PHYSICAL_CLASSES = {"eclip", "other_clip", "rip", "rna_capture", "other_physical"}
NODE_COLUMNS = ["node_id", "node_type", "canonical_id", "node_index_within_type"]


def canonical_json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _require_boolean(frame, name):
    if frame[name].isna().any() or not pd.api.types.is_bool_dtype(frame[name].dtype):
        raise ValueError(f"Binding source {name} must be a complete boolean")


def adapt_october_edges(edges, lineage):
    """Read native October edge/lineage products and recover assay classes.

Recomputes their ordered event hashes instead of accepting a checksum-shaped
column as evidence. Predicted and experimental flags must agree with the native
event records. An RBP/TF partner remains a protein endpoint.
"""
    edge_columns = {"edge_evidence_id", "lncrna_id", "protein_id", "evidence_class", "mapping_multiplicity",
                    "source_lineage_sha256", "source_event_count", "cancer_id", "is_context_specific", "weight"}
    lineage_columns = {"edge_evidence_id", "lncrna_id", "protein_id", "evidence_class", "experiment_type", "source_database",
                       "is_experimental", "is_computational", "is_physical", "family_broadcast_used", *LINEAGE_HASH_COLUMNS}
    if not edge_columns.issubset(edges) or not lineage_columns.issubset(lineage):
        raise ValueError("Native October binding source lacks full event lineage")
    if edges.empty or lineage.empty:
        raise ValueError("Native binding expansion is empty")
    if edges.edge_evidence_id.duplicated().any() or edges[["edge_evidence_id", "lncrna_id", "protein_id"]].isna().any().any():
        raise ValueError("Native expansion edges must have unique complete source identities")
    if set(edges.edge_evidence_id) != set(lineage.edge_evidence_id):
        raise ValueError("Native edge and event-lineage universes disagree")
    _require_boolean(edges, "is_context_specific")
    if edges.is_context_specific.any() or edges.cancer_id.notna().any(): raise ValueError("October expansion input is not global molecular binding")
    if not pd.to_numeric(edges.mapping_multiplicity, errors="coerce").eq(1).all(): raise ValueError("October expansion requires unique protein mapping")
    for col in ["is_experimental", "is_computational", "is_physical", "family_broadcast_used"]:
        _require_boolean(lineage, col)
    if not lineage.is_physical.all() or lineage.family_broadcast_used.any():
        raise ValueError("Functional or family-broadcast evidence cannot become a binding edge")
    if not lineage.partner_type.isin(["protein", "tf", "rbp"]).all(): raise ValueError("Unsupported binding endpoint type")
    for col in LINEAGE_HASH_COLUMNS:
        if lineage[col].isna().any() or lineage[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing native lineage component: {col}")
    for col in ["source_row_sha256", "source_sha256"]:
        if not lineage[col].astype(str).str.fullmatch("[0-9a-f]{64}").all(): raise ValueError("Malformed native source hash")
    ordered = lineage.sort_values(["edge_evidence_id", *LINEAGE_HASH_COLUMNS], kind="stable")
    summaries = []
    for evidence_id, group in ordered.groupby("edge_evidence_id", sort=False):
        digest = hashlib.sha256()
        for index, row in enumerate(group[list(LINEAGE_HASH_COLUMNS)].itertuples(index=False, name=None)):
            if index: digest.update(b"\x1e")
            digest.update("\x1f".join(map(str, row)).encode())
        summaries.append({"edge_evidence_id": evidence_id, "observed_lineage_sha256": digest.hexdigest(),
                          "observed_event_count": group.event_id.nunique()})
    checks = edges.merge(pd.DataFrame(summaries), on="edge_evidence_id", validate="one_to_one")
    if (not checks.source_lineage_sha256.eq(checks.observed_lineage_sha256).all()
            or not checks.source_event_count.eq(checks.observed_event_count).all()):
        raise ValueError("Native expansion event hash/count mismatch")
    expected = "EVIDEDGE:"+edges.apply(lambda r: hashlib.sha256("\x1f".join([r.evidence_class, r.lncrna_id, r.protein_id]).encode()).hexdigest(), axis=1)
    if not edges.edge_evidence_id.eq(expected).all(): raise ValueError("Native evidence identifier does not bind its endpoints")
    linked = lineage.merge(edges[["edge_evidence_id", "lncrna_id", "protein_id", "evidence_class", "weight", "source_lineage_sha256"]],
                           on=["edge_evidence_id", "lncrna_id", "protein_id", "evidence_class"], how="left", validate="many_to_one")
    if linked.weight.isna().any(): raise ValueError("Native lineage relabelled an edge endpoint/class")
    predicted = linked.evidence_class.eq("predicted_physical")
    experimental = linked.evidence_class.eq("experimental_physical")
    if (not (predicted | experimental).all() or not linked.is_computational.eq(predicted).all()
            or not linked.is_experimental.eq(experimental).all()): raise ValueError("Native predicted/experimental evidence flags disagree")
    assignments = {raw: classify_assay(raw) for raw in linked.experiment_type.fillna("").unique()}
    classes = []
    for raw, pred in zip(linked.experiment_type.fillna(""), predicted):
        assignment = assignments[raw]
        if pred: classes.append("predicted"); continue
        if assignment.experiment_family in {"computational_prediction", "functional_perturbation", "reporter_assay", "expression_or_abundance"}:
            raise ValueError("Native experimental_physical label contradicts its assay method")
        classes.append(assignment.graph_assay_class if assignment.graph_assay_class in PHYSICAL_CLASSES else "other_physical")
    linked["graph_assay_class"] = classes
    output = linked.groupby(["lncrna_id", "protein_id", "graph_assay_class"], as_index=False).agg(
        weight=("weight", "max"), source_database=("source_database", lambda x: "|".join(sorted(set(x.astype(str))))),
        source_lineage_sha256=("source_lineage_sha256", lambda x: "|".join(sorted(set(x)))),
        edge_evidence_id=("edge_evidence_id", lambda x: "|".join(sorted(set(x)))))
    output["cancer_id"] = pd.NA; output["is_context_specific"] = False
    return output


def normalize_binding_source(frame, *, source_id, predicted_weight_cap=.25):
    required = {"lncrna_id", "protein_id", "graph_assay_class", "weight", "source_database", "cancer_id", "is_context_specific"}
    if not required.issubset(frame): raise ValueError("Expansion requires typed global binding inputs")
    values = frame.copy().reset_index(drop=True)
    _require_boolean(values, "is_context_specific")
    if values.is_context_specific.any() or values.cancer_id.notna().any(): raise ValueError("Expansion cannot silently globalize context-specific evidence")
    if not values.graph_assay_class.isin(PHYSICAL_CLASSES | {"predicted"}).all(): raise ValueError("Expansion contains nonphysical or untyped evidence")
    if values[list(required-{"cancer_id"})].isna().any().any(): raise ValueError("Incomplete expansion binding")
    if (not values.lncrna_id.astype(str).str.startswith("LNC:").all()
            or not values.protein_id.astype(str).str.startswith(("UNIPROT:", "PROTEIN:")).all()):
        raise ValueError("Expansion endpoints must use registered lncRNA/protein namespaces")
    weights = pd.to_numeric(values.weight, errors="coerce")
    if not np.isfinite(weights).all() or weights.le(0).any() or not 0 < predicted_weight_cap <= .25:
        raise ValueError("Invalid expansion weight or predicted weight cap")
    if "source_weight" not in values:
        values["source_weight"] = weights
    else:
        original_weights = pd.to_numeric(values.source_weight, errors="coerce")
        if not np.isfinite(original_weights).all() or original_weights.le(0).any() or original_weights.lt(weights).any():
            raise ValueError("Original source weights contradict the admitted weights")
    values["weight"] = weights.where(values.graph_assay_class.ne("predicted"), weights.clip(upper=predicted_weight_cap))
    values["expansion_source_id"] = str(source_id)
    return values


def _pairs(frame, left="lncrna_id", right="protein_id"):
    return pd.MultiIndex.from_frame(frame[[left, right]].astype(str))


@dataclass
class CurrentBindingArms:
    nodes: pd.DataFrame
    master_edges: pd.DataFrame
    relation_schema: pd.DataFrame
    arm_edge_ids: dict
    source_rows: pd.DataFrame
    audit: dict
    base_node_degrees: dict
    master_schedule: pd.DataFrame
    chunk_positions: dict
    predicted_dropout: float


def build_current_binding_arms(base_bundle, sources, *, fold, predicted_dropout=.5, edge_chunk_size=250000):
    from ..gnn import fold_node_degree_map
    if not 0 < predicted_dropout < 1: raise ValueError("Predicted dropout must be between zero and one")
    base = base_bundle.edges.copy()
    build_safe_graph(base, outer_fold=fold)
    binding = base.loc[base.edge_role.isin(TYPED_BINDING_ROLES)]
    if (binding.empty or binding.relation_type.isin(["binds_protein", "binds_protein_predicted"]).any()
            or binding.cancer_id.notna().any() or binding.is_context_specific.any()):
        raise ValueError("The baseline must be the current typed global C graph without predicted/legacy flat binding")
    if not sources: raise ValueError("At least one source-bound expansion table is required")
    rows = pd.concat(sources, ignore_index=True)
    if "expansion_source_id" not in rows or rows.expansion_source_id.isna().any():
        raise ValueError("Expansion tables must be normalized with named source identities")
    rows = pd.concat([normalize_binding_source(group, source_id=name)
                      for name, group in rows.groupby("expansion_source_id", sort=True)], ignore_index=True)
    current_pairs = _pairs(binding, "source_id", "target_id")
    rows["admitted"] = ~_pairs(rows).isin(current_pairs)
    rows["exclusion_reason"] = np.where(rows.admitted, None, "PAIR_ALREADY_IN_CURRENT_C")
    new_measured_pairs = _pairs(rows.loc[rows.admitted & rows.graph_assay_class.ne("predicted")])
    overlaps = rows.graph_assay_class.eq("predicted") & _pairs(rows).isin(new_measured_pairs)
    rows.loc[overlaps, "admitted"] = False
    rows.loc[overlaps, "exclusion_reason"] = "EXPERIMENTAL_EXPANSION_TAKES_PRECEDENCE"
    accepted = rows.loc[rows.admitted].copy()
    if accepted.empty or not accepted.graph_assay_class.eq("predicted").any() or not accepted.graph_assay_class.ne("predicted").any():
        raise ValueError("Three-arm comparison requires both new experimental and new predicted pairs")
    new_edges = materialize_typed_lnc_protein_binding(accepted, include_predicted=True, require_global_binding=True)
    authority = build_formal_graph_authority([base, new_edges], outer_fold=fold,
                                            required_nodes=base_bundle.nodes[["node_type", "canonical_id"]])
    # Append new nodes after all existing per-type indices. Reusing sorted
    # full-universe positions would silently change the candidate tensors.
    old_nodes = base_bundle.nodes[NODE_COLUMNS].copy()
    for kind, group in old_nodes.groupby("node_type"):
        if sorted(group.node_index_within_type) != list(range(len(group))):
            raise ValueError(f"Current-C {kind} node indices are not a contiguous unique range")
    old_keys = pd.MultiIndex.from_frame(old_nodes[["node_type", "canonical_id"]])
    new = authority.nodes.loc[~pd.MultiIndex.from_frame(authority.nodes[["node_type", "canonical_id"]]).isin(old_keys)].copy()
    if set(new.node_type)-set(old_nodes.node_type): raise ValueError("Expansion introduced a new node type/model architecture")
    for kind in sorted(new.node_type.unique()):
        pos = new.index[new.node_type.eq(kind)]
        new.loc[pos, "node_index_within_type"] = np.arange(len(pos))+int(old_nodes.node_type.eq(kind).sum())
    nodes = pd.concat([old_nodes, new[NODE_COLUMNS]], ignore_index=True)
    # All C edges survive byte-for-byte, including their source metadata.
    added = authority.edges.loc[~authority.edges.edge_id.isin(base.edge_id)].copy()
    master = pd.concat([base, added], ignore_index=True).sort_values(list(EDGE_KEYS), kind="stable").reset_index(drop=True)
    new_pred = set(added.loc[added.relation_type.eq("binds_protein_predicted"), "edge_id"])
    new_exp = set(added.edge_id)-new_pred; current = set(base.edge_id)
    arms = {"current_c": current, "experimental": current | new_exp, "experimental_predicted": current | new_exp | new_pred}
    # Common G0 features contain no old or new binding, encoding, or PPI edges.
    feature_edges = base.loc[~base.edge_role.isin(TYPED_BINDING_ROLES | {"static_protein_gene_encoding", "static_symmetric_ppi"})]
    degree = fold_node_degree_map(nodes, feature_edges)
    if base_bundle.node_degrees is None:
        raise ValueError("Current C lacks its registered G0 node features")
    for kind, old in base_bundle.node_degrees.items():
        if not np.array_equal(np.asarray(old), degree[kind][:len(old)]):
            raise ValueError("Current-C node features disagree with the common G0 feature contract")
    policy = RuntimeSamplingPolicy(edge_chunk_size=edge_chunk_size, resident_backbone=True,
                                   variable_relation_types=("coexpressed_positive", "coexpressed_negative"))
    schedule = schedule_runtime_edges(master, policy)
    positions = runtime_chunk_positions(schedule)
    if len(positions) != int(base_bundle.runtime_num_chunks):
        raise ValueError("Expansion changed runtime chunk count; use the current C edge chunk size")
    # The canonical rotating source rows must preserve their registered chunks.
    if base_bundle.runtime_schedule is not None:
        old_schedule = base_bundle.runtime_schedule
        variable = old_schedule.relation_type.isin(policy.variable_relation_types)
        if variable.any():
            columns = [c for c in old_schedule if c in schedule and c.startswith("runtime_")]
            compare = old_schedule.loc[variable, ["edge_id", *columns]].merge(schedule[["edge_id", *columns]], on="edge_id", suffixes=("_old", "_new"), validate="one_to_one")
            for column in columns:
                if not compare[column+"_old"].equals(compare[column+"_new"]): raise ValueError("Coexpression runtime schedule drift")
    audit = {"fold": fold, "node_sha256": dataframe_sha256(nodes, ["node_type", "canonical_id"]),
        "base_node_sha256": dataframe_sha256(old_nodes, ["node_type", "canonical_id"]),
        "base_edge_sha256": dataframe_sha256(base, list(EDGE_KEYS)), "master_edge_sha256": dataframe_sha256(master, list(EDGE_KEYS)),
        "appended_nodes": len(new), "current_candidate_indices_preserved": True,
        "arm_edge_counts": {arm: len(ids) for arm, ids in arms.items()},
        "source_rows": len(rows), "excluded_rows": int((~rows.admitted).sum()),
        "runtime_chunks": len(positions), "node_features_from_common_g0": True,
        "training_started": False, "predicted_dropout": predicted_dropout}
    return CurrentBindingArms(nodes, master, authority.relation_schema, arms, rows, audit, degree, schedule, positions, predicted_dropout)


def binding_arm_runtime_bundle(arms, arm):
    from ..gnn import _materialize_graph_bundle
    if arm not in ARMS: raise ValueError("Unknown current-C binding comparison arm")
    selected = arms.master_edges.loc[arms.master_edges.edge_id.isin(arms.arm_edge_ids[arm])].copy()
    schedule = arms.master_schedule.loc[arms.master_schedule.edge_id.isin(arms.arm_edge_ids[arm])].reset_index(drop=True)
    positions = runtime_chunk_positions(schedule)
    if set(positions) != set(arms.chunk_positions): raise ValueError("A comparison arm lost a runtime chunk")
    bundle = _materialize_graph_bundle(arms.nodes, schedule.iloc[positions[0]], canonical_edges=selected,
        feature_edges=selected.iloc[:0], node_degree_override=arms.base_node_degrees,
        runtime_schedule=schedule, relation_schema_edges=arms.relation_schema,
        runtime_chunk_positions=positions, active_runtime_chunk=0, graph_contract="CONTRACT-S")
    bundle.binding_evidence_arm = arm
    bundle.relation_edge_dropout = {"binds_protein_predicted": arms.predicted_dropout} if arm == "experimental_predicted" else {}
    return bundle
