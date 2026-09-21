from __future__ import annotations

import json
import heapq
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .common import read_table, write_table


def feature_attributions(cfg: dict[str, Any], predictions: pd.DataFrame, top_n: int = 100000) -> pd.DataFrame:
    features = [x for x in cfg["training"]["feature_columns"] if x in predictions.columns]
    weights_cfg = cfg["pair_evidence"]["weights"]
    feature_weights = {
        "bulk_support": weights_cfg.get("bulk", 0),
        "sc_ssgsea_support": weights_cfg.get("sc_ssgsea", 0),
        "ucell_support": weights_cfg.get("ucell", 0),
        "replication_support": weights_cfg.get("replication", 0),
        "interaction_support": weights_cfg.get("interaction", 0),
        "perturbation_support": weights_cfg.get("perturbation", 0),
        "drug_support": weights_cfg.get("drug", 0),
        "clinical_support": weights_cfg.get("clinical", 0),
        "state_support": weights_cfg.get("state", 0),
        "direction_consistency": 0.05,
        "independent_pmid_score": 0.04,
        "independent_dataset_score": 0.04,
        "bulk_detection_rate": 0.02,
        "sc_detection_rate": 0.02,
    }
    selected = predictions.nlargest(min(top_n, len(predictions)), "calibrated_probability")
    rows = []
    for row in selected.itertuples(index=False):
        contributions = []
        for feature in features:
            value = float(getattr(row, feature, 0.0) or 0.0)
            contribution = value * float(feature_weights.get(feature, 0.01))
            contributions.append((feature, value, contribution))
        total = sum(abs(x[2]) for x in contributions) or 1.0
        for rank, (feature, value, contribution) in enumerate(sorted(contributions, key=lambda x: abs(x[2]), reverse=True)[:10], 1):
            rows.append({
                "candidate_id": row.candidate_id,
                "cancer_id": row.cancer_id,
                "lncrna_id": row.lncrna_id,
                "pathway_family_id": row.pathway_family_id,
                "feature_rank": rank,
                "feature_name": feature,
                "feature_value": value,
                "attribution": contribution / total,
                "explanation_method": "evidence_weight_normalized",
            })
    return pd.DataFrame(rows)


DEFAULT_EXPLANATION_RELATIONS = [
    "coexpressed_with",
    "curated_relation",
    "curated_drug_relation",
    "targets",
    "member_of",
    "physical_interaction",
]


def read_explanation_edges(cfg: dict[str, Any]) -> pd.DataFrame:
    """Materialize a bounded, direction-aware observed explanation subgraph."""
    settings = cfg.get("explanation", {})
    allowed = settings.get("allowed_relations", DEFAULT_EXPLANATION_RELATIONS)
    max_context = int(settings.get("max_context_edges_per_source_relation", 50))
    max_other = int(settings.get("max_other_edges_per_source_relation", 100))
    edge_path = cfg["_results"] / "tables" / "graph_edge.parquet"
    relation_sql = ", ".join("'" + str(x).replace("'", "''") + "'" for x in allowed)
    con = duckdb.connect()
    query = f"""
    WITH selected AS (
      SELECT source_node_id, target_node_id, relation_type, weight, edge_id,
             row_number() OVER (
               PARTITION BY source_node_id, relation_type
               ORDER BY abs(weight) DESC, edge_id
             ) AS source_rank
      FROM read_parquet('{str(edge_path).replace("'", "''")}')
      WHERE observed
        AND relation_type IN ({relation_sql})
    )
    SELECT source_node_id, target_node_id, relation_type, weight, edge_id
    FROM selected
    WHERE relation_type IN ('member_of', 'curated_drug_relation')
       OR (
         source_rank <= CASE
           WHEN relation_type = 'coexpressed_with' THEN {max_context}
           ELSE {max_other}
         END
       )
    """
    out = con.execute(query).df()
    con.close()
    return out


def build_adjacency(edges: pd.DataFrame, reverse_relations: set[str] | None = None):
    reverse_relations = {"physical_interaction"} if reverse_relations is None else reverse_relations
    adjacency = defaultdict(list)
    for row in edges.itertuples(index=False):
        adjacency[str(row.source_node_id)].append((str(row.target_node_id), str(row.relation_type), float(row.weight), str(row.edge_id)))
        if str(row.relation_type) in reverse_relations:
            adjacency[str(row.target_node_id)].append((str(row.source_node_id), f"rev_{row.relation_type}", float(row.weight), str(row.edge_id)))
    for node in adjacency:
        adjacency[node].sort(key=lambda item: abs(item[2]), reverse=True)
    return adjacency


def top_paths(
    adjacency,
    source: str,
    targets: set[str],
    max_hops: int = 4,
    top_k: int = 5,
    max_expansions: int = 1000,
):
    counter = itertools.count()
    queue = [(-1.0, next(counter), source, [source], [])]
    found: list[tuple[float, list[str], list[tuple[str, str, float]]]] = []
    expansions = 0
    while queue and expansions < max_expansions:
        neg_score, _, node, nodes, edge_meta = heapq.heappop(queue)
        score = -neg_score
        if len(found) >= top_k and score <= min(x[0] for x in found):
            break
        if len(edge_meta) >= max_hops:
            continue
        expansions += 1
        for nxt, relation, weight, edge_id in adjacency.get(node, []):
            if nxt in nodes:
                continue
            new_nodes = nodes + [nxt]
            new_edges = edge_meta + [(edge_id, relation, weight)]
            new_score = score * max(min(abs(weight), 1.0), 1e-4)
            if nxt in targets:
                found.append((new_score, new_nodes, new_edges))
                found = sorted(found, key=lambda x: x[0], reverse=True)[:top_k]
            else:
                heapq.heappush(queue, (-new_score, next(counter), nxt, new_nodes, new_edges))
    found.sort(key=lambda x: x[0], reverse=True)
    return found[:top_k]


def graph_explanations(cfg: dict[str, Any], predictions: pd.DataFrame, top_n: int = 5000) -> pd.DataFrame:
    nodes = read_table(cfg["_results"] / "tables" / "graph_node.parquet")
    edges = read_explanation_edges(cfg)
    lookup = {(r.node_type, str(r.canonical_id)): str(r.node_id) for r in nodes.itertuples(index=False)}
    family_member = read_table(cfg["_results"] / "tables" / "pathway_family_member.parquet")
    family_targets = family_member.groupby("pathway_family_id").pathway_id.apply(lambda x: {lookup.get(("pathway", str(v))) for v in x if lookup.get(("pathway", str(v))) is not None}).to_dict()
    adjacency = build_adjacency(edges)
    rows = []
    settings = cfg.get("explanation", {})
    max_hops = int(settings.get("max_hops", 4))
    top_k = int(settings.get("top_k_paths", 5))
    max_expansions = int(settings.get("max_expansions_per_candidate", 1000))
    for row in predictions.nlargest(min(top_n, len(predictions)), "calibrated_probability").itertuples(index=False):
        source = lookup.get(("lncRNA", str(row.lncrna_id)))
        targets = family_targets.get(str(row.pathway_family_id), set())
        if not source or not targets:
            continue
        for rank, (score, node_path, edge_path) in enumerate(
            top_paths(
                adjacency,
                source,
                targets,
                max_hops=max_hops,
                top_k=top_k,
                max_expansions=max_expansions,
            ),
            1,
        ):
            rows.append({
                "candidate_id": row.candidate_id,
                "path_rank": rank,
                "path_score": score,
                "node_path_json": json.dumps(node_path),
                "edge_path_json": json.dumps([x[0] for x in edge_path]),
                "relation_path_json": json.dumps([x[1] for x in edge_path]),
            })
    return pd.DataFrame(rows)
