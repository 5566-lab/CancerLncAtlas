from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import LOGGER, read_table, stable_id, write_json, write_table
from .interaction_context import apply_strict_cancer_context
from .protein_layer import build_contextual_lnc_protein, require_protein_assets


def _node_frame(ids: pd.Series, node_type: str, labels: pd.Series | None = None) -> pd.DataFrame:
    values = ids.dropna().astype(str).drop_duplicates().sort_values().reset_index(drop=True)
    frame = pd.DataFrame({"canonical_id": values, "node_type": node_type})
    frame["node_id"] = [stable_id("NODE", node_type, x) for x in values]
    if labels is not None:
        lookup = dict(zip(ids.astype(str), labels.astype(str)))
        frame["label"] = frame.canonical_id.map(lookup).fillna(frame.canonical_id)
    else:
        frame["label"] = frame.canonical_id
    frame["node_index_within_type"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _edge_frame(source_ids, target_ids, source_type: str, target_type: str, relation: str, weight=None, cancer_id=None, observed=True, source_database=None) -> pd.DataFrame:
    n = len(source_ids)
    cancer_series = pd.Series([pd.NA] * n, dtype="string") if cancer_id is None else (pd.Series([cancer_id] * n, dtype="string") if np.isscalar(cancer_id) else pd.Series(cancer_id, dtype="string").reset_index(drop=True))
    source_series = pd.Series([pd.NA] * n, dtype="string") if source_database is None else (pd.Series([source_database] * n, dtype="string") if np.isscalar(source_database) else pd.Series(source_database, dtype="string").reset_index(drop=True))
    out = pd.DataFrame({
        "source_canonical_id": pd.Series(source_ids, dtype="string").reset_index(drop=True),
        "target_canonical_id": pd.Series(target_ids, dtype="string").reset_index(drop=True),
        "source_type": source_type,
        "target_type": target_type,
        "relation_type": relation,
        "weight": np.ones(n, dtype=float) if weight is None else pd.to_numeric(pd.Series(weight).reset_index(drop=True), errors="coerce").fillna(1.0),
        "cancer_id": cancer_series,
        "is_context_specific": cancer_series.notna().to_numpy(),
        "observed": observed,
        "source_database": source_series,
    })
    out = out.dropna(subset=["source_canonical_id", "target_canonical_id"])
    out["edge_id"] = [stable_id("EDGE", st, s, r, tt, t, c) for st, s, r, tt, t, c in zip(out.source_type, out.source_canonical_id, out.relation_type, out.target_type, out.target_canonical_id, out.cancer_id)]
    return out.drop_duplicates("edge_id")


def build_graph_tables(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    std = cfg["_standardized"]
    tables = cfg["_results"] / "tables"
    dim_lnc = read_table(std / "dim_lncRNA.parquet")
    dim_gene = read_table(std / "dim_gene.parquet")
    dim_path = read_table(std / "dim_pathway.parquet")
    dim_drug = read_table(std / "dim_drug.parquet")
    dim_cancer = read_table(std / "dim_cancer.parquet")
    families = read_table(tables / "pathway_family.parquet")
    state_catalog_path = tables / "state_catalog.parquet"
    states_path = std / "tumor_state_long.parquet"
    if state_catalog_path.exists():
        catalog = read_table(state_catalog_path)
        states = catalog[["state_id"]].drop_duplicates()
    elif states_path.exists():
        states = read_table(states_path, columns=["state_id"]).drop_duplicates()
    else:
        states = pd.DataFrame({"state_id": []})
    required_states = set(map(str, cfg.get("state_graph", {}).get("required_states", [])))
    missing_required = sorted(required_states - set(states.state_id.astype(str)))
    if missing_required:
        raise RuntimeError(f"V2.9 graph build missing required state nodes: {missing_required}")

    node_frames = [
        _node_frame(dim_lnc.lncrna_id, "lncRNA", dim_lnc.gene_symbol),
        _node_frame(dim_gene.gene_id, "gene", dim_gene.gene_symbol),
        _node_frame(dim_path.pathway_id, "pathway", dim_path.pathway_name),
        _node_frame(families.pathway_family_id, "pathway_family", families.family_name),
        _node_frame(dim_drug.drug_id, "drug", dim_drug.drug_name),
        _node_frame(dim_cancer.cancer_id, "cancer", dim_cancer.english_name if "english_name" in dim_cancer else dim_cancer.cancer_id),
        _node_frame(states.state_id, "state"),
    ]
    relation = read_table(std / "interaction_relation.parquet")
    protein_assets = require_protein_assets(std)
    protein_gene = protein_assets["protein_gene"]
    lnc_protein, protein_context_audit = build_contextual_lnc_protein(
        std,
        tables,
        relation,
        dim_lnc,
        dim_cancer,
        protein_gene,
        protein_assets["lnc_protein_validated"],
    )
    string_protein = protein_assets["string_protein"]
    drug_protein = protein_assets["drug_protein"]
    protein_ids = pd.concat(
        [
            protein_gene.protein_id,
            lnc_protein.protein_id,
            string_protein.protein_id_a,
            string_protein.protein_id_b,
            drug_protein.protein_id,
        ],
        ignore_index=True,
    ).dropna().astype(str)
    if not protein_ids.str.startswith("UNIPROT:").all():
        raise ValueError(
            "All protein nodes must use canonical UNIPROT:* IDs; examples="
            f"{protein_ids.loc[~protein_ids.str.startswith('UNIPROT:')].head(10).tolist()}"
        )
    if not protein_ids.empty:
        node_frames.append(_node_frame(protein_ids, "protein"))
    nodes = pd.concat(node_frames, ignore_index=True)
    node_lookup = {(r.node_type, str(r.canonical_id)): r.node_id for r in nodes.itertuples(index=False)}

    edges = []
    member = read_table(std / "pathway_gene_member.parquet").dropna(subset=["gene_id"])
    edges.append(_edge_frame(member.gene_id, member.pathway_id, "gene", "pathway", "member_of", member.weight.fillna(1.0), source_database="MSigDB/Reactome"))
    fam_member = read_table(tables / "pathway_family_member.parquet")
    edges.append(_edge_frame(fam_member.pathway_id, fam_member.pathway_family_id, "pathway", "pathway_family", "member_of_family", fam_member.membership_weight.fillna(1.0), source_database="CancerLncAtlas"))

    coex_root = tables / "lnc_gene_coexpression"
    if coex_root.exists():
        coex = read_table(coex_root)
        edges.append(_edge_frame(coex.lncrna_id, coex.gene_id, "lncRNA", "gene", "coexpressed_with", coex.rho.abs(), coex.cancer_id, source_database="TCGA"))

    pstate_root = tables / "pathway_state_edge"
    if pstate_root.exists():
        ps = read_table(pstate_root)
        ps_relation = np.where(pd.to_numeric(ps.effect, errors="coerce").fillna(0).ge(0), "positively_associated_with_state", "negatively_associated_with_state")
        edges.append(_edge_frame(ps.pathway_id, ps.state_id, "pathway", "state", ps_relation, ps.effect.abs(), ps.cancer_id, source_database="TCGA_sample_scores"))

    pfstate_root = tables / "pathway_family_state_edge"
    if pfstate_root.exists():
        pfs = read_table(pfstate_root)
        pfs_relation = np.where(pd.to_numeric(pfs.effect, errors="coerce").fillna(0).ge(0), "positively_associated_with_state", "negatively_associated_with_state")
        edges.append(_edge_frame(pfs.pathway_family_id, pfs.state_id, "pathway_family", "state", pfs_relation, pfs.effect.abs(), pfs.cancer_id, source_database="CancerLncAtlas_state_aggregation"))

    gstate_root = tables / "gene_state_edge"
    if gstate_root.exists():
        gs = read_table(gstate_root)
        gs_relation = np.where(pd.to_numeric(gs.effect, errors="coerce").fillna(0).ge(0), "positively_associated_with_state", "negatively_associated_with_state")
        edges.append(_edge_frame(gs.gene_id, gs.state_id, "gene", "state", gs_relation, gs.effect.abs(), gs.cancer_id, source_database="TCGA_sample_scores"))

    cstate_path = tables / "cancer_state_edge.parquet"
    if cstate_path.exists():
        cs = read_table(cstate_path)
        cstate = _edge_frame(
            cs.cancer_id,
            cs.state_id,
            "cancer",
            "state",
            "state_profile_requires_fold_localization",
            np.ones(len(cs)),
            cs.cancer_id,
            source_database="TCGA_sample_scores",
        )
        raw_lookup = dict(zip(zip(cs.cancer_id.astype(str), cs.state_id.astype(str)), pd.to_numeric(cs.effect, errors="coerce")))
        cstate["raw_effect"] = [
            raw_lookup.get((str(cancer), str(state)), np.nan)
            for cancer, state in zip(cstate.source_canonical_id, cstate.target_canonical_id)
        ]
        cstate["requires_fold_localization"] = True
        edges.append(cstate)

    # Canonical V2.4 protein mechanism layer. Gene-collapsed STRING and drug
    # targets are intentionally not duplicated in the production graph.
    edges.append(
        _edge_frame(
            protein_gene.protein_id,
            protein_gene.gene_id,
            "protein",
            "gene",
            "encoded_by",
            protein_gene.mapping_weight,
            source_database=protein_gene.source_database,
        )
    )
    edges.append(
        _edge_frame(
            string_protein.protein_id_a,
            string_protein.protein_id_b,
            "protein",
            "protein",
            "physical_interaction",
            string_protein.weight,
            source_database=string_protein.source_database,
        )
    )
    edges.append(
        _edge_frame(
            drug_protein.drug_id,
            drug_protein.protein_id,
            "drug",
            "protein",
            "targets",
            drug_protein.weight,
            source_database=drug_protein.source_database,
        )
    )

    # Curated interaction relations; no direct lncRNA-pathway-family target edge is added.
    rel_gene = relation.loc[relation.partner_type.astype(str).str.lower().str.contains("gene|mrna") & relation.partner_id.isin(set(dim_gene.gene_id.astype(str)))]
    if not rel_gene.empty:
        rel_gene, curated_context_audit = apply_strict_cancer_context(rel_gene, dim_cancer)
        weight = np.where(rel_gene.is_experimental.fillna(False), 1.0, 0.4)
        edges.append(_edge_frame(rel_gene.lncrna_id, rel_gene.partner_id, "lncRNA", "gene", "curated_relation", weight, rel_gene.cancer_id, source_database=rel_gene.source_database))
    else:
        curated_context_audit = {"input_rows": 0, "context_specific_lost_to_global": 0}
    edges.append(
        _edge_frame(
            lnc_protein.lncrna_id,
            lnc_protein.protein_id,
            "lncRNA",
            "protein",
            "binds_protein",
            lnc_protein.weight,
            lnc_protein.cancer_id,
            source_database=lnc_protein.source_database,
        )
    )

    cur_path = std / "lncRNA_drug_curated_evidence.parquet"
    if cur_path.exists():
        cur = read_table(cur_path).dropna(subset=["lncrna_id", "drug_id"])
        edges.append(_edge_frame(cur.lncrna_id, cur.drug_id, "lncRNA", "drug", "curated_drug_relation", np.full(len(cur), 0.8), cur.cancer_id, source_database=cur.source_database))

    # Detection edges provide context without using the target relation.
    bulk_det_path = tables / "bulk_lnc_detection.parquet"
    if bulk_det_path.exists():
        det = read_table(bulk_det_path)
        det = det.loc[det.bulk_detection_rate >= cfg["candidate_universe"]["bulk_min_detection_rate"]]
        edges.append(_edge_frame(det.lncrna_id, det.cancer_id, "lncRNA", "cancer", "expressed_in", det.bulk_detection_rate, det.cancer_id, source_database="TCGA"))

    edge = pd.concat([x for x in edges if x is not None and not x.empty], ignore_index=True)
    # V3 LOCO contract: only tumor-expression detection is target-independent.
    # Every other cancer-context edge is removed for held-out cancers.
    edge["target_independent"] = edge.relation_type.astype(str).eq("expressed_in")
    edge["source_node_id"] = [node_lookup.get((t, str(i))) for t, i in zip(edge.source_type, edge.source_canonical_id)]
    edge["target_node_id"] = [node_lookup.get((t, str(i))) for t, i in zip(edge.target_type, edge.target_canonical_id)]
    edge = edge.dropna(subset=["source_node_id", "target_node_id"]).drop_duplicates("edge_id")

    degree = pd.concat([
        edge.groupby("source_node_id").size().rename("out_degree"),
        edge.groupby("target_node_id").size().rename("in_degree"),
    ], axis=1).fillna(0)
    nodes = nodes.merge(degree, left_on="node_id", right_index=True, how="left").fillna({"out_degree": 0, "in_degree": 0})
    nodes["log_degree"] = np.log1p(nodes.out_degree + nodes.in_degree)
    nodes["bias_feature"] = 1.0
    write_table(nodes, tables / "graph_node.parquet")
    write_table(edge, tables / "graph_edge.parquet")
    state_nodes = nodes.loc[nodes.node_type.astype(str).eq("state")]
    state_edges = edge.loc[(edge.source_type.astype(str).eq("state")) | (edge.target_type.astype(str).eq("state"))]
    required_states = set(map(str, cfg.get("state_graph", {}).get("required_states", [])))
    missing = sorted(required_states - set(state_nodes.canonical_id.astype(str)))
    relation_counts = edge.groupby(["source_type", "relation_type", "target_type"], observed=True).size()
    protein_required = {
        "protein_to_gene": int(relation_counts.get(("protein", "encoded_by", "gene"), 0)),
        "protein_to_protein": int(relation_counts.get(("protein", "physical_interaction", "protein"), 0)),
        "drug_to_protein": int(relation_counts.get(("drug", "targets", "protein"), 0)),
        "lncRNA_to_protein": int(relation_counts.get(("lncRNA", "binds_protein", "protein"), 0)),
    }
    context_lost = int(curated_context_audit.get("context_specific_lost_to_global", 0)) + int(
        protein_context_audit.get("context_specific_lost_to_global", 0)
    )
    passed = (
        not missing
        and len(state_nodes) > 0
        and len(state_edges) > 0
        and all(value > 0 for value in protein_required.values())
        and context_lost == 0
    )
    audit = {
        "status": "PASS" if passed else "FAIL",
        "n_state_nodes": int(len(state_nodes)),
        "n_state_edges": int(len(state_edges)),
        "state_relation_counts": state_edges.relation_type.astype(str).value_counts().to_dict(),
        "missing_required_states": missing,
        **protein_required,
        "curated_context_audit": curated_context_audit,
        "protein_context_audit": protein_context_audit,
        "context_specific_interaction_lost_to_global": context_lost,
    }
    write_json(audit, tables / "graph_state_asset_audit.json")
    if audit["status"] != "PASS":
        raise RuntimeError(f"V2.9 state graph audit failed: {audit}")
    return nodes, edge


def build_fold_manifests(cfg: dict[str, Any]) -> pd.DataFrame:
    cancers = read_table(cfg["_standardized"] / "dim_cancer.parquet").cancer_id.astype(str).tolist()
    reference = set(cfg["analysis_cancers"].get("reference_only", []))
    excluded = set(cfg["analysis_cancers"].get("exclude_from_training", []))
    primary = [x for x in cancers if x not in reference | excluded]
    validation_pool = set(primary)
    eligibility_path = cfg["_results"] / "tables" / "state_model_eligibility.tsv"
    required_states = set(map(str, cfg.get("sample_contract", {}).get("target_states", [])))
    if eligibility_path.exists() and required_states:
        eligibility = read_table(eligibility_path)
        eligible = eligibility.loc[
            eligibility.cancer_id.astype(str).isin(primary)
            & eligibility.state_id.astype(str).isin(required_states)
            & eligibility.evaluation_eligibility.astype(str).eq("ELIGIBLE")
        ]
        counts = eligible.groupby("cancer_id", observed=True).state_id.nunique()
        validation_pool = set(counts.loc[counts.eq(len(required_states))].index.astype(str))
        if len(validation_pool) < 2:
            raise RuntimeError(
                "Fewer than two cancers have both proxy classes for every required validation state: "
                f"{sorted(validation_pool)}"
            )
    rows = []
    for index, test in enumerate(primary):
        validation = next(
            candidate
            for offset in range(1, len(primary) + 1)
            if (candidate := primary[(index + offset) % len(primary)]) != test
            and candidate in validation_pool
        )
        train = [x for x in primary if x not in {test, validation}]
        fold_id = f"LOCO_{test}"
        rows.append({
            "fold_id": fold_id,
            "test_cancer": test,
            "validation_cancer": validation,
            "train_cancers": ";".join(train),
            "reference_cancers": ";".join(sorted(reference)),
            "n_train_cancers": len(train),
            "split_seed": cfg["random_seed"] + index,
        })
    out = pd.DataFrame(rows)
    write_table(out, cfg["_results"] / "tables" / "fold_manifest.tsv")
    edge_path = cfg["_results"] / "tables" / "graph_edge.parquet"
    if edge_path.exists():
        edge = read_table(edge_path, columns=["cancer_id", "is_context_specific", "relation_type"])
        graph_rows = []
        for row in out.itertuples(index=False):
            excluded = {str(row.test_cancer), str(row.validation_cancer)}
            excluded.update(map(str, cfg.get("analysis_cancers", {}).get("reference_only", [])))
            excluded.update(map(str, cfg.get("analysis_cancers", {}).get("exclude_from_training", [])))
            context = edge.is_context_specific.fillna(False).astype(bool)
            excluded_mask = context & edge.cancer_id.astype(str).isin(excluded)
            graph_rows.append({
                "fold_id": row.fold_id,
                "test_cancer": row.test_cancer,
                "validation_cancer": row.validation_cancer,
                "n_total_edges": len(edge),
                "n_excluded_context_edges": int(excluded_mask.sum()),
                "n_training_graph_edges": int((~excluded_mask).sum()),
                "exclusion_policy": "exclude test, validation, reference-only and exclude-from-training context edges; never include direct lncRNA-pathway-family or lncRNA-state target edges",
            })
        write_table(pd.DataFrame(graph_rows), cfg["_results"] / "tables" / "fold_graph_edge_manifest.tsv")
    return out
