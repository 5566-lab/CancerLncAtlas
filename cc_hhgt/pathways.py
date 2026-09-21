from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .common import LOGGER, input_path, read_table, stable_id, write_table
from .io import read_cancer_partition
from .stats import minmax01, robust_z


def _top_pairs_from_dense(matrix: np.ndarray, ids: np.ndarray, top_k: int, min_similarity: float, view: str, absolute: bool = False) -> pd.DataFrame:
    rows = []
    for i in range(matrix.shape[0]):
        values = np.abs(matrix[i]) if absolute else matrix[i]
        order = np.argpartition(values, -min(top_k + 1, len(values)))[-min(top_k + 1, len(values)):]
        for j in order:
            if i >= j:
                continue
            sim = float(values[j])
            if not np.isfinite(sim) or sim < min_similarity:
                continue
            rows.append({"pathway_a": ids[i], "pathway_b": ids[j], "view": view, "similarity": sim, "signed_similarity": float(matrix[i, j])})
    return pd.DataFrame(rows)


def gene_jaccard(pathway_member: pd.DataFrame, top_k: int, min_similarity: float) -> pd.DataFrame:
    mapped = pathway_member.dropna(subset=["gene_id"]).drop_duplicates(["pathway_id", "gene_id"])
    pcat = pd.Categorical(mapped.pathway_id)
    gcat = pd.Categorical(mapped.gene_id)
    x = sparse.csr_matrix((np.ones(len(mapped), dtype=np.float32), (pcat.codes, gcat.codes)), shape=(len(pcat.categories), len(gcat.categories)))
    intersections = x @ x.T
    sizes = np.asarray(x.sum(axis=1)).ravel()
    rows = []
    for i in range(intersections.shape[0]):
        start, stop = intersections.indptr[i], intersections.indptr[i + 1]
        js = intersections.indices[start:stop]
        vals = intersections.data[start:stop]
        candidates = []
        for j, inter in zip(js, vals):
            if i >= j:
                continue
            union = sizes[i] + sizes[j] - inter
            sim = float(inter / union) if union else 0.0
            if sim >= min_similarity:
                candidates.append((sim, j))
        candidates.sort(reverse=True)
        for sim, j in candidates[:top_k]:
            rows.append({"pathway_a": str(pcat.categories[i]), "pathway_b": str(pcat.categories[j]), "view": "gene_jaccard", "similarity": sim, "signed_similarity": sim})
    return pd.DataFrame(rows)


def bulk_activity_similarity(cfg: dict[str, Any], top_k: int, min_similarity: float) -> pd.DataFrame:
    path = input_path(cfg, "bulk_pathway_activity")
    cancers = read_table(cfg["_standardized"] / "dim_cancer.parquet").cancer_id.astype(str)
    accum: dict[tuple[str, str], list[float]] = defaultdict(list)
    for cancer in cancers:
        frame = read_cancer_partition(path, cancer)
        if frame.empty:
            continue
        value_col = "activity_score" if "activity_score" in frame.columns else "value"
        mat = frame.pivot_table(index="sample_id", columns="pathway_id", values=value_col, aggfunc="mean")
        if len(mat) < 20 or mat.shape[1] < 2:
            continue
        mat = mat.loc[:, mat.var(ddof=1) > 1e-8]
        corr = mat.corr(method="spearman").to_numpy(dtype=np.float32)
        ids = mat.columns.to_numpy()
        pairs = _top_pairs_from_dense(corr, ids, top_k, min_similarity, "bulk_activity", absolute=True)
        for r in pairs.itertuples(index=False):
            key = tuple(sorted([str(r.pathway_a), str(r.pathway_b)]))
            accum[key].append(float(abs(r.signed_similarity)))
    rows = [{"pathway_a": a, "pathway_b": b, "view": "bulk_activity", "similarity": float(np.median(v)), "signed_similarity": float(np.median(v)), "n_cancers": len(v)} for (a, b), v in accum.items()]
    return pd.DataFrame(rows)


def profile_similarity(table: pd.DataFrame, pathway_col: str, feature_col: str, value_col: str, view: str, top_k: int, min_similarity: float) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    matrix = table.pivot_table(index=pathway_col, columns=feature_col, values=value_col, aggfunc="mean", fill_value=0.0)
    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        return pd.DataFrame()
    sim = cosine_similarity(matrix.to_numpy(float))
    return _top_pairs_from_dense(sim, matrix.index.to_numpy(), top_k, min_similarity, view)


def drug_overlap(cfg: dict[str, Any], member: pd.DataFrame, top_k: int, min_similarity: float) -> pd.DataFrame:
    target_path = cfg["_standardized"] / "drug_gene_target.parquet"
    if not target_path.exists():
        return pd.DataFrame()
    targets = read_table(target_path)
    targets = targets.dropna(subset=["drug_id", "gene_id"])
    joined = member.dropna(subset=["gene_id"])[["pathway_id", "gene_id"]].drop_duplicates().merge(targets[["drug_id", "gene_id"]].drop_duplicates(), on="gene_id")
    if joined.empty:
        return pd.DataFrame()
    fake = joined[["pathway_id", "drug_id"]].rename(columns={"drug_id": "gene_id"})
    out = gene_jaccard(fake, top_k, min_similarity)
    if not out.empty:
        out["view"] = "drug_overlap"
    return out


def ontology_similarity(cfg: dict[str, Any]) -> pd.DataFrame:
    path = input_path(cfg, "reactome_hierarchy")
    if not path or not path.exists():
        return pd.DataFrame()
    raw = read_table(path)
    parent = next((c for c in ["parent_pathway_id", "parent_id", "parent"] if c in raw.columns), None)
    child = next((c for c in ["child_pathway_id", "child_id", "child"] if c in raw.columns), None)
    if not parent or not child:
        return pd.DataFrame()
    out = raw[[parent, child]].dropna().rename(columns={parent: "pathway_a", child: "pathway_b"}).drop_duplicates()
    out["view"] = "ontology"
    out["similarity"] = 1.0
    out["signed_similarity"] = 1.0
    return out


def build_pathway_similarity(cfg: dict[str, Any]) -> pd.DataFrame:
    settings = cfg["pathway_similarity"]
    member = read_table(cfg["_standardized"] / "pathway_gene_member.parquet")
    views = []
    LOGGER.info("Pathway similarity: gene Jaccard")
    views.append(gene_jaccard(member, settings["top_k_per_view"], settings["min_view_similarity"]))
    LOGGER.info("Pathway similarity: ontology")
    views.append(ontology_similarity(cfg))
    LOGGER.info("Pathway similarity: bulk activity")
    views.append(bulk_activity_similarity(cfg, settings["top_k_per_view"], settings["min_view_similarity"]))

    pstate_root = cfg["_results"] / "tables" / "pathway_state_edge"
    if pstate_root.exists():
        pstate = read_table(pstate_root)
        pstate["feature"] = pstate.cancer_id.astype(str) + "::" + pstate.state_id.astype(str)
        views.append(profile_similarity(pstate, "pathway_id", "feature", "effect", "state_profile", settings["top_k_per_view"], settings["min_view_similarity"]))

    sc_path = cfg["_standardized"] / "sc_pseudotime_pathway.parquet"
    if sc_path.exists():
        sc = read_table(sc_path)
        value_col = "standardized_effect" if "standardized_effect" in sc.columns else "effect"
        analysis_type = sc["analysis_type"].astype(str) if "analysis_type" in sc.columns else pd.Series("all", index=sc.index)
        contrast = sc["contrast"].astype(str) if "contrast" in sc.columns else pd.Series("all", index=sc.index)
        sc["feature"] = sc.cancer_id.astype(str) + "::" + analysis_type + "::" + contrast
        views.append(profile_similarity(sc, "pathway_id", "feature", value_col, "sc_pseudotime", settings["top_k_per_view"], settings["min_view_similarity"]))

    ucell_path = cfg["_results"] / "tables" / "ucell_pathway_support.parquet"
    if ucell_path.exists():
        uc = read_table(ucell_path)
        uc["feature"] = uc.cancer_id.astype(str)
        views.append(profile_similarity(uc, "pathway_id", "feature", "ucell_effect", "ucell_profile", settings["top_k_per_view"], settings["min_view_similarity"]))

    views.append(drug_overlap(cfg, member, settings["top_k_per_view"], settings["min_view_similarity"]))
    all_views = pd.concat([x for x in views if x is not None and not x.empty], ignore_index=True)
    if all_views.empty:
        raise RuntimeError("No pathway similarity view could be constructed")
    all_views[["pathway_a", "pathway_b"]] = np.sort(all_views[["pathway_a", "pathway_b"]].astype(str).to_numpy(), axis=1)
    all_views = all_views.groupby(["pathway_a", "pathway_b", "view"], as_index=False).agg(similarity=("similarity", "max"), signed_similarity=("signed_similarity", "mean"))
    write_table(all_views, cfg["_results"] / "tables" / "pathway_similarity_view.parquet")

    pivot = all_views.pivot_table(index=["pathway_a", "pathway_b"], columns="view", values="similarity", aggfunc="max").reset_index()
    weights = settings["weights"]
    numerator = np.zeros(len(pivot), dtype=float)
    denominator = np.zeros(len(pivot), dtype=float)
    for view, weight in weights.items():
        if view not in pivot.columns:
            pivot[view] = np.nan
        finite = pivot[view].notna().to_numpy()
        numerator[finite] += weight * pivot.loc[finite, view].to_numpy(float)
        denominator[finite] += weight
    pivot["combined_similarity"] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    pivot["n_views"] = pivot[[v for v in weights if v in pivot.columns]].notna().sum(axis=1)
    pivot["similarity_edge_id"] = [stable_id("PSIM", a, b) for a, b in zip(pivot.pathway_a, pivot.pathway_b)]
    pivot = pivot.loc[pivot.combined_similarity >= settings["min_fused_similarity"]]
    write_table(pivot, cfg["_results"] / "tables" / "pathway_similarity_edge.parquet")
    return pivot


def build_pathway_families(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim = read_table(cfg["_results"] / "tables" / "pathway_similarity_edge.parquet")
    dim = read_table(cfg["_standardized"] / "dim_pathway.parquet")
    graph = nx.Graph()
    graph.add_nodes_from(dim.pathway_id.astype(str))
    graph.add_weighted_edges_from(sim[["pathway_a", "pathway_b", "combined_similarity"]].itertuples(index=False, name=None))
    settings = cfg["pathway_family"]
    target_mid = (settings["target_min_families"] + settings["target_max_families"]) / 2
    candidates = []
    for resolution in settings["resolution_grid"]:
        communities = nx.community.louvain_communities(graph, weight="weight", resolution=float(resolution), seed=cfg["random_seed"])
        candidates.append((abs(len(communities) - target_mid), resolution, communities))
        if settings["target_min_families"] <= len(communities) <= settings["target_max_families"]:
            break
    _, resolution, communities = min(candidates, key=lambda x: x[0])
    dim_lookup = dim.set_index("pathway_id")
    family_rows = []
    member_rows = []
    for idx, community in enumerate(sorted(communities, key=lambda x: (-len(x), sorted(x)[0])), 1):
        pathways = sorted(str(x) for x in community)
        family_id = f"PF:{idx:04d}"
        subset = dim_lookup.reindex(pathways)
        representative = subset.sort_values("gene_count", ascending=False).iloc[0] if "gene_count" in subset.columns and subset.gene_count.notna().any() else subset.iloc[0]
        family_rows.append({
            "pathway_family_id": family_id,
            "family_name": str(representative.get("pathway_name", family_id)),
            "representative_pathway_id": str(representative.name),
            "n_pathways": len(pathways),
            "resolution": resolution,
            "analysis_version": cfg["analysis_version"],
        })
        subgraph = graph.subgraph(pathways)
        degree = dict(subgraph.degree(weight="weight"))
        max_degree = max(degree.values()) if degree and max(degree.values()) > 0 else 1.0
        for pathway in pathways:
            member_rows.append({
                "pathway_family_id": family_id,
                "pathway_id": pathway,
                "membership_weight": float(degree.get(pathway, 0.0) / max_degree),
                "is_representative": pathway == str(representative.name),
            })
    families = pd.DataFrame(family_rows)
    members = pd.DataFrame(member_rows)
    write_table(families, cfg["_results"] / "tables" / "pathway_family.parquet")
    write_table(members, cfg["_results"] / "tables" / "pathway_family_member.parquet")
    return families, members
