#!/usr/bin/env python3
"""Prepare a bounded, local V3.2 one-seed pilot from existing inputs.

The pathway activity parquet is treated as an immutable precomputed endpoint;
this script only validates/renames ``activity_score`` to the V3.2 interface.
The current migrated lncRNA parquet contains the measured 7,066-ID universe,
so the run is explicitly recorded as an input-aligned diagnostic pilot rather
than silently claiming the stale 16,889/4,712 contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ENV_PY = Path(r"D:\model\CC_HHGT_v2_6_strict_adapter_query\.conda\python.exe")
ACTIVITY_ROOT = ROOT / "artifacts" / "input" / "bulk_pathway_activity"
EXPR_ROOT = ROOT / "artifacts" / "input" / "bulk_lncRNA_expression"
GRAPH_ROOT = Path(
    r"D:\model\V3_1_EXACT_33C_Formal_20260823\migration\V3STATE-exact_pathway_33c_site-20260822T165034Z-7f67e8ec56f9\assets\results\tables"
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sql_ids(values: list[str]) -> str:
    return ",".join("'" + str(v).replace("'", "''") + "'" for v in values)


def load_input_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    expr = con.execute(
        f"""
        select cancer_id, sample_id, patient_id, lncrna_id, logcpm, tpm
        from read_parquet('{EXPR_ROOT.as_posix()}/**/*.parquet')
        """
    ).fetchdf()
    activity = con.execute(
        f"""
        select cancer_id, sample_id, patient_id, pathway_id,
               activity_score as pathway_activity
        from read_parquet('{ACTIVITY_ROOT.as_posix()}/**/*.parquet')
        """
    ).fetchdf()
    expr[["cancer_id", "sample_id", "patient_id", "lncrna_id"]] = expr[
        ["cancer_id", "sample_id", "patient_id", "lncrna_id"]
    ].astype(str)
    activity[["cancer_id", "sample_id", "patient_id", "pathway_id"]] = activity[
        ["cancer_id", "sample_id", "patient_id", "pathway_id"]
    ].astype(str)
    activity["pathway_activity"] = pd.to_numeric(activity.pathway_activity, errors="raise")
    if activity.duplicated(["sample_id", "pathway_id"]).any():
        raise RuntimeError("Existing pathway activity contains duplicate sample/pathway keys")
    return expr, activity


def build_scope(expr: pd.DataFrame, samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cancers = sorted(samples.cancer_id.unique())
    annotation = sorted(expr.lncrna_id.unique())
    total = samples.groupby("cancer_id", observed=True).sample_id.nunique()
    detected = (
        expr.loc[expr.logcpm.gt(0)]
        .groupby(["cancer_id", "lncrna_id"], observed=True)
        .sample_id.nunique()
    )
    idx = pd.MultiIndex.from_product([cancers, annotation], names=["cancer_id", "lncrna_id"])
    det = detected.reindex(idx, fill_value=0).rename("n_detected_samples").reset_index()
    det["n_canonical_samples"] = det.cancer_id.map(total).astype(int)
    det["detection_rate"] = det.n_detected_samples / det.n_canonical_samples
    det["within_cancer_eligible"] = det.detection_rate.ge(0.10)
    counts = det.groupby("lncrna_id", observed=True).within_cancer_eligible.sum()
    det["detected_cancers"] = det.lncrna_id.map(counts).astype(int)
    det["shared_or_local_scope"] = np.select(
        [det.within_cancer_eligible & det.detected_cancers.ge(3), det.within_cancer_eligible],
        ["shared", "cancer_local"],
        default="not_expression_eligible",
    )
    # Keep a compact eligibility table for association code and manifests.
    return det, pd.DataFrame(
        {
            "lncrna_id": annotation,
            "detected_cancers": [int(counts.get(x, 0)) for x in annotation],
        }
    )


def choose_candidates(
    det: pd.DataFrame, activity: pd.DataFrame, *, seed: int, lnc_per_cancer: int, n_pathways: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    pathways = sorted(activity.pathway_id.unique(), key=lambda x: _digest(f"{seed}|path|{x}"))[:n_pathways]
    selected: list[pd.DataFrame] = []
    for cancer, frame in det.groupby("cancer_id", observed=True, sort=True):
        eligible = frame.loc[frame.within_cancer_eligible].copy()
        eligible = eligible.sort_values(
            ["detection_rate", "lncrna_id"], ascending=[False, True], kind="stable"
        )
        # Deterministic tie shuffle prevents selecting only lexicographic IDs.
        if len(eligible) > lnc_per_cancer:
            top = eligible.iloc[: min(len(eligible), lnc_per_cancer * 4)].copy()
            order = rng.permutation(len(top))
            eligible = top.iloc[order].sort_values("lncrna_id", kind="stable")
        selected.append(eligible.head(lnc_per_cancer))
    chosen = pd.concat(selected, ignore_index=True)
    chosen = chosen[["cancer_id", "lncrna_id", "detection_rate", "detected_cancers", "shared_or_local_scope"]]
    pairs = chosen.assign(_key=1).merge(pd.DataFrame({"pathway_id": pathways, "_key": 1}), on="_key").drop(columns="_key")
    return chosen, pairs, pathways


def hierarchy(pathways: list[str]) -> pd.DataFrame:
    con = duckdb.connect()
    ids = _sql_ids(pathways)
    edge_path = (GRAPH_ROOT / "graph_edge.parquet").as_posix()
    frame = con.execute(
        f"""
        select distinct source_canonical_id as pathway_id, target_canonical_id as pathway_family_id
        from read_parquet('{edge_path}')
        where relation_type='member_of_family' and source_canonical_id in ({ids})
        """
    ).fetchdf()
    frame[["pathway_id", "pathway_family_id"]] = frame[["pathway_id", "pathway_family_id"]].astype(str)
    frame = frame.drop_duplicates("pathway_id")
    missing = sorted(set(pathways) - set(frame.pathway_id))
    if missing:
        frame = pd.concat(
            [frame, pd.DataFrame({"pathway_id": missing, "pathway_family_id": [f"FAMILY_UNMAPPED:{x}" for x in missing]})],
            ignore_index=True,
        )
    return frame.sort_values("pathway_id", kind="stable").reset_index(drop=True)


def graph_bundle(lncs: list[str], pathways: list[str], cancers: list[str]):
    import torch  # noqa: F401
    from cc_hhgt.gnn import _materialize_graph_bundle
    from cc_hhgt.v32.safe_graph import build_safe_graph

    con = duckdb.connect()
    edge_path = (GRAPH_ROOT / "graph_edge.parquet").as_posix()
    psql = _sql_ids(pathways)
    edges = con.execute(
        f"""
        select * from read_parquet('{edge_path}')
        where (relation_type='member_of' and target_type='pathway' and target_canonical_id in ({psql}))
           or (relation_type='member_of_family' and source_type='pathway' and source_canonical_id in ({psql}))
        """
    ).fetchdf()
    edges = edges.drop_duplicates(["source_type", "source_canonical_id", "relation_type", "target_type", "target_canonical_id"])
    edges["weight"] = pd.to_numeric(edges.weight, errors="coerce").fillna(1.0).clip(lower=1e-5).astype(float)
    # Run the V3.2 safety validator on a role-labelled view; no lncRNA-pair,
    # literature, drug, perturbation, validation, or test edge is admitted.
    safe = edges.rename(columns={"source_canonical_id": "source_id", "target_canonical_id": "target_id"})[
        ["source_type", "source_id", "relation_type", "target_type", "target_id", "weight"]
    ].copy()
    safe["edge_role"] = np.where(safe.relation_type.eq("member_of_family"), "pathway_hierarchy", "protein_pathway")
    safe["source_split"] = "static"
    safe["outcome_derived"] = False
    build_safe_graph(safe, outer_fold=0)

    node_path = (GRAPH_ROOT / "graph_node.parquet").as_posix()
    needed = set(lncs) | set(pathways) | set(edges.source_canonical_id) | set(edges.target_canonical_id)
    nodes = pd.read_parquet(node_path)
    nodes = nodes.loc[nodes.canonical_id.astype(str).isin(needed)].copy()
    # Cancer nodes are part of the decoder contract even without context edges.
    all_nodes = pd.read_parquet(node_path, columns=["canonical_id", "node_type", "node_id", "label", "node_index_within_type", "out_degree", "in_degree", "log_degree", "bias_feature"])
    all_nodes = all_nodes.loc[all_nodes.node_type.eq("cancer") & all_nodes.canonical_id.astype(str).isin(cancers)]
    nodes = pd.concat([nodes, all_nodes], ignore_index=True).drop_duplicates("canonical_id")
    nodes = nodes.sort_values(["node_type", "node_index_within_type", "canonical_id"], kind="stable").reset_index(drop=True)
    return _materialize_graph_bundle(nodes, edges, feature_edges=edges, canonical_edges=edges)


def _split_associations(expr: pd.DataFrame, activity: pd.DataFrame, split_manifest: pd.DataFrame, hierarchy_frame: pd.DataFrame, eligibility: pd.DataFrame, split: str):
    from cc_hhgt.v32.associations import compute_fold_associations

    split_ids = split_manifest.loc[split_manifest.split.eq(split), ["cancer_id", "sample_id", "split", "outer_fold"]]
    lnc_ids = set(eligibility.loc[eligibility.within_cancer_eligible, "lncrna_id"].astype(str))
    expr = expr.loc[expr.lncrna_id.astype(str).isin(lnc_ids)].copy()
    activity = activity.rename(columns={"pathway_activity": "pathway_activity_scaled"})
    return compute_fold_associations(
        expr, activity, split_ids, hierarchy_frame, eligibility,
        split=split, expression_value_column="logcpm", activity_value_column="pathway_activity_scaled",
        pathway_block_size=64,
    )


def _candidate_frame(pairs: pd.DataFrame, assoc: pd.DataFrame, *, fold: int, split: str) -> pd.DataFrame:
    keys = ["cancer_id", "lncrna_id", "pathway_id"]
    frame = pairs.merge(assoc, on=keys, how="left", suffixes=("", "_assoc"))
    frame["pathway_family_id"] = frame.pathway_family_id.astype(str)
    frame["discovery_effect"] = pd.to_numeric(frame.discovery_effect, errors="coerce").fillna(0.0)
    frame["discovery_fdr"] = pd.to_numeric(frame.discovery_fdr, errors="coerce").fillna(1.0)
    frame["discovery_pvalue"] = pd.to_numeric(frame.discovery_pvalue, errors="coerce").fillna(1.0)
    frame["association_n_samples"] = pd.to_numeric(frame.association_n_samples, errors="coerce").fillna(0).astype(int)
    frame["label_class"] = frame.label_class.fillna("unlabeled")
    frame["proxy_label"] = frame.label_class.isin(["strong_positive", "weak_positive"]).astype("int8")
    frame["weak_positive"] = frame.label_class.eq("weak_positive")
    frame["association_direction"] = np.where(frame.discovery_effect.ge(0), "positive", "negative")
    frame["outer_fold"] = int(fold)
    frame["association_split"] = split
    return frame


def _make_batches(frame: pd.DataFrame, base_logit: np.ndarray, bundle: Any, *, batch_size: int = 2048):
    import torch
    maps = bundle.node_maps
    rows = []
    for start in range(0, len(frame), batch_size):
        sub = frame.iloc[start:start + batch_size]
        l = torch.tensor([maps["lncRNA"][str(x)] for x in sub.lncrna_id], dtype=torch.long)
        p = torch.tensor([maps["pathway"][str(x)] for x in sub.pathway_id], dtype=torch.long)
        c = torch.tensor([maps["cancer"][str(x)] for x in sub.cancer_id], dtype=torch.long)
        detection = pd.to_numeric(sub.detection_rate, errors="coerce").fillna(0).to_numpy(float)
        effect = pd.to_numeric(sub.discovery_effect, errors="coerce").fillna(0).to_numpy(float)
        shared = sub.shared_or_local_scope.astype(str).eq("shared").to_numpy(float)
        context = np.column_stack([shared, detection, np.clip(np.abs(effect), 0, 5), (sub.detected_cancers.to_numpy(float) / 33.0)])
        rows.append({
            "candidate_batch": {"l": l, "p": p, "c": c},
            "base_logit": torch.tensor(base_logit[start:start + len(sub)], dtype=torch.float32),
            "conservation_context": torch.tensor(context, dtype=torch.float32),
            "proxy_label": torch.tensor(sub.proxy_label.to_numpy(float), dtype=torch.float32),
            "weak_positive": torch.tensor(sub.weak_positive.to_numpy(bool), dtype=torch.bool),
            "graph_available": torch.ones(len(sub), dtype=torch.bool),
            "direction_label": torch.tensor((effect >= 0).astype(float), dtype=torch.float32),
            "direction_available": torch.tensor((np.abs(effect) >= 0.15), dtype=torch.bool),
        })
    return rows


def prepare_fold(fold: int, expr: pd.DataFrame, activity: pd.DataFrame, fold_manifest: pd.DataFrame, pairs: pd.DataFrame, hierarchy_frame: pd.DataFrame, eligibility: pd.DataFrame, artifact_hashes: dict[str, str], output: Path, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from cc_hhgt.v32.baselines import build_safe_hashed_design
    from cc_hhgt.v32.contracts import candidate_key_sha256
    from cc_hhgt.v32.patient_folds import assign_outer_split
    from cc_hhgt.v32.prepared import build_prepared_fold_payload

    split_manifest = assign_outer_split(fold_manifest, fold, n_folds=5, validation_offset=1)
    assoc = {split: _split_associations(expr, activity, split_manifest, hierarchy_frame, eligibility, split) for split in ("train", "validation", "test")}
    frames = {split: _candidate_frame(pairs, assoc[split], fold=fold, split=split) for split in assoc}
    # Replication labels must share the exact same candidate universe.
    keys = ["cancer_id", "lncrna_id", "pathway_id"]
    common = frames["train"][keys].merge(frames["validation"][keys], on=keys).merge(frames["test"][keys], on=keys)
    frames = {split: frames[split].merge(common, on=keys, how="inner", validate="one_to_one") for split in frames}
    for frame in frames.values():
        frame["cross_cancer_support_frequency"] = frame.lncrna_id.map(eligibility.groupby("lncrna_id").detected_cancers.max()).fillna(0).to_numpy(float) / 33.0
        frame["cross_cancer_direction_consistency"] = 0.0
        frame["cross_cancer_i2"] = 0.0
    # A fast liblinear L1 fit is sufficient for the bounded pilot and avoids
    # spending hours on the full saga C-grid before the graph run starts.
    x_train = build_safe_hashed_design(frames["train"], n_features=4096)
    x_val = build_safe_hashed_design(frames["validation"], n_features=4096)
    y_train = frames["train"].proxy_label.to_numpy(int)
    sample_weight = np.where(frames["train"].label_class.astype(str).eq("weak_positive"), 0.35, np.where(y_train == 0, 0.12, 1.0))
    model = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=250, random_state=seed)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    val_probability = model.predict_proba(x_val)[:, 1]
    val_y = frames["validation"].proxy_label.to_numpy(int)
    baseline = type("PilotBaseline", (), {})()
    baseline.model = model
    baseline.selected_c = 0.1
    baseline.candidate_sha256 = candidate_key_sha256(frames["train"])
    baseline.validation_auprc = float(average_precision_score(val_y, val_probability))
    baseline.validation_auroc = float(roc_auc_score(val_y, val_probability)) if np.unique(val_y).size == 2 else float("nan")
    train_logit = model.decision_function(x_train)
    val_logit = model.decision_function(x_val)
    bundle = graph_bundle(sorted(pairs.lncrna_id.unique()), sorted(pairs.pathway_id.unique()), sorted(pairs.cancer_id.unique()))
    legacy_cfg = {"training": {"hidden_channels": 96, "pair_hidden_channels": 96, "num_layers": 2, "num_heads": 2, "dropout": 0.20}, "residual_learning": {"enabled": False}}
    payload = build_prepared_fold_payload(
        patient_fold=fold, bundle=bundle, feature_dim=4, legacy_model_config=legacy_cfg,
        train_batches=_make_batches(frames["train"], train_logit, bundle),
        validation_batches=_make_batches(frames["validation"], val_logit, bundle),
        artifact_hashes=artifact_hashes, conservation_context_features=4, graph=None,
    )
    payload["input_scope"] = {"annotation_source_n": int(expr.lncrna_id.nunique()), "shared_n": int(eligibility.loc[eligibility.detected_cancers.ge(3), "lncrna_id"].nunique()), "candidate_rows": int(len(frames["train"]))}
    output.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(payload, output / f"PATIENT_FOLD_{fold}.pt")
    metrics = {"fold": fold, "train_rows": len(frames["train"]), "validation_rows": len(frames["validation"]), "test_rows": len(frames["test"]), "l1_selected_c": baseline.selected_c, "l1_validation_auprc": baseline.validation_auprc, "l1_validation_auroc": baseline.validation_auroc, "candidate_sha256": baseline.candidate_sha256}
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hashes", required=True, help="JSON with the four authorization hashes")
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "prepared"))
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--lnc-per-cancer", type=int, default=64)
    parser.add_argument("--pathways", type=int, default=256)
    args = parser.parse_args()
    hashes = json.loads(Path(args.hashes).read_text(encoding="utf-8"))
    expr, activity = load_input_tables()
    source_lnc_n = int(expr.lncrna_id.nunique())
    samples = activity[["cancer_id", "sample_id", "patient_id"]].drop_duplicates().sort_values(["cancer_id", "patient_id", "sample_id"])
    det, _ = build_scope(expr, samples)
    chosen, pairs, pathways = choose_candidates(det, activity, seed=args.seed, lnc_per_cancer=args.lnc_per_cancer, n_pathways=args.pathways)
    # Release the full measured matrix before any pivot operation.  The pilot
    # intentionally trains on a deterministic candidate slice; the complete
    # universe remains represented in the scope parquet and is not silently
    # substituted into the formal website output.
    chosen_lnc = set(chosen.lncrna_id.astype(str))
    expr = expr.loc[expr.lncrna_id.astype(str).isin(chosen_lnc)].copy()
    activity = activity.loc[activity.pathway_id.astype(str).isin(set(pathways))].copy()
    hierarchy_frame = hierarchy(pathways)
    pairs = pairs.merge(hierarchy_frame, on="pathway_id", how="left", validate="many_to_one")
    # The activity endpoint already exists; only validate its schema/finite values.
    if not np.isfinite(activity.pathway_activity.to_numpy(float)).all():
        raise RuntimeError("Existing pathway activity has non-finite values")
    from cc_hhgt.v32.patient_folds import build_patient_fold_manifest
    fold_manifest = build_patient_fold_manifest(samples, n_folds=5, seed=args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fold_manifest.to_csv(out / "PATIENT_FOLD_MANIFEST.tsv", sep="\t", index=False)
    det.to_parquet(out / "LNCRNA_SCOPE_7066_INPUT.parquet", index=False)
    hierarchy_frame.to_parquet(out / "PATHWAY_HIERARCHY.parquet", index=False)
    summary = {"source_lnc_n": source_lnc_n, "pilot_lnc_n": int(expr.lncrna_id.nunique()), "shared_lnc_n_ge3": int(det.loc[det.detected_cancers.ge(3), "lncrna_id"].nunique()), "eligible_pairs": int(len(pairs)), "pathways_sampled": len(pathways), "lnc_per_cancer": args.lnc_per_cancer}
    (out / "PREP_INPUT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    metrics = []
    for fold in range(5):
        print(json.dumps({"status": "PREPARING", "fold": fold, **summary}), flush=True)
        metrics.append(prepare_fold(fold, expr, activity, fold_manifest, pairs, hierarchy_frame, det, hashes, out, args.seed))
        print(json.dumps({"status": "PREPARED", **metrics[-1]}), flush=True)
    (out / "PREP_METRICS.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
