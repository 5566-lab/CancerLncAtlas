#!/usr/bin/env python3
"""Prepare the formal V3.2 one-seed, five-fold local run.

This uses the 149-generated edgeR TMM logCPM matrices and the existing
precomputed exact-pathway activity table.  Candidate pairs are deterministic
and capped at 100,000 per cancer (below the frozen 250,000 evaluation cap),
with every within-cancer eligible lncRNA represented in the cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FORMAL_EXPR = ROOT / "artifacts" / "input" / "formal_lncRNA_expression"
FORMAL_ACTIVITY = ROOT / "artifacts" / "input" / "formal_pathway_activity"
CANDIDATE_SAMPLER = "sha256_affine_cycle_v1"
KEYS = ["cancer_id", "lncrna_id", "pathway_id"]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def load_scope(rates: pd.DataFrame) -> pd.DataFrame:
    rates = rates.copy()
    required = {"cancer_id", "lncrna_id"}
    if missing := sorted(required - set(rates)):
        raise RuntimeError(f"Pinned detection authority lacks columns: {missing}")
    if rates.lncrna_id.nunique() != 16889 or rates.cancer_id.nunique() != 33:
        raise RuntimeError("Formal rates table is not the complete 16,889 x 33 grid")
    detection_column = (
        "detection_rate_logcpm_gt0"
        if "detection_rate_logcpm_gt0" in rates
        else "detection_rate"
    )
    if detection_column not in rates:
        raise RuntimeError("Pinned detection authority lacks a detection-rate column")
    det = rates[["cancer_id", "lncrna_id", detection_column]].rename(
        columns={detection_column: "detection_rate"}
    )
    det["within_cancer_eligible"] = det.detection_rate.ge(0.10)
    counts = det.groupby("lncrna_id", observed=True).within_cancer_eligible.sum()
    det["detected_cancers"] = det.lncrna_id.map(counts).astype(int)
    det["shared_or_local_scope"] = np.select(
        [det.within_cancer_eligible & det.detected_cancers.ge(3), det.within_cancer_eligible],
        ["shared", "cancer_local"],
        default="not_expression_eligible",
    )
    observed_shared = int(det.loc[det.detected_cancers.ge(3), "lncrna_id"].nunique())
    if observed_shared != 4712:
        raise RuntimeError(f"Formal eligibility mismatch: expected 4712, observed {observed_shared}")
    return det.sort_values(["cancer_id", "lncrna_id"], kind="stable").reset_index(drop=True)


def load_activity(root: str | Path = FORMAL_ACTIVITY) -> pd.DataFrame:
    import duckdb
    activity_root = Path(root).resolve(strict=True)
    con = duckdb.connect()
    activity = con.execute(
        f"""
        select cancer_id::varchar as cancer_id, sample_id::varchar as sample_id,
               patient_id::varchar as patient_id, pathway_id::varchar as pathway_id,
               activity_score::double as pathway_activity_scaled
        from read_parquet('{activity_root.as_posix()}/**/*.parquet')
        """
    ).fetchdf()
    if activity.duplicated(["sample_id", "pathway_id"]).any():
        raise RuntimeError("Filtered precomputed activity has duplicate keys")
    if not np.isfinite(activity.pathway_activity_scaled.to_numpy(float)).all():
        raise RuntimeError("Filtered precomputed activity has non-finite values")
    return activity


def load_expression_cancer(
    cancer: str, root: str | Path = FORMAL_EXPR
) -> pd.DataFrame:
    expression_root = Path(root).resolve(strict=True)
    return pd.read_parquet(
        expression_root / f"cancer_id={cancer}" / "part-0.parquet"
    )


def build_hierarchy(pathways: list[str], static_authority: pd.DataFrame) -> pd.DataFrame:
    """Build the candidate hierarchy only from the pinned fresh authority."""

    required = {"pathway_id", "pathway_family_id", "weight"}
    if missing := sorted(required - set(static_authority)):
        raise RuntimeError(f"Pinned pathway hierarchy lacks columns: {missing}")
    result = static_authority.loc[
        static_authority.pathway_id.astype(str).isin(set(map(str, pathways)))
    ].copy()
    result["weight"] = pd.to_numeric(result.weight, errors="coerce")
    if result.weight.isna().any() or not np.isfinite(result.weight.to_numpy(float)).all():
        raise RuntimeError("Pinned candidate hierarchy has nonfinite weights")
    if result.weight.lt(0).any():
        raise RuntimeError("Pinned candidate hierarchy has negative weights")
    result = result[["pathway_id", "pathway_family_id"]].astype(str).drop_duplicates()
    if result.duplicated("pathway_id").any():
        raise RuntimeError("Pinned hierarchy maps one exact pathway to multiple families")
    missing = sorted(set(map(str, pathways)) - set(result.pathway_id.astype(str)))
    if missing:
        raise RuntimeError(
            f"Pinned hierarchy lacks candidate exact pathways: {missing[:10]}"
        )
    # Exact zero-weight family rows retain the pathway-to-family lookup used by
    # the association baseline, but ``materialize_pathway_hierarchy`` excludes
    # them from graph messages.  This preserves all 2,135 exact pathways while
    # emitting the source-audited 2,114 nonzero hierarchy edges.
    return result.sort_values("pathway_id", kind="stable").reset_index(drop=True)


def make_candidate_pairs(scope: pd.DataFrame, pathways: list[str], *, budget: int, seed: int) -> pd.DataFrame:
    """Build a deterministic, balanced candidate universe in O(output rows).

    The previous implementation hashed and sorted every pathway separately for
    every eligible lncRNA.  With 33 cancers and 2,135 pathways that expanded a
    3.3-million-row output into hundreds of millions of SHA256 evaluations.
    A SHA256-derived affine cycle is a full permutation of the pathway indices
    whenever ``step`` is coprime to the pathway count.  It therefore retains
    deterministic seed binding, no replacement, and per-lncRNA coverage while
    doing one digest per lncRNA plus exactly the rows that will be emitted.
    """

    rows = []
    path_values = sorted(set(map(str, pathways)))
    if not path_values:
        raise RuntimeError("Formal candidate sampler requires at least one pathway")
    pathway_count = len(path_values)
    for cancer, frame in scope.loc[scope.within_cancer_eligible].groupby("cancer_id", observed=True, sort=True):
        lncs = sorted(frame.lncrna_id.astype(str).unique())
        if not lncs:
            continue
        if budget < len(lncs):
            raise RuntimeError(f"Budget {budget} cannot cover every eligible lncRNA in {cancer}")
        if budget > len(lncs) * pathway_count:
            raise RuntimeError(
                f"Budget {budget} exceeds the no-replacement lncRNA-pathway "
                f"universe for {cancer}: {len(lncs) * pathway_count}"
            )
        base, extra = divmod(int(budget), len(lncs))
        for index, lnc in enumerate(lncs):
            take = base + (1 if index < extra else 0)
            digest = hashlib.sha256(
                f"{CANDIDATE_SAMPLER}|{seed}|{cancer}|{lnc}".encode()
            ).digest()
            offset = int.from_bytes(digest[:8], "big") % pathway_count
            step = int.from_bytes(digest[8:16], "big") % pathway_count
            if step == 0:
                step = 1
            while math.gcd(step, pathway_count) != 1:
                step += 1
                if step == pathway_count:
                    step = 1
            ordered = [
                path_values[(offset + j * step) % pathway_count]
                for j in range(take)
            ]
            rows.extend((str(cancer), lnc, p) for p in ordered)
    result = pd.DataFrame(rows, columns=["cancer_id", "lncrna_id", "pathway_id"])
    if result.empty or result.duplicated().any():
        raise RuntimeError("Formal candidate sampler produced an invalid universe")
    return result.sort_values(["cancer_id", "lncrna_id", "pathway_id"], kind="stable").reset_index(drop=True)


def _rank_standardize(matrix: pd.DataFrame) -> np.ndarray:
    from cc_hhgt.v32.associations import _rank_and_standardize
    # A lncRNA/pathway may be absent from every sample in a small validation
    # split; represent that column as zero information instead of propagating
    # all-NaN warnings through the correlation calculation.
    matrix = matrix.copy()
    matrix = matrix.fillna(matrix.mean(axis=0)).fillna(0.0)
    return _rank_and_standardize(matrix, np.ones((len(matrix), 1), dtype=float))


def sampled_associations(
    expr_by_cancer: dict[str, pd.DataFrame], activity_by_cancer: dict[str, pd.DataFrame],
    candidates: pd.DataFrame, split_manifest: pd.DataFrame, split: str,
    hierarchy: pd.DataFrame, scope: pd.DataFrame,
) -> pd.DataFrame:
    from cc_hhgt.v32.associations import benjamini_hochberg
    from cc_hhgt.stats import correlation_p_values

    rows = []
    for cancer, pairs in candidates.groupby("cancer_id", observed=True, sort=True):
        sample_ids = split_manifest.loc[(split_manifest.cancer_id == cancer) & (split_manifest.split == split), "sample_id"].astype(str).tolist()
        expr = expr_by_cancer[str(cancer)]
        activity = activity_by_cancer[str(cancer)]
        expr = expr.loc[expr.sample_id.astype(str).isin(sample_ids)]
        activity = activity.loc[activity.sample_id.astype(str).isin(sample_ids)]
        lnc_ids = sorted(pairs.lncrna_id.astype(str).unique())
        # Keep the candidate pathway universe fixed across train/validation/
        # test even when a rare pathway is absent from one split's activity
        # rows; the missing column becomes an all-NaN, zero-information vector.
        pathway_ids = sorted(candidates.pathway_id.astype(str).unique())
        common = sorted(set(sample_ids) & set(expr.sample_id.astype(str)) & set(activity.sample_id.astype(str)))
        if len(common) < 5:
            raise RuntimeError(f"{cancer}/{split} has only {len(common)} aligned samples")
        lnc_wide = expr.pivot(index="sample_id", columns="lncrna_id", values="logcpm").reindex(index=common, columns=lnc_ids)
        path_wide = activity.pivot(index="sample_id", columns="pathway_id", values="pathway_activity_scaled").reindex(index=common, columns=pathway_ids)
        lnc_values = _rank_standardize(lnc_wide)
        path_values = _rank_standardize(path_wide)
        residual_design_rank = 1
        degrees = len(common) - residual_design_rank - 1
        if degrees <= 0:
            raise RuntimeError(
                f"No residual correlation degrees of freedom for {cancer}/{split}: "
                f"n={len(common)}, design_rank={residual_design_rank}"
            )
        rho = np.clip(lnc_values.T @ path_values, -0.999999, 0.999999)
        pvalue = correlation_p_values(
            rho,
            len(common),
            residual_design_rank=residual_design_rank,
        )
        fdr = benjamini_hochberg(pvalue.reshape(-1)).reshape(pvalue.shape)
        lmap = {x: i for i, x in enumerate(lnc_ids)}
        pmap = {x: i for i, x in enumerate(pathway_ids)}
        li = pairs.lncrna_id.astype(str).map(lmap).to_numpy(int)
        pi = pairs.pathway_id.astype(str).map(pmap).to_numpy(int)
        rows.append(pd.DataFrame({
            "cancer_id": str(cancer), "lncrna_id": pairs.lncrna_id.astype(str).to_numpy(),
            "pathway_id": pairs.pathway_id.astype(str).to_numpy(),
            "discovery_effect": rho[li, pi], "discovery_pvalue": pvalue[li, pi],
            "discovery_fdr": fdr[li, pi], "association_n_samples": len(common),
            "residual_design_rank": residual_design_rank, "correlation_df": degrees,
        }))
    result = pd.concat(rows, ignore_index=True)
    result = result.merge(hierarchy, on="pathway_id", how="left", validate="many_to_one")
    result = result.merge(scope[["cancer_id", "lncrna_id", "detection_rate", "detected_cancers", "shared_or_local_scope"]], on=["cancer_id", "lncrna_id"], how="left", validate="many_to_one")
    strong = result.discovery_fdr.le(0.05) & result.discovery_effect.abs().ge(0.20)
    weak = result.discovery_fdr.le(0.10) & result.discovery_effect.abs().ge(0.15)
    result["label_class"] = np.select([strong, weak], ["strong_positive", "weak_positive"], default="unlabeled")
    result["proxy_label"] = result.label_class.isin(["strong_positive", "weak_positive"]).astype("int8")
    result["weak_positive"] = result.label_class.eq("weak_positive")
    result["association_direction"] = np.where(result.discovery_effect.ge(0), "positive", "negative")
    return result.sort_values(["cancer_id", "lncrna_id", "pathway_id"], kind="stable").reset_index(drop=True)


def attach_replication_labels(discovery: pd.DataFrame, replication: pd.DataFrame) -> pd.DataFrame:
    """Keep train-patient discovery features and attach held-out labels only.

    Replication effects are attached under label-only names.  They never
    replace the train-derived discovery feature.
    """
    keys = KEYS
    label_columns = keys + [
        "proxy_label",
        "weak_positive",
        "discovery_effect",
        "association_available",
        "association_unavailable_reason",
    ]
    if missing := sorted(set(label_columns) - set(replication.columns)):
        raise RuntimeError(f"Replication labels lack corrected fields: {missing}")
    labels = replication[label_columns].copy().rename(
        columns={
            "discovery_effect": "replication_effect",
            "association_available": "replication_association_available",
            "association_unavailable_reason": "replication_unavailable_reason",
        }
    )
    if labels.duplicated(keys).any():
        raise RuntimeError("Replication labels contain duplicate candidate keys")
    result = discovery.rename(
        columns={"association_available": "train_association_available"}
    ).drop(
        columns=[
            "proxy_label",
            "weak_positive",
            "label_class",
            "association_unavailable_reason",
        ],
        errors="ignore",
    ).merge(
        labels, on=keys, how="inner", validate="one_to_one"
    )
    if len(result) != len(discovery):
        raise RuntimeError("Replication labels do not cover the discovery candidate universe")
    result["label_class"] = np.where(
        result["weak_positive"].astype(bool), "weak_positive",
        np.where(result["proxy_label"].astype(bool), "strong_positive", "unlabeled")
    )
    result["train_discovery_effect"] = pd.to_numeric(
        result["discovery_effect"], errors="raise"
    )
    replication_effect = pd.to_numeric(result["replication_effect"], errors="coerce")
    replication_available = result.replication_association_available.astype(bool)
    if replication_effect.loc[~replication_available].notna().any():
        raise RuntimeError("Unavailable replication association carries a numeric effect")
    result["association_available"] = replication_available
    result["association_unavailable_reason"] = result.replication_unavailable_reason
    result["replication_direction_label"] = replication_effect.ge(0).astype("int8")
    result["replication_direction_available"] = (
        replication_available & replication_effect.abs().ge(0.15)
    )
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _fast_l1(frames: dict[str, pd.DataFrame], *, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from cc_hhgt.v32.baselines import build_safe_hashed_design
    from cc_hhgt.v32.contracts import candidate_key_sha256
    # Check exact key equality without holding a second merged copy.
    keys = ["cancer_id", "lncrna_id", "pathway_id"]
    reference = frames["train"][keys]
    for name in ("validation",):
        if not reference.equals(frames[name][keys]):
            raise RuntimeError(f"Candidate identity drift in {name}")
    x_train = build_safe_hashed_design(frames["train"], n_features=4096)
    x_val = build_safe_hashed_design(frames["validation"], n_features=4096)
    y = frames["train"].proxy_label.to_numpy(int)
    train_available = frames["train"].association_available.to_numpy(bool)
    if np.unique(y[train_available]).size != 2:
        raise RuntimeError("Available adjusted training associations require both classes")
    weight = np.where(frames["train"].label_class.astype(str).eq("weak_positive"), 0.35, np.where(y == 0, 0.12, 1.0))
    weight = np.where(train_available, weight, 0.0)
    model = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=250, random_state=seed)
    model.fit(x_train, y, sample_weight=weight)
    val_prob = model.predict_proba(x_val)[:, 1]
    val_available = frames["validation"].association_available.to_numpy(bool)
    val_y = frames["validation"].proxy_label.to_numpy(int)[val_available]
    val_probability = val_prob[val_available]
    metrics = {
        "selected_c": 0.1,
        "candidate_sha256": candidate_key_sha256(frames["train"]),
        "validation_available_rows": int(val_available.sum()),
        "validation_auprc": float(average_precision_score(val_y, val_probability)),
        "validation_auroc": float(roc_auc_score(val_y, val_probability)) if np.unique(val_y).size == 2 else None,
        "test_metrics_computed": False,
    }
    return model, metrics, x_train, x_val


def build_graph(
    graph_inputs: Any,
    *,
    outer_fold: int,
    split_manifest: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
):
    """Build one hash-bound fold authority; no implicit graph source exists."""

    from cc_hhgt.v32.formal_graph_authority import build_bound_formal_graph

    return build_bound_formal_graph(
        graph_inputs,
        outer_fold=int(outer_fold),
        split_manifest=split_manifest,
        candidate_pairs=candidate_pairs,
    )


def make_batches(
    frame: pd.DataFrame,
    logits: np.ndarray,
    bundle: Any,
    *,
    split: str,
    batch_size: int = 8192,
):
    import torch
    maps = bundle.node_maps
    result = []
    for start in range(0, len(frame), batch_size):
        sub = frame.iloc[start:start + batch_size]
        effect = pd.to_numeric(sub.discovery_effect, errors="coerce").fillna(0).to_numpy(float)
        context = np.column_stack([
            sub.shared_or_local_scope.astype(str).eq("shared").to_numpy(float),
            pd.to_numeric(sub.detection_rate, errors="coerce").fillna(0).to_numpy(float),
            np.clip(np.abs(effect), 0, 5),
            pd.to_numeric(sub.detected_cancers, errors="coerce").fillna(0).to_numpy(float) / 33.0,
        ])
        if split == "train":
            direction_label = (effect >= 0).astype(float)
            direction_available = np.abs(effect) >= 0.15
        elif split == "validation":
            required = {
                "replication_direction_label",
                "replication_direction_available",
            }
            if missing := sorted(required - set(sub.columns)):
                raise RuntimeError(
                    f"Validation batches lack replication direction fields: {missing}"
                )
            direction_label = sub.replication_direction_label.to_numpy(float)
            direction_available = sub.replication_direction_available.to_numpy(bool)
        else:
            raise RuntimeError(
                f"Training preparation cannot materialize sealed split {split!r}"
            )
        result.append({
            "candidate_batch": {
                "l": torch.tensor([maps["lncRNA"][str(x)] for x in sub.lncrna_id], dtype=torch.long),
                "p": torch.tensor([maps["pathway"][str(x)] for x in sub.pathway_id], dtype=torch.long),
                "c": torch.tensor([maps["cancer"][str(x)] for x in sub.cancer_id], dtype=torch.long),
            },
            "base_logit": torch.tensor(logits[start:start + len(sub)], dtype=torch.float32),
            "conservation_context": torch.tensor(context, dtype=torch.float32),
            "proxy_label": torch.tensor(sub.proxy_label.to_numpy(float), dtype=torch.float32),
            "weak_positive": torch.tensor(sub.weak_positive.to_numpy(bool), dtype=torch.bool),
            "association_available": torch.tensor(
                sub.association_available.to_numpy(bool), dtype=torch.bool
            ),
            "graph_available": torch.ones(len(sub), dtype=torch.bool),
            "direction_label": torch.tensor(direction_label, dtype=torch.float32),
            "direction_available": torch.tensor(direction_available, dtype=torch.bool),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hashes", required=True)
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--graph-authority-receipt", required=True)
    parser.add_argument("--graph-authority-receipt-sha256", required=True)
    parser.add_argument("--graph-detection", required=True)
    parser.add_argument("--graph-detection-sha256", required=True)
    parser.add_argument("--graph-signed-membership", required=True)
    parser.add_argument("--graph-signed-membership-sha256", required=True)
    parser.add_argument("--graph-pathway-hierarchy", required=True)
    parser.add_argument("--graph-pathway-hierarchy-sha256", required=True)
    parser.add_argument("--graph-lnc-protein-binding", required=True)
    parser.add_argument("--graph-lnc-protein-binding-sha256", required=True)
    parser.add_argument("--graph-protein-gene-encoding", required=True)
    parser.add_argument("--graph-protein-gene-encoding-sha256", required=True)
    parser.add_argument("--graph-ppi", required=True)
    parser.add_argument("--graph-ppi-sha256", required=True)
    parser.add_argument(
        "--graph-fold-expression-pattern",
        required=True,
        help="Explicit outer-train expression path containing one literal {fold}",
    )
    parser.add_argument(
        "--graph-fold-coexpression-pattern",
        required=True,
        help="Explicit outer-train coexpression path containing one literal {fold}",
    )
    parser.add_argument(
        "--formal-expression-root",
        default=str(FORMAL_EXPR),
        help="Explicit patient-level lncRNA expression Parquet root",
    )
    parser.add_argument(
        "--formal-activity-root",
        default=str(FORMAL_ACTIVITY),
        help="Explicit patient-level exact-pathway activity Parquet root",
    )
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "formal_prepared"))
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--budget-per-cancer", type=int, default=100000)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument(
        "--association-mode",
        choices=("legacy", "local-cnv-adjusted"),
        default="legacy",
    )
    parser.add_argument("--formal-candidates")
    parser.add_argument("--formal-candidates-sha256")
    parser.add_argument("--local-cnv-audit-root")
    parser.add_argument("--local-cnv-audit-success-sha256")
    parser.add_argument("--local-cnv-validation-root")
    parser.add_argument("--local-cnv-validation-success-sha256")
    args = parser.parse_args()
    from cc_hhgt.v32.patient_fold_authority import (
        DEFAULT_SEED,
        validate_frozen_v32_patient_fold_binding,
    )

    if int(args.seed) != DEFAULT_SEED:
        raise RuntimeError(
            f"Formal V3.2 preparation requires frozen seed {DEFAULT_SEED}"
        )
    patient_folds_path = Path(args.patient_folds).resolve()
    patient_fold_receipt_path = Path(args.patient_fold_authority_receipt).resolve()
    patient_authority_audit = validate_frozen_v32_patient_fold_binding(
        patient_folds_path, patient_fold_receipt_path
    )
    hashes = json.loads(Path(args.hashes).read_text(encoding="utf-8"))
    if not isinstance(hashes, dict):
        raise RuntimeError("Formal authorization hashes must be a JSON mapping")
    graph_receipt_sha = str(args.graph_authority_receipt_sha256).lower()
    registered_graph_receipt = hashes.get("formal_graph_authority_receipt_sha256")
    if registered_graph_receipt is not None and registered_graph_receipt != graph_receipt_sha:
        raise RuntimeError("Training authorization and graph receipt SHA256 differ")
    hashes["formal_graph_authority_receipt_sha256"] = graph_receipt_sha
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Formal prepared output reuse is forbidden: {out.resolve()}")
    fold_manifest = pd.read_csv(
        patient_folds_path,
        sep="\t",
        dtype={"cancer_id": str, "sample_id": str, "patient_id": str},
    )
    from cc_hhgt.v32.formal_graph_authority import (
        load_formal_graph_input_authority,
        read_formal_graph_table,
    )

    graph_inputs = load_formal_graph_input_authority(
        receipt_path=args.graph_authority_receipt,
        receipt_sha256=graph_receipt_sha,
        patient_authority_audit=patient_authority_audit,
        static_inputs={
            "detection": (args.graph_detection, args.graph_detection_sha256),
            "signed_membership": (
                args.graph_signed_membership,
                args.graph_signed_membership_sha256,
            ),
            "pathway_hierarchy": (
                args.graph_pathway_hierarchy,
                args.graph_pathway_hierarchy_sha256,
            ),
            "lnc_protein_binding": (
                args.graph_lnc_protein_binding,
                args.graph_lnc_protein_binding_sha256,
            ),
            "protein_gene_encoding": (
                args.graph_protein_gene_encoding,
                args.graph_protein_gene_encoding_sha256,
            ),
            "ppi": (args.graph_ppi, args.graph_ppi_sha256),
        },
        fold_expression_pattern=args.graph_fold_expression_pattern,
        fold_coexpression_pattern=args.graph_fold_coexpression_pattern,
    )
    scope = load_scope(read_formal_graph_table(graph_inputs, "detection"))
    adjusted_mode = args.association_mode == "local-cnv-adjusted"
    if adjusted_mode:
        from cc_hhgt.v32.local_cnv_core_preparation import (
            adjusted_association_frame,
            sha256_file,
            validate_local_cnv_audit,
            validate_validation_a1_authority,
        )

        required_adjusted = {
            "formal_candidates": args.formal_candidates,
            "formal_candidates_sha256": args.formal_candidates_sha256,
            "local_cnv_audit_root": args.local_cnv_audit_root,
            "local_cnv_audit_success_sha256": args.local_cnv_audit_success_sha256,
            "local_cnv_validation_root": args.local_cnv_validation_root,
            "local_cnv_validation_success_sha256": args.local_cnv_validation_success_sha256,
        }
        if missing := sorted(key for key, value in required_adjusted.items() if not value):
            raise RuntimeError(f"Adjusted association mode lacks arguments: {missing}")
        candidate_path = Path(args.formal_candidates).resolve(strict=True)
        if sha256_file(candidate_path) != args.formal_candidates_sha256.lower():
            raise RuntimeError("Formal candidate authority SHA256 drift")
        candidates = pd.read_parquet(candidate_path)
        if (
            len(candidates) != 3_300_000
            or candidates.duplicated(KEYS).any()
            or candidates.cancer_id.astype(str).nunique() != 33
            or not candidates.groupby("cancer_id", observed=True).size().eq(100_000).all()
        ):
            raise RuntimeError("Formal candidate authority closure failed")
        local_cnv_authority = validate_local_cnv_audit(
            args.local_cnv_audit_root,
            expected_success_sha256=args.local_cnv_audit_success_sha256,
        )
        validation_a1_authority = validate_validation_a1_authority(
            args.local_cnv_validation_root,
            expected_success_sha256=args.local_cnv_validation_success_sha256,
        )
        if validation_a1_authority.get("candidate_sha256") != args.formal_candidates_sha256.lower():
            raise RuntimeError("Validation-A1 candidate authority differs from formal candidates")
        cancers = sorted(candidates.cancer_id.astype(str).unique())
        pathways = sorted(candidates.pathway_id.astype(str).unique())
    else:
        activity = load_activity(args.formal_activity_root)
        cancers = sorted(activity.cancer_id.unique())
        pathways = sorted(activity.pathway_id.unique())
    hierarchy = build_hierarchy(
        pathways, read_formal_graph_table(graph_inputs, "pathway_hierarchy")
    )
    if adjusted_mode:
        candidates = candidates.drop(columns=["pathway_family_id"], errors="ignore").merge(
            hierarchy, on="pathway_id", how="left", validate="many_to_one"
        )
    else:
        candidates = make_candidate_pairs(scope, pathways, budget=args.budget_per_cancer, seed=args.seed).merge(hierarchy, on="pathway_id", how="left", validate="many_to_one")
    out.mkdir(parents=True)
    candidates.to_parquet(out / "FORMAL_CANDIDATE_UNIVERSE.parquet", index=False)
    if adjusted_mode:
        expr_sample = fold_manifest[["cancer_id", "sample_id", "patient_id"]].drop_duplicates()
    else:
        expr_sample = activity[["cancer_id", "sample_id", "patient_id"]].drop_duplicates()
    from cc_hhgt.v32.patient_folds import assign_outer_split
    identity_columns = ["cancer_id", "sample_id", "patient_id"]
    observed_identity = expr_sample[identity_columns].sort_values(
        identity_columns, kind="stable"
    ).reset_index(drop=True)
    authority_identity = fold_manifest[identity_columns].sort_values(
        identity_columns, kind="stable"
    ).reset_index(drop=True)
    if not observed_identity.equals(authority_identity):
        raise RuntimeError(
            "Formal activity sample/patient identities differ from the frozen authority"
        )
    (out / "SAMPLE_PATIENT_FOLD_MAP.tsv").write_bytes(
        patient_folds_path.read_bytes()
    )
    (out / "PATIENT_FOLD_AUTHORITY_RECEIPT.json").write_bytes(
        patient_fold_receipt_path.read_bytes()
    )
    patient_binding = {
        "format": "CC_HHGT_V3_2_FORMAL_PREPARED_PATIENT_FOLD_BINDING_V1",
        "status": "PASS_FROZEN_V32_PATIENT_FIRST_AUTHORITY",
        "authority": patient_authority_audit,
        "sample_patient_fold_map": {
            "path": str(patient_folds_path),
            "sha256": patient_authority_audit["manifest_sha256"],
        },
        "authority_receipt": {
            "path": str(patient_fold_receipt_path),
            "sha256": patient_authority_audit["receipt_sha256"],
        },
        "sample_id_patient_fallback_used": False,
        "legacy_patient_fold_manifest_used": False,
    }
    (out / "PATIENT_FOLD_BINDING.json").write_text(
        json.dumps(patient_binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Each arm is a standalone prepared root.  Existing downstream consumers
    # can therefore be pointed explicitly at ``output/G0``, ``output/G1`` or
    # ``output/G2`` without teaching them to guess a graph variant.
    for variant in ("G0", "G1", "G2"):
        variant_root = out / variant
        variant_root.mkdir(parents=True)
        for filename in (
            "SAMPLE_PATIENT_FOLD_MAP.tsv",
            "PATIENT_FOLD_AUTHORITY_RECEIPT.json",
            "PATIENT_FOLD_BINDING.json",
        ):
            (variant_root / filename).write_bytes((out / filename).read_bytes())
        (variant_root / "FORMAL_GRAPH_VARIANT.json").write_text(
            json.dumps(
                {
                    "format": "CANCERLNCATLAS_V32_FORMAL_GRAPH_VARIANT_ROOT_V1",
                    "variant": variant,
                    "graph_authority_receipt_sha256": graph_inputs.receipt_sha256,
                    "legacy_root_fold_payloads_allowed": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    hierarchy.to_parquet(out / "PATHWAY_HIERARCHY.parquet", index=False)
    scope.to_parquet(out / "FORMAL_LNCRNA_SCOPE.parquet", index=False)
    if not adjusted_mode:
        expr_by = {
            c: load_expression_cancer(c, args.formal_expression_root) for c in cancers
        }
        activity_by = {c: activity.loc[activity.cancer_id.eq(c)].copy() for c in cancers}
    metrics = []
    for fold in range(args.start_fold, min(5, args.start_fold + args.fold_count)):
        print(json.dumps({"status": "PREPARING_FORMAL", "fold": fold, "candidates": len(candidates), "source_lnc": int(scope.lncrna_id.nunique()), "shared_lnc": int(scope.loc[scope.detected_cancers.ge(3), "lncrna_id"].nunique())}), flush=True)
        split_manifest = assign_outer_split(fold_manifest, fold, n_folds=5, validation_offset=1)
        if adjusted_mode:
            train_parts = []
            validation_parts = []
            for cancer in cancers:
                train_raw = pd.read_parquet(
                    Path(args.local_cnv_audit_root)
                    / "pair_level" / f"cancer={cancer}" / f"fold={fold}" / "part.parquet"
                )
                validation_raw = pd.read_parquet(
                    Path(args.local_cnv_validation_root)
                    / "validation_associations" / f"cancer={cancer}" / f"fold={fold}" / "part.parquet"
                )
                train_parts.append(
                    adjusted_association_frame(
                        train_raw, prefix="A1_local", expected_cancer=cancer,
                        expected_fold=fold,
                    )
                )
                validation_parts.append(
                    adjusted_association_frame(
                        validation_raw, prefix="A1_validation", expected_cancer=cancer,
                        expected_fold=fold,
                    )
                )
            metadata = candidates.merge(
                scope[["cancer_id", "lncrna_id", "detection_rate", "detected_cancers", "shared_or_local_scope"]],
                on=["cancer_id", "lncrna_id"], how="left", validate="many_to_one",
            )
            train_discovery = metadata.merge(
                pd.concat(train_parts, ignore_index=True), on=KEYS,
                how="left", validate="one_to_one",
            )
            validation_replication = metadata.merge(
                pd.concat(validation_parts, ignore_index=True), on=KEYS,
                how="left", validate="one_to_one",
            )
            if train_discovery.association_available.isna().any() or validation_replication.association_available.isna().any():
                raise RuntimeError("Adjusted association authority does not close over candidates")
        else:
            train_discovery = sampled_associations(expr_by, activity_by, candidates, split_manifest, "train", hierarchy, scope)
            validation_replication = sampled_associations(expr_by, activity_by, candidates, split_manifest, "validation", hierarchy, scope)
            train_discovery["association_available"] = True
            train_discovery["association_unavailable_reason"] = ""
            validation_replication["association_available"] = True
            validation_replication["association_unavailable_reason"] = ""
        frames = {
            "train": train_discovery,
            "validation": attach_replication_labels(train_discovery, validation_replication),
        }
        for split, frame in frames.items():
            frame["cross_cancer_support_frequency"] = frame.lncrna_id.map(scope.groupby("lncrna_id").detected_cancers.max()).fillna(0).to_numpy(float) / 33.0
            frame["cross_cancer_direction_consistency"] = 0.0; frame["cross_cancer_i2"] = 0.0
        model, baseline_metrics, x_train, x_val = _fast_l1(frames, seed=args.seed)
        train_logits = model.decision_function(x_train); val_logits = model.decision_function(x_val)
        from cc_hhgt.v32.training import PREPARED_FORMAT
        bound_graph = build_graph(
            graph_inputs,
            outer_fold=fold,
            split_manifest=split_manifest,
            candidate_pairs=candidates,
        )
        graph_binding_path = out / "GRAPH_AUTHORITIES" / f"PATIENT_FOLD_{fold}.json"
        graph_binding_path.parent.mkdir(parents=True, exist_ok=True)
        graph_binding_path.write_text(
            json.dumps(dict(bound_graph.binding), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        from cc_hhgt.v32.formal_graph import build_variant_runtime_bundle
        from cc_hhgt.v32.formal_graph_authority import bind_formal_graph_variant
        import torch
        for variant in ("G0", "G1", "G2"):
            bundle = build_variant_runtime_bundle(bound_graph.authority, variant)
            graph_binding = bind_formal_graph_variant(
                bound_graph.binding, bound_graph.authority, variant
            )
            payload = {
                "prepared_format": PREPARED_FORMAT,
                "patient_fold": fold, "bundle": bundle, "graph": None, "feature_dim": 4,
                "formal_graph_variant": variant,
                "formal_graph_authority": graph_binding,
                "conservation_context_features": 4,
                "legacy_model_config": {"training": {"hidden_channels": 96, "pair_hidden_channels": 96, "num_layers": 2, "num_heads": 2, "dropout": 0.20}, "residual_learning": {"enabled": False}},
                "train_batches": make_batches(frames["train"], train_logits, bundle, split="train"),
                "validation_batches": make_batches(frames["validation"], val_logits, bundle, split="validation"),
                "label_contract": {
                    "train_direction_label_source": "train_local_cnv_adjusted_effect" if adjusted_mode else "train_discovery_effect",
                    "validation_direction_label_source": "validation_local_cnv_adjusted_replication_effect" if adjusted_mode else "validation_replication_effect",
                    **({
                        "association_target": "A1_LOCAL",
                        "association_available_mask": "association_available",
                    } if adjusted_mode else {}),
                    "heldout_direction_is_model_input": False,
                    "test_labels_in_training_payload": False,
                    "test_logits_in_training_payload": False,
                    "test_metrics_computed_before_winner_lock": False,
                },
                "artifact_hashes": {
                    **hashes,
                    **({
                        "formal_candidate_authority_sha256": args.formal_candidates_sha256.lower(),
                        "local_cnv_audit_success_sha256": local_cnv_authority["success_sha256"],
                        "local_cnv_validation_success_sha256": validation_a1_authority["success_sha256"],
                    } if adjusted_mode else {}),
                }, "contains_optimizer_state": False, "contains_trained_parameters": False,
                "patient_fold_authority": patient_binding,
                "input_scope": {"source_lnc_n": int(scope.lncrna_id.nunique()), "shared_lnc_n": int(scope.loc[scope.detected_cancers.ge(3), "lncrna_id"].nunique()), "budget_per_cancer": args.budget_per_cancer, "candidate_rows_per_split": int(len(frames["train"])), "discovery_features_split": "train_patients_only", "replication_labels_splits": ["validation"], "held_out_effects_as_features": False, "sealed_test_accessed": False, "association_mode": args.association_mode, "unavailable_rows_loss_masked": adjusted_mode},
            }
            variant_root = out / variant
            torch.save(payload, variant_root / f"PATIENT_FOLD_{fold}.pt")
            fold_metrics = {"fold": fold, "variant": variant, **baseline_metrics, "rows": {s: int(len(f)) for s, f in frames.items()}, "graph_variant_edge_count": int(bound_graph.authority.manifest["variant_edge_counts"][variant])}
            metrics.append(fold_metrics)
            print(json.dumps({"status": "PREPARED_FORMAL", **fold_metrics}), flush=True)
    (out / "PREP_METRICS.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (out / "PREP_INPUT_SUMMARY.json").write_text(json.dumps({"source_lnc_n": int(scope.lncrna_id.nunique()), "shared_lnc_n": int(scope.loc[scope.detected_cancers.ge(3), "lncrna_id"].nunique()), "cancers": len(cancers), "pathways": len(pathways), "budget_per_cancer": args.budget_per_cancer, "candidate_rows": int(len(candidates)), "candidate_sampler": "pinned_formal_candidate_authority" if adjusted_mode else CANDIDATE_SAMPLER, "association_mode": args.association_mode, "patient_fold_authority": patient_authority_audit, "formal_graph_authority_receipt_sha256": graph_inputs.receipt_sha256, "graph_variants": ["G0", "G1", "G2"], "prepared_layout": "{G0,G1,G2}/PATIENT_FOLD_{fold}.pt", "legacy_root_fold_payloads_written": False, **({"local_cnv_audit": local_cnv_authority, "validation_a1_authority": validation_a1_authority} if adjusted_mode else {})}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
