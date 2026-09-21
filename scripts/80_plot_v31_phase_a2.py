#!/usr/bin/env python3
"""Generate the two required Graph Repair figures from frozen audit tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cc_hhgt.graph_contract import build_fold_graph, materialize_relation_provenance


PILOT = ("BRCA", "COAD", "KIRP")


def save_figure(figure, root: Path, stem: str) -> None:
    figure.savefig(root / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2-root", required=True)
    parser.add_argument("--prior-audit-root", required=True)
    parser.add_argument("--graph-edge", required=True)
    parser.add_argument("--fold-manifest", required=True)
    args = parser.parse_args()

    a2 = Path(args.a2_root).resolve()
    prior = Path(args.prior_audit_root).resolve()
    output = a2 / "FIGURES"
    output.mkdir(parents=True, exist_ok=True)
    edges = materialize_relation_provenance(pd.read_parquet(args.graph_edge))
    folds = pd.read_csv(args.fold_manifest, sep="\t")
    old = pd.read_csv(
        prior / "PHASE_A/CC_HHGT_TRAIN_TEST_CONTRACT_AUDIT.tsv", sep="\t"
    )

    contract_rows = []
    for cancer in PILOT:
        fold = folds.loc[folds.test_cancer.astype(str).eq(cancer)].iloc[0]
        old_rows = old.loc[
            old.test_cancer.astype(str).eq(cancer)
            & old.is_dynamic_context.astype(bool)
            & old.relation_family.astype(str).ne("__CANDIDATE_CONTEXT_SUMMARY__")
        ]
        old_train = float(old_rows.train_cancer_mean_visible_edges.sum())
        old_real = float(old_rows.heldout_test_visible_edges.sum())
        raw = edges.loc[
            edges.cancer_id.astype("string").eq(cancer)
            & edges.relation_provenance.astype(str).ne("global_static")
        ]
        repaired = build_fold_graph(
            edges,
            heldout_cancers={cancer, str(fold.validation_cancer)},
            mode="real_test",
            contract="CONTRACT-T",
            reference_only=set(),
            compute_fingerprint=False,
        ).edges
        repaired_visible = repaired.loc[
            repaired.cancer_id.astype("string").eq(cancer)
            & repaired.relation_provenance.astype(str).ne("global_static")
        ]
        contract_rows.extend(
            [
                {"cancer": cancer, "version": "Old P2", "stage": "train", "edges": old_train},
                {"cancer": cancer, "version": "Old P2", "stage": "pseudoheldout", "edges": old_train},
                {"cancer": cancer, "version": "Old P2", "stage": "real heldout", "edges": old_real},
                {"cancer": cancer, "version": "A2 Contract-T", "stage": "train", "edges": float(len(raw))},
                {"cancer": cancer, "version": "A2 Contract-T", "stage": "pseudoheldout", "edges": float(len(repaired_visible))},
                {"cancer": cancer, "version": "A2 Contract-T", "stage": "real heldout", "edges": float(len(repaired_visible))},
            ]
        )
    contract = pd.DataFrame(contract_rows)
    contract.to_csv(output / "FIGURE_1_GRAPH_CONTRACT_REPAIR_DATA.tsv", sep="\t", index=False)
    summary = contract.groupby(["version", "stage"], observed=True).edges.median().unstack()
    stages = ["train", "pseudoheldout", "real heldout"]
    x = np.arange(len(stages))
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for offset, (version, color) in zip((-0.18, 0.18), (("Old P2", "#B85C5C"), ("A2 Contract-T", "#2878B5")), strict=True):
        values = [float(summary.loc[version, stage]) for stage in stages]
        axis.bar(x + offset, values, width=0.34, label=version, color=color)
    axis.set_yscale("log")
    axis.set_xticks(x, ["Training cancer", "Pseudo-heldout c*", "Real held-out"])
    axis.set_ylabel("Median cancer-sensitive edges (log scale)")
    axis.set_title("Graph contract before and after repair")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    save_figure(figure, output, "FIGURE_1_GRAPH_CONTRACT_REPAIR")

    truncation = pd.read_csv(
        prior / "PHASE_A/CC_HHGT_TRUNCATION_AUDIT.tsv", sep="\t"
    )
    retention = pd.read_csv(a2 / "AUDITS/EDGE_SIGNAL_RETENTION_AUDIT.tsv", sep="\t")
    families = {
        "Coexpression": "lncRNA__coexpressed_with__gene",
        "lncRNA-protein": "lncRNA__binds_protein__protein",
        "PPI": "protein__physical_interaction__protein",
        "Gene-pathway": "gene__member_of__pathway",
    }
    retention_rows = []
    for label, family in families.items():
        old_hit = truncation.loc[truncation.truncation_name.astype(str).eq(family)]
        old_value = (
            float((1 - old_hit.fraction_signal_mass_removed.astype(float)).median())
            if len(old_hit)
            else 1.0
        )
        new_hit = retention.loc[
            retention.relation_family.astype(str).eq(family)
            & retention.contract.astype(str).eq("CONTRACT-T")
        ]
        new_value = float(new_hit.weighted_signal_retention.astype(float).min())
        retention_rows.extend(
            [
                {"relation": label, "strategy": "Old 300k cap", "weighted_signal_retention": old_value},
                {"relation": label, "strategy": "A2 runtime cycle", "weighted_signal_retention": new_value},
            ]
        )
    retention_plot = pd.DataFrame(retention_rows)
    retention_plot.to_csv(output / "FIGURE_2_EDGE_SIGNAL_RETENTION_DATA.tsv", sep="\t", index=False)
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    labels = list(families)
    old_values = retention_plot.loc[retention_plot.strategy.eq("Old 300k cap")].weighted_signal_retention.to_numpy()
    new_values = retention_plot.loc[retention_plot.strategy.eq("A2 runtime cycle")].weighted_signal_retention.to_numpy()
    x = np.arange(len(labels))
    axis.bar(x - 0.18, old_values, width=0.34, label="Old 300k cap", color="#B85C5C")
    axis.bar(x + 0.18, new_values, width=0.34, label="A2 runtime cycle", color="#2A9D8F")
    axis.axhline(0.90, color="#333333", linestyle="--", linewidth=1, label="90% target")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Weighted signal retained")
    axis.set_title("Edge signal retention: destructive cap vs runtime coverage cycle")
    axis.legend(frameon=False, ncol=3, fontsize=9)
    axis.grid(axis="y", alpha=0.2)
    save_figure(figure, output, "FIGURE_2_EDGE_SIGNAL_RETENTION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
