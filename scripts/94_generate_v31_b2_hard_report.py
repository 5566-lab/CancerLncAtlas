#!/usr/bin/env python3
"""Generate the B2 HARD REPORT and the only authorization token for B3."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from cc_hhgt.common import file_sha256
from cc_hhgt.v30_integrity import atomic_write_bytes, atomic_write_json, merkle_sha256
from cc_hhgt.v31_b2_hard_report import build_b2_hard_report


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def _task_audits(run_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_root.glob("lambda_*/cc_hhgt/LOCO_*/seed_*/SUCCESS.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit = payload.get("post_task_audit", {})
        rows.append(
            {
                "loco_cancer": payload.get("test_cancer"),
                "seed": payload.get("seed"),
                "residual_shrinkage_lambda": payload.get(
                    "residual_shrinkage_lambda"
                ),
                "status": audit.get("status"),
                "residual_initialization_max_abs": audit.get(
                    "residual_initialization_max_abs"
                ),
                "success_json": str(path),
                "success_sha256": file_sha256(path),
            }
        )
    return pd.DataFrame(rows)


def _markdown(
    payload: dict,
    summary: pd.DataFrame,
    cancer: pd.DataFrame,
) -> str:
    lines = [
        "# B2 HARD REPORT",
        "",
        f"Decision: **{payload['decision']}**",
        "",
        f"Primary scope: **{payload['primary_scope']}**",
        "",
        f"BestSimple AUPRC: {payload['primary_best_simple_auprc']:.6f}",
        f"Selected B2 AUPRC: {payload['primary_selected_b2_auprc']:.6f}",
        (
            "Paired mean ΔAUPRC: "
            f"{payload['primary_mean_delta_auprc']:+.6f}"
        ),
        (
            "Cancer-cluster bootstrap 95% CI: "
            f"[{payload['cluster_aware_ci95_lower']:+.6f}, "
            f"{payload['cluster_aware_ci95_upper']:+.6f}]"
        ),
        (
            "Positive cancers: "
            f"{payload['positive_cancers']}/{payload['total_cancers']}"
        ),
        f"Prevalence: {payload['primary_prevalence']:.6f}",
        (
            "BestSimple AUPRC/prevalence: "
            f"{payload['primary_best_simple_auprc_over_prevalence']:.6f}"
        ),
        (
            "Selected B2 AUPRC/prevalence: "
            f"{payload['primary_selected_b2_auprc_over_prevalence']:.6f}"
        ),
        "",
        "| Secondary metric | BestSimple | Selected B2 | Paired delta |",
        "|---|---:|---:|---:|",
        (
            f"| AUROC | {payload['primary_best_simple_auroc']:.6f} | "
            f"{payload['primary_selected_b2_auroc']:.6f} | "
            f"{payload['primary_mean_delta_auroc']:+.6f} |"
        ),
        (
            f"| Brier | {payload['primary_best_simple_brier']:.6f} | "
            f"{payload['primary_selected_b2_brier']:.6f} | "
            f"{payload['primary_mean_delta_brier']:+.6f} |"
        ),
        (
            f"| ECE | {payload['primary_best_simple_ece']:.6f} | "
            f"{payload['primary_selected_b2_ece']:.6f} | "
            f"{payload['primary_mean_delta_ece']:+.6f} |"
        ),
        "",
        (
            "| Scope | BestSimple AUPRC | Selected B2 AUPRC | Paired ΔAUPRC | "
            "CI95 lower | CI95 upper | Positive cancers | Prevalence | "
            "B2 AUPRC/prevalence | ΔAUROC | ΔBrier | ΔECE |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.scope} | {row.best_simple_auprc:.6f} | "
            f"{row.selected_b2_auprc:.6f} | {row.mean_delta_auprc:+.6f} | "
            f"{row.cancer_cluster_bootstrap_ci95_lower:+.6f} | "
            f"{row.cancer_cluster_bootstrap_ci95_upper:+.6f} | "
            f"{int(row.positive_cancers)}/{int(row.total_cancers)} | "
            f"{row.prevalence:.6f} | "
            f"{row.selected_b2_auprc_over_prevalence:.6f} | "
            f"{row.mean_delta_auroc:+.6f} | "
            f"{row.mean_delta_brier:+.6f} | {row.mean_delta_ece:+.6f} |"
        )
    pathway_cancer = cancer.loc[cancer.scope.eq("Pathway")].sort_values(
        "loco_cancer"
    )
    lines.extend(
        [
            "",
            "## Pathway cancer-direction audit",
            "",
            (
                "| Cancer | BestSimple AUPRC | Selected B2 AUPRC | "
                "Paired ΔAUPRC | Direction | Prevalence | "
                "B2 AUPRC/prevalence | ΔAUROC | ΔBrier | ΔECE |"
            ),
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pathway_cancer.itertuples(index=False):
        direction = "positive" if row.mean_delta_auprc > 0 else "non-positive"
        lines.append(
            f"| {row.loco_cancer} | {row.best_simple_auprc:.6f} | "
            f"{row.selected_b2_auprc:.6f} | {row.mean_delta_auprc:+.6f} | "
            f"{direction} | {row.prevalence:.6f} | "
            f"{row.selected_b2_auprc_over_prevalence:.6f} | "
            f"{row.mean_delta_auroc:+.6f} | "
            f"{row.mean_delta_brier:+.6f} | {row.mean_delta_ece:+.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "Primary inference uses paired candidate rows and a cancer-cluster "
                "bootstrap. RNAss, DNAss, and ALL_TARGETS_MACRO are diagnostics and "
                "cannot compensate for an insufficient Pathway gain."
            ),
            "",
            "B3 is authorized only when every hard gate in B2_HARD_REPORT.json is true.",
            "No full-cancer training is authorized.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b2-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise RuntimeError("HARD REPORT requires a clean analysis worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    b2 = Path(args.b2_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    predictions = pd.read_parquet(b2 / "GRAPH_RESIDUAL_PREDICTIONS.parquet")
    admission = pd.read_csv(b2 / "GRAPH_RESIDUAL_ADMISSION.tsv", sep="\t")
    try:
        fallback = pd.read_csv(
            b2 / "GRAPH_RESIDUAL_FALLBACK_AUDIT.tsv", sep="\t"
        )
    except pd.errors.EmptyDataError:
        fallback = pd.DataFrame()
    gate_path = b2 / "GRAPH_RESIDUAL_GATE.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    task_audits = _task_audits(Path(args.run_root).resolve())
    metrics, cancer, summary, payload = build_b2_hard_report(
        predictions,
        admission,
        fallback,
        gate,
        task_audits,
        iterations=args.iterations,
        seed=args.seed,
    )

    paths = {
        "group_metrics": output / "B2_HARD_REPORT_GROUP_METRICS.tsv",
        "cancer_metrics": output / "B2_HARD_REPORT_CANCER_METRICS.tsv",
        "scope_summary": output / "B2_HARD_REPORT_SCOPE_SUMMARY.tsv",
        "task_audits": output / "B2_HARD_REPORT_TASK_AUDITS.tsv",
        "markdown": output / "B2_HARD_REPORT.md",
        "json": output / "B2_HARD_REPORT.json",
        "manifest": output / "B2_HARD_REPORT_SHA256.tsv",
    }
    for frame, key in (
        (metrics, "group_metrics"),
        (cancer, "cancer_metrics"),
        (summary, "scope_summary"),
        (task_audits, "task_audits"),
    ):
        _atomic_table(frame, paths[key])
    atomic_write_bytes(
        paths["markdown"], _markdown(payload, summary, cancer).encode("utf-8")
    )
    payload.update(
        {
            "analysis_git_commit": commit,
            "b2_gate_sha256": file_sha256(gate_path),
        }
    )
    atomic_write_json(paths["json"], payload)
    manifest_rows = [
        {
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for key, path in paths.items()
        if key not in {"manifest", "json"}
    ]
    manifest = pd.DataFrame(manifest_rows)
    _atomic_table(manifest, paths["manifest"])
    payload["manifest_sha256"] = file_sha256(paths["manifest"])
    payload["manifest_merkle_sha256"] = merkle_sha256(manifest_rows)
    atomic_write_json(paths["json"], payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
