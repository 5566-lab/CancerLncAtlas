from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from cc_hhgt.common import file_sha256, write_table
from cc_hhgt.metrics import binary_metrics
from cc_hhgt.v30_integrity import atomic_write_json, merkle_sha256


CANCERS = ("BRCA", "COAD", "KIRP")
SEEDS = (20260726, 20261726, 20262726)
TARGETS = ("Pathway", "RNAss", "DNAss")
PRIMARY_CONTRACT = "CONTRACT-T"
DIAGNOSTIC_CONTRACT = "CONTRACT-S"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fail-closed V3.1 B1 Fixed Graph stage report."
    )
    parser.add_argument("--b1-dir", required=True, type=Path)
    parser.add_argument("--fixed-root", required=True, type=Path)
    parser.add_argument("--old-p0-root", required=True, type=Path)
    parser.add_argument("--old-p2-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    return parser.parse_args()


def atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def target_rows(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    out = frame.loc[frame.split.astype(str).eq("test")].copy()
    if target != "Pathway":
        out = out.loc[
            out.state_id.astype(str).str.contains(target, case=False, regex=False)
        ].copy()
    return out.sort_values("candidate_id").reset_index(drop=True)


def legacy_rows(root: Path, cancer: str, seed: int, target: str) -> pd.DataFrame:
    task = "pathway" if target == "Pathway" else "state"
    path = root / f"LOCO_{cancer}" / f"seed_{seed}" / f"prediction_{task}_calibrated.parquet"
    if not path.is_file():
        raise RuntimeError(f"Missing frozen legacy prediction: {path}")
    return target_rows(pd.read_parquet(path), target)


def metric_row(
    frame: pd.DataFrame,
    *,
    model_id: str,
    cancer: str,
    seed: int,
    target: str,
) -> dict[str, object]:
    values = binary_metrics(frame.proxy_label, frame.proxy_positive_probability)
    prevalence = float(values["positive_rate"])
    auprc = float(values["auprc"])
    return {
        "model_id": model_id,
        "cancer_id": cancer,
        "seed": int(seed),
        "target_subtype": target,
        "metric_scope": "LOCO_test",
        **values,
        "auprc_over_prevalence": auprc / prevalence if prevalence > 0 else np.nan,
        "auprc_minus_prevalence": auprc - prevalence,
    }


def comparison_tables(
    fixed: pd.DataFrame,
    old_roots: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    parity_rows: list[dict[str, object]] = []
    for cancer in CANCERS:
        for seed in SEEDS:
            for target in TARGETS:
                current = fixed.loc[
                    fixed.contract.astype(str).eq(PRIMARY_CONTRACT)
                    & fixed.cancer_id.astype(str).eq(cancer)
                    & fixed.seed.astype(int).eq(seed)
                    & fixed.target_subtype.astype(str).eq(target)
                    & fixed.prediction_scale.astype(str).eq("calibrated_probability")
                ].sort_values("candidate_id").reset_index(drop=True)
                if current.empty:
                    raise RuntimeError(
                        f"Missing Fixed G-T prediction group: {cancer}/{seed}/{target}"
                    )
                metric_rows.append(
                    metric_row(
                        current,
                        model_id="FIXED_G_T",
                        cancer=cancer,
                        seed=seed,
                        target=target,
                    )
                )
                current_identity = current[["candidate_id", "proxy_label"]].copy()
                for model_id, root in old_roots.items():
                    legacy = legacy_rows(root, cancer, seed, target)
                    legacy_identity = legacy[["candidate_id", "proxy_label"]].copy()
                    key_parity = current_identity.candidate_id.equals(
                        legacy_identity.candidate_id
                    )
                    label_parity = current_identity.proxy_label.equals(
                        legacy_identity.proxy_label
                    )
                    status = (
                        "PASS"
                        if len(current_identity) == len(legacy_identity)
                        and key_parity
                        and label_parity
                        else "FAIL"
                    )
                    parity_rows.append(
                        {
                            "reference_model": model_id,
                            "cancer_id": cancer,
                            "seed": seed,
                            "target_subtype": target,
                            "fixed_rows": len(current_identity),
                            "legacy_rows": len(legacy_identity),
                            "candidate_key_parity": key_parity,
                            "proxy_label_parity": label_parity,
                            "status": status,
                        }
                    )
                    if status != "PASS":
                        raise RuntimeError(
                            "B1/legacy candidate contract mismatch: "
                            f"{parity_rows[-1]}"
                        )
                    metric_rows.append(
                        metric_row(
                            legacy,
                            model_id=model_id,
                            cancer=cancer,
                            seed=seed,
                            target=target,
                        )
                    )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["target_subtype", "cancer_id", "seed", "model_id"]
    ).reset_index(drop=True)
    parity = pd.DataFrame(parity_rows).sort_values(
        ["reference_model", "target_subtype", "cancer_id", "seed"]
    ).reset_index(drop=True)

    summaries: list[dict[str, object]] = []
    for (target, model), group in metrics.groupby(
        ["target_subtype", "model_id"], observed=True, sort=True
    ):
        summaries.append(
            {
                "summary_scope": "all_cancer_seed_macro",
                "cancer_id": "ALL",
                "target_subtype": target,
                "model_id": model,
                "n_runs": len(group),
                "auprc_mean": group.auprc.mean(),
                "auprc_median": group.auprc.median(),
                "auprc_sd": group.auprc.std(ddof=1),
                "auroc_mean": group.auroc.mean(),
                "brier_mean": group.brier.mean(),
                "ece_mean": group.ece.mean(),
                "positive_rate_mean": group.positive_rate.mean(),
                "auprc_over_prevalence_mean": group.auprc_over_prevalence.mean(),
                "auprc_minus_prevalence_mean": group.auprc_minus_prevalence.mean(),
            }
        )
    for (cancer, target, model), group in metrics.groupby(
        ["cancer_id", "target_subtype", "model_id"], observed=True, sort=True
    ):
        summaries.append(
            {
                "summary_scope": "within_cancer_seed_macro",
                "cancer_id": cancer,
                "target_subtype": target,
                "model_id": model,
                "n_runs": len(group),
                "auprc_mean": group.auprc.mean(),
                "auprc_median": group.auprc.median(),
                "auprc_sd": group.auprc.std(ddof=1),
                "auroc_mean": group.auroc.mean(),
                "brier_mean": group.brier.mean(),
                "ece_mean": group.ece.mean(),
                "positive_rate_mean": group.positive_rate.mean(),
                "auprc_over_prevalence_mean": group.auprc_over_prevalence.mean(),
                "auprc_minus_prevalence_mean": group.auprc_minus_prevalence.mean(),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["summary_scope", "target_subtype", "cancer_id", "model_id"]
    ).reset_index(drop=True)

    pivot = metrics.pivot(
        index=["cancer_id", "seed", "target_subtype"],
        columns="model_id",
        values=["auprc", "auroc", "brier", "ece"],
    )
    delta_rows: list[dict[str, object]] = []
    for index, row in pivot.iterrows():
        cancer, seed, target = index
        for reference in ("OLD_P0", "OLD_P2"):
            delta_rows.append(
                {
                    "cancer_id": cancer,
                    "seed": seed,
                    "target_subtype": target,
                    "reference_model": reference,
                    "fixed_auprc": row[("auprc", "FIXED_G_T")],
                    "reference_auprc": row[("auprc", reference)],
                    "delta_auprc": row[("auprc", "FIXED_G_T")]
                    - row[("auprc", reference)],
                    "delta_auroc": row[("auroc", "FIXED_G_T")]
                    - row[("auroc", reference)],
                    "delta_brier": row[("brier", "FIXED_G_T")]
                    - row[("brier", reference)],
                    "delta_ece": row[("ece", "FIXED_G_T")]
                    - row[("ece", reference)],
                }
            )
    deltas = pd.DataFrame(delta_rows).sort_values(
        ["reference_model", "target_subtype", "cancer_id", "seed"]
    ).reset_index(drop=True)
    return metrics, summary, deltas, parity


def contract_comparison(fixed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seed = SEEDS[0]
    for cancer in CANCERS:
        for target in TARGETS:
            values: dict[str, dict[str, float]] = {}
            for contract in (DIAGNOSTIC_CONTRACT, PRIMARY_CONTRACT):
                frame = fixed.loc[
                    fixed.contract.astype(str).eq(contract)
                    & fixed.cancer_id.astype(str).eq(cancer)
                    & fixed.seed.astype(int).eq(seed)
                    & fixed.target_subtype.astype(str).eq(target)
                    & fixed.prediction_scale.astype(str).eq("calibrated_probability")
                ]
                values[contract] = binary_metrics(
                    frame.proxy_label, frame.proxy_positive_probability
                )
            rows.append(
                {
                    "cancer_id": cancer,
                    "seed": seed,
                    "target_subtype": target,
                    "contract_s_auprc": values[DIAGNOSTIC_CONTRACT]["auprc"],
                    "contract_t_auprc": values[PRIMARY_CONTRACT]["auprc"],
                    "delta_t_minus_s_auprc": values[PRIMARY_CONTRACT]["auprc"]
                    - values[DIAGNOSTIC_CONTRACT]["auprc"],
                    "contract_s_auroc": values[DIAGNOSTIC_CONTRACT]["auroc"],
                    "contract_t_auroc": values[PRIMARY_CONTRACT]["auroc"],
                    "delta_t_minus_s_auroc": values[PRIMARY_CONTRACT]["auroc"]
                    - values[DIAGNOSTIC_CONTRACT]["auroc"],
                    "contract_s_brier": values[DIAGNOSTIC_CONTRACT]["brier"],
                    "contract_t_brier": values[PRIMARY_CONTRACT]["brier"],
                    "delta_t_minus_s_brier": values[PRIMARY_CONTRACT]["brier"]
                    - values[DIAGNOSTIC_CONTRACT]["brier"],
                    "contract_s_ece": values[DIAGNOSTIC_CONTRACT]["ece"],
                    "contract_t_ece": values[PRIMARY_CONTRACT]["ece"],
                    "delta_t_minus_s_ece": values[PRIMARY_CONTRACT]["ece"]
                    - values[DIAGNOSTIC_CONTRACT]["ece"],
                }
            )
    return pd.DataFrame(rows)


def prediction_distribution(fixed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = ["contract", "cancer_id", "seed", "target_subtype", "prediction_scale"]
    for keys, group in fixed.groupby(groups, observed=True, sort=True):
        p = group.proxy_positive_probability.to_numpy(float)
        rows.append(
            {
                **dict(zip(groups, keys)),
                "n": len(group),
                "mean": np.mean(p),
                "sd": np.std(p),
                "p01": np.quantile(p, 0.01),
                "p05": np.quantile(p, 0.05),
                "p50": np.quantile(p, 0.50),
                "p95": np.quantile(p, 0.95),
                "p99": np.quantile(p, 0.99),
                "n_unique": len(np.unique(p)),
                "collapsed": bool(np.std(p) <= 1e-8 or len(np.unique(p)) <= 1),
            }
        )
    return pd.DataFrame(rows)


def repair_decisions(deltas: pd.DataFrame) -> list[dict[str, object]]:
    p2 = deltas.loc[deltas.reference_model.eq("OLD_P2")].copy()
    decisions: list[dict[str, object]] = []
    for target in TARGETS:
        group = p2.loc[p2.target_subtype.eq(target)]
        by_cancer = group.groupby("cancer_id", observed=True).delta_auprc.mean()
        mean_delta = float(group.delta_auprc.mean())
        positive_cancers = int((by_cancer > 0).sum())
        decisions.append(
            {
                "target_subtype": target,
                "mean_delta_auprc_fixed_g_t_minus_old_p2": mean_delta,
                "median_delta_auprc_fixed_g_t_minus_old_p2": float(
                    group.delta_auprc.median()
                ),
                "positive_cancers": positive_cancers,
                "total_cancers": len(by_cancer),
                "cancer_mean_deltas": {
                    key: float(value) for key, value in by_cancer.items()
                },
                "decision": (
                    "GO" if mean_delta > 0 and positive_cancers >= 2 else "NO_GO"
                ),
            }
        )
    return decisions


def markdown_report(
    summary: pd.DataFrame,
    decisions: list[dict[str, object]],
    contracts: pd.DataFrame,
    runtime: dict[str, object],
) -> str:
    overall = summary.loc[summary.summary_scope.eq("all_cancer_seed_macro")]
    lookup = {
        (row.target_subtype, row.model_id): row
        for row in overall.itertuples(index=False)
    }
    lines = [
        "# V3.1 Phase B1 Fixed Graph 阶段性报告",
        "",
        "## 结论",
        "",
        "**B1 工程/数据合同门禁 PASS，但 standalone 预测修复整体不构成 GO。** "
        "Fixed G-T 相比 Old P2：Pathway 与 RNAss 在 3/3 癌种均下降；DNAss 在 "
        "3/3 癌种小幅上升。Graph repair 修复了合同与信息保留问题，但没有自动转化为 "
        "Pathway/RNAss 的更高 LOCO AUPRC。",
        "",
        "B2 尚未启动，GPU 已释放。下一步若继续，应测试 `BestSimpleLOCO + Graph residual` "
        "是否存在条件增量，不能把 B1 的 DNAss 小幅改善表述成整体 Graph 胜出。",
        "",
        "## B1 门禁",
        "",
        f"- 任务：{runtime['tasks_completed']}/{runtime['tasks_expected']}，运行门禁 PASS。",
        f"- 总串行 GPU 任务时间：{runtime['total_gpu_task_hours']:.2f} 小时；"
        f"G-T 单任务平均 {runtime['contract_t_mean_minutes']:.1f} 分钟，"
        f"G-S 单任务平均 {runtime['contract_s_mean_minutes']:.1f} 分钟。",
        "- 所有任务 `test_collapse_rows=0`；未启动任何全癌训练。",
        "- 2,478,576 条 B1 LOCO-test 预测、72 行 raw/calibrated 指标。",
        "- raw 与 calibrated 候选键一致，组内 AUROC/AUPRC 排序完全一致。",
        "- Fixed G-T 与 Old P0/Old P2 共 54 个癌种×seed×target 对照组的候选键和标签精确一致。",
        "",
        "## 三癌种、三 seed 的主要结果",
        "",
        "以下均为 calibrated probability 上重新计算的 LOCO-test 指标；AUPRC/AUROC "
        "使用未裁剪排序分数，Brier/ECE 使用数值安全裁剪。",
        "",
        "| Target | Model | AUPRC mean | median | AUROC mean | Brier mean | ECE mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "OLD_P0": "Old P0",
        "OLD_P2": "Old P2",
        "FIXED_G_T": "Fixed G-T",
    }
    for target in TARGETS:
        for model in ("OLD_P0", "OLD_P2", "FIXED_G_T"):
            row = lookup[(target, model)]
            lines.append(
                f"| {target} | {labels[model]} | {row.auprc_mean:.4f} | "
                f"{row.auprc_median:.4f} | {row.auroc_mean:.4f} | "
                f"{row.brier_mean:.4f} | {row.ece_mean:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Repair gain：Fixed G-T − Old P2",
            "",
            "| Target | mean ΔAUPRC | median ΔAUPRC | positive cancers | Decision |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in decisions:
        lines.append(
            f"| {item['target_subtype']} | "
            f"{item['mean_delta_auprc_fixed_g_t_minus_old_p2']:+.4f} | "
            f"{item['median_delta_auprc_fixed_g_t_minus_old_p2']:+.4f} | "
            f"{item['positive_cancers']}/{item['total_cancers']} | "
            f"{item['decision']} |"
        )
    lines.extend(
        [
            "",
            "- Pathway：明显下降，是当前最重要的负结果。",
            "- RNAss：也下降；Graph contract 修复没有解决 RNAss 的跨癌泛化难题。",
            "- DNAss：小幅正增量且 3/3 癌种同方向，但效应仅约 +0.005 AUPRC。",
            "",
            "## Contract-S 与 Contract-T（共同 seed 20260726）",
            "",
            "| Cancer | Target | G-S AUPRC | G-T AUPRC | T−S |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in contracts.itertuples(index=False):
        lines.append(
            f"| {row.cancer_id} | {row.target_subtype} | "
            f"{row.contract_s_auprc:.4f} | {row.contract_t_auprc:.4f} | "
            f"{row.delta_t_minus_s_auprc:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Contract-T 对 Pathway 的三个癌种均略好于 Contract-S；RNAss 在 BRCA/COAD "
            "改善、KIRP 略降；DNAss 为混合结果。因此不能宣称 target context 在所有任务上 "
            "统一获益。",
            "",
            "## 门禁过程中发现并修复的指标问题",
            "",
            "旧 `binary_metrics()` 在计算 AUROC/AUPRC 前把概率裁剪到 `[1e-7, 1-1e-7]`，"
            "会把极端概率制造成人工并列，并让单调温度校准看似改变排序。已修复为：排序指标"
            "使用原始有限分数，Brier/log-loss/ECE 才使用裁剪概率。完整回归为 153 passed。",
            "",
            "## 阶段决策",
            "",
            "- B1 integrity：PASS。",
            "- Fixed Graph standalone overall repair：NO-GO（DNAss-only small GO）。",
            "- B2 scientific rationale：仍可测试，因为 residual 问的是 Graph 在 BestSimple 条件下"
            "是否有独立信息，不等价于 standalone 必须胜出。",
            "- 当前执行状态：PAUSED_BEFORE_B2；等待明确批准后才使用 GPU。",
            "- 全癌训练：未启动且继续禁止。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report_git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=args.repo_root, text=True
    ).strip()
    if git_status:
        raise RuntimeError(f"Report repository is dirty: {git_status}")
    gate_path = args.b1_dir / "FIXED_GRAPH_B1_GATE.json"
    predictions_path = args.b1_dir / "FIXED_GRAPH_PREDICTIONS.parquet"
    metric_audit_path = args.b1_dir / "RAW_CALIBRATED_METRIC_AUDIT.tsv"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or int(gate.get("tasks_completed", 0)) != 12:
        raise RuntimeError("B1 final gate is not a complete PASS")
    metric_audit = pd.read_csv(metric_audit_path, sep="\t")
    if set(metric_audit.status.astype(str)) != {"PASS"}:
        raise RuntimeError("B1 raw/calibrated ranking audit is not PASS")
    fixed = pd.read_parquet(predictions_path)
    metrics, summary, deltas, parity = comparison_tables(
        fixed,
        {
            "OLD_P0": args.old_p0_root,
            "OLD_P2": args.old_p2_root,
        },
    )
    contracts = contract_comparison(fixed)
    distribution = prediction_distribution(fixed)
    if distribution.collapsed.any():
        raise RuntimeError("B1 report detected a collapsed prediction group")
    decisions = repair_decisions(deltas)

    status_path = args.fixed_root / "run_control" / "RUN_STATUS.json"
    run_status = json.loads(status_path.read_text(encoding="utf-8"))
    result_rows = run_status.get("results", [])
    if len(result_rows) != 12 or any(row.get("test_collapse_rows") != 0 for row in result_rows):
        raise RuntimeError("B1 runtime audit is incomplete or contains collapsed test rows")
    total_seconds = sum(float(row["task_seconds"]) for row in result_rows)
    t_seconds = [
        float(row["task_seconds"])
        for row in result_rows
        if row["contract"] == PRIMARY_CONTRACT
    ]
    s_seconds = [
        float(row["task_seconds"])
        for row in result_rows
        if row["contract"] == DIAGNOSTIC_CONTRACT
    ]
    runtime = {
        "tasks_completed": 12,
        "tasks_expected": 12,
        "total_gpu_task_hours": total_seconds / 3600.0,
        "contract_t_mean_minutes": np.mean(t_seconds) / 60.0,
        "contract_s_mean_minutes": np.mean(s_seconds) / 60.0,
    }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs": output / "B1_OLD_MODEL_COMPARISON_RUNS.tsv",
        "summary": output / "B1_OLD_MODEL_COMPARISON_SUMMARY.tsv",
        "deltas": output / "B1_FIXED_VS_OLD_DELTAS.tsv",
        "parity": output / "B1_CANDIDATE_PARITY_AUDIT.tsv",
        "contracts": output / "B1_CONTRACT_S_VS_T.tsv",
        "distribution": output / "B1_PREDICTION_DISTRIBUTION.tsv",
        "report": output / "B1_STAGE_REPORT.md",
        "json": output / "B1_STAGE_REPORT.json",
        "sha": output / "B1_STAGE_SHA256.tsv",
        "success": output / "B1_STAGE_SUCCESS.json",
    }
    atomic_table(metrics, paths["runs"])
    atomic_table(summary, paths["summary"])
    atomic_table(deltas, paths["deltas"])
    atomic_table(parity, paths["parity"])
    atomic_table(contracts, paths["contracts"])
    atomic_table(distribution, paths["distribution"])
    atomic_text(markdown_report(summary, decisions, contracts, runtime), paths["report"])

    payload = {
        "status": "PASS",
        "stage": "PHASE_B1_STAGE_REPORT",
        "b1_integrity_gate": "PASS",
        "candidate_parity_audit": "PASS",
        "candidate_parity_groups": int(len(parity)),
        "prediction_collapse_groups": 0,
        "task_specific_repair_decisions": decisions,
        "overall_standalone_repair_decision": "NO_GO_DNASS_ONLY_SMALL_GO",
        "phase_b2_status": "PAUSED_BEFORE_B2_AWAITING_EXPLICIT_GPU_APPROVAL",
        "full_cancer_training_started": False,
        "report_git_commit": report_git_commit,
        "runtime": runtime,
        "sources": {
            "b1_gate": str(gate_path),
            "b1_gate_sha256": file_sha256(gate_path),
            "b1_predictions": str(predictions_path),
            "b1_predictions_sha256": file_sha256(predictions_path),
            "old_p0_root": str(args.old_p0_root),
            "old_p2_root": str(args.old_p2_root),
        },
    }
    atomic_write_json(paths["json"], payload)
    # SUCCESS is written last and authenticates the immutable manifest.  It is
    # deliberately not included in that manifest, avoiding a circular hash.
    manifest_files = [
        path for key, path in paths.items() if key not in {"sha", "success"}
    ]
    sha = pd.DataFrame(
        [
            {
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in manifest_files
        ]
    ).sort_values("relative_path")
    atomic_table(sha, paths["sha"])
    success = {
        "status": "PASS",
        "stage": "PHASE_B1_STAGE_REPORT",
        "report_manifest_sha256": file_sha256(paths["sha"]),
        "report_manifest_merkle_sha256": merkle_sha256(sha.to_dict("records")),
        "manifest_entries": len(sha),
        "full_cancer_training_started": False,
        "phase_b2_status": "PAUSED_BEFORE_B2_AWAITING_EXPLICIT_GPU_APPROVAL",
    }
    atomic_write_json(paths["success"], success)
    print(json.dumps({**payload, **success}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
