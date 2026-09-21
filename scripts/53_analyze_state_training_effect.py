#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


COLORS = {
    "navy": "#16324F",
    "blue": "#2F6690",
    "teal": "#3A9D8F",
    "orange": "#E07A5F",
    "gold": "#E9C46A",
    "gray": "#7A8491",
    "light": "#EAF0F5",
    "red": "#C44536",
}


def configure_plotting() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = family
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(score)
    return float(roc_auc_score(y[valid], score[valid])) if np.unique(y[valid]).size > 1 else float("nan")


def binary_auprc(y: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(score)
    return float(average_precision_score(y[valid], score[valid])) if y[valid].sum() > 0 else float("nan")


def collect_old_loco(strict_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in strict_root.glob("*/LOCO_*/seed_*/metrics.tsv"):
        frame = pd.read_csv(path, sep="\t")
        for split in ("train", "test"):
            hit = frame.loc[frame.task.eq("state") & frame.split.eq(split)]
            if hit.empty:
                continue
            row = hit.iloc[0].to_dict()
            row["metric_path"] = str(path)
            rows.append(row)
    return pd.DataFrame(rows)


def collect_patient_fold(pf_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict] = []
    state_rows: list[dict] = []
    for path in pf_root.rglob("prediction.parquet"):
        frame = pd.read_parquet(path)
        probability_columns = [
            column
            for column in frame.columns
            if column.startswith("state_")
            and column.endswith("_probability")
            and column != "state_direction_probability"
        ]
        y = frame.test_membership_label.to_numpy(float)
        model = frame[probability_columns].mean(axis=1, skipna=True).to_numpy(float)
        patient = frame.state_patient_probability.to_numpy(float)
        baseline = frame.train_effect.abs().to_numpy(float)
        valid = np.isfinite(y) & np.isfinite(model)
        meta = {
            "cancer_id": str(frame.cancer_id.iloc[0]),
            "patient_fold_id": str(frame.patient_fold_id.iloc[0]),
            "seed": int(frame.seed.iloc[0]),
            "n": int(valid.sum()),
            "n_positive": int(y[valid].sum()),
        }
        macro_values: list[float] = []
        for state_id, group in frame.loc[valid].groupby("state_id"):
            state_y = group.test_membership_label.to_numpy(float)
            state_model = group[probability_columns].mean(axis=1, skipna=True).to_numpy(float)
            state_baseline = group.train_effect.abs().to_numpy(float)
            model_auc = binary_auc(state_y, state_model)
            macro_values.append(model_auc)
            state_rows.append(
                {
                    **meta,
                    "state_id": str(state_id),
                    "n_state": len(group),
                    "n_positive_state": int(state_y.sum()),
                    "model_auroc": model_auc,
                    "baseline_auroc": binary_auc(state_y, state_baseline),
                }
            )
        finite_macro = [value for value in macro_values if np.isfinite(value)]
        run_rows.append(
            {
                **meta,
                "model_auroc": binary_auc(y, model),
                "model_auprc": binary_auprc(y, model),
                "patient_auroc": binary_auc(y, patient),
                "baseline_auroc": binary_auc(y, baseline),
                "macro_state_model_auroc": float(np.mean(finite_macro)) if finite_macro else float("nan"),
            }
        )
    return pd.DataFrame(run_rows), pd.DataFrame(state_rows)


def collect_minimal_runs(export_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = {"minimal_loco": 80, "minimal_loco_140": 140, "minimal_loco_250": 250}
    rows: list[dict] = []
    histories: list[pd.DataFrame] = []
    for folder, limit in labels.items():
        model_dir = export_dir / folder / "rgcn" / "LOCO_BLCA" / "seed_20260726"
        history = pd.read_csv(model_dir / "training_history.tsv", sep="\t")
        metrics = pd.read_csv(model_dir / "metrics.tsv", sep="\t")
        success = json.loads((model_dir / "SUCCESS.json").read_text(encoding="utf-8"))
        history["configured_epoch_limit"] = limit
        histories.append(history)
        for task in ("pathway", "state"):
            for split in ("val", "test"):
                metric = metrics.loc[metrics.task.eq(task) & metrics.split.eq(split)].iloc[0]
                rows.append(
                    {
                        "configured_epoch_limit": limit,
                        "epochs_run": len(history),
                        "best_epoch": int(success["best_epoch"]),
                        "task": task,
                        "split": split,
                        "n": int(metric["n"]),
                        "positive_rate": float(metric["positive_rate"]),
                        "auroc": float(metric["auroc"]),
                        "auprc": float(metric["auprc"]),
                    }
                )
    return pd.DataFrame(rows), pd.concat(histories, ignore_index=True)


def collect_per_state(old_prediction: Path, fixed_prediction: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for version, path in [("旧版：全正训练", old_prediction), ("修复：PU 1:4", fixed_prediction)]:
        frame = pd.read_parquet(path)
        frame = frame.loc[frame.split.astype(str).eq("test")]
        for state_id, group in frame.groupby("state_id"):
            y = group.proxy_label.to_numpy(float)
            probability = group.raw_probability.to_numpy(float)
            rows.append(
                {
                    "version": version,
                    "state_id": str(state_id),
                    "n": len(group),
                    "n_positive": int(y.sum()),
                    "positive_rate": float(y.mean()),
                    "auroc": binary_auc(y, probability),
                    "auprc": binary_auprc(y, probability),
                }
            )
    return pd.DataFrame(rows)


def save_sampling_plot(export_dir: Path) -> None:
    names = ["旧 sampler\n生产折", "修复 sampler\n生产 cap", "本次最小实验\n20k cap"]
    positive = np.asarray([182_669, 30_000, 4_000])
    unlabeled = np.asarray([0, 120_000, 16_000])
    totals = positive + unlabeled
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})
    axes[0].bar(x, positive / totals, color=COLORS["orange"], label="positive")
    axes[0].bar(x, unlabeled / totals, bottom=positive / totals, color=COLORS["teal"], label="unlabeled")
    for index, total in enumerate(totals):
        axes[0].text(index, 1.04, f"P={positive[index]:,}\nU={unlabeled[index]:,}", ha="center", va="bottom", fontsize=10)
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 1.23)
    axes[0].set_ylabel("训练样本构成")
    axes[0].set_title("state 训练集：旧逻辑被正样本挤满")
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].axis("off")
    boxes = [
        (0.05, 0.72, 0.90, 0.18, "旧逻辑：先保留全部 positive", COLORS["orange"]),
        (0.05, 0.42, 0.90, 0.18, "positive 数 > cap → U 配额 = 0", COLORS["red"]),
        (0.05, 0.12, 0.90, 0.18, "nnPU 无 negative risk → 退化成只推高分数", COLORS["navy"]),
    ]
    for x0, y0, width, height, text, color in boxes:
        axes[1].add_patch(plt.Rectangle((x0, y0), width, height, color=color, alpha=0.95, transform=axes[1].transAxes))
        axes[1].text(x0 + width / 2, y0 + height / 2, text, color="white", ha="center", va="center", fontsize=12, transform=axes[1].transAxes)
    axes[1].annotate("", xy=(0.5, 0.60), xytext=(0.5, 0.72), arrowprops={"arrowstyle": "->", "lw": 2}, xycoords="axes fraction")
    axes[1].annotate("", xy=(0.5, 0.30), xytext=(0.5, 0.42), arrowprops={"arrowstyle": "->", "lw": 2}, xycoords="axes fraction")
    fig.tight_layout()
    fig.savefig(export_dir / "01_sampling_bug_and_fix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_global_plot(old_loco: pd.DataFrame, export_dir: Path) -> None:
    test = old_loco.loc[old_loco.split.eq("test")]
    models = ["rgcn", "hgt", "cc_hhgt_strict"]
    labels = ["RGCN", "HGT", "CC-HHGT"]
    arrays = [test.loc[test.model_name.eq(model), "auroc"].dropna().to_numpy() for model in models]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.25, 0.75]})
    box = axes[0].boxplot(arrays, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], [COLORS["blue"], COLORS["teal"], COLORS["orange"]], strict=True):
        patch.set_facecolor(color); patch.set_alpha(0.72)
    rng = np.random.default_rng(7)
    for index, values in enumerate(arrays, start=1):
        axes[0].scatter(rng.normal(index, 0.055, len(values)), values, s=10, alpha=0.32, color=COLORS["navy"])
    axes[0].axhline(0.5, color=COLORS["red"], linestyle="--", linewidth=1.5, label="随机水平 0.5")
    axes[0].set_ylim(0.38, 0.64)
    axes[0].set_ylabel("LOCO test AUROC")
    axes[0].set_title("旧 strict_release：跨癌种 state 指标")
    axes[0].legend(frameon=False)
    axes[1].axis("off")
    median = test.auroc.median(); mean = test.auroc.mean(); std = test.auroc.std()
    lines = [
        ("297", "test runs\n3模型 × 33癌种 × 3 seed"),
        ("297/297", "训练 positive_rate = 1.0"),
        (f"{median:.3f}", f"median AUROC\nmean {mean:.3f} ± {std:.3f}"),
        (f"{(test.auroc.ge(.6).mean()*100):.1f}%", "AUROC ≥ 0.6"),
    ]
    for index, (value, label) in enumerate(lines):
        y = 0.87 - index * 0.23
        axes[1].text(0.08, y, value, fontsize=23, fontweight="bold", color=COLORS["navy"], transform=axes[1].transAxes)
        axes[1].text(0.40, y + 0.005, label, fontsize=11, color=COLORS["gray"], va="center", transform=axes[1].transAxes)
    fig.tight_layout()
    fig.savefig(export_dir / "02_old_loco_global_audit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history: pd.DataFrame, success: dict, export_dir: Path) -> None:
    best_epoch = int(success["best_epoch"])
    stop_epoch = int(history.epoch.max())
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    axes[0, 0].plot(history.epoch, history.path_loss, label="pathway loss", color=COLORS["blue"])
    axes[0, 0].plot(history.epoch, history.state_loss, label="state loss", color=COLORS["orange"])
    axes[0, 0].plot(history.epoch, history.loss, label="joint loss", color=COLORS["navy"], alpha=0.75)
    axes[0, 0].set_title("训练损失")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(history.epoch, history.pathway_auroc, label="pathway", color=COLORS["blue"])
    axes[0, 1].plot(history.epoch, history.state_auroc, label="state", color=COLORS["orange"])
    axes[0, 1].axhline(0.5, color=COLORS["gray"], linestyle="--")
    axes[0, 1].set_title("验证 AUROC")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(history.epoch, history.pathway_auprc, label="pathway AUPRC", color=COLORS["blue"])
    axes[1, 0].plot(history.epoch, history.state_auprc, label="state AUPRC", color=COLORS["orange"])
    axes[1, 0].plot(history.epoch, history.joint_validation_score, label="joint checkpoint score", color=COLORS["teal"], linewidth=2)
    axes[1, 0].axvline(best_epoch, color=COLORS["red"], linestyle="--", label=f"best={best_epoch}")
    axes[1, 0].set_title("验证 AUPRC 与 checkpoint 分数")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 1].set_xlim(0, 250)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].hlines(0.55, 0, 250, color=COLORS["light"], linewidth=12)
    axes[1, 1].hlines(0.55, 0, stop_epoch, color=COLORS["teal"], linewidth=12)
    axes[1, 1].scatter([best_epoch, stop_epoch, 250], [0.55, 0.55, 0.55], s=120, color=[COLORS["red"], COLORS["teal"], COLORS["gray"]], zorder=3)
    axes[1, 1].text(best_epoch, 0.68, f"最佳 {best_epoch}", ha="center", color=COLORS["red"], fontweight="bold")
    axes[1, 1].text(stop_epoch, 0.36, f"早停 {stop_epoch}", ha="center", color=COLORS["teal"], fontweight="bold")
    axes[1, 1].text(248, 0.68, "上限 250", ha="right", color=COLORS["gray"])
    axes[1, 1].text(0.02, 0.10, f"best 后连续 {stop_epoch-best_epoch} epoch 无提升\n→ 250 上限对本实验足够", transform=axes[1, 1].transAxes, fontsize=12, color=COLORS["navy"])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_title("训练步数充分性")
    for ax in axes.flat:
        if ax is not axes[1, 1]:
            ax.axvline(best_epoch, color=COLORS["red"], linestyle=":", alpha=0.7)
            ax.set_xlabel("epoch")
    fig.suptitle("修复后最小 LOCO 实验：RGCN / BLCA / seed 20260726 / state 20k", fontsize=16, fontweight="bold", color=COLORS["navy"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(export_dir / "03_minimal_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def short_state(name: str) -> str:
    return name.split("::")[-1]


def save_per_state_plot(per_state: pd.DataFrame, old_auc: float, fixed_auc: float, export_dir: Path) -> None:
    pivot = per_state.pivot(index="state_id", columns="version", values="auroc")
    states = list(pivot.index)
    x = np.arange(len(states))
    width = 0.36
    old_label = "旧版：全正训练"; fixed_label = "修复：PU 1:4"
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5), gridspec_kw={"width_ratios": [1.4, 0.6]})
    axes[0].bar(x - width / 2, pivot[old_label], width, label=old_label, color=COLORS["gray"])
    axes[0].bar(x + width / 2, pivot[fixed_label], width, label=fixed_label, color=COLORS["teal"])
    axes[0].axhline(0.5, color=COLORS["red"], linestyle="--")
    axes[0].set_xticks(x, [short_state(state) for state in states], rotation=28, ha="right")
    axes[0].set_ylim(0.40, 0.64)
    axes[0].set_ylabel("BLCA test AUROC（state 内）")
    axes[0].set_title("EXTEND 与 6 个 stemness state 均使用同一修复")
    axes[0].legend(frameon=False, fontsize=9)
    macro_old = float(pivot[old_label].mean()); macro_fixed = float(pivot[fixed_label].mean())
    labels = ["pooled\n跨 state", "macro\nstate 内均值"]
    axes[1].bar(np.arange(2) - width / 2, [old_auc, macro_old], width, color=COLORS["gray"], label="旧版")
    axes[1].bar(np.arange(2) + width / 2, [fixed_auc, macro_fixed], width, color=COLORS["teal"], label="修复")
    axes[1].axhline(0.5, color=COLORS["red"], linestyle="--")
    axes[1].set_xticks(np.arange(2), labels)
    axes[1].set_ylim(0.40, 0.72)
    axes[1].set_title("汇总方式会改变数值")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.3f", fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(export_dir / "04_stemness_extend_old_vs_fixed.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pf_plot(pf_runs: pd.DataFrame, pf_states: pd.DataFrame, export_dir: Path) -> None:
    valid = pf_runs.dropna(subset=["model_auroc", "baseline_auroc"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    columns = ["model_auroc", "patient_auroc", "baseline_auroc", "macro_state_model_auroc"]
    labels = ["4-head\nensemble", "patient\nhead", "abs(train_effect)\nbaseline", "state内\nmacro"]
    arrays = [valid[column].dropna().to_numpy() for column in columns]
    box = axes[0].boxplot(arrays, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], [COLORS["orange"], COLORS["blue"], COLORS["teal"], COLORS["gold"]], strict=True):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    axes[0].axhline(0.5, color=COLORS["red"], linestyle="--")
    axes[0].set_ylim(0.25, 0.92)
    axes[0].set_ylabel("PF test AUROC")
    axes[0].set_title("肿瘤内 held-out 患者子集：495 runs")
    axes[1].scatter(valid.baseline_auroc, valid.model_auroc, s=16, alpha=0.48, color=COLORS["blue"])
    axes[1].plot([0, 1], [0, 1], linestyle="--", color=COLORS["red"])
    axes[1].set_xlim(0.15, 0.95); axes[1].set_ylim(0.15, 0.95)
    axes[1].set_xlabel("abs(train_effect) baseline AUROC")
    axes[1].set_ylabel("4-head ensemble AUROC")
    axes[1].set_title("多数点未超过效应量基线")
    delta = valid.model_auroc - valid.baseline_auroc
    axes[1].text(0.04, 0.92, f"median Δ = {delta.median():+.3f}\nmean Δ = {delta.mean():+.3f}", transform=axes[1].transAxes, va="top", fontsize=12, color=COLORS["navy"])
    fig.tight_layout()
    fig.savefig(export_dir / "05_patient_fold_model_vs_baseline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--export-dir", default=r"D:\model\exports\state_training_effect_fixed")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    export_dir = Path(args.export_dir).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    result_root = project / "results/model/cc_hhgt_v2_9_state_graph"
    strict_root = result_root / "v2_9_strict"
    pf_root = result_root / "v2_9_downstream/lncrna_state_experts"
    minimal_dir = export_dir / "minimal_loco_250/rgcn/LOCO_BLCA/seed_20260726"
    old_dir = strict_root / "rgcn/LOCO_BLCA/seed_20260726"

    old_loco = collect_old_loco(strict_root)
    pf_runs, pf_states = collect_patient_fold(pf_root)
    minimal_runs, histories = collect_minimal_runs(export_dir)
    per_state = collect_per_state(old_dir / "prediction_state_raw.parquet", minimal_dir / "prediction_state_raw.parquet")

    old_loco.to_csv(export_dir / "old_loco_state_metrics.tsv", sep="\t", index=False)
    pf_runs.to_csv(export_dir / "patient_fold_metrics_recomputed.tsv", sep="\t", index=False)
    pf_states.to_csv(export_dir / "patient_fold_per_state_metrics.tsv", sep="\t", index=False)
    minimal_runs.to_csv(export_dir / "minimal_loco_epoch_experiments.tsv", sep="\t", index=False)
    histories.to_csv(export_dir / "minimal_loco_histories.tsv", sep="\t", index=False)
    per_state.to_csv(export_dir / "per_state_old_vs_fixed.tsv", sep="\t", index=False)

    old_test = old_loco.loc[old_loco.split.eq("test")]
    fixed_metric = minimal_runs.loc[
        minimal_runs.configured_epoch_limit.eq(250) & minimal_runs.task.eq("state") & minimal_runs.split.eq("test")
    ].iloc[0]
    old_metric = pd.read_csv(old_dir / "metrics.tsv", sep="\t").query("task == 'state' and split == 'test'").iloc[0]
    history_250 = pd.read_csv(minimal_dir / "training_history.tsv", sep="\t")
    success_250 = json.loads((minimal_dir / "SUCCESS.json").read_text(encoding="utf-8"))
    valid_pf = pf_runs.dropna(subset=["model_auroc", "baseline_auroc"])
    summary = {
        "old_loco": {
            "runs": int(len(old_test)),
            "all_positive_training_runs": int(old_loco.loc[old_loco.split.eq("train"), "positive_rate"].eq(1.0).sum()),
            "median_auroc": float(old_test.auroc.median()),
            "mean_auroc": float(old_test.auroc.mean()),
            "std_auroc": float(old_test.auroc.std()),
            "fraction_ge_0_6": float(old_test.auroc.ge(0.6).mean()),
        },
        "minimal_fixed_blca": {
            "old_test_auroc": float(old_metric.auroc),
            "fixed_test_auroc": float(fixed_metric.auroc),
            "fixed_test_auprc": float(fixed_metric.auprc),
            "fixed_val_auroc": float(minimal_runs.query("configured_epoch_limit == 250 and task == 'state' and split == 'val'").iloc[0].auroc),
            "best_epoch": int(success_250["best_epoch"]),
            "stop_epoch": int(history_250.epoch.max()),
            "epoch_limit": 250,
            "state_training_rows": int(success_250["state_training_rows"]),
            "state_training_positive": int(success_250["state_training_positive"]),
            "state_training_unlabeled": int(success_250["state_training_unlabeled"]),
            "macro_state_old_auroc": float(per_state.query("version == '旧版：全正训练'").auroc.mean()),
            "macro_state_fixed_auroc": float(per_state.query("version == '修复：PU 1:4'").auroc.mean()),
        },
        "patient_fold": {
            "runs": int(len(pf_runs)),
            "valid_runs": int(pf_runs.model_auroc.notna().sum()),
            "model_median_auroc": float(pf_runs.model_auroc.median()),
            "model_mean_auroc": float(pf_runs.model_auroc.mean()),
            "model_std_auroc": float(pf_runs.model_auroc.std()),
            "model_fraction_ge_0_6": float(pf_runs.model_auroc.ge(0.6).sum() / pf_runs.model_auroc.notna().sum()),
            "baseline_median_auroc": float(pf_runs.baseline_auroc.median()),
            "patient_head_median_auroc": float(pf_runs.patient_auroc.median()),
            "macro_state_median_auroc": float(pf_runs.macro_state_model_auroc.median()),
            "median_model_minus_baseline": float((valid_pf.model_auroc - valid_pf.baseline_auroc).median()),
            "mean_model_minus_baseline": float((valid_pf.model_auroc - valid_pf.baseline_auroc).mean()),
        },
    }
    (export_dir / "summary_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    save_sampling_plot(export_dir)
    save_global_plot(old_loco, export_dir)
    save_training_curves(history_250, success_250, export_dir)
    save_per_state_plot(per_state, float(old_metric.auroc), float(fixed_metric.auroc), export_dir)
    save_pf_plot(pf_runs, pf_states, export_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
