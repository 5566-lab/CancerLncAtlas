"""Current R_SELECTED benchmarks for specificity, conservation, and reversal.

The score construction is deliberately label-free.  Frozen patient-fold
Observed scores and validation-selected simple baselines are combined with the
current B3 R_SELECTED membership probability.  Direction comes from the
validation-selected B2 state head because B3 context residuals do not train a
new direction head.  Held-out patient labels are read only after every score is
materialized and are used solely for evaluation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import file_sha256, require_columns, write_table
from .metrics import binary_metrics
from .v30_integrity import atomic_write_json, merkle_sha256
from .v31_phase_c import PILOT_CANCERS, PILOT_SEEDS


BENCHMARK_SUBTYPES = ("RNAss", "DNAss")
DOWNSTREAM_TASKS = ("SPECIFICITY", "CONSERVATION", "REVERSAL")
TASK_CONTRACTS: dict[str, dict[str, str]] = {
    "SPECIFICITY": {
        "label": "heldout_specificity_label",
        "simple_method": "BestSimpleSpecificity",
        "simple_score": "best_simple_specificity_score",
        "new_method": "R_SELECTED_DualAxis_specificity",
        "new_score": "specificity_index",
    },
    "CONSERVATION": {
        "label": "heldout_conservation_label",
        "simple_method": "BestSimpleConservation",
        "simple_score": "best_simple_conservation_score",
        "new_method": "R_SELECTED_DualAxis_conservation",
        "new_score": "conservation_index",
    },
    "REVERSAL": {
        "label": "heldout_reversal_label",
        "simple_method": "BestSimpleReversal",
        "simple_score": "best_simple_reversal_score",
        "new_method": "R_SELECTED_DualAxis_reversal",
        "new_score": "reversal_index",
    },
}
REFERENCE_FILES = (
    "tables/ATLAS_REFERENCE_ALL.parquet",
    "SPECIFICITY_SIMPLE_BASELINE_PREDICTIONS.parquet",
    "OUTPUT_SHA256_MANIFEST.tsv",
    "V31_DUAL_AXIS_PILOT_SUMMARY.json",
)
FROZEN_REFERENCE_SHA256 = {
    "tables/ATLAS_REFERENCE_ALL.parquet": (
        "42db8c9bb06791bf6da8e170f50f5d357c41cd430741f2b9d8e30b9f2fd778a1"
    ),
    "SPECIFICITY_SIMPLE_BASELINE_PREDICTIONS.parquet": (
        "69644b041eaf664955d7607aaa128787ad998fc88f4b73622be984780ef6d9f5"
    ),
    "OUTPUT_SHA256_MANIFEST.tsv": (
        "adc508a0458d14bf2940607bcecf0c4aaf9a94495d40ec0dca719ca7797dd8de"
    ),
    "V31_DUAL_AXIS_PILOT_SUMMARY.json": (
        "be77492f689f15d55e2023f7fb4d69d88246d5a2ce871682e3e6feaf40dcca19"
    ),
}


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pass_json(path: Path, stage: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{stage} gate is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("failures"):
        raise RuntimeError(f"{stage} gate is not a clean PASS")
    if payload.get("full_cancer_training_started") is True:
        raise RuntimeError(f"{stage} reports forbidden full-cancer training")
    return payload


def _verify_frozen_reference(root: Path) -> dict[str, str]:
    """Verify the historical patient-fold reference against its own manifest."""

    summary_path = root / "V31_DUAL_AXIS_PILOT_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "COMPLETE_AND_STOPPED"
        or summary.get("full_cancer_started") is not False
        or summary.get("website_modified") is not False
        or set(map(str, summary.get("scope_cancers", []))) != set(PILOT_CANCERS)
    ):
        raise RuntimeError("Frozen Dual-Axis reference provenance is unsafe")
    manifest_path = root / "OUTPUT_SHA256_MANIFEST.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t")
    require_columns(manifest, ["path", "bytes", "sha256"], "Dual-Axis output manifest")
    indexed = manifest.set_index(manifest.path.astype(str))
    required = (
        "tables/ATLAS_REFERENCE_ALL.parquet",
        "SPECIFICITY_SIMPLE_BASELINE_PREDICTIONS.parquet",
    )
    hashes: dict[str, str] = {}
    for relative in required:
        path = root / relative
        if relative not in indexed.index or not path.is_file():
            raise RuntimeError(f"Frozen Dual-Axis reference asset is missing: {relative}")
        row = indexed.loc[relative]
        observed = file_sha256(path)
        if (
            int(row["bytes"]) != path.stat().st_size
            or str(row["sha256"]).lower() != observed.lower()
        ):
            raise RuntimeError(f"Frozen Dual-Axis reference asset changed: {relative}")
        hashes[relative] = observed
    hashes["OUTPUT_SHA256_MANIFEST.tsv"] = file_sha256(manifest_path)
    hashes["V31_DUAL_AXIS_PILOT_SUMMARY.json"] = file_sha256(summary_path)
    if hashes != FROZEN_REFERENCE_SHA256:
        drift = {
            name: {
                "expected": FROZEN_REFERENCE_SHA256.get(name),
                "observed": hashes.get(name),
            }
            for name in sorted(set(hashes) | set(FROZEN_REFERENCE_SHA256))
            if hashes.get(name) != FROZEN_REFERENCE_SHA256.get(name)
        }
        raise RuntimeError(f"Frozen Dual-Axis external SHA256 anchor drift: {drift}")
    return hashes


def _normalize_expected(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame,
        [
            "loco_cancer", "seed", "target_subtype", "split", "lncrna_id",
            "state_id", "proxy_positive_probability", "model_variant",
        ],
        "B3 R_SELECTED prediction",
    )
    selected = frame.loc[
        frame.split.astype(str).eq("test")
        & frame.target_subtype.astype(str).isin(BENCHMARK_SUBTYPES)
    ].copy()
    if set(selected.loco_cancer.astype(str)) != set(PILOT_CANCERS):
        raise RuntimeError("R_SELECTED downstream cancer scope drift")
    if set(selected.seed.astype(int)) != set(PILOT_SEEDS):
        raise RuntimeError("R_SELECTED downstream seed scope drift")
    if set(selected.target_subtype.astype(str)) != set(BENCHMARK_SUBTYPES):
        raise RuntimeError("R_SELECTED lacks RNAss/DNAss benchmark predictions")
    if not selected.model_variant.astype(str).eq("R_SELECTED").all():
        raise RuntimeError("Downstream expected input is not uniformly R_SELECTED")
    selected["cancer_id"] = selected.loco_cancer.astype(str)
    selected["target_id"] = selected.state_id.astype(str)
    selected["pan_cancer_expected_probability"] = pd.to_numeric(
        selected.proxy_positive_probability, errors="raise"
    ).astype(float)
    keys = ["cancer_id", "seed", "lncrna_id", "target_id"]
    if selected.duplicated(keys).any():
        raise RuntimeError("R_SELECTED has duplicate state candidate keys")
    if not selected.pan_cancer_expected_probability.between(0, 1).all():
        raise RuntimeError("R_SELECTED probability is outside [0,1]")
    return selected[keys + ["target_subtype", "pan_cancer_expected_probability"]]


def _normalize_direction(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame,
        [
            "loco_cancer", "seed", "target_subtype", "split", "lncrna_id",
            "state_id", "direction_positive_probability", "prediction_scale",
            "module_admitted",
        ],
        "B2 selected Graph direction prediction",
    )
    direction = frame.loc[
        frame.split.astype(str).eq("test")
        & frame.target_subtype.astype(str).isin(BENCHMARK_SUBTYPES)
        & frame.prediction_scale.astype(str).eq("raw_probability")
    ].copy()
    direction["cancer_id"] = direction.loco_cancer.astype(str)
    direction["target_id"] = direction.state_id.astype(str)
    direction["pan_cancer_direction_positive_probability"] = pd.to_numeric(
        direction.direction_positive_probability, errors="raise"
    ).astype(float)
    # A rejected Graph membership module has no admitted direction head for the
    # atlas.  Neutral 0.5 is the exact direction-off fallback.
    admitted = direction.module_admitted.astype(bool)
    direction.loc[
        ~admitted, "pan_cancer_direction_positive_probability"
    ] = 0.5
    keys = ["cancer_id", "seed", "lncrna_id", "target_id"]
    if direction.duplicated(keys).any():
        raise RuntimeError("B2 selected Graph has duplicate state direction keys")
    if not direction.pan_cancer_direction_positive_probability.between(0, 1).all():
        raise RuntimeError("B2 direction probability is outside [0,1]")
    return direction[keys + ["pan_cancer_direction_positive_probability"]]


def build_current_atlas_scores(
    reference: pd.DataFrame,
    simple: pd.DataFrame,
    selected_expected: pd.DataFrame,
    graph_direction: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct scores first, then attach held-out labels for evaluation."""

    reference_keys = ["cancer_id", "patient_fold_id", "lncrna_id", "target_id"]
    required_reference = [
        *reference_keys, "target_type", "observed_effect_signed",
        "observed_strength_score", "observed_reproducibility_probability",
        "heldout_specificity_label", "heldout_conservation_label",
        "heldout_reversal_label",
    ]
    require_columns(reference, required_reference, "frozen patient-fold atlas reference")
    require_columns(
        simple,
        [
            *reference_keys,
            "best_simple_specificity_method", "best_simple_specificity_score",
            "best_simple_conservation_method", "best_simple_conservation_score",
            "best_simple_reversal_method", "best_simple_reversal_score",
        ],
        "frozen atlas simple baselines",
    )
    if reference.duplicated(reference_keys).any() or simple.duplicated(reference_keys).any():
        raise RuntimeError("Frozen atlas reference/simple keys are not unique")
    reference = reference.loc[
        reference.cancer_id.astype(str).isin(PILOT_CANCERS)
        & reference.target_type.astype(str).eq("state")
    ].copy()
    label_columns = [contract["label"] for contract in TASK_CONTRACTS.values()]
    score_reference_columns = [
        *reference_keys, "target_type", "observed_effect_signed",
        "observed_strength_score", "observed_reproducibility_probability",
    ]
    # Deliberately exclude held-out labels while constructing every score.
    simple_columns = sorted(
        {
            value
            for contract in TASK_CONTRACTS.values()
            for key, value in contract.items()
            if key == "simple_score"
        }
        | {
            "best_simple_specificity_method",
            "best_simple_conservation_method",
            "best_simple_reversal_method",
        }
    )
    score_base = reference[score_reference_columns].merge(
        simple[reference_keys + simple_columns],
        on=reference_keys,
        how="inner",
        validate="one_to_one",
    )
    expected = _normalize_expected(selected_expected).merge(
        _normalize_direction(graph_direction),
        on=["cancer_id", "seed", "lncrna_id", "target_id"],
        how="inner",
        validate="one_to_one",
    )
    atlas = score_base.merge(
        expected,
        on=["cancer_id", "lncrna_id", "target_id"],
        how="inner",
        validate="many_to_many",
    )
    if atlas.empty:
        raise RuntimeError("Current Expected atlas candidate intersection is empty")
    o = pd.to_numeric(atlas.observed_strength_score, errors="raise").clip(0, 1)
    e = pd.to_numeric(
        atlas.pan_cancer_expected_probability, errors="raise"
    ).clip(0, 1)
    d = pd.to_numeric(
        atlas.pan_cancer_direction_positive_probability, errors="raise"
    ).clip(0, 1)
    observed_positive = atlas.observed_effect_signed.astype(float).ge(0)
    agreement = np.where(observed_positive, d, 1.0 - d)
    atlas["direction_agreement_confidence"] = agreement
    atlas["direction_disagreement_confidence"] = 1.0 - agreement
    atlas["specificity_index"] = o * (1.0 - e)
    atlas["conservation_index"] = o * e * agreement
    atlas["reversal_index"] = o * e * (1.0 - agreement)
    observed_probability = pd.to_numeric(
        atlas.observed_reproducibility_probability, errors="raise"
    ).clip(1e-6, 1 - 1e-6)
    expected_probability = e.clip(1e-6, 1 - 1e-6)
    atlas["specificity_logit_delta"] = (
        np.log(observed_probability / (1 - observed_probability))
        - np.log(expected_probability / (1 - expected_probability))
    )
    atlas["score_construction_used_heldout_labels"] = False
    score_hash_columns = [
        "cancer_id", "patient_fold_id", "lncrna_id", "target_id", "seed",
        "specificity_index", "conservation_index", "reversal_index",
    ]
    score_only = atlas[score_hash_columns].copy()
    labels = reference[reference_keys + label_columns]
    evaluated = atlas.merge(labels, on=reference_keys, how="left", validate="many_to_one")
    if evaluated[label_columns].isna().any().any():
        raise RuntimeError("Held-out atlas reference labels are incomplete")
    for column in label_columns:
        evaluated[column] = evaluated[column].astype(np.int8)
    return evaluated, score_only


def _metric_rows(atlas: pd.DataFrame, task: str) -> pd.DataFrame:
    contract = TASK_CONTRACTS[task]
    rows: list[dict[str, Any]] = []
    group_columns = ["cancer_id", "patient_fold_id", "target_id", "seed"]
    methods = {
        contract["simple_method"]: contract["simple_score"],
        contract["new_method"]: contract["new_score"],
    }
    for keys, group in atlas.groupby(group_columns, observed=True, sort=True):
        y = group[contract["label"]].astype(int)
        for method, column in methods.items():
            metrics = binary_metrics(y, group[column].astype(float))
            if not np.isfinite(metrics["auprc"]):
                continue
            prevalence = float(metrics["positive_rate"])
            auprc = float(metrics["auprc"])
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "target_subtype": (
                        "RNAss" if "RNAss" in str(keys[2]) else "DNAss"
                    ),
                    "task": task,
                    "method": method,
                    "metric_scope": "PF_outer_test",
                    "positive_prevalence": prevalence,
                    "auprc": auprc,
                    "auroc": metrics["auroc"],
                    "brier": metrics["brier"],
                    "ece": metrics["ece"],
                    "auprc_over_prevalence": (
                        auprc / prevalence if prevalence > 0 else np.nan
                    ),
                    "auprc_minus_prevalence": auprc - prevalence,
                    "n": metrics["n"],
                    "n_positive": metrics["n_positive"],
                }
            )
    result = pd.DataFrame(rows)
    if result.empty or set(result.cancer_id.astype(str)) != set(PILOT_CANCERS):
        raise RuntimeError(f"{task} metrics lack the exact three-cancer scope")
    return result


def _cluster_bootstrap(
    paired: pd.DataFrame, *, resamples: int, seed: int
) -> dict[str, Any]:
    cluster = paired.groupby(
        ["cancer_id", "patient_fold_id"], observed=True
    ).delta_auprc.mean()
    if len(cluster) != 15:
        raise RuntimeError("Downstream cluster contract is not 3 cancers x 5 PF")
    values = cluster.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    by_cancer = cluster.groupby(level="cancer_id").mean()
    return {
        "delta_auprc": float(values.mean()),
        "median_delta_auprc": float(np.median(values)),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "n_clusters": int(len(values)),
        "positive_clusters": int((values > 0).sum()),
        "positive_cancers": int((by_cancer > 0).sum()),
        "total_cancers": int(len(by_cancer)),
    }


def _acceptance(metrics: Mapping[str, pd.DataFrame], resamples: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(DOWNSTREAM_TASKS):
        contract = TASK_CONTRACTS[task]
        frame = metrics[task]
        keys = ["cancer_id", "patient_fold_id", "target_id", "seed", "task"]
        new = frame.loc[frame.method.eq(contract["new_method"]), keys + ["auprc"]]
        base = frame.loc[frame.method.eq(contract["simple_method"]), keys + ["auprc"]]
        paired = new.merge(base, on=keys, suffixes=("_new", "_base"), validate="one_to_one")
        paired["delta_auprc"] = paired.auprc_new - paired.auprc_base
        summary = _cluster_bootstrap(
            paired, resamples=resamples, seed=310731 + index
        )
        if (
            summary["delta_auprc"] >= 0.02
            and summary["positive_cancers"] >= 2
            and summary["ci95_lower"] > 0
        ):
            decision = "STRONG_GO"
        elif summary["delta_auprc"] >= 0.02 and summary["positive_cancers"] >= 2:
            decision = "GO"
        elif summary["delta_auprc"] > 0 and summary["positive_cancers"] >= 2:
            decision = "BORDERLINE"
        else:
            decision = "NO_GO"
        rows.append(
            {
                "task": task,
                "best_simple_baseline": contract["simple_method"],
                "new_score": contract["new_method"],
                **summary,
                "decision": decision,
                "test_metric_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows)


def run_current_expected_downstream(
    *,
    selected_predictions_path: Path,
    b3_gate_path: Path,
    graph_predictions_path: Path,
    graph_gate_path: Path,
    hard_report_path: Path,
    frozen_dual_axis_root: Path,
    output_dir: Path,
    aggregation_git_commit: str,
    bootstrap_resamples: int = 5000,
) -> dict[str, Any]:
    """Build the exact HARD-GO-only current Expected downstream benchmarks."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RuntimeError(f"Current Expected downstream refuses reuse: {output_dir}")
    b3 = _pass_json(Path(b3_gate_path), "B3 context")
    graph = _pass_json(Path(graph_gate_path), "B2 Graph")
    hard = _pass_json(Path(hard_report_path), "B2 HARD REPORT")
    if (
        b3.get("stage") != "PHASE_B3_MULTIMODAL_CONTEXT_RESIDUAL"
        or b3.get("selection_used_test_labels") is not False
        or hard.get("stage") != "B2_HARD_REPORT"
        or hard.get("decision") != "HARD_GO_B3"
        or hard.get("b3_authorized") is not True
        or graph.get("stage") != "PHASE_B2_GRAPH_RESIDUAL"
        or graph.get("selection_used_test_labels") is not False
        or b3.get("graph_gate_sha256") != file_sha256(graph_gate_path)
        or b3.get("b2_hard_report_sha256") != file_sha256(hard_report_path)
        or hard.get("b2_gate_sha256") != file_sha256(graph_gate_path)
    ):
        raise RuntimeError("Current Expected downstream lacks canonical HARD-GO lineage")
    reference_hashes = _verify_frozen_reference(Path(frozen_dual_axis_root))
    selected = pd.read_parquet(selected_predictions_path)
    graph_predictions = pd.read_parquet(graph_predictions_path)
    reference = pd.read_parquet(
        Path(frozen_dual_axis_root) / "tables/ATLAS_REFERENCE_ALL.parquet"
    )
    simple = pd.read_parquet(
        Path(frozen_dual_axis_root) / "SPECIFICITY_SIMPLE_BASELINE_PREDICTIONS.parquet"
    )
    atlas, score_only = build_current_atlas_scores(
        reference, simple, selected, graph_predictions
    )
    metric_frames = {task: _metric_rows(atlas, task) for task in DOWNSTREAM_TASKS}
    acceptance = _acceptance(metric_frames, int(bootstrap_resamples))
    output_dir.mkdir(parents=True)
    paths = {
        "atlas": output_dir / "CURRENT_EXPECTED_DUAL_AXIS_SCORES.parquet",
        "score_only": output_dir / "SCORE_CONSTRUCTION_WITHOUT_HELDOUT_LABELS.parquet",
        "specificity": output_dir / "SPECIFICITY_BENCHMARK_METRICS.tsv",
        "conservation": output_dir / "CONSERVATION_BENCHMARK_METRICS.tsv",
        "reversal": output_dir / "REVERSAL_BENCHMARK_METRICS.tsv",
        "acceptance": output_dir / "PRIMARY_AUPRC_ACCEPTANCE_MATRIX.tsv",
        "audit": output_dir / "CURRENT_EXPECTED_CANDIDATE_AUDIT.tsv",
        "manifest": output_dir / "CURRENT_EXPECTED_SHA256.tsv",
        "gate": output_dir / "CURRENT_EXPECTED_LINEAGE.json",
    }
    audit = (
        atlas.groupby(["cancer_id", "target_subtype", "seed"], observed=True)
        .agg(
            candidate_rows=("lncrna_id", "size"),
            patient_folds=("patient_fold_id", "nunique"),
            unique_lncrnas=("lncrna_id", "nunique"),
            score_used_heldout_labels=(
                "score_construction_used_heldout_labels", "max"
            ),
        )
        .reset_index()
    )
    if (
        len(audit) != 18
        or not audit.patient_folds.eq(5).all()
        or audit.score_used_heldout_labels.any()
    ):
        raise RuntimeError("Current Expected downstream candidate/fold audit failed")
    for frame, key in (
        (atlas, "atlas"), (score_only, "score_only"),
        (metric_frames["SPECIFICITY"], "specificity"),
        (metric_frames["CONSERVATION"], "conservation"),
        (metric_frames["REVERSAL"], "reversal"),
        (acceptance, "acceptance"), (audit, "audit"),
    ):
        _atomic_table(frame, paths[key])
    output_files = [
        paths[key]
        for key in (
            "atlas", "score_only", "specificity", "conservation", "reversal",
            "acceptance", "audit",
        )
    ]
    manifest_rows = [
        {
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in output_files
    ]
    _atomic_table(pd.DataFrame(manifest_rows), paths["manifest"])
    payload = {
        "status": "PASS",
        "stage": "PHASE_B3_CURRENT_EXPECTED_DOWNSTREAM",
        "expected_model": "R_SELECTED",
        "expected_prediction_sha256": file_sha256(selected_predictions_path),
        "b3_gate_sha256": file_sha256(b3_gate_path),
        "b2_graph_prediction_sha256": file_sha256(graph_predictions_path),
        "b2_graph_gate_sha256": file_sha256(graph_gate_path),
        "b2_hard_report_sha256": file_sha256(hard_report_path),
        "frozen_dual_axis_reference_hashes": reference_hashes,
        "pilot_cancers": list(PILOT_CANCERS),
        "target_subtypes": ["Pathway", "RNAss", "DNAss"],
        "benchmark_target_subtypes": list(BENCHMARK_SUBTYPES),
        "benchmark_tasks": list(DOWNSTREAM_TASKS),
        "reference_scope": "patient_fold_outer_test_RNAss_DNAss",
        "score_formula_scope": "preregistered_fixed_formula_no_test_tuning",
        "direction_source": "B2_validation_selected_state_direction_head",
        "test_labels_used_for_score_construction": False,
        "test_labels_used_for_evaluation_only": True,
        "test_metric_used_for_selection": False,
        "candidate_audit_rows": int(len(audit)),
        "atlas_rows": int(len(atlas)),
        "bootstrap_resamples": int(bootstrap_resamples),
        "aggregation_git_commit": aggregation_git_commit,
        "manifest_sha256": file_sha256(paths["manifest"]),
        "manifest_merkle_sha256": merkle_sha256(manifest_rows),
        "full_cancer_training_started": False,
        "website_modified": False,
        "failures": [],
    }
    atomic_write_json(paths["gate"], payload)
    return payload
