"""Aggregate patient OOF outputs and train availability-masked MoE gates.

Key changes relative to the legacy aggregator:
- the final universe is the union of strict, patient-native, and evidence pairs;
- strict OOF is an optional expert, not the left/mandatory table;
- the patient-native probability is independent of strict scores/embeddings;
- final scores are learned from OOF expert predictions with masked gates,
  never a simple arithmetic mean.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cc_hhgt_v26.adapter_data import load_evidence_candidates  # noqa: E402
from cc_hhgt_v26.moe_model import (  # noqa: E402
    AvailabilityMaskedMoE,
    fit_moe,
    predict_moe,
    probability_to_logit,
    sigmoid_numpy,
)
from cc_hhgt.v30_integrity import stable_partition  # noqa: E402

RUN = Path(os.getenv("CC_HHGT_ADAPTER_RUN_ROOT", str(ROOT / "results" / "cancer_adapter_run")))
ADAPTER = Path(os.getenv("CC_HHGT_ADAPTER_RESULT_ROOT", str(ROOT / "results" / "cancer_adapter")))
RELEASE = Path(os.getenv("CC_HHGT_RELEASE_DIR", str(ROOT / "results" / "v2_8_cancer_native_moe_release")))
STRICT_OOF = Path(os.getenv("CC_HHGT_STRICT_OOF_PATH", str(ROOT / "results" / "strict_release" / "strict_cross_cancer_oof_prediction.parquet")))
NEW_MEMBER = Path(os.getenv("CC_HHGT_PATHWAY_FAMILY_MEMBER", str(ROOT / "results" / "static_pathway" / "static_pathway_family_member.parquet")))
OLD_ROOT = Path(os.getenv("CC_HHGT_OLD_EVIDENCE_ROOT", str(ROOT.parent / "CC_HHGT_GPU_4070TiS_v2_3_multiseed_external_validation")))
OLD_MEMBER = OLD_ROOT / "results" / "model" / "cc_hhgt_v2_1_gpu" / "tables" / "pathway_family_member.parquet"
OLD_PREDICTION = (
    OLD_ROOT
    / "results"
    / "model"
    / "cc_hhgt_v2_1_gpu"
    / "tables"
    / "all_candidate_prediction"
    / "cancer_id=*"
    / "*.parquet"
)
PRECOMPUTED_EVIDENCE = Path(os.getenv("CC_HHGT_PRECOMPUTED_EVIDENCE", "")) if os.getenv("CC_HHGT_PRECOMPUTED_EVIDENCE") else None
GRAPH_NODE = Path(os.getenv("CC_HHGT_GRAPH_NODE", str(ROOT / "results" / "strict_graph" / "graph_node.parquet")))
T95_DF14 = 2.14478668792
EXPECTED_CANCERS = 33
EXPECTED_PATIENT_FOLDS = 5
EXPECTED_SEEDS = 3
MOE_MAX_TRAIN_ROWS = int(os.getenv("CC_HHGT_MOE_MAX_TRAIN_ROWS", "2000000"))
MOE_SEED = int(os.getenv("CC_HHGT_MOE_SEED", "20260731"))

KEYS = ["cancer_id", "lncrna_id", "pathway_family_id"]
DISCOVERY_QUALITY_COLUMNS = [
    "cross_cancer_probability_sd",
    "cancer_native_probability_sd",
    "train_detection_rate",
    "train_n_patients",
    "candidate_from_patient",
    "strict_available",
]
CONFIDENCE_QUALITY_COLUMNS = [
    *DISCOVERY_QUALITY_COLUMNS,
    "evidence_crosswalk_max_coverage",
    "candidate_from_evidence",
]


def validate() -> None:
    if not (RUN / "SUCCESS.json").exists():
        raise RuntimeError("patient adapter run lacks SUCCESS.json")
    if (RUN / "FAILED.txt").exists():
        raise RuntimeError("patient adapter run contains FAILED.txt")
    manifest = pd.read_csv(RUN / "task_manifest.tsv", sep="\t")
    expected_rows = EXPECTED_CANCERS * EXPECTED_PATIENT_FOLDS * EXPECTED_SEEDS
    required = {"cancer_id", "patient_fold_id", "seed", "status"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise RuntimeError(f"patient adapter manifest lacks required columns: {missing}")
    combination_counts = manifest.groupby(
        ["cancer_id", "patient_fold_id", "seed"], dropna=False
    ).size()
    cancer_fold_counts = manifest.groupby("cancer_id")["patient_fold_id"].nunique()
    cancer_seed_counts = manifest.groupby("cancer_id")["seed"].nunique()
    complete = (
        len(manifest) == expected_rows
        and manifest["cancer_id"].nunique() == EXPECTED_CANCERS
        and cancer_fold_counts.eq(EXPECTED_PATIENT_FOLDS).all()
        and cancer_seed_counts.eq(EXPECTED_SEEDS).all()
        and combination_counts.eq(1).all()
        and manifest["status"].eq("COMPLETED").all()
    )
    if not complete:
        raise RuntimeError(
            "patient adapter is incomplete or structurally invalid: "
            f"rows={len(manifest)}/{expected_rows}, "
            f"cancers={manifest['cancer_id'].nunique()}, "
            f"status={manifest['status'].value_counts().to_dict()}"
        )


def _build_candidate_union(summary_output: Path) -> pd.DataFrame:
    strict = pd.read_parquet(STRICT_OOF, columns=KEYS).drop_duplicates(KEYS)
    strict["candidate_from_strict"] = 1
    adapter = pd.read_parquet(summary_output, columns=KEYS).drop_duplicates(KEYS)
    adapter["candidate_from_patient"] = 1
    cancers = sorted(set(strict["cancer_id"]) | set(adapter["cancer_id"]))
    evidence_frames = [load_evidence_candidates(cancer) for cancer in cancers]
    evidence = (
        pd.concat(evidence_frames, ignore_index=True, sort=False)
        if evidence_frames
        else pd.DataFrame(columns=KEYS)
    )
    if not evidence.empty:
        evidence = evidence[KEYS].drop_duplicates(KEYS)
        evidence["candidate_from_evidence"] = 1
    frames = [strict, adapter]
    if not evidence.empty:
        frames.append(evidence)
    union = pd.concat(frames, ignore_index=True, sort=False)
    for column in [
        "candidate_from_strict",
        "candidate_from_patient",
        "candidate_from_evidence",
    ]:
        if column not in union:
            union[column] = 0
    union = (
        union.groupby(KEYS, observed=True, as_index=False)[
            [
                "candidate_from_strict",
                "candidate_from_patient",
                "candidate_from_evidence",
            ]
        ]
        .max()
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    for column in [
        "candidate_from_strict",
        "candidate_from_patient",
        "candidate_from_evidence",
    ]:
        union[column] = union[column].fillna(0).astype("int8")
    union["candidate_source"] = union.apply(
        lambda row: "|".join(
            source
            for source, column in [
                ("strict", "candidate_from_strict"),
                ("patient_native", "candidate_from_patient"),
                ("evidence", "candidate_from_evidence"),
            ]
            if int(row[column]) == 1
        ),
        axis=1,
    )
    return union


def _fold_index(value: object) -> int:
    match = re.search(r"PF[_-]?(\d+)", str(value), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return stable_partition(str(value), EXPECTED_PATIENT_FOLDS, 20260810) + 1


def _quality_matrix(
    frame: pd.DataFrame,
    quality_columns: list[str],
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quality = frame.reindex(columns=quality_columns).copy()
    for column in quality:
        quality[column] = pd.to_numeric(quality[column], errors="coerce")
    quality = quality.replace([np.inf, -np.inf], np.nan)
    if mean is None:
        mean = quality.mean(axis=0).fillna(0).to_numpy(dtype=np.float32)
    if scale is None:
        scale = quality.std(axis=0).replace(0, 1).fillna(1).to_numpy(dtype=np.float32)
    values = quality.fillna(pd.Series(mean, index=quality.columns)).to_numpy(dtype=np.float32)
    values = (values - mean) / scale
    return values.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def _expert_arrays(
    frame: pd.DataFrame, expert_columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    probability = frame.reindex(columns=expert_columns).apply(
        pd.to_numeric, errors="coerce"
    )
    availability = probability.notna().to_numpy(dtype=np.float32)
    logits = np.full(probability.shape, np.nan, dtype=np.float32)
    for index, column in enumerate(expert_columns):
        valid = probability[column].notna().to_numpy()
        if valid.any():
            logits[valid, index] = probability_to_logit(
                probability.loc[valid, column].to_numpy(dtype=np.float32)
            )
    return logits, availability


def _subsample_training(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    if len(frame) <= MOE_MAX_TRAIN_ROWS:
        return frame
    rng = np.random.default_rng(seed)
    positive = frame[frame["test_membership_label"].ge(0.5)]
    negative = frame[frame["test_membership_label"].lt(0.5)]
    keep_positive = positive
    remaining = max(MOE_MAX_TRAIN_ROWS - len(keep_positive), 0)
    if remaining == 0:
        return keep_positive.sample(MOE_MAX_TRAIN_ROWS, random_state=seed)
    chosen_negative = negative.iloc[
        rng.choice(len(negative), size=min(remaining, len(negative)), replace=False)
    ]
    return pd.concat([keep_positive, chosen_negative], ignore_index=True).sample(
        frac=1.0, random_state=seed
    )


def _train_crossfit_gate(
    frame: pd.DataFrame,
    expert_columns: list[str],
    gate_name: str,
    quality_columns: list[str],
) -> tuple[pd.DataFrame, AvailabilityMaskedMoE, dict[str, object]]:
    frame = frame.copy()
    frame["moe_fold_index"] = frame["patient_fold_id"].map(_fold_index)
    frame["test_membership_label"] = pd.to_numeric(
        frame["test_membership_label"], errors="coerce"
    )
    oof_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []
    expert_names = list(expert_columns)

    for heldout in sorted(frame["moe_fold_index"].dropna().unique()):
        validation_fold = heldout % EXPECTED_PATIENT_FOLDS + 1
        train = frame[
            ~frame["moe_fold_index"].isin([heldout, validation_fold])
        ].copy()
        validation = frame[frame["moe_fold_index"].eq(validation_fold)].copy()
        test = frame[frame["moe_fold_index"].eq(heldout)].copy()
        train = _subsample_training(train, MOE_SEED + int(heldout))
        train_quality, mean, scale = _quality_matrix(train, quality_columns)
        validation_quality, _, _ = _quality_matrix(validation, quality_columns, mean, scale)
        test_quality, _, _ = _quality_matrix(test, quality_columns, mean, scale)
        train_logits, train_available = _expert_arrays(train, expert_names)
        validation_logits, validation_available = _expert_arrays(validation, expert_names)
        test_logits, test_available = _expert_arrays(test, expert_names)
        fit = fit_moe(
            train_logits,
            train_available,
            train_quality,
            train["test_membership_label"].to_numpy(dtype=np.float32),
            validation_logits,
            validation_available,
            validation_quality,
            validation["test_membership_label"].to_numpy(dtype=np.float32),
            seed=MOE_SEED + int(heldout),
        )
        test_logit, test_weight = predict_moe(
            fit.model, test_logits, test_available, test_quality
        )
        part = test[[*KEYS, "patient_fold_id", "test_membership_label"]].copy()
        part[f"{gate_name}_oof_probability"] = sigmoid_numpy(test_logit)
        for index, expert in enumerate(expert_names):
            part[f"{gate_name}_weight_{expert}"] = test_weight[:, index].astype(
                "float32"
            )
        oof_parts.append(part)
        fold_metrics.append(
            {
                "gate": gate_name,
                "heldout_fold": int(heldout),
                "validation_fold": int(validation_fold),
                "n_train": int(len(train)),
                "n_validation": int(len(validation)),
                "n_test": int(len(test)),
                **fit.validation_metrics,
            }
        )

    # Final deployable gate: four fold groups train, one group validation.  OOF
    # performance above remains the unbiased report used for model selection.
    validation_fold = EXPECTED_PATIENT_FOLDS
    final_train = frame[~frame["moe_fold_index"].eq(validation_fold)].copy()
    final_validation = frame[frame["moe_fold_index"].eq(validation_fold)].copy()
    final_train = _subsample_training(final_train, MOE_SEED + 100)
    train_quality, mean, scale = _quality_matrix(final_train, quality_columns)
    validation_quality, _, _ = _quality_matrix(final_validation, quality_columns, mean, scale)
    train_logits, train_available = _expert_arrays(final_train, expert_names)
    validation_logits, validation_available = _expert_arrays(
        final_validation, expert_names
    )
    final_fit = fit_moe(
        train_logits,
        train_available,
        train_quality,
        final_train["test_membership_label"].to_numpy(dtype=np.float32),
        validation_logits,
        validation_available,
        validation_quality,
        final_validation["test_membership_label"].to_numpy(dtype=np.float32),
        seed=MOE_SEED + 100,
    )
    gate_root = RELEASE / "moe_gate"
    gate_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": final_fit.state_dict,
            "n_experts": len(expert_names),
            "quality_dim": len(quality_columns),
            "expert_columns": expert_names,
            "quality_columns": quality_columns,
            "quality_mean": mean,
            "quality_scale": scale,
            "training_source": "OOF expert predictions only",
            "availability_mask": "masked_softmax; unavailable expert weight is zero",
            "seed": MOE_SEED + 100,
        },
        gate_root / f"{gate_name}.pt",
    )
    pd.DataFrame(final_fit.history).to_csv(
        gate_root / f"{gate_name}_history.tsv", sep="\t", index=False
    )
    pd.DataFrame(fold_metrics).to_csv(
        gate_root / f"{gate_name}_crossfit_metrics.tsv", sep="\t", index=False
    )
    metadata = {
        "gate": gate_name,
        "expert_columns": expert_names,
        "quality_columns": quality_columns,
        "quality_mean": mean.tolist(),
        "quality_scale": scale.tolist(),
        "validation_metrics": final_fit.validation_metrics,
    }
    return pd.concat(oof_parts, ignore_index=True), final_fit.model, metadata


def _apply_gate(
    model: AvailabilityMaskedMoE,
    frame: pd.DataFrame,
    metadata: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    expert_columns = list(metadata["expert_columns"])
    quality_columns = list(metadata["quality_columns"])
    quality, _, _ = _quality_matrix(
        frame,
        quality_columns,
        np.asarray(metadata["quality_mean"], dtype=np.float32),
        np.asarray(metadata["quality_scale"], dtype=np.float32),
    )
    logits, availability = _expert_arrays(frame, expert_columns)
    valid = availability.sum(axis=1) > 0
    final_logit = np.full(len(frame), np.nan, dtype=np.float32)
    weights = np.zeros((len(frame), len(expert_columns)), dtype=np.float32)
    if valid.any():
        predicted_logit, predicted_weight = predict_moe(
            model, logits[valid], availability[valid], quality[valid]
        )
        final_logit[valid] = predicted_logit.astype(np.float32)
        weights[valid] = predicted_weight.astype(np.float32)
    return final_logit, weights


def main() -> int:
    validate()
    RELEASE.mkdir(parents=True, exist_ok=True)
    database = RELEASE / "release_build.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute("PRAGMA threads=8")
    connection.execute("PRAGMA memory_limit='12GB'")
    prediction_glob = ADAPTER / "*" / "LOCO_*__PF*" / "seed_*" / "prediction.parquet"
    metric_glob = ADAPTER / "*" / "LOCO_*__PF*" / "seed_*" / "metrics.tsv"
    oof_output = RELEASE / "cancer_specific_oof_prediction.parquet"
    summary_output = RELEASE / "cancer_specific_multiseed_summary.parquet"
    calibration_output = RELEASE / "cancer_specific_calibration.parquet"
    membership_output = RELEASE / "cancer_specific_membership.parquet"
    metric_output = RELEASE / "patient_level_oof_metrics.parquet"
    attribution_source = ADAPTER / "cancer_adapter_feature_attribution.parquet"
    attribution_output = RELEASE / "cancer_adapter_feature_attribution.parquet"

    connection.execute(
        f"""
        COPY (
          SELECT *
          FROM read_parquet('{prediction_glob.as_posix()}', union_by_name=true)
          ORDER BY cancer_id, patient_fold_id, seed, lncrna_id, pathway_family_id
        ) TO '{oof_output.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    connection.execute(
        f"""
        COPY (
          SELECT
            cancer_id,
            lncrna_id,
            pathway_family_id,
            count(*)::INTEGER AS n_patient_fold_seed_predictions,
            count(DISTINCT patient_fold_id)::INTEGER AS n_patient_folds_available,
            count(DISTINCT seed)::INTEGER AS n_seeds_available,
            avg(cancer_specific_probability)::FLOAT AS cancer_specific_probability,
            stddev_samp(cancer_specific_probability)::FLOAT AS cancer_specific_probability_sd,
            avg(cancer_native_probability)::FLOAT AS cancer_native_probability,
            stddev_samp(cancer_native_probability)::FLOAT AS cancer_native_probability_sd,
            avg(cancer_joint_probability)::FLOAT AS cancer_joint_probability,
            stddev_samp(cancer_joint_probability)::FLOAT AS cancer_joint_probability_sd,
            avg(patient_gate_native_weight)::FLOAT AS patient_gate_native_weight,
            avg(patient_gate_joint_weight)::FLOAT AS patient_gate_joint_weight,
            greatest(0.0, avg(cancer_specific_probability)
              - {T95_DF14} * stddev_samp(cancer_specific_probability) / sqrt(count(*)))::FLOAT
              AS cancer_specific_probability_ci95_low,
            least(1.0, avg(cancer_specific_probability)
              + {T95_DF14} * stddev_samp(cancer_specific_probability) / sqrt(count(*)))::FLOAT
              AS cancer_specific_probability_ci95_high,
            avg(direction_probability)::FLOAT AS cancer_direction_probability,
            avg(specificity_score)::FLOAT AS specificity_score,
            avg(test_membership_label)::FLOAT AS heldout_patient_support_rate,
            avg(train_detection_rate)::FLOAT AS train_detection_rate,
            avg(train_n_patients)::FLOAT AS train_n_patients,
            max(candidate_from_strict)::TINYINT AS candidate_from_strict,
            max(candidate_from_patient)::TINYINT AS candidate_from_patient,
            max(candidate_from_evidence)::TINYINT AS candidate_from_evidence,
            max(strict_available)::TINYINT AS strict_available
          FROM read_parquet('{oof_output.as_posix()}')
          GROUP BY cancer_id, lncrna_id, pathway_family_id
          HAVING count(*) = count(DISTINCT patient_fold_id) * {EXPECTED_SEEDS}
          ORDER BY cancer_id, lncrna_id, pathway_family_id
        ) TO '{summary_output.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    connection.execute(
        f"""
        COPY (
          SELECT
            cancer_id, patient_fold_id, seed::BIGINT AS seed,
            temperature::DOUBLE AS temperature, calibration_status,
            'validation_patients_only' AS calibration_source
          FROM read_csv_auto('{metric_glob.as_posix()}', delim='\t', header=true, union_by_name=true)
          ORDER BY cancer_id, patient_fold_id, seed
        ) TO '{calibration_output.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    connection.execute(
        f"""
        COPY (
          WITH fold_metric AS (
            SELECT * FROM read_csv_auto('{metric_glob.as_posix()}', delim='\t', header=true, union_by_name=true)
          ), seed_metric AS (
            SELECT cancer_id, seed, count(*)::INTEGER AS n_patient_folds,
              avg(test_auprc)::DOUBLE AS auprc,
              avg(test_auroc)::DOUBLE AS auroc,
              avg(test_brier)::DOUBLE AS brier,
              avg(test_ece)::DOUBLE AS ece,
              avg(native_test_auprc)::DOUBLE AS native_auprc
            FROM fold_metric GROUP BY cancer_id, seed
          )
          SELECT cancer_id, count(*)::INTEGER AS n_seeds,
            avg(auprc)::DOUBLE AS auprc_mean, stddev_samp(auprc)::DOUBLE AS auprc_sd,
            avg(native_auprc)::DOUBLE AS native_auprc_mean,
            stddev_samp(native_auprc)::DOUBLE AS native_auprc_sd,
            avg(auroc)::DOUBLE AS auroc_mean, stddev_samp(auroc)::DOUBLE AS auroc_sd,
            avg(brier)::DOUBLE AS brier_mean, stddev_samp(brier)::DOUBLE AS brier_sd,
            avg(ece)::DOUBLE AS ece_mean, stddev_samp(ece)::DOUBLE AS ece_sd,
            min(n_patient_folds)::INTEGER AS n_patient_folds
          FROM seed_metric GROUP BY cancer_id HAVING count(*) = 3 ORDER BY cancer_id
        ) TO '{metric_output.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    connection.execute(
        f"""
        COPY (
          SELECT *, cancer_specific_probability >= 0.5 AS predicted_membership,
            CASE WHEN cancer_direction_probability >= 0.55 THEN 'positive'
                 WHEN cancer_direction_probability <= 0.45 THEN 'negative'
                 ELSE 'uncertain' END AS direction
          FROM read_parquet('{summary_output.as_posix()}')
        ) TO '{membership_output.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    if attribution_source.exists():
        connection.execute(
            f"""
            COPY (
              SELECT cancer_id, patient_fold_id, seed, feature_group,
                normalized_weight_attribution
              FROM read_parquet('{attribution_source.as_posix()}')
            ) TO '{attribution_output.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    candidate_union = _build_candidate_union(summary_output)
    candidate_union_output = RELEASE / "candidate_union.parquet"
    candidate_union.to_parquet(candidate_union_output, index=False, compression="zstd")

    evidence_output = RELEASE / "evidence_integrated_spf_probability.parquet"
    if PRECOMPUTED_EVIDENCE is not None and PRECOMPUTED_EVIDENCE.exists():
        precomputed = pd.read_parquet(PRECOMPUTED_EVIDENCE)
        required = {*KEYS, "evidence_integrated_probability"}
        missing = sorted(required - set(precomputed.columns))
        if missing:
            raise RuntimeError(
                f"precomputed evidence table lacks columns {missing}: {PRECOMPUTED_EVIDENCE}"
            )
        keep = [
            *KEYS,
            "evidence_integrated_probability",
            "evidence_crosswalk_max_coverage",
            "evidence_crosswalk_weight",
            "n_contributing_old_families",
        ]
        keep = [column for column in keep if column in precomputed.columns]
        precomputed = precomputed[keep].drop_duplicates(KEYS)
        precomputed.to_parquet(evidence_output, index=False, compression="zstd")
    else:
        old_prediction_files = list(OLD_PREDICTION.parent.parent.glob("cancer_id=*/*.parquet"))
        if not OLD_MEMBER.exists() or not old_prediction_files or not NEW_MEMBER.exists():
            # Historical evidence is an optional confidence expert in V2.9.
            # Missing historical assets must not block discovery or be silently
            # converted to zero support; emit an empty, unavailable expert table.
            pd.DataFrame(
                columns=[
                    *KEYS,
                    "evidence_integrated_probability",
                    "evidence_crosswalk_max_coverage",
                    "evidence_crosswalk_weight",
                    "n_contributing_old_families",
                ]
            ).to_parquet(evidence_output, index=False, compression="zstd")
            (RELEASE / "HISTORICAL_EVIDENCE_UNAVAILABLE.json").write_text(
                json.dumps(
                    {
                        "status": "UNAVAILABLE",
                        "reason": "historical evidence assets not supplied",
                        "old_member": str(OLD_MEMBER),
                        "old_prediction_root": str(OLD_PREDICTION.parent.parent),
                        "new_member": str(NEW_MEMBER),
                        "policy": "availability-masked; never fill missing evidence with zero",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            crosswalk = RELEASE / "pf_to_spf_exact_pathway_crosswalk.parquet"
            connection.execute(
            f"""
            COPY (
              WITH old_member AS (
                SELECT pathway_family_id AS old_pathway_family_id, pathway_id
                FROM read_parquet('{OLD_MEMBER.as_posix()}')
              ), new_member AS (
                SELECT pathway_family_id, pathway_id FROM read_parquet('{NEW_MEMBER.as_posix()}')
              ), old_size AS (
                SELECT old_pathway_family_id, count(*) AS old_n FROM old_member GROUP BY old_pathway_family_id
              ), new_size AS (
                SELECT pathway_family_id, count(*) AS new_n FROM new_member GROUP BY pathway_family_id
              ), overlap AS (
                SELECT o.old_pathway_family_id, n.pathway_family_id, count(*) AS overlap_n
                FROM old_member o JOIN new_member n USING(pathway_id)
                GROUP BY o.old_pathway_family_id, n.pathway_family_id
              )
              SELECT overlap.*, old_size.old_n, new_size.new_n,
                overlap_n::DOUBLE / (old_n + new_n - overlap_n) AS pathway_jaccard,
                overlap_n::DOUBLE / new_n AS new_family_coverage
              FROM overlap JOIN old_size USING(old_pathway_family_id)
              JOIN new_size USING(pathway_family_id)
              ORDER BY old_pathway_family_id, pathway_family_id
            ) TO '{crosswalk.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
            )
            connection.execute(
            f"""
            COPY (
              WITH wanted AS (
                SELECT cancer_id, lncrna_id, pathway_family_id
                FROM read_parquet('{candidate_union_output.as_posix()}')
              ), old_prediction AS (
                SELECT cancer_id, lncrna_id,
                  pathway_family_id AS old_pathway_family_id,
                  calibrated_probability
                FROM read_parquet('{OLD_PREDICTION.as_posix()}', hive_partitioning=true, union_by_name=true)
              ), mapped AS (
                SELECT wanted.cancer_id, wanted.lncrna_id, wanted.pathway_family_id,
                  old_prediction.calibrated_probability, crosswalk.pathway_jaccard,
                  crosswalk.new_family_coverage
                FROM wanted JOIN read_parquet('{crosswalk.as_posix()}') crosswalk USING(pathway_family_id)
                JOIN old_prediction ON wanted.cancer_id = old_prediction.cancer_id
                  AND wanted.lncrna_id = old_prediction.lncrna_id
                  AND crosswalk.old_pathway_family_id = old_prediction.old_pathway_family_id
              )
              SELECT cancer_id, lncrna_id, pathway_family_id,
                (sum(calibrated_probability * pathway_jaccard)
                  / nullif(sum(pathway_jaccard), 0))::FLOAT AS evidence_integrated_probability,
                max(new_family_coverage)::FLOAT AS evidence_crosswalk_max_coverage,
                sum(pathway_jaccard)::FLOAT AS evidence_crosswalk_weight,
                count(*)::INTEGER AS n_contributing_old_families
              FROM mapped GROUP BY cancer_id, lncrna_id, pathway_family_id
            ) TO '{evidence_output.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
            )
    # Build one OOF stacking row per patient fold and candidate, averaging the
    # three seed predictions.  These are the only rows used to train the MoE.
    moe_training_output = RELEASE / "moe_oof_training_frame.parquet"
    connection.execute(
        f"""
        COPY (
          WITH patient AS (
            SELECT cancer_id, patient_fold_id, lncrna_id, pathway_family_id,
              avg(cancer_native_probability)::FLOAT AS cancer_native_probability,
              stddev_samp(cancer_native_probability)::FLOAT AS cancer_native_probability_sd,
              avg(cancer_specific_probability)::FLOAT AS cancer_specific_probability,
              avg(test_membership_label)::FLOAT AS test_membership_label,
              avg(train_detection_rate)::FLOAT AS train_detection_rate,
              avg(train_n_patients)::FLOAT AS train_n_patients,
              max(candidate_from_strict)::TINYINT AS candidate_from_strict,
              max(candidate_from_patient)::TINYINT AS candidate_from_patient,
              max(candidate_from_evidence)::TINYINT AS candidate_from_evidence,
              max(strict_available)::TINYINT AS strict_available
            FROM read_parquet('{oof_output.as_posix()}')
            GROUP BY cancer_id, patient_fold_id, lncrna_id, pathway_family_id
            HAVING count(DISTINCT seed) = {EXPECTED_SEEDS}
          )
          SELECT patient.*, strict.cross_cancer_probability,
            strict.cross_cancer_probability_sd,
            evidence.evidence_integrated_probability,
            evidence.evidence_crosswalk_max_coverage
          FROM patient
          LEFT JOIN read_parquet('{STRICT_OOF.as_posix()}') strict
            USING(cancer_id, lncrna_id, pathway_family_id)
          LEFT JOIN read_parquet('{evidence_output.as_posix()}') evidence
            USING(cancer_id, lncrna_id, pathway_family_id)
        ) TO '{moe_training_output.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    connection.close()

    moe_frame = pd.read_parquet(moe_training_output)
    discovery_oof, discovery_model, discovery_metadata = _train_crossfit_gate(
        moe_frame,
        ["cross_cancer_probability", "cancer_native_probability"],
        "discovery",
        DISCOVERY_QUALITY_COLUMNS,
    )
    confidence_oof, confidence_model, confidence_metadata = _train_crossfit_gate(
        moe_frame,
        [
            "cross_cancer_probability",
            "cancer_native_probability",
            "evidence_integrated_probability",
        ],
        "confidence",
        CONFIDENCE_QUALITY_COLUMNS,
    )
    moe_oof = discovery_oof.merge(
        confidence_oof,
        on=[*KEYS, "patient_fold_id", "test_membership_label"],
        how="outer",
        validate="one_to_one",
    )
    moe_oof.to_parquet(
        RELEASE / "moe_oof_prediction.parquet", index=False, compression="zstd"
    )

    strict = pd.read_parquet(STRICT_OOF)
    adapter = pd.read_parquet(summary_output)
    evidence = pd.read_parquet(evidence_output)
    final = candidate_union.merge(strict, on=KEYS, how="left", suffixes=("", "_strict"))
    final = final.merge(adapter, on=KEYS, how="left", suffixes=("", "_adapter"))
    final = final.merge(evidence, on=KEYS, how="left")
    final["strict_available"] = final["cross_cancer_probability"].notna().astype("int8")
    final["patient_available"] = final["cancer_native_probability"].notna().astype("int8")
    final["evidence_available"] = final["evidence_integrated_probability"].notna().astype("int8")
    model_score_available = final[[
        "strict_available", "patient_available", "evidence_available"
    ]].sum(axis=1).gt(0)
    unscored = final.loc[~model_score_available].copy()
    unscored["failure_reason"] = "no_strict_patient_or_numeric_evidence_expert"
    unscored.to_parquet(
        RELEASE / "candidate_union_unscored.parquet",
        index=False,
        compression="zstd",
    )
    final = final.loc[model_score_available].copy().reset_index(drop=True)

    discovery_logit, discovery_weight = _apply_gate(
        discovery_model, final, discovery_metadata
    )
    confidence_logit, confidence_weight = _apply_gate(
        confidence_model, final, confidence_metadata
    )
    final["target_evidence_masked_fused_probability"] = sigmoid_numpy(discovery_logit)
    final["discovery_ranking_probability"] = final[
        "target_evidence_masked_fused_probability"
    ]
    final["fused_confidence_probability"] = sigmoid_numpy(confidence_logit)
    final["expert_weight_strict_discovery"] = discovery_weight[:, 0].astype("float32")
    final["expert_weight_cancer_discovery"] = discovery_weight[:, 1].astype("float32")
    final["expert_weight_strict"] = confidence_weight[:, 0].astype("float32")
    final["expert_weight_cancer"] = confidence_weight[:, 1].astype("float32")
    final["expert_weight_evidence"] = confidence_weight[:, 2].astype("float32")

    node = pd.read_parquet(GRAPH_NODE, columns=["canonical_id", "node_type", "log_degree"])
    node = node[node["node_type"].eq("lncRNA")].rename(columns={"canonical_id": "lncrna_id"})
    final = final.merge(node[["lncrna_id", "log_degree"]], on="lncrna_id", how="left")
    final["cold_start_status"] = np.where(
        final["cross_cancer_probability"].isna(),
        "strict_unavailable_patient_rescued",
        np.where(final["log_degree"].fillna(0).eq(0), "static_graph_cold_start", "static_graph_observed"),
    )
    direction = pd.to_numeric(final.get("cancer_direction_probability"), errors="coerce")
    final["direction"] = np.select(
        [direction.ge(0.55), direction.le(0.45)],
        ["positive", "negative"],
        default="uncertain",
    )
    final["availability"] = [
        {
            "cross_cancer": bool(strict_value),
            "cancer_native": bool(patient_value),
            "evidence_integrated": bool(evidence_value),
        }
        for strict_value, patient_value, evidence_value in zip(
            final["strict_available"],
            final["patient_available"],
            final["evidence_available"],
            strict=True,
        )
    ]
    final["prediction_scope"] = (
        "candidate_union_plus_independent_patient_native_plus_availability_masked_moe"
    )
    final = final.drop(columns=["log_degree"], errors="ignore")
    final = final.sort_values(KEYS).reset_index(drop=True)
    final_output = RELEASE / "final_three_probability_table.parquet"
    final.to_parquet(final_output, index=False, compression="zstd")

    rescued = final[
        final["strict_available"].eq(0) & final["patient_available"].eq(1)
    ].copy()
    rescued.to_parquet(
        RELEASE / "cancer_specific_rescued_candidates.parquet",
        index=False,
        compression="zstd",
    )
    gate_summary = {
        "completed_at": datetime.now().isoformat(),
        "status": "COMPLETED",
        "candidate_union_rows": int(len(candidate_union)),
        "patient_oof_rows": int(len(pd.read_parquet(oof_output, columns=["cancer_id"]))),
        "cancer_specific_pairs": int(len(adapter)),
        "final_pairs": int(len(final)),
        "unscored_candidate_union_rows": int(len(unscored)),
        "strict_unavailable_patient_rescued": int(len(rescued)),
        "simple_mean_removed": True,
        "patient_native_independent_of_strict": True,
        "moe_availability_mask": True,
        "moe_training_source": "OOF expert predictions only",
        "discovery_experts": discovery_metadata["expert_columns"],
        "confidence_experts": confidence_metadata["expert_columns"],
        "old_pf_direct_join_forbidden": True,
        "evidence_mapping": "many_to_many_exact_pathway_jaccard",
    }
    (RELEASE / "PROBABILITY_RELEASE_SUCCESS.json").write_text(
        json.dumps(gate_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
