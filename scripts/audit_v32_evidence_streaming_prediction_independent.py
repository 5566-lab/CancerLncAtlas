#!/usr/bin/env python3
"""Independently audit the 2026-08-30 streaming Evidence outputs.

This auditor deliberately does not import either Evidence training module.  It
recomputes row, fold, availability, direction and confidence-calibration facts
from the immutable Parquet products with DuckDB, and treats provenance as a
separate fail-closed publication gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import duckdb


FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_INDEPENDENT_AUDIT_V1"
EXPECTED_ANALYSIS = "CancerLncAtlas_V3.2_FULL_MULTITASK"
EXPECTED_ROWS = 3_300_000
EXPECTED_CANCERS = 33
EXPECTED_FOLDS = set(range(5))
FROZEN_PATIENT_RECEIPT_SHA256 = (
    "1ef32bda9d14d87997f3b81c34b8aacce172de390db5c29aee011ca73ba317e0"
)
FROZEN_SAMPLE_MAP_SHA256 = (
    "e05c20085be532159bb6a51f200a27923975c3caa6a90910290e10a5b60e6253"
)
FROZEN_PATIENT_LOGICAL_SHA256 = (
    "3c8bee3de5af3c5886ce8ef748c14471b449db5bbf6da886522e8adc540fa527"
)
CURRENT_G2_LINEAGE_FORMAT = (
    "CANCERLNCATLAS_V32_CURRENT_G2_CORE_EMBEDDING_LINEAGE_V1"
)
KEYS = ("cancer_id", "lncrna_id", "pathway_id")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def record_check(checks: dict[str, Any], name: str, passed: bool, **detail: Any) -> None:
    checks[name] = {"passed": bool(passed), **detail}


def inspect_stage(stage: Path, checks: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = stage / "STAGING_MANIFEST.json"
    manifest = read_json(manifest_path)
    declared = manifest.get("artifacts", {})
    hashes: dict[str, Any] = {}
    all_match = isinstance(declared, Mapping) and bool(declared)
    if isinstance(declared, Mapping):
        for relative, item in sorted(declared.items()):
            path = stage / str(relative)
            expected = str(item.get("sha256", "")) if isinstance(item, Mapping) else ""
            observed = sha256(path) if path.is_file() else None
            hashes[str(relative)] = {
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": observed,
                "declared_sha256": expected,
                "match": observed == expected,
            }
            all_match &= observed == expected
    record_check(checks, "stage_declared_artifact_hashes", all_match, artifacts=len(hashes))
    status = manifest.get("status")
    formal_preflight = (
        status == "PASS_FRESH_V32_EVIDENCE_STREAMING_PREFLIGHT"
        and manifest.get("real_6160707_server_preflight_executed") is True
        and manifest.get("server_accessed_by_this_materialization") is True
    )
    record_check(
        checks,
        "formal_preflight_receipt",
        formal_preflight,
        observed_status=status,
        real_6160707_server_preflight_executed=manifest.get(
            "real_6160707_server_preflight_executed"
        ),
        server_accessed_by_this_materialization=manifest.get(
            "server_accessed_by_this_materialization"
        ),
        interpretation=(
            "FULL_SCALE_STAGE_MATERIALIZATION_SUCCEEDED_BUT_NO_FORMAL_PREFLIGHT_PASS_RECEIPT"
        ),
    )
    candidate = manifest.get("inputs", {}).get("formal_candidates", {})
    candidate_path = str(candidate.get("path", ""))
    candidate_authoritative = (
        "/runtime/smoke/" not in candidate_path.replace("\\", "/")
        and isinstance(manifest.get("formal_candidate_authority"), Mapping)
        and bool(manifest.get("formal_candidate_authority", {}).get("receipt_sha256"))
    )
    record_check(
        checks,
        "formal_candidate_authority",
        candidate_authoritative,
        declared_path=candidate_path,
        sha256=candidate.get("sha256"),
        source_is_runtime_smoke="/runtime/smoke/" in candidate_path.replace("\\", "/"),
        authority_receipt_bound=isinstance(
            manifest.get("formal_candidate_authority"), Mapping
        ),
    )
    return manifest, {"manifest_sha256": sha256(manifest_path), "artifacts": hashes}


def inspect_core(core: Path, checks: dict[str, Any]) -> dict[str, Any]:
    path = core / "CORE_EMBEDDING_MANIFEST.json"
    payload = read_json(path)
    formal = payload.get("formal_lineage")
    strict_top = isinstance(formal, Mapping) and (
        formal.get("format") == CURRENT_G2_LINEAGE_FORMAT
        and formal.get("formal_graph_variant") == "G2"
        and formal.get("patient_fold_authority_receipt_sha256")
        == FROZEN_PATIENT_RECEIPT_SHA256
        and formal.get("sample_patient_fold_map_sha256") == FROZEN_SAMPLE_MAP_SHA256
        and formal.get("patient_authority_logical_sha256")
        == FROZEN_PATIENT_LOGICAL_SHA256
        and isinstance(formal.get("graph_authority_receipt_sha256"), str)
        and len(formal.get("graph_authority_receipt_sha256", "")) == 64
        and formal.get("sealed_test_opened") is False
    )
    folds: dict[str, Any] = {}
    all_exports_match = True
    all_fold_strict = strict_top
    for fold in range(5):
        item = payload.get("folds", {}).get(str(fold), {})
        exports: dict[str, Any] = {}
        for node_type in ("cancer", "lncRNA", "pathway"):
            declared = item.get("exports", {}).get(node_type, {})
            local = core / f"patient_fold={fold}" / f"{node_type}.parquet"
            observed = sha256(local) if local.is_file() else None
            expected = declared.get("sha256")
            exports[node_type] = {
                "path": str(local),
                "sha256": observed,
                "declared_sha256": expected,
                "match": observed == expected,
            }
            all_exports_match &= observed == expected
        fold_strict = strict_top and (
            item.get("formal_graph_variant") == "G2"
            and item.get("graph_authority_receipt_sha256")
            == formal.get("graph_authority_receipt_sha256")
            and item.get("patient_fold_authority_receipt_sha256")
            == FROZEN_PATIENT_RECEIPT_SHA256
            and isinstance(item.get("prepared_sha256"), str)
            and len(item.get("prepared_sha256", "")) == 64
        )
        all_fold_strict &= fold_strict
        folds[str(fold)] = {
            "checkpoint_path": item.get("checkpoint_path"),
            "checkpoint_sha256": item.get("checkpoint_sha256"),
            "prepared_path": item.get("prepared_path"),
            "prepared_sha256": item.get("prepared_sha256"),
            "formal_graph_variant": item.get("formal_graph_variant"),
            "artifact_hashes": item.get("artifact_hashes"),
            "strict_current_g2_lineage": fold_strict,
            "exports": exports,
        }
    record_check(checks, "core_export_hashes", all_exports_match, folds=5)
    record_check(
        checks,
        "fresh_current_g2_core_lineage",
        all_fold_strict,
        manifest_sha256=sha256(path),
        formal_lineage_present=isinstance(formal, Mapping),
        observed_checkpoint_parent=(
            payload.get("folds", {}).get("0", {}).get("checkpoint_path")
        ),
        missing_required_fields=(
            []
            if all_fold_strict
            else [
                "formal_lineage.format",
                "formal_lineage.formal_graph_variant=G2",
                "formal_lineage.graph_authority_receipt_sha256",
                "formal_lineage frozen patient authority hashes",
                "fold[*].formal_graph_variant=G2",
                "fold[*].graph/patient authority bindings",
            ]
        ),
    )
    return {
        "manifest_path": str(path),
        "manifest_sha256": sha256(path),
        "manifest_mtime": path.stat().st_mtime,
        "export_format": payload.get("export_format"),
        "module_id": payload.get("module_id"),
        "formal_lineage": formal,
        "folds": folds,
    }


def inspect_training(training: Path, checks: dict[str, Any]) -> dict[str, Any]:
    manifest_path = training / "TRAINING_MANIFEST.json"
    manifest = read_json(manifest_path)
    folds: dict[str, Any] = {}
    all_receipts = True
    for fold in range(5):
        receipt_path = training / f"patient_fold={fold}" / "HEAD_FIT.json"
        state_path = training / f"patient_fold={fold}" / "private_head_state.pt"
        receipt = read_json(receipt_path)
        state_hash = sha256(state_path)
        passed = (
            receipt.get("patient_fold") == fold
            and receipt.get("state_sha256") == state_hash
            and int(receipt.get("optimizer_steps", 0)) > 0
            and receipt.get("confidence_supervision_available") is True
            and receipt.get("direction_supervision_available") is True
            and int(receipt.get("train_missing_core", -1)) == 0
            and int(receipt.get("evaluation_missing_core", -1)) == 0
        )
        all_receipts &= passed
        folds[str(fold)] = {
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256(receipt_path),
            "state_path": str(state_path),
            "state_sha256": state_hash,
            "declared_state_sha256": receipt.get("state_sha256"),
            "final_parameter_sha256": receipt.get("final_parameter_sha256"),
            "optimizer_steps": receipt.get("optimizer_steps"),
            "train_examples": receipt.get("train_examples"),
            "evaluation_examples": receipt.get("evaluation_examples"),
            "history_epochs": len(receipt.get("history", [])),
            "core_embedding_lineage": receipt.get("core_embedding_lineage"),
            "receipt_contract_passed": passed,
        }
    record_check(checks, "training_receipts_and_state_hashes", all_receipts, folds=5)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "status": manifest.get("status"),
        "stage_validation": manifest.get("stage_validation"),
        "device": manifest.get("device"),
        "folds": folds,
    }


def inspect_predictions(
    con: duckdb.DuckDBPyConnection,
    stage: Path,
    training: Path,
    prediction: Path,
    checks: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = prediction / "PREDICTION_MANIFEST.json"
    manifest = read_json(manifest_path)
    pred_path = prediction / "EVIDENCE_PREDICTIONS.parquet"
    pred = f"read_parquet('{sql_path(pred_path)}')"
    availability_path = stage / "candidate_event_availability.parquet"
    availability = f"read_parquet('{sql_path(availability_path)}')"
    parts_glob = prediction / "pair_fold=*_predictions.parquet"
    parts = f"read_parquet('{sql_path(parts_glob)}')"
    summary = con.execute(
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT (cancer_id,lncrna_id,pathway_id)) AS unique_keys,
               count(DISTINCT cancer_id) AS cancers,
               count_if(availability) AS available_rows,
               count_if(NOT availability) AS unavailable_rows,
               count_if(availability AND (
                 evidence_confidence_probability IS NULL OR
                 NOT isfinite(evidence_confidence_probability) OR
                 evidence_confidence_probability<0 OR evidence_confidence_probability>1 OR
                 direction NOT IN ('negative','neutral','positive') OR
                 uncertainty IS NULL OR NOT isfinite(uncertainty) OR uncertainty<0 OR
                 event_count<1 OR event_count>64 OR
                 evidence_fold NOT BETWEEN 0 AND 4 OR
                 coalesce(unavailable_reason,'')<>'' OR coalesce(failure_reason,'')<>'')) AS bad_available,
               count_if(NOT availability AND (
                 evidence_confidence_probability IS NOT NULL OR direction IS NOT NULL OR
                 uncertainty IS NOT NULL OR event_count<>0 OR evidence_fold IS NOT NULL OR
                 unavailable_reason IS NULL OR unavailable_reason='' OR
                 failure_reason IS DISTINCT FROM unavailable_reason)) AS bad_unavailable,
               count_if(analysis_version<>? OR changes_primary_ranking IS DISTINCT FROM FALSE OR
                 main_ranking_modified IS DISTINCT FROM FALSE) AS bad_flags,
               avg(evidence_confidence_probability) FILTER(WHERE availability) AS confidence_mean,
               avg(uncertainty) FILTER(WHERE availability) AS uncertainty_mean
        FROM {pred}
        """,
        [EXPECTED_ANALYSIS],
    ).fetchone()
    names = [
        "rows", "unique_keys", "cancers", "available_rows", "unavailable_rows",
        "bad_available", "bad_unavailable", "bad_flags", "confidence_mean",
        "uncertainty_mean",
    ]
    values = dict(zip(names, summary, strict=True))
    semantic_pass = (
        int(values["rows"]) == EXPECTED_ROWS
        and int(values["unique_keys"]) == EXPECTED_ROWS
        and int(values["cancers"]) == EXPECTED_CANCERS
        and int(values["bad_available"]) == 0
        and int(values["bad_unavailable"]) == 0
        and int(values["bad_flags"]) == 0
    )
    record_check(checks, "prediction_row_and_typed_availability_semantics", semantic_pass, **values)
    alignment = con.execute(
        f"""
        SELECT
          (SELECT count(*)-count(DISTINCT (cancer_id,lncrna_id,pathway_id)) FROM {parts}),
          (SELECT count(*) FROM {parts} p ANTI JOIN {pred} x USING(cancer_id,lncrna_id,pathway_id)),
          (SELECT count(*) FROM {pred} x WHERE availability AND NOT EXISTS (
             SELECT 1 FROM {parts} p WHERE p.cancer_id=x.cancer_id
               AND p.lncrna_id=x.lncrna_id AND p.pathway_id=x.pathway_id)),
          (SELECT count(*) FROM {pred} x JOIN {parts} p USING(cancer_id,lncrna_id,pathway_id)
             WHERE NOT x.availability OR
               abs(x.evidence_confidence_probability-p.evidence_confidence_probability)>1e-12 OR
               x.direction IS DISTINCT FROM p.direction OR
               abs(x.uncertainty-p.uncertainty)>1e-12 OR
               x.event_count IS DISTINCT FROM p.event_count OR
               cast(x.evidence_fold AS BIGINT) IS DISTINCT FROM p.evidence_fold),
          (SELECT count(*) FROM {pred} x FULL JOIN {availability} a
             USING(cancer_id,lncrna_id,pathway_id) WHERE x.cancer_id IS NULL OR a.cancer_id IS NULL),
          (SELECT count(*) FROM {pred} x JOIN {availability} a
             USING(cancer_id,lncrna_id,pathway_id)
             WHERE x.availability IS DISTINCT FROM a.event_bag_available OR
               x.event_count IS DISTINCT FROM least(a.event_count,64))
        """
    ).fetchone()
    alignment_names = (
        "partition_duplicate_keys", "partition_missing_full", "available_missing_partition",
        "partition_value_drift", "candidate_key_symmetric_difference",
        "candidate_availability_or_count_drift",
    )
    alignment_values = dict(zip(alignment_names, map(int, alignment), strict=True))
    record_check(
        checks,
        "prediction_partition_and_stage_alignment",
        all(value == 0 for value in alignment_values.values()),
        **alignment_values,
    )
    reasons = con.execute(
        f"SELECT availability,unavailable_reason,count(*) AS row_count FROM {pred} GROUP BY ALL ORDER BY 1,2"
    ).fetchall()
    stage_reason = (
        read_json(stage / "STAGING_MANIFEST.json")
        .get("typed_unavailable_contract", {})
        .get("no_event_failure_reason")
    )
    observed_reason = next((row[1] for row in reasons if row[0] is False), None)
    record_check(
        checks,
        "typed_unavailable_reason_contract",
        observed_reason == stage_reason,
        stage_declared_reason=stage_reason,
        prediction_observed_reason=observed_reason,
        counts=[{"availability": row[0], "reason": row[1], "rows": row[2]} for row in reasons],
    )
    direction_counts = dict(
        con.execute(
            f"SELECT direction,count(*) FROM {pred} WHERE availability GROUP BY 1 ORDER BY 1"
        ).fetchall()
    )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "predictions_path": str(pred_path),
        "predictions_sha256": sha256(pred_path),
        "declared_predictions_sha256": manifest.get("predictions_sha256"),
        "summary": values,
        "direction_counts": direction_counts,
        "alignment": alignment_values,
        "unavailable_reasons": [
            {"availability": row[0], "reason": row[1], "rows": row[2]} for row in reasons
        ],
    }


def inspect_folds_and_metrics(
    con: duckdb.DuckDBPyConnection,
    stage: Path,
    training: Path,
    prediction: Path,
    checks: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    events_path = stage / "candidate_exact_events_pair_folded.parquet"
    pred_path = prediction / "EVIDENCE_PREDICTIONS.parquet"
    events = f"read_parquet('{sql_path(events_path)}')"
    pred = f"read_parquet('{sql_path(pred_path)}')"
    fold_rows = con.execute(
        f"SELECT leakage_fold,count(*),count(DISTINCT (cancer_id,lncrna_id,pathway_id)) "
        f"FROM {events} GROUP BY 1 ORDER BY 1"
    ).fetchall()
    pair_cross_fold = con.execute(
        f"SELECT count(*) FROM (SELECT lncrna_id,pathway_id FROM {events} "
        "GROUP BY 1,2 HAVING count(DISTINCT leakage_fold)<>1)"
    ).fetchone()[0]
    per_fold: dict[str, Any] = {}
    all_fold_counts = int(pair_cross_fold) == 0
    for fold, event_rows, evaluation_examples in fold_rows:
        receipt = read_json(training / f"patient_fold={int(fold)}" / "HEAD_FIT.json")
        train_exclusions = stage / f"pair_fold={int(fold)}" / "train_provenance_exclusions.parquet"
        exclusions = f"read_parquet('{sql_path(train_exclusions)}')"
        train_examples = con.execute(
            f"SELECT count(*) FROM (SELECT DISTINCT e.cancer_id,e.lncrna_id,e.pathway_id "
            f"FROM {events} e ANTI JOIN {exclusions} x USING(event_id) "
            f"WHERE e.leakage_fold<>{int(fold)})"
        ).fetchone()[0]
        count_match = (
            int(receipt.get("train_examples", -1)) == int(train_examples)
            and int(receipt.get("evaluation_examples", -1)) == int(evaluation_examples)
        )
        all_fold_counts &= count_match
        per_fold[str(int(fold))] = {
            "event_rows": int(event_rows),
            "evaluation_examples": int(evaluation_examples),
            "declared_evaluation_examples": receipt.get("evaluation_examples"),
            "train_examples_after_exclusion": int(train_examples),
            "declared_train_examples": receipt.get("train_examples"),
            "counts_match": count_match,
        }
    record_check(
        checks,
        "pair_blocked_fold_and_training_example_counts",
        all_fold_counts,
        pair_cross_fold_count=int(pair_cross_fold),
        folds=per_fold,
    )

    cap_row = con.execute(
        f"""
        WITH bags AS (
          SELECT cancer_id,lncrna_id,pathway_id,count(*) AS raw_event_count,
                 count(DISTINCT event_id) AS unique_event_ids,
                 count_if(event_id IS NULL OR event_id='') AS null_event_ids
          FROM {events} GROUP BY 1,2,3
        )
        SELECT count_if(raw_event_count>64) AS capped_bags,
               sum(greatest(raw_event_count-64,0)) AS truncated_event_rows,
               max(raw_event_count) AS max_raw_event_count,
               count_if(unique_event_ids<>raw_event_count OR null_event_ids<>0)
                 AS nondeterministic_key_bags
        FROM bags
        """
    ).fetchone()
    cap = {
        "capped_bags": int(cap_row[0]),
        "truncated_event_rows": int(cap_row[1]),
        "max_raw_event_count": int(cap_row[2]),
        "nondeterministic_key_bags": int(cap_row[3]),
    }
    repo = Path(__file__).resolve().parents[1]
    stream_code_path = repo / "cc_hhgt" / "v32" / "evidence_streaming_training.py"
    train_runner_path = repo / "scripts" / "run_v32_evidence_streaming_training.py"
    predict_runner_path = repo / "scripts" / "run_v32_evidence_streaming_predict.py"
    stream_code = stream_code_path.read_text(encoding="utf-8")
    train_runner = train_runner_path.read_text(encoding="utf-8")
    predict_runner = predict_runner_path.read_text(encoding="utf-8")
    shared_deterministic_path = (
        "row_number() OVER (PARTITION BY " in stream_code
        and "ORDER BY event_id" in stream_code
        and "iter_fold_trainer_batches(" in train_runner
        and "iter_fold_trainer_batches(" in predict_runner
        and cap["nondeterministic_key_bags"] == 0
    )
    record_check(
        checks,
        "event_cap_deterministic_and_shared_by_training_prediction",
        shared_deterministic_path,
        **cap,
        cap=64,
        selection_order="event_id ASC",
        prediction_event_count="min(raw_event_count,64)",
        stream_code_sha256=sha256(stream_code_path),
        training_runner_sha256=sha256(train_runner_path),
        prediction_runner_sha256=sha256(predict_runner_path),
    )
    # The stable hash-like event_id order prevents run-to-run drift, but it is
    # not a quality-aware choice (for example direct experimental evidence is
    # not ranked ahead of computational evidence).  Preserve that distinction.
    record_check(
        checks,
        "event_cap_quality_priority",
        cap["capped_bags"] == 0,
        **cap,
        selection_order="event_id ASC_ONLY",
        quality_columns_in_selection_order=[],
        risk=(
            "DETERMINISTIC_BUT_NOT_QUALITY_PRIORITIZED; EVENTS_AFTER_FIRST_64_ARE_IGNORED"
        ),
    )

    metric_row = con.execute(
        f"""
        WITH ranked AS (
          SELECT *,row_number() OVER(PARTITION BY cancer_id,lncrna_id,pathway_id
                                     ORDER BY event_id) AS rn
          FROM {events}
        ), bags AS (
          SELECT cancer_id,lncrna_id,pathway_id,count(*) AS event_count,
                 avg(confidence_target) FILTER(WHERE isfinite(confidence_target)) AS target_confidence,
                 count_if(direction_target=0) AS n_negative,
                 count_if(direction_target=1) AS n_neutral,
                 count_if(direction_target=2) AS n_positive
          FROM ranked WHERE rn<=64 GROUP BY 1,2,3
        ), joined AS (
          SELECT p.*,b.target_confidence,
            CASE
              WHEN greatest(n_negative,n_neutral,n_positive)=0 THEN NULL
              WHEN cast(n_negative=greatest(n_negative,n_neutral,n_positive) AS INT)+
                   cast(n_neutral=greatest(n_negative,n_neutral,n_positive) AS INT)+
                   cast(n_positive=greatest(n_negative,n_neutral,n_positive) AS INT)<>1 THEN NULL
              WHEN n_negative=greatest(n_negative,n_neutral,n_positive) THEN 'negative'
              WHEN n_neutral=greatest(n_negative,n_neutral,n_positive) THEN 'neutral'
              ELSE 'positive'
            END AS target_direction
          FROM {pred} p JOIN bags b USING(cancer_id,lncrna_id,pathway_id)
          WHERE p.availability
        )
        SELECT count(*) AS available_rows,
               count(target_direction) AS direction_labeled_rows,
               count_if(direction=target_direction) FILTER(WHERE target_direction IS NOT NULL)
                 AS direction_correct_rows,
               avg(cast(direction=target_direction AS INT))
                 FILTER(WHERE target_direction IS NOT NULL) AS direction_accuracy,
               count(target_confidence) AS confidence_labeled_rows,
               avg(target_confidence) FILTER(WHERE target_confidence IS NOT NULL)
                 AS target_confidence_mean,
               avg(evidence_confidence_probability) FILTER(WHERE target_confidence IS NOT NULL)
                 AS predicted_confidence_mean,
               avg(pow(evidence_confidence_probability-target_confidence,2))
                 FILTER(WHERE target_confidence IS NOT NULL) AS confidence_brier,
               avg(abs(evidence_confidence_probability-target_confidence))
                 FILTER(WHERE target_confidence IS NOT NULL) AS confidence_mae,
               corr(evidence_confidence_probability,target_confidence)
                 FILTER(WHERE target_confidence IS NOT NULL) AS confidence_correlation,
               regr_slope(target_confidence,evidence_confidence_probability)
                 FILTER(WHERE target_confidence IS NOT NULL) AS calibration_slope,
               regr_intercept(target_confidence,evidence_confidence_probability)
                 FILTER(WHERE target_confidence IS NOT NULL) AS calibration_intercept
        FROM joined
        """
    ).fetchone()
    metric_names = (
        "available_rows", "direction_labeled_rows", "direction_correct_rows",
        "direction_accuracy", "confidence_labeled_rows", "target_confidence_mean",
        "predicted_confidence_mean", "confidence_brier", "confidence_mae",
        "confidence_correlation", "calibration_slope", "calibration_intercept",
    )
    metrics = dict(zip(metric_names, metric_row, strict=True))
    metrics["event_cap"] = cap
    confusion_rows = con.execute(
        f"""
        WITH ranked AS (
          SELECT *,row_number() OVER(PARTITION BY cancer_id,lncrna_id,pathway_id
                                     ORDER BY event_id) AS rn FROM {events}
        ), bags AS (
          SELECT cancer_id,lncrna_id,pathway_id,
                 count_if(direction_target=0) n0,count_if(direction_target=1) n1,
                 count_if(direction_target=2) n2 FROM ranked WHERE rn<=64 GROUP BY 1,2,3
        ), truth AS (
          SELECT *,CASE WHEN greatest(n0,n1,n2)=0 OR
             cast(n0=greatest(n0,n1,n2) AS INT)+cast(n1=greatest(n0,n1,n2) AS INT)+
             cast(n2=greatest(n0,n1,n2) AS INT)<>1 THEN NULL
             WHEN n0=greatest(n0,n1,n2) THEN 'negative'
             WHEN n1=greatest(n0,n1,n2) THEN 'neutral' ELSE 'positive' END target_direction
          FROM bags
        )
        SELECT target_direction,direction,count(*) FROM {pred} p JOIN truth t
          USING(cancer_id,lncrna_id,pathway_id)
        WHERE p.availability AND target_direction IS NOT NULL GROUP BY 1,2 ORDER BY 1,2
        """
    ).fetchall()
    metrics["direction_confusion"] = [
        {"target": row[0], "predicted": row[1], "rows": int(row[2])}
        for row in confusion_rows
    ]
    # Metrics are descriptive.  No acceptance threshold was preregistered, so
    # a finite metric is not silently relabelled as calibrated.
    finite = all(
        value is None or (isinstance(value, (int, float)) and math.isfinite(float(value)))
        for key, value in metrics.items()
        if key not in {"direction_confusion", "event_cap"}
    )
    record_check(
        checks,
        "direction_recovery_and_confidence_metrics_recomputed",
        finite and int(metrics["available_rows"]) > 0,
        preregistered_acceptance_threshold_present=False,
        verdict="DESCRIPTIVE_ONLY_NOT_A_CALIBRATION_PASS",
    )
    return per_fold, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--server-observation", type=Path)
    args = parser.parse_args()
    stage, training, prediction, core = (
        args.stage_root.resolve(), args.training_root.resolve(),
        args.prediction_root.resolve(), args.core_root.resolve(),
    )
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Audit refuses output reuse: {output}")
    checks: dict[str, Any] = {}
    stage_manifest, stage_hashes = inspect_stage(stage, checks)
    core_report = inspect_core(core, checks)
    training_report = inspect_training(training, checks)
    con = duckdb.connect(":memory:")
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")
    try:
        prediction_report = inspect_predictions(con, stage, training, prediction, checks)
        folds, metrics = inspect_folds_and_metrics(
            con, stage, training, prediction, checks
        )
    finally:
        con.close()
    server = None
    if args.server_observation:
        server = read_json(args.server_observation.resolve())
        local_expected = {
            "PREDICTION_MANIFEST.json": prediction_report["manifest_sha256"],
            "EVIDENCE_PREDICTIONS.parquet": prediction_report["predictions_sha256"],
            "TRAINING_MANIFEST.json": training_report["manifest_sha256"],
        }
        observed = server.get("sha256", {})
        server_match = all(observed.get(name) == value for name, value in local_expected.items())
        record_check(
            checks, "server_copy_critical_hashes", server_match,
            expected=local_expected, observed={name: observed.get(name) for name in local_expected},
            path_layout_matches_completion=server.get("path_layout_matches_completion"),
        )
    technical_failed = sorted(
        name for name, item in checks.items()
        if name not in {
            "formal_preflight_receipt", "formal_candidate_authority",
            "fresh_current_g2_core_lineage", "typed_unavailable_reason_contract",
            "event_cap_quality_priority",
            "direction_recovery_and_confidence_metrics_recomputed",
        }
        and item.get("passed") is not True
    )
    publication_blockers = sorted(
        name for name in (
            "formal_preflight_receipt", "formal_candidate_authority",
            "fresh_current_g2_core_lineage", "typed_unavailable_reason_contract",
            "event_cap_quality_priority",
        )
        if checks.get(name, {}).get("passed") is not True
    )
    publication_allowed = not technical_failed and not publication_blockers
    status = (
        "PASS_INDEPENDENT_AUDIT_PUBLICATION_ALLOWED"
        if publication_allowed
        else "BLOCKED_INDEPENDENT_AUDIT_NOT_PUBLISHABLE"
    )
    report = {
        "format": FORMAT,
        "status": status,
        "analysis_version": EXPECTED_ANALYSIS,
        "auditor_independent_of_training_implementation": True,
        "training_module_imported": False,
        "universe_rows": int(prediction_report["summary"]["rows"]),
        "prediction_rows": int(prediction_report["summary"]["rows"]),
        "available_rows": int(prediction_report["summary"]["available_rows"]),
        "unavailable_rows": int(prediction_report["summary"]["unavailable_rows"]),
        "technical_acceptance": not technical_failed,
        "technical_prediction_pass": not technical_failed,
        "publication_allowed": publication_allowed,
        "closure_allowed": publication_allowed,
        "technical_failures": technical_failed,
        "publication_blockers": publication_blockers,
        "checks": checks,
        "stage": {
            "root": str(stage),
            "status": stage_manifest.get("status"),
            **stage_hashes,
        },
        "core_embedding_lineage": core_report,
        "training": training_report,
        "prediction": prediction_report,
        "folds": folds,
        "direction_recovery_and_confidence_calibration": metrics,
        "server_copy": server,
        "required_remediation": [
            "Materialize and train the current patient-first G0/G1/G2 arms, then export a hash-bound fresh G2 core.",
            "Bind the fresh G2 core to the frozen patient authority and formal graph authority receipt for every fold.",
            "Replace the runtime/smoke candidate source with a formal candidate authority receipt, or independently prove and bind equivalence.",
            "Issue a separate formal preflight PASS receipt; stage materialization READY is not a preflight PASS.",
            "Retrain and repredict Evidence from the corrected authorities; do not relabel this run in place.",
            "Unify the no-event typed reason and preregister direction/calibration acceptance thresholds before publication.",
            "Define and preregister a quality-aware top-64 event policy; event_id order is deterministic but not evidence-quality priority.",
        ],
        "production_deployed": False,
        "port_8260_touched": False,
        "sealed_test_opened": False,
    }
    output.mkdir(parents=True)
    audit_path = output / "AUDIT.json"
    audit_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sums: list[tuple[str, str]] = [(sha256(audit_path), "AUDIT.json")]
    for path in (
        stage / "STAGING_MANIFEST.json",
        core / "CORE_EMBEDDING_MANIFEST.json",
        training / "TRAINING_MANIFEST.json",
        prediction / "PREDICTION_MANIFEST.json",
        prediction / "EVIDENCE_PREDICTIONS.parquet",
    ):
        sums.append((sha256(path), str(path)))
    (output / "SHA256SUMS.tsv").write_text(
        "\n".join(f"{digest}\t{name}" for digest, name in sums) + "\n",
        encoding="utf-8",
    )
    success = {
        "status": status,
        "publication_allowed": publication_allowed,
        "closure_allowed": publication_allowed,
        "audit_path": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "prediction_sha256": prediction_report["predictions_sha256"],
        "publication_blockers": publication_blockers,
    }
    (output / "RESULT.json").write_text(
        json.dumps(success, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(success, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
