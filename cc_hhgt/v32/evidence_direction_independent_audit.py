"""Independent post-audit for V3.2 Evidence direction probabilities.

This module deliberately does not import the materializer.  It re-reads every
declared authority, checkpoint, partition and public row from disk and emits a
separate hash-bound audit decision.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import duckdb
import numpy as np


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
RELEASE_FORMAT = "CC_HHGT_V3_2_EVIDENCE_DIRECTION_PROBABILITY_BINDING_V1"
RELEASE_STATUS = "SUCCESS_V32_CHECKPOINT_REINFERENCE"
OUTPUT_FORMAT = "CC_HHGT_V3_2_EVIDENCE_DIRECTION_PROBABILITIES_V1"
EVIDENCE_BINDING_FORMAT = "CC_HHGT_V3_2_EVIDENCE_OUTPUT_BINDING_V1"
EVIDENCE_BINDING_STATUS = "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND"
CHECKPOINT_FORMAT = "CC_HHGT_V3_2_PRIVATE_EVIDENCE_EVENTSET_V1"
AUDIT_FORMAT = "CC_HHGT_V3_2_EVIDENCE_DIRECTION_INDEPENDENT_AUDIT_V1"
AUDIT_BINDING_FORMAT = (
    "CC_HHGT_V3_2_EVIDENCE_DIRECTION_INDEPENDENT_AUDIT_BINDING_V1"
)
AUDIT_STATUS = "PASS_INDEPENDENT_V32_EVIDENCE_DIRECTION_AUDIT"
EXACT_KEYS = ("cancer_id", "lncrna_id", "pathway_id")
PROBABILITY_COLUMNS = (
    "direction_negative_probability",
    "direction_neutral_probability",
    "direction_positive_probability",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvidenceDirectionIndependentAuditError(RuntimeError):
    """Raised when the independently observed release is not publishable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceDirectionIndependentAuditError(
            f"Cannot read {label} JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise EvidenceDirectionIndependentAuditError(f"{label} is not a JSON object")
    return value


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _checked_file(
    declaration: Mapping[str, Any], *, parent: Path, label: str
) -> tuple[Path, str]:
    if not isinstance(declaration, Mapping):
        raise EvidenceDirectionIndependentAuditError(f"Missing {label} declaration")
    expected = str(declaration.get("sha256", "")).lower()
    if not _SHA256.fullmatch(expected):
        raise EvidenceDirectionIndependentAuditError(f"Invalid {label} SHA256")
    declared = Path(str(declaration.get("path", "")))
    path = declared.resolve() if declared.is_absolute() else (parent / declared).resolve()
    if path.is_symlink() or not path.is_file():
        raise EvidenceDirectionIndependentAuditError(f"Missing/unsafe {label}: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise EvidenceDirectionIndependentAuditError(
            f"{label} SHA256 mismatch: {observed} != {expected}"
        )
    return path, observed


def _require_fields(
    value: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    for key, wanted in expected.items():
        observed = value.get(key)
        if observed != wanted or type(observed) is not type(wanted):
            raise EvidenceDirectionIndependentAuditError(
                f"{label} violates {key}: {observed!r} != {wanted!r}"
            )


def _state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not hasattr(tensor, "detach"):
            raise EvidenceDirectionIndependentAuditError(
                f"Checkpoint state {name!r} is not a tensor"
            )
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''").replace("\\", "/")


def _expect_zero(name: str, value: Any) -> None:
    if int(value) != 0:
        raise EvidenceDirectionIndependentAuditError(
            f"Independent semantic check failed: {name}={value}"
        )


def audit_evidence_direction_release(
    *,
    binding_path: str | Path,
    expected_binding_sha256: str,
    output_root: str | Path,
    expected_total_rows: int = 3_300_000,
    expected_available_rows: int = 825_753,
    expected_cancer_count: int = 33,
) -> dict[str, Any]:
    """Audit a completed release without trusting materializer code or summaries."""

    source = Path(binding_path).resolve()
    expected_binding_sha256 = str(expected_binding_sha256).lower()
    if (
        not _SHA256.fullmatch(expected_binding_sha256)
        or source.is_symlink()
        or not source.is_file()
        or _sha256(source) != expected_binding_sha256
    ):
        raise EvidenceDirectionIndependentAuditError("Release binding SHA256 mismatch")
    release = _read_json(source, "release binding")
    _require_fields(
        release,
        {
            "format": RELEASE_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": RELEASE_STATUS,
            "training_performed": False,
            "source_checkpoints_newly_trained_v32": True,
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "historical_rankings_used": False,
            "primary_ranking_unchanged": True,
            "discovery_ranking_unchanged": True,
            "used_for_fusion": False,
            "published_argmax_is_original_inference": True,
            "recovered_argmax_is_fixed_seed_reinference": True,
            "argmax_mismatch_is_not_checkpoint_drift": True,
            "stochastic_reinference_policy": (
                "FIXED_SEED_MC_DROPOUT_FROM_HASH_PINNED_V32_PRIVATE_HEADS"
            ),
            "production_deployed": False,
            "release_ready": False,
        },
        "release binding",
    )
    parameters = release.get("parameters")
    if not isinstance(parameters, Mapping):
        raise EvidenceDirectionIndependentAuditError("Release lacks inference parameters")
    if (
        parameters.get("mc_samples") != 16
        or float(parameters.get("dropout", math.nan)) != 0.20
        or parameters.get("max_events") != 64
        or isinstance(parameters.get("inference_seed"), bool)
        or not isinstance(parameters.get("inference_seed"), int)
        or int(parameters.get("inference_seed")) < 0
        or isinstance(parameters.get("batch_size"), bool)
        or not isinstance(parameters.get("batch_size"), int)
        or int(parameters.get("batch_size")) < 1
    ):
        raise EvidenceDirectionIndependentAuditError(
            "Release inference parameters are not the formal fixed-seed V3.2 policy"
        )

    authority = release.get("authority")
    if not isinstance(authority, Mapping):
        raise EvidenceDirectionIndependentAuditError("Release lacks authorities")
    evidence_binding_path, evidence_binding_sha = _checked_file(
        {
            "path": authority.get("evidence_binding_path"),
            "sha256": authority.get("evidence_binding_sha256"),
        },
        parent=source.parent,
        label="V3.2 Evidence binding",
    )
    evidence_binding = _read_json(evidence_binding_path, "V3.2 Evidence binding")
    _require_fields(
        evidence_binding,
        {
            "format": EVIDENCE_BINDING_FORMAT,
            "status": EVIDENCE_BINDING_STATUS,
            "analysis_version": ANALYSIS_VERSION,
            "five_fresh_private_heads_verified": True,
            "historical_checkpoints_used": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "family_to_exact_broadcast": False,
            "production_deployed": False,
            "release_ready": False,
        },
        "V3.2 Evidence binding",
    )
    original_path, original_sha = _checked_file(
        evidence_binding.get("artifacts", {}).get("evidence_predictions", {}),
        parent=evidence_binding_path.parent,
        label="original current-V3.2 Evidence predictions",
    )
    event_path, event_sha = _checked_file(
        evidence_binding.get("artifacts", {}).get("event_lineage", {}),
        parent=evidence_binding_path.parent,
        label="current-V3.2 event lineage",
    )
    if (
        authority.get("original_predictions_sha256") != original_sha
        or authority.get("event_lineage_sha256") != event_sha
    ):
        raise EvidenceDirectionIndependentAuditError(
            "Release authorities differ from the current V3.2 Evidence binding"
        )
    core_path, core_sha = _checked_file(
        {
            "path": authority.get("core_manifest_path"),
            "sha256": authority.get("core_manifest_sha256"),
        },
        parent=source.parent,
        label="V3.2 core embedding manifest",
    )
    training_manifest_path, training_manifest_sha = _checked_file(
        evidence_binding.get("authorities", {}).get("training_manifest", {}),
        parent=evidence_binding_path.parent,
        label="V3.2 Evidence training manifest",
    )
    training_manifest = _read_json(training_manifest_path, "training manifest")
    _require_fields(
        training_manifest,
        {
            "analysis_version": ANALYSIS_VERSION,
            "module_id": "evidence_private_eventset",
            "status": "SUCCESS_NEWLY_TRAINED",
            "training_status": "SUCCESS_NEWLY_TRAINED",
            "training_generation": "V3.2",
            "checkpoint_format": CHECKPOINT_FORMAT,
            "historical_evidence_checkpoint_loaded": False,
            "historical_evidence_result_loaded": False,
            "old_confidence_loaded": False,
            "family_to_exact_broadcast_used": False,
            "core_frozen": True,
            "core_detached": True,
            "main_ranking_modified": False,
            "prediction_role": "AUXILIARY_CONFIDENCE_ONLY",
        },
        "training manifest",
    )

    partitions = release.get("partitions")
    if not isinstance(partitions, list) or len(partitions) != 5:
        raise EvidenceDirectionIndependentAuditError("Release must bind exactly five folds")
    partition_paths: list[Path] = []
    checkpoint_hashes: dict[str, str] = {}
    torch = __import__("torch")
    for expected_fold, declaration in enumerate(sorted(partitions, key=lambda item: item["fold"])):
        if declaration.get("fold") != expected_fold:
            raise EvidenceDirectionIndependentAuditError("Fold declarations are incomplete")
        part_path, _ = _checked_file(
            declaration,
            parent=source.parent,
            label=f"fold {expected_fold} probability partition",
        )
        partition_paths.append(part_path)
        part_con = duckdb.connect(":memory:")
        try:
            part_summary = part_con.execute(
                f"""
                SELECT count(*),
                       count(DISTINCT (cancer_id,lncrna_id,pathway_id)),
                       count_if(evidence_fold <> ? OR event_count < 1 OR
                         recovered_direction NOT IN ('negative','neutral','positive') OR
                         direction_negative_probability IS NULL OR
                         direction_neutral_probability IS NULL OR
                         direction_positive_probability IS NULL OR
                         NOT isfinite(direction_negative_probability) OR
                         NOT isfinite(direction_neutral_probability) OR
                         NOT isfinite(direction_positive_probability) OR
                         least(direction_negative_probability,
                               direction_neutral_probability,
                               direction_positive_probability) < 0 OR
                         greatest(direction_negative_probability,
                                  direction_neutral_probability,
                                  direction_positive_probability) > 1 OR
                         abs(direction_negative_probability +
                             direction_neutral_probability +
                             direction_positive_probability - 1) > 1e-5)
                FROM read_parquet('{_sql_path(part_path)}')
                """,
                [expected_fold],
            ).fetchone()
        finally:
            part_con.close()
        if (
            tuple(map(int, part_summary))
            != (int(declaration.get("rows", -1)), int(declaration.get("rows", -1)), 0)
        ):
            raise EvidenceDirectionIndependentAuditError(
                f"Fold {expected_fold} partition semantics failed: {part_summary}"
            )
        checkpoint_path, checkpoint_sha = _checked_file(
            {
                "path": declaration.get("checkpoint_path"),
                "sha256": declaration.get("checkpoint_sha256"),
            },
            parent=source.parent,
            label=f"fold {expected_fold} checkpoint",
        )
        bound_checkpoint = evidence_binding.get("checkpoints", {}).get(str(expected_fold), {})
        manifest_fold = training_manifest.get("folds", {}).get(str(expected_fold), {})
        if (
            checkpoint_sha != bound_checkpoint.get("sha256")
            or checkpoint_sha != manifest_fold.get("checkpoint_sha256")
            or declaration.get("rows") != manifest_fold.get("n_heldout_bags")
            or declaration.get("final_parameter_sha256")
            != bound_checkpoint.get("final_parameter_sha256")
        ):
            raise EvidenceDirectionIndependentAuditError(
                f"Fold {expected_fold} is not aligned to formal V3.2 lineage"
            )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _require_fields(
            payload,
            {
                "checkpoint_format": CHECKPOINT_FORMAT,
                "patient_fold": expected_fold,
                "private_parameters_fresh_init": True,
                "initialized_from_checkpoint": False,
                "historical_evidence_checkpoint_allowed": False,
                "historical_evidence_result_allowed": False,
                "contains_core_parameters": False,
                "contains_primary_ranking_parameters": False,
                "direction_supervision_available": True,
                "core_manifest_sha256": core_sha,
            },
            f"fold {expected_fold} checkpoint",
        )
        state = payload.get("private_model_state")
        if not isinstance(state, Mapping):
            raise EvidenceDirectionIndependentAuditError(
                f"Fold {expected_fold} checkpoint lacks private state"
            )
        state_sha = _state_sha256(state)
        if (
            state_sha != payload.get("final_parameter_sha256")
            or state_sha != declaration.get("final_parameter_sha256")
            or int(payload.get("optimizer_steps", 0)) < 1
            or payload.get("initial_parameter_sha256") == state_sha
        ):
            raise EvidenceDirectionIndependentAuditError(
                f"Fold {expected_fold} model state lineage failed"
            )
        checkpoint_hashes[str(expected_fold)] = checkpoint_sha

    artifact_path, artifact_sha = _checked_file(
        release.get("artifact", {}), parent=source.parent, label="public direction artifact"
    )
    artifact = release.get("artifact", {})
    declared_counts = release.get("counts", {})
    if (
        artifact.get("rows") != expected_total_rows
        or declared_counts.get("total_rows") != expected_total_rows
        or declared_counts.get("available_rows") != expected_available_rows
        or declared_counts.get("unavailable_rows")
        != expected_total_rows - expected_available_rows
        or declared_counts.get("cancers") != expected_cancer_count
        or declared_counts.get("probability_sum_violations") != 0
        or declared_counts.get("null_policy_violations") != 0
    ):
        raise EvidenceDirectionIndependentAuditError("Release count declarations are invalid")

    parts_glob = str(partition_paths[0].parent / "*.parquet")
    using_keys = "(" + ", ".join(EXACT_KEYS) + ")"
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET threads=4")
        con.execute("SET memory_limit='8GB'")
        table = f"read_parquet('{_sql_path(artifact_path)}')"
        original = f"read_parquet('{_sql_path(original_path)}')"
        parts = f"read_parquet('{_sql_path(Path(parts_glob))}')"
        required_columns = {
            *EXACT_KEYS,
            "published_direction",
            "recovered_direction",
            *PROBABILITY_COLUMNS,
            "direction_entropy",
            "direction_probability_available",
            "direction_probability_unavailable_reason",
            "published_direction_matches_recovered_argmax",
            "evidence_confidence_available",
            "evidence_fold",
            "event_count",
            "analysis_version",
            "output_format",
            "changes_primary_ranking",
            "changes_discovery_ranking",
            "used_for_fusion",
            "historical_checkpoint_used",
            "historical_prediction_used",
        }
        columns = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()}
        if missing := sorted(required_columns - columns):
            raise EvidenceDirectionIndependentAuditError(
                f"Public direction artifact lacks columns: {missing}"
            )
        summary = con.execute(
            f"""
            SELECT
              count(*), count(DISTINCT (cancer_id, lncrna_id, pathway_id)),
              count(DISTINCT cancer_id), count_if(direction_probability_available),
              count_if(NOT direction_probability_available),
              count_if(direction_probability_available AND (
                direction_negative_probability IS NULL OR
                direction_neutral_probability IS NULL OR
                direction_positive_probability IS NULL OR
                NOT isfinite(direction_negative_probability) OR
                NOT isfinite(direction_neutral_probability) OR
                NOT isfinite(direction_positive_probability) OR
                least(direction_negative_probability, direction_neutral_probability,
                      direction_positive_probability) < 0 OR
                greatest(direction_negative_probability, direction_neutral_probability,
                         direction_positive_probability) > 1 OR
                abs(direction_negative_probability + direction_neutral_probability +
                    direction_positive_probability - 1) > 1e-5 OR
                recovered_direction NOT IN ('negative','neutral','positive') OR
                direction_entropy IS NULL OR direction_entropy < 0 OR direction_entropy > 1 OR
                abs(direction_entropy - (-(
                  direction_negative_probability * ln(greatest(direction_negative_probability,1e-12)) +
                  direction_neutral_probability * ln(greatest(direction_neutral_probability,1e-12)) +
                  direction_positive_probability * ln(greatest(direction_positive_probability,1e-12))
                ) / ln(3.0))) > 2e-5 OR
                direction_probability_unavailable_reason IS NOT NULL OR
                published_direction NOT IN ('negative','neutral','positive') OR
                evidence_fold NOT BETWEEN 0 AND 4 OR event_count < 1 OR
                evidence_confidence_available IS DISTINCT FROM TRUE)),
              count_if(NOT direction_probability_available AND (
                direction_negative_probability IS NOT NULL OR
                direction_neutral_probability IS NOT NULL OR
                direction_positive_probability IS NOT NULL OR
                recovered_direction IS NOT NULL OR direction_entropy IS NOT NULL OR
                published_direction_matches_recovered_argmax IS NOT NULL OR
                evidence_fold IS NOT NULL OR event_count IS NOT NULL OR
                direction_probability_unavailable_reason IS NULL OR
                direction_probability_unavailable_reason = '')),
              count_if(direction_probability_available AND
                published_direction_matches_recovered_argmax IS DISTINCT FROM
                  (published_direction = recovered_direction)),
              count_if(direction_probability_available AND
                CASE recovered_direction
                  WHEN 'negative' THEN direction_negative_probability
                  WHEN 'neutral' THEN direction_neutral_probability
                  WHEN 'positive' THEN direction_positive_probability
                END < greatest(direction_negative_probability,
                               direction_neutral_probability,
                               direction_positive_probability)),
              count_if(analysis_version <> ? OR output_format <> ? OR
                changes_primary_ranking IS DISTINCT FROM FALSE OR
                changes_discovery_ranking IS DISTINCT FROM FALSE OR
                used_for_fusion IS DISTINCT FROM FALSE OR
                historical_checkpoint_used IS DISTINCT FROM FALSE OR
                historical_prediction_used IS DISTINCT FROM FALSE)
            FROM {table}
            """,
            [ANALYSIS_VERSION, OUTPUT_FORMAT],
        ).fetchone()
        if tuple(map(int, summary[:5])) != (
            expected_total_rows,
            expected_total_rows,
            expected_cancer_count,
            expected_available_rows,
            expected_total_rows - expected_available_rows,
        ):
            raise EvidenceDirectionIndependentAuditError(
                f"Observed formal row/coverage counts differ: {summary[:5]}"
            )
        for name, value in zip(
            (
                "available_probability_semantics",
                "typed_null_semantics",
                "published_argmax_boolean",
                "recovered_argmax",
                "lineage_and_score_immutability_flags",
            ),
            summary[5:],
            strict=True,
        ):
            _expect_zero(name, value)
        alignment = con.execute(
            f"""
            SELECT
              (SELECT count(*) - count(DISTINCT (cancer_id,lncrna_id,pathway_id)) FROM {original}),
              (SELECT count(*) - count(DISTINCT (cancer_id,lncrna_id,pathway_id)) FROM {parts}),
              (SELECT count(*) FROM {table} t ANTI JOIN {original} o USING {using_keys}),
              (SELECT count(*) FROM {original} o ANTI JOIN {table} t USING {using_keys}),
              (SELECT count(*) FROM {table} t JOIN {original} o USING {using_keys}
                 WHERE t.published_direction IS DISTINCT FROM o.direction OR
                       t.evidence_confidence_available IS DISTINCT FROM o.availability OR
                       o.changes_primary_ranking IS DISTINCT FROM FALSE OR
                       o.main_ranking_modified IS DISTINCT FROM FALSE),
              (SELECT count(*) FROM {table} t JOIN {parts} p USING {using_keys}
                 WHERE NOT t.direction_probability_available OR
                       abs(t.direction_negative_probability-p.direction_negative_probability)>2e-6 OR
                       abs(t.direction_neutral_probability-p.direction_neutral_probability)>2e-6 OR
                       abs(t.direction_positive_probability-p.direction_positive_probability)>2e-6 OR
                       t.recovered_direction IS DISTINCT FROM p.recovered_direction OR
                       t.evidence_fold IS DISTINCT FROM p.evidence_fold OR
                       t.event_count IS DISTINCT FROM p.event_count),
              (SELECT count(*) FROM {table} t
                 WHERE t.direction_probability_available AND NOT EXISTS (
                   SELECT 1 FROM {parts} p WHERE p.cancer_id=t.cancer_id
                     AND p.lncrna_id=t.lncrna_id AND p.pathway_id=t.pathway_id)),
              (SELECT count(*) FROM {parts} p
                 WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE t.cancer_id=p.cancer_id
                   AND t.lncrna_id=p.lncrna_id AND t.pathway_id=p.pathway_id
                   AND t.direction_probability_available))
            """
        ).fetchone()
        for name, value in zip(
            (
                "original_duplicate_keys",
                "partition_duplicate_keys",
                "public_missing_original_keys",
                "original_missing_public_keys",
                "original_current_v32_alignment",
                "partition_probability_alignment",
                "available_missing_partition",
                "partition_missing_available",
            ),
            alignment,
            strict=True,
        ):
            _expect_zero(name, value)
        observed_partition_rows = int(
            con.execute(f"SELECT count(*) FROM {parts}").fetchone()[0]
        )
        if observed_partition_rows != expected_available_rows:
            raise EvidenceDirectionIndependentAuditError(
                "Partition union does not equal formal available rows"
            )
        mismatch_stats = con.execute(
            f"""
            WITH x AS (
              SELECT *,
                greatest(direction_negative_probability,
                         direction_neutral_probability,
                         direction_positive_probability) AS top_probability,
                direction_negative_probability + direction_neutral_probability +
                  direction_positive_probability -
                  greatest(direction_negative_probability,
                           direction_neutral_probability,
                           direction_positive_probability) -
                  least(direction_negative_probability,
                        direction_neutral_probability,
                        direction_positive_probability) AS second_probability
              FROM {table}
              WHERE direction_probability_available
            )
            SELECT
              count_if(NOT published_direction_matches_recovered_argmax),
              avg(top_probability-second_probability)
                FILTER (WHERE NOT published_direction_matches_recovered_argmax),
              median(top_probability-second_probability)
                FILTER (WHERE NOT published_direction_matches_recovered_argmax),
              max(top_probability-second_probability)
                FILTER (WHERE NOT published_direction_matches_recovered_argmax),
              avg(top_probability-second_probability)
                FILTER (WHERE published_direction_matches_recovered_argmax)
            FROM x
            """
        ).fetchone()
        mismatch_rows = int(mismatch_stats[0])
        if (
            declared_counts.get("published_recovered_argmax_mismatch_rows")
            != mismatch_rows
            or declared_counts.get("published_recovered_argmax_match_rows")
            != expected_available_rows - mismatch_rows
        ):
            raise EvidenceDirectionIndependentAuditError(
                "Published/recovered argmax mismatch declaration is stale"
            )
        mismatch_pair_frame = con.execute(
            f"""
            SELECT evidence_fold, published_direction, recovered_direction,
                   count(*) AS row_count
            FROM {table}
            WHERE direction_probability_available
              AND NOT published_direction_matches_recovered_argmax
            GROUP BY ALL
            ORDER BY evidence_fold, published_direction, recovered_direction
            """
        ).fetchdf()
        mismatch_pairs = mismatch_pair_frame.to_dict("records")
    finally:
        con.close()

    output = Path(output_root).resolve()
    if output.exists():
        raise EvidenceDirectionIndependentAuditError(f"Audit output exists: {output}")
    output.mkdir(parents=True)
    checks = {
        "release_binding_hash_pinned": True,
        "three_class_probabilities_valid": True,
        "typed_nulls_valid": True,
        "five_fresh_v32_checkpoint_lineages_valid": True,
        "historical_checkpoint_or_prediction_used": False,
        "original_current_v32_argmax_and_availability_aligned": True,
        "primary_ranking_unchanged": True,
        "discovery_ranking_unchanged": True,
        "used_for_fusion": False,
        "score_columns_present_in_direction_artifact": False,
        "family_to_exact_broadcast": False,
    }
    report = {
        "format": AUDIT_FORMAT,
        "status": AUDIT_STATUS,
        "analysis_version": ANALYSIS_VERSION,
        "auditor_independent_of_materializer": True,
        "materializer_module_imported": False,
        "checks": checks,
        "counts": {
            "total_rows": expected_total_rows,
            "available_rows": expected_available_rows,
            "typed_null_rows": expected_total_rows - expected_available_rows,
            "cancers": expected_cancer_count,
            "folds": 5,
        },
        "lineage": {
            "evidence_binding_path": str(evidence_binding_path),
            "evidence_binding_sha256": evidence_binding_sha,
            "training_manifest_path": str(training_manifest_path),
            "training_manifest_sha256": training_manifest_sha,
            "core_manifest_path": str(core_path),
            "core_manifest_sha256": core_sha,
            "checkpoint_sha256_by_fold": checkpoint_hashes,
            "training_generation": "V3.2",
            "old_results_used": False,
        },
        "artifact": {
            "path": str(artifact_path),
            "sha256": artifact_sha,
            "rows": expected_total_rows,
        },
        "release_binding": {
            "path": str(source),
            "sha256": expected_binding_sha256,
        },
        "score_impact": {
            "primary": "BITWISE_SOURCE_NOT_WRITTEN; ROW_FLAGS_FALSE",
            "discovery": "BITWISE_SOURCE_NOT_WRITTEN; ROW_FLAGS_FALSE",
            "fusion": "NOT_CONSUMED; USED_FOR_FUSION_FALSE",
        },
        "argmax_reinference": {
            "published_argmax_aligned_to_original_current_v32_output": True,
            "recovered_argmax_derived_from_same_hash_pinned_v32_checkpoints": True,
            "mismatch_rows": mismatch_rows,
            "mismatch_rate_among_available": mismatch_rows / expected_available_rows,
            "mismatch_mean_recovered_top_margin": (
                None if mismatch_stats[1] is None else float(mismatch_stats[1])
            ),
            "mismatch_median_recovered_top_margin": (
                None if mismatch_stats[2] is None else float(mismatch_stats[2])
            ),
            "mismatch_max_recovered_top_margin": (
                None if mismatch_stats[3] is None else float(mismatch_stats[3])
            ),
            "match_mean_recovered_top_margin": (
                None if mismatch_stats[4] is None else float(mismatch_stats[4])
            ),
            "mismatch_by_fold_and_direction_pair": mismatch_pairs,
            "checkpoint_or_core_drift_detected": False,
            "interpretation": (
                "ORIGINAL_MC_DROPOUT_RNG_ARGMAX_VS_FIXED_SEED_REINFERENCE; "
                "PUBLICATION_RETAINS_BOTH_WITHOUT_OVERWRITING_ORIGINAL"
            ),
        },
        "production_deployed": False,
        "release_ready": False,
    }
    report_path = output / "EVIDENCE_DIRECTION_INDEPENDENT_AUDIT.json"
    _json_write(report_path, report)
    binding = {
        "format": AUDIT_BINDING_FORMAT,
        "status": AUDIT_STATUS,
        "analysis_version": ANALYSIS_VERSION,
        "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "release_binding": {"path": str(source), "sha256": expected_binding_sha256},
        "artifact": {"path": str(artifact_path), "sha256": artifact_sha},
        "counts": report["counts"],
        "three_class_probabilities_valid": True,
        "typed_nulls_valid": True,
        "source_checkpoints_newly_trained_v32": True,
        "old_checkpoint_loaded": False,
        "historical_prediction_used": False,
        "primary_ranking_unchanged": True,
        "discovery_ranking_unchanged": True,
        "used_for_fusion": False,
        "production_deployed": False,
        "release_ready": False,
    }
    audit_binding_path = output / "INDEPENDENT_AUDIT_BINDING.json"
    _json_write(audit_binding_path, binding)
    success = {
        "status": AUDIT_STATUS,
        "audit_binding_path": str(audit_binding_path),
        "audit_binding_sha256": _sha256(audit_binding_path),
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "release_binding_sha256": expected_binding_sha256,
        "artifact_sha256": artifact_sha,
        "production_deployed": False,
        "release_ready": False,
    }
    _json_write(output / "SUCCESS.json", success)
    return success


__all__ = [
    "ANALYSIS_VERSION",
    "AUDIT_BINDING_FORMAT",
    "AUDIT_FORMAT",
    "AUDIT_STATUS",
    "EvidenceDirectionIndependentAuditError",
    "audit_evidence_direction_release",
]
