"""Fail-closed staging of public fold predictions for V3.2 web materialization.

The source fold files contain ``held_out_proxy_label`` and are therefore
private training artifacts.  This module never edits or publishes them.  It
uses DuckDB to write immutable public-only fold copies and only emits the
formal web-materializer input manifest after a separately hash-pinned,
validation-only external-router versus hierarchical-gate winner is proved.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .web_relationship_materialization import (
    ANALYSIS_VERSION,
    INPUT_MANIFEST_FORMAT,
    TCGA_CANCERS,
)


SOURCE_MANIFEST_FORMAT = "CancerLncAtlas.v32.web_relationship_source_staging.v1"
STAGING_MANIFEST_FORMAT = "CancerLncAtlas.v32.web_relationship_sanitized_inputs.v1"
READINESS_FORMAT = "CancerLncAtlas.v32.web_relationship_selection_readiness.v1"
SELECTION_AUTHORITY_FORMAT = "CancerLncAtlas.v32.routing_winner_selection.v1"
EXPECTED_ROWS = 3_300_000
EXPECTED_FOLDS = tuple(range(5))
KEYS = ("cancer_id", "lncrna_id", "pathway_id")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OLD_FLAGS = (
    "old_checkpoint_loaded",
    "old_predictions_used_as_features",
    "old_rankings_used_as_outputs",
    "historical_predictions_used",
    "historical_rankings_used",
)
PRIVATE_FOLD_COLUMNS = frozenset(
    {
        "association_proxy_label",
        "fusion_target",
        "ground_truth",
        "held_out_label",
        "held_out_proxy_label",
        "label",
        "label_class",
        "outcome",
        "proxy_label",
        "sample_weight",
        "strong_association_label",
        "target_label",
        "test_label",
        "y_true",
    }
)
PUBLIC_FOLD_COLUMNS = (
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
    "shared_or_local_scope",
    "association_membership_probability",
    "association_direction",
    "association_direction_probability",
    "l1_probability",
    "ridge_probability",
    "graph_residual",
    "graph_gate",
    "regulatory_evidence_confidence",
    "regulatory_evidence_available",
    "pathway_target_level",
)
SANITIZED_FOLD_COLUMNS = (
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "association_membership_probability",
)
RAW_FOLD_CONTROL_COLUMNS = ("patient_fold_id", "held_out_proxy_label")


class WebRelationshipInputStagingError(RuntimeError):
    pass


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_sha(value: object, label: str) -> str:
    candidate = str(value or "").lower()
    if not _SHA256.fullmatch(candidate):
        raise WebRelationshipInputStagingError(f"{label} is not a SHA-256")
    return candidate


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WebRelationshipInputStagingError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise WebRelationshipInputStagingError(f"{label} must be a JSON object")
    return payload


def _false_only(payload: Mapping[str, Any], label: str) -> None:
    for field in _OLD_FLAGS:
        if payload.get(field) is not False:
            raise WebRelationshipInputStagingError(
                f"{label} must explicitly declare {field}=false"
            )


def _resolve_declared_file(
    declaration: Mapping[str, Any],
    *,
    relative_to: Path,
    label: str,
    allowed_suffixes: Sequence[str] | None = None,
) -> tuple[Path, str]:
    raw = declaration.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise WebRelationshipInputStagingError(f"{label} lacks path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = relative_to / candidate
    if candidate.is_symlink():
        raise WebRelationshipInputStagingError(f"{label} may not be a symlink")
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise WebRelationshipInputStagingError(f"{label} is missing: {candidate}")
    if allowed_suffixes and candidate.suffix.lower() not in set(allowed_suffixes):
        raise WebRelationshipInputStagingError(f"{label} has invalid file type")
    expected = _require_sha(declaration.get("sha256"), f"{label} SHA-256")
    observed = _sha256(candidate)
    if observed != expected:
        raise WebRelationshipInputStagingError(
            f"{label} SHA-256 mismatch: {observed} != {expected}"
        )
    return candidate, expected


def _evidence_interpretation(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        key: payload.get(key)
        for key in (
            "status",
            "module_id",
            "training_run_id",
            "primary_ranking_unchanged",
            "primary_score_preserved",
            "discovery_performance_outcome",
            "confidence_performance_outcome",
            "release_ready",
            "production_deployed",
        )
        if key in payload
    }
    diagnostic_only = (
        payload.get("primary_ranking_unchanged") is True
        or payload.get("primary_score_preserved") is True
        or payload.get("discovery_performance_outcome") == "NO_INCREMENT"
        or payload.get("confidence_performance_outcome") == "NO_INCREMENT"
    )
    return {
        "observed_fields": fields,
        "diagnostic_primary_unchanged_or_no_increment": diagnostic_only,
        "proves_validation_only_router_vs_hierarchical_winner": False,
    }


def _validate_selection_authority(payload: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    expected = {
        "format": SELECTION_AUTHORITY_FORMAT,
        "status": "PASS",
        "analysis_version": ANALYSIS_VERSION,
        "comparison_status": "SUCCESS",
        "compared_methods": [
            "current_primary",
            "external_router",
            "hierarchical_end_to_end",
        ],
        "same_candidate_universe": True,
        "same_outer_pair_folds": True,
        "same_patient_level_modality_oof": True,
        "same_seed_and_training_budget": True,
        "selection_split": "validation_only",
        "heldout_test_metrics_used_for_selection": False,
        "production_deployed": False,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            reasons.append(
                f"selection authority {field}={payload.get(field)!r}, expected {value!r}"
            )
    if payload.get("winner_id") not in expected["compared_methods"]:
        reasons.append("selection authority lacks a recognized winner_id")
    for field in (
        "candidate_universe_sha256",
        "comparison_manifest_sha256",
        "fair_comparison_metrics_sha256",
        "winner_score_sha256",
    ):
        try:
            _require_sha(payload.get(field), f"selection authority {field}")
        except WebRelationshipInputStagingError as exc:
            reasons.append(str(exc))
    for field in ("winner_fold_prediction_sha256", "winner_checkpoint_sha256"):
        values = payload.get(field)
        if not isinstance(values, list) or len(values) != 5:
            reasons.append(f"selection authority {field} must contain five SHA-256 values")
            continue
        try:
            hashes = [_require_sha(value, f"selection authority {field}") for value in values]
        except WebRelationshipInputStagingError as exc:
            reasons.append(str(exc))
            continue
        if len(set(hashes)) != 5:
            reasons.append(f"selection authority {field} must contain five distinct values")
    try:
        _false_only(payload, "selection authority")
    except WebRelationshipInputStagingError as exc:
        reasons.append(str(exc))
    return reasons


def audit_web_relationship_selection_readiness(
    *,
    evidence_declarations: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    selection_authority_declaration: Mapping[str, Any] | None = None,
    relative_to: str | Path | None = None,
) -> dict[str, Any]:
    """Audit small selection documents without scanning 3.3M-row Parquet inputs."""

    destination = Path(output_root).resolve()
    if destination.exists():
        raise WebRelationshipInputStagingError(
            f"Refusing to overwrite readiness output: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    implementation_path = Path(__file__).resolve()
    implementation_sha256 = _sha256(implementation_path)
    base = Path(relative_to).resolve() if relative_to else Path.cwd().resolve()
    inspected: list[dict[str, Any]] = []
    for index, declaration in enumerate(evidence_declarations):
        if not isinstance(declaration, Mapping):
            raise WebRelationshipInputStagingError("Selection evidence declaration is invalid")
        source, digest = _resolve_declared_file(
            declaration,
            relative_to=base,
            label=f"selection evidence {index}",
            allowed_suffixes=(".json",),
        )
        payload = _read_json(source, f"selection evidence {index}")
        inspected.append(
            {
                "path": str(source),
                "sha256": digest,
                **_evidence_interpretation(payload),
            }
        )

    reasons: list[str] = []
    authority_record: dict[str, Any] | None = None
    if selection_authority_declaration is None:
        reasons.extend(
            [
                "no hash-pinned routing winner selection authority was supplied",
                "PRIMARY_UNCHANGED and NO_INCREMENT only describe a prior secondary fusion; they do not compare external routing with end-to-end hierarchical gating",
                "validation-only selection and held-out-test non-use are therefore unproved",
            ]
        )
    else:
        authority_path, authority_sha = _resolve_declared_file(
            selection_authority_declaration,
            relative_to=base,
            label="selection authority",
            allowed_suffixes=(".json",),
        )
        authority = _read_json(authority_path, "selection authority")
        reasons.extend(_validate_selection_authority(authority))
        authority_record = {
            "path": str(authority_path),
            "sha256": authority_sha,
            "winner_id": authority.get("winner_id"),
            "winner_score_sha256": authority.get("winner_score_sha256"),
        }
    ready = not reasons
    report = {
        "format": READINESS_FORMAT,
        "status": "PASS_SELECTION_AUTHORITY" if ready else "BLOCKED_WAITING_FOR_VALIDATION_ONLY_ROUTING_WINNER",
        "analysis_version": ANALYSIS_VERSION,
        "ready_for_formal_input_staging": ready,
        "may_claim_final_selected": ready,
        "inspected_evidence": inspected,
        "selection_authority": authority_record,
        "blocking_reasons": reasons,
        "required_next_evidence": []
        if ready
        else [
            "hash-pinned fair external-router versus hierarchical-end-to-end comparison",
            "same candidate universe, outer folds, patient modality OOF, seed and training budget",
            "validation-only winner declaration with held-out test metrics excluded",
            "winner score, five winner fold predictions and five winner checkpoints",
        ],
        "full_3_3m_inputs_scanned": False,
        "sanitized_folds_materialized": False,
        "formal_input_manifest_emitted": False,
        "implementation": {
            "module": "cc_hhgt/v32/web_relationship_input_staging.py",
            "sha256": implementation_sha256,
        },
        "production_deployed": False,
    }
    report_path = destination / "WEB_RELATIONSHIP_INPUT_READINESS.json"
    _atomic_json(report_path, report)
    _atomic_json(
        destination / "AUDIT_RESULT.json",
        {
            "status": report["status"],
            "readiness_report": report_path.name,
            "readiness_report_sha256": _sha256(report_path),
            "production_deployed": False,
        },
    )
    return {**report, "report_path": str(report_path), "report_sha256": _sha256(report_path)}


def _load_source_manifest(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    if path.is_symlink():
        raise WebRelationshipInputStagingError("Source manifest may not be a symlink")
    source = path.resolve()
    if not source.is_file():
        raise WebRelationshipInputStagingError(f"Source manifest is missing: {source}")
    expected = _require_sha(expected_sha256, "expected source manifest SHA-256")
    observed = _sha256(source)
    if observed != expected:
        raise WebRelationshipInputStagingError("Source manifest SHA-256 mismatch")
    payload = _read_json(source, "source manifest")
    required = {
        "format": SOURCE_MANIFEST_FORMAT,
        "status": "CANDIDATE_INPUTS_NOT_YET_SELECTED",
        "analysis_version": ANALYSIS_VERSION,
        "pathway_target_level": "exact_pathway",
        "family_to_exact_broadcast": False,
        "expected_rows": EXPECTED_ROWS,
        "expected_cancer_ids": list(TCGA_CANCERS),
    }
    for field, value in required.items():
        if payload.get(field) != value:
            raise WebRelationshipInputStagingError(
                f"Source manifest invalid {field}: {payload.get(field)!r}"
            )
    _false_only(payload, "source manifest")
    for field in ("primary_training_run_id", "evidence_training_run_id"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise WebRelationshipInputStagingError(f"Source manifest lacks {field}")
    if not isinstance(payload.get("artifacts"), dict):
        raise WebRelationshipInputStagingError("Source manifest lacks artifacts")
    return payload, observed


def _manifest_records(payload: Mapping[str, Any], label: str) -> dict[int, Mapping[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 5:
        raise WebRelationshipInputStagingError(f"{label} must contain five records")
    result: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise WebRelationshipInputStagingError(f"{label} record is invalid")
        try:
            fold = int(record.get("patient_fold"))
        except (TypeError, ValueError) as exc:
            raise WebRelationshipInputStagingError(f"{label} record lacks patient_fold") from exc
        if fold in result or fold not in EXPECTED_FOLDS:
            raise WebRelationshipInputStagingError(f"{label} has invalid/duplicate fold {fold}")
        result[fold] = record
    if sorted(result) != list(EXPECTED_FOLDS):
        raise WebRelationshipInputStagingError(f"{label} folds are not 0..4")
    return result


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _sql_string(value: str) -> str:
    return value.replace("'", "''")


def _columns(connection: Any, view: str) -> list[str]:
    return [str(row[0]) for row in connection.execute(f"DESCRIBE {view}").fetchall()]


def _scalar(connection: Any, query: str) -> Any:
    row = connection.execute(query).fetchone()
    return None if row is None else row[0]


def _validate_unique_keys(connection: Any, view: str, expected_rows: int) -> None:
    rows = int(_scalar(connection, f"SELECT count(*) FROM {view}"))
    if rows != expected_rows:
        raise WebRelationshipInputStagingError(
            f"{view} rows={rows}, expected={expected_rows}"
        )
    bad_keys = int(
        _scalar(
            connection,
            f"""SELECT count(*) FROM {view}
                 WHERE cancer_id IS NULL OR trim(CAST(cancer_id AS VARCHAR))=''
                    OR lncrna_id IS NULL OR trim(CAST(lncrna_id AS VARCHAR))=''
                    OR pathway_id IS NULL OR trim(CAST(pathway_id AS VARCHAR))=''""",
        )
    )
    duplicate_groups = int(
        _scalar(
            connection,
            f"""SELECT count(*) FROM (
                 SELECT {', '.join(KEYS)}, count(*) n FROM {view}
                 GROUP BY {', '.join(KEYS)} HAVING count(*) <> 1)""",
        )
    )
    if bad_keys or duplicate_groups:
        raise WebRelationshipInputStagingError(
            f"{view} invalid keys={bad_keys}, duplicate groups={duplicate_groups}"
        )


def _assert_same_keys(connection: Any, left: str, right: str) -> None:
    using = ", ".join(KEYS)
    left_only = int(_scalar(connection, f"SELECT count(*) FROM {left} ANTI JOIN {right} USING ({using})"))
    right_only = int(_scalar(connection, f"SELECT count(*) FROM {right} ANTI JOIN {left} USING ({using})"))
    if left_only or right_only:
        raise WebRelationshipInputStagingError(
            f"Exact-key mismatch {left}<->{right}: {left_only}/{right_only}"
        )


def _validate_lineage_rows(
    connection: Any, view: str, training_run_id: str, *, require_exact: bool = True
) -> None:
    columns = set(_columns(connection, view))
    required = {*KEYS, "analysis_version", "training_run_id"}
    if missing := sorted(required - columns):
        raise WebRelationshipInputStagingError(f"{view} lacks lineage columns: {missing}")
    exact_clause = ""
    if require_exact:
        if "pathway_target_level" not in columns:
            raise WebRelationshipInputStagingError(f"{view} lacks pathway_target_level")
        exact_clause = " OR CAST(pathway_target_level AS VARCHAR) <> 'exact_pathway'"
    invalid = int(
        _scalar(
            connection,
            f"""SELECT count(*) FROM {view}
                 WHERE CAST(analysis_version AS VARCHAR) <> '{_sql_string(ANALYSIS_VERSION)}'
                    OR CAST(training_run_id AS VARCHAR) <> '{_sql_string(training_run_id)}'
                    {exact_clause}""",
        )
    )
    if invalid:
        raise WebRelationshipInputStagingError(f"{view} has {invalid} row-lineage violations")


def stage_fresh_v32_web_relationship_inputs(
    *,
    source_manifest_path: str | Path,
    expected_source_manifest_sha256: str,
    output_root: str | Path,
    memory_limit: str = "1GB",
    temp_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Sanitize real fold files and emit a formal input manifest only if selected."""

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise WebRelationshipInputStagingError("duckdb is required") from exc
    if not re.fullmatch(r"[1-9][0-9]*(?:MB|GB)", str(memory_limit).upper()):
        raise WebRelationshipInputStagingError("memory_limit must look like 512MB or 1GB")
    source_input = Path(source_manifest_path)
    source, source_sha = _load_source_manifest(
        source_input, expected_source_manifest_sha256
    )
    source_path = source_input.resolve()
    destination = Path(output_root).resolve()
    if destination.exists():
        raise WebRelationshipInputStagingError(
            f"Refusing to overwrite/reuse input staging: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    base = source_path.parent
    evidence_declarations = source.get("selection_evidence", [])
    authority_decl = source.get("selection_authority")
    readiness_dir = destination / "selection_readiness"
    readiness = audit_web_relationship_selection_readiness(
        evidence_declarations=evidence_declarations,
        selection_authority_declaration=authority_decl
        if isinstance(authority_decl, Mapping)
        else None,
        output_root=readiness_dir,
        relative_to=base,
    )
    if not readiness["ready_for_formal_input_staging"]:
        result = {
            "status": readiness["status"],
            "ready_for_formal_input_staging": False,
            "readiness_report": str(Path(readiness["report_path"]).relative_to(destination)),
            "readiness_report_sha256": readiness["report_sha256"],
            "sanitized_folds_materialized": False,
            "formal_input_manifest_emitted": False,
            "production_deployed": False,
        }
        _atomic_json(destination / "BLOCKED.json", result)
        return result

    artifacts = source["artifacts"]
    required_artifacts = {
        "primary",
        "exact_evidence",
        "candidate_universe",
        "exact_module_lineage",
        "evidence_lineage",
        "checkpoint_manifest",
        "ensemble_source_manifest",
        "task_manifest",
        "comparison_manifest",
        "comparison_metrics",
        "raw_fold_predictions",
        "checkpoint_files",
    }
    if missing := sorted(required_artifacts - set(artifacts)):
        raise WebRelationshipInputStagingError(f"Source manifest lacks artifacts: {missing}")
    resolved: dict[str, tuple[Path, str]] = {}
    for role in required_artifacts - {"raw_fold_predictions", "checkpoint_files"}:
        declaration = artifacts[role]
        if not isinstance(declaration, Mapping):
            raise WebRelationshipInputStagingError(f"Invalid {role} declaration")
        suffixes = (".parquet", ".pq") if role in {
            "primary", "exact_evidence", "candidate_universe", "comparison_metrics"
        } else (".json", ".tsv")
        resolved[role] = _resolve_declared_file(
            declaration, relative_to=base, label=role, allowed_suffixes=suffixes
        )

    raw_declarations = artifacts["raw_fold_predictions"]
    checkpoint_declarations = artifacts["checkpoint_files"]
    if not isinstance(raw_declarations, list) or len(raw_declarations) != 5:
        raise WebRelationshipInputStagingError("Exactly five raw fold predictions are required")
    if not isinstance(checkpoint_declarations, list) or len(checkpoint_declarations) != 5:
        raise WebRelationshipInputStagingError("Exactly five checkpoint files are required")
    raw_folds: dict[int, tuple[Path, str]] = {}
    checkpoints: dict[int, tuple[Path, str]] = {}
    for label, declarations, target, suffixes in (
        ("raw fold", raw_declarations, raw_folds, (".parquet", ".pq")),
        ("checkpoint", checkpoint_declarations, checkpoints, (".pt", ".pth", ".bin")),
    ):
        for declaration in declarations:
            if not isinstance(declaration, Mapping):
                raise WebRelationshipInputStagingError(f"Invalid {label} declaration")
            try:
                fold = int(declaration.get("fold_id"))
            except (TypeError, ValueError) as exc:
                raise WebRelationshipInputStagingError(f"{label} lacks fold_id") from exc
            if fold in target or fold not in EXPECTED_FOLDS:
                raise WebRelationshipInputStagingError(f"Invalid/duplicate {label} fold {fold}")
            target[fold] = _resolve_declared_file(
                declaration,
                relative_to=base,
                label=f"{label} {fold}",
                allowed_suffixes=suffixes,
            )
    if sorted(raw_folds) != list(EXPECTED_FOLDS) or sorted(checkpoints) != list(EXPECTED_FOLDS):
        raise WebRelationshipInputStagingError("Raw fold/checkpoint IDs must be 0..4")
    if len({value[1] for value in raw_folds.values()}) != 5:
        raise WebRelationshipInputStagingError("Raw fold prediction artifacts are not distinct")
    if len({value[1] for value in checkpoints.values()}) != 5:
        raise WebRelationshipInputStagingError("Checkpoint artifacts are not distinct")

    checkpoint_manifest = _read_json(resolved["checkpoint_manifest"][0], "checkpoint manifest")
    ensemble_manifest = _read_json(resolved["ensemble_source_manifest"][0], "ensemble source manifest")
    module_lineage = _read_json(resolved["exact_module_lineage"][0], "exact module lineage")
    evidence_lineage = _read_json(resolved["evidence_lineage"][0], "evidence lineage")
    authority_path, authority_sha = _resolve_declared_file(
        authority_decl, relative_to=base, label="selection authority", allowed_suffixes=(".json",)
    )
    authority = _read_json(authority_path, "selection authority")
    if _validate_selection_authority(authority):
        raise WebRelationshipInputStagingError("Selection authority changed after readiness audit")
    comparison_manifest_sha = resolved["comparison_manifest"][1]
    comparison_metrics_sha = resolved["comparison_metrics"][1]
    if authority["comparison_manifest_sha256"] != comparison_manifest_sha:
        raise WebRelationshipInputStagingError("Selection authority comparison manifest SHA drift")
    if authority["fair_comparison_metrics_sha256"] != comparison_metrics_sha:
        raise WebRelationshipInputStagingError("Selection authority comparison metrics SHA drift")
    if authority["candidate_universe_sha256"] != resolved["candidate_universe"][1]:
        raise WebRelationshipInputStagingError("Selection authority candidate universe SHA drift")
    if authority["winner_score_sha256"] != resolved["primary"][1]:
        raise WebRelationshipInputStagingError("Declared primary is not the selected winner score")
    if authority["winner_fold_prediction_sha256"] != [raw_folds[i][1] for i in EXPECTED_FOLDS]:
        raise WebRelationshipInputStagingError("Selected winner fold prediction SHA drift")
    if authority["winner_checkpoint_sha256"] != [checkpoints[i][1] for i in EXPECTED_FOLDS]:
        raise WebRelationshipInputStagingError("Selected winner checkpoint SHA drift")

    checkpoint_records = _manifest_records(checkpoint_manifest, "checkpoint manifest")
    ensemble_records = _manifest_records(ensemble_manifest, "ensemble source manifest")
    for field in (
        "old_checkpoint_loaded",
        "old_predictions_used_as_features",
        "old_rankings_used_as_outputs",
    ):
        if ensemble_manifest.get(field) is not False:
            raise WebRelationshipInputStagingError(
                f"ensemble source manifest must declare {field}=false"
            )
    if (
        checkpoint_manifest.get("analysis_version") != ANALYSIS_VERSION
        or ensemble_manifest.get("analysis_version") != ANALYSIS_VERSION
        or ensemble_manifest.get("training_run_id") != source["primary_training_run_id"]
        or ensemble_manifest.get("ensemble_formula")
        != "arithmetic_mean_of_five_patient_fold_predictions"
        or ensemble_manifest.get("all_sources_current_v32") is not True
        or ensemble_manifest.get("all_five_folds_trained_from_random_initialization") is not True
    ):
        raise WebRelationshipInputStagingError("Exact checkpoint/ensemble lineage is invalid")
    task_manifest_sha = resolved["task_manifest"][1]
    partition_hashes: dict[int, str] = {}
    for fold in EXPECTED_FOLDS:
        checkpoint_record = checkpoint_records[fold]
        ensemble_record = ensemble_records[fold]
        if (
            checkpoint_record.get("checkpoint_sha256") != checkpoints[fold][1]
            or ensemble_record.get("checkpoint_sha256") != checkpoints[fold][1]
            or ensemble_record.get("prediction_sha256") != raw_folds[fold][1]
            or int(ensemble_record.get("prediction_rows", -1)) != EXPECTED_ROWS
            or ensemble_record.get("candidate_alignment") != "FULL_ROW_EXACT"
            or ensemble_record.get("trained_from_random_initialization") is not True
            or ensemble_record.get("old_checkpoint_loaded") is not False
            or ensemble_record.get("old_predictions_used_as_features") is not False
            or ensemble_record.get("old_rankings_used_as_outputs") is not False
        ):
            raise WebRelationshipInputStagingError(f"Fold {fold} checkpoint/ensemble drift")
        source_hashes = ensemble_record.get("artifact_hashes")
        if not isinstance(source_hashes, Mapping) or source_hashes.get("task_manifest_sha256") != task_manifest_sha:
            raise WebRelationshipInputStagingError(f"Fold {fold} task partition lineage drift")
        partition_hashes[fold] = _canonical_sha256(
            {
                "source_task_manifest_sha256": task_manifest_sha,
                "held_out_patient_fold": fold,
                "semantics": "training partition complement of held-out patient fold",
            }
        )
    if len(set(partition_hashes.values())) != 5:
        raise WebRelationshipInputStagingError("Derived training partition bindings are not distinct")
    if (
        module_lineage.get("analysis_version") != ANALYSIS_VERSION
        or module_lineage.get("training_run_id") != source["primary_training_run_id"]
        or module_lineage.get("prediction_sha256") != resolved["primary"][1]
        or module_lineage.get("checkpoint_manifest_sha256") != resolved["checkpoint_manifest"][1]
        or module_lineage.get("ensemble_source_manifest_sha256") != resolved["ensemble_source_manifest"][1]
        or module_lineage.get("training_status") != "SUCCESS"
        or module_lineage.get("old_checkpoint_loaded") is not False
        or module_lineage.get("old_predictions_used_as_features") is not False
        or module_lineage.get("old_rankings_used_as_outputs") is not False
    ):
        raise WebRelationshipInputStagingError("Exact module lineage is invalid")
    if (
        evidence_lineage.get("analysis_version") != ANALYSIS_VERSION
        or evidence_lineage.get("training_run_id") != source["evidence_training_run_id"]
        or evidence_lineage.get("prediction_sha256") != resolved["exact_evidence"][1]
        or int(evidence_lineage.get("prediction_rows", -1)) != EXPECTED_ROWS
        or evidence_lineage.get("pathway_target_level") != "exact_pathway"
        or evidence_lineage.get("family_to_exact_broadcast") is not False
    ):
        raise WebRelationshipInputStagingError("Exact evidence lineage is invalid")
    _false_only(evidence_lineage, "exact evidence lineage")

    work = destination / f".materializing.{os.getpid()}"
    work.mkdir()
    if temp_directory:
        spill_base = Path(temp_directory).resolve()
        spill_base.mkdir(parents=True, exist_ok=True)
        spill = spill_base / f".v32_web_input_staging_spill.{os.getpid()}"
    else:
        spill = work / "duckdb_spill"
    spill.mkdir(exist_ok=False)
    database = work / "staging.duckdb"
    connection = duckdb.connect(str(database))
    sanitized: dict[int, Path] = {}
    try:
        connection.execute("SET threads=1")
        connection.execute(f"SET memory_limit='{str(memory_limit).upper()}'")
        connection.execute(f"SET temp_directory='{_sql_path(spill)}'")
        table_paths = {
            "primary_input": resolved["primary"][0],
            "evidence_input": resolved["exact_evidence"][0],
            "candidate_input": resolved["candidate_universe"][0],
            **{f"raw_fold_{fold}": raw_folds[fold][0] for fold in EXPECTED_FOLDS},
        }
        for view, path in table_paths.items():
            connection.execute(
                f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{_sql_path(path)}')"
            )
            _validate_unique_keys(connection, view, EXPECTED_ROWS)
        _validate_lineage_rows(connection, "primary_input", source["primary_training_run_id"])
        _validate_lineage_rows(connection, "evidence_input", source["evidence_training_run_id"])
        primary_columns = set(_columns(connection, "primary_input"))
        primary_required = {
            *KEYS,
            "pathway_family_id",
            "association_membership_probability",
            "association_direction_probability",
            "association_direction",
            "n_folds_available",
        }
        if missing := sorted(primary_required - primary_columns):
            raise WebRelationshipInputStagingError(f"Primary lacks columns: {missing}")
        evidence_columns = set(_columns(connection, "evidence_input"))
        evidence_required = {
            *KEYS,
            "evidence_confidence_probability",
            "availability",
            "direct_target_evidence",
            "family_to_exact_broadcast",
            "direction",
        }
        if missing := sorted(evidence_required - evidence_columns):
            raise WebRelationshipInputStagingError(f"Evidence lacks columns: {missing}")
        bad_evidence = int(
            _scalar(
                connection,
                """SELECT count(*) FROM evidence_input
                   WHERE try_cast(family_to_exact_broadcast AS BOOLEAN) IS DISTINCT FROM FALSE
                      OR try_cast(availability AS BOOLEAN) IS NULL
                      OR try_cast(direct_target_evidence AS BOOLEAN) IS NULL
                      OR try_cast(availability AS BOOLEAN) <> try_cast(direct_target_evidence AS BOOLEAN)""",
            )
        )
        if bad_evidence:
            raise WebRelationshipInputStagingError("Evidence availability/broadcast contract failed")
        cancers = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT CAST(cancer_id AS VARCHAR) FROM primary_input ORDER BY 1"
            ).fetchall()
        ]
        if cancers != list(TCGA_CANCERS):
            raise WebRelationshipInputStagingError("Primary cancer set is not canonical 33 TCGA")
        for view in ("evidence_input", "candidate_input", *[f"raw_fold_{i}" for i in EXPECTED_FOLDS]):
            _assert_same_keys(connection, "primary_input", view)
        for fold in EXPECTED_FOLDS:
            view = f"raw_fold_{fold}"
            columns = _columns(connection, view)
            lower = {column.lower(): column for column in columns}
            allowed = set(PUBLIC_FOLD_COLUMNS) | set(RAW_FOLD_CONTROL_COLUMNS)
            if set(columns) != allowed:
                raise WebRelationshipInputStagingError(
                    f"Raw fold {fold} schema drift: extra={sorted(set(columns)-allowed)}, missing={sorted(allowed-set(columns))}"
                )
            if "held_out_proxy_label" not in lower:
                raise WebRelationshipInputStagingError(f"Raw fold {fold} private label was not identified")
            invalid_fold = int(
                _scalar(
                    connection,
                    f"SELECT count(*) FROM {view} WHERE try_cast(patient_fold_id AS INTEGER) IS DISTINCT FROM {fold}",
                )
            )
            if invalid_fold:
                raise WebRelationshipInputStagingError(f"Raw fold {fold} patient_fold_id drift")
            output_path = work / f"FOLD_{fold}_PUBLIC_SANITIZED.parquet"
            public_projection = ", ".join(SANITIZED_FOLD_COLUMNS)
            connection.execute(
                f"""COPY (
                    SELECT {public_projection},
                           {fold}::INTEGER AS fold_id,
                           '{_sql_string(ANALYSIS_VERSION)}' AS analysis_version,
                           '{_sql_string(source['primary_training_run_id'])}' AS training_run_id,
                           false AS old_checkpoint_loaded,
                           false AS old_predictions_used_as_features,
                           false AS old_rankings_used_as_outputs,
                           false AS historical_predictions_used,
                           false AS historical_rankings_used
                    FROM {view}
                ) TO '{_sql_path(output_path)}'
                  (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"""
            )
            sanitized[fold] = output_path
            public_columns = {row[0] for row in connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(output_path)}')"
            ).fetchall()}
            if public_columns & PRIVATE_FOLD_COLUMNS:
                raise WebRelationshipInputStagingError(f"Sanitized fold {fold} leaked private labels")
            _validate_unique_keys(
                connection,
                f"read_parquet('{_sql_path(output_path)}')",
                EXPECTED_ROWS,
            )
        ensemble_columns = ensemble_manifest.get("ensemble_columns")
        if not isinstance(ensemble_columns, list) or not ensemble_columns:
            raise WebRelationshipInputStagingError("Ensemble manifest lacks columns")
        for column in ensemble_columns:
            if column not in primary_columns or any(
                column not in set(_columns(connection, f"raw_fold_{fold}"))
                for fold in EXPECTED_FOLDS
            ):
                raise WebRelationshipInputStagingError(f"Ensemble column missing: {column}")
            average = " + ".join(
                f"try_cast(raw_fold_{fold}.{column} AS DOUBLE)" for fold in EXPECTED_FOLDS
            )
            joins = " ".join(
                f"JOIN raw_fold_{fold} USING ({', '.join(KEYS)})" for fold in EXPECTED_FOLDS
            )
            maximum = float(
                _scalar(
                    connection,
                    f"""SELECT max(abs(try_cast(primary_input.{column} AS DOUBLE) - (({average})/5.0)))
                         FROM primary_input {joins}""",
                )
            )
            if maximum > 2.0e-6:
                raise WebRelationshipInputStagingError(
                    f"Primary is not the declared five-fold mean for {column}: max_abs={maximum}"
                )
    finally:
        connection.close()

    # Recheck every immutable source after the scan before emitting authority.
    if _sha256(source_path) != source_sha:
        raise WebRelationshipInputStagingError("Source manifest changed during staging")
    for role, (path, digest) in resolved.items():
        if _sha256(path) != digest:
            raise WebRelationshipInputStagingError(f"{role} changed during staging")
    for fold in EXPECTED_FOLDS:
        if _sha256(raw_folds[fold][0]) != raw_folds[fold][1]:
            raise WebRelationshipInputStagingError(f"Raw fold {fold} changed during staging")
        if _sha256(checkpoints[fold][0]) != checkpoints[fold][1]:
            raise WebRelationshipInputStagingError(f"Checkpoint {fold} changed during staging")
    if _sha256(authority_path) != authority_sha:
        raise WebRelationshipInputStagingError("Selection authority changed during staging")

    database.unlink(missing_ok=True)
    fold_artifacts: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        final_path = destination / sanitized[fold].name
        os.replace(sanitized[fold], final_path)
        fold_artifacts.append(
            {
                "role": "independent_fold_prediction",
                "fold_id": fold,
                "path": final_path.name,
                "sha256": _sha256(final_path),
                "analysis_version": ANALYSIS_VERSION,
                "training_run_id": source["primary_training_run_id"],
                "selection_sha256": authority_sha,
                "candidate_universe_sha256": resolved["candidate_universe"][1],
                "checkpoint_sha256": checkpoints[fold][1],
                "training_partition_sha256": partition_hashes[fold],
                "training_partition_semantics": "derived binding of source task manifest SHA and held-out patient fold",
                "source_task_manifest_sha256": task_manifest_sha,
                **{field: False for field in _OLD_FLAGS},
            }
        )
    for child in spill.iterdir():
        if child.is_dir() and not child.is_symlink():
            raise WebRelationshipInputStagingError(f"Unexpected nested spill directory: {child}")
        child.unlink()
    spill.rmdir()
    work.rmdir()

    common = {
        "analysis_version": ANALYSIS_VERSION,
        "selection_sha256": authority_sha,
        "candidate_universe_sha256": resolved["candidate_universe"][1],
        **{field: False for field in _OLD_FLAGS},
    }
    formal_input = {
        "manifest_format": INPUT_MANIFEST_FORMAT,
        "status": "PASS",
        "analysis_version": ANALYSIS_VERSION,
        "pathway_target_level": "exact_pathway",
        "pathway_family_role": "auxiliary_hierarchy_only",
        "family_to_exact_broadcast": False,
        "primary_score_status": "FINAL_SELECTED",
        "selection_status": "PASS",
        "selection_split": "validation_only",
        "test_metrics_used_for_selection": False,
        "expected_cancers": 33,
        "expected_cancer_ids": list(TCGA_CANCERS),
        "required_folds": list(EXPECTED_FOLDS),
        "primary_training_run_id": source["primary_training_run_id"],
        "evidence_training_run_id": source["evidence_training_run_id"],
        "selection_sha256": authority_sha,
        "candidate_universe_sha256": resolved["candidate_universe"][1],
        **{field: False for field in _OLD_FLAGS},
        "artifacts": {
            "primary": {
                "role": "final_selected_primary",
                "path": str(resolved["primary"][0]),
                "sha256": resolved["primary"][1],
                "training_run_id": source["primary_training_run_id"],
                **common,
            },
            "exact_evidence": {
                "role": "exact_target_evidence",
                "path": str(resolved["exact_evidence"][0]),
                "sha256": resolved["exact_evidence"][1],
                "training_run_id": source["evidence_training_run_id"],
                "family_to_exact_broadcast": False,
                **common,
            },
            "fold_predictions": fold_artifacts,
        },
    }
    formal_path = destination / "WEB_RELATIONSHIP_INPUT_MANIFEST.json"
    _atomic_json(formal_path, formal_input)
    staging_manifest = {
        "format": STAGING_MANIFEST_FORMAT,
        "status": "PASS_SANITIZED_PUBLIC_FOLDS",
        "analysis_version": ANALYSIS_VERSION,
        "source_manifest": {"path": str(source_path), "sha256": source_sha},
        "selection_authority": {"path": str(authority_path), "sha256": authority_sha},
        "source_raw_folds_contained_private_label": True,
        "private_label_column": "held_out_proxy_label",
        "private_labels_copied_to_output": False,
        "source_files_modified": False,
        "sanitized_folds": fold_artifacts,
        "formal_input_manifest": {
            "path": formal_path.name,
            "sha256": _sha256(formal_path),
        },
        "production_deployed": False,
        "release_ready": False,
    }
    staging_path = destination / "STAGING_MANIFEST.json"
    _atomic_json(staging_path, staging_manifest)
    success = {
        "status": "PASS_SANITIZED_PUBLIC_FOLDS",
        "staging_manifest": staging_path.name,
        "staging_manifest_sha256": _sha256(staging_path),
        "formal_input_manifest": formal_path.name,
        "formal_input_manifest_sha256": _sha256(formal_path),
        "production_deployed": False,
        "release_ready": False,
    }
    _atomic_json(destination / "SUCCESS.json", success)
    return {**success, "output_root": str(destination)}


__all__ = [
    "PRIVATE_FOLD_COLUMNS",
    "PUBLIC_FOLD_COLUMNS",
    "SANITIZED_FOLD_COLUMNS",
    "READINESS_FORMAT",
    "SELECTION_AUTHORITY_FORMAT",
    "SOURCE_MANIFEST_FORMAT",
    "STAGING_MANIFEST_FORMAT",
    "WebRelationshipInputStagingError",
    "audit_web_relationship_selection_readiness",
    "stage_fresh_v32_web_relationship_inputs",
]
