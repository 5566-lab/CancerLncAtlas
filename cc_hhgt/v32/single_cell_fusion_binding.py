"""Hash-bound materialisation of the audited V3.2 single-cell fusion expert."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import duckdb

from .single_cell_fusion_adapter import (
    ADAPTER_FORMAT,
    ANALYSIS_VERSION,
    AVAILABILITY_COLUMN,
    PROBABILITY_COLUMN,
    REASON_COLUMN,
    build_single_cell_exact_fusion_expert,
)


BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_FUSION_INPUT_BINDING_V1"
AUDIT_BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_INDEPENDENT_AUDIT_BINDING_V1"
AUDIT_REPORT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_INDEPENDENT_AUDIT_V1"
FORMAL_CANDIDATE_SHA256 = "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
FORMAL_EXACT_SOURCE_SHA256 = "83e25646ef604b9d1a76f5872c73421ed0041c2fe097549e7c8838848ede7112"
FORMAL_CANDIDATE_ROWS = 3_300_000
FORMAL_EXACT_SOURCE_ROWS = 2_554_541
FORMAL_AVAILABLE_ROWS = 954_541
FORMAL_UNAVAILABLE_ROWS = 2_345_459


class SingleCellFusionBindingError(RuntimeError):
    """Raised when an adapter cannot be bound to the independent audit."""


def artifact_sha256(path: str | Path) -> str:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise SingleCellFusionBindingError(f"Missing or unsafe file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path, role: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise SingleCellFusionBindingError(f"{role} is missing or unsafe: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleCellFusionBindingError(f"{role} is invalid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise SingleCellFusionBindingError(f"{role} must be a JSON object")
    return source, value


def _require(payload: Mapping[str, Any], expected: Mapping[str, Any], role: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SingleCellFusionBindingError(
                f"{role}.{key} drift: observed={payload.get(key)!r}, expected={value!r}"
            )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _validate_audit(
    *,
    binding_path: Path,
    binding: Mapping[str, Any],
    expected_binding_sha256: str,
    candidates_path: Path,
    exact_source_path: Path,
    strict_formal: bool,
) -> tuple[Path, dict[str, Any], dict[str, int]]:
    if artifact_sha256(binding_path) != expected_binding_sha256.lower():
        raise SingleCellFusionBindingError("Independent audit binding SHA256 drift")
    _require(
        binding,
        {
            "format": AUDIT_BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS",
            "fail_closed": True,
        },
        "audit binding",
    )
    decision = binding.get("release_decision")
    if not isinstance(decision, Mapping):
        raise SingleCellFusionBindingError("Independent audit binding lacks release_decision")
    _require(
        decision,
        {
            "formal_single_cell_head_accepted": True,
            "full_universe_fusion_adapter_materialization": "AUTHORIZED",
            "primary_ranking_may_be_changed": False,
        },
        "audit binding release_decision",
    )
    report_path = Path(str(binding.get("audit_report_path", ""))).resolve()
    report_sha = str(binding.get("audit_report_sha256", "")).lower()
    if artifact_sha256(report_path) != report_sha:
        raise SingleCellFusionBindingError("Independent audit report SHA256 drift")
    _, report = _read_json(report_path, "independent audit report")
    _require(
        report,
        {
            "format": AUDIT_REPORT_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS",
            "fail_closed": True,
        },
        "independent audit report",
    )
    failed = report.get("failed_checks")
    checks = report.get("checks")
    if failed not in ([], None) or not isinstance(checks, list) or len(checks) < 200:
        raise SingleCellFusionBindingError("Independent audit did not prove the full check set")
    if any(not isinstance(item, Mapping) or item.get("status") != "PASS" for item in checks):
        raise SingleCellFusionBindingError("Independent audit contains a non-PASS check")
    report_decision = report.get("release_decision")
    if report_decision != decision:
        raise SingleCellFusionBindingError("Audit report/binding release decisions differ")

    snapshot = report.get("hash_evidence", {}).get("authoritative_snapshot_after", {})
    candidate_record = snapshot.get("candidate_universe")
    exact_record = snapshot.get("exact_aggregate")
    if not isinstance(candidate_record, Mapping) or not isinstance(exact_record, Mapping):
        raise SingleCellFusionBindingError("Independent audit lacks authoritative input hashes")
    if (
        Path(str(candidate_record.get("path", ""))).resolve() != candidates_path
        or candidate_record.get("sha256") != artifact_sha256(candidates_path)
        or binding.get("candidate_universe_sha256") != candidate_record.get("sha256")
    ):
        raise SingleCellFusionBindingError("Candidate authority differs from independent audit")
    if (
        Path(str(exact_record.get("path", ""))).resolve() != exact_source_path
        or exact_record.get("sha256") != artifact_sha256(exact_source_path)
    ):
        raise SingleCellFusionBindingError("Exact single-cell aggregate differs from audit")

    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise SingleCellFusionBindingError("Independent audit lacks row-count summary")
    counts = {
        "candidate_rows": int(summary.get("candidate_rows", -1)),
        "exact_source_rows": int(summary.get("exact_aggregate_rows", -1)),
        "available_rows": int(summary.get("adapter_available_rows", -1)),
        "unavailable_rows": int(summary.get("adapter_unavailable_rows", -1)),
    }
    if counts["candidate_rows"] != counts["available_rows"] + counts["unavailable_rows"]:
        raise SingleCellFusionBindingError("Independent adapter counts are inconsistent")
    if strict_formal:
        expected = {
            "candidate_rows": FORMAL_CANDIDATE_ROWS,
            "exact_source_rows": FORMAL_EXACT_SOURCE_ROWS,
            "available_rows": FORMAL_AVAILABLE_ROWS,
            "unavailable_rows": FORMAL_UNAVAILABLE_ROWS,
        }
        if counts != expected:
            raise SingleCellFusionBindingError(
                f"Independent audit counts differ from formal authority: {counts}"
            )
        if artifact_sha256(candidates_path) != FORMAL_CANDIDATE_SHA256:
            raise SingleCellFusionBindingError("Formal candidate SHA256 drift")
        if artifact_sha256(exact_source_path) != FORMAL_EXACT_SOURCE_SHA256:
            raise SingleCellFusionBindingError("Formal single-cell exact SHA256 drift")
    return report_path, report, counts


def materialize_single_cell_fusion_binding(
    *,
    candidates_path: str | Path,
    exact_source_path: str | Path,
    independent_audit_binding_path: str | Path,
    expected_audit_binding_sha256: str,
    output_root: str | Path,
    strict_formal: bool = True,
) -> dict[str, Any]:
    """Materialise one full-universe expert only after independent authorization."""

    candidates = Path(candidates_path).resolve()
    exact_source = Path(exact_source_path).resolve()
    audit_binding_path, audit_binding = _read_json(
        independent_audit_binding_path, "independent audit binding"
    )
    report_path, _, counts = _validate_audit(
        binding_path=audit_binding_path,
        binding=audit_binding,
        expected_binding_sha256=expected_audit_binding_sha256,
        candidates_path=candidates,
        exact_source_path=exact_source,
        strict_formal=strict_formal,
    )

    output = Path(output_root).resolve()
    if output.exists():
        raise SingleCellFusionBindingError(f"Refusing to reuse adapter output: {output}")
    output.mkdir(parents=True)
    try:
        frame = build_single_cell_exact_fusion_expert(candidates, exact_source)
        prediction_path = output / "single_cell_exact_fusion_expert.parquet"
        temporary = output / ".single_cell_exact_fusion_expert.parquet.tmp"
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, prediction_path)
        del frame

        connection = duckdb.connect(database=":memory:")
        try:
            audit = connection.execute(
                f"""
                SELECT count(*) AS rows,
                       count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS distinct_keys,
                       count_if({AVAILABILITY_COLUMN}) AS available_rows,
                       count_if(NOT {AVAILABILITY_COLUMN}) AS unavailable_rows,
                       count_if({AVAILABILITY_COLUMN} AND
                                ({PROBABILITY_COLUMN} IS NULL OR NOT isfinite({PROBABILITY_COLUMN})
                                 OR {PROBABILITY_COLUMN} < 0 OR {PROBABILITY_COLUMN} > 1))
                         AS bad_available,
                       count_if(NOT {AVAILABILITY_COLUMN} AND {PROBABILITY_COLUMN} IS NOT NULL)
                         AS filled_unavailable,
                       count_if(NOT {AVAILABILITY_COLUMN} AND
                                ({REASON_COLUMN} IS NULL OR trim({REASON_COLUMN}) = ''))
                         AS missing_reason,
                       count_if(analysis_version <> ?) AS version_mismatch,
                       count_if(changes_primary_ranking) AS primary_changes,
                       count_if(family_to_exact_broadcast) AS family_broadcasts,
                       count_if(direct_target_evidence) AS direct_evidence_rows
                FROM read_parquet({_sql_path(prediction_path)})
                """,
                [ANALYSIS_VERSION],
            ).fetchone()
        finally:
            connection.close()
        observed = {
            "rows": int(audit[0]),
            "distinct_keys": int(audit[1]),
            "available_rows": int(audit[2]),
            "unavailable_rows": int(audit[3]),
            "bad_available_rows": int(audit[4]),
            "filled_unavailable_rows": int(audit[5]),
            "missing_unavailable_reason_rows": int(audit[6]),
            "analysis_version_mismatch_rows": int(audit[7]),
            "changes_primary_ranking_rows": int(audit[8]),
            "family_broadcast_rows": int(audit[9]),
            "direct_target_evidence_rows": int(audit[10]),
        }
        if (
            observed["rows"] != counts["candidate_rows"]
            or observed["distinct_keys"] != counts["candidate_rows"]
            or observed["available_rows"] != counts["available_rows"]
            or observed["unavailable_rows"] != counts["unavailable_rows"]
            or any(
                observed[key] != 0
                for key in (
                    "bad_available_rows",
                    "filled_unavailable_rows",
                    "missing_unavailable_reason_rows",
                    "analysis_version_mismatch_rows",
                    "changes_primary_ranking_rows",
                    "family_broadcast_rows",
                    "direct_target_evidence_rows",
                )
            )
        ):
            raise SingleCellFusionBindingError(f"Materialized adapter failed audit: {observed}")

        audit_path = output / "ADAPTER_AUDIT.json"
        _atomic_json(
            audit_path,
            {
                "format": "CC_HHGT_V3_2_SINGLE_CELL_FUSION_ADAPTER_AUDIT_V1",
                "analysis_version": ANALYSIS_VERSION,
                "status": "PASS",
                "observed": observed,
                "expected_from_independent_audit": counts,
                "primary_ranking_unchanged": True,
                "unavailable_encoding": "null_with_reason",
            },
        )
        binding_path = output / "SINGLE_CELL_FUSION_BINDING.json"
        binding = {
            "format": BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS_HASH_BOUND_FUSION_INPUT",
            "training_run_id": audit_binding["audited_training_run_id"],
            "adapter_format": ADAPTER_FORMAT,
            "target_level": "cancer_x_lncrna_x_exact_pathway",
            "source_target_level": "dataset_x_celltype_x_lncrna_x_exact_pathway",
            "candidate_rows": counts["candidate_rows"],
            "available_rows": counts["available_rows"],
            "unavailable_rows": counts["unavailable_rows"],
            "prediction": {
                "path": str(prediction_path),
                "sha256": artifact_sha256(prediction_path),
                "rows": counts["candidate_rows"],
            },
            "adapter_audit": {
                "path": str(audit_path),
                "sha256": artifact_sha256(audit_path),
            },
            "independent_audit_binding": {
                "path": str(audit_binding_path),
                "sha256": artifact_sha256(audit_binding_path),
            },
            "independent_audit_report": {
                "path": str(report_path),
                "sha256": artifact_sha256(report_path),
            },
            "candidate_authority": {
                "path": str(candidates),
                "sha256": artifact_sha256(candidates),
            },
            "single_cell_exact_source": {
                "path": str(exact_source),
                "sha256": artifact_sha256(exact_source),
                "rows": counts["exact_source_rows"],
            },
            "adapter_code_sha256": artifact_sha256(Path(__file__).with_name("single_cell_fusion_adapter.py")),
            "binding_code_sha256": artifact_sha256(Path(__file__)),
            "fusion_input_eligible": True,
            "direct_target_evidence": False,
            "affects_discovery": True,
            "affects_confidence": True,
            "changes_primary_ranking": False,
            "family_to_exact_broadcast": False,
            "unavailable_encoding": "null_with_reason",
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "release_ready": False,
            "production_deployed": False,
        }
        _atomic_json(binding_path, binding)
        success_path = output / "SUCCESS.json"
        _atomic_json(
            success_path,
            {
                "status": binding["status"],
                "binding": binding_path.name,
                "binding_sha256": artifact_sha256(binding_path),
                "fusion_input_eligible": True,
                "primary_ranking_unchanged": True,
                "release_ready": False,
                "production_deployed": False,
            },
        )
        return {
            "output_root": str(output),
            "binding_path": str(binding_path),
            "binding_sha256": artifact_sha256(binding_path),
            "prediction_path": str(prediction_path),
            "prediction_sha256": artifact_sha256(prediction_path),
            **counts,
        }
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


__all__ = [
    "AUDIT_BINDING_FORMAT",
    "BINDING_FORMAT",
    "SingleCellFusionBindingError",
    "artifact_sha256",
    "materialize_single_cell_fusion_binding",
]
