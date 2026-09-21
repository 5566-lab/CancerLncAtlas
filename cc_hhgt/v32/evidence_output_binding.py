"""Post-training hash binding for the fresh pair-blocked V3.2 Evidence run.

The pair-blocked trainer predates explicit hashes for its generated physical
facts and execution code in ``TRAINING_MANIFEST.json``.  This verifier does
not alter that completed run.  It creates a separate immutable binding after
rehashing the successful training markers, five checkpoints, code, preflight,
and every downstream artifact needed by Interaction release materialisation.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from .input_lineage import artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CC_HHGT_V3_2_EVIDENCE_OUTPUT_BINDING_V1"
FORMAL_CANDIDATE_SHA256 = (
    "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
)
FORMAL_MEMBERSHIP_SHA256 = (
    "0ae85904df979046fcbfb7f781977947392e99735b819f4d90803869831871ef"
)
FORMAL_CANDIDATE_ROWS = 3_300_000
FORMAL_CANCER_COUNT = 33
FORMAL_EVIDENCE_CODE_SHA256 = (
    "4490a8a8cb5d1b4fe5e08bf3180909a152d56748b321144133ab077d9d278295"
)
FORMAL_EVIDENCE_RUNNER_SHA256 = (
    "3a9dfc7232cf58de0bc8df962c3f08e5ef9d1a7478812c4dbb7ee1e067a3273e"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OUTPUT_ARTIFACTS = {
    "physical_facts": "physical_interaction_facts.parquet",
    "event_lineage": "event_lineage.parquet",
    "rejected_mappings": "rejected_event_mappings.parquet",
    "evidence_predictions": "evidence_private_predictions.parquet",
    "split_integrity_audit": "PAIR_BLOCKED_SPLIT_AUDIT.json",
}


class EvidenceOutputBindingError(RuntimeError):
    """Raised when a completed Evidence output cannot be cryptographically bound."""


def _read_json(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise EvidenceOutputBindingError(f"Evidence {role} is missing/unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceOutputBindingError(f"Evidence {role} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise EvidenceOutputBindingError(f"Evidence {role} is not a JSON object")
    return value


def _resolve_declared_path(
    value: Any,
    relocations: Mapping[str, Path],
) -> Path:
    declared = str(value or "")
    if declared in relocations:
        return relocations[declared]
    return Path(declared).resolve()


def _declared_file(
    value: Any,
    expected: Path,
    role: str,
    relocations: Mapping[str, Path],
) -> None:
    if _resolve_declared_path(value, relocations) != expected.resolve():
        raise EvidenceOutputBindingError(f"Evidence {role} path drift")


def _declared_input(
    declaration: Any,
    *,
    expected_path: Path,
    expected_sha256: str,
    role: str,
    relocations: Mapping[str, Path],
) -> None:
    if not isinstance(declaration, Mapping):
        raise EvidenceOutputBindingError(f"Evidence manifest lacks {role}")
    _declared_file(declaration.get("path"), expected_path, role, relocations)
    if declaration.get("sha256") != expected_sha256:
        raise EvidenceOutputBindingError(f"Evidence {role} SHA256 drift")


def _parquet_rows(path: Path) -> int:
    if not path.is_file() or path.is_symlink():
        raise EvidenceOutputBindingError(f"Evidence parquet is missing/unsafe: {path}")
    return int(pq.ParquetFile(path).metadata.num_rows)


def _parquet_schema(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def materialize_evidence_output_binding(
    *,
    evidence_output_root: str | Path,
    evidence_code_path: str | Path,
    evidence_runner_path: str | Path,
    exact_membership_path: str | Path,
    exact_candidates_path: str | Path,
    output_root: str | Path,
    strict_formal_authority: bool = True,
    path_relocations: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Verify one successful pair-blocked run and write a separate binding."""

    evidence_root = Path(evidence_output_root).resolve()
    code_path = Path(evidence_code_path).resolve()
    runner_path = Path(evidence_runner_path).resolve()
    membership_path = Path(exact_membership_path).resolve()
    candidates_path = Path(exact_candidates_path).resolve()
    relocations: dict[str, Path] = {}
    for declared, local_value in (path_relocations or {}).items():
        declared_text = str(declared).strip()
        local = Path(local_value).resolve()
        if not declared_text or declared_text in relocations:
            raise EvidenceOutputBindingError("Evidence path relocation key is empty/duplicate")
        if not local.is_file() or local.is_symlink():
            raise EvidenceOutputBindingError(
                f"Evidence relocated source is missing/unsafe: {local}"
            )
        relocations[declared_text] = local
    for path, role in (
        (evidence_root, "output root"),
        (code_path, "training code"),
        (runner_path, "runner"),
        (membership_path, "exact membership"),
        (candidates_path, "exact candidates"),
    ):
        if (role == "output root" and not path.is_dir()) or (
            role != "output root" and (not path.is_file() or path.is_symlink())
        ):
            raise EvidenceOutputBindingError(f"Evidence {role} is missing/unsafe: {path}")
    hashes = {
        "code": artifact_sha256(code_path),
        "runner": artifact_sha256(runner_path),
        "membership": artifact_sha256(membership_path),
        "candidates": artifact_sha256(candidates_path),
    }
    if strict_formal_authority:
        expected = {
            "code": FORMAL_EVIDENCE_CODE_SHA256,
            "runner": FORMAL_EVIDENCE_RUNNER_SHA256,
            "membership": FORMAL_MEMBERSHIP_SHA256,
            "candidates": FORMAL_CANDIDATE_SHA256,
        }
        drift = {key: (hashes[key], value) for key, value in expected.items() if hashes[key] != value}
        if drift:
            raise EvidenceOutputBindingError(f"Formal Evidence authority hash drift: {drift}")

    training_manifest_path = evidence_root / "TRAINING_MANIFEST.json"
    training_success_path = evidence_root / "TRAINING_SUCCESS.json"
    manifest = _read_json(training_manifest_path, "TRAINING_MANIFEST.json")
    success = _read_json(training_success_path, "TRAINING_SUCCESS.json")
    manifest_sha = artifact_sha256(training_manifest_path)
    required_manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "evidence_private_eventset",
        "status": "SUCCESS_NEWLY_TRAINED",
        "training_status": "SUCCESS_NEWLY_TRAINED",
        "release_ready": False,
        "partial_not_publishable": True,
        "training_generation": "V3.2",
        "five_fresh_private_heads_attempted": True,
        "historical_evidence_checkpoint_loaded": False,
        "historical_evidence_result_loaded": False,
        "old_confidence_loaded": False,
        "pair_evidence_loaded": False,
        "family_spf_loaded": False,
        "family_to_exact_broadcast_used": False,
        "core_frozen": True,
        "core_detached": True,
        "physical_facts_separate_from_predictions": True,
        "main_ranking_modified": False,
        "prediction_role": "AUXILIARY_CONFIDENCE_ONLY",
    }
    for key, expected_value in required_manifest.items():
        if manifest.get(key) != expected_value:
            raise EvidenceOutputBindingError(
                f"Evidence training manifest has invalid {key}: {manifest.get(key)!r}"
            )
    if (
        success.get("status") != "SUCCESS_NEWLY_TRAINED"
        or success.get("release_ready") is not False
        or success.get("partial_not_publishable") is not True
        or success.get("trained_private_heads") != 5
        or int(success.get("optimizer_steps", 0)) <= 0
        or success.get("manifest_sha256") != manifest_sha
    ):
        raise EvidenceOutputBindingError("Evidence TRAINING_SUCCESS is invalid/stale")
    _declared_file(
        success.get("manifest"), training_manifest_path, "training manifest", relocations
    )

    authority = manifest.get("candidate_authority")
    if not isinstance(authority, Mapping):
        raise EvidenceOutputBindingError("Evidence candidate authority is missing")
    if (
        authority.get("sha256") != hashes["candidates"]
        or authority.get("sha256_pinned") is not True
        or int(authority.get("rows", -1)) != (FORMAL_CANDIDATE_ROWS if strict_formal_authority else int(authority.get("rows", -1)))
        or int(authority.get("cancers", -1)) != (FORMAL_CANCER_COUNT if strict_formal_authority else int(authority.get("cancers", -1)))
        or authority.get("old_prediction_or_ranking_columns_read") is not False
    ):
        raise EvidenceOutputBindingError("Evidence candidate authority is not formal/current")
    inputs = manifest.get("inputs")
    optional_inputs = manifest.get("optional_inputs")
    if not isinstance(inputs, Mapping) or not isinstance(optional_inputs, Mapping):
        raise EvidenceOutputBindingError("Evidence manifest input declarations are missing")
    _declared_input(
        inputs.get("pathway_members"),
        expected_path=membership_path,
        expected_sha256=hashes["membership"],
        role="exact membership",
        relocations=relocations,
    )
    _declared_input(
        optional_inputs.get("candidates"),
        expected_path=candidates_path,
        expected_sha256=hashes["candidates"],
        role="exact candidates",
        relocations=relocations,
    )

    preflight = manifest.get("split_preflight_gate")
    if not isinstance(preflight, Mapping) or any(
        preflight.get(key) is not True
        for key in (
            "pair_isolation_verified",
            "five_fold_provenance_exclusion_verified",
            "no_prediction_or_checkpoint_artifacts_verified",
        )
    ):
        raise EvidenceOutputBindingError("Evidence split preflight binding is incomplete")
    preflight_path = _resolve_declared_path(preflight.get("path", ""), relocations)
    preflight_success_path = _resolve_declared_path(
        preflight.get("success_marker_path", ""), relocations
    )
    if artifact_sha256(preflight_path) != preflight.get("sha256"):
        raise EvidenceOutputBindingError("Evidence split preflight manifest SHA drift")
    if artifact_sha256(preflight_success_path) != preflight.get("success_marker_sha256"):
        raise EvidenceOutputBindingError("Evidence split preflight success SHA drift")

    split = manifest.get("split_integrity_audit")
    split_path = evidence_root / _OUTPUT_ARTIFACTS["split_integrity_audit"]
    if (
        not isinstance(split, Mapping)
        or split.get("hard_pair_cross_fold_count") != 0
        or split.get("all_five_folds_populated") is not True
        or sorted(split.get("active_folds", [])) != list(range(5))
        or split.get("sha256") != artifact_sha256(split_path)
    ):
        raise EvidenceOutputBindingError("Evidence pair-blocked split integrity proof failed")
    _declared_file(
        split.get("path"), split_path, "split integrity audit", relocations
    )

    folds = manifest.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != {str(index) for index in range(5)}:
        raise EvidenceOutputBindingError("Evidence training manifest lacks five folds")
    checkpoint_records: dict[str, Any] = {}
    optimizer_steps = 0
    for fold_id in range(5):
        fold = folds[str(fold_id)]
        if not isinstance(fold, Mapping):
            raise EvidenceOutputBindingError(f"Evidence fold {fold_id} is invalid")
        checkpoint = _resolve_declared_path(fold.get("checkpoint_path", ""), relocations)
        if checkpoint.parent != evidence_root / f"patient_fold={fold_id}":
            raise EvidenceOutputBindingError(f"Evidence fold {fold_id} checkpoint escaped fold root")
        observed_checkpoint_sha = artifact_sha256(checkpoint)
        history = checkpoint.parent / "training_history.tsv"
        if not history.is_file() or history.is_symlink():
            raise EvidenceOutputBindingError(f"Evidence fold {fold_id} training history is missing")
        history_rows = max(
            sum(1 for line in history.read_text(encoding="utf-8").splitlines() if line.strip()) - 1,
            0,
        )
        if history_rows <= 0:
            raise EvidenceOutputBindingError(f"Evidence fold {fold_id} training history is empty")
        initial = str(fold.get("initial_parameter_sha256", ""))
        final = str(fold.get("final_parameter_sha256", ""))
        fold_steps = int(fold.get("optimizer_steps", 0))
        required_true = (
            fold.get("private_parameters_fresh_init") is True
            and fold.get("initialized_from_checkpoint") is False
            and fold.get("historical_evidence_checkpoint_allowed") is False
            and fold.get("historical_evidence_result_allowed") is False
            and fold.get("family_spf_allowed") is False
            and fold.get("pair_evidence_supervision_allowed") is False
            and fold.get("core_frozen") is True
            and fold.get("core_detached") is True
            and fold.get("confidence_supervision_available") is True
        )
        if fold.get("checkpoint_sha256") != observed_checkpoint_sha:
            raise EvidenceOutputBindingError(
                f"Evidence fold {fold_id} checkpoint SHA256 drift"
            )
        if (
            int(fold.get("patient_fold", -1)) != fold_id
            or not required_true
            or fold_steps <= 0
            or not _SHA256.fullmatch(initial)
            or not _SHA256.fullmatch(final)
            or initial == final
        ):
            raise EvidenceOutputBindingError(f"Evidence fold {fold_id} is not fresh/successful")
        optimizer_steps += fold_steps
        checkpoint_records[str(fold_id)] = {
            "path": str(checkpoint),
            "sha256": observed_checkpoint_sha,
            "optimizer_steps": fold_steps,
            "initial_parameter_sha256": initial,
            "final_parameter_sha256": final,
            "core_checkpoint_sha256": fold.get("core_checkpoint_sha256"),
            "core_parameter_sha256": fold.get("core_parameter_sha256"),
            "training_history_path": str(history),
            "training_history_sha256": artifact_sha256(history),
            "training_history_rows": history_rows,
        }
    counts = manifest.get("counts")
    if (
        not isinstance(counts, Mapping)
        or int(counts.get("trained_private_heads", -1)) != 5
        or int(counts.get("optimizer_steps", -1)) != optimizer_steps
    ):
        raise EvidenceOutputBindingError("Evidence aggregate optimizer/head counts drifted")

    artifacts: dict[str, Any] = {}
    rows_by_role: dict[str, int] = {}
    for role, name in _OUTPUT_ARTIFACTS.items():
        path = evidence_root / name
        if role == "split_integrity_audit":
            rows = None
        else:
            rows = _parquet_rows(path)
            rows_by_role[role] = rows
        artifacts[role] = {
            "path": str(path),
            "sha256": artifact_sha256(path),
            "rows": rows,
            "columns": _parquet_schema(path) if rows is not None else None,
        }
    expected_rows = {
        "physical_facts": int(counts.get("physical_facts", -1)),
        "event_lineage": int(counts.get("materialized_bag_events", -1)),
        "rejected_mappings": int(counts.get("rejected_rows", -1)),
        "evidence_predictions": int(counts.get("candidate_rows", -1)),
    }
    if rows_by_role != expected_rows or rows_by_role["physical_facts"] <= 0:
        raise EvidenceOutputBindingError(
            f"Evidence output row-count drift: observed={rows_by_role}, expected={expected_rows}"
        )

    prediction = evidence_root / _OUTPUT_ARTIFACTS["evidence_predictions"]
    physical = evidence_root / _OUTPUT_ARTIFACTS["physical_facts"]
    lineage = evidence_root / _OUTPUT_ARTIFACTS["event_lineage"]
    con = duckdb.connect(":memory:")
    try:
        prediction_audit = con.execute(
            f"""
            SELECT count(DISTINCT training_run_id), min(training_run_id),
                   count_if(analysis_version <> ?),
                   count_if(changes_primary_ranking IS DISTINCT FROM false)
            FROM read_parquet({_sql_path(prediction)})
            """,
            [ANALYSIS_VERSION],
        ).fetchone()
        physical_bad = int(con.execute(
            f"SELECT count(*) FROM read_parquet({_sql_path(physical)}) WHERE is_prediction IS DISTINCT FROM false"
        ).fetchone()[0])
        lineage_bad = int(con.execute(
            f"SELECT count(*) FROM read_parquet({_sql_path(lineage)}) WHERE family_broadcast_used IS DISTINCT FROM false"
        ).fetchone()[0])
    finally:
        con.close()
    if (
        int(prediction_audit[0]) != 1
        or not str(prediction_audit[1] or "").startswith("V32-EVIDENCE-TRAIN-")
        or int(prediction_audit[2])
        or int(prediction_audit[3])
        or physical_bad
        or lineage_bad
    ):
        raise EvidenceOutputBindingError("Evidence output semantic validation failed")
    training_run_id = str(prediction_audit[1])

    output = Path(output_root).resolve()
    if output.exists():
        raise EvidenceOutputBindingError(f"Evidence binding output reuse is forbidden: {output}")
    output.mkdir(parents=True)
    try:
        binding = {
            "format": BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND",
            "release_ready": False,
            "production_deployed": False,
            "evidence_training_run_id": training_run_id,
            "evidence_module_still_auxiliary_partial": True,
            "interaction_materialization_input_eligible": True,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "family_to_exact_broadcast": False,
            "five_fresh_private_heads_verified": True,
            "optimizer_steps_total": optimizer_steps,
            "authorities": {
                "evidence_code": {"path": str(code_path), "sha256": hashes["code"]},
                "evidence_runner": {"path": str(runner_path), "sha256": hashes["runner"]},
                "exact_membership": {"path": str(membership_path), "sha256": hashes["membership"]},
                "exact_candidates": {"path": str(candidates_path), "sha256": hashes["candidates"]},
                "training_manifest": {"path": str(training_manifest_path), "sha256": manifest_sha},
                "training_success": {"path": str(training_success_path), "sha256": artifact_sha256(training_success_path)},
                "split_preflight": {"path": str(preflight_path), "sha256": preflight.get("sha256")},
                "split_preflight_success": {"path": str(preflight_success_path), "sha256": preflight.get("success_marker_sha256")},
            },
            "artifacts": artifacts,
            "checkpoints": checkpoint_records,
            "counts": rows_by_role,
            "source_path_relocations": [
                {
                    "declared_path": declared,
                    "local_path": str(local),
                    "sha256": artifact_sha256(local),
                }
                for declared, local in sorted(relocations.items())
            ],
        }
        binding_path = output / "EVIDENCE_OUTPUT_BINDING.json"
        temporary = output / ".EVIDENCE_OUTPUT_BINDING.json.tmp"
        temporary.write_text(
            json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, binding_path)
        success_binding = {
            "status": binding["status"],
            "release_ready": False,
            "binding": binding_path.name,
            "binding_sha256": artifact_sha256(binding_path),
        }
        success_path = output / "SUCCESS.json"
        success_path.write_text(
            json.dumps(success_binding, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return binding
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


__all__ = [
    "ANALYSIS_VERSION",
    "BINDING_FORMAT",
    "EvidenceOutputBindingError",
    "FORMAL_CANDIDATE_SHA256",
    "FORMAL_EVIDENCE_CODE_SHA256",
    "FORMAL_EVIDENCE_RUNNER_SHA256",
    "FORMAL_MEMBERSHIP_SHA256",
    "materialize_evidence_output_binding",
]
