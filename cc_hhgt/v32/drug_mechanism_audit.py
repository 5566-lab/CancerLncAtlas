"""Independent fail-closed audit for a formal V3.2 Drug mechanism release.

This verifier intentionally does not import the mechanism materialiser or its
query loader.  It rechecks the pinned manifest, source hashes, formal Drug run,
and every materialised row from first principles.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .input_lineage import artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
MECHANISM_FORMAT = "CC_HHGT_V3_2_LNCRNA_EXACT_PATHWAY_TARGET_DRUG_MECHANISM_V1"
FORMAL_CANDIDATE_SHA256 = "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
FORMAL_MEMBERSHIP_SHA256 = "0ae85904df979046fcbfb7f781977947392e99735b819f4d90803869831871ef"
PROBABILITY_COLUMN = "drug_response_association_probability"
NATIVE_TARGET_KEYS = ["cancer_id", "lncrna_id", "drug_id"]
SEMANTIC_DECLARATIONS: dict[str, Any] = {
    "native_target_keys": NATIVE_TARGET_KEYS,
    "actionability_separate_from_exact_pathway": True,
    "does_not_change_primary_pathway_ranking": True,
    "drug_response_association_probability_not_efficacy_or_direction": True,
    "signed_rho_private_only": True,
}
EXPECTED_COLUMNS = (
    "mechanism_path_id",
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "target_gene_id",
    "target_gene_symbol",
    "drug_id",
    "drug_name",
    PROBABILITY_COLUMN,
    "training_run_id",
    "prediction_fold_mask",
    "cell_line_folds_with_prediction",
    "target_mapping_method",
    "mechanism_type",
    "mechanism_semantics",
    "causal_mechanism_claimed",
    "target_contribution_claimed",
    "family_to_exact_broadcast",
    "analysis_version",
    "generation",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DrugMechanismAuditError(RuntimeError):
    """Raised when an independently audited release fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DrugMechanismAuditError(message)


def _json_object(path: Path, role: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"Missing or unsafe {role}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DrugMechanismAuditError(f"Invalid {role}: {path}") from exc
    _require(isinstance(payload, dict), f"{role} is not a JSON object")
    return payload


def _declared_path(value: Any, role: str) -> Path:
    _require(isinstance(value, str) and value.strip() != "", f"{role} path missing")
    path = Path(value).resolve()
    _require(path.exists() and not path.is_symlink(), f"{role} path missing/unsafe: {path}")
    return path


def _declared_sha(value: Any, role: str) -> str:
    digest = str(value or "").lower()
    _require(bool(_SHA256.fullmatch(digest)), f"{role} SHA-256 is invalid")
    return digest


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"Independent audit output reuse is forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit_drug_mechanism_release(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    audit_output: str | Path | None = None,
    expected_candidate_sha256: str = FORMAL_CANDIDATE_SHA256,
    expected_membership_sha256: str = FORMAL_MEMBERSHIP_SHA256,
) -> dict[str, Any]:
    """Verify one formal mechanism release without trusting its query loader."""

    manifest_source = Path(manifest_path).resolve()
    expected_manifest = _declared_sha(expected_manifest_sha256, "expected manifest")
    _require(
        artifact_sha256(manifest_source) == expected_manifest,
        "Drug mechanism manifest SHA-256 mismatch",
    )
    manifest = _json_object(manifest_source, "Drug mechanism manifest")
    required_manifest = {
        "format": MECHANISM_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "drug",
        "status": "SUCCESS_NEWLY_MATERIALIZED_V32",
        "release_ready": False,
        "production_deployed": False,
        "mechanism_semantics": "STRUCTURAL_HYPOTHESIS_NOT_MODEL_ATTRIBUTION",
        "causal_mechanism_claimed": False,
        "target_contribution_claimed": False,
        "family_to_exact_broadcast": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "current_v32_drug_predictions_used_as_binding": True,
        **SEMANTIC_DECLARATIONS,
    }
    for field, expected in required_manifest.items():
        _require(manifest.get(field) == expected, f"Drug mechanism manifest invalid {field}")

    binding = manifest.get("formal_drug_bundle_binding")
    _require(isinstance(binding, dict), "Drug mechanism manifest lacks formal Drug binding")
    training_run_id = str(binding.get("training_run_id", "")).strip()
    _require(training_run_id != "", "Formal Drug binding lacks training_run_id")
    success_path = _declared_path(binding.get("success_path"), "formal Drug SUCCESS")
    lineage_path = _declared_path(binding.get("lineage_path"), "formal Drug lineage")
    success_sha = _declared_sha(binding.get("success_sha256"), "formal Drug SUCCESS")
    lineage_sha = _declared_sha(binding.get("lineage_sha256"), "formal Drug lineage")
    _require(artifact_sha256(success_path) == success_sha, "Formal Drug SUCCESS SHA drift")
    _require(artifact_sha256(lineage_path) == lineage_sha, "Formal Drug lineage SHA drift")
    success = _json_object(success_path, "formal Drug SUCCESS")
    lineage = _json_object(lineage_path, "formal Drug lineage")
    _require(success.get("status") == "SUCCESS", "Formal Drug status is not SUCCESS")
    _require(success.get("release_ready") is True, "Formal Drug SUCCESS is not release-ready")
    _require(lineage.get("training_status") == "SUCCESS", "Formal Drug lineage is not SUCCESS")
    _require(lineage.get("release_ready") is True, "Formal Drug lineage is not release-ready")
    for payload, role in ((success, "SUCCESS"), (lineage, "lineage")):
        _require(payload.get("analysis_version") == ANALYSIS_VERSION, f"Formal Drug {role} is not V3.2")
        _require(payload.get("module_id") == "drug", f"Formal Drug {role} module drift")
        _require(payload.get("training_run_id") == training_run_id, f"Formal Drug {role} run-id drift")
    required_truths = (
        "private_head_trained_from_scratch",
        "all_five_folds_have_optimizer_updates",
        "all_non_null_predictions_newly_trained_v32",
        "available_keys_unique",
    )
    required_falses = (
        "old_checkpoint_loaded",
        "old_predictions_used_as_features",
        "old_rankings_used_as_outputs",
        "old_gdsc_prism_association_tables_used",
        "partial_not_publishable",
    )
    _require(all(lineage.get(field) is True for field in required_truths), "Fresh Drug truths missing")
    _require(all(lineage.get(field) is False for field in required_falses), "Historical/partial Drug flag present")

    inputs = manifest.get("inputs")
    _require(isinstance(inputs, dict), "Drug mechanism manifest lacks input declarations")
    expected_authorities = {
        "current_exact_candidates": _declared_sha(expected_candidate_sha256, "candidate authority"),
        "current_exact_membership": _declared_sha(expected_membership_sha256, "membership authority"),
    }
    input_audit: dict[str, dict[str, Any]] = {}
    for role in (
        "current_v32_drug_predictions",
        "current_exact_candidates",
        "current_exact_membership",
        "curated_drug_targets",
    ):
        declaration = inputs.get(role)
        _require(isinstance(declaration, dict), f"Drug mechanism input missing: {role}")
        path = _declared_path(declaration.get("path"), role)
        declared = _declared_sha(declaration.get("sha256"), role)
        observed = artifact_sha256(path)
        _require(observed == declared, f"Drug mechanism input SHA drift: {role}")
        if role in expected_authorities:
            _require(observed == expected_authorities[role], f"Drug mechanism input is not formal authority: {role}")
        input_audit[role] = {"path": str(path), "sha256": observed}
    prediction_path = Path(input_audit["current_v32_drug_predictions"]["path"])
    _require(Path(str(success.get("prediction_path", ""))).resolve() == prediction_path, "SUCCESS prediction path drift")
    prediction_sha = input_audit["current_v32_drug_predictions"]["sha256"]
    _require(success.get("prediction_sha256") == prediction_sha, "SUCCESS prediction SHA drift")
    _require(lineage.get("prediction_sha256") == prediction_sha, "Lineage prediction SHA drift")
    _require(success.get("lineage_sha256") == lineage_sha, "SUCCESS lineage SHA drift")
    _require(not (success_path.parent / "RUN_IN_PROGRESS.json").exists(), "Formal Drug run still in progress")
    _require(not (success_path.parent / "RUN_FAILED.json").exists(), "Formal Drug run has failure marker")

    declaration = manifest.get("artifact")
    _require(isinstance(declaration, dict), "Drug mechanism artifact declaration missing")
    relative = Path(str(declaration.get("path", "")))
    _require(not relative.is_absolute(), "Drug mechanism artifact path must be relative")
    release_root = manifest_source.parent.resolve()
    artifact = (release_root / relative).resolve()
    try:
        artifact.relative_to(release_root)
    except ValueError as exc:
        raise DrugMechanismAuditError("Drug mechanism artifact escaped release root") from exc
    _require(artifact.is_file() and not artifact.is_symlink(), "Drug mechanism artifact missing/unsafe")
    artifact_sha = _declared_sha(declaration.get("sha256"), "Drug mechanism artifact")
    _require(artifact_sha256(artifact) == artifact_sha, "Drug mechanism artifact SHA drift")
    declared_rows = declaration.get("rows")
    _require(isinstance(declared_rows, int) and not isinstance(declared_rows, bool) and declared_rows > 0,
             "Drug mechanism artifact row declaration invalid")

    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise DrugMechanismAuditError("duckdb is required for independent Drug mechanism audit") from exc
    relation = f"read_parquet({_sql_path(artifact)})"
    con = duckdb.connect(":memory:", config={"threads": "2", "preserve_insertion_order": "false"})
    try:
        columns = tuple(row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall())
        _require(columns == EXPECTED_COLUMNS, f"Drug mechanism artifact schema drift: {columns}")
        audit = con.execute(
            f"""
            SELECT
              count(*) AS rows,
              count(*) FILTER (
                WHERE mechanism_path_id IS NOT NULL AND mechanism_path_id <> ''
                  AND cancer_id IS NOT NULL AND cancer_id <> ''
                  AND lncrna_id IS NOT NULL AND lncrna_id <> ''
                  AND pathway_id IS NOT NULL AND pathway_id <> ''
                  AND target_gene_id IS NOT NULL AND target_gene_id <> ''
                  AND drug_id IS NOT NULL AND drug_id <> ''
              ) AS complete_key_rows,
              count(*) FILTER (
                WHERE {PROBABILITY_COLUMN} IS NOT NULL
                  AND isfinite({PROBABILITY_COLUMN})
                  AND {PROBABILITY_COLUMN} BETWEEN 0 AND 1
              ) AS valid_probability_rows,
              count(*) FILTER (
                WHERE analysis_version = ?
                  AND training_run_id = ?
                  AND mechanism_type = 'CURATED_DRUG_TARGET_MEMBER_OF_CURRENT_EXACT_PATHWAY'
                  AND mechanism_semantics = 'STRUCTURAL_HYPOTHESIS_NOT_MODEL_ATTRIBUTION'
                  AND NOT causal_mechanism_claimed
                  AND NOT target_contribution_claimed
                  AND NOT family_to_exact_broadcast
                  AND generation = 'V3.2_FRESH_STRUCTURAL_MECHANISM_BINDING'
              ) AS valid_semantic_rows,
              count(DISTINCT mechanism_path_id) AS unique_path_ids
            FROM {relation}
            """,
            [ANALYSIS_VERSION, training_run_id],
        ).fetchone()
        duplicates = int(
            con.execute(
                f"""
                SELECT count(*) FROM (
                  SELECT cancer_id, lncrna_id, pathway_id, target_gene_id, drug_id
                  FROM {relation}
                  GROUP BY ALL HAVING count(*) <> 1
                )
                """
            ).fetchone()[0]
        )
    finally:
        con.close()
    row_counts = tuple(map(int, audit))
    _require(row_counts == (declared_rows,) * 5, f"Drug mechanism row/semantic audit failed: {row_counts}")
    _require(duplicates == 0, f"Drug mechanism duplicate structural paths: {duplicates}")

    result = {
        "status": "PASS",
        "analysis_version": ANALYSIS_VERSION,
        "audit_kind": "INDEPENDENT_FORMAL_DRUG_MECHANISM_RELEASE",
        "manifest_path": str(manifest_source),
        "manifest_sha256": expected_manifest,
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_sha,
        "mechanism_paths": declared_rows,
        "training_run_id": training_run_id,
        "semantic_declarations": SEMANTIC_DECLARATIONS,
        "formal_candidate_sha256": expected_authorities["current_exact_candidates"],
        "formal_membership_sha256": expected_authorities["current_exact_membership"],
        "input_artifacts": input_audit,
        "all_rows_valid": True,
        "duplicate_structural_paths": 0,
        "release_ready": False,
        "production_deployed": False,
    }
    if audit_output is not None:
        _write_new_json(Path(audit_output).resolve(), result)
    return result


__all__ = [
    "DrugMechanismAuditError",
    "FORMAL_CANDIDATE_SHA256",
    "FORMAL_MEMBERSHIP_SHA256",
    "NATIVE_TARGET_KEYS",
    "SEMANTIC_DECLARATIONS",
    "audit_drug_mechanism_release",
]
