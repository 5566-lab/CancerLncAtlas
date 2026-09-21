#!/usr/bin/env python
"""Independent fail-closed, read-only audit of formal V3.2 Evidence outputs.

The script never writes files.  It emits a wrapper-compatible post-audit JSON
object on stdout and can optionally bind that audit to an already materialized
R2 Evidence output binding.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import duckdb
import numpy as np
import pyarrow.parquet as pq
import torch


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
POST_AUDIT_FORMAT = "CC_HHGT_V3_2_EVIDENCE_INDEPENDENT_POST_AUDIT_V1"
R2_BINDING_FORMAT = "CC_HHGT_V3_2_EVIDENCE_OUTPUT_BINDING_V1"
FORMAL_CANDIDATE_SHA256 = "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
FORMAL_MEMBERSHIP_SHA256 = "0ae85904df979046fcbfb7f781977947392e99735b819f4d90803869831871ef"
FORMAL_PREFLIGHT_SHA256 = "e450d4eadf087e94db0a8d6d09dc37552cd658b7842bdac2ffa8ef6361e12922"
FORMAL_PREFLIGHT_SUCCESS_SHA256 = "a2b5ef1b33ec3d829a47f13f588831e3981f342c83f65e33ad01fa056ab2d446"
FORMAL_CORE_MANIFEST_SHA256 = "1c53fe66dde2c74eca9eb3d6f2d7dc52f3651b9cf80c138656d00f3c5bdf0e3a"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESIDUALS = {
    "residual_pmid_overlap_count": 0,
    "residual_source_event_overlap_count": 0,
    "residual_source_record_overlap_count": 0,
    "residual_train_evaluation_provenance_overlap_count": 0,
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def file_sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def history_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return len(list(csv.DictReader(handle, delimiter="\t")))


def audit_fold(root: Path, fold_id: int, declared: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = root / f"patient_fold={fold_id}" / "private_eventset_state.pt"
    history = checkpoint.parent / "training_history.tsv"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(payload, Mapping), f"fold {fold_id} checkpoint is not a mapping")
    state = payload.get("private_model_state")
    require(isinstance(state, Mapping) and bool(state), f"fold {fold_id} lacks private state")
    checkpoint_sha = file_sha256(checkpoint)
    history_sha = file_sha256(history)
    recomputed = state_sha256(state)
    initial = str(payload.get("initial_parameter_sha256", ""))
    final = str(payload.get("final_parameter_sha256", ""))
    steps = int(payload.get("optimizer_steps", 0))
    required = {
        "patient_fold": fold_id,
        "private_parameters_fresh_init": True,
        "initialized_from_checkpoint": False,
        "historical_evidence_checkpoint_allowed": False,
        "historical_evidence_result_allowed": False,
        "pair_evidence_supervision_allowed": False,
        "family_spf_allowed": False,
        "core_frozen": True,
        "core_detached": True,
        "contains_core_parameters": False,
        "contains_primary_ranking_parameters": False,
        "confidence_supervision_available": True,
        "direction_supervision_available": True,
        "core_manifest_sha256": FORMAL_CORE_MANIFEST_SHA256,
    }
    for key, expected in required.items():
        require(payload.get(key) == expected, f"fold {fold_id} invalid {key}")
    provenance = payload.get("provenance_exclusion_audit")
    require(isinstance(provenance, Mapping), f"fold {fold_id} lacks provenance audit")
    residuals = {key: int(provenance.get(key, -1)) for key in _RESIDUALS}
    require(residuals == _RESIDUALS, f"fold {fold_id} provenance residual")
    require(_SHA256.fullmatch(initial) is not None, f"fold {fold_id} invalid initial hash")
    require(_SHA256.fullmatch(final) is not None, f"fold {fold_id} invalid final hash")
    require(initial != final and recomputed == final and steps > 0, f"fold {fold_id} state/steps failed")
    require(history_rows(history) > 0, f"fold {fold_id} empty history")
    comparisons = {
        "checkpoint_sha256": checkpoint_sha,
        "initial_parameter_sha256": initial,
        "final_parameter_sha256": final,
        "optimizer_steps": steps,
        "core_checkpoint_sha256": payload.get("core_checkpoint_sha256"),
        "core_parameter_sha256": payload.get("core_parameter_sha256"),
        "core_manifest_sha256": payload.get("core_manifest_sha256"),
    }
    for key, observed in comparisons.items():
        require(declared.get(key) == observed, f"fold {fold_id} manifest/checkpoint drift: {key}")
    return {
        "patient_fold": fold_id,
        "checkpoint_sha256": checkpoint_sha,
        "training_history_sha256": history_sha,
        "initial_parameter_sha256": initial,
        "final_parameter_sha256": final,
        "recomputed_final_parameter_sha256": recomputed,
        "optimizer_steps": steps,
        "core_checkpoint_sha256": payload.get("core_checkpoint_sha256"),
        "core_parameter_sha256": payload.get("core_parameter_sha256"),
        "private_parameters_fresh_init": True,
        "initialized_from_checkpoint": False,
        "historical_evidence_checkpoint_allowed": False,
        "historical_evidence_result_allowed": False,
        "pair_evidence_supervision_allowed": False,
        "family_spf_allowed": False,
        "core_frozen": True,
        "core_detached": True,
        "contains_core_parameters": False,
        "contains_primary_ranking_parameters": False,
        "confidence_supervision_available": True,
        "direction_supervision_available": True,
        "provenance_residuals_zero": True,
        "provenance_residuals": residuals,
    }


def audit_predictions(path: Path, candidates: Path) -> dict[str, Any]:
    metadata = pq.ParquetFile(path)
    columns = list(metadata.schema_arrow.names)
    required_columns = {
        "cancer_id", "lncrna_id", "pathway_id", "evidence_confidence_probability",
        "uncertainty", "direction", "availability", "unavailable_reason", "failure_reason",
        "analysis_version", "training_run_id", "changes_primary_ranking", "main_ranking_modified",
    }
    require(required_columns.issubset(columns), "prediction schema is incomplete")
    con = duckdb.connect(":memory:")
    try:
        row = con.execute(f"""
            SELECT
              count(*), count_if(availability IS TRUE), count_if(availability IS FALSE),
              count_if(availability IS FALSE AND evidence_confidence_probability IS NOT NULL),
              count_if(availability IS FALSE AND trim(coalesce(unavailable_reason, '')) = ''),
              count_if(availability IS FALSE AND trim(coalesce(failure_reason, '')) = ''),
              count_if(availability IS TRUE AND evidence_confidence_probability IS NULL),
              count_if(evidence_confidence_probability IS NOT NULL AND
                       (NOT isfinite(evidence_confidence_probability) OR
                        evidence_confidence_probability < 0 OR evidence_confidence_probability > 1)),
              count_if(changes_primary_ranking IS DISTINCT FROM false),
              count_if(main_ranking_modified IS DISTINCT FROM false),
              count_if(analysis_version <> ?),
              count(DISTINCT training_run_id), min(training_run_id), max(training_run_id),
              count(DISTINCT cancer_id),
              count_if(cancer_id IS NULL OR lncrna_id IS NULL OR pathway_id IS NULL),
              count_if(availability IS FALSE AND uncertainty IS NOT NULL),
              count_if(availability IS TRUE AND uncertainty IS NULL),
              count_if(uncertainty IS NOT NULL AND (NOT isfinite(uncertainty) OR uncertainty < 0 OR uncertainty > 1)),
              count_if(coalesce(failure_reason, '') <> coalesce(unavailable_reason, '')),
              min(evidence_confidence_probability), max(evidence_confidence_probability)
            FROM read_parquet({sql_path(path)})
        """, [ANALYSIS_VERSION]).fetchone()
        unique_keys = int(con.execute(f"""
            SELECT count(*) FROM (
              SELECT cancer_id, lncrna_id, pathway_id
              FROM read_parquet({sql_path(path)}) GROUP BY ALL
            )
        """).fetchone()[0])
        normalized_candidates = f"""
            SELECT upper(trim(cancer_id)) AS cancer_id,
                   regexp_replace(upper(trim(lncrna_id)), '^LNC:', '') AS lncrna_id,
                   upper(trim(pathway_id)) AS pathway_id
            FROM read_parquet({sql_path(candidates)})
        """
        prediction_only = int(con.execute(f"""
            SELECT count(*) FROM (
              SELECT cancer_id, lncrna_id, pathway_id FROM read_parquet({sql_path(path)})
              EXCEPT
              {normalized_candidates}
            )
        """).fetchone()[0])
        candidate_only = int(con.execute(f"""
            SELECT count(*) FROM (
              {normalized_candidates}
              EXCEPT
              SELECT cancer_id, lncrna_id, pathway_id FROM read_parquet({sql_path(path)})
            )
        """).fetchone()[0])
        prediction_noncanonical_lnc = int(con.execute(f"""
            SELECT count(*) FROM read_parquet({sql_path(path)})
            WHERE NOT regexp_full_match(lncrna_id, 'ENSG[0-9]+')
        """).fetchone()[0])
        candidate_noncanonical_lnc = int(con.execute(f"""
            SELECT count(*) FROM read_parquet({sql_path(candidates)})
            WHERE NOT regexp_full_match(lncrna_id, 'LNC:ENSG[0-9]+')
        """).fetchone()[0])
        candidate_lnc_prefixed_rows = int(con.execute(f"""
            SELECT count(*) FROM read_parquet({sql_path(candidates)})
            WHERE regexp_full_match(lncrna_id, 'LNC:ENSG[0-9]+')
        """).fetchone()[0])
        prediction_lnc_unprefixed_rows = int(con.execute(f"""
            SELECT count(*) FROM read_parquet({sql_path(path)})
            WHERE regexp_full_match(lncrna_id, 'ENSG[0-9]+')
        """).fetchone()[0])
        candidate_pathway_case_normalized_rows = int(con.execute(f"""
            SELECT count(*) FROM read_parquet({sql_path(candidates)})
            WHERE pathway_id <> upper(trim(pathway_id))
        """).fetchone()[0])
        reasons = dict(con.execute(f"""
            SELECT unavailable_reason, count(*)
            FROM read_parquet({sql_path(path)}) WHERE availability IS FALSE
            GROUP BY unavailable_reason ORDER BY count(*) DESC
        """).fetchall())
    finally:
        con.close()
    names = (
        "rows", "available_rows", "unavailable_rows",
        "unavailable_probability_nonnull_rows", "unavailable_reason_missing_rows",
        "unavailable_failure_reason_missing_rows", "available_probability_null_rows",
        "probability_out_of_range_rows", "changes_primary_ranking_true_rows",
        "main_ranking_modified_true_rows", "analysis_version_mismatch_rows",
        "distinct_training_run_ids", "training_run_id", "max_training_run_id",
        "distinct_cancers", "null_exact_key_rows", "unavailable_uncertainty_nonnull_rows",
        "available_uncertainty_null_rows", "uncertainty_out_of_range_rows",
        "reason_mismatch_rows", "minimum_probability", "maximum_probability",
    )
    result = dict(zip(names, row, strict=True))
    result.update({
        "columns": columns,
        "unique_exact_keys": unique_keys,
        "duplicate_exact_key_rows": int(result["rows"]) - unique_keys,
        "prediction_only_exact_keys": prediction_only,
        "candidate_only_exact_keys": candidate_only,
        "prediction_noncanonical_lnc_rows": prediction_noncanonical_lnc,
        "candidate_noncanonical_lnc_rows": candidate_noncanonical_lnc,
        "candidate_lnc_prefixed_rows": candidate_lnc_prefixed_rows,
        "prediction_lnc_unprefixed_rows": prediction_lnc_unprefixed_rows,
        "candidate_pathway_case_normalized_rows": candidate_pathway_case_normalized_rows,
        "candidate_identifier_normalization": "TRIM_UPPERCASE_AND_STRIP_REQUIRED_LNC_PREFIX",
        "unavailable_reason_counts": reasons,
    })
    require(int(result["rows"]) == 3_300_000, "prediction row count is not 3.3M")
    require(int(result["distinct_cancers"]) == 33, "prediction cancer count is not 33")
    require(str(result["training_run_id"]).startswith("V32-EVIDENCE-TRAIN-"), "run id is invalid")
    require(result["training_run_id"] == result["max_training_run_id"], "multiple run IDs")
    zero_fields = (
        "unavailable_probability_nonnull_rows", "unavailable_reason_missing_rows",
        "unavailable_failure_reason_missing_rows", "available_probability_null_rows",
        "probability_out_of_range_rows", "changes_primary_ranking_true_rows",
        "main_ranking_modified_true_rows", "analysis_version_mismatch_rows",
        "null_exact_key_rows", "unavailable_uncertainty_nonnull_rows",
        "available_uncertainty_null_rows", "uncertainty_out_of_range_rows",
        "reason_mismatch_rows", "duplicate_exact_key_rows", "prediction_only_exact_keys",
        "candidate_only_exact_keys",
        "prediction_noncanonical_lnc_rows", "candidate_noncanonical_lnc_rows",
    )
    for field in zero_fields:
        require(int(result[field]) == 0, f"prediction audit failed: {field}")
    require(int(result["distinct_training_run_ids"]) == 1, "prediction run-id cardinality failed")
    require(candidate_lnc_prefixed_rows == 3_300_000, "candidate LNC prefix closure failed")
    require(prediction_lnc_unprefixed_rows == 3_300_000, "prediction ENSG closure failed")
    require(candidate_pathway_case_normalized_rows == 12_141, "pathway case-normalization count drift")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_output_root")
    parser.add_argument("--r2-binding")
    parser.add_argument("--output", help="Optional new JSON path; overwrite is forbidden")
    args = parser.parse_args()
    root = Path(args.evidence_output_root).resolve()
    require(root.is_dir() and not root.is_symlink(), "invalid Evidence output root")
    binding_path = Path(args.r2_binding).resolve() if args.r2_binding else None
    binding = read_json(binding_path) if binding_path is not None else None
    if binding is not None:
        require(binding.get("format") == R2_BINDING_FORMAT, "R2 binding format failed")
        require(binding.get("analysis_version") == ANALYSIS_VERSION, "R2 binding version failed")
        require(binding.get("status") == "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND", "R2 binding status failed")
        require(binding.get("release_ready") is False, "R2 binding release flag failed")
    manifest_path = root / "TRAINING_MANIFEST.json"
    success_path = root / "TRAINING_SUCCESS.json"
    manifest = read_json(manifest_path)
    success = read_json(success_path)
    manifest_sha = file_sha256(manifest_path)
    required_manifest = {
        "analysis_version": ANALYSIS_VERSION, "status": "SUCCESS_NEWLY_TRAINED",
        "training_status": "SUCCESS_NEWLY_TRAINED", "training_generation": "V3.2",
        "release_ready": False, "partial_not_publishable": True,
        "historical_evidence_checkpoint_loaded": False,
        "historical_evidence_result_loaded": False, "old_confidence_loaded": False,
        "pair_evidence_loaded": False, "family_spf_loaded": False,
        "family_to_exact_broadcast_used": False, "core_frozen": True,
        "core_detached": True, "main_ranking_modified": False,
        "physical_facts_separate_from_predictions": True,
        "prediction_role": "AUXILIARY_CONFIDENCE_ONLY",
    }
    for key, expected in required_manifest.items():
        require(manifest.get(key) == expected, f"manifest invalid {key}")
    require(success.get("status") == "SUCCESS_NEWLY_TRAINED", "success status failed")
    require(success.get("manifest_sha256") == manifest_sha, "success manifest hash drift")
    require(success.get("release_ready") is False and success.get("partial_not_publishable") is True, "success release flags failed")
    authority = manifest.get("candidate_authority", {})
    require(authority.get("sha256") == FORMAL_CANDIDATE_SHA256, "candidate hash failed")
    require(authority.get("rows") == 3_300_000 and authority.get("cancers") == 33, "candidate dimensions failed")
    membership = manifest.get("inputs", {}).get("pathway_members", {})
    require(membership.get("sha256") == FORMAL_MEMBERSHIP_SHA256, "membership hash failed")
    preflight = manifest.get("split_preflight_gate", {})
    require(preflight.get("sha256") == FORMAL_PREFLIGHT_SHA256, "preflight hash failed")
    require(preflight.get("success_marker_sha256") == FORMAL_PREFLIGHT_SUCCESS_SHA256, "preflight success hash failed")
    split = manifest.get("split_integrity_audit", {})
    require(split.get("hard_pair_cross_fold_count") == 0 and split.get("all_five_folds_populated") is True, "pair split failed")
    folds_declared = manifest.get("folds")
    require(isinstance(folds_declared, Mapping) and set(folds_declared) == {str(i) for i in range(5)}, "five folds missing")
    folds = {str(i): audit_fold(root, i, folds_declared[str(i)]) for i in range(5)}
    optimizer_steps = sum(int(fold["optimizer_steps"]) for fold in folds.values())
    counts = manifest.get("counts", {})
    require(optimizer_steps == 17_567 == int(counts.get("optimizer_steps", -1)), "optimizer total failed")
    require(int(success.get("optimizer_steps", -1)) == optimizer_steps, "success optimizer total failed")
    if binding is not None:
        bound_candidates = binding.get("authorities", {}).get("exact_candidates", {})
        require(bound_candidates.get("sha256") == FORMAL_CANDIDATE_SHA256, "bound candidate hash failed")
        candidates = Path(str(bound_candidates.get("path", ""))).resolve()
    else:
        candidates = Path(str(authority.get("path", ""))).resolve()
    require(file_sha256(candidates) == FORMAL_CANDIDATE_SHA256, "candidate file hash failed")
    predictions = root / "evidence_private_predictions.parquet"
    prediction = audit_predictions(predictions, candidates)
    require(int(prediction["available_rows"]) == int(counts.get("available_predictions", -1)), "available count drift")
    require(int(prediction["unavailable_rows"]) == int(counts.get("unavailable_predictions", -1)), "unavailable count drift")
    artifact_paths = {
        "evidence_predictions": predictions,
        "event_lineage": root / "event_lineage.parquet",
        "physical_facts": root / "physical_interaction_facts.parquet",
        "rejected_mappings": root / "rejected_event_mappings.parquet",
        "split_integrity_audit": root / "PAIR_BLOCKED_SPLIT_AUDIT.json",
        "training_manifest": manifest_path,
        "training_success": success_path,
    }
    artifacts = {role: {"path": str(path), "sha256": file_sha256(path)} for role, path in artifact_paths.items()}
    physical = artifact_paths["physical_facts"]
    lineage = artifact_paths["event_lineage"]
    rejected = artifact_paths["rejected_mappings"]
    con = duckdb.connect(":memory:")
    try:
        physical_audit = con.execute(f"SELECT count(*), count(DISTINCT physical_fact_id), count_if(is_prediction IS DISTINCT FROM false) FROM read_parquet({sql_path(physical)})").fetchone()
        lineage_audit = con.execute(f"SELECT count(*), count_if(family_broadcast_used IS DISTINCT FROM false), count_if(is_model_prediction IS DISTINCT FROM false) FROM read_parquet({sql_path(lineage)})").fetchone()
        rejected_rows = int(con.execute(f"SELECT count(*) FROM read_parquet({sql_path(rejected)})").fetchone()[0])
    finally:
        con.close()
    require(tuple(map(int, physical_audit)) == (1_926_537, 1_926_537, 0), "physical fact audit failed")
    require(tuple(map(int, lineage_audit)) == (12_887_868, 0, 0), "event lineage audit failed")
    require(rejected_rows == 5_364_477, "rejected row audit failed")
    result: dict[str, Any] = {
        "format": POST_AUDIT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS",
        "semantic_policy": {
            "direct_target_evidence": True,
            "direct_target_scope": "LNCRNA_PARTNER_MAPPED_VIA_EXACT_MEMBER",
            "direct_exact_pathway_assertion": False,
            "unavailable_encoding": "null_with_reason",
            "confidence_only": True,
            "changes_primary_ranking": False,
            "affects_discovery": False,
            "affects_primary_ranking": False,
            "raw_prediction_direct_fusion_allowed": False,
            "canonical_candidate_adapter_required": True,
            "lncrna_identifier_normalization": "LNC:ENSG_TO_ENSG_FOR_TRAINING__RESTORE_LNC_PREFIX_FOR_FUSION",
            "pathway_identifier_normalization": "TRIM_AND_UPPERCASE_TO_CANONICAL_PATHWAY_FOR_FUSION",
        },
        "prediction_semantics": prediction,
        "folds": folds,
        "optimizer_steps_total": optimizer_steps,
        "training_manifest_sha256": manifest_sha,
        "training_success_sha256": file_sha256(success_path),
        "artifacts": artifacts,
        "physical_fact_audit": {"rows": int(physical_audit[0]), "unique_ids": int(physical_audit[1]), "prediction_rows": int(physical_audit[2])},
        "event_lineage_audit": {"rows": int(lineage_audit[0]), "family_broadcast_rows": int(lineage_audit[1]), "model_prediction_rows": int(lineage_audit[2])},
        "rejected_mapping_rows": rejected_rows,
        "release_ready": False,
    }
    if binding is not None and binding_path is not None:
        require(binding.get("optimizer_steps_total") == optimizer_steps, "R2 binding optimizer drift")
        bound_artifacts = binding.get("artifacts", {})
        role_map = {
            "evidence_predictions": "evidence_predictions",
            "event_lineage": "event_lineage",
            "physical_facts": "physical_facts",
            "rejected_mappings": "rejected_mappings",
            "split_integrity_audit": "split_integrity_audit",
        }
        for bound_role, audit_role in role_map.items():
            require(
                bound_artifacts.get(bound_role, {}).get("sha256")
                == artifacts[audit_role]["sha256"],
                f"R2 binding artifact drift: {bound_role}",
            )
        bound_folds = binding.get("checkpoints", {})
        require(set(bound_folds) == {str(index) for index in range(5)}, "R2 binding fold set failed")
        for fold_id, fold in folds.items():
            require(bound_folds[fold_id].get("sha256") == fold["checkpoint_sha256"], f"R2 fold {fold_id} checkpoint drift")
            require(bound_folds[fold_id].get("training_history_sha256") == fold["training_history_sha256"], f"R2 fold {fold_id} history drift")
            require(bound_folds[fold_id].get("final_parameter_sha256") == fold["final_parameter_sha256"], f"R2 fold {fold_id} state drift")
        result["r2_binding_sha256"] = file_sha256(binding_path)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        require(not output.exists() and not output.parent.exists(), "audit output reuse is forbidden")
        output.parent.mkdir(parents=True, exist_ok=False)
        output.write_text(serialized, encoding="utf-8")
        print(json.dumps({"status": "PASS", "output": str(output), "sha256": file_sha256(output)}, sort_keys=True))
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        raise SystemExit(f"AUDIT_FAIL: {exc}") from exc
