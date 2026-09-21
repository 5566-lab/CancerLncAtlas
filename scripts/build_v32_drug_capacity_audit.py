#!/usr/bin/env python3
"""Build a contract-valid V3.2 Drug capacity/runtime unavailable lineage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--audit-spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo))
    from cc_hhgt.v32.full_model_contract import validate_module_lineage
    from cc_hhgt.v32.input_lineage import artifact_sha256

    spec_path = args.audit_spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite capacity audit: {output}")
    output.mkdir(parents=True, exist_ok=True)

    reason = str(spec["reason_code"])
    if not re.fullmatch(r"[A-Z0-9_]+", reason):
        raise ValueError("reason_code must be an uppercase machine-readable token")
    if spec.get("staging_started") is not False or spec.get("training_started") is not False:
        raise ValueError("Capacity audit cannot claim a started staging/training run")
    if int(spec["candidate_cancers"]) != 33 or spec.get("missing_cancers") != []:
        raise ValueError("Native-assayed preflight must preserve all 33 cancers")
    scope = str(spec["remote_scope_root"]).rstrip("/") + "/"
    input_artifacts = list(spec["input_artifacts"])
    for artifact in input_artifacts:
        path = str(artifact["path"])
        if not path.startswith(scope):
            raise ValueError(f"Remote input escapes the authorized scope: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(artifact["sha256"])):
            raise ValueError(f"Input lacks SHA256: {path}")

    core_path = (repo / spec["v32_core_manifest"]).resolve()
    core = json.loads(core_path.read_text(encoding="utf-8"))
    if core.get("all_embeddings_from_newly_trained_v32_core") is not True:
        raise ValueError("Core is not attested as newly trained V3.2")
    if core.get("historical_checkpoint_loaded") is not False:
        raise ValueError("Core loaded a historical checkpoint")
    if core.get("historical_prediction_loaded") is not False:
        raise ValueError("Core loaded historical predictions")
    core_sha = artifact_sha256(core_path)
    parameter_hashes = [str(core["folds"][str(fold)]["core_parameter_sha256"]) for fold in range(5)]
    core_parameter_composite = canonical_hash(parameter_hashes)
    input_artifacts.append(
        {
            "artifact_kind": "v32_core_checkpoint",
            "generation": "V3.2",
            "path": str(core_path),
            "sha256": core_sha,
        }
    )

    capacity = {
        key: spec[key]
        for key in (
            "reason_code",
            "native_assayed_drugs",
            "native_assayed_pathway_drug_rows",
            "native_assayed_candidate_rows",
            "candidate_cancers",
            "missing_cancers",
            "all_drugcentral_candidate_rows",
            "all_drugcentral_drugs",
            "public8_free_bytes_at_audit",
            "server_available_memory_bytes_at_audit",
            "minimum_numeric_working_set_bytes",
            "candidate_core_float32_bytes",
            "candidate_split_iterations",
            "estimated_total_disk_bytes_low",
            "estimated_total_disk_bytes_high",
            "staging_started",
            "training_started",
            "native_assay_intersection_uses_response_values",
            "preflight",
        )
    }
    atomic_json(output / "CAPACITY_AUDIT.json", capacity)
    atomic_json(output / "INPUT_MANIFEST.json", {"artifacts": input_artifacts})
    atomic_json(
        output / "CODE_MANIFEST.json",
        {
            "remote_code_root": (
                "./data/CancerLncAtlas/results/model/v32_full_multitask/"
                "drug_inputs/code_v32_drug_20260825"
            ),
            "artifacts": spec["remote_code_artifacts"],
        },
    )
    checkpoint_manifest = {
        "analysis_version": spec["analysis_version"],
        "module_id": "drug",
        "status": "AUDITED_UNAVAILABLE",
        "records": [],
        "checkpoint_files": 0,
        "trained_folds": 0,
        "reason_code": reason,
    }
    atomic_json(output / "CHECKPOINT_MANIFEST.json", checkpoint_manifest)
    audit_config = {
        "analysis_version": spec["analysis_version"],
        "training_run_id": spec["training_run_id"],
        "mode": "CAPACITY_AND_RUNTIME_FAIL_CLOSED_AUDIT",
        "reason_code": reason,
        "candidate_universe_materialized": False,
        "staging_started": False,
        "training_started": False,
        "old_results_permitted": False,
    }
    atomic_json(output / "AUDIT_CONFIG.json", audit_config)

    lineage = {
        "module_id": "drug",
        "analysis_version": spec["analysis_version"],
        "training_run_id": spec["training_run_id"],
        "training_status": "AUDITED_UNAVAILABLE",
        "statistical_training_status": "UNAVAILABLE_CAPACITY_RUNTIME_GATE",
        "initialization_policy": "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        "folds": 5,
        "seeds": [20260825 + fold * 101 for fold in range(5)],
        "code_sha256": artifact_sha256(output / "CODE_MANIFEST.json"),
        "config_sha256": artifact_sha256(output / "AUDIT_CONFIG.json"),
        "input_manifest_sha256": artifact_sha256(output / "INPUT_MANIFEST.json"),
        "checkpoint_manifest_sha256": artifact_sha256(output / "CHECKPOINT_MANIFEST.json"),
        "input_artifacts": input_artifacts,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "old_gdsc_prism_association_tables_used": False,
        "private_head_trained_from_scratch": False,
        "trained_folds": 0,
        "checkpoint_files": 0,
        "release_ready": False,
        "all_probabilities_null": True,
        "all_unavailable_rows_have_reason": True,
        "prediction_rows": 0,
        "available_rows": 0,
        "reason_applies_to_unmaterialized_candidate_universe": True,
        "failure_reason": reason,
        "reason_code": reason,
        "staging_started": False,
        "training_started": False,
        "candidate_rows_audited": int(spec["native_assayed_candidate_rows"]),
        "candidate_cancers_audited": int(spec["candidate_cancers"]),
        "preflight_path": spec["preflight"]["path"],
        "preflight_sha256": spec["preflight"]["sha256"],
        "preflight_bytes": int(spec["preflight"]["bytes"]),
        "core_parameters_frozen": True,
        "v32_core_checkpoint_sha256": core_sha,
        "core_parameters_before_sha256": core_parameter_composite,
        "core_parameters_after_sha256": core_parameter_composite,
        "supersedes_lineage_path": str((repo / spec["supersedes_lineage"]).resolve()),
        "supersession_reason": "REMOTE_PERMISSION_GATE_RESOLVED; CAPACITY_RUNTIME_GATE_IS_CURRENT",
        "public_prediction_path": None,
    }
    validate_module_lineage("drug", lineage)
    atomic_json(output / "MODULE_LINEAGE.json", lineage)
    validation = {
        "status": "PASS",
        "validator": "cc_hhgt.v32.full_model_contract.validate_module_lineage",
        "module_id": "drug",
        "training_status": "AUDITED_UNAVAILABLE",
        "lineage_path": str(output / "MODULE_LINEAGE.json"),
        "lineage_sha256": artifact_sha256(output / "MODULE_LINEAGE.json"),
        "reason_code": reason,
    }
    atomic_json(output / "FULL_MODEL_CONTRACT_VALIDATION.json", validation)
    success = {
        "status": "AUDITED_UNAVAILABLE",
        "analysis_version": spec["analysis_version"],
        "module_id": "drug",
        "training_run_id": spec["training_run_id"],
        "release_ready": False,
        "staging_started": False,
        "training_started": False,
        "prediction_rows": 0,
        "checkpoint_files": 0,
        "trained_folds": 0,
        "all_probabilities_null": True,
        "reason_code": reason,
        "lineage_path": str(output / "MODULE_LINEAGE.json"),
        "lineage_sha256": artifact_sha256(output / "MODULE_LINEAGE.json"),
        "contract_validation_path": str(output / "FULL_MODEL_CONTRACT_VALIDATION.json"),
        "contract_validation_status": "PASS",
    }
    atomic_json(output / "SUCCESS.json", success)
    print(json.dumps(success, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
