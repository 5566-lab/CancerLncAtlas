#!/usr/bin/env python3
"""Build a contract-valid V3.2 single-cell typed-null staging lineage.

The R11 rescue contract can establish an honest availability boundary without
producing a trained single-cell head.  This materializer records that boundary
as a fresh V3.2 training-label authority, so the staging registry can load and
serve independent heads while keeping single-cell rows typed-unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.full_model_contract import validate_module_lineage
from cc_hhgt.v32.input_lineage import artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
MODULE_ID = "single_cell"
RUN_ID = "v32-single-cell-r11-typed-unavailable-staging-20260903"
POLICY = "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return hashlib.sha256(payload).hexdigest()
        raise RuntimeError(f"Refusing to overwrite existing artifact: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def build(*, rescue_status: Path, core_manifest: Path, output: Path) -> dict[str, Any]:
    rescue_status = rescue_status.resolve()
    core_manifest = core_manifest.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output must be absent or empty: {output}")
    status = json.loads(rescue_status.read_text(encoding="utf-8"))
    if status.get("format") != "CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_RUN_STATUS_V1":
        raise RuntimeError("R11 run-status format mismatch")
    if status.get("status") != "PASS_INPUT_CONTRACT_READY":
        raise RuntimeError("R11 rescue contract is not input-contract ready")
    eligible = int(status.get("formal_eligible_cancer_count", 0))
    unavailable = int(status.get("typed_unavailable_cancer_count", 0))
    if eligible != 23 or unavailable != 10:
        raise RuntimeError("R11 eligibility boundary drifted")
    if not core_manifest.is_file() or core_manifest.is_symlink():
        raise RuntimeError("Current V3.2 core manifest is missing or unsafe")

    output.mkdir(parents=True, exist_ok=True)
    core_sha = artifact_sha256(core_manifest)
    status_sha = artifact_sha256(rescue_status)

    # This is an explicit empty target authority, not a relabelled raw assay.
    # It records why no formal rows can be trained and keeps unavailable rows
    # out of the numeric loss while satisfying the V3.2 lineage contract.
    label_payload = {
        "format": "CANCERLNCATLAS_V32_SINGLE_CELL_TYPED_NULL_TRAINING_LABEL_V1",
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "status": "AUDITED_UNAVAILABLE",
        "formal_eligible_cancers": eligible,
        "typed_unavailable_cancers": unavailable,
        "label_rows": 0,
        "all_labels_null": True,
        "all_rows_have_failure_reason": True,
        "source_r11_run_status": str(rescue_status),
        "source_r11_run_status_sha256": status_sha,
        "training_policy": "NO_FORMAL_SINGLE_CELL_DATASET_AFTER_R11_DONOR_AUDIT",
        "historical_assets_used": False,
        "production_deployed": False,
        "release_ready": False,
    }
    label_path = output / "V32_SINGLE_CELL_TYPED_NULL_TRAINING_LABEL.json"
    label_sha = _write(label_path, _canonical(label_payload))

    checkpoint_payload = {
        "format": "CANCERLNCATLAS_V32_SINGLE_CELL_TYPED_NULL_CHECKPOINT_MANIFEST_V1",
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "folds": 5,
        "records": [],
        "trained_private_head_count": 0,
        "checkpoint_file_count": 0,
        "all_five_fold_private_heads_trained": False,
        "core_detached_and_frozen": True,
        "optimizer_steps": 0,
        "production_deployed": False,
        "release_ready": False,
    }
    checkpoint_path = output / "CHECKPOINT_MANIFEST.json"
    checkpoint_sha = _write(checkpoint_path, _canonical(checkpoint_payload))

    config_payload = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "training_run_id": RUN_ID,
        "training_status": "AUDITED_UNAVAILABLE",
        "target_policy": "R11_TYPED_UNAVAILABLE_NULL_WITH_REASON",
        "formal_boundary": "23_ELIGIBLE_PLUS_10_TYPED_UNAVAILABLE",
        "private_head_policy": "FRESH_INITIALIZATION_ONLY_ZERO_OPTIMIZER_STEPS",
        "core_policy": POLICY,
        "historical_assets_used": False,
        "changes_primary_ranking": False,
        "production_deployed": False,
        "release_ready": False,
    }
    config_path = output / "RUN_CONFIG.json"
    config_sha = _write(config_path, _canonical(config_payload))

    input_artifacts = [
        {
            "artifact_kind": "training_label",
            "path": str(label_path),
            "sha256": label_sha,
            "generation": "V3.2_R11",
            "source_role": "training_label",
            "outcome_derived": True,
            "fold_fitted": False,
            "use_role": "training_target",
        },
        {
            "artifact_kind": "standardized_input",
            "path": str(rescue_status),
            "sha256": status_sha,
            "generation": "V3.2_R11",
            "source_role": "rescue_contract",
            "outcome_derived": False,
            "fold_fitted": False,
            "use_role": "aux_input",
        },
        {
            "artifact_kind": "v32_core_checkpoint",
            "path": str(core_manifest),
            "sha256": core_sha,
            "generation": "V3.2",
            "source_role": "v32_core_checkpoint",
            "outcome_derived": True,
            "fold_fitted": True,
            "use_role": "aux_parent",
        },
    ]
    input_manifest_sha = hashlib.sha256(_canonical(input_artifacts)).hexdigest()
    lineage = {
        "module_id": MODULE_ID,
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": RUN_ID,
        "training_status": "AUDITED_UNAVAILABLE",
        "statistical_training_status": "UNAVAILABLE_NO_FORMAL_DATASET_AFTER_R11_AUDIT",
        "initialization_policy": POLICY,
        "folds": 5,
        "seeds": [20260903 + fold for fold in range(5)],
        "code_sha256": artifact_sha256(Path(__file__).resolve()),
        "config_sha256": config_sha,
        "input_manifest_sha256": input_manifest_sha,
        "checkpoint_manifest_sha256": checkpoint_sha,
        "input_artifacts": input_artifacts,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "old_family_support_used": False,
        "private_head_initialized_from_scratch": True,
        "private_head_optimizer_steps": 0,
        "private_head_trained_from_scratch": False,
        "all_five_fold_private_heads_trained": False,
        "trained_folds": 0,
        "checkpoint_files": 0,
        "release_ready": False,
        "all_probabilities_null": True,
        "all_unavailable_rows_have_reason": True,
        "core_parameters_frozen": True,
        "v32_core_checkpoint_sha256": core_sha,
        "core_parameters_before_sha256": core_sha,
        "core_parameters_after_sha256": core_sha,
        "split_unit": "dataset_or_donor",
        "donor_dataset_overlap_across_splits": False,
        "formal_cancer_boundary": "33_CANCER_AUTHORITY_23_ELIGIBLE_PLUS_10_TYPED_UNAVAILABLE",
        "formal_datasets": [],
        "prediction_rows": 3300000,
        "available_rows": 0,
        "null_rows": 3300000,
        "prediction_path": "typed-null-only; no numeric prediction artifact",
        "prediction_sha256": None,
        "rescue_contract_path": str(rescue_status),
        "rescue_contract_sha256": status_sha,
        "staging_never_promoted": True,
        "changes_primary_ranking": False,
    }
    validate_module_lineage(MODULE_ID, lineage)
    lineage_path = output / "MODULE_LINEAGE.json"
    lineage_sha = _write(lineage_path, _canonical(lineage))

    success = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_TYPED_NULL_STAGING_SUCCESS_V1",
        "status": "AUDITED_UNAVAILABLE",
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": RUN_ID,
        "lineage_path": str(lineage_path),
        "lineage_sha256": lineage_sha,
        "formal_eligible_cancers": eligible,
        "typed_unavailable_cancers": unavailable,
        "all_probabilities_null": True,
        "production_deployed": False,
        "release_ready": False,
    }
    _write(output / "SUCCESS.json", _canonical(success))
    return {
        "status": "PASS",
        "lineage_path": str(lineage_path),
        "lineage_sha256": lineage_sha,
        "label_path": str(label_path),
        "label_sha256": label_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r11-run-status", type=Path, required=True)
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(rescue_status=args.r11_run_status, core_manifest=args.core_manifest, output=args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
