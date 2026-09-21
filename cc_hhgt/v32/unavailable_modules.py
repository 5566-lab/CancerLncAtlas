"""Honest V3.2 audit materialization for modules with no trainable head."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .full_model_contract import MODULE_CONTRACTS, validate_module_lineage, validate_public_module_frame
from .input_lineage import artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def audited_unavailable_lineage(
    *,
    module_id: str,
    training_run_id: str,
    failure_reason: str,
    core_manifest_path: str | Path,
    input_artifacts: list[dict[str, Any]],
    output_root: str | Path,
    prediction_path: str | Path | None,
    prediction_rows: int,
) -> dict[str, Any]:
    if module_id not in MODULE_CONTRACTS or module_id == "exact_pathway":
        raise ValueError(module_id)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    core_path = Path(core_manifest_path).resolve()
    core_sha = artifact_sha256(core_path)
    checkpoint_manifest_path = output / "CHECKPOINT_MANIFEST.json"
    _atomic_json(
        {
            "analysis_version": ANALYSIS_VERSION,
            "module_id": module_id,
            "records": [],
            "trained_folds": 0,
            "checkpoint_files": 0,
        },
        checkpoint_manifest_path,
    )
    artifacts = list(input_artifacts) + [
        {
            "path": str(core_path),
            "sha256": core_sha,
            "artifact_kind": "v32_core_checkpoint",
        }
    ]
    lineage: dict[str, Any] = {
        "module_id": module_id,
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": str(training_run_id),
        "training_status": "AUDITED_UNAVAILABLE",
        "initialization_policy": MODULE_CONTRACTS[module_id].core_policy,
        "folds": 5,
        "seeds": [20260825 + fold for fold in range(5)],
        "code_sha256": artifact_sha256(Path(__file__).resolve()),
        "config_sha256": _canonical_hash(
            {"module_id": module_id, "failure_reason": failure_reason, "policy": "fail_closed"}
        ),
        "input_manifest_sha256": _canonical_hash(artifacts),
        "checkpoint_manifest_sha256": artifact_sha256(checkpoint_manifest_path),
        "input_artifacts": artifacts,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "private_head_trained_from_scratch": False,
        "trained_folds": 0,
        "checkpoint_files": 0,
        "release_ready": False,
        "all_probabilities_null": True,
        "all_unavailable_rows_have_reason": True,
        "failure_reason": str(failure_reason),
        "core_parameters_frozen": True,
        "v32_core_checkpoint_sha256": core_sha,
        "core_parameters_before_sha256": core_sha,
        "core_parameters_after_sha256": core_sha,
        "prediction_rows": int(prediction_rows),
        "available_rows": 0,
    }
    if prediction_path is not None:
        path = Path(prediction_path).resolve()
        lineage["prediction_path"] = str(path)
        lineage["prediction_sha256"] = artifact_sha256(path)
    validate_module_lineage(module_id, lineage)
    _atomic_json(lineage, output / "MODULE_LINEAGE.json")
    return lineage


def materialize_interaction_audit(
    *,
    physical_facts_path: str | Path,
    core_manifest_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    facts_path = Path(physical_facts_path).resolve()
    facts = pd.read_parquet(facts_path, columns=["lncrna_id", "partner_id"])
    pairs = facts.dropna().astype(str).rename(columns={"partner_id": "protein_id"})
    pairs = pairs.loc[
        pairs.lncrna_id.str.strip().ne("") & pairs.protein_id.str.strip().ne("")
    ].drop_duplicates(["lncrna_id", "protein_id"])
    pairs["physical_interaction_probability"] = pd.NA
    pairs["availability"] = False
    pairs["failure_reason"] = "STRICT_CONNECTED_COMPONENT_COUNT_1_LT_5"
    pairs["analysis_version"] = ANALYSIS_VERSION
    pairs["training_run_id"] = "v32-interaction-audit-20260825"
    pairs["changes_primary_ranking"] = False
    output = Path(output_root).resolve()
    prediction_path = output / "interaction_typed_predictions.parquet"
    _atomic_parquet(pairs, prediction_path)
    validate_public_module_frame("interaction", pairs)
    lineage = audited_unavailable_lineage(
        module_id="interaction",
        training_run_id="v32-interaction-audit-20260825",
        failure_reason="STRICT_CONNECTED_COMPONENT_COUNT_1_LT_5",
        core_manifest_path=core_manifest_path,
        input_artifacts=[
            {
                "path": str(facts_path),
                "sha256": artifact_sha256(facts_path),
                "artifact_kind": "standardized_input",
            }
        ],
        output_root=output,
        prediction_path=prediction_path,
        prediction_rows=len(pairs),
    )
    return {
        "status": "AUDITED_UNAVAILABLE",
        "prediction_rows": int(len(pairs)),
        "prediction_path": str(prediction_path),
        "lineage_path": str(output / "MODULE_LINEAGE.json"),
        "contract_validation": "PASS",
        "prediction_sha256": lineage["prediction_sha256"],
    }


def materialize_drug_permission_audit(
    *,
    static_drug_target_path: str | Path,
    core_manifest_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    target_path = Path(static_drug_target_path).resolve()
    output = Path(output_root).resolve()
    lineage = audited_unavailable_lineage(
        module_id="drug",
        training_run_id="v32-drug-permission-audit-20260825",
        failure_reason="REMOTE_NATIVE_RAW_STAGING_AND_TRAINING_NOT_AUTHORIZED",
        core_manifest_path=core_manifest_path,
        input_artifacts=[
            {
                "path": str(target_path),
                "sha256": artifact_sha256(target_path),
                "artifact_kind": "annotation",
            }
        ],
        output_root=output,
        prediction_path=None,
        prediction_rows=0,
    )
    return {
        "status": "AUDITED_UNAVAILABLE",
        "prediction_rows": 0,
        "lineage_path": str(output / "MODULE_LINEAGE.json"),
        "failure_reason": lineage["failure_reason"],
        "contract_validation": "PASS",
    }


__all__ = [
    "audited_unavailable_lineage",
    "materialize_drug_permission_audit",
    "materialize_interaction_audit",
]
