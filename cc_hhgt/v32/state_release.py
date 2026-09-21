"""Materialize the complete, contract-compatible V3.2 State release."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .full_model_contract import validate_module_lineage, validate_public_module_frame
from .input_lineage import artifact_sha256
from .state_training import HISTORICAL_STATE_IDS


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve_export(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    for ancestor in (manifest_path.parent, *manifest_path.parents):
        resolved = ancestor / candidate
        if resolved.is_file():
            return resolved.resolve()
    raise FileNotFoundError(value)


def core_lncrna_universe(core_manifest_path: str | Path) -> tuple[list[str], dict[str, Any]]:
    source = Path(core_manifest_path).resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    fold_sets: dict[int, set[str]] = {}
    for fold in range(5):
        declaration = manifest["folds"][str(fold)]["exports"]["lncRNA"]
        path = _resolve_export(source, str(declaration["path"]))
        if artifact_sha256(path) != str(declaration["sha256"]):
            raise ValueError(f"Fold {fold} lncRNA embedding SHA256 mismatch")
        values = set(pd.read_parquet(path, columns=["node_id"]).node_id.astype(str))
        fold_sets[fold] = values
    reference = fold_sets[0]
    if any(values != reference for values in fold_sets.values()):
        raise ValueError("V3.2 fold lncRNA universes differ")
    return sorted(reference), manifest


def complete_state_universe(
    release: pd.DataFrame,
    *,
    cancers: list[str],
    lncrnas: list[str],
    training_run_id: str,
) -> pd.DataFrame:
    keys = ["cancer_id", "lncrna_id", "state_id"]
    if release[keys].duplicated().any():
        raise ValueError("State source release has duplicate keys")
    base = pd.MultiIndex.from_product(
        [sorted(cancers), sorted(lncrnas), HISTORICAL_STATE_IDS], names=keys
    ).to_frame(index=False)
    output = base.merge(release, on=keys, how="left", validate="one_to_one", sort=False)
    output["availability"] = output.availability.fillna(False).astype(bool)
    missing = output.state_membership_probability.isna() & output.availability_reason.isna()
    output.loc[missing, "availability_reason"] = "LNCRNA_EXPRESSION_UNAVAILABLE_FOR_CANCER"
    output.loc[~output.availability, "state_membership_probability"] = np.nan
    output.loc[~output.availability, "state_effect"] = np.nan
    output.loc[~output.availability, "association_direction"] = pd.NA
    output["folds_available"] = pd.to_numeric(output.folds_available, errors="coerce").fillna(0).astype(int)
    output["folds_expected"] = pd.to_numeric(output.folds_expected, errors="coerce").fillna(5).astype(int)
    output["model_version"] = "V3.2"
    output["probability_source"] = output.probability_source.fillna("unavailable_null")
    output["effect_source"] = output.effect_source.fillna("unavailable_null")
    output["analysis_version"] = ANALYSIS_VERSION
    output["training_run_id"] = str(training_run_id)
    output["changes_primary_ranking"] = False
    return output.sort_values(keys, kind="stable").reset_index(drop=True)


def materialize_state_release(
    *,
    source_release_path: str | Path,
    source_lineage_path: str | Path,
    fold_manifest_path: str | Path,
    core_manifest_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    source_release_path = Path(source_release_path).resolve()
    source_lineage_path = Path(source_lineage_path).resolve()
    fold_manifest_path = Path(fold_manifest_path).resolve()
    core_manifest_path = Path(core_manifest_path).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_lineage = json.loads(source_lineage_path.read_text(encoding="utf-8"))
    release = pd.read_parquet(source_release_path)
    folds = pd.read_csv(fold_manifest_path, sep="\t")
    lncrnas, _ = core_lncrna_universe(core_manifest_path)
    cancers = sorted(folds.cancer_id.astype(str).unique())
    complete = complete_state_universe(
        release,
        cancers=cancers,
        lncrnas=lncrnas,
        training_run_id=str(source_lineage["training_run_id"]),
    )
    prediction_path = output / "state_typed_predictions.parquet"
    _atomic_parquet(complete, prediction_path)

    checkpoint_manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "state",
        "records": source_lineage["new_private_head_checkpoints"],
        "all_private_heads_fresh": all(
            row.get("source_checkpoint_sha256") is None
            for row in source_lineage["new_private_head_checkpoints"].values()
        ),
    }
    checkpoint_manifest_path = output / "CHECKPOINT_MANIFEST.json"
    _atomic_json(checkpoint_manifest, checkpoint_manifest_path)
    input_artifacts = []
    role_kind = {
        "tumor_state_long_measurement": "standardized_input",
        "v32_patient_fold_manifest": "split_manifest",
        "v32_core_embedding_manifest": "v32_core_checkpoint",
        "lncrna_expression": "standardized_input",
    }
    for artifact in source_lineage["input_artifacts"]:
        role = str(artifact["role"])
        input_artifacts.append(
            {
                "path": artifact["path"],
                "sha256": artifact["sha256"],
                "artifact_kind": role_kind[role],
            }
        )
    core_artifact = next(
        item for item in input_artifacts if item["artifact_kind"] == "v32_core_checkpoint"
    )
    before = [
        source_lineage["fold_lineage"][str(fold)]["core_parameters_before_sha256"]
        for fold in range(5)
    ]
    after = [
        source_lineage["fold_lineage"][str(fold)]["core_parameters_after_sha256"]
        for fold in range(5)
    ]
    module_lineage: dict[str, Any] = {
        "module_id": "state",
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": source_lineage["training_run_id"],
        "training_status": "SUCCESS",
        "initialization_policy": "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        "folds": 5,
        "seeds": source_lineage["seeds"],
        "code_sha256": artifact_sha256(Path(__file__).resolve().with_name("state_training.py")),
        "config_sha256": source_lineage["config_sha256"],
        "input_manifest_sha256": _canonical_hash(input_artifacts),
        "checkpoint_manifest_sha256": artifact_sha256(checkpoint_manifest_path),
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "private_head_trained_from_scratch": True,
        "core_parameters_frozen": True,
        "v32_core_checkpoint_sha256": core_artifact["sha256"],
        "core_parameters_before_sha256": _canonical_hash(before),
        "core_parameters_after_sha256": _canonical_hash(after),
        "input_artifacts": input_artifacts,
        "source_training_lineage_path": str(source_lineage_path),
        "source_training_lineage_sha256": artifact_sha256(source_lineage_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": artifact_sha256(prediction_path),
        "prediction_rows": int(len(complete)),
        "available_rows": int(complete.availability.sum()),
        "unavailable_rows": int((~complete.availability).sum()),
        "full_cancer_lncrna_state_universe": True,
    }
    validate_module_lineage("state", module_lineage)
    validate_public_module_frame("state", complete)
    lineage_path = output / "MODULE_LINEAGE.json"
    _atomic_json(module_lineage, lineage_path)
    return {
        "status": "SUCCESS",
        "prediction_path": str(prediction_path),
        "prediction_rows": int(len(complete)),
        "available_rows": int(complete.availability.sum()),
        "lineage_path": str(lineage_path),
        "contract_validation": "PASS",
    }


__all__ = ["complete_state_universe", "core_lncrna_universe", "materialize_state_release"]
