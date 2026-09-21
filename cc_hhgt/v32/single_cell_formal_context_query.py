"""Hash-bound formal single-cell context release and read-only queries.

This module deliberately keeps three different single-cell quantities separate:

* lncRNA-by-cell-type expression/detection facts;
* protein-only donor-pseudobulk exact-pathway rank-mean activity; and
* the learned dataset-by-cell-type-by-lncRNA-by-exact-pathway replication
  probability from the fresh V3.2 private heads.

None of these values is the immutable exact-pathway primary score.  The formal
multimodal checkpoint currently learned a zero coefficient for the single-cell
expert, so the release also records that these values do not currently alter
the secondary score.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import duckdb
import numpy as np
import pandas as pd

ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_FORMAL_CONTEXT_BINDING_V1"
MAX_QUERY_LIMIT = 500
MAX_QUERY_OFFSET = 10_000_000

FORMAL_CANCERS = frozenset(
    {
        "ACC", "CHOL", "DLBC", "ESCA", "GBM", "HNSC", "KIRC", "LAML",
        "LGG", "LUSC", "MESO", "PCPG", "READ", "SARC", "SKCM", "THYM",
        "UCEC",
    }
)
ALL_CANCERS = frozenset(
    {
        "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
        "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
        "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
        "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
        "UVM",
    }
)

IMMUNE_CELL_TYPES = frozenset(
    {
        "B_cell", "T_cell", "NK_cell", "Myeloid", "Plasma_cell",
        "Hematopoietic_other", "Hematopoietic_progenitor", "Lymphoid",
    }
)
STROMAL_CELL_TYPES = frozenset({"Endothelial", "Fibroblast_stromal"})
COMPARTMENTS = frozenset(
    {
        "IMMUNE", "STROMAL", "MALIGNANT", "MALIGNANT_CANDIDATE",
        "OTHER_UNRESOLVED", "UNAVAILABLE",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:\-]+$")

EXPECTED_COUNTS = {
    "typed_predictions": {"rows": 7_814_014, "available_rows": 6_214_014},
    "lncrna_celltype": {"rows": 229_104, "available_rows": 194_082},
    "pathway_activity": {"rows": 4_354_871, "available_rows": 4_320_711},
    "exact_association": {"rows": 2_554_541, "available_rows": 954_541},
}


class SingleCellFormalContextError(RuntimeError):
    """Base formal-context error."""


class SingleCellFormalContextAssetError(SingleCellFormalContextError):
    """Raised when a source, binding, or semantic invariant is invalid."""


class SingleCellFormalContextInputError(SingleCellFormalContextError):
    """Raised when query filters are invalid."""


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise SingleCellFormalContextAssetError(f"{label} is missing or unsafe: {source}")
    return source


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = _safe_file(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleCellFormalContextAssetError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise SingleCellFormalContextAssetError(f"{label} must be a JSON object")
    return value


def _sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def cell_type_compartment(cell_type: Any) -> str:
    value = str(cell_type or "").split("::", 1)[0]
    if value in IMMUNE_CELL_TYPES:
        return "IMMUNE"
    if value in STROMAL_CELL_TYPES:
        return "STROMAL"
    if value == "Malignant":
        return "MALIGNANT"
    if value == "Malignant_candidate":
        return "MALIGNANT_CANDIDATE"
    if value == "UNAVAILABLE" or not value:
        return "UNAVAILABLE"
    return "OTHER_UNRESOLVED"


def _compartment_sql(column: str) -> str:
    immune = ", ".join("'" + value + "'" for value in sorted(IMMUNE_CELL_TYPES))
    stromal = ", ".join("'" + value + "'" for value in sorted(STROMAL_CELL_TYPES))
    return (
        f"CASE WHEN split_part(coalesce({column}, ''), '::', 1) IN ({immune}) "
        "THEN 'IMMUNE' "
        f"WHEN split_part(coalesce({column}, ''), '::', 1) IN ({stromal}) "
        "THEN 'STROMAL' "
        f"WHEN split_part(coalesce({column}, ''), '::', 1) = 'Malignant' "
        "THEN 'MALIGNANT' "
        f"WHEN split_part(coalesce({column}, ''), '::', 1) = 'Malignant_candidate' "
        "THEN 'MALIGNANT_CANDIDATE' "
        f"WHEN split_part(coalesce({column}, ''), '::', 1) IN ('', 'UNAVAILABLE') "
        "THEN 'UNAVAILABLE' ELSE 'OTHER_UNRESOLVED' END"
    )


def _canonical_cancer(value: Any) -> str:
    cancer = str(value or "").strip().upper()
    if cancer not in ALL_CANCERS:
        raise SingleCellFormalContextInputError("cancer_id is not in the 33-cancer authority")
    return cancer


def _optional_identifier(value: Any | None, label: str) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token or len(token) > 256 or not _IDENTIFIER.fullmatch(token):
        raise SingleCellFormalContextInputError(f"{label} is invalid")
    return token


def _optional_cell_type(value: Any | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token or len(token) > 256 or any(ord(char) < 32 for char in token):
        raise SingleCellFormalContextInputError("cell_type is invalid")
    return token


def _availability(value: Any) -> str:
    token = str(value or "ALL").strip().upper()
    if token not in {"ALL", "AVAILABLE", "UNAVAILABLE"}:
        raise SingleCellFormalContextInputError(
            "availability must be ALL, AVAILABLE, or UNAVAILABLE"
        )
    return token


def _optional_compartment(value: Any | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip().upper()
    if token not in COMPARTMENTS:
        raise SingleCellFormalContextInputError(
            f"compartment must be one of {sorted(COMPARTMENTS)}"
        )
    return token


def _bounds(limit: Any, offset: Any) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise SingleCellFormalContextInputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise SingleCellFormalContextInputError(f"offset must be 0..{MAX_QUERY_OFFSET}")
    return int(limit), int(offset)


def _artifact(path: Path, **extra: Any) -> dict[str, Any]:
    result = {
        "path": str(path),
        "sha256": artifact_sha256(path),
        "bytes": int(path.stat().st_size),
    }
    result.update(extra)
    return result


def _parquet_metrics(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    relations = {name: f"read_parquet({_sql_path(path)})" for name, path in paths.items()}
    connection = duckdb.connect(database=":memory:")
    try:
        typed = connection.execute(
            f"""
            SELECT count(*), count_if(single_cell_available),
                   count(DISTINCT (dataset_id, cell_type, lncrna_id, pathway_id)),
                   count_if(single_cell_available AND
                     (single_cell_replication_probability IS NULL OR
                      NOT isfinite(single_cell_replication_probability) OR
                      single_cell_replication_probability < 0 OR
                      single_cell_replication_probability > 1)),
                   count_if(NOT single_cell_available AND
                     single_cell_replication_probability IS NOT NULL),
                   count_if(changes_primary_ranking),
                   count_if(analysis_version <> ? OR
                     target_level <> 'dataset_x_celltype_x_lncrna_x_exact_pathway')
            FROM {relations['typed_predictions']}
            """,
            [ANALYSIS_VERSION],
        ).fetchone()
        celltype = connection.execute(
            f"""
            SELECT count(*), count_if(lnc_celltype_available),
                   count(DISTINCT CASE WHEN lnc_celltype_available THEN cancer_id END),
                   count(DISTINCT CASE WHEN lnc_celltype_available THEN lncrna_id END),
                   count(DISTINCT CASE WHEN lnc_celltype_available THEN cell_type END),
                   count_if(NOT source_is_nonpredictive),
                   count_if(lnc_celltype_available AND
                     (lnc_detection_rate IS NULL OR lnc_mean_log_expression IS NULL OR
                      lnc_specificity_tau IS NULL OR lnc_n_cells IS NULL)),
                   count_if(NOT lnc_celltype_available AND
                     (lnc_detection_rate IS NOT NULL OR lnc_mean_log_expression IS NOT NULL OR
                      lnc_specificity_tau IS NOT NULL OR lnc_n_cells IS NOT NULL))
            FROM {relations['lncrna_celltype']}
            """
        ).fetchone()
        activity = connection.execute(
            f"""
            SELECT count(*), count_if(activity_available),
                   count(DISTINCT CASE WHEN activity_available THEN cancer_id END),
                   count(DISTINCT CASE WHEN activity_available THEN donor_id END),
                   count(DISTINCT CASE WHEN activity_available THEN cell_type END),
                   count(DISTINCT CASE WHEN activity_available THEN pathway_id END),
                   count_if(NOT source_is_nonpredictive),
                   count_if(activity_available AND
                     (activity_value IS NULL OR NOT isfinite(activity_value))),
                   count_if(activity_available AND activity_kind <> 'activity')
            FROM {relations['pathway_activity']}
            """
        ).fetchone()
        methods = connection.execute(
            f"""
            SELECT DISTINCT activity_kind, activity_source_tier
            FROM {relations['pathway_activity']}
            WHERE activity_available
            ORDER BY 1, 2
            """
        ).fetchall()
        exact = connection.execute(
            f"""
            SELECT count(*), count_if(single_cell_available),
                   count(DISTINCT CASE WHEN single_cell_available THEN cancer_id END),
                   count(DISTINCT (cancer_id, lncrna_id, pathway_id)),
                   count_if(NOT exact_pathway_only),
                   count_if(single_cell_available AND
                     (single_cell_replication_probability IS NULL OR
                      NOT isfinite(single_cell_replication_probability) OR
                      single_cell_replication_probability < 0 OR
                      single_cell_replication_probability > 1))
            FROM {relations['exact_association']}
            """
        ).fetchone()
        formal_cancers = {
            str(row[0])
            for row in connection.execute(
                f"SELECT DISTINCT cancer_id FROM {relations['lncrna_celltype']} "
                "WHERE lnc_celltype_available"
            ).fetchall()
        }
        labels = connection.execute(
            f"""
            SELECT cell_type, count(*), count(DISTINCT cancer_id)
            FROM {relations['lncrna_celltype']}
            WHERE lnc_celltype_available
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
        compartments = connection.execute(
            f"""
            SELECT {_compartment_sql('cell_type')} AS compartment,
                   count(*), count(DISTINCT cancer_id)
            FROM {relations['lncrna_celltype']}
            WHERE lnc_celltype_available
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
    finally:
        connection.close()
    if tuple(map(int, typed[0:2])) != (7_814_014, 6_214_014) or tuple(map(int, typed[2:])) != (
        7_814_014, 0, 0, 0, 0
    ):
        raise SingleCellFormalContextAssetError(f"Typed prediction invariants failed: {typed}")
    if tuple(map(int, celltype)) != (229_104, 194_082, 17, 5_424, 16, 0, 0, 0):
        raise SingleCellFormalContextAssetError(f"lncRNA-cell-type invariants failed: {celltype}")
    if tuple(map(int, activity)) != (4_354_871, 4_320_711, 17, 274, 16, 2_135, 0, 0, 0):
        raise SingleCellFormalContextAssetError(f"Pathway-activity invariants failed: {activity}")
    if methods != [("activity", "approved"), ("activity", "primary_raw_count")]:
        raise SingleCellFormalContextAssetError(f"Unexpected activity source kinds: {methods}")
    if tuple(map(int, exact)) != (2_554_541, 954_541, 17, 2_554_541, 0, 0):
        raise SingleCellFormalContextAssetError(f"Exact association invariants failed: {exact}")
    if formal_cancers != set(FORMAL_CANCERS):
        raise SingleCellFormalContextAssetError(
            f"Formal single-cell cancer set mismatch: {sorted(formal_cancers)}"
        )
    return {
        "typed_predictions": {
            **EXPECTED_COUNTS["typed_predictions"],
            "target_level": "dataset_x_celltype_x_lncrna_x_exact_pathway",
            "learned_association": True,
        },
        "lncrna_celltype": {
            **EXPECTED_COUNTS["lncrna_celltype"],
            "cancers_available": 17,
            "distinct_lncrna_available": 5_424,
            "cell_type_labels_available": 16,
            "learned_association": False,
        },
        "pathway_activity": {
            **EXPECTED_COUNTS["pathway_activity"],
            "cancers_available": 17,
            "donors_available": 274,
            "cell_type_labels_available": 16,
            "exact_pathways_available": 2_135,
            "method": "V3.2_PROTEIN_ONLY_WITHIN_PSEUDOBULK_RANK_MEAN_V1",
            "protein_only": True,
            "learned_association": False,
        },
        "exact_association": {
            **EXPECTED_COUNTS["exact_association"],
            "cancers_available": 17,
            "target_level": "cancer_x_lncrna_x_exact_pathway",
            "aggregation": "MEAN_ACROSS_AVAILABLE_DATASET_CELLTYPE_TARGETS",
            "learned_association": True,
        },
        "cell_type_labels": [
            {
                "cell_type": str(label),
                "compartment": cell_type_compartment(label),
                "rows": int(rows),
                "cancers": int(cancers),
            }
            for label, rows, cancers in labels
        ],
        "compartments": [
            {"compartment": str(name), "rows": int(rows), "cancers": int(cancers)}
            for name, rows, cancers in compartments
        ],
    }


def build_single_cell_formal_context_binding(
    *,
    fresh_root: str | Path,
    dataset_manifest_path: str | Path,
    detection_path: str | Path,
    remote_observation_path: str | Path,
    fusion_adapter_binding_path: str | Path,
    multimodal_checkpoint_path: str | Path,
    hnsc_ucell_binding_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create a binding after full source/hash/semantic validation."""

    root = Path(fresh_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise SingleCellFormalContextAssetError(f"Fresh single-cell root is invalid: {root}")
    source_paths = {
        "success": _safe_file(root / "SUCCESS.json", "fresh SUCCESS"),
        "lineage": _safe_file(root / "LINEAGE.json", "fresh LINEAGE"),
        "input_lineage_audit": _safe_file(
            root / "INPUT_LINEAGE_AUDIT.json", "fresh input lineage audit"
        ),
        "contract_validation": _safe_file(
            root / "FULL_MODEL_CONTRACT_VALIDATION.json", "fresh contract validation"
        ),
        "checkpoint_manifest": _safe_file(
            root / "CHECKPOINT_MANIFEST.json", "fresh checkpoint manifest"
        ),
        "typed_predictions": _safe_file(
            root / "single_cell_typed_predictions.parquet", "typed predictions"
        ),
        "lncrna_celltype": _safe_file(
            root / "lnc_celltype_summary.parquet", "lncRNA-cell-type table"
        ),
        "pathway_activity": _safe_file(root / "activity.parquet", "pathway activity"),
        "exact_association": _safe_file(
            root / "lnc_exact_pathway.parquet", "exact association"
        ),
        "dataset_manifest": _safe_file(dataset_manifest_path, "dataset manifest"),
        "detection_33c": _safe_file(detection_path, "33-cancer detection audit"),
        "remote_observation": _safe_file(
            remote_observation_path, "remote metadata observation"
        ),
        "fusion_adapter_binding": _safe_file(
            fusion_adapter_binding_path, "single-cell exact adapter binding"
        ),
        "multimodal_checkpoint": _safe_file(
            multimodal_checkpoint_path, "multimodal checkpoint"
        ),
        "hnsc_ucell_binding": _safe_file(hnsc_ucell_binding_path, "HNSC UCell binding"),
    }
    success = _read_json(source_paths["success"], "fresh SUCCESS")
    lineage = _read_json(source_paths["lineage"], "fresh LINEAGE")
    if (
        success.get("analysis_version") != ANALYSIS_VERSION
        or success.get("status") != "SUCCESS"
        or success.get("trained_folds") != 5
        or success.get("formal_datasets") != 17
        or success.get("available_rows") != 6_214_014
        or success.get("exact_pathway_only") is not True
        or lineage.get("training_status") != "SUCCESS"
        or lineage.get("private_head_trained_from_scratch") is not True
        or lineage.get("old_checkpoint_loaded") is not False
        or lineage.get("old_predictions_used_as_features") is not False
        or lineage.get("old_family_support_used") is not False
    ):
        raise SingleCellFormalContextAssetError("Fresh single-cell lineage is not formal V3.2")
    if artifact_sha256(source_paths["lineage"]) != success.get("lineage_sha256"):
        raise SingleCellFormalContextAssetError("Fresh SUCCESS does not bind LINEAGE")
    for role, lineage_key in {
        "typed_predictions": "prediction_sha256",
        "lncrna_celltype": "lnc_celltype_sha256",
        "pathway_activity": "activity_sha256",
        "exact_association": "lnc_exact_pathway_sha256",
    }.items():
        if artifact_sha256(source_paths[role]) != lineage.get(lineage_key):
            raise SingleCellFormalContextAssetError(f"LINEAGE hash mismatch for {role}")

    metrics = _parquet_metrics(
        {key: source_paths[key] for key in EXPECTED_COUNTS}
    )
    detection = _read_json(source_paths["detection_33c"], "33-cancer detection audit")
    remote = _read_json(source_paths["remote_observation"], "remote metadata observation")
    if (
        detection.get("analysis_version") != ANALYSIS_VERSION
        or detection.get("counts", {}).get("cancers") != 33
        or detection.get("counts", {}).get("formal_eligible_cancers") != 17
        or detection.get("counts", {}).get("detected_union") != 15_879
        or remote.get("analysis_version") != ANALYSIS_VERSION
        or remote.get("current_processed_metadata", {}).get("metadata_files") != 23
        or remote.get("current_processed_metadata", {}).get("formal_eligible_metadata_files") != 17
    ):
        raise SingleCellFormalContextAssetError("Raw-H5/metadata coverage authorities are invalid")

    ucell = _read_json(source_paths["hnsc_ucell_binding"], "HNSC UCell binding")
    ucell_counts = ucell.get("validated_counts", {})
    if (
        ucell.get("status") != "SUCCESS_HNSC_UCELL_PILOT_HASH_BOUND"
        or ucell.get("cancer_id") != "HNSC"
        or ucell_counts.get("cells") != 5_902
        or ucell_counts.get("pathways_total") != 2_135
        or ucell_counts.get("pathways_available") != 2_120
        or ucell_counts.get("cell_level_numeric_rows") != 12_512_240
        or ucell_counts.get("cell_level_coverage_rows") != 12_600_770
    ):
        raise SingleCellFormalContextAssetError("HNSC UCell binding invariants failed")

    checkpoint = _read_json(source_paths["multimodal_checkpoint"], "multimodal checkpoint")
    fusion_weights: dict[str, Any] = {}
    for endpoint, expected_experts in {
        "discovery": ["genomic", "single_cell"],
        "confidence": ["genomic", "single_cell", "evidence_transformer"],
    }.items():
        record = checkpoint.get(endpoint, {})
        if record.get("expert_ids") != expected_experts or record.get(
            "primary_probability_offset_fixed"
        ) is not True:
            raise SingleCellFormalContextAssetError(
                f"Multimodal checkpoint has invalid {endpoint} contract"
            )
        weights = record.get("weights")
        sc_index = expected_experts.index("single_cell")
        if not isinstance(weights, list) or float(weights[sc_index]) != 0.0:
            raise SingleCellFormalContextAssetError(
                f"Single-cell {endpoint} weight is not the audited learned zero"
            )
        fusion_weights[endpoint] = {
            "expert_ids": expected_experts,
            "weights": [float(value) for value in weights],
            "single_cell_weight": 0.0,
        }

    detection_by_cancer = {
        str(row["cancer_id"]): row for row in detection.get("entries", [])
        if isinstance(row, dict)
    }
    metadata_by_cancer = {
        str(row["cancer_id"]): row
        for row in remote.get("current_processed_metadata", {}).get("files", [])
        if isinstance(row, dict)
    }
    expansion = []
    for cancer in sorted(FORMAL_CANCERS):
        raw = detection_by_cancer.get(cancer, {})
        metadata = metadata_by_cancer.get(cancer, {})
        if not raw.get("raw_h5_sha256") or not metadata.get("sha256"):
            raise SingleCellFormalContextAssetError(
                f"Formal UCell expansion source is incomplete for {cancer}"
            )
        expansion.append(
            {
                "cancer_id": cancer,
                "raw_h5_path": raw.get("raw_h5_path"),
                "raw_h5_sha256": raw.get("raw_h5_sha256"),
                "cells_scanned": int(raw.get("cells", 0)),
                "cell_metadata_path": (
                    f"./data/CancerLncAtlas/processed/sc_tool_input/"
                    f"{cancer}/cell_metadata.parquet"
                ),
                "cell_metadata_sha256": metadata.get("sha256"),
                "cell_level_ucell_status": (
                    "FRESH_HASH_BOUND_COMPLETE"
                    if cancer == "HNSC"
                    else "PENDING_PER_CANCER_PREFLIGHT_AND_REMOTE_COMPUTE"
                ),
            }
        )

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    binding_path = destination / "SINGLE_CELL_FORMAL_CONTEXT_BINDING.json"
    success_path = destination / "SUCCESS.json"
    if binding_path.exists() or success_path.exists():
        raise SingleCellFormalContextAssetError(f"Refusing to overwrite {destination}")
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL_CONTEXT_READY_WITH_CELL_LEVEL_UCELL_PARTIAL",
        "training_run_id": success.get("training_run_id"),
        "formal_context_cancers": sorted(FORMAL_CANCERS),
        "formal_context_cancer_count": 17,
        "raw_h5_cancer_count": 33,
        "raw_h5_detected_lncrna_union": 15_879,
        "metrics": metrics,
        "semantics": {
            "lncrna_celltype": "NONPREDICTIVE_EXPRESSION_DETECTION_FACT",
            "pathway_activity": "NONPREDICTIVE_PROTEIN_ONLY_DONOR_PSEUDOBULK_RANK_MEAN",
            "celltype_prediction": "LEARNED_REPLICATION_PROBABILITY",
            "training_target": "CLIP_ABS_SPEARMAN_RHO_X_ONE_MINUS_FDR",
            "celltype_id_has_dedicated_embedding": False,
            "pathway_activity_training_feature_aggregation": "DATASET_X_PATHWAY_MEAN",
            "exact_association_aggregation": "CANCER_X_LNCRNA_X_EXACT_PATHWAY_MEAN",
            "activity_is_not_learned_association": True,
        },
        "model_gaps": {
            "celltype_specific_pathway_activity_present_in_source": True,
            "celltype_specific_pathway_activity_consumed_by_current_private_head": False,
            "current_activity_feature_aggregation": "DATASET_X_PATHWAY_MEAN",
            "celltype_specific_pathway_feature_requires_retraining": True,
            "current_celltype_specific_prediction_available": True,
            "current_celltype_specific_prediction_basis": (
                "CELLTYPE_MATCHED_LNCRNA_EXPRESSION_FEATURES_PLUS_"
                "DATASET_LEVEL_PATHWAY_ACTIVITY"
            ),
        },
        "fusion": {
            "weights": fusion_weights,
            "single_cell_currently_changes_secondary_score": False,
            "changes_exact_primary_score": False,
            "changes_exact_primary_ranking": False,
            "zero_weight_is_a_learned_result_not_missing_data": True,
        },
        "cell_level_ucell": {
            "status": "PARTIAL_1_OF_17_FORMAL_CANCERS",
            "covered_cancers": ["HNSC"],
            "formal_cancers_total": 17,
            "cells_hnsc": 5_902,
            "pathways_total": 2_135,
            "pathways_available_hnsc": 2_120,
            "numeric_rows_hnsc": 12_512_240,
            "typed_unavailable_rows_hnsc": 88_530,
            "other_formal_cancers_require_remote_compute": 16,
        },
        "ucell_expansion_inputs": expansion,
        "remaining_gaps": [
            "CELLTYPE_SPECIFIC_PATHWAY_ACTIVITY_REQUIRES_PRIVATE_HEAD_V2_RETRAINING",
            "CELL_LEVEL_UCELL_MISSING_FOR_16_OF_17_FORMAL_CANCERS",
            "PSEUDOTIME_HAS_NO_EXPLICIT_ROOT_OR_ORDERED_SOURCE_STATE",
            "FRESH_HASH_BOUND_SINGLE_CELL_FIGURES_MISSING",
            "SINGLE_CELL_FUSION_WEIGHT_LEARNED_AS_ZERO",
        ],
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "exact_primary_modified": False,
        "release_ready": False,
        "production_deployed": False,
        "artifacts": {
            role: _artifact(
                path,
                **(
                    metrics.get(role, {})
                    if role in metrics
                    else {}
                ),
            )
            for role, path in source_paths.items()
        },
    }
    binding_path.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    binding_sha = artifact_sha256(binding_path)
    success_payload = {
        "status": binding["status"],
        "binding": binding_path.name,
        "binding_sha256": binding_sha,
        "formal_context_ready": True,
        "single_cell_module_complete": False,
        "release_ready": False,
        "production_deployed": False,
    }
    success_path.write_text(
        json.dumps(success_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "binding_path": str(binding_path),
        "binding_sha256": binding_sha,
        "success_path": str(success_path),
        "binding": binding,
    }


class SingleCellFormalContextQuery:
    """Validated read-only query surface for the formal 17-cancer context."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
    ) -> None:
        source = _safe_file(binding_path, "single-cell formal-context binding")
        expected = str(expected_binding_sha256 or "").strip().lower()
        if not _SHA256.fullmatch(expected):
            raise SingleCellFormalContextAssetError("expected binding SHA256 is required")
        observed = artifact_sha256(source)
        if observed != expected:
            raise SingleCellFormalContextAssetError(
                f"Formal-context binding SHA mismatch: {observed} != {expected}"
            )
        binding = _read_json(source, "single-cell formal-context binding")
        required = {
            "format": BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "FORMAL_CONTEXT_READY_WITH_CELL_LEVEL_UCELL_PARTIAL",
            "formal_context_cancer_count": 17,
            "raw_h5_cancer_count": 33,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "exact_primary_modified": False,
            "release_ready": False,
            "production_deployed": False,
        }
        for key, value in required.items():
            if binding.get(key) != value:
                raise SingleCellFormalContextAssetError(
                    f"Formal-context binding has invalid {key}: {binding.get(key)!r}"
                )
        if set(binding.get("formal_context_cancers", [])) != set(FORMAL_CANCERS):
            raise SingleCellFormalContextAssetError("Formal-context cancer set is invalid")
        artifacts = binding.get("artifacts")
        if not isinstance(artifacts, dict):
            raise SingleCellFormalContextAssetError("Formal-context binding lacks artifacts")
        paths: dict[str, Path] = {}
        for role in (
            "typed_predictions", "lncrna_celltype", "pathway_activity", "exact_association"
        ):
            declaration = artifacts.get(role)
            if not isinstance(declaration, dict):
                raise SingleCellFormalContextAssetError(f"Missing artifact declaration: {role}")
            path = _safe_file(declaration.get("path", ""), role)
            digest = str(declaration.get("sha256", "")).lower()
            if not _SHA256.fullmatch(digest) or artifact_sha256(path) != digest:
                raise SingleCellFormalContextAssetError(f"Artifact SHA drift: {role}")
            expected_counts = EXPECTED_COUNTS[role]
            if any(declaration.get(key) != value for key, value in expected_counts.items()):
                raise SingleCellFormalContextAssetError(f"Artifact count drift: {role}")
            paths[role] = path
        success = _read_json(source.parent / "SUCCESS.json", "formal-context SUCCESS")
        if (
            success.get("binding") != source.name
            or success.get("binding_sha256") != observed
            or success.get("formal_context_ready") is not True
            or success.get("single_cell_module_complete") is not False
        ):
            raise SingleCellFormalContextAssetError("Formal-context SUCCESS is stale")
        fusion = binding.get("fusion", {})
        if (
            fusion.get("single_cell_currently_changes_secondary_score") is not False
            or fusion.get("changes_exact_primary_score") is not False
            or fusion.get("weights", {}).get("discovery", {}).get("single_cell_weight") != 0.0
            or fusion.get("weights", {}).get("confidence", {}).get("single_cell_weight") != 0.0
        ):
            raise SingleCellFormalContextAssetError("Formal-context score semantics are invalid")
        self.binding = binding
        self.binding_path = source
        self.binding_sha256 = observed
        self.paths = paths

    @staticmethod
    def _connect():
        return duckdb.connect(database=":memory:")

    def capability_status(self) -> dict[str, Any]:
        return {
            "module": "single_cell_formal_context",
            "status": self.binding["status"],
            "formal_context_cancers": self.binding["formal_context_cancers"],
            "formal_context_cancer_count": 17,
            "raw_h5_cancer_count": 33,
            "metrics": self.binding["metrics"],
            "cell_level_ucell": self.binding["cell_level_ucell"],
            "remaining_gaps": self.binding["remaining_gaps"],
            "single_cell_currently_changes_secondary_score": False,
            "changes_exact_primary_score": False,
            "release_ready": False,
            "production_deployed": False,
            "binding_sha256": self.binding_sha256,
        }

    def _query(
        self,
        *,
        role: str,
        select: str,
        cell_column: str,
        availability_column: str,
        cancer_id: Any,
        lncrna_id: Any | None = None,
        pathway_id: Any | None = None,
        donor_id: Any | None = None,
        cell_type: Any | None = None,
        compartment: Any | None = None,
        availability: Any = "ALL",
        limit: Any = 100,
        offset: Any = 0,
        order_by: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cancer = _canonical_cancer(cancer_id)
        lnc = _optional_identifier(lncrna_id, "lncrna_id")
        pathway = _optional_identifier(pathway_id, "pathway_id")
        donor = _optional_identifier(donor_id, "donor_id")
        cell = _optional_cell_type(cell_type)
        compartment_value = _optional_compartment(compartment)
        availability_value = _availability(availability)
        limit_value, offset_value = _bounds(limit, offset)
        clauses = ["cancer_id = ?"]
        parameters: list[Any] = [cancer]
        for column, value in (
            ("lncrna_id", lnc), ("pathway_id", pathway), ("donor_id", donor)
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if cell is not None:
            clauses.append(f"({cell_column} = ? OR split_part({cell_column}, '::', 1) = ?)")
            parameters.extend([cell, cell])
        compartment_expression = _compartment_sql(cell_column)
        if compartment_value is not None:
            clauses.append(f"({compartment_expression}) = ?")
            parameters.append(compartment_value)
        if availability_value != "ALL":
            clauses.append(f"{availability_column} = ?")
            parameters.append(availability_value == "AVAILABLE")
        relation = f"read_parquet({_sql_path(self.paths[role])})"
        connection = self._connect()
        try:
            frame = connection.execute(
                f"""
                SELECT {select}, {compartment_expression} AS compartment
                FROM {relation}
                WHERE {' AND '.join(clauses)}
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit_value, offset_value],
            ).fetchdf()
        except duckdb.BinderException as exc:
            raise SingleCellFormalContextInputError(
                "A filter is not applicable to this query kind"
            ) from exc
        finally:
            connection.close()
        rows = [
            {str(key): _json_value(value) for key, value in record.items()}
            for record in frame.to_dict("records")
        ]
        filters = {
            "cancer_id": cancer,
            "lncrna_id": lnc,
            "pathway_id": pathway,
            "donor_id": donor,
            "cell_type": cell,
            "compartment": compartment_value,
            "availability": availability_value,
            "limit": limit_value,
            "offset": offset_value,
        }
        return rows, filters

    def query_lncrna_celltype(
        self,
        *,
        cancer_id: Any,
        lncrna_id: Any | None = None,
        cell_type: Any | None = None,
        compartment: Any | None = None,
        availability: Any = "ALL",
        limit: Any = 100,
        offset: Any = 0,
    ) -> dict[str, Any]:
        rows, filters = self._query(
            role="lncrna_celltype",
            select=(
                "dataset_id, cancer_id, lncrna_id, cell_type, cell_key, "
                "lnc_detection_rate, lnc_mean_log_expression, lnc_specificity_tau, "
                "lnc_n_cells, lnc_celltype_available, lnc_celltype_unavailable_reason"
            ),
            cell_column="cell_type",
            availability_column="lnc_celltype_available",
            cancer_id=cancer_id,
            lncrna_id=lncrna_id,
            cell_type=cell_type,
            compartment=compartment,
            availability=availability,
            limit=limit,
            offset=offset,
            order_by="lnc_celltype_available DESC, lncrna_id, cell_type",
        )
        return self._response(
            "lncrna_celltype_expression_fact", rows, filters,
            learned_association=False,
            value_semantics="NONPREDICTIVE_EXPRESSION_DETECTION_FACT",
        )

    def query_pathway_activity(
        self,
        *,
        cancer_id: Any,
        pathway_id: Any | None = None,
        donor_id: Any | None = None,
        cell_type: Any | None = None,
        compartment: Any | None = None,
        availability: Any = "ALL",
        limit: Any = 100,
        offset: Any = 0,
    ) -> dict[str, Any]:
        rows, filters = self._query(
            role="pathway_activity",
            select=(
                "dataset_id, cancer_id, donor_id, cell_type, cell_key, pathway_id, "
                "activity_value, activity_kind, activity_source_tier, "
                "effective_source_tier, n_cells, activity_available, "
                "activity_unavailable_reason"
            ),
            cell_column="cell_type",
            availability_column="activity_available",
            cancer_id=cancer_id,
            pathway_id=pathway_id,
            donor_id=donor_id,
            cell_type=cell_type,
            compartment=compartment,
            availability=availability,
            limit=limit,
            offset=offset,
            order_by="activity_available DESC, pathway_id, cell_type, donor_id",
        )
        return self._response(
            "protein_only_donor_pseudobulk_exact_pathway_activity",
            rows,
            filters,
            learned_association=False,
            value_semantics="V3.2_PROTEIN_ONLY_WITHIN_PSEUDOBULK_RANK_MEAN_V1",
        )

    def query_celltype_predictions(
        self,
        *,
        cancer_id: Any,
        lncrna_id: Any | None = None,
        pathway_id: Any | None = None,
        cell_type: Any | None = None,
        compartment: Any | None = None,
        availability: Any = "ALL",
        limit: Any = 100,
        offset: Any = 0,
    ) -> dict[str, Any]:
        rows, filters = self._query(
            role="typed_predictions",
            select=(
                "dataset_id, cancer_id, cell_type, cell_type_major, cell_state, "
                "lncrna_id, pathway_id, single_cell_replication_probability, "
                "single_cell_available, single_cell_unavailable_reason, "
                "target_level, changes_primary_ranking"
            ),
            cell_column="coalesce(cell_type_major, cell_type)",
            availability_column="single_cell_available",
            cancer_id=cancer_id,
            lncrna_id=lncrna_id,
            pathway_id=pathway_id,
            cell_type=cell_type,
            compartment=compartment,
            availability=availability,
            limit=limit,
            offset=offset,
            order_by=(
                "single_cell_available DESC, single_cell_replication_probability DESC "
                "NULLS LAST, lncrna_id, pathway_id, cell_type"
            ),
        )
        return self._response(
            "learned_celltype_lncrna_exact_pathway_replication_probability",
            rows,
            filters,
            learned_association=True,
            value_semantics=(
                "FRESH_FIVE_FOLD_PRIVATE_HEAD_PROBABILITY_OF_DIRECTION_AGNOSTIC_"
                "DONOR_PSEUDOBULK_REPLICATION_STRENGTH"
            ),
        )

    def _response(
        self,
        query_kind: str,
        rows: list[dict[str, Any]],
        filters: dict[str, Any],
        *,
        learned_association: bool,
        value_semantics: str,
    ) -> dict[str, Any]:
        return {
            "module": "single_cell_formal_context",
            "query_kind": query_kind,
            "learned_association": learned_association,
            "value_semantics": value_semantics,
            "activity_is_not_learned_association": query_kind.startswith("protein_only"),
            "single_cell_currently_changes_secondary_score": False,
            "changes_exact_primary_score": False,
            "typed_unavailable_nulls": True,
            "filters": filters,
            "returned_rows": len(rows),
            "rows": rows,
            "provenance": {
                "analysis_version": ANALYSIS_VERSION,
                "training_run_id": self.binding["training_run_id"],
                "binding_path": str(self.binding_path),
                "binding_sha256": self.binding_sha256,
                "historical_results_used": False,
                "release_ready": False,
                "production_deployed": False,
            },
        }


__all__ = [
    "ALL_CANCERS",
    "ANALYSIS_VERSION",
    "BINDING_FORMAT",
    "COMPARTMENTS",
    "EXPECTED_COUNTS",
    "FORMAL_CANCERS",
    "IMMUNE_CELL_TYPES",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
    "STROMAL_CELL_TYPES",
    "SingleCellFormalContextAssetError",
    "SingleCellFormalContextError",
    "SingleCellFormalContextInputError",
    "SingleCellFormalContextQuery",
    "artifact_sha256",
    "build_single_cell_formal_context_binding",
    "cell_type_compartment",
]
