"""Reproducible, hash-bound audit of the remaining V3.2 single-cell gaps.

The audit is deliberately non-predictive.  It does not infer a trajectory
root, reuse historical trajectory/UCell outputs, create figures, or change a
primary/fusion score.  A source dataset with longitudinal metadata may be
recorded as a future candidate, but it cannot be promoted when it is not the
current V3.2 cell authority.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
OBSERVATION_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_GAP_REMOTE_OBSERVATION_V1"
REPORT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_GAP_AUDIT_V1"
LNCRNA_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_LNCRNA_DETECTION_33C_V1"
BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_GAP_AUDIT_BINDING_V1"
INDEPENDENT_REPORT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_GAP_INDEPENDENT_AUDIT_V1"
INDEPENDENT_BINDING_FORMAT = (
    "CC_HHGT_V3_2_SINGLE_CELL_GAP_INDEPENDENT_AUDIT_BINDING_V1"
)

FORMAL_CANCERS = (
    "ACC",
    "BLCA",
    "BRCA",
    "CESC",
    "CHOL",
    "COAD",
    "DLBC",
    "ESCA",
    "GBM",
    "HNSC",
    "KICH",
    "KIRC",
    "KIRP",
    "LAML",
    "LGG",
    "LIHC",
    "LUAD",
    "LUSC",
    "MESO",
    "OV",
    "PAAD",
    "PCPG",
    "PRAD",
    "READ",
    "SARC",
    "SKCM",
    "STAD",
    "TGCT",
    "THCA",
    "THYM",
    "UCEC",
    "UCS",
    "UVM",
)

ROOT_COLUMNS = {
    "trajectory_root",
    "is_trajectory_root",
    "root_cell",
    "root_state",
    "pseudotime_root",
}
ORDER_COLUMNS = {
    "ordered_state",
    "state_order",
    "trajectory_order",
    "timepoint",
    "time_point",
    "collection_day",
    "day",
}
PRECOMPUTED_TIME_COLUMNS = {"pseudotime", "latent_time"}
IMAGE_SUFFIXES = {".png", ".pdf", ".svg", ".tif", ".tiff", ".jpg", ".jpeg"}


class SingleCellGapAuditError(RuntimeError):
    """Raised when a gap audit would overstate or lose source integrity."""


@dataclass(frozen=True)
class SingleCellGapInputs:
    remote_observation: Path
    remote_live_recheck: Path
    lncrna_table: Path
    lncrna_summary: Path
    lncrna_binding: Path
    hnsc_ucell_binding: Path
    figure_manifest: Path

    def resolved(self) -> "SingleCellGapInputs":
        return SingleCellGapInputs(
            **{name: Path(value).resolve() for name, value in self.__dict__.items()}
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleCellGapAuditError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SingleCellGapAuditError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _valid_sha(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _require_file(path: Path, role: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SingleCellGapAuditError(f"Missing/empty {role}: {path}")


def _require_hash(path: Path, expected: Any, role: str) -> None:
    if not _valid_sha(expected):
        raise SingleCellGapAuditError(f"Invalid expected SHA for {role}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise SingleCellGapAuditError(
            f"{role} hash drift: observed={observed} expected={expected}"
        )


def _prepare_output(path: Path) -> Path:
    output = path.resolve()
    if output.exists() and any(output.iterdir()):
        raise SingleCellGapAuditError(f"Output reuse is forbidden: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _read_lncrna_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "cancer_id",
            "source_dataset_id",
            "raw_h5_path",
            "raw_h5_sha256",
            "raw_h5_size_bytes",
            "cell_count",
            "lncRNA_universe_count",
            "detected_lncRNA_count",
            "detected_lncRNA_set_sha256",
            "formal_eligible",
            "failure_or_limitation_reason",
            "h5_coverage_status",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SingleCellGapAuditError(
                f"lncRNA table lacks columns: {sorted(missing)}"
            )
        rows: list[dict[str, Any]] = []
        for raw in reader:
            cancer = str(raw["cancer_id"]).upper()
            detected = int(raw["detected_lncRNA_count"])
            universe = int(raw["lncRNA_universe_count"])
            cells = int(raw["cell_count"])
            if detected < 0 or detected > universe or universe <= 0 or cells <= 0:
                raise SingleCellGapAuditError(
                    f"Invalid detection counts for {cancer}: {detected}/{universe}"
                )
            if not _valid_sha(raw["raw_h5_sha256"]):
                raise SingleCellGapAuditError(f"Invalid H5 SHA for {cancer}")
            if not _valid_sha(raw["detected_lncRNA_set_sha256"]):
                raise SingleCellGapAuditError(f"Invalid detected-set SHA for {cancer}")
            h5_path = str(raw["raw_h5_path"]).replace("\\", "/")
            if not h5_path.startswith(
                "./data/CancerLncAtlas/processed/sc_tool_input/"
            ):
                raise SingleCellGapAuditError(
                    f"H5 path is outside audited scope for {cancer}: {h5_path}"
                )
            rows.append(
                {
                    "cancer_id": cancer,
                    "dataset_id": str(raw["source_dataset_id"]),
                    "cells": cells,
                    "lncrna_universe": universe,
                    "detected_lncrna": detected,
                    "detected_fraction": detected / universe,
                    "detected_lncrna_set_sha256": str(
                        raw["detected_lncRNA_set_sha256"]
                    ),
                    "raw_h5_path": h5_path,
                    "raw_h5_sha256": str(raw["raw_h5_sha256"]),
                    "raw_h5_size_bytes": int(raw["raw_h5_size_bytes"]),
                    "formal_eligible": str(raw["formal_eligible"]).lower()
                    in {"true", "1", "yes"},
                    "limitation": str(raw["failure_or_limitation_reason"]),
                    "coverage_status": str(raw["h5_coverage_status"]),
                }
            )
    cancers = [row["cancer_id"] for row in rows]
    if len(rows) != 33 or len(set(cancers)) != 33 or set(cancers) != set(FORMAL_CANCERS):
        raise SingleCellGapAuditError("lncRNA table is not a unique 33-cancer authority")
    if any(row["coverage_status"] != "COVERED_RAW_H5_READ_ONLY_SCANNED" for row in rows):
        raise SingleCellGapAuditError("At least one raw H5 was not read-only scanned")
    return sorted(rows, key=lambda row: row["cancer_id"])


def _validate_observation(
    observation: dict[str, Any], lncrna_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if observation.get("format") != OBSERVATION_FORMAT:
        raise SingleCellGapAuditError("Wrong remote observation format")
    if observation.get("analysis_version") != ANALYSIS_VERSION:
        raise SingleCellGapAuditError("Wrong remote observation analysis version")
    if observation.get("remote_scope") != "./data/CancerLncAtlas":
        raise SingleCellGapAuditError("Remote observation exceeds the authorised scope")
    policy = observation.get("collection_policy", {})
    required_true = (
        "aggregate_or_schema_only_stdout",
        "patient_or_cell_identifiers_returned",
        "remote_data_files_copied_locally",
        "remote_project_tree_writes_performed",
        "remote_persistent_outputs_written",
    )
    expected = {
        "aggregate_or_schema_only_stdout": True,
        "patient_or_cell_identifiers_returned": False,
        "remote_data_files_copied_locally": False,
        "remote_project_tree_writes_performed": False,
        "remote_persistent_outputs_written": False,
    }
    for key in required_true:
        if policy.get(key) is not expected[key]:
            raise SingleCellGapAuditError(f"Unsafe/ambiguous collection policy: {key}")

    metadata = observation.get("current_processed_metadata", {})
    files = metadata.get("files", [])
    if not isinstance(files, list) or len(files) != 23:
        raise SingleCellGapAuditError("Expected 23 current metadata schema observations")
    metadata_cancers = {str(row.get("cancer_id")) for row in files}
    if len(metadata_cancers) != 23:
        raise SingleCellGapAuditError("Current metadata cancer IDs are duplicated")
    absent = set(metadata.get("cancers_without_current_metadata", []))
    if absent != set(FORMAL_CANCERS) - metadata_cancers or len(absent) != 10:
        raise SingleCellGapAuditError("Current metadata missing-cancer set is inconsistent")
    common = {str(column).lower() for column in metadata.get("common_columns", [])}
    observed_columns: set[str] = set(common)
    for row in files:
        if int(row.get("rows", 0)) <= 0 or not _valid_sha(row.get("sha256")):
            raise SingleCellGapAuditError("Invalid metadata row count or SHA")
        observed_columns.update(
            str(column).lower() for column in row.get("extra_columns", [])
        )
    computed_candidates = {
        "explicit_root": sorted(observed_columns & ROOT_COLUMNS),
        "ordered_state": sorted(observed_columns & ORDER_COLUMNS),
        "precomputed_time": sorted(observed_columns & PRECOMPUTED_TIME_COLUMNS),
    }
    if any(computed_candidates.values()):
        raise SingleCellGapAuditError(
            f"Current metadata unexpectedly gained trajectory columns: {computed_candidates}"
        )
    if metadata.get("candidate_columns_found") != computed_candidates:
        raise SingleCellGapAuditError("Reported metadata candidate columns are inconsistent")
    eligible = {row["cancer_id"] for row in lncrna_rows if row["formal_eligible"]}
    if len(eligible) != 17 or not eligible.issubset(metadata_cancers):
        raise SingleCellGapAuditError("Formal metadata coverage is not 17/17")
    if int(metadata.get("formal_eligible_metadata_files", -1)) != 17:
        raise SingleCellGapAuditError("Formal metadata file count drift")

    raw = observation.get("raw_metadata_header_audit", {})
    candidate = raw.get("candidate_dataset", {})
    if int(raw.get("current_v32_authority_files_with_candidate_categories", -1)) != 0:
        raise SingleCellGapAuditError("Current authority contains an unreviewed time candidate")
    if candidate.get("status") != (
        "CANDIDATE_SEPARATE_READ_LONGITUDINAL_BRANCH_NOT_CURRENT_V32_AUTHORITY"
    ):
        raise SingleCellGapAuditError("READ longitudinal candidate was promoted or lost")
    if candidate.get("same_cells_as_current_v32_read_authority") is not False:
        raise SingleCellGapAuditError("READ candidate cannot be transferred to current cells")
    if candidate.get("numeric_pseudotime_published") is not False:
        raise SingleCellGapAuditError("Candidate must not publish numeric pseudotime")
    if candidate.get("strict_total_order_encoded_by_labels") is not False:
        raise SingleCellGapAuditError("Treatment branches cannot be represented as one total order")
    if candidate.get("current_v32_read_dataset_id") != "SC_GSE302903_READ":
        raise SingleCellGapAuditError("Current READ authority drift")
    if candidate.get("dataset_id") != "SC_GSE254249_READ":
        raise SingleCellGapAuditError("Longitudinal candidate identity drift")
    if not _valid_sha(candidate.get("metadata_sha256")):
        raise SingleCellGapAuditError("Longitudinal candidate metadata SHA is invalid")

    current = observation.get("current_v32_remote_inventory", {})
    if int(current.get("matching_files", -1)) != 0:
        raise SingleCellGapAuditError("Unreviewed V3.2 remote result files appeared")
    historical = observation.get("excluded_historical_results", {})
    if historical.get("status") != "HISTORICAL_DERIVED_RESULTS_EXCLUDED_FROM_V32":
        raise SingleCellGapAuditError("Historical trajectory results were not excluded")
    if historical.get("trajectory_script", {}).get("explicit_source_root_used") is not False:
        raise SingleCellGapAuditError("Historical inferred root was mislabelled as explicit")
    if int(historical.get("pseudotime_files", -1)) != 33:
        raise SingleCellGapAuditError("Historical pseudotime inventory drift")
    if int(historical.get("ucell_aggregate_files", -1)) != 33:
        raise SingleCellGapAuditError("Historical UCell inventory drift")
    for key in ("pseudotime_sha256_manifest_sha256", "ucell_sha256_manifest_sha256"):
        if not _valid_sha(historical.get(key)):
            raise SingleCellGapAuditError(f"Invalid historical manifest hash: {key}")

    figures = observation.get("remote_figure_inventory", {})
    if int(figures.get("current_v32_publishable_figure_files", -1)) != 0:
        raise SingleCellGapAuditError("Current V3.2 figure inventory is not zero")
    for key, count in (
        ("historical_trajectory_figures", 428),
        ("historical_web_umap_figures", 197),
    ):
        entry = figures.get(key, {})
        if int(entry.get("files", -1)) != count:
            raise SingleCellGapAuditError(f"Historical figure inventory drift: {key}")
        if entry.get("status") != "PHYSICAL_FILES_EXIST_HISTORICAL_NOT_V32_PUBLISHABLE":
            raise SingleCellGapAuditError(f"Historical figure was promoted: {key}")
        if not _valid_sha(entry.get("sha256_manifest_sha256")):
            raise SingleCellGapAuditError(f"Invalid historical figure hash: {key}")
    return {
        "metadata_cancers": sorted(metadata_cancers),
        "metadata_candidate_columns": computed_candidates,
        "eligible_cancers": sorted(eligible),
        "longitudinal_candidate": candidate,
    }


def _validate_upstream_bindings(
    inputs: SingleCellGapInputs,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prior = _load_json(inputs.lncrna_binding)
    if prior.get("format") != "CC_HHGT_V3_2_SINGLE_CELL_EXPLICIT_ID_REAUDIT_BINDING_V1":
        raise SingleCellGapAuditError("Wrong explicit-ID reaudit binding")
    artifacts = prior.get("artifacts", {})
    for key, path in (
        ("per_cancer_table", inputs.lncrna_table),
        ("summary", inputs.lncrna_summary),
    ):
        entry = artifacts.get(key, {})
        _require_hash(path, entry.get("sha256"), f"explicit-ID {key}")
    summary = _load_json(inputs.lncrna_summary)
    if summary.get("status") != "SUCCESS_CODE_BOUND_33_CANCER_EXPLICIT_ID_REAUDIT":
        raise SingleCellGapAuditError("Explicit-ID reaudit is not successful")
    if int(summary.get("cancer_count", 0)) != 33:
        raise SingleCellGapAuditError("Explicit-ID summary is not 33-cancer")
    if int(summary.get("r1_count_and_set_hashes_reproduced_cancers", 0)) != 33:
        raise SingleCellGapAuditError("Explicit-ID counts were not reproduced for all cancers")

    ucell = _load_json(inputs.hnsc_ucell_binding)
    if ucell.get("format") != "CC_HHGT_V3_2_HNSC_UCELL_PILOT_QUERY_BINDING_V1":
        raise SingleCellGapAuditError("Wrong HNSC UCell binding")
    if ucell.get("status") != "SUCCESS_HNSC_UCELL_PILOT_HASH_BOUND":
        raise SingleCellGapAuditError("HNSC UCell is not hash-bound success")
    if ucell.get("cancer_id") != "HNSC" or ucell.get("scope") != "HNSC_ONLY_PILOT":
        raise SingleCellGapAuditError("UCell scope is not exactly HNSC")
    if ucell.get("pseudotime_numeric_output") is not False:
        raise SingleCellGapAuditError("HNSC UCell binding unexpectedly contains pseudotime")
    counts = ucell.get("validated_counts", {})
    if int(counts.get("cells", -1)) != 5902:
        raise SingleCellGapAuditError("HNSC UCell cell count drift")
    if int(counts.get("pathways_available", -1)) != 2120:
        raise SingleCellGapAuditError("HNSC UCell pathway coverage drift")
    if int(counts.get("pseudotime_numeric_values", -1)) != 0:
        raise SingleCellGapAuditError("HNSC pseudotime typed gap drift")
    independent = ucell.get("sources", {}).get("independent_audit", {})
    independent_path = Path(str(independent.get("path", "")))
    _require_file(independent_path, "HNSC UCell independent audit")
    _require_hash(
        independent_path,
        independent.get("sha256"),
        "HNSC UCell independent audit",
    )

    figures = _load_json(inputs.figure_manifest)
    if figures.get("format") != "CC_HHGT_V3_2_SINGLE_CELL_FIGURE_MANIFEST_V1":
        raise SingleCellGapAuditError("Wrong V3.2 figure manifest")
    entries = figures.get("entries", [])
    if len(entries) != 33 or {entry.get("cancer_id") for entry in entries} != set(
        FORMAL_CANCERS
    ):
        raise SingleCellGapAuditError("Figure manifest is not unique 33-cancer")
    for entry in entries:
        if (
            entry.get("status") != "NULL_WITH_REASON"
            or entry.get("path") is not None
            or entry.get("sha256") is not None
            or entry.get("reason") != "NO_REGISTERED_SINGLE_CELL_FIGURE"
        ):
            raise SingleCellGapAuditError("A figure entry is not a preserved typed gap")
    return summary, ucell, figures


def materialize_single_cell_gap_audit(
    *,
    inputs: SingleCellGapInputs,
    output_root: str | Path,
) -> dict[str, Any]:
    resolved = inputs.resolved()
    for name, path in resolved.__dict__.items():
        _require_file(path, name)
    output = _prepare_output(Path(output_root))
    summary, ucell, figures = _validate_upstream_bindings(resolved)
    lncrna_rows = _read_lncrna_rows(resolved.lncrna_table)
    observation = _load_json(resolved.remote_observation)
    observation_audit = _validate_observation(observation, lncrna_rows)
    live_recheck = _load_json(resolved.remote_live_recheck)
    if live_recheck.get("format") != (
        "CC_HHGT_V3_2_SINGLE_CELL_GAP_REMOTE_LIVE_RECHECK_V1"
    ):
        raise SingleCellGapAuditError("Wrong remote live-recheck format")
    if live_recheck.get("status") != "PASS" or int(
        live_recheck.get("failed_checks", -1)
    ) != 0:
        raise SingleCellGapAuditError("Remote live recheck did not pass")
    if int(live_recheck.get("checks", -1)) != 35:
        raise SingleCellGapAuditError("Remote live-recheck check count drift")
    if live_recheck.get("remote_scope") != observation.get("remote_scope"):
        raise SingleCellGapAuditError("Remote live-recheck scope drift")
    if live_recheck.get("remote_writes_performed") is not False:
        raise SingleCellGapAuditError("Remote live recheck wrote remote data")
    if live_recheck.get("patient_or_cell_records_returned") is not False:
        raise SingleCellGapAuditError("Remote live recheck returned protected records")
    if live_recheck.get("observation_content_sha256") != _json_sha256(observation):
        raise SingleCellGapAuditError("Remote live recheck is bound to another observation")

    detected = [int(row["detected_lncrna"]) for row in lncrna_rows]
    total_cells = sum(int(row["cells"]) for row in lncrna_rows)
    if total_cells != int(summary.get("total_cells", -1)) or total_cells != 1_919_578:
        raise SingleCellGapAuditError("Total single-cell count drift")
    if int(summary.get("detected_union_count", -1)) != 15_879:
        raise SingleCellGapAuditError("Detected lncRNA union count drift")
    if int(summary.get("detected_intersection_count", -1)) != 4:
        raise SingleCellGapAuditError("Detected lncRNA intersection count drift")

    lncrna_path = output / "LNCRNA_DETECTION_33C.json"
    lncrna_payload = {
        "format": LNCRNA_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "VERIFIED_33_CANCER_RAW_H5_EXPLICIT_ID_COUNTS",
        "definition": "GENCODE_V50_STRICT_PRIMARY_LNCRNA_WITH_FINITE_VALUE_GT_ZERO_IN_AT_LEAST_ONE_CELL",
        "mapping_policy": "VERSION_STRIP;DIRECT_STABLE_ENSEMBL;EXACT_UNIQUE_FULL_ANNOTATION_SYMBOL;STABLE_ID_DEDUP",
        "source_table": _artifact(resolved.lncrna_table),
        "source_summary": _artifact(resolved.lncrna_summary),
        "counts": {
            "cancers": 33,
            "formal_eligible_cancers": sum(
                1 for row in lncrna_rows if row["formal_eligible"]
            ),
            "cells": total_cells,
            "annotation_strict_lncrna": int(
                summary.get("annotation_strict_lncRNA_count", -1)
            ),
            "detected_union": int(summary["detected_union_count"]),
            "detected_intersection": int(summary["detected_intersection_count"]),
            "detected_min": min(detected),
            "detected_median": statistics.median(detected),
            "detected_mean": statistics.mean(detected),
            "detected_max": max(detected),
        },
        "global_set_hashes": {
            "detected_union_sha256": summary["detected_union_sha256"],
            "detected_intersection_sha256": summary["detected_intersection_sha256"],
        },
        "entries": lncrna_rows,
        "remote_h5_files_rehashed_in_this_materialization": False,
        "remote_h5_hashes_bound_from_independently_reproduced_explicit_id_reaudit": True,
        "production_deployed": False,
    }
    _write_json(lncrna_path, lncrna_payload)

    observation_candidate = observation_audit["longitudinal_candidate"]
    report_path = output / "SINGLE_CELL_GAP_AUDIT.json"
    report = {
        "format": REPORT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PARTIAL_WITH_TYPED_GAPS_AND_UNMOUNTED_LONGITUDINAL_CANDIDATE",
        "audit_only": True,
        "changes_primary_score": False,
        "changes_fusion_score": False,
        "numeric_values_invented": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "production_deployed": False,
        "release_ready": False,
        "pseudotime": {
            "status": "GAP_TYPED_UNAVAILABLE_CURRENT_V32",
            "numeric_rows": 0,
            "path": None,
            "unavailable_value": None,
            "unavailable_reason": "NO_EXPLICIT_TRAJECTORY_ROOT_OR_ORDERED_SOURCE_STATE_IN_CURRENT_V32_AUTHORITIES",
            "current_metadata_files_audited": 23,
            "current_metadata_cancers": observation_audit["metadata_cancers"],
            "current_formal_metadata_cancers": observation_audit["eligible_cancers"],
            "candidate_columns_found": observation_audit[
                "metadata_candidate_columns"
            ],
            "historical_inferred_root_results_excluded": True,
            "separate_read_candidate": {
                "dataset_id": observation_candidate["dataset_id"],
                "accession": observation_candidate["accession"],
                "metadata_sha256": observation_candidate["metadata_sha256"],
                "patients": observation_candidate["patients"],
                "patients_with_multiple_timepoint_categories": observation_candidate[
                    "patients_with_multiple_timepoint_categories"
                ],
                "timepoint_levels": observation_candidate["timepoint_levels"],
                "status": observation_candidate["status"],
                "same_cells_as_current_v32_read_authority": False,
                "strict_total_order_encoded_by_labels": False,
                "numeric_pseudotime_published": False,
                "permitted_future_role": "FRESH_READ_TREATMENT_BRANCH_ANALYSIS_AFTER_SEPARATE_REGISTRATION",
            },
        },
        "ucell": {
            "status": "PARTIAL_SCOPE_HNSC_ONLY",
            "formal_cancers_covered": 1,
            "formal_cancers_total": 33,
            "covered_cancers": ["HNSC"],
            "cells": int(ucell["validated_counts"]["cells"]),
            "pathways_total": int(ucell["validated_counts"]["pathways_total"]),
            "pathways_available": int(
                ucell["validated_counts"]["pathways_available"]
            ),
            "cell_level_numeric_rows": int(
                ucell["validated_counts"]["cell_level_numeric_rows"]
            ),
            "other_32_cancers": "GAP_TYPED_UNAVAILABLE_NO_FRESH_FORMAL_CELL_LEVEL_UCELL",
            "historical_33_cancer_ucell_excluded": True,
        },
        "lncrna_detection": {
            "status": lncrna_payload["status"],
            "artifact": _artifact(lncrna_path),
            "counts": lncrna_payload["counts"],
            "global_set_hashes": lncrna_payload["global_set_hashes"],
        },
        "figures": {
            "status": "GAP_TYPED_UNAVAILABLE_CURRENT_V32",
            "manifest_entries": len(figures["entries"]),
            "current_v32_figure_files": 0,
            "typed_unavailable_entries": 33,
            "historical_physical_files_exist": True,
            "historical_trajectory_figure_files": int(
                observation["remote_figure_inventory"][
                    "historical_trajectory_figures"
                ]["files"]
            ),
            "historical_web_umap_files": int(
                observation["remote_figure_inventory"][
                    "historical_web_umap_figures"
                ]["files"]
            ),
            "historical_files_publishable_as_v32": False,
            "unavailable_reason": "NO_HASH_BOUND_FRESH_V32_SINGLE_CELL_FIGURE_FILES",
        },
        "typed_gaps": [
            {
                "artifact_id": "v32_sc_pseudotime",
                "status": "GAP_TYPED_UNAVAILABLE",
                "numeric_rows": 0,
                "reason": "NO_EXPLICIT_TRAJECTORY_ROOT_OR_ORDERED_SOURCE_STATE_IN_CURRENT_V32_AUTHORITIES",
            },
            {
                "artifact_id": "v32_sc_ucell_other_32_cancers",
                "status": "GAP_TYPED_UNAVAILABLE",
                "numeric_rows": 0,
                "reason": "NO_FRESH_FORMAL_CELL_LEVEL_UCELL_OUTSIDE_HNSC",
            },
            {
                "artifact_id": "v32_sc_figures",
                "status": "GAP_TYPED_UNAVAILABLE",
                "files": 0,
                "reason": "NO_HASH_BOUND_FRESH_V32_SINGLE_CELL_FIGURE_FILES",
            },
        ],
        "excluded_historical_results": observation["excluded_historical_results"],
        "remote_observation": _artifact(resolved.remote_observation),
        "remote_live_recheck": _artifact(resolved.remote_live_recheck),
    }
    _write_json(report_path, report)

    repo_root = resolved.remote_observation.parent.parent
    code_paths = {
        "audit_module": Path(__file__).resolve(),
        "materializer": repo_root
        / "scripts"
        / "materialize_v32_single_cell_gap_audit.py",
        "independent_auditor": repo_root
        / "scripts"
        / "audit_v32_single_cell_gap_release.py",
        "remote_readonly_verifier": repo_root
        / "scripts"
        / "verify_v32_single_cell_gap_remote_readonly.py",
        "tests": repo_root / "tests" / "test_v32_single_cell_gap_audit.py",
    }
    for name, path in code_paths.items():
        _require_file(path, f"audit code {name}")
    binding_path = output / "SINGLE_CELL_GAP_AUDIT_BINDING.json"
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "SUCCESS_HASH_BOUND_TYPED_GAP_AUDIT",
        "inputs": {
            name: _artifact(path) for name, path in resolved.__dict__.items()
        },
        "code": {name: _artifact(path) for name, path in code_paths.items()},
        "outputs": {
            lncrna_path.name: _artifact(lncrna_path),
            report_path.name: _artifact(report_path),
        },
        "capability_status": {
            "pseudotime": report["pseudotime"]["status"],
            "ucell": report["ucell"]["status"],
            "lncrna_detection": report["lncrna_detection"]["status"],
            "figures": report["figures"]["status"],
        },
        "remote_live_rehash_performed_by_materializer": False,
        "remote_live_recheck_bound": True,
        "remote_observation_sha256": sha256_file(resolved.remote_observation),
        "historical_results_promoted": False,
        "candidate_results_promoted": False,
        "supersedes_local_draft": "v32_single_cell_gap_audit_20260826_r1",
        "release_ready": False,
        "production_deployed": False,
    }
    _write_json(binding_path, binding)
    success_path = output / "SUCCESS.json"
    success = {
        "format": BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "SUCCESS_HASH_BOUND_TYPED_GAP_AUDIT",
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "report_sha256": sha256_file(report_path),
        "lncrna_detection_sha256": sha256_file(lncrna_path),
        "release_ready": False,
        "production_deployed": False,
    }
    _write_json(success_path, success)
    return success


def _check(
    checks: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


def audit_single_cell_gap_release(
    *,
    release_root: str | Path,
    audit_root: str | Path,
    expected_binding_sha256: str | None = None,
) -> dict[str, Any]:
    release = Path(release_root).resolve()
    output = _prepare_output(Path(audit_root))
    binding_path = release / "SINGLE_CELL_GAP_AUDIT_BINDING.json"
    success_path = release / "SUCCESS.json"
    _require_file(binding_path, "gap audit binding")
    _require_file(success_path, "gap audit success")
    binding_sha = sha256_file(binding_path)
    if expected_binding_sha256 is not None and binding_sha != expected_binding_sha256:
        raise SingleCellGapAuditError(
            f"external binding hash mismatch: {binding_sha} != {expected_binding_sha256}"
        )
    binding = _load_json(binding_path)
    success = _load_json(success_path)
    checks: list[dict[str, Any]] = []
    _check(checks, "binding_format", binding.get("format") == BINDING_FORMAT, binding.get("format"), BINDING_FORMAT)
    _check(checks, "success_binding_sha", success.get("binding_sha256") == binding_sha, success.get("binding_sha256"), binding_sha)
    _check(checks, "release_not_ready", binding.get("release_ready") is False, binding.get("release_ready"), False)
    _check(checks, "production_not_deployed", binding.get("production_deployed") is False, binding.get("production_deployed"), False)
    _check(checks, "historical_not_promoted", binding.get("historical_results_promoted") is False, binding.get("historical_results_promoted"), False)
    _check(checks, "candidate_not_promoted", binding.get("candidate_results_promoted") is False, binding.get("candidate_results_promoted"), False)

    for group_name in ("inputs", "outputs", "code"):
        group = binding.get(group_name, {})
        for name, entry in sorted(group.items()):
            path = Path(str(entry.get("path", "")))
            exists = path.is_file()
            _check(checks, f"{group_name}.{name}.exists", exists, exists, True)
            if exists:
                observed = sha256_file(path)
                _check(checks, f"{group_name}.{name}.sha256", observed == entry.get("sha256"), observed, entry.get("sha256"))
                _check(checks, f"{group_name}.{name}.bytes", path.stat().st_size == int(entry.get("bytes", -1)), path.stat().st_size, entry.get("bytes"))

    report_path = release / "SINGLE_CELL_GAP_AUDIT.json"
    lncrna_path = release / "LNCRNA_DETECTION_33C.json"
    report = _load_json(report_path)
    lncrna = _load_json(lncrna_path)
    _check(checks, "report_format", report.get("format") == REPORT_FORMAT, report.get("format"), REPORT_FORMAT)
    _check(checks, "audit_only", report.get("audit_only") is True, report.get("audit_only"), True)
    _check(checks, "primary_unchanged", report.get("changes_primary_score") is False, report.get("changes_primary_score"), False)
    _check(checks, "fusion_unchanged", report.get("changes_fusion_score") is False, report.get("changes_fusion_score"), False)
    _check(checks, "numeric_not_invented", report.get("numeric_values_invented") is False, report.get("numeric_values_invented"), False)
    _check(checks, "pseudotime_numeric_zero", int(report.get("pseudotime", {}).get("numeric_rows", -1)) == 0, report.get("pseudotime", {}).get("numeric_rows"), 0)
    _check(checks, "pseudotime_path_null", report.get("pseudotime", {}).get("path") is None, report.get("pseudotime", {}).get("path"), None)
    _check(checks, "read_candidate_not_same_cells", report.get("pseudotime", {}).get("separate_read_candidate", {}).get("same_cells_as_current_v32_read_authority") is False, report.get("pseudotime", {}).get("separate_read_candidate", {}).get("same_cells_as_current_v32_read_authority"), False)
    _check(checks, "read_candidate_no_numeric_output", report.get("pseudotime", {}).get("separate_read_candidate", {}).get("numeric_pseudotime_published") is False, report.get("pseudotime", {}).get("separate_read_candidate", {}).get("numeric_pseudotime_published"), False)
    _check(checks, "ucell_hnsc_only", report.get("ucell", {}).get("covered_cancers") == ["HNSC"], report.get("ucell", {}).get("covered_cancers"), ["HNSC"])
    _check(checks, "ucell_one_of_33", (report.get("ucell", {}).get("formal_cancers_covered"), report.get("ucell", {}).get("formal_cancers_total")) == (1, 33), [report.get("ucell", {}).get("formal_cancers_covered"), report.get("ucell", {}).get("formal_cancers_total")], [1, 33])
    _check(checks, "figures_current_zero", int(report.get("figures", {}).get("current_v32_figure_files", -1)) == 0, report.get("figures", {}).get("current_v32_figure_files"), 0)
    _check(checks, "historical_figures_not_publishable", report.get("figures", {}).get("historical_files_publishable_as_v32") is False, report.get("figures", {}).get("historical_files_publishable_as_v32"), False)
    entries = lncrna.get("entries", [])
    _check(checks, "lncrna_format", lncrna.get("format") == LNCRNA_FORMAT, lncrna.get("format"), LNCRNA_FORMAT)
    _check(checks, "lncrna_33_unique", len(entries) == 33 and len({row.get("cancer_id") for row in entries}) == 33, len(entries), 33)
    _check(checks, "lncrna_cancer_universe", {row.get("cancer_id") for row in entries} == set(FORMAL_CANCERS), sorted({row.get("cancer_id") for row in entries}), list(FORMAL_CANCERS))
    _check(checks, "lncrna_cells", int(lncrna.get("counts", {}).get("cells", -1)) == 1_919_578, lncrna.get("counts", {}).get("cells"), 1_919_578)
    _check(checks, "lncrna_union", int(lncrna.get("counts", {}).get("detected_union", -1)) == 15_879, lncrna.get("counts", {}).get("detected_union"), 15_879)
    _check(checks, "lncrna_intersection", int(lncrna.get("counts", {}).get("detected_intersection", -1)) == 4, lncrna.get("counts", {}).get("detected_intersection"), 4)
    _check(checks, "lncrna_min_max", (lncrna.get("counts", {}).get("detected_min"), lncrna.get("counts", {}).get("detected_max")) == (40, 14863), [lncrna.get("counts", {}).get("detected_min"), lncrna.get("counts", {}).get("detected_max")], [40, 14863])

    failed = [item for item in checks if not item["passed"]]
    audit_report_path = output / "INDEPENDENT_AUDIT.json"
    audit_report = {
        "format": INDEPENDENT_REPORT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS" if not failed else "FAIL",
        "checks": len(checks),
        "passed_checks": len(checks) - len(failed),
        "failed_checks": len(failed),
        "check_results": checks,
        "release_binding_path": str(binding_path),
        "release_binding_sha256": binding_sha,
        "remote_live_rehash_performed": False,
        "frozen_remote_observation_revalidated": True,
        "historical_results_promoted": False,
        "production_deployed": False,
    }
    _write_json(audit_report_path, audit_report)
    audit_binding_path = output / "INDEPENDENT_AUDIT_BINDING.json"
    audit_binding = {
        "format": INDEPENDENT_BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS_HASH_BOUND" if not failed else "FAIL_HASH_BOUND",
        "checks": len(checks),
        "failed_checks": len(failed),
        "release_binding_path": str(binding_path),
        "release_binding_sha256": binding_sha,
        "report_path": str(audit_report_path),
        "report_sha256": sha256_file(audit_report_path),
        "production_deployed": False,
    }
    _write_json(audit_binding_path, audit_binding)
    _write_json(
        output / "SUCCESS.json",
        {
            "format": INDEPENDENT_BINDING_FORMAT,
            "status": audit_binding["status"],
            "binding_path": str(audit_binding_path),
            "binding_sha256": sha256_file(audit_binding_path),
            "report_sha256": sha256_file(audit_report_path),
            "checks": len(checks),
            "failed_checks": len(failed),
            "production_deployed": False,
        },
    )
    if failed:
        raise SingleCellGapAuditError(
            "Independent audit failed: " + ", ".join(item["name"] for item in failed)
        )
    return audit_binding
