#!/usr/bin/env python3
"""Run resumable-status, non-overwriting V3.2 single-cell cancer partitions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.input_lineage import artifact_sha256  # noqa: E402
from cc_hhgt.v32.single_cell_input_builder import EXPECTED_CANCERS  # noqa: E402
from cc_hhgt.v32.single_cell_partition_builder import (  # noqa: E402
    build_single_cell_partition,
    load_current_v32_authority,
    prepare_gencode_annotation,
)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _run_one(arguments: dict) -> dict:
    return build_single_cell_partition(**arguments)


def _summary(scope: list[str], records: dict[str, dict], run_id: str) -> dict:
    success = sorted(
        cancer for cancer, row in records.items()
        if str(row.get("status", "")).startswith("SUCCESS_PARTITION_BUILT")
    )
    formal = sorted(cancer for cancer in success if records[cancer].get("formal_eligible") is True)
    blocked = sorted(
        cancer for cancer, row in records.items()
        if str(row.get("status", "")).startswith("BLOCKED_")
    )
    failed = sorted(
        cancer for cancer, row in records.items()
        if str(row.get("status", "")).startswith("FAILED_")
    )
    pending = sorted(set(scope) - set(records))
    return {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_33C_PARTITION_RUN_STATUS_V1",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "run_id": run_id,
        "scope": scope,
        "scope_count": len(scope),
        "scope_is_full_33": set(scope) == EXPECTED_CANCERS,
        "completed_cancers": success,
        "completed_count": len(success),
        "formal_eligible_cancers": formal,
        "formal_eligible_count": len(formal),
        "blocked_cancers": blocked,
        "blocked_count": len(blocked),
        "failed_cancers": failed,
        "failed_count": len(failed),
        "pending_cancers": pending,
        "pending_count": len(pending),
        "per_cancer": records,
        "expression_audit_is_not_training_completion": True,
        "training_started": False,
        "fresh_ucell_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
        "fresh_pseudotime_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
        "single_cell_module_complete": False,
        "release_ready": False,
        "production_deployed": False,
    }


def build_feature_count_delta_table(
    scope: list[str], records: dict[str, dict], audit: pd.DataFrame
) -> pd.DataFrame:
    """Build an honest fresh-vs-audit table.

    Only a successfully rebuilt partition has a fresh direct-ID count.  A
    metadata-blocked or failed partition did not scan the H5 feature IDs and
    therefore must remain typed unavailable even if its status carries a
    legacy/audit feature count for manifest compatibility.
    """

    rows = []
    for cancer in scope:
        record = records[cancer]
        audit_count = int(audit.loc[cancer, "canonical_lnc_feature_unique"])
        success = str(record.get("status", "")).startswith(
            "SUCCESS_PARTITION_BUILT"
        )
        if success:
            fresh_count = record.get("fresh_direct_id_lncrna_feature_count")
            if fresh_count is None:
                # Backward-compatible reading of successful r5 controls only;
                # this fallback is never used for blocked/failed partitions.
                fresh_count = record.get("lncrna_feature_universe_count")
            delta = record.get("fresh_minus_audit_feature_count")
            tolerance = record.get("feature_count_tolerance")
            within = (
                abs(int(delta)) <= int(tolerance)
                if fresh_count is not None
                and delta is not None
                and tolerance is not None
                else pd.NA
            )
        else:
            fresh_count = pd.NA
            delta = pd.NA
            tolerance = pd.NA
            within = pd.NA
        rows.append(
            {
                "cancer_id": cancer,
                "audit_symbol_first_lncrna_feature_count": audit_count,
                "fresh_direct_id_lncrna_feature_count": fresh_count,
                "fresh_minus_audit": delta,
                "allowed_delta_max_25_or_2pct": tolerance,
                "within_delta_gate": within,
                "partition_status": record.get("status"),
                "formal_eligible": record.get("formal_eligible", False),
                "blocking_reason": record.get("blocking_reason"),
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sc-root", required=True, type=Path)
    parser.add_argument("--expression-audit-tsv", required=True, type=Path)
    parser.add_argument("--gencode-gtf", required=True, type=Path)
    parser.add_argument("--exact-membership", required=True, type=Path)
    parser.add_argument("--exact-candidates", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cancers", nargs="*", default=None)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--min-association-observations", type=int, default=5)
    parser.add_argument("--association-chunk-size", type=int, default=100_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    scope = sorted(
        {str(value).upper().strip() for value in (args.cancers or EXPECTED_CANCERS)}
    )
    unexpected = sorted(set(scope) - EXPECTED_CANCERS)
    if unexpected:
        raise ValueError(f"Unexpected cancers: {unexpected}")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Output reuse is forbidden: {output}")

    # Complete the immutable authority preflight before creating result state.
    exact_membership, _, authority = load_current_v32_authority(
        args.exact_membership, args.exact_candidates
    )
    audit = pd.read_csv(args.expression_audit_tsv, sep="\t")
    required = {
        "cancer_id", "source_tier", "measurement_scale",
        "canonical_lnc_feature_unique",
    }
    if missing := sorted(required - set(audit.columns)):
        raise ValueError(f"Expression audit table lacks columns: {missing}")
    audit["cancer_id"] = audit.cancer_id.astype(str).str.upper().str.strip()
    if set(audit.cancer_id) != EXPECTED_CANCERS or audit.cancer_id.duplicated().any():
        raise ValueError("Expression audit must contain exactly one row for each of 33 cancers")
    audit = audit.set_index("cancer_id")

    output.mkdir(parents=True)
    authority_root = output / "authority"
    partitions_root = output / "partitions"
    authority_root.mkdir()
    partitions_root.mkdir()
    annotation_path = authority_root / "gencode_v50_gene_annotation.parquet"
    annotation_provenance = authority_root / "GENCODE_ANNOTATION_PROVENANCE.json"
    annotation = prepare_gencode_annotation(
        gtf_path=args.gencode_gtf,
        output_parquet=annotation_path,
        provenance_path=annotation_provenance,
    )
    filtered_membership_path = authority_root / "exact_pathway_membership_2135.parquet"
    _atomic_parquet(filtered_membership_path, exact_membership)
    authority["filtered_exact_membership_path"] = str(filtered_membership_path)
    authority["filtered_exact_membership_sha256"] = artifact_sha256(
        filtered_membership_path
    )
    authority_payload = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_33C_AUTHORITY_V1",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "run_id": args.run_id,
        "exact_authority": authority,
        "gencode_annotation": annotation,
        "expression_audit_path": str(args.expression_audit_tsv.resolve()),
        "expression_audit_sha256": artifact_sha256(args.expression_audit_tsv.resolve()),
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "historical_sc_trajectory_used": False,
    }
    _atomic_json(authority_root / "AUTHORITY.json", authority_payload)

    records: dict[str, dict] = {}
    _atomic_json(output / "RUN_STATUS.json", _summary(scope, records, args.run_id))
    futures = {}
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for cancer in scope:
            row = audit.loc[cancer]
            arguments = {
                "cancer_id": cancer,
                "h5_path": args.sc_root / cancer / "raw_feature_bc_matrix.h5",
                "metadata_path": args.sc_root / cancer / "cell_metadata.parquet",
                "annotation_parquet_path": annotation_path,
                "annotation_provenance_path": annotation_provenance,
                "membership_path": args.exact_membership,
                "candidate_path": args.exact_candidates,
                "output_root": partitions_root / f"cancer_id={cancer}",
                "source_tier": str(row.source_tier),
                "measurement_scale": str(row.measurement_scale),
                "audited_lnc_feature_count": int(row.canonical_lnc_feature_unique),
                "min_association_observations": args.min_association_observations,
                "association_chunk_size": args.association_chunk_size,
            }
            futures[executor.submit(_run_one, arguments)] = (cancer, arguments)
        for future in as_completed(futures):
            cancer, arguments = futures[future]
            try:
                records[cancer] = future.result()
            except Exception as exc:
                failure_root = Path(arguments["output_root"])
                failure_root.mkdir(parents=True, exist_ok=False)
                failure = {
                    "format": "CC_HHGT_V3_2_SINGLE_CELL_PARTITION_FAILURE_V1",
                    "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
                    "cancer_id": cancer,
                    "status": "FAILED_PARTITION_BUILD",
                    "formal_eligible": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": "".join(traceback.format_exception(exc)),
                    "release_ready": False,
                    "production_deployed": False,
                }
                _atomic_json(failure_root / "PARTITION_STATUS.json", failure)
                records[cancer] = failure
            _atomic_json(output / "RUN_STATUS.json", _summary(scope, records, args.run_id))

    summary = _summary(scope, records, args.run_id)
    delta_path = output / "FEATURE_COUNT_DELTA.tsv"
    _atomic_tsv(delta_path, build_feature_count_delta_table(scope, records, audit))

    handoff_root = output / "training_handoff"
    handoff_root.mkdir()
    manifest_rows = []
    successful_partitions = sorted(
        cancer for cancer, row in records.items()
        if str(row.get("status", "")).startswith("SUCCESS_PARTITION_BUILT")
    )
    for cancer in scope:
        source = partitions_root / f"cancer_id={cancer}" / "dataset_manifest_row.parquet"
        if source.is_file():
            manifest_rows.append(pd.read_parquet(source))
    if len(manifest_rows) != len(scope):
        missing = sorted(
            cancer for cancer in scope
            if not (
                partitions_root / f"cancer_id={cancer}" / "dataset_manifest_row.parquet"
            ).is_file()
        )
        # Failed partitions have no trusted manifest row.  Persist the gap in
        # the handoff rather than inventing a formal declaration.
    else:
        missing = []
    manifest_path = handoff_root / "dataset_manifest_33c.parquet"
    if manifest_rows:
        _atomic_parquet(manifest_path, pd.concat(manifest_rows, ignore_index=True))
    asset_map = {
        "association": "single_cell_association.parquet",
        "lnc_celltype": "lnc_celltype.parquet",
        "activity": "activity.parquet",
        "expression_facts": "gene_expression_facts.parquet",
        "exact_pathway_availability": "exact_pathway_availability.parquet",
    }
    handoff_assets: dict[str, dict] = {}
    for asset, filename in asset_map.items():
        asset_root = handoff_root / asset
        asset_root.mkdir()
        for cancer in successful_partitions:
            source = partitions_root / f"cancer_id={cancer}" / filename
            if source.is_file():
                _link_or_copy(source, asset_root / f"{cancer}.parquet")
        files = sorted(asset_root.glob("*.parquet"))
        handoff_assets[asset] = {
            "path": str(asset_root),
            "partition_files": len(files),
            "partition_cancers": [path.stem for path in files],
            "composite_sha256": (
                hashlib.sha256(
                    "\n".join(
                        f"{path.name}\t{artifact_sha256(path)}" for path in files
                    ).encode("utf-8")
                ).hexdigest()
                if files
                else None
            ),
        }
    training_handoff = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_PARTITIONED_TRAINING_HANDOFF_V1",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "run_id": args.run_id,
        "scope": scope,
        "scope_is_full_33": set(scope) == EXPECTED_CANCERS,
        "candidate_path": str(args.exact_candidates.resolve()),
        "candidate_sha256": authority["candidate_sha256"],
        "exact_membership_path": str(filtered_membership_path),
        "exact_membership_sha256": authority["filtered_exact_membership_sha256"],
        "dataset_manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "dataset_manifest_sha256": (
            artifact_sha256(manifest_path) if manifest_path.is_file() else None
        ),
        "manifest_missing_failed_cancers": missing,
        "assets": handoff_assets,
        "association_generation": "V3.2_FRESH_FROM_RAW_FEATURE_BC_MATRIX_H5",
        "lnc_celltype_generation": "V3.2_FRESH_FROM_RAW_FEATURE_BC_MATRIX_H5",
        "activity_generation": "V3.2_FRESH_FROM_RAW_FEATURE_BC_MATRIX_H5",
        "availability_policy": (
            "EXPLICIT_EXACT_PATHWAY_NULL_REASON_NO_ZERO_OR_HALF_IMPUTATION"
        ),
        "core_embedding_manifest": "REQUIRED_CURRENT_V32_FRESH_CORE_NOT_YET_BOUND",
        "private_head_training_ready_for_available_partitions": (
            set(scope) == EXPECTED_CANCERS
            and not missing
            and summary["failed_count"] == 0
            and summary["formal_eligible_count"] > 0
        ),
        "all_33_formal_eligible": summary["formal_eligible_count"] == 33,
        "fresh_ucell_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
        "fresh_pseudotime_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
        "historical_sc_trajectory_allowed": False,
        "single_cell_module_complete": False,
        "training_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _atomic_json(handoff_root / "TRAINING_HANDOFF.json", training_handoff)
    gate = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_FORMAL_33C_INPUT_GATE_V1",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "run_id": args.run_id,
        "scope_is_full_33": summary["scope_is_full_33"],
        "all_partitions_built": summary["completed_count"] == 33,
        "all_33_formal_eligible": summary["formal_eligible_count"] == 33,
        "formal_33c_training_ready": (
            summary["scope_is_full_33"]
            and summary["completed_count"] == 33
            and summary["formal_eligible_count"] == 33
            and summary["failed_count"] == 0
            and summary["blocked_count"] == 0
        ),
        "formal_eligible_cancers": summary["formal_eligible_cancers"],
        "blocked_cancers": summary["blocked_cancers"],
        "failed_cancers": summary["failed_cancers"],
        "quality_limited_cancers": sorted(
            cancer for cancer, row in records.items()
            if row.get("blocking_reason") == "LIMITED_LNCRNA_FEATURE_UNIVERSE"
        ),
        "missing_metadata_cancers": sorted(
            cancer for cancer, row in records.items()
            if row.get("blocking_reason") == "DONOR_CELLTYPE_METADATA_UNAVAILABLE"
        ),
        "fresh_direct_id_counts_authoritative_for_built_partitions": True,
        "feature_count_delta_gate": "ABS_FRESH_MINUS_AUDIT_LE_MAX_25_OR_2_PERCENT",
        "feature_count_delta_path": str(delta_path),
        "feature_count_delta_sha256": artifact_sha256(delta_path),
        "training_handoff_path": str(handoff_root / "TRAINING_HANDOFF.json"),
        "training_handoff_sha256": artifact_sha256(
            handoff_root / "TRAINING_HANDOFF.json"
        ),
        "expression_audit_is_not_training_completion": True,
        "fresh_ucell_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
        "fresh_pseudotime_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
        "historical_sc_trajectory_allowed": False,
        "single_cell_module_complete": False,
        "training_started": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _atomic_json(output / "FORMAL_33C_INPUT_GATE.json", gate)
    summary["formal_33c_input_gate"] = gate
    _atomic_json(output / "RUN_STATUS.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
