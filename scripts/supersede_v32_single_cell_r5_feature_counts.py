#!/usr/bin/env python3
"""Create a non-overwriting control-layer supersession for the r5 count audit.

The r5 data partitions remain immutable.  This script corrects the control
semantics that had exposed the read-only audit count as a fresh direct-ID
count for metadata-blocked cancers.  It writes a new run status, feature-count
table, manifest, training handoff and gate, all with explicit old-to-new
lineage and SHA-256 records.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from cc_hhgt.v32.input_lineage import artifact_sha256  # noqa: E402
from cc_hhgt.v32.single_cell_input_builder import EXPECTED_CANCERS  # noqa: E402
from scripts.run_v32_single_cell_33c_partition_build import (  # noqa: E402
    build_feature_count_delta_table,
)


SOURCE_HASHES = {
    "RUN_STATUS.json": "e7dc42bc9a4fc8b9e426e778e71034babe38ee31de4053932057ec9fdaba46ed",
    "FORMAL_33C_INPUT_GATE.json": "73f730ecac8833a7657932493010feda63f4ebe86d343477b4f9d49acfe0b7db",
    "FEATURE_COUNT_DELTA.tsv": "8d757152c9fd743a2c1e13b7443ac5333f063b58891cc9768e087239ed4d682f",
    "training_handoff/TRAINING_HANDOFF.json": "968e5b02d7b09c022ab32ac5b5cffc9fdfc81b044a053f3539a4b2f4aacfa6df",
    "training_handoff/dataset_manifest_33c.parquet": "641536abbaf1127aa740a46f88b70b379cfd1996ac678fb65be73b35598a6b94",
}
EXPECTED_BLOCKED = frozenset(
    {"BLCA", "BRCA", "COAD", "LIHC", "LUAD", "OV", "PAAD", "PRAD", "STAD", "THCA"}
)
SOURCE_RUN_ID = "v32-single-cell-33c-partitions-20260826-r5"
STATUS_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_33C_RUN_STATUS_SEMANTIC_SUPERSESSION_V2"
HANDOFF_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_PARTITIONED_TRAINING_HANDOFF_V2"
GATE_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_FORMAL_33C_INPUT_GATE_V2"
SUPERSESSION_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_CONTROL_SUPERSESSION_V1"


class SupersessionError(RuntimeError):
    """Raised when immutable r5 lineage or correction invariants fail."""


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _attach_contract(payload: dict[str, Any], key: str) -> dict[str, Any]:
    payload.pop(key, None)
    payload[key] = _canonical_sha256(payload)
    return payload


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_source(
    source: Path,
    expected_hashes: Mapping[str, str],
    *,
    require_full_authority: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    for relative, expected in expected_hashes.items():
        path = source / relative
        if not path.is_file():
            raise SupersessionError(f"Required immutable r5 control is missing: {path}")
        observed = artifact_sha256(path)
        if observed != expected:
            raise SupersessionError(
                f"Immutable r5 hash drift for {relative}: {observed} != {expected}"
            )
    run = _load_json(source / "RUN_STATUS.json")
    gate = _load_json(source / "FORMAL_33C_INPUT_GATE.json")
    handoff = _load_json(source / "training_handoff" / "TRAINING_HANDOFF.json")
    delta = pd.read_csv(source / "FEATURE_COUNT_DELTA.tsv", sep="\t")
    manifest = pd.read_parquet(
        source / "training_handoff" / "dataset_manifest_33c.parquet"
    )
    scope = set(map(str, run.get("scope", [])))
    blocked = set(map(str, run.get("blocked_cancers", [])))
    if require_full_authority:
        if run.get("run_id") != SOURCE_RUN_ID:
            raise SupersessionError(f"Unexpected source run_id: {run.get('run_id')}")
        if scope != EXPECTED_CANCERS or len(scope) != 33:
            raise SupersessionError("Source is not the exact V3.2 33-cancer authority")
        if blocked != EXPECTED_BLOCKED:
            raise SupersessionError(f"Unexpected metadata-blocked cancers: {sorted(blocked)}")
        if run.get("failed_count") != 0 or run.get("pending_count") != 0:
            raise SupersessionError("Source r5 run is not terminal and failure-free")
        if len(delta) != 33 or manifest.cancer_id.astype(str).nunique() != 33:
            raise SupersessionError("Source r5 controls do not cover all 33 cancers")
    return run, gate, handoff, delta, manifest


def _correct_status_records(
    run: Mapping[str, Any], source_root: Path
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    corrected: dict[str, dict[str, Any]] = {}
    lineage_rows: list[dict[str, Any]] = []
    for cancer in map(str, run["scope"]):
        original = copy.deepcopy(run["per_cancer"][cancer])
        row = copy.deepcopy(original)
        original_contract = row.get("contract_sha256")
        status = str(row.get("status", ""))
        success = status.startswith("SUCCESS_PARTITION_BUILT")
        blocked = status.startswith("BLOCKED_")
        generic_count = row.get("lncrna_feature_universe_count")
        audited_count = row.get(
            "audited_lncrna_feature_universe_count", generic_count
        )
        if success:
            fresh_count = row.get(
                "fresh_direct_id_lncrna_feature_count", generic_count
            )
            row["fresh_direct_id_lncrna_feature_count"] = int(fresh_count)
            row["lncrna_feature_universe_count_semantics"] = (
                "FRESH_DIRECT_STABLE_ID_THEN_UNIQUE_GENCODE_SYMBOL_COUNT"
            )
        else:
            row["fresh_direct_id_lncrna_feature_count"] = None
            row["fresh_minus_audit_feature_count"] = None
            row["feature_count_tolerance"] = None
            row["lncrna_feature_universe_count_semantics"] = (
                "READ_ONLY_AUDIT_COUNT_METADATA_BLOCKED_H5_NOT_SCANNED"
                if blocked
                else "FRESH_DIRECT_ID_COUNT_UNAVAILABLE_PARTITION_NOT_BUILT"
            )
        row["audited_lncrna_feature_universe_count"] = (
            int(audited_count) if audited_count is not None else None
        )
        row["supersedes_contract_sha256"] = original_contract
        row["supersession_reason"] = (
            "BLOCKED_AUDIT_COUNT_WAS_MISLABELLED_AS_FRESH_DIRECT_ID_COUNT"
        )
        row["source_partition_root"] = str(
            source_root / "partitions" / f"cancer_id={cancer}"
        )
        _attach_contract(row, "contract_sha256")
        corrected[cancer] = row
        lineage_rows.append(
            {
                "cancer_id": cancer,
                "partition_status": status,
                "formal_eligible": bool(row.get("formal_eligible", False)),
                "audit_feature_count": row.get(
                    "audited_lncrna_feature_universe_count"
                ),
                "fresh_direct_id_feature_count": row.get(
                    "fresh_direct_id_lncrna_feature_count"
                ),
                "old_contract_sha256": original_contract,
                "new_contract_sha256": row["contract_sha256"],
                "source_partition_root": row["source_partition_root"],
                "data_partition_recomputed": False,
                "control_semantics_corrected": True,
            }
        )
    return corrected, pd.DataFrame(lineage_rows)


def supersede_controls(
    *,
    source_root: str | Path,
    output_root: str | Path,
    run_id: str,
    expected_hashes: Mapping[str, str] = SOURCE_HASHES,
    require_full_authority: bool = True,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise SupersessionError(f"Supersession output reuse is forbidden: {output}")
    if source == output or source in output.parents:
        raise SupersessionError("Supersession output must not be inside immutable r5")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise SupersessionError(f"Temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        run, gate, handoff, source_delta, source_manifest = _validate_source(
            source, expected_hashes, require_full_authority=require_full_authority
        )
        scope = list(map(str, run["scope"]))
        corrected_records, status_lineage = _correct_status_records(run, source)

        audit = source_delta[
            ["cancer_id", "audit_symbol_first_lncrna_feature_count"]
        ].rename(
            columns={
                "audit_symbol_first_lncrna_feature_count": (
                    "canonical_lnc_feature_unique"
                )
            }
        )
        audit["cancer_id"] = audit.cancer_id.astype(str)
        audit = audit.set_index("cancer_id")
        corrected_delta = build_feature_count_delta_table(
            scope, corrected_records, audit
        )
        blocked_mask = corrected_delta.partition_status.astype(str).str.startswith(
            "BLOCKED_"
        )
        unavailable_columns = [
            "fresh_direct_id_lncrna_feature_count",
            "fresh_minus_audit",
            "allowed_delta_max_25_or_2pct",
            "within_delta_gate",
        ]
        if not corrected_delta.loc[blocked_mask, unavailable_columns].isna().all().all():
            raise SupersessionError("Blocked fresh-count fields are not all typed unavailable")
        success_mask = corrected_delta.partition_status.astype(str).str.startswith(
            "SUCCESS_PARTITION_BUILT"
        )
        if corrected_delta.loc[success_mask, "fresh_direct_id_lncrna_feature_count"].isna().any():
            raise SupersessionError("A successfully rebuilt partition lost its fresh count")

        corrected_manifest = source_manifest.copy()
        corrected_manifest["cancer_id"] = corrected_manifest.cancer_id.astype(str)
        fresh_map = {
            cancer: corrected_records[cancer].get(
                "fresh_direct_id_lncrna_feature_count"
            )
            for cancer in scope
        }
        audit_map = {
            cancer: corrected_records[cancer].get(
                "audited_lncrna_feature_universe_count"
            )
            for cancer in scope
        }
        semantics_map = {
            cancer: corrected_records[cancer][
                "lncrna_feature_universe_count_semantics"
            ]
            for cancer in scope
        }
        corrected_manifest["audited_lncrna_feature_universe_count"] = (
            corrected_manifest.cancer_id.map(audit_map).astype("Int64")
        )
        corrected_manifest["fresh_direct_id_lncrna_feature_count"] = (
            corrected_manifest.cancer_id.map(fresh_map).astype("Int64")
        )
        corrected_manifest["lncrna_feature_universe_count_semantics"] = (
            corrected_manifest.cancer_id.map(semantics_map)
        )
        corrected_manifest["control_supersession_run_id"] = str(run_id)

        status_path = temporary / "RUN_STATUS.json"
        delta_path = temporary / "FEATURE_COUNT_DELTA.tsv"
        manifest_path = temporary / "dataset_manifest_33c.parquet"
        status_lineage_path = temporary / "PARTITION_STATUS_SUPERSESSION.parquet"

        corrected_run = copy.deepcopy(run)
        corrected_run.update(
            {
                "format": STATUS_FORMAT,
                "run_id": str(run_id),
                "per_cancer": corrected_records,
                "supersedes_run_id": run.get("run_id"),
                "supersedes_run_status_path": str(source / "RUN_STATUS.json"),
                "supersedes_run_status_sha256": expected_hashes["RUN_STATUS.json"],
                "control_semantics_correction_only": True,
                "data_partitions_recomputed": False,
                "blocked_fresh_direct_id_counts_typed_unavailable": True,
                "source_data_partitions_remain_immutable": True,
            }
        )
        _attach_contract(corrected_run, "status_contract_sha256")
        _atomic_json(status_path, corrected_run)
        _atomic_tsv(delta_path, corrected_delta)
        _atomic_parquet(manifest_path, corrected_manifest)
        _atomic_parquet(status_lineage_path, status_lineage)

        handoff_path = temporary / "TRAINING_HANDOFF.json"
        corrected_handoff = copy.deepcopy(handoff)
        corrected_handoff.update(
            {
                "format": HANDOFF_FORMAT,
                "run_id": str(run_id),
                "supersedes_run_id": handoff.get("run_id"),
                "supersedes_handoff_path": str(
                    source / "training_handoff" / "TRAINING_HANDOFF.json"
                ),
                "supersedes_handoff_sha256": expected_hashes[
                    "training_handoff/TRAINING_HANDOFF.json"
                ],
                "dataset_manifest_path": str(output / "dataset_manifest_33c.parquet"),
                "dataset_manifest_sha256": artifact_sha256(manifest_path),
                "feature_count_delta_path": str(output / "FEATURE_COUNT_DELTA.tsv"),
                "feature_count_delta_sha256": artifact_sha256(delta_path),
                "partition_status_supersession_path": str(
                    output / "PARTITION_STATUS_SUPERSESSION.parquet"
                ),
                "partition_status_supersession_sha256": artifact_sha256(
                    status_lineage_path
                ),
                "assets_reused_by_reference_from_fresh_r5_partitions": True,
                "data_partitions_recomputed": False,
                "blocked_fresh_direct_id_counts_typed_unavailable": True,
                "fresh_ucell_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
                "fresh_pseudotime_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
                "single_cell_module_complete": False,
                "training_started": False,
                "release_ready": False,
                "production_deployed": False,
            }
        )
        _attach_contract(corrected_handoff, "handoff_contract_sha256")
        _atomic_json(handoff_path, corrected_handoff)

        gate_path = temporary / "FORMAL_33C_INPUT_GATE.json"
        corrected_gate = copy.deepcopy(gate)
        corrected_gate.update(
            {
                "format": GATE_FORMAT,
                "run_id": str(run_id),
                "supersedes_run_id": gate.get("run_id"),
                "supersedes_gate_path": str(
                    source / "FORMAL_33C_INPUT_GATE.json"
                ),
                "supersedes_gate_sha256": expected_hashes[
                    "FORMAL_33C_INPUT_GATE.json"
                ],
                "feature_count_delta_path": str(output / "FEATURE_COUNT_DELTA.tsv"),
                "feature_count_delta_sha256": artifact_sha256(delta_path),
                "training_handoff_path": str(output / "TRAINING_HANDOFF.json"),
                "training_handoff_sha256": artifact_sha256(handoff_path),
                "dataset_manifest_path": str(output / "dataset_manifest_33c.parquet"),
                "dataset_manifest_sha256": artifact_sha256(manifest_path),
                "run_status_path": str(output / "RUN_STATUS.json"),
                "run_status_sha256": artifact_sha256(status_path),
                "partition_status_supersession_path": str(
                    output / "PARTITION_STATUS_SUPERSESSION.parquet"
                ),
                "partition_status_supersession_sha256": artifact_sha256(
                    status_lineage_path
                ),
                "blocked_fresh_direct_id_counts_typed_unavailable": True,
                "control_semantics_correction_only": True,
                "data_partitions_recomputed": False,
                "fresh_ucell_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
                "fresh_pseudotime_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
                "single_cell_module_complete": False,
                "training_started": False,
                "release_ready": False,
                "production_deployed": False,
            }
        )
        _attach_contract(corrected_gate, "gate_contract_sha256")
        _atomic_json(gate_path, corrected_gate)

        new_paths = {
            "RUN_STATUS.json": status_path,
            "FORMAL_33C_INPUT_GATE.json": gate_path,
            "FEATURE_COUNT_DELTA.tsv": delta_path,
            "TRAINING_HANDOFF.json": handoff_path,
            "dataset_manifest_33c.parquet": manifest_path,
            "PARTITION_STATUS_SUPERSESSION.parquet": status_lineage_path,
        }
        supersession = {
            "format": SUPERSESSION_FORMAT,
            "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
            "run_id": str(run_id),
            "reason": "BLOCKED_AUDIT_COUNT_WAS_MISLABELLED_AS_FRESH_DIRECT_ID_COUNT",
            "source_root": str(source),
            "output_root": str(output),
            "old_run_id": run.get("run_id"),
            "old_controls": {
                relative: {
                    "path": str(source / relative),
                    "sha256": expected,
                }
                for relative, expected in expected_hashes.items()
            },
            "new_controls": {
                name: {"path": str(output / name), "sha256": artifact_sha256(path)}
                for name, path in new_paths.items()
            },
            "blocked_cancers": sorted(
                corrected_delta.loc[blocked_mask, "cancer_id"].astype(str)
            ),
            "blocked_count": int(blocked_mask.sum()),
            "success_count": int(success_mask.sum()),
            "failed_count": int(corrected_run.get("failed_count", 0)),
            "pending_count": int(corrected_run.get("pending_count", 0)),
            "blocked_fresh_direct_id_counts_all_unavailable": True,
            "successful_fresh_direct_id_counts_all_present": True,
            "source_data_partitions_remain_immutable": True,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "historical_sc_trajectory_used": False,
            "fresh_ucell_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
            "fresh_pseudotime_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
            "single_cell_module_complete": False,
            "release_ready": False,
            "production_deployed": False,
        }
        _attach_contract(supersession, "supersession_contract_sha256")
        _atomic_json(temporary / "SUPERSESSION.json", supersession)

        sha_rows = [
            {"artifact": path.name, "sha256": artifact_sha256(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        _atomic_tsv(temporary / "CONTROL_SHA256.tsv", pd.DataFrame(sha_rows))
        os.replace(temporary, output)
        return supersession
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = supersede_controls(
        source_root=args.source_root,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
