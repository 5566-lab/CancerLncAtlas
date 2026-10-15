#!/usr/bin/env python3
"""Fail-closed validation for fold-local metacell materialization."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import pandas as pd


MATERIALIZATION_FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_FOLD_LOCAL_V1"
PREFLIGHT_FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_LAUNCH_PREFLIGHT_V1"
VALIDATION_FORMAT = "CANCERLNCATLAS_V32_SC_METACELL_MATERIALIZATION_VALIDATION_V1"
EXPECTED_FULL_CANCERS = 28
DEFAULT_ACTIVE_STATUS = Path(
    "/dell_2/DSC/CancerLncAtlas/runtime/"
    "single_cell_cell_level_r11_rescued_20260903_r1/RUN_STATUS.json"
)
DEFAULT_REPAIR_ROOT = Path("/dsk2/user/dengsc/sc_preprocessing_repair_20261004_r1")
REQUIRED_JOB_FILES = (
    "RUN_STATUS.json",
    "SERVER_PREFLIGHT.json",
    "TRAINING_HANDOFF.json",
    "dataset_manifest_33c.parquet",
)
REQUIRED_FOLD_FILES = (
    "CONTEXT_COVERAGE.json",
    "GEOMETRY_AUDIT.json",
    "METACELL_MEMBERSHIP.parquet",
    "METACELL_METADATA.parquet",
    "QUERY_ROWS.parquet",
    "TRAIN_ROWS.parquet",
    "UNIVERSE.parquet",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def verify_file_record(
    record: dict,
    expected_path: Path,
    *,
    label: str,
) -> dict[str, object]:
    """Verify that a receipt binds the exact intended file, size, and digest."""
    if not isinstance(record, dict):
        raise RuntimeError(f"{label}: file record is not an object")
    path = Path(str(record.get("path", "")))
    if _resolved(path) != _resolved(expected_path):
        raise RuntimeError(
            f"{label}: receipt path drift: {path} != {expected_path}"
        )
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label}: artifact is missing or symlinked: {path}")
    observed_bytes = path.stat().st_size
    if int(record.get("bytes", -1)) != observed_bytes:
        raise RuntimeError(f"{label}: byte count drift")
    observed_sha = sha256(path)
    if str(record.get("sha256", "")) != observed_sha:
        raise RuntimeError(f"{label}: file hash drift")
    return {"path": str(path), "bytes": observed_bytes, "sha256": observed_sha}


def verify_receipt_files(
    receipt: dict,
    base: Path,
    *,
    label: str,
    required: tuple[str, ...] = (),
) -> dict[str, dict[str, object]]:
    records = receipt.get("files")
    if not isinstance(records, dict):
        raise RuntimeError(f"{label}: receipt lacks file records")
    missing = sorted(set(required) - set(records))
    if missing:
        raise RuntimeError(f"{label}: receipt lacks required files {missing}")
    verified: dict[str, dict[str, object]] = {}
    for name, record in sorted(records.items()):
        if not name or Path(name).name != name:
            raise RuntimeError(f"{label}: unsafe receipt file key {name!r}")
        verified[name] = verify_file_record(
            record, base / name, label=f"{label}/{name}"
        )
    return verified


def authorized_cancers(
    active_status_path: Path,
    repair_root: Path,
    *,
    require_full: bool,
) -> tuple[list[str], dict[str, object]]:
    """Derive authority from active per-cancer flags intersected with complete jobs."""
    active = read_json(active_status_path, "active RUN_STATUS")
    per_cancer = active.get("per_cancer")
    if not isinstance(per_cancer, dict):
        raise RuntimeError("active RUN_STATUS lacks per_cancer authority")
    active_formal = {
        str(cancer).upper()
        for cancer, record in per_cancer.items()
        if isinstance(record, dict) and record.get("formal_eligible") is True
    }
    jobs_root = repair_root / "jobs"
    if not jobs_root.is_dir():
        raise RuntimeError(f"repair jobs root is absent: {jobs_root}")
    job_directories = {
        path.name.upper(): path
        for path in jobs_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    complete_jobs = {
        cancer
        for cancer, path in job_directories.items()
        if all((path / name).is_file() for name in REQUIRED_JOB_FILES)
    }
    incomplete_formal_jobs = sorted(
        cancer
        for cancer in active_formal & set(job_directories)
        if cancer not in complete_jobs
    )
    expected = sorted(active_formal & complete_jobs)
    if incomplete_formal_jobs:
        raise RuntimeError(
            "formal repair jobs are incomplete: " + ", ".join(incomplete_formal_jobs)
        )
    if require_full and len(expected) != EXPECTED_FULL_CANCERS:
        raise RuntimeError(
            f"full authority closure requires {EXPECTED_FULL_CANCERS} cancers, "
            f"observed {len(expected)}"
        )
    evidence = {
        "active_run_status_path": str(active_status_path),
        "active_run_status_sha256": sha256(active_status_path),
        "repair_root": str(repair_root),
        "active_formal_cancers": sorted(active_formal),
        "complete_repair_jobs": sorted(complete_jobs),
        "authorized_cancers": expected,
        "authorized_cancer_count": len(expected),
        "authority_rule": (
            "ACTIVE_RUN_STATUS_PER_CANCER_FORMAL_ELIGIBLE_INTERSECT_"
            "COMPLETE_REPAIR_JOBS"
        ),
    }
    return expected, evidence


def validate_launch_preflight(
    output_root: Path,
    core_root: Path,
    candidate_path: Path | None = None,
) -> dict[str, object]:
    preflight_path = output_root / "LAUNCH_PREFLIGHT.json"
    preflight = read_json(preflight_path, "launch preflight")
    if preflight.get("format") != PREFLIGHT_FORMAT:
        raise RuntimeError("launch preflight format drift")
    if (
        preflight.get("target_host") != "149"
        or preflight.get("cpu_only") is not True
        or preflight.get("paid_compute") is not False
    ):
        raise RuntimeError("launch preflight compute contract drift")
    if _resolved(Path(str(preflight.get("output_root", "")))) != _resolved(output_root):
        raise RuntimeError("launch preflight output root drift")
    if preflight.get("old_result_overwritten") is not False:
        raise RuntimeError("launch preflight does not preserve old results")
    if preflight.get("old_result_backed_up") is not False:
        raise RuntimeError("launch preflight backup policy drift")

    binding_path = Path(str(preflight.get("preprocessing_binding_path", "")))
    candidate_declared = Path(str(preflight.get("candidate_path", "")))
    core_manifest = Path(str(preflight.get("core_manifest_path", "")))
    if candidate_path is not None and _resolved(candidate_declared) != _resolved(candidate_path):
        raise RuntimeError("candidate path differs from launch preflight")
    expected_core_manifest = core_root / "CORE_EMBEDDING_MANIFEST.json"
    if _resolved(core_manifest) != _resolved(expected_core_manifest):
        raise RuntimeError("core manifest path differs from requested core root")
    bindings = {
        "preprocessing_binding": (
            binding_path,
            str(preflight.get("preprocessing_binding_sha256", "")),
        ),
        "candidate_universe": (
            candidate_declared,
            str(preflight.get("candidate_sha256", "")),
        ),
        "core_manifest": (
            core_manifest,
            str(preflight.get("core_manifest_sha256", "")),
        ),
    }
    verified: dict[str, object] = {}
    for label, (path, declared_sha) in bindings.items():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"preflight-bound {label} is missing or symlinked: {path}")
        observed_sha = sha256(path)
        if not declared_sha or observed_sha != declared_sha:
            raise RuntimeError(f"preflight-bound {label} hash drift")
        verified[label] = {
            "path": str(path),
            "sha256": observed_sha,
            "bytes": path.stat().st_size,
        }
    return {
        "path": str(preflight_path),
        "sha256": sha256(preflight_path),
        "format": preflight["format"],
        "run_id": preflight.get("run_id"),
        "target_host": preflight["target_host"],
        "cpu_only": preflight["cpu_only"],
        "paid_compute": preflight["paid_compute"],
        "formal_pointer_update_authorized": preflight.get(
            "formal_pointer_update_authorized"
        ),
        "bindings": verified,
    }


def validation_roots(
    output_root: Path,
    expected: list[str],
    cancer: str | None,
) -> list[Path]:
    cancers_root = output_root / "cancers"
    if not cancers_root.is_dir():
        raise RuntimeError(f"materialization cancer root is absent: {cancers_root}")
    expected_set = set(expected)
    if cancer:
        selected = cancer.upper()
        if selected not in expected_set:
            raise RuntimeError(f"pilot cancer is not authorized: {selected}")
        root = cancers_root / selected
        if not (root / "SUCCESS.json").is_file():
            raise RuntimeError(f"pilot cancer is incomplete: {selected}")
        return [root]
    actual_directories = {
        path.name
        for path in cancers_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    missing = sorted(expected_set - actual_directories)
    extra = sorted(actual_directories - expected_set)
    incomplete = sorted(
        cancer_id
        for cancer_id in expected_set & actual_directories
        if not (cancers_root / cancer_id / "SUCCESS.json").is_file()
    )
    if missing or extra or incomplete:
        raise RuntimeError(
            "full cancer directory closure failed: "
            f"missing={missing}, extra={extra}, incomplete={incomplete}"
        )
    return [cancers_root / cancer_id for cancer_id in expected]


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalize_lnc(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    for prefix in ("LNCRNA:", "LNC:"):
        if upper.startswith(prefix):
            return text[len(prefix):]
    return text


def core_lnc_ids(core_root: Path) -> set[str]:
    values = None
    for fold in range(5):
        frame = pd.read_parquet(core_root / f"patient_fold={fold}" / "lncRNA.parquet")
        column = "node_id" if "node_id" in frame else "lncrna_id"
        local = {normalize_lnc(value) for value in frame[column]}
        values = local if values is None else values & local
    return values or set()


def validate_cancer(root: Path, core_lncs: set[str]) -> tuple[dict, pd.DataFrame]:
    success_path = root / "SUCCESS.json"
    receipt = read_json(success_path, f"{root.name} cancer receipt")
    cancer_id = root.name
    if (
        receipt.get("format") != MATERIALIZATION_FORMAT
        or receipt.get("target_host") != "149"
        or receipt.get("state") != "CANCER_METACELL_OOF_INPUTS_COMPLETE"
        or receipt.get("cancer_id") != cancer_id
    ):
        raise RuntimeError(f"{cancer_id}: cancer receipt identity drift")
    verified_cancer_files = verify_receipt_files(
        receipt,
        root,
        label=f"{cancer_id} cancer receipt",
        required=("DONOR_FOLDS.parquet",),
    )
    folds = pd.read_parquet(root / "DONOR_FOLDS.parquet")
    required_fold_columns = {
        "cancer_id", "donor_id", "outer_fold_id", "cohort_fold_count",
        "limited_replication",
    }
    if missing := sorted(required_fold_columns - set(folds.columns)):
        raise RuntimeError(f"{cancer_id}: donor folds lack columns {missing}")
    if folds.empty or folds.donor_id.astype(str).duplicated().any():
        raise RuntimeError(f"{cancer_id}: donor fold map is empty or duplicated")
    if not folds.cancer_id.astype(str).eq(cancer_id).all():
        raise RuntimeError(f"{cancer_id}: donor fold cancer identity drift")
    donor_to_fold = dict(zip(folds.donor_id.astype(str), folds.outer_fold_id.astype(int)))
    expected_active_folds = list(range(min(5, len(donor_to_fold))))
    observed_active_folds = sorted(set(donor_to_fold.values()))
    if observed_active_folds != expected_active_folds:
        raise RuntimeError(f"{cancer_id}: active fold closure drift")
    if not folds.cohort_fold_count.astype(int).eq(len(expected_active_folds)).all():
        raise RuntimeError(f"{cancer_id}: cohort fold count drift")
    limited_replication = len(donor_to_fold) < 5
    if not folds.limited_replication.astype(bool).eq(limited_replication).all():
        raise RuntimeError(f"{cancer_id}: limited-replication flag drift")
    if int(receipt.get("donors", -1)) != len(donor_to_fold):
        raise RuntimeError(f"{cancer_id}: receipt donor count drift")
    if list(map(int, receipt.get("active_outer_folds", []))) != expected_active_folds:
        raise RuntimeError(f"{cancer_id}: receipt active folds drift")
    if receipt.get("limited_replication") is not limited_replication:
        raise RuntimeError(f"{cancer_id}: receipt replication flag drift")

    fold_directories: dict[int, Path] = {}
    for path in root.glob("outer_fold=*"):
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            fold = int(path.name.split("=", 1)[1])
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"{cancer_id}: invalid fold directory {path.name}") from error
        if fold in fold_directories:
            raise RuntimeError(f"{cancer_id}: duplicate fold directory {fold}")
        fold_directories[fold] = path
    if sorted(fold_directories) != expected_active_folds:
        raise RuntimeError(f"{cancer_id}: fold directory closure drift")
    declared_fold_hashes = receipt.get("fold_receipt_sha256")
    if not isinstance(declared_fold_hashes, dict) or set(declared_fold_hashes) != {
        str(fold) for fold in expected_active_folds
    }:
        raise RuntimeError(f"{cancer_id}: fold receipt hash closure drift")

    query_parts = []
    universe_parts = []
    fold_reports = []
    fold_success_hashes: dict[str, str] = {}
    membership_cell_ids: set[str] | None = None
    for fold in expected_active_folds:
        fold_dir = fold_directories[fold]
        fold_success = fold_dir / "SUCCESS.json"
        fold_success_sha = sha256(fold_success)
        if str(declared_fold_hashes[str(fold)]) != fold_success_sha:
            raise RuntimeError(f"{cancer_id} fold {fold}: SUCCESS hash drift")
        fold_success_hashes[str(fold)] = fold_success_sha
        fold_receipt = read_json(fold_success, f"{cancer_id} fold {fold} receipt")
        if (
            fold_receipt.get("format") != MATERIALIZATION_FORMAT
            or fold_receipt.get("target_host") != "149"
            or fold_receipt.get("state") != "FOLD_LOCAL_METACELL_TARGETS_READY"
            or fold_receipt.get("cancer_id") != cancer_id
            or int(fold_receipt.get("outer_fold_id", -1)) != fold
        ):
            raise RuntimeError(f"{cancer_id} fold {fold}: receipt identity drift")
        if (
            fold_receipt.get("target_policy")
            != "SIGNED_WITHIN_DONOR_CELLTYPE_METACELL_SPEARMAN_RHO"
            or fold_receipt.get("signed_rho_preserved") is not True
            or fold_receipt.get("query_donor_used_for_geometry_fit") is not False
            or fold_receipt.get("query_donor_used_for_candidate_selection") is not False
        ):
            raise RuntimeError(f"{cancer_id} fold {fold}: scientific receipt drift")
        source = set(map(str, fold_receipt["source_donors"]))
        query = set(map(str, fold_receipt["query_donors"]))
        if source & query or source | query != set(donor_to_fold):
            raise RuntimeError(f"{cancer_id} fold {fold}: donor partition drift")
        if query != {donor for donor, assigned in donor_to_fold.items() if assigned == fold}:
            raise RuntimeError(f"{cancer_id} fold {fold}: query assignment drift")
        if canonical_hash(sorted(source)) != fold_receipt.get("source_donor_hash"):
            raise RuntimeError(f"{cancer_id} fold {fold}: source donor hash drift")
        if canonical_hash(sorted(query)) != fold_receipt.get("query_donor_hash"):
            raise RuntimeError(f"{cancer_id} fold {fold}: query donor hash drift")
        verified_fold_files = verify_receipt_files(
            fold_receipt,
            fold_dir,
            label=f"{cancer_id} fold {fold} receipt",
            required=REQUIRED_FOLD_FILES,
        )
        membership = pd.read_parquet(fold_dir / "METACELL_MEMBERSHIP.parquet")
        required_membership = {
            "cell_id", "donor_id", "cell_type", "state_id", "metacell_id",
        }
        if missing := sorted(required_membership - set(membership.columns)):
            raise RuntimeError(
                f"{cancer_id} fold {fold}: membership lacks columns {missing}"
            )
        if membership.cell_id.astype(str).duplicated().any():
            raise RuntimeError(f"{cancer_id} fold {fold}: duplicate input cell")
        local_cell_ids = set(membership.cell_id.astype(str))
        if membership_cell_ids is None:
            membership_cell_ids = local_cell_ids
        elif membership_cell_ids != local_cell_ids:
            raise RuntimeError(f"{cancer_id} fold {fold}: eligible cell closure drift")
        if len(membership) != int(receipt.get("eligible_cells", -1)):
            raise RuntimeError(f"{cancer_id} fold {fold}: eligible cell count drift")
        if set(membership.donor_id.astype(str)) != set(donor_to_fold):
            raise RuntimeError(f"{cancer_id} fold {fold}: membership donor closure drift")
        used = membership.loc[membership.metacell_id.ge(0)].copy()
        grouped = used.groupby("metacell_id", observed=True).agg(
            donor_count=("donor_id", "nunique"),
            celltype_count=("cell_type", "nunique"),
            state_count=("state_id", "nunique"),
            cells=("cell_id", "size"),
        )
        if len(grouped) and (
            grouped[["donor_count", "celltype_count", "state_count"]].ne(1).any().any()
            or grouped.cells.lt(3).any()
            or grouped.cells.gt(5).any()
        ):
            raise RuntimeError(f"{cancer_id} fold {fold}: invalid metacell membership")
        metacell_metadata = pd.read_parquet(fold_dir / "METACELL_METADATA.parquet")
        if (
            metacell_metadata.metacell_id.duplicated().any()
            or set(metacell_metadata.metacell_id.astype(int))
            != set(map(int, grouped.index))
            or len(metacell_metadata) != int(fold_receipt.get("metacells", -1))
        ):
            raise RuntimeError(f"{cancer_id} fold {fold}: metacell metadata closure drift")
        expected_sizes = grouped.cells.astype(int).to_dict()
        observed_sizes = dict(
            zip(
                metacell_metadata.metacell_id.astype(int),
                metacell_metadata.n_cells.astype(int),
            )
        )
        if expected_sizes != observed_sizes:
            raise RuntimeError(f"{cancer_id} fold {fold}: metacell size drift")
        geometry = read_json(
            fold_dir / "GEOMETRY_AUDIT.json", f"{cancer_id} fold {fold} geometry audit"
        )
        if (
            set(map(str, geometry.get("training_donors", []))) != source
            or geometry.get("lncrna_targets_used") is not False
            or int(geometry.get("overlapping_cells", -1)) != 0
        ):
            raise RuntimeError(f"{cancer_id} fold {fold}: geometry leakage contract drift")
        query_rows = pd.read_parquet(fold_dir / "QUERY_ROWS.parquet")
        universe = pd.read_parquet(fold_dir / "UNIVERSE.parquet")
        if len(query_rows):
            if set(query_rows.query_donor_id.astype(str)) - query:
                raise RuntimeError(f"{cancer_id} fold {fold}: source donor in query rows")
            key = ["cell_key", "query_donor_id", "lncrna_id", "pathway_id"]
            if query_rows.duplicated(key).any():
                raise RuntimeError(f"{cancer_id} fold {fold}: duplicate query target")
            if not query_rows.outer_fold_id.astype(int).eq(fold).all():
                raise RuntimeError(f"{cancer_id} fold {fold}: row fold drift")
            if not np.isfinite(query_rows[["rho", "association_target"]].to_numpy(float)).all():
                raise RuntimeError(f"{cancer_id} fold {fold}: non-finite target")
            if not np.allclose(
                query_rows.association_target,
                query_rows.rho.abs(),
                rtol=0.0,
                atol=1e-7,
            ):
                raise RuntimeError(f"{cancer_id} fold {fold}: abs target drift")
            if query_rows.rho.abs().gt(1.000001).any():
                raise RuntimeError(f"{cancer_id} fold {fold}: rho outside domain")
            if query_rows.candidate_selection_query_donor_used.astype(bool).any():
                raise RuntimeError(f"{cancer_id} fold {fold}: query selection leakage")
            if not query_rows.cancer_id.astype(str).eq(cancer_id).all():
                raise RuntimeError(f"{cancer_id} fold {fold}: query cancer drift")
            query_parts.append(query_rows)
        if len(universe):
            universe_key = ["cell_key", "lncrna_id", "pathway_id"]
            if universe.duplicated(universe_key).any():
                raise RuntimeError(f"{cancer_id} fold {fold}: duplicate universe pair")
            if (
                not universe.outer_fold_id.astype(int).eq(fold).all()
                or not universe.cancer_id.astype(str).eq(cancer_id).all()
                or universe.candidate_selection_query_donor_used.astype(bool).any()
            ):
                raise RuntimeError(f"{cancer_id} fold {fold}: universe identity drift")
            universe_parts.append(universe)
        if len(query_rows) != int(fold_receipt.get("query_rows", -1)):
            raise RuntimeError(f"{cancer_id} fold {fold}: query row count drift")
        if len(universe) != int(fold_receipt.get("universe_rows", -1)):
            raise RuntimeError(f"{cancer_id} fold {fold}: universe row count drift")
        fold_reports.append(
            {
                "outer_fold_id": fold,
                "metacells": len(grouped),
                "assigned_cells": len(used),
                "unassigned_cells": int(membership.metacell_id.lt(0).sum()),
                "query_rows": len(query_rows),
                "query_donors_with_rows": int(query_rows.query_donor_id.nunique()) if len(query_rows) else 0,
                "success_sha256": fold_success_sha,
                "verified_file_count": len(verified_fold_files),
            }
        )
    query = pd.concat(query_parts, ignore_index=True) if query_parts else pd.DataFrame()
    universe = pd.concat(universe_parts, ignore_index=True) if universe_parts else pd.DataFrame()
    if len(query):
        key = ["cancer_id", "cell_key", "query_donor_id", "lncrna_id", "pathway_id"]
        if query.duplicated(key).any():
            raise RuntimeError(f"{cancer_id}: query target occurs in more than one fold")
        wrong = query.apply(
            lambda row: donor_to_fold[str(row.query_donor_id)] != int(row.outer_fold_id), axis=1
        )
        if wrong.any():
            raise RuntimeError(f"{cancer_id}: OOF donor/fold mismatch")
    lnc_values = set(map(str, universe.lncrna_id)) if len(universe) else set()
    oov = pd.DataFrame(
        {
            "cancer_id": cancer_id,
            "lncrna_id": sorted(lnc_values),
        }
    )
    oov["core_available_all_folds"] = oov.lncrna_id.isin(core_lncs)
    oov["embedding_route"] = np.where(
        oov.core_available_all_folds,
        "FROZEN_CORE_PLUS_PRIVATE_ID_ADAPTER",
        "SINGLE_CELL_PRIVATE_OOV_ID_EMBEDDING",
    )
    donors_with_rows = set(map(str, query.query_donor_id)) if len(query) else set()
    report = {
        "cancer_id": cancer_id,
        "success_path": str(success_path),
        "success_sha256": sha256(success_path),
        "verified_cancer_files": verified_cancer_files,
        "fold_success_sha256": fold_success_hashes,
        "eligible_donors": len(donor_to_fold),
        "donors_with_oof_rows": len(donors_with_rows),
        "donors_without_estimable_oof_rows": sorted(set(donor_to_fold) - donors_with_rows),
        "query_rows": len(query),
        "query_unique_pairs": int(
            query[["lncrna_id", "pathway_id"]].drop_duplicates().shape[0]
        ) if len(query) else 0,
        "new_query_rows": int(
            ((~query.is_bulk_candidate) & query.is_sc_discovery_candidate).sum()
        ) if len(query) else 0,
        "new_unique_pairs": int(
            query.loc[
                (~query.is_bulk_candidate) & query.is_sc_discovery_candidate,
                ["lncrna_id", "pathway_id"],
            ].drop_duplicates().shape[0]
        ) if len(query) else 0,
        "lncrnas_in_universe": len(lnc_values),
        "oov_lncrnas_in_universe": int((~oov.core_available_all_folds).sum()),
        "folds": fold_reports,
    }
    return report, oov


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--core-root", required=True, type=Path)
    parser.add_argument("--candidate-path", type=Path)
    parser.add_argument("--active-status", type=Path, default=DEFAULT_ACTIVE_STATUS)
    parser.add_argument("--repair-root", type=Path, default=DEFAULT_REPAIR_ROOT)
    parser.add_argument("--cancer")
    args = parser.parse_args()
    if socket.gethostname() != "149":
        raise SystemExit("WRONG_HOST")
    args.output_root = _resolved(args.output_root)
    args.core_root = _resolved(args.core_root)
    args.active_status = _resolved(args.active_status)
    args.repair_root = _resolved(args.repair_root)
    preflight_evidence = validate_launch_preflight(
        args.output_root, args.core_root, args.candidate_path
    )
    expected, authority_evidence = authorized_cancers(
        args.active_status,
        args.repair_root,
        require_full=args.cancer is None,
    )
    core_lncs = core_lnc_ids(args.core_root)
    roots = validation_roots(args.output_root, expected, args.cancer)
    reports = []
    oov_parts = []
    for root in roots:
        report, oov = validate_cancer(root, core_lncs)
        reports.append(report)
        oov_parts.append(oov)
    oov = pd.concat(oov_parts, ignore_index=True).drop_duplicates()
    suffix = f".{args.cancer.upper()}" if args.cancer else ""
    oov_path = args.output_root / f"OOV_AUDIT{suffix}.parquet"
    oov.to_parquet(oov_path, index=False)
    full_closure = args.cancer is None
    result = {
        "format": VALIDATION_FORMAT,
        "target_host": "149",
        "status": "PASS",
        "closure_status": (
            "PASS_FULL_28_CANCER_MATERIALIZATION_CLOSURE"
            if full_closure
            else "PASS_SINGLE_CANCER_PILOT_VALIDATION"
        ),
        "validation_scope": "FULL" if full_closure else "SINGLE_CANCER_PILOT",
        "fail_closed": True,
        "expected_cancer_count": len(expected),
        "validated_cancer_count": len(reports),
        "expected_cancers": expected,
        "validated_cancers": [report["cancer_id"] for report in reports],
        "authority": authority_evidence,
        "launch_preflight": preflight_evidence,
        "validator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "cancers": reports,
        "core_lncrnas": len(core_lncs),
        "oov_audit": {"path": str(oov_path), "sha256": sha256(oov_path), "rows": len(oov)},
    }
    result_path = args.output_root / f"MATERIALIZATION_VALIDATION{suffix}.json"
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
