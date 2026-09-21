#!/usr/bin/env python3
"""Lightweight server preflight for the portable single-cell r7 authority.

This command verifies copy/control/authority hashes and checks the formal-17
10x H5 plus metadata schemas.  It reads barcodes and the five required metadata
columns to prove one-to-one alignment, but never reads the expression ``data``
or ``indices`` arrays and never calls UCell or pseudotime code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_METADATA = (
    "cell_id",
    "patient_id",
    "cell_type_major",
    "dataset_id",
    "cancer_id",
)
DESTINATION_ROOT = (
    "./data/CancerLncAtlas/runtime/"
    "single_cell_cell_level_r7_portable_supersession_20260829"
)
DERIVED_ASSET_FAMILIES = (
    "activity",
    "association",
    "exact_pathway_availability",
    "expression_facts",
    "lnc_celltype",
)
EXACT_MEMBERSHIP_SHA256 = (
    "22b6215920f58b13d5e80f8fe93f6964ec0ea0fc14288c64aa758509e4ace975"
)
FORMAL_17 = (
    "ACC",
    "CHOL",
    "DLBC",
    "ESCA",
    "GBM",
    "HNSC",
    "KIRC",
    "LAML",
    "LGG",
    "LUSC",
    "MESO",
    "PCPG",
    "READ",
    "SARC",
    "SKCM",
    "THYM",
    "UCEC",
)
_HASH_CHUNK = 4 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SESSION = re.compile(r"^[0-9a-f]{32}$")


class ServerPreflightError(RuntimeError):
    """Raised when a portable authority or raw-input gate fails closed."""


def artifact_sha256(path: str | Path) -> str:
    """Hash a regular file without relying on the project package."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ServerPreflightError(f"Cannot hash missing/unsafe file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while block := stream.read(_HASH_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _decode(values: Any) -> list[str]:
    """Decode HDF5 byte strings without importing CC-HHGT internals."""

    return [
        item.decode("utf-8", errors="replace")
        if isinstance(item, (bytes, np.bytes_))
        else str(item)
        for item in values
    ]


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _assert_no_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise ServerPreflightError(f"r7 root cannot be a symlink: {root}")
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*names, *files]:
            path = base / name
            if path.is_symlink():
                raise ServerPreflightError(f"Symlink is forbidden in r7: {path}")


def _portable_relative(portable_path: str) -> Path:
    prefix = DESTINATION_ROOT + "/"
    normalized = str(portable_path).replace("\\", "/")
    if not normalized.startswith(prefix):
        raise ServerPreflightError(f"Authority path escapes fixed r7 root: {portable_path}")
    relative = Path(normalized[len(prefix) :])
    if relative.is_absolute() or ".." in relative.parts:
        raise ServerPreflightError(f"Invalid portable relative path: {portable_path}")
    return relative


def _verify_transfer_residues(
    root: Path,
    expected: dict[str, dict[str, Any]],
    actual_files: set[str],
) -> tuple[int, int, int, int, bool]:
    """Admit only byte-identical ``<final>.partial.<sha>`` transfer residue.

    The transfer helper normally publishes a partial with ``os.link``.  A
    verified target may already exist, however, in which case the helper keeps
    the independently uploaded partial after confirming the existing target's
    digest. Inode identity is therefore recorded rather than used as the
    integrity proof; the exact residue name, byte size, and SHA256 are. A
    separate inode is charged as additional storage.
    """

    finals = set(expected)
    if missing := sorted(finals - actual_files):
        raise ServerPreflightError(f"Bundle final files are missing: {missing}")
    extras = actual_files - finals
    verified_files = 0
    verified_bytes = 0
    hardlink_files = 0
    additional_storage_bytes = 0
    admitted: set[str] = set()
    for relative, record in expected.items():
        residue_relative = f"{relative}.partial.{record['sha256']}"
        if residue_relative not in extras:
            continue
        residue = root / residue_relative
        if residue.is_symlink() or not residue.is_file():
            raise ServerPreflightError(
                f"Transfer residue is absent/non-regular/symlinked: {residue_relative}"
            )
        observed_bytes = int(residue.stat().st_size)
        if observed_bytes != int(record["bytes"]):
            raise ServerPreflightError(
                f"Transfer residue byte-size drift: {residue_relative}"
            )
        if artifact_sha256(residue) != str(record["sha256"]):
            raise ServerPreflightError(f"Transfer residue SHA drift: {residue_relative}")
        final = root / relative
        if os.path.samefile(residue, final):
            hardlink_files += 1
        else:
            additional_storage_bytes += observed_bytes
        admitted.add(residue_relative)
        verified_files += 1
        verified_bytes += observed_bytes
    if unknown := sorted(extras - admitted):
        raise ServerPreflightError(f"Unknown files or malformed transfer residue: {unknown}")
    return (
        verified_files,
        verified_bytes,
        hardlink_files,
        additional_storage_bytes,
        hardlink_files == verified_files,
    )


def _normalized_bridge_entries(copy_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_path": str(record["source_path"]),
            "target_path": str(record["target_path"]).replace("\\", "/"),
            "bytes": int(record["bytes"]),
            "sha256": str(record["sha256"]).lower(),
        }
        for record in copy_manifest.get("entries", [])
    ]


def _verify_transfer_controls(
    root: Path,
    expected_bridge_sha256: str,
    bridge_manifest: dict[str, Any],
) -> tuple[set[str], int, int, int]:
    """Validate the transfer client's exact bootstrap-control tree."""

    transfer_root = root / "_transfer"
    if not transfer_root.exists():
        return set(), 0, 0, 0
    if transfer_root.is_symlink() or not transfer_root.is_dir():
        raise ServerPreflightError("_transfer is not a regular directory")
    if not _SHA256.fullmatch(expected_bridge_sha256):
        raise ServerPreflightError("Expected bridge SHA is malformed")
    expected_entries = _normalized_bridge_entries(bridge_manifest)
    if len(expected_entries) != 13:
        raise ServerPreflightError("Bridge manifest is not the exact 13-entry delivery")
    expected_plan = {
        "format": "CANCERLNCATLAS_V32_REMOTE_TRANSFER_PLAN_V1",
        "source_copy_manifest_sha256": expected_bridge_sha256,
        "target_root": DESTINATION_ROOT,
        "entry_count": 13,
        "total_bytes": sum(row["bytes"] for row in expected_entries),
        "entries": expected_entries,
        "production_deployed": False,
    }

    accepted_files: set[str] = set()
    sessions: dict[str, dict[str, Path]] = {}
    allowed_directories = {
        "_transfer",
        "_transfer/control",
        f"_transfer/control/{expected_bridge_sha256}",
    }
    for path in transfer_root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ServerPreflightError(f"Symlink is forbidden in transfer controls: {relative}")
        parts = Path(relative).parts
        if path.is_dir():
            if len(parts) == 4:
                if (
                    parts[:3] != ("_transfer", "control", expected_bridge_sha256)
                    or not _SESSION.fullmatch(parts[3])
                ):
                    raise ServerPreflightError(f"Unexpected transfer-control directory: {relative}")
                allowed_directories.add(relative)
                sessions.setdefault(parts[3], {})
            elif relative not in allowed_directories:
                raise ServerPreflightError(f"Unexpected transfer-control directory: {relative}")
            continue
        if not path.is_file():
            raise ServerPreflightError(f"Non-regular transfer-control object: {relative}")
        if (
            len(parts) != 5
            or parts[:3] != ("_transfer", "control", expected_bridge_sha256)
            or not _SESSION.fullmatch(parts[3])
        ):
            raise ServerPreflightError(f"Unexpected transfer-control file: {relative}")
        session = parts[3]
        filename = parts[4]
        match = re.fullmatch(
            r"(remote_helper|plan)\.([0-9a-f]{64})\.(py|json)", filename
        )
        if not match:
            raise ServerPreflightError(f"Unexpected transfer-control filename: {relative}")
        role, filename_sha, extension = match.groups()
        if (role, extension) not in {("remote_helper", "py"), ("plan", "json")}:
            raise ServerPreflightError(f"Transfer-control role/extension drift: {relative}")
        observed_sha = artifact_sha256(path)
        if observed_sha != filename_sha:
            raise ServerPreflightError(f"Transfer-control filename SHA drift: {relative}")
        session_files = sessions.setdefault(session, {})
        if role in session_files:
            raise ServerPreflightError(f"Duplicate {role} in transfer session: {session}")
        session_files[role] = path
        accepted_files.add(relative)

    if not sessions:
        raise ServerPreflightError("_transfer exists without a valid control session")
    for session, files in sessions.items():
        if set(files) != {"remote_helper", "plan"}:
            raise ServerPreflightError(
                f"Transfer session lacks exact helper+plan pair: {session} {sorted(files)}"
            )
        plan = _load_json(files["plan"])
        if plan != expected_plan:
            raise ServerPreflightError(f"Transfer plan contract drift: {session}")
    return (
        accepted_files,
        len(sessions),
        len(accepted_files),
        sum(int((root / relative).stat().st_size) for relative in accepted_files),
    )


def verify_bundle_controls(
    root: Path,
    copy_manifest_path: Path,
    expected_copy_manifest_sha256: str,
) -> dict[str, Any]:
    _assert_no_symlinks(root)
    copy_path = copy_manifest_path.resolve()
    observed_copy_sha = artifact_sha256(copy_path)
    if observed_copy_sha != expected_copy_manifest_sha256:
        raise ServerPreflightError(
            f"COPY_MANIFEST SHA drift: {observed_copy_sha} != {expected_copy_manifest_sha256}"
        )
    copy_manifest = _load_json(copy_path)
    if copy_manifest.get("format") != "CANCERLNCATLAS_V32_PORTABLE_COPY_MANIFEST_V1":
        raise ServerPreflightError("COPY_MANIFEST production format mismatch")
    if copy_manifest.get("target_root") != DESTINATION_ROOT:
        raise ServerPreflightError("COPY_MANIFEST destination root drift")
    if copy_manifest.get("symlinks_permitted") is not False:
        raise ServerPreflightError("COPY_MANIFEST does not forbid symlinks")
    if copy_manifest.get("overwrite_permitted") is not False:
        raise ServerPreflightError("COPY_MANIFEST permits destination overwrite")
    if copy_manifest.get("payloads_are_byte_identical") is not True:
        raise ServerPreflightError("COPY_MANIFEST does not require byte identity")

    (
        transfer_control_files,
        transfer_control_sessions,
        transfer_control_file_count,
        transfer_control_bytes,
    ) = _verify_transfer_controls(root, observed_copy_sha, copy_manifest)

    listed: set[str] = set()
    expected_files: dict[str, dict[str, Any]] = {}
    for entry in copy_manifest.get("entries", []):
        if entry.get("kind") != "file":
            raise ServerPreflightError("COPY_MANIFEST contains a non-leaf entry")
        target = str(entry.get("target_path", "")).replace("\\", "/")
        prefix = DESTINATION_ROOT + "/"
        if not target.startswith(prefix):
            raise ServerPreflightError(f"COPY_MANIFEST target escapes r7 root: {target}")
        relative = target[len(prefix) :]
        if not relative or ".." in Path(relative).parts:
            raise ServerPreflightError(f"Invalid COPY_MANIFEST relative path: {relative}")
        if relative in listed:
            raise ServerPreflightError(f"Duplicate copy entry: {relative}")
        listed.add(relative)
        expected_files[relative] = {
            "bytes": int(entry["bytes"]),
            "sha256": str(entry["sha256"]).lower(),
        }
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ServerPreflightError(f"Copied file absent or symlinked: {path}")
        if int(path.stat().st_size) != int(entry["bytes"]):
            raise ServerPreflightError(f"Copied byte-size drift: {relative}")
        if artifact_sha256(path) != entry["sha256"]:
            raise ServerPreflightError(f"Copied SHA drift: {relative}")
        if target != f"{DESTINATION_ROOT}/{relative}":
            raise ServerPreflightError(f"Copied destination drift: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    (
        residue_files,
        residue_bytes,
        residue_hardlink_files,
        residue_additional_storage_bytes,
        residue_all_hardlinks,
    ) = _verify_transfer_residues(root, expected_files, actual - transfer_control_files)

    control_path = root / "CONTROL_SHA256.tsv"
    controls = pd.read_csv(control_path, sep="\t")
    if list(controls.columns) != ["artifact", "sha256"]:
        raise ServerPreflightError("CONTROL_SHA256.tsv schema drift")
    control_seen: set[str] = set()
    for row in controls.itertuples(index=False):
        relative = str(row.artifact)
        if relative in control_seen:
            raise ServerPreflightError(f"Duplicate CONTROL_SHA entry: {relative}")
        control_seen.add(relative)
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ServerPreflightError(f"Controlled file absent or symlinked: {relative}")
        if artifact_sha256(path) != str(row.sha256):
            raise ServerPreflightError(f"CONTROL_SHA drift: {relative}")

    authority = _load_json(root / "PORTABLE_AUTHORITY.json")
    if authority.get("portable_server_root") != DESTINATION_ROOT:
        raise ServerPreflightError("Portable authority root drift")
    authority_results: dict[str, Any] = {}
    for name in ("gencode_gtf", "gencode_annotation", "gencode_provenance", "exact_membership"):
        record = authority[name]
        relative = _portable_relative(record["path"])
        path = root / relative
        observed = artifact_sha256(path)
        if observed != record["sha256"]:
            raise ServerPreflightError(f"Authority SHA drift: {name}")
        authority_results[name] = {
            "relative_path": relative.as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": observed,
        }
    if authority_results["exact_membership"]["sha256"] != EXACT_MEMBERSHIP_SHA256:
        raise ServerPreflightError("Exact membership is not the pinned 2,135-pathway file")

    handoff = _load_json(root / "TRAINING_HANDOFF.json")
    if set(handoff.get("assets", {})) != set(DERIVED_ASSET_FAMILIES):
        raise ServerPreflightError("Derived asset-family set drift")
    if any(asset.get("path") is not None for asset in handoff["assets"].values()):
        raise ServerPreflightError("An unrecomputed derived asset is unexpectedly bound")
    if any(
        asset.get("status") != "UNBOUND_FRESH_RECOMPUTE_REQUIRED"
        for asset in handoff["assets"].values()
    ):
        raise ServerPreflightError("A derived asset lost its fail-closed status")
    return {
        "copy_manifest_sha256": observed_copy_sha,
        "copy_entries_verified": len(listed),
        "verified_transfer_residue_files": residue_files,
        "verified_transfer_residue_bytes": residue_bytes,
        "verified_transfer_residue_hardlink_files": residue_hardlink_files,
        "verified_transfer_residue_independent_files": residue_files
        - residue_hardlink_files,
        "verified_transfer_residue_additional_storage_bytes": (
            residue_additional_storage_bytes
        ),
        "verified_transfer_residue_hardlinks": residue_all_hardlinks,
        "verified_transfer_control_sessions": transfer_control_sessions,
        "verified_transfer_control_files": transfer_control_file_count,
        "verified_transfer_control_bytes": transfer_control_bytes,
        "control_entries_verified": len(control_seen),
        "authority": authority_results,
        "derived_assets_fail_closed": True,
        "symlinks_found": 0,
    }


def audit_raw_cancer(
    cancer: str,
    record: dict[str, Any],
    expected_h5_bytes: int,
    expected_metadata_bytes: int,
    *,
    enforce_authorized_paths: bool = True,
) -> dict[str, Any]:
    h5_path = Path(record["h5_path"])
    metadata_path = Path(record["metadata_path"])
    if enforce_authorized_paths:
        for path, role in ((h5_path, "H5"), (metadata_path, "metadata")):
            normalized = str(path).replace("\\", "/")
            if not normalized.startswith("./data/CancerLncAtlas/"):
                raise ServerPreflightError(f"{cancer} {role} is outside authorized /public8")
    for path, role in ((h5_path, "H5"), (metadata_path, "metadata")):
        if path.is_symlink() or not path.is_file():
            raise ServerPreflightError(f"{cancer} {role} absent or symlinked: {path}")
    if h5_path.stat().st_size != int(expected_h5_bytes):
        raise ServerPreflightError(f"{cancer} H5 byte-size drift")
    if metadata_path.stat().st_size != int(expected_metadata_bytes):
        raise ServerPreflightError(f"{cancer} metadata byte-size drift")
    if artifact_sha256(metadata_path) != record["metadata_sha256"]:
        raise ServerPreflightError(f"{cancer} metadata SHA drift")

    try:
        import h5py
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - server dependency gate
        raise ServerPreflightError("h5py and pyarrow are required") from exc

    parquet = pq.ParquetFile(metadata_path)
    columns = set(parquet.schema_arrow.names)
    missing = sorted(set(REQUIRED_METADATA) - columns)
    if missing:
        raise ServerPreflightError(f"{cancer} metadata lacks columns: {missing}")
    metadata = pd.read_parquet(metadata_path, columns=list(REQUIRED_METADATA))
    if metadata.empty:
        raise ServerPreflightError(f"{cancer} metadata is empty")
    if metadata.cell_id.isna().any() or metadata.cell_id.astype(str).duplicated().any():
        raise ServerPreflightError(f"{cancer} metadata cell IDs are null/duplicated")
    dataset_id = metadata.dataset_id.fillna("").astype(str).str.strip()
    cancer_id = metadata.cancer_id.fillna("").astype(str).str.strip().str.upper()
    if dataset_id.eq("").any() or dataset_id.nunique() != 1:
        raise ServerPreflightError(f"{cancer} metadata does not have one nonempty dataset_id")
    if cancer_id.eq("").any() or set(cancer_id) != {cancer}:
        raise ServerPreflightError(f"{cancer} metadata cancer_id drift")
    # Missing donor/cell-type annotations are an explicit cell-level exclusion,
    # not a whole-dataset schema failure.  This matches the formal partition
    # builder and avoids mistaking a processing gate for absent raw data.
    donor = metadata.patient_id.fillna("").astype(str).str.strip()
    cell_type = metadata.cell_type_major.fillna("").astype(str).str.strip()
    usable_metadata = donor.ne("") & cell_type.ne("")
    if not usable_metadata.any():
        raise ServerPreflightError(f"{cancer} has no cells with donor and cell-type metadata")

    with h5py.File(h5_path, "r") as handle:
        group = handle["matrix"] if "matrix" in handle else handle
        required = {"data", "indices", "indptr", "shape", "barcodes", "features"}
        if not required.issubset(group.keys()):
            raise ServerPreflightError(f"{cancer} H5 is not a 10x sparse matrix")
        features = group["features"]
        if not {"id", "name"}.issubset(features.keys()):
            raise ServerPreflightError(f"{cancer} H5 feature schema lacks id/name")
        shape = tuple(int(value) for value in np.asarray(group["shape"][:]))
        if len(shape) != 2 or min(shape) <= 0:
            raise ServerPreflightError(f"{cancer} H5 shape is invalid: {shape}")
        if group["barcodes"].shape[0] != shape[1]:
            raise ServerPreflightError(f"{cancer} barcode count disagrees with shape")
        if features["id"].shape[0] != shape[0] or features["name"].shape[0] != shape[0]:
            raise ServerPreflightError(f"{cancer} feature count disagrees with shape")
        if group["indptr"].shape[0] != shape[1] + 1:
            raise ServerPreflightError(f"{cancer} indptr length disagrees with CSC shape")
        if group["data"].shape != group["indices"].shape:
            raise ServerPreflightError(f"{cancer} data/indices stored lengths disagree")
        barcodes = pd.Index(_decode(group["barcodes"][:]), dtype="object")
        matrix_header = {
            "features": shape[0],
            "cells": shape[1],
            "stored_nnz_from_dataset_shape_only": int(group["data"].shape[0]),
            "data_dtype": str(group["data"].dtype),
            "indices_dtype": str(group["indices"].dtype),
            "indptr_dtype": str(group["indptr"].dtype),
        }
    if barcodes.has_duplicates:
        raise ServerPreflightError(f"{cancer} H5 barcodes are duplicated")
    metadata_ids = pd.Index(metadata.cell_id.astype(str), dtype="object")
    if len(metadata_ids) != len(barcodes) or (metadata_ids.get_indexer(barcodes) < 0).any():
        raise ServerPreflightError(f"{cancer} H5 barcodes and metadata are not one-to-one")

    return {
        "cancer_id": cancer,
        "h5_path": str(h5_path),
        "h5_bytes": int(h5_path.stat().st_size),
        "h5_bound_sha256": record.get("h5_sha256"),
        "h5_sha256_verified": False,
        "h5_sha256_verification_reason": "DEFERRED_TO_FORMAL_INGEST_TO_AVOID_FULL_MATRIX_FILE_SCAN",
        "metadata_path": str(metadata_path),
        "metadata_bytes": int(metadata_path.stat().st_size),
        "metadata_sha256_verified": True,
        "matrix_header": matrix_header,
        "metadata_rows": int(len(metadata)),
        "cells_with_donor_and_celltype": int(usable_metadata.sum()),
        "cells_missing_donor_or_celltype": int((~usable_metadata).sum()),
        "donors": int(donor.loc[usable_metadata].nunique()),
        "cell_types": int(cell_type.loc[usable_metadata].nunique()),
        "dataset_id": str(dataset_id.iloc[0]),
        "barcode_metadata_one_to_one": True,
        "expression_data_array_read": False,
        "indices_array_read": False,
        "ucell_started": False,
        "pseudotime_started": False,
    }


def run_preflight(
    *,
    root: Path,
    copy_manifest_path: Path,
    expected_copy_manifest_sha256: str,
) -> dict[str, Any]:
    controls = verify_bundle_controls(
        root, copy_manifest_path, expected_copy_manifest_sha256
    )
    run = _load_json(root / "RUN_STATUS.json")
    if set(run.get("formal_eligible_cancers", [])) != set(FORMAL_17):
        raise ServerPreflightError("Formal-17 scope drift")
    size_rows = pd.read_csv(root / "SERVER_READABILITY_PREFLIGHT.tsv", sep="\t")
    size_rows["cancer_id"] = size_rows.cancer_id.astype(str)
    if set(size_rows.cancer_id) != set(FORMAL_17) or size_rows.cancer_id.duplicated().any():
        raise ServerPreflightError("Raw-input byte-size table scope drift")
    size_rows = size_rows.set_index("cancer_id")
    cancers = []
    for cancer in FORMAL_17:
        size = size_rows.loc[cancer]
        cancers.append(
            audit_raw_cancer(
                cancer,
                run["per_cancer"][cancer],
                int(size.h5_bytes),
                int(size.metadata_bytes),
            )
        )
    result = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R7_PORTABLE_SERVER_PREFLIGHT_V1",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "run_id": run["run_id"],
        "r7_root": str(root),
        "controls": controls,
        "formal_cancers": cancers,
        "formal_cancer_count": len(cancers),
        "all_formal_raw_schemas_pass": True,
        "all_barcode_metadata_one_to_one": True,
        "matrix_payload_scanned": False,
        "raw_h5_full_sha_scanned": False,
        "ucell_started": False,
        "pseudotime_started": False,
        "heavy_recompute_started": False,
        "safe_to_start_fresh_cell_level_compute": True,
        "single_cell_module_complete": False,
        "release_ready": False,
        "production_deployed": False,
    }
    result["preflight_contract_sha256"] = _canonical_sha256(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r7-root", required=True, type=Path)
    parser.add_argument("--copy-manifest", required=True, type=Path)
    parser.add_argument("--expected-copy-manifest-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.r7_root.resolve()
    if str(root).replace("\\", "/") != DESTINATION_ROOT:
        raise ServerPreflightError(
            f"Server preflight must run on the fixed destination: {DESTINATION_ROOT}"
        )
    output = args.output_json.resolve()
    if output == root or root in output.parents:
        raise ServerPreflightError(
            "Preflight output must be outside the immutable r7 bundle"
        )
    normalized_output = str(output).replace("\\", "/")
    if not normalized_output.startswith(
        ("./data/CancerLncAtlas/", "./data/CancerLncAtlas/")
    ):
        raise ServerPreflightError("Preflight output is outside authorized server roots")
    result = run_preflight(
        root=root,
        copy_manifest_path=args.copy_manifest.resolve(),
        expected_copy_manifest_sha256=args.expected_copy_manifest_sha256,
    )
    _exclusive_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
