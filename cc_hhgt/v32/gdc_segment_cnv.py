"""GDC masked segment manifest and recoverable download helpers.

This module only stages candidate inputs.  It never writes a formal V3.2 release.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
TCGA_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
)
TUMOUR_SAMPLE_PRIORITY = {
    "Primary Tumor": 0,
    "Primary Blood Derived Cancer - Peripheral Blood": 0,
    "Primary Blood Derived Cancer - Bone Marrow": 0,
    "Additional - New Primary": 1,
    "Recurrent Tumor": 2,
    "Metastatic": 3,
}
FIELDS = (
    "file_id,file_name,md5sum,file_size,data_type,data_format,access,"
    "analysis.workflow_type,cases.case_id,cases.submitter_id,"
    "cases.project.project_id,cases.samples.sample_id,"
    "cases.samples.submitter_id,cases.samples.sample_type"
)


class SegmentCNVError(RuntimeError):
    pass


UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required by the GDC manifest contract
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_segment_target(row: Mapping[str, Any], output_root: Path) -> Path:
    """Return the only permitted local path for one selected GDC UUID."""

    cancer = str(row["cancer_id"]).upper()
    prefix = str(
        row.get("sample_submitter_id")
        or row.get("case_submitter_id")
        or row["file_id"]
    )
    return output_root / cancer / f"{prefix}__{row['file_name']}"


def selected_gdc_download_candidates(row: Mapping[str, Any], download_root: Path) -> tuple[Path, ...]:
    """Exact supported gdc-client layouts, in strict priority order."""

    file_id = str(row["file_id"])
    file_name = str(row["file_name"])
    return (
        download_root / file_id / file_id / file_name,
        download_root / file_id / file_name,
    )


def _unique_quarantine(path: Path, suffix: str) -> Path:
    candidate = path.with_name(f"{path.name}.{suffix}")
    revision = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{suffix}.r{revision}")
        revision += 1
    return candidate


def _marker_last_directory_publish(source: Path, target: Path) -> str:
    """NFS-safe, no-replace publish with the readiness marker moved last.

    Some network filesystems return ``EINVAL`` for
    ``renameat2(RENAME_NOREPLACE)``.  Creating the final directory is still an
    atomic no-replace operation there.  Consumers already require a top-level
    readiness marker, so reserve the final name first, move every payload entry,
    and expose that marker only after the payload is complete.
    """

    terminal_names = [
        name for name in ("STAGING_COMPLETE.json", "SUCCESS.json")
        if (source / name).is_file()
    ]
    if len(terminal_names) != 1:
        raise SegmentCNVError(
            "Marker-last directory publish requires exactly one top-level readiness marker"
        )
    terminal_name = terminal_names[0]
    try:
        target.mkdir()
    except FileExistsError as exc:
        raise SegmentCNVError(f"Directory publish collision: {target}") from exc

    incomplete = target / "PUBLISH_INCOMPLETE.json"
    state = {
        "format": "CC_HHGT_V3_2_DIRECTORY_PUBLISH_INCOMPLETE_V1",
        "status": "BUILDING",
        "source": str(source),
        "target": str(target),
        "terminal_marker": terminal_name,
        "moved_entries": 0,
    }
    incomplete.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        entries = sorted(source.iterdir(), key=lambda path: path.name)
        for entry in entries:
            if entry.name == terminal_name:
                continue
            destination = target / entry.name
            if destination.exists():
                raise SegmentCNVError(f"Directory publish child collision: {destination}")
            os.rename(entry, destination)
            state["moved_entries"] = int(state["moved_entries"]) + 1

        # There must never be a window in which the ready marker is visible
        # while payload entries are still moving.  A crash between these two
        # operations leaves a marker-less, therefore unusable, final root.
        incomplete.unlink()
        os.rename(source / terminal_name, target / terminal_name)
        source.rmdir()
        return "MKDIR_MARKER_LAST_NOREPLACE"
    except Exception as exc:
        if not (target / terminal_name).exists():
            state.update({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)})
            try:
                incomplete.write_text(
                    json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            except OSError:
                # The absence of the readiness marker remains fail-closed even
                # if the diagnostic record itself cannot be refreshed.
                pass
        raise


def _rename_directory_noreplace(source: Path, target: Path) -> str:
    """Publish a directory without replacing an existing final name."""

    if os.name == "nt":
        os.rename(source, target)
        return "OS_RENAME_NOREPLACE"
    if not sys.platform.startswith("linux"):
        raise SegmentCNVError("Atomic no-replace directory publish is unsupported on this platform")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return _marker_last_directory_publish(source, target)
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(target), 1  # AT_FDCWD, RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            return _marker_last_directory_publish(source, target)
        raise OSError(error, os.strerror(error), str(target))
    return "RENAMEAT2_NOREPLACE"


def full33_gate_implementation_hashes(repo_root: str | Path | None = None) -> dict[str, str]:
    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
    paths = {
        "gdc_segment_cnv.py": Path(__file__).resolve(),
        "genomic_training.py": root / "cc_hhgt/v32/genomic_training.py",
        "preflight_v32_full33_segment_cnv.py": root / "scripts/preflight_v32_full33_segment_cnv.py",
        "download_v32_pancancer_segment_cnv.py": root / "scripts/download_v32_pancancer_segment_cnv.py",
        "run_v32_genomic_training.py": root / "scripts/run_v32_genomic_training.py",
        "server_launch_v32_cnv_router_cpu.sh": root / "scripts/server_launch_v32_cnv_router_cpu.sh",
        "segment_cnv_streaming.py": root / "cc_hhgt/v32/segment_cnv_streaming.py",
        "run_v32_streaming_segment_cnv.py": root / "scripts/run_v32_streaming_segment_cnv.py",
    }
    if missing := [label for label, path in paths.items() if not path.is_file()]:
        raise SegmentCNVError(f"Full33 gate implementation files are missing: {missing}")
    return {label: sha256_file(path) for label, path in paths.items()}


def _selected_manifest_rows(manifest_tsv: Path) -> tuple[list[str], list[dict[str, str]]]:
    with manifest_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = list(reader.fieldnames or [])
        rows = [
            dict(row)
            for row in reader
            if str(row.get("selected_for_patient", "")).lower() in {"true", "1"}
        ]
    required = {
        "cancer_id", "file_id", "file_name", "md5sum", "file_size",
        "case_id", "case_submitter_id", "sample_submitter_id",
        "selected_for_patient",
    }
    if missing := sorted(required - set(columns)):
        raise SegmentCNVError(f"Selected segment manifest lacks columns: {missing}")
    if not rows:
        raise SegmentCNVError("Selected segment manifest is empty")
    file_ids = [str(row["file_id"]).lower() for row in rows]
    if any(UUID_PATTERN.fullmatch(value) is None for value in file_ids):
        raise SegmentCNVError("Selected segment manifest contains a malformed GDC UUID")
    if len(file_ids) != len(set(file_ids)):
        raise SegmentCNVError("Selected segment manifest contains duplicate GDC UUIDs")
    patient_keys = [(row["cancer_id"], row["case_id"]) for row in rows]
    if len(patient_keys) != len(set(patient_keys)):
        raise SegmentCNVError("Selected segment manifest contains multiple files for one patient")
    for row in rows:
        if re.fullmatch(r"[0-9a-fA-F]{32}", str(row["md5sum"])) is None:
            raise SegmentCNVError(f"GDC UUID {row['file_id']} lacks a valid MD5")
        if int(row["file_size"]) <= 0:
            raise SegmentCNVError(f"GDC UUID {row['file_id']} lacks a positive size")
    return columns, rows


def audit_selected_segment_download(
    manifest_tsv: str | Path,
    manifest_json: str | Path,
    download_root: str | Path,
) -> dict[str, Any]:
    """Re-hash every selected UUID and fail closed on size or MD5 drift."""

    table = Path(manifest_tsv).resolve()
    authority_path = Path(manifest_json).resolve()
    root = Path(download_root).resolve()
    if not table.is_file() or not authority_path.is_file():
        raise SegmentCNVError("Segment manifest table/authority is missing")
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    bundle_inventory = {
        "MANIFEST.json": sha256_file(authority_path), table.name: sha256_file(table),
    }
    summary_path = table.parent / "GDC_CLIENT_MANIFEST_SUMMARY.json"
    chunk_hashes: dict[str, str] = {}
    if int(authority.get("selected_patient_files", -1)) == 11057:
        chunk_root = table.parent / "gdc_client_chunks"
        chunk_index_path = chunk_root / "CHUNK_INDEX.tsv"
        if not summary_path.is_file() or not chunk_index_path.is_file():
            raise SegmentCNVError("Complete r3 GDC client summary/chunk index is missing")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_summary = {
            "format": "CC_HHGT_V3_2_GDC_CLIENT_SEGMENT_MANIFEST_V1",
            "selected_files": 11057, "reusable_files": 141, "missing_files": 10916,
            "chunk_size": 250, "chunks": 44,
            "gdc_client_uuid_directory_layout_preserved": True,
            "file_case_sample_mapping": table.name,
        }
        if any(summary.get(key) != value for key, value in expected_summary.items()):
            raise SegmentCNVError("GDC client summary is not the complete frozen r3 full33 version")
        with chunk_index_path.open("r", encoding="utf-8", newline="") as handle:
            chunk_index = list(csv.DictReader(handle, delimiter="\t"))
        if (
            len(chunk_index) != 44
            or sum(int(row["files"]) for row in chunk_index) != 10916
            or sum(int(row["bytes"]) for row in chunk_index) != int(summary.get("missing_bytes", -1))
        ):
            raise SegmentCNVError("GDC chunk index does not declare exact 44/10916 coverage")
        for row in chunk_index:
            chunk = chunk_root / row["path"]
            if not chunk.is_file():
                raise SegmentCNVError(f"GDC chunk missing: {chunk}")
            with chunk.open("r", encoding="utf-8", newline="") as handle:
                if sum(1 for _ in handle) - 1 != int(row["files"]):
                    raise SegmentCNVError(f"GDC chunk row-count drift: {chunk}")
            chunk_hashes[row["path"]] = sha256_file(chunk)
        bundle_inventory.update({
            "GDC_CLIENT_MANIFEST_SUMMARY.json": sha256_file(summary_path),
            "gdc_client_chunks/CHUNK_INDEX.tsv": sha256_file(chunk_index_path),
            **{f"gdc_client_chunks/{name}": digest for name, digest in chunk_hashes.items()},
        })
    declared_table = authority.get("table", {})
    if declared_table.get("sha256") != sha256_file(table):
        raise SegmentCNVError("Segment manifest TSV disagrees with MANIFEST.json SHA256")
    columns, rows = _selected_manifest_rows(table)
    if int(authority.get("selected_patient_files", -1)) != len(rows):
        raise SegmentCNVError("Selected UUID count disagrees with MANIFEST.json")
    records: list[dict[str, Any]] = []
    retry_rows: list[dict[str, str]] = []
    inventory = hashlib.sha256()
    for row in rows:
        expected_size = int(row["file_size"])
        expected_md5 = str(row["md5sum"]).lower()
        status = "VERIFIED"
        observed_size: int | None = None
        observed_md5: str | None = None
        candidates = tuple(path.resolve() for path in selected_gdc_download_candidates(row, root))
        existing = [path for path in candidates if path.is_file()]
        source = existing[0] if existing else candidates[0]
        if len(existing) > 1:
            status = "LAYOUT_CONFLICT"
        elif not existing:
            status = "MISSING"
        else:
            observed_size = int(source.stat().st_size)
            if observed_size != expected_size:
                status = "SIZE_MISMATCH"
            else:
                observed_md5 = md5_file(source)
                if observed_md5 != expected_md5:
                    status = "MD5_MISMATCH"
        if status != "VERIFIED":
            retry = dict(row)
            retry["preflight_retry_reason"] = status
            retry_rows.append(retry)
        inventory.update(
            (f"{row['file_id'].lower()}\t{expected_size}\t{expected_md5}\t"
             f"{source.as_posix()}\n").encode("utf-8")
        )
        records.append(
            {
                "file_id": row["file_id"],
                "cancer_id": row["cancer_id"],
                "path": str(source),
                "source_layout": (
                    "CONFLICT" if len(existing) > 1
                    else "MISSING" if not existing
                    else "UUID_UUID_FILENAME" if source == candidates[0]
                    else "UUID_FILENAME"
                ),
                "sample_submitter_id": row["sample_submitter_id"],
                "case_submitter_id": row["case_submitter_id"],
                "status": status,
                "expected_size": expected_size,
                "observed_size": observed_size,
                "expected_md5": expected_md5,
                "observed_md5": observed_md5,
            }
        )
    return {
        "status": "READY" if not retry_rows else "BLOCKED",
        "manifest_tsv": str(table),
        "manifest_tsv_sha256": sha256_file(table),
        "manifest_json": str(authority_path),
        "manifest_json_sha256": sha256_file(authority_path),
        "manifest_bundle_inventory_sha256": hashlib.sha256(
            json.dumps(bundle_inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "manifest_bundle_files": len(bundle_inventory),
        "gdc_client_summary_sha256": sha256_file(summary_path) if summary_path.is_file() else None,
        "gdc_client_chunks": len(chunk_hashes),
        "download_root": str(root),
        "selected_files": len(rows),
        "verified_files": len(rows) - len(retry_rows),
        "retry_files": len(retry_rows),
        "selected_total_bytes": sum(int(row["file_size"]) for row in rows),
        "inventory_sha256": inventory.hexdigest(),
        "columns": columns,
        "retry_rows": retry_rows,
        "records": records,
    }


def materialize_verified_segment_staging(
    audit: Mapping[str, Any],
    staging_root: str | Path,
) -> dict[str, Any]:
    """Publish only verified UUID inputs into a new training-compatible root."""

    if audit.get("status") != "READY" or int(audit.get("retry_files", -1)) != 0:
        raise SegmentCNVError("Unverified segment inventory cannot be staged")
    staging = Path(staging_root).resolve()
    if staging.exists():
        raise SegmentCNVError(f"Segment staging refuses output reuse: {staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    token = str(audit.get("inventory_sha256", "unknown"))[:12]
    revision = 0
    while True:
        suffix = f".{token}" if revision == 0 else f".{token}.r{revision}"
        building = staging.with_name(f".{staging.name}.building{suffix}")
        try:
            building.mkdir()
            break
        except FileExistsError:
            revision += 1
    incomplete = building / "STAGING_INCOMPLETE.json"
    state: dict[str, Any] = {
        "format": "CC_HHGT_V3_2_SEGMENT_STAGING_INCOMPLETE_V1",
        "status": "BUILDING",
        "final_staging_root": str(staging),
        "source_inventory_sha256": audit.get("inventory_sha256"),
        "expected_files": int(audit.get("selected_files", -1)),
        "published_files": 0,
    }
    incomplete.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    published: list[dict[str, Any]] = []
    inventory = hashlib.sha256()
    publish_lock = staging.with_name(f".{staging.name}.publish.lock")
    lock_owned = False
    try:
        for record in audit.get("records", []):
            source = Path(record["path"]).resolve()
            row = {
                "cancer_id": record["cancer_id"],
                "file_id": record["file_id"],
                "file_name": source.name,
                "sample_submitter_id": record["sample_submitter_id"],
                "case_submitter_id": record["case_submitter_id"],
            }
            final_target = selected_segment_target(row, staging).resolve()
            relative = final_target.relative_to(staging)
            target = building / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(f".{target.name}.partial")
            try:
                os.link(source, partial)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                with source.open("rb") as source_handle, partial.open("xb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            if partial.stat().st_size != int(record["expected_size"]):
                raise SegmentCNVError(f"Staged segment size drift: {record['file_id']}")
            observed_md5 = md5_file(partial)
            if observed_md5 != record["expected_md5"]:
                raise SegmentCNVError(f"Staged segment MD5 drift: {record['file_id']}")
            os.rename(partial, target)
            inventory.update(
                (f"{record['file_id'].lower()}\t{record['expected_size']}\t"
                 f"{record['expected_md5']}\t{final_target.as_posix()}\n").encode("utf-8")
            )
            published.append(
                {
                    "file_id": record["file_id"],
                    "cancer_id": record["cancer_id"],
                    "source_path": str(source),
                    "staged_path": str(final_target),
                    "size": int(record["expected_size"]),
                    "md5": record["expected_md5"],
                }
            )
            state["published_files"] = len(published)
        if len(published) != int(audit.get("selected_files", -1)):
            raise SegmentCNVError("Staging file count disagrees with verified selected inventory")
        complete = {
            "format": "CC_HHGT_V3_2_SEGMENT_STAGING_V1",
            "status": "READY",
            "source_inventory_sha256": audit.get("inventory_sha256"),
            "files": len(published),
            "inventory_sha256": inventory.hexdigest(),
        }
        (building / "STAGING_COMPLETE.json").write_text(
            json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        incomplete.unlink()
        lock_fd = os.open(publish_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        lock_owned = True
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(lock_fd)
        if staging.exists():
            raise SegmentCNVError(f"Segment staging publish collision: {staging}")
        publish_method = _rename_directory_noreplace(building, staging)
        return {
            "status": "READY",
            "staging_root": str(staging),
            "files": len(published),
            "inventory_sha256": inventory.hexdigest(),
            "publish_method": publish_method,
            "records": published,
        }
    except Exception as exc:
        if building.exists():
            state.update({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)})
            incomplete.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    finally:
        if lock_owned:
            publish_lock.unlink(missing_ok=True)


def validate_segment_download_gate(
    marker_path: str | Path,
    *,
    staging_root: str | Path,
    bindings: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Re-verify a READY marker and every bound downstream input."""

    marker = Path(marker_path).resolve()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("status") != "READY":
        raise SegmentCNVError("CNV segment download gate is not READY")
    if payload.get("format") != "CC_HHGT_V3_2_FULL33_SEGMENT_DOWNLOAD_GATE_V1":
        raise SegmentCNVError("CNV segment download gate format is invalid")
    if set(payload.get("required_cancers", [])) != set(TCGA_CANCERS):
        raise SegmentCNVError("CNV segment download gate is not full33")
    if int(payload.get("patient_folds", -1)) != 5:
        raise SegmentCNVError("CNV segment download gate is not five-fold bound")
    if payload.get("implementation_sha256") != full33_gate_implementation_hashes():
        raise SegmentCNVError("CNV segment gate implementation hash drift")
    if Path(payload.get("staging_root", "")).resolve() != Path(staging_root).resolve():
        raise SegmentCNVError("CNV segment download gate points to another staging root")
    current = audit_selected_segment_download(
        payload["manifest_tsv"], payload["manifest_json"], payload["source_download_root"]
    )
    if current["status"] != "READY" or current["inventory_sha256"] != payload.get("source_inventory_sha256"):
        raise SegmentCNVError("CNV source segment inventory changed after preflight")
    _, rows = _selected_manifest_rows(Path(payload["manifest_tsv"]))
    staged_inventory = hashlib.sha256()
    for row in rows:
        target = selected_segment_target(row, Path(staging_root).resolve()).resolve()
        if not target.is_file() or target.stat().st_size != int(row["file_size"]):
            raise SegmentCNVError(f"CNV staged segment missing/size drift: {row['file_id']}")
        if md5_file(target) != str(row["md5sum"]).lower():
            raise SegmentCNVError(f"CNV staged segment MD5 drift: {row['file_id']}")
        staged_inventory.update(
            (f"{str(row['file_id']).lower()}\t{int(row['file_size'])}\t"
             f"{str(row['md5sum']).lower()}\t{target.as_posix()}\n").encode("utf-8")
        )
    if staged_inventory.hexdigest() != payload.get("staging_inventory_sha256"):
        raise SegmentCNVError("CNV staged segment inventory hash drift")
    callability = payload.get("patient_segment_callability", {})
    callability_path = Path(callability.get("path", "")).resolve()
    if (
        not callability_path.is_file()
        or callability.get("sha256") != sha256_file(callability_path)
        or int(callability.get("fold_patients", -1)) <= 0
        or int(callability.get("typed_unavailable", -1)) < 0
    ):
        raise SegmentCNVError("CNV patient typed-callability manifest drift")
    declared_bindings = payload.get("bindings", {})
    for label, raw_path in bindings.items():
        path = Path(raw_path).resolve()
        declared = declared_bindings.get(label, {})
        if declared.get("path") != str(path) or declared.get("sha256") != sha256_file(path):
            raise SegmentCNVError(f"CNV segment gate binding drift: {label}")
    return payload


def gdc_filter(cancers: Sequence[str]) -> dict[str, Any]:
    projects = [f"TCGA-{str(value).upper()}" for value in cancers]
    return {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": projects}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Masked Copy Number Segment"]}},
            {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["DNAcopy"]}},
            {"op": "in", "content": {"field": "files.access", "value": ["open"]}},
        ],
    }


def _request_page(
    cancers: Sequence[str], offset: int, size: int, *, timeout: int = 120
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "filters": json.dumps(gdc_filter(cancers), separators=(",", ":")),
            "fields": FIELDS,
            "expand": "cases.project,cases.samples",
            "from": offset,
            "size": size,
        }
    )
    request = urllib.request.Request(
        f"{GDC_FILES_ENDPOINT}?{query}",
        headers={"Accept": "application/json", "User-Agent": "CancerLncAtlas-V3.2-CNV-audit/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def query_gdc_masked_segments(
    cancers: Sequence[str], *, page_size: int = 5000, timeout: int = 120
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cancers = tuple(dict.fromkeys(str(value).upper() for value in cancers))
    unknown = sorted(set(cancers) - set(TCGA_CANCERS))
    if unknown:
        raise SegmentCNVError(f"Unknown TCGA cancer IDs: {unknown}")
    hits: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        payload = _request_page(cancers, offset, page_size, timeout=timeout)
        data = payload.get("data", {})
        page = list(data.get("hits", []))
        pagination = data.get("pagination", {})
        total = int(pagination.get("total", len(page)))
        hits.extend(page)
        offset += len(page)
        if not page:
            break
    if total is not None and len(hits) != total:
        raise SegmentCNVError(f"GDC pagination incomplete: expected {total}, received {len(hits)}")
    return hits, {"api_total": total or 0, "pages_complete": True}


def _tumour_samples(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    samples = []
    for sample in case.get("samples") or []:
        sample_type = str(sample.get("sample_type") or "")
        if sample_type in TUMOUR_SAMPLE_PRIORITY:
            samples.append(dict(sample))
    return sorted(
        samples,
        key=lambda row: (
            TUMOUR_SAMPLE_PRIORITY[str(row.get("sample_type"))],
            str(row.get("submitter_id") or ""),
            str(row.get("sample_id") or ""),
        ),
    )


def flatten_tumour_manifest(hits: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    excluded_no_tumour_sample = 0
    ambiguous_sample_files = 0
    for hit in hits:
        cases = hit.get("cases") or []
        for case in cases:
            tumour = _tumour_samples(case)
            if not tumour:
                excluded_no_tumour_sample += 1
                continue
            if len(tumour) > 1:
                ambiguous_sample_files += 1
            sample = tumour[0]
            project = case.get("project") or {}
            project_id = str(project.get("project_id") or "")
            cancer = project_id.removeprefix("TCGA-")
            rows.append(
                {
                    "cancer_id": cancer,
                    "project_id": project_id,
                    "file_id": str(hit.get("file_id") or hit.get("id") or ""),
                    "file_name": str(hit.get("file_name") or ""),
                    "md5sum": str(hit.get("md5sum") or "").lower(),
                    "file_size": int(hit.get("file_size") or 0),
                    "case_id": str(case.get("case_id") or ""),
                    "case_submitter_id": str(case.get("submitter_id") or ""),
                    "sample_id": str(sample.get("sample_id") or ""),
                    "sample_submitter_id": str(sample.get("submitter_id") or ""),
                    "sample_type": str(sample.get("sample_type") or ""),
                    "sample_metadata_ambiguous": len(tumour) > 1,
                    "data_type": str(hit.get("data_type") or ""),
                    "workflow_type": str((hit.get("analysis") or {}).get("workflow_type") or ""),
                    "existing_reusable": False,
                    "existing_path": "",
                }
            )
    rows.sort(key=lambda row: (row["cancer_id"], row["case_submitter_id"], row["sample_submitter_id"], row["file_name"], row["file_id"]))
    seen_cases: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["cancer_id"], row["case_id"])
        row["selected_for_patient"] = key not in seen_cases
        seen_cases.add(key)
    return rows, {
        "excluded_files_without_tumour_sample": excluded_no_tumour_sample,
        "files_with_ambiguous_tumour_sample_metadata": ambiguous_sample_files,
    }


def reconcile_existing_files(
    rows: Sequence[dict[str, Any]], roots: Sequence[Path]
) -> dict[str, Any]:
    """Mark byte-identical cached files; never infer coverage from cancer alone."""

    by_name: dict[str, list[Path]] = {}
    by_file_id: dict[str, list[Path]] = {}
    scanned = 0
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.seg*.txt"):
            scanned += 1
            by_name.setdefault(path.name, []).append(path)
            for part in path.parts:
                if len(part) == 36 and part.count("-") == 4:
                    by_file_id.setdefault(part.lower(), []).append(path)
            for match in re.findall(
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                str(path),
            ):
                by_file_id.setdefault(match.lower(), []).append(path)
    reused = 0
    for row in rows:
        candidates = list(by_file_id.get(str(row["file_id"]).lower(), []))
        candidates.extend(by_name.get(str(row["file_name"]), []))
        unique = list(dict.fromkeys(candidates))
        for path in unique:
            if md5_file(path) == str(row["md5sum"]).lower():
                row["existing_reusable"] = True
                row["existing_path"] = str(path)
                reused += 1
                break
    # Prefer a verified cached aliquot when a case has more than one tumour file.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["cancer_id"]), str(row["case_id"])), []).append(row)
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda row: (
                not bool(row["existing_reusable"]),
                TUMOUR_SAMPLE_PRIORITY.get(str(row["sample_type"]), 99),
                str(row["sample_submitter_id"]),
                str(row["file_name"]),
                str(row["file_id"]),
            ),
        )
        for index, row in enumerate(ordered):
            row["selected_for_patient"] = index == 0
    return {"existing_files_scanned": scanned, "existing_files_reusable": reused}


def reconcile_existing_inventory(
    rows: Sequence[dict[str, Any]], inventory: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    """Reconcile server-produced ``(md5, path)`` records by UUID and hash."""

    by_uuid: dict[str, list[tuple[str, str]]] = {}
    for observed_md5, observed_path in inventory:
        for match in re.findall(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            observed_path,
        ):
            by_uuid.setdefault(match.lower(), []).append((observed_md5.lower(), observed_path))
    reused = 0
    for row in rows:
        for observed_md5, observed_path in by_uuid.get(str(row["file_id"]).lower(), []):
            if observed_md5 == str(row["md5sum"]).lower():
                row["existing_reusable"] = True
                row["existing_path"] = observed_path
                reused += 1
                break
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["cancer_id"]), str(row["case_id"])), []).append(row)
    for group in grouped.values():
        ordered = sorted(group, key=lambda row: (
            not bool(row["existing_reusable"]), str(row["sample_submitter_id"]),
            str(row["file_name"]), str(row["file_id"]),
        ))
        for index, row in enumerate(ordered):
            row["selected_for_patient"] = index == 0
    return {"existing_files_scanned": len(inventory), "existing_files_reusable": reused}


def write_manifest(rows: Sequence[Mapping[str, Any]], output_dir: Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else [
        "cancer_id", "project_id", "file_id", "file_name", "md5sum", "file_size",
        "case_id", "case_submitter_id", "sample_id", "sample_submitter_id",
        "sample_type", "sample_metadata_ambiguous", "data_type", "workflow_type",
        "selected_for_patient",
    ]
    table = output_dir / "gdc_masked_segment_file_case_sample_manifest.tsv"
    temporary = table.with_suffix(".tsv.partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(table)
    selected = [row for row in rows if bool(row.get("selected_for_patient"))]
    by_cancer: dict[str, dict[str, int]] = {}
    for row in selected:
        item = by_cancer.setdefault(str(row["cancer_id"]), {"files": 0, "bytes": 0})
        item["files"] += 1
        item["bytes"] += int(row["file_size"])
    payload = {
        "format": "CC_HHGT_V3_2_GDC_MASKED_SEGMENT_MANIFEST_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "formal_v32_unchanged": True,
        "gdc_endpoint": GDC_FILES_ENDPOINT,
        "filters": audit.get("filters"),
        "queried_cancers": audit.get("queried_cancers"),
        "all_tumour_file_rows": len(rows),
        "selected_patient_files": len(selected),
        "selected_total_bytes": sum(int(row["file_size"]) for row in selected),
        "selected_existing_reusable_files": sum(bool(row.get("existing_reusable")) for row in selected),
        "selected_download_required_files": sum(not bool(row.get("existing_reusable")) for row in selected),
        "selected_download_required_bytes": sum(
            int(row["file_size"]) for row in selected if not bool(row.get("existing_reusable"))
        ),
        "by_cancer": by_cancer,
        "audit": dict(audit),
        "table": {"path": table.name, "sha256": sha256_file(table), "bytes": table.stat().st_size},
    }
    manifest = output_dir / "MANIFEST.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _download_one(
    row: Mapping[str, str],
    output_root: Path,
    retries: int,
    chunk_size: int,
    lock_ttl_seconds: int,
) -> dict[str, Any]:
    target = selected_segment_target(row, output_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_md5 = str(row["md5sum"]).lower()
    expected_size = int(row["file_size"]) if str(row.get("file_size", "")).strip() else None
    if target.is_file():
        observed_size = int(target.stat().st_size)
        observed_md5 = md5_file(target)
        if observed_md5 == expected_md5 and (expected_size is None or observed_size == expected_size):
            return {"file_id": row["file_id"], "status": "ALREADY_VERIFIED", "path": str(target)}
        return {
            "file_id": row["file_id"], "status": "TARGET_CONFLICT", "path": str(target),
            "expected_size": expected_size, "observed_size": observed_size,
            "expected_md5": expected_md5, "observed_md5": observed_md5,
        }
    partial = target.with_suffix(target.suffix + ".partial")
    lock = target.with_suffix(target.suffix + ".lock")
    if lock.exists():
        age = max(0.0, time.time() - lock.stat().st_mtime)
        if age > lock_ttl_seconds:
            lock.unlink(missing_ok=True)
        else:
            return {
                "file_id": row["file_id"], "status": "LOCKED_ACTIVE",
                "lock_age_seconds": round(age, 3), "lock_ttl_seconds": lock_ttl_seconds,
                "path": str(target),
            }
    try:
        with lock.open("x", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "created_at": time.time(), "file_id": row["file_id"]}, handle)
    except FileExistsError:
        return {"file_id": row["file_id"], "status": "LOCK_RACE", "path": str(target)}
    try:
        for attempt in range(retries + 1):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"User-Agent": "CancerLncAtlas-V3.2-CNV-download/1"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(f"{GDC_DATA_ENDPOINT}/{row['file_id']}", headers=headers)
                with urllib.request.urlopen(request, timeout=180) as response:
                    status = int(getattr(response, "status", 200))
                    mode = "ab" if offset and status == 206 else "wb"
                    with partial.open(mode) as handle:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            handle.write(chunk)
                if (expected_size is not None and partial.stat().st_size != expected_size) or md5_file(partial) != expected_md5:
                    quarantine = _unique_quarantine(partial, f"corrupt.attempt{attempt + 1}")
                    partial.replace(quarantine)
                    raise SegmentCNVError(
                        f"Size/MD5 mismatch after download: {row['file_id']}; quarantined={quarantine}"
                    )
                if target.exists():
                    quarantine = _unique_quarantine(partial, "verified.target_conflict")
                    partial.replace(quarantine)
                    return {
                        "file_id": row["file_id"], "status": "TARGET_CONFLICT",
                        "path": str(target), "verified_partial_quarantine": str(quarantine),
                    }
                os.rename(partial, target)
                return {"file_id": row["file_id"], "status": "DOWNLOADED_VERIFIED", "path": str(target)}
            except (OSError, urllib.error.URLError, SegmentCNVError) as exc:
                if attempt >= retries:
                    return {"file_id": row["file_id"], "status": "FAILED", "error": str(exc), "path": str(target)}
                time.sleep(min(60, 2 ** attempt))
    finally:
        lock.unlink(missing_ok=True)


def download_manifest(
    manifest_tsv: Path,
    output_root: Path,
    *,
    cancers: Sequence[str] = (),
    workers: int = 3,
    retries: int = 4,
    chunk_size: int = 1024 * 1024,
    max_files: int | None = None,
    lock_ttl_seconds: int = 21600,
) -> list[dict[str, Any]]:
    if "candidate" not in str(output_root).lower():
        raise SegmentCNVError("Download output must be an explicitly candidate-only path")
    with manifest_tsv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    allowed = {value.upper() for value in cancers}
    rows = [row for row in rows if str(row.get("selected_for_patient", "")).lower() in {"true", "1"}]
    if allowed:
        rows = [row for row in rows if row["cancer_id"].upper() in allowed]
    if max_files is not None:
        rows = rows[:max_files]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 6))) as pool:
        futures = [
            pool.submit(_download_one, row, output_root, retries, chunk_size, lock_ttl_seconds)
            for row in rows
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda row: str(row["file_id"]))


__all__ = [
    "GDC_DATA_ENDPOINT", "GDC_FILES_ENDPOINT", "SegmentCNVError", "TCGA_CANCERS",
    "audit_selected_segment_download", "download_manifest", "flatten_tumour_manifest",
    "full33_gate_implementation_hashes", "materialize_verified_segment_staging",
    "gdc_filter", "md5_file", "selected_gdc_download_candidates", "selected_segment_target",
    "reconcile_existing_files", "reconcile_existing_inventory",
    "query_gdc_masked_segments", "sha256_file", "validate_segment_download_gate",
    "write_manifest",
]
