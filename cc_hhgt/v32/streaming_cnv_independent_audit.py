"""Independent, hash-bound acceptance audit for the full-33 streaming CNV store.

The materializer's own ``SUCCESS.json`` is necessary but not sufficient: this
auditor does not import the materializer and independently checks directory
containment, every declared file hash, fold coverage, compact-array semantics,
aggregate counts, the frozen run context, and the external process-memory
supervisor receipt.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TCGA_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
)
STORE_FORMAT = "CC_HHGT_V3_2_STREAMING_SEGMENT_CNV_V1"
PARTITION_FORMAT = "CC_HHGT_V3_2_STREAMING_SEGMENT_CNV_PARTITION_V1"
FOLD_FORMAT = "CC_HHGT_V3_2_STREAMING_SEGMENT_CNV_FOLD_V1"
SUPERVISOR_FORMAT = "CC_HHGT_V3_2_PROCESS_MEMORY_SUPERVISOR_V1"
AUDIT_FORMAT = "CC_HHGT_V3_2_STREAMING_SEGMENT_CNV_INDEPENDENT_AUDIT_V1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class StreamingCNVIndependentAuditError(RuntimeError):
    """Raised when independent acceptance evidence is incomplete or drifts."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise StreamingCNVIndependentAuditError(f"Missing/unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StreamingCNVIndependentAuditError(f"Unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise StreamingCNVIndependentAuditError(f"{label} is not a JSON object")
    return value


def _contained(path: Path, root: Path, label: str) -> Path:
    if path.is_symlink():
        raise StreamingCNVIndependentAuditError(f"{label} is a symlink: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise StreamingCNVIndependentAuditError(
            f"{label} escapes its expected root: {resolved}"
        ) from exc
    return resolved


def _expected_declared_partition_files() -> set[str]:
    """Files hashed by the materializer's parent partition receipt.

    The writer deliberately excludes every file named ``SUCCESS.json`` from
    this map.  Fold receipts are nevertheless bound by the parent receipt via
    its embedded ``folds`` records and are independently compared byte-for-
    semantics below.
    """

    result = {
        "patient_ids.json",
        "lncrna_ids.json",
        "pathway_ids.json",
        "lncrna_event.npy",
        "lncrna_callable.npy",
        "lncrna_burden.npy",
        "pathway_event.npy",
        "pathway_callable.npy",
        "pathway_burden.npy",
    }
    for fold in range(5):
        result.add(f"fold={fold}/test_patient_rows.npy")
    return result


def _expected_partition_inventory() -> set[str]:
    """All files below a published partition, excluding its parent receipt."""

    result = _expected_declared_partition_files()
    for fold in range(5):
        result.add(f"fold={fold}/SUCCESS.json")
    return result


def _read_string_list(path: Path, label: str, expected: int) -> list[str]:
    if not path.is_file() or path.is_symlink():
        raise StreamingCNVIndependentAuditError(f"Missing/unsafe {label}: {path}")
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StreamingCNVIndependentAuditError(f"Unreadable {label}: {path}") from exc
    if (
        not isinstance(values, list)
        or len(values) != expected
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
        or values != sorted(values)
    ):
        raise StreamingCNVIndependentAuditError(f"Invalid ordered unique {label}")
    return values


def _audit_matrix_triplet(
    event_path: Path,
    callable_path: Path,
    burden_path: Path,
    *,
    shape: tuple[int, int],
    row_chunk: int = 64,
) -> tuple[int, int, int]:
    event = np.load(event_path, mmap_mode="r", allow_pickle=False)
    callable_value = np.load(callable_path, mmap_mode="r", allow_pickle=False)
    burden = np.load(burden_path, mmap_mode="r", allow_pickle=False)
    if (
        tuple(event.shape) != shape
        or tuple(callable_value.shape) != shape
        or tuple(burden.shape) != shape
        or event.dtype != np.dtype("uint8")
        or callable_value.dtype != np.dtype("uint8")
        or burden.dtype != np.dtype("float32")
    ):
        raise StreamingCNVIndependentAuditError("Compact CNV matrix shape/dtype drift")
    completely_unavailable = 0
    available_values = 0
    for start in range(0, shape[0], max(1, int(row_chunk))):
        stop = min(shape[0], start + max(1, int(row_chunk)))
        local_event = np.asarray(event[start:stop])
        local_callable = np.asarray(callable_value[start:stop])
        local_burden = np.asarray(burden[start:stop])
        if (
            not np.isin(local_event, (0, 1)).all()
            or not np.isin(local_callable, (0, 1)).all()
            or np.any((local_event == 1) & (local_callable != 1))
            or np.any((local_callable == 1) & ~np.isfinite(local_burden))
            or np.any((local_callable == 0) & ~np.isnan(local_burden))
            or np.any(np.isfinite(local_burden) & (local_burden < 0))
        ):
            raise StreamingCNVIndependentAuditError("Compact CNV typed-null semantics drift")
        completely_unavailable += int(
            np.count_nonzero(
                (local_callable == 0).all(axis=1)
                & np.isnan(local_burden).all(axis=1)
            )
        )
        available_values += int(np.count_nonzero(local_callable))
    raw_bytes = int(event.nbytes + callable_value.nbytes + burden.nbytes)
    del event, callable_value, burden
    return completely_unavailable, available_values, raw_bytes


def _audit_supervisor(
    marker_path: Path,
    *,
    expected_store_root: Path,
    expected_supervisor_sha256: str,
    max_memory_bytes: int,
) -> dict[str, Any]:
    marker = _read_json(marker_path, "memory supervisor SUCCESS")
    audit_root = marker_path.parent.resolve()
    if marker_path.name != "SUCCESS.json":
        raise StreamingCNVIndependentAuditError("Supervisor marker is not SUCCESS.json")
    if any(
        (audit_root / name).exists()
        for name in ("RUNNING.json", "TYPED_FAILURE.json", "SUPERVISOR_FAILURE.json")
    ):
        raise StreamingCNVIndependentAuditError("Supervisor root contains a conflicting state")
    if (
        marker.get("format") != SUPERVISOR_FORMAT
        or marker.get("status") != "SUCCESS"
        or int(marker.get("child_return_code", -1)) != 0
        or marker.get("reason") != "CHILD_EXITED"
        or marker.get("termination") is not None
        or int(marker.get("max_memory_bytes", 0)) != int(max_memory_bytes)
        or marker.get("supervisor_sha256") != str(expected_supervisor_sha256).lower()
    ):
        raise StreamingCNVIndependentAuditError("Memory supervisor SUCCESS contract drift")
    maximum_rss = int(marker.get("max_rss_bytes", -1))
    maximum_hwm = int(marker.get("max_hwm_bytes", -1))
    if max(maximum_rss, maximum_hwm) > int(max_memory_bytes):
        raise StreamingCNVIndependentAuditError("Supervisor observed memory above the gate")
    command = marker.get("command")
    if not isinstance(command, list) or str(expected_store_root) not in map(str, command):
        raise StreamingCNVIndependentAuditError("Supervisor command does not bind the audited store")
    log_path = audit_root / "CHILD.log"
    sample_path = audit_root / "MEMORY_SAMPLES.tsv"
    if (
        sha256_file(log_path) != marker.get("child_log_sha256")
        or sha256_file(sample_path) != marker.get("memory_samples_sha256")
    ):
        raise StreamingCNVIndependentAuditError("Supervisor log/sample hash drift")
    lines = sample_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "elapsed_seconds\trss_bytes\thwm_bytes":
        raise StreamingCNVIndependentAuditError("Supervisor memory sample header drift")
    samples = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 3:
            raise StreamingCNVIndependentAuditError("Malformed supervisor memory sample")
        samples.append((float(fields[0]), int(fields[1]), int(fields[2])))
    if len(samples) != int(marker.get("memory_samples", -1)) or not samples:
        raise StreamingCNVIndependentAuditError("Supervisor memory sample count drift")
    if max(row[1] for row in samples) != maximum_rss or max(row[2] for row in samples) != maximum_hwm:
        raise StreamingCNVIndependentAuditError("Supervisor retained maxima do not match samples")
    return {
        "success_sha256": sha256_file(marker_path),
        "samples": len(samples),
        "max_rss_bytes": maximum_rss,
        "max_hwm_bytes": maximum_hwm,
        "max_memory_bytes": int(max_memory_bytes),
        "poll_seconds": marker.get("poll_seconds"),
        "child_pid": marker.get("child_pid"),
    }


def audit_streaming_cnv_store(
    *,
    aggregate_success_path: str | Path,
    supervisor_success_path: str | Path,
    expected_store_root: str | Path,
    expected_context_sha256: str,
    expected_candidate_sha256: str,
    expected_streaming_code_sha256: str,
    expected_supervisor_sha256: str,
    expected_total_patients: int,
    expected_total_processed: int,
    expected_total_typed_unavailable: int,
    max_memory_bytes: int = 536_870_912,
    expected_cancers: Sequence[str] = TCGA_CANCERS,
) -> dict[str, Any]:
    """Recompute the complete full-scope CNV storage and resource contract."""

    root = Path(expected_store_root).resolve()
    success_path = Path(aggregate_success_path).resolve()
    if success_path != root / "SUCCESS.json":
        raise StreamingCNVIndependentAuditError("Aggregate SUCCESS path/root mismatch")
    cancers = tuple(str(value).upper() for value in expected_cancers)
    if not cancers or len(cancers) != len(set(cancers)):
        raise StreamingCNVIndependentAuditError("Expected cancer contract is empty/duplicated")
    for digest, label in (
        (expected_context_sha256, "context"),
        (expected_candidate_sha256, "candidate"),
        (expected_streaming_code_sha256, "streaming code"),
        (expected_supervisor_sha256, "memory supervisor"),
    ):
        if not HEX64.fullmatch(str(digest).lower()):
            raise StreamingCNVIndependentAuditError(f"Invalid expected {label} SHA256")
    aggregate = _read_json(success_path, "aggregate streaming CNV SUCCESS")
    context_path = root / "RUN_CONTEXT.json"
    context = _read_json(context_path, "streaming CNV RUN_CONTEXT")
    context_without_self = dict(context)
    observed_context_sha = str(context_without_self.pop("context_sha256", ""))
    if canonical_sha256(context_without_self) != observed_context_sha:
        raise StreamingCNVIndependentAuditError("RUN_CONTEXT canonical self-hash drift")
    if (
        observed_context_sha != str(expected_context_sha256).lower()
        or aggregate.get("context_sha256") != observed_context_sha
        or aggregate.get("run_context_sha256") != sha256_file(context_path)
        or context.get("format") != STORE_FORMAT
        or context.get("candidate_loading") != "CANCER_FILTERED_NO_GLOBAL_OBJECT_TABLE"
        or context.get("observed_process_memory_gate_enforced") is not True
        or int(context.get("max_memory_bytes", 0)) != int(max_memory_bytes)
        or context.get("bindings", {}).get("candidates", {}).get("sha256")
        != str(expected_candidate_sha256).lower()
        or context.get("implementation_sha256", {}).get("segment_cnv_streaming.py")
        != str(expected_streaming_code_sha256).lower()
    ):
        raise StreamingCNVIndependentAuditError("Frozen RUN_CONTEXT authority drift")
    for name, declaration in sorted(context.get("bindings", {}).items()):
        if not isinstance(declaration, Mapping):
            raise StreamingCNVIndependentAuditError(f"Invalid context binding: {name}")
        source = Path(str(declaration.get("path", "")))
        expected_hash = str(declaration.get("sha256", "")).lower()
        if (
            not HEX64.fullmatch(expected_hash)
            or not source.is_file()
            or source.is_symlink()
            or sha256_file(source) != expected_hash
        ):
            raise StreamingCNVIndependentAuditError(f"Context binding hash drift: {name}")
    download_gate_path = Path(str(context.get("download_gate", "")))
    if (
        not download_gate_path.is_file()
        or download_gate_path.is_symlink()
        or sha256_file(download_gate_path) != context.get("download_gate_sha256")
    ):
        raise StreamingCNVIndependentAuditError("Download gate hash drift")
    if (
        aggregate.get("format") != STORE_FORMAT
        or aggregate.get("status") != "SUCCESS"
        or aggregate.get("typed_unavailable_is_never_zero") is not True
        or int(aggregate.get("patient_folds", 0)) != 5
        or tuple(aggregate.get("required_cancers", ())) != cancers
        or set(aggregate.get("cancers", {})) != set(cancers)
    ):
        raise StreamingCNVIndependentAuditError("Aggregate full-scope contract drift")
    if list(root.glob(".cancer=*.building.*")):
        raise StreamingCNVIndependentAuditError("Unpublished building directories remain")

    total_patients = 0
    total_processed = 0
    total_unavailable = 0
    max_partition_peak = 0
    cancer_records: list[dict[str, Any]] = []
    expected_declared_files = _expected_declared_partition_files()
    expected_inventory = _expected_partition_inventory()
    for cancer in cancers:
        aggregate_record = aggregate["cancers"][cancer]
        partition_link = root / f"cancer={cancer}"
        if partition_link.is_symlink():
            raise StreamingCNVIndependentAuditError(f"Partition is a symlink: {cancer}")
        partition = partition_link.resolve()
        declared_partition = Path(str(aggregate_record.get("path", ""))).resolve()
        if declared_partition != partition:
            raise StreamingCNVIndependentAuditError(f"Aggregate partition path drift: {cancer}")
        _contained(partition, root, f"{cancer} partition")
        payload_path = partition / "SUCCESS.json"
        payload = _read_json(payload_path, f"{cancer} partition SUCCESS")
        if (
            sha256_file(payload_path) != aggregate_record.get("success_sha256")
            or payload.get("format") != PARTITION_FORMAT
            or payload.get("status") != "SUCCESS"
            or payload.get("cancer_id") != cancer
            or payload.get("context_sha256") != observed_context_sha
            or int(payload.get("memory_gate_bytes", 0)) != int(max_memory_bytes)
            or payload.get("observed_process_memory_gate_enforced") is not True
            or int(payload.get("observed_peak_process_bytes", max_memory_bytes + 1))
            > int(max_memory_bytes)
            or int(payload.get("max_materialized_long_rows", -1)) != 0
        ):
            raise StreamingCNVIndependentAuditError(f"Partition contract drift: {cancer}")
        declared_files = payload.get("files")
        if (
            not isinstance(declared_files, Mapping)
            or set(declared_files) != expected_declared_files
        ):
            raise StreamingCNVIndependentAuditError(f"Partition file contract drift: {cancer}")
        actual_files = {
            path.relative_to(partition).as_posix()
            for path in partition.rglob("*")
            if path.is_file() and path.resolve() != payload_path.resolve()
        }
        if actual_files != expected_inventory:
            raise StreamingCNVIndependentAuditError(f"Partition file inventory drift: {cancer}")
        for relative in sorted(expected_declared_files):
            path = _contained(partition / relative, partition, f"{cancer}/{relative}")
            if path.is_symlink() or sha256_file(path) != declared_files[relative]:
                raise StreamingCNVIndependentAuditError(f"Partition file hash drift: {path}")
        if canonical_sha256(dict(declared_files)) != aggregate_record.get("files_composite_sha256"):
            raise StreamingCNVIndependentAuditError(f"Partition composite hash drift: {cancer}")

        patients = int(payload.get("patients", -1))
        processed = int(payload.get("processed_segment_patients", -1))
        unavailable = int(payload.get("typed_unavailable_patients", -1))
        lncrnas = int(payload.get("lncrnas", -1))
        pathways = int(payload.get("pathways", -1))
        if min(patients, processed, unavailable, lncrnas, pathways) < 0 or processed + unavailable != patients:
            raise StreamingCNVIndependentAuditError(f"Partition patient/entity counts drift: {cancer}")
        patient_ids = _read_string_list(partition / "patient_ids.json", "patient IDs", patients)
        _read_string_list(partition / "lncrna_ids.json", "lncRNA IDs", lncrnas)
        _read_string_list(partition / "pathway_ids.json", "pathway IDs", pathways)
        lnc_unavailable, lnc_available, lnc_raw_bytes = _audit_matrix_triplet(
            partition / "lncrna_event.npy",
            partition / "lncrna_callable.npy",
            partition / "lncrna_burden.npy",
            shape=(patients, lncrnas),
        )
        pathway_unavailable, pathway_available, pathway_raw_bytes = _audit_matrix_triplet(
            partition / "pathway_event.npy",
            partition / "pathway_callable.npy",
            partition / "pathway_burden.npy",
            shape=(patients, pathways),
        )
        if (
            lnc_unavailable < unavailable
            or pathway_unavailable < unavailable
            or lnc_raw_bytes + pathway_raw_bytes != int(payload.get("matrix_disk_bytes_estimate", -1))
        ):
            raise StreamingCNVIndependentAuditError(f"Typed-unavailable matrix count drift: {cancer}")

        fold_payloads = payload.get("folds")
        if not isinstance(fold_payloads, list) or len(fold_payloads) != 5:
            raise StreamingCNVIndependentAuditError(f"Fold manifest count drift: {cancer}")
        fold_by_id = {int(row.get("patient_fold", -1)): row for row in fold_payloads}
        if set(fold_by_id) != set(range(5)):
            raise StreamingCNVIndependentAuditError(f"Fold IDs drift: {cancer}")
        observed_indices: list[int] = []
        for fold in range(5):
            fold_root = partition / f"fold={fold}"
            fold_marker = _read_json(fold_root / "SUCCESS.json", f"{cancer}/fold={fold}")
            if fold_marker != fold_by_id[fold] or (
                fold_marker.get("format") != FOLD_FORMAT
                or fold_marker.get("status") != "SUCCESS"
                or fold_marker.get("cancer_id") != cancer
                or int(fold_marker.get("patient_fold", -1)) != fold
                or fold_marker.get("context_sha256") != observed_context_sha
            ):
                raise StreamingCNVIndependentAuditError(f"Fold SUCCESS drift: {cancer}/{fold}")
            index_path = fold_root / "test_patient_rows.npy"
            indices = np.load(index_path, allow_pickle=False)
            if (
                indices.dtype != np.dtype("int32")
                or indices.ndim != 1
                or not np.array_equal(indices, np.sort(indices))
                or len(indices) != len(np.unique(indices))
                or (len(indices) and (int(indices[0]) < 0 or int(indices[-1]) >= patients))
                or len(indices) != int(fold_marker.get("test_patients", -1))
                or sha256_file(index_path) != fold_marker.get("test_patient_rows_sha256")
            ):
                raise StreamingCNVIndependentAuditError(f"Fold index drift: {cancer}/{fold}")
            observed_indices.extend(map(int, indices.tolist()))
        if sorted(observed_indices) != list(range(len(patient_ids))):
            raise StreamingCNVIndependentAuditError(f"Fold partition is not exact/disjoint: {cancer}")

        peak = int(payload["observed_peak_process_bytes"])
        max_partition_peak = max(max_partition_peak, peak)
        total_patients += patients
        total_processed += processed
        total_unavailable += unavailable
        cancer_records.append(
            {
                "cancer_id": cancer,
                "partition_success_sha256": sha256_file(payload_path),
                "patients": patients,
                "processed_segment_patients": processed,
                "typed_unavailable_patients": unavailable,
                "lncrnas": lncrnas,
                "pathways": pathways,
                "lncrna_callable_values": lnc_available,
                "pathway_callable_values": pathway_available,
                "observed_peak_process_bytes": peak,
                "fold_test_patients": [int(fold_by_id[index]["test_patients"]) for index in range(5)],
            }
        )

    expected_totals = (
        int(expected_total_patients),
        int(expected_total_processed),
        int(expected_total_typed_unavailable),
    )
    observed_totals = (total_patients, total_processed, total_unavailable)
    if observed_totals != expected_totals:
        raise StreamingCNVIndependentAuditError(
            f"Full33 patient totals drift: {observed_totals} != {expected_totals}"
        )
    supervisor = _audit_supervisor(
        Path(supervisor_success_path).resolve(),
        expected_store_root=root,
        expected_supervisor_sha256=str(expected_supervisor_sha256).lower(),
        max_memory_bytes=int(max_memory_bytes),
    )
    return {
        "format": AUDIT_FORMAT,
        "status": "PASS",
        "aggregate_success_path": str(success_path),
        "aggregate_success_sha256": sha256_file(success_path),
        "store_root": str(root),
        "context_sha256": observed_context_sha,
        "run_context_file_sha256": sha256_file(context_path),
        "candidate_sha256": str(expected_candidate_sha256).lower(),
        "streaming_code_sha256": str(expected_streaming_code_sha256).lower(),
        "supervisor_code_sha256": str(expected_supervisor_sha256).lower(),
        "cancers": cancer_records,
        "cancer_count": len(cancer_records),
        "patient_totals": {
            "patients": total_patients,
            "processed_segment_patients": total_processed,
            "typed_unavailable_patients": total_unavailable,
        },
        "max_partition_observed_peak_process_bytes": max_partition_peak,
        "memory_limit_bytes": int(max_memory_bytes),
        "supervisor": supervisor,
        "all_declared_files_rehashed": True,
        "all_fold_success_receipts_semantically_verified": True,
        "partition_inventory_exact_including_fold_receipts": True,
        "all_compact_arrays_semantically_scanned": True,
        "typed_unavailable_never_zero": True,
        "folds_exact_disjoint_exhaustive": True,
        "production_deployment_assessed_by_this_audit": False,
        "port_8260_assessed_by_this_audit": False,
    }


__all__ = [
    "AUDIT_FORMAT",
    "StreamingCNVIndependentAuditError",
    "TCGA_CANCERS",
    "audit_streaming_cnv_store",
    "canonical_sha256",
    "sha256_file",
]
