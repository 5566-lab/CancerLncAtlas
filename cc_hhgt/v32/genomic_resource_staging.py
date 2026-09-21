"""Bounded cancer partitions and disk-backed genomic OOF prediction arrays.

This adapter is intentionally separate from the frozen CNV/genomic training
bundle.  It provides a resource-gated path for a future rerun without changing
or blessing any existing prediction.  Candidate strings are held one cancer
at a time and fold probabilities live in ``.npy`` memmaps.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


TARGET_KEYS = ("cancer_id", "lncrna_id", "pathway_id")
MODALITIES = ("mutation", "cnv")
FOLDS = 5
BUDGET_FORMAT = "CC_HHGT_V3_2_GENOMIC_RESOURCE_BUDGET_V1"
STAGE_FORMAT = "CC_HHGT_V3_2_GENOMIC_CANCER_MEMMAP_STAGE_V1"
READY_FORMAT = "CC_HHGT_V3_2_GENOMIC_MEMMAP_PREDICTIONS_READY_V1"
REASSEMBLY_FORMAT = "CC_HHGT_V3_2_GENOMIC_MEMMAP_REASSEMBLY_V1"
CNV_STREAMING_FORMAT = "CC_HHGT_V3_2_STREAMING_SEGMENT_CNV_V1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GenomicResourceError(RuntimeError):
    """Raised when resource staging, writes, or reassembly fail closed."""


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise GenomicResourceError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenomicResourceError(f"Unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise GenomicResourceError(f"{label} is not a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_budget(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("format") != BUDGET_FORMAT:
        raise GenomicResourceError("Genomic resource budget format drift")
    for key in (
        "scope",
        "adapter_budget",
        "downstream_budget",
        "memmap_layout",
        "input_bindings",
    ):
        if not isinstance(value.get(key), Mapping):
            raise GenomicResourceError(f"Genomic resource budget lacks {key}")
    scope = value["scope"]
    cancers = [str(item).upper() for item in scope.get("formal_cancers", [])]
    rows = int(scope.get("candidate_rows", 0))
    per_cancer = int(scope.get("candidate_rows_per_cancer", 0))
    if not cancers or len(cancers) != len(set(cancers)) or rows != len(cancers) * per_cancer:
        raise GenomicResourceError("Genomic resource scope is internally inconsistent")
    layout = value["memmap_layout"]
    if (
        tuple(layout.get("modalities", ())) != MODALITIES
        or int(layout.get("patient_folds", -1)) != FOLDS
        or layout.get("probability_dtype") != "float32"
        or layout.get("available_dtype") != "bool"
        or layout.get("processed_dtype") != "bool"
    ):
        raise GenomicResourceError("Genomic memmap layout drift")
    adapter = value["adapter_budget"]
    downstream = value["downstream_budget"]
    if int(adapter.get("max_candidate_partition_rows", 0)) < per_cancer:
        raise GenomicResourceError("Candidate partition exceeds the declared row budget")
    for key in (
        "max_declared_adapter_peak_rss_bytes",
        "max_memmap_bytes",
        "max_staging_disk_bytes",
    ):
        if int(adapter.get(key, 0)) <= 0:
            raise GenomicResourceError(f"Invalid adapter budget: {key}")
    for key in (
        "max_declared_process_rss_bytes",
        "minimum_host_available_memory_bytes",
        "minimum_host_free_disk_bytes",
    ):
        if int(downstream.get(key, 0)) <= 0:
            raise GenomicResourceError(f"Invalid downstream budget: {key}")
    if downstream.get("forbid_launch_without_resource_success") is not True:
        raise GenomicResourceError("Undeclared downstream launch is not forbidden")
    bindings = value["input_bindings"]
    candidate = bindings.get("candidate_authority")
    cnv = bindings.get("cnv_streaming_aggregate_success")
    if not isinstance(candidate, Mapping) or not isinstance(cnv, Mapping):
        raise GenomicResourceError("Genomic resource input bindings are incomplete")
    if not str(candidate.get("path", "")):
        raise GenomicResourceError("Candidate authority path is not pinned")
    if not _SHA256.fullmatch(str(candidate.get("sha256", "")).lower()):
        raise GenomicResourceError("Candidate authority SHA-256 is not pinned")
    if not _SHA256.fullmatch(
        str(candidate.get("ordered_exact_candidate_key_sha256", "")).lower()
    ):
        raise GenomicResourceError("Candidate ordered-key SHA-256 is not pinned")
    if not str(cnv.get("path", "")):
        raise GenomicResourceError("CNV streaming aggregate SUCCESS path is not pinned")
    if cnv.get("sha256_policy") != "CALLER_MUST_PIN_EXACT_64_HEX_AT_LAUNCH":
        raise GenomicResourceError("CNV streaming aggregate hash policy drift")
    if cnv.get("format") != CNV_STREAMING_FORMAT or cnv.get("status") != "SUCCESS":
        raise GenomicResourceError("CNV streaming aggregate semantic binding drift")
    return dict(value)


def _require_exact_path(path: Path, expected: Path, label: str) -> None:
    if path.resolve() != expected.resolve():
        raise GenomicResourceError(f"{label} path drift: {path} != {expected}")


def _require_inside(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise GenomicResourceError(f"{label} escapes its resource root: {path}") from exc


def _validate_cnv_aggregate_gate(
    path: Path, *, expected_sha256: str, expected_cancers: Sequence[str]
) -> dict[str, Any]:
    expected_hash = str(expected_sha256).lower()
    if not _SHA256.fullmatch(expected_hash):
        raise GenomicResourceError(
            "CNV streaming aggregate SUCCESS SHA-256 must be caller-pinned"
        )
    observed_hash = artifact_sha256(path)
    if observed_hash != expected_hash:
        raise GenomicResourceError("CNV streaming aggregate SUCCESS SHA-256 drift")
    payload = _read_json(path, "CNV streaming aggregate SUCCESS")
    if payload.get("format") != CNV_STREAMING_FORMAT or payload.get("status") != "SUCCESS":
        raise GenomicResourceError("CNV streaming aggregate is not a formal SUCCESS")
    cancer_payload = payload.get("cancers")
    if not isinstance(cancer_payload, Mapping):
        raise GenomicResourceError("CNV streaming aggregate cancer inventory is invalid")
    cancers = {str(item).upper() for item in cancer_payload}
    required = {str(item).upper() for item in expected_cancers}
    if cancers != required or {str(item).upper() for item in payload.get("required_cancers", [])} != required:
        raise GenomicResourceError("CNV streaming aggregate does not cover the exact cancer scope")
    if int(payload.get("patient_folds", -1)) != FOLDS:
        raise GenomicResourceError("CNV streaming aggregate patient-fold count drift")
    if payload.get("typed_unavailable_is_never_zero") is not True:
        raise GenomicResourceError("CNV streaming aggregate typed-missingness contract drift")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": observed_hash,
        "format": CNV_STREAMING_FORMAT,
        "status": "SUCCESS",
        "cancers": len(cancers),
        "patient_folds": FOLDS,
    }


def _available_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError):
        if hasattr(os, "sysconf"):
            try:
                return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
            except (OSError, ValueError):
                return None
    return None


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, AttributeError, OSError):
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except (AttributeError, OSError, ValueError):
            pass
    try:
        import resource

        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return maximum if sys.platform == "darwin" else maximum * 1024
    except (ImportError, AttributeError, OSError):
        return None


def audit_genomic_training_residency(
    source_path: str | Path, *, candidate_rows: int, candidate_string_bytes_estimate: int
) -> dict[str, Any]:
    """Machine-audit the known full-residency constructs in genomic_training.py."""

    path = Path(source_path).resolve()
    text = path.read_text(encoding="utf-8")
    patterns = {
        "full_candidate_table_loaded": 'candidates = normalise_candidates(_read_table(paths["candidates"]))',
        "two_float64_probability_sums": "np.zeros(len(candidates), np.float64)",
        "two_int16_probability_counts": "np.zeros(len(candidates), np.int16)",
        "ten_full_float32_fold_arrays": "np.full((N_FOLDS, len(candidates)), np.nan, np.float32)",
        "typed_full_candidate_copy": "typed = candidates.copy()",
        "per_fold_full_candidate_copy": "fold_frame = candidates.copy()",
    }
    evidence = {key: token in text for key, token in patterns.items()}
    if not all(evidence.values()):
        raise GenomicResourceError(
            f"genomic_training.py residency structure changed: "
            f"{[key for key, found in evidence.items() if not found]}"
        )
    n = int(candidate_rows)
    numeric_arrays = {
        "probability_sums_float64_bytes": 2 * n * 8,
        "probability_counts_int16_bytes": 2 * n * 2,
        "probability_by_fold_float32_bytes": 2 * FOLDS * n * 4,
    }
    numeric_total = int(sum(numeric_arrays.values()))
    return {
        "source_path": str(path),
        "source_sha256": artifact_sha256(path),
        "evidence": evidence,
        "candidate_rows": n,
        "candidate_string_table_estimated_bytes": int(candidate_string_bytes_estimate),
        "fixed_full_length_numeric_array_bytes": numeric_arrays,
        "fixed_full_length_numeric_array_total_bytes": numeric_total,
        "minimum_resident_bytes_before_typed_reason_call_core_temps": int(
            numeric_total + candidate_string_bytes_estimate
        ),
        "full_candidate_and_fold_arrays_resident": True,
        "typed_and_fold_candidate_copies_present": True,
        "resource_safe_adapter_required": True,
    }


def _canonical_partition(
    frame: pd.DataFrame, cancer: str, *, sort_keys: bool = True
) -> pd.DataFrame:
    if missing := sorted(set(TARGET_KEYS) - set(frame.columns)):
        raise GenomicResourceError(f"Candidate partition lacks keys: {missing}")
    result = frame[list(TARGET_KEYS)].copy()
    result["cancer_id"] = result.cancer_id.astype(str).str.upper()
    result["lncrna_id"] = result.lncrna_id.astype(str)
    result["pathway_id"] = result.pathway_id.astype(str)
    if not result.cancer_id.eq(cancer).all():
        raise GenomicResourceError(f"Candidate filter returned another cancer for {cancer}")
    if result.duplicated(list(TARGET_KEYS)).any():
        raise GenomicResourceError(f"Candidate partition {cancer} has duplicate exact keys")
    if sort_keys:
        result = result.sort_values(list(TARGET_KEYS), kind="stable")
    return result.reset_index(drop=True)


def _read_candidate_cancer(path: Path, cancer: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    return pq.read_table(
        path, columns=list(TARGET_KEYS), filters=[("cancer_id", "=", cancer)]
    ).to_pandas()


def stage_genomic_cancer_memmaps(
    *, candidate_authority_path: str | Path, genomic_training_source_path: str | Path,
    cnv_streaming_success_path: str | Path,
    cnv_streaming_success_sha256: str,
    budget_contract_path: str | Path, output_root: str | Path,
    enforce_host_budget: bool = True,
) -> dict[str, Any]:
    """Partition candidate strings by cancer and initialize bounded memmaps."""

    candidate_path = Path(candidate_authority_path).resolve()
    source_path = Path(genomic_training_source_path).resolve()
    cnv_success_path = Path(cnv_streaming_success_path).resolve()
    budget_path = Path(budget_contract_path).resolve()
    for label, path in (
        ("candidate authority", candidate_path),
        ("genomic training source", source_path),
        ("CNV streaming aggregate SUCCESS", cnv_success_path),
        ("resource budget", budget_path),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise GenomicResourceError(f"Missing {label}: {path}")
    budget = _validate_budget(_read_json(budget_path, "genomic resource budget"))
    bindings = budget["input_bindings"]
    candidate_binding = bindings["candidate_authority"]
    cnv_binding = bindings["cnv_streaming_aggregate_success"]
    _require_exact_path(
        candidate_path, Path(str(candidate_binding["path"])), "Candidate authority"
    )
    candidate_sha256 = artifact_sha256(candidate_path)
    if candidate_sha256 != str(candidate_binding["sha256"]).lower():
        raise GenomicResourceError("Candidate authority SHA-256 drift")
    _require_exact_path(
        cnv_success_path,
        Path(str(cnv_binding["path"])),
        "CNV streaming aggregate SUCCESS",
    )
    cancers = [str(item).upper() for item in budget["scope"]["formal_cancers"]]
    cnv_gate = _validate_cnv_aggregate_gate(
        cnv_success_path,
        expected_sha256=cnv_streaming_success_sha256,
        expected_cancers=cancers,
    )
    output = Path(output_root).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() or staging.exists():
        raise GenomicResourceError(f"Genomic resource staging refuses reuse: {output}")
    available_memory = _available_memory_bytes()
    free_disk = int(shutil.disk_usage(output.parent).free)
    downstream = budget["downstream_budget"]
    host_checks = {
        "available_memory_known": available_memory is not None,
        "available_memory_meets_budget": (
            available_memory is not None
            and available_memory >= int(downstream["minimum_host_available_memory_bytes"])
        ),
        "free_disk_meets_budget": free_disk >= int(downstream["minimum_host_free_disk_bytes"]),
    }
    if enforce_host_budget and not all(host_checks.values()):
        raise GenomicResourceError(f"Host resource budget failed before output creation: {host_checks}")

    rows_per_cancer = int(budget["scope"]["candidate_rows_per_cancer"])
    expected_rows = int(budget["scope"]["candidate_rows"])
    staging.mkdir(parents=True)
    partition_root = staging / "candidate_partitions"
    partition_root.mkdir()
    records: list[dict[str, Any]] = []
    global_digest = hashlib.sha256()
    row_start = 0
    candidate_memory_sum = 0
    candidate_memory_max = 0
    observed_process_rss = _process_rss_bytes()
    observed_process_rss_max = observed_process_rss
    for cancer in cancers:
        frame = _canonical_partition(_read_candidate_cancer(candidate_path, cancer), cancer)
        local_rss = _process_rss_bytes()
        if local_rss is not None:
            observed_process_rss_max = max(observed_process_rss_max or 0, local_rss)
        if len(frame) != rows_per_cancer:
            raise GenomicResourceError(
                f"{cancer} candidate rows {len(frame)} != declared {rows_per_cancer}"
            )
        memory_bytes = int(frame.memory_usage(index=True, deep=True).sum())
        candidate_memory_sum += memory_bytes
        candidate_memory_max = max(candidate_memory_max, memory_bytes)
        for row in frame[list(TARGET_KEYS)].itertuples(index=False):
            global_digest.update("\t".join(map(str, row)).encode("utf-8") + b"\n")
        partition = partition_root / f"cancer_id={cancer}"
        partition.mkdir()
        path = partition / "part-0.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        local_rss = _process_rss_bytes()
        if local_rss is not None:
            observed_process_rss_max = max(observed_process_rss_max or 0, local_rss)
        row_stop = row_start + len(frame)
        records.append(
            {
                "cancer_id": cancer,
                "global_row_start": row_start,
                "global_row_stop_exclusive": row_stop,
                "rows": len(frame),
                "candidate_pandas_deep_bytes": memory_bytes,
                "path": str(output / "candidate_partitions" / f"cancer_id={cancer}" / "part-0.parquet"),
                "bytes": int(path.stat().st_size),
                "sha256": artifact_sha256(path),
            }
        )
        row_start = row_stop
    if row_start != expected_rows:
        raise GenomicResourceError("Cancer candidate partitions do not cover the declared scope")
    ordered_key_sha256 = global_digest.hexdigest()
    if ordered_key_sha256 != str(
        candidate_binding["ordered_exact_candidate_key_sha256"]
    ).lower():
        raise GenomicResourceError("Candidate ordered exact-key SHA-256 drift")

    memmap_root = staging / "memmaps"
    memmap_root.mkdir()
    shape = (len(MODALITIES), FOLDS, expected_rows)
    probability_path = memmap_root / "fold_probability.float32.npy"
    available_path = memmap_root / "fold_available.bool.npy"
    processed_path = memmap_root / "fold_processed.bool.npy"
    probability = np.lib.format.open_memmap(
        probability_path, mode="w+", dtype=np.float32, shape=shape
    )
    available = np.lib.format.open_memmap(
        available_path, mode="w+", dtype=np.bool_, shape=shape
    )
    processed = np.lib.format.open_memmap(
        processed_path, mode="w+", dtype=np.bool_, shape=shape
    )
    chunk = max(1, rows_per_cancer)
    for start in range(0, expected_rows, chunk):
        stop = min(expected_rows, start + chunk)
        probability[:, :, start:stop] = np.nan
        available[:, :, start:stop] = False
        processed[:, :, start:stop] = False
    probability.flush()
    available.flush()
    processed.flush()
    del probability, available, processed
    memmap_records = {
        "fold_probability": {
            "path": str(output / "memmaps" / probability_path.name),
            "dtype": "float32",
            "shape": list(shape),
            "bytes": int(probability_path.stat().st_size),
            "initial_sha256": artifact_sha256(probability_path),
        },
        "fold_available": {
            "path": str(output / "memmaps" / available_path.name),
            "dtype": "bool",
            "shape": list(shape),
            "bytes": int(available_path.stat().st_size),
            "initial_sha256": artifact_sha256(available_path),
        },
        "fold_processed": {
            "path": str(output / "memmaps" / processed_path.name),
            "dtype": "bool",
            "shape": list(shape),
            "bytes": int(processed_path.stat().st_size),
            "initial_sha256": artifact_sha256(processed_path),
        },
    }
    memmap_bytes = int(sum(item["bytes"] for item in memmap_records.values()))
    adapter = budget["adapter_budget"]
    if memmap_bytes > int(adapter["max_memmap_bytes"]):
        raise GenomicResourceError("Initialized memmaps exceed the declared disk budget")
    partition_bytes = int(sum(item["bytes"] for item in records))
    if memmap_bytes + partition_bytes > int(adapter["max_staging_disk_bytes"]):
        raise GenomicResourceError("Resource stage exceeds the declared staging disk budget")
    if (
        observed_process_rss_max is not None
        and observed_process_rss_max > int(adapter["max_declared_adapter_peak_rss_bytes"])
    ):
        raise GenomicResourceError(
            "Observed adapter RSS exceeds the declared adapter process budget"
        )
    source_audit = audit_genomic_training_residency(
        source_path,
        candidate_rows=expected_rows,
        candidate_string_bytes_estimate=candidate_memory_sum,
    )
    manifest_path = staging / "RESOURCE_MANIFEST.json"
    manifest = {
        "format": STAGE_FORMAT,
        "status": "RESOURCE_STAGING_READY_NO_PREDICTIONS",
        "stage_root": str(output),
        "candidate_authority": {
            "path": str(candidate_path),
            "bytes": int(candidate_path.stat().st_size),
            "sha256": candidate_sha256,
            "ordered_exact_candidate_key_sha256": ordered_key_sha256,
        },
        "cnv_streaming_aggregate_success": cnv_gate,
        "genomic_training_source_audit": source_audit,
        "resource_budget": {
            "path": str(budget_path),
            "sha256": artifact_sha256(budget_path),
            "resource_budget_id": budget["resource_budget_id"],
        },
        "candidate_rows": expected_rows,
        "candidate_rows_per_cancer": rows_per_cancer,
        "formal_cancers": cancers,
        "ordered_exact_candidate_key_sha256": ordered_key_sha256,
        "candidate_partitions": records,
        "candidate_pandas_deep_bytes_estimated_full": candidate_memory_sum,
        "max_single_cancer_candidate_pandas_deep_bytes": candidate_memory_max,
        "memmaps": memmap_records,
        "memmap_bytes": memmap_bytes,
        "partition_bytes": partition_bytes,
        "host_resource_observation": {
            "available_memory_bytes": available_memory,
            "free_disk_bytes": free_disk,
            "process_rss_bytes_at_start": observed_process_rss,
            "sampled_process_rss_max_bytes": observed_process_rss_max,
            "checks": host_checks,
            "enforced": bool(enforce_host_budget),
        },
        "resident_candidate_scope": "ONE_CANCER_AT_A_TIME",
        "full_fold_arrays_resident_in_ram": False,
        "memmap_initial_state": "ALL_UNPROCESSED_TYPED_FALSE_PROBABILITY_NAN",
        "old_predictions_or_checkpoints_used": False,
        "frozen_cnv_r3_modified": False,
        "formal_training_started": False,
    }
    _atomic_json(manifest_path, manifest)
    success = {
        "format": STAGE_FORMAT,
        "status": "RESOURCE_STAGING_READY_NO_PREDICTIONS",
        "stage_root": str(output),
        "resource_manifest_path": str(output / manifest_path.name),
        "resource_manifest_sha256": artifact_sha256(manifest_path),
        "candidate_rows": expected_rows,
        "candidate_authority_sha256": candidate_sha256,
        "ordered_exact_candidate_key_sha256": ordered_key_sha256,
        "cnv_streaming_success_sha256": cnv_gate["sha256"],
        "memmap_bytes": memmap_bytes,
        "resident_candidate_scope": "ONE_CANCER_AT_A_TIME",
        "formal_training_started": False,
        "predictions_complete": False,
        "success_written_last": True,
    }
    _atomic_json(staging / "SUCCESS.json", success)
    os.replace(staging, output)
    return success


def _validate_stage_manifest_paths(
    stage_root: Path,
    manifest: Mapping[str, Any],
    *,
    verify_external_hashes: bool,
) -> dict[str, Any]:
    if (
        manifest.get("format") != STAGE_FORMAT
        or manifest.get("status") != "RESOURCE_STAGING_READY_NO_PREDICTIONS"
    ):
        raise GenomicResourceError("Genomic resource manifest format/status drift")
    _require_exact_path(
        Path(str(manifest.get("stage_root", ""))), stage_root, "Manifest stage root"
    )
    budget_record = manifest.get("resource_budget", {})
    budget_path = Path(str(budget_record.get("path", ""))).resolve()
    if not budget_path.is_file() or artifact_sha256(budget_path) != budget_record.get("sha256"):
        raise GenomicResourceError("Genomic resource budget path/hash drift")
    budget = _validate_budget(_read_json(budget_path, "bound genomic resource budget"))
    candidate_binding = budget["input_bindings"]["candidate_authority"]
    candidate_record = manifest.get("candidate_authority", {})
    candidate_path = Path(str(candidate_record.get("path", ""))).resolve()
    _require_exact_path(
        candidate_path, Path(str(candidate_binding["path"])), "Bound candidate authority"
    )
    if (
        candidate_record.get("sha256") != str(candidate_binding["sha256"]).lower()
        or candidate_record.get("ordered_exact_candidate_key_sha256")
        != str(candidate_binding["ordered_exact_candidate_key_sha256"]).lower()
        or manifest.get("ordered_exact_candidate_key_sha256")
        != str(candidate_binding["ordered_exact_candidate_key_sha256"]).lower()
    ):
        raise GenomicResourceError("Bound candidate authority hash contract drift")
    cnv_binding = budget["input_bindings"]["cnv_streaming_aggregate_success"]
    cnv_record = manifest.get("cnv_streaming_aggregate_success", {})
    cnv_path = Path(str(cnv_record.get("path", ""))).resolve()
    _require_exact_path(
        cnv_path, Path(str(cnv_binding["path"])), "Bound CNV streaming aggregate SUCCESS"
    )
    if not _SHA256.fullmatch(str(cnv_record.get("sha256", ""))):
        raise GenomicResourceError("Bound CNV streaming aggregate hash is invalid")
    if verify_external_hashes:
        if not candidate_path.is_file() or artifact_sha256(candidate_path) != candidate_record["sha256"]:
            raise GenomicResourceError("Bound candidate authority content drift")
        if not cnv_path.is_file() or artifact_sha256(cnv_path) != cnv_record["sha256"]:
            raise GenomicResourceError("Bound CNV streaming aggregate content drift")

    expected_cancers = [str(item).upper() for item in budget["scope"]["formal_cancers"]]
    records = manifest.get("candidate_partitions", [])
    if not isinstance(records, list) or [item.get("cancer_id") for item in records] != expected_cancers:
        raise GenomicResourceError("Candidate partition scope/order drift")
    for record in records:
        cancer = str(record["cancer_id"])
        path = Path(str(record.get("path", ""))).resolve()
        expected = stage_root / "candidate_partitions" / f"cancer_id={cancer}" / "part-0.parquet"
        _require_inside(path, stage_root, "Candidate partition")
        _require_exact_path(path, expected, "Candidate partition")
    expected_memmaps = {
        "fold_probability": "fold_probability.float32.npy",
        "fold_available": "fold_available.bool.npy",
        "fold_processed": "fold_processed.bool.npy",
    }
    memmaps = manifest.get("memmaps", {})
    if set(memmaps) != set(expected_memmaps):
        raise GenomicResourceError("Genomic memmap inventory drift")
    for key, filename in expected_memmaps.items():
        path = Path(str(memmaps[key].get("path", ""))).resolve()
        _require_inside(path, stage_root, f"{key} memmap")
        _require_exact_path(path, stage_root / "memmaps" / filename, f"{key} memmap")
    return budget


def _load_stage(
    success_path: Path, *, verify_external_hashes: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    success_path = success_path.resolve()
    stage_root = success_path.parent
    _require_exact_path(success_path, stage_root / "SUCCESS.json", "Resource stage SUCCESS")
    success = _read_json(success_path, "genomic resource staging SUCCESS")
    if success.get("format") != STAGE_FORMAT or success.get("status") != "RESOURCE_STAGING_READY_NO_PREDICTIONS":
        raise GenomicResourceError("Genomic resource stage is not writable staging")
    _require_exact_path(
        Path(str(success.get("stage_root", ""))), stage_root, "SUCCESS stage root"
    )
    manifest_path = Path(str(success.get("resource_manifest_path", ""))).resolve()
    _require_inside(manifest_path, stage_root, "Resource manifest")
    _require_exact_path(
        manifest_path, stage_root / "RESOURCE_MANIFEST.json", "Resource manifest"
    )
    if artifact_sha256(manifest_path) != success.get("resource_manifest_sha256"):
        raise GenomicResourceError("Genomic resource manifest hash drift")
    manifest = _read_json(manifest_path, "genomic resource manifest")
    _validate_stage_manifest_paths(
        stage_root, manifest, verify_external_hashes=verify_external_hashes
    )
    for key in (
        "candidate_rows",
        "candidate_authority_sha256",
        "ordered_exact_candidate_key_sha256",
        "cnv_streaming_success_sha256",
    ):
        expected = {
            "candidate_rows": manifest.get("candidate_rows"),
            "candidate_authority_sha256": manifest.get("candidate_authority", {}).get("sha256"),
            "ordered_exact_candidate_key_sha256": manifest.get("ordered_exact_candidate_key_sha256"),
            "cnv_streaming_success_sha256": manifest.get("cnv_streaming_aggregate_success", {}).get("sha256"),
        }[key]
        if success.get(key) != expected:
            raise GenomicResourceError(f"Resource SUCCESS/manifest binding drift: {key}")
    return success, manifest


def write_genomic_memmap_partition(
    *, staging_success_path: str | Path, modality: str, patient_fold: int,
    cancer_id: str, candidate_keys: pd.DataFrame | str | Path,
    probability: Sequence[float], available: Sequence[bool],
) -> dict[str, Any]:
    """Write one keyed cancer/modality/fold slice; overwrites are forbidden.

    A bare probability vector is deliberately insufficient: callers must also
    supply the exact candidate keys in vector order.  This prevents a valid-
    looking but silently row-shifted prediction from entering the memmap.
    """

    success_path = Path(staging_success_path).resolve()
    _, manifest = _load_stage(success_path)
    modality = str(modality).lower()
    cancer = str(cancer_id).upper()
    if modality not in MODALITIES or int(patient_fold) not in range(FOLDS):
        raise GenomicResourceError("Memmap write modality/fold is invalid")
    lookup = {item["cancer_id"]: item for item in manifest["candidate_partitions"]}
    if cancer not in lookup:
        raise GenomicResourceError(f"Memmap write cancer is outside scope: {cancer}")
    record = lookup[cancer]
    start = int(record["global_row_start"])
    stop = int(record["global_row_stop_exclusive"])
    expected_key_path = Path(str(record["path"]))
    if artifact_sha256(expected_key_path) != record["sha256"]:
        raise GenomicResourceError(f"Candidate partition hash drift: {cancer}")
    supplied_key_frame = (
        pd.read_parquet(Path(candidate_keys), columns=list(TARGET_KEYS))
        if isinstance(candidate_keys, (str, Path))
        else candidate_keys
    )
    if not isinstance(supplied_key_frame, pd.DataFrame):
        raise GenomicResourceError("Memmap write candidate_keys must be a table or Parquet path")
    supplied_keys = _canonical_partition(supplied_key_frame, cancer, sort_keys=False)
    expected_keys = pd.read_parquet(expected_key_path, columns=list(TARGET_KEYS))
    if not supplied_keys[list(TARGET_KEYS)].equals(expected_keys[list(TARGET_KEYS)]):
        raise GenomicResourceError(
            f"Memmap write exact candidate key/order drift for {cancer}"
        )
    values = np.asarray(probability, dtype=np.float32)
    availability = np.asarray(available)
    if values.ndim != 1 or availability.ndim != 1:
        raise GenomicResourceError("Memmap probability and availability must be vectors")
    if availability.dtype != np.bool_:
        if not all(isinstance(value, (bool, np.bool_)) for value in availability.tolist()):
            raise GenomicResourceError("Memmap availability must be strictly boolean")
        availability = availability.astype(np.bool_)
    if len(values) != stop - start or len(availability) != stop - start:
        raise GenomicResourceError("Memmap write length differs from the cancer partition")
    if (
        not np.isfinite(values[availability]).all()
        or ((values[availability] < 0) | (values[availability] > 1)).any()
        or not np.isnan(values[~availability]).all()
    ):
        raise GenomicResourceError("Memmap probability violates typed availability")
    paths = {key: Path(value["path"]) for key, value in manifest["memmaps"].items()}
    receipt_path = (
        success_path.parent
        / "write_receipts"
        / f"modality={modality}"
        / f"patient_fold={int(patient_fold)}"
        / f"cancer_id={cancer}.json"
    )
    if receipt_path.exists():
        raise GenomicResourceError(f"Memmap write receipt already exists: {receipt_path}")
    probability_map = np.load(paths["fold_probability"], mmap_mode="r+")
    available_map = np.load(paths["fold_available"], mmap_mode="r+")
    processed_map = np.load(paths["fold_processed"], mmap_mode="r+")
    modality_index = MODALITIES.index(modality)
    fold = int(patient_fold)
    if processed_map[modality_index, fold, start:stop].any():
        raise GenomicResourceError("Memmap cancer/modality/fold slice was already processed")
    probability_map[modality_index, fold, start:stop] = values
    available_map[modality_index, fold, start:stop] = availability
    processed_map[modality_index, fold, start:stop] = True
    probability_map.flush()
    available_map.flush()
    processed_map.flush()
    del probability_map, available_map, processed_map
    receipt = {
        "format": STAGE_FORMAT,
        "status": "PARTITION_WRITTEN",
        "modality": modality,
        "patient_fold": fold,
        "cancer_id": cancer,
        "global_row_start": start,
        "global_row_stop_exclusive": stop,
        "rows": stop - start,
        "available_rows": int(availability.sum()),
        "probability_slice_sha256": _array_sha256(values),
        "availability_slice_sha256": _array_sha256(availability),
        "candidate_partition_sha256": record["sha256"],
        "overwrite_permitted": False,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def seal_genomic_memmap_predictions(
    *, staging_success_path: str | Path, output_path: str | Path | None = None
) -> dict[str, Any]:
    """Seal memmaps only after every cancer/modality/fold slice was processed."""

    success_path = Path(staging_success_path).resolve()
    _, manifest = _load_stage(success_path, verify_external_hashes=True)
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else success_path.parent / "PREDICTIONS_READY.json"
    )
    _require_inside(destination, success_path.parent, "Prediction seal")
    _require_exact_path(
        destination,
        success_path.parent / "PREDICTIONS_READY.json",
        "Prediction seal",
    )
    if destination.exists():
        raise GenomicResourceError(f"Prediction seal refuses output reuse: {destination}")
    paths = {key: Path(value["path"]) for key, value in manifest["memmaps"].items()}
    probability = np.load(paths["fold_probability"], mmap_mode="r")
    available = np.load(paths["fold_available"], mmap_mode="r")
    processed = np.load(paths["fold_processed"], mmap_mode="r")
    expected_shape = (len(MODALITIES), FOLDS, int(manifest["candidate_rows"]))
    if (
        tuple(probability.shape) != expected_shape
        or tuple(available.shape) != expected_shape
        or tuple(processed.shape) != expected_shape
        or probability.dtype != np.dtype("float32")
        or available.dtype != np.dtype("bool")
        or processed.dtype != np.dtype("bool")
    ):
        raise GenomicResourceError("Genomic memmap shape/dtype drift")
    for record in manifest["candidate_partitions"]:
        candidate_path = Path(str(record["path"])).resolve()
        if artifact_sha256(candidate_path) != record["sha256"]:
            raise GenomicResourceError(
                f"Candidate partition hash drift: {record['cancer_id']}"
            )
    counts: list[dict[str, Any]] = []
    expected_receipt_paths: set[Path] = set()
    for modality_index, modality in enumerate(MODALITIES):
        for fold in range(FOLDS):
            processed_rows = 0
            available_rows = 0
            for record in manifest["candidate_partitions"]:
                start = int(record["global_row_start"])
                stop = int(record["global_row_stop_exclusive"])
                local_processed = np.asarray(processed[modality_index, fold, start:stop])
                local_available = np.asarray(available[modality_index, fold, start:stop])
                local_probability = np.asarray(probability[modality_index, fold, start:stop])
                if not local_processed.all():
                    raise GenomicResourceError(
                        f"Unprocessed memmap rows remain: {modality}/fold={fold}"
                    )
                if (
                    not np.isfinite(local_probability[local_available]).all()
                    or ((local_probability[local_available] < 0) | (local_probability[local_available] > 1)).any()
                    or not np.isnan(local_probability[~local_available]).all()
                ):
                    raise GenomicResourceError(
                        f"Memmap typed availability drift: {modality}/fold={fold}"
                    )
                receipt_path = (
                    success_path.parent
                    / "write_receipts"
                    / f"modality={modality}"
                    / f"patient_fold={fold}"
                    / f"cancer_id={record['cancer_id']}.json"
                )
                expected_receipt_paths.add(receipt_path.resolve())
                receipt = _read_json(receipt_path, "memmap write receipt")
                expected_receipt_fields = {
                    "format": STAGE_FORMAT,
                    "status": "PARTITION_WRITTEN",
                    "modality": modality,
                    "patient_fold": fold,
                    "cancer_id": record["cancer_id"],
                    "global_row_start": start,
                    "global_row_stop_exclusive": stop,
                    "rows": stop - start,
                    "available_rows": int(local_available.sum()),
                    "probability_slice_sha256": _array_sha256(local_probability),
                    "availability_slice_sha256": _array_sha256(local_available),
                    "candidate_partition_sha256": record["sha256"],
                    "overwrite_permitted": False,
                }
                if any(receipt.get(key) != value for key, value in expected_receipt_fields.items()):
                    raise GenomicResourceError(
                        f"Memmap receipt/content drift: {modality}/fold={fold}/"
                        f"{record['cancer_id']}"
                    )
                processed_rows += int(local_processed.sum())
                available_rows += int(local_available.sum())
            counts.append(
                {
                    "modality": modality,
                    "patient_fold": fold,
                    "processed_rows": processed_rows,
                    "available_rows": available_rows,
                }
            )
    del probability, available, processed
    expected_receipts = len(expected_receipt_paths)
    actual_receipt_paths = {
        path.resolve()
        for path in (success_path.parent / "write_receipts").rglob("cancer_id=*.json")
    }
    if actual_receipt_paths != expected_receipt_paths:
        raise GenomicResourceError(
            "Memmap receipt set differs from the exact cancer/modality/fold contract"
        )
    memmaps = {
        key: {
            **record,
            "sha256": artifact_sha256(record["path"]),
        }
        for key, record in manifest["memmaps"].items()
    }
    ready = {
        "format": READY_FORMAT,
        "status": "PREDICTIONS_COMPLETE",
        "resource_manifest_path": str(success_path.parent / "RESOURCE_MANIFEST.json"),
        "resource_manifest_sha256": artifact_sha256(success_path.parent / "RESOURCE_MANIFEST.json"),
        "candidate_rows": int(manifest["candidate_rows"]),
        "ordered_exact_candidate_key_sha256": manifest["ordered_exact_candidate_key_sha256"],
        "memmaps": memmaps,
        "fold_counts": counts,
        "write_receipts": expected_receipts,
        "all_rows_processed": True,
        "typed_unavailable_never_zero": True,
        "frozen_cnv_r3_modified": False,
        "success_written_last": True,
    }
    _atomic_json(destination, ready)
    return ready


def reassemble_genomic_memmap_outputs(
    *, predictions_ready_path: str | Path, output_root: str | Path,
    training_run_id: str,
) -> dict[str, Any]:
    """Stream exact-order combined and five fold outputs from sealed memmaps."""

    ready_path = Path(predictions_ready_path).resolve()
    stage_root = ready_path.parent
    _require_inside(ready_path, stage_root, "Sealed predictions")
    _require_exact_path(
        ready_path, stage_root / "PREDICTIONS_READY.json", "Sealed predictions"
    )
    ready = _read_json(ready_path, "sealed genomic memmap predictions")
    if ready.get("format") != READY_FORMAT or ready.get("status") != "PREDICTIONS_COMPLETE":
        raise GenomicResourceError("Genomic memmap predictions are not sealed")
    manifest_path = Path(str(ready.get("resource_manifest_path", ""))).resolve()
    _require_inside(manifest_path, stage_root, "Sealed resource manifest")
    _require_exact_path(
        manifest_path,
        stage_root / "RESOURCE_MANIFEST.json",
        "Sealed resource manifest",
    )
    if artifact_sha256(manifest_path) != ready.get("resource_manifest_sha256"):
        raise GenomicResourceError("Sealed resource manifest hash drift")
    manifest = _read_json(manifest_path, "sealed resource manifest")
    _validate_stage_manifest_paths(
        stage_root, manifest, verify_external_hashes=True
    )
    if (
        ready.get("candidate_rows") != manifest.get("candidate_rows")
        or ready.get("ordered_exact_candidate_key_sha256")
        != manifest.get("ordered_exact_candidate_key_sha256")
    ):
        raise GenomicResourceError("Sealed prediction/candidate binding drift")
    output = Path(output_root).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() or staging.exists():
        raise GenomicResourceError(f"Genomic memmap reassembly refuses output reuse: {output}")
    if set(ready.get("memmaps", {})) != set(manifest.get("memmaps", {})):
        raise GenomicResourceError("Sealed genomic memmap inventory drift")
    memmap_paths = {}
    for key, record in ready["memmaps"].items():
        path = Path(str(record["path"])).resolve()
        expected_path = Path(str(manifest["memmaps"][key]["path"])).resolve()
        _require_inside(path, stage_root, f"Sealed {key} memmap")
        _require_exact_path(path, expected_path, f"Sealed {key} memmap")
        if artifact_sha256(path) != record.get("sha256"):
            raise GenomicResourceError(f"Sealed genomic memmap hash drift: {key}")
        memmap_paths[key] = path
    probability = np.load(memmap_paths["fold_probability"], mmap_mode="r")
    available = np.load(memmap_paths["fold_available"], mmap_mode="r")
    processed = np.load(memmap_paths["fold_processed"], mmap_mode="r")
    expected_shape = (len(MODALITIES), FOLDS, int(manifest["candidate_rows"]))
    if (
        tuple(probability.shape) != expected_shape
        or tuple(available.shape) != expected_shape
        or tuple(processed.shape) != expected_shape
        or probability.dtype != np.dtype("float32")
        or available.dtype != np.dtype("bool")
        or processed.dtype != np.dtype("bool")
    ):
        raise GenomicResourceError("Sealed genomic memmap shape/dtype drift")
    chunk = int(manifest["candidate_rows_per_cancer"])
    for start in range(0, expected_shape[2], chunk):
        stop = min(expected_shape[2], start + chunk)
        if not np.asarray(processed[:, :, start:stop]).all():
            raise GenomicResourceError("Sealed processed memmap contains false rows")
    staging.mkdir(parents=True)
    combined_path = staging / "mutation_cnv_typed_predictions.parquet"
    fold_root = staging / "patient_fold_oof_predictions"
    fold_root.mkdir()
    import pyarrow as pa
    import pyarrow.parquet as pq

    combined_writer = None
    fold_writers: list[Any] = [None] * FOLDS
    fold_paths: list[Path] = []
    for fold in range(FOLDS):
        partition = fold_root / f"patient_fold={fold}"
        partition.mkdir()
        fold_paths.append(partition / "part-0.parquet")
    key_digest = hashlib.sha256()
    try:
        for record in manifest["candidate_partitions"]:
            candidate_path = Path(str(record["path"]))
            if artifact_sha256(candidate_path) != record["sha256"]:
                raise GenomicResourceError(f"Candidate partition hash drift: {candidate_path}")
            keys = pd.read_parquet(candidate_path, columns=list(TARGET_KEYS))
            start = int(record["global_row_start"])
            stop = int(record["global_row_stop_exclusive"])
            if len(keys) != stop - start:
                raise GenomicResourceError("Candidate partition interval drift")
            for row in keys[list(TARGET_KEYS)].itertuples(index=False):
                key_digest.update("\t".join(map(str, row)).encode("utf-8") + b"\n")
            combined = keys.copy()
            modality_fold_values: dict[str, np.ndarray] = {}
            modality_fold_available: dict[str, np.ndarray] = {}
            for modality_index, modality in enumerate(MODALITIES):
                fold_values = np.asarray(probability[modality_index, :, start:stop]).astype(
                    np.float32, copy=True
                )
                fold_available = np.asarray(available[modality_index, :, start:stop]).astype(
                    bool, copy=True
                )
                count = fold_available.sum(axis=0).astype(np.int16)
                total = np.where(fold_available, fold_values, 0.0).sum(axis=0, dtype=np.float64)
                mean = np.divide(
                    total,
                    count,
                    out=np.full(stop - start, np.nan, dtype=np.float64),
                    where=count > 0,
                )
                combined[f"{modality}_context_probability"] = mean.astype(np.float32)
                combined[f"{modality}_available"] = count > 0
                combined[f"{modality}_unavailable_reason"] = pd.Series(
                    np.where(
                        count > 0,
                        None,
                        f"{modality.upper()}_NO_OOF_PREDICTION_FROM_MEMMAP_STAGE",
                    ),
                    dtype="string",
                )
                combined[f"{modality}_patient_folds_with_prediction"] = count
                modality_fold_values[modality] = fold_values
                modality_fold_available[modality] = fold_available
            combined["analysis_version"] = "CancerLncAtlas_V3.2_FULL_MULTITASK"
            combined["training_run_id"] = str(training_run_id)
            combined["module_id"] = "genomic_context"
            combined["changes_primary_ranking"] = False
            table = pa.Table.from_pandas(combined, preserve_index=False)
            if combined_writer is None:
                combined_writer = pq.ParquetWriter(combined_path, table.schema, compression="zstd")
            combined_writer.write_table(table, row_group_size=len(combined))
            for fold in range(FOLDS):
                fold_frame = keys.copy()
                fold_frame["patient_fold_id"] = fold
                for modality in MODALITIES:
                    fold_frame[f"{modality}_context_probability"] = modality_fold_values[modality][fold]
                    fold_frame[f"{modality}_available"] = modality_fold_available[modality][fold]
                fold_table = pa.Table.from_pandas(fold_frame, preserve_index=False)
                if fold_writers[fold] is None:
                    fold_writers[fold] = pq.ParquetWriter(
                        fold_paths[fold], fold_table.schema, compression="zstd"
                    )
                fold_writers[fold].write_table(fold_table, row_group_size=len(fold_frame))
    finally:
        if combined_writer is not None:
            combined_writer.close()
        for writer in fold_writers:
            if writer is not None:
                writer.close()
        del probability, available, processed
    if key_digest.hexdigest() != manifest["ordered_exact_candidate_key_sha256"]:
        raise GenomicResourceError("Reassembled exact candidate order drift")
    expected_rows = int(manifest["candidate_rows"])
    if combined_writer is None or pq.ParquetFile(combined_path).metadata.num_rows != expected_rows:
        raise GenomicResourceError("Reassembled combined prediction rows are incomplete")
    fold_records = []
    for fold, path in enumerate(fold_paths):
        if fold_writers[fold] is None or pq.ParquetFile(path).metadata.num_rows != expected_rows:
            raise GenomicResourceError(f"Reassembled fold {fold} rows are incomplete")
        fold_records.append(
            {
                "patient_fold": fold,
                "path": str(output / "patient_fold_oof_predictions" / f"patient_fold={fold}" / "part-0.parquet"),
                "rows": expected_rows,
                "sha256": artifact_sha256(path),
            }
        )
    lineage_path = staging / "LINEAGE.json"
    lineage = {
        "format": REASSEMBLY_FORMAT,
        "status": "SUCCESS",
        "training_run_id": str(training_run_id),
        "candidate_rows": expected_rows,
        "ordered_exact_candidate_key_sha256": key_digest.hexdigest(),
        "combined_predictions": {
            "path": str(output / combined_path.name),
            "sha256": artifact_sha256(combined_path),
        },
        "patient_fold_predictions": fold_records,
        "resident_candidate_scope": "ONE_CANCER_AT_A_TIME",
        "fold_arrays_read_from_memmap": True,
        "frozen_cnv_r3_modified": False,
    }
    _atomic_json(lineage_path, lineage)
    success = {
        "format": REASSEMBLY_FORMAT,
        "status": "SUCCESS",
        "lineage_path": str(output / lineage_path.name),
        "lineage_sha256": artifact_sha256(lineage_path),
        "prediction_path": str(output / combined_path.name),
        "prediction_sha256": artifact_sha256(combined_path),
        "prediction_rows": expected_rows,
        "folds": 5,
        "exact_candidate_order_preserved": True,
        "success_written_last": True,
    }
    _atomic_json(staging / "SUCCESS.json", success)
    os.replace(staging, output)
    return success


__all__ = [
    "GenomicResourceError",
    "artifact_sha256",
    "audit_genomic_training_residency",
    "reassemble_genomic_memmap_outputs",
    "seal_genomic_memmap_predictions",
    "stage_genomic_cancer_memmaps",
    "write_genomic_memmap_partition",
]
