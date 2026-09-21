"""Memory-bounded fresh V3.2 Drug training over a factored candidate universe.

The conceptual target-derived universe can exceed sixty million rows.  This
runner never creates an ``N x core_features`` array for that universe.  It
uses native-assay-only fold eligibility, expands the factored relation in
bounded record batches, trains five fresh private heads, and writes only
available public predictions as atomic cancer partitions.  A missing public
key is explicitly *unavailable*, never a zero or a historical prediction.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import drug_training as base
from .input_lineage import artifact_sha256, audit_input_lineage
from .drug_staging import STAGING_FORMAT, TCGA_CANCERS


FACTORED_CANDIDATE_FORMAT = "CC_HHGT_V3_2_FACTORED_DRUG_CANDIDATES_V1"
STREAMING_PREDICTION_FORMAT = "CC_HHGT_V3_2_DRUG_SPARSE_CELL_LINE_ASSOCIATION_V1"
IMPLICIT_UNAVAILABLE_POLICY = (
    "Keys in the conceptual native-target-derived universe that are absent from "
    "the sparse public dataset are unavailable, never probability zero. They lack "
    "a held-out canonical-cell-line fold with sufficient matched native response, "
    "lncRNA expression, and current V3.2 core support."
)
FORMAL_MIN_PUBLIC_BYTES_PER_ROW = 160
FORMAL_MIN_ATOMIC_WRITE_MULTIPLIER = 2.25
FORMAL_MIN_DISK_RESERVE_BYTES = 2 * 1024**3
FORMAL_MIN_DISK_RESERVE_FRACTION = 0.15
FORMAL_MIN_TARGET_MAPPING_FRACTION = 0.15
FORMAL_MIN_TARGET_MAPPED_DRUGS = 1_000
RAM_INPUT_EXPANSION_MULTIPLIER = 8.0
RAM_FIXED_OVERHEAD_BYTES = 512 * 1024**2
RAM_RESERVE_BYTES = 2 * 1024**3
RAM_MAX_AVAILABLE_FRACTION = 0.50
DUCKDB_MAX_TEMP_BYTES = 128 * 1024**3
DUCKDB_MAX_MEMORY_BYTES = 16 * 1024**3
EXACT_BINDING_KEYS = (
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
)
FORMAL_STAGING_INPUT_ROLES = {
    "gdsc1_native",
    "gdsc2_native",
    "prism_matrix_native",
    "prism_compound_native",
    "cmp_expression_native",
    "cmp_models_native",
    "drugcentral_target_native",
    "hgnc_native",
    "v32_exact_candidates",
    "exact_pathway_membership",
    "exact_release_prediction_proof",
    "exact_release_lineage_proof",
}


@dataclass(frozen=True)
class StreamingDrugTrainingConfig(base.DrugTrainingConfig):
    candidate_chunk_rows: int = 25_000
    private_evaluation_rows_per_fold: int = 50_000
    estimated_public_bytes_per_row: int = 160
    atomic_write_multiplier: float = 2.25
    disk_reserve_bytes: int = 2 * 1024**3
    disk_reserve_fraction: float = 0.15
    preflight_only: bool = False

    def validate(self) -> None:
        super().validate()
        if self.candidate_chunk_rows < 1 or self.private_evaluation_rows_per_fold < 1:
            raise ValueError("Streaming chunk/evaluation limits must be positive")
        if self.estimated_public_bytes_per_row < FORMAL_MIN_PUBLIC_BYTES_PER_ROW:
            raise ValueError(
                f"formal estimated_public_bytes_per_row must be >= "
                f"{FORMAL_MIN_PUBLIC_BYTES_PER_ROW}"
            )
        if self.atomic_write_multiplier < FORMAL_MIN_ATOMIC_WRITE_MULTIPLIER:
            raise ValueError(
                f"formal atomic_write_multiplier must be >= "
                f"{FORMAL_MIN_ATOMIC_WRITE_MULTIPLIER}"
            )
        if self.disk_reserve_bytes < FORMAL_MIN_DISK_RESERVE_BYTES:
            raise ValueError(
                f"formal disk_reserve_bytes must be >= {FORMAL_MIN_DISK_RESERVE_BYTES}"
            )
        if not FORMAL_MIN_DISK_RESERVE_FRACTION <= self.disk_reserve_fraction < 1:
            raise ValueError(
                f"formal disk_reserve_fraction must be >= "
                f"{FORMAL_MIN_DISK_RESERVE_FRACTION} and < 1"
            )
        if not self.require_all_folds:
            raise ValueError("formal streaming Drug training requires all five folds")


@dataclass(frozen=True)
class _AssociationContext:
    cancer_id: str
    datasets: tuple[str, ...]
    expression_by_dataset: Mapping[str, pd.DataFrame]
    response_by_dataset_drug: Mapping[tuple[str, str], pd.Series]


@dataclass(frozen=True)
class _PreparedAssociationCache:
    """One-pass raw-assay linkage used by every train/validation/test context.

    The large response and expression tables are linked to canonical models,
    cancers and folds exactly once.  Context construction below only slices
    these prepared matrices/series by canonical model ID; it never merges or
    pivots either full input again.
    """

    canonical_expression: pd.DataFrame | None
    expression_by_cancer_dataset: Mapping[tuple[str, str], pd.DataFrame]
    response_by_cancer_dataset_drug: Mapping[tuple[str, str, str], pd.Series]
    response_drugs_by_cancer_dataset: Mapping[tuple[str, str], tuple[str, ...]]
    models_by_cancer_dataset: Mapping[tuple[str, str], tuple[str, ...]]
    datasets_by_cancer: Mapping[str, tuple[str, ...]]
    fold_eligible: pd.DataFrame
    expression_fold_coverage: pd.DataFrame
    instrumentation: dict[str, Any]


class _BalancedReservoir:
    """Bounded, deterministic two-class reservoir for head fitting."""

    def __init__(self, maximum: int, seed: int) -> None:
        self.per_class = max(1, int(maximum) // 2)
        self.rng = np.random.default_rng(seed)
        self.parts: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def offer(
        self, core: np.ndarray, domain: np.ndarray, labels: np.ndarray
    ) -> None:
        for label in (0, 1):
            positions = np.flatnonzero(labels == float(label))
            if not len(positions):
                continue
            priority = self.rng.random(len(positions))
            incoming = (
                core[positions], domain[positions], labels[positions], priority
            )
            if label in self.parts:
                previous = self.parts[label]
                joined = tuple(
                    np.concatenate([previous[index], incoming[index]], axis=0)
                    for index in range(4)
                )
            else:
                joined = incoming
            if len(joined[2]) > self.per_class:
                keep = np.argpartition(joined[3], self.per_class - 1)[: self.per_class]
                joined = tuple(value[keep] for value in joined)
            self.parts[label] = joined  # type: ignore[assignment]

    def arrays(self, core_width: int, domain_width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        present = [self.parts[label] for label in (0, 1) if label in self.parts]
        if not present:
            return (
                np.empty((0, core_width), dtype=np.float32),
                np.empty((0, domain_width), dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
        return tuple(
            np.concatenate([part[index] for part in present], axis=0)
            for index in range(3)
        )  # type: ignore[return-value]


class _FrameReservoir:
    def __init__(self, maximum: int, seed: int) -> None:
        self.maximum = int(maximum)
        self.rng = np.random.default_rng(seed)
        self.frame = pd.DataFrame()
        self.priority = np.empty(0, dtype=float)

    def offer(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        priorities = self.rng.random(len(frame))
        combined = pd.concat([self.frame, frame], ignore_index=True)
        combined_priority = np.concatenate([self.priority, priorities])
        if len(combined) > self.maximum:
            keep = np.argpartition(combined_priority, self.maximum - 1)[: self.maximum]
            combined = combined.iloc[keep].reset_index(drop=True)
            combined_priority = combined_priority[keep]
        self.frame = combined
        self.priority = combined_priority


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _canonical_staging_sha(value: Any) -> str:
    """Match drug_staging._canonical_sha, including non-ASCII runtime strings."""

    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _available_memory_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if Path("/proc/meminfo").is_file():
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
    raise base.DrugTrainingError("Available system memory could not be measured")


def _runtime_fingerprint() -> dict[str, Any]:
    import duckdb
    import pyarrow
    import torch

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "pyarrow": str(pyarrow.__version__),
        "duckdb": str(duckdb.__version__),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": int(os.cpu_count() or 1),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def _execution_code_paths() -> dict[str, Path]:
    module_root = Path(__file__).resolve().parent
    paths = {
        "drug_training_streaming.py": Path(__file__).resolve(),
        "drug_training.py": module_root / "drug_training.py",
        "drug_staging.py": module_root / "drug_staging.py",
        "drug_sparse_query.py": module_root / "drug_sparse_query.py",
        "integrated_model.py": module_root / "integrated_model.py",
        "input_lineage.py": module_root / "input_lineage.py",
        "full_model_contract.py": module_root / "full_model_contract.py",
    }
    runner = Path(__file__).resolve().parents[2] / "scripts" / "run_v32_drug_training.py"
    if runner.is_file():
        paths["run_v32_drug_training.py"] = runner
    return paths


def _hash_snapshot(paths: Mapping[str, Path]) -> dict[str, str]:
    return {key: artifact_sha256(path) for key, path in paths.items()}


def _assert_hash_snapshot_unchanged(
    paths: Mapping[str, Path], start: Mapping[str, str], *, label: str
) -> dict[str, str]:
    end = _hash_snapshot(paths)
    drift = {
        key: {"start": start.get(key), "end": end.get(key)}
        for key in sorted(set(start) | set(end))
        if start.get(key) != end.get(key)
    }
    if drift:
        raise base.DrugTrainingError(f"{label} hash drift during Drug run: {drift}")
    return end


def _require_sha256(value: Any, label: str) -> str:
    text = str(value).lower() if isinstance(value, str) else ""
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise base.DrugTrainingError(f"{label} must be a strict 64-hex SHA-256")
    return text


def _revalidate_declared_snapshot(
    snapshot: Any,
    *,
    label: str,
    required_roles: set[str],
) -> dict[str, Path]:
    """Rehash every declared staging artifact; never trust top-level booleans."""

    if not isinstance(snapshot, Mapping):
        raise base.DrugTrainingError(f"{label} snapshot is missing or invalid")
    start = snapshot.get("start")
    end = snapshot.get("end")
    if not isinstance(start, Mapping) or not start:
        raise base.DrugTrainingError(f"{label} start snapshot must be a non-empty mapping")
    if not isinstance(end, Mapping) or not end:
        raise base.DrugTrainingError(f"{label} end snapshot must be a non-empty mapping")
    if dict(start) != dict(end):
        raise base.DrugTrainingError(f"{label} staging start/end snapshots differ")
    if snapshot.get("unchanged") is not True:
        raise base.DrugTrainingError(f"{label} staging snapshot lacks unchanged=true")
    missing = sorted(required_roles - set(map(str, start)))
    if missing:
        raise base.DrugTrainingError(f"{label} snapshot lacks required roles: {missing}")

    resolved: dict[str, Path] = {}
    for raw_role, raw_record in sorted(start.items(), key=lambda item: str(item[0])):
        role = str(raw_role)
        if not isinstance(raw_record, Mapping):
            raise base.DrugTrainingError(f"{label} role {role} is not a mapping")
        required_fields = {"path", "artifact_type", "bytes", "sha256"}
        missing_fields = sorted(required_fields - set(raw_record))
        if missing_fields:
            raise base.DrugTrainingError(
                f"{label} role {role} lacks fields: {missing_fields}"
            )
        path = Path(str(raw_record["path"])).resolve()
        if not path.exists():
            raise base.DrugTrainingError(f"{label} role {role} is missing: {path}")
        artifact_type = "file" if path.is_file() else "directory" if path.is_dir() else ""
        declared_type = str(raw_record["artifact_type"])
        if artifact_type == "" or declared_type != artifact_type:
            raise base.DrugTrainingError(f"{label} role {role} artifact type drifted")
        declared_bytes = raw_record["bytes"]
        if not isinstance(declared_bytes, int) or isinstance(declared_bytes, bool):
            raise base.DrugTrainingError(f"{label} role {role} bytes is not an integer")
        declared_sha = _require_sha256(raw_record["sha256"], f"{label} role {role} SHA")
        actual_bytes = int(_path_size_bytes(path))
        actual_sha = artifact_sha256(path)
        if int(declared_bytes) != actual_bytes or declared_sha != actual_sha:
            raise base.DrugTrainingError(f"{label} role {role} current artifact drifted")
        resolved[role] = path
    return resolved


def _early_resource_preflight(
    output: Path,
    conceptual_candidate_rows: int,
    pandas_input_paths: Mapping[str, Path],
    config: StreamingDrugTrainingConfig,
) -> dict[str, Any]:
    disk = _disk_preflight(output, conceptual_candidate_rows, config)
    input_sizes = {key: _path_size_bytes(path) for key, path in pandas_input_paths.items()}
    available_ram = _available_memory_bytes()
    estimated_ram = int(
        sum(input_sizes.values()) * RAM_INPUT_EXPANSION_MULTIPLIER
        + RAM_FIXED_OVERHEAD_BYTES
        + int(config.candidate_chunk_rows) * 2048
    )
    ram_budget = max(
        0,
        min(
            int(available_ram * RAM_MAX_AVAILABLE_FRACTION),
            int(available_ram - RAM_RESERVE_BYTES),
        ),
    )
    residual_disk = max(
        0,
        int(disk["available_after_reserve_bytes"])
        - int(disk["estimated_peak_incremental_bytes"]),
    )
    duckdb_temp_limit = min(DUCKDB_MAX_TEMP_BYTES, residual_disk)
    residual_ram = max(0, ram_budget - estimated_ram)
    duckdb_memory_limit = min(DUCKDB_MAX_MEMORY_BYTES, residual_ram)
    ram_status = "PASS" if estimated_ram <= ram_budget else "FAIL"
    duckdb_status = (
        "PASS"
        if duckdb_temp_limit >= 1024**3 and duckdb_memory_limit >= 256 * 1024**2
        else "FAIL"
    )
    result = {
        **disk,
        "guard_timing": (
            "BEFORE_FULL_PANDAS_READ_BEFORE_DUCKDB_DISTINCT_OR_SPILL_"
            "BEFORE_FOLD_TABLE_WRITE"
        ),
        "pandas_input_sizes_bytes": input_sizes,
        "pandas_input_total_bytes": int(sum(input_sizes.values())),
        "ram_input_expansion_multiplier": RAM_INPUT_EXPANSION_MULTIPLIER,
        "ram_fixed_overhead_bytes": RAM_FIXED_OVERHEAD_BYTES,
        "available_ram_bytes": available_ram,
        "ram_reserve_bytes": RAM_RESERVE_BYTES,
        "ram_max_available_fraction": RAM_MAX_AVAILABLE_FRACTION,
        "estimated_peak_ram_bytes": estimated_ram,
        "ram_budget_bytes": ram_budget,
        "ram_status": ram_status,
        "duckdb_temp_directory": str(output / ".duckdb_tmp"),
        "duckdb_max_temp_directory_size_bytes": int(duckdb_temp_limit),
        "duckdb_memory_limit_bytes": int(duckdb_memory_limit),
        "duckdb_threads": int(max(1, min(os.cpu_count() or 1, 8))),
        "sparse_query_duckdb_threads": int(
            max(1, min(os.cpu_count() or 1, 2))
        ),
        "sparse_query_validation_partitioning": (
            "EXACT_ALL_ROWS_BY_OBSERVED_CANCER_WITH_ALL_MASK_FOLDS"
        ),
        "duckdb_status": duckdb_status,
    }
    result["status"] = (
        "PASS"
        if disk["status"] == "PASS" and ram_status == "PASS" and duckdb_status == "PASS"
        else "FAIL"
    )
    return result


@contextmanager
def _bounded_duckdb(output: Path, resource_audit: Mapping[str, Any]):
    import duckdb

    if resource_audit.get("status") != "PASS":
        raise base.DrugTrainingError("DuckDB cannot start before a passing resource guard")
    temporary = output / ".duckdb_tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"SET temp_directory='{_sql_path(temporary)}'")
        con.execute(
            "SET max_temp_directory_size='"
            f"{int(resource_audit['duckdb_max_temp_directory_size_bytes'])}B'"
        )
        con.execute(
            f"SET memory_limit='{int(resource_audit['duckdb_memory_limit_bytes'])}B'"
        )
        con.execute(f"SET threads={int(resource_audit['duckdb_threads'])}")
        yield con
    finally:
        con.close()
        if temporary.is_dir():
            shutil.rmtree(temporary)


@contextmanager
def _sparse_query_resource_environment(
    output: Path, resource_audit: Mapping[str, Any]
):
    """Bind sparse-query DuckDB validation to the formal resource budget."""

    if resource_audit.get("status") != "PASS":
        raise base.DrugTrainingError(
            "Sparse Drug query validation requires a passing resource guard"
        )
    temporary = output / ".sparse_query_duckdb_tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    settings = {
        "CC_HHGT_DRUG_SPARSE_DUCKDB_MEMORY_LIMIT": (
            f"{int(resource_audit['duckdb_memory_limit_bytes'])}B"
        ),
        "CC_HHGT_DRUG_SPARSE_DUCKDB_THREADS": str(
            int(
                resource_audit.get(
                    "sparse_query_duckdb_threads",
                    resource_audit["duckdb_threads"],
                )
            )
        ),
        "CC_HHGT_DRUG_SPARSE_DUCKDB_TEMP_DIRECTORY": str(temporary),
        "CC_HHGT_DRUG_SPARSE_DUCKDB_MAX_TEMP_DIRECTORY_SIZE": (
            f"{int(resource_audit['duckdb_max_temp_directory_size_bytes'])}B"
        ),
    }
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if temporary.is_dir() and not any(temporary.iterdir()):
            temporary.rmdir()


def _load_factored_definition(
    root: Path, exact_path: Path
) -> tuple[dict[str, Any], Path]:
    definition_path = root / "FACTORED_UNIVERSE.json"
    if not definition_path.is_file():
        raise base.DrugTrainingError(
            f"Streaming Drug requires FACTORED_UNIVERSE.json: {root}"
        )
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if definition.get("candidate_format") != FACTORED_CANDIDATE_FORMAT:
        raise base.DrugTrainingError("Unsupported factored Drug candidate format")
    if definition.get("analysis_version") != base.ANALYSIS_VERSION:
        raise base.DrugTrainingError("Factored Drug universe is not current V3.2")
    if definition.get("storage_mode") != "factored":
        raise base.DrugTrainingError("Formal Drug candidate storage_mode must be factored")
    if definition.get("target_keys") != list(base.TARGET_KEYS):
        raise base.DrugTrainingError("Factored Drug target_keys are not the formal key contract")
    if int(definition.get("candidate_cancers", -1)) != len(TCGA_CANCERS):
        raise base.DrugTrainingError("Factored Drug universe must declare all 33 cancers")
    if any(
        definition.get(key) is not False
        for key in ("old_association_tables_read", "old_predictions_used", "old_checkpoints_used")
    ):
        raise base.DrugTrainingError("Factored Drug universe did not exclude old results")
    declared_exact = Path(str(definition.get("exact_candidate_path", ""))).resolve()
    if declared_exact != exact_path.resolve():
        raise base.DrugTrainingError("Factored Drug exact-candidate path mismatch")
    if artifact_sha256(exact_path) != definition.get("exact_candidate_sha256"):
        raise base.DrugTrainingError("Factored Drug exact-candidate hash mismatch")
    edge = Path(str(definition["pathway_drug_edge_path"]))
    if not edge.is_absolute():
        edge = (root / edge).resolve()
    else:
        edge = edge.resolve()
    if not edge.is_relative_to(root.resolve()):
        raise base.DrugTrainingError("Factored pathway-drug edge must be inside factored root")
    if not edge.is_file() or artifact_sha256(edge) != definition.get("pathway_drug_edge_sha256"):
        raise base.DrugTrainingError("Factored pathway-drug edge is missing or hash-mismatched")
    if int(definition.get("pathway_drug_edge_rows", 0)) < 1:
        raise base.DrugTrainingError("Factored pathway-drug edge row declaration is invalid")
    if int(definition.get("conceptual_candidate_rows", 0)) < 1:
        raise base.DrugTrainingError("Factored conceptual candidate row declaration is invalid")
    return definition, edge


def _audit_factored_relation(
    definition: Mapping[str, Any],
    exact_path: Path,
    edge_path: Path,
    output: Path,
    resource_audit: Mapping[str, Any],
) -> dict[str, Any]:
    with _bounded_duckdb(output, resource_audit) as con:
        edge_rows, edge_distinct = con.execute(
            f"""
            SELECT count(*), count(DISTINCT (pathway_id, drug_id))
            FROM read_parquet('{_sql_path(edge_path)}')
            """
        ).fetchone()
        rows = con.execute(
            f"""
            SELECT cancer_id, count(*) AS candidate_rows
            FROM (
              SELECT DISTINCT c.cancer_id, c.lncrna_id, p.drug_id
              FROM read_parquet('{_sql_path(exact_path)}') c
              JOIN read_parquet('{_sql_path(edge_path)}') p USING(pathway_id)
            ) q
            GROUP BY cancer_id
            ORDER BY cancer_id
            """
        ).fetchall()
    edge_rows = int(edge_rows)
    edge_distinct = int(edge_distinct)
    cancers = [str(row[0]).upper() for row in rows]
    conceptual_rows = int(sum(int(row[1]) for row in rows))
    if edge_rows != edge_distinct:
        raise base.DrugTrainingError("Factored pathway-drug edge contains duplicate keys")
    if edge_rows != int(definition["pathway_drug_edge_rows"]):
        raise base.DrugTrainingError(
            "Factored pathway-drug edge row count differs from declaration"
        )
    if cancers != sorted(TCGA_CANCERS):
        raise base.DrugTrainingError(
            f"Factored exact x edge relation does not cover the 33 cancers: {cancers}"
        )
    if conceptual_rows != int(definition["conceptual_candidate_rows"]):
        raise base.DrugTrainingError(
            "Independent exact x edge DISTINCT row count differs from declaration"
        )
    return {
        "method": "INDEPENDENT_DUCKDB_EXACT_X_EDGE_DISTINCT",
        "edge_rows_recomputed": edge_rows,
        "edge_distinct_rows_recomputed": edge_distinct,
        "conceptual_candidate_rows_recomputed": conceptual_rows,
        "candidate_cancers_recomputed": len(cancers),
        "candidate_cancer_ids_recomputed": cancers,
        "declarations_match": True,
    }


def _recompute_exact_candidate_binding(
    *,
    binding: Any,
    exact_path: Path,
    prediction_path: Path,
    lineage_path: Path,
    expected_prediction_sha256: str,
    expected_lineage_sha256: str,
    output: Path,
    resource_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently bind staging candidates to the specified exact-pathway release."""

    if not isinstance(binding, Mapping):
        raise base.DrugTrainingError("Drug staging lacks exact candidate source binding")
    if binding.get("status") != "PASS":
        raise base.DrugTrainingError("Drug staging exact candidate binding is not PASS")
    if binding.get("binding_policy") != (
        "LITERAL_FOUR_KEY_BIDIRECTIONAL_EXCEPT_AND_ROW_UNIQUENESS"
    ):
        raise base.DrugTrainingError("Drug staging exact binding policy drifted")
    if list(binding.get("key_columns", [])) != list(EXACT_BINDING_KEYS):
        raise base.DrugTrainingError("Drug staging exact binding key order drifted")

    declared_candidate = Path(str(binding.get("candidate_path", ""))).resolve()
    declared_prediction = Path(str(binding.get("release_prediction_path", ""))).resolve()
    declared_lineage = Path(str(binding.get("release_lineage_path", ""))).resolve()
    if declared_candidate != exact_path.resolve():
        raise base.DrugTrainingError("Drug staging exact binding candidate path drifted")
    if declared_prediction != prediction_path.resolve():
        raise base.DrugTrainingError("Drug staging exact binding prediction path drifted")
    if declared_lineage != lineage_path.resolve():
        raise base.DrugTrainingError("Drug staging exact binding lineage path drifted")

    candidate_sha = artifact_sha256(exact_path)
    prediction_sha = artifact_sha256(prediction_path)
    lineage_sha = artifact_sha256(lineage_path)
    if _require_sha256(binding.get("candidate_sha256"), "Exact candidate binding SHA") != candidate_sha:
        raise base.DrugTrainingError("Drug staging exact candidate SHA drifted")
    if (
        _require_sha256(binding.get("release_prediction_sha256"), "Exact prediction binding SHA")
        != prediction_sha
        or prediction_sha != expected_prediction_sha256
    ):
        raise base.DrugTrainingError("Specified exact release prediction SHA mismatch")
    if (
        _require_sha256(binding.get("release_lineage_sha256"), "Exact lineage binding SHA")
        != lineage_sha
        or lineage_sha != expected_lineage_sha256
    ):
        raise base.DrugTrainingError("Specified exact release lineage SHA mismatch")

    try:
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise base.DrugTrainingError(f"Invalid exact release lineage JSON: {error}") from error
    if not isinstance(lineage, Mapping):
        raise base.DrugTrainingError("Exact release lineage must be a JSON object")
    from .full_model_contract import validate_module_lineage

    try:
        validate_module_lineage("exact_pathway", lineage)
    except Exception as error:
        raise base.DrugTrainingError(
            f"Exact release lineage failed the full V3.2 contract: {error}"
        ) from error
    if lineage.get("prediction_sha256") != prediction_sha:
        raise base.DrugTrainingError("Exact lineage prediction SHA does not match proof")
    if binding.get("lineage_training_run_id") != lineage.get("training_run_id"):
        raise base.DrugTrainingError("Exact binding training_run_id differs from lineage")
    if binding.get("lineage_newly_trained_v32") is not True:
        raise base.DrugTrainingError("Exact binding lacks newly-trained V3.2 attestation")
    input_artifacts = lineage.get("input_artifacts")
    if not isinstance(input_artifacts, Sequence) or isinstance(input_artifacts, (str, bytes)):
        raise base.DrugTrainingError("Exact release lineage input_artifacts is invalid")
    if not any(
        isinstance(record, Mapping)
        and record.get("sha256") == candidate_sha
        and record.get("artifact_kind") == "standardized_input"
        for record in input_artifacts
    ):
        raise base.DrugTrainingError(
            "Exact release lineage does not bind the staged candidate SHA as standardized input"
        )

    candidate_sql = _sql_path(exact_path)
    prediction_sql = _sql_path(prediction_path)
    with _bounded_duckdb(output, resource_audit) as con:
        con.execute(
            f"""
            CREATE TEMP VIEW staging_exact_keys AS
            SELECT
              CAST(cancer_id AS VARCHAR) AS cancer_id,
              CAST(lncrna_id AS VARCHAR) AS lncrna_id,
              CAST(pathway_id AS VARCHAR) AS pathway_id,
              CAST(pathway_family_id AS VARCHAR) AS pathway_family_id
            FROM read_parquet('{candidate_sql}')
            """
        )
        con.execute(
            f"""
            CREATE TEMP VIEW release_exact_keys AS
            SELECT
              CAST(cancer_id AS VARCHAR) AS cancer_id,
              CAST(lncrna_id AS VARCHAR) AS lncrna_id,
              CAST(pathway_id AS VARCHAR) AS pathway_id,
              CAST(pathway_family_id AS VARCHAR) AS pathway_family_id
            FROM read_parquet('{prediction_sql}')
            """
        )
        candidate_rows = int(con.execute("SELECT count(*) FROM staging_exact_keys").fetchone()[0])
        prediction_rows = int(con.execute("SELECT count(*) FROM release_exact_keys").fetchone()[0])
        candidate_unique = int(
            con.execute("SELECT count(*) FROM (SELECT DISTINCT * FROM staging_exact_keys) q").fetchone()[0]
        )
        prediction_unique = int(
            con.execute("SELECT count(*) FROM (SELECT DISTINCT * FROM release_exact_keys) q").fetchone()[0]
        )
        empty_predicate = " OR ".join(
            f"{key} IS NULL OR {key} = ''" for key in EXACT_BINDING_KEYS
        )
        candidate_null_keys = int(
            con.execute(f"SELECT count(*) FROM staging_exact_keys WHERE {empty_predicate}").fetchone()[0]
        )
        prediction_null_keys = int(
            con.execute(f"SELECT count(*) FROM release_exact_keys WHERE {empty_predicate}").fetchone()[0]
        )
        candidate_minus_prediction = int(
            con.execute(
                "SELECT count(*) FROM (SELECT DISTINCT * FROM staging_exact_keys "
                "EXCEPT SELECT DISTINCT * FROM release_exact_keys) q"
            ).fetchone()[0]
        )
        prediction_minus_candidate = int(
            con.execute(
                "SELECT count(*) FROM (SELECT DISTINCT * FROM release_exact_keys "
                "EXCEPT SELECT DISTINCT * FROM staging_exact_keys) q"
            ).fetchone()[0]
        )
        candidate_cancers, candidate_lncrnas, candidate_pathways = con.execute(
            "SELECT count(DISTINCT cancer_id), count(DISTINCT lncrna_id), "
            "count(DISTINCT pathway_id) FROM staging_exact_keys"
        ).fetchone()

    recomputed = {
        "candidate_rows": candidate_rows,
        "candidate_unique_keys": candidate_unique,
        "release_prediction_rows": prediction_rows,
        "release_prediction_unique_keys": prediction_unique,
        "candidate_null_keys": candidate_null_keys,
        "prediction_null_keys": prediction_null_keys,
        "candidate_minus_prediction": candidate_minus_prediction,
        "prediction_minus_candidate": prediction_minus_candidate,
    }
    for key, observed in recomputed.items():
        declared = binding.get(key)
        if not isinstance(declared, int) or isinstance(declared, bool) or int(declared) != observed:
            raise base.DrugTrainingError(f"Drug staging exact binding declaration drifted for {key}")
    if (
        candidate_rows < 1
        or len({candidate_rows, candidate_unique, prediction_rows, prediction_unique}) != 1
        or candidate_null_keys != 0
        or prediction_null_keys != 0
        or candidate_minus_prediction != 0
        or prediction_minus_candidate != 0
        or int(candidate_cancers) != len(TCGA_CANCERS)
        or int(candidate_lncrnas) < 1
        or int(candidate_pathways) < 1
    ):
        raise base.DrugTrainingError("Independent exact candidate four-key binding failed")
    if int(lineage.get("prediction_rows", -1)) != prediction_rows:
        raise base.DrugTrainingError("Exact lineage prediction row count does not match proof")

    verified = dict(binding)
    verified.update(recomputed)
    verified.update(
        {
            "status": "PASS",
            "candidate_sha256": candidate_sha,
            "release_prediction_sha256": prediction_sha,
            "release_lineage_sha256": lineage_sha,
            "candidate_cancers_recomputed": int(candidate_cancers),
            "candidate_lncrnas_recomputed": int(candidate_lncrnas),
            "candidate_pathways_recomputed": int(candidate_pathways),
            "lineage_contract_validated": True,
            "four_key_binding_recomputed": True,
        }
    )
    return verified


def _validate_staging_manifest(
    candidate_root: Path,
    paths: Mapping[str, Path],
    *,
    output: Path,
    resource_audit: Mapping[str, Any],
    expected_staging_manifest_sha256: str,
    expected_exact_release_prediction_sha256: str,
    expected_exact_release_lineage_sha256: str,
) -> dict[str, Any]:
    expected_manifest_sha = _require_sha256(
        expected_staging_manifest_sha256, "Expected staging manifest SHA"
    )
    expected_prediction_sha = _require_sha256(
        expected_exact_release_prediction_sha256,
        "Expected exact release prediction SHA",
    )
    expected_lineage_sha = _require_sha256(
        expected_exact_release_lineage_sha256,
        "Expected exact release lineage SHA",
    )
    manifest_path = candidate_root.parent / "STAGING_MANIFEST.json"
    if not manifest_path.is_file():
        raise base.DrugTrainingError(
            f"Formal streaming Drug input lacks STAGING_MANIFEST.json: {manifest_path}"
        )
    manifest_sha = artifact_sha256(manifest_path)
    if manifest_sha != expected_manifest_sha:
        raise base.DrugTrainingError("Specified staging manifest SHA mismatch")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise base.DrugTrainingError(f"Invalid staging manifest JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise base.DrugTrainingError("Drug staging manifest must be a JSON object")
    if payload.get("staging_format") != STAGING_FORMAT:
        raise base.DrugTrainingError("Drug staging manifest format drifted")
    if payload.get("analysis_version") != base.ANALYSIS_VERSION:
        raise base.DrugTrainingError("Drug staging manifest is not current V3.2")
    if payload.get("status") != "SUCCESS" or payload.get("formal") is not True:
        raise base.DrugTrainingError("Drug staging manifest is not a formal successful run")
    if payload.get("training_ready") is not True:
        raise base.DrugTrainingError("Drug staging manifest is not training-ready")
    if sorted(map(str, payload.get("candidate_cancers", []))) != sorted(TCGA_CANCERS):
        raise base.DrugTrainingError("Drug staging manifest does not cover the 33 cancers")

    input_snapshot = payload.get("input_artifact_snapshot")
    input_snapshot_paths = _revalidate_declared_snapshot(
        input_snapshot,
        label="Drug staging input/static/proof",
        required_roles=FORMAL_STAGING_INPUT_ROLES,
    )
    if payload.get("input_artifacts_unchanged") is not True:
        raise base.DrugTrainingError("Drug staging input unchanged attestation is false")
    code_snapshot = payload.get("execution_code_snapshot")
    code_snapshot_paths = _revalidate_declared_snapshot(
        code_snapshot,
        label="Drug staging execution code",
        required_roles={"drug_staging_module"},
    )
    if not any(role.startswith("execution_entrypoint_") for role in code_snapshot_paths):
        raise base.DrugTrainingError("Drug staging code snapshot lacks an execution entrypoint")
    if payload.get("execution_code_unchanged") is not True:
        raise base.DrugTrainingError("Drug staging code unchanged attestation is false")

    runtime_fingerprint = payload.get("runtime_fingerprint")
    if not isinstance(runtime_fingerprint, Mapping) or not runtime_fingerprint:
        raise base.DrugTrainingError("Drug staging runtime fingerprint is invalid")
    runtime_sha = _require_sha256(
        payload.get("runtime_fingerprint_sha256"), "Drug staging runtime fingerprint SHA"
    )
    if _canonical_staging_sha(runtime_fingerprint) != runtime_sha:
        raise base.DrugTrainingError("Drug staging runtime fingerprint SHA mismatch")

    provenance_path = Path(
        str(payload.get("staging_provenance_snapshot_path", ""))
    ).resolve()
    if not provenance_path.is_relative_to(candidate_root.parent.resolve()):
        raise base.DrugTrainingError("Drug staging provenance snapshot is outside staging root")
    if not provenance_path.is_file():
        raise base.DrugTrainingError("Drug staging provenance snapshot is missing")
    provenance_sha = _require_sha256(
        payload.get("staging_provenance_snapshot_sha256"),
        "Drug staging provenance snapshot SHA",
    )
    if artifact_sha256(provenance_path) != provenance_sha:
        raise base.DrugTrainingError("Drug staging provenance snapshot SHA mismatch")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise base.DrugTrainingError(f"Invalid staging provenance JSON: {error}") from error
    expected_provenance = {
        "status": "PASS",
        "input_artifacts": input_snapshot,
        "execution_code": code_snapshot,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": runtime_sha,
        "exact_candidate_source_binding": payload.get("exact_candidate_source_binding"),
    }
    if provenance != expected_provenance:
        raise base.DrugTrainingError(
            "Drug staging provenance snapshot does not exactly match manifest declarations"
        )

    exact_prediction_path = input_snapshot_paths["exact_release_prediction_proof"]
    exact_lineage_path = input_snapshot_paths["exact_release_lineage_proof"]
    if input_snapshot_paths["v32_exact_candidates"] != paths["exact"].resolve():
        raise base.DrugTrainingError("Drug staging snapshot candidate path differs from training")
    if (
        artifact_sha256(exact_prediction_path) != expected_prediction_sha
        or artifact_sha256(exact_lineage_path) != expected_lineage_sha
    ):
        raise base.DrugTrainingError("Specified exact release proof hashes do not match snapshots")
    verified_binding = _recompute_exact_candidate_binding(
        binding=payload.get("exact_candidate_source_binding"),
        exact_path=paths["exact"],
        prediction_path=exact_prediction_path,
        lineage_path=exact_lineage_path,
        expected_prediction_sha256=expected_prediction_sha,
        expected_lineage_sha256=expected_lineage_sha,
        output=output,
        resource_audit=resource_audit,
    )
    mapping_coverage = payload.get("native_assayed_to_target_mapping_coverage")
    if not isinstance(mapping_coverage, Mapping):
        raise base.DrugTrainingError("Drug staging lacks native-assayed target coverage")
    if mapping_coverage.get("identity_scope") != (
        "EXACT_DRUG_NAME_AFTER_UPPERCASE_ALPHANUMERIC_NORMALIZATION_NO_SYNONYM_CROSSWALK"
    ):
        raise base.DrugTrainingError("Drug target identity scope is not explicit/fail-closed")
    if int(mapping_coverage.get("minimum_mapped_drugs", 0)) < FORMAL_MIN_TARGET_MAPPED_DRUGS:
        raise base.DrugTrainingError("Drug target mapped-drug threshold is below formal floor")
    if float(mapping_coverage.get("minimum_mapping_fraction", 0.0)) < FORMAL_MIN_TARGET_MAPPING_FRACTION:
        raise base.DrugTrainingError("Drug target mapping fraction threshold is below formal floor")
    if mapping_coverage.get("threshold_pass") is not True:
        raise base.DrugTrainingError("Native-assayed to target identity coverage failed")
    if mapping_coverage.get("candidate_scope") != "TARGET_ANNOTATED_NATIVE_ASSAYED_DRUGS":
        raise base.DrugTrainingError("Drug candidate scope is not target-annotated native assays")
    silent_path = Path(
        str(mapping_coverage.get("silent_nonmatch_sidecar_path", ""))
    ).resolve()
    if not silent_path.is_relative_to(candidate_root.parent.resolve()):
        raise base.DrugTrainingError("Drug silent-nonmatch sidecar is outside staging root")
    if not silent_path.is_file():
        raise base.DrugTrainingError("Drug silent-nonmatch sidecar is missing")
    if artifact_sha256(silent_path) != mapping_coverage.get(
        "silent_nonmatch_sidecar_sha256"
    ):
        raise base.DrugTrainingError("Drug silent-nonmatch sidecar hash mismatch")
    import pyarrow.parquet as pq

    silent_rows = int(pq.ParquetFile(silent_path).metadata.num_rows)
    if silent_rows != int(mapping_coverage.get("silent_nonmatch_sidecar_rows", -1)):
        raise base.DrugTrainingError("Drug silent-nonmatch sidecar row count mismatch")
    if any(
        payload.get(key) is not False
        for key in (
            "old_association_tables_read", "old_predictions_used", "old_checkpoints_used"
        )
    ):
        raise base.DrugTrainingError("Drug staging manifest did not exclude old results")
    declarations = payload.get("artifacts")
    if not isinstance(declarations, Mapping):
        raise base.DrugTrainingError("Drug staging manifest lacks artifact declarations")
    expected = {
        "raw_cell_line_drug_response": paths["response"],
        "raw_cell_line_lncrna_expression": paths["expression"],
        "cell_line_map": paths["mapping"],
        "drug_gene_target": paths["targets"],
        "drug_candidate_universe": paths["candidates"],
    }
    for role, actual_path in expected.items():
        declaration = declarations.get(role)
        if not isinstance(declaration, Mapping):
            raise base.DrugTrainingError(f"Drug staging manifest lacks {role}")
        declared_path = Path(str(declaration.get("path", ""))).resolve()
        if declared_path != actual_path.resolve():
            raise base.DrugTrainingError(
                f"Drug staging {role} path drift: {declared_path} != {actual_path}"
            )
        if declaration.get("sha256") != artifact_sha256(actual_path):
            raise base.DrugTrainingError(f"Drug staging {role} SHA256 mismatch")
    revalidation_audit = {
        "status": "PASS",
        "analysis_version": base.ANALYSIS_VERSION,
        "staging_manifest_path": str(manifest_path.resolve()),
        "staging_manifest_sha256": manifest_sha,
        "expected_staging_manifest_sha256": expected_manifest_sha,
        "staging_provenance_snapshot_path": str(provenance_path),
        "staging_provenance_snapshot_sha256": provenance_sha,
        "staging_input_artifacts_rehashed": True,
        "staging_input_artifact_roles_rehashed": sorted(input_snapshot_paths),
        "staging_execution_code_rehashed": True,
        "staging_execution_code_roles_rehashed": sorted(code_snapshot_paths),
        "staging_runtime_fingerprint_sha256": runtime_sha,
        "exact_release_lineage_contract_validated": True,
        "exact_candidate_four_key_binding_recomputed": True,
        "expected_exact_release_prediction_sha256": expected_prediction_sha,
        "expected_exact_release_lineage_sha256": expected_lineage_sha,
        "exact_candidate_source_binding": verified_binding,
    }
    audit_path = output / "STAGING_REVALIDATION_AUDIT.json"
    _atomic_json(audit_path, revalidation_audit)
    return {
        "manifest": dict(payload),
        "manifest_path": manifest_path.resolve(),
        "manifest_sha256": manifest_sha,
        "provenance_path": provenance_path,
        "provenance_sha256": provenance_sha,
        "input_snapshot_paths": input_snapshot_paths,
        "code_snapshot_paths": code_snapshot_paths,
        "exact_prediction_path": exact_prediction_path,
        "exact_lineage_path": exact_lineage_path,
        "exact_candidate_source_binding": verified_binding,
        "staging_runtime_fingerprint_sha256": runtime_sha,
        "revalidation_audit": revalidation_audit,
        "revalidation_audit_path": audit_path,
    }


def _build_context(
    expression: pd.DataFrame,
    response: pd.DataFrame,
    mapping: pd.DataFrame,
    models: Sequence[str],
    cancer_id: str,
) -> _AssociationContext:
    selected = set(map(str, models))
    local_map = mapping.loc[
        mapping.canonical_model_id.astype(str).isin(selected)
        & mapping.cancer_id.astype(str).eq(str(cancer_id))
    ].copy()
    if local_map.empty:
        return _AssociationContext(str(cancer_id), (), {}, {})
    canonical_expression = "canonical_model_id" in expression
    expression_by_dataset: dict[str, pd.DataFrame] = {}
    if canonical_expression:
        model_ids = set(local_map.canonical_model_id.astype(str))
        local_expression = expression.loc[
            expression.canonical_model_id.astype(str).isin(model_ids)
        ]
        matrix = local_expression.pivot_table(
            index="canonical_model_id", columns="lncrna_id",
            values="expression_value", aggfunc="mean", observed=True,
        )
        for dataset in sorted(local_map.dataset_id.astype(str).unique()):
            dataset_models = local_map.loc[
                local_map.dataset_id.astype(str).eq(dataset), "canonical_model_id"
            ].astype(str).unique()
            expression_by_dataset[dataset] = matrix.reindex(dataset_models)
    else:
        linked_expression = expression.merge(
            local_map[["dataset_id", "cell_line_id", "canonical_model_id"]].drop_duplicates(),
            on=["dataset_id", "cell_line_id"], how="inner", validate="many_to_many",
        )
        for dataset, local in linked_expression.groupby("dataset_id", observed=True, sort=False):
            expression_by_dataset[str(dataset)] = local.pivot_table(
                index="canonical_model_id", columns="lncrna_id",
                values="expression_value", aggfunc="mean", observed=True,
            )
    linked_response = response.merge(
        local_map[["dataset_id", "cell_line_id", "canonical_model_id"]].drop_duplicates(),
        on=["dataset_id", "cell_line_id"], how="inner", validate="many_to_many",
    )
    response_groups: dict[tuple[str, str], pd.Series] = {}
    for (dataset, drug), local in linked_response.groupby(
        ["dataset_id", "drug_id"], observed=True, sort=False
    ):
        response_groups[(str(dataset), str(drug))] = local.groupby(
            "canonical_model_id", observed=True
        ).sensitivity_value.mean()
    return _AssociationContext(
        str(cancer_id), tuple(sorted(expression_by_dataset)),
        expression_by_dataset, response_groups,
    )


def _prepare_association_cache(
    expression: pd.DataFrame,
    response: pd.DataFrame,
    mapping: pd.DataFrame,
    config: StreamingDrugTrainingConfig,
) -> _PreparedAssociationCache:
    """Link the two large assay tables to mapping metadata exactly once."""

    map_columns = [
        "dataset_id", "cell_line_id", "canonical_model_id", "cancer_id",
        "cell_line_fold_id",
    ]
    missing_map = sorted(set(map_columns) - set(mapping.columns))
    if missing_map:
        raise base.DrugTrainingError(
            f"Prepared Drug context mapping lacks columns: {missing_map}"
        )
    mapping_keys = mapping[map_columns].drop_duplicates().copy()
    mapping_keys["dataset_id"] = mapping_keys.dataset_id.astype(str)
    mapping_keys["cell_line_id"] = mapping_keys.cell_line_id.astype(str)
    mapping_keys["canonical_model_id"] = mapping_keys.canonical_model_id.astype(str)
    mapping_keys["cancer_id"] = mapping_keys.cancer_id.astype(str).str.upper()
    fold_conflicts = mapping_keys.groupby(
        "canonical_model_id", observed=True
    ).cell_line_fold_id.nunique()
    if (fold_conflicts > 1).any():
        raise base.DrugTrainingError(
            "Prepared Drug context found a canonical model in multiple folds"
        )

    models_by_cancer_dataset: dict[tuple[str, str], tuple[str, ...]] = {}
    datasets_by_cancer_sets: dict[str, set[str]] = {}
    for (cancer, dataset), local in mapping_keys.groupby(
        ["cancer_id", "dataset_id"], observed=True, sort=False
    ):
        key = (str(cancer), str(dataset))
        models_by_cancer_dataset[key] = tuple(
            local.canonical_model_id.astype(str).drop_duplicates().tolist()
        )
        datasets_by_cancer_sets.setdefault(str(cancer), set()).add(str(dataset))
    datasets_by_cancer = {
        cancer: tuple(sorted(datasets))
        for cancer, datasets in datasets_by_cancer_sets.items()
    }

    linked_response = response.merge(
        mapping_keys,
        on=["dataset_id", "cell_line_id"],
        how="inner",
        validate="many_to_many",
    )
    response_counts = linked_response.groupby(
        ["cancer_id", "dataset_id", "drug_id", "cell_line_fold_id"],
        observed=True,
    ).canonical_model_id.nunique().rename("model_count").reset_index()
    qualified = response_counts.loc[
        response_counts.model_count >= int(config.min_cell_lines_per_dataset),
        ["cancer_id", "dataset_id", "drug_id", "cell_line_fold_id"],
    ]
    qualified_linked = linked_response.merge(
        qualified,
        on=["cancer_id", "dataset_id", "drug_id", "cell_line_fold_id"],
        how="inner",
        validate="many_to_one",
    )
    response_totals = qualified_linked.groupby(
        ["cancer_id", "drug_id", "cell_line_fold_id"], observed=True
    ).agg(
        total_models=("canonical_model_id", "nunique"),
        eligible_datasets=("dataset_id", "nunique"),
    ).reset_index()
    fold_eligible = response_totals.loc[
        response_totals.total_models >= int(config.min_total_cell_lines),
        [
            "cancer_id", "drug_id", "cell_line_fold_id", "total_models",
            "eligible_datasets",
        ],
    ].copy()
    fold_eligible["cancer_id"] = fold_eligible.cancer_id.astype(str).str.upper()

    response_aggregate = linked_response.groupby(
        ["cancer_id", "dataset_id", "drug_id", "canonical_model_id"],
        observed=True,
        sort=False,
    ).sensitivity_value.mean().reset_index()
    response_groups: dict[tuple[str, str, str], pd.Series] = {}
    response_drugs: dict[tuple[str, str], list[str]] = {}
    for (cancer, dataset, drug), local in response_aggregate.groupby(
        ["cancer_id", "dataset_id", "drug_id"], observed=True, sort=False
    ):
        key = (str(cancer), str(dataset), str(drug))
        response_groups[key] = pd.Series(
            local.sensitivity_value.to_numpy(dtype=float),
            index=pd.Index(
                local.canonical_model_id.astype(str).to_numpy(),
                name="canonical_model_id",
            ),
            name="sensitivity_value",
        )
        response_drugs.setdefault((str(cancer), str(dataset)), []).append(str(drug))
    response_drugs_by_cancer_dataset = {
        key: tuple(values) for key, values in response_drugs.items()
    }

    canonical_expression: pd.DataFrame | None = None
    expression_by_cancer_dataset: dict[tuple[str, str], pd.DataFrame] = {}
    if "canonical_model_id" in expression.columns:
        canonical_expression = expression.pivot_table(
            index="canonical_model_id",
            columns="lncrna_id",
            values="expression_value",
            aggfunc="mean",
            observed=True,
        )
        expression_link = expression.merge(
            mapping_keys[[
                "canonical_model_id", "cancer_id", "cell_line_fold_id"
            ]].drop_duplicates(),
            on="canonical_model_id",
            how="inner",
            validate="many_to_many",
        ).drop_duplicates(
            [
                "canonical_model_id", "cancer_id", "lncrna_id",
                "cell_line_fold_id",
            ]
        )
        coverage = expression_link.groupby(
            ["cancer_id", "lncrna_id", "cell_line_fold_id"], observed=True
        ).canonical_model_id.nunique().rename(
            "matched_expression_models"
        ).reset_index()
    else:
        expression_link = expression.merge(
            mapping_keys,
            on=["dataset_id", "cell_line_id"],
            how="inner",
            validate="many_to_many",
        )
        for (cancer, dataset), local in expression_link.groupby(
            ["cancer_id", "dataset_id"], observed=True, sort=False
        ):
            expression_by_cancer_dataset[(str(cancer), str(dataset))] = (
                local.pivot_table(
                    index="canonical_model_id",
                    columns="lncrna_id",
                    values="expression_value",
                    aggfunc="mean",
                    observed=True,
                )
            )
        expression_models = expression_link.drop_duplicates(
            [
                "dataset_id", "canonical_model_id", "cancer_id", "lncrna_id",
                "cell_line_fold_id",
            ]
        )
        by_dataset = expression_models.groupby(
            [
                "cancer_id", "lncrna_id", "cell_line_fold_id", "dataset_id"
            ],
            observed=True,
        ).canonical_model_id.nunique().rename("dataset_models").reset_index()
        eligible_expression_datasets = by_dataset.loc[
            by_dataset.dataset_models >= int(config.min_cell_lines_per_dataset),
            ["cancer_id", "lncrna_id", "cell_line_fold_id", "dataset_id"],
        ]
        eligible_expression_models = expression_models.merge(
            eligible_expression_datasets,
            on=["cancer_id", "lncrna_id", "cell_line_fold_id", "dataset_id"],
            how="inner",
            validate="many_to_one",
        )
        coverage = eligible_expression_models.groupby(
            ["cancer_id", "lncrna_id", "cell_line_fold_id"], observed=True
        ).canonical_model_id.nunique().rename(
            "matched_expression_models"
        ).reset_index()
    expression_fold_coverage = coverage.loc[
        coverage.matched_expression_models >= int(config.min_total_cell_lines)
    ].rename(columns={"cell_line_fold_id": "fold_id"}).copy()
    expression_fold_coverage["cancer_id"] = (
        expression_fold_coverage.cancer_id.astype(str).str.upper()
    )
    expression_fold_coverage["expression_available"] = True
    expression_fold_coverage = expression_fold_coverage.sort_values(
        ["cancer_id", "lncrna_id", "fold_id"], kind="stable"
    ).reset_index(drop=True)

    instrumentation: dict[str, Any] = {
        "cache_format": "CC_HHGT_V3_2_DRUG_PREPARED_ASSOCIATION_CACHE_V1",
        "full_table_prepare_calls": 1,
        "response_full_table_mapping_links": 1,
        "expression_full_table_mapping_links": 1,
        "response_full_table_groupby_calls": 2,
        "expression_full_table_pivot_calls": (
            1 if canonical_expression is not None else len(expression_by_cancer_dataset)
        ),
        "expression_mode": (
            "CANONICAL_MODEL" if canonical_expression is not None
            else "DATASET_CELL_LINE"
        ),
        "response_input_rows": int(len(response)),
        "response_linked_rows": int(len(linked_response)),
        "expression_input_rows": int(len(expression)),
        "expression_linked_rows": int(len(expression_link)),
        "mapping_rows": int(len(mapping_keys)),
        "prepared_response_groups": int(len(response_groups)),
        "prepared_expression_matrices": int(
            1 if canonical_expression is not None
            else len(expression_by_cancer_dataset)
        ),
        "prepared_cancers": int(len(datasets_by_cancer)),
        "context_slice_calls": 0,
        "legacy_full_table_context_build_calls": 0,
        "formal_path_uses_prepared_cache": True,
        "context_operations_after_prepare": "CANONICAL_MODEL_INDEX_SLICE_ONLY",
    }
    return _PreparedAssociationCache(
        canonical_expression=canonical_expression,
        expression_by_cancer_dataset=expression_by_cancer_dataset,
        response_by_cancer_dataset_drug=response_groups,
        response_drugs_by_cancer_dataset=response_drugs_by_cancer_dataset,
        models_by_cancer_dataset=models_by_cancer_dataset,
        datasets_by_cancer=datasets_by_cancer,
        fold_eligible=fold_eligible,
        expression_fold_coverage=expression_fold_coverage,
        instrumentation=instrumentation,
    )


def _build_context_from_prepared(
    prepared: _PreparedAssociationCache,
    models: Sequence[str],
    cancer_id: str,
) -> _AssociationContext:
    """Build a context using index slices only (no full merge or pivot)."""

    prepared.instrumentation["context_slice_calls"] = int(
        prepared.instrumentation.get("context_slice_calls", 0)
    ) + 1
    cancer = str(cancer_id).upper()
    selected = set(map(str, models))
    expression_by_dataset: dict[str, pd.DataFrame] = {}
    response_by_dataset_drug: dict[tuple[str, str], pd.Series] = {}
    for dataset in prepared.datasets_by_cancer.get(cancer, ()):
        key = (cancer, str(dataset))
        dataset_models = tuple(
            model
            for model in prepared.models_by_cancer_dataset.get(key, ())
            if model in selected
        )
        if not dataset_models:
            continue
        if prepared.canonical_expression is not None:
            expression_by_dataset[str(dataset)] = (
                prepared.canonical_expression.reindex(dataset_models)
            )
        else:
            matrix = prepared.expression_by_cancer_dataset.get(key)
            if matrix is None:
                continue
            expression_by_dataset[str(dataset)] = matrix.reindex(dataset_models)
        selected_dataset_models = set(dataset_models)
        for drug in prepared.response_drugs_by_cancer_dataset.get(key, ()):
            response_values = prepared.response_by_cancer_dataset_drug[
                (cancer, str(dataset), str(drug))
            ]
            local = response_values.loc[
                response_values.index.astype(str).isin(selected_dataset_models)
            ]
            if not local.empty:
                response_by_dataset_drug[(str(dataset), str(drug))] = local
    return _AssociationContext(
        cancer,
        tuple(sorted(expression_by_dataset)),
        expression_by_dataset,
        response_by_dataset_drug,
    )


def _association_cache_audit(
    prepared: _PreparedAssociationCache,
) -> dict[str, Any]:
    return {
        **prepared.instrumentation,
        "status": "PASS",
        "response_or_expression_full_table_reprocessed_per_context": False,
    }


def _preeligible_drugs(
    context: _AssociationContext, config: StreamingDrugTrainingConfig
) -> set[str]:
    canonical_models: dict[str, set[str]] = {}
    for (_, drug), response in context.response_by_dataset_drug.items():
        if len(response) >= int(config.min_cell_lines_per_dataset):
            canonical_models.setdefault(drug, set()).update(map(str, response.index))
    return {
        drug for drug, models in canonical_models.items()
        if len(models) >= int(config.min_total_cell_lines)
    }


def _rank_correlation_columns(matrix: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Column-wise Spearman correlation with a dense fast path."""
    return base._rank_correlation_columns(matrix, response)


def _preeligible_lncrnas(*contexts: _AssociationContext) -> set[str]:
    """Return lncRNAs with finite expression in at least one supplied context.

    A candidate absent from every context matrix cannot meet the downstream
    matched-expression requirements.  Filtering it before exact×Drug expansion
    is therefore lossless and materially reduces formal streaming work.
    """

    result: set[str] = set()
    for context in contexts:
        for matrix in context.expression_by_dataset.values():
            if matrix.empty:
                continue
            finite = np.isfinite(matrix.to_numpy(dtype=float)).any(axis=0)
            result.update(
                str(value)
                for value in matrix.columns[np.flatnonzero(finite)].tolist()
            )
    return result


def _association_statistics_vectorized(
    candidates: pd.DataFrame,
    context: _AssociationContext,
    drug_target_counts: Mapping[str, int],
    drug_target_embedding_available: Mapping[str, bool],
    config: StreamingDrugTrainingConfig,
) -> base.AssociationStatistics:
    size = len(candidates)
    domain = np.zeros((size, len(base._DOMAIN_FEATURES)), dtype=np.float32)
    labels = np.full(size, np.nan, dtype=np.float32)
    assay_available = np.zeros(size, dtype=bool)
    reasons = np.full(size, "NO_MATCHED_CELL_LINE_ASSAY", dtype=object)
    rho_values = np.full(size, np.nan, dtype=np.float32)
    n_values = np.zeros(size, dtype=np.int32)
    dataset_values = np.zeros(size, dtype=np.int16)
    if not size or not context.datasets:
        return base.AssociationStatistics(
            domain, labels, assay_available, reasons, rho_values, n_values, dataset_values
        )
    drug_ids = candidates.drug_id.astype(str).to_numpy(object)
    lncrna_ids = candidates.lncrna_id.astype(str).to_numpy(object)
    target_count = np.asarray(
        [int(drug_target_counts.get(str(value), 0)) for value in drug_ids], dtype=np.int32
    )
    target_available = np.asarray(
        [bool(drug_target_embedding_available.get(str(value), False)) for value in drug_ids],
        dtype=bool,
    )
    domain[:, 4] = np.log1p(target_count)
    domain[:, 5] = target_available.astype(np.float32)
    combined = np.zeros(size, dtype=np.float64)
    expression_mean = np.zeros(size, dtype=np.float64)
    expression_sd = np.zeros(size, dtype=np.float64)
    groups = candidates.groupby("drug_id", observed=True, sort=False).indices
    for drug, raw_positions in groups.items():
        positions = np.asarray(raw_positions, dtype=int)
        local_lncrnas = lncrna_ids[positions]
        records: list[base.DatasetAssociationArrays] = []
        for dataset in context.datasets:
            response = context.response_by_dataset_drug.get((str(dataset), str(drug)))
            expression = context.expression_by_dataset.get(str(dataset))
            if response is None or expression is None:
                continue
            common = expression.index.intersection(response.index)
            y = response.reindex(common).to_numpy(float)
            x = expression.reindex(index=common, columns=local_lncrnas).to_numpy(float)
            records.append(
                base.DatasetAssociationArrays(
                    dataset_id=str(dataset),
                    canonical_model_ids=common.astype(str).to_numpy(object),
                    expression=x,
                    sensitivity=y,
                )
            )
        summary = base.combine_cross_dataset_associations(
            records,
            candidate_count=len(positions),
            min_cell_lines_per_dataset=int(config.min_cell_lines_per_dataset),
        )
        combined[positions] = summary.rho
        n_values[positions] = summary.n_cell_lines
        dataset_values[positions] = summary.dataset_count
        expression_mean[positions] = summary.expression_mean
        expression_sd[positions] = summary.expression_sd
    has_dataset = dataset_values > 0
    reasons[has_dataset] = "INSUFFICIENT_TOTAL_MATCHED_CELL_LINES"
    consensus_undefined = (
        has_dataset
        & (n_values >= int(config.min_total_cell_lines))
        & ~np.isfinite(combined)
    )
    reasons[consensus_undefined] = "CROSS_DATASET_CONSENSUS_UNDEFINED"
    sufficient = (
        has_dataset
        & (n_values >= int(config.min_total_cell_lines))
        & np.isfinite(combined)
    )
    rho_values[sufficient] = combined[sufficient].astype(np.float32)
    assay_available[sufficient] = True
    reasons[sufficient] = ""
    domain[:, 0] = np.log1p(n_values)
    domain[:, 1] = dataset_values
    domain[:, 2] = expression_mean.astype(np.float32)
    domain[:, 3] = expression_sd.astype(np.float32)
    magnitude = np.abs(combined)
    labels[sufficient & (magnitude >= float(config.positive_abs_rho))] = 1.0
    labels[sufficient & (magnitude <= float(config.negative_abs_rho))] = 0.0
    return base.AssociationStatistics(
        domain, labels, assay_available, reasons, rho_values, n_values, dataset_values
    )


def _candidate_batches(
    exact_path: Path,
    edge_path: Path,
    cancer_id: str,
    eligible_drugs: Iterable[str],
    eligible_lncrnas: Iterable[str] | None,
    chunk_rows: int,
    output: Path,
    resource_audit: Mapping[str, Any],
) -> Iterable[pd.DataFrame]:
    drugs = pd.DataFrame({"drug_id": sorted(set(map(str, eligible_drugs)))})
    if drugs.empty:
        return
    lncrnas = None
    if eligible_lncrnas is not None:
        lncrnas = pd.DataFrame(
            {
                "lncrna_id": sorted(
                    {base._canonical_lncrna(value) for value in eligible_lncrnas}
                )
            }
        )
        if lncrnas.empty:
            return
    with _bounded_duckdb(output, resource_audit) as con:
        con.register("eligible_drugs", drugs)
        lncrna_join = ""
        if lncrnas is not None:
            con.register("eligible_lncrnas", lncrnas)
            lncrna_join = "JOIN eligible_lncrnas l ON l.lncrna_id = c.lncrna_id"
        query = f"""
          SELECT DISTINCT c.cancer_id, c.lncrna_id, p.drug_id
          FROM read_parquet('{_sql_path(exact_path)}') c
          JOIN read_parquet('{_sql_path(edge_path)}') p USING(pathway_id)
          JOIN eligible_drugs d ON d.drug_id = p.drug_id
          {lncrna_join}
          WHERE c.cancer_id = '{str(cancer_id).replace("'", "''")}'
          ORDER BY p.drug_id, c.lncrna_id
        """
        reader = con.execute(query).to_arrow_reader(batch_size=int(chunk_rows))
        for batch in reader:
            frame = batch.to_pandas()
            if not frame.empty:
                frame["cancer_id"] = frame.cancer_id.astype(str).str.upper()
                frame["lncrna_id"] = frame.lncrna_id.map(base._canonical_lncrna)
                frame["drug_id"] = frame.drug_id.astype(str)
                yield frame.reset_index(drop=True)


def _eligible_candidate_count(
    exact_path: Path,
    edge_path: Path,
    eligible_pairs: pd.DataFrame,
    output: Path,
    resource_audit: Mapping[str, Any],
) -> int:
    if eligible_pairs.empty:
        return 0
    with _bounded_duckdb(output, resource_audit) as con:
        total = 0
        for cancer, local in eligible_pairs.groupby("cancer_id", observed=True):
            drugs = local[["drug_id"]].drop_duplicates().reset_index(drop=True)
            con.register("eligible_drugs_for_cancer", drugs)
            safe_cancer = str(cancer).replace("'", "''")
            total += int(
                con.execute(
                    f"""
                    SELECT count(*) FROM (
                      SELECT DISTINCT c.lncrna_id, p.drug_id
                      FROM read_parquet('{_sql_path(exact_path)}') c
                      JOIN read_parquet('{_sql_path(edge_path)}') p USING(pathway_id)
                      JOIN eligible_drugs_for_cancer e ON e.drug_id = p.drug_id
                      WHERE c.cancer_id = '{safe_cancer}'
                    ) q
                    """
                ).fetchone()[0]
            )
            con.unregister("eligible_drugs_for_cancer")
        return total


def _expression_aware_candidate_count(
    exact_path: Path,
    edge_path: Path,
    assay_fold_eligibility: pd.DataFrame,
    expression_fold_coverage: pd.DataFrame,
    output: Path,
    resource_audit: Mapping[str, Any],
) -> int:
    """Count the lossless same-fold assay/expression candidate upper bound."""

    assay = _fold_bitmask_relation(
        assay_fold_eligibility,
        key_columns=("cancer_id", "drug_id"),
        fold_column="cell_line_fold_id",
        mask_column="assay_fold_mask",
        relation_label="assay fold eligibility",
    )
    expression = _fold_bitmask_relation(
        expression_fold_coverage,
        key_columns=("cancer_id", "lncrna_id"),
        fold_column="fold_id",
        mask_column="expression_fold_mask",
        relation_label="expression fold coverage",
    )
    if assay.empty or expression.empty:
        return 0
    common_cancers = sorted(
        set(assay.cancer_id).intersection(expression.cancer_id), key=str
    )
    total = 0
    for cancer_id in common_cancers:
        local_assay = assay.loc[
            assay.cancer_id.eq(cancer_id), ["drug_id", "assay_fold_mask"]
        ].reset_index(drop=True)
        local_expression = expression.loc[
            expression.cancer_id.eq(cancer_id),
            ["lncrna_id", "expression_fold_mask"],
        ].reset_index(drop=True)
        if local_assay.empty or local_expression.empty:
            continue
        # The global DISTINCT key includes cancer_id, so its cardinality is
        # exactly the sum of these disjoint per-cancer DISTINCT(l, d) sets.
        # A fresh bounded connection per cancer also releases every spill file
        # before the next partition instead of accumulating one global hash.
        with _bounded_duckdb(output, resource_audit) as con:
            con.register("assay_fold_masks", local_assay)
            con.register("expression_fold_masks", local_expression)
            result = con.execute(
                f"""
                SELECT count(*)
                FROM (
                  SELECT DISTINCT c.lncrna_id, p.drug_id
                  FROM read_parquet('{_sql_path(exact_path)}') c
                  JOIN read_parquet('{_sql_path(edge_path)}') p USING(pathway_id)
                  JOIN assay_fold_masks a ON a.drug_id = p.drug_id
                  JOIN expression_fold_masks x ON x.lncrna_id = c.lncrna_id
                  WHERE c.cancer_id = ?
                    AND (
                      CAST(a.assay_fold_mask AS UBIGINT)
                      & CAST(x.expression_fold_mask AS UBIGINT)
                    ) != 0
                ) q
                """,
                [str(cancer_id)],
            ).fetchone()
            if result is None or len(result) != 1:
                raise base.DrugTrainingError(
                    f"Missing expression-aware candidate count for {cancer_id}"
                )
            local_count = int(result[0])
            if local_count < 0:
                raise base.DrugTrainingError(
                    f"Negative expression-aware candidate count for {cancer_id}"
                )
            total += local_count
    return total


def _fold_bitmask_relation(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    fold_column: str,
    mask_column: str,
    relation_label: str,
) -> pd.DataFrame:
    """Collapse five-fold rows to one lossless bit mask per relation key."""

    required = [*key_columns, fold_column]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise base.DrugTrainingError(
            f"{relation_label} is missing required columns: {missing}"
        )
    local = frame[required].copy()
    if local.empty:
        return pd.DataFrame(columns=[*key_columns, mask_column])
    raw_folds = local[fold_column]
    if raw_folds.isna().any():
        raise base.DrugTrainingError(f"{relation_label} contains null fold ids")
    if pd.api.types.is_bool_dtype(raw_folds.dtype) or raw_folds.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).any():
        raise base.DrugTrainingError(
            f"{relation_label} fold ids must be integers in [0, 4]"
        )
    try:
        numeric_folds = pd.to_numeric(raw_folds, errors="raise")
        fold_values = numeric_folds.to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise base.DrugTrainingError(
            f"{relation_label} fold ids must be integers in [0, 4]"
        ) from exc
    if (
        not np.isfinite(fold_values).all()
        or not np.equal(fold_values, np.floor(fold_values)).all()
        or (fold_values < 0).any()
        or (fold_values > 4).any()
    ):
        raise base.DrugTrainingError(
            f"{relation_label} fold ids must be integers in [0, 4]"
        )

    local[fold_column] = fold_values.astype(np.uint8)
    # SQL equality never matches null relation keys, so discard them before
    # grouping while still validating every supplied fold id above.
    local = local.dropna(subset=list(key_columns)).drop_duplicates(required)
    if local.empty:
        return pd.DataFrame(columns=[*key_columns, mask_column])
    local[mask_column] = np.left_shift(
        np.uint8(1), local[fold_column].to_numpy(dtype=np.uint8, copy=False)
    )
    masks = local.groupby(
        list(key_columns), observed=True, sort=False, dropna=True
    )[mask_column].sum().reset_index()
    # Fold rows were de-duplicated, therefore summing their disjoint powers of
    # two is exactly the same as bitwise OR (and is bounded by 0b11111).
    masks[mask_column] = masks[mask_column].astype(np.uint8)
    return masks


def _response_fold_eligible_pairs(
    response: pd.DataFrame,
    mapping: pd.DataFrame,
    config: StreamingDrugTrainingConfig,
) -> pd.DataFrame:
    linked = response.merge(
        mapping[[
            "dataset_id", "cell_line_id", "canonical_model_id", "cancer_id",
            "cell_line_fold_id",
        ]].drop_duplicates(),
        on=["dataset_id", "cell_line_id"], how="inner", validate="many_to_many",
    )
    counts = linked.groupby(
        ["cancer_id", "dataset_id", "drug_id", "cell_line_fold_id"],
        observed=True,
    ).canonical_model_id.nunique().rename("model_count").reset_index()
    qualified = counts.loc[
        counts.model_count >= int(config.min_cell_lines_per_dataset)
    , ["cancer_id", "dataset_id", "drug_id", "cell_line_fold_id"]]
    qualified_linked = linked.merge(
        qualified,
        on=["cancer_id", "dataset_id", "drug_id", "cell_line_fold_id"],
        how="inner",
        validate="many_to_one",
    )
    totals = qualified_linked.groupby(
        ["cancer_id", "drug_id", "cell_line_fold_id"], observed=True
    ).agg(
        total_models=("canonical_model_id", "nunique"),
        eligible_datasets=("dataset_id", "nunique"),
    ).reset_index()
    eligible = totals.loc[
        totals.total_models >= int(config.min_total_cell_lines),
        [
            "cancer_id", "drug_id", "cell_line_fold_id", "total_models",
            "eligible_datasets",
        ],
    ].copy()
    eligible["cancer_id"] = eligible.cancer_id.astype(str).str.upper()
    return eligible


def _load_and_audit_core_support(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    targets: pd.DataFrame,
    required_drugs: Iterable[str],
) -> tuple[
    dict[int, base.FoldCoreEmbeddings],
    dict[int, tuple[dict[str, np.ndarray], dict[str, int], dict[str, bool]]],
    dict[str, str],
    dict[str, Any],
]:
    """Actually load all five core bundles and audit Drug target support."""

    target_drugs = set(targets.drug_id.astype(str))
    required = set(map(str, required_drugs))
    if not target_drugs or not required:
        raise base.DrugTrainingError(
            "Drug core preflight requires non-empty annotated/native target drugs"
        )
    missing_annotations = sorted(required - target_drugs)
    if missing_annotations:
        raise base.DrugTrainingError(
            "Drug core preflight lost required target annotations: "
            f"{missing_annotations[:5]}"
        )
    all_target_genes = set(targets.gene_id.astype(str))
    cores: dict[int, base.FoldCoreEmbeddings] = {}
    drug_features: dict[
        int, tuple[dict[str, np.ndarray], dict[str, int], dict[str, bool]]
    ] = {}
    frozen_hashes: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for fold in range(base.N_FOLDS):
        core = base.load_fold_core_embeddings(manifest_path, manifest, fold)
        cores[fold] = core
        for path_text, digest in core.input_hashes.items():
            if path_text in frozen_hashes and frozen_hashes[path_text] != digest:
                raise base.DrugTrainingError(
                    "Frozen core embedding hash drifted between folds"
                )
            frozen_hashes[path_text] = digest
        vectors, target_counts, target_available = base._drug_target_vectors(
            targets, core
        )
        if set(target_counts) != target_drugs or set(target_available) != target_drugs:
            raise base.DrugTrainingError(
                f"Drug core preflight fold {fold} did not account for every target drug"
            )
        if set(vectors) != target_drugs:
            raise base.DrugTrainingError(
                f"Drug core preflight fold {fold} target vector keys drifted"
            )
        vector_widths = {int(value.shape[0]) for value in vectors.values()}
        if vector_widths != {int(core.gene.values.shape[1])}:
            raise base.DrugTrainingError(
                f"Drug core preflight fold {fold} target vector widths drifted"
            )
        if any(not np.isfinite(value).all() for value in vectors.values()):
            raise base.DrugTrainingError(
                f"Drug core preflight fold {fold} produced a non-finite target vector"
            )
        required_available = sum(
            bool(target_available.get(drug, False)) for drug in required
        )
        if required_available < 1:
            raise base.DrugTrainingError(
                f"Drug core preflight fold {fold} has no native target/core support"
            )
        drug_features[fold] = (vectors, target_counts, target_available)
        core_genes = set(map(str, core.gene.index))
        records.append(
            {
                "cell_line_fold": int(fold),
                "status": "PASS",
                "core_bundle_loaded": True,
                "export_hashes_verified": True,
                "lncrna_rows": int(len(core.lncrna.index)),
                "gene_rows": int(len(core.gene.index)),
                "cancer_rows": int(len(core.cancer.index)),
                "feature_width": int(core.gene.values.shape[1]),
                "annotated_target_drugs": int(len(target_drugs)),
                "required_native_target_drugs": int(len(required)),
                "required_drugs_with_core_target": int(required_available),
                "required_drug_core_fraction": float(
                    required_available / len(required)
                ),
                "annotated_genes": int(len(all_target_genes)),
                "annotated_genes_in_core": int(len(all_target_genes & core_genes)),
                "annotated_gene_core_fraction": float(
                    len(all_target_genes & core_genes) / max(len(all_target_genes), 1)
                ),
                "checkpoint_sha256": core.checkpoint_sha256,
                "core_parameter_sha256": core.core_parameter_sha256,
                "export_artifact_sha256s": dict(core.input_hashes),
            }
        )
    audit = {
        "audit_format": "CC_HHGT_V3_2_DRUG_CORE_TARGET_PREFLIGHT_V1",
        "status": "PASS",
        "fold_bundles_loaded": int(len(cores)),
        "all_five_core_fold_bundles_loaded": set(cores) == set(range(base.N_FOLDS)),
        "all_declared_export_hashes_verified": True,
        "target_vectors_computed_during_preflight": True,
        "target_core_unavailability_is_typed_not_imputed": True,
        "required_native_target_drugs": int(len(required)),
        "fold_records": records,
    }
    return cores, drug_features, frozen_hashes, audit


def _disk_preflight(
    output: Path,
    sparse_candidate_upper_bound: int,
    config: StreamingDrugTrainingConfig,
) -> dict[str, Any]:
    usage = shutil.disk_usage(output.parent)
    reserve = max(
        int(config.disk_reserve_bytes), int(usage.free * config.disk_reserve_fraction)
    )
    estimated = int(
        sparse_candidate_upper_bound
        * int(config.estimated_public_bytes_per_row)
        * float(config.atomic_write_multiplier)
        + config.private_evaluation_rows_per_fold * base.N_FOLDS * 96
        + 512 * 1024**2
    )
    available_after_reserve = max(0, int(usage.free) - reserve)
    fixed_bytes = int(
        config.private_evaluation_rows_per_fold * base.N_FOLDS * 96
        + 512 * 1024**2
    )
    per_candidate_peak_bytes = int(
        int(config.estimated_public_bytes_per_row)
        * float(config.atomic_write_multiplier)
    )
    maximum_sparse_rows_under_budget = max(
        0, (available_after_reserve - fixed_bytes) // max(per_candidate_peak_bytes, 1)
    )
    audit = {
        "free_bytes": int(usage.free),
        "reserve_bytes": reserve,
        "available_after_reserve_bytes": available_after_reserve,
        "sparse_candidate_upper_bound": int(sparse_candidate_upper_bound),
        "estimated_peak_incremental_bytes": estimated,
        "estimated_public_bytes_per_row": int(config.estimated_public_bytes_per_row),
        "atomic_write_multiplier": float(config.atomic_write_multiplier),
        "reserve_fraction_basis": "CURRENT_FREE_BYTES",
        "fixed_peak_incremental_bytes": fixed_bytes,
        "per_sparse_candidate_peak_bytes": per_candidate_peak_bytes,
        "maximum_sparse_rows_under_budget": int(maximum_sparse_rows_under_budget),
        "status": "PASS" if estimated <= available_after_reserve else "FAIL",
    }
    return audit


def _run_streaming_drug_training_impl(
    *,
    exact_candidates_path: str | Path,
    factored_drug_candidates_path: str | Path,
    raw_drug_response_path: str | Path,
    raw_lncrna_expression_path: str | Path,
    cell_line_map_path: str | Path,
    drug_gene_target_path: str | Path,
    core_embedding_manifest_path: str | Path,
    output_root: str | Path,
    training_run_id: str,
    curated_drug_response_path: str | Path | None = None,
    expected_staging_manifest_sha256: str | None = None,
    expected_exact_release_prediction_sha256: str | None = None,
    expected_exact_release_lineage_sha256: str | None = None,
    config: StreamingDrugTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train and publish the sparse, availability-aware V3.2 Drug module."""

    settings = config or StreamingDrugTrainingConfig()
    settings.validate()
    if not re.fullmatch(r"v32-[a-z0-9][a-z0-9._-]*", str(training_run_id).strip()):
        raise base.DrugTrainingError(
            "training_run_id must be lowercase and start with 'v32-'"
        )
    paths = {
        "exact": Path(exact_candidates_path).resolve(),
        "candidates": Path(factored_drug_candidates_path).resolve(),
        "response": Path(raw_drug_response_path).resolve(),
        "expression": Path(raw_lncrna_expression_path).resolve(),
        "mapping": Path(cell_line_map_path).resolve(),
        "targets": Path(drug_gene_target_path).resolve(),
        "core_manifest": Path(core_embedding_manifest_path).resolve(),
    }
    if curated_drug_response_path is not None:
        paths["curated"] = Path(curated_drug_response_path).resolve()
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        if key not in {"core_manifest", "candidates"}:
            base._assert_source_path(
                path, static_annotation=key in {"mapping", "targets", "curated"}
            )
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    definition, edge_path = _load_factored_definition(paths["candidates"], paths["exact"])
    pandas_input_paths = {
        key: paths[key]
        for key in ("response", "expression", "mapping", "targets", "curated")
        if key in paths
    }
    budget = _early_resource_preflight(
        output,
        int(definition["conceptual_candidate_rows"]),
        pandas_input_paths,
        settings,
    )
    budget.update(
        {
            "legacy_dense_core_bytes": int(
                int(definition["conceptual_candidate_rows"]) * 288 * 4
            ),
            "legacy_dense_core_assumed_features": 288,
            "conceptual_candidate_rows": int(definition["conceptual_candidate_rows"]),
            "resource_policy": "FAIL_CLOSED_BEFORE_ANY_LARGE_READ_DISTINCT_OR_WRITE",
        }
    )
    _atomic_json(output / "EXECUTION_BUDGET_AUDIT.json", budget)
    if budget["status"] != "PASS":
        raise base.DrugTrainingError(
            "Drug streaming early resource preflight failed: "
            f"disk={budget.get('status')}, ram={budget.get('ram_status')}, "
            f"duckdb={budget.get('duckdb_status')}"
        )
    staging_validation = _validate_staging_manifest(
        paths["candidates"],
        paths,
        output=output,
        resource_audit=budget,
        expected_staging_manifest_sha256=expected_staging_manifest_sha256,
        expected_exact_release_prediction_sha256=expected_exact_release_prediction_sha256,
        expected_exact_release_lineage_sha256=expected_exact_release_lineage_sha256,
    )
    staging_manifest = staging_validation["manifest"]
    staging_manifest_path = staging_validation["manifest_path"]
    staging_revalidation_audit = staging_validation["revalidation_audit"]
    staging_revalidation_audit_path = staging_validation["revalidation_audit_path"]
    staging_provenance_path = staging_validation["provenance_path"]
    exact_release_prediction_path = staging_validation["exact_prediction_path"]
    exact_release_lineage_path = staging_validation["exact_lineage_path"]
    input_hash_paths = {
        "exact_candidates": paths["exact"],
        "factored_candidate_root": paths["candidates"],
        "pathway_drug_edge": edge_path,
        "raw_drug_response": paths["response"],
        "raw_lncrna_expression": paths["expression"],
        "cell_line_map": paths["mapping"],
        "drug_gene_target": paths["targets"],
        "core_embedding_manifest": paths["core_manifest"],
        "staging_manifest": staging_manifest_path,
        "staging_provenance_snapshot": staging_provenance_path,
        "exact_release_prediction_proof": exact_release_prediction_path,
        "exact_release_lineage_proof": exact_release_lineage_path,
        "exact_pathway_membership": staging_validation["input_snapshot_paths"][
            "exact_pathway_membership"
        ],
    }
    if "curated" in paths:
        input_hash_paths["curated_drug_response"] = paths["curated"]
    code_paths = _execution_code_paths()
    for role, path in staging_validation["code_snapshot_paths"].items():
        if path not in code_paths.values():
            code_paths[f"staging::{role}"] = path
    input_hashes_start = _hash_snapshot(input_hash_paths)
    code_hashes_start = _hash_snapshot(code_paths)
    runtime_fingerprint = _runtime_fingerprint()
    factored_relation_audit = _audit_factored_relation(
        definition, paths["exact"], edge_path, output, budget
    )
    _atomic_json(output / "FACTORED_RELATION_AUDIT.json", factored_relation_audit)
    response = base.normalise_raw_response(base._read_table(paths["response"]))
    expression = base.normalise_raw_expression(base._read_table(paths["expression"]))
    mapping = base.assign_cell_line_folds(
        base.normalise_cell_line_map(base._read_table(paths["mapping"])), settings.seed
    )
    targets = base.normalise_drug_targets(base._read_table(paths["targets"]))
    native_drugs = set(response.drug_id.astype(str))
    targeted_drugs = set(targets.drug_id.astype(str))
    matched_drugs = native_drugs & targeted_drugs
    unmatched_drugs = native_drugs - targeted_drugs
    target_mapping_coverage = {
        "identity_scope": (
            "EXACT_DRUG_NAME_AFTER_UPPERCASE_ALPHANUMERIC_NORMALIZATION_"
            "NO_SYNONYM_CROSSWALK"
        ),
        "synonym_crosswalk_applied": False,
        "native_assayed_unique_drugs": int(len(native_drugs)),
        "native_assayed_with_target_mapping": int(len(matched_drugs)),
        "native_assayed_without_target_mapping_silent_nonmatch": int(
            len(unmatched_drugs)
        ),
        "mapping_fraction": float(len(matched_drugs) / max(len(native_drugs), 1)),
        "minimum_mapped_drugs": FORMAL_MIN_TARGET_MAPPED_DRUGS,
        "minimum_mapping_fraction": FORMAL_MIN_TARGET_MAPPING_FRACTION,
        "threshold_pass": bool(
            len(matched_drugs) >= FORMAL_MIN_TARGET_MAPPED_DRUGS
            and len(matched_drugs) / max(len(native_drugs), 1)
            >= FORMAL_MIN_TARGET_MAPPING_FRACTION
        ),
        "silent_nonmatch_examples": sorted(unmatched_drugs)[:20],
        "recomputed_from_normalized_response_and_target_tables": True,
    }
    declared_target_coverage = staging_manifest[
        "native_assayed_to_target_mapping_coverage"
    ]
    target_mapping_coverage.update(
        {
            "candidate_scope": str(declared_target_coverage["candidate_scope"]),
            "silent_nonmatch_sidecar_path": str(
                declared_target_coverage["silent_nonmatch_sidecar_path"]
            ),
            "silent_nonmatch_sidecar_sha256": str(
                declared_target_coverage["silent_nonmatch_sidecar_sha256"]
            ),
            "silent_nonmatch_sidecar_rows": int(
                declared_target_coverage["silent_nonmatch_sidecar_rows"]
            ),
        }
    )
    for key in (
        "native_assayed_unique_drugs",
        "native_assayed_with_target_mapping",
        "native_assayed_without_target_mapping_silent_nonmatch",
    ):
        if int(declared_target_coverage.get(key, -1)) != int(
            target_mapping_coverage[key]
        ):
            raise base.DrugTrainingError(
                f"Native-assayed target mapping coverage drifted for {key}"
            )
    if not np.isclose(
        float(declared_target_coverage.get("mapping_fraction", -1.0)),
        float(target_mapping_coverage["mapping_fraction"]),
        atol=1e-12,
        rtol=0,
    ):
        raise base.DrugTrainingError("Native-assayed target mapping fraction drifted")
    if target_mapping_coverage["threshold_pass"] is not True:
        raise base.DrugTrainingError("Native-assayed target mapping failed formal threshold")
    curated = (
        base.normalise_curated_response(base._read_table(paths["curated"]))
        if "curated" in paths
        else pd.DataFrame(columns=[*base.TARGET_KEYS, "pmid", "source_database"])
    )
    source_specs: list[dict[str, Any]] = [
        {"path": paths["exact"], "generation": "V3.2", "source_role": "standardized_input", "outcome_derived": False, "fold_fitted": False, "use_role": "aux_input", "artifact_id": "v32_exact_candidate_universe"},
        {"path": paths["candidates"], "generation": "V3.2", "source_role": "standardized_input", "outcome_derived": False, "fold_fitted": False, "use_role": "aux_input", "artifact_id": "v32_factored_drug_candidate_universe"},
        {"path": paths["response"], "generation": "raw_source", "source_role": "raw_data", "outcome_derived": True, "fold_fitted": False, "use_role": "training_target", "artifact_id": "raw_cell_line_drug_response"},
        {"path": paths["expression"], "generation": "raw_source", "source_role": "raw_data", "outcome_derived": False, "fold_fitted": False, "use_role": "aux_input", "artifact_id": "raw_cell_line_lncrna_expression"},
        {"path": paths["mapping"], "generation": "historical_static", "source_role": "static_annotation", "outcome_derived": False, "fold_fitted": False, "use_role": "split_control", "artifact_id": "cell_line_cancer_mapping"},
        {"path": paths["targets"], "generation": "historical_static", "source_role": "static_annotation", "outcome_derived": False, "fold_fitted": False, "use_role": "aux_input", "artifact_id": "curated_drug_gene_targets"},
        {"path": staging_manifest_path, "generation": "V3.2", "source_role": "standardized_input", "outcome_derived": False, "fold_fitted": False, "use_role": "aux_input", "artifact_id": "v32_native_drug_staging_manifest"},
    ]
    if "curated" in paths:
        source_specs.append(
            {"path": paths["curated"], "generation": "historical_curated_observation", "source_role": "raw_data", "outcome_derived": True, "fold_fitted": False, "use_role": "training_target", "artifact_id": "curated_lncrna_drug_response_observations"}
        )
    source_audit = audit_input_lineage(source_specs)
    if source_audit["status"] != "PASS":
        failures = {
            row["artifact_id"]: row["reasons"]
            for row in source_audit["artifacts"] if row["status"] != "PASS"
        }
        raise base.DrugTrainingError(f"V3.2 Drug input lineage rejected: {failures}")
    core_manifest, core_manifest_sha, core_parameter_composite = base._validate_core_manifest(
        paths["core_manifest"]
    )
    _atomic_parquet(
        mapping[[
            "dataset_id", "cell_line_id", "canonical_model_id", "cancer_id",
            "cell_line_fold_id",
        ]],
        output / "CELL_LINE_FOLD_MANIFEST.parquet",
    )
    prepared_contexts = _prepare_association_cache(
        expression, response, mapping, settings
    )
    fold_eligible = prepared_contexts.fold_eligible
    _atomic_parquet(fold_eligible, output / "NATIVE_ASSAY_FOLD_ELIGIBILITY.parquet")
    expression_fold_coverage = prepared_contexts.expression_fold_coverage
    cores, drug_features, frozen_hashes, core_target_preflight_audit = (
        _load_and_audit_core_support(
            paths["core_manifest"], core_manifest, targets, matched_drugs
        )
    )
    core_target_preflight_path = output / "CORE_TARGET_PREFLIGHT_AUDIT.json"
    _atomic_json(core_target_preflight_path, core_target_preflight_audit)
    expression_fold_coverage_path = output / "LNCRNA_EXPRESSION_FOLD_COVERAGE.parquet"
    _atomic_parquet(expression_fold_coverage, expression_fold_coverage_path)
    eligible_pairs = fold_eligible[["cancer_id", "drug_id"]].drop_duplicates()
    sparse_upper_bound = _eligible_candidate_count(
        paths["exact"], edge_path, eligible_pairs, output, budget
    )
    expression_aware_upper_bound = _expression_aware_candidate_count(
        paths["exact"], edge_path, fold_eligible, expression_fold_coverage,
        output, budget,
    )
    if expression_aware_upper_bound > sparse_upper_bound:
        raise base.DrugTrainingError(
            "Expression-aware Drug candidate bound exceeds assay-only bound"
        )
    budget.update(
        {
            "native_assay_sparse_worst_case_upper_bound_rows": int(
                sparse_upper_bound
            ),
            "same_fold_assay_expression_sparse_upper_bound_rows": int(
                expression_aware_upper_bound
            ),
            "expression_aware_upper_bound_is_lossless": True,
            "lncrna_expression_fold_coverage_path": str(
                expression_fold_coverage_path
            ),
            "lncrna_expression_fold_coverage_sha256": artifact_sha256(
                expression_fold_coverage_path
            ),
            "lncrna_expression_fold_coverage_rows": int(
                len(expression_fold_coverage)
            ),
            "factored_relation_audit": factored_relation_audit,
            "core_target_preflight_audit_path": str(core_target_preflight_path),
            "core_target_preflight_audit_sha256": artifact_sha256(
                core_target_preflight_path
            ),
            "all_five_core_fold_bundles_loaded": True,
            "prepared_association_cache": {
                key: value
                for key, value in _association_cache_audit(
                    prepared_contexts
                ).items()
                if key != "context_slice_calls"
            },
        }
    )
    _atomic_json(output / "EXECUTION_BUDGET_AUDIT.json", budget)
    if settings.preflight_only:
        for path_text, before in frozen_hashes.items():
            if artifact_sha256(path_text) != before:
                raise base.DrugTrainingError(
                    "Frozen V3.2 core embedding changed during Drug preflight: "
                    f"{path_text}"
                )
        context_cache_audit_path = output / "CONTEXT_CACHE_AUDIT.json"
        _atomic_json(
            context_cache_audit_path,
            _association_cache_audit(prepared_contexts),
        )
        input_hashes_end = _assert_hash_snapshot_unchanged(
            input_hash_paths, input_hashes_start, label="Input"
        )
        code_hashes_end = _assert_hash_snapshot_unchanged(
            code_paths, code_hashes_start, label="Code"
        )
        result = {
            "status": "PREFLIGHT_PASS",
            "analysis_version": base.ANALYSIS_VERSION,
            "module_id": base.MODULE_ID,
            "training_run_id": str(training_run_id),
            "cross_dataset_association_policy": (
                base.CROSS_DATASET_ASSOCIATION_POLICY
            ),
            "expression_moment_policy": base.EXPRESSION_MOMENT_POLICY,
            "conceptual_candidate_rows": int(definition["conceptual_candidate_rows"]),
            "native_assay_sparse_upper_bound_rows": int(sparse_upper_bound),
            "same_fold_assay_expression_sparse_upper_bound_rows": int(
                expression_aware_upper_bound
            ),
            "execution_budget_audit_path": str(output / "EXECUTION_BUDGET_AUDIT.json"),
            "heads_trained": 0,
            "predictions_written": 0,
            "old_results_used": False,
            "release_ready": False,
            "partial_not_publishable": True,
            "input_hashes_start": input_hashes_start,
            "input_hashes_end": input_hashes_end,
            "code_hashes_start": code_hashes_start,
            "code_hashes_end": code_hashes_end,
            "runtime_fingerprint": runtime_fingerprint,
            "runtime_fingerprint_sha256": _canonical_sha(runtime_fingerprint),
            "staging_revalidation_audit_path": str(staging_revalidation_audit_path),
            "staging_revalidation_audit_sha256": artifact_sha256(
                staging_revalidation_audit_path
            ),
            "staging_revalidation_audit_status": "PASS",
            "staging_revalidation_audit": staging_revalidation_audit,
            "staging_manifest_path": str(staging_manifest_path),
            "staging_manifest_sha256": staging_validation["manifest_sha256"],
            "staging_provenance_snapshot_path": str(staging_provenance_path),
            "staging_provenance_snapshot_sha256": staging_validation[
                "provenance_sha256"
            ],
            "expected_staging_manifest_sha256": staging_revalidation_audit[
                "expected_staging_manifest_sha256"
            ],
            "expected_exact_release_prediction_sha256": staging_revalidation_audit[
                "expected_exact_release_prediction_sha256"
            ],
            "expected_exact_release_lineage_sha256": staging_revalidation_audit[
                "expected_exact_release_lineage_sha256"
            ],
            "exact_candidate_source_binding": staging_validation[
                "exact_candidate_source_binding"
            ],
            "native_assayed_to_target_mapping_coverage": target_mapping_coverage,
            "core_target_preflight_audit_path": str(core_target_preflight_path),
            "core_target_preflight_audit_sha256": artifact_sha256(
                core_target_preflight_path
            ),
            "core_target_preflight_audit": core_target_preflight_audit,
            "context_cache_audit_path": str(context_cache_audit_path),
            "context_cache_audit_sha256": artifact_sha256(
                context_cache_audit_path
            ),
            "context_cache_audit": _association_cache_audit(
                prepared_contexts
            ),
        }
        _atomic_json(output / "PREFLIGHT_SUCCESS.json", result)
        return result

    cancers = sorted(eligible_pairs.cancer_id.astype(str).unique())
    checkpoint_rows: list[dict[str, Any]] = []
    fold_status: list[dict[str, Any]] = []
    heads: dict[int, Any] = {}
    domain_normalisation: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for fold in range(base.N_FOLDS):
        core = cores[fold]
        features = drug_features[fold]
        core_width = int(core.lncrna.values.shape[1] * 3)
        train_reservoir = _BalancedReservoir(settings.max_train_rows, settings.seed + fold)
        validation_reservoir = _BalancedReservoir(
            settings.max_validation_rows, settings.seed + 50_000 + fold
        )
        for cancer in cancers:
            train_context = _build_context_from_prepared(
                prepared_contexts,
                base._split_models(mapping, fold, "train"), cancer,
            )
            validation_context = _build_context_from_prepared(
                prepared_contexts,
                base._split_models(mapping, fold, "validation"), cancer,
            )
            eligible_drugs = _preeligible_drugs(train_context, settings) | _preeligible_drugs(
                validation_context, settings
            )
            eligible_lncrnas = _preeligible_lncrnas(
                train_context, validation_context
            )
            if not eligible_drugs or not eligible_lncrnas:
                continue
            vectors, target_counts, target_available = features
            for candidates in _candidate_batches(
                paths["exact"], edge_path, cancer, eligible_drugs,
                eligible_lncrnas,
                settings.candidate_chunk_rows, output, budget,
            ):
                candidate_core, core_available = base._candidate_core(candidates, core, vectors)
                target_core_available = candidates.drug_id.astype(str).map(
                    target_available
                ).fillna(False).to_numpy(dtype=bool)
                core_available &= target_core_available
                train_stats = _association_statistics_vectorized(
                    candidates, train_context, target_counts, target_available, settings
                )
                train_labels = train_stats.labels.copy()
                train_labels[~core_available] = np.nan
                train_reservoir.offer(candidate_core, train_stats.domain, train_labels)
                validation_stats = _association_statistics_vectorized(
                    candidates, validation_context, target_counts, target_available, settings
                )
                validation_labels = validation_stats.labels.copy()
                validation_labels[~core_available] = np.nan
                validation_reservoir.offer(
                    candidate_core, validation_stats.domain, validation_labels
                )
        train_core, train_domain, train_labels = train_reservoir.arrays(
            core_width, len(base._DOMAIN_FEATURES)
        )
        validation_core, validation_domain, validation_labels = validation_reservoir.arrays(
            core_width, len(base._DOMAIN_FEATURES)
        )
        if set(np.unique(train_labels)) != {0.0, 1.0}:
            fold_status.append(
                {"cell_line_fold": fold, "status": "UNAVAILABLE", "reason": "TWO_TRAINING_CLASSES_NOT_AVAILABLE"}
            )
            if settings.require_all_folds:
                raise base.DrugTrainingError(
                    f"Fresh V3.2 Drug fold {fold} lacks two training classes"
                )
            continue
        head, initialization, mean, scale, history = base._fit_head(
            train_core, train_domain, train_labels,
            validation_core, validation_domain, validation_labels,
            fold=fold, config=settings,
        )
        import torch
        from .integrated_model import (
            module_state_sha256,
            validate_private_head_checkpoint,
        )

        optimizer_steps = int(
            len(history)
            * ((len(train_labels) + int(settings.batch_size) - 1) // int(settings.batch_size))
        )
        final_parameter_sha256 = module_state_sha256(head)
        validate_private_head_checkpoint(initialization, head)
        if optimizer_steps < 1:
            raise base.DrugTrainingError(
                f"Fresh V3.2 Drug fold {fold} completed no optimizer steps"
            )
        if final_parameter_sha256 == initialization["initial_parameter_sha256"]:
            raise base.DrugTrainingError(
                f"Fresh V3.2 Drug fold {fold} parameters did not update"
            )

        checkpoint_path = output / f"cell_line_fold={fold}" / "private_head.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_temporary = checkpoint_path.with_name(".private_head.pt.tmp")
        torch.save(
            {
                "checkpoint_format": base.PRIVATE_CHECKPOINT_FORMAT,
                "analysis_version": base.ANALYSIS_VERSION,
                "module_id": base.MODULE_ID,
                "cell_line_fold": fold,
                "model_state": head.state_dict(),
                "initialization": initialization,
                "domain_features": list(base._DOMAIN_FEATURES),
                "domain_mean": mean,
                "domain_scale": scale,
                "history": history,
                "optimizer_steps": optimizer_steps,
                "final_parameter_sha256": final_parameter_sha256,
                "core_checkpoint_sha256": core.checkpoint_sha256,
                "core_parameter_sha256": core.core_parameter_sha256,
                "old_checkpoint_loaded": False,
                "old_predictions_used": False,
                "streaming_candidate_chunks": True,
            },
            checkpoint_temporary,
        )
        os.replace(checkpoint_temporary, checkpoint_path)
        checkpoint_rows.append(
            {
                "cell_line_fold": fold,
                "path": str(checkpoint_path),
                "sha256": artifact_sha256(checkpoint_path),
                "initial_parameter_sha256": initialization["initial_parameter_sha256"],
                "final_parameter_sha256": final_parameter_sha256,
                "optimizer_steps": optimizer_steps,
                "source_checkpoint_sha256": None,
                "core_checkpoint_sha256": core.checkpoint_sha256,
                "core_parameter_sha256": core.core_parameter_sha256,
                "train_rows": int(len(train_labels)),
                "validation_rows": int(len(validation_labels)),
                "status": "SUCCESS",
            }
        )
        fold_status.append(
            {
                "cell_line_fold": fold,
                "status": "SUCCESS",
                "train_rows": int(len(train_labels)),
                "validation_rows": int(len(validation_labels)),
                "optimizer_steps": optimizer_steps,
                "initial_parameter_sha256": initialization[
                    "initial_parameter_sha256"
                ],
                "final_parameter_sha256": final_parameter_sha256,
                "parameters_changed": True,
            }
        )
        heads[fold] = head
        domain_normalisation[fold] = (mean, scale)

    if settings.require_all_folds and len(heads) != base.N_FOLDS:
        raise base.DrugTrainingError(
            f"Fresh V3.2 Drug training requires five heads; observed={len(heads)}"
        )
    if not heads:
        raise base.DrugTrainingError("No fresh V3.2 Drug private head was trainable")

    from .drug_sparse_query import factor_core_entity_availability

    core_entity_availability = factor_core_entity_availability(
        lncrna_ids_by_fold={
            fold: cores[fold].lncrna.index.keys() for fold in sorted(heads)
        },
        cancer_ids_by_fold={
            fold: cores[fold].cancer.index.keys() for fold in sorted(heads)
        },
        drug_target_ids_by_fold={
            fold: (
                drug_id
                for drug_id, available in drug_features[fold][2].items()
                if available
            )
            for fold in sorted(heads)
        },
    )
    core_entity_availability_path = output / "CORE_ENTITY_FOLD_AVAILABILITY.parquet"
    _atomic_parquet(core_entity_availability, core_entity_availability_path)

    prediction_root = output / "drug_response_association"
    public_rows = 0
    public_parts = 0
    private_reservoirs = {
        fold: _FrameReservoir(
            settings.private_evaluation_rows_per_fold, settings.seed + 90_000 + fold
        ) for fold in heads
    }
    curated_count = (
        curated.groupby(list(base.TARGET_KEYS), observed=True).size().to_dict()
        if not curated.empty else {}
    )
    for cancer in cancers:
        contexts = {
            fold: _build_context_from_prepared(
                prepared_contexts,
                base._split_models(mapping, fold, "test"), cancer,
            ) for fold in heads
        }
        eligible_drugs: set[str] = set()
        for context in contexts.values():
            eligible_drugs |= _preeligible_drugs(context, settings)
        eligible_lncrnas = _preeligible_lncrnas(*contexts.values())
        part_index = 0
        for candidates in _candidate_batches(
            paths["exact"], edge_path, cancer, eligible_drugs,
            eligible_lncrnas,
            settings.candidate_chunk_rows, output, budget,
        ):
            probability_sum = np.zeros(len(candidates), dtype=np.float64)
            probability_count = np.zeros(len(candidates), dtype=np.int16)
            prediction_fold_mask = np.zeros(len(candidates), dtype=np.uint8)
            for fold, head in heads.items():
                vectors, target_counts, target_available = drug_features[fold]
                candidate_core, core_available = base._candidate_core(
                    candidates, cores[fold], vectors
                )
                target_core_available = candidates.drug_id.astype(str).map(
                    target_available
                ).fillna(False).to_numpy(dtype=bool)
                core_available &= target_core_available
                stats = _association_statistics_vectorized(
                    candidates, contexts[fold], target_counts, target_available, settings
                )
                mask = stats.assay_available & core_available
                if not mask.any():
                    continue
                mean, scale = domain_normalisation[fold]
                probability = base._predict_head(
                    head, candidate_core[mask], stats.domain[mask], mean, scale,
                    settings.prediction_batch_size,
                )
                positions = np.flatnonzero(mask)
                probability_sum[positions] += probability
                probability_count[positions] += 1
                prediction_fold_mask[positions] |= np.uint8(1 << int(fold))
                private_reservoirs[fold].offer(
                    candidates.iloc[positions].assign(
                        cell_line_fold=fold,
                        association_proxy_label=stats.labels[positions],
                        held_out_rho=stats.rho[positions],
                        held_out_n_cell_lines=stats.n_cell_lines[positions],
                        held_out_probability=probability,
                    )
                )
            available_positions = np.flatnonzero(probability_count > 0)
            if not len(available_positions):
                continue
            public = candidates.iloc[available_positions].copy()
            public["drug_response_association_probability"] = (
                probability_sum[available_positions] / probability_count[available_positions]
            )
            public["availability"] = True
            public["failure_reason"] = pd.NA
            public["cell_line_folds_with_prediction"] = probability_count[available_positions].astype(int)
            public["prediction_fold_mask"] = prediction_fold_mask[
                available_positions
            ].astype(np.uint8)
            public["evidence_scope"] = "CELL_LINE_ASSOCIATION_NOT_TCGA_PATIENT_RESPONSE"
            public["cell_line_response_only"] = True
            public["tcga_patient_response_claimed"] = False
            public["scientific_status"] = "diagnostic_only"
            public["changes_primary_ranking"] = False
            public["analysis_version"] = base.ANALYSIS_VERSION
            public["training_run_id"] = str(training_run_id)
            public["module_id"] = base.MODULE_ID
            public["target_level"] = base.TARGET_LEVEL
            public["prediction_format"] = STREAMING_PREDICTION_FORMAT
            public["curated_evidence_count"] = [
                int(curated_count.get(tuple(row), 0))
                for row in public[list(base.TARGET_KEYS)].itertuples(index=False, name=None)
            ]
            public["curated_evidence_used_as_model_feature"] = False
            public["implicit_unavailable_policy"] = IMPLICIT_UNAVAILABLE_POLICY
            from .full_model_contract import validate_public_module_frame

            validate_public_module_frame(base.MODULE_ID, public)
            destination = (
                prediction_root / f"cancer-{cancer}" / f"part-{part_index:05d}.parquet"
            )
            _atomic_parquet(public, destination)
            public_rows += len(public)
            public_parts += 1
            part_index += 1
    if public_rows == 0:
        raise base.DrugTrainingError("Streaming Drug run produced no held-out public prediction")

    private_frames = [
        reservoir.frame for reservoir in private_reservoirs.values()
        if not reservoir.frame.empty
    ]
    private_evaluation = pd.concat(private_frames, ignore_index=True) if private_frames else pd.DataFrame(
        columns=[*base.TARGET_KEYS, "cell_line_fold", "association_proxy_label", "held_out_rho", "held_out_n_cell_lines", "held_out_probability"]
    )
    private_evaluation_path = output / "drug_oof_evaluation.PRIVATE.parquet"
    _atomic_parquet(private_evaluation, private_evaluation_path)

    from .drug_sparse_query import (
        MANIFEST_FORMAT as DRUG_SPARSE_QUERY_MANIFEST_FORMAT,
        write_drug_sparse_query_sidecars,
    )

    with _sparse_query_resource_environment(output, budget):
        sparse_query_result = write_drug_sparse_query_sidecars(
            bundle_root=output,
            exact_candidates=paths["exact"],
            pathway_drug_edges=edge_path,
            assay_fold_eligibility=fold_eligible,
            expression_fold_coverage=expression_fold_coverage_path,
            core_entity_availability=core_entity_availability_path,
            available_predictions_path=prediction_root,
            conceptual_candidate_rows=int(definition["conceptual_candidate_rows"]),
            training_run_id=str(training_run_id),
            model_run_status="SUCCESS",
        )
    sparse_query_manifest_path = Path(
        str(sparse_query_result["manifest_path"])
    ).resolve()
    sparse_query_manifest = json.loads(
        sparse_query_manifest_path.read_text(encoding="utf-8")
    )
    if (
        sparse_query_manifest.get("training_run_id") != str(training_run_id)
        or sparse_query_manifest.get("analysis_version") != base.ANALYSIS_VERSION
        or sparse_query_manifest.get("module_id") != base.MODULE_ID
        or sparse_query_manifest.get("model_run_status") != "SUCCESS"
        or sparse_query_manifest.get("manifest_format")
        != DRUG_SPARSE_QUERY_MANIFEST_FORMAT
    ):
        raise base.DrugTrainingError(
            "Drug sparse query manifest drifted from the successful training run"
        )
    sparse_query_validation = sparse_query_manifest.get("validation_audit")
    if (
        not isinstance(sparse_query_validation, Mapping)
        or sparse_query_validation.get("status") != "PASS"
        or sparse_query_validation.get("sha256")
        != sparse_query_result["validation_audit_sha256"]
    ):
        raise base.DrugTrainingError(
            "Drug sparse query validation audit is absent or not PASS"
        )

    for path_text, before in frozen_hashes.items():
        if artifact_sha256(path_text) != before:
            raise base.DrugTrainingError(
                f"Frozen V3.2 core embedding changed during Drug training: {path_text}"
            )
    if artifact_sha256(paths["core_manifest"]) != core_manifest_sha:
        raise base.DrugTrainingError("Core embedding manifest changed during Drug training")
    checkpoint_manifest = {
        "checkpoint_format": base.PRIVATE_CHECKPOINT_FORMAT,
        "analysis_version": base.ANALYSIS_VERSION,
        "module_id": base.MODULE_ID,
        "folds": base.N_FOLDS,
        "records": checkpoint_rows,
        "fold_status": fold_status,
        "all_private_heads_random_initialization": all(
            row["source_checkpoint_sha256"] is None for row in checkpoint_rows
        ),
        "core_detached_and_frozen": True,
        "canonical_cell_line_fold_isolation": True,
        "optimizer_steps_total": int(
            sum(int(row["optimizer_steps"]) for row in checkpoint_rows)
        ),
        "all_heads_completed_optimizer_steps": all(
            int(row["optimizer_steps"]) > 0 for row in checkpoint_rows
        ),
        "all_head_parameters_changed": all(
            row["initial_parameter_sha256"] != row["final_parameter_sha256"]
            for row in checkpoint_rows
        ),
    }
    checkpoint_manifest_path = output / "CHECKPOINT_MANIFEST.json"
    _atomic_json(checkpoint_manifest_path, checkpoint_manifest)
    config_path = output / "RUN_CONFIG.json"
    _atomic_json(
        config_path,
        {
            "analysis_version": base.ANALYSIS_VERSION,
            "training_run_id": training_run_id,
            "training": asdict(settings),
            "estimand": "held_out_cell_line_lncrna_expression_x_drug_sensitivity_association",
            "patient_response_estimand": False,
            "candidate_storage": "FACTORED_INPUT_SPARSE_AVAILABLE_PUBLIC",
            "implicit_unavailable_policy": IMPLICIT_UNAVAILABLE_POLICY,
            "old_association_tables_allowed": False,
        },
    )
    _atomic_json(output / "INPUT_LINEAGE_AUDIT.json", source_audit)
    input_artifacts = [
        {
            "path": row["path"],
            "sha256": row["sha256"],
            "artifact_kind": (
                "raw_data" if row["source_role"] == "raw_data"
                else "annotation" if row["source_role"] == "static_annotation"
                else "standardized_input"
            ),
            "generation": row["generation"],
            "source_role": row["source_role"],
            "outcome_derived": row["outcome_derived"],
            "fold_fitted": row["fold_fitted"],
        }
        for row in source_audit["artifacts"]
    ]
    input_artifacts.append(
        {
            "path": str(paths["core_manifest"]),
            "sha256": core_manifest_sha,
            "artifact_kind": "v32_core_checkpoint",
            "generation": "V3.2",
            "source_role": "v32_core_checkpoint",
            "outcome_derived": True,
            "fold_fitted": True,
            "use_role": "aux_parent",
        }
    )
    context_cache_audit_path = output / "CONTEXT_CACHE_AUDIT.json"
    _atomic_json(
        context_cache_audit_path,
        _association_cache_audit(prepared_contexts),
    )
    input_hashes_end = _assert_hash_snapshot_unchanged(
        input_hash_paths, input_hashes_start, label="Input"
    )
    code_hashes_end = _assert_hash_snapshot_unchanged(
        code_paths, code_hashes_start, label="Code"
    )
    conceptual_rows = int(definition["conceptual_candidate_rows"])
    lineage = {
        "module_id": base.MODULE_ID,
        "analysis_version": base.ANALYSIS_VERSION,
        "training_run_id": str(training_run_id),
        "training_status": "SUCCESS",
        "initialization_policy": "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        "cross_dataset_association_policy": base.CROSS_DATASET_ASSOCIATION_POLICY,
        "expression_moment_policy": base.EXPRESSION_MOMENT_POLICY,
        "folds": base.N_FOLDS,
        "seeds": [int(settings.seed + fold * 101) for fold in range(base.N_FOLDS)],
        "code_sha256": _canonical_sha(code_hashes_start),
        "code_artifact_sha256s": code_hashes_start,
        "code_hashes_start": code_hashes_start,
        "code_hashes_end": code_hashes_end,
        "code_hashes_unchanged": code_hashes_start == code_hashes_end,
        "input_hashes_start": input_hashes_start,
        "input_hashes_end": input_hashes_end,
        "input_hashes_unchanged": input_hashes_start == input_hashes_end,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": _canonical_sha(runtime_fingerprint),
        "staging_revalidation_audit_path": str(staging_revalidation_audit_path),
        "staging_revalidation_audit_sha256": artifact_sha256(
            staging_revalidation_audit_path
        ),
        "staging_revalidation_audit_status": "PASS",
        "staging_manifest_path": str(staging_manifest_path),
        "staging_manifest_sha256": staging_validation["manifest_sha256"],
        "staging_provenance_snapshot_path": str(staging_provenance_path),
        "staging_provenance_snapshot_sha256": staging_validation[
            "provenance_sha256"
        ],
        "staging_input_artifacts_rehashed": True,
        "staging_execution_code_rehashed": True,
        "staging_runtime_fingerprint_sha256": staging_validation[
            "staging_runtime_fingerprint_sha256"
        ],
        "exact_release_lineage_contract_validated": True,
        "exact_candidate_four_key_binding_recomputed": True,
        "exact_release_prediction_sha256": staging_revalidation_audit[
            "expected_exact_release_prediction_sha256"
        ],
        "exact_release_lineage_sha256": staging_revalidation_audit[
            "expected_exact_release_lineage_sha256"
        ],
        "expected_staging_manifest_sha256": staging_revalidation_audit[
            "expected_staging_manifest_sha256"
        ],
        "expected_exact_release_prediction_sha256": staging_revalidation_audit[
            "expected_exact_release_prediction_sha256"
        ],
        "expected_exact_release_lineage_sha256": staging_revalidation_audit[
            "expected_exact_release_lineage_sha256"
        ],
        "exact_candidate_source_binding": staging_validation[
            "exact_candidate_source_binding"
        ],
        "config_sha256": artifact_sha256(config_path),
        "input_manifest_sha256": source_audit["lineage_sha256"],
        "checkpoint_manifest_sha256": artifact_sha256(checkpoint_manifest_path),
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "old_gdsc_prism_association_tables_used": False,
        "private_head_trained_from_scratch": len(checkpoint_rows) == base.N_FOLDS,
        "core_parameters_frozen": True,
        "v32_core_checkpoint_sha256": core_manifest_sha,
        "core_parameters_before_sha256": core_parameter_composite,
        "core_parameters_after_sha256": core_parameter_composite,
        "input_artifacts": input_artifacts,
        "fold_status": fold_status,
        "optimizer_steps_total": int(
            sum(int(row["optimizer_steps"]) for row in checkpoint_rows)
        ),
        "all_five_folds_have_optimizer_updates": bool(
            len(checkpoint_rows) == base.N_FOLDS
            and all(
                int(row["optimizer_steps"]) > 0
                and row["initial_parameter_sha256"]
                != row["final_parameter_sha256"]
                for row in checkpoint_rows
            )
        ),
        "split_unit": "canonical_cell_line_model",
        "canonical_model_never_crosses_folds": True,
        "cell_line_response_never_claimed_as_patient_response": True,
        "scientific_status": "diagnostic_only",
        "changes_primary_ranking": False,
        "candidate_representation": "FACTORED_NATIVE_ASSAY_ELIGIBLE",
        "candidate_scope": "TARGET_ANNOTATED_NATIVE_ASSAYED_DRUGS",
        "factored_relation_audit": factored_relation_audit,
        "native_assayed_to_target_mapping_coverage": target_mapping_coverage,
        "core_target_preflight_audit_path": str(core_target_preflight_path),
        "core_target_preflight_audit_sha256": artifact_sha256(
            core_target_preflight_path
        ),
        "core_target_preflight_audit": core_target_preflight_audit,
        "context_cache_audit_path": str(context_cache_audit_path),
        "context_cache_audit_sha256": artifact_sha256(
            context_cache_audit_path
        ),
        "context_cache_audit": _association_cache_audit(prepared_contexts),
        "conceptual_candidate_rows": conceptual_rows,
        "native_assay_sparse_upper_bound_rows": int(sparse_upper_bound),
        "same_fold_assay_expression_sparse_upper_bound_rows": int(
            expression_aware_upper_bound
        ),
        "expression_aware_candidate_prefilter": True,
        "prediction_rows": int(public_rows),
        "available_rows": int(public_rows),
        "unavailable_rows": int(conceptual_rows - public_rows),
        "implicit_unavailable_policy": IMPLICIT_UNAVAILABLE_POLICY,
        "absent_key_means_unavailable_not_zero": True,
        "all_non_null_predictions_newly_trained_v32": True,
        "candidate_chunk_rows": int(settings.candidate_chunk_rows),
        "max_candidate_core_rows_resident": int(settings.candidate_chunk_rows),
        "legacy_dense_core_materialized": False,
        "dense_candidate_table_materialized": False,
        "drug_sparse_query_manifest_path": str(sparse_query_manifest_path),
        "drug_sparse_query_manifest_sha256": str(
            sparse_query_result["manifest_sha256"]
        ),
        "drug_sparse_query_manifest_format": DRUG_SPARSE_QUERY_MANIFEST_FORMAT,
        "drug_sparse_query_manifest_status": "SUCCESS",
        "drug_sparse_query_validation_audit_path": str(
            sparse_query_result["validation_audit_path"]
        ),
        "drug_sparse_query_validation_audit_sha256": str(
            sparse_query_result["validation_audit_sha256"]
        ),
        "drug_sparse_query_validation_audit_status": "PASS",
        "drug_sparse_query_artifacts": sparse_query_manifest["artifacts"],
        "typed_absence_resolver": True,
        "available_keys_unique": True,
        "available_keys_subset_of_conceptual": True,
        "all_prediction_fold_masks_valid": True,
        "all_contributing_folds_same_fold_supported": True,
        "resolver_requires_expected_manifest_sha256": True,
        "execution_budget_audit": budget,
        "prediction_path": str(prediction_root),
        "prediction_sha256": artifact_sha256(prediction_root),
        "prediction_partitions": int(public_parts),
        "private_evaluation_path": str(private_evaluation_path),
        "private_evaluation_sha256": artifact_sha256(private_evaluation_path),
        "private_evaluation_sampling": f"bounded_reservoir_{settings.private_evaluation_rows_per_fold}_per_fold",
        "release_ready": True,
        "partial_not_publishable": False,
    }
    from .full_model_contract import validate_module_lineage

    validate_module_lineage(base.MODULE_ID, lineage)
    lineage_path = output / "MODULE_LINEAGE.json"
    _atomic_json(lineage_path, lineage)
    success = {
        "status": "SUCCESS",
        "analysis_version": base.ANALYSIS_VERSION,
        "module_id": base.MODULE_ID,
        "training_run_id": str(training_run_id),
        "cross_dataset_association_policy": base.CROSS_DATASET_ASSOCIATION_POLICY,
        "expression_moment_policy": base.EXPRESSION_MOMENT_POLICY,
        "prediction_path": str(prediction_root),
        "prediction_sha256": lineage["prediction_sha256"],
        "lineage_path": str(lineage_path),
        "lineage_sha256": artifact_sha256(lineage_path),
        "conceptual_candidate_rows": conceptual_rows,
        "available_rows": int(public_rows),
        "implicit_unavailable_rows": int(conceptual_rows - public_rows),
        "scientific_status": "diagnostic_only",
        "tcga_patient_response_claimed": False,
        "release_ready": True,
        "partial_not_publishable": False,
        "code_sha256": lineage["code_sha256"],
        "code_artifact_sha256s": code_hashes_start,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": lineage["runtime_fingerprint_sha256"],
        "staging_revalidation_audit_path": str(staging_revalidation_audit_path),
        "staging_revalidation_audit_sha256": lineage[
            "staging_revalidation_audit_sha256"
        ],
        "staging_revalidation_audit_status": "PASS",
        "staging_manifest_path": str(staging_manifest_path),
        "staging_manifest_sha256": lineage["staging_manifest_sha256"],
        "staging_provenance_snapshot_path": str(staging_provenance_path),
        "staging_provenance_snapshot_sha256": lineage[
            "staging_provenance_snapshot_sha256"
        ],
        "exact_release_prediction_sha256": lineage[
            "exact_release_prediction_sha256"
        ],
        "exact_release_lineage_sha256": lineage["exact_release_lineage_sha256"],
        "expected_staging_manifest_sha256": lineage[
            "expected_staging_manifest_sha256"
        ],
        "optimizer_steps_total": lineage["optimizer_steps_total"],
        "all_five_folds_have_optimizer_updates": lineage[
            "all_five_folds_have_optimizer_updates"
        ],
        "native_assayed_to_target_mapping_coverage": target_mapping_coverage,
        "core_target_preflight_audit_path": str(core_target_preflight_path),
        "core_target_preflight_audit_sha256": lineage[
            "core_target_preflight_audit_sha256"
        ],
        "context_cache_audit_path": str(context_cache_audit_path),
        "context_cache_audit_sha256": lineage["context_cache_audit_sha256"],
        "drug_sparse_query_manifest_path": str(sparse_query_manifest_path),
        "drug_sparse_query_manifest_sha256": lineage[
            "drug_sparse_query_manifest_sha256"
        ],
        "drug_sparse_query_validation_audit_path": lineage[
            "drug_sparse_query_validation_audit_path"
        ],
        "drug_sparse_query_validation_audit_sha256": lineage[
            "drug_sparse_query_validation_audit_sha256"
        ],
    }
    _atomic_json(output / "SUCCESS.json", success)
    return success


def run_streaming_drug_training(
    *,
    exact_candidates_path: str | Path,
    factored_drug_candidates_path: str | Path,
    raw_drug_response_path: str | Path,
    raw_lncrna_expression_path: str | Path,
    cell_line_map_path: str | Path,
    drug_gene_target_path: str | Path,
    core_embedding_manifest_path: str | Path,
    output_root: str | Path,
    training_run_id: str,
    curated_drug_response_path: str | Path | None = None,
    expected_staging_manifest_sha256: str | None = None,
    expected_exact_release_prediction_sha256: str | None = None,
    expected_exact_release_lineage_sha256: str | None = None,
    config: StreamingDrugTrainingConfig | None = None,
) -> dict[str, Any]:
    """Fail-closed public wrapper with non-reusable output and terminal markers."""

    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise base.DrugTrainingError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    in_progress = output / "RUN_IN_PROGRESS.json"
    _atomic_json(
        in_progress,
        {
            "status": "RUN_IN_PROGRESS",
            "analysis_version": base.ANALYSIS_VERSION,
            "module_id": base.MODULE_ID,
            "training_run_id": str(training_run_id),
            "release_ready": False,
            "partial_not_publishable": True,
            "success_marker_present": False,
        },
    )
    try:
        result = _run_streaming_drug_training_impl(
            exact_candidates_path=exact_candidates_path,
            factored_drug_candidates_path=factored_drug_candidates_path,
            raw_drug_response_path=raw_drug_response_path,
            raw_lncrna_expression_path=raw_lncrna_expression_path,
            cell_line_map_path=cell_line_map_path,
            drug_gene_target_path=drug_gene_target_path,
            core_embedding_manifest_path=core_embedding_manifest_path,
            output_root=output,
            training_run_id=training_run_id,
            curated_drug_response_path=curated_drug_response_path,
            expected_staging_manifest_sha256=expected_staging_manifest_sha256,
            expected_exact_release_prediction_sha256=(
                expected_exact_release_prediction_sha256
            ),
            expected_exact_release_lineage_sha256=(
                expected_exact_release_lineage_sha256
            ),
            config=config,
        )
    except Exception as error:
        for marker in (output / "SUCCESS.json", output / "PREFLIGHT_SUCCESS.json"):
            marker.unlink(missing_ok=True)
        _atomic_json(
            output / "RUN_FAILED.json",
            {
                "status": "FAILED",
                "analysis_version": base.ANALYSIS_VERSION,
                "module_id": base.MODULE_ID,
                "training_run_id": str(training_run_id),
                "error_type": type(error).__name__,
                "error": str(error),
                "release_ready": False,
                "partial_not_publishable": True,
                "success_marker_present": False,
            },
        )
        in_progress.unlink(missing_ok=True)
        raise
    in_progress.unlink(missing_ok=True)
    return result


__all__ = [
    "FACTORED_CANDIDATE_FORMAT",
    "IMPLICIT_UNAVAILABLE_POLICY",
    "StreamingDrugTrainingConfig",
    "run_streaming_drug_training",
]
