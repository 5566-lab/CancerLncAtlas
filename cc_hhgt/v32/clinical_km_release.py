"""Fresh, non-model V3.2 lncRNA Kaplan--Meier survival materialization.

Only current TCGA-CDR outcomes, current patient-level V3.2 logCPM, and the
current exact candidate universe are admitted.  No prediction, checkpoint,
ranking, or web table is an input to this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.special import erfc

if __package__:
    from .clinical_training import CLINICAL_ENDPOINTS, standardize_tcga_cdr_workbook
else:  # pragma: no cover - exercised by the standalone runner
    from clinical_training import (  # type: ignore
        CLINICAL_ENDPOINTS,
        standardize_tcga_cdr_workbook,
    )


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CC_HHGT_V3_2_FRESH_CLINICAL_KM_BINDING_V1"
RELEASE_STATUS = "SUCCESS_FRESH_V32_CLINICAL_KM_HASH_BOUND"
WORKBOOK_SHA256 = "ea594c0fbb6731477c7ac511fab449ca9c38b0d42d269591ed9f5c4090e75a5a"
EXPRESSION_ROOT_SHA256 = "9b7907242f99be0964311c66573dd799805cd3e857ba1d804e5556611a575b99"
FORMAL_CANDIDATE_SHA256 = "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
FORMAL_CANDIDATE_ROWS = 3_300_000
FORMAL_EXPRESSION_ROWS = 23_955_621
FORMAL_CANDIDATE_PAIRS = 76_734
FORMAL_STATISTICS_ROWS = FORMAL_CANDIDATE_PAIRS * len(CLINICAL_ENDPOINTS)
FORMAL_CANCERS = 33
FORMAL_LNCRNAS = 8_541
HORIZON_YEARS = (0, 1, 2, 3, 5, 10)
DAYS_PER_YEAR = 365.25
DFS_FAILURE_REASON = "NO_DISTINCT_DFS_SOURCE"
_EXPECTED_CANCERS = {
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ClinicalKMReleaseError(RuntimeError):
    """Raised when a fresh clinical-KM release fails closed."""


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def artifact_sha256(path: str | Path) -> str:
    """Hash a regular file or a symlink-free directory tree deterministically."""

    source = Path(path)
    if source.is_symlink():
        raise ClinicalKMReleaseError(f"Symlink artifacts are forbidden: {source}")
    if source.is_file():
        return _file_sha256(source)
    if not source.is_dir():
        raise ClinicalKMReleaseError(f"Artifact is missing: {source}")
    files = [item for item in source.rglob("*") if item.is_file()]
    if not files:
        raise ClinicalKMReleaseError(f"Artifact directory is empty: {source}")
    # ``WindowsPath`` ordering is case-insensitive while ``PosixPath``
    # ordering is case-sensitive.  Sorting Path objects directly therefore
    # made a formal directory hash generated on Windows impossible to verify
    # on Linux (for example ``SUCCESS.json`` versus ``cancer_id=*``).  Bind
    # the order to normalized relative POSIX names instead.
    files.sort(key=lambda item: item.relative_to(source).as_posix().casefold())
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise ClinicalKMReleaseError(f"Symlink artifacts are forbidden: {item}")
        digest.update(item.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ClinicalKMReleaseError(f"{label} is missing or unsafe: {source}")
    return source


def _safe_directory(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_dir() or source.is_symlink():
        raise ClinicalKMReleaseError(f"{label} is missing or unsafe: {source}")
    for item in source.rglob("*"):
        if item.is_symlink():
            raise ClinicalKMReleaseError(f"{label} contains a symlink: {item}")
    return source


def _sql_path(path: Path) -> str:
    quote = chr(39)
    return quote + path.resolve().as_posix().replace(quote, quote * 2) + quote


def _parquet_list(paths: list[Path]) -> str:
    return "[" + ",".join(_sql_path(path) for path in paths) + "]"


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_authority_hash(
    path: Path,
    expected: str,
    label: str,
    strict_formal_authority: bool,
) -> str:
    observed = artifact_sha256(path)
    if strict_formal_authority and observed != expected:
        raise ClinicalKMReleaseError(
            f"Formal {label} SHA256 mismatch: {observed} != {expected}"
        )
    return observed


def _validate_candidates(
    path: Path,
    *,
    strict_formal_authority: bool,
) -> tuple[pd.DataFrame, int]:
    con = duckdb.connect(":memory:")
    relation = f"read_parquet({_sql_path(path)})"
    try:
        columns = {
            row[0]
            for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        }
        if not {"cancer_id", "lncrna_id", "pathway_id"}.issubset(columns):
            raise ClinicalKMReleaseError(
                "Formal candidate universe lacks cancer/lncRNA/exact-pathway keys"
            )
        audit = con.execute(
            f"""
            SELECT count(*) AS rows,
                   count(DISTINCT cancer_id) AS cancers,
                   count(DISTINCT lncrna_id) AS lncrnas,
                   count(DISTINCT (cancer_id, lncrna_id)) AS pairs,
                   count_if(cancer_id IS NULL OR lncrna_id IS NULL
                            OR pathway_id IS NULL) AS invalid_rows
            FROM {relation}
            """
        ).fetchone()
        pairs = con.execute(
            f"""
            SELECT DISTINCT cancer_id::VARCHAR AS cancer_id,
                            lncrna_id::VARCHAR AS lncrna_id
            FROM {relation}
            ORDER BY cancer_id, lncrna_id
            """
        ).fetchdf()
    finally:
        con.close()
    row_count, cancers, lncrnas, pair_count, invalid = map(int, audit)
    if invalid or pairs.duplicated(["cancer_id", "lncrna_id"]).any():
        raise ClinicalKMReleaseError("Candidate universe has invalid or duplicate pair keys")
    if strict_formal_authority and (
        row_count != FORMAL_CANDIDATE_ROWS
        or cancers != FORMAL_CANCERS
        or lncrnas != FORMAL_LNCRNAS
        or pair_count != FORMAL_CANDIDATE_PAIRS
        or set(pairs["cancer_id"]) != _EXPECTED_CANCERS
    ):
        raise ClinicalKMReleaseError(
            f"Formal candidate counts are invalid: {audit}"
        )
    return pairs, row_count


def _validate_expression_root(
    root: Path,
    candidate_pairs: pd.DataFrame,
    *,
    strict_formal_authority: bool,
) -> tuple[dict[str, Path], int, int]:
    paths = sorted(root.glob("cancer_id=*/part-0.parquet"))
    expected = FORMAL_CANCERS if strict_formal_authority else candidate_pairs.cancer_id.nunique()
    if not paths or len(paths) != expected:
        raise ClinicalKMReleaseError(
            f"Expected {expected} expression partitions, observed {len(paths)}"
        )
    by_cancer: dict[str, Path] = {}
    for path in paths:
        cancer = path.parent.name.split("=", 1)[-1]
        if cancer in by_cancer:
            raise ClinicalKMReleaseError(f"Duplicate expression partition: {cancer}")
        by_cancer[cancer] = _safe_file(path, f"{cancer} expression partition")
    candidate_cancers = set(candidate_pairs.cancer_id.astype(str))
    if set(by_cancer) != candidate_cancers:
        raise ClinicalKMReleaseError(
            "Expression partitions do not equal the current candidate cancer scope"
        )
    con = duckdb.connect(":memory:")
    relation = f"read_parquet({_parquet_list(list(by_cancer.values()))}, hive_partitioning=false)"
    try:
        columns = {
            row[0]
            for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        }
        required = {"cancer_id", "sample_id", "patient_id", "lncrna_id", "logcpm"}
        if missing := sorted(required - columns):
            raise ClinicalKMReleaseError(
                f"Expression partitions lack columns: {missing}"
            )
        audit = con.execute(
            f"""
            SELECT count(*) AS rows,
                   count(DISTINCT (cancer_id, patient_id, lncrna_id)) AS patient_keys,
                   count(DISTINCT (cancer_id, lncrna_id)) AS pairs,
                   count(DISTINCT patient_id) AS patients,
                   count_if(cancer_id IS NULL OR patient_id IS NULL
                            OR lncrna_id IS NULL OR logcpm IS NULL
                            OR NOT isfinite(logcpm)) AS invalid_rows
            FROM {relation}
            """
        ).fetchone()
        expression_pairs = con.execute(
            f"""
            SELECT DISTINCT cancer_id::VARCHAR AS cancer_id,
                            lncrna_id::VARCHAR AS lncrna_id
            FROM {relation}
            ORDER BY cancer_id, lncrna_id
            """
        ).fetchdf()
    finally:
        con.close()
    row_count, patient_keys, pair_count, patient_count, invalid = map(int, audit)
    if invalid or row_count != patient_keys:
        raise ClinicalKMReleaseError(
            "Current expression must have one finite value per cancer/patient/lncRNA"
        )
    comparison = candidate_pairs.merge(
        expression_pairs,
        on=["cancer_id", "lncrna_id"],
        how="outer",
        indicator=True,
    )
    if not comparison["_merge"].eq("both").all():
        raise ClinicalKMReleaseError(
            "Expression cancer-lncRNA pairs do not equal the candidate universe"
        )
    if strict_formal_authority and (
        row_count != FORMAL_EXPRESSION_ROWS
        or pair_count != FORMAL_CANDIDATE_PAIRS
    ):
        raise ClinicalKMReleaseError(f"Formal expression counts are invalid: {audit}")
    return by_cancer, row_count, patient_count


def _validate_endpoints(workbook: Path, *, strict_formal_authority: bool) -> pd.DataFrame:
    endpoints, _ = standardize_tcga_cdr_workbook(workbook)
    required = {
        "cancer_id", "patient_id", "clinical_endpoint", "time_days", "event",
        "endpoint_available", "failure_reason", "source_sheet",
    }
    if missing := sorted(required - set(endpoints.columns)):
        raise ClinicalKMReleaseError(f"Standardized clinical endpoints lack: {missing}")
    if endpoints.duplicated(
        ["cancer_id", "patient_id", "clinical_endpoint"]
    ).any():
        raise ClinicalKMReleaseError("Clinical endpoint keys are not unique")
    if set(endpoints.clinical_endpoint.astype(str)) != set(CLINICAL_ENDPOINTS):
        raise ClinicalKMReleaseError("Clinical endpoint universe is not exactly six endpoints")
    dfs = endpoints.loc[endpoints.clinical_endpoint.eq("DFS")]
    if (
        dfs.empty
        or dfs.endpoint_available.astype(bool).any()
        or dfs.time_days.notna().any()
        or dfs.event.notna().any()
        or not dfs.failure_reason.astype(str).eq(DFS_FAILURE_REASON).all()
        or not dfs.source_sheet.astype(str).eq(DFS_FAILURE_REASON).all()
    ):
        raise ClinicalKMReleaseError(
            "DFS must remain unavailable because TCGA-CDR has no distinct DFS source"
        )
    available = endpoints.loc[endpoints.endpoint_available.astype(bool)]
    time = pd.to_numeric(available.time_days, errors="coerce")
    event = pd.to_numeric(available.event, errors="coerce")
    if time.isna().any() or not time.gt(0).all() or not event.isin([0, 1]).all():
        raise ClinicalKMReleaseError("Available clinical endpoints contain invalid outcomes")
    if strict_formal_authority and set(endpoints.cancer_id.astype(str)) != _EXPECTED_CANCERS:
        raise ClinicalKMReleaseError("Clinical workbook cancer universe is not formal 33-cancer scope")
    return endpoints


def _empty_endpoint_statistics(
    *,
    cancer: str,
    lncrnas: np.ndarray,
    endpoint: str,
    source_sheet: str,
    n_expression_patients: int,
    n_endpoint_patients: int,
    n_events: int,
    failure_reason: str,
    run_id: str,
) -> pd.DataFrame:
    size = len(lncrnas)
    return pd.DataFrame(
        {
            "cancer_id": np.repeat(cancer, size),
            "lncrna_id": lncrnas,
            "clinical_endpoint": np.repeat(endpoint, size),
            "availability": np.zeros(size, dtype=bool),
            "failure_reason": np.repeat(failure_reason, size),
            "endpoint_source": np.repeat(source_sheet, size),
            "n_expression_patients": np.repeat(n_expression_patients, size),
            "n_endpoint_patients": np.repeat(n_endpoint_patients, size),
            "n_events": np.repeat(n_events, size),
            "median_logcpm": np.full(size, np.nan),
            "n_high": np.full(size, np.nan),
            "n_low": np.full(size, np.nan),
            "events_high": np.full(size, np.nan),
            "events_low": np.full(size, np.nan),
            "logrank_observed_minus_expected_high": np.full(size, np.nan),
            "logrank_variance": np.full(size, np.nan),
            "logrank_z_high_vs_low": np.full(size, np.nan),
            "logrank_chi_square": np.full(size, np.nan),
            "logrank_p_value": np.full(size, np.nan),
            "grouping_method": np.repeat("ENDPOINT_COHORT_PATIENT_LOGCPM_MEDIAN", size),
            "statistical_method": np.repeat(
                "TWO_SIDED_LOGRANK_MANTEL_HAENSZEL_AND_KAPLAN_MEIER", size
            ),
            "analysis_version": np.repeat(ANALYSIS_VERSION, size),
            "computation_run_id": np.repeat(run_id, size),
        }
    )


def _risk_and_death_matrices(
    high: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return event times, total risk/death, and high-group risk/death matrices."""

    order = np.argsort(time, kind="stable")
    ordered_time = time[order]
    ordered_event = event[order].astype(bool, copy=False)
    ordered_high = high[order]
    unique_time, starts = np.unique(ordered_time, return_index=True)
    total_risk = (len(time) - starts).astype(np.int32)
    total_death = np.add.reduceat(
        ordered_event.astype(np.int16), starts
    ).astype(np.int32)
    reverse_risk = np.cumsum(
        ordered_high[::-1], axis=0, dtype=np.int32
    )[::-1]
    high_risk = reverse_risk[starts]
    high_death = np.add.reduceat(
        (ordered_high & ordered_event[:, None]).astype(np.int16),
        starts,
        axis=0,
    ).astype(np.int32)
    has_event = total_death > 0
    return (
        unique_time[has_event],
        total_risk[has_event],
        total_death[has_event],
        high_risk[has_event],
        high_death[has_event],
    )


def vectorized_logrank_km(
    expression: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    horizons_years: tuple[int, ...] = HORIZON_YEARS,
) -> dict[str, np.ndarray]:
    """Calculate all lncRNA median splits, log-ranks, and fixed-horizon KMs.

    The only loops are over the six requested horizons.  Patient-at-risk and
    event-time operations are vectorized across every lncRNA in the cancer.
    """

    values = np.asarray(expression, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.int8)
    if values.ndim != 2 or len(values) != len(time) or len(time) != len(event):
        raise ClinicalKMReleaseError("Expression/outcome arrays are not aligned")
    if not len(time) or not np.isfinite(values).all():
        raise ClinicalKMReleaseError("Vectorized KM requires finite expression patients")
    if not np.isfinite(time).all() or not (time > 0).all() or not np.isin(event, [0, 1]).all():
        raise ClinicalKMReleaseError("Vectorized KM received invalid outcome values")

    median = np.median(values, axis=0)
    high = values >= median[None, :]
    n_patients = len(time)
    n_high = high.sum(axis=0, dtype=np.int32)
    n_low = n_patients - n_high
    events_high = high[event.astype(bool)].sum(axis=0, dtype=np.int32)
    total_events = int(event.sum())
    events_low = total_events - events_high

    event_time, total_risk, total_death, high_risk, high_death = (
        _risk_and_death_matrices(high, time, event)
    )
    if not len(event_time):
        width = values.shape[1]
        nan = np.full(width, np.nan)
        return {
            "median": median,
            "n_high": n_high,
            "n_low": n_low,
            "events_high": events_high,
            "events_low": events_low,
            "observed_minus_expected_high": nan.copy(),
            "variance": nan.copy(),
            "z": nan.copy(),
            "chi_square": nan.copy(),
            "p_value": nan.copy(),
            "logrank_computable": np.zeros(width, dtype=bool),
            "survival_high": np.ones((len(horizons_years), width)),
            "survival_low": np.ones((len(horizons_years), width)),
            "at_risk_high": np.zeros((len(horizons_years), width), dtype=np.int32),
            "at_risk_low": np.zeros((len(horizons_years), width), dtype=np.int32),
            "cumulative_events_high": np.zeros(
                (len(horizons_years), width), dtype=np.int32
            ),
            "cumulative_events_low": np.zeros(
                (len(horizons_years), width), dtype=np.int32
            ),
        }

    n = total_risk.astype(np.float64)[:, None]
    d = total_death.astype(np.float64)[:, None]
    r_high = high_risk.astype(np.float64)
    d_high = high_death.astype(np.float64)
    r_low = n - r_high
    d_low = d - d_high
    expected_high = np.sum(d * r_high / n, axis=0)
    variance_terms = np.zeros_like(r_high, dtype=np.float64)
    usable_risk = n[:, 0] > 1
    variance_terms[usable_risk] = (
        r_high[usable_risk]
        * r_low[usable_risk]
        * d[usable_risk]
        * (n[usable_risk] - d[usable_risk])
        / (n[usable_risk] ** 2 * (n[usable_risk] - 1.0))
    )
    variance = variance_terms.sum(axis=0)
    observed_minus_expected = events_high.astype(float) - expected_high
    split_valid = (n_high > 0) & (n_low > 0)
    computable = split_valid & np.isfinite(variance) & (variance > 0)
    z = np.full(values.shape[1], np.nan)
    z[computable] = observed_minus_expected[computable] / np.sqrt(variance[computable])
    chi_square = np.square(z)
    p_value = np.full(values.shape[1], np.nan)
    p_value[computable] = erfc(np.abs(z[computable]) / math.sqrt(2.0))

    factor_high = np.ones_like(r_high)
    factor_low = np.ones_like(r_low)
    np.divide(d_high, r_high, out=factor_high, where=r_high > 0)
    np.divide(d_low, r_low, out=factor_low, where=r_low > 0)
    factor_high = np.where(r_high > 0, 1.0 - factor_high, 1.0)
    factor_low = np.where(r_low > 0, 1.0 - factor_low, 1.0)
    survival_event_high = np.cumprod(factor_high, axis=0)
    survival_event_low = np.cumprod(factor_low, axis=0)
    cumulative_high = np.cumsum(high_death, axis=0, dtype=np.int32)
    cumulative_low = np.cumsum(
        total_death[:, None] - high_death, axis=0, dtype=np.int32
    )

    horizon_count = len(horizons_years)
    width = values.shape[1]
    survival_high = np.ones((horizon_count, width), dtype=np.float64)
    survival_low = np.ones((horizon_count, width), dtype=np.float64)
    at_risk_high = np.zeros((horizon_count, width), dtype=np.int32)
    at_risk_low = np.zeros((horizon_count, width), dtype=np.int32)
    cumulative_events_high = np.zeros((horizon_count, width), dtype=np.int32)
    cumulative_events_low = np.zeros((horizon_count, width), dtype=np.int32)
    for horizon_index, years in enumerate(horizons_years):
        days = float(years) * DAYS_PER_YEAR
        event_index = int(np.searchsorted(event_time, days, side="right") - 1)
        if event_index >= 0:
            survival_high[horizon_index] = survival_event_high[event_index]
            survival_low[horizon_index] = survival_event_low[event_index]
            cumulative_events_high[horizon_index] = cumulative_high[event_index]
            cumulative_events_low[horizon_index] = cumulative_low[event_index]
        still_at_risk = time >= days
        at_risk_high[horizon_index] = high[still_at_risk].sum(
            axis=0, dtype=np.int32
        )
        at_risk_low[horizon_index] = int(still_at_risk.sum()) - at_risk_high[
            horizon_index
        ]

    return {
        "median": median,
        "n_high": n_high,
        "n_low": n_low,
        "events_high": events_high,
        "events_low": events_low,
        "observed_minus_expected_high": observed_minus_expected,
        "variance": variance,
        "z": z,
        "chi_square": chi_square,
        "p_value": p_value,
        "logrank_computable": computable,
        "survival_high": survival_high,
        "survival_low": survival_low,
        "at_risk_high": at_risk_high,
        "at_risk_low": at_risk_low,
        "cumulative_events_high": cumulative_events_high,
        "cumulative_events_low": cumulative_events_low,
    }


def _endpoint_statistics_and_curves(
    *,
    cancer: str,
    lncrnas: np.ndarray,
    expression: np.ndarray,
    expression_patients: np.ndarray,
    endpoint_rows: pd.DataFrame,
    endpoint: str,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    width = len(lncrnas)
    if endpoint == "DFS":
        return (
            _empty_endpoint_statistics(
                cancer=cancer,
                lncrnas=lncrnas,
                endpoint=endpoint,
                source_sheet=DFS_FAILURE_REASON,
                n_expression_patients=len(expression_patients),
                n_endpoint_patients=0,
                n_events=0,
                failure_reason=DFS_FAILURE_REASON,
                run_id=run_id,
            ),
            pd.DataFrame(),
        )

    source_sheets = endpoint_rows.source_sheet.dropna().astype(str).unique()
    source_sheet = source_sheets[0] if len(source_sheets) == 1 else "UNKNOWN_ENDPOINT_SOURCE"
    indexed = endpoint_rows.drop_duplicates("patient_id").set_index("patient_id")
    aligned = indexed.reindex(expression_patients)
    time = pd.to_numeric(aligned.time_days, errors="coerce").to_numpy(float)
    event = pd.to_numeric(aligned.event, errors="coerce").to_numpy(float)
    declared_available = aligned.endpoint_available.fillna(False).astype(bool).to_numpy()
    valid = (
        declared_available
        & np.isfinite(time)
        & (time > 0)
        & np.isin(event, [0.0, 1.0])
    )
    n_endpoint = int(valid.sum())
    total_events = int(event[valid].sum()) if n_endpoint else 0
    if not n_endpoint:
        return (
            _empty_endpoint_statistics(
                cancer=cancer,
                lncrnas=lncrnas,
                endpoint=endpoint,
                source_sheet=source_sheet,
                n_expression_patients=len(expression_patients),
                n_endpoint_patients=0,
                n_events=0,
                failure_reason="ENDPOINT_UNAVAILABLE_FOR_CANCER",
                run_id=run_id,
            ),
            pd.DataFrame(),
        )

    calculation = vectorized_logrank_km(
        expression[valid],
        time[valid],
        event[valid].astype(np.int8),
    )
    split_valid = (calculation["n_high"] > 0) & (calculation["n_low"] > 0)
    availability = calculation["logrank_computable"].copy()
    reason = np.full(width, "LOGRANK_VARIANCE_ZERO", dtype=object)
    reason[~split_valid] = "MEDIAN_SPLIT_SINGLE_GROUP"
    if total_events == 0:
        availability[:] = False
        reason[:] = "NO_OBSERVED_EVENTS"
    reason[availability] = None
    null_if_unavailable = lambda values: np.where(availability, values, np.nan)
    stats = pd.DataFrame(
        {
            "cancer_id": np.repeat(cancer, width),
            "lncrna_id": lncrnas,
            "clinical_endpoint": np.repeat(endpoint, width),
            "availability": availability,
            "failure_reason": reason,
            "endpoint_source": np.repeat(source_sheet, width),
            "n_expression_patients": np.repeat(len(expression_patients), width),
            "n_endpoint_patients": np.repeat(n_endpoint, width),
            "n_events": np.repeat(total_events, width),
            "median_logcpm": calculation["median"],
            "n_high": calculation["n_high"],
            "n_low": calculation["n_low"],
            "events_high": calculation["events_high"],
            "events_low": calculation["events_low"],
            "logrank_observed_minus_expected_high": null_if_unavailable(
                calculation["observed_minus_expected_high"]
            ),
            "logrank_variance": null_if_unavailable(calculation["variance"]),
            "logrank_z_high_vs_low": null_if_unavailable(calculation["z"]),
            "logrank_chi_square": null_if_unavailable(calculation["chi_square"]),
            "logrank_p_value": null_if_unavailable(calculation["p_value"]),
            "grouping_method": np.repeat(
                "ENDPOINT_COHORT_PATIENT_LOGCPM_MEDIAN", width
            ),
            "statistical_method": np.repeat(
                "TWO_SIDED_LOGRANK_MANTEL_HAENSZEL_AND_KAPLAN_MEIER", width
            ),
            "analysis_version": np.repeat(ANALYSIS_VERSION, width),
            "computation_run_id": np.repeat(run_id, width),
        }
    )

    available_index = np.flatnonzero(availability)
    if not len(available_index):
        return stats, pd.DataFrame()
    curve_parts: list[pd.DataFrame] = []
    for group in ("HIGH", "LOW"):
        survival = calculation[f"survival_{group.lower()}"]
        at_risk = calculation[f"at_risk_{group.lower()}"]
        cumulative = calculation[f"cumulative_events_{group.lower()}"]
        for horizon_index, years in enumerate(HORIZON_YEARS):
            curve_parts.append(
                pd.DataFrame(
                    {
                        "cancer_id": np.repeat(cancer, len(available_index)),
                        "lncrna_id": lncrnas[available_index],
                        "clinical_endpoint": np.repeat(
                            endpoint, len(available_index)
                        ),
                        "expression_group": np.repeat(group, len(available_index)),
                        "horizon_years": np.repeat(years, len(available_index)),
                        "horizon_days": np.repeat(
                            float(years) * DAYS_PER_YEAR, len(available_index)
                        ),
                        "n_at_risk": at_risk[horizon_index, available_index],
                        "cumulative_events": cumulative[
                            horizon_index, available_index
                        ],
                        "survival_probability": survival[
                            horizon_index, available_index
                        ],
                        "analysis_version": np.repeat(
                            ANALYSIS_VERSION, len(available_index)
                        ),
                        "computation_run_id": np.repeat(
                            run_id, len(available_index)
                        ),
                    }
                )
            )
    return stats, pd.concat(curve_parts, ignore_index=True)


def _load_expression_matrix(
    path: Path,
    cancer: str,
    lncrnas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(
        path,
        columns=["cancer_id", "patient_id", "lncrna_id", "logcpm"],
    )
    frame = frame.loc[frame.cancer_id.astype(str).eq(cancer)].copy()
    if frame.empty:
        raise ClinicalKMReleaseError(f"Expression partition is empty for {cancer}")
    if frame.duplicated(["patient_id", "lncrna_id"]).any():
        raise ClinicalKMReleaseError(f"Expression patient-lncRNA keys duplicate for {cancer}")
    observed_lncrnas = set(frame.lncrna_id.astype(str))
    if observed_lncrnas != set(map(str, lncrnas)):
        raise ClinicalKMReleaseError(
            f"Expression lncRNA scope does not equal candidate scope for {cancer}"
        )
    matrix = frame.pivot(
        index="patient_id", columns="lncrna_id", values="logcpm"
    ).sort_index(axis=0)
    matrix = matrix.reindex(columns=lncrnas)
    values = matrix.to_numpy(dtype=np.float64, copy=False)
    if values.size != len(matrix.index) * len(lncrnas) or not np.isfinite(values).all():
        raise ClinicalKMReleaseError(
            f"Expression matrix is incomplete or non-finite for {cancer}"
        )
    return matrix.index.astype(str).to_numpy(), values


def _cast_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    for column in (
        "n_expression_patients", "n_endpoint_patients", "n_events", "n_high",
        "n_low", "events_high", "events_low",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame["availability"] = frame.availability.astype(bool)
    numeric = (
        "median_logcpm", "logrank_observed_minus_expected_high",
        "logrank_variance", "logrank_z_high_vs_low", "logrank_chi_square",
        "logrank_p_value",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return frame


def _curve_schema_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cancer_id": pd.Series(dtype="string"),
            "lncrna_id": pd.Series(dtype="string"),
            "clinical_endpoint": pd.Series(dtype="string"),
            "expression_group": pd.Series(dtype="string"),
            "horizon_years": pd.Series(dtype="int16"),
            "horizon_days": pd.Series(dtype="float64"),
            "n_at_risk": pd.Series(dtype="int64"),
            "cumulative_events": pd.Series(dtype="int64"),
            "survival_probability": pd.Series(dtype="float64"),
            "analysis_version": pd.Series(dtype="string"),
            "computation_run_id": pd.Series(dtype="string"),
        }
    )


def materialize_clinical_km_release(
    *,
    workbook_path: str | Path,
    expression_root: str | Path,
    exact_candidate_path: str | Path,
    output_root: str | Path,
    runner_path: str | Path,
    strict_formal_authority: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    """Materialize fresh log-rank/KM facts into a new immutable release root."""

    workbook = _safe_file(workbook_path, "TCGA-CDR workbook")
    expression = _safe_directory(expression_root, "Current V3.2 expression root")
    candidates_path = _safe_file(exact_candidate_path, "Exact candidate universe")
    code_path = _safe_file(Path(__file__), "Clinical KM materializer code")
    runner = _safe_file(runner_path, "Clinical KM runner")
    destination = Path(output_root).resolve()
    if destination.exists():
        raise ClinicalKMReleaseError(
            f"Refusing to overwrite existing clinical KM release: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    workbook_sha = _validate_authority_hash(
        workbook, WORKBOOK_SHA256, "TCGA-CDR workbook", strict_formal_authority
    )
    expression_sha = _validate_authority_hash(
        expression,
        EXPRESSION_ROOT_SHA256,
        "expression tree",
        strict_formal_authority,
    )
    candidate_sha = _validate_authority_hash(
        candidates_path,
        FORMAL_CANDIDATE_SHA256,
        "candidate universe",
        strict_formal_authority,
    )
    candidate_pairs, candidate_rows = _validate_candidates(
        candidates_path, strict_formal_authority=strict_formal_authority
    )
    expression_paths, expression_rows, expression_patients = _validate_expression_root(
        expression,
        candidate_pairs,
        strict_formal_authority=strict_formal_authority,
    )
    endpoints = _validate_endpoints(
        workbook, strict_formal_authority=strict_formal_authority
    )
    algorithm_policy = {
        "grouping": "median patient logCPM among patients with that endpoint; HIGH >= median; LOW < median",
        "patient_expression_aggregation": "not_applicable_one_current_value_per_cancer_patient_lncrna",
        "logrank": "two-sided Mantel-Haenszel log-rank with tied event times",
        "km_estimator": "Kaplan-Meier product limit",
        "horizon_years": list(HORIZON_YEARS),
        "days_per_year": DAYS_PER_YEAR,
        "dfs_policy": DFS_FAILURE_REASON,
        "vectorization": "cancer_endpoint_patient_by_lncrna_matrix",
    }
    run_id = "V32-CLINICAL-KM-" + _canonical_json_sha256(
        {
            "workbook": workbook_sha,
            "expression": expression_sha,
            "candidates": candidate_sha,
            "algorithm": algorithm_policy,
            "code": artifact_sha256(code_path),
        }
    )[:16].upper()

    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    stage.mkdir(parents=False, exist_ok=False)
    curves_root = stage / "clinical_km_curves"
    curves_root.mkdir()
    statistics_parts: list[pd.DataFrame] = []
    curve_rows = 0
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        cancers = sorted(candidate_pairs.cancer_id.astype(str).unique())
        for cancer_index, cancer in enumerate(cancers, start=1):
            if progress:
                print(
                    f"[clinical-km] cancer={cancer} ({cancer_index}/{len(cancers)})",
                    flush=True,
                )
            lncrnas = (
                candidate_pairs.loc[candidate_pairs.cancer_id.eq(cancer), "lncrna_id"]
                .astype(str)
                .sort_values(kind="stable")
                .to_numpy()
            )
            patient_ids, matrix = _load_expression_matrix(
                expression_paths[cancer], cancer, lncrnas
            )
            cancer_curves: list[pd.DataFrame] = []
            for endpoint in CLINICAL_ENDPOINTS:
                endpoint_rows = endpoints.loc[
                    endpoints.cancer_id.astype(str).eq(cancer)
                    & endpoints.clinical_endpoint.astype(str).eq(endpoint)
                ].copy()
                if endpoint != "DFS" and endpoint_rows.empty:
                    raise ClinicalKMReleaseError(
                        f"Clinical source lacks {cancer}/{endpoint} rows"
                    )
                stats, curves = _endpoint_statistics_and_curves(
                    cancer=cancer,
                    lncrnas=lncrnas,
                    expression=matrix,
                    expression_patients=patient_ids,
                    endpoint_rows=endpoint_rows,
                    endpoint=endpoint,
                    run_id=run_id,
                )
                statistics_parts.append(stats)
                if not curves.empty:
                    cancer_curves.append(curves)
            if cancer_curves:
                curve_frame = pd.concat(cancer_curves, ignore_index=True)
                endpoint_order = {value: index for index, value in enumerate(CLINICAL_ENDPOINTS)}
                curve_frame["_endpoint_order"] = curve_frame.clinical_endpoint.map(
                    endpoint_order
                )
                curve_frame = curve_frame.sort_values(
                    [
                        "lncrna_id", "_endpoint_order", "expression_group",
                        "horizon_years",
                    ],
                    kind="stable",
                ).drop(columns="_endpoint_order")
                curve_frame.to_parquet(
                    curves_root / f"part-{cancer}.parquet",
                    index=False,
                    compression="zstd",
                )
                curve_rows += len(curve_frame)
            del matrix

        statistics = _cast_statistics(pd.concat(statistics_parts, ignore_index=True))
        endpoint_order = {value: index for index, value in enumerate(CLINICAL_ENDPOINTS)}
        statistics["_endpoint_order"] = statistics.clinical_endpoint.map(endpoint_order)
        statistics = statistics.sort_values(
            ["cancer_id", "lncrna_id", "_endpoint_order"], kind="stable"
        ).drop(columns="_endpoint_order").reset_index(drop=True)
        expected_rows = len(candidate_pairs) * len(CLINICAL_ENDPOINTS)
        if (
            len(statistics) != expected_rows
            or statistics.duplicated(
                ["cancer_id", "lncrna_id", "clinical_endpoint"]
            ).any()
        ):
            raise ClinicalKMReleaseError("Statistics do not cover the typed pair-endpoint universe")
        dfs = statistics.loc[statistics.clinical_endpoint.eq("DFS")]
        if (
            len(dfs) != len(candidate_pairs)
            or dfs.availability.any()
            or not dfs.failure_reason.astype(str).eq(DFS_FAILURE_REASON).all()
            or dfs.logrank_p_value.notna().any()
        ):
            raise ClinicalKMReleaseError("DFS output semantics were violated")
        available = statistics.availability.astype(bool)
        if (
            statistics.loc[available, "failure_reason"].notna().any()
            or statistics.loc[available, "logrank_p_value"].isna().any()
            or not statistics.loc[available, "logrank_p_value"].between(0, 1).all()
            or statistics.loc[~available, "failure_reason"].isna().any()
            or statistics.loc[~available, "logrank_p_value"].notna().any()
            or curve_rows != int(available.sum()) * 2 * len(HORIZON_YEARS)
        ):
            raise ClinicalKMReleaseError("Availability or curve cardinality semantics failed")
        if strict_formal_authority and len(statistics) != FORMAL_STATISTICS_ROWS:
            raise ClinicalKMReleaseError("Formal statistics row count is not 460,404")

        statistics_path = stage / "clinical_km_statistics.parquet"
        statistics.to_parquet(statistics_path, index=False, compression="zstd")
        if not any(curves_root.rglob("*.parquet")):
            _curve_schema_frame().to_parquet(
                curves_root / "part-empty.parquet", index=False, compression="zstd"
            )

        final_statistics_path = destination / statistics_path.name
        final_curves_root = destination / curves_root.name
        final_source_path = destination / "SOURCE_INPUTS.json"
        final_lineage_path = destination / "MODULE_LINEAGE.json"
        final_binding_path = destination / "CLINICAL_KM_BINDING.json"
        source_inputs = {
            "analysis_version": ANALYSIS_VERSION,
            "input_policy": "ONLY_CURRENT_RAW_OUTCOMES_CURRENT_PATIENT_LOGCPM_AND_CURRENT_EXACT_CANDIDATES",
            "fresh_statistical_calculation": True,
            "training_not_applicable": True,
            "historical_derived_outputs_used": False,
            "historical_predictions_used": False,
            "historical_checkpoints_used": False,
            "historical_rankings_used": False,
            "historical_web_tables_used": False,
            "inputs": {
                "tcga_cdr_workbook": {
                    "path": str(workbook), "sha256": workbook_sha, "role": "raw_outcomes"
                },
                "current_patient_logcpm": {
                    "path": str(expression),
                    "sha256": expression_sha,
                    "role": "current_v32_standardized_measurement",
                    "rows": expression_rows,
                    "patients": expression_patients,
                },
                "current_exact_candidates": {
                    "path": str(candidates_path),
                    "sha256": candidate_sha,
                    "role": "current_v32_static_scope",
                    "rows": candidate_rows,
                    "cancer_lncrna_pairs": len(candidate_pairs),
                },
            },
        }
        source_path = stage / final_source_path.name
        _json_write(source_path, source_inputs)
        completed_at = datetime.now(timezone.utc).isoformat()
        lineage = {
            "analysis_version": ANALYSIS_VERSION,
            "module_id": "clinical_lncrna_kaplan_meier",
            "status": "SUCCESS_FRESH_V32_STATISTICAL_CALCULATION",
            "computation_run_id": run_id,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "fresh_statistical_calculation": True,
            "training_not_applicable": True,
            "all_output_rows_generated_current_run": True,
            "historical_derived_outputs_used": False,
            "historical_predictions_used": False,
            "historical_checkpoints_used": False,
            "historical_rankings_used": False,
            "historical_web_tables_used": False,
            "changes_primary_ranking": False,
            "release_ready": False,
            "production_deployed": False,
            "algorithm_policy": algorithm_policy,
            "code": {
                "path": str(code_path), "sha256": artifact_sha256(code_path)
            },
            "runner": {"path": str(runner), "sha256": artifact_sha256(runner)},
        }
        lineage_path = stage / final_lineage_path.name
        _json_write(lineage_path, lineage)
        counts = {
            "candidate_pairs": len(candidate_pairs),
            "statistics_rows": len(statistics),
            "available_rows": int(available.sum()),
            "unavailable_rows": int((~available).sum()),
            "curve_rows": int(curve_rows),
            "cancers": int(candidate_pairs.cancer_id.nunique()),
            "lncrnas": int(candidate_pairs.lncrna_id.nunique()),
            "endpoints": len(CLINICAL_ENDPOINTS),
            "dfs_rows": len(dfs),
            "dfs_available_rows": int(dfs.availability.sum()),
            "horizons": len(HORIZON_YEARS),
        }
        binding = {
            "format": BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": RELEASE_STATUS,
            "module_id": "clinical_lncrna_kaplan_meier",
            "result_role": "SECONDARY_FRESH_LNCRNA_SURVIVAL_STATISTICS",
            "computation_run_id": run_id,
            "formal_authority": bool(strict_formal_authority),
            "fresh_statistical_calculation": True,
            "training_not_applicable": True,
            "all_output_rows_generated_current_run": True,
            "historical_derived_outputs_used": False,
            "historical_predictions_used": False,
            "historical_checkpoints_used": False,
            "historical_rankings_used": False,
            "historical_web_tables_used": False,
            "changes_primary_ranking": False,
            "release_ready": False,
            "production_deployed": False,
            "clinical_endpoints": list(CLINICAL_ENDPOINTS),
            "horizon_years": list(HORIZON_YEARS),
            "dfs_policy": DFS_FAILURE_REASON,
            "algorithm_policy": algorithm_policy,
            "counts": counts,
            "authorities": source_inputs["inputs"],
            "artifacts": {
                "statistics": {
                    "path": str(final_statistics_path),
                    "sha256": artifact_sha256(statistics_path),
                    "rows": len(statistics),
                },
                "curves": {
                    "path": str(final_curves_root),
                    "sha256": artifact_sha256(curves_root),
                    "rows": int(curve_rows),
                },
                "source_inputs": {
                    "path": str(final_source_path),
                    "sha256": artifact_sha256(source_path),
                },
                "module_lineage": {
                    "path": str(final_lineage_path),
                    "sha256": artifact_sha256(lineage_path),
                },
            },
        }
        binding_path = stage / final_binding_path.name
        _json_write(binding_path, binding)
        binding_sha = artifact_sha256(binding_path)
        success = {
            "status": RELEASE_STATUS,
            "analysis_version": ANALYSIS_VERSION,
            "computation_run_id": run_id,
            "binding": final_binding_path.name,
            "binding_sha256": binding_sha,
            "statistics_rows": len(statistics),
            "curve_rows": int(curve_rows),
            "fresh_statistical_calculation": True,
            "training_not_applicable": True,
            "changes_primary_ranking": False,
            "release_ready": False,
            "production_deployed": False,
        }
        _json_write(stage / "SUCCESS.json", success)
        stage.replace(destination)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    return {
        "output_root": str(destination),
        "binding": str(destination / "CLINICAL_KM_BINDING.json"),
        "binding_sha256": binding_sha,
        "computation_run_id": run_id,
        "counts": counts,
        "authorities": {
            "tcga_cdr_workbook": workbook_sha,
            "current_patient_logcpm": expression_sha,
            "current_exact_candidates": candidate_sha,
        },
        "release_ready": False,
        "production_deployed": False,
    }


__all__ = [
    "ANALYSIS_VERSION",
    "BINDING_FORMAT",
    "ClinicalKMReleaseError",
    "DAYS_PER_YEAR",
    "DFS_FAILURE_REASON",
    "EXPRESSION_ROOT_SHA256",
    "FORMAL_CANDIDATE_PAIRS",
    "FORMAL_CANDIDATE_SHA256",
    "FORMAL_STATISTICS_ROWS",
    "HORIZON_YEARS",
    "RELEASE_STATUS",
    "WORKBOOK_SHA256",
    "artifact_sha256",
    "materialize_clinical_km_release",
    "vectorized_logrank_km",
]
