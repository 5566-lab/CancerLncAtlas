"""Fresh, hash-bound V3.2 tumour lncRNA-gene coexpression release.

This module consumes expression measurements and covariates only.  Historical
coexpression edges, predictions, rankings, checkpoints and web tables are never
accepted as inputs.  Coexpression is an observational association and is not a
claim of physical binding or causality.
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
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from cc_hhgt.stats import (
    correlation_p_values,
    design_rank,
    prepare_design,
    rank_transform,
    residualize,
    standardize,
)


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CC_HHGT_V3_2_BULK_COEXPRESSION_BINDING_V1"
FORMAL_CANDIDATE_SHA256 = (
    "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
)
FORMAL_LNCRNA_EXPRESSION_TREE_SHA256 = (
    "9b7907242f99be0964311c66573dd799805cd3e857ba1d804e5556611a575b99"
)
FORMAL_GENE_EXPRESSION_TREE_SHA256 = (
    "cbec5777b511e65adcc892daf1aa2c2dc2ea236d4f3cccd9a9a3f1368587747c"
)
FORMAL_COVARIATE_SHA256 = (
    "72bb9b268926af7accd5c090a3e22afc5bb13a81df5e158cb38f037adceff948"
)
FORMAL_CANDIDATE_ROWS = 3_300_000
FORMAL_CANDIDATE_PAIRS = 76_734
FORMAL_LNCRNAS = 8_541
FORMAL_CANCERS = 33
EXPECTED_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
)
DEFAULT_COVARIATES = (
    "purity",
    "age_years",
    "sex",
    "stage",
    "molecular_subtype",
    "clinical_subtype",
    "technical_batch",
    "leukocyte_fraction",
    "cell_fraction_Macrophages.M0",
    "cell_fraction_Macrophages.M1",
    "cell_fraction_Macrophages.M2",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BulkCoexpressionReleaseError(RuntimeError):
    """Raised when fresh coexpression materialisation fails closed."""


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def artifact_sha256(path: str | Path) -> str:
    """Hash a regular file or an entire symlink-free tree deterministically."""

    source = Path(path).resolve()
    if source.is_symlink():
        raise BulkCoexpressionReleaseError(f"Symlink artifacts are forbidden: {source}")
    if source.is_file():
        return _file_sha256(source)
    if not source.is_dir():
        raise BulkCoexpressionReleaseError(f"Artifact is missing: {source}")
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise BulkCoexpressionReleaseError(f"Artifact directory is empty: {source}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise BulkCoexpressionReleaseError(f"Symlink artifacts are forbidden: {item}")
        digest.update(item.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise BulkCoexpressionReleaseError(f"{label} is missing or unsafe: {source}")
    return source


def _safe_dir(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_dir() or source.is_symlink():
        raise BulkCoexpressionReleaseError(f"{label} is missing or unsafe: {source}")
    return source


def _sql_path(path: Path) -> str:
    quote = chr(39)
    return quote + path.resolve().as_posix().replace(quote, quote * 2) + quote


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:24]}"


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def _candidate_pairs(path: Path, strict: bool) -> tuple[pd.DataFrame, int]:
    if strict and artifact_sha256(path) != FORMAL_CANDIDATE_SHA256:
        raise BulkCoexpressionReleaseError("Formal exact candidate SHA256 mismatch")
    con = duckdb.connect(":memory:")
    try:
        relation = f"read_parquet({_sql_path(path)})"
        columns = {
            row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        }
        if not {"cancer_id", "lncrna_id", "pathway_id"}.issubset(columns):
            raise BulkCoexpressionReleaseError("Exact candidates lack required keys")
        rows = int(con.execute(f"SELECT count(*) FROM {relation}").fetchone()[0])
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
    if pairs.duplicated(["cancer_id", "lncrna_id"]).any():
        raise BulkCoexpressionReleaseError("Exact candidate pairs are duplicated")
    if strict and (
        rows != FORMAL_CANDIDATE_ROWS
        or len(pairs) != FORMAL_CANDIDATE_PAIRS
        or pairs["cancer_id"].nunique() != FORMAL_CANCERS
        or pairs["lncrna_id"].nunique() != FORMAL_LNCRNAS
        or set(pairs["cancer_id"]) != set(EXPECTED_CANCERS)
    ):
        raise BulkCoexpressionReleaseError("Formal exact candidate universe is invalid")
    return pairs, rows


def _partition_map(root: Path, entity_label: str, cancers: Iterable[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for cancer in cancers:
        path = root / f"cancer_id={cancer}" / "part-0.parquet"
        if not path.is_file() or path.is_symlink():
            raise BulkCoexpressionReleaseError(
                f"Missing {entity_label} partition for {cancer}: {path}"
            )
        output[cancer] = path
    return output


def _tumour_mask(sample_ids: pd.Series) -> pd.Series:
    codes = sample_ids.astype(str).str.slice(13, 15)
    numeric = pd.to_numeric(codes, errors="coerce")
    return numeric.between(1, 9, inclusive="both")


def _patient_matrix(
    path: Path,
    *,
    cancer: str,
    entity_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = ["cancer_id", "sample_id", "patient_id", entity_column, "logcpm"]
    frame = pd.read_parquet(path, columns=required)
    if set(frame["cancer_id"].astype(str)) != {cancer}:
        raise BulkCoexpressionReleaseError(f"{cancer} partition contains another cancer")
    invalid = (
        frame[["sample_id", "patient_id", entity_column, "logcpm"]].isna().any(axis=1)
        | ~np.isfinite(pd.to_numeric(frame["logcpm"], errors="coerce"))
    )
    if invalid.any():
        raise BulkCoexpressionReleaseError(
            f"{cancer}/{entity_column} contains {int(invalid.sum())} invalid measurements"
        )
    source_samples = int(frame["sample_id"].nunique())
    frame = frame.loc[_tumour_mask(frame["sample_id"])].copy()
    if frame.empty:
        raise BulkCoexpressionReleaseError(f"{cancer}/{entity_column} has no tumour rows")
    tumour_samples = int(frame["sample_id"].nunique())
    duplicate = frame.duplicated(["patient_id", entity_column], keep=False)
    duplicate_rows = int(duplicate.sum())
    if duplicate_rows:
        frame = (
            frame.groupby(["patient_id", entity_column], as_index=False, observed=True)[
                "logcpm"
            ]
            .mean()
        )
    else:
        frame = frame[["patient_id", entity_column, "logcpm"]]
    matrix = frame.pivot(index="patient_id", columns=entity_column, values="logcpm")
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.sort_index().sort_index(axis=1).astype(np.float32)
    return matrix, {
        "source_samples": source_samples,
        "tumour_samples": tumour_samples,
        "patients": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "duplicate_patient_feature_rows_averaged": duplicate_rows,
    }


def _variance_filter(matrix: pd.DataFrame, minimum: float) -> tuple[pd.DataFrame, set[str]]:
    variance = matrix.var(axis=0, ddof=1)
    keep = variance.index[variance.ge(minimum) & np.isfinite(variance)]
    dropped = set(matrix.columns) - set(keep)
    return matrix.loc[:, keep], dropped


def _empty_edges() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "edge_id", "cancer_id", "lncrna_id", "gene_id", "rho", "p_value",
            "fdr", "direction", "n_patients", "residual_design_rank",
            "correlation_df", "method", "analysis_version", "computation_run_id",
            "physical_binding_claimed", "causal_effect_claimed",
        ]
    )


def _empty_cluster_membership() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cancer_id", "cluster_id", "lncrna_id", "member_rank",
            "member_strength", "n_edges", "positive_edges", "negative_edges",
            "representative", "analysis_version", "computation_run_id",
        ]
    )


def _empty_cluster_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cancer_id", "cluster_id", "n_lncrnas", "n_unique_genes", "n_edges",
            "mean_abs_rho", "median_abs_rho", "representative_lncrna_id",
            "top_gene_ids_json", "clustering_method", "analysis_version",
            "computation_run_id",
        ]
    )


def _covariate_design(
    covariates: pd.DataFrame,
    *,
    cancer: str,
    patients: list[str],
    columns: tuple[str, ...],
) -> tuple[np.ndarray, list[str], int]:
    frame = covariates.loc[
        covariates["cancer_id"].astype(str).eq(cancer)
    ].copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    if frame.duplicated("patient_id").any():
        aggregations: dict[str, str] = {}
        for column in frame.columns:
            if column in {"cancer_id", "sample_id", "patient_id"}:
                continue
            aggregations[column] = (
                "mean" if pd.api.types.is_numeric_dtype(frame[column]) else "first"
            )
        frame = frame.groupby("patient_id", as_index=False).agg(aggregations)
    frame = frame.set_index("patient_id").reindex(patients)
    used = [column for column in columns if column in frame.columns]
    missing_patients = int(frame[used].isna().all(axis=1).sum()) if used else len(patients)
    design = (
        prepare_design(frame[used].reset_index(drop=True))
        if used
        else np.ones((len(patients), 1), dtype=float)
    )
    return design, used, missing_patients


def _residual_rank_matrix(matrix: pd.DataFrame, design: np.ndarray) -> np.ndarray:
    filled = matrix.copy()
    for column in filled.columns[filled.isna().any(axis=0)]:
        median = filled[column].median()
        filled[column] = filled[column].fillna(median if np.isfinite(median) else 0.0)
    ranked = rank_transform(filled.to_numpy(dtype=np.float32, copy=False))
    return standardize(residualize(ranked, design))


def _conservative_bh_candidates(
    lnc_values: np.ndarray,
    gene_values: np.ndarray,
    *,
    n_patients: int,
    residual_rank: int,
    block_size: int,
    min_abs_rho: float,
    max_fdr: float,
) -> tuple[np.ndarray, np.ndarray, float | None, int]:
    """Compute conservative global-BH values after the effect-size filter."""

    denominator = max(n_patients - 1, 1)
    total_tests = int(lnc_values.shape[1] * gene_values.shape[1])
    candidates: list[np.ndarray] = []
    for start in range(0, lnc_values.shape[1], block_size):
        stop = min(start + block_size, lnc_values.shape[1])
        corr = (lnc_values[:, start:stop].T @ gene_values) / denominator
        p_values = correlation_p_values(
            corr,
            n_patients,
            residual_design_rank=residual_rank,
        )
        mask = (np.abs(corr) >= min_abs_rho) & (p_values <= max_fdr)
        if mask.any():
            candidates.append(p_values[mask].astype(np.float64, copy=False))
    if not candidates:
        return np.empty(0), np.empty(0), None, total_tests
    sorted_p = np.sort(np.concatenate(candidates), kind="stable")
    ranks = np.arange(1, len(sorted_p) + 1, dtype=np.float64)
    adjusted = sorted_p * float(total_tests) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    passing = np.flatnonzero(adjusted <= max_fdr)
    cutoff = float(sorted_p[passing[-1]]) if len(passing) else None
    return sorted_p, adjusted, cutoff, total_tests


def _select_edges(
    lnc_values: np.ndarray,
    gene_values: np.ndarray,
    *,
    lnc_ids: np.ndarray,
    gene_ids: np.ndarray,
    cancer: str,
    n_patients: int,
    residual_rank: int,
    block_size: int,
    min_abs_rho: float,
    max_fdr: float,
    max_edges_per_direction: int,
    sorted_candidate_p: np.ndarray,
    adjusted_candidate_p: np.ndarray,
    p_cutoff: float | None,
    run_id: str,
) -> pd.DataFrame:
    if p_cutoff is None:
        return _empty_edges()
    denominator = max(n_patients - 1, 1)
    rows: dict[str, list[Any]] = {
        "cancer_id": [], "lncrna_id": [], "gene_id": [], "rho": [],
        "p_value": [], "fdr": [], "direction": [],
    }
    for start in range(0, lnc_values.shape[1], block_size):
        stop = min(start + block_size, lnc_values.shape[1])
        corr = (lnc_values[:, start:stop].T @ gene_values) / denominator
        p_values = correlation_p_values(
            corr,
            n_patients,
            residual_design_rank=residual_rank,
        )
        for local_index in range(stop - start):
            rho_row = corr[local_index]
            p_row = p_values[local_index]
            candidate = np.flatnonzero(
                (np.abs(rho_row) >= min_abs_rho) & (p_row <= p_cutoff)
            )
            if not len(candidate):
                continue
            q_index = np.searchsorted(
                sorted_candidate_p,
                p_row[candidate],
                side="left",
            )
            q_index = np.clip(q_index, 0, len(adjusted_candidate_p) - 1)
            q_values = adjusted_candidate_p[q_index]
            candidate = candidate[q_values <= max_fdr]
            q_values = q_values[q_values <= max_fdr]
            if not len(candidate):
                continue
            for positive in (True, False):
                sign_mask = rho_row[candidate] >= 0 if positive else rho_row[candidate] < 0
                selected = candidate[sign_mask]
                selected_q = q_values[sign_mask]
                if not len(selected):
                    continue
                order = np.argsort(-np.abs(rho_row[selected]), kind="stable")
                order = order[:max_edges_per_direction]
                selected = selected[order]
                selected_q = selected_q[order]
                count = len(selected)
                rows["cancer_id"].extend([cancer] * count)
                rows["lncrna_id"].extend([str(lnc_ids[start + local_index])] * count)
                rows["gene_id"].extend(gene_ids[selected].astype(str).tolist())
                rows["rho"].extend(rho_row[selected].astype(float).tolist())
                rows["p_value"].extend(p_row[selected].astype(float).tolist())
                rows["fdr"].extend(selected_q.astype(float).tolist())
                rows["direction"].extend(["positive" if positive else "negative"] * count)
    if not rows["lncrna_id"]:
        return _empty_edges()
    edges = pd.DataFrame(rows)
    edges["edge_id"] = [
        _stable_id("COEX", cancer, lnc, gene)
        for lnc, gene in zip(edges["lncrna_id"], edges["gene_id"])
    ]
    edges["n_patients"] = n_patients
    edges["residual_design_rank"] = residual_rank
    edges["correlation_df"] = n_patients - residual_rank - 1
    edges["method"] = (
        "patient_aligned_covariate_residual_spearman_conservative_global_bh"
    )
    edges["analysis_version"] = ANALYSIS_VERSION
    edges["computation_run_id"] = run_id
    edges["physical_binding_claimed"] = False
    edges["causal_effect_claimed"] = False
    return edges[
        [
            "edge_id", "cancer_id", "lncrna_id", "gene_id", "rho", "p_value",
            "fdr", "direction", "n_patients", "residual_design_rank",
            "correlation_df", "method", "analysis_version", "computation_run_id",
            "physical_binding_claimed", "causal_effect_claimed",
        ]
    ].sort_values(
        ["lncrna_id", "direction", "fdr", "rho"],
        ascending=[True, True, True, False],
        kind="stable",
    ).reset_index(drop=True)


def _cluster_edges(
    edges: pd.DataFrame,
    *,
    cancer: str,
    run_id: str,
    max_clusters: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if edges.empty:
        return _empty_cluster_membership(), _empty_cluster_summary()
    lncs = np.sort(edges["lncrna_id"].astype(str).unique())
    genes = np.sort(edges["gene_id"].astype(str).unique())
    lnc_index = {value: index for index, value in enumerate(lncs)}
    gene_index = {value: index for index, value in enumerate(genes)}
    matrix = sparse.csr_matrix(
        (
            edges["rho"].to_numpy(float),
            (
                edges["lncrna_id"].map(lnc_index).to_numpy(int),
                edges["gene_id"].map(gene_index).to_numpy(int),
            ),
        ),
        shape=(len(lncs), len(genes)),
    )
    if len(lncs) == 1:
        labels = np.zeros(1, dtype=int)
        method = "single_nonempty_profile"
    else:
        n_components = max(1, min(32, len(lncs) - 1, len(genes) - 1))
        if n_components >= 1 and min(matrix.shape) > 1:
            embedding = TruncatedSVD(
                n_components=n_components,
                algorithm="randomized",
                n_iter=7,
                random_state=random_state,
            ).fit_transform(matrix)
            embedding = normalize(embedding, norm="l2", axis=1)
            method = f"signed_edge_profile_svd{n_components}_minibatch_kmeans"
        else:
            embedding = matrix.toarray()
            method = "signed_edge_profile_minibatch_kmeans"
        n_clusters = min(
            max_clusters,
            len(lncs),
            max(2, int(round(math.sqrt(len(lncs) / 2.0)))),
        )
        labels = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            batch_size=min(1024, max(64, len(lncs))),
        ).fit_predict(embedding)

    strength = edges.assign(abs_rho=edges["rho"].abs()).groupby(
        "lncrna_id", observed=True
    )["abs_rho"].sum()
    representative: dict[int, str] = {}
    for label in np.unique(labels):
        members = lncs[labels == label]
        representative[int(label)] = sorted(
            members,
            key=lambda value: (-float(strength.get(value, 0.0)), str(value)),
        )[0]
    canonical_order = sorted(representative, key=lambda label: representative[label])
    canonical = {label: index + 1 for index, label in enumerate(canonical_order)}
    cluster_by_lnc = {
        str(lnc): f"COEXC:{cancer}:{canonical[int(label)]:03d}"
        for lnc, label in zip(lncs, labels)
    }

    edge_stats = edges.assign(abs_rho=edges["rho"].abs()).groupby(
        "lncrna_id", as_index=False, observed=True
    ).agg(
        member_strength=("abs_rho", "sum"),
        n_edges=("edge_id", "size"),
        positive_edges=("direction", lambda values: int((values == "positive").sum())),
        negative_edges=("direction", lambda values: int((values == "negative").sum())),
    )
    edge_stats["cancer_id"] = cancer
    edge_stats["cluster_id"] = edge_stats["lncrna_id"].map(cluster_by_lnc)
    edge_stats = edge_stats.sort_values(
        ["cluster_id", "member_strength", "lncrna_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    edge_stats["member_rank"] = edge_stats.groupby("cluster_id").cumcount() + 1
    representative_by_cluster = {
        f"COEXC:{cancer}:{canonical[label]:03d}": value
        for label, value in representative.items()
    }
    edge_stats["representative"] = [
        representative_by_cluster[str(cluster_id)] == str(lnc)
        for cluster_id, lnc in zip(edge_stats["cluster_id"], edge_stats["lncrna_id"])
    ]
    edge_stats["analysis_version"] = ANALYSIS_VERSION
    edge_stats["computation_run_id"] = run_id
    membership = edge_stats[
        [
            "cancer_id", "cluster_id", "lncrna_id", "member_rank",
            "member_strength", "n_edges", "positive_edges", "negative_edges",
            "representative", "analysis_version", "computation_run_id",
        ]
    ].reset_index(drop=True)

    summaries: list[dict[str, Any]] = []
    work = edges.copy()
    work["cluster_id"] = work["lncrna_id"].map(cluster_by_lnc)
    work["abs_rho"] = work["rho"].abs()
    for cluster_id, group in work.groupby("cluster_id", sort=True, observed=True):
        top_genes = (
            group.groupby("gene_id", observed=True)["abs_rho"]
            .sum()
            .sort_values(ascending=False, kind="stable")
            .head(25)
            .index.astype(str)
            .tolist()
        )
        summaries.append(
            {
                "cancer_id": cancer,
                "cluster_id": str(cluster_id),
                "n_lncrnas": int(group["lncrna_id"].nunique()),
                "n_unique_genes": int(group["gene_id"].nunique()),
                "n_edges": int(len(group)),
                "mean_abs_rho": float(group["abs_rho"].mean()),
                "median_abs_rho": float(group["abs_rho"].median()),
                "representative_lncrna_id": representative_by_cluster[str(cluster_id)],
                "top_gene_ids_json": json.dumps(top_genes, separators=(",", ":")),
                "clustering_method": method,
                "analysis_version": ANALYSIS_VERSION,
                "computation_run_id": run_id,
            }
        )
    summary = pd.DataFrame(summaries).sort_values("cluster_id").reset_index(drop=True)
    return membership, summary


def _availability_table(
    candidate_lncs: list[str],
    *,
    cancer: str,
    run_id: str,
    n_patients: int,
    residual_rank_value: int | None,
    correlation_df: int | None,
    variable_lncs: set[str],
    edge_counts: pd.Series,
    cluster_by_lnc: dict[str, str],
    global_failure: str | None,
    min_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lncrna_id in candidate_lncs:
        if global_failure is not None:
            available = False
            failure_reason = global_failure
            result_status = "UNAVAILABLE"
        elif lncrna_id not in variable_lncs:
            available = False
            failure_reason = "LNC_LOW_VARIANCE_OR_MISSING_EXPRESSION"
            result_status = "UNAVAILABLE"
        else:
            available = True
            failure_reason = None
            result_status = (
                "PASS_WITH_SIGNIFICANT_EDGES"
                if int(edge_counts.get(lncrna_id, 0)) > 0
                else "PASS_NO_SIGNIFICANT_EDGES"
            )
        rows.append(
            {
                "cancer_id": cancer,
                "lncrna_id": lncrna_id,
                "availability": available,
                "failure_reason": failure_reason,
                "result_status": result_status,
                "n_patients": n_patients,
                "minimum_required_patients": min_samples,
                "residual_design_rank": residual_rank_value,
                "correlation_df": correlation_df,
                "n_edges": int(edge_counts.get(lncrna_id, 0)),
                "cluster_available": lncrna_id in cluster_by_lnc,
                "cluster_id": cluster_by_lnc.get(lncrna_id),
                "analysis_version": ANALYSIS_VERSION,
                "computation_run_id": run_id,
                "historical_derived_outputs_used": False,
                "physical_binding_claimed": False,
                "causal_effect_claimed": False,
            }
        )
    return pd.DataFrame(rows)


def _analyse_cancer(
    *,
    cancer: str,
    candidate_lncs: list[str],
    lnc_path: Path,
    gene_path: Path,
    covariates: pd.DataFrame,
    run_id: str,
    min_samples: int,
    min_variance: float,
    min_abs_rho: float,
    max_fdr: float,
    max_edges_per_direction: int,
    block_size: int,
    max_clusters: int,
    random_state: int,
    covariate_columns: tuple[str, ...],
    strict_formal_authority: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    lnc, lnc_audit = _patient_matrix(
        lnc_path,
        cancer=cancer,
        entity_column="lncrna_id",
    )
    gene, gene_audit = _patient_matrix(
        gene_path,
        cancer=cancer,
        entity_column="gene_id",
    )
    candidate_set = set(candidate_lncs)
    observed_lnc = set(lnc.columns)
    if strict_formal_authority and observed_lnc != candidate_set:
        raise BulkCoexpressionReleaseError(
            f"{cancer} lncRNA expression/candidate mismatch: "
            f"missing={len(candidate_set - observed_lnc)}, extra={len(observed_lnc - candidate_set)}"
        )
    available_lnc = sorted(candidate_set & observed_lnc)
    lnc = lnc.loc[:, available_lnc]
    lnc_patients = set(lnc.index)
    gene_patients = set(gene.index)
    patients = sorted(lnc_patients & gene_patients)
    n_patients = len(patients)
    patient_audit = {
        "lnc_only_patients": len(lnc_patients - gene_patients),
        "gene_only_patients": len(gene_patients - lnc_patients),
        "aligned_patients": n_patients,
    }

    global_failure: str | None = None
    residual_rank_value: int | None = None
    correlation_df: int | None = None
    variable_lncs: set[str] = set()
    variable_gene_count = 0
    total_tests = 0
    bh_candidate_count = 0
    bh_p_cutoff: float | None = None
    covariates_used: list[str] = []
    missing_covariate_patients = 0
    edges = _empty_edges()
    membership = _empty_cluster_membership()
    clusters = _empty_cluster_summary()

    if n_patients < min_samples:
        global_failure = f"INSUFFICIENT_PATIENTS_MIN_{min_samples}"
    elif not available_lnc:
        global_failure = "NO_CANDIDATE_LNCRNA_EXPRESSION"
    else:
        lnc = lnc.reindex(patients)
        gene = gene.reindex(patients)
        lnc, _ = _variance_filter(lnc, min_variance)
        gene, _ = _variance_filter(gene, min_variance)
        variable_lncs = set(lnc.columns.astype(str))
        variable_gene_count = int(gene.shape[1])
        if not variable_lncs:
            global_failure = "NO_VARIABLE_LNCRNAS"
        elif gene.empty:
            global_failure = "NO_VARIABLE_GENES"
        else:
            design, covariates_used, missing_covariate_patients = _covariate_design(
                covariates,
                cancer=cancer,
                patients=patients,
                columns=covariate_columns,
            )
            residual_rank_value = design_rank(design)
            correlation_df = n_patients - residual_rank_value - 1
            if correlation_df <= 0:
                global_failure = "NO_RESIDUAL_CORRELATION_DEGREES_OF_FREEDOM"
            else:
                lnc_values = _residual_rank_matrix(lnc, design)
                gene_values = _residual_rank_matrix(gene, design)
                sorted_p, adjusted_p, bh_p_cutoff, total_tests = (
                    _conservative_bh_candidates(
                        lnc_values,
                        gene_values,
                        n_patients=n_patients,
                        residual_rank=residual_rank_value,
                        block_size=block_size,
                        min_abs_rho=min_abs_rho,
                        max_fdr=max_fdr,
                    )
                )
                bh_candidate_count = len(sorted_p)
                edges = _select_edges(
                    lnc_values,
                    gene_values,
                    lnc_ids=lnc.columns.to_numpy(str),
                    gene_ids=gene.columns.to_numpy(str),
                    cancer=cancer,
                    n_patients=n_patients,
                    residual_rank=residual_rank_value,
                    block_size=block_size,
                    min_abs_rho=min_abs_rho,
                    max_fdr=max_fdr,
                    max_edges_per_direction=max_edges_per_direction,
                    sorted_candidate_p=sorted_p,
                    adjusted_candidate_p=adjusted_p,
                    p_cutoff=bh_p_cutoff,
                    run_id=run_id,
                )
                membership, clusters = _cluster_edges(
                    edges,
                    cancer=cancer,
                    run_id=run_id,
                    max_clusters=max_clusters,
                    random_state=random_state,
                )

    edge_counts = (
        edges.groupby("lncrna_id", observed=True).size()
        if not edges.empty
        else pd.Series(dtype="int64")
    )
    cluster_by_lnc = (
        membership.set_index("lncrna_id")["cluster_id"].astype(str).to_dict()
        if not membership.empty
        else {}
    )
    availability = _availability_table(
        candidate_lncs,
        cancer=cancer,
        run_id=run_id,
        n_patients=n_patients,
        residual_rank_value=residual_rank_value,
        correlation_df=correlation_df,
        variable_lncs=variable_lncs,
        edge_counts=edge_counts,
        cluster_by_lnc=cluster_by_lnc,
        global_failure=global_failure,
        min_samples=min_samples,
    )
    summary = {
        "cancer_id": cancer,
        "status": (
            "UNAVAILABLE" if global_failure else "PASS_FRESH_V32_COEXPRESSION"
        ),
        "failure_reason": global_failure,
        "candidate_lncrnas": len(candidate_lncs),
        "variable_lncrnas": len(variable_lncs),
        "variable_genes": variable_gene_count,
        "n_patients": n_patients,
        "residual_design_rank": residual_rank_value,
        "correlation_df": correlation_df,
        "total_tests": total_tests,
        "bh_effect_candidates": bh_candidate_count,
        "bh_p_cutoff": bh_p_cutoff,
        "edges": len(edges),
        "lncrnas_with_edges": int(edges["lncrna_id"].nunique()) if not edges.empty else 0,
        "clusters": len(clusters),
        "covariates_used_json": json.dumps(covariates_used, separators=(",", ":")),
        "missing_covariate_patients": missing_covariate_patients,
        "lnc_source_samples": lnc_audit["source_samples"],
        "lnc_tumour_samples": lnc_audit["tumour_samples"],
        "gene_source_samples": gene_audit["source_samples"],
        "gene_tumour_samples": gene_audit["tumour_samples"],
        **patient_audit,
        "analysis_version": ANALYSIS_VERSION,
        "computation_run_id": run_id,
    }
    return edges, availability, membership, clusters, summary


def _write_partition(root: Path, cancer: str, frame: pd.DataFrame) -> Path | None:
    if frame.empty:
        return None
    destination = root / f"cancer_id={cancer}" / "part-0.parquet"
    destination.parent.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(destination, index=False, compression="zstd")
    return destination


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def materialize_bulk_coexpression_release(
    *,
    lncrna_expression_root: str | Path,
    gene_expression_root: str | Path,
    covariates_path: str | Path,
    exact_candidate_path: str | Path,
    output_root: str | Path,
    runner_path: str | Path,
    strict_formal_authority: bool = True,
    min_samples: int = 40,
    min_variance: float = 0.01,
    min_abs_rho: float = 0.20,
    max_fdr: float = 0.05,
    max_edges_per_direction: int = 75,
    block_size: int = 128,
    max_clusters: int = 32,
    random_state: int = 20260826,
    covariate_columns: tuple[str, ...] = DEFAULT_COVARIATES,
) -> dict[str, Any]:
    """Compute a new V3.2 coexpression release from patient-level measurements."""

    if (
        min_samples < 4
        or min_variance < 0
        or not 0 < min_abs_rho < 1
        or not 0 < max_fdr <= 1
        or max_edges_per_direction < 1
        or block_size < 1
        or max_clusters < 1
    ):
        raise BulkCoexpressionReleaseError("Invalid coexpression configuration")
    lnc_root = _safe_dir(lncrna_expression_root, "lncRNA expression root")
    gene_root = _safe_dir(gene_expression_root, "gene expression root")
    cov_path = _safe_file(covariates_path, "Covariate table")
    candidate_path = _safe_file(exact_candidate_path, "Exact candidates")
    runner = _safe_file(runner_path, "Coexpression runner")
    code = _safe_file(Path(__file__), "Coexpression code")
    destination = Path(output_root).resolve()
    if destination.exists():
        raise BulkCoexpressionReleaseError(
            f"Refusing to overwrite existing coexpression release: {destination}"
        )

    input_hashes = {
        "lncrna_expression_tree": artifact_sha256(lnc_root),
        "gene_expression_tree": artifact_sha256(gene_root),
        "covariates": artifact_sha256(cov_path),
        "exact_candidates": artifact_sha256(candidate_path),
    }
    if strict_formal_authority:
        expected = {
            "lncrna_expression_tree": FORMAL_LNCRNA_EXPRESSION_TREE_SHA256,
            "gene_expression_tree": FORMAL_GENE_EXPRESSION_TREE_SHA256,
            "covariates": FORMAL_COVARIATE_SHA256,
            "exact_candidates": FORMAL_CANDIDATE_SHA256,
        }
        if input_hashes != expected:
            raise BulkCoexpressionReleaseError(
                f"Formal source authority mismatch: {input_hashes} != {expected}"
            )
    pairs, candidate_rows = _candidate_pairs(candidate_path, strict_formal_authority)
    cancers = (
        list(EXPECTED_CANCERS)
        if strict_formal_authority
        else sorted(pairs["cancer_id"].astype(str).unique())
    )
    if not cancers:
        raise BulkCoexpressionReleaseError("No cancers are available")
    lnc_partitions = _partition_map(lnc_root, "lncRNA expression", cancers)
    gene_partitions = _partition_map(gene_root, "gene expression", cancers)
    covariates = pd.read_parquet(cov_path)
    if not {"cancer_id", "sample_id", "patient_id"}.issubset(covariates.columns):
        raise BulkCoexpressionReleaseError("Covariate table lacks required identifiers")
    if covariates.duplicated(["cancer_id", "sample_id"]).any():
        raise BulkCoexpressionReleaseError("Covariate sample keys are duplicated")

    config = {
        "min_samples": min_samples,
        "min_variance": min_variance,
        "min_abs_rho": min_abs_rho,
        "max_fdr": max_fdr,
        "max_edges_per_direction": max_edges_per_direction,
        "block_size": block_size,
        "max_clusters": max_clusters,
        "random_state": random_state,
        "covariates": list(covariate_columns),
        "patient_alignment": "patient_id_tumour_samples_only_mean_duplicate_aliquots",
        "method": "covariate_residual_spearman_conservative_global_bh",
    }
    authority = {
        "analysis_version": ANALYSIS_VERSION,
        "inputs": input_hashes,
        "candidate_rows": candidate_rows,
        "config": config,
        "code_sha256": artifact_sha256(code),
        "runner_sha256": artifact_sha256(runner),
    }
    run_id = "V32-BULK-COEXPRESSION-" + _canonical_hash(authority)[:16].upper()
    stage = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.partial"
    stage.mkdir(parents=True, exist_ok=False)
    edge_root = stage / "tumor_lncrna_gene_coexpression"
    availability_root = stage / "coexpression_availability"
    membership_root = stage / "coexpression_cluster_membership"
    cluster_root = stage / "coexpression_cluster_summary"
    for root in (edge_root, availability_root, membership_root, cluster_root):
        root.mkdir()
    status_path = stage / "RUN_STATUS.json"
    _json_dump(
        status_path,
        {
            "status": "RUNNING",
            "analysis_version": ANALYSIS_VERSION,
            "computation_run_id": run_id,
            "completed_cancers": [],
            "production_deployed": False,
        },
    )

    summaries: list[dict[str, Any]] = []
    total_edges = 0
    total_availability = 0
    total_available = 0
    total_membership = 0
    total_clusters = 0
    completed: list[str] = []
    try:
        for cancer in cancers:
            candidate_lncs = (
                pairs.loc[pairs["cancer_id"].astype(str).eq(cancer), "lncrna_id"]
                .astype(str)
                .sort_values()
                .tolist()
            )
            edges, availability, membership, clusters, summary = _analyse_cancer(
                cancer=cancer,
                candidate_lncs=candidate_lncs,
                lnc_path=lnc_partitions[cancer],
                gene_path=gene_partitions[cancer],
                covariates=covariates,
                run_id=run_id,
                min_samples=min_samples,
                min_variance=min_variance,
                min_abs_rho=min_abs_rho,
                max_fdr=max_fdr,
                max_edges_per_direction=max_edges_per_direction,
                block_size=block_size,
                max_clusters=max_clusters,
                random_state=random_state,
                covariate_columns=covariate_columns,
                strict_formal_authority=strict_formal_authority,
            )
            if (
                len(availability) != len(candidate_lncs)
                or availability.duplicated(["cancer_id", "lncrna_id"]).any()
                or int(availability["n_edges"].sum()) != len(edges)
                or (not edges.empty and (
                    edges["edge_id"].duplicated().any()
                    or not edges["rho"].abs().ge(min_abs_rho - 1e-12).all()
                    or not edges["fdr"].le(max_fdr + 1e-12).all()
                    or edges.groupby(["lncrna_id", "direction"]).size().max()
                    > max_edges_per_direction
                ))
            ):
                raise BulkCoexpressionReleaseError(
                    f"{cancer} coexpression semantic audit failed"
                )
            _write_partition(edge_root, cancer, edges)
            _write_partition(availability_root, cancer, availability)
            _write_partition(membership_root, cancer, membership)
            _write_partition(cluster_root, cancer, clusters)
            summaries.append(summary)
            total_edges += len(edges)
            total_availability += len(availability)
            total_available += int(availability["availability"].sum())
            total_membership += len(membership)
            total_clusters += len(clusters)
            completed.append(cancer)
            _json_dump(
                status_path,
                {
                    "status": "RUNNING",
                    "analysis_version": ANALYSIS_VERSION,
                    "computation_run_id": run_id,
                    "completed_cancers": completed,
                    "last_cancer_summary": summary,
                    "production_deployed": False,
                },
            )
            print(
                f"[bulk-coexpression] {cancer} patients={summary['n_patients']} "
                f"edges={len(edges)} clusters={len(clusters)}",
                flush=True,
            )
    except Exception as exc:
        _json_dump(
            stage / "FAILURE.json",
            {
                "status": "FAILED",
                "analysis_version": ANALYSIS_VERSION,
                "computation_run_id": run_id,
                "completed_cancers": completed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "production_deployed": False,
            },
        )
        raise

    cancer_summary = pd.DataFrame(summaries).sort_values("cancer_id").reset_index(drop=True)
    cancer_summary_path = stage / "coexpression_cancer_summary.parquet"
    cancer_summary.to_parquet(cancer_summary_path, index=False, compression="zstd")
    if (
        len(cancer_summary) != len(cancers)
        or total_availability != len(pairs)
        or total_edges <= 0
        or total_membership <= 0
        or total_clusters <= 0
    ):
        raise BulkCoexpressionReleaseError("Aggregate coexpression semantic audit failed")
    if strict_formal_authority and (
        len(cancers) != FORMAL_CANCERS
        or total_availability != FORMAL_CANDIDATE_PAIRS
        or set(cancer_summary["cancer_id"]) != set(EXPECTED_CANCERS)
    ):
        raise BulkCoexpressionReleaseError("Formal coexpression coverage is invalid")

    source_path = stage / "SOURCE_INPUTS.json"
    lineage_path = stage / "MODULE_LINEAGE.json"
    binding_path = stage / "BULK_COEXPRESSION_BINDING.json"
    success_path = stage / "SUCCESS.json"
    source_inputs = {
        "analysis_version": ANALYSIS_VERSION,
        "computation_run_id": run_id,
        "input_policy": {
            "allowed_roles": [
                "PROVENANCE_AUDITED_STANDARDIZED_SOURCE",
                "RAW_OUTCOME_FREE_MEASUREMENT",
                "TASK_DEFINITION_EXACT_CANDIDATE_UNIVERSE",
            ],
            "historical_coexpression_edges_used": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "historical_web_tables_used": False,
        },
        "inputs": {
            "lncrna_expression": {
                "path": str(lnc_root),
                "sha256": input_hashes["lncrna_expression_tree"],
                "role": "CURRENT_V32_FORMAL_LNCRNA_MEASUREMENTS",
                "partitions": len(lnc_partitions),
            },
            "gene_expression": {
                "path": str(gene_root),
                "sha256": input_hashes["gene_expression_tree"],
                "role": "PROVENANCE_AUDITED_STANDARDIZED_GENE_MEASUREMENTS",
                "partitions": len(gene_partitions),
                "derived_coexpression_output": False,
            },
            "covariates": {
                "path": str(cov_path),
                "sha256": input_hashes["covariates"],
                "role": "PROVENANCE_AUDITED_STANDARDIZED_COVARIATES",
            },
            "exact_candidates": {
                "path": str(candidate_path),
                "sha256": input_hashes["exact_candidates"],
                "role": "CURRENT_V32_TASK_DEFINITION",
                "rows": candidate_rows,
                "pairs": len(pairs),
            },
        },
        "configuration": config,
    }
    _json_dump(source_path, source_inputs)
    lineage = {
        "analysis_version": ANALYSIS_VERSION,
        "computation_run_id": run_id,
        "status": "SUCCESS_FRESH_V32_BULK_COEXPRESSION",
        "result_role": "TUMOUR_LNCRNA_GENE_COEXPRESSION_AND_PROFILE_CLUSTERS",
        "fresh_statistical_calculation": True,
        "training_not_applicable": True,
        "all_output_rows_generated_current_run": True,
        "historical_derived_outputs_used": False,
        "historical_coexpression_edges_used": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "patient_alignment": config["patient_alignment"],
        "changes_primary_exact_pathway_ranking": False,
        "physical_binding_claimed": False,
        "causal_effect_claimed": False,
        "release_ready": False,
        "production_deployed": False,
        "source_input_sha256": artifact_sha256(source_path),
    }
    _json_dump(lineage_path, lineage)

    def final_path(item: Path) -> str:
        return str(destination / item.relative_to(stage))

    artifact_declarations = {
        "tumor_lncrna_gene_coexpression": {
            "path": final_path(edge_root),
            "sha256": artifact_sha256(edge_root),
            "rows": total_edges,
        },
        "coexpression_availability": {
            "path": final_path(availability_root),
            "sha256": artifact_sha256(availability_root),
            "rows": total_availability,
        },
        "coexpression_cluster_membership": {
            "path": final_path(membership_root),
            "sha256": artifact_sha256(membership_root),
            "rows": total_membership,
        },
        "coexpression_cluster_summary": {
            "path": final_path(cluster_root),
            "sha256": artifact_sha256(cluster_root),
            "rows": total_clusters,
        },
        "coexpression_cancer_summary": {
            "path": final_path(cancer_summary_path),
            "sha256": artifact_sha256(cancer_summary_path),
            "rows": len(cancer_summary),
        },
        "source_inputs": {
            "path": final_path(source_path),
            "sha256": artifact_sha256(source_path),
        },
        "module_lineage": {
            "path": final_path(lineage_path),
            "sha256": artifact_sha256(lineage_path),
        },
    }
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "SUCCESS_FRESH_V32_BULK_COEXPRESSION_HASH_BOUND",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "computation_run_id": run_id,
        "result_role": lineage["result_role"],
        "fresh_statistical_calculation": True,
        "training_not_applicable": True,
        "all_output_rows_generated_current_run": True,
        "formal_authority": strict_formal_authority,
        "historical_derived_outputs_used": False,
        "historical_coexpression_edges_used": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "family_to_exact_broadcast": False,
        "changes_primary_exact_pathway_ranking": False,
        "physical_binding_claimed": False,
        "causal_effect_claimed": False,
        "release_ready": False,
        "production_deployed": False,
        "counts": {
            "cancers": len(cancers),
            "candidate_pairs": len(pairs),
            "edge_rows": total_edges,
            "lncrnas_with_edges": total_membership,
            "cluster_membership_rows": total_membership,
            "cluster_summary_rows": total_clusters,
            "available_pairs": total_available,
            "unavailable_cancers": int(
                sum(item["status"] == "UNAVAILABLE" for item in summaries)
            ),
        },
        "semantics": {
            "association_not_physical_binding": True,
            "association_not_causality": True,
            "sample_scope": "PRIMARY_TUMOUR_OR_OTHER_TUMOUR_SAMPLE_CODES_01_TO_09",
            "alignment_key": "patient_id",
            "duplicate_tumour_aliquots": "MEAN_WITHIN_PATIENT_FEATURE",
            "correlation": "SPEARMAN_ON_COVARIATE_RESIDUAL_RANKS",
            "multiple_testing": (
                "CONSERVATIVE_BH_RANKED_OVER_EFFECT_FILTERED_CANDIDATES_"
                "WITH_FULL_LNCRNA_X_GENE_DENOMINATOR"
            ),
            "unavailable_encoding": "NULL_WITH_TYPED_REASON",
            "cluster_basis": "SIGNED_SIGNIFICANT_EDGE_PROFILE",
        },
        "configuration": config,
        "authorities": {
            "materializer_code": {
                "path": str(code),
                "sha256": artifact_sha256(code),
            },
            "materializer_runner": {
                "path": str(runner),
                "sha256": artifact_sha256(runner),
            },
            "lncrna_expression_tree": {
                "path": str(lnc_root),
                "sha256": input_hashes["lncrna_expression_tree"],
            },
            "gene_expression_tree": {
                "path": str(gene_root),
                "sha256": input_hashes["gene_expression_tree"],
            },
            "covariates": {
                "path": str(cov_path),
                "sha256": input_hashes["covariates"],
            },
            "exact_candidates": {
                "path": str(candidate_path),
                "sha256": input_hashes["exact_candidates"],
            },
        },
        "artifacts": artifact_declarations,
    }
    _json_dump(binding_path, binding)
    binding_sha = artifact_sha256(binding_path)
    success = {
        "status": binding["status"],
        "analysis_version": ANALYSIS_VERSION,
        "computation_run_id": run_id,
        "binding": binding_path.name,
        "binding_sha256": binding_sha,
        "release_ready": False,
        "production_deployed": False,
    }
    _json_dump(success_path, success)
    _json_dump(
        status_path,
        {
            "status": "SUCCESS",
            "analysis_version": ANALYSIS_VERSION,
            "computation_run_id": run_id,
            "completed_cancers": completed,
            "counts": binding["counts"],
            "binding_sha256": binding_sha,
            "release_ready": False,
            "production_deployed": False,
        },
    )
    stage.replace(destination)
    return {
        "binding_path": str(destination / binding_path.name),
        "binding_sha256": binding_sha,
        "success_path": str(destination / success_path.name),
        "computation_run_id": run_id,
        **binding["counts"],
    }


__all__ = [
    "ANALYSIS_VERSION",
    "BINDING_FORMAT",
    "BulkCoexpressionReleaseError",
    "DEFAULT_COVARIATES",
    "EXPECTED_CANCERS",
    "FORMAL_CANDIDATE_PAIRS",
    "FORMAL_CANDIDATE_ROWS",
    "FORMAL_CANDIDATE_SHA256",
    "FORMAL_CANCERS",
    "FORMAL_COVARIATE_SHA256",
    "FORMAL_GENE_EXPRESSION_TREE_SHA256",
    "FORMAL_LNCRNA_EXPRESSION_TREE_SHA256",
    "FORMAL_LNCRNAS",
    "artifact_sha256",
    "materialize_bulk_coexpression_release",
]
