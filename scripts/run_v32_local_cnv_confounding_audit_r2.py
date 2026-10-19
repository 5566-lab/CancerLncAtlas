#!/usr/bin/env python3
"""Strict fold-local V3.2 local-CNV confounding and real-G0 audit.

One invocation processes one cancer across all five outer folds.  Association
A0_FULL uses the shared fold-local covariate design used by the formal
association path; A0_MATCHED/A1 use the same local-CNV-callable patients,
while A2 uses its own
pathway-CNV-complete cohort and retains an A1-on-A2 matched comparator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cc_hhgt.stats import (
    bh_fdr as conservative_bh,
    correlation_p_values,
    design_rank,
    rank_transform,
    residualize,
    standardize,
)
from cc_hhgt.v32 import bulk_coexpression_release as bulk
from cc_hhgt.v32.associations import _rank_and_standardize
from cc_hhgt.v32.cnv_only_training import TCGA_CANCERS
from cc_hhgt.v32.directional_cnv import DirectionalCNVStore
from cc_hhgt.v32.local_cnv_confounding import (
    COVARIATE_CONTRACT_VERSION,
    DEFAULT_COVARIATES,
    EFFECTIVE_VARIANCE_TOL,
    association_label,
    bh_fdr,
    direction,
    fold_local_pair_models,
    pathway_within_cancer_fdr,
)


FORMAT = "CC_HHGT_V3_2_LOCAL_CNV_CONFOUNDING_AUDIT_V2"
RUN_ID = "V32_LOCAL_CNV_CONFOUNDING_20260902_R2"
CANDIDATES_PER_CANCER = 100_000


@contextmanager
def cancer_run_lock(output: Path, cancer: str):
    """Serialize duplicate launchers for one cancer on the Linux server."""
    lock_path = output / "locks" / f"{cancer}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "posix":
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_covariate_authority(
    source_config_path: Path,
    sources: dict[str, Any],
    argument_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Bind the CLI covariates input to the frozen source-config authority.

    The association audit accepts a CLI path for convenience, but that path
    must be exactly the path declared by the immutable source manifest.  A
    manifest checksum, when present, is also checked before any data are
    loaded.  This prevents an accidentally substituted covariate table from
    silently changing the formal association design.
    """
    spec = sources.get("covariates")
    if spec is None:
        raise RuntimeError("Source config lacks the frozen covariates authority")
    if isinstance(spec, dict):
        declared_raw = spec.get("path")
        if not declared_raw:
            raise RuntimeError("Source config covariates authority lacks path")
        declared_sha = str(spec.get("sha256", "")).strip().lower()
    elif isinstance(spec, (str, os.PathLike)):
        # Older manifests may have recorded only a path.  Preserve support for
        # those manifests while still enforcing the path binding below.
        declared_raw = spec
        declared_sha = ""
    else:
        raise RuntimeError("Source config covariates authority must be a path record")

    declared_path = Path(declared_raw).resolve(strict=True)
    argument_resolved = Path(argument_path).resolve(strict=True)
    if declared_path != argument_resolved:
        raise RuntimeError(
            "Covariates CLI path does not match source-config authority: "
            f"{argument_resolved} != {declared_path}"
        )
    if declared_sha and not re.fullmatch(r"[0-9a-f]{64}", declared_sha):
        raise RuntimeError("Source config covariates sha256 is not a 64-hex digest")
    observed_sha = sha256(argument_resolved)
    if declared_sha and observed_sha != declared_sha:
        raise RuntimeError(
            "Covariates source-config sha256 mismatch: "
            f"declared={declared_sha} observed={observed_sha}"
        )
    binding = {
        "version": COVARIATE_CONTRACT_VERSION,
        "source_config": str(source_config_path.resolve()),
        "declared_path": str(declared_path),
        "argument_path": str(argument_resolved),
        "declared_sha256": declared_sha or None,
        "observed_sha256": observed_sha,
        "path_match": True,
        "sha256_verified": bool(declared_sha),
    }
    return argument_resolved, binding


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def canonical_patient(value: Any) -> str:
    text = str(value).strip().replace(".", "-").upper()
    return text[:12] if text.startswith("TCGA-") else text


def patient_matrix(
    path: Path, cancer: str, entity_column: str, value_column: str,
    *, capture=None, role="", fold=-1,
) -> tuple[pd.DataFrame, int]:
    columns = ["cancer_id", "sample_id", "patient_id", entity_column, value_column]
    from cc_hhgt.v32.fold_local_expression import read_analysis_frame, analysis_columns
    columns = analysis_columns(path, columns)
    reader = lambda p: read_analysis_frame(p, columns)
    frame = (reader(path) if capture is None else
             capture.read(path,reader,role=role,cancer=cancer,fold=fold,columns=columns))
    exclusions = frame.attrs.get("expression_exclusions", {})
    if not frame.empty and set(frame.cancer_id.astype(str)) != {cancer}:
        raise RuntimeError(f"cancer drift in {path}")
    frame["patient_id"] = frame.patient_id.map(canonical_patient)
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.loc[np.isfinite(frame[value_column])]
    duplicate = int(frame.duplicated(["patient_id", entity_column], keep=False).sum())
    matrix = frame.pivot_table(
        index="patient_id", columns=entity_column, values=value_column,
        aggfunc="mean", observed=True,
    )
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.sort_index().sort_index(axis=1).astype(np.float32)
    matrix.attrs["expression_exclusions"] = exclusions
    return matrix, duplicate


def split_patients(patient_map: pd.DataFrame, cancer: str, fold: int) -> list[str]:
    validation = (fold + 1) % 5
    local = patient_map.loc[patient_map.cancer_id.eq(cancer)]
    return sorted(local.loc[
        ~local.patient_fold_id.isin([fold, validation]), "patient_id"
    ].astype(str))


def fdr_and_labels(frame: pd.DataFrame, prefix: str) -> None:
    p_column = f"p_{prefix}"
    cancer_fdr = bh_fdr(frame[p_column].to_numpy(float))
    frame[f"fdr_cancer_global_{prefix}"] = cancer_fdr
    frame[f"fdr_pathway_within_cancer_{prefix}"] = pathway_within_cancer_fdr(
        frame, p_column
    )
    frame[f"fdr_official_{prefix}"] = cancer_fdr
    frame[f"label_{prefix}"] = [
        association_label(rho, fdr) for rho, fdr in zip(
            frame[f"rho_{prefix}"], cancer_fdr, strict=True
        )
    ]
    frame[f"direction_{prefix}"] = [direction(value) for value in frame[f"rho_{prefix}"]]


def a0_full(
    lnc: pd.DataFrame, pathway: pd.DataFrame, pairs: pd.DataFrame, patients: list[str],
    *, covariates: pd.DataFrame, cancer: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    common = sorted(set(patients) & set(lnc.index) & set(pathway.index))
    if len(common) < 5:
        return np.full(len(pairs), np.nan), np.full(len(pairs), np.nan), len(common), 0
    lnc_ids = sorted(pairs.lncrna_id.astype(str).unique())
    pathway_ids = sorted(pairs.pathway_id.astype(str).unique())
    x = lnc.reindex(index=common, columns=lnc_ids)
    y = pathway.reindex(index=common, columns=pathway_ids)
    x = x.fillna(x.mean(axis=0)).fillna(0.0)
    y = y.fillna(y.mean(axis=0)).fillna(0.0)
    design, _, _ = bulk._covariate_design(
        covariates, cancer=cancer, patients=common, columns=DEFAULT_COVARIATES,
    )
    xv = _rank_and_standardize(x, design)
    yv = _rank_and_standardize(y, design)
    # Normalization must not promote numerical projection noise to a signal.
    def estimable(frame):
        ranked = frame.rank(axis=0, method="average").to_numpy(float)
        residual = ranked - design @ (np.linalg.pinv(design) @ ranked)
        residual -= residual.mean(axis=0, keepdims=True)
        return np.sum(residual * residual, axis=0) > EFFECTIVE_VARIANCE_TOL
    x_estimable, y_estimable = estimable(x), estimable(y)
    li = pairs.lncrna_id.astype(str).map({v: i for i, v in enumerate(lnc_ids)}).to_numpy(int)
    pi = pairs.pathway_id.astype(str).map({v: i for i, v in enumerate(pathway_ids)}).to_numpy(int)
    rho = np.full(len(pairs), np.nan)
    for start in range(0, len(pairs), 4096):
        stop = min(start + 4096, len(pairs))
        rho[start:stop] = np.einsum(
            "ij,ij->j", xv[:, li[start:stop]], yv[:, pi[start:stop]]
        )
    rho = np.clip(rho, -0.999999, 0.999999)
    # Entirely absent/filtered features cannot become measured zero effects
    # through the historical missing-value imputation above.
    measured = pairs.lncrna_id.isin(lnc.columns) & pairs.pathway_id.isin(pathway.columns)
    rho[~measured.to_numpy()] = np.nan
    rho[~(x_estimable[li] & y_estimable[pi])] = np.nan
    residual_rank = design_rank(design)
    p = correlation_p_values(rho, len(common), residual_design_rank=residual_rank)
    return rho, p, len(common), residual_rank


def association_fold(
    cancer: str, fold: int, pairs: pd.DataFrame, lnc: pd.DataFrame,
    pathway: pd.DataFrame, patient_map: pd.DataFrame, cnv: DirectionalCNVStore,
    covariates: pd.DataFrame,
) -> pd.DataFrame:
    declared = split_patients(patient_map, cancer, fold)
    common = sorted(set(declared) & set(lnc.index) & set(pathway.index) & set(cnv.patient_ids))
    design, covariates_used, missing_covariate_patients = bulk._covariate_design(
        covariates, cancer=cancer, patients=common, columns=DEFAULT_COVARIATES,
    )
    out = pairs.copy().reset_index(drop=True)
    rho_full, p_full, n_full, a0_train_design_rank = a0_full(
        lnc, pathway, out, declared, covariates=covariates, cancer=cancer
    )
    # Preserve the current formal preparation contract for the downstream
    # CNV-only head: train associations are discovery targets and the adjacent
    # fold is validation-only replication.  No designated OOF/test labels are
    # computed or stored here.
    validation_patients = sorted(
        patient_map.loc[
            patient_map.cancer_id.eq(cancer)
            & patient_map.patient_fold_id.eq((fold + 1) % 5),
            "patient_id",
        ].astype(str)
    )
    validation_common = sorted(
        set(validation_patients) & set(lnc.index) & set(pathway.index) & set(cnv.patient_ids)
    )
    validation_design, validation_covariates_used, validation_missing_covariate_patients = (
        bulk._covariate_design(
            covariates, cancer=cancer, patients=validation_common,
            columns=DEFAULT_COVARIATES,
        )
    )
    rho_validation, p_validation, n_validation, a0_validation_design_rank = a0_full(
        lnc, pathway, out, validation_patients,
        covariates=covariates, cancer=cancer,
    )
    out["fold_id"] = int(fold)
    out["split_role"] = "train"
    out["n_A0_full"] = n_full
    out["rho_A0_full"] = rho_full
    out["p_A0_full"] = p_full
    out["n_A0_validation"] = n_validation
    out["rho_A0_validation"] = rho_validation
    out["p_A0_validation"] = p_validation

    lnc_ids = list(lnc.columns.astype(str))
    pathway_ids = list(pathway.columns.astype(str))
    lmap = {value: index for index, value in enumerate(lnc_ids)}
    pmap = {value: index for index, value in enumerate(pathway_ids)}
    cnv_rows = np.asarray([cnv.patient_index[value] for value in common], dtype=int)
    lnc_matrix = lnc.reindex(index=common, columns=lnc_ids).to_numpy(float)
    pathway_matrix = pathway.reindex(index=common, columns=pathway_ids).to_numpy(float)
    cli = np.asarray([cnv.lncrna_index.get(value, -1) for value in lnc_ids], dtype=int)
    cpi = np.asarray([cnv.pathway_index.get(value, -1) for value in pathway_ids], dtype=int)
    local_signed = np.full((len(common), len(lnc_ids)), np.nan, dtype=np.float32)
    local_callable = np.zeros((len(common), len(lnc_ids)), dtype=bool)
    local_amp = np.zeros_like(local_callable)
    local_del = np.zeros_like(local_callable)
    valid_lnc = cli >= 0
    if valid_lnc.any():
        local_signed[:, valid_lnc] = cnv.lncrna_signed[np.ix_(cnv_rows, cli[valid_lnc])]
        local_callable[:, valid_lnc] = cnv.lncrna_callable[np.ix_(cnv_rows, cli[valid_lnc])]
        local_amp[:, valid_lnc] = cnv.lncrna_amplification[np.ix_(cnv_rows, cli[valid_lnc])]
        local_del[:, valid_lnc] = cnv.lncrna_deletion[np.ix_(cnv_rows, cli[valid_lnc])]
    pathway_signed = np.full((len(common), len(pathway_ids)), np.nan, dtype=np.float32)
    pathway_callable = np.zeros((len(common), len(pathway_ids)), dtype=bool)
    valid_pathway = cpi >= 0
    if valid_pathway.any():
        pathway_signed[:, valid_pathway] = cnv.pathway_signed[np.ix_(cnv_rows, cpi[valid_pathway])]
        pathway_callable[:, valid_pathway] = cnv.pathway_callable[np.ix_(cnv_rows, cpi[valid_pathway])]

    fields = {
        "n_A0_matched": np.zeros(len(out), np.int32),
        "n_A1_local": np.zeros(len(out), np.int32),
        "n_A2_local_pathway": np.zeros(len(out), np.int32),
        "rho_A0_matched": np.full(len(out), np.nan), "p_A0_matched": np.full(len(out), np.nan),
        "rho_A1_local": np.full(len(out), np.nan), "p_A1_local": np.full(len(out), np.nan),
        "rho_A1_on_A2_matched": np.full(len(out), np.nan), "p_A1_on_A2_matched": np.full(len(out), np.nan),
        "rho_A2_local_pathway": np.full(len(out), np.nan), "p_A2_local_pathway": np.full(len(out), np.nan),
        "local_cnv_beta": np.full(len(out), np.nan), "local_cnv_beta_se": np.full(len(out), np.nan),
        "pathway_cnv_beta": np.full(len(out), np.nan),
        "residual_design_rank_A0_matched": np.zeros(len(out), np.int16),
        "residual_design_rank_A1_local": np.zeros(len(out), np.int16),
        "residual_design_rank_A1_on_A2_matched": np.zeros(len(out), np.int16),
        "residual_design_rank_A2_local_pathway": np.zeros(len(out), np.int16),
        "correlation_df_A0_matched": np.zeros(len(out), np.int32),
        "correlation_df_A1_local": np.zeros(len(out), np.int32),
        "correlation_df_A1_on_A2_matched": np.zeros(len(out), np.int32),
        "correlation_df_A2_local_pathway": np.zeros(len(out), np.int32),
        "local_cnv_callable_n": np.zeros(len(out), np.int32),
        "pathway_cnv_callable_n": np.zeros(len(out), np.int32),
        "local_amplification_n": np.zeros(len(out), np.int32),
        "local_deletion_n": np.zeros(len(out), np.int32),
        "association_available": np.zeros(len(out), dtype=bool),
    }
    for start in range(0, len(out), 1024):
        stop = min(start + 1024, len(out))
        li = out.lncrna_id.iloc[start:stop].astype(str).map(lmap).fillna(-1).to_numpy(int)
        pi = out.pathway_id.iloc[start:stop].astype(str).map(pmap).fillna(-1).to_numpy(int)
        valid = (li >= 0) & (pi >= 0)
        if not valid.any():
            continue
        destination = start + np.flatnonzero(valid)
        lix, pix = li[valid], pi[valid]
        result = fold_local_pair_models(
            lnc_matrix[:, lix], pathway_matrix[:, pix], local_signed[:, lix],
            pathway_signed[:, pix], design, local_callable=local_callable[:, lix],
            pathway_callable=pathway_callable[:, pix],
        )
        mapping = {
            "n_A0_matched": "n_a0_matched", "n_A1_local": "n_a1_local",
            "n_A2_local_pathway": "n_a2_local_pathway",
            "rho_A0_matched": "rho_a0_matched", "p_A0_matched": "p_a0_matched",
            "rho_A1_local": "rho_a1_local", "p_A1_local": "p_a1_local",
            "rho_A1_on_A2_matched": "rho_a1_on_a2_matched",
            "p_A1_on_A2_matched": "p_a1_on_a2_matched",
            "rho_A2_local_pathway": "rho_a2_local_pathway",
            "p_A2_local_pathway": "p_a2_local_pathway",
            "local_cnv_beta": "local_cnv_beta", "local_cnv_beta_se": "local_cnv_beta_se",
            "pathway_cnv_beta": "pathway_cnv_beta",
            "residual_design_rank_A0_matched": "residual_design_rank_a0",
            "residual_design_rank_A1_local": "residual_design_rank_a1",
            "residual_design_rank_A1_on_A2_matched": "residual_design_rank_a1_on_a2",
            "residual_design_rank_A2_local_pathway": "residual_design_rank_a2",
            "correlation_df_A0_matched": "correlation_df_a0",
            "correlation_df_A1_local": "correlation_df_a1",
            "correlation_df_A1_on_A2_matched": "correlation_df_a1_on_a2",
            "correlation_df_A2_local_pathway": "correlation_df_a2",
        }
        for target, source in mapping.items():
            fields[target][destination] = result[source]
        fields["association_available"][destination] = result["available"]
        fields["local_cnv_callable_n"][destination] = local_callable[:, lix].sum(axis=0)
        fields["pathway_cnv_callable_n"][destination] = pathway_callable[:, pix].sum(axis=0)
        fields["local_amplification_n"][destination] = local_amp[:, lix].sum(axis=0)
        fields["local_deletion_n"][destination] = local_del[:, lix].sum(axis=0)
    for name, value in fields.items():
        out[name] = value
    for prefix in ("A0_full", "A0_validation", "A0_matched", "A1_local", "A1_on_A2_matched", "A2_local_pathway"):
        fdr_and_labels(out, prefix)
    out["delta_rho_local"] = out.rho_A1_local - out.rho_A0_matched
    out["delta_abs_rho_local"] = out.rho_A1_local.abs() - out.rho_A0_matched.abs()
    out["attenuation_ratio_local"] = np.divide(
        out.rho_A1_local.abs(), out.rho_A0_matched.abs(),
        out=np.full(len(out), np.nan), where=out.rho_A0_matched.abs().to_numpy() > 1e-12,
    )
    out["delta_rho_pathway_cnv"] = out.rho_A2_local_pathway - out.rho_A1_on_A2_matched
    out["label_flip_local"] = out.label_A0_matched.ne(out.label_A1_local)
    out["direction_flip_local"] = out.direction_A0_matched.ne(out.direction_A1_local) & out.direction_A0_matched.ne("unavailable") & out.direction_A1_local.ne("unavailable")
    out["label_flip_pathway_cnv"] = out.label_A1_on_A2_matched.ne(out.label_A2_local_pathway)
    out["rank_A0_matched"] = out.rho_A0_matched.abs().rank(ascending=False, method="min", na_option="bottom")
    out["rank_A1_local"] = out.rho_A1_local.abs().rank(ascending=False, method="min", na_option="bottom")
    out["rank_change"] = out.rank_A1_local - out.rank_A0_matched
    out["cnv_sensitive"] = out.delta_abs_rho_local.abs().ge(0.05) | out.label_flip_local
    out["cnv_mediated_candidate"] = out.label_A0_matched.isin(["strong_positive", "weak_positive"]) & out.label_A1_local.eq("unlabeled")
    out["cnv_independent_candidate"] = out.label_A0_matched.eq(out.label_A1_local) & out.label_A1_local.isin(["strong_positive", "weak_positive"])
    out["availability"] = np.where(
        out.association_available, "available", "typed_unavailable"
    )
    out["unavailable_reason"] = np.where(
        out.local_cnv_callable_n.lt(12), "LOCAL_CNV_INSUFFICIENT_MATCHED_PATIENTS",
        np.where(~out.association_available, "ASSOCIATION_NOT_ESTIMABLE", ""),
    )
    out["run_id"] = RUN_ID
    out["local_cnv_continuous_definition"] = "signed_mean_segment_value"
    out["formal_C0"] = "shared_covariate_current_v32_formal_association"
    out.attrs["covariates_used"] = list(covariates_used)
    out.attrs["missing_covariate_patients"] = int(missing_covariate_patients)
    out.attrs["base_design_rank"] = int(np.linalg.matrix_rank(design)) if len(design) else 0
    out.attrs["a0_train_design_rank"] = int(a0_train_design_rank)
    out.attrs["a0_validation_design_rank"] = int(a0_validation_design_rank)
    out.attrs["validation_covariates_used"] = list(validation_covariates_used)
    out.attrs["validation_missing_covariate_patients"] = int(
        validation_missing_covariate_patients
    )
    out.attrs["validation_base_design_rank"] = (
        int(np.linalg.matrix_rank(validation_design)) if len(validation_design) else 0
    )
    from cc_hhgt.v32.fold_local_expression import mark_excluded
    return mark_excluded(mark_excluded(out, pathway, entity="pathway_id"), lnc)


def edge_candidate_frame(
    x: np.ndarray, genes: np.ndarray, lnc_ids: np.ndarray, gene_ids: np.ndarray,
    *, n: int, rank: int | np.ndarray, model: str,
) -> tuple[pd.DataFrame, int]:
    total = int(x.shape[1] * genes.shape[1])
    rows = []
    denominator = max(n - 1, 1)
    rank_values = np.asarray(rank, dtype=int).reshape(-1)
    if rank_values.size not in {1, x.shape[1]}:
        raise ValueError("edge residual rank is not aligned to lncRNA columns")
    for start in range(0, x.shape[1], 128):
        stop = min(start + 128, x.shape[1])
        rho = (x[:, start:stop].T @ genes) / denominator
        block_rank = rank_values[0] if rank_values.size == 1 else rank_values[start:stop]
        if np.ndim(block_rank) == 0:
            p = correlation_p_values(rho, n, residual_design_rank=int(block_rank))
            block_rank_matrix = np.full(rho.shape, int(block_rank), dtype=int)
        else:
            block_rank_array = np.asarray(block_rank, dtype=int)
            p = np.full(rho.shape, np.nan, dtype=float)
            for rank_value in np.unique(block_rank_array):
                selected = block_rank_array == int(rank_value)
                p[selected, :] = correlation_p_values(
                    rho[selected, :], n, residual_design_rank=int(rank_value)
                )
            block_rank_matrix = np.broadcast_to(
                block_rank_array[:, None], rho.shape
            )
        li, gi = np.where((np.abs(rho) >= 0.20) & (p <= 0.05))
        if len(li):
            rows.append(pd.DataFrame({
                "lncrna_id": lnc_ids[start:stop][li], "gene_id": gene_ids[gi],
                f"rho_{model}": rho[li, gi], f"p_{model}": p[li, gi],
                f"n_{model}": n,
                f"residual_design_rank_{model}": block_rank_matrix[li, gi],
                f"correlation_df_{model}": n - block_rank_matrix[li, gi] - 1,
            }))
    columns = [
        "lncrna_id", "gene_id", f"rho_{model}", f"p_{model}", f"n_{model}",
        f"residual_design_rank_{model}", f"correlation_df_{model}",
    ]
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)), total


def select_edges(candidates: pd.DataFrame, total_tests: int, model: str) -> pd.DataFrame:
    if candidates.empty:
        columns = [
            "lncrna_id", "gene_id", f"rho_{model}", f"p_{model}",
            f"n_{model}", f"residual_design_rank_{model}", f"fdr_{model}",
            f"correlation_df_{model}", f"direction_{model}",
        ]
        return pd.DataFrame(columns=columns)
    frame = candidates.copy()
    frame[f"fdr_{model}"] = conservative_bh(frame[f"p_{model}"], total_tests=total_tests)
    frame = frame.loc[frame[f"fdr_{model}"].le(0.05)].copy()
    frame[f"direction_{model}"] = np.where(frame[f"rho_{model}"].ge(0), "positive", "negative")
    frame["abs_rho"] = frame[f"rho_{model}"].abs()
    frame = frame.sort_values(
        ["lncrna_id", f"direction_{model}", "abs_rho", f"fdr_{model}", "gene_id"],
        ascending=[True, True, False, True, True], kind="stable",
    ).groupby(["lncrna_id", f"direction_{model}"], observed=True, sort=False).head(75)
    return frame.drop(columns="abs_rho")


def g0_fold(
    cancer: str, fold: int, lnc: pd.DataFrame, gene: pd.DataFrame,
    patient_map: pd.DataFrame, covariates: pd.DataFrame, cnv: DirectionalCNVStore,
    allowed_genes: set[str],
) -> pd.DataFrame:
    declared = split_patients(patient_map, cancer, fold)
    common = sorted(set(declared) & set(lnc.index) & set(gene.index) & set(cnv.patient_ids))
    lnc_ids = sorted(set(lnc.columns.astype(str)) & set(cnv.lncrna_ids))
    gene_ids = sorted(set(gene.columns.astype(str)) & allowed_genes)
    if not common or not lnc_ids or not gene_ids:
        return merge_edge_models(cancer,fold,select_edges(pd.DataFrame(),0,"E0"),select_edges(pd.DataFrame(),0,"E1"))
    lnc_local = lnc.reindex(index=common, columns=lnc_ids)
    gene_local = gene.reindex(index=common, columns=gene_ids)
    rows = np.asarray([cnv.patient_index[p] for p in common], dtype=int)
    positions = np.asarray([cnv.lncrna_index[v] for v in lnc_ids], dtype=int)
    callable_matrix = np.asarray(cnv.lncrna_callable[np.ix_(rows, positions)], dtype=bool)
    signed_matrix = np.asarray(cnv.lncrna_signed[np.ix_(rows, positions)], dtype=float)
    packed = np.packbits(callable_matrix, axis=0).T
    _, inverse = np.unique(packed, axis=0, return_inverse=True)
    e0_parts, e1_parts = [], []
    total_tests = 0
    for group in range(int(inverse.max()) + 1 if len(inverse) else 0):
        columns = np.flatnonzero(inverse == group)
        mask = callable_matrix[:, columns[0]]
        if int(mask.sum()) < 40:
            continue
        patients = np.asarray(common)[mask].tolist()
        x_frame, _ = bulk._variance_filter(lnc_local.iloc[mask, columns], 0.01)
        g_frame, _ = bulk._variance_filter(gene_local.iloc[mask], 0.01)
        if x_frame.empty or g_frame.empty:
            continue
        kept = np.asarray([lnc_ids.index(value) for value in x_frame.columns], dtype=int)
        z = signed_matrix[np.ix_(mask, kept)]
        design, _, _ = bulk._covariate_design(
            covariates, cancer=cancer, patients=patients, columns=DEFAULT_COVARIATES,
        )
        rank = design_rank(design)
        x_filled = x_frame.fillna(x_frame.median()).fillna(0.0)
        g_filled = g_frame.fillna(g_frame.median()).fillna(0.0)
        xr = rank_transform(x_filled.to_numpy(float))
        gr = rank_transform(g_filled.to_numpy(float))
        zr = rank_transform(z)
        x0_raw = residualize(xr, design)
        g0 = standardize(residualize(gr, design))
        z0 = residualize(zr, design)
        denominator = np.sum(z0 * z0, axis=0)
        beta = np.divide(
            np.sum(z0 * x0_raw, axis=0), denominator,
            out=np.zeros(len(denominator)), where=denominator > EFFECTIVE_VARIANCE_TOL,
        )
        x0 = standardize(x0_raw)
        x1 = standardize(x0_raw - z0 * beta)
        e0, tests = edge_candidate_frame(
            x0, g0, x_frame.columns.to_numpy(str), g_frame.columns.to_numpy(str),
            n=len(patients), rank=rank, model="E0",
        )
        effective = denominator > EFFECTIVE_VARIANCE_TOL
        e1, _ = edge_candidate_frame(
            x1, g0, x_frame.columns.to_numpy(str), g_frame.columns.to_numpy(str),
            n=len(patients), rank=rank + effective.astype(np.int16), model="E1",
        )
        e0_parts.append(e0); e1_parts.append(e1); total_tests += tests
    e0 = select_edges(pd.concat(e0_parts, ignore_index=True) if e0_parts else pd.DataFrame(), total_tests, "E0")
    e1 = select_edges(pd.concat(e1_parts, ignore_index=True) if e1_parts else pd.DataFrame(), total_tests, "E1")
    return merge_edge_models(cancer,fold,e0,e1)


def merge_edge_models(cancer: str, fold: int, e0: pd.DataFrame, e1: pd.DataFrame) -> pd.DataFrame:
    """Keep the complete E0/E1 schema even when no expression features qualify."""
    keys = ["lncrna_id", "gene_id"]
    merged = e0.merge(e1, on=keys, how="outer", validate="one_to_one")
    merged.insert(0, "fold_id", int(fold)); merged.insert(0, "cancer_id", cancer)
    for column in ("rho_E0", "rho_E1"):
        if column not in merged:
            merged[column] = np.nan
        merged[column] = pd.to_numeric(merged[column], errors="coerce").astype(float)
    merged["present_E0"] = np.isfinite(merged["rho_E0"].to_numpy(float))
    merged["present_E1"] = np.isfinite(merged["rho_E1"].to_numpy(float))
    merged["edge_removed"] = merged.present_E0 & ~merged.present_E1
    merged["edge_added"] = ~merged.present_E0 & merged.present_E1
    merged["direction_flip"] = merged.present_E0 & merged.present_E1 & (np.sign(merged.rho_E0) != np.sign(merged.rho_E1))
    for model in ("E0", "E1"):
        merged[f"rank_{model}"] = merged.groupby("lncrna_id", observed=True)[f"rho_{model}"].transform(lambda value: value.abs().rank(ascending=False, method="min", na_option="bottom"))
    merged["rank_change"] = merged.rank_E1 - merged.rank_E0
    merged["local_cnv_continuous_definition"] = "signed_mean_segment_value"
    merged["edge_universe"] = "real_lncrna_gene_G0"
    merged["run_id"] = RUN_ID
    return merged


def run_cancer(args: argparse.Namespace) -> dict[str, Any]:
    from cc_hhgt.v32.local_cnv_sources import TransformationCapture,preflight,validate,OUTPUT_ROLES
    cancer = args.cancer.upper()
    if cancer not in TCGA_CANCERS:
        raise RuntimeError(f"unknown cancer: {cancer}")
    output = Path(args.output_root).resolve()
    preflight(output,cancer,"association")
    capture=TransformationCapture()
    source_config_path = Path(args.source_config).resolve(strict=True)
    config = capture.json(source_config_path,role="source_config")
    from cc_hhgt.v32.local_cnv_cohort import from_config, require_receipt
    cohort = from_config(config, source_config_path.parent, args.candidates, args.patient_map)
    if cohort is not None and cancer not in cohort["cancers"]:
        raise ValueError("Requested cancer is outside the declared local-CNV cohort")
    sources = config.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("Source config lacks a sources mapping")
    covariates_path, covariate_binding = bind_covariate_authority(
        source_config_path, sources, args.covariates
    )
    receipt_path = output / "receipts" / f"{cancer}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        require_receipt(receipt, cohort)
        if "consumed_sources" not in receipt:raise RuntimeError("Historical local-CNV output lacks consumed-source binding; use a new output")
        validate(receipt["consumed_sources"],receipt_path.parent,expected_inputs={
            "source_config":source_config_path,"covariates":covariates_path,
            "candidates":args.candidates,"patient_map":args.patient_map,
            "signed_membership":Path(args.graph_root)/"static/signed_membership.parquet",
            "protein_gene_encoding":Path(args.graph_root)/"static/protein_gene_encoding.parquet"},
            compact_root=Path(args.directional_root)/f"cancer={cancer}")
        if receipt.get("run_id")!=RUN_ID:raise ValueError("Local-CNV reused run identity differs")
        if receipt.get("status") != "PASS" or receipt.get("cancer_id") != cancer:
            raise RuntimeError(f"existing cancer receipt is not reusable: {receipt_path}")
        if receipt.get("covariate_contract_version") != COVARIATE_CONTRACT_VERSION:
            raise RuntimeError(
                f"existing cancer receipt lacks {COVARIATE_CONTRACT_VERSION}: {receipt_path}"
            )
        if receipt.get("covariate_contract") != covariate_binding:
            raise RuntimeError(
                f"existing cancer receipt covariate authority drift: {receipt_path}"
            )
        for collection in ("pair_outputs", "g0_outputs"):
            records = receipt.get(collection, [])
            if len(records) != 5:
                raise RuntimeError(f"existing cancer receipt fold closure drift: {cancer}/{collection}")
            for record in records:
                path = Path(record["path"])
                if not path.is_file() or sha256(path) != record["sha256"]:
                    raise RuntimeError(f"existing immutable output SHA drift: {cancer}/{collection}/{record.get('fold')}")
        receipt = dict(receipt)
        receipt["reused_after_sha_verification"] = True
        return receipt
    columns=["cancer_id","lncrna_id","pathway_id"]
    candidates = capture.read(args.candidates,lambda p:pd.read_parquet(p,columns=columns,
        filters=[("cancer_id","==",cancer)]),role="candidates",cancer=cancer,columns=columns).astype(str)
    expected_rows = cohort["candidate_rows_by_cancer"][cancer] if cohort is not None else CANDIDATES_PER_CANCER
    if len(candidates) != expected_rows or candidates.duplicated().any():
        raise RuntimeError(f"candidate authority drift for {cancer}: {len(candidates)}")
    patient_map = capture.read(args.patient_map,lambda p:pd.read_csv(p,sep="\t"),role="patient_map")
    patient_map["cancer_id"] = patient_map.cancer_id.astype(str)
    patient_map["patient_id"] = patient_map.patient_id.map(canonical_patient)
    patient_map["patient_fold_id"] = pd.to_numeric(patient_map.patient_fold_id).astype(int)
    covariates = capture.read(covariates_path,pd.read_parquet,role="covariates")
    covariates["patient_id"] = covariates.patient_id.map(canonical_patient)
    covariates["cancer_id"] = covariates.cancer_id.astype(str).str.upper()
    cnv = DirectionalCNVStore(Path(args.directional_root) / f"cancer={cancer}",source_capture=capture,cancer=cancer)
    signed_membership = capture.read(Path(args.graph_root)/"static/signed_membership.parquet",
        lambda p:pd.read_parquet(p,columns=["gene_id"]),role="signed_membership",columns=["gene_id"])
    encoding = capture.read(Path(args.graph_root)/"static/protein_gene_encoding.parquet",
        lambda p:pd.read_parquet(p,columns=["gene_id"]),role="protein_gene_encoding",columns=["gene_id"])
    allowed_genes = set(signed_membership.gene_id.astype(str)) | set(encoding.gene_id.astype(str))
    pair_records, edge_records = [], []
    from cc_hhgt.v32.fold_local_expression import expression_path
    from cc_hhgt.v32.fold_local_activity import activity_path
    expression_cache, duplicate_by_fold = {}, {}
    for fold in range(5):
        matrices = {}
        for key, entity, role in (("lnc_expression_root","lncrna_id","lncrna_expression"),
                                  ("gene_expression_root","gene_id","gene_expression"),
                                  ("pathway_activity_root","pathway_id","pathway_activity")):
            path = (activity_path(sources,cancer,fold,args.patient_map) if entity=="pathway_id" else
                    expression_path(sources,key,cancer,fold,args.patient_map))
            column = "activity_score" if entity=="pathway_id" else "logcpm"
            scope = fold if sources[key].get("layout") == "fold_local" else -1
            if scope >= 0:
                matrices[key] = patient_matrix(path,cancer,entity,column,capture=capture,role=role,fold=scope)
                continue
            if path not in expression_cache:
                expression_cache[path] = patient_matrix(path,cancer,entity,column,capture=capture,role=role,fold=scope)
            matrices[key] = expression_cache[path]
        lnc, lnc_duplicates = matrices["lnc_expression_root"]
        gene, gene_duplicates = matrices["gene_expression_root"]
        pathway, pathway_duplicates = matrices["pathway_activity_root"]
        duplicate_by_fold[str(fold)] = {"lncrna":lnc_duplicates,"gene":gene_duplicates,"pathway":pathway_duplicates}
        pair = association_fold(
            cancer, fold, candidates, lnc, pathway, patient_map, cnv, covariates
        )
        pair_metadata = {
            "covariates_used": pair.attrs.get("covariates_used", []),
            "missing_covariate_patients": int(pair.attrs.get("missing_covariate_patients", 0)),
            "base_design_rank": int(pair.attrs.get("base_design_rank", 0)),
            "a0_train_design_rank": int(pair.attrs.get("a0_train_design_rank", 0)),
            "a0_validation_design_rank": int(
                pair.attrs.get("a0_validation_design_rank", 0)
            ),
            "validation_covariates_used": pair.attrs.get("validation_covariates_used", []),
            "validation_missing_covariate_patients": int(
                pair.attrs.get("validation_missing_covariate_patients", 0)
            ),
            "validation_base_design_rank": int(
                pair.attrs.get("validation_base_design_rank", 0)
            ),
        }
        pair_path = output / "pair_level" / f"cancer={cancer}" / f"fold={fold}" / "part.parquet"
        atomic_parquet(pair_path, pair)
        edge = g0_fold(cancer, fold, lnc, gene, patient_map, covariates, cnv, allowed_genes)
        edge_path = output / "g0_edge_stability" / f"cancer={cancer}" / f"fold={fold}" / "part.parquet"
        atomic_parquet(edge_path, edge)
        train_count = len(split_patients(patient_map, cancer, fold))
        edge_status = (
            "TYPED_UNAVAILABLE_INSUFFICIENT_TRAIN_PATIENTS"
            if train_count < 40 else
            ("PASS_EMPTY_NO_SIGNIFICANT_EDGES" if edge.empty else "PASS_REAL_G0_EDGE_COMPARISON")
        )
        pair_records.append({"fold": fold, "path": str(pair_path), "rows": len(pair), "sha256": sha256(pair_path),
            "output_kind":"pair","consumed_roles":sorted(OUTPUT_ROLES["pair"]),**pair_metadata})
        edge_records.append({
            "fold": fold, "path": str(edge_path), "rows": len(edge),
            "sha256": sha256(edge_path), "status": edge_status,
            "declared_train_patients": train_count,
            "output_kind":"graph","consumed_roles":sorted(OUTPUT_ROLES["graph"]),
        })
    receipt = {
        "format": FORMAT, "status": "PASS", "cancer_id": cancer,"run_id":RUN_ID,
        **({"cohort": cohort} if cohort is not None else {}),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "covariate_contract_version": COVARIATE_CONTRACT_VERSION,
        "covariate_contract": covariate_binding,
        "covariates": {
            "path": str(covariates_path),
            "sha256": covariate_binding["observed_sha256"],
            "source_config_path": str(source_config_path),
            "source_config_declared_path": covariate_binding["declared_path"],
            "source_config_declared_sha256": covariate_binding["declared_sha256"],
            "declared": list(DEFAULT_COVARIATES),
        },
        "pair_outputs": pair_records, "g0_outputs": edge_records,
        "duplicate_patient_feature_rows_averaged": {
            "expression_by_fold": duplicate_by_fold,
        },
        "association_C0": "shared_covariate_current_v32_formal_association",
        "association_A1_A2_covariates": list(DEFAULT_COVARIATES),
        "formal_validation_labels_materialized": True,
        "designated_oof_labels_materialized": False,
        "g0_C0": list(DEFAULT_COVARIATES),
        "signed_continuous_local_cnv": True, "amplification_deletion_separate": True,
        "pathway_callability_not_required_for_A0_MATCHED_or_A1": True,
    }
    receipt["consumed_sources"]=capture.publish(output/"source_reads"/(cancer+".json"),
        producer="association",cancer=cancer,outputs=pair_records+edge_records,runner=Path(__file__))
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> None:
    from cc_hhgt.v32.cnv_input_sources import require_host
    require_host()
    global RUN_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--patient-map", required=True)
    parser.add_argument("--covariates", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--directional-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--run-id", default=RUN_ID,
        help="Immutable run identifier written into every pair/edge row.",
    )
    args = parser.parse_args()
    RUN_ID = str(args.run_id)
    output = Path(args.output_root).resolve()
    with cancer_run_lock(output, str(args.cancer).upper()):
        result = run_cancer(args)
    print(json.dumps({"status": result["status"], "cancer": args.cancer}, sort_keys=True))


if __name__ == "__main__":
    main()
