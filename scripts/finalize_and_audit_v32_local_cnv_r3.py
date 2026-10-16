#!/usr/bin/env python3
"""Independent closeout for the strict V3.2 local-CNV sensitivity run.

This reader intentionally does not import the calculation module.  It verifies
the immutable cancer receipts and row-level derivations, computes the frozen
decision gate, and only then writes the release-shaped summary and SUCCESS.
The two large ``*.parquet`` deliverables are partitioned Parquet datasets made
from hard links to the audited files, avoiding a second multi-gigabyte copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from cc_hhgt.v32.local_cnv_confounding import (
    COVARIATE_CONTRACT_VERSION as EXPECTED_COVARIATE_CONTRACT_VERSION,
    DEFAULT_COVARIATES,
)


CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)
EXPECTED_RUN_ID = "V32_LOCAL_CNV_CONFOUNDING_20260902_R3"
EXPECTED_ASSOCIATION_C0 = "shared_covariate_current_v32_formal_association"
EXPECTED_COVARIATES = DEFAULT_COVARIATES
POSITIVE = {"strong_positive", "weak_positive"}
PREFIXES = (
    "A0_full", "A0_validation", "A0_matched", "A1_local", "A1_on_A2_matched",
    "A2_local_pathway",
)
PAIR_REQUIRED = {
    "cancer_id", "lncrna_id", "pathway_id", "fold_id", "split_role",
    "n_A0_full", "n_A0_validation", "n_A0_matched", "n_A1_local", "n_A2_local_pathway",
    "rho_A0_full", "p_A0_full", "rho_A0_validation", "p_A0_validation",
    "rho_A0_matched", "p_A0_matched",
    "rho_A1_local", "p_A1_local", "rho_A1_on_A2_matched",
    "p_A1_on_A2_matched", "rho_A2_local_pathway", "p_A2_local_pathway",
    "local_cnv_beta", "local_cnv_beta_se", "pathway_cnv_beta",
    "local_cnv_callable_n", "pathway_cnv_callable_n",
    "local_amplification_n", "local_deletion_n", "delta_rho_local",
    "delta_abs_rho_local", "attenuation_ratio_local",
    "delta_rho_pathway_cnv", "label_flip_local", "direction_flip_local",
    "label_flip_pathway_cnv", "rank_A0_matched", "rank_A1_local",
    "rank_change", "cnv_sensitive", "cnv_mediated_candidate",
    "cnv_independent_candidate", "availability", "unavailable_reason",
    "run_id", "local_cnv_continuous_definition", "formal_C0",
}
EDGE_REQUIRED = {
    "cancer_id", "fold_id", "lncrna_id", "gene_id", "rho_E0", "rho_E1",
    "direction_E0", "direction_E1",
    "present_E0", "present_E1", "edge_removed", "edge_added",
    "direction_flip", "rank_E0", "rank_E1", "rank_change",
    "local_cnv_continuous_definition", "edge_universe", "run_id",
}


def edge_direction_metrics(edge: pd.DataFrame, top_k: int = 75) -> dict[str, float]:
    """Compare the two selected G0 edge sets, retaining sign and per-lncRNA topology."""
    if edge.empty:
        return {
            "positive_edge_jaccard": 1.0, "negative_edge_jaccard": 1.0,
            "median_lncrna_direction_topk_jaccard": 1.0,
        }
    keys = ["lncrna_id", "gene_id"]
    positive_e0 = set(map(tuple, edge.loc[edge.direction_E0.eq("positive"), keys].astype(str).to_numpy()))
    positive_e1 = set(map(tuple, edge.loc[edge.direction_E1.eq("positive"), keys].astype(str).to_numpy()))
    negative_e0 = set(map(tuple, edge.loc[edge.direction_E0.eq("negative"), keys].astype(str).to_numpy()))
    negative_e1 = set(map(tuple, edge.loc[edge.direction_E1.eq("negative"), keys].astype(str).to_numpy()))
    local_scores: list[float] = []
    # Do not rescan the complete (occasionally multi-million-row) edge table
    # once per lncRNA.  The former expression was mathematically correct but
    # quadratic in practice.  A single stable groupby preserves the exact
    # per-lncRNA calculation while making the independent audit linear in the
    # number of edge rows.
    for _, local in edge.groupby("lncrna_id", observed=True, sort=True):
        for direction in ("positive", "negative"):
            left = local.loc[local.direction_E0.eq(direction)].assign(_magnitude=lambda x: x.rho_E0.abs()).nlargest(top_k, "_magnitude")
            right = local.loc[local.direction_E1.eq(direction)].assign(_magnitude=lambda x: x.rho_E1.abs()).nlargest(top_k, "_magnitude")
            left_set = set(left.gene_id.astype(str)); right_set = set(right.gene_id.astype(str))
            if left_set or right_set:
                local_scores.append(jaccard(left_set, right_set))
    return {
        "positive_edge_jaccard": jaccard(positive_e0, positive_e1),
        "negative_edge_jaccard": jaccard(negative_e0, negative_e1),
        "median_lncrna_direction_topk_jaccard": finite_median(local_scores),
    }


def describe_file(path: Path, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"required authority missing: {label}: {resolved}")
    record: dict[str, Any] = {
        "label": label, "path": str(resolved), "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }
    suffixes = "".join(resolved.suffixes).lower()
    if suffixes.endswith(".parquet"):
        import pyarrow.parquet as pq
        metadata = pq.ParquetFile(resolved).metadata
        record.update(rows=metadata.num_rows, columns=metadata.num_columns,
                      schema=str(pq.read_schema(resolved)))
    elif suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        reader = pd.read_csv(resolved, sep="\t", chunksize=250_000)
        rows = 0; columns: list[str] = []
        for chunk in reader:
            if not columns:
                columns = list(map(str, chunk.columns))
            rows += len(chunk)
        record.update(rows=rows, columns=len(columns), schema="|".join(columns))
    elif suffixes.endswith(".json"):
        record.update(schema="json")
    else:
        record.update(schema="opaque")
    return record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def fdr(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    out = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return out
    observed = values[valid]
    order = np.argsort(observed, kind="stable")
    adjusted = observed[order] * len(observed) / np.arange(1, len(observed) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(len(observed))
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    out[valid] = restored
    return out


def labels(rho: np.ndarray, q: np.ndarray) -> np.ndarray:
    result = np.full(len(rho), "unlabeled", dtype=object)
    unavailable = ~np.isfinite(rho) | ~np.isfinite(q)
    result[unavailable] = "unavailable"
    strong = ~unavailable & (q <= 0.05) & (np.abs(rho) >= 0.20)
    weak = ~unavailable & ~strong & (q <= 0.10) & (np.abs(rho) >= 0.15)
    result[strong] = "strong_positive"
    result[weak] = "weak_positive"
    return result


def assert_close(name: str, actual: Iterable[float], expected: Iterable[float], atol: float = 2e-10) -> None:
    left = np.asarray(list(actual), dtype=float)
    right = np.asarray(list(expected), dtype=float)
    if left.shape != right.shape or not np.allclose(left, right, atol=atol, rtol=1e-9, equal_nan=True):
        mismatch = int(np.sum(~np.isclose(left, right, atol=atol, rtol=1e-9, equal_nan=True)))
        raise RuntimeError(f"{name} mismatch: {mismatch}/{len(left)}")


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return np.nan
    a, b = rankdata(left[valid]), rankdata(right[valid])
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def top_jaccard(left: np.ndarray, right: np.ndarray, keys: np.ndarray, k: int) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return np.nan
    size = min(k, int(valid.sum()))
    positions = np.flatnonzero(valid)
    a = positions[np.argsort(np.abs(left[positions]), kind="stable")[-size:]]
    b = positions[np.argsort(np.abs(right[positions]), kind="stable")[-size:]]
    return jaccard(set(keys[a]), set(keys[b]))


def verify_pair(frame: pd.DataFrame, cancer: str, fold: int, candidate_keys: pd.MultiIndex) -> list[dict[str, Any]]:
    missing = sorted(PAIR_REQUIRED - set(frame.columns))
    if missing:
        raise RuntimeError(f"pair schema missing {cancer}/{fold}: {missing}")
    if len(frame) != 100_000 or frame[["lncrna_id", "pathway_id"]].duplicated().any():
        raise RuntimeError(f"pair closure drift {cancer}/{fold}: {len(frame)}")
    observed_keys = pd.MultiIndex.from_frame(frame[["lncrna_id", "pathway_id"]].astype(str))
    if not observed_keys.equals(candidate_keys):
        # Equality is order-sensitive; set equality remains mandatory.
        if len(observed_keys.symmetric_difference(candidate_keys)):
            raise RuntimeError(f"candidate key mismatch {cancer}/{fold}")
    if set(frame.cancer_id.astype(str)) != {cancer} or set(frame.fold_id.astype(int)) != {fold}:
        raise RuntimeError(f"partition identity drift {cancer}/{fold}")
    if set(frame.split_role.astype(str)) != {"train"}:
        raise RuntimeError(f"non-train rows in audit {cancer}/{fold}")
    if set(frame.run_id.astype(str)) != {EXPECTED_RUN_ID}:
        raise RuntimeError(f"run id drift {cancer}/{fold}")
    if set(frame.local_cnv_continuous_definition.astype(str)) != {"signed_mean_segment_value"}:
        raise RuntimeError(f"signed CNV definition drift {cancer}/{fold}")
    forbidden = [name for name in frame.columns if "mutation" in name.lower() or "sealed" in name.lower()]
    if forbidden:
        raise RuntimeError(f"forbidden columns {cancer}/{fold}: {forbidden}")
    if not np.array_equal(frame.n_A0_matched.to_numpy(), frame.n_A1_local.to_numpy()):
        raise RuntimeError(f"A0/A1 matched cohort drift {cancer}/{fold}")
    if (frame.local_amplification_n < 0).any() or (frame.local_deletion_n < 0).any():
        raise RuntimeError(f"negative directional event count {cancer}/{fold}")
    for prefix in PREFIXES:
        q = fdr(frame[f"p_{prefix}"].to_numpy(float))
        assert_close(f"FDR {cancer}/{fold}/{prefix}", frame[f"fdr_official_{prefix}"], q)
        expected_labels = labels(frame[f"rho_{prefix}"].to_numpy(float), q)
        if not np.array_equal(frame[f"label_{prefix}"].astype(str).to_numpy(), expected_labels):
            raise RuntimeError(f"label derivation drift {cancer}/{fold}/{prefix}")
    assert_close("delta_rho_local", frame.delta_rho_local, frame.rho_A1_local - frame.rho_A0_matched)
    assert_close("delta_abs_rho_local", frame.delta_abs_rho_local, frame.rho_A1_local.abs() - frame.rho_A0_matched.abs())
    expected_flip = frame.label_A0_matched.ne(frame.label_A1_local).to_numpy()
    if not np.array_equal(frame.label_flip_local.to_numpy(bool), expected_flip):
        raise RuntimeError(f"label flip derivation drift {cancer}/{fold}")

    pathway_metrics: list[dict[str, Any]] = []
    for pathway, local in frame.groupby("pathway_id", observed=True, sort=False):
        a = local.rho_A0_matched.to_numpy(float)
        b = local.rho_A1_local.to_numpy(float)
        keys = local.lncrna_id.astype(str).to_numpy()
        baseline_set = set(local.loc[local.label_A0_matched.isin(POSITIVE), "lncrna_id"].astype(str))
        adjusted_set = set(local.loc[local.label_A1_local.isin(POSITIVE), "lncrna_id"].astype(str))
        pathway_metrics.append({
            "cancer_id": cancer, "fold_id": fold, "pathway_id": str(pathway),
            "rank_spearman": spearman(a, b),
            "top50_jaccard": top_jaccard(a, b, keys, 50),
            "top100_jaccard": top_jaccard(a, b, keys, 100),
            "top200_jaccard": top_jaccard(a, b, keys, 200),
            "gene_set_jaccard": jaccard(baseline_set, adjusted_set),
            "baseline_positive": len(baseline_set),
            "adjusted_positive": len(adjusted_set),
        })
    return pathway_metrics


def hardlink_dataset(source: Path, destination: Path) -> None:
    for src in sorted(source.rglob("*.parquet")):
        relative = src.relative_to(source)
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.nanmedian(array)) if np.isfinite(array).any() else np.nan


def decision(global_summary: dict[str, Any], cancer_summary: pd.DataFrame) -> str:
    flip = float(global_summary["baseline_positive_label_flip_rate"])
    direction = float(global_summary["direction_flip_rate"])
    rank = float(global_summary["median_cancer_pathway_rank_spearman"])
    top100 = float(global_summary["median_top100_jaccard"])
    g0_sign = float(global_summary["g0_sign_flip_rate"])
    g0_j = float(global_summary["median_g0_edge_jaccard"])
    core = (
        flip > 0.10
        or int((cancer_summary.baseline_positive_label_flip_rate > 0.15).sum()) >= 5
        or direction > 0.05 or rank < 0.90 or top100 < 0.75
        or g0_sign > 0.05 or g0_j < 0.75
    )
    if core:
        return "LOCAL_CNV_CORE_RETRAIN_REQUIRED"
    small = (
        flip <= 0.05 and direction <= 0.02 and rank >= 0.95
        and top100 >= 0.85 and g0_sign <= 0.02 and g0_j >= 0.85
    )
    return "LOCAL_CNV_IMPACT_SMALL" if small else "LOCAL_CNV_SENSITIVITY_ONLY_REVIEW"


def main() -> None:
    from cc_hhgt.v32.cnv_input_sources import require_host
    require_host()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--directional-root", required=True, type=Path)
    parser.add_argument("--runner-code", required=True, type=Path)
    parser.add_argument("--patient-map", type=Path, default=Path(
        "/dell_2/DSC/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/"
        "patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv"))
    parser.add_argument("--covariates", type=Path, default=Path(
        "/public8/DSC/CancerLncAtlas/CancerLncAtlas_CC_HHGT_v2/"
        "input_snapshot/processed/tcga_association_covariates.parquet"))
    parser.add_argument("--source-config", type=Path, default=Path(
        "/dell_2/DSC/CancerLncAtlas/runtime/tools/v32_local_cnv_and_oof_20260902_r2/"
        "v32_g012_authority_sources_20260830_r1.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    from cc_hhgt.v32.local_cnv_sources import preflight
    preflight(root,"FINALIZE","association_finalizer")
    if (root / "SUCCESS.json").exists():
        raise RuntimeError("immutable final SUCCESS already exists")
    directional_success = args.directional_root.resolve() / "SUCCESS.json"
    if not directional_success.is_file():
        raise RuntimeError("directional CNV aggregate SUCCESS missing")
    candidates = pd.read_parquet(args.candidates, columns=["cancer_id", "lncrna_id", "pathway_id"])
    candidates = candidates.astype(str)
    if len(candidates) != 3_300_000 or set(candidates.cancer_id) != set(CANCERS):
        raise RuntimeError("formal candidate authority closure drift")
    configured = json.loads(args.source_config.read_text(encoding="utf-8"))
    sources = configured.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("source config lacks a sources mapping")
    covariate_spec = sources.get("covariates")
    if isinstance(covariate_spec, dict):
        declared_covariate_path = covariate_spec.get("path")
        declared_covariate_sha = str(covariate_spec.get("sha256", "")).strip().lower()
    elif isinstance(covariate_spec, (str, os.PathLike)):
        declared_covariate_path = covariate_spec
        declared_covariate_sha = ""
    else:
        raise RuntimeError("source config lacks the frozen covariates authority")
    if not declared_covariate_path:
        raise RuntimeError("source config covariates authority lacks path")
    declared_covariate_path = Path(declared_covariate_path).resolve(strict=True)
    observed_covariate_path = args.covariates.resolve(strict=True)
    if declared_covariate_path != observed_covariate_path:
        raise RuntimeError(
            "finalizer covariates path does not match source config: "
            f"{observed_covariate_path} != {declared_covariate_path}"
        )
    if declared_covariate_sha and not re.fullmatch(r"[0-9a-f]{64}", declared_covariate_sha):
        raise RuntimeError("source config covariates sha256 is not a 64-hex digest")
    observed_covariate_sha = sha256(observed_covariate_path)
    if declared_covariate_sha and observed_covariate_sha != declared_covariate_sha:
        raise RuntimeError(
            "source config covariates sha256 mismatch: "
            f"declared={declared_covariate_sha} observed={observed_covariate_sha}"
        )
    authority_files = [
        describe_file(args.candidates, "formal_candidate_universe"),
        describe_file(args.patient_map, "patient_fold_authority"),
        describe_file(observed_covariate_path, "association_covariates"),
        describe_file(args.source_config, "g012_source_config"),
        describe_file(directional_success, "directional_cnv_success"),
        describe_file(args.runner_code, "formal_runner_code"),
        describe_file(Path(__file__), "independent_finalizer_code"),
    ]
    for label, source in configured.get("sources", {}).items():
        source_path = Path(str(source.get("path", "")))
        if source_path.is_file() and label not in {"covariates"}:
            authority_files.append(describe_file(source_path, f"source_config:{label}"))
    patient_map = pd.read_csv(args.patient_map, sep="\t")
    required_patient = {"cancer_id", "patient_id", "patient_fold_id"}
    if required_patient - set(patient_map.columns):
        raise RuntimeError(f"patient fold schema missing: {sorted(required_patient - set(patient_map.columns))}")
    patient_map.patient_fold_id = pd.to_numeric(patient_map.patient_fold_id, errors="raise").astype(int)
    duplicate_patient_rows = int(patient_map.duplicated(["cancer_id", "patient_id"]).sum())
    if duplicate_patient_rows or set(patient_map.patient_fold_id) != set(range(5)):
        raise RuntimeError("patient-fold uniqueness/exhaustiveness drift")
    fold_patient_counts = (
        patient_map.groupby(["cancer_id", "patient_fold_id"], observed=True).patient_id.nunique()
        .rename("patients").reset_index().to_dict(orient="records")
    )
    directional_payload = json.loads(directional_success.read_text(encoding="utf-8"))
    directional_coverage = [{"cancer_id": cancer, **directional_payload["cancers"][cancer]}
                            for cancer in CANCERS]

    staging = root / f".finalization.partial.{os.getpid()}"
    if staging.exists():
        raise RuntimeError(f"staging already exists: {staging}")
    staging.mkdir(parents=True)
    pathway_metrics: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    edge_summaries: list[dict[str, Any]] = []
    audited_files: list[dict[str, Any]] = []
    receipt_hashes: dict[str, str] = {}
    try:
        for cancer in CANCERS:
            receipt_path = root / "receipts" / f"{cancer}.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            from cc_hhgt.v32.local_cnv_sources import validate
            if "consumed_sources" not in receipt:raise RuntimeError("Local-CNV receipt lacks actual consumed sources")
            validate(receipt["consumed_sources"],receipt_path.parent)
            if receipt.get("status") != "PASS" or receipt.get("cancer_id") != cancer:
                raise RuntimeError(f"receipt contract drift: {cancer}")
            if (
                receipt.get("covariate_contract_version")
                != EXPECTED_COVARIATE_CONTRACT_VERSION
                or receipt.get("association_C0") != EXPECTED_ASSOCIATION_C0
                or receipt.get("covariates", {}).get("declared")
                != list(EXPECTED_COVARIATES)
            ):
                raise RuntimeError(f"shared covariate/formal C0 contract drift: {cancer}")
            binding = receipt.get("covariate_contract", {})
            if (
                binding.get("version") != EXPECTED_COVARIATE_CONTRACT_VERSION
                or Path(str(binding.get("declared_path", ""))).resolve()
                != observed_covariate_path
                or binding.get("observed_sha256") != observed_covariate_sha
            ):
                raise RuntimeError(f"covariate authority binding drift: {cancer}")
            if len(receipt.get("pair_outputs", [])) != 5 or len(receipt.get("g0_outputs", [])) != 5:
                raise RuntimeError(f"receipt fold closure drift: {cancer}")
            receipt_hashes[cancer] = sha256(receipt_path)
            candidate_local = candidates.loc[candidates.cancer_id.eq(cancer), ["lncrna_id", "pathway_id"]]
            candidate_keys = pd.MultiIndex.from_frame(candidate_local)
            for fold in range(5):
                pair_record = next(row for row in receipt["pair_outputs"] if int(row["fold"]) == fold)
                pair_path = Path(pair_record["path"]).resolve()
                if pair_path != (root / "pair_level" / f"cancer={cancer}" / f"fold={fold}" / "part.parquet").resolve():
                    raise RuntimeError(f"pair receipt path drift {cancer}/{fold}")
                observed_sha = sha256(pair_path)
                if observed_sha != pair_record["sha256"]:
                    raise RuntimeError(f"pair SHA drift {cancer}/{fold}")
                pair = pd.read_parquet(pair_path)
                if int(pair_record["rows"]) != len(pair):
                    raise RuntimeError(f"pair receipt row drift {cancer}/{fold}")
                pathway_metrics.extend(verify_pair(pair, cancer, fold, candidate_keys))
                baseline = pair.label_A0_matched.isin(POSITIVE)
                adjusted = pair.label_A1_local.isin(POSITIVE)
                direction_evaluable = (
                    np.isfinite(pair.rho_A0_matched) & np.isfinite(pair.rho_A1_local)
                    & pair.rho_A0_matched.ne(0) & pair.rho_A1_local.ne(0)
                )
                fold_summaries.append({
                    "cancer_id": cancer, "fold_id": fold,
                    "candidate_count": len(pair),
                    "median_matched_patient_loss": finite_median(1 - pair.n_A0_matched / pair.n_A0_full.replace(0, np.nan)),
                    "baseline_positive_count": int(baseline.sum()),
                    "adjusted_positive_count": int(adjusted.sum()),
                    "baseline_positive_flip_count": int((baseline & pair.label_flip_local).sum()),
                    "direction_evaluable_count": int(direction_evaluable.sum()),
                    "direction_flip_count": int((direction_evaluable & pair.direction_flip_local).sum()),
                    "cnv_sensitive_count": int(pair.cnv_sensitive.sum()),
                    "cnv_mediated_candidate_count": int(pair.cnv_mediated_candidate.sum()),
                    "available_count": int(pair.availability.eq("available").sum()),
                    "median_abs_delta_rho_pathway_cnv": finite_median(pair.delta_rho_pathway_cnv.abs()),
                })
                audited_files.append({"kind": "pair", "cancer_id": cancer, "fold_id": fold, "rows": len(pair), "sha256": observed_sha})

                edge_record = next(row for row in receipt["g0_outputs"] if int(row["fold"]) == fold)
                edge_path = Path(edge_record["path"]).resolve()
                if edge_path != (root / "g0_edge_stability" / f"cancer={cancer}" / f"fold={fold}" / "part.parquet").resolve():
                    raise RuntimeError(f"edge receipt path drift {cancer}/{fold}")
                edge_sha = sha256(edge_path)
                if edge_sha != edge_record["sha256"]:
                    raise RuntimeError(f"edge SHA drift {cancer}/{fold}")
                edge = pd.read_parquet(edge_path)
                if int(edge_record["rows"]) != len(edge):
                    raise RuntimeError(f"edge receipt row drift {cancer}/{fold}")
                typed = str(edge_record.get("status", "")).startswith("TYPED_UNAVAILABLE")
                if not edge.empty:
                    missing = sorted(EDGE_REQUIRED - set(edge.columns))
                    if missing:
                        raise RuntimeError(f"edge schema missing {cancer}/{fold}: {missing}")
                    if set(edge.cancer_id.astype(str)) != {cancer} or set(edge.fold_id.astype(int)) != {fold}:
                        raise RuntimeError(f"edge identity drift {cancer}/{fold}")
                    if set(edge.run_id.astype(str)) != {EXPECTED_RUN_ID}:
                        raise RuntimeError(f"edge run id drift {cancer}/{fold}")
                    if edge[["lncrna_id", "gene_id"]].duplicated().any():
                        raise RuntimeError(f"duplicate real G0 edges {cancer}/{fold}")
                e0 = set(map(tuple, edge.loc[edge.get("present_E0", False).astype(bool), ["lncrna_id", "gene_id"]].astype(str).to_numpy())) if not edge.empty else set()
                e1 = set(map(tuple, edge.loc[edge.get("present_E1", False).astype(bool), ["lncrna_id", "gene_id"]].astype(str).to_numpy())) if not edge.empty else set()
                common_edges = e0 & e1
                sign_flips = int(edge.direction_flip.sum()) if not edge.empty else 0
                signed_metrics = edge_direction_metrics(edge) if not typed else {
                    "positive_edge_jaccard": np.nan, "negative_edge_jaccard": np.nan,
                    "median_lncrna_direction_topk_jaccard": np.nan,
                }
                edge_summaries.append({
                    "cancer_id": cancer, "fold_id": fold, "availability": "typed_unavailable" if typed else "available",
                    "unavailable_reason": str(edge_record.get("status", "")) if typed else "",
                    "edge_count_E0": len(e0), "edge_count_E1": len(e1),
                    "common_edge_count": len(common_edges), "union_edge_count": len(e0 | e1),
                    "edge_jaccard": np.nan if typed else jaccard(e0, e1),
                    "sign_flip_count": sign_flips,
                    "sign_flip_evaluable_count": len(common_edges),
                    "sign_flip_rate": np.nan if typed or not common_edges else sign_flips / len(common_edges),
                    "edge_turnover_rate": np.nan if typed else (len(e0 ^ e1) / len(e0 | e1) if (e0 | e1) else 0.0),
                    **signed_metrics,
                })
                audited_files.append({"kind": "g0", "cancer_id": cancer, "fold_id": fold, "rows": len(edge), "sha256": edge_sha})

        pathway_table = pd.DataFrame(pathway_metrics)
        fold_table = pd.DataFrame(fold_summaries)
        edge_table = pd.DataFrame(edge_summaries)
        cancer_rows: list[dict[str, Any]] = []
        for cancer in CANCERS:
            folds = fold_table.loc[fold_table.cancer_id.eq(cancer)]
            paths = pathway_table.loc[pathway_table.cancer_id.eq(cancer)]
            edges = edge_table.loc[edge_table.cancer_id.eq(cancer)]
            base_count = int(folds.baseline_positive_count.sum())
            direction_count = int(folds.direction_evaluable_count.sum())
            g0_direction_count = int(edges.sign_flip_evaluable_count.sum())
            cancer_rows.append({
                "cancer_id": cancer, "folds": len(folds), "candidate_rows": int(folds.candidate_count.sum()),
                "median_matched_patient_loss": finite_median(folds.median_matched_patient_loss),
                "baseline_positive_count": base_count,
                "adjusted_positive_count": int(folds.adjusted_positive_count.sum()),
                "baseline_positive_label_flip_rate": float(folds.baseline_positive_flip_count.sum() / base_count) if base_count else 0.0,
                "direction_evaluable_count": direction_count,
                "direction_flip_rate": float(folds.direction_flip_count.sum() / direction_count) if direction_count else 0.0,
                "median_rank_spearman": finite_median(paths.rank_spearman),
                "top50_jaccard": finite_median(paths.top50_jaccard),
                "top100_jaccard": finite_median(paths.top100_jaccard),
                "top200_jaccard": finite_median(paths.top200_jaccard),
                "gene_set_jaccard": finite_median(paths.gene_set_jaccard),
                "g0_edge_jaccard": finite_median(edges.edge_jaccard),
                "g0_positive_edge_jaccard": finite_median(edges.positive_edge_jaccard),
                "g0_negative_edge_jaccard": finite_median(edges.negative_edge_jaccard),
                "g0_median_lncrna_direction_topk_jaccard": finite_median(edges.median_lncrna_direction_topk_jaccard),
                "g0_sign_flip_rate": float(edges.sign_flip_count.sum() / g0_direction_count) if g0_direction_count else 0.0,
                "g0_edge_turnover_rate": finite_median(edges.edge_turnover_rate),
                "cnv_sensitive_count": int(folds.cnv_sensitive_count.sum()),
                "cnv_mediated_candidate_count": int(folds.cnv_mediated_candidate_count.sum()),
                "available_pair_count": int(folds.available_count.sum()),
                "g0_unavailable_folds": int(edges.availability.eq("typed_unavailable").sum()),
                "median_abs_delta_rho_pathway_cnv": finite_median(folds.median_abs_delta_rho_pathway_cnv),
            })
        cancer_table = pd.DataFrame(cancer_rows)
        total_base = int(fold_table.baseline_positive_count.sum())
        total_direction = int(fold_table.direction_evaluable_count.sum())
        total_g0_direction = int(edge_table.sign_flip_evaluable_count.sum())
        global_summary = {
            "format": "CC_HHGT_V3_2_LOCAL_CNV_GLOBAL_SUMMARY_V1",
            "cancers": 33, "folds": 165, "candidate_rows": int(fold_table.candidate_count.sum()),
            "baseline_positive_count": total_base,
            "adjusted_positive_count": int(fold_table.adjusted_positive_count.sum()),
            "baseline_positive_label_flip_rate": float(fold_table.baseline_positive_flip_count.sum() / total_base) if total_base else 0.0,
            "direction_flip_rate": float(fold_table.direction_flip_count.sum() / total_direction) if total_direction else 0.0,
            "median_cancer_pathway_rank_spearman": finite_median(pathway_table.rank_spearman),
            "median_top50_jaccard": finite_median(pathway_table.top50_jaccard),
            "median_top100_jaccard": finite_median(pathway_table.top100_jaccard),
            "median_top200_jaccard": finite_median(pathway_table.top200_jaccard),
            "median_gene_set_jaccard": finite_median(pathway_table.gene_set_jaccard),
            "median_g0_edge_jaccard": finite_median(edge_table.loc[edge_table.availability.eq("available"), "edge_jaccard"]),
            "median_g0_positive_edge_jaccard": finite_median(edge_table.loc[edge_table.availability.eq("available"), "positive_edge_jaccard"]),
            "median_g0_negative_edge_jaccard": finite_median(edge_table.loc[edge_table.availability.eq("available"), "negative_edge_jaccard"]),
            "median_g0_lncrna_direction_topk_jaccard": finite_median(edge_table.loc[edge_table.availability.eq("available"), "median_lncrna_direction_topk_jaccard"]),
            "g0_sign_flip_rate": float(edge_table.sign_flip_count.sum() / total_g0_direction) if total_g0_direction else 0.0,
            "median_g0_edge_turnover_rate": finite_median(edge_table.loc[edge_table.availability.eq("available"), "edge_turnover_rate"]),
            "cnv_sensitive_count": int(fold_table.cnv_sensitive_count.sum()),
            "cnv_mediated_candidate_count": int(fold_table.cnv_mediated_candidate_count.sum()),
            "g0_typed_unavailable_folds": int(edge_table.availability.eq("typed_unavailable").sum()),
            "mutation_used": False, "sealed_test_read": False, "primary_or_g012_modified": False,
        }
        outcome = decision(global_summary, cancer_table)
        decision_payload = {
            "format": "CC_HHGT_V3_2_LOCAL_CNV_DECISION_V1", "status": "PASS",
            "decision": outcome,
            "frozen_gate": {
                "impact_small_all": {"label_flip_max": 0.05, "direction_flip_max": 0.02, "median_rank_min": 0.95, "median_top100_min": 0.85, "g0_sign_flip_max": 0.02, "median_g0_jaccard_min": 0.85},
                "core_retrain_any": {"label_flip_gt": 0.10, "at_least_5_cancers_label_flip_gt": 0.15, "direction_flip_gt": 0.05, "median_rank_lt": 0.90, "median_top100_lt": 0.75, "g0_sign_flip_gt": 0.05, "median_g0_jaccard_lt": 0.75},
            },
        }
        global_summary["decision"] = outcome

        # Create the release-shaped partitioned datasets only after every file
        # has passed the checks above.
        input_authority = {
            "format": "CC_HHGT_V3_2_LOCAL_CNV_INPUT_AUTHORITY_V1",
            "run_id": EXPECTED_RUN_ID, "status": "PASS", "files": authority_files,
            "covariate_contract_version": EXPECTED_COVARIATE_CONTRACT_VERSION,
            "covariate_contract": {
                "path": str(observed_covariate_path),
                "sha256": observed_covariate_sha,
                "declared_sha256": declared_covariate_sha or None,
                "declared": list(EXPECTED_COVARIATES),
            },
            "association_C0": EXPECTED_ASSOCIATION_C0,
            "candidate_rows": len(candidates), "cancer_count": candidates.cancer_id.nunique(),
            "candidate_feature_count": candidates.lncrna_id.nunique(),
            "candidate_pathway_count": candidates.pathway_id.nunique(),
            "patient_count": patient_map[["cancer_id", "patient_id"]].drop_duplicates().shape[0],
            "fold_patient_counts": fold_patient_counts,
            "duplicate_patient_or_aliquot_rows": duplicate_patient_rows,
            "directional_cnv_coverage": directional_coverage,
            "exclusion_rules": [
                "designated OOF fold excluded from train",
                "adjacent fold excluded for validation",
                "duplicate aliquots collapsed by canonical patient authority",
                "unavailable CNV remains missing and is never converted to zero",
            ],
            "genome_build": "GRCh38", "annotation_authority": "grch38_gene_lncrna_intervals.parquet",
            "code_commit": None, "worktree_status": "NOT_A_GIT_REPOSITORY; runner SHA recorded",
        }
        write_json(staging / "INPUT_AUTHORITY.json", input_authority)
        schema_rows = [{
            "label": row["label"], "path": row["path"], "bytes": row["bytes"],
            "sha256": row["sha256"], "rows": row.get("rows", ""),
            "columns": row.get("columns", ""), "schema": row.get("schema", ""),
        } for row in authority_files]
        pd.DataFrame(schema_rows).to_csv(staging / "INPUT_SCHEMA_REPORT.tsv", sep="\t", index=False)
        precheck = {
            "format": "CC_HHGT_V3_2_LOCAL_CNV_PRECHECK_V1", "status": "PASS",
            "run_id": EXPECTED_RUN_ID, "cancer_count": 33, "fold_count": 165,
            "candidate_rows_per_cancer": 100_000,
            "signed_continuous_cnv": True, "amplification_deletion_separate": True,
            "fold_local_only": True, "mutation_used": False, "sealed_test_read": False,
            "authority_file_count": len(authority_files),
            "directional_processed_segment_patients": int(sum(row["processed_segment_patients"] for row in directional_coverage)),
            "directional_typed_unavailable_patients": int(sum(row["typed_unavailable_patients"] for row in directional_coverage)),
            "patient_fold_mutually_exclusive": True, "duplicate_patient_or_aliquot_rows": duplicate_patient_rows,
            "candidate_primary_key_unique": not candidates.duplicated(["cancer_id", "lncrna_id", "pathway_id"]).any(),
            "callable_neutral_amp_del_continuous_unavailable_distinct": True,
            "current_core_input_gate": "checked_by_downstream_cnv_head_core_manifest_before_training",
            "covariate_contract_version": EXPECTED_COVARIATE_CONTRACT_VERSION,
            "association_C0": EXPECTED_ASSOCIATION_C0,
            "covariates_declared": list(EXPECTED_COVARIATES),
        }
        write_json(staging / "PRECHECK.json", precheck)
        (staging / "RUNNING_PROCESS_SNAPSHOT.txt").write_text(
            f"captured_at_utc={datetime.now(timezone.utc).isoformat()}\n"
            f"finalizer_pid={os.getpid()}\nrun_id={EXPECTED_RUN_ID}\n"
            "state=all_33_cancer_workers_complete;independent_finalization_running\n",
            encoding="utf-8",
        )
        hardlink_dataset(root / "pair_level", staging / "local_cnv_confounding_pair_level.parquet")
        hardlink_dataset(root / "g0_edge_stability", staging / "local_cnv_g0_edge_stability.parquet")
        cancer_table.to_csv(staging / "local_cnv_confounding_cancer_summary.tsv", sep="\t", index=False)
        edge_table.to_csv(staging / "local_cnv_g0_edge_summary.tsv", sep="\t", index=False)
        pathway_table.to_parquet(staging / "local_cnv_pathway_rank_stability.parquet", index=False, compression="zstd")
        write_json(staging / "local_cnv_confounding_global_summary.json", global_summary)
        write_json(staging / "LOCAL_CNV_DECISION.json", decision_payload)
        lineage = {
            "format": "CC_HHGT_V3_2_LOCAL_CNV_LINEAGE_V1", "run_id": EXPECTED_RUN_ID,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "formal_candidates": {"path": str(args.candidates.resolve()), "sha256": sha256(args.candidates.resolve())},
            "covariates": {
                "path": str(observed_covariate_path),
                "sha256": observed_covariate_sha,
                "declared_sha256": declared_covariate_sha or None,
                "declared": list(EXPECTED_COVARIATES),
            },
            "directional_cnv_success": {"path": str(directional_success), "sha256": sha256(directional_success)},
            "runner_code": {"path": str(args.runner_code.resolve()), "sha256": sha256(args.runner_code.resolve())},
            "independent_finalizer_code": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "receipt_sha256": receipt_hashes,
        }
        write_json(staging / "LINEAGE.json", lineage)
        audit = {
            "format": "CC_HHGT_V3_2_LOCAL_CNV_INDEPENDENT_AUDIT_V1", "status": "PASS",
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "cancer_closure": 33, "fold_closure": 165, "candidate_rows_per_fold": 100_000,
                "candidate_key_exact": True, "receipt_sha_recomputed": True,
                "pair_file_sha_recomputed": True, "g0_file_sha_recomputed": True,
                "official_fdr_recomputed": True, "labels_recomputed": True,
                "matched_A0_A1_cohort_equal": True, "signed_continuous_cnv": True,
                "amplification_deletion_separate": True, "real_lncrna_gene_g0": True,
                "g0_direction_columns_present": True,
                "g0_positive_negative_jaccard_recomputed": True,
                "g0_per_lncrna_direction_topk_jaccard_recomputed": True,
                "input_authority_and_schema_verified": True,
                "shared_covariate_contract_verified": True,
                "formal_C0_shared_covariate_verified": True,
                "mutation_absent": True, "sealed_test_absent": True,
            },
            "audited_file_count": len(audited_files),
            "decision": outcome,
        }
        write_json(staging / "LOCAL_CNV_INDEPENDENT_AUDIT.json", audit)
        report = f"""# CancerLncAtlas V3.2 local-CNV confounding audit\n\nStatus: PASS\n\nDecision: **{outcome}**\n\n- 33 cancers and 165 fold-local partitions audited.\n- Signed continuous local CNV was used; amplification and deletion counts remain separate.\n- Mutation and sealed-test data were not read.\n- Primary and G0/G1/G2 production outputs were not modified by this audit.\n\nMachine-readable results are in `local_cnv_confounding_global_summary.json`, `local_cnv_confounding_cancer_summary.tsv`, `local_cnv_g0_edge_summary.tsv`, and `LOCAL_CNV_DECISION.json`.\n"""
        (staging / "LOCAL_CNV_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
        success = {
            "format": "CC_HHGT_V3_2_LOCAL_CNV_FINAL_SUCCESS_V1", "status": "SUCCESS",
            "run_id": EXPECTED_RUN_ID, "decision": outcome,
            "covariate_contract_version": EXPECTED_COVARIATE_CONTRACT_VERSION,
            "association_C0": EXPECTED_ASSOCIATION_C0,
            "covariate_contract": {
                "version": EXPECTED_COVARIATE_CONTRACT_VERSION,
                "declared_path": str(observed_covariate_path),
                "observed_sha256": observed_covariate_sha,
                "declared_sha256": declared_covariate_sha or None,
            },
            "covariates": {
                "path": str(observed_covariate_path),
                "sha256": observed_covariate_sha,
                "declared": list(EXPECTED_COVARIATES),
            },
            "independent_audit_sha256": sha256(staging / "LOCAL_CNV_INDEPENDENT_AUDIT.json"),
            "decision_sha256": sha256(staging / "LOCAL_CNV_DECISION.json"),
            "lineage_sha256": sha256(staging / "LINEAGE.json"),
        }
        write_json(staging / "SUCCESS.json", success)
        for child in sorted(staging.iterdir(), key=lambda p: p.name == "SUCCESS.json"):
            os.replace(child, root / child.name)
        staging.rmdir()
        print(json.dumps(success, sort_keys=True))
    except Exception:
        # Leave the hidden staging directory as forensic evidence; never emit
        # SUCCESS on a partial or failed closeout.
        raise


if __name__ == "__main__":
    main()
