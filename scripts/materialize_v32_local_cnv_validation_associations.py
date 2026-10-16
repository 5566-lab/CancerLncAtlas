#!/usr/bin/env python3
"""Materialize label-only validation A1 associations for corrected V3.2 core.

Each outer fold uses only its adjacent validation patient partition to produce
an independent replication target.  These values are labels, never model
features or fitted model parameters.  The designated OOF/test partition is not
read, unavailable local CNV remains null, and every candidate key is retained.
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

from cc_hhgt.v32 import bulk_coexpression_release as bulk
from cc_hhgt.v32.directional_cnv import DirectionalCNVStore
from cc_hhgt.v32.local_cnv_confounding import (
    COVARIATE_CONTRACT_VERSION,
    association_label,
    bh_fdr,
    canonical_patient,
    DEFAULT_COVARIATES,
    direction,
    fold_local_pair_models,
    pathway_within_cancer_fdr,
    patient_first_matrix,
)
from cc_hhgt.v32.local_cnv_core_preparation import (
    sha256_file,
    validate_local_cnv_audit,
)


CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)
CANDIDATES_PER_CANCER = 100_000


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


@contextmanager
def _cancer_lock(root: Path, cancer: str):
    path = root / "locks" / f"{cancer}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "posix":
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _patient_hash(values: list[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode()).hexdigest()


def _bind_covariate_authority(
    source_config_path: Path, sources: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Resolve and verify the covariate authority declared by the manifest."""
    spec = sources.get("covariates")
    if spec is None:
        raise RuntimeError("Source config lacks the frozen covariates authority")
    if isinstance(spec, dict):
        raw_path = spec.get("path")
        if not raw_path:
            raise RuntimeError("Source config covariates authority lacks path")
        declared_sha = str(spec.get("sha256", "")).strip().lower()
    elif isinstance(spec, (str, os.PathLike)):
        raw_path = spec
        declared_sha = ""
    else:
        raise RuntimeError("Source config covariates authority must be a path record")
    path = Path(raw_path).resolve(strict=True)
    if declared_sha and not re.fullmatch(r"[0-9a-f]{64}", declared_sha):
        raise RuntimeError("Source config covariates sha256 is not a 64-hex digest")
    observed_sha = sha256_file(path)
    if declared_sha and observed_sha != declared_sha:
        raise RuntimeError(
            "Covariates source-config sha256 mismatch: "
            f"declared={declared_sha} observed={observed_sha}"
        )
    binding = {
        "version": COVARIATE_CONTRACT_VERSION,
        "source_config": str(source_config_path.resolve()),
        "declared_path": str(path),
        "argument_path": str(path),
        "declared_sha256": declared_sha or None,
        "observed_sha256": observed_sha,
        "path_match": True,
        "sha256_verified": bool(declared_sha),
    }
    return path, binding


def _matrix(path: Path, cancer: str, entity: str, value: str, *,capture=None,role="") -> pd.DataFrame:
    columns = ["cancer_id", "patient_id", entity, value]
    frame = (pd.read_parquet(path,columns=columns) if capture is None else
             capture.read(path,lambda p:pd.read_parquet(p,columns=columns),role=role,cancer=cancer,columns=columns))
    if set(frame.cancer_id.astype(str)) != {cancer}:
        raise RuntimeError(f"Cancer identity drift in {path}")
    matrix, _ = patient_first_matrix(
        frame, feature_column=entity, value_column=value
    )
    return matrix.astype(np.float32)


def _validation_a1(
    pairs: pd.DataFrame,
    lnc: pd.DataFrame,
    pathway: pd.DataFrame,
    cnv: DirectionalCNVStore,
    patients: list[str],
    design: np.ndarray,
) -> pd.DataFrame:
    common = sorted(set(patients) & set(lnc.index) & set(pathway.index) & set(cnv.patient_ids))
    if design.ndim != 2 or design.shape[0] != len(common):
        raise ValueError("validation covariate design is not aligned to common patients")
    out = pairs.copy().reset_index(drop=True)
    lnc_ids = sorted(out.lncrna_id.astype(str).unique())
    pathway_ids = sorted(out.pathway_id.astype(str).unique())
    lmap = {value: index for index, value in enumerate(lnc_ids)}
    pmap = {value: index for index, value in enumerate(pathway_ids)}
    cli = np.asarray([cnv.lncrna_index.get(value, -1) for value in lnc_ids], dtype=int)
    rows = np.asarray([cnv.patient_index[value] for value in common], dtype=int)
    x = lnc.reindex(index=common, columns=lnc_ids).to_numpy(float)
    y = pathway.reindex(index=common, columns=pathway_ids).to_numpy(float)
    local_signed = np.full((len(common), len(lnc_ids)), np.nan, dtype=np.float32)
    local_callable = np.zeros((len(common), len(lnc_ids)), dtype=bool)
    valid_lnc = cli >= 0
    if len(common) and valid_lnc.any():
        local_signed[:, valid_lnc] = cnv.lncrna_signed[np.ix_(rows, cli[valid_lnc])]
        local_callable[:, valid_lnc] = cnv.lncrna_callable[np.ix_(rows, cli[valid_lnc])]

    rho = np.full(len(out), np.nan)
    pvalue = np.full(len(out), np.nan)
    n_samples = np.zeros(len(out), dtype=np.int32)
    beta = np.full(len(out), np.nan)
    callable_n = np.zeros(len(out), dtype=np.int32)
    rank = np.zeros(len(out), dtype=np.int16)
    degrees = np.zeros(len(out), dtype=np.int32)
    association_available = np.zeros(len(out), dtype=bool)
    for start in range(0, len(out), 1024):
        stop = min(start + 1024, len(out))
        li = out.lncrna_id.iloc[start:stop].astype(str).map(lmap).to_numpy(int)
        pi = out.pathway_id.iloc[start:stop].astype(str).map(pmap).to_numpy(int)
        expression = x[:, li]
        activity = y[:, pi]
        signed = local_signed[:, li]
        callable_value = local_callable[:, li]
        result = fold_local_pair_models(
            expression,
            activity,
            signed,
            np.full_like(signed, np.nan),
            design,
            local_callable=callable_value,
            pathway_callable=np.zeros_like(callable_value),
        )
        rho[start:stop] = result["rho_a1_local"]
        pvalue[start:stop] = result["p_a1_local"]
        n_samples[start:stop] = result["n_a1_local"]
        beta[start:stop] = result["local_cnv_beta"]
        rank[start:stop] = result["residual_design_rank_a1"]
        degrees[start:stop] = result["correlation_df_a1"]
        association_available[start:stop] = result["available"]
        callable_n[start:stop] = callable_value.sum(axis=0)

    out["n_A1_validation"] = n_samples
    out["rho_A1_validation"] = rho
    out["p_A1_validation"] = pvalue
    out["local_cnv_beta_A1_validation"] = beta
    out["residual_design_rank_A1_validation"] = rank
    out["correlation_df_A1_validation"] = degrees
    out["local_cnv_callable_n"] = callable_n
    out["fdr_cancer_global_A1_validation"] = bh_fdr(pvalue)
    out["fdr_pathway_within_cancer_A1_validation"] = pathway_within_cancer_fdr(
        out, "p_A1_validation"
    )
    out["fdr_official_A1_validation"] = out.fdr_cancer_global_A1_validation
    out["label_A1_validation"] = [
        association_label(effect, fdr)
        for effect, fdr in zip(rho, out.fdr_official_A1_validation, strict=True)
    ]
    out["direction_A1_validation"] = [direction(value) for value in rho]
    available = association_available
    out["availability"] = np.where(available, "available", "typed_unavailable")
    out["unavailable_reason"] = np.where(
        callable_n < 12,
        "LOCAL_CNV_INSUFFICIENT_VALIDATION_PATIENTS",
        np.where(~available, "ASSOCIATION_NOT_ESTIMABLE", ""),
    )
    out["validation_patient_n"] = len(common)
    return out


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from cc_hhgt.v32.local_cnv_sources import TransformationCapture,preflight,validate,OUTPUT_ROLES
    cancer = args.cancer.upper()
    if cancer not in CANCERS:
        raise RuntimeError(f"Unknown cancer: {cancer}")
    output = args.output_root.resolve()
    preflight(output,cancer,"validation")
    capture=TransformationCapture()
    source_config_path = args.source_config.resolve(strict=True)
    config = capture.json(source_config_path,role="source_config")
    sources = config.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("Source config lacks a sources mapping")
    covariates_path, covariate_binding = _bind_covariate_authority(
        source_config_path, sources
    )
    receipt_path = output / "receipts" / f"{cancer}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if "consumed_sources" not in receipt:raise RuntimeError("Historical validation output lacks consumed-source binding; use a new output")
        validate(receipt["consumed_sources"],receipt_path.parent,expected_inputs={
            "source_config":source_config_path,"covariates":covariates_path,
            "candidates":args.candidates,"patient_map":args.patient_map},
            compact_root=args.directional_root/f"cancer={cancer}")
        if receipt.get("run_id")!=args.run_id:raise ValueError("Local-CNV validation reused run identity differs")
        if receipt.get("status") != "PASS" or receipt.get("cancer_id") != cancer:
            raise RuntimeError(f"Existing validation receipt is invalid: {receipt_path}")
        if receipt.get("covariate_contract_version") != COVARIATE_CONTRACT_VERSION:
            raise RuntimeError(
                f"Existing validation receipt lacks {COVARIATE_CONTRACT_VERSION}: {receipt_path}"
            )
        if receipt.get("covariate_contract") != covariate_binding:
            raise RuntimeError(
                f"Existing validation receipt covariate authority drift: {receipt_path}"
            )
        for record in receipt.get("outputs", []):
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Existing validation output drift: {path}")
        return receipt

    audit = validate_local_cnv_audit(
        args.local_cnv_audit_root,
        expected_success_sha256=args.local_cnv_success_sha256,
    )
    covariates = capture.read(covariates_path,pd.read_parquet,role="covariates")
    if not {"cancer_id", "patient_id"}.issubset(covariates.columns):
        raise RuntimeError("Covariates authority lacks cancer_id/patient_id")
    covariates["cancer_id"] = covariates.cancer_id.astype(str).str.upper()
    covariates["patient_id"] = covariates.patient_id.map(canonical_patient)
    lnc_path = Path(sources["lnc_expression_root"]["path"]) / f"cancer_id={cancer}" / "part-0.parquet"
    pathway_path = Path(sources["pathway_activity_root"]["path"]) / f"cancer_id={cancer}" / "part-0.parquet"
    lnc = _matrix(lnc_path, cancer, "lncrna_id", "logcpm",capture=capture,role="lncrna_expression")
    pathway = _matrix(pathway_path, cancer, "pathway_id", "activity_score",capture=capture,role="pathway_activity")
    columns=["cancer_id","lncrna_id","pathway_id"]
    pairs = capture.read(args.candidates,lambda p:pd.read_parquet(p,columns=columns,
        filters=[("cancer_id","==",cancer)]),role="candidates",cancer=cancer,columns=columns).astype(str)
    if len(pairs) != CANDIDATES_PER_CANCER or pairs.duplicated().any():
        raise RuntimeError(f"Candidate authority drift for {cancer}: {len(pairs)}")
    patient_map = capture.read(args.patient_map,lambda p:pd.read_csv(p,sep="\t"),role="patient_map")
    patient_map["cancer_id"] = patient_map.cancer_id.astype(str)
    patient_map["patient_id"] = patient_map.patient_id.map(canonical_patient)
    patient_map["patient_fold_id"] = pd.to_numeric(
        patient_map.patient_fold_id, errors="raise"
    ).astype(int)
    cnv = DirectionalCNVStore(args.directional_root / f"cancer={cancer}",source_capture=capture,cancer=cancer)
    records = []
    for fold in range(5):
        validation_fold = (fold + 1) % 5
        patients = sorted(
            patient_map.loc[
                patient_map.cancer_id.eq(cancer)
                & patient_map.patient_fold_id.eq(validation_fold),
                "patient_id",
            ].astype(str)
        )
        common = sorted(set(patients) & set(lnc.index) & set(pathway.index) & set(cnv.patient_ids))
        design, covariates_used, missing_covariate_patients = bulk._covariate_design(
            covariates,
            cancer=cancer,
            patients=common,
            columns=DEFAULT_COVARIATES,
        )
        result = _validation_a1(pairs, lnc, pathway, cnv, patients, design)
        result.insert(3, "fold_id", int(fold))
        result.insert(4, "split_role", "validation_label_only")
        result["validation_patient_fold_id"] = validation_fold
        result["validation_patient_sha256"] = _patient_hash(patients)
        result["run_id"] = args.run_id
        result["local_cnv_audit_success_sha256"] = audit["success_sha256"]
        result["sealed_test_accessed"] = False
        result["covariates_used"] = json.dumps(covariates_used, separators=(",", ":"))
        path = output / "validation_associations" / f"cancer={cancer}" / f"fold={fold}" / "part.parquet"
        _atomic_parquet(path, result)
        records.append(
            {
                "fold": fold,
                "output_kind":"validation","consumed_roles":sorted(OUTPUT_ROLES["validation"]),
                "validation_fold": validation_fold,
                "validation_patient_n": len(patients),
                "validation_patient_sha256": _patient_hash(patients),
                "rows": len(result),
                "available": int(result.availability.eq("available").sum()),
                "typed_unavailable": int(result.availability.eq("typed_unavailable").sum()),
                "covariates_used": list(covariates_used),
                "missing_covariate_patients": int(missing_covariate_patients),
                "base_design_rank": int(np.linalg.matrix_rank(design)) if len(design) else 0,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    receipt = {
        "format": "CC_HHGT_V3_2_LOCAL_CNV_VALIDATION_A1_CANCER_V1",
        "status": "PASS",
        "run_id": args.run_id,
        "cancer_id": cancer,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": sha256_file(args.candidates),
        "patient_map_sha256": sha256_file(args.patient_map),
        "directional_cnv_success_sha256": sha256_file(args.directional_root / "SUCCESS.json"),
        "covariate_contract_version": COVARIATE_CONTRACT_VERSION,
        "covariate_contract": covariate_binding,
        "covariates": {
            "path": str(covariates_path.resolve()),
            "sha256": covariate_binding["observed_sha256"],
            "source_config_path": str(source_config_path),
            "source_config_declared_path": covariate_binding["declared_path"],
            "source_config_declared_sha256": covariate_binding["declared_sha256"],
            "declared": list(DEFAULT_COVARIATES),
        },
        "local_cnv_authority": audit,
        "outputs": records,
        "gates": {
            "candidate_closure_per_fold": CANDIDATES_PER_CANCER,
            "validation_label_only": True,
            "validation_partition_independent_of_train": True,
            "designated_oof_partition_read": False,
            "model_feature_written": False,
            "mutation_used": False,
            "unavailable_preserved_as_null": True,
        },
    }
    receipt["consumed_sources"]=capture.publish(output/"source_reads"/(cancer+".json"),
        producer="validation",cancer=cancer,outputs=records,runner=Path(__file__))
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    from cc_hhgt.v32.cnv_input_sources import require_host
    require_host()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", required=True)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--patient-map", required=True, type=Path)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--directional-root", required=True, type=Path)
    parser.add_argument("--local-cnv-audit-root", required=True, type=Path)
    parser.add_argument("--local-cnv-success-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--run-id", default="V32_LOCAL_CNV_VALIDATION_A1_20260903_R1"
    )
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    with _cancer_lock(args.output_root, args.cancer.upper()):
        receipt = _run(args)
    print(json.dumps({"status": receipt["status"], "cancer_id": receipt["cancer_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
