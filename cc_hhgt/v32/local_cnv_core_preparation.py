"""Strict adapters from the local-CNV audit to fresh core preparation.

The audit tables deliberately keep unavailable rows in the 3.3M candidate
universe.  These helpers preserve that closure while producing the field
contract consumed by the formal G0/G1/G2 materializer.  A placeholder binary
label is carried for tensor shape only; ``association_available`` is the
authoritative loss mask and unavailable effects remain NaN.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cc_hhgt.v32.local_cnv_confounding import (
    COVARIATE_CONTRACT_VERSION,
    DEFAULT_COVARIATES,
)


KEYS = ("cancer_id", "lncrna_id", "pathway_id")
POSITIVE_LABELS = frozenset({"strong_positive", "weak_positive"})
VALID_LABELS = POSITIVE_LABELS | {"unlabeled"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_local_cnv_audit(
    root: str | Path, *, expected_success_sha256: str | None = None,
    require_correction: bool = True,
) -> dict[str, Any]:
    audit_root = Path(root).resolve()
    success_path = audit_root / "SUCCESS.json"
    independent_path = audit_root / "LOCAL_CNV_INDEPENDENT_AUDIT.json"
    decision_path = audit_root / "LOCAL_CNV_DECISION.json"
    for path in (success_path, independent_path, decision_path):
        if not path.is_file():
            raise RuntimeError(f"Local-CNV authority is incomplete: {path}")
    success_sha = sha256_file(success_path)
    if expected_success_sha256 and success_sha != str(expected_success_sha256).lower():
        raise RuntimeError("Local-CNV SUCCESS SHA256 drift")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if success.get("status") not in {"SUCCESS", "PASS"}:
        raise RuntimeError("Local-CNV SUCCESS authority did not pass")
    if independent.get("status") != "PASS":
        raise RuntimeError("Local-CNV independent audit did not pass")
    if decision.get("status") != "PASS":
        raise RuntimeError("Local-CNV decision receipt did not pass")
    if decision.get("decision") not in {
        "LOCAL_CNV_CORE_RETRAIN_REQUIRED", "LOCAL_CNV_IMPACT_SMALL", "LOCAL_CNV_SENSITIVITY_ONLY_REVIEW"
    }:
        raise RuntimeError("Unknown local-CNV audit decision")
    if require_correction and decision.get("decision") != "LOCAL_CNV_CORE_RETRAIN_REQUIRED":
        raise RuntimeError("Local-CNV authority does not require corrected core inputs")
    if success.get("covariate_contract_version") != COVARIATE_CONTRACT_VERSION:
        raise RuntimeError("Local-CNV SUCCESS lacks the shared covariate contract")
    if success.get("association_C0") != "shared_covariate_current_v32_formal_association":
        raise RuntimeError("Local-CNV SUCCESS is not from the shared-covariate C0 path")
    covariates = success.get("covariates", {})
    if covariates.get("declared") != list(DEFAULT_COVARIATES):
        raise RuntimeError("Local-CNV SUCCESS covariate declaration drift")
    binding = success.get("covariate_contract", {})
    if binding.get("version") != COVARIATE_CONTRACT_VERSION:
        raise RuntimeError("Local-CNV SUCCESS covariate authority binding missing")
    cohort = success.get("cohort")
    if cohort is None and (independent.get("cohort") is not None or decision.get("cohort") is not None):
        raise RuntimeError("Local-CNV cohort declaration was removed from SUCCESS")
    if cohort is not None:
        from .local_cnv_cohort import replay, require_receipt
        replay(cohort, audit_root)
        require_receipt(independent, cohort); require_receipt(decision, cohort)
        if (success.get("independent_audit_sha256") != sha256_file(independent_path)
                or success.get("decision_sha256") != sha256_file(decision_path)
                or success.get("decision") != decision["decision"]
                or independent.get("checks", {}).get("cancer_closure") != len(cohort["cancers"])
                or independent.get("checks", {}).get("fold_closure") != 5 * len(cohort["cancers"])):
            raise RuntimeError("Local-CNV cohort closeout binding differs")
    return {
        **({"cohort": cohort} if cohort is not None else {}),
        "root": str(audit_root),
        "success_sha256": success_sha,
        "independent_audit_sha256": sha256_file(independent_path),
        "decision_sha256": sha256_file(decision_path),
        "decision": decision["decision"],
        "covariate_contract_version": COVARIATE_CONTRACT_VERSION,
        "covariate_contract": binding,
        "covariates_declared": list(DEFAULT_COVARIATES),
    }


def validate_validation_a1_authority(
    root: str | Path, *, expected_success_sha256: str | None = None
) -> dict[str, Any]:
    """Validate the independently audited validation-A1 authority."""

    authority_root = Path(root).resolve()
    success_path = authority_root / "SUCCESS.json"
    independent_path = authority_root / "INDEPENDENT_AUDIT.json"
    for path in (success_path, independent_path):
        if not path.is_file():
            raise RuntimeError(f"Validation-A1 authority is incomplete: {path}")
    success_sha = sha256_file(success_path)
    if expected_success_sha256 and success_sha != str(expected_success_sha256).lower():
        raise RuntimeError("Validation-A1 SUCCESS SHA256 drift")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    if success.get("status") != "PASS":
        raise RuntimeError("Validation-A1 SUCCESS authority did not pass")
    if independent.get("status") != "PASS":
        raise RuntimeError("Validation-A1 independent audit did not pass")
    if success.get("covariate_contract_version") != COVARIATE_CONTRACT_VERSION:
        raise RuntimeError("Validation-A1 SUCCESS lacks the shared covariate contract")
    if success.get("covariates", {}).get("declared") != list(DEFAULT_COVARIATES):
        raise RuntimeError("Validation-A1 SUCCESS covariate declaration drift")
    gates = success.get("gates", {})
    if (
        gates.get("validation_label_only") is not True
        or gates.get("designated_oof_partition_read") is not False
        or gates.get("sealed_test_accessed") is not False
        or gates.get("mutation_used") is not False
        or gates.get("candidate_closure") is not True
        or gates.get("typed_unavailable_preserved") is not True
        or gates.get("shared_covariate_contract") != COVARIATE_CONTRACT_VERSION
        or gates.get("covariates_declared") != list(DEFAULT_COVARIATES)
    ):
        raise RuntimeError("Validation-A1 governance contract failed")
    cohort = success.get("cohort")
    if cohort is None and independent.get("cohort") is not None:
        raise RuntimeError("Validation-A1 cohort declaration was removed from SUCCESS")
    if cohort is not None:
        from .local_cnv_cohort import replay, require_receipt
        replay(cohort, authority_root); require_receipt(independent, cohort)
        if (success.get("independent_audit_sha256") != sha256_file(independent_path)
                or independent.get("rows") != cohort["expected_pair_rows_all_folds"]
                or independent.get("fold_partitions") != 5 * len(cohort["cancers"])
                or success.get("candidate_sha256") != cohort["candidates"]["sha256"]
                or success.get("patient_map_sha256") != cohort["patient_map"]["sha256"]):
            raise RuntimeError("Validation-A1 cohort closeout binding differs")
    return {
        **({"cohort": cohort, "local_cnv_audit_success_sha256": success.get("local_cnv_audit_success_sha256")}
           if cohort is not None else {}),
        "root": str(authority_root),
        "success_sha256": success_sha,
        "independent_audit_sha256": sha256_file(independent_path),
        "candidate_sha256": success.get("candidate_sha256"),
        "patient_map_sha256": success.get("patient_map_sha256"),
        "covariate_contract_version": COVARIATE_CONTRACT_VERSION,
        "covariates_declared": list(DEFAULT_COVARIATES),
        "covariate_contract": success.get("covariate_contract", {}),
    }


def adjusted_association_frame(
    frame: pd.DataFrame,
    *,
    prefix: str,
    expected_cancer: str | None = None,
    expected_fold: int | None = None,
) -> pd.DataFrame:
    """Map one audited A1 table to the formal association feature contract."""

    columns = {
        "rho": f"rho_{prefix}",
        "p": f"p_{prefix}",
        "fdr": f"fdr_official_{prefix}",
        "label": f"label_{prefix}",
        "direction": f"direction_{prefix}",
        "n": f"n_{prefix}",
    }
    required = set(KEYS) | set(columns.values()) | {
        "fold_id",
        "availability",
        "unavailable_reason",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Adjusted association table lacks columns: {missing}")
    value = frame.copy().reset_index(drop=True)
    if value.duplicated(list(KEYS)).any():
        raise RuntimeError("Adjusted association candidate keys are not unique")
    if expected_cancer is not None and set(value.cancer_id.astype(str)) != {
        str(expected_cancer)
    }:
        raise RuntimeError("Adjusted association cancer identity drift")
    folds = pd.to_numeric(value.fold_id, errors="coerce")
    if expected_fold is not None and (
        folds.isna().any() or not folds.eq(int(expected_fold)).all()
    ):
        raise RuntimeError("Adjusted association fold identity drift")

    rho = pd.to_numeric(value[columns["rho"]], errors="coerce")
    pvalue = pd.to_numeric(value[columns["p"]], errors="coerce")
    fdr = pd.to_numeric(value[columns["fdr"]], errors="coerce")
    n_samples = pd.to_numeric(value[columns["n"]], errors="coerce").fillna(0).astype(int)
    available = value.availability.astype(str).eq("available")
    available &= np.isfinite(rho) & np.isfinite(pvalue) & np.isfinite(fdr)
    declared_unavailable = value.availability.astype(str).eq("typed_unavailable")
    if not (available | declared_unavailable).all():
        raise RuntimeError("Adjusted association has inconsistent availability semantics")
    if value.loc[declared_unavailable, "unavailable_reason"].astype(str).str.strip().eq("").any():
        raise RuntimeError("Typed-unavailable association lacks a reason")
    if rho.loc[declared_unavailable].notna().any():
        raise RuntimeError("Typed-unavailable association carries a numeric effect")

    labels = value[columns["label"]].astype(str)
    if not set(labels.loc[available]).issubset(VALID_LABELS):
        raise RuntimeError("Adjusted association contains an unknown available label")
    # The effective residual design is candidate-specific: a constant or
    # design-collinear CNV column does not increase the rank, whereas a
    # variable CNV column does.  The audit must carry the rank/df it actually
    # used; deriving ``2``/``n-3`` here silently misstates constant-CNV rows.
    rank_column = next(
        (
            candidate
            for candidate in (
                f"residual_design_rank_{prefix}",
                "residual_design_rank",
            )
            if candidate in value.columns
        ),
        None,
    )
    df_column = next(
        (
            candidate
            for candidate in (
                f"correlation_df_{prefix}",
                "correlation_df",
            )
            if candidate in value.columns
        ),
        None,
    )
    if rank_column is None or df_column is None:
        raise ValueError(
            "Adjusted association must carry per-candidate residual rank and "
            "correlation df; re-materialize with the corrected CNV audit"
        )
    residual_rank = pd.to_numeric(value[rank_column], errors="coerce")
    correlation_df = pd.to_numeric(value[df_column], errors="coerce")
    if available.any():
        if (
            residual_rank.loc[available].isna().any()
            or correlation_df.loc[available].isna().any()
        ):
            raise RuntimeError("Available adjusted association lacks finite rank/df")
        rank_float = residual_rank.loc[available].to_numpy(float)
        df_float = correlation_df.loc[available].to_numpy(float)
        n_float = n_samples.loc[available].to_numpy(float)
        if (
            (rank_float < 1).any()
            or (df_float <= 0).any()
            or not np.equal(rank_float, np.floor(rank_float)).all()
            or not np.equal(df_float, np.floor(df_float)).all()
            or not np.equal(df_float, n_float - rank_float - 1).all()
        ):
            raise RuntimeError("Adjusted association rank/df contract failed")
    label_class = labels.where(available, "unavailable")
    result = value[list(KEYS)].astype(str).copy()
    result["discovery_effect"] = rho
    result["discovery_pvalue"] = pvalue
    result["discovery_fdr"] = fdr
    result["association_n_samples"] = n_samples
    result["residual_design_rank"] = np.where(
        available, residual_rank.to_numpy(float), 0
    ).astype("int16")
    result["correlation_df"] = np.where(
        available, correlation_df.to_numpy(float), 0
    ).astype("int32")
    result["label_class"] = label_class
    result["proxy_label"] = (
        label_class.isin(POSITIVE_LABELS) & available
    ).astype("int8")
    result["weak_positive"] = (
        label_class.eq("weak_positive") & available
    )
    result["association_direction"] = value[columns["direction"]].astype(str).where(
        available, "unavailable"
    )
    result["association_available"] = available.astype(bool)
    result["association_unavailable_reason"] = value.unavailable_reason.astype(str).where(
        ~available, ""
    )
    return result.sort_values(list(KEYS), kind="stable").reset_index(drop=True)


def e1_coexpression_frame(
    frame: pd.DataFrame,
    *,
    expected_fold: int,
) -> pd.DataFrame:
    """Select signed E1 edges without falling back to confounded E0 rows."""

    required = {
        "cancer_id",
        "fold_id",
        "lncrna_id",
        "gene_id",
        "rho_E1",
        "present_E1",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Local-CNV G0 table lacks columns: {missing}")
    value = frame.copy().reset_index(drop=True)
    folds = pd.to_numeric(value.fold_id, errors="coerce")
    if folds.isna().any() or not folds.eq(int(expected_fold)).all():
        raise RuntimeError("Local-CNV G0 fold identity drift")
    selected = value.loc[value.present_E1.astype(bool)].copy()
    selected["rho"] = pd.to_numeric(selected.rho_E1, errors="coerce")
    if selected.rho.isna().any() or selected.rho.eq(0).any():
        raise RuntimeError("Selected E1 edge lacks a finite non-zero signed effect")
    if selected.duplicated(["cancer_id", "lncrna_id", "gene_id"]).any():
        raise RuntimeError("Selected E1 edge keys are not unique")
    result = selected[["cancer_id", "lncrna_id", "gene_id", "rho"]].astype(
        {"cancer_id": str, "lncrna_id": str, "gene_id": str}
    )
    result["source_split"] = "train"
    result["edge_outer_fold"] = int(expected_fold)
    return result.sort_values(
        ["cancer_id", "lncrna_id", "gene_id"], kind="stable"
    ).reset_index(drop=True)


__all__ = [
    "KEYS",
    "adjusted_association_frame",
    "e1_coexpression_frame",
    "sha256_file",
    "validate_local_cnv_audit",
    "validate_validation_a1_authority",
]
