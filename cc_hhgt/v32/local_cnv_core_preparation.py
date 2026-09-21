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
    root: str | Path, *, expected_success_sha256: str | None = None
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
    if decision.get("decision") != "LOCAL_CNV_CORE_RETRAIN_REQUIRED":
        raise RuntimeError("Local-CNV authority does not require corrected core inputs")
    return {
        "root": str(audit_root),
        "success_sha256": success_sha,
        "independent_audit_sha256": sha256_file(independent_path),
        "decision_sha256": sha256_file(decision_path),
        "decision": decision["decision"],
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
    gates = success.get("gates", {})
    if (
        gates.get("validation_label_only") is not True
        or gates.get("designated_oof_partition_read") is not False
        or gates.get("sealed_test_accessed") is not False
        or gates.get("mutation_used") is not False
        or gates.get("candidate_closure") is not True
        or gates.get("typed_unavailable_preserved") is not True
    ):
        raise RuntimeError("Validation-A1 governance contract failed")
    return {
        "root": str(authority_root),
        "success_sha256": success_sha,
        "independent_audit_sha256": sha256_file(independent_path),
        "candidate_sha256": success.get("candidate_sha256"),
        "patient_map_sha256": success.get("patient_map_sha256"),
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
    label_class = labels.where(available, "unavailable")
    result = value[list(KEYS)].astype(str).copy()
    result["discovery_effect"] = rho
    result["discovery_pvalue"] = pvalue
    result["discovery_fdr"] = fdr
    result["association_n_samples"] = n_samples
    result["residual_design_rank"] = np.where(available, 2, 0).astype("int16")
    result["correlation_df"] = np.where(
        available, np.maximum(n_samples.to_numpy() - 3, 0), 0
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
