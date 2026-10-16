#!/usr/bin/env python3
"""Independently audit the five-fold local-CNV-adjusted validation labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cc_hhgt.v32.local_cnv_confounding import (
    COVARIATE_CONTRACT_VERSION,
    DEFAULT_COVARIATES,
)


CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)
KEYS = ["cancer_id", "lncrna_id", "pathway_id"]
VALID_LABELS = {"strong_positive", "weak_positive", "unlabeled"}
EXPECTED_COVARIATES = DEFAULT_COVARIATES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_hash(values: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in values).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def canonical_patient(value: Any) -> str:
    text = str(value).strip().upper().replace(".", "-")
    parts = text.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 and parts[0] == "TCGA" else text


def _covariates_used(value: Any) -> list[str]:
    """Decode the per-part contract column and reject silent omissions."""
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Validation-A1 covariates_used is not valid JSON") from exc
    if list(decoded or []) != list(EXPECTED_COVARIATES):
        raise RuntimeError(
            "Validation-A1 covariates_used does not match the shared contract: "
            f"{decoded!r}"
        )
    return list(decoded)


def main() -> int:
    from cc_hhgt.v32.cnv_input_sources import require_host
    require_host()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--patient-map", required=True, type=Path)
    parser.add_argument("--local-cnv-audit-success", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    from cc_hhgt.v32.local_cnv_sources import preflight
    preflight(root,"FINALIZE","validation_finalizer")
    if (root / "SUCCESS.json").exists() or (root / "INDEPENDENT_AUDIT.json").exists():
        raise FileExistsError("Validation-A1 terminal authority reuse is forbidden")

    candidates = pd.read_parquet(args.candidates, columns=KEYS).astype(str)
    if (
        len(candidates) != 3_300_000
        or candidates.duplicated(KEYS).any()
        or set(candidates.cancer_id) != set(CANCERS)
        or not candidates.groupby("cancer_id", observed=True).size().eq(100_000).all()
    ):
        raise RuntimeError("Formal candidate authority closure failed")
    candidate_sha = sha256_file(args.candidates)
    candidate_by_cancer = {
        cancer: candidates.loc[candidates.cancer_id.eq(cancer), KEYS]
        .sort_values(KEYS, kind="stable").reset_index(drop=True)
        for cancer in CANCERS
    }
    patient_map = pd.read_csv(args.patient_map, sep="\t")
    required_patient = {"cancer_id", "patient_id", "patient_fold_id"}
    if missing := sorted(required_patient - set(patient_map.columns)):
        raise RuntimeError(f"Patient authority lacks columns: {missing}")
    patient_map["cancer_id"] = patient_map.cancer_id.astype(str)
    patient_map["patient_id"] = patient_map.patient_id.map(canonical_patient)
    patient_map["patient_fold_id"] = pd.to_numeric(
        patient_map.patient_fold_id, errors="raise"
    ).astype(int)
    patient_map = patient_map.drop_duplicates(
        ["cancer_id", "patient_id", "patient_fold_id"]
    )
    if patient_map.duplicated(["cancer_id", "patient_id"]).any():
        raise RuntimeError("Patient authority assigns one patient to multiple folds")

    local_success = json.loads(args.local_cnv_audit_success.read_text(encoding="utf-8"))
    if (
        local_success.get("status") != "SUCCESS"
        or local_success.get("decision") != "LOCAL_CNV_CORE_RETRAIN_REQUIRED"
    ):
        raise RuntimeError("Pinned local-CNV audit is not a passed retrain authority")
    local_success_sha = sha256_file(args.local_cnv_audit_success)

    records = []
    total_available = 0
    total_unavailable = 0
    for cancer in CANCERS:
        receipt_path = root / "receipts" / f"{cancer}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        from cc_hhgt.v32.local_cnv_sources import validate
        if "consumed_sources" not in receipt:raise RuntimeError("Validation-A1 receipt lacks actual consumed sources")
        validate(receipt["consumed_sources"],receipt_path.parent)
        if (
            receipt.get("status") != "PASS"
            or receipt.get("cancer_id") != cancer
            or receipt.get("candidate_sha256") != candidate_sha
            or receipt.get("patient_map_sha256") != sha256_file(args.patient_map)
            or receipt.get("local_cnv_authority", {}).get("success_sha256")
            != local_success_sha
            or receipt.get("gates", {}).get("validation_label_only") is not True
            or receipt.get("gates", {}).get("designated_oof_partition_read") is not False
            or receipt.get("gates", {}).get("mutation_used") is not False
            or receipt.get("covariate_contract_version") != COVARIATE_CONTRACT_VERSION
            or receipt.get("covariates", {}).get("declared") != list(EXPECTED_COVARIATES)
        ):
            raise RuntimeError(f"Invalid validation-A1 receipt: {cancer}")
        outputs = receipt.get("outputs", [])
        if {int(item.get("fold", -1)) for item in outputs} != set(range(5)):
            raise RuntimeError(f"Validation-A1 receipt lacks five folds: {cancer}")
        for record in sorted(outputs, key=lambda item: int(item["fold"])):
            fold = int(record["fold"])
            validation_fold = (fold + 1) % 5
            expected_patients = sorted(
                patient_map.loc[
                    patient_map.cancer_id.eq(cancer)
                    & patient_map.patient_fold_id.eq(validation_fold),
                    "patient_id",
                ].astype(str)
            )
            path = Path(record["path"]).resolve()
            expected_path = (
                root / "validation_associations" / f"cancer={cancer}"
                / f"fold={fold}" / "part.parquet"
            ).resolve()
            if path != expected_path or sha256_file(path) != record.get("sha256"):
                raise RuntimeError(f"Validation-A1 output path/hash drift: {cancer}/{fold}")
            frame = pd.read_parquet(path)
            required = set(KEYS) | {
                "fold_id", "split_role", "validation_patient_fold_id",
                "validation_patient_sha256", "rho_A1_validation",
                "p_A1_validation", "fdr_official_A1_validation",
                "label_A1_validation", "direction_A1_validation",
                "availability", "unavailable_reason", "sealed_test_accessed",
                "covariates_used",
            }
            if missing := sorted(required - set(frame.columns)):
                raise RuntimeError(f"Validation-A1 schema drift {cancer}/{fold}: {missing}")
            used_contracts = frame["covariates_used"].map(
                lambda value: tuple(_covariates_used(value))
            )
            if used_contracts.nunique() != 1:
                raise RuntimeError(f"Validation-A1 covariate contract varies within {cancer}/{fold}")
            keys = frame[KEYS].astype(str).sort_values(KEYS, kind="stable").reset_index(drop=True)
            if not keys.equals(candidate_by_cancer[cancer]):
                raise RuntimeError(f"Candidate key closure failed: {cancer}/{fold}")
            if (
                not pd.to_numeric(frame.fold_id).eq(fold).all()
                or not frame.split_role.astype(str).eq("validation_label_only").all()
                or not pd.to_numeric(frame.validation_patient_fold_id).eq(validation_fold).all()
                or not frame.validation_patient_sha256.astype(str).eq(patient_hash(expected_patients)).all()
                or frame.sealed_test_accessed.astype(bool).any()
            ):
                raise RuntimeError(f"Validation role/firewall drift: {cancer}/{fold}")
            available = frame.availability.astype(str).eq("available")
            unavailable = frame.availability.astype(str).eq("typed_unavailable")
            numeric = frame[[
                "rho_A1_validation", "p_A1_validation", "fdr_official_A1_validation"
            ]].apply(pd.to_numeric, errors="coerce")
            if (
                not (available | unavailable).all()
                or not np.isfinite(numeric.loc[available].to_numpy(float)).all()
                or numeric.loc[unavailable].notna().any().any()
                or frame.loc[unavailable, "unavailable_reason"].astype(str).str.strip().eq("").any()
                or not set(frame.loc[available, "label_A1_validation"].astype(str)).issubset(VALID_LABELS)
            ):
                raise RuntimeError(f"Validation-A1 null semantics failed: {cancer}/{fold}")
            observed_available = int(available.sum())
            observed_unavailable = int(unavailable.sum())
            if (
                observed_available != int(record.get("available", -1))
                or observed_unavailable != int(record.get("typed_unavailable", -1))
                or len(frame) != 100_000
            ):
                raise RuntimeError(f"Validation-A1 receipt count drift: {cancer}/{fold}")
            total_available += observed_available
            total_unavailable += observed_unavailable
            records.append({
                "cancer_id": cancer,
                "fold": fold,
                "validation_fold": validation_fold,
                "rows": len(frame),
                "available": observed_available,
                "typed_unavailable": observed_unavailable,
                "sha256": record["sha256"],
            })

    audit = {
        "format": "CC_HHGT_V3_2_LOCAL_CNV_VALIDATION_A1_INDEPENDENT_AUDIT_V1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": candidate_sha,
        "patient_map_sha256": sha256_file(args.patient_map),
        "local_cnv_audit_success_sha256": local_success_sha,
        "cancers": len(CANCERS),
        "fold_partitions": len(records),
        "rows": total_available + total_unavailable,
        "available": total_available,
        "typed_unavailable": total_unavailable,
        "partitions": records,
        "gates": {
            "five_folds": len(records) == 165,
            "candidate_closure": total_available + total_unavailable == 16_500_000,
            "validation_label_only": True,
            "patient_roles_recomputed": True,
            "typed_unavailable_preserved": True,
            "shared_covariate_contract": COVARIATE_CONTRACT_VERSION,
            "covariates_declared": list(EXPECTED_COVARIATES),
            "designated_oof_partition_read": False,
            "sealed_test_accessed": False,
            "mutation_used": False,
        },
    }
    atomic_json(root / "INDEPENDENT_AUDIT.json", audit)
    success = {
        "format": "CC_HHGT_V3_2_LOCAL_CNV_VALIDATION_A1_SUCCESS_V1",
        "status": "PASS",
        "covariate_contract_version": COVARIATE_CONTRACT_VERSION,
        "covariate_contract": {"version": COVARIATE_CONTRACT_VERSION},
        "covariates": {"declared": list(EXPECTED_COVARIATES)},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": candidate_sha,
        "patient_map_sha256": sha256_file(args.patient_map),
        "local_cnv_audit_success_sha256": local_success_sha,
        "independent_audit_sha256": sha256_file(root / "INDEPENDENT_AUDIT.json"),
        "gates": audit["gates"],
    }
    atomic_json(root / "SUCCESS.json", success)
    print(json.dumps({
        "status": "PASS",
        "partitions": 165,
        "rows": total_available + total_unavailable,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
