"""Audited materialization of official TCGA ATAC into patient-gene matrices."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .atac_training import (
    ANALYSIS_VERSION,
    FORMAL_CANCERS,
    MATERIALIZATION_FORMAT,
    RAW_COVERED_CANCERS,
    RAW_GAP_CANCERS,
    AtacTrainingError,
    _assert_nonpredictive_columns,
    _assert_nonpredictive_path,
    _gene_id,
    _patient_id,
    file_sha256,
    normalise_membership,
    normalise_patient_folds,
)


INPUT_CONTRACT_FORMAT = "CC_HHGT_V3_2_ATAC_AUTHORITATIVE_INPUT_CONTRACT_V1"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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


def _verify_input_contract(
    contract_path: Path, paths: Mapping[str, Path], observed_hashes: Mapping[str, str]
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("format") != INPUT_CONTRACT_FORMAT:
        raise AtacTrainingError("ATAC authoritative input contract format is invalid")
    records = contract.get("files")
    if not isinstance(records, list):
        raise AtacTrainingError("ATAC authoritative input contract lacks files")
    by_role = {str(record.get("role")): record for record in records}
    if set(by_role) != set(paths):
        raise AtacTrainingError("ATAC input contract roles do not match formal inputs")
    for role, path in paths.items():
        record = by_role[role]
        if str(record.get("basename")) != path.name:
            raise AtacTrainingError(f"ATAC {role} basename differs from frozen contract")
        if int(record.get("bytes", -1)) != path.stat().st_size:
            raise AtacTrainingError(f"ATAC {role} byte size differs from frozen contract")
        if str(record.get("sha256")) != observed_hashes[role]:
            raise AtacTrainingError(f"ATAC {role} SHA256 differs from frozen contract")
    return contract


def _candidate_gene_ids(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    genes: set[str] = set()
    parquet = pq.ParquetFile(path)
    if "lncrna_id" not in parquet.schema_arrow.names:
        raise AtacTrainingError("Candidate universe lacks lncrna_id")
    _assert_nonpredictive_columns(parquet.schema_arrow.names, "Candidate universe")
    for batch in parquet.iter_batches(columns=["lncrna_id"], batch_size=65_536):
        for value in batch.column(0).to_pylist():
            if value is not None:
                genes.add(_gene_id(value))
    if not genes:
        raise AtacTrainingError("Candidate universe has no lncRNA genes")
    return genes


def run_atac_materialization(
    *,
    candidates_path: str | Path,
    patient_folds_path: str | Path,
    pathway_membership_path: str | Path,
    matrix_path: str | Path,
    mapping_path: str | Path,
    peak_set_path: str | Path,
    promoters_path: str | Path,
    extractor_script_path: str | Path,
    input_contract_path: str | Path,
    output_root: str | Path,
    rscript_bin: str = "/usr/bin/Rscript",
) -> dict[str, Any]:
    """Hash-gate raw inputs, run the R aggregation, and publish Parquet matrices."""

    paths = {
        "candidates": Path(candidates_path).resolve(),
        "patient_folds": Path(patient_folds_path).resolve(),
        "pathway_membership": Path(pathway_membership_path).resolve(),
        "normalized_atac_matrix": Path(matrix_path).resolve(),
        "identifier_mapping": Path(mapping_path).resolve(),
        "peak_set": Path(peak_set_path).resolve(),
        "promoters": Path(promoters_path).resolve(),
    }
    extractor = Path(extractor_script_path).resolve()
    contract_path = Path(input_contract_path).resolve()
    for role, path in paths.items():
        _assert_nonpredictive_path(path, role)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    for path in (extractor, contract_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    output = Path(output_root).resolve()
    if output.exists():
        raise AtacTrainingError(f"ATAC materializer refuses output reuse: {output}")

    # Full SHA256 of the 706 MB RDS and every other authority is deliberately
    # completed before the output directory is created.
    observed_hashes = {role: file_sha256(path) for role, path in paths.items()}
    contract = _verify_input_contract(contract_path, paths, observed_hashes)
    folds = normalise_patient_folds(pd.read_csv(paths["patient_folds"], sep="\t"))
    if set(folds.cancer_id.unique()) != set(FORMAL_CANCERS):
        raise AtacTrainingError("ATAC folds are not the exact 33-cancer scope")
    membership = normalise_membership(pd.read_parquet(paths["pathway_membership"]))
    required_genes = _candidate_gene_ids(paths["candidates"]) | set(
        membership.gene_id.astype(str)
    )

    output.mkdir(parents=True)
    intermediate = output / "intermediate"
    r_output = intermediate / "r_gene_accessibility"
    intermediate.mkdir()
    r_output.mkdir()
    genes_path = intermediate / "required_gene_ids.tsv.gz"
    with gzip.open(genes_path, "wt", encoding="utf-8", newline="") as handle:
        pd.DataFrame({"gene_id": sorted(required_genes)}).to_csv(
            handle, sep="\t", index=False, lineterminator="\n"
        )
    fold_path = intermediate / "canonical_patient_folds.tsv.gz"
    with gzip.open(fold_path, "wt", encoding="utf-8", newline="") as handle:
        folds.to_csv(handle, sep="\t", index=False, lineterminator="\n")

    command = [
        str(rscript_bin),
        str(extractor),
        str(paths["normalized_atac_matrix"]),
        str(paths["identifier_mapping"]),
        str(paths["peak_set"]),
        str(paths["promoters"]),
        str(genes_path),
        str(fold_path),
        str(r_output),
        "CC_HHGT_V3_2_ATAC_MATERIALIZE_V1",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": environment.get("OMP_NUM_THREADS", "16"),
            "OPENBLAS_NUM_THREADS": environment.get("OPENBLAS_NUM_THREADS", "16"),
        }
    )
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    log_path = output / "ATAC_MATERIALIZATION.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise AtacTrainingError(
            f"ATAC R materialization failed ({completed.returncode}); see {log_path}"
        )
    r_audit_path = r_output / "ATAC_GENE_ACCESSIBILITY_AUDIT.json"
    if not r_audit_path.is_file():
        raise AtacTrainingError("ATAC R materializer did not write its audit")
    r_audit = json.loads(r_audit_path.read_text(encoding="utf-8"))
    if r_audit.get("status") != "PASS":
        raise AtacTrainingError("ATAC R materialization audit is not PASS")
    if tuple(sorted(r_audit.get("covered_cancers", []))) != tuple(
        sorted(RAW_COVERED_CANCERS)
    ):
        raise AtacTrainingError("ATAC R materializer did not recover exact 23-cancer coverage")
    if tuple(sorted(r_audit.get("raw_gap_cancers", []))) != tuple(sorted(RAW_GAP_CANCERS)):
        raise AtacTrainingError("ATAC R materializer raw-gap list differs from exact typed 10")

    matrix_root = output / "gene_accessibility"
    records: list[dict[str, Any]] = []
    folds_by_cancer = {
        cancer: set(group.patient_id.astype(str))
        for cancer, group in folds.groupby("cancer_id", observed=True, sort=False)
    }
    for cancer in RAW_COVERED_CANCERS:
        source = r_output / f"{cancer}.gene_accessibility.tsv.gz"
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(source)
        frame = pd.read_csv(source, sep="\t", compression="gzip")
        if "gene_id" not in frame.columns or frame.empty:
            raise AtacTrainingError(f"ATAC {cancer} R matrix is empty/malformed")
        frame["gene_id"] = frame.gene_id.map(_gene_id)
        if frame.gene_id.duplicated().any():
            raise AtacTrainingError(f"ATAC {cancer} R matrix has duplicate genes")
        patients = [_patient_id(column) for column in frame.columns if column != "gene_id"]
        if len(patients) != len(set(patients)) or not patients:
            raise AtacTrainingError(f"ATAC {cancer} patient columns are empty/duplicated")
        if not set(patients).issubset(folds_by_cancer[cancer]):
            raise AtacTrainingError(f"ATAC {cancer} R matrix has non-canonical patients")
        numeric = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=np.float32)).all():
            raise AtacTrainingError(f"ATAC {cancer} R matrix contains non-finite values")
        for column in frame.columns[1:]:
            frame[column] = numeric[column].astype(np.float32)
        destination = matrix_root / f"cancer_id={cancer}" / "part-0.parquet"
        _atomic_parquet(frame, destination)
        records.append(
            {
                "cancer_id": cancer,
                "path": str(destination),
                "sha256": file_sha256(destination),
                "bytes": destination.stat().st_size,
                "genes": int(len(frame)),
                "patients": int(len(patients)),
                "patient_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(patients)).encode("utf-8")
                ).hexdigest(),
                "r_intermediate_path": str(source),
                "r_intermediate_sha256": file_sha256(source),
            }
        )

    materialized_paths = [Path(record["path"]) for record in records]
    materialized_paths.extend(Path(record["r_intermediate_path"]) for record in records)
    materialized_paths.extend([log_path, r_audit_path, genes_path, fold_path])
    manifest_entries = [
        {
            "relative_path": str(path.relative_to(output)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(materialized_paths, key=lambda value: str(value))
    ]
    manifest_path = output / "MANIFEST.json"
    _atomic_json(
        manifest_path,
        {
            "format": "CC_HHGT_V3_2_ATAC_MATERIALIZATION_MANIFEST_V1",
            "entries": manifest_entries,
        },
    )
    success = {
        "format": MATERIALIZATION_FORMAT,
        "status": "SUCCESS",
        "analysis_version": ANALYSIS_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "covered_cancers": list(RAW_COVERED_CANCERS),
        "raw_gap_cancers": list(RAW_GAP_CANCERS),
        "records": records,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "input_contract_path": str(contract_path),
        "input_contract_sha256": file_sha256(contract_path),
        "input_contract_id": contract.get("contract_id"),
        "input_artifacts": [
            {
                "role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": observed_hashes[role],
                "raw_or_static": True,
                "historical_prediction": False,
                "historical_checkpoint": False,
            }
            for role, path in paths.items()
        ],
        "extractor_script_path": str(extractor),
        "extractor_script_sha256": file_sha256(extractor),
        "r_audit_path": str(r_audit_path),
        "r_audit_sha256": file_sha256(r_audit_path),
        "technical_replicate_policy": "MEAN_WITHIN_EXACT_ALIQUOT_ID",
        "multiple_aliquot_policy": (
            "EXPLICIT_EQUAL_ALIQUOT_WEIGHT_MEAN_WITHIN_PATIENT_AFTER_TECHNICAL_REPLICATE_MEAN"
        ),
        "canonical_patient_alignment_only": True,
        "old_predictions_used": False,
        "old_checkpoints_used": False,
        "old_rankings_used": False,
        "training_started": False,
        "missing_raw_cancer_assumed_zero": False,
        "success_written_last": True,
    }
    _atomic_json(output / "GENE_ACCESSIBILITY_SUCCESS.json", success)
    return success


__all__ = ["INPUT_CONTRACT_FORMAT", "run_atac_materialization"]
