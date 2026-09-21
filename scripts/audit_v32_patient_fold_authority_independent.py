#!/usr/bin/env python3
"""Independently audit the immutable V3.2 patient-first fold authority.

The implementation intentionally does not import either the authority builder
or the historical patient-fold helper.  It reloads explicit source patient IDs,
reimplements assignment and hashing, and compares both published tables exactly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import pandas as pd


AUTHORITY_FORMAT = "CANCERLNCATLAS_V32_PATIENT_FIRST_FOLD_AUTHORITY_V1"
RECEIPT_FORMAT = "CANCERLNCATLAS_V32_PATIENT_FIRST_FOLD_RECEIPT_V1"
SUCCESS_FORMAT = "CANCERLNCATLAS_V32_PATIENT_FIRST_FOLD_SUCCESS_V1"
AUDIT_FORMAT = "CANCERLNCATLAS_V32_PATIENT_FIRST_FOLD_INDEPENDENT_AUDIT_V1"
AUDIT_SUCCESS_FORMAT = "CANCERLNCATLAS_V32_PATIENT_FIRST_FOLD_INDEPENDENT_AUDIT_SUCCESS_V1"
DEFAULT_SEED = 20260726
N_FOLDS = 5
TCGA_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)


class IndependentAuditError(RuntimeError):
    """Raised on any drift from the independently reconstructed authority."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _table_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    for column in columns:
        selected[column] = selected[column].astype(str)
    selected = selected.sort_values(list(columns), kind="stable")
    payload = "\n".join(
        "\t".join(map(str, row))
        for row in selected.itertuples(index=False, name=None)
    ) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _inventory(activity_root: Path) -> tuple[list[dict[str, Any]], str]:
    if not activity_root.is_dir() or activity_root.is_symlink():
        raise IndependentAuditError("Unsafe formal activity root")
    records = []
    for path in sorted(
        activity_root.rglob("*.parquet"),
        key=lambda item: item.relative_to(activity_root).as_posix(),
    ):
        if path.is_symlink() or not path.is_file():
            raise IndependentAuditError(f"Unsafe activity file: {path}")
        records.append(
            {
                "relative_path": path.relative_to(activity_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise IndependentAuditError("No source Parquet files")
    return records, _json_sha256(records)


def _load_source(activity_root: Path, memory_limit: str) -> pd.DataFrame:
    pattern = (activity_root / "**" / "*.parquet").as_posix()
    connection = duckdb.connect()
    connection.execute("SET threads=1")
    connection.execute("SET memory_limit=?", [memory_limit])
    try:
        described = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [pattern]
        ).fetchall()
        columns = {str(row[0]) for row in described}
        missing = sorted({"cancer_id", "sample_id", "patient_id"} - columns)
        if missing:
            raise IndependentAuditError(f"Source lacks explicit columns: {missing}")
        frame = connection.execute(
            """
            SELECT upper(trim(cast(cancer_id AS VARCHAR))) AS cancer_id,
                   trim(cast(sample_id AS VARCHAR)) AS sample_id,
                   trim(cast(patient_id AS VARCHAR)) AS patient_id
            FROM read_parquet(?)
            GROUP BY ALL
            ORDER BY cancer_id, patient_id, sample_id
            """,
            [pattern],
        ).fetchdf()
    finally:
        connection.close()
    if frame.empty or frame.isna().any().any():
        raise IndependentAuditError("Source explicit patient map is empty or null")
    if frame[["cancer_id", "sample_id", "patient_id"]].eq("").any().any():
        raise IndependentAuditError("Source explicit patient map has empty IDs")
    if frame.groupby(["cancer_id", "sample_id"], observed=True).patient_id.nunique().gt(1).any():
        raise IndependentAuditError("Source sample maps to multiple patients")
    if frame.groupby("sample_id", observed=True).cancer_id.nunique().gt(1).any():
        raise IndependentAuditError("Source sample crosses cancers")
    if frame.groupby("patient_id", observed=True).cancer_id.nunique().gt(1).any():
        raise IndependentAuditError("Source patient crosses cancers")
    return frame


def _reconstruct(source: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = source[["cancer_id", "patient_id"]].drop_duplicates().copy()
    patients["_order"] = [
        hashlib.sha256(f"{seed}|{cancer}|{patient}".encode("utf-8")).hexdigest()
        for cancer, patient in patients.itertuples(index=False, name=None)
    ]
    patients = patients.sort_values(
        ["cancer_id", "_order", "patient_id"], kind="stable"
    )
    patients["patient_fold_id"] = (
        patients.groupby("cancer_id", observed=True).cumcount() % N_FOLDS
    ).astype("int8")
    patients["fold_seed"] = int(seed)
    patients = patients.drop(columns="_order").sort_values(
        ["cancer_id", "patient_id"], kind="stable"
    ).reset_index(drop=True)
    samples = source.merge(
        patients,
        on=["cancer_id", "patient_id"],
        how="left",
        validate="many_to_one",
    ).sort_values(["cancer_id", "patient_id", "sample_id"], kind="stable")
    return patients, samples.reset_index(drop=True)


def _read_table(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.is_file() or path.is_symlink():
        raise IndependentAuditError(f"Missing or unsafe authority table: {path}")
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype={"cancer_id": "string", "sample_id": "string", "patient_id": "string"},
    )
    if list(frame.columns) != list(columns):
        raise IndependentAuditError(
            f"Authority schema drift for {path.name}: {list(frame.columns)}"
        )
    if frame.empty or frame.isna().any().any():
        raise IndependentAuditError(f"Authority table empty/null: {path.name}")
    for column in ("cancer_id", "sample_id", "patient_id"):
        if column in frame:
            frame[column] = frame[column].astype(str).str.strip()
            if frame[column].eq("").any():
                raise IndependentAuditError(f"Authority has empty {column}")
    frame["cancer_id"] = frame.cancer_id.str.upper()
    frame["patient_fold_id"] = pd.to_numeric(frame.patient_fold_id, errors="raise").astype(int)
    frame["fold_seed"] = pd.to_numeric(frame.fold_seed, errors="raise").astype(int)
    return frame


def _require_artifact(record: Any, path: Path) -> None:
    if not isinstance(record, Mapping):
        raise IndependentAuditError(f"Missing artifact record for {path.name}")
    if (
        record.get("filename") != path.name
        or int(record.get("bytes", -1)) != path.stat().st_size
        or record.get("sha256") != _sha256_file(path)
    ):
        raise IndependentAuditError(f"Artifact hash/size drift: {path.name}")


def audit_authority(
    *,
    activity_root: str | Path,
    authority_root: str | Path,
    output_root: str | Path,
    seed: int = DEFAULT_SEED,
    expected_patients: int = 10_432,
    expected_cancers: Sequence[str] = TCGA_CANCERS,
    memory_limit: str = "2GB",
) -> dict[str, Any]:
    source_root = Path(activity_root).resolve()
    authority = Path(authority_root).resolve()
    output = Path(output_root).resolve()
    temporary = output.with_name(f".{output.name}.tmp")
    if output.exists():
        raise FileExistsError(f"Independent audit refuses output reuse: {output}")
    if temporary.exists():
        raise FileExistsError(f"Independent audit temporary directory exists: {temporary}")
    if not authority.is_dir() or authority.is_symlink():
        raise IndependentAuditError("Authority root is missing or unsafe")

    patient_path = authority / "PATIENT_FOLD_AUTHORITY.tsv"
    sample_path = authority / "SAMPLE_PATIENT_FOLD_MAP.tsv"
    inventory_path = authority / "SOURCE_INVENTORY.json"
    receipt_path = authority / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"
    success_path = authority / "SUCCESS.json"
    for path in (patient_path, sample_path, inventory_path, receipt_path, success_path):
        if not path.is_file() or path.is_symlink():
            raise IndependentAuditError(f"Required authority artifact missing/unsafe: {path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        success = json.loads(success_path.read_text(encoding="utf-8"))
        published_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndependentAuditError("Authority JSON is unreadable") from exc

    required_cancers = sorted(str(value).upper() for value in expected_cancers)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("format") != RECEIPT_FORMAT
        or receipt.get("authority_format") != AUTHORITY_FORMAT
        or receipt.get("status") != "PASS_EXPLICIT_PATIENT_FIRST_5FOLD_AUTHORITY"
        or int(receipt.get("seed", -1)) != int(seed)
        or int(receipt.get("n_folds", -1)) != N_FOLDS
        or receipt.get("expected_cancers") != required_cancers
        or int(receipt.get("expected_patients", -1)) != int(expected_patients)
    ):
        raise IndependentAuditError("Receipt identity/scope/seed contract drift")
    gates = receipt.get("gates", {})
    required_true_gates = {
        "explicit_patient_id_source_required",
        "sample_barcode_patient_fallback_forbidden",
        "all_33_cancers_present",
        "all_five_folds_per_cancer",
        "patient_cross_cancer_count_zero",
        "patient_cross_fold_count_zero",
        "output_reuse_forbidden",
    }
    if not isinstance(gates, Mapping) or any(gates.get(key) is not True for key in required_true_gates):
        raise IndependentAuditError("Receipt has an open patient-authority gate")
    implementation = receipt.get("implementation", {})
    if (
        not isinstance(implementation, Mapping)
        or implementation.get("old_prepare_v32_formal_executed") is not False
        or implementation.get("assignment_algorithm")
        != "SHA256(seed|cancer_id|explicit_patient_id)_SORT_THEN_ROUND_ROBIN"
    ):
        raise IndependentAuditError("Receipt implementation contract drift")
    for path_key, sha_key in (
        ("builder_module", "builder_module_sha256"),
        ("launcher", "launcher_sha256"),
    ):
        code_path = Path(str(implementation.get(path_key, ""))).resolve()
        if not code_path.is_file() or implementation.get(sha_key) != _sha256_file(code_path):
            raise IndependentAuditError(f"Bound implementation changed: {path_key}")

    _require_artifact(receipt.get("artifacts", {}).get("patient_fold_authority"), patient_path)
    _require_artifact(receipt.get("artifacts", {}).get("sample_patient_fold_map"), sample_path)
    _require_artifact(receipt.get("source_inventory"), inventory_path)
    if (
        not isinstance(success, Mapping)
        or success.get("format") != SUCCESS_FORMAT
        or success.get("status") != "SUCCESS"
        or success.get("formal_training_started") is not False
        or success.get("production_8260_touched") is not False
    ):
        raise IndependentAuditError("Authority SUCCESS marker contract drift")
    for key, path in (
        ("receipt", receipt_path),
        ("patient_fold_authority", patient_path),
        ("sample_patient_fold_map", sample_path),
        ("source_inventory", inventory_path),
    ):
        _require_artifact(success.get(key), path)

    inventory, tree_sha256 = _inventory(source_root)
    expected_inventory = {
        "format": "CANCERLNCATLAS_V32_FORMAL_ACTIVITY_INVENTORY_V1",
        "source_root": str(source_root),
        "files": len(inventory),
        "bytes": sum(int(record["bytes"]) for record in inventory),
        "tree_sha256": tree_sha256,
        "records": inventory,
    }
    if published_inventory != expected_inventory:
        raise IndependentAuditError("Published source inventory differs from current formal activity")
    if (
        receipt.get("source_inventory", {}).get("activity_tree_sha256") != tree_sha256
        or int(receipt.get("source_inventory", {}).get("activity_files", -1)) != len(inventory)
    ):
        raise IndependentAuditError("Receipt source inventory binding drift")

    source = _load_source(source_root, memory_limit)
    expected_patient, expected_sample = _reconstruct(source, int(seed))
    patient_columns = ["cancer_id", "patient_id", "patient_fold_id", "fold_seed"]
    sample_columns = ["cancer_id", "sample_id", "patient_id", "patient_fold_id", "fold_seed"]
    actual_patient = _read_table(patient_path, patient_columns)
    actual_sample = _read_table(sample_path, sample_columns)
    observed_cancers = sorted(actual_patient.cancer_id.unique().tolist())
    if observed_cancers != required_cancers or len(actual_patient) != int(expected_patients):
        raise IndependentAuditError("Published cancer/patient scope drift")
    if set(actual_patient.patient_fold_id) != set(range(N_FOLDS)):
        raise IndependentAuditError("Published fold domain is not exactly 0..4")
    if set(actual_patient.fold_seed) != {int(seed)} or set(actual_sample.fold_seed) != {int(seed)}:
        raise IndependentAuditError("Published fold seed drift")
    if actual_patient.duplicated(["cancer_id", "patient_id"]).any():
        raise IndependentAuditError("Published patient authority duplicates patient keys")
    if actual_sample.duplicated(["cancer_id", "sample_id"]).any():
        raise IndependentAuditError("Published sample map duplicates sample keys")
    if actual_patient.groupby("patient_id", observed=True).cancer_id.nunique().gt(1).any():
        raise IndependentAuditError("Published patient crosses cancers")
    if actual_sample.groupby(["cancer_id", "patient_id"], observed=True).patient_fold_id.nunique().gt(1).any():
        raise IndependentAuditError("Published patient crosses folds")
    per_cancer_folds = actual_patient.groupby("cancer_id", observed=True).patient_fold_id.nunique()
    if not per_cancer_folds.eq(N_FOLDS).all():
        raise IndependentAuditError("A cancer is missing a fold")
    pd.testing.assert_frame_equal(
        actual_patient.sort_values(patient_columns).reset_index(drop=True),
        expected_patient.astype({"patient_fold_id": int, "fold_seed": int})
        .sort_values(patient_columns).reset_index(drop=True),
        check_dtype=False,
        check_like=False,
    )
    pd.testing.assert_frame_equal(
        actual_sample.sort_values(sample_columns).reset_index(drop=True),
        expected_sample.astype({"patient_fold_id": int, "fold_seed": int})
        .sort_values(sample_columns).reset_index(drop=True),
        check_dtype=False,
        check_like=False,
    )
    patient_logical = _table_sha256(actual_patient, patient_columns)
    sample_logical = _table_sha256(actual_sample, sample_columns)
    if (
        patient_logical != receipt.get("logical_hashes", {}).get("patient_authority_sha256")
        or sample_logical != receipt.get("logical_hashes", {}).get("sample_patient_map_sha256")
    ):
        raise IndependentAuditError("Logical authority hash drift")

    audit = {
        "format": AUDIT_FORMAT,
        "status": "PASS_INDEPENDENT_EXPLICIT_PATIENT_FIRST_5FOLD_AUDIT",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "LOCAL_NEW_ARTIFACT_ONLY_NO_TRAINING_NO_SERVER_NO_8260",
        "authority_root": str(authority),
        "source_activity_root": str(source_root),
        "observed": {
            "cancers": len(observed_cancers),
            "patients": len(actual_patient),
            "samples": len(actual_sample),
            "source_files": len(inventory),
            "source_bytes": sum(int(record["bytes"]) for record in inventory),
            "patients_crossing_cancers": 0,
            "patients_crossing_folds": 0,
        },
        "contract": {
            "explicit_source_patient_id_reloaded": True,
            "sample_id_patient_derivation_used": False,
            "assignment_independently_reimplemented": True,
            "production_fold_helper_imported": False,
            "authority_builder_imported": False,
            "old_prepare_v32_formal_executed": False,
            "all_33_cancers_present": True,
            "all_five_folds_per_cancer": True,
            "exact_patient_table_match": True,
            "exact_sample_patient_table_match": True,
            "source_inventory_match": True,
            "code_hashes_match": True,
            "production_8260_touched": False,
        },
        "hashes": {
            "source_tree_sha256": tree_sha256,
            "receipt_sha256": _sha256_file(receipt_path),
            "patient_authority_file_sha256": _sha256_file(patient_path),
            "sample_patient_map_file_sha256": _sha256_file(sample_path),
            "patient_authority_logical_sha256": patient_logical,
            "sample_patient_map_logical_sha256": sample_logical,
            "independent_auditor_sha256": _sha256_file(Path(__file__).resolve()),
        },
    }
    temporary.mkdir(parents=True)
    audit_path = temporary / "AUDIT.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_success = {
        "format": AUDIT_SUCCESS_FORMAT,
        "status": "SUCCESS",
        "audit": {
            "filename": audit_path.name,
            "bytes": audit_path.stat().st_size,
            "sha256": _sha256_file(audit_path),
        },
        "formal_training_started": False,
        "server_accessed": False,
        "production_8260_touched": False,
    }
    (temporary / "SUCCESS.json").write_text(
        json.dumps(audit_success, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {**audit, "output_root": str(output)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-root", required=True)
    parser.add_argument("--authority-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-patients", type=int, default=10_432)
    parser.add_argument("--memory-limit", default="2GB")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit = audit_authority(
        activity_root=args.activity_root,
        authority_root=args.authority_root,
        output_root=args.output_root,
        seed=args.seed,
        expected_patients=args.expected_patients,
        memory_limit=args.memory_limit,
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output_root": audit["output_root"],
                **audit["observed"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
