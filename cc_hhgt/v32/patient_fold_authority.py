"""Immutable patient-first fold authority built from explicit source patient IDs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


AUTHORITY_FORMAT = "CANCERLNCATLAS_V32_PATIENT_FIRST_FOLD_AUTHORITY_V1"
RECEIPT_FORMAT = "CANCERLNCATLAS_V32_PATIENT_FIRST_FOLD_RECEIPT_V1"
SUCCESS_FORMAT = "CANCERLNCATLAS_V32_PATIENT_FIRST_FOLD_SUCCESS_V1"
DEFAULT_SEED = 20260726
N_FOLDS = 5
TCGA_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)

# Frozen consumer identity for the patient-first authority independently
# audited and deployed on 2026-08-29.  ``validate_patient_fold_authority``
# remains the general builder/auditor validator; every formal V3.2 consumer
# must enter through ``validate_frozen_v32_patient_fold_binding`` so that a
# different, structurally valid five-fold table cannot silently replace the
# authority used by the other heads.
FROZEN_V32_PATIENTS = 10_432
FROZEN_V32_SAMPLE_PATIENT_MAP_FILENAME = "SAMPLE_PATIENT_FOLD_MAP.tsv"
FROZEN_V32_RECEIPT_FILENAME = "PATIENT_FOLD_AUTHORITY_RECEIPT.json"
FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256 = (
    "e05c20085be532159bb6a51f200a27923975c3caa6a90910290e10a5b60e6253"
)
FROZEN_V32_RECEIPT_SHA256 = (
    "1ef32bda9d14d87997f3b81c34b8aacce172de390db5c29aee011ca73ba317e0"
)
FROZEN_V32_PATIENT_AUTHORITY_LOGICAL_SHA256 = (
    "3c8bee3de5af3c5886ce8ef748c14471b449db5bbf6da886522e8adc540fa527"
)
FROZEN_V32_SAMPLE_PATIENT_MAP_LOGICAL_SHA256 = (
    "9117526520f280c1de9127a6e890e26526e91398d993a817d9073279139c651f"
)
FROZEN_V32_BINDING_FORMAT = "CANCERLNCATLAS_V32_FROZEN_PATIENT_FOLD_BINDING_V1"
FORMAL_PREPARED_BINDING_FORMAT = (
    "CC_HHGT_V3_2_FORMAL_PREPARED_PATIENT_FOLD_BINDING_V1"
)


class PatientFoldAuthorityError(RuntimeError):
    """Raised when an explicit patient authority fails closed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _table_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    for column in columns:
        selected[column] = selected[column].astype(str)
    selected = selected.sort_values(list(columns), kind="stable")
    lines = ["\t".join(map(str, row)) for row in selected.itertuples(index=False, name=None)]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _assignment_order(seed: int, cancer_id: str, patient_id: str) -> str:
    return hashlib.sha256(f"{seed}|{cancer_id}|{patient_id}".encode("utf-8")).hexdigest()


def activity_inventory(activity_root: str | Path) -> tuple[list[dict[str, Any]], str]:
    root = Path(activity_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise PatientFoldAuthorityError(f"Unsafe activity root: {root}")
    files = sorted(root.rglob("*.parquet"), key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise PatientFoldAuthorityError("Formal pathway activity has no Parquet files")
    records: list[dict[str, Any]] = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise PatientFoldAuthorityError(f"Unsafe activity input: {path}")
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records, _json_sha256(records)


def _load_explicit_sample_patients(
    activity_root: Path, *, memory_limit: str
) -> pd.DataFrame:
    import duckdb

    pattern = (activity_root / "**" / "*.parquet").as_posix()
    connection = duckdb.connect()
    connection.execute("SET threads=1")
    connection.execute("SET memory_limit=?", [memory_limit])
    try:
        columns = {
            str(row[0])
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [pattern]
            ).fetchall()
        }
        required = {"cancer_id", "sample_id", "patient_id"}
        if missing := sorted(required - columns):
            raise PatientFoldAuthorityError(
                f"Formal pathway activity lacks explicit authority columns: {missing}"
            )
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
        raise PatientFoldAuthorityError("Explicit sample/patient authority is empty or null")
    if frame[["cancer_id", "sample_id", "patient_id"]].eq("").any().any():
        raise PatientFoldAuthorityError("Explicit sample/patient authority contains empty IDs")
    sample_patient = frame.groupby(["cancer_id", "sample_id"], observed=True).patient_id.nunique()
    if sample_patient.gt(1).any():
        raise PatientFoldAuthorityError("A cancer/sample maps to multiple explicit patient IDs")
    sample_cancer = frame.groupby("sample_id", observed=True).cancer_id.nunique()
    if sample_cancer.gt(1).any():
        raise PatientFoldAuthorityError("One sample occurs in multiple cancers")
    patient_cancer = frame.groupby("patient_id", observed=True).cancer_id.nunique()
    if patient_cancer.gt(1).any():
        raise PatientFoldAuthorityError("One explicit patient occurs in multiple cancers")
    return frame


def _build_assignments(
    samples: pd.DataFrame, *, seed: int, n_folds: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_folds != N_FOLDS:
        raise PatientFoldAuthorityError("V3.2 patient authority requires exactly five folds")
    patients = samples[["cancer_id", "patient_id"]].drop_duplicates().copy()
    counts = patients.groupby("cancer_id", observed=True).patient_id.nunique()
    if (counts < n_folds).any():
        raise PatientFoldAuthorityError(
            f"Cancers with fewer patients than folds: {counts[counts < n_folds].to_dict()}"
        )
    patients["_assignment_order"] = [
        _assignment_order(seed, cancer, patient)
        for cancer, patient in patients[["cancer_id", "patient_id"]].itertuples(
            index=False, name=None
        )
    ]
    patients = patients.sort_values(
        ["cancer_id", "_assignment_order", "patient_id"], kind="stable"
    )
    patients["patient_fold_id"] = (
        patients.groupby("cancer_id", observed=True).cumcount() % n_folds
    ).astype("int8")
    patients["fold_seed"] = int(seed)
    patients = patients.drop(columns="_assignment_order").sort_values(
        ["cancer_id", "patient_id"], kind="stable"
    ).reset_index(drop=True)
    sample_map = samples.merge(
        patients,
        on=["cancer_id", "patient_id"],
        how="left",
        validate="many_to_one",
    ).sort_values(["cancer_id", "patient_id", "sample_id"], kind="stable")
    sample_map = sample_map.reset_index(drop=True)
    conflicts = sample_map.groupby(
        ["cancer_id", "patient_id"], observed=True
    ).patient_fold_id.nunique()
    if conflicts.gt(1).any():
        raise PatientFoldAuthorityError("Samples from one patient cross folds")
    return patients, sample_map


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_patient_fold_authority(
    *,
    activity_root: str | Path,
    output_root: str | Path,
    seed: int = DEFAULT_SEED,
    expected_cancers: Sequence[str] = TCGA_CANCERS,
    expected_patients: int = 10_432,
    memory_limit: str = "2GB",
    launcher_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a new authority directory; existing paths are never reused."""

    source = Path(activity_root).resolve()
    output = Path(output_root).resolve()
    temporary = output.with_name(f".{output.name}.tmp")
    if output.exists():
        raise FileExistsError(f"Patient authority refuses output reuse: {output}")
    if temporary.exists():
        raise FileExistsError(f"Patient authority temporary directory exists: {temporary}")
    inventory, inventory_sha256 = activity_inventory(source)
    samples = _load_explicit_sample_patients(source, memory_limit=memory_limit)
    observed_cancers = sorted(samples.cancer_id.unique().tolist())
    required_cancers = sorted(str(value).upper() for value in expected_cancers)
    if observed_cancers != required_cancers:
        raise PatientFoldAuthorityError(
            f"Activity cancer authority mismatch: observed={observed_cancers}, expected={required_cancers}"
        )
    patients, sample_map = _build_assignments(samples, seed=int(seed), n_folds=N_FOLDS)
    if len(patients) != int(expected_patients):
        raise PatientFoldAuthorityError(
            f"Explicit patient count drift: {len(patients)} != {expected_patients}"
        )
    cross_cancer = int(patients.groupby("patient_id", observed=True).cancer_id.nunique().gt(1).sum())
    cross_fold = int(
        sample_map.groupby(["cancer_id", "patient_id"], observed=True)
        .patient_fold_id.nunique().gt(1).sum()
    )
    fold_sets = patients.groupby("cancer_id", observed=True).patient_fold_id.agg(
        lambda values: sorted(set(map(int, values)))
    )
    if cross_cancer or cross_fold or any(value != list(range(N_FOLDS)) for value in fold_sets):
        raise PatientFoldAuthorityError("Patient authority failed cancer/fold isolation")
    temporary.mkdir(parents=True)
    inventory_path = temporary / "SOURCE_INVENTORY.json"
    patient_path = temporary / "PATIENT_FOLD_AUTHORITY.tsv"
    sample_path = temporary / "SAMPLE_PATIENT_FOLD_MAP.tsv"
    inventory_payload = {
        "format": "CANCERLNCATLAS_V32_FORMAL_ACTIVITY_INVENTORY_V1",
        "source_root": str(source),
        "files": len(inventory),
        "bytes": sum(int(record["bytes"]) for record in inventory),
        "tree_sha256": inventory_sha256,
        "records": inventory,
    }
    inventory_path.write_text(
        json.dumps(inventory_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    patients.to_csv(patient_path, sep="\t", index=False, lineterminator="\n")
    sample_map.to_csv(sample_path, sep="\t", index=False, lineterminator="\n")
    patient_columns = ["cancer_id", "patient_id", "patient_fold_id", "fold_seed"]
    sample_columns = [
        "cancer_id", "sample_id", "patient_id", "patient_fold_id", "fold_seed"
    ]
    code_path = Path(__file__).resolve()
    implementation = {
        "builder_module": str(code_path),
        "builder_module_sha256": sha256_file(code_path),
        "launcher": str(Path(launcher_path).resolve()) if launcher_path else None,
        "launcher_sha256": sha256_file(launcher_path) if launcher_path else None,
        "old_prepare_v32_formal_executed": False,
        "assignment_algorithm": "SHA256(seed|cancer_id|explicit_patient_id)_SORT_THEN_ROUND_ROBIN",
    }
    per_cancer = []
    for cancer, group in patients.groupby("cancer_id", observed=True, sort=True):
        counts = group.groupby("patient_fold_id", observed=True).size().reindex(range(N_FOLDS), fill_value=0)
        per_cancer.append(
            {
                "cancer_id": str(cancer),
                "patients": int(len(group)),
                "fold_counts": {str(int(key)): int(value) for key, value in counts.items()},
                "max_minus_min": int(counts.max() - counts.min()),
            }
        )
    receipt = {
        "format": RECEIPT_FORMAT,
        "authority_format": AUTHORITY_FORMAT,
        "status": "PASS_EXPLICIT_PATIENT_FIRST_5FOLD_AUTHORITY",
        "analysis_version": "CancerLncAtlas_V3.2_PATIENT_FIRST_FOLD_AUTHORITY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "n_folds": N_FOLDS,
        "expected_cancers": required_cancers,
        "expected_patients": int(expected_patients),
        "observed": {
            "cancers": len(observed_cancers),
            "patients": int(len(patients)),
            "samples": int(len(sample_map)),
            "patients_with_multiple_samples": int(
                sample_map.groupby(["cancer_id", "patient_id"], observed=True)
                .sample_id.nunique().gt(1).sum()
            ),
            "patients_crossing_cancers": cross_cancer,
            "patients_crossing_folds": cross_fold,
            "ambiguous_sample_patient_keys": int(
                sample_map.groupby(["cancer_id", "sample_id"], observed=True)
                .patient_id.nunique().gt(1).sum()
            ),
        },
        "source_inventory": {
            **_artifact_record(inventory_path),
            "activity_tree_sha256": inventory_sha256,
            "activity_files": len(inventory),
            "activity_bytes": inventory_payload["bytes"],
        },
        "artifacts": {
            "patient_fold_authority": _artifact_record(patient_path),
            "sample_patient_fold_map": _artifact_record(sample_path),
        },
        "logical_hashes": {
            "patient_authority_sha256": _table_sha256(patients, patient_columns),
            "sample_patient_map_sha256": _table_sha256(sample_map, sample_columns),
        },
        "implementation": implementation,
        "per_cancer": per_cancer,
        "gates": {
            "explicit_patient_id_source_required": True,
            "sample_barcode_patient_fallback_forbidden": True,
            "all_33_cancers_present": len(observed_cancers) == 33,
            "all_five_folds_per_cancer": True,
            "patient_cross_cancer_count_zero": cross_cancer == 0,
            "patient_cross_fold_count_zero": cross_fold == 0,
            "output_reuse_forbidden": True,
        },
    }
    receipt_path = temporary / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    success = {
        "format": SUCCESS_FORMAT,
        "status": "SUCCESS",
        "receipt": _artifact_record(receipt_path),
        "patient_fold_authority": _artifact_record(patient_path),
        "sample_patient_fold_map": _artifact_record(sample_path),
        "source_inventory": _artifact_record(inventory_path),
        "formal_training_started": False,
        "production_8260_touched": False,
    }
    (temporary / "SUCCESS.json").write_text(
        json.dumps(success, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {**receipt, "output_root": str(output)}


def validate_patient_fold_authority(
    manifest_path: str | Path,
    receipt_path: str | Path,
    *,
    expected_seed: int = DEFAULT_SEED,
    expected_cancers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate a mounted authority without deriving patients from sample IDs."""

    manifest = Path(manifest_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    if manifest.is_symlink() or not manifest.is_file():
        raise PatientFoldAuthorityError("Patient fold manifest is missing or unsafe")
    if receipt_file.is_symlink() or not receipt_file.is_file():
        raise PatientFoldAuthorityError("Patient fold authority receipt is missing or unsafe")
    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PatientFoldAuthorityError("Patient authority receipt is invalid JSON") from exc
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("format") != RECEIPT_FORMAT
        or receipt.get("authority_format") != AUTHORITY_FORMAT
        or receipt.get("status") != "PASS_EXPLICIT_PATIENT_FIRST_5FOLD_AUTHORITY"
        or int(receipt.get("seed", -1)) != int(expected_seed)
        or int(receipt.get("n_folds", -1)) != N_FOLDS
        or receipt.get("gates", {}).get("explicit_patient_id_source_required") is not True
        or receipt.get("gates", {}).get("sample_barcode_patient_fallback_forbidden") is not True
    ):
        raise PatientFoldAuthorityError("Patient authority receipt contract failed")
    artifact = receipt.get("artifacts", {}).get("sample_patient_fold_map")
    if not isinstance(artifact, Mapping):
        raise PatientFoldAuthorityError("Receipt lacks sample/patient fold artifact")
    if (
        artifact.get("filename") != manifest.name
        or artifact.get("sha256") != sha256_file(manifest)
        or int(artifact.get("bytes", -1)) != manifest.stat().st_size
    ):
        raise PatientFoldAuthorityError("Sample/patient fold manifest hash drift")
    frame = pd.read_csv(
        manifest,
        sep="\t",
        dtype={"cancer_id": "string", "sample_id": "string", "patient_id": "string"},
    )
    columns = ["cancer_id", "sample_id", "patient_id", "patient_fold_id", "fold_seed"]
    if missing := sorted(set(columns) - set(frame.columns)):
        raise PatientFoldAuthorityError(f"Explicit patient fold manifest lacks: {missing}")
    if frame[columns].isna().any().any():
        raise PatientFoldAuthorityError("Explicit patient fold manifest contains nulls")
    for column in ("cancer_id", "sample_id", "patient_id"):
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise PatientFoldAuthorityError(f"Explicit patient fold manifest has empty {column}")
    frame["cancer_id"] = frame.cancer_id.str.upper()
    frame["patient_fold_id"] = pd.to_numeric(frame.patient_fold_id, errors="raise").astype(int)
    frame["fold_seed"] = pd.to_numeric(frame.fold_seed, errors="raise").astype(int)
    if set(frame.patient_fold_id) != set(range(N_FOLDS)) or set(frame.fold_seed) != {int(expected_seed)}:
        raise PatientFoldAuthorityError("Explicit patient fold/seed domain drift")
    if frame[["cancer_id", "sample_id"]].duplicated().any():
        raise PatientFoldAuthorityError("Explicit patient fold manifest duplicates samples")
    if frame.groupby(["cancer_id", "sample_id"], observed=True).patient_id.nunique().gt(1).any():
        raise PatientFoldAuthorityError("A sample maps to multiple explicit patients")
    if frame.groupby("patient_id", observed=True).cancer_id.nunique().gt(1).any():
        raise PatientFoldAuthorityError("An explicit patient crosses cancers")
    if frame.groupby(["cancer_id", "patient_id"], observed=True).patient_fold_id.nunique().gt(1).any():
        raise PatientFoldAuthorityError("An explicit patient crosses folds")
    observed_cancers = sorted(frame.cancer_id.unique().tolist())
    if expected_cancers is not None and observed_cancers != sorted(
        str(value).upper() for value in expected_cancers
    ):
        raise PatientFoldAuthorityError("Explicit patient authority cancer scope drift")
    logical = _table_sha256(frame, columns)
    if logical != receipt.get("logical_hashes", {}).get("sample_patient_map_sha256"):
        raise PatientFoldAuthorityError("Explicit sample/patient logical hash drift")
    patients = frame[["cancer_id", "patient_id", "patient_fold_id", "fold_seed"]].drop_duplicates()
    patient_logical = _table_sha256(
        patients, ["cancer_id", "patient_id", "patient_fold_id", "fold_seed"]
    )
    if patient_logical != receipt.get("logical_hashes", {}).get("patient_authority_sha256"):
        raise PatientFoldAuthorityError("Explicit patient logical hash drift")
    return {
        "rows": int(len(frame)),
        "samples": int(len(frame)),
        "patients": int(len(patients)),
        "cancers": observed_cancers,
        "folds": N_FOLDS,
        "seed": int(expected_seed),
        "manifest_sha256": sha256_file(manifest),
        "receipt_sha256": sha256_file(receipt_file),
        "patient_authority_sha256": patient_logical,
        "sample_patient_map_sha256": logical,
        "explicit_patient_id": True,
        "sample_fallback_used": False,
    }


def validate_frozen_v32_patient_fold_binding(
    manifest_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Fail closed unless both files are the one frozen V3.2 authority.

    This deliberately performs the complete semantic validator first and then
    pins the physical and logical identities.  Basename checks reject the old
    ``PATIENT_FOLD_MANIFEST.tsv`` surface even if somebody copies the new bytes
    into that legacy name.  No patient identifier is ever derived from a
    sample/barcode in this path.
    """

    manifest = Path(manifest_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    if manifest.name != FROZEN_V32_SAMPLE_PATIENT_MAP_FILENAME:
        raise PatientFoldAuthorityError(
            "Formal V3.2 consumers require SAMPLE_PATIENT_FOLD_MAP.tsv; "
            "legacy PATIENT_FOLD_MANIFEST.tsv is forbidden"
        )
    if receipt_file.name != FROZEN_V32_RECEIPT_FILENAME:
        raise PatientFoldAuthorityError(
            "Formal V3.2 consumers require PATIENT_FOLD_AUTHORITY_RECEIPT.json"
        )
    if manifest.parent != receipt_file.parent:
        raise PatientFoldAuthorityError(
            "Patient fold map and authority receipt must be co-located"
        )

    audit = validate_patient_fold_authority(
        manifest,
        receipt_file,
        expected_seed=DEFAULT_SEED,
        expected_cancers=TCGA_CANCERS,
    )
    physical = {
        "manifest_sha256": sha256_file(manifest),
        "receipt_sha256": sha256_file(receipt_file),
    }
    expected_physical = {
        "manifest_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        "receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
    }
    if physical != expected_physical:
        raise PatientFoldAuthorityError(
            f"Frozen V3.2 patient authority SHA drift: {physical}"
        )
    if (
        int(audit.get("seed", -1)) != DEFAULT_SEED
        or int(audit.get("folds", -1)) != N_FOLDS
        or int(audit.get("patients", -1)) != FROZEN_V32_PATIENTS
        or int(audit.get("samples", -1)) != FROZEN_V32_PATIENTS
        or list(audit.get("cancers", [])) != sorted(TCGA_CANCERS)
        or audit.get("patient_authority_sha256")
        != FROZEN_V32_PATIENT_AUTHORITY_LOGICAL_SHA256
        or audit.get("sample_patient_map_sha256")
        != FROZEN_V32_SAMPLE_PATIENT_MAP_LOGICAL_SHA256
        or audit.get("explicit_patient_id") is not True
        or audit.get("sample_fallback_used") is not False
    ):
        raise PatientFoldAuthorityError(
            "Frozen V3.2 patient authority semantic identity drift"
        )
    return {
        **audit,
        "binding_format": FROZEN_V32_BINDING_FORMAT,
        "status": "PASS_FROZEN_V32_PATIENT_FIRST_AUTHORITY",
        "expected_patients": FROZEN_V32_PATIENTS,
        "expected_cancer_count": len(TCGA_CANCERS),
        "legacy_patient_fold_manifest_accepted": False,
        "receipt_required": True,
        "files_co_located": True,
    }


def validate_frozen_v32_patient_fold_payload_binding(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen authority record embedded in a prepared payload.

    A training payload is a serialized object and cannot safely re-resolve the
    source paths recorded when it was prepared.  It must nevertheless carry
    the exact physical authority identities and the complete semantic audit.
    This gate is deliberately independent of the payload location and rejects
    the legacy manifest even when the surrounding training authorization is
    otherwise hash-consistent.
    """

    if not isinstance(binding, Mapping):
        raise PatientFoldAuthorityError(
            "Prepared payload lacks the frozen patient-fold authority binding"
        )
    authority = binding.get("authority")
    if not isinstance(authority, Mapping):
        raise PatientFoldAuthorityError(
            "Prepared payload patient-fold authority audit is missing"
        )
    if (
        binding.get("format") != FORMAL_PREPARED_BINDING_FORMAT
        or binding.get("status") != "PASS_FROZEN_V32_PATIENT_FIRST_AUTHORITY"
        or binding.get("sample_id_patient_fallback_used") is not False
        or binding.get("legacy_patient_fold_manifest_used") is not False
        or binding.get("sample_patient_fold_map", {}).get("sha256")
        != FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256
        or binding.get("authority_receipt", {}).get("sha256")
        != FROZEN_V32_RECEIPT_SHA256
        or authority.get("binding_format") != FROZEN_V32_BINDING_FORMAT
        or authority.get("status") != "PASS_FROZEN_V32_PATIENT_FIRST_AUTHORITY"
        or authority.get("manifest_sha256")
        != FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256
        or authority.get("receipt_sha256") != FROZEN_V32_RECEIPT_SHA256
        or authority.get("patient_authority_sha256")
        != FROZEN_V32_PATIENT_AUTHORITY_LOGICAL_SHA256
        or authority.get("sample_patient_map_sha256")
        != FROZEN_V32_SAMPLE_PATIENT_MAP_LOGICAL_SHA256
        or int(authority.get("seed", -1)) != DEFAULT_SEED
        or int(authority.get("folds", -1)) != N_FOLDS
        or int(authority.get("patients", -1)) != FROZEN_V32_PATIENTS
        or int(authority.get("samples", -1)) != FROZEN_V32_PATIENTS
        or list(authority.get("cancers", [])) != sorted(TCGA_CANCERS)
        or authority.get("explicit_patient_id") is not True
        or authority.get("sample_fallback_used") is not False
        or authority.get("legacy_patient_fold_manifest_accepted") is not False
        or authority.get("receipt_required") is not True
    ):
        raise PatientFoldAuthorityError(
            "Prepared payload patient-fold authority binding drift"
        )
    return {
        "status": "PASS_FROZEN_V32_PAYLOAD_FOLD_BINDING",
        "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        "authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        "patient_authority_logical_sha256": (
            FROZEN_V32_PATIENT_AUTHORITY_LOGICAL_SHA256
        ),
        "sample_patient_map_logical_sha256": (
            FROZEN_V32_SAMPLE_PATIENT_MAP_LOGICAL_SHA256
        ),
    }


def validate_frozen_v32_prepared_fold_binding(
    prepared_root: str | Path,
) -> dict[str, Any]:
    """Require a formal prepared tree to carry the frozen authority copies."""

    root = Path(prepared_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise PatientFoldAuthorityError("Formal prepared root is missing or unsafe")
    audit = validate_frozen_v32_patient_fold_binding(
        root / FROZEN_V32_SAMPLE_PATIENT_MAP_FILENAME,
        root / FROZEN_V32_RECEIPT_FILENAME,
    )
    binding_path = root / "PATIENT_FOLD_BINDING.json"
    if binding_path.is_symlink() or not binding_path.is_file():
        raise PatientFoldAuthorityError("Formal prepared patient-fold binding is missing")
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PatientFoldAuthorityError(
            "Formal prepared patient-fold binding is invalid JSON"
        ) from exc
    validate_frozen_v32_patient_fold_payload_binding(binding)
    return {
        "status": "PASS_FROZEN_V32_PREPARED_FOLD_BINDING",
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "authority": audit,
    }


__all__ = [
    "AUTHORITY_FORMAT",
    "RECEIPT_FORMAT",
    "SUCCESS_FORMAT",
    "DEFAULT_SEED",
    "FROZEN_V32_BINDING_FORMAT",
    "FORMAL_PREPARED_BINDING_FORMAT",
    "FROZEN_V32_PATIENTS",
    "FROZEN_V32_PATIENT_AUTHORITY_LOGICAL_SHA256",
    "FROZEN_V32_RECEIPT_FILENAME",
    "FROZEN_V32_RECEIPT_SHA256",
    "FROZEN_V32_SAMPLE_PATIENT_MAP_FILENAME",
    "FROZEN_V32_SAMPLE_PATIENT_MAP_LOGICAL_SHA256",
    "FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256",
    "N_FOLDS",
    "TCGA_CANCERS",
    "PatientFoldAuthorityError",
    "activity_inventory",
    "build_patient_fold_authority",
    "sha256_file",
    "validate_patient_fold_authority",
    "validate_frozen_v32_patient_fold_binding",
    "validate_frozen_v32_patient_fold_payload_binding",
    "validate_frozen_v32_prepared_fold_binding",
]
