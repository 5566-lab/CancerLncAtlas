"""Fail-closed lineage binding for current patient-first V3.2 products.

The patient-fold authority gate proves that the *requested* fold table is the
frozen 33-cancer/10,432-patient authority.  This module closes the other half
of the contract: a newly produced modality artifact must also record that
same table in its input lineage.  Merely presenting the new authority next to
an old output is therefore insufficient.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .patient_fold_authority import (
    FROZEN_V32_PATIENTS,
    FROZEN_V32_RECEIPT_SHA256,
    FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
    TCGA_CANCERS,
    sha256_file,
    validate_frozen_v32_patient_fold_binding,
)


LINEAGE_AUDIT_FORMAT = "CC_HHGT_V3_2_PATIENT_FIRST_OUTPUT_LINEAGE_AUDIT_V1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_FILENAME = "patient_fold_manifest.tsv"
_CURRENT_FILENAME = "sample_patient_fold_map.tsv"


class PatientFirstLineageError(RuntimeError):
    """Raised when an output cannot be tied to the frozen patient authority."""


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _load_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise PatientFirstLineageError(f"Lineage JSON is missing or unsafe: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PatientFirstLineageError(f"Lineage JSON is unreadable: {path}") from exc


def _walk(value: Any, context: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield context, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, context + (_normalise(key),))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _walk(child, context + (str(index),))


def _sha(value: Any) -> str | None:
    text = str(value).strip().lower()
    return text if _SHA256.fullmatch(text) else None


def _record_claims(payload: Any) -> dict[str, Any]:
    map_claims: list[dict[str, str]] = []
    receipt_claims: list[dict[str, str]] = []
    legacy_references: list[str] = []

    for context, value in _walk(payload):
        if isinstance(value, str) and _LEGACY_FILENAME in value.lower():
            legacy_references.append(value)
        if not isinstance(value, Mapping):
            continue

        record = dict(value)
        path_text = str(
            record.get("path")
            or record.get("filename")
            or record.get("artifact_id")
            or ""
        ).strip()
        basename = Path(path_text.replace("\\", "/")).name.lower()
        role = _normalise(record.get("role", record.get("source_role", "")))
        context_text = ".".join(context)
        binding_like = (
            "patient_fold" in context_text
            or "sample_patient_fold_map" in context_text
            or str(record.get("binding_format", "")).startswith("CANCERLNCATLAS_V32_FROZEN")
            or str(record.get("status", "")) == "PASS_FROZEN_V32_PATIENT_FIRST_AUTHORITY"
        )

        record_sha = _sha(record.get("sha256"))
        map_record = basename == _CURRENT_FILENAME or (
            role in {"folds", "patient_folds", "patient_fold_manifest", "sample_patient_fold_map"}
            and (not basename or "fold" in basename)
        )
        if map_record and record_sha:
            map_claims.append(
                {"field": f"{context_text or '<root>'}.sha256", "sha256": record_sha}
            )

        for key in (
            "patient_fold_manifest_sha256",
            "sample_patient_fold_map_sha256",
            "sample_patient_fold_map_physical_sha256",
        ):
            claim = _sha(record.get(key))
            if claim:
                map_claims.append({"field": f"{context_text or '<root>'}.{key}", "sha256": claim})
        if binding_like:
            claim = _sha(record.get("manifest_sha256"))
            if claim:
                map_claims.append(
                    {"field": f"{context_text or '<root>'}.manifest_sha256", "sha256": claim}
                )

        receipt_record = basename == "patient_fold_authority_receipt.json"
        if receipt_record and record_sha:
            receipt_claims.append(
                {"field": f"{context_text or '<root>'}.sha256", "sha256": record_sha}
            )
        for key in ("receipt_sha256", "authority_receipt_sha256"):
            claim = _sha(record.get(key))
            if claim:
                receipt_claims.append(
                    {"field": f"{context_text or '<root>'}.{key}", "sha256": claim}
                )

    def unique(records: list[dict[str, str]]) -> list[dict[str, str]]:
        return [dict(item) for item in {tuple(sorted(item.items())) for item in records}]

    return {
        "map_claims": sorted(unique(map_claims), key=lambda item: (item["field"], item["sha256"])),
        "receipt_claims": sorted(
            unique(receipt_claims), key=lambda item: (item["field"], item["sha256"])
        ),
        "legacy_references": sorted(set(legacy_references)),
    }


def _audit_document(path: Path, *, fold_bearing: bool) -> dict[str, Any]:
    payload = _load_json(path)
    claims = _record_claims(payload)
    observed_map = sorted({item["sha256"] for item in claims["map_claims"]})
    observed_receipt = sorted({item["sha256"] for item in claims["receipt_claims"]})
    reasons: list[str] = []
    if claims["legacy_references"]:
        reasons.append("LEGACY_PATIENT_FOLD_MANIFEST_REFERENCED")
    wrong_map = sorted(set(observed_map) - {FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256})
    wrong_receipt = sorted(set(observed_receipt) - {FROZEN_V32_RECEIPT_SHA256})
    if wrong_map:
        reasons.append("PATIENT_FOLD_MAP_SHA_DRIFT:" + ",".join(wrong_map))
    if wrong_receipt:
        reasons.append("PATIENT_FOLD_RECEIPT_SHA_DRIFT:" + ",".join(wrong_receipt))
    if fold_bearing and FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256 not in observed_map:
        reasons.append("FOLD_BEARING_LINEAGE_LACKS_FROZEN_MAP_SHA")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "fold_bearing": bool(fold_bearing),
        "map_claims": claims["map_claims"],
        "receipt_claims": claims["receipt_claims"],
        "legacy_references": claims["legacy_references"],
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
    }


def validate_patient_first_output_lineage(
    *,
    patient_folds_path: str | Path,
    patient_fold_authority_receipt_path: str | Path,
    fold_lineage_paths: Sequence[str | Path],
    run_lineage_paths: Sequence[str | Path] = (),
    component: str,
) -> dict[str, Any]:
    """Validate one current output chain without accepting legacy fold products.

    ``fold_lineage_paths`` are the input audits/bindings that directly declare
    the fold table used to build the output.  Each is required to carry the
    physical frozen map SHA.  ``run_lineage_paths`` are downstream manifests;
    they are recorded and are rejected if they contain a conflicting fold or
    receipt claim, but they need not duplicate upstream declarations.
    """

    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", str(component)):
        raise PatientFirstLineageError("component must be a stable lowercase identifier")
    if not fold_lineage_paths:
        raise PatientFirstLineageError("At least one fold-bearing lineage JSON is required")

    authority = validate_frozen_v32_patient_fold_binding(
        Path(patient_folds_path).resolve(),
        Path(patient_fold_authority_receipt_path).resolve(),
    )
    if (
        int(authority.get("patients", -1)) != FROZEN_V32_PATIENTS
        or int(authority.get("expected_cancer_count", -1)) != len(TCGA_CANCERS)
        or len(authority.get("cancers", [])) != len(TCGA_CANCERS)
        or authority.get("manifest_sha256") != FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256
        or authority.get("receipt_sha256") != FROZEN_V32_RECEIPT_SHA256
    ):
        raise PatientFirstLineageError("Frozen authority semantic identity drift")

    fold_records = [
        _audit_document(Path(path).resolve(), fold_bearing=True)
        for path in fold_lineage_paths
    ]
    run_records = [
        _audit_document(Path(path).resolve(), fold_bearing=False)
        for path in run_lineage_paths
    ]
    failed = [record for record in fold_records + run_records if record["status"] != "PASS"]
    if failed:
        compact = {record["path"]: record["reasons"] for record in failed}
        raise PatientFirstLineageError(f"Patient-first output lineage rejected: {compact}")

    return {
        "format": LINEAGE_AUDIT_FORMAT,
        "status": "PASS_FROZEN_PATIENT_FIRST_OUTPUT_LINEAGE",
        "component": str(component),
        "authority": {
            "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
            "authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
            "cancers": len(TCGA_CANCERS),
            "patients": FROZEN_V32_PATIENTS,
            "folds": 5,
            "legacy_patient_fold_manifest_accepted": False,
        },
        "fold_lineage": fold_records,
        "run_lineage": run_records,
        "old_fold_output_accepted": False,
        "output_reuse_authorized": False,
        "production_8260_touched": False,
    }


def write_patient_first_output_lineage_audit(
    audit: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Write an audit once; an existing path is never replaced."""

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(audit), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as exc:
        raise PatientFirstLineageError(
            f"Lineage audit refuses output reuse: {destination}"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return destination


__all__ = [
    "LINEAGE_AUDIT_FORMAT",
    "PatientFirstLineageError",
    "validate_patient_first_output_lineage",
    "write_patient_first_output_lineage_audit",
]
